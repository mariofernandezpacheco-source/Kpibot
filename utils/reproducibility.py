# utils/reproducibility.py
import json
import os
import platform
import random

import numpy as np


def set_global_determinism(seed: int, set_pythonhash: bool = True):
    """
    Fija semillas globales para random y numpy.
    Si set_pythonhash=True, fija PYTHONHASHSEED (debe ocurrir lo antes posible en el arranque del proceso).
    """
    if set_pythonhash:
        os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def xgb_deterministic_params(seed: int, n_jobs: int):
    """
    Parámetros seguros para XGBoost sklearn-wrapper (CPU).
    Nota: para resultados 100% reproducibles, mantener n_jobs=1.
    """
    return dict(
        random_state=seed,
        seed=seed,  # redundante pero inofensivo
        n_jobs=n_jobs,  # usa 1 si quieres determinismo extremo
        tree_method="hist",  # CPU; evita GPU por no determinista
        predictor="cpu_predictor",
        verbosity=0,
        # Si usas subsample/colsample, con seed + 1 hilo siguen siendo deterministas
    )


def lgbm_deterministic_params(seed: int, n_jobs: int):
    """
    Parámetros seguros para LightGBM (si lo incorporas ahora o más adelante).
    """
    return dict(
        random_state=seed,
        num_threads=n_jobs,  # 1 para determinismo fuerte
        deterministic=True,  # fuerza determinismo interno
        bagging_seed=seed,
        feature_fraction_seed=seed,
        data_random_seed=seed,
        # force_row_wise=True  # opcional en versiones antiguas
    )


def collect_env_versions():
    """
    Devuelve un dict con versiones de librerías clave + SO.
    """
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import numpy as _np

        info["numpy"] = _np.__version__
    except Exception:
        pass
    try:
        import pandas as _pd

        info["pandas"] = _pd.__version__
    except Exception:
        pass
    try:
        import sklearn as _sk

        info["sklearn"] = _sk.__version__
    except Exception:
        pass
    try:
        import xgboost as _xgb

        info["xgboost"] = _xgb.__version__
    except Exception:
        pass
    try:
        import lightgbm as _lgb

        info["lightgbm"] = _lgb.__version__
    except Exception:
        pass
    return info


def write_env_versions(path: str | os.PathLike, extra: dict | None = None):
    """
    Escribe las versiones a disco en JSON (crea carpetas si faltan).
    """
    info = collect_env_versions()
    if extra:
        info.update(extra)
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info
