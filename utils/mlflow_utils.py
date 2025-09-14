from __future__ import annotations

import contextlib
from typing import Any

import mlflow

# Carga settings del proyecto
try:
    import settings as settings

    S = settings.S
except Exception:
    # fallback muy defensivo
    class _S:
        mlflow_enabled = True
        mlflow_tracking_uri = None
        mlflow_experiment = "PHIBOT"
        mlflow_nested = False
        mlflow_tags = {"project": "phibot"}

    S = _S()

# ---------- Internals ----------


def _enabled() -> bool:
    return bool(getattr(S, "mlflow_enabled", True))


def _ensure_setup():
    """Configura tracking URI y experimento (idempotente)."""
    uri = getattr(S, "mlflow_tracking_uri", None) or "file:./mlruns"
    try:
        mlflow.set_tracking_uri(uri)
    except Exception:
        pass
    exp = getattr(S, "mlflow_experiment", "PHIBOT")
    try:
        mlflow.set_experiment(exp)
    except Exception:
        pass


@contextlib.contextmanager
def _dummy_run():
    """Contexto no-op cuando MLflow está desactivado."""
    yield


# ---------- API pública usada en el repo ----------


def start_run(
    run_name: str | None = None, tags: dict[str, Any] | None = None, nested: bool | None = None
):
    """Abre un run de MLflow si está habilitado; si no, devuelve un contexto no-op."""
    if not _enabled():
        return _dummy_run()
    _ensure_setup()
    try:
        ctx = mlflow.start_run(
            run_name=run_name,
            nested=(getattr(S, "mlflow_nested", False) if nested is None else nested),
        )
        if tags:
            try:
                mlflow.set_tags(tags)
            except Exception:
                pass
        # tags globales del proyecto (si las hubiera)
        proj_tags = getattr(S, "mlflow_tags", None)
        if proj_tags:
            try:
                mlflow.set_tags(proj_tags)
            except Exception:
                pass
        return ctx
    except Exception:
        # en caso de error, no rompemos el flujo
        return _dummy_run()


def end_run():
    """Cierra el run activo si corresponde."""
    if not _enabled():
        return
    try:
        mlflow.end_run()
    except Exception:
        pass


def log_params(params: dict[str, Any]):
    """Loguea parámetros en el run activo (silencioso si no hay run/está deshabilitado)."""
    if not _enabled():
        return
    try:
        # convertir Path y tipos no serializables
        _p = {
            k: (str(v) if not isinstance(v, (int, float, str, bool)) else v)
            for k, v in params.items()
        }
        mlflow.log_params(_p)
    except Exception:
        pass


def log_metrics(metrics: dict[str, float], step: int | None = None):
    """Loguea métricas en el run activo (silencioso si no hay run/está deshabilitado)."""
    if not _enabled():
        return
    try:
        # filtra NaNs/None
        clean = {
            k: float(v)
            for k, v in metrics.items()
            if v is not None and not (isinstance(v, float) and (v != v))
        }
        if not clean:
            return
        if step is None:
            mlflow.log_metrics(clean)
        else:
            mlflow.log_metrics(clean, step=step)
    except Exception:
        pass
