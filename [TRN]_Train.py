# [TRN]_Train.py — Entrenamiento con MLflow tracking (experimentos, artefactos y registro opcional)
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

# === Settings y rutas
import settings as settings

S = settings.S
DATA_DIR = Path(S.data_path)
MODELS_DIR = Path(S.models_path)
CONFIG_DIR = Path(S.config_path)
LOGS_DIR = Path(S.logs_path)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# === Utils del proyecto
from sklearn import metrics as skmetrics

# === Sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.A_data_loader import load_data
from utils.B_feature_engineering import add_technical_indicators
from utils.C_label_generator import generate_triple_barrier_labels
from utils.logging_cfg import get_logger
from utils.schemas import LabelsSchema, OHLCVSchema, validate_df

# === XGBoost (opcional)
try:
    from xgboost import XGBClassifier

    _XGB_AVAILABLE = True
except Exception:
    _XGB_AVAILABLE = False

# === MLflow ===
import hashlib
import os
import subprocess
import tempfile

import mlflow
from mlflow.models import infer_signature

LOGGER = get_logger("trainer")


# =========================
# Helpers
# =========================
def _sha256(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "unknown"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def ensure_atr14(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date")
    if "atr_14" not in out.columns:
        prev_close = out["close"].shift(1)
        tr = pd.concat(
            [
                (out["high"] - out["low"]).abs(),
                (out["high"] - prev_close).abs(),
                (out["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return out


def _recent_slice(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if not days or days <= 0 or "date" not in df.columns:
        return df
    cutoff = pd.to_datetime(df["date"], utc=True).max() - pd.Timedelta(days=days)
    return df[df["date"] >= cutoff]


def _slice_by_dates(df: pd.DataFrame, date_from: str | None, date_to: str | None) -> pd.DataFrame:
    if "date" not in df.columns:
        return df
    z = df.copy()
    z["date"] = pd.to_datetime(z["date"], utc=True, errors="coerce")
    if date_from:
        z = z[z["date"] >= pd.Timestamp(date_from, tz="UTC")]
    if date_to:
        z = z[z["date"] <= pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1)]
    return z


def _feature_cols_for_b2(df: pd.DataFrame) -> list[str]:
    prefer = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "atr_14",
        "ret_1",
        "ret_5",
        "ma_5",
        "ma_10",
        "rsi_14",
        "stoch_k",
        "stoch_d",
        "willr_14",
        "bb_up",
        "bb_mid",
        "bb_low",
    ]
    return [c for c in prefer if c in df.columns]


FEATURE_SETS: dict[str, list[str] | None] = {
    "core": [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "atr_14",
        "ret_1",
        "ma_5",
        "ma_10",
        "rsi_14",
    ],
    "core+vol": [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "atr_14",
        "ret_1",
        "ret_5",
        "ma_5",
        "ma_10",
        "rsi_14",
        "bb_up",
        "bb_low",
    ],
    "all": None,  # usa todas las columnas calculadas en add_technical_indicators/_feature_cols_for_b2
}


def apply_feature_set(dfX: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    cols = FEATURE_SETS.get(feature_set)
    if cols is None:
        return dfX
    keep = [c for c in cols if c in dfX.columns]
    return dfX.reindex(columns=keep, fill_value=0.0)


def _map_labels(y_raw: pd.Series, label_map_mode: str) -> tuple[pd.Series, dict, list]:
    mode = str(label_map_mode).lower()
    if mode == "multiclass_3way":
        mapping = {-1: 0, 0: 1, 1: 2}
        y = y_raw.map(mapping).astype(int)
        class_order = [0, 1, 2]
    else:
        mapping = {-1: 0, 0: 0, 1: 1}
        y = y_raw.map(mapping).astype(int)
        class_order = [0, 1]
    return y, mapping, class_order


def _build_estimator(
    model_type: str, n_classes: int, random_state: int, hparams: dict[str, Any] | None = None
):
    hparams = hparams or {}
    m = str(model_type).lower()
    if m in ("xgb", "xgboost") and _XGB_AVAILABLE:
        params = dict(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=0,
        )
        params.update(hparams)
        if n_classes > 2:
            params.update(objective="multi:softprob", num_class=n_classes)
        else:
            params.update(objective="binary:logistic")
        return XGBClassifier(**params)
    elif m == "rf":
        from sklearn.ensemble import RandomForestClassifier

        params = dict(n_estimators=400, max_depth=8, random_state=random_state, n_jobs=-1)
        params.update(hparams)
        return RandomForestClassifier(**params)
    elif m == "logreg":
        params = dict(max_iter=1000, random_state=random_state, solver="lbfgs")
        params.update(hparams)
        return LogisticRegression(**params)
    else:
        solver = "lbfgs"
        multi = "multinomial" if n_classes > 2 else "auto"
        return LogisticRegression(
            max_iter=2000, solver=solver, multi_class=multi, random_state=random_state
        )


def _build_pipeline(
    feature_cols: list[str], y: pd.Series, model_type: str, hparams: dict[str, Any] | None = None
) -> Pipeline:
    n_classes = int(np.unique(y).size)
    est = _build_estimator(
        model_type, n_classes, random_state=getattr(S, "seed", 42), hparams=hparams or {}
    )

    num_proc = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    prep = ColumnTransformer(
        transformers=[("num", num_proc, feature_cols)], remainder="drop", sparse_threshold=0.0
    )

    pipe = Pipeline(steps=[("prep", prep), ("clf", est)])
    return pipe


def _pipeline_dir(ticker: str) -> Path:
    return MODELS_DIR / ticker.upper()


def _save_pipeline(ticker: str, pipe: Pipeline, meta: dict):
    out_dir = _pipeline_dir(ticker)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "pipeline.pkl")
    (out_dir / "pipeline_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


# =========================
# MLflow helpers
# =========================
def _mlflow_setup():
    uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    exp = os.getenv("MLFLOW_EXPERIMENT", "PHIBOT")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp)


def _log_config_and_fingerprints():
    cfg = CONFIG_DIR / "config.yaml"
    if cfg.exists():
        mlflow.log_artifact(str(cfg), artifact_path="config")
        mlflow.set_tag("config_sha256", _sha256(cfg))
    if Path(DATA_DIR).exists():
        mlflow.set_tag("data_dir", str(DATA_DIR))


try:
    import matplotlib

    matplotlib.use("Agg")
except Exception:
    pass


def _log_plots(y_true, y_proba):
    import matplotlib.pyplot as plt

    try:
        fpr, tpr, _ = skmetrics.roc_curve(y_true, y_proba)
        auc = skmetrics.auc(fpr, tpr)
        fig = plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], "--")
        plt.title(f"ROC AUC={auc:.3f}")
        fig.savefig("roc.png", bbox_inches="tight")
        mlflow.log_artifact("roc.png", artifact_path="plots")
        plt.close(fig)
    except Exception:
        pass
    try:
        precision, recall, _ = skmetrics.precision_recall_curve(y_true, y_proba)
        pr_auc = skmetrics.auc(recall, precision)
        fig = plt.figure()
        plt.plot(recall, precision)
        plt.title(f"PR AUC={pr_auc:.3f}")
        fig.savefig("pr.png", bbox_inches="tight")
        mlflow.log_artifact("pr.png", artifact_path="plots")
        plt.close(fig)
    except Exception:
        pass


# =========================
# Entrenamiento por ticker (con MLflow)
# =========================
def train_ticker(
    ticker: str,
    timeframe: str,
    *,
    overrides: dict[str, Any] | None = None,
    force_relabel: bool = False,
    clean: bool = False,
    print_stats: bool = True,
) -> Path:
    """
    Entrena y guarda 02_models/{TICKER}/pipeline.pkl + pipeline_meta.json
    admite overrides: tp_multiplier, sl_multiplier, time_limit_candles, days, date_from, date_to,
                      model, hparams (dict o json str), feature_set, inference_threshold
    """
    _mlflow_setup()
    ov = overrides or {}

    LABEL_MODE = str(getattr(S, "label_map_mode", "multiclass_3way")).lower()
    MODEL_TYPE = str(ov.get("model", getattr(S, "model_type", "xgb"))).lower()
    FEATURE_SET = str(ov.get("feature_set", "core"))
    DAYS = int(ov.get("days", getattr(S, "days_of_data", 90)))
    DATE_FROM = ov.get("date_from")
    DATE_TO = ov.get("date_to")

    TP_MULT = float(ov.get("tp_multiplier", getattr(S, "tp_multiplier", 3.0)))
    SL_MULT = float(ov.get("sl_multiplier", getattr(S, "sl_multiplier", 2.0)))
    TIME_LIMIT = int(ov.get("time_limit_candles", getattr(S, "time_limit_candles", 16)))
    INF_THR = ov.get("inference_threshold", None)

    # 1) Cargar datos
    df = load_data(ticker=ticker, timeframe=timeframe, use_local=True, base_path=DATA_DIR)
    if df.empty:
        raise RuntimeError(f"[{ticker}] No hay datos.")
    df = validate_df(df, OHLCVSchema, name="OHLCV(training/input)")
    df = ensure_atr14(df)

    # Filtro temporal
    df = _recent_slice(df, DAYS)
    df = _slice_by_dates(df, DATE_FROM, DATE_TO)

    # 2) Enriquecer
    try:
        df = add_technical_indicators(df)
    except Exception:
        pass

    # 3) Etiquetado
    if force_relabel or ("label" not in df.columns):
        label_kwargs = {}
        if "label_window" in generate_triple_barrier_labels.__code__.co_varnames:
            label_kwargs["label_window"] = int(getattr(S, "label_window", 5))
        df = generate_triple_barrier_labels(
            data=df,
            volatility_col=str(getattr(S, "volatility_col", "atr_14")),
            tp_multiplier=TP_MULT,
            sl_multiplier=SL_MULT,
            time_limit_candles=TIME_LIMIT,
            **label_kwargs,
        )
    df = validate_df(df, LabelsSchema, name="labels(training)")

    # 4) Features
    feat_all = _feature_cols_for_b2(df)
    if not feat_all:
        raise ValueError(f"[{ticker}] Sin columnas de features válidas tras ingeniería.")
    X = df[[c for c in feat_all if c in df.columns]].copy()
    X = apply_feature_set(X, FEATURE_SET)
    y_raw = df["label"].astype(int)

    # 5) Mapeo de etiquetas
    y, mapping_dict, class_order = _map_labels(y_raw, LABEL_MODE)
    if y.nunique() < 2:
        raise ValueError(
            f"[{ticker}] Solo una clase tras mapeo ({LABEL_MODE}); ajusta TP/SL/time_limit/label_window."
        )

    if print_stats:
        vc = y.value_counts().sort_index()
        print(f"[{ticker}] clases={sorted(y.unique().tolist())} dist={vc.to_dict()}")

    # 6) Pipeline y fit
    hparams = ov.get("hparams")
    if isinstance(hparams, str):
        try:
            hparams = json.loads(hparams)
        except Exception:
            hparams = {}
    if hparams is None:
        hparams = {}

    pipe = _build_pipeline(
        feature_cols=list(X.columns), y=y, model_type=MODEL_TYPE, hparams=hparams
    )

    # === MLflow run ===
    run_name = f"train/{ticker}/{timeframe}"

    import importlib.metadata as imd
    import subprocess

    import mlflow

    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        git = "unknown"
    mlflow.set_tag("git_commit", git)
    mlflow.set_tag("package_version", imd.version("phibot"))

    with mlflow.start_run(run_name=run_name):
        # tags/params
        mlflow.set_tags(
            {
                "phase": "training",
                "ticker": ticker.upper(),
                "timeframe": timeframe,
                "strategy": getattr(S, "strategy", "default"),
                "git_commit": _git_commit(),
            }
        )
        log_params = {
            "model_type": MODEL_TYPE,
            "label_mode": LABEL_MODE,
            "days_history": DAYS,
            "tp_multiplier": TP_MULT,
            "sl_multiplier": SL_MULT,
            "time_limit_candles": TIME_LIMIT,
            "label_window": int(getattr(S, "label_window", 5)),
            "n_features": int(X.shape[1]),
            "feature_set": FEATURE_SET,
            "date_from": DATE_FROM or "",
            "date_to": DATE_TO or "",
        }
        if INF_THR is not None:
            log_params["inference_threshold"] = float(INF_THR)
        mlflow.log_params(log_params)
        if hparams:
            mlflow.log_params({f"hparam_{k}": v for k, v in hparams.items()})
        _log_config_and_fingerprints()

        # Fit + tiempo
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X, y)
        t_fit = time.time() - t0
        mlflow.log_metric("train_fit_seconds", float(t_fit))

        # --- probabilidades in-sample + métricas
        try:
            y_proba = pipe.predict_proba(X)
            proba_pos = (
                y_proba[:, -1] if (y_proba.ndim == 2 and y_proba.shape[1] > 1) else y_proba.ravel()
            )
            y_bin = (y == (y.max())).astype(int).values

            roc_auc = float(skmetrics.roc_auc_score(y_bin, proba_pos))
            precision, recall, _ = skmetrics.precision_recall_curve(y_bin, proba_pos)
            pr_auc = float(skmetrics.auc(recall, precision))
            mlflow.log_metrics({"train_roc_auc": roc_auc, "train_pr_auc": pr_auc})

            try:
                mlflow.log_metric(
                    "train_log_loss", float(log_loss(y_bin, proba_pos, labels=[0, 1]))
                )
            except Exception:
                pass
            try:
                mlflow.log_metric("train_brier", float(brier_score_loss(y_bin, proba_pos)))
            except Exception:
                pass

            try:
                ks = float(
                    np.max(
                        np.abs(
                            np.cumsum(np.sort(proba_pos[y_bin == 1])) / np.sum(y_bin == 1)
                            - np.cumsum(np.sort(proba_pos[y_bin == 0])) / np.sum(y_bin == 0)
                        )
                    )
                )
                mlflow.log_metric("train_ks", ks)
            except Exception:
                pass

            # Métricas a umbral (si hay)
            thr_for_metrics = None
            if INF_THR is not None:
                thr_for_metrics = float(INF_THR)
            elif getattr(S, "threshold_default", None) is not None:
                thr_for_metrics = float(S.threshold_default)

            if thr_for_metrics is not None:
                y_hat = (proba_pos >= thr_for_metrics).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_bin, y_hat, labels=[0, 1]).ravel()
                mlflow.log_metrics(
                    {
                        "train_tn": int(tn),
                        "train_fp": int(fp),
                        "train_fn": int(fn),
                        "train_tp": int(tp),
                        "train_accuracy_at_thr": float(accuracy_score(y_bin, y_hat)),
                        "train_precision_at_thr": float(
                            precision_score(y_bin, y_hat, zero_division=0)
                        ),
                        "train_recall_at_thr": float(recall_score(y_bin, y_hat, zero_division=0)),
                        "train_f1_at_thr": float(f1_score(y_bin, y_hat, zero_division=0)),
                        "train_balanced_acc_at_thr": float(balanced_accuracy_score(y_bin, y_hat)),
                        "train_specificity_at_thr": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                    }
                )
        except Exception as e:
            mlflow.set_tag("metrics_error", repr(e))
            print(f"[{ticker}] metrics_error: {e}")

        # Artefactos meta
        try:
            with tempfile.TemporaryDirectory() as td:
                td = Path(td)
                (td / "features.txt").write_text("\n".join(list(X.columns)), encoding="utf-8")
                mlflow.log_artifact(str(td / "features.txt"), artifact_path="meta")
                meta_dict = {
                    "label_map_mode": LABEL_MODE,
                    "class_order": class_order,
                    "model_type": MODEL_TYPE,
                    "seed": int(getattr(S, "seed", 42)),
                    "days_history": DAYS,
                    "tp_multiplier": TP_MULT,
                    "sl_multiplier": SL_MULT,
                    "time_limit_candles": TIME_LIMIT,
                    "feature_set": FEATURE_SET,
                }
                (td / "pipeline_meta.json").write_text(
                    json.dumps(meta_dict, indent=2), encoding="utf-8"
                )
                mlflow.log_artifact(str(td / "pipeline_meta.json"), artifact_path="meta")
        except Exception:
            pass

        # Guardar modelo (MLflow 3 usa 'name='; evitamos artifact_path deprecado)
        try:
            sig = infer_signature(X.head(100), pipe.predict_proba(X.head(100)))
        except Exception:
            sig = None
        input_example = X.head(5)
        reg_name = f"phibot_{getattr(S,'strategy','default')}_{ticker.upper()}"
        register = bool(int(os.getenv("USE_MLFLOW_REGISTRY", "0")))
        info = mlflow.sklearn.log_model(
            sk_model=pipe,
            name="model",
            signature=sig,
            input_example=input_example,
            registered_model_name=(reg_name if register else None),
        )
        mlflow.set_tag("logged_model_uri", info.model_uri)

        # Persistencia local habitual (si clean/overwrite ⇒ borra antes)
        out_dir = _pipeline_dir(ticker)
        if clean and out_dir.exists():
            import shutil

            shutil.rmtree(out_dir, ignore_errors=True)

        meta = {
            "timeframe": timeframe,
            "features": list(X.columns),
            "label_map_mode": LABEL_MODE,
            "class_order": class_order,
            "model_type": MODEL_TYPE,
            "seed": int(getattr(S, "seed", 42)),
            "days_history": DAYS,
            "tp_multiplier": TP_MULT,
            "sl_multiplier": SL_MULT,
            "time_limit_candles": TIME_LIMIT,
            "feature_set": FEATURE_SET,
        }
        _save_pipeline(ticker, pipe, meta)

        try:
            clf = pipe.named_steps["clf"]
            features = list(X.columns)
            imp_df = None
            if hasattr(clf, "feature_importances_"):
                imp_df = pd.DataFrame({"feature": features, "importance": clf.feature_importances_})
            elif hasattr(clf, "coef_"):
                coef = np.ravel(clf.coef_) if clf.coef_.ndim == 2 else clf.coef_
                imp_df = pd.DataFrame(
                    {"feature": features, "coef": coef, "importance": np.abs(coef)}
                )
            if imp_df is not None:
                imp_df = imp_df.sort_values("importance", ascending=False)
                tmp_path = Path(tempfile.mkdtemp()) / "feature_importance.csv"
                imp_df.to_csv(tmp_path, index=False)
                mlflow.log_artifact(str(tmp_path), artifact_path="importance")
        except Exception:
            pass

        # Distribución de clases
        vc = y.value_counts().to_dict()
        mlflow.log_params({f"class_count_{k}": int(v) for k, v in vc.items()})

    return _pipeline_dir(ticker) / "pipeline.pkl"


def train_single_ticker(ticker: str, timeframe: str) -> Path:
    return train_ticker(ticker, timeframe)


# =========================
# CLI batch
# =========================
def _tickers_from_file(name: str) -> list[str]:
    candidate = Path(name)
    if candidate.exists():
        file_path = candidate
    else:
        file_path = CONFIG_DIR / name

    if not file_path.exists():
        raise FileNotFoundError(f"No existe {file_path}")

    with open(file_path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and ln.strip().lower() != "ticker"]
    out, seen = [], set()
    for t in lines:
        tu = t.upper()
        if tu not in seen:
            seen.add(tu)
            out.append(tu)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Entrenamiento B2 por ticker (pipeline.pkl + meta) + MLflow."
    )
    ap.add_argument("--timeframe", default=str(getattr(S, "timeframe_default", "5mins")))
    ap.add_argument("--tickers_file", default="top_100_robustos.txt")
    # overrides de labeling/model/features/periodo
    ap.add_argument("--tp", type=float, default=None, help="TP multiplier override")
    ap.add_argument("--sl", type=float, default=None, help="SL multiplier override")
    ap.add_argument("--time_limit", type=int, default=None, help="time limit candles override")
    ap.add_argument("--model", default=None, choices=["xgb", "rf", "logreg"], help="modelo ML")
    ap.add_argument(
        "--hparams", type=str, default=None, help='JSON de hiperparámetros, p.ej. {"max_depth":4}'
    )
    ap.add_argument(
        "--feature_set",
        default=None,
        choices=["core", "core+vol", "all"],
        help="subset de features",
    )
    ap.add_argument("--days", type=int, default=None, help="últimos N días")
    ap.add_argument("--date_from", type=str, default=None, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--date_to", type=str, default=None, help="YYYY-MM-DD (UTC)")
    ap.add_argument(
        "--inference_threshold", type=float, default=None, help="umbral que quieres usar en vivo"
    )

    ap.add_argument(
        "--force_relabel", action="store_true", help="Ignora 'label' existente y relabela de nuevo."
    )
    # Alias claro para sobreescritura (equivale a --clean)
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Borra 02_models/{TICKER} antes de guardar (sobrescribe).",
    )
    ap.add_argument("--clean", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument(
        "--print_stats", action="store_true", help="Imprime distribución de clases tras el mapeo."
    )
    args = ap.parse_args()

    # Normaliza overwrite → clean
    if args.overwrite:
        args.clean = True

    # === Lecturas automáticas de selección ===
    # 1) Global
    try:
        _sel_path = Path(S.config_path) / "optimizer_selected.json"
        if _sel_path.exists():
            _sel = json.loads(_sel_path.read_text(encoding="utf-8"))

            def _set_if_missing(attr, key, *, dump_json=False):
                cur = getattr(args, attr, None)
                if cur in (None, ""):
                    val = _sel.get(key, None)
                    if val is not None:
                        setattr(args, attr, json.dumps(val) if dump_json else val)

            _set_if_missing("tp", "tp_multiplier")
            _set_if_missing("sl", "sl_multiplier")
            _set_if_missing("time_limit", "time_limit_candles")
            _set_if_missing("days", "days")
            _set_if_missing("date_from", "date_from")
            _set_if_missing("date_to", "date_to")
            _set_if_missing("model", "model")
            _set_if_missing("feature_set", "feature_set")
            _set_if_missing("hparams", "hparams", dump_json=True)
            _set_if_missing("inference_threshold", "inference_threshold")
    except Exception as _e:
        print(f"⚠️ No se pudo leer optimizer_selected.json: {_e}")

    # 2) Por-ticker (si existe)
    per_ticker_sel: dict = {}
    try:
        _sel_bt_path = Path(S.config_path) / "optimizer_selected_by_ticker.json"
        if _sel_bt_path.exists():
            _selbt = json.loads(_sel_bt_path.read_text(encoding="utf-8"))
            per_ticker_sel = _selbt.get("per_ticker", {})
    except Exception as _e:
        print(f"⚠️ No se pudo leer optimizer_selected_by_ticker.json: {_e}")

    base_timeframe = args.timeframe
    tickers = _tickers_from_file(args.tickers_file)

    # Overrides base (global)
    base_overrides = {
        "tp_multiplier": args.tp,
        "sl_multiplier": args.sl,
        "time_limit_candles": args.time_limit,
        "model": args.model,
        "hparams": args.hparams,
        "feature_set": args.feature_set or "core",
        "days": args.days if args.days is not None else getattr(S, "days_of_data", 90),
        "date_from": args.date_from,
        "date_to": args.date_to,
        "inference_threshold": args.inference_threshold,
    }

    print(
        f"Iniciando entrenamiento | timeframe(base)={base_timeframe} | TP={base_overrides['tp_multiplier'] or S.tp_multiplier} "
        f"| SL={base_overrides['sl_multiplier'] or S.sl_multiplier} | TL={base_overrides['time_limit_candles'] or S.time_limit_candles} "
        f"| days={base_overrides['days']} | model={base_overrides['model'] or getattr(S,'model_type','xgb')} "
        f"| feature_set={base_overrides['feature_set']} | overwrite={'YES' if args.clean else 'NO'}"
    )

    ok, fail = 0, 0
    for t in tickers:
        try:
            # Aplica selección por-ticker si existe
            ov_t = dict(base_overrides)
            tf_t = base_timeframe
            if t in per_ticker_sel:
                sel = per_ticker_sel[t]
                # timeframe por ticker (si viene y quieres permitirlo)
                tf_t = sel.get("timeframe") or base_timeframe
                for k_src, k_dst in [
                    ("tp_multiplier", "tp_multiplier"),
                    ("sl_multiplier", "sl_multiplier"),
                    ("time_limit_candles", "time_limit_candles"),
                    ("model", "model"),
                    ("feature_set", "feature_set"),
                ]:
                    if sel.get(k_src) is not None:
                        ov_t[k_dst] = sel.get(k_src)
                # threshold recomendado (si no lo pasaste ya)
                if ov_t.get("inference_threshold") in (None, "", 0):
                    thr = sel.get("recommended_threshold")
                    if thr is not None:
                        try:
                            ov_t["inference_threshold"] = float(thr)
                        except Exception:
                            pass

            p = train_ticker(
                t,
                tf_t,
                overrides=ov_t,
                force_relabel=bool(args.force_relabel),
                clean=bool(args.clean),
                print_stats=bool(args.print_stats) or True,
            )
            print(f"[{t}] OK → {p}")
            ok += 1
        except Exception as e:
            print(f"[{t}] ERROR entrenando: {e}")
            fail += 1

    print(f"Resumen entreno: OK={ok} | FAIL={fail}")


if __name__ == "__main__":
    main()
