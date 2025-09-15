# RSH_TimeSeriesCV.py — TSCV con selección robusta de threshold + MLflow + logging estructurado
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import json as _json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics

# NUEVO: features seleccionables y métricas de clasificación
from engine.features import apply_features  # aplica un registro de features seleccionadas
from engine.metrics import classification_metrics  # (no imprescindible aquí, pero útil si amplías)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# settings & rutas
import settings as settings

S = settings.S
DATA_DIR = Path(S.data_path)
MODELS_DIR = Path(S.models_path)
CONFIG_DIR = Path(S.config_path)
LOGS_DIR = Path(S.logs_path)
CV_DIR = Path(getattr(S, "cv_dir", LOGS_DIR / "cv"))
CV_DIR.mkdir(parents=True, exist_ok=True)

# ========= Trainer import robusto =========
# Intentamos módulos “importables” y legacy en orden.
_train_mod = None
for _cand in ("trn_train", "TRN_Train", "04_model_training"):
    try:
        _train_mod = importlib.import_module(_cand)
        break
    except ModuleNotFoundError:
        _train_mod = None

if _train_mod is None:
    # Intento por ruta (por si alguien mantiene nombres raros)
    for p in (ROOT, ROOT.parent):
        f = p / "[TRN]_Train.py"
        if f.exists():
            spec = importlib.util.spec_from_file_location("trn_train_bridge", f)
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            _train_mod = mod
            break

if _train_mod is None:
    raise ImportError(
        "No encuentro módulo de entrenamiento. Crea 'trn_train.py' (preferido) "
        "o conserva 'TRN_Train.py'/'04_model_training.py' con las mismas funciones."
    )

# Reutiliza piezas del trainer (exportadas)
ensure_atr14 = _train_mod.ensure_atr14
_feature_cols_for_b2 = _train_mod._feature_cols_for_b2
_map_labels = _train_mod._map_labels
_build_pipeline = _train_mod._build_pipeline
apply_feature_set = _train_mod.apply_feature_set

# Loader, etiquetas, validación y splitter
from utils.A_data_loader import load_data
from utils.C_label_generator import generate_triple_barrier_labels

# Logging estructurado
from utils.logging_cfg import get_logger

# MLflow (opcional): funciones finas del proyecto
from utils.mlflow_utils import end_run, log_metrics, log_params, start_run
from utils.schemas import LabelsSchema, OHLCVSchema, validate_df
from utils.tscv import PurgedWalkForwardSplit

# =========================
# Parámetros base (se pueden override por CLI)
# =========================
LABEL_WINDOW = int(getattr(S, "label_window", 5))
DAYS_HISTORY = int(getattr(S, "days_of_data", 90))
TP_MULT = float(getattr(S, "tp_multiplier", 3.0))
SL_MULT = float(getattr(S, "sl_multiplier", 2.0))
TIME_LIMIT_CANDLES = int(getattr(S, "time_limit_candles", 16))
LABEL_MAP_MODE = str(getattr(S, "label_map_mode", "binary_up_vs_rest")).lower()
MODEL_TYPE = str(getattr(S, "model_type", "xgb")).lower()
FEATURE_SET = "core"
HPARAMS = {}

N_SPLITS = int(getattr(S, "n_splits_cv", 5))
TEST_SIZE = int(getattr(S, "cv_test_size", 500))
SCHEME = str(getattr(S, "cv_scheme", "expanding"))
EMBARGO_DEFAULT = int(getattr(S, "cv_embargo", -1))
PURGE_DEFAULT = int(getattr(S, "cv_purge", -1))
THRESHOLDS = list(getattr(S, "cv_threshold_grid", [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]))

DATE_FROM = None  # "YYYY-MM-DD" opcional
DATE_TO = None  # "YYYY-MM-DD" opcional

# NUEVO: selección explícita de features (por defecto, vacío → usa FEATURE_SET legacy)
SELECTED_FEATURES: list[str] = []


# =========================
# Helpers
# =========================
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


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1 + (z**2) / n
    centre = p + (z**2) / (2 * n)
    adj = z * np.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)
    return max(0.0, (centre - adj) / denom)


def evaluate_thresholds_oof(
    y_true: np.ndarray, proba_up: np.ndarray, thresholds: list[float]
) -> dict:
    res = []
    for thr in thresholds:
        sel = proba_up >= thr
        n_tr = int(sel.sum())
        if n_tr == 0:
            res.append(
                {
                    "thr": float(thr),
                    "n_trades": 0,
                    "hit_rate": np.nan,
                    "ev": -SL_MULT,
                    "ev_lb": -SL_MULT,
                }
            )
            continue
        hits = int(((y_true == 1) & sel).sum())
        hr = hits / n_tr
        ev = hr * TP_MULT - (1 - hr) * SL_MULT
        hr_lb = wilson_lower_bound(hits, n_tr, 1.96)
        ev_lb = hr_lb * TP_MULT - (1 - hr_lb) * SL_MULT
        res.append(
            {
                "thr": float(thr),
                "n_trades": n_tr,
                "hits": hits,
                "hit_rate": float(hr),
                "hit_rate_lb": float(hr_lb),
                "ev": float(ev),
                "ev_lb": float(ev_lb),
            }
        )
    best = max(res, key=lambda d: (d["ev_lb"], d["n_trades"], d["thr"])) if res else None
    return {"oof_thresholds": res, "best_oof_by_ev_lb": best}


def _nanmean(xs):
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _nanstd(xs):
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    return float(np.std(xs, ddof=1)) if xs else float("nan")

def _ensure_mlflow_uri():
    import mlflow
    from pathlib import Path
    # Si no viene configurado, forzamos ./mlruns relativo al repo
    uri = getattr(S, "mlflow_tracking_uri", None)
    if not uri or uri.startswith("file:"):
        mlflow.set_tracking_uri((Path(__file__).resolve().parents[1] / "mlruns").resolve().as_uri())
    else:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(getattr(S, "mlflow_experiment", "PHIBOT"))

def _ks_statistic(y_true_bin: np.ndarray, proba: np.ndarray) -> float:
    """KS = max(TPR - FPR) a partir de la curva ROC (sin SciPy)."""
    try:
        from sklearn import metrics as _skm

        fpr, tpr, _ = _skm.roc_curve(y_true_bin, proba)
        return float(np.max(tpr - fpr))
    except Exception:
        return float("nan")


def _metrics_at_threshold(
    y_true_bin: np.ndarray, proba: np.ndarray, thr: float, tp_mult: float, sl_mult: float
) -> dict:
    from sklearn import metrics as _skm

    y_hat = (proba >= float(thr)).astype(int)
    tn, fp, fn, tp = _skm.confusion_matrix(y_true_bin, y_hat, labels=[0, 1]).ravel()
    out = {}
    try:
        out["accuracy_at_thr"] = float(_skm.accuracy_score(y_true_bin, y_hat))
    except Exception:
        pass
    try:
        out["precision_at_thr"] = float(_skm.precision_score(y_true_bin, y_hat, zero_division=0))
    except Exception:
        pass
    try:
        out["recall_at_thr"] = float(_skm.recall_score(y_true_bin, y_hat, zero_division=0))
    except Exception:
        pass
    try:
        out["f1_at_thr"] = float(_skm.f1_score(y_true_bin, y_hat, zero_division=0))
    except Exception:
        pass
    try:
        out["balanced_acc_at_thr"] = float(_skm.balanced_accuracy_score(y_true_bin, y_hat))
    except Exception:
        pass
    out["tn"] = int(tn)
    out["fp"] = int(fp)
    out["fn"] = int(fn)
    out["tp"] = int(tp)
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    out["specificity_at_thr"] = float(specificity)
    n_tr = int((proba >= float(thr)).sum())
    out["n_trades_at_thr"] = n_tr
    hit_rate = (tp / n_tr) if n_tr > 0 else float("nan")
    out["hit_rate_at_thr"] = float(hit_rate) if not np.isnan(hit_rate) else float("nan")
    ev = (
        hit_rate * tp_mult - (1.0 - hit_rate) * sl_mult
        if n_tr > 0 and not np.isnan(hit_rate)
        else -sl_mult
    )
    out["ev_at_thr"] = float(ev) if not np.isnan(ev) else float("nan")
    return out


def _prepare_dataset(ticker: str, timeframe: str) -> pd.DataFrame:
    df = load_data(ticker=ticker, timeframe=timeframe, use_local=True, base_path=DATA_DIR)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = _recent_slice(df, DAYS_HISTORY)
    df = _slice_by_dates(df, DATE_FROM, DATE_TO)
    df = validate_df(df, OHLCVSchema, name="OHLCV(cv/input)")
    df = ensure_atr14(df)

    label_kwargs = {}
    if "label_window" in generate_triple_barrier_labels.__code__.co_varnames:
        label_kwargs["label_window"] = LABEL_WINDOW

    df = generate_triple_barrier_labels(
        data=df,
        volatility_col=str(getattr(S, "volatility_col", "atr_14")),
        tp_multiplier=TP_MULT,
        sl_multiplier=SL_MULT,
        time_limit_candles=TIME_LIMIT_CANDLES,
        **label_kwargs,
    )
    df = validate_df(df, LabelsSchema, name="labels(cv)")
    return df.reset_index(drop=True)


# =========================
# Core CV
# =========================
def run_cv(
    ticker: str,
    timeframe: str,
    n_splits: int,
    test_size: int,
    scheme: str,
    embargo: int | None,
    purge: int | None,
    thresholds: list[float],
) -> dict:
    logger = get_logger("cv", ticker=ticker, timeframe=timeframe)
    logger.info("cv_start", event="cv_start", n_splits=n_splits, test_size=test_size, scheme=scheme)

    run = start_run(
        run_name=f"cv_{ticker}_{timeframe}",
        tags={"phase": "cv", "ticker": ticker, "timeframe": timeframe},
        nested=True,
    )
    # log params de experimento/overrides
    base_params = {
        "n_splits": n_splits,
        "test_size": test_size,
        "scheme": scheme,
        "embargo": -1 if embargo is None else embargo,
        "purge": -1 if purge is None else purge,
        "threshold_grid": thresholds,
        "tp_multiplier": TP_MULT,
        "sl_multiplier": SL_MULT,
        "label_window": LABEL_WINDOW,
        "time_limit_candles": TIME_LIMIT_CANDLES,
        "model_type": MODEL_TYPE,
        "label_map_mode": LABEL_MAP_MODE,
        "feature_set": FEATURE_SET,
        "days_history": DAYS_HISTORY,
        "date_from": DATE_FROM or "",
        "date_to": DATE_TO or "",
    }
    if SELECTED_FEATURES:
        base_params["selected_features"] = ",".join(SELECTED_FEATURES)
    log_params(base_params)
    if HPARAMS:
        log_params({f"hparam_{k}": v for k, v in HPARAMS.items()})

    df = _prepare_dataset(ticker, timeframe)

    # ===== FEATURES =====
    if SELECTED_FEATURES:
        # aplica features seleccionadas sobre OHLCV y toma solo las nuevas columnas
        cols_before = set(df.columns)
        df = apply_features(df, SELECTED_FEATURES)
        feat_cols = [c for c in df.columns if c not in cols_before]
        if not feat_cols:
            raise ValueError("No se generaron columnas de features a partir de 'SELECTED_FEATURES'.")
        X_all = df[feat_cols].copy()
    else:
        # flujo legacy: columnas del trainer + subset por FEATURE_SET
        feat_cols = _feature_cols_for_b2(df)
        X_all = df[[c for c in feat_cols if c in df.columns]].copy()
        X_all = apply_feature_set(X_all, FEATURE_SET)

    y_raw = df["label"].astype(int)
    y, mapping_dict, class_order = _map_labels(y_raw, LABEL_MAP_MODE)
    if y.nunique() < 2:
        end_run()
        logger.error("cv_error", event="cv_error", msg="Solo una clase tras mapeo")
        raise ValueError("Tras el mapeo de etiquetas solo hay una clase; ajusta parámetros.")

    if embargo is None or embargo < 0:
        embargo = int(TIME_LIMIT_CANDLES)
    if purge is None or purge < 0:
        purge = int(max(TIME_LIMIT_CANDLES, LABEL_WINDOW))

    splitter = PurgedWalkForwardSplit(
        n_splits=n_splits,
        test_size=test_size,
        train_min_size=max(test_size, 10 * LABEL_WINDOW),
        scheme=scheme,
        embargo=int(embargo),
        purge=int(purge),
    )

    folds, oof_idx, oof_proba_up, oof_y = [], [], [], []
    for i, (tr_idx, te_idx) in enumerate(splitter.split(X_all)):
        X_tr, y_tr = X_all.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X_all.iloc[te_idx], y.iloc[te_idx]

        pipe = _build_pipeline(
            feature_cols=list(X_tr.columns), y=y_tr, model_type=MODEL_TYPE, hparams=HPARAMS
        )

        pipe.fit(X_tr, y_tr)

        if len(np.unique(y_tr)) == 2:
            proba_up = pipe.predict_proba(X_te)[:, 1]
            up_idx = 1
        else:
            proba = pipe.predict_proba(X_te)
            up_idx = 2  # clase mapeada a "up" en _map_labels (0:-1,1:0,2:1)
            proba_up = proba[:, up_idx]

        y_te_bin = (y_te == (1 if len(np.unique(y)) == 2 else up_idx)).astype(int).values
        try:
            roc_auc = float(metrics.roc_auc_score(y_te_bin, proba_up))
            precision, recall, _ = metrics.precision_recall_curve(y_te_bin, proba_up)
            pr_auc = float(metrics.auc(recall, precision))
        except Exception:
            roc_auc, pr_auc = float("nan"), float("nan")

        fold_metrics = []
        for thr in thresholds:
            sel = proba_up >= thr
            n_tr = int(sel.sum())
            if n_tr == 0:
                fold_metrics.append({"thr": float(thr), "n_trades": 0, "ev": -SL_MULT})
                continue
            hits = int(((y_te_bin == 1) & sel).sum())
            ev = (hits / n_tr) * TP_MULT - (1 - hits / n_tr) * SL_MULT
            fold_metrics.append({"thr": float(thr), "n_trades": n_tr, "ev": float(ev)})

        folds.append(
            {
                "fold": i,
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "thresholds": fold_metrics,
            }
        )

        oof_idx.append(te_idx)
        oof_proba_up.append(proba_up)
        oof_y.append(y_te_bin)

    order = np.argsort(np.concatenate(oof_idx))
    proba_oof = np.concatenate(oof_proba_up)[order]
    y_oof = np.concatenate(oof_y)[order]

    oof_eval = evaluate_thresholds_oof(y_oof, proba_oof, thresholds)

    fold_roc = [f.get("roc_auc") for f in folds]
    fold_pr = [f.get("pr_auc") for f in folds]
    metrics_mean = {"roc_auc": _nanmean(fold_roc), "pr_auc": _nanmean(fold_pr)}
    metrics_std = {"roc_auc": _nanstd(fold_roc), "pr_auc": _nanstd(fold_pr)}

    best = oof_eval.get("best_oof_by_ev_lb")
    recommended = None
    reason = None
    if best:
        ntr = int(best["n_trades"])
        folds_ok = sum(
            1
            for f in folds
            if any(
                d["thr"] == best["thr"]
                and d["n_trades"] >= int(getattr(S, "cv_min_trades_per_fold", 5))
                for d in f["thresholds"]
            )
        )
        if ntr >= int(getattr(S, "cv_min_total_trades", 100)) and folds_ok >= int(
            getattr(S, "cv_min_folds_covered", 3)
        ):
            recommended = float(best["thr"])
            reason = "oof_ev_lower_bound_with_coverage"
        else:
            per_fold_best = []
            for f in folds:
                good = [
                    d
                    for d in f["thresholds"]
                    if d["n_trades"] >= int(getattr(S, "cv_min_trades_per_fold", 20))
                ]
                if good:
                    per_fold_best.append(max(good, key=lambda d: (d["ev"], d["n_trades"]))["thr"])
            if len(per_fold_best) >= int(getattr(S, "cv_min_folds_covered", 3)):
                recommended = float(np.median(per_fold_best))
                reason = "median_of_fold_best_ev_with_coverage"

    ev_at_rec_per_fold = []
    rec_thr = (
        recommended
        if recommended is not None
        else (best["thr"] if best and best.get("thr") is not None else None)
    )
    if rec_thr is not None:
        for f in folds:
            ev_entry = next((d for d in f["thresholds"] if abs(d["thr"] - rec_thr) < 1e-12), None)
            if ev_entry is not None and ev_entry.get("n_trades", 0) > 0:
                ev_at_rec_per_fold.append(float(ev_entry["ev"]))
    ev_stats = {
        "per_fold": ev_at_rec_per_fold,
        "mean": _nanmean(ev_at_rec_per_fold),
        "std": _nanstd(ev_at_rec_per_fold),
        "folds_covered": int(len(ev_at_rec_per_fold)),
    }

    # === OOF-level metrics (agregadas) ===
    try:
        oof_roc_auc = float(metrics.roc_auc_score(y_oof, proba_oof))
    except Exception:
        oof_roc_auc = float("nan")
    try:
        _prec, _rec, _ = metrics.precision_recall_curve(y_oof, proba_oof)
        oof_pr_auc = float(metrics.auc(_rec, _prec))
    except Exception:
        oof_pr_auc = float("nan")
    try:
        oof_log_loss = float(metrics.log_loss(y_oof, proba_oof, labels=[0, 1]))
    except Exception:
        oof_log_loss = float("nan")
    try:
        from sklearn.metrics import brier_score_loss as _brier

        oof_brier = float(_brier(y_oof, proba_oof))
    except Exception:
        oof_brier = float("nan")
    try:
        oof_ks = _ks_statistic(y_oof, proba_oof)
    except Exception:
        oof_ks = float("nan")

    # Métricas a umbral recomendado (si lo hay)
    oof_thr_metrics = {}
    if rec_thr is not None:
        oof_thr_metrics = _metrics_at_threshold(y_oof, proba_oof, float(rec_thr), TP_MULT, SL_MULT)

    # Log en MLflow: medias por fold + OOF + EV/coverage
    _to_log = {}
    if metrics_mean:
        _to_log.update({f"mean_{k}": v for k, v in metrics_mean.items() if v is not None})
    if metrics_std:
        _to_log.update({f"std_{k}": v for k, v in metrics_std.items() if v is not None})
    # EV a umbral recomendado + folds cubiertos
    if ev_stats.get("mean") is not None:
        _to_log["ev_mean_at_thr"] = float(ev_stats.get("mean"))
    if ev_stats.get("folds_covered") is not None:
        _to_log["folds_covered"] = int(ev_stats.get("folds_covered"))
    # OOF agregadas
    for k, v in {
        "oof_roc_auc": oof_roc_auc,
        "oof_pr_auc": oof_pr_auc,
        "oof_log_loss": oof_log_loss,
        "oof_brier": oof_brier,
        "oof_ks": oof_ks,
    }.items():
        if v is not None and not np.isnan(v):
            _to_log[k] = float(v)
    # OOF @ threshold recomendado
    for k, v in oof_thr_metrics.items():
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            _to_log[f"oof_{k}"] = float(v) if isinstance(v, (int, float, np.floating)) else v
    if _to_log:
        log_metrics(_to_log)

    if recommended is not None:
        log_params({"recommended_threshold": recommended, "recommended_by": reason})

    result = {
        "ticker": ticker.upper(),
        "timeframe": timeframe,
        "n_splits": n_splits,
        "test_size": test_size,
        "scheme": scheme,
        "embargo": int(embargo),
        "purge": int(purge),
        "threshold_grid": thresholds,
        "folds": folds,
        "oof": {"n_samples": int(len(y_oof)), "oof_eval": oof_eval},
        "metrics_mean": metrics_mean,
        "metrics_std": metrics_std,
        "ev_at_recommended": ev_stats,
        "recommended_threshold": recommended,
        "recommended_by": reason,
    }

    end_run()

    logger.info(
        "cv_done",
        event="cv_done",
        roc_auc_mean=float(metrics_mean["roc_auc"]),
        pr_auc_mean=float(metrics_mean["pr_auc"]),
    )

    return result


def _save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _parse_thresholds(s: str) -> list[float]:
    s = s.strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [float(x) for x in obj]
    except Exception:
        pass
    # CSV
    return [float(x) for x in s.split(",")]


def main():
    # Usaremos UN SOLO parser (ap) y aquí decidimos si usar MLflow o no.
    global TP_MULT, SL_MULT, TIME_LIMIT_CANDLES, MODEL_TYPE, FEATURE_SET, HPARAMS
    global DAYS_HISTORY, DATE_FROM, DATE_TO, THRESHOLDS, SELECTED_FEATURES
    # IMPORTANTE: NO llames a build_parser() aquí. Ese era el origen del error.

    ap = argparse.ArgumentParser(
        description="Time Series CV con purge+embargo y selección robusta de threshold."
    )
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--timeframe", default=str(getattr(S, "timeframe_default", "5mins")))
    ap.add_argument("--n_splits", type=int, default=N_SPLITS)
    ap.add_argument("--test_size", type=int, default=TEST_SIZE)
    ap.add_argument("--scheme", choices=["expanding", "rolling"], default=SCHEME)
    ap.add_argument("--embargo", type=int, default=EMBARGO_DEFAULT)
    ap.add_argument("--purge", type=int, default=PURGE_DEFAULT)

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
        help="subset de features (legacy)",
    )
    # NUEVO: lista de features del registry (CSV)
    ap.add_argument("--features", type=str, default=None,
                    help="Features seleccionadas (CSV) del registry de engine.features")
    ap.add_argument("--days", type=int, default=None, help="últimos N días")
    ap.add_argument("--date_from", type=str, default=None, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--date_to", type=str, default=None, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--thresholds", type=str, default=None,
                    help="Lista (JSON o CSV) de thresholds para la grid")

    # ✅ CLAVE: flag para desactivar completamente el logging a MLflow (modo “silencioso”)
    ap.add_argument("--no_mlflow", action="store_true",
                    help="Si se pasa, NO crea run en MLflow (solo genera el JSON de CV).")

    args = ap.parse_args()
    use_mlflow = not args.no_mlflow

    # Si NO queremos MLflow, “apagar” las funciones importadas de utils.mlflow_utils
    # re-vinculándolas a no-ops antes de llamar a run_cv()
    if not use_mlflow:
        # Estas asignaciones sustituyen las funciones importadas arriba del módulo
        # para que run_cv() no loguee nada.
        from types import SimpleNamespace
        global start_run, log_params, log_metrics, end_run
        def _noop(*a, **k): return None
        start_run = lambda *a, **k: SimpleNamespace(info="mlflow_disabled")  # devuelve algo “truthy”
        log_params = _noop
        log_metrics = _noop
        end_run = _noop
    else:
        # Solo configurar MLflow si se usa
        _ensure_mlflow_uri()

    # ------ Overrides globales con CLI / optimizer_selected.json ------
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

            # label & operativa
            _set_if_missing("tp", "tp_multiplier")
            _set_if_missing("sl", "sl_multiplier")
            _set_if_missing("time_limit", "time_limit_candles")

            # periodo
            _set_if_missing("days", "days")
            _set_if_missing("date_from", "date_from")
            _set_if_missing("date_to", "date_to")

            # modelo/feats
            _set_if_missing("model", "model")
            _set_if_missing("feature_set", "feature_set")
            _set_if_missing("hparams", "hparams", dump_json=True)

            # grid de thresholds
            if getattr(args, "thresholds", None) in (None, ""):
                thr = _sel.get("thresholds")
                if thr:
                    args.thresholds = json.dumps(thr)
    except Exception as _e:
        print(f"⚠️ No se pudo leer optimizer_selected.json: {_e}")

    # ------ Aplicar overrides a variables globales ------
    if args.tp is not None:
        TP_MULT = float(args.tp)
    if args.sl is not None:
        SL_MULT = float(args.sl)
    if args.time_limit is not None:
        TIME_LIMIT_CANDLES = int(args.time_limit)
    if args.model is not None:
        MODEL_TYPE = args.model
    if args.feature_set is not None:
        FEATURE_SET = args.feature_set
    if args.features:
        SELECTED_FEATURES = [s.strip() for s in args.features.split(",") if s.strip()]
    else:
        SELECTED_FEATURES = []
    if args.hparams:
        try:
            HPARAMS = json.loads(args.hparams)
        except Exception:
            HPARAMS = {}
    if args.days is not None:
        DAYS_HISTORY = int(args.days)
    if args.date_from:
        DATE_FROM = args.date_from
    if args.date_to:
        DATE_TO = args.date_to
    if args.thresholds:
        THRESHOLDS = _parse_thresholds(args.thresholds)

    emb = None if args.embargo < 0 else args.embargo
    pur = None if args.purge < 0 else args.purge

    # ------ Ejecutar la CV ------
    try:
        res = run_cv(
            ticker=args.ticker.upper(),
            timeframe=args.timeframe,
            n_splits=args.n_splits,
            test_size=args.test_size,
            scheme=args.scheme,
            embargo=emb,
            purge=pur,
            thresholds=THRESHOLDS,
        )
        out = CV_DIR / f"{args.ticker.upper()}_{args.timeframe}_cv.json"
        _save_json(res, out)
        msg = f"✅ CV guardada en {out}"
        if res.get("recommended_threshold") is not None:
            msg += f" | threshold recomendado: {res['recommended_threshold']:.2f} ({res.get('recommended_by')})"
        print(msg)
    except Exception:
        logger = get_logger("cv", ticker=args.ticker.upper(), timeframe=args.timeframe)
        logger.error("cv_error", event="cv_error", exc_info=True)
        raise


if __name__ == "__main__":
    main()
