# RSH_Scenarios.py — Un run por combinación (ticker × modelo × thr × tp × sl × tl), con OOF de CV y métricas de backtest
from __future__ import annotations


import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import mlflow




# Paths / imports base
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Settings del proyecto
import settings as settings
S = settings.S

# Backtest + métricas
from engine.backtest_runner import run_backtest_for_ticker
from engine.metrics import trading_metrics


# --------------------------
# Utilidades
# --------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def as_abs_file_uri(path: Path) -> str:
    return path.resolve().as_uri()

def mlflow_setup():
    """Asegura una única URI consistente (file://.../mlruns) y el experimento."""
    tracking_uri_cfg = getattr(S, "mlflow_tracking_uri", None)
    default_abs_uri = (ROOT / "mlruns").resolve().as_uri()
    try:
        if tracking_uri_cfg:
            if tracking_uri_cfg.startswith("file:"):
                mlflow.set_tracking_uri(default_abs_uri)
            else:
                mlflow.set_tracking_uri(tracking_uri_cfg)
        else:
            mlflow.set_tracking_uri(default_abs_uri)
    except Exception:
        mlflow.set_tracking_uri(default_abs_uri)

    mlflow.set_experiment(getattr(S, "mlflow_experiment", "PHIBOT"))

def _is_finite(x) -> bool:
    try:
        if x is None:
            return False
        return np.isfinite(float(x))
    except Exception:
        return False

def _len_obj(x) -> int:
    if x is None:
        return 0
    if isinstance(x, (pd.DataFrame, pd.Series, list, tuple)):
        return int(len(x))
    try:
        return int(len(x))
    except Exception:
        return 0


def parse_csv_floats(s: Optional[str]) -> List[float]:
    if not s or not str(s).strip(): return []
    return [float(x) for x in str(s).split(",") if str(x).strip() != ""]

def parse_csv_ints(s: Optional[str]) -> List[int]:
    if not s or not str(s).strip(): return []
    return [int(x) for x in str(s).split(",") if str(x).strip() != ""]


# --------------------------
# CV “silenciosa” (sin MLflow)
# --------------------------
def run_cv_silent(
    ticker: str,
    timeframe: str,
    *,
    model: str,
    features_csv: Optional[str],
    feature_set: Optional[str],
    thresholds_csv: Optional[str],
    days: Optional[int],
    date_from: Optional[str],
    date_to: Optional[str],
    n_splits: Optional[int],
    test_size: Optional[int],
    scheme: Optional[str],
    embargo: Optional[int],
    purge: Optional[int],
    tp_mult: Optional[float],
    sl_mult: Optional[float],
    time_limit: Optional[int],
    python_exe: Optional[str] = None,
) -> dict:
    """
    Lanza RSH_TimeSeriesCV.py con --no_mlflow para obtener OOF (ROC/PR) y threshold recomendado,
    sin crear runs en MLflow. Devuelve el dict del JSON generado por la CV.
    """
    py = python_exe or sys.executable
    script = ROOT / "RSH_TimeSeriesCV.py"
    if not script.exists():
        raise FileNotFoundError(f"No encuentro RSH_TimeSeriesCV.py en {script}")

    args = [py, str(script),
            "--ticker", ticker, "--timeframe", timeframe,
            "--model", model, "--no_mlflow"]

    if features_csv:
        args += ["--features", features_csv]
    elif feature_set:
        args += ["--feature_set", feature_set]

    if thresholds_csv:
        args += ["--thresholds", thresholds_csv]
    if days is not None:
        args += ["--days", str(int(days))]
    if date_from:
        args += ["--date_from", date_from]
    if date_to:
        args += ["--date_to", date_to]
    if n_splits is not None:
        args += ["--n_splits", str(int(n_splits))]
    if test_size is not None:
        args += ["--test_size", str(int(test_size))]
    if scheme:
        args += ["--scheme", scheme]
    if embargo is not None:
        args += ["--embargo", str(int(embargo))]
    if purge is not None:
        args += ["--purge", str(int(purge))]
    if tp_mult is not None:
        args += ["--tp", str(float(tp_mult))]
    if sl_mult is not None:
        args += ["--sl", str(float(sl_mult))]
    if time_limit is not None:
        args += ["--time_limit", str(int(time_limit))]

    proc = subprocess.run(args, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"CV falló para {ticker}: {proc.stdout}\n{proc.stderr}")

    # Localiza el JSON que escribe la CV
    cv_dir = Path(getattr(S, "cv_dir", Path(S.logs_path) / "cv"))
    cv_json = cv_dir / f"{ticker.upper()}_{timeframe}_cv.json"
    if not cv_json.exists():
        raise FileNotFoundError(f"No se generó el JSON de CV: {cv_json}")
    return json.loads(cv_json.read_text(encoding="utf-8"))


# --------------------------
# Núcleo: un run por combinación
# --------------------------
def optimize_per_combo(
    ticker: str,
    timeframe: str,
    *,
    models: List[str],
    features_csv: Optional[str],
    feature_set: Optional[str],
    thresholds_csv: Optional[str],
    tp_grid_csv: Optional[str],
    sl_grid_csv: Optional[str],
    tl_grid_csv: Optional[str],
    # CV period & scheme
    days: Optional[int],
    date_from: Optional[str],
    date_to: Optional[str],
    n_splits: Optional[int],
    test_size: Optional[int],
    scheme: Optional[str],
    embargo: Optional[int],
    purge: Optional[int],
    # labeling params usados en la CV
    tp_mult: Optional[float],
    sl_mult: Optional[float],
    time_limit_lbl: Optional[int],
    # flags/UI
    iter_thr_in_bt: bool,
    primary_metric: str,
    min_trades: int,
) -> None:
    """
    Genera UN RUN por cada combinación (thr, tp, sl, tl) y por cada modelo seleccionado.
    Cada run incluye:
      - params: ticker, timeframe, fechas, modelo, features, thr/tp/sl/tl...
      - metrics: oof_roc_auc, oof_pr_auc (de la CV silenciosa), y métricas de backtest
      - artifacts: equity.csv, trades.csv, metrics.json
    """
    mlflow_setup()

    # 1) Ejecuta UNA SOLA CV “silenciosa” por cada modelo (para OOF y threshold recomendado)
    thresholds_list = parse_csv_floats(thresholds_csv) or getattr(S, "cv_threshold_grid", [0.55,0.6,0.65,0.7,0.75,0.8])
    cv_by_model: Dict[str, dict] = {}
    for model in models:
        try:
            cv_res = run_cv_silent(
                ticker, timeframe,
                model=model,
                features_csv=features_csv,
                feature_set=feature_set,
                thresholds_csv=",".join(str(x) for x in thresholds_list),
                days=days, date_from=date_from, date_to=date_to,
                n_splits=n_splits, test_size=test_size, scheme=scheme, embargo=embargo, purge=purge,
                tp_mult=tp_mult, sl_mult=sl_mult, time_limit=time_limit_lbl,
            )
        except Exception:
            cv_res = {"metrics_mean": {"roc_auc": float('nan'), "pr_auc": float('nan')}}
        cv_by_model[model] = cv_res

    # 2) Construye grids
    tp_grid = parse_csv_floats(tp_grid_csv) or [0.003, 0.005, 0.008]
    sl_grid = parse_csv_floats(sl_grid_csv) or [0.003, 0.005, 0.008]
    tl_grid = parse_csv_ints(tl_grid_csv) or [8, 12, 16]

    # 3) Para cada modelo y cada combinación, crea UN RUN
    for model in models:
        roc = float(cv_by_model[model].get("metrics_mean",{}).get("roc_auc", np.nan))
        prc = float(cv_by_model[model].get("metrics_mean",{}).get("pr_auc", np.nan))
        thr_rec = cv_by_model[model].get("recommended_threshold")
        thr_rec = float(thr_rec) if _is_finite(thr_rec) else None

        if iter_thr_in_bt:
            thr_iter = thresholds_list[:]  # itera todos los thresholds
        else:
            thr_iter = [thr_rec] if _is_finite(thr_rec) else ([thresholds_list[-1]] if thresholds_list else [0.8])

        for thr in thr_iter:
            thr_val = float(thr)

            for tp_ in tp_grid:
                for sl_ in sl_grid:
                    for tl_ in tl_grid:
                        # Ejecuta backtest para ESTA combinación
                        params_bt = {
                            "threshold": float(thr_val),
                            # TP/SL como porcentaje decimal (0.003 = 0.3%) y sinónimos
                            "tp_pct": float(tp_), "sl_pct": float(sl_),
                            "take_profit": float(tp_), "stop_loss": float(sl_),
                            "tp": float(tp_), "sl": float(sl_),
                            # Time-limit (barras) + sinónimos
                            "time_limit": int(tl_), "time_limit_bars": int(tl_), "bars_limit": int(tl_),
                            # Otras opciones comunes
                            "cooldown_bars": int(getattr(S, "bt_cooldown_bars", 0)),
                            "allow_short": bool(getattr(S, "allow_short", True)),
                            "slippage_bps": float(getattr(S, "slippage_bps", 0.0)),
                            "capital_per_trade": float(getattr(S, "capital_per_trade", 1000.0)),
                            "commission_per_trade": float(getattr(S, "commission_per_trade", 0.35)),
                            # Contexto por si tu runner los usa para señales
                            "model": model,
                            "features": features_csv or "",
                            "feature_set": feature_set or "",
                        }

                        def _len_obj(x) -> int:
                            if x is None: return 0
                            import pandas as pd  # seguro si no estaba importado en este scope
                            if isinstance(x, (pd.DataFrame, pd.Series)): return int(len(x))
                            if isinstance(x, (list, tuple)): return int(len(x))
                            try:
                                return int(len(x))
                            except Exception:
                                return 0

                        def _is_num_finite(x) -> bool:
                            import numpy as np
                            try:
                                xv = float(x)
                                return np.isfinite(xv)
                            except Exception:
                                return False

                        bt = {}
                        try:
                            bt = run_backtest_for_ticker(ticker, timeframe, params_bt)
                        except Exception:
                            bt = {}

                        # extrae equity/trades
                        eq = bt.get("equity") if isinstance(bt, dict) else None
                        tr = bt.get("trades") if isinstance(bt, dict) else None
                        n_trades_val = _len_obj(tr)
                        eq_len_val = _len_obj(eq)

                        # extrae métricas (o calcula si faltan)
                        metrics_bt = {}
                        for cand in (bt.get("metrics"), bt.get("stats"), bt.get("summary")) if isinstance(bt,
                                                                                                          dict) else []:
                            if isinstance(cand, dict) and cand:
                                metrics_bt = cand
                                break

                        if not any(k in metrics_bt for k in
                                   ("net_return", "sharpe", "max_drawdown", "win_rate", "profit_factor", "n_trades")):
                            try:
                                eq_arr = eq.values if isinstance(eq, pd.Series) else (
                                    np.asarray(eq) if eq is not None else None)
                                tr_list = tr.to_dict("records") if isinstance(tr, pd.DataFrame) else (
                                    tr if isinstance(tr, list) else None)
                                m_calc = trading_metrics(eq_arr, tr_list)
                                if isinstance(m_calc, dict):
                                    metrics_bt = {**metrics_bt, **m_calc}
                            except Exception:
                                pass

                        # ===== Un run MLflow (no anidado) para esta combinación =====
                        run_name = f"{ticker.upper()}__{timeframe}__{model}__thr{thr_val:.3f}_tp{float(tp_):.4f}_sl{float(sl_):.4f}_tl{int(tl_)}"
                        with mlflow.start_run(run_name=run_name, nested=False):
                            # Tags
                            mlflow.set_tags({
                                "component": "optimizer",
                                "phase": "scenarios_per_combo",
                                "ticker": ticker.upper(),
                                "timeframe": timeframe,
                                "model": model,
                                "features": features_csv or "",
                                "feature_set": feature_set or "",
                                "primary_metric": primary_metric,
                            })

                            # Params (incluye fechas/ventana/CV)
                            mlflow.log_params({
                                "threshold": float(thr_val),
                                "tp_pct": float(tp_), "sl_pct": float(sl_), "time_limit": int(tl_),
                                "date_from": date_from or "", "date_to": date_to or "",
                                "days": days if days is not None else "",
                                "n_splits": n_splits if n_splits is not None else "",
                                "test_size": test_size if test_size is not None else "",
                                "scheme": scheme or "", "embargo": embargo if embargo is not None else "",
                                "purge": purge if purge is not None else "",
                                "tp_multiplier_lbl": tp_mult if tp_mult is not None else "",
                                "sl_multiplier_lbl": sl_mult if sl_mult is not None else "",
                                "time_limit_lbl": time_limit_lbl if time_limit_lbl is not None else "",
                                "iter_thr_in_bt": bool(iter_thr_in_bt),
                            })

                            # --- SIEMPRE registrar métricas de “vida” para evitar “No metrics recorded” ---
                            mlflow.log_metric("smoke_metric", 1.0)
                            mlflow.log_metric("oof_roc_auc", float(roc) if _is_num_finite(roc) else -1.0)
                            mlflow.log_metric("oof_pr_auc", float(prc) if _is_num_finite(prc) else -1.0)
                            mlflow.log_metric("debug_trades_len", float(n_trades_val))
                            mlflow.log_metric("debug_equity_len", float(eq_len_val))
                            mlflow.log_metric("debug_has_metrics", 1.0 if bool(metrics_bt) else 0.0)

                            # Métricas de backtest (normalizadas)
                            mapping = {
                                "ret": "net_return", "return": "net_return", "netret": "net_return",
                                "pnl": "net_return",
                                "max_dd": "max_drawdown", "drawdown": "max_drawdown",
                                "ntrades": "n_trades", "trades": "n_trades",
                            }
                            for k, v in (metrics_bt.items() if isinstance(metrics_bt, dict) else []):
                                kk = mapping.get(str(k).lower(), str(k).lower())
                                if kk in {"net_return", "sharpe", "max_drawdown", "win_rate", "profit_factor",
                                          "n_trades"} and _is_num_finite(v):
                                    mlflow.log_metric(kk, float(v))

                            # Artefactos
                            outdir = Path("artifacts");
                            ensure_dir(outdir)
                            if hasattr(eq, "empty") and not eq.empty:
                                eq.to_csv(outdir / "equity.csv")
                                mlflow.log_artifact(str(outdir / "equity.csv"))
                            if hasattr(tr, "empty") and not tr.empty:
                                tr.to_csv(outdir / "trades.csv", index=False)
                                mlflow.log_artifact(str(outdir / "trades.csv"))

                            # metrics.json seguro
                            def _to_json_safe(d: dict) -> dict:
                                safe = {}
                                for k, v in (d.items() if isinstance(d, dict) else []):
                                    if hasattr(v, "item"):
                                        try:
                                            v = v.item()
                                        except Exception:
                                            pass
                                    if isinstance(v, (int, float, str, bool)):
                                        if isinstance(v, float) and not _is_num_finite(v):  # filtra NaN/Inf
                                            continue
                                        safe[k] = v
                                return safe

                            (outdir / "metrics.json").write_text(
                                json.dumps(_to_json_safe(metrics_bt), ensure_ascii=False, indent=2), encoding="utf-8")
                            mlflow.log_artifact(str(outdir / "metrics.json"))


# --------------------------
# CLI
# --------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Research/Scenarios — Un run por combinación (ticker × thr × tp × sl × tl)")
    p.add_argument("--ticker", required=True)
    p.add_argument("--timeframe", default=getattr(S, "timeframe_default", "5mins"))
    # grids / opciones
    p.add_argument("--models", default="xgb", help="CSV de modelos (xgb,rf,logreg)")
    p.add_argument("--features", default=None, help="CSV de features (engine.features); si se usa, ignora feature_set")
    p.add_argument("--feature_set", default="core", choices=["core","core+vol","all"], help="Usado si no pasas --features")
    p.add_argument("--thresholds", default=None, help="CSV de thresholds para CV y grid (ej. 0.6,0.7,0.8)")
    p.add_argument("--tp_grid", default="0.003,0.005,0.008")
    p.add_argument("--sl_grid", default="0.003,0.005,0.008")
    p.add_argument("--tl_grid", default="8,12,16")

    # CV ventana y esquema (para calcular OOF ROC/PR — no crea runs)
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--date_from", type=str, default=None)
    p.add_argument("--date_to", type=str, default=None)
    p.add_argument("--n_splits", type=int, default=getattr(S,"n_splits_cv",5))
    p.add_argument("--test_size", type=int, default=getattr(S,"cv_test_size",500))
    p.add_argument("--scheme", type=str, default=getattr(S,"cv_scheme","expanding"))
    p.add_argument("--embargo", type=int, default=getattr(S,"cv_embargo",-1))
    p.add_argument("--purge", type=int, default=getattr(S,"cv_purge",-1))
    p.add_argument(
        "--min_trades", type=int, default=0,
        help="Sólo para logging/filtrado en UI; no afecta al backtest."
    )

    # labeling defaults (triple barrier usados en CV)
    p.add_argument("--tp", type=float, default=None)
    p.add_argument("--sl", type=float, default=None)
    p.add_argument("--time_limit", type=int, default=None)

    # flags/UI
    p.add_argument("--iter_thr_in_bt", action="store_true", help="Si se pasa, iterar thresholds también en el backtest; si no, usar el recomendado.")
    p.add_argument("--primary_metric", default="net_return", choices=["net_return","sharpe","win_rate","profit_factor"])

    return p

def main():
    args = build_parser().parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    optimize_per_combo(
        ticker=args.ticker.upper(),
        timeframe=args.timeframe,
        models=models,
        features_csv=args.features,
        feature_set=(None if args.features else args.feature_set),
        thresholds_csv=args.thresholds,
        tp_grid_csv=args.tp_grid,
        sl_grid_csv=args.sl_grid,
        tl_grid_csv=args.tl_grid,
        days=args.days,
        date_from=args.date_from,
        date_to=args.date_to,
        n_splits=args.n_splits,
        test_size=args.test_size,
        scheme=args.scheme,
        embargo=(None if args.embargo is not None and args.embargo < 0 else args.embargo),
        purge=(None if args.purge is not None and args.purge < 0 else args.purge),
        tp_mult=args.tp,
        sl_mult=args.sl,
        time_limit_lbl=args.time_limit,
        iter_thr_in_bt=bool(args.iter_thr_in_bt),
        primary_metric=args.primary_metric,
        min_trades=args.min_trades,
    )
    print("\n✅ OK. Abre MLflow para ver los runs (uno por combinación):")
    print(f"    mlflow ui --backend-store-uri {as_abs_file_uri(ROOT / 'mlruns')} -p 5000")

if __name__ == "__main__":
    main()
