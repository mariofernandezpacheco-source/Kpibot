# [RSH]_Scenarios.py — Optimización por ticker integrando 03_time_series_cv y MLflow (con manejo robusto)
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from engine.backtest_runner import run_backtest_for_ticker

# ----------------------------------------------------------------------
# Rutas base y settings
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import settings as settings

    S = settings.S
except Exception:
    print("ERROR: no se pudo importar settings.py. Asegúrate de que está junto a este script.")
    raise


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _save_best_params(ticker: str, timeframe: str, params: dict, models_path: Path) -> Path:
    model_dir = Path(models_path) / ticker.upper() / timeframe
    model_dir.mkdir(parents=True, exist_ok=True)
    out = model_dir / "best_params.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    return out


def read_lines_maybe_json_list(path: Path) -> list[str]:
    """
    Lee tickers desde un .txt (uno por línea) o una lista JSON en texto.
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de tickers: {path}")
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    # Si viene como lista JSON (["AAPL","MSFT"])
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            return [str(x).strip().upper() for x in obj if str(x).strip()]
    except Exception:
        pass
    # Si viene como líneas
    return [line.strip().upper() for line in txt.splitlines() if line.strip()]


def mlflow_setup():
    """
    Configura MLflow con URI ABSOLUTA (file:///C:/.../mlruns) para evitar
    desajustes entre procesos/UI y el script.
    """
    tracking_uri_cfg = getattr(S, "mlflow_tracking_uri", None)
    # Construimos la URI absoluta por defecto: <repo>/mlruns
    abs_uri = (ROOT / "mlruns").resolve().as_uri()

    try:
        if tracking_uri_cfg:
            # Si el usuario ya puso file:./mlruns u otra file:..., normalizamos a absoluta
            if tracking_uri_cfg.startswith("file:"):
                if tracking_uri_cfg in (
                    "file:./mlruns",
                    "file:mlruns",
                ) or tracking_uri_cfg.startswith("file:./"):
                    mlflow.set_tracking_uri(abs_uri)
                else:
                    mlflow.set_tracking_uri(tracking_uri_cfg)
            else:
                # Si puso una ruta no-file (ej. http://), respetamos
                mlflow.set_tracking_uri(tracking_uri_cfg)
        else:
            mlflow.set_tracking_uri(abs_uri)
    except Exception:
        # Fallback
        mlflow.set_tracking_uri(abs_uri)

    experiment = getattr(S, "mlflow_experiment", "PHIBOT")
    mlflow.set_experiment(experiment)

    # DEBUG visible
    try:
        print("MLflow tracking_uri =>", mlflow.get_tracking_uri())
        exp = mlflow.get_experiment_by_name(experiment)
        if exp:
            print(f"MLflow experiment   => name={exp.name} id={exp.experiment_id}")
    except Exception:
        pass


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def cv_json_path(ticker: str, timeframe: str) -> Path:
    """
    Localiza el JSON de resultados de la CV (lo genera [RSH]_TimeSeriesCV.py).
    """
    cv_dir = getattr(S, "cv_dir", (S.logs_path / "cv"))
    return Path(cv_dir) / f"{ticker.upper()}_{timeframe}_cv.json"


def run_cv_cli(ticker: str, timeframe: str, python_executable: str | None = None) -> dict:
    """
    Lanza [RSH]_TimeSeriesCV.py como proceso hijo para (re)generar la CV del ticker/timeframe
    y devuelve el dict cargado del JSON de salida.
    """
    py = python_executable or sys.executable
    script = ROOT / "[RSH]_TimeSeriesCV.py"
    if not script.exists():
        raise FileNotFoundError(f"No encuentro [RSH]_TimeSeriesCV.py en {script}")
    cmd = [py, str(script), "--ticker", ticker, "--timeframe", timeframe]
    print(f"→ Ejecutando CV: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    out_json = cv_json_path(ticker, timeframe)
    if not out_json.exists():
        raise FileNotFoundError(f"No se generó el JSON de CV esperado: {out_json}")
    with open(out_json, encoding="utf-8") as f:
        return json.load(f)


def _is_finite(x) -> bool:
    try:
        return (x is not None) and not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))
    except Exception:
        return False


def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        try:
            xf = float(str(x).strip())
            return xf
        except Exception:
            return float(default)


def _safe_int(x, default=None) -> int | None:
    try:
        if x is None:
            return default
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return default
        return int(x)
    except Exception:
        try:
            xf = float(x)
            if np.isnan(xf) or np.isinf(xf):
                return default
            return int(xf)
        except Exception:
            return default


def _log_metric_safe(name: str, value, step: int | None = None):
    if not _is_finite(value):
        return
    try:
        if step is None:
            mlflow.log_metric(name, float(value))
        else:
            mlflow.log_metric(name, float(value), step=step)
    except Exception:
        pass


def dataframe_from_thresholds(cv_res: dict) -> pd.DataFrame:
    """
    Convierte la sección 'oof_thresholds' (si existe) en DataFrame.
    Estructura esperada por 03_time_series_cv: [
      {'thr','n_trades','hits','hit_rate','hit_rate_lb','ev','ev_lb'}, ...
    ]
    """
    thr_list = cv_res.get("oof_thresholds") or cv_res.get("oof_thresholds_by_ev_lb") or []
    if not isinstance(thr_list, list):
        return pd.DataFrame()
    df = pd.DataFrame(thr_list)
    # Asegura tipos
    for col in ("thr", "n_trades", "hits", "hit_rate", "hit_rate_lb", "ev", "ev_lb"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@dataclass
class TickerSummary:
    ticker: str
    timeframe: str
    roc_auc_mean: float | None = None
    pr_auc_mean: float | None = None
    recommended_threshold: float | None = None
    recommended_by: str | None = None
    ev_lb_best: float | None = None
    n_trades_best: int | None = None


# ----------------------------------------------------------------------
# Núcleo: optimización por ticker (apoyado en CV) con logging a MLflow
# ----------------------------------------------------------------------
def optimize_one_ticker(
    ticker: str,
    timeframe: str,
    skip_cv: bool,
    parent_tags: dict | None = None,
) -> TickerSummary:
    """
    - Ejecuta (opcional) la CV del ticker/timeframe
    - Lee el JSON de CV
    - Lo registra en MLflow (run padre por ticker + run hijo para la CV)
    - Devuelve un resumen para agregación
    """
    mlflow_setup()

    # Run padre por ticker
    run_name_parent = f"{ticker.upper()}__{timeframe}__{datetime.now():%Y-%m-%d}"
    with mlflow.start_run(run_name=run_name_parent, nested=False):
        try:
            # Tags globales
            tags = dict(parent_tags or {})
            tags.update(
                {
                    "ticker": ticker.upper(),
                    "timeframe": timeframe,
                    "component": "optimizer",
                    "universe": "SP500",
                }
            )
            try:
                mlflow.set_tags(tags)
            except Exception:
                pass

            # --- DEBUG: identifica run padre y smoke test ---
            try:
                active = mlflow.active_run()
                if active:
                    print("MLflow parent run_id:", active.info.run_id)
                mlflow.log_param("smoke_parent", "ok")
                mlflow.log_metric("smoke_parent_metric", 1.0)
            except Exception as _e:
                print("WARN: no pude loguear en run padre:", _e)

            # Lanza/lee CV
            if not skip_cv:
                cv_res = run_cv_cli(ticker, timeframe)
            else:
                # Carga sin ejecutar
                with open(cv_json_path(ticker, timeframe), encoding="utf-8") as f:
                    cv_res = json.load(f)

            # Run hijo para la CV
            with mlflow.start_run(run_name="cv", nested=True):
                try:
                    # DEBUG del run hijo + smoke test
                    try:
                        active = mlflow.active_run()
                        if active:
                            print("MLflow child(run=cv) run_id:", active.info.run_id)
                        mlflow.log_param("smoke_child", "ok")
                        mlflow.log_metric("smoke_child_metric", 1.0)
                    except Exception as _e:
                        print("WARN: no pude loguear en run hijo:", _e)

                    # Params relevantes de CV y estrategia (para rastreo)
                    mlflow.log_params(
                        {
                            "ticker": ticker.upper(),
                            "timeframe": timeframe,
                            "n_splits_cv": getattr(S, "n_splits_cv", None),
                            "cv_test_size": getattr(S, "cv_test_size", None),
                            "cv_scheme": getattr(S, "cv_scheme", None),
                            "cv_embargo": getattr(S, "cv_embargo", None),
                            "cv_purge": getattr(S, "cv_purge", None),
                            "cv_threshold_grid": ",".join(
                                map(str, getattr(S, "cv_threshold_grid", []))
                            ),
                            "label_window": getattr(S, "label_window", None),
                            "time_limit_candles": getattr(S, "time_limit_candles", None),
                            "tp_multiplier": getattr(S, "tp_multiplier", None),
                            "sl_multiplier": getattr(S, "sl_multiplier", None),
                            "allow_short": getattr(S, "allow_short", None),
                            "prefer_stronger_side": getattr(S, "prefer_stronger_side", None),
                        }
                    )

                    # Métricas agregadas de la CV
                    roc = _safe_float(cv_res.get("roc_auc_mean"))
                    prc = _safe_float(cv_res.get("pr_auc_mean"))
                    rec_thr = _safe_float(cv_res.get("recommended_threshold"))
                    rec_by = str(cv_res.get("recommended_by") or "")

                    # guardamos recommended_threshold también como param
                    mlflow.log_params(
                        {
                            "recommended_threshold": None if np.isnan(rec_thr) else rec_thr,
                            "recommended_by": rec_by,
                        }
                    )
                    _log_metric_safe("roc_auc_mean", roc)
                    _log_metric_safe("pr_auc_mean", prc)
                    _log_metric_safe("recommended_threshold", rec_thr)

                    # Curva de thresholds → artefacto CSV + métricas por step
                    thr_df = dataframe_from_thresholds(cv_res)
                    if not thr_df.empty:
                        for idx, row in thr_df.iterrows():
                            _log_metric_safe("thr_ev_lb", row.get("ev_lb"), step=idx)
                            _log_metric_safe("thr_n_trades", row.get("n_trades"), step=idx)
                        thr_csv = ROOT / "threshold_curve.csv"
                        thr_df.to_csv(thr_csv, index=False)
                        mlflow.log_artifact(str(thr_csv))

                    # Guarda el JSON de CV como artefacto
                    cv_json_out = ROOT / "cv_result.json"
                    with open(cv_json_out, "w", encoding="utf-8") as f:
                        json.dump(cv_res, f, ensure_ascii=False, indent=2)
                    mlflow.log_artifact(str(cv_json_out))

                except Exception as e:
                    print("WARN: fallo dentro del run hijo (cv):", e)
                    mlflow.set_tag("cv_error", str(e))

            # ----------------------------
            # Elegimos "mejor escenario"
            # ----------------------------
            # 1) Threshold recomendado (si no hay, cogemos el mejor por ev_lb)
            best = cv_res.get("best_oof_by_ev_lb") or {}
            rec_thr_safe = float(rec_thr) if _is_finite(rec_thr) else None

            if rec_thr_safe is None:
                # fallback a mejor por EV_lb de la curva
                thr_df = dataframe_from_thresholds(cv_res)
                if not thr_df.empty and "ev_lb" in thr_df.columns and "thr" in thr_df.columns:
                    thr_df = thr_df.sort_values("ev_lb", ascending=False)
                    top = thr_df.iloc[0]
                    rec_thr_safe = float(top["thr"]) if _is_finite(top["thr"]) else None
                    best = {
                        "ev_lb": float(top.get("ev_lb", np.nan)),
                        "n_trades": int(top.get("n_trades", 0))
                        if _is_finite(top.get("n_trades", np.nan))
                        else None,
                    }

            # 2) Construimos params mínimos del "mejor"
            best_params = {
                "ticker": ticker.upper(),
                "timeframe": timeframe,
                "threshold": rec_thr_safe
                if rec_thr_safe is not None
                else float(getattr(S, "threshold_default", 0.8)),
                # parámetros de backtest (puedes afinarlos desde config.yaml si quieres)
                "tp_pct": 0.005,  # 0.5%
                "sl_pct": 0.005,  # 0.5%
                "cooldown_bars": 0,
                "allow_short": getattr(S, "allow_short", True),
                "slippage_bps": 0.0,
                "capital_per_trade": float(getattr(S, "capital_per_trade", 1000.0)),
                "commission_per_trade": float(getattr(S, "commission_per_trade", 0.35)),
                # (opcional) añade aquí features/ventanas si tu CV las reporta
            }

            # 3) Guardamos best_params.json para enlazar TRN y LIV
            best_params_path = _save_best_params(ticker, timeframe, best_params, S.models_path)

            # 4) Backtest OOS simple con ese escenario (runner nuevo)
            bt = run_backtest_for_ticker(ticker, timeframe, best_params)
            eq, tr, m = bt.get("equity"), bt.get("trades"), bt.get("metrics", {})

            # 5) Artefactos y métricas del backtest en MLflow (run hijo)
            with mlflow.start_run(run_name="oos_backtest", nested=True):
                # params
                mlflow.log_params(
                    {
                        "bt_threshold": best_params["threshold"],
                        "bt_tp_pct": best_params["tp_pct"],
                        "bt_sl_pct": best_params["sl_pct"],
                        "bt_cooldown_bars": best_params["cooldown_bars"],
                        "bt_allow_short": best_params["allow_short"],
                        "bt_capital_per_trade": best_params["capital_per_trade"],
                        "bt_commission_per_trade": best_params["commission_per_trade"],
                    }
                )
                # metrics (solo si existen y son finitas)
                for k in (
                    "net_return",
                    "sharpe",
                    "max_drawdown",
                    "win_rate",
                    "profit_factor",
                    "n_trades",
                ):
                    v = m.get(k)
                    if v is not None and not (
                        isinstance(v, float) and (np.isnan(v) or np.isinf(v))
                    ):
                        mlflow.log_metric(k, float(v))

                # artefactos
                outdir = (
                    S.logs_path
                    / "backtests"
                    / f"{ticker.upper()}_{timeframe}_{datetime.now():%Y%m%d-%H%M%S}"
                )
                outdir.mkdir(parents=True, exist_ok=True)
                # equity/trades
                if isinstance(eq, pd.Series) and not eq.empty:
                    eq.to_csv(outdir / "equity.csv")
                    mlflow.log_artifact(str(outdir / "equity.csv"))
                if isinstance(tr, pd.DataFrame) and not tr.empty:
                    tr.to_csv(outdir / "trades.csv", index=False)
                    mlflow.log_artifact(str(outdir / "trades.csv"))
                # metrics.json (comodidad)
                (outdir / "metrics.json").write_text(
                    json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                mlflow.log_artifact(str(outdir / "metrics.json"))
                # guardamos también los params elegidos
                mlflow.log_artifact(str(best_params_path))

            # 6) Resumen para este ticker
            summ = TickerSummary(
                ticker=ticker.upper(),
                timeframe=timeframe,
                roc_auc_mean=(float(roc) if _is_finite(roc) else None),
                pr_auc_mean=(float(prc) if _is_finite(prc) else None),
                recommended_threshold=(
                    float(best_params["threshold"])
                    if _is_finite(best_params["threshold"])
                    else None
                ),
                recommended_by=(rec_by or None)
                if rec_by
                else "ev_lb"
                if rec_thr_safe is None
                else "cv",
                ev_lb_best=_safe_float(best.get("ev_lb"), default=np.nan),
                n_trades_best=_safe_int(best.get("n_trades"), default=None),
            )

            # Marca estado ok
            mlflow.set_tag("optimizer_status", "ok")
            return summ

        except Exception as e:
            print("WARN: fallo dentro del run padre:", e)
            mlflow.set_tag("optimizer_error", str(e))
            # Devolvemos un resumen vacío para no romper el agregador
            return TickerSummary(ticker=ticker.upper(), timeframe=timeframe)


# ----------------------------------------------------------------------
# CLI & agregación
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="02c — Optimización por ticker con MLflow (apoyado en 03_time_series_cv)"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ticker", type=str, help="Ticker único (ej. AAPL)")
    g.add_argument(
        "--tickers-file", type=str, help="Ruta a .txt con 1 ticker por línea o lista JSON"
    )

    p.add_argument("--timeframe", type=str, default=getattr(S, "timeframe_default", "5mins"))
    p.add_argument(
        "--skip-cv",
        action="store_true",
        help="No relanzar [RSH]_TimeSeriesCV.py; reutiliza el JSON si existe",
    )
    p.add_argument(
        "--tags",
        type=str,
        default="",
        help='Tags adicionales para MLflow en JSON (ej. \'{"batch":"N1"}\')',
    )
    return p


def main():
    args = build_parser().parse_args()

    if args.ticker:
        tickers = [args.ticker.strip().upper()]
    else:
        tickers = read_lines_maybe_json_list(Path(args.tickers_file))

    timeframe = args.timeframe
    # Tags opcionales
    tags = {}
    if args.tags:
        try:
            tags = json.loads(args.tags)
            if not isinstance(tags, dict):
                tags = {}
        except Exception:
            tags = {}

    # Ejecuta por ticker y agrega
    summaries: list[TickerSummary] = []
    robust_counter = Counter()

    for tk in tickers:
        try:
            summ = optimize_one_ticker(tk, timeframe, skip_cv=args.skip_cv, parent_tags=tags)
            summaries.append(summ)

            # criterio de "robustez": EV_lb > 0 y nº trades >= umbral (configurable)
            ev_lb = summ.ev_lb_best if summ.ev_lb_best is not None else float("nan")
            n_tr = summ.n_trades_best if summ.n_trades_best is not None else 0
            min_total = getattr(S, "cv_min_total_trades", 100)
            if _is_finite(ev_lb) and (ev_lb > 0) and (n_tr >= min_total):
                robust_counter[tk] += 1

        except subprocess.CalledProcessError:
            print(f"⚠️  Falló la CV para {tk}. Revisa el log de [RSH]_TimeSeriesCV.py.")
        except Exception as e:
            print(f"⚠️  Error en optimizer para {tk}: {e}")

    # Guardar resumen global
    ensure_dir(S.logs_path)
    summary_path = S.logs_path / "optimizer_summary.csv"
    if summaries:
        df = pd.DataFrame([asdict(s) for s in summaries])
        df.to_csv(summary_path, index=False)
        print(f"\n✅ Resumen guardado en {summary_path}")
    else:
        df = pd.DataFrame(columns=[f.name for f in TickerSummary.__dataclass_fields__.values()])
        df.to_csv(summary_path, index=False)
        print(f"\n⚠️  No se generaron resúmenes. Archivo vacío en {summary_path}")

    # Top 100 robustos (si aplica)
    robust_path = S.logs_path / "top_100_robustos.txt"
    if robust_counter:
        top100 = [t for t, _ in robust_counter.most_common(100)]
        robust_path.write_text("\n".join(top100), encoding="utf-8")
        print(
            f"✅ Tickers robustos (criterio EV_lb>0 & n_trades≥{getattr(S,'cv_min_total_trades',100)}): {robust_path}"
        )

    # Pista para la UI con URI absoluta
    ui_uri = (ROOT / "mlruns").resolve().as_uri()
    print("\nListo. Abre la UI de MLflow para comparar por ticker:")
    print(f"    mlflow ui --backend-store-uri {ui_uri} -p 5000")


if __name__ == "__main__":
    main()
