# LIV_PaperWorker.py — worker.sh con Parquet, MLflow (vía entrenos externos), TSCV thresholds y logging estructurado
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import mlflow
import nest_asyncio
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from ib_insync import IB
from mlflow import MlflowClient

from settings import S
from utils.A_data_loader import load_data
from utils.B_feature_engineering import add_technical_indicators, load_context_data
from utils.io_utils import atomic_write_csv
from utils.logging_cfg import get_logger, shutdown_logging
from utils.parquet_store import migrate_csv_folder_to_parquet
from utils.schemas import SignalsSchema, TradesSchema, validate_df
from utils.time_utils import (
    get_session_bounds,
    minutes_until_close,
    within_close_buffer,
)

# Opcional: silenciar barra de progreso de artifacts
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
CONFIG_DIR = Path(S.config_path)

# =========================
# Selección por-ticker (JSON)
# =========================
SEL_BY_TICKER_PATH = CONFIG_DIR / "optimizer_selected_by_ticker.json"
SEL_BY_TICKER: dict = {}
try:
    if SEL_BY_TICKER_PATH.exists():
        obj = json.loads(SEL_BY_TICKER_PATH.read_text(encoding="utf-8"))
        SEL_BY_TICKER = obj.get("per_ticker", {})
except Exception as e:
    print(f"⚠️ No pude leer optimizer_selected_by_ticker.json: {e}")


def get_selected_for(ticker: str) -> dict | None:
    return SEL_BY_TICKER.get(ticker.upper())


# =========================
# MLflow: cargar el modelo ganador (best_run_<TICKER>.txt) + threshold del run
# con fallback a pipeline.pkl local y a thresholds de CV/JSON
# =========================
_MODEL_CACHE: dict[str, tuple[object, float]] = {}  # {TICKER: (modelo, threshold)}
_LAST_THR: dict[tuple[str, str], float] = {}  # (TICKER, TIMEFRAME) -> last_thr_loggado


def _load_best_run_id(ticker: str) -> str:
    path = CONFIG_DIR / f"best_run_{ticker.upper()}.txt"
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def _load_model_and_thr_from_run(run_id: str, default_thr: float = 0.80):
    client = MlflowClient()
    run = client.get_run(run_id)
    uri = run.data.tags.get("logged_model_uri") or f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(uri)
    thr = run.data.params.get("inference_threshold") or run.data.params.get("recommended_threshold")
    thr = float(thr) if thr is not None else default_thr
    return model, thr


def _pipeline_dir(ticker: str) -> Path:
    return Path(S.models_path) / ticker.upper()


def _pipeline_pkl(ticker: str) -> Path:
    return _pipeline_dir(ticker) / "pipeline.pkl"


def _pipeline_meta(ticker: str) -> Path:
    return _pipeline_dir(ticker) / "pipeline_meta.json"


def _cv_json(ticker: str, timeframe: str) -> Path:
    return Path(S.cv_dir) / f"{ticker.upper()}_{timeframe}_cv.json"


_CV_CACHE: dict[tuple[str, str], dict] = {}
_PIPE_CACHE: dict[str, dict] = {}


def load_cv_threshold_cached(ticker: str, timeframe: str) -> float | None:
    key = (ticker.upper(), timeframe)
    path = _cv_json(*key)
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    cached = _CV_CACHE.get(key)
    if cached and cached.get("mtime") == mtime:
        return cached.get("thr")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        thr = obj.get("recommended_threshold", None)
        thr = float(thr) if isinstance(thr, (int, float)) else None
    except Exception:
        thr = None
    _CV_CACHE[key] = {"mtime": mtime, "thr": thr}
    return thr


def _load_local_pipeline_and_thr(ticker: str, timeframe: str, default_thr: float = 0.80):
    """Carga 02_models/{ticker}/pipeline.pkl y threshold: JSON por-ticker → CV → default."""
    pkl = _pipeline_pkl(ticker)
    if not pkl.exists():
        raise FileNotFoundError(f"pipeline.pkl no encontrado para {ticker}")
    pipe = joblib.load(pkl)

    # 1) JSON por-ticker
    sel = get_selected_for(ticker)
    if sel and (sel.get("recommended_threshold") is not None):
        try:
            thr = float(sel.get("recommended_threshold"))
            return pipe, thr
        except Exception:
            pass

    # 2) CV por ticker/timeframe
    thr_cv = load_cv_threshold_cached(ticker, timeframe)
    if thr_cv is not None:
        return pipe, float(thr_cv)

    # 3) default
    return pipe, float(default_thr)


def get_model_and_threshold(ticker: str, timeframe: str, default_thr: float = 0.80):
    t = ticker.upper()
    if t in _MODEL_CACHE:
        return _MODEL_CACHE[t]
    # Orden: MLflow → local
    try:
        run_id = _load_best_run_id(t)
        model, thr = _load_model_and_thr_from_run(run_id, default_thr=default_thr)
    except Exception:
        model, thr = _load_local_pipeline_and_thr(t, timeframe, default_thr=default_thr)
    _MODEL_CACHE[t] = (model, thr)
    return model, thr


def load_pipeline_cached(ticker: str):
    pkl = _pipeline_pkl(ticker)
    if not pkl.exists():
        return None
    mtime = pkl.stat().st_mtime
    cached = _PIPE_CACHE.get(ticker)
    if cached and cached.get("mtime") == mtime:
        return cached["pipe"]
    pipe = joblib.load(pkl)

    # cache meta y up_idx
    up_idx = _resolve_up_index(pipe, ticker)
    meta = {}
    mp = _pipeline_meta(ticker)
    if mp.exists():
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    _PIPE_CACHE[ticker] = {"pipe": pipe, "mtime": mtime, "up_idx": up_idx, "meta": meta}
    return pipe


def get_meta_cached(ticker: str) -> dict:
    obj = _PIPE_CACHE.get(ticker)
    if obj is not None and "meta" in obj:
        return obj["meta"]
    mp = _pipeline_meta(ticker)
    meta = {}
    if mp.exists():
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    _PIPE_CACHE.setdefault(ticker, {}).update({"meta": meta})
    return meta


def get_up_index_cached(ticker: str) -> int:
    obj = _PIPE_CACHE.get(ticker)
    if obj is not None and "up_idx" in obj:
        return obj["up_idx"]
    pipe = load_pipeline_cached(ticker)
    if pipe is None:
        return 1
    up_idx = _resolve_up_index(pipe, ticker)
    _PIPE_CACHE.setdefault(ticker, {}).update({"up_idx": up_idx})
    return up_idx


def _resolve_up_index(pipe, ticker: str) -> int:
    """
    Resuelve el índice de clase "UP" para predict_proba:
      - Si multiclase: usa class_order/meta para decidir (up = máx clase).
      - Si binario: up = clase 1 (o la mayor si no existe 1).
    """
    up_class = 1
    try:
        meta_p = _pipeline_meta(ticker)
        if meta_p.exists():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            mode = str(meta.get("label_map_mode", "binary_up_vs_rest")).lower()
            if mode == "multiclass_3way":
                order = meta.get("class_order", [0, 1, 2])
                up_class = max(order)
            else:
                up_class = 1
    except Exception:
        pass
    up_idx = None
    try:
        classes = getattr(pipe.named_steps["clf"], "classes_", None)
        if classes is not None:
            for i, c in enumerate(classes):
                try:
                    if int(c) == int(up_class):
                        up_idx = i
                        break
                except Exception:
                    continue
    except Exception:
        pass
    if up_idx is None:
        try:
            k = len(getattr(pipe.named_steps["clf"], "classes_", []))
            return 1 if k == 2 else (k - 1 if k > 0 else 1)
        except Exception:
            return 1
    return up_idx


def _clamp_threshold(x: float) -> float:
    return max(float(S.threshold_min), min(float(S.threshold_max), float(x)))


def get_threshold_for(ticker: str, timeframe: str, thr_from_run: float) -> float:
    """CV → JSON por-ticker → run → default, con clamp."""
    thr = None
    # 1) JSON por-ticker
    sel = get_selected_for(ticker)
    if sel and (sel.get("recommended_threshold") is not None):
        try:
            thr = float(sel.get("recommended_threshold"))
        except Exception:
            thr = None
    # 2) CV
    if thr is None and bool(getattr(S, "use_cv_threshold_first", True)):
        thr = load_cv_threshold_cached(ticker, timeframe)
    # 3) Run
    if thr is None:
        thr = thr_from_run if thr_from_run is not None else float(S.threshold_default)
    return _clamp_threshold(float(thr))


def get_threshold_for_verbose(ticker: str, timeframe: str, thr_from_run: float):
    """
    Igual que get_threshold_for, pero devuelve (thr, source) para poder loggear el origen.
    Orden: JSON por ticker -> CV -> run -> default.
    """
    source = None
    thr = None

    sel = get_selected_for(ticker)
    if sel and (sel.get("recommended_threshold") is not None):
        try:
            thr = float(sel.get("recommended_threshold"))
            source = "per_ticker_json"
        except Exception:
            thr = None

    if thr is None and bool(getattr(S, "use_cv_threshold_first", True)):
        thr_cv = load_cv_threshold_cached(ticker, timeframe)
        if thr_cv is not None:
            thr = float(thr_cv)
            source = "cv"

    if thr is None:
        if thr_from_run is not None:
            thr = float(thr_from_run)
            source = "run"
        else:
            thr = float(S.threshold_default)
            source = "default"

    return _clamp_threshold(float(thr)), source


def compute_atr14_abs(df: pd.DataFrame) -> pd.Series:
    z = df.copy().sort_values("date")
    high = z["high"].astype(float)
    low = z["low"].astype(float)
    close = z["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False).mean()


# =========================
# Parámetros de operación (global por defecto; por-ticker se aplican más abajo)
# =========================
OPERATING_TIMEFRAME = S.timeframe_default
TP_MULT_DEFAULT = S.tp_multiplier
SL_MULT_DEFAULT = S.sl_multiplier
TIME_LIMIT_CANDLES_DEFAULT = S.time_limit_candles

BAR_SIZE_SETTING = S.bar_size_by_tf.get(OPERATING_TIMEFRAME, "5 mins")
CANDLE_SIZE_MINUTES = int(S.candle_size_min_by_tf.get(OPERATING_TIMEFRAME, 5))

CAPITAL_PER_TRADE = float(S.capital_per_trade)
COMMISSION_PER_TRADE = float(S.commission_per_trade)

today_str = datetime.utcnow().strftime("%d_%m_%y")
LOGS_DIR = S.logs_path
LIVE_DATA_DIR = LOGS_DIR / "data_live"
CHARTS_DATA_DIR = LIVE_DATA_DIR / "live_charts_data"
MODELS_PATH = S.models_path
DATA_PATH = S.data_path
CV_DIR = S.cv_dir

TRADES_LOG_PATH = LOGS_DIR / f"paper_trades_{today_str}.csv"
OPEN_POSITIONS_LOG_PATH = LIVE_DATA_DIR / "live_open_positions.csv"
PROBS_LOG_PATH = LIVE_DATA_DIR / "live_probabilities.csv"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
CV_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Universo de tickers
# =========================
def get_tickers_from_file(file_path: Path) -> list[str]:
    if not file_path.exists():
        print(f"Error: Fichero de tickers no se encontró en {file_path}")
        return []
    with open(file_path, encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip() and line.strip().lower() != "ticker"]
    out, seen = [], set()
    for t in tickers:
        tu = t.upper()
        if tu not in seen:
            seen.add(tu)
            out.append(tu)
    return out


tickers_filepath = CONFIG_DIR / "top_100_robustos.txt"
# si quieres fallback a otro archivo, ajusta aquí:
# if not tickers_filepath.exists():
#     tickers_filepath = CONFIG_DIR / "top_100_robustos.txt"
TICKERS = get_tickers_from_file(tickers_filepath)

_nyse = mcal.get_calendar(S.calendar)


# =========================
# Daily Prep (downloader + parquet + training + CV)
# =========================
def run_daily_prep_once(logger):
    marker = LOGS_DIR / f"daily_prep_{datetime.utcnow():%Y%m%d}.ok"
    if marker.exists():
        print("🟢 Daily prep ya ejecutado hoy. Saltando.")
        return

    py = sys.executable
    project_root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + (os.pathsep + env.get("PYTHONPATH", ""))

    # Estados de paso (si el paso está desactivado, lo consideramos OK)
    downloader_ok = True
    training_ok = True
    cv_ok = True

    # 1) Downloader (solo si está activado)
    if bool(getattr(S, "downloader_on_start", False)):
        try:
            downloader_path = project_root / "utils" / "data_update.py"
            print("⬇️  Ejecutando downloader:", downloader_path)
            logger.info("download_start", event="download_start")
            subprocess.run([py, "-u", str(downloader_path)], check=True, cwd=project_root, env=env)
            print("✅ Downloader completado.")
            logger.info("download_end", event="download_end")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Downloader terminó con error: {e}")
            logger.error("download_error", event="download_error", exc_info=True)
            downloader_ok = False
    else:
        print("↓ Downloader desactivado por settings.downloader_on_start=False")

    # 1.5) CSV → Parquet (opcional; no bloquea el marker)
    if bool(getattr(S, "parquet_enabled", True)):
        try:
            csv_root = Path(S.data_path) / OPERATING_TIMEFRAME
            print(f"🗂️  Sincronizando CSV→Parquet en {csv_root} …")
            migrate_csv_folder_to_parquet(
                csv_root, timeframe=OPERATING_TIMEFRAME, glob_pat="*.csv", ticker_from_name=True
            )
            print("✅ CSV→Parquet sincronizado.")
        except Exception as e:
            print(f"ℹ️ CSV→Parquet omitido ({e})")

    # 2) Training batch (solo si está activado)
    if bool(getattr(S, "training_on_start", False)):
        try:
            training_path = project_root / "TRN_Train.py"
            training_cmd = [
                py,
                "-u",
                str(training_path),
                "--timeframe",
                OPERATING_TIMEFRAME,
                "--tickers_file",
                tickers_filepath.name,
            ]
            print("🧠 Ejecutando training batch:", " ".join(training_cmd))
            subprocess.run(training_cmd, check=True, cwd=project_root, env=env)
            print("✅ Training completado.")
            logger.info("pipeline_trained_batch", event="pipeline_trained_batch")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Training falló: {e}")
            logger.error("training_error", event="training_error", exc_info=True)
            training_ok = False
    else:
        print("↓ Training desactivado por settings.training_on_start=False")

    # Helper CV (usa cierres de py/project_root/env)
    def _run_cv_for(ticker: str) -> bool:
        cv_path = project_root / "RSH_TimeSeriesCV.py"
        args = [
            py,
            "-u",
            str(cv_path),
            "--ticker",
            ticker,
            "--timeframe",
            OPERATING_TIMEFRAME,
            "--n_splits",
            str(int(getattr(S, "n_splits_cv", 5))),
            "--test_size",
            str(int(getattr(S, "cv_test_size", 500))),
            "--scheme",
            str(getattr(S, "cv_scheme", "expanding")),
            "--embargo",
            str(int(getattr(S, "cv_embargo", -1))),
            "--purge",
            str(int(getattr(S, "cv_purge", -1))),
        ]
        try:
            subprocess.run(args, check=True, cwd=project_root, env=env)
            return True
        except subprocess.CalledProcessError as e:
            print(f"⚠️ CV falló para {ticker}: {e}")
            _lg = get_logger("cv", ticker=ticker, timeframe=OPERATING_TIMEFRAME)
            _lg.error("cv_error", event="cv_error", exc_info=True)
            return False

    # 3) CV por ticker (solo si está activado)
    if bool(getattr(S, "cv_update_on_start", False)):
        print("📊 Ejecutando CV por ticker para actualizar thresholds…")
        cv_ok = True
        for t in TICKERS:
            ok = _run_cv_for(t)
            cv_ok = cv_ok and ok
        logger.info("cv_done_batch", event="cv_done_batch", ok=cv_ok)
        print("✅ CV completada." if cv_ok else "⚠️ CV terminó con errores en algún ticker.")
    else:
        print("↓ CV desactivada por settings.cv_update_on_start=False")

    # Marker final (solo si lo crítico no falló)
    if downloader_ok and training_ok:
        try:
            marker.write_text("ok", encoding="utf-8")
            print("📌 Daily prep marcado como completado.")
        except Exception:
            pass
    else:
        print("❌ Daily prep NO completado, no se crea marker.")


# =========================
# Bucle principal
# =========================
def main():
    logger = get_logger(
        "worker.sh",
        session_id=f"wrk_{datetime.utcnow():%Y%m%d_%H%M}",
        run_mode="paper",  # o "live"
        timeframe=OPERATING_TIMEFRAME,
    )
    logger.info("worker_start", event="worker_start", universe=len(TICKERS))
    nest_asyncio.apply()
    ib = IB()
    open_positions: dict[str, dict] = {}
    closed_trades_df = pd.DataFrame()

    try:
        print("🔌 Conectando a Interactive Brokers...")
        ib.connect(S.ib_host, S.ib_port, clientId=int(time.time() % 1000))
        print("✅ Conexión exitosa.")

        run_daily_prep_once(logger)

        # Espera a apertura de mercado
        while True:
            now_utc = pd.Timestamp.now(tz="UTC")
            open_ts, close_ts = get_session_bounds(now_utc)
            is_trading_day = open_ts is not None and close_ts is not None
            if is_trading_day and (open_ts <= now_utc < close_ts):
                print(
                    f"✅ Mercado abierto ({open_ts.strftime('%H:%M')}–{close_ts.strftime('%H:%M')} UTC). Iniciando…"
                )
                break
            msg = (
                "festivo"
                if not is_trading_day
                else f"cierra a las {close_ts.strftime('%H:%M')} UTC"
            )
            print(f"⌛ Mercado cerrado ({msg}). Hora actual: {now_utc.strftime('%H:%M:%S')} UTC")
            time.sleep(60)

        while True:
            now_utc = pd.Timestamp.now(tz="UTC")
            open_ts, close_ts = get_session_bounds(now_utc)
            if (open_ts is None) or (close_ts is None) or (now_utc >= close_ts):
                print("🌙 Mercado cerrado. Finalizando operativa por hoy.")
                break

            cycle_start_time = time.monotonic()
            print(f"\n--- Ciclo {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC ---")

            # Contexto (VIX/SPY). Si falla, solo avisamos; NO usamos 'lg' aquí
            try:
                context_data = load_context_data(OPERATING_TIMEFRAME, DATA_PATH)
            except Exception as e:
                logger.info("context_missing", event="context_missing", reason=str(e))
                context_data = None

            all_probs = []
            for ticker in TICKERS:
                lg = logger.bind(ticker=ticker)

                # === Parámetros por ticker (desde JSON si existe) ===
                sel = get_selected_for(ticker) or {}
                timeframe_t = sel.get("timeframe") or OPERATING_TIMEFRAME
                tp_mul_t = float(sel.get("tp_multiplier", TP_MULT_DEFAULT))
                sl_mul_t = float(sel.get("sl_multiplier", SL_MULT_DEFAULT))
                tl_candles_t = int(sel.get("time_limit_candles", TIME_LIMIT_CANDLES_DEFAULT))
                candle_minutes_t = int(
                    S.candle_size_min_by_tf.get(timeframe_t, CANDLE_SIZE_MINUTES)
                )

                # 1) Modelo + threshold (MLflow → local) y umbral final (CV/JSON/run/default)
                try:
                    pipe, thr_run = get_model_and_threshold(
                        ticker, timeframe_t, default_thr=float(S.threshold_default)
                    )
                except Exception as e:
                    lg.warning("model_missing", event="model_missing", reason=str(e))
                    continue

                thr, thr_src = get_threshold_for_verbose(ticker, timeframe_t, thr_from_run=thr_run)

                prev = _LAST_THR.get((ticker, timeframe_t))
                if prev is None:
                    lg.info(
                        "threshold_set", threshold=float(thr), source=thr_src, timeframe=timeframe_t
                    )
                elif abs(prev - thr) > 1e-9:
                    lg.info(
                        "threshold_changed",
                        old=float(prev),
                        new=float(thr),
                        source=thr_src,
                        timeframe=timeframe_t,
                    )
                _LAST_THR[(ticker, timeframe_t)] = float(thr)

                try:
                    mlflow.set_tracking_uri(S.mlflow_tracking_uri)
                    mlflow.set_experiment(S.mlflow_experiment)
                    # run efímero por ticker/timeframe (nested para no molestar otros runs)
                    run_name = f"infer:{ticker}:{timeframe_t}"
                    with mlflow.start_run(
                        run_name=run_name, tags={"phase": "inference"}, nested=True
                    ):
                        mlflow.set_tags(
                            {
                                "ticker": ticker,
                                "timeframe": timeframe_t,
                                "inference_threshold_effective": float(thr),
                                "inference_threshold_source": str(thr_src),
                            }
                        )
                        # Si tienes a mano la URI del modelo cargado, añádela:
                        try:
                            model_uri = locals().get("logged_model_uri") or getattr(
                                pipe, "model_uri_", None
                            )
                            if model_uri:
                                mlflow.set_tag("model_uri_loaded", str(model_uri))
                        except Exception:
                            pass
                except Exception as e:
                    lg.warning("mlflow_tag_failed", error=str(e))

                # === Datos live + FEATURES exactamente como en training
                try:
                    df_live = load_data(
                        ticker=ticker,
                        timeframe=timeframe_t,
                        use_local=True,
                        base_path=Path(S.data_path),
                    )
                    if df_live is None or df_live.empty:
                        lg.warning("live_data_missing", event="live_data_missing")
                        continue

                    df_live = df_live.sort_values("date")

                    # 1) Misma ingeniería de features que en 04_model_training
                    try:
                        # 1) Misma ingeniería de features que en 04_model_training
                        df_live = add_technical_indicators(
                            df_live, context_data=context_data
                        )  # <--- CORREGIDO
                    except Exception as e:
                        lg.warning(
                            "fe_engineering_error", event="fe_engineering_error", reason=str(e)
                        )

                    # 2) Asegurar ATR14 (si no vino en los indicadores)
                    if "atr_14" not in df_live.columns:
                        try:
                            df_live["atr_14"] = compute_atr14_abs(df_live)
                        except Exception:
                            df_live["atr_14"] = np.nan  # el SimpleImputer del pipeline lo cubrirá

                    # 3) Alinear columnas a las del modelo entrenado
                    meta = get_meta_cached(ticker)
                    feat_cols = meta.get("features", None)
                    if not feat_cols:
                        # fallback por si el meta no existe (raro)
                        base_candidates = [
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "atr_14",
                            "rsi_14",
                            "ret_1",
                            "ma_5",
                            "ma_10",
                            "bb_up",
                            "bb_low",
                        ]
                        feat_cols = [c for c in base_candidates if c in df_live.columns]

                    # Crea columnas faltantes como NaN (las imputará el pipeline)
                    for c in feat_cols:
                        if c not in df_live.columns:
                            df_live[c] = np.nan

                    # 4) Fila de inferencia EXACTA y precio último
                    row = df_live.tail(1)[feat_cols].copy()
                    last_price = float(df_live["close"].iloc[-1])

                    # (opcional) diagnostico de columnas
                    # miss = [c for c in feat_cols if c not in df_live.columns]  # debería ser []
                    # lg.info("inference_features", used=len(feat_cols), missing=miss)

                except Exception as e:
                    lg.warning("live_data_error", event="live_data_error", reason=str(e))
                    continue

                # === Predicción de probabilidades con columnas correctas
                try:
                    proba = pipe.predict_proba(row)
                except Exception as e:
                    lg.error("predict_error", event="predict_error", reason=str(e))
                    continue

                proba = np.asarray(proba)
                k = proba.shape[1]

                if k == 2:
                    up_idx = get_up_index_cached(ticker)  # normalmente 1
                    p_up = float(proba[0, up_idx])
                    p_down = 1.0 - p_up
                    p_hold = 0.0
                else:
                    # multiclase: usa meta para localizar up/down; hold = 1 - (up+down)
                    up_idx = get_up_index_cached(ticker)
                    p_up = float(proba[0, up_idx])

                    meta = get_meta_cached(ticker)
                    order = meta.get("class_order", [0, 1, 2])
                    down_cls = min(order)
                    classes = getattr(pipe.named_steps["clf"], "classes_", None)
                    if classes is not None:
                        down_idx_list = [
                            i for i, c in enumerate(classes) if int(c) == int(down_cls)
                        ]
                        down_idx = down_idx_list[0] if down_idx_list else 0
                    else:
                        down_idx = 0
                    p_down = float(proba[0, down_idx])
                    p_hold = max(0.0, 1.0 - (p_up + p_down))

                long_ok = p_up >= thr
                short_ok = bool(getattr(S, "allow_short", True)) and (p_down >= thr)
                if long_ok and short_ok:
                    if bool(getattr(S, "prefer_stronger_side", True)):
                        margin_up = p_up - thr
                        margin_dn = p_down - thr
                        signal = 1 if margin_up >= margin_dn else -1
                    else:
                        signal = 1
                elif long_ok:
                    signal = 1
                elif short_ok:
                    signal = -1
                else:
                    signal = 0

                lg.info(
                    "signal",
                    event="signal",
                    prob_up=float(p_up),
                    prob_hold=float(p_hold),
                    prob_down=float(p_down),
                    threshold_used=float(thr),
                    chosen_signal=int(signal),
                    params=dict(tp=tp_mul_t, sl=sl_mul_t, tl=tl_candles_t, timeframe=timeframe_t),
                )

                all_probs.append(
                    {
                        "ticker": ticker,
                        "prob_down": float(p_down),
                        "prob_hold": float(p_hold),
                        "prob_up": float(p_up),
                        "signal": int(signal),  # -1/0/1
                        "threshold_used": float(thr),
                    }
                )

                # --- gestión de posiciones abiertas ---
                if ticker in open_positions:
                    pos = open_positions[ticker]
                    should_close, trigger = False, ""
                    if pos["signal"] == 1 and last_price >= pos["tp_price"]:
                        should_close, trigger = True, "Take Profit"
                    elif pos["signal"] == 1 and last_price <= pos["sl_price"]:
                        should_close, trigger = True, "Stop Loss"
                    elif now_utc.to_pydatetime() >= pos["time_limit"]:
                        should_close, trigger = True, "Time Limit"
                    if pos["signal"] == -1 and last_price <= pos["tp_price"]:
                        should_close, trigger = True, "Take Profit (short)"
                    elif pos["signal"] == -1 and last_price >= pos["sl_price"]:
                        should_close, trigger = True, "Stop Loss (short)"

                    if should_close:
                        pnl_gross = (
                            (last_price - pos["entry_price"]) * pos["quantity"] * pos["signal"]
                        )
                        total_commission = COMMISSION_PER_TRADE * 2
                        pnl_net = pnl_gross - total_commission
                        trade = pd.DataFrame(
                            [
                                {
                                    "entry_time": pos["entry_time"],
                                    "exit_time": now_utc.to_pydatetime(),
                                    "ticker": ticker,
                                    "signal": pos["signal"],
                                    "quantity": pos["quantity"],
                                    "entry_price": pos["entry_price"],
                                    "exit_price": last_price,
                                    "pnl": pnl_net,
                                    "exit_reason": trigger,
                                }
                            ]
                        )
                        trade = validate_df(trade, TradesSchema, name="trades(paper/single)")
                        closed_trades_df = pd.concat([closed_trades_df, trade], ignore_index=True)
                        open_positions.pop(ticker)
                        lg.info(
                            "position_closed",
                            event="position_closed",
                            exit_reason=trigger,
                            pnl_abs=float(pnl_net),
                        )

                # --- nuevas entradas ---
                safe_margin_minutes = tl_candles_t * candle_minutes_t
                last_trade_open_time = (
                    close_ts - timedelta(minutes=safe_margin_minutes)
                ).to_pydatetime()

                # Bloqueo adicional por buffer de cierre configurable (settings.market_close_buffer_min)
                in_close_buffer = within_close_buffer()
                if in_close_buffer:
                    lg.info(
                        "skip_entry_close_buffer",
                        event="skip_entry_close_buffer",
                        minutes_to_close=minutes_until_close(now_utc),
                    )

                can_open_new_trade = (now_utc.to_pydatetime() < last_trade_open_time) and (
                    not in_close_buffer
                )

                if ticker not in open_positions and signal != 0 and can_open_new_trade:
                    quantity = max(0, int(CAPITAL_PER_TRADE // last_price))
                    if quantity == 0:
                        lg.warning(
                            "order_rejected", event="order_rejected", reason="capital_too_small"
                        )
                        continue

                    atr_series = compute_atr14_abs(df_live)
                    atr_at_entry = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
                    if atr_at_entry <= 0:
                        lg.warning(
                            "order_rejected", event="order_rejected", reason="atr_non_positive"
                        )
                        continue

                    if signal == 1:
                        tp_price = last_price + (atr_at_entry * tp_mul_t)
                        sl_price = last_price - (atr_at_entry * sl_mul_t)
                        side_str = "BUY"
                    else:  # corto
                        tp_price = last_price - (atr_at_entry * tp_mul_t)
                        sl_price = last_price + (atr_at_entry * sl_mul_t)
                        side_str = "SELL_SHORT"

                    time_limit_exit = (
                        now_utc + timedelta(minutes=tl_candles_t * candle_minutes_t)
                    ).to_pydatetime()

                    open_positions[ticker] = {
                        "entry_time": now_utc.to_pydatetime(),
                        "ticker": ticker,
                        "entry_price": last_price,
                        "signal": int(signal),
                        "quantity": quantity,
                        "tp_price": tp_price,
                        "sl_price": sl_price,
                        "time_limit": time_limit_exit,
                    }
                    lg.info(
                        "order_sent",
                        event="order_sent",
                        side=side_str,
                        qty=int(quantity),
                        entry_price=float(last_price),
                        sl_price=float(sl_price),
                        tp_price=float(tp_price),
                    )

                # (Opcional) data para dashboard
                try:
                    atomic_write_csv(
                        df_live.tail(200), CHARTS_DATA_DIR / f"{ticker}_chart_data.csv", index=False
                    )
                except Exception:
                    pass

            # Persistencia de estado/logs CSV
            if open_positions:
                atomic_write_csv(
                    pd.DataFrame.from_dict(open_positions, orient="index"),
                    OPEN_POSITIONS_LOG_PATH,
                    index=True,
                )
            elif OPEN_POSITIONS_LOG_PATH.exists():
                OPEN_POSITIONS_LOG_PATH.unlink(missing_ok=True)

            if not closed_trades_df.empty:
                closed_trades_df = validate_df(
                    closed_trades_df, TradesSchema, name="trades(paper/batch)"
                )
                atomic_write_csv(closed_trades_df, TRADES_LOG_PATH, index=False)

            if all_probs:
                out = pd.DataFrame(all_probs)
                for c in ["prob_down", "prob_hold", "prob_up"]:
                    out[c] = out[c].astype(float).clip(0.0, 1.0)
                out["timestamp"] = pd.Timestamp.utcnow()
                out = out.drop_duplicates(subset=["ticker", "timestamp"], keep="last")
                out = validate_df(out, SignalsSchema, name="signals(live)")
                out = out.set_index("ticker")
                atomic_write_csv(out, PROBS_LOG_PATH, index=True)

            # Sincroniza con la duración de la vela global (asumiendo timeframe homogéneo)
            cycle_duration = time.monotonic() - cycle_start_time
            sleep_duration = (CANDLE_SIZE_MINUTES * 60) - cycle_duration
            print(
                f"Duración del ciclo: {cycle_duration:.2f}s. Esperando {max(0, sleep_duration):.2f}s…"
            )
            if sleep_duration > 0:
                time.sleep(sleep_duration)

    except Exception:
        logger.error("fatal_error", event="error", exc_info=True)
        print("🔥 ERROR FATAL: ver detalles en logs.")
    finally:
        if ib.isConnected():
            print("🔌 Desconectando de IBKR.")
            ib.disconnect()
        shutdown_logging()


if __name__ == "__main__":
    main()
