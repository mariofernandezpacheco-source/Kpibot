# engine/strategy.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

import settings as settings
from engine.events import MarketBar, Signal

S = settings.S

CONFIG_DIR = Path(S.config_path)
CV_DIR = Path(getattr(S, "cv_dir", S.logs_path / "cv"))
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))

# ---- Carga modelo+threshold (caché por ticker)
_MODEL_CACHE: dict[str, tuple[object, float]] = {}


def _best_run_id_for(ticker: str) -> str:
    p = CONFIG_DIR / f"best_run_{ticker.upper()}.txt"
    return p.read_text(encoding="utf-8").strip()


def _cv_thr_for(ticker: str, timeframe: str) -> float | None:
    p = CV_DIR / f"{ticker.upper()}_{timeframe}_cv.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        t = obj.get("recommended_threshold", None)
        return float(t) if t is not None else None
    except Exception:
        return None


def _load_model_from_run(run_id: str):
    c = MlflowClient()
    run = c.get_run(run_id)
    uri = run.data.tags.get("logged_model_uri") or f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(uri), run


def _get_model_and_thr(ticker: str, timeframe: str) -> tuple[object, float]:
    key = f"{ticker.upper()}::{timeframe}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    rid = _best_run_id_for(ticker)
    model, run = _load_model_from_run(rid)
    thr_run = run.data.params.get("inference_threshold") or run.data.params.get(
        "recommended_threshold"
    )
    thr_cv = _cv_thr_for(ticker, timeframe)
    thr = float(
        thr_cv if thr_cv is not None else (thr_run if thr_run is not None else S.threshold_default)
    )
    # clamp
    thr = max(float(S.threshold_min), min(float(S.threshold_max), thr))
    _MODEL_CACHE[key] = (model, thr)
    return _MODEL_CACHE[key]


def _probs(model, bar_df: pd.DataFrame) -> tuple[float, float, float]:
    """Devuelve (p_down, p_hold, p_up) soportando binario o multiclase."""
    try:
        proba = model.predict_proba(bar_df.set_index("date", drop=False))
    except Exception:
        proba = model.predict_proba(bar_df)
    proba = np.asarray(proba)
    k = proba.shape[1]
    if k == 2:
        p_up = float(proba[0, 1])
        p_down = 1.0 - p_up
        return p_down, 0.0, p_up
    # multiclase: asumimos clases ordenadas [down, hold, up]
    return float(proba[0, 0]), float(proba[0, 1]), float(proba[0, 2])


@dataclass
class StrategyConfig:
    timeframe: str = S.timeframe_default
    allow_short: bool = bool(getattr(S, "allow_short", True))
    prefer_stronger_side: bool = True


class MLFlowStrategy:
    """Estrategia que usa el modelo MLflow + threshold por ticker."""

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def on_bar(self, mb: MarketBar, has_open_position: bool) -> Signal | None:
        if has_open_position:
            return None  # una posición por ticker
        model, thr = _get_model_and_thr(mb.ticker, self.cfg.timeframe)
        # construir mini dataframe con la barra
        row = pd.DataFrame(
            [
                {
                    "date": pd.to_datetime(mb.ts, utc=True),
                    "open": mb.open,
                    "high": mb.high,
                    "low": mb.low,
                    "close": mb.close,
                    "volume": mb.volume,
                }
            ]
        )
        p_down, p_hold, p_up = _probs(model, row)

        long_ok = p_up >= thr
        short_ok = self.cfg.allow_short and (p_down >= thr)

        if long_ok and short_ok and self.cfg.prefer_stronger_side:
            # el margen sobre el umbral decide
            side = 1 if (p_up - thr) >= (p_down - thr) else -1
        elif long_ok:
            side = 1
        elif short_ok:
            side = -1
        else:
            return None

        return Signal(
            ticker=mb.ticker, ts=mb.ts, side=side, prob_up=p_up, prob_down=p_down, threshold=thr
        )
