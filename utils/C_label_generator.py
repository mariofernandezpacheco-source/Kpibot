# utils/C_label_generator.py
from __future__ import annotations

import numpy as np
import pandas as pd

# numba opcional (fallback a Python puro si no está)
try:
    import numba

    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False

# Config
from settings import S

# Schemas (Pandera)
from utils.schemas import LabelsSchema, OHLCVSchema, validate_df


# ==========================
# Helpers internos
# ==========================
def ensure_atr14(df: pd.DataFrame) -> pd.DataFrame:
    """Alias público para _compute_atr14"""
    if "atr_14" not in df.columns:
        df = df.copy()
        df["atr_14"] = _compute_atr14(df)
    return df

def _compute_atr14(df: pd.DataFrame, *, high="high", low="low", close="close") -> pd.Series:
    """
    ATR(14) determinista (Wilder ≈ EMA con alpha=1/14).
    """
    prev_close = df[close].shift(1)
    tr1 = (df[high] - df[low]).abs()
    tr2 = (df[high] - prev_close).abs()
    tr3 = (df[low] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False).mean()


def _validate_inputs(df: pd.DataFrame, volatility_col: str) -> pd.DataFrame:
    """
    Validación interna minimal para triple barrera.
    (OJO: aquí NO se pasa 'name', esa etiqueta es para validate_df de Pandera.)
    """
    required = ["high", "low", "close", volatility_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para triple barrera: {missing}")

    out = df.copy()
    # coerción numérica básica
    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=required)
    return out


# ==========================
# Núcleo de etiquetado
# ==========================
def _labels_python(
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    tp_levels: np.ndarray,
    sl_levels: np.ndarray,
    horizon_candles: int,
) -> np.ndarray:
    """
    Implementación en Python puro: etiqueta
      1 si toca TP antes, -1 si toca SL antes, 0 si nada (time limit).
    """
    n = len(high_prices)
    labels = np.zeros(n, dtype=np.int8)
    max_i = max(0, n - horizon_candles)

    for i in range(max_i):
        lab = 0
        for j in range(1, horizon_candles + 1):
            # TP primero
            if high_prices[i + j] >= tp_levels[i]:
                lab = 1
                break
            # SL primero
            if low_prices[i + j] <= sl_levels[i]:
                lab = -1
                break
        labels[i] = lab
    return labels


if _NUMBA_AVAILABLE:

    @numba.njit(cache=True)
    def _labels_numba(
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        tp_levels: np.ndarray,
        sl_levels: np.ndarray,
        horizon_candles: int,
    ) -> np.ndarray:
        n = len(high_prices)
        labels = np.zeros(n, dtype=np.int8)
        max_i = n - horizon_candles
        if max_i < 0:
            max_i = 0

        for i in range(max_i):
            lab = 0
            for j in range(1, horizon_candles + 1):
                if high_prices[i + j] >= tp_levels[i]:
                    lab = 1
                    break
                if low_prices[i + j] <= sl_levels[i]:
                    lab = -1
                    break
            labels[i] = lab
        return labels

else:
    _labels_numba = None  # type: ignore


def _compute_labels(
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    tp_levels: np.ndarray,
    sl_levels: np.ndarray,
    horizon_candles: int,
) -> np.ndarray:
    """
    Wrapper: usa numba si está disponible, si no, Python puro.
    """
    if _labels_numba is not None:
        return _labels_numba(high_prices, low_prices, tp_levels, sl_levels, horizon_candles)
    return _labels_python(high_prices, low_prices, tp_levels, sl_levels, horizon_candles)


# ==========================
# API principal
# ==========================
def generate_triple_barrier_labels(
    data: pd.DataFrame,
    *,
    volatility_col: str = "atr_14",
    tp_multiplier: float | None = None,
    sl_multiplier: float | None = None,
    time_limit_candles: int | None = None,
    label_window: int | None = None,
    compute_volatility_if_missing: bool = True,
) -> pd.DataFrame:
    """
    Genera etiquetas con Triple Barrera y devuelve el DF original + 'label' (int8 en {-1,0,1})

    Args:
        data: DataFrame OHLC(V) con al menos ['high','low','close'] y la columna 'volatility_col'.
        volatility_col: nombre de la columna de volatilidad (default 'atr_14').
        tp_multiplier: multiplicador de volatilidad para TP (default S.tp_multiplier).
        sl_multiplier: multiplicador de volatilidad para SL (default S.sl_multiplier).
        time_limit_candles: horizonte máximo en velas (default S.time_limit_candles).
        label_window: límite alternativo (se usa min con time_limit_candles si ambos >0).
        compute_volatility_if_missing: si True y volatility_col=='atr_14', la calcula si falta.

    Returns:
        DataFrame con columna 'label' en {-1,0,1} dtype=int8.
    """
    df = data.copy()

    # 0) Validación de entrada con Pandera (OHLCV). 'ticker' es opcional.
    df = validate_df(df, OHLCVSchema, name="OHLCV->labels(input)")

    # 1) Completar volatilidad si falta y procede
    if (
        volatility_col not in df.columns
        and compute_volatility_if_missing
        and volatility_col == "atr_14"
    ):
        # ordenar temporalmente para un ATR correcto
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
            df.sort_values("date", inplace=True)
        else:
            df.sort_index(inplace=True)
        df["atr_14"] = _compute_atr14(df)

    # 2) Validación interna mínima (sin 'name')
    df = _validate_inputs(df, volatility_col)

    # 3) Parámetros efectivos
    tp_mult = float(tp_multiplier if tp_multiplier is not None else S.tp_multiplier)
    sl_mult = float(sl_multiplier if sl_multiplier is not None else S.sl_multiplier)
    tlimit = int(time_limit_candles if time_limit_candles is not None else S.time_limit_candles)
    lwin = int(label_window if label_window is not None else S.label_window)

    horizon = tlimit if tlimit and tlimit > 0 else 0
    if lwin and lwin > 0:
        horizon = min(horizon, lwin) if horizon > 0 else lwin
    if horizon <= 0:
        raise ValueError("El horizonte de velas (time_limit/label_window) debe ser > 0.")

    # 4) Barreras
    df["tp_level"] = df["close"] + (df[volatility_col] * tp_mult)
    df["sl_level"] = df["close"] - (df[volatility_col] * sl_mult)

    # 5) A NumPy
    high_np = df["high"].to_numpy(dtype=np.float64, copy=False)
    low_np = df["low"].to_numpy(dtype=np.float64, copy=False)
    tp_np = df["tp_level"].to_numpy(dtype=np.float64, copy=False)
    sl_np = df["sl_level"].to_numpy(dtype=np.float64, copy=False)

    # 6) Calcular etiquetas
    labels_np = _compute_labels(high_np, low_np, tp_np, sl_np, horizon)
    df["label"] = labels_np.astype(np.int8, copy=False)

    # 7) Limpieza de columnas temporales
    df.drop(columns=["tp_level", "sl_level"], inplace=True, errors="ignore")

    # 8) Validación de salida con Pandera (Labels)
    df = validate_df(df, LabelsSchema, name="labels(output)")
    return df
