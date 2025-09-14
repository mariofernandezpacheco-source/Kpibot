# utils/B_feature_engineering.py

from pathlib import Path

import numpy as np
import pandas as pd
import ta

# Config (settings.py en la raíz)
from settings import S


# -----------------------------
# Helpers
# -----------------------------
def _normalize_timeframe(tf: str) -> str:
    """Normaliza el timeframe a formato '5mins'/'10mins' (sin espacios)."""
    return str(tf).lower().replace(" ", "")


def _safe_read_csv(path: Path) -> pd.DataFrame:
    """Lee CSV de forma segura con parseo de fecha y normalización a UTC."""
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"El archivo {path} no contiene columna 'date'.")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df


# -----------------------------
# Contexto (VIX, SPY)
# -----------------------------
def load_context_data(timeframe: str, data_path: Path | None = None) -> dict:
    """
    Carga DataFrames de contexto (VIX, SPY) desde S.data_path (o data_path si se pasa).
    Devuelve {'vix': df_vix, 'spy': df_spy} con columnas ['date', '<ticker>_close'].
    Si faltan, retorna el dict sin esa clave.
    """
    tf = _normalize_timeframe(timeframe)
    base = Path(data_path) if data_path is not None else S.data_path

    context_data: dict[str, pd.DataFrame] = {}

    # VIX
    try:
        vix_file = base / "VIX.csv"
        vix_df = _safe_read_csv(vix_file)
        context_data["vix"] = vix_df[["date", "close"]].rename(columns={"close": "vix_close"})
        print("✅ Datos del VIX cargados.")
    except FileNotFoundError:
        print("⚠️ No se encontraron datos del VIX. Continuando sin VIX.")
    except Exception as e:
        print(f"⚠️ Error leyendo VIX: {e}. Continuando sin VIX.")

    # SPY
    try:
        spy_file = base / "SPY.csv"
        spy_df = _safe_read_csv(spy_file)
        context_data["spy"] = spy_df[["date", "close"]].rename(columns={"close": "spy_close"})
        print("✅ Datos del SPY cargados.")
    except FileNotFoundError:
        print("⚠️ No se encontraron datos del SPY. Continuando sin SPY.")
    except Exception as e:
        print(f"⚠️ Error leyendo SPY: {e}. Continuando sin SPY.")

    return context_data


# -----------------------------
# Ingeniería de características
# -----------------------------
def add_technical_indicators(df: pd.DataFrame, context_data: dict | None = None) -> pd.DataFrame:
    """
    Calcula indicadores técnicos + fusiona datos de contexto (VIX, SPY).
    - No hace look-ahead: después del merge, sólo ffill() en columnas de contexto.
    - Devuelve un DataFrame con nuevas columnas (RSI, MACD diff, ATR, Bollinger, etc.).
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # Orden temporal y tipos robustos
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    out.sort_values("date", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # === Indicadores (dataset propio) ===
    # Momentum
    out["rsi_14"] = ta.momentum.RSIIndicator(out["close"], window=14, fillna=True).rsi()
    stoch = ta.momentum.StochasticOscillator(
        out["high"], out["low"], out["close"], window=14, smooth_window=3, fillna=True
    )
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()
    out["willr_14"] = ta.momentum.WilliamsRIndicator(
        out["high"], out["low"], out["close"], lbp=14, fillna=True
    ).williams_r()

    # Tendencia
    macd = ta.trend.MACD(out["close"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    out["macd_diff"] = macd.macd_diff()
    out["trend_adx_14"] = ta.trend.ADXIndicator(
        out["high"], out["low"], out["close"], window=14, fillna=True
    ).adx()
    out["trend_cci_20"] = ta.trend.CCIIndicator(
        out["high"], out["low"], out["close"], window=20, constant=0.015, fillna=True
    ).cci()

    # Volatilidad
    out["atr_14"] = ta.volatility.AverageTrueRange(
        out["high"], out["low"], out["close"], window=14, fillna=True
    ).average_true_range()
    bb = ta.volatility.BollingerBands(out["close"], window=20, window_dev=2, fillna=True)
    out["bb_bb_high"] = bb.bollinger_hband()
    out["bb_bb_low"] = bb.bollinger_lband()

    # Volumen
    out["obv"] = ta.volume.OnBalanceVolumeIndicator(
        out["close"], out["volume"], fillna=True
    ).on_balance_volume()
    out["mfi_14"] = ta.volume.MFIIndicator(
        out["high"], out["low"], out["close"], out["volume"], window=14, fillna=True
    ).money_flow_index()
    out["volume_avg_20"] = out["volume"].rolling(window=20, min_periods=1).mean()

    # ROC
    out["roc_10"] = ta.momentum.ROCIndicator(out["close"], window=10, fillna=True).roc()

    # === Contexto (VIX, SPY) ===
    ctx = context_data or {}

    context_cols_to_ffill: list[str] = []

    vix_df = ctx.get("vix")
    if vix_df is not None and not vix_df.empty:
        out = pd.merge_asof(out, vix_df.sort_values("date"), on="date", direction="backward")
        out["vix_roc_5"] = out["vix_close"].pct_change(periods=5) * 100
        context_cols_to_ffill += ["vix_close", "vix_roc_5"]

    spy_df = ctx.get("spy")
    if spy_df is not None and not spy_df.empty:
        out = pd.merge_asof(out, spy_df.sort_values("date"), on="date", direction="backward")
        # RSI de SPY como contexto
        out["spy_rsi_14"] = ta.momentum.RSIIndicator(out["spy_close"], window=14, fillna=True).rsi()
        # fuerza relativa simple close% - spy%
        out["relative_strength"] = out["close"].pct_change(1) - out["spy_close"].pct_change(1)
        context_cols_to_ffill += ["spy_close", "spy_rsi_14", "relative_strength"]

    # Sin look-ahead: sólo ffill sobre columnas de contexto recién creadas
    if context_cols_to_ffill:
        existing = [c for c in context_cols_to_ffill if c in out.columns]
        out[existing] = out[existing].ffill()

    # Opcional: limpia columnas crudas de contexto si no quieres exponerlas
    out.drop(columns=["vix_close", "spy_close"], errors="ignore", inplace=True)

    # Limpieza final segura
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out = out.reset_index(drop=True)
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Devuelve la lista ORDENADA de columnas de features:
    - Excluye columnas no predictoras conocidas.
    - Excluye 'ticker' explícitamente.
    - Filtra SOLO columnas numéricas (number/bool) para que XGBoost no reciba 'object'.
    """
    exclude = {"date", "label", "open", "high", "low", "close", "volume", "index", "ticker"}
    # candidatos por exclusión
    cand = [c for c in df.columns if c not in exclude]
    # quedarnos solo con numéricas/bool (nada de object)
    numeric = df[cand].select_dtypes(include=["number", "bool"]).columns.tolist()
    return numeric
