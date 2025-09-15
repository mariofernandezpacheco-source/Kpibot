# engine/features.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _rolling_z(x, win):
    s = pd.Series(x)
    return (s - s.rolling(win, min_periods=win//2).mean()) / (s.rolling(win, min_periods=win//2).std(ddof=0) + 1e-12)

def sma(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df = df.copy()
    df[f"sma_{n}"] = df["close"].rolling(n, min_periods=n//2).mean()
    return df

def ema(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df = df.copy()
    df[f"ema_{n}"] = df["close"].ewm(span=n, adjust=False, min_periods=max(2, n//3)).mean()
    return df

def rsi(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["close"].diff()
    up = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / (down + 1e-12)
    df[f"rsi_{n}"] = 100 - (100 / (1 + rs))
    return df

def macd(df: pd.DataFrame, fast=12, slow=26, sig=9) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal = macd_line.ewm(span=sig, adjust=False).mean()
    df[f"macd_{fast}_{slow}"] = macd_line
    df[f"macd_signal_{sig}"] = signal
    df[f"macd_hist"] = macd_line - signal
    return df

def bbands(df: pd.DataFrame, n=20, k=2.0) -> pd.DataFrame:
    df = df.copy()
    ma = df["close"].rolling(n, min_periods=n//2).mean()
    sd = df["close"].rolling(n, min_periods=n//2).std(ddof=0)
    df[f"bb_up_{n}_{k}"] = ma + k*sd
    df[f"bb_dn_{n}_{k}"] = ma - k*sd
    df[f"bb_z_{n}_{k}"]  = (df["close"] - ma) / (sd + 1e-12)
    return df

def atr(df: pd.DataFrame, n=14) -> pd.DataFrame:
    df = df.copy()
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df[f"atr_{n}"] = tr.ewm(alpha=1/n, adjust=False).mean()
    return df

def garman_klass(df: pd.DataFrame, n=14) -> pd.DataFrame:
    df = df.copy()
    log_hl = np.log(df["high"]/df["low"])**2
    log_co = np.log(df["close"]/df["open"])**2
    gk = 0.5*log_hl - (2*np.log(2)-1)*log_co
    df[f"gk_vol_{n}"] = gk.rolling(n, min_periods=n//2).mean().pow(0.5)
    return df

def vwap_dev(df: pd.DataFrame, n=20) -> pd.DataFrame:
    df = df.copy()
    pv = df["close"] * df["volume"].clip(lower=0)
    v = df["volume"].clip(lower=0)
    vwap = pv.rolling(n, min_periods=n//2).sum() / (v.rolling(n, min_periods=n//2).sum() + 1e-12)
    df[f"vwap_dev_{n}"] = (df["close"] - vwap) / (vwap.abs() + 1e-12)
    return df

def vol_zscore(df: pd.DataFrame, n=20) -> pd.DataFrame:
    df = df.copy()
    df[f"vol_z_{n}"] = _rolling_z(df["volume"].astype(float).replace([np.inf,-np.inf], np.nan).fillna(0), n)
    return df

# ------- registro: nombre -> (callable, kwargs por defecto) -------
FEATURES = {
    # tendencia / momentum
    "sma_5":   (sma, dict(n=5)),
    "sma_20":  (sma, dict(n=20)),
    "ema_12":  (ema, dict(n=12)),
    "ema_26":  (ema, dict(n=26)),
    "rsi_14":  (rsi, dict(n=14)),
    "macd":    (macd, dict(fast=12, slow=26, sig=9)),
    "bb_20_2": (bbands, dict(n=20, k=2.0)),
    # volatilidad
    "atr_14":  (atr, dict(n=14)),
    "gk_14":   (garman_klass, dict(n=14)),
    # volumen
    "vwap_20": (vwap_dev, dict(n=20)),
    "volz_20": (vol_zscore, dict(n=20)),
}

def apply_features(df: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    out = df.copy()
    for key in selected:
        if key not in FEATURES:
            continue
        fn, kwargs = FEATURES[key]
        out = fn(out, **kwargs)
    # opcional: reemplazar inf/NaN
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.ffill().bfill()
        return out