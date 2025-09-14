# utils/D_backtest_utils.py

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Config (settings.py en la raíz)
from settings import S
from utils.B_feature_engineering import add_technical_indicators


# ------------------------------------------------------------
# Carga de modelo entrenado (fallback simple para pruebas)
# ------------------------------------------------------------
def load_trained_model(filename: str = "intraday_model.pkl"):
    """
    Carga un modelo desde la carpeta de modelos configurada.
    Nota: si estás usando bundles (modelo+features) usa los loaders específicos.
    """
    model_path = Path(S.models_path) / filename
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado en: {model_path}. Entrena el modelo antes de hacer backtest.sh."
        )
    return joblib.load(model_path)


# ------------------------------------------------------------
# Simulación rápida (demo) basada en probs (no triple barrera)
# ------------------------------------------------------------
def simulate_trades(
    df: pd.DataFrame, model, window: int, threshold: float, capital: float, risk_pct: float
):
    """
    Simulación simple: entrena features y abre/cierra en la vela siguiente.
    Útil para pruebas rápidas. Para tu flujo actual usa backtest_signals.
    """
    df_feat = add_technical_indicators(df)
    X_all = df_feat.iloc[window:]
    X_arr = X_all.values

    # Si tu modelo espera un nº concreto de columnas, recorta/pad si fuese necesario
    try:
        expected = model.coef_.shape[1]  # para modelos lineales
        if X_arr.shape[1] != expected:
            X_arr = X_arr[:, :expected]
    except Exception:
        pass

    probs = model.predict_proba(X_arr)[:, 1]  # binario (ejemplo)

    prices = df["close"].iloc[window:-1].values
    next_prices = df["close"].iloc[window + 1 :].values
    timestamps = df.index[window:-1]

    trades = []
    for ts, p, price, nxt in tqdm(
        zip(timestamps, probs, prices, next_prices, strict=False),
        total=len(prices),
        desc="Simulating trades",
    ):
        if (p >= threshold) or (p <= (1 - threshold)):
            factor = p if p >= threshold else (1 - p)
            qty = int((capital * risk_pct * factor) / price)
            if qty <= 0:
                continue
            pnl = (nxt - price) * qty if p >= threshold else (price - nxt) * qty
            trades.append({"timestamp": ts, "pnl": pnl})

    return pd.DataFrame(trades)


# ------------------------------------------------------------
# Backtest de señales (compatible con tu app y worker.sh)
# ------------------------------------------------------------
def backtest_signals(
    df_signals: pd.DataFrame,
    capital_per_trade: float | None = None,
    commission_per_trade: float | None = None,
):
    """
    Ejecuta un backtest.sh realista a partir de señales + precios de cierre.

    Espera un DataFrame con:
      - 'timestamp' (datetime): marca temporal de la vela
      - 'close' (float): precio de cierre de la vela
      - 'signal' (int): {-1, 0, 1} (venta, nada, compra)

    Reglas:
      - Abre posición cuando signal != 0 y no hay posición abierta.
      - Cierra cuando aparece la señal contraria (flip). Entrada y salida al precio 'close' de la vela.
      - Tamaño = floor(capital_per_trade / entry_price).
      - Comisión fija por trade (se descuentan 2 comisiones por round-trip).
      - Expone eventos para calcular la exposición máxima de capital.

    Returns
    -------
    metrics : dict
        {'profit', 'num_trades', 'win_rate', 'sharpe_ratio', 'max_exposure'}
    trades_df : pd.DataFrame
        columnas: ['entry_time','exit_time','entry_price','exit_price','quantity','signal_type','pnl']
    events_df : pd.DataFrame
        columnas: ['timestamp','capital_change','capital_in_use']
    """
    # Defaults desde settings.yaml si no se pasan
    capital = float(capital_per_trade if capital_per_trade is not None else S.capital_per_trade)
    commission = float(
        commission_per_trade if commission_per_trade is not None else S.commission_per_trade
    )

    if df_signals is None or df_signals.empty:
        metrics = {
            "profit": 0.0,
            "num_trades": 0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "max_exposure": 0.0,
        }
        return metrics, pd.DataFrame(), pd.DataFrame()

    # Normalizaciones
    df = df_signals.copy()
    # Garantiza columnas
    required_cols = {"timestamp", "close", "signal"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"backtest_signals: faltan columnas requeridas: {missing}")

    # Tipos
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce").fillna(0).astype(int)

    # Orden temporal
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)

    trades = []
    position = 0  # -1 corto, 0 plano, 1 largo
    entry_price = None
    entry_time = None

    for _, row in df.iterrows():
        sig = int(row["signal"])
        price = float(row["close"])
        ts = row["timestamp"]

        is_closing_signal = position != 0 and sig == -position
        is_opening_signal = position == 0 and sig != 0

        # Cierre por señal opuesta
        if is_closing_signal:
            quantity = int(capital // entry_price)
            if quantity > 0:
                pnl_gross = (price - entry_price) * quantity * position
                total_commission = commission * 2  # entrada + salida
                pnl_net = pnl_gross - total_commission
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": ts,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "quantity": quantity,
                        "signal_type": position,
                        "pnl": pnl_net,
                    }
                )
            # Reset estado
            position = 0
            entry_price = None
            entry_time = None

        # Apertura si estamos planos y llega señal válida
        if is_opening_signal:
            position = sig
            entry_price = price
            entry_time = ts

    # Si al final queda algo abierto, NO lo cerramos (política conservadora).
    if not trades:
        metrics = {
            "profit": 0.0,
            "num_trades": 0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "max_exposure": 0.0,
        }
        return metrics, pd.DataFrame(), pd.DataFrame()

    trades_df = pd.DataFrame(trades)

    # --- Exposición máxima de capital vía eventos (aperturas/cierres) ---
    opens = trades_df[["entry_time"]].rename(columns={"entry_time": "timestamp"})
    opens["capital_change"] = capital

    closes = trades_df[["exit_time"]].rename(columns={"exit_time": "timestamp"})
    closes["capital_change"] = -capital

    events_df = pd.concat([opens, closes]).sort_values("timestamp").reset_index(drop=True)
    events_df["capital_in_use"] = events_df["capital_change"].cumsum()
    max_exposure = float(events_df["capital_in_use"].max() if not events_df.empty else 0.0)

    # --- Métricas ---
    total_pnl = float(trades_df["pnl"].sum())
    num_trades = int(len(trades_df))
    win_rate = float((trades_df["pnl"] > 0).mean() * 100) if num_trades > 0 else 0.0

    # Sharpe Ratio (agregando PnL por día de salida)
    daily_returns = trades_df.set_index("exit_time")["pnl"].resample("D").sum()
    if daily_returns.std() > 0 and len(daily_returns) > 1:
        sharpe_ratio = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))
    else:
        sharpe_ratio = 0.0

    metrics = {
        "profit": total_pnl,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_exposure": max_exposure,
    }

    return metrics, trades_df, events_df


# ------------------------------------------------------------
# Plot helpers (opcionales)
# ------------------------------------------------------------
def plot_pnl(df_trades: pd.DataFrame, ticker: str, days: int):
    """
    Visualización rápida de PnL (históricos y acumulados).
    """
    if df_trades is None or df_trades.empty:
        print("No hay trades para graficar.")
        return

    df = df_trades.copy().sort_values("exit_time").reset_index(drop=True)
    df["cumulative_pnl"] = df["pnl"].cumsum()

    plt.figure()
    plt.plot(df["exit_time"], df["cumulative_pnl"], label="Cumulative PnL")
    plt.title(f"Cumulative PnL for {ticker} ({days}d)")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.bar(df["exit_time"], df["pnl"], width=0.0005)
    plt.title(f"PnL per Trade for {ticker} ({days}d)")
    plt.xlabel("Time")
    plt.ylabel("PnL")
    plt.tight_layout()
    plt.show()
