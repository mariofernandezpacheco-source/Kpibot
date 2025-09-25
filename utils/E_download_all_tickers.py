# data_update.py — parametrizado + descarga SPY/VIX

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Config y utilidades del proyecto
from settings import S
from utils.ib_connection import fetch_intraday


def _normalize_timeframe(tf: str) -> str:
    return str(tf).lower().replace(" ", "")


def _bar_size_for(tf: str) -> str:
    """BarSize string para IB a partir del timeframe ('5mins' -> '5 mins')."""
    return S.bar_size_by_tf.get(tf, "5 mins")


def _load_existing_csv(path: Path) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    if not path.exists():
        return pd.DataFrame(), None
    df = pd.read_csv(path)
    # 'date' puede venir como columna o índice; lo normalizamos
    if "date" not in df.columns and df.index.name in ["date", "timestamp", None]:
        df = df.reset_index()
    if "date" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        return df, df["date"].max()
    return pd.DataFrame(), None


def _update_single_ticker(ticker: str, tf: str, bar_size: str, data_folder: Path):
    ticker = ticker.upper().strip()
    csv_path = data_folder / f"{ticker}_{tf}.csv"

    df_old, last_date = _load_existing_csv(csv_path)

    if last_date is not None:
        now_utc = datetime.now(UTC)
        days_since = max(1, (now_utc - last_date).days + 1)
        start_days = days_since
        print(
            f"📅 {ticker}: última fecha {last_date.date()}, descargando últimos {start_days} días…"
        )
    else:
        start_days = 365
        print(f"📅 {ticker}: sin histórico previo. Descargando {start_days} días…")

    # Descargar desde IBKR
    df_new = fetch_intraday(ticker, days=start_days, bar_size=bar_size)
    if df_new is None or df_new.empty:
        print(f"⚠️ Sin datos nuevos para {ticker}")
        return

    # Normalización de columnas
    if df_new.index.name in ["date", "timestamp", None]:
        df_new = df_new.reset_index()
    if "date" not in df_new.columns and "timestamp" in df_new.columns:
        df_new.rename(columns={"timestamp": "date"}, inplace=True)
    if "date" not in df_new.columns:
        raise ValueError(
            f"❌ {ticker}: El DataFrame descargado no contiene columna 'date'. "
            f"Columnas: {df_new.columns.tolist()}"
        )

    df_new["date"] = pd.to_datetime(df_new["date"], utc=True, errors="coerce")
    df_new = df_new.dropna(subset=["date"])
    # Filtra solo posteriores a la última fecha conocida
    if last_date is not None:
        df_new = df_new[df_new["date"] > last_date]

    print(f"✅ {ticker}: filas nuevas tras filtro por fecha: {len(df_new)}")

    if df_new.empty:
        print(f"⚠️ {ticker}: no hay nuevas filas que añadir.")
        return

    # Combina, ordena y guarda
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined.drop_duplicates(subset="date", inplace=True)
    df_combined.sort_values("date", inplace=True)
    df_combined.to_csv(csv_path, index=False)
    print(f"💾 {ticker} actualizado: {len(df_combined)} filas totales → {csv_path.name}")


def main():
    # Parámetros desde config_
    tf = _normalize_timeframe(S.timeframe_default)  # '5mins' / '10mins'
    bar_size = _bar_size_for(tf)  # '5 mins' / '10 mins'
    data_folder = Path(S.data_path)
    config_folder = Path(S.config_path)

    tickers_file = config_folder / "sp500_tickers.txt"
    if not tickers_file.exists():
        print(f"❌ Archivo de tickers no encontrado: {tickers_file}")
        return

    with open(tickers_file) as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    # Aseguramos SPY y VIX en la lista
    base_universe = set(tickers)
    base_universe.update({"SPY", "VIX"})
    tickers_all = sorted(base_universe)

    print(f"📥 Actualizando datos ({bar_size}) para {len(tickers_all)} símbolos en {data_folder}…")
    data_folder.mkdir(parents=True, exist_ok=True)

    for ticker in tqdm(tickers_all, desc="Símbolos procesados"):
        try:
            _update_single_ticker(ticker, tf, bar_size, data_folder)
        except Exception as e:
            print(f"❌ Error al actualizar {ticker}: {e}")


if __name__ == "__main__":
    main()
