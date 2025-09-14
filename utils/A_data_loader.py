# utils/A_data_loader.py

from pathlib import Path

import pandas as pd

# Config (settings.py en la raíz del proyecto)
from settings import S
from utils.io_utils import safe_read_csv
from utils.schemas import OHLCVSchema, validate_df


def _normalize_timeframe(tf: str) -> str:
    """Normaliza el timeframe a formato '5mins'/'10mins' (sin espacios)."""
    return str(tf).lower().replace(" ", "")


def load_local_data(ticker: str, timeframe: str, base_path: Path | None = None) -> pd.DataFrame:
    """
    Carga datos históricos desde un CSV local de forma robusta.
    - Normaliza 'date' a timezone UTC.
    - Ordena por fecha y resetea índice.
    - Limpia la columna 'index' si venía en el CSV.
    """
    ticker = str(ticker).upper().strip()
    timeframe = _normalize_timeframe(timeframe)
    base_path = Path(base_path) if base_path is not None else S.data_path

    filename = f"{ticker}.csv"
    filepath = base_path / timeframe / filename
    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

    try:
        df = safe_read_csv(filepath)
    except pd.errors.EmptyDataError:
        # CSV vacío → devolver DF vacío con columnas estándar mínimas
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    # Limpieza básica
    if "index" in df.columns:
        df = df.drop(columns=["index"], errors="ignore")

    # Normalizamos la fecha a UTC
    if "date" not in df.columns:
        raise ValueError(f"El archivo {filepath} no contiene columna 'date'.")
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # Tipos numéricos (silencioso si faltan)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Orden y limpieza final
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["date"])  # elimina fechas inválidas

    df = validate_df(df, OHLCVSchema, name="OHLCV")
    return df


def load_data(
    ticker: str, timeframe: str, use_local: bool = True, base_path: Path | None = None
) -> pd.DataFrame:
    """
    Punto de entrada unificado.
    - Si use_local=True (por defecto), carga desde CSV en S.data_path (o base_path si se pasa).
    - Si en el futuro quieres IBKR, aquí hacemos el branching.
    """
    if use_local:
        return load_local_data(ticker=ticker, timeframe=timeframe, base_path=base_path)
    # Placeholder para futuras fuentes (IBKR, API, etc.)
    raise NotImplementedError("Carga remota deshabilitada en esta versión. Usa use_local=True.")
