# utils/parquet_store.py
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import settings as settings

S = settings.S


def _ohlcv_root() -> Path:
    return Path(S.parquet_base_path) / "ohlcv"


def _ensure_cols(df: pd.DataFrame, ticker: str, timeframe: str) -> pd.DataFrame:
    z = df.copy()
    # Normaliza timestamp y crea columna 'date' (día) para la partición
    if "date" in z.columns:
        z["date"] = pd.to_datetime(z["date"], utc=True, errors="coerce")
        z = z.rename(columns={"date": "timestamp"})
    elif "timestamp" in z.columns:
        z["timestamp"] = pd.to_datetime(z["timestamp"], utc=True, errors="coerce")
    else:
        raise ValueError("Se requiere columna 'date' o 'timestamp' en el dataframe de OHLCV.")

    z["date"] = z["timestamp"].dt.floor("D")  # día UTC (YYYY-MM-DD)
    z["ticker"] = ticker.upper()
    z["timeframe"] = timeframe
    # Orden más común: timestamp ascendente
    z = z.sort_values("timestamp").reset_index(drop=True)
    return z


def write_ohlcv_parquet(
    df: pd.DataFrame, *, ticker: str, timeframe: str, mode: str = "append"
) -> None:
    """
    Escribe un df OHLCV al dataset particionado:
      dataset/ohlcv/ticker=XYZ/date=YYYY-MM-DD/part-*.parquet
    Usa pyarrow.parquet.write_to_dataset (compatible con más versiones).
    """
    if not bool(getattr(S, "parquet_enabled", True)):
        return

    root = _ohlcv_root()
    root.mkdir(parents=True, exist_ok=True)

    # Normaliza columnas y crea 'timestamp' + 'date' (día) + 'ticker' + 'timeframe'
    z = _ensure_cols(df, ticker, timeframe)
    # Para particionar por día con write_to_dataset es mejor que 'date' sea date (no timestamp)
    z["date"] = pd.to_datetime(z["timestamp"], utc=True, errors="coerce").dt.date

    # Si nos piden overwrite del ticker, borramos sus particiones
    if mode == "overwrite":
        tdir = root / f"ticker={ticker.upper()}"
        if tdir.exists():
            for p in tdir.rglob("*.parquet"):
                try:
                    p.unlink()
                except Exception:
                    pass
            for d in sorted(tdir.rglob("*"), reverse=True):
                try:
                    d.rmdir()
                except Exception:
                    pass

    # Escribe usando el API clásico de parquet (muy compatible)
    table = pa.Table.from_pandas(z, preserve_index=False)
    kwargs = dict(
        root_path=str(root),
        partition_cols=["ticker", "date"],
        compression=str(getattr(S, "parquet_compression", "zstd")),
    )
    try:
        # versiones con flag use_legacy_dataset
        pq.write_to_dataset(table, use_legacy_dataset=False, **kwargs)
    except TypeError:
        # versiones que no aceptan use_legacy_dataset
        pq.write_to_dataset(table, **kwargs)


def read_ohlcv_parquet(
    *,
    ticker: str,
    timeframe: str | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Lee del dataset aplicando filtros a particiones: ticker y rango de 'date' (día).
    Devuelve un DataFrame con columna 'date' = timestamp (UTC) para compatibilidad.
    """
    if not bool(getattr(S, "parquet_enabled", True)):
        raise RuntimeError("Parquet está deshabilitado en settings.")

    import pyarrow as pa
    import pyarrow.dataset as ds

    root = _ohlcv_root()
    dataset = ds.dataset(str(root), partitioning="hive", format="parquet")

    # Filtros a nivel de partición
    filt = ds.field("ticker") == ticker.upper()
    if start is not None:
        start = pd.to_datetime(start, utc=True)
        filt = filt & (ds.field("date") >= pa.scalar(start.date(), type=pa.date32()))
    if end is not None:
        end = pd.to_datetime(end, utc=True)
        filt = filt & (ds.field("date") <= pa.scalar(end.date(), type=pa.date32()))

    # Proyección de columnas (minimizamos lectura)
    cols = list(columns) if columns else None

    table = dataset.to_table(filter=filt, columns=cols)
    df = table.to_pandas(types_mapper=pd.ArrowDtype)  # rápido

    # Compatibilidad: reponemos 'date' (timestamp) desde 'timestamp'
    if "timestamp" in df.columns and "date" in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    elif "timestamp" in df.columns:
        df["date"] = df["timestamp"]
        df.drop(columns=["timestamp"], inplace=True, errors="ignore")

    # Filtra timeframe si viene como columna
    if timeframe is not None and "timeframe" in df.columns:
        df = df[df["timeframe"] == timeframe]

    # Orden natural
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    return df


def migrate_csv_folder_to_parquet(
    csv_root: Path, timeframe: str, glob_pat: str = "*.csv", ticker_from_name: bool = True
) -> None:
    """
    Migra ficheros CSV de 01_data/<timeframe>/ a dataset/ohlcv/.
    Asume que cada CSV corresponde a un ticker. Si no, ajusta la lógica.
    """
    csv_root = Path(csv_root)
    files = sorted(csv_root.glob(glob_pat))
    for f in files:
        try:
            tkr = f.stem.upper() if ticker_from_name else "UNKNOWN"
            df = pd.read_csv(f)
            write_ohlcv_parquet(df, ticker=tkr, timeframe=timeframe, mode="append")
            print(f"✅ {tkr}: migrado {f.name}")
        except Exception as e:
            print(f"⚠️ {f.name}: error migrando — {e}")
