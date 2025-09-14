# utils/io_utils.py
from __future__ import annotations

import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import pandas as pd
from filelock import FileLock, Timeout

# --- Config por defecto ---
DEFAULT_LOCK_TIMEOUT = 10.0  # segundos
DEFAULT_READ_RETRIES = 3
DEFAULT_READ_RETRY_SLEEP = 0.2  # s


def _lock_for(path: Path) -> FileLock:
    return FileLock(str(path) + ".lock")


def _atomic_replace(tmp_path: Path, final_path: Path):
    # Reemplazo atómico (misma partición)
    os.replace(str(tmp_path), str(final_path))


def atomic_write_csv(
    df: pd.DataFrame,
    path: Path,
    mode: Literal["w", "x"] = "w",
    index: bool = False,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    **to_csv_kwargs,
) -> None:
    """
    Escribe CSV de forma atómica: escribe a *.tmp y luego os.replace().
    Protegido por FileLock(path+'.lock').
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    lock = _lock_for(path)
    try:
        with lock.acquire(timeout=lock_timeout):
            df.to_csv(tmp_path, index=index, **to_csv_kwargs)
            _atomic_replace(tmp_path, path)
    except Timeout:
        raise Timeout(f"Timeout adquiriendo lock para escribir: {path}")
    finally:
        # Limpieza de tmp si quedó colgado por excepción
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def safe_read_csv(
    path: Path,
    parse_dates: Iterable[str] | None = None,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    retries: int = DEFAULT_READ_RETRIES,
    retry_sleep: float = DEFAULT_READ_RETRY_SLEEP,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Lee CSV protegido con FileLock(path+'.lock') y reintentos ante EmptyDataError / parcial.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    last_err = None
    for attempt in range(1, retries + 1):
        lock = _lock_for(path)
        try:
            with lock.acquire(timeout=lock_timeout):
                return pd.read_csv(path, parse_dates=parse_dates, **read_csv_kwargs)
        except Timeout as e:
            last_err = e
        except pd.errors.EmptyDataError as e:
            # Posible escritor a medio volcado; esperamos y reintentamos
            last_err = e
        except Exception as e:
            last_err = e

        time.sleep(retry_sleep)

    # Último intento sin lock (best-effort) por si el lock quedó huérfano
    try:
        return pd.read_csv(path, parse_dates=parse_dates, **read_csv_kwargs)
    except Exception:
        # Fallo definitivo
        raise RuntimeError(f"No se pudo leer CSV consistente: {path}. Último error: {last_err}")


def atomic_write_parquet(
    df: pd.DataFrame,
    path: Path,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    compression: str = "snappy",
    **to_parquet_kwargs,
) -> None:
    """
    Escribe Parquet de forma atómica con lock y compresión snappy.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    lock = _lock_for(path)
    try:
        with lock.acquire(timeout=lock_timeout):
            df.to_parquet(tmp_path, compression=compression, **to_parquet_kwargs)
            _atomic_replace(tmp_path, path)
    except Timeout:
        raise Timeout(f"Timeout adquiriendo lock para escribir: {path}")
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def safe_read_parquet(
    path: Path,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    retries: int = DEFAULT_READ_RETRIES,
    retry_sleep: float = DEFAULT_READ_RETRY_SLEEP,
    **read_parquet_kwargs,
) -> pd.DataFrame:
    """
    Lee Parquet protegido con lock y reintentos.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    last_err = None
    for attempt in range(1, retries + 1):
        lock = _lock_for(path)
        try:
            with lock.acquire(timeout=lock_timeout):
                return pd.read_parquet(path, **read_parquet_kwargs)
        except Timeout as e:
            last_err = e
        except Exception as e:
            last_err = e
        time.sleep(retry_sleep)

    # Último intento sin lock
    try:
        return pd.read_parquet(path, **read_parquet_kwargs)
    except Exception:
        raise RuntimeError(f"No se pudo leer Parquet consistente: {path}. Último error: {last_err}")
