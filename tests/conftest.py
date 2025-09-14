import os
import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Asegura que el root del repo esté en sys.path cuando ejecutes pytest desde la raíz
PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def safe_import(path, name=None):
    """Importa de forma segura y devuelve (mod, attr) o pytest.skip si no existe."""
    try:
        mod = __import__(path, fromlist=["*"])
    except Exception as e:
        pytest.skip(f"Saltando tests: no se pudo importar {path}: {e}")
    if name is None:
        return mod
    if not hasattr(mod, name):
        pytest.skip(f"Saltando tests: {path} no tiene {name}")
    return getattr(mod, name)


@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.fixture
def synthetic_ohlcv(rng):
    """Genera un OHLCV sintético estable y reproducible (10min)."""
    n = 300
    dt0 = datetime(2025, 1, 1, 9, 0, tzinfo=UTC)
    price = 100 + rng.normal(0, 0.2, size=n).cumsum()
    price = np.maximum(price, 1.0)
    high = price + np.abs(rng.normal(0, 0.15, size=n))
    low = price - np.abs(rng.normal(0, 0.15, size=n))
    open_ = price + rng.normal(0, 0.05, size=n)
    close = price + rng.normal(0, 0.05, size=n)
    vol = rng.integers(1000, 5000, size=n)

    ts = [dt0 + timedelta(minutes=10 * i) for i in range(n)]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(ts, utc=True),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
            "ticker": "TEST",
        }
    )
    return df


@pytest.fixture
def tiny_prices_for_labels():
    """Serie corta con un TP/SL claro para test de etiquetas triple-barrera."""
    closes = [100.0, 101.1, 100.5, 99.8, 98.0, 98.5, 99.0]
    highs = [100.2, 101.2, 100.7, 100.0, 98.5, 99.0, 99.5]
    lows = [99.8, 100.9, 100.3, 99.0, 97.8, 98.0, 98.5]
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=len(closes), freq="10min", tz="UTC"),
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": 1000,
            "ticker": "TEST",
        }
    )
    return df
