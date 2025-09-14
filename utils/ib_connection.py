# utils/ib_connection.py — reintentos/backoff + reconexión + circuit breaker

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd
from ib_insync import IB, Index, Stock, util
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from settings import S

log = logging.getLogger("ib")
log.setLevel(logging.INFO)


@dataclass
class IBConfig:
    host: str = S.ib_host
    port: int = S.ib_port
    client_id: int = S.ib_client_id
    max_retries: int = S.ib_max_retries
    backoff_min_s: float = S.ib_backoff_min_s
    backoff_max_s: float = S.ib_backoff_max_s
    circuit_fail_threshold: int = S.ib_circuit_fail_threshold
    circuit_open_seconds: int = S.ib_circuit_open_seconds


_ib: IB | None = None
_fail_count: int = 0
_circuit_open_until: float = 0.0


def _contract_for(symbol: str):
    sym = symbol.upper().strip()
    if sym in {"VIX", "^VIX"}:
        return Index("VIX", "CBOE"), "MIDPOINT"
    return Stock(sym, "SMART", "USD"), "TRADES"


def _ensure_not_open_circuit(cfg: IBConfig):
    global _circuit_open_until
    now = time.time()
    if now < _circuit_open_until:
        remaining = int(_circuit_open_until - now)
        raise RuntimeError(f"Circuit breaker abierto {remaining}s")


def _open_circuit(cfg: IBConfig):
    global _circuit_open_until
    _circuit_open_until = time.time() + cfg.circuit_open_seconds
    log.warning(f"[IB] Circuit breaker ABIERTO por {cfg.circuit_open_seconds}s")


def _record_failure(cfg: IBConfig):
    global _fail_count
    _fail_count += 1
    if _fail_count >= cfg.circuit_fail_threshold:
        _open_circuit(cfg)
        _fail_count = 0  # reseteamos después de abrir


def _record_success():
    global _fail_count
    _fail_count = 0


def get_ib(cfg: IBConfig = IBConfig()) -> IB:
    global _ib
    if _ib and _ib.isConnected():
        return _ib
    _ensure_not_open_circuit(cfg)

    ib = IB()
    try:
        ib.connect(cfg.host, cfg.port, clientId=cfg.client_id, readonly=True)
        if not ib.isConnected():
            raise RuntimeError("No conectado")
        _record_success()
        _ib = ib
        return _ib
    except Exception:
        _record_failure(cfg)
        raise


def reconnect_if_needed(cfg: IBConfig = IBConfig()):
    """Reconecta si la sesión se cayó."""
    global _ib
    if not _ib or not _ib.isConnected():
        return get_ib(cfg)


@retry(
    reraise=True,
    stop=stop_after_attempt(S.ib_max_retries),
    wait=wait_exponential_jitter(initial=S.ib_backoff_min_s, max=S.ib_backoff_max_s),
    retry=retry_if_exception_type((Exception,)),  # puedes afinar tipos si quieres
    before_sleep=before_sleep_log(log, logging.WARNING),
)
def req_historical_with_retry(symbol: str, duration: str, bar_size: str) -> pd.DataFrame:
    """
    Llama a reqHistoricalData con reintentos, reconexión y circuit breaker.
    Devuelve DF normalizado (date UTC, OHLCV).
    """
    cfg = IBConfig()
    _ensure_not_open_circuit(cfg)
    ib = get_ib(cfg)

    contract, what = _contract_for(symbol)
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what,
            useRTH=True,
            formatDate=1,
            keepUpToDate=False,
        )
        if not bars:
            raise RuntimeError("Respuesta vacía")
        df = util.df(bars)
        if "date" not in df.columns:
            df = df.rename(columns={"time": "date"})
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = (
            df[["date", "open", "high", "low", "close", "volume"]]
            .dropna(subset=["date"])
            .sort_values("date")
        )
        _record_success()
        return df.reset_index(drop=True)
    except Exception:
        reconnect_if_needed(cfg)  # intenta reloguear en el siguiente intento
        _record_failure(cfg)
        raise
