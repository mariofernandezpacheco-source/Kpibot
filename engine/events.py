# engine/events.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

Side = int  # 1 = long, -1 = short


@dataclass
class MarketBar:
    ticker: str
    ts: datetime  # UTC
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    ticker: str
    ts: datetime  # UTC
    side: Side  # 1 (long) o -1 (short)
    prob_up: float
    prob_down: float
    threshold: float


@dataclass
class Order:
    ticker: str
    ts: datetime  # UTC
    side: Side
    qty: int
    type: str = "MKT"
    tp: float | None = None
    sl: float | None = None
    time_limit: datetime | None = None  # UTC


@dataclass
class Fill:
    ticker: str
    ts: datetime  # UTC
    side: Side
    qty: int
    price: float
    fee: float


@dataclass
class EOD:
    ts: datetime  # UTC fin de sesión (aprox) para forzar cierres
