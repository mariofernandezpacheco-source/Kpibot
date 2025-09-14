from __future__ import annotations

from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal

from settings import S


def _to_tz(dt: pd.Timestamp, tz: str) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(tz)


@lru_cache(maxsize=8)
def _calendar():
    cal_name = getattr(S, "calendar", None) or getattr(S, "exchange_calendar", "XNYS")
    return mcal.get_calendar(cal_name)


def get_session_bounds(day: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Devuelve (open, close) del día en TZ de mercado."""
    tz_mkt = S.market_timezone
    now = pd.Timestamp.now(tz=tz_mkt)
    day = now.normalize() if day is None else _to_tz(pd.Timestamp(day), tz_mkt).normalize()

    cal = _calendar()
    sched = cal.schedule(start_date=day.date(), end_date=day.date())
    if sched.empty:
        next_days = cal.valid_days(
            start_date=day.date(), end_date=day.date() + pd.Timedelta(days=7)
        )
        if len(next_days) == 0:
            raise RuntimeError("No hay sesiones próximas")
        nd = pd.Timestamp(next_days[0])
        sched = cal.schedule(start_date=nd.date(), end_date=nd.date())

    o = _to_tz(pd.Timestamp(sched.iloc[0]["market_open"]), tz_mkt)
    c = _to_tz(pd.Timestamp(sched.iloc[0]["market_close"]), tz_mkt)
    return o, c


def is_market_open(ts: pd.Timestamp | None = None) -> bool:
    tz_mkt = S.market_timezone
    now = pd.Timestamp.now(tz=tz_mkt) if ts is None else _to_tz(pd.Timestamp(ts), tz_mkt)
    o, c = get_session_bounds(now)
    return (now >= o) and (now <= c)


def minutes_until_close(ts: pd.Timestamp | None = None) -> int:
    tz_mkt = S.market_timezone
    now = pd.Timestamp.now(tz=tz_mkt) if ts is None else _to_tz(pd.Timestamp(ts), tz_mkt)
    _, c = get_session_bounds(now)
    return max(0, int((c - now).total_seconds() // 60))


def within_close_buffer(ts: pd.Timestamp | None = None, buffer_min: int | None = None) -> bool:
    if buffer_min is None:
        buffer_min = getattr(S, "market_close_buffer_min", 10)
    return minutes_until_close(ts) <= buffer_min


# Helpers para mostrar en hora local (opcional)
def to_local(ts: pd.Timestamp) -> pd.Timestamp:
    return _to_tz(pd.Timestamp(ts), S.local_timezone)
