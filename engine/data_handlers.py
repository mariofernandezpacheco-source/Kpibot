# engine/data_handlers.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

import settings as settings
from engine.events import EOD, MarketBar
from utils.A_data_loader import load_data

S = settings.S


class BacktestDataHandler:
    """
    Lee CSV/parquet via load_data(ticker, timeframe) y emite MarketBar por orden temporal.
    EOD: por simplicidad, emite un EOD cuando cambia el día (UTC) en el timeline combinado.
    """

    def __init__(self, tickers: list[str], timeframe: str):
        self.tickers = [t.upper() for t in tickers]
        self.timeframe = timeframe
        self.base_path = Path(S.data_path)

    def _load_all(self) -> dict[str, pd.DataFrame]:
        out = {}
        for t in self.tickers:
            df = load_data(
                ticker=t, timeframe=self.timeframe, use_local=True, base_path=self.base_path
            )
            if df is None or df.empty:
                continue
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
            out[t] = df
        return out

    async def stream_to(self, bus):
        frames = self._load_all()
        if not frames:
            return
        # timeline combinado (todas las marcas temporales)
        all_ts = sorted(set(pd.concat([df["date"] for df in frames.values()]).tolist()))
        last_day = None
        for ts in all_ts:
            day = ts.date()
            for t, df in frames.items():
                row = df[df["date"] == ts]
                if row.empty:
                    continue
                r = row.iloc[0]
                ev = MarketBar(
                    ticker=t,
                    ts=ts,
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    volume=float(getattr(r, "volume", 0.0)),
                )
                await bus.put(ev)
            # EOD cuando cambia el día
            if last_day is not None and day != last_day:
                await bus.put(EOD(ts=ts))
            last_day = day
        # EOD final
        await bus.put(EOD(ts=all_ts[-1]))
