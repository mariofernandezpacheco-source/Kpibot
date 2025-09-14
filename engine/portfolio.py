# engine/portfolio.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import settings as settings
from engine.events import Fill, MarketBar, Order, Signal

S = settings.S


@dataclass
class Position:
    ticker: str
    entry_time: datetime
    side: int  # 1 long, -1 short
    qty: int
    entry_price: float
    tp: float
    sl: float
    time_limit: datetime


class Portfolio:
    def __init__(
        self,
        capital_per_trade: float = float(S.capital_per_trade),
        tp_mult: float = float(S.tp_multiplier),
        sl_mult: float = float(S.sl_multiplier),
        time_limit_candles: int = int(S.time_limit_candles),
        candle_minutes: int = int(S.candle_size_min_by_tf.get(S.timeframe_default, 5)),
    ):
        self.capital_per_trade = float(capital_per_trade)
        self.tp_mult = float(tp_mult)
        self.sl_mult = float(sl_mult)
        self.time_limit_candles = int(time_limit_candles)
        self.candle_minutes = int(candle_minutes)
        self.positions: dict[str, Position] = {}

    def has_position(self, ticker: str) -> bool:
        return ticker in self.positions

    def on_signal(self, sig: Signal, last_price: float, atr_abs: float) -> Order | None:
        if self.has_position(sig.ticker):
            return None
        qty = int(max(0, self.capital_per_trade // max(1e-9, last_price)))
        if qty <= 0 or atr_abs <= 0:
            return None

        if sig.side == 1:
            tp = last_price + self.tp_mult * atr_abs
            sl = last_price - self.sl_mult * atr_abs
        else:
            tp = last_price - self.tp_mult * atr_abs
            sl = last_price + self.sl_mult * atr_abs

        tlimit = sig.ts + timedelta(minutes=self.time_limit_candles * self.candle_minutes)
        return Order(
            ticker=sig.ticker, ts=sig.ts, side=sig.side, qty=qty, tp=tp, sl=sl, time_limit=tlimit
        )

    def on_fill(self, fill: Fill):
        # registra/actualiza posición
        if fill.qty == 0:
            return
        if fill.ticker not in self.positions:
            # nueva
            self.positions[fill.ticker] = Position(
                ticker=fill.ticker,
                entry_time=fill.ts,
                side=fill.side,
                qty=fill.qty,
                entry_price=fill.price,
                tp=0.0,
                sl=0.0,
                time_limit=fill.ts,  # tp/sl se ajustan al crear orden
            )

    def register_entry(self, order: Order, fill: Fill):
        # guarda tp/sl/time_limit exactos en la posición (desde la orden original)
        pos = self.positions.get(order.ticker)
        if pos:
            pos.tp = float(order.tp or 0.0)
            pos.sl = float(order.sl or 0.0)
            pos.time_limit = order.time_limit or pos.time_limit

    def check_exits(self, mb: MarketBar) -> Order | None:
        """Devuelve una orden de cierre si se dispara TP/SL o time_limit."""
        pos = self.positions.get(mb.ticker)
        if not pos:
            return None
        price = mb.close
        should_close = False

        if pos.side == 1:
            if price >= pos.tp:
                should_close = True
            if price <= pos.sl:
                should_close = True
        else:
            if price <= pos.tp:
                should_close = True
            if price >= pos.sl:
                should_close = True

        if mb.ts >= pos.time_limit:
            should_close = True

        if should_close:
            # cerrar con qty opuesta
            return Order(ticker=pos.ticker, ts=mb.ts, side=-pos.side, qty=pos.qty, type="MKT")

        return None

    def on_exit_fill(self, fill: Fill):
        # retira la posición al cerrarla
        if fill.ticker in self.positions:
            self.positions.pop(fill.ticker, None)

    def open_positions(self) -> dict[str, Position]:
        return dict(self.positions)
