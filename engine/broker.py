# engine/broker.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import settings as settings
from engine.events import Fill, MarketBar, Order

S = settings.S


@dataclass
class SimConfig:
    commission_per_trade: float = float(S.commission_per_trade)
    slippage_bps: float = 1.0  # 1 = 1 punto básico ~ 0.01%
    min_slippage_abs: float = 0.0  # puedes fijar un mínimo absoluto si quieres
    latency_ms: int = 200


class SimBroker:
    """
    Broker simulado muy simple:
    - Ejecuta la orden al precio de cierre de la barra (con slippage).
    - Aplica comisión fija por orden.
    - Simula latencia con timestamp + latency_ms.
    """

    def __init__(self, cfg: SimConfig = SimConfig()):
        self.cfg = cfg

    def _apply_slippage(self, side: int, price: float) -> float:
        slip = price * (self.cfg.slippage_bps / 1e4)
        if self.cfg.min_slippage_abs > 0:
            slip = max(slip, self.cfg.min_slippage_abs)
        return price + (slip if side == 1 else -slip)

    async def execute(self, order: Order, ref_bar: MarketBar) -> Fill:
        px = self._apply_slippage(order.side, ref_bar.close)
        fee = float(self.cfg.commission_per_trade)
        ts = ref_bar.ts + timedelta(milliseconds=self.cfg.latency_ms)
        return Fill(
            ticker=order.ticker, ts=ts, side=order.side, qty=order.qty, price=float(px), fee=fee
        )

    async def close_all(
        self, positions: dict[str, any], ref_bar: MarketBar | None = None
    ) -> dict[str, Fill]:
        fills = {}
        if not positions:
            return fills
        if ref_bar is None:
            # si no nos pasan una barra referencia, no aplicamos slippage/latencia bien; lo dejamos en 0
            for t, pos in positions.items():
                fills[t] = Fill(
                    ticker=t,
                    ts=datetime.utcnow(),
                    side=-pos.side,
                    qty=pos.qty,
                    price=pos.entry_price,
                    fee=self.cfg.commission_per_trade,
                )
            return fills
        for t, pos in positions.items():
            px = self._apply_slippage(-pos.side, ref_bar.close)
            ts = ref_bar.ts + timedelta(milliseconds=self.cfg.latency_ms)
            fills[t] = Fill(
                ticker=t,
                ts=ts,
                side=-pos.side,
                qty=pos.qty,
                price=float(px),
                fee=float(self.cfg.commission_per_trade),
            )
        return fills
