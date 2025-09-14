# utils/threshold_sweep.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    commission_bps: float = 1.0  # 1 bps = 0.01%
    slippage_bps: float = 2.0


def _trade_pnl(label: int, side: int, tp_pct: float, sl_pct: float) -> float:
    # label: +1 tocó TP, -1 tocó SL, 0 timeout | side: +1 long, -1 short, 0 flat
    if side == 0 or label == 0:
        return 0.0
    return tp_pct if (label == side) else -sl_pct


def _apply_costs(r: np.ndarray, cost: CostModel) -> np.ndarray:
    # coste round-trip aprox: 2 comisiones + 1 slippage (en bps)
    rt = (2 * cost.commission_bps + cost.slippage_bps) / 10000.0
    return r - rt


def _sharpe(returns: np.ndarray, eps: float = 1e-12) -> float:
    if returns.size == 0:
        return float("nan")
    mu = np.nanmean(returns)
    sd = np.nanstd(returns, ddof=1)
    return float("nan") if sd < eps else float(mu / sd)


def sweep_thresholds(
    y_true: pd.Series | np.ndarray,
    proba: pd.Series | np.ndarray,
    *,
    tp_pct: float,
    sl_pct: float,
    primary: str = "oof_net_sharpe",
    cost: CostModel = CostModel(),
    grid: np.ndarray | None = None,
) -> tuple[pd.DataFrame, float, str]:
    """
    Genera señales con umbral simétrico:
      +1 si p>=t, -1 si p<=1-t, 0 si queda en medio.
    Calcula métricas netas por trade según label de triple-barrera.
    """
    y = pd.Series(y_true).astype(float).values
    p = pd.Series(proba).astype(float).values
    if grid is None:
        grid = np.round(np.arange(0.50, 0.81, 0.01), 2)

    rows = []
    for t in grid:
        sig = np.where(p >= t, 1, np.where(p <= (1.0 - t), -1, 0))
        m = sig != 0
        if m.sum() == 0:
            rows.append(
                {
                    "thr": t,
                    "oof_trades": 0,
                    "oof_accept_rate": 0.0,
                    "oof_net_return": np.nan,
                    "oof_net_sharpe": np.nan,
                    "oof_max_drawdown": np.nan,
                    "oof_win_rate": np.nan,
                }
            )
            continue

        r = np.array(
            [_trade_pnl(lbl, sd, tp_pct, sl_pct) for lbl, sd in zip(y[m], sig[m], strict=False)],
            dtype=float,
        )
        r = _apply_costs(r, cost)

        ret = float(np.nanmean(r)) if r.size else np.nan
        shp = _sharpe(r)
        eq = np.cumsum(r)
        peak = np.maximum.accumulate(eq) if eq.size else np.array([])
        dd = float(np.nanmax(peak - eq)) if eq.size else np.nan
        win = float((r > 0).mean()) if r.size else np.nan

        rows.append(
            {
                "thr": t,
                "oof_trades": int(m.sum()),
                "oof_accept_rate": float(m.mean()),
                "oof_net_return": ret,
                "oof_net_sharpe": shp,
                "oof_max_drawdown": dd,
                "oof_win_rate": win,
            }
        )

    df = pd.DataFrame(rows).sort_values("thr").reset_index(drop=True)
    primary = primary if primary in df.columns else "oof_net_sharpe"
    best_row = (
        df.loc[df["oof_max_drawdown"].idxmin()]
        if primary == "oof_max_drawdown"
        else df.loc[df[primary].idxmax()]
    )
    return df, float(best_row["thr"]), primary
