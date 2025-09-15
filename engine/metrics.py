# engine/metrics.py
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, matthews_corrcoef, brier_score_loss,
)

def classification_metrics(y_true, y_proba, thr=0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(y_proba).astype(float)
    p = np.clip(p, 1e-9, 1-1e-9)

    out = {}
    # threshold-free
    out["oof_roc_auc"] = float(roc_auc_score(y_true, p)) if len(np.unique(y_true))>1 else np.nan
    out["oof_pr_auc"]  = float(average_precision_score(y_true, p)) if len(np.unique(y_true))>1 else np.nan
    out["brier"]       = float(brier_score_loss(y_true, p))

    # thresholded
    y_pred = (p >= float(thr)).astype(int)
    out["f1"]        = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"]    = float(recall_score(y_true, y_pred, zero_division=0))
    out["mcc"]       = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true))>1 else np.nan
    return out

def trading_metrics(equity_curve: np.ndarray, trades: list[dict] | None) -> dict:
    """
    equity_curve: vector de equity (p.ej. 1.0 inicial, multiplicativo) o PnL acumulado.
    trades: lista de operaciones con {profit} o {ret}.
    """
    out = {}
    if equity_curve is None or len(equity_curve) == 0:
        return out
    eq = np.asarray(equity_curve, dtype=float)
    ret = np.diff(np.log(eq + 1e-12), prepend=np.log(eq[0]+1e-12))

    out["net_return"] = float(eq[-1] / (eq[0] + 1e-12) - 1.0)

    # Sharpe (intradia: sin RF o RF≈0) - anualiza si quieres (√(bars_per_year))
    sd = np.std(ret, ddof=1)
    out["sharpe"] = float(np.mean(ret) / (sd + 1e-12)) if len(ret) > 1 else np.nan

    # Max drawdown
    roll_max = np.maximum.accumulate(eq)
    drawdown = eq / (roll_max + 1e-12) - 1.0
    out["max_drawdown"] = float(drawdown.min())

    # Profit factor & win rate
    if trades:
        p = np.array([t.get("profit", t.get("ret", 0.0)) for t in trades], dtype=float)
        wins = p[p > 0].sum()
        losses = -p[p < 0].sum()
        out["profit_factor"] = float(wins / (losses + 1e-12)) if losses > 0 else np.nan
        out["win_rate"] = float((p > 0).mean()) if len(p) else np.nan
        out["n_trades"] = int(len(p))
    return out
