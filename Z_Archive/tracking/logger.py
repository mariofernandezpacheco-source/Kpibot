import numpy as np
import pandas as pd


def trading_metrics(equity: pd.Series, trades: pd.DataFrame, rf_daily: float = 0.0) -> dict:
    r = equity.pct_change().dropna()
    daily = r.resample("1D").sum(min_count=1).dropna()
    net_ret = equity.iloc[-1] / equity.iloc[0] - 1
    sharpe = ((daily.mean() - rf_daily) / daily.std() * np.sqrt(252)) if daily.std() > 0 else np.nan
    max_dd = (equity / equity.cummax() - 1).min()
    win_rate = float((trades["pnl"] > 0).mean()) if len(trades) else np.nan
    pf = (
        (trades.loc[trades.pnl > 0, "pnl"].sum() / abs(trades.loc[trades.pnl < 0, "pnl"].sum()))
        if (len(trades) and (trades.loc[trades.pnl < 0, "pnl"].sum() != 0))
        else np.nan
    )
    return dict(
        net_return=float(net_ret),
        sharpe=float(sharpe),
        max_drawdown=float(max_dd),
        win_rate=win_rate,
        profit_factor=float(pf),
        n_trades=int(len(trades)),
    )


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    out = {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if (y_proba is not None) and (len(np.unique(y_true)) == 2):
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
    return out
