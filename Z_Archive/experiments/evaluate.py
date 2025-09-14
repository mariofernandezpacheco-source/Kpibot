from dataclasses import dataclass
from typing import Any


@dataclass
class EvalResult:
    metrics: dict[str, float]
    artifacts: dict[
        str, Any
    ]  # p.ej. {"equity": pd.Series, "trades": pd.DataFrame, "figs": {"equity_png": "path"}}
    model: Any | None


def evaluate_model_for_ticker(ticker: str, params: dict) -> EvalResult:
    """
    1) Carga datos del ticker (tu pipeline actual)
    2) Construye features según params['features_set'] y ventanas
    3) CV temporal purgado / walk-forward
    4) Entrena (sklearn/XGB/lo que uses)
    5) Predice → señales usando params['threshold']
    6) Backtest con tus reglas → equity, trades
    7) Calcula métricas de trading + clasificación
    8) Devuelve EvalResult
    """
    # ... aquí llamas a tu motor actual
    return EvalResult(metrics={}, artifacts={}, model=None)
