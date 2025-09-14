# utils/schemas.py
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa  # ✅ import recomendado para pandas

# Si quieres que el mínimo de features exigidas sea configurable:
try:
    from settings import S

    _REQUIRED_FEATURES_MIN: list[str] = getattr(
        S, "required_features_min", ["atr_14", "rsi_14", "macd_diff"]
    )
except Exception:
    _REQUIRED_FEATURES_MIN = ["atr_14", "rsi_14", "macd_diff"]


# ---------------------------
# Helpers de checks
# ---------------------------
def _unique_date_or_ticker_date(df: pd.DataFrame) -> bool:
    """
    Si existe 'ticker', valida unicidad en (ticker, date).
    Si no existe, valida unicidad solo en date.
    """
    if "date" not in df.columns:
        return False
    if "ticker" in df.columns:
        return df[["ticker", "date"]].drop_duplicates().shape[0] == len(df)
    return df[["date"]].drop_duplicates().shape[0] == len(df)


# ---------------------------
# Esquemas
# ---------------------------
OHLCVSchema = pa.DataFrameSchema(
    {
        "date": pa.Column(pa.DateTime, nullable=False),
        "open": pa.Column(float, nullable=False),
        "high": pa.Column(float, nullable=False),
        "low": pa.Column(float, nullable=False),
        "close": pa.Column(float, nullable=False),
        "volume": pa.Column(float, nullable=False),  # algunos feeds entregan float
        # 'ticker' es OPCIONAL (permitido por strict=False). Si está, se valida en el check.
    },
    checks=[
        pa.Check(
            _unique_date_or_ticker_date,
            error="Duplicados en date (o en ticker+date si 'ticker' existe)",
        ),
        pa.Check(lambda df: (df["high"] >= df["low"]).all(), error="high < low"),
        pa.Check(
            lambda df: (df["open"] > 0).all() and (df["close"] > 0).all(), error="precios <= 0"
        ),
        pa.Check(lambda df: (df["volume"] >= 0).all(), error="volumen negativo"),
    ],
    strict=False,  # permite columnas extra como 'ticker'
    coerce=True,  # intenta convertir tipos (str→float, etc.)
)

# Construimos dinámicamente el mínimo de columnas de features a exigir
_features_cols = {"date": pa.Column(pa.DateTime, nullable=False)}
for feat in _REQUIRED_FEATURES_MIN:
    _features_cols[feat] = pa.Column(float, nullable=False)

FeaturesSchema = pa.DataFrameSchema(
    _features_cols,
    checks=[
        pa.Check(lambda df: df.isna().sum().sum() == 0, error="NaN en features"),
    ],
    strict=False,
    coerce=True,
)

LabelsSchema = pa.DataFrameSchema(
    {
        "date": pa.Column(pa.DateTime, nullable=False),
        "label": pa.Column(int, nullable=False),  # {-1,0,1} o {0,1,2}
    },
    checks=[
        pa.Check(
            lambda df: df["label"].isin([-1, 0, 1, 0, 1, 2]).all(),
            error="label fuera de {-1,0,1} o {0,1,2}",
        ),
    ],
    strict=False,
    coerce=True,
)

SignalsSchema = pa.DataFrameSchema(
    {
        "timestamp": pa.Column(pa.DateTime, nullable=False),
        "ticker": pa.Column(str, nullable=False),
        "prob_down": pa.Column(float, nullable=False),
        "prob_hold": pa.Column(float, nullable=False),
        "prob_up": pa.Column(float, nullable=False),
        "signal": pa.Column(int, nullable=False),  # {-1,0,1}
    },
    checks=[
        pa.Check(
            lambda df: (
                df[["prob_down", "prob_hold", "prob_up"]].ge(0).all(axis=1)
                & df[["prob_down", "prob_hold", "prob_up"]].le(1).all(axis=1)
            ).all(),
            error="probabilidades fuera de [0,1]",
        ),
        pa.Check(lambda df: df["signal"].isin([-1, 0, 1]).all(), error="signal fuera de {-1,0,1}"),
        pa.Check(
            lambda df: df[["ticker", "timestamp"]].drop_duplicates().shape[0] == len(df),
            error="Duplicados en (ticker, timestamp)",
        ),
    ],
    strict=False,
    coerce=True,
)

TradesSchema = pa.DataFrameSchema(
    {
        "entry_time": pa.Column(pa.DateTime, nullable=False),
        "exit_time": pa.Column(pa.DateTime, nullable=False),
        "ticker": pa.Column(str, nullable=False),
        "signal": pa.Column(int, nullable=False),  # {-1,1}
        "quantity": pa.Column(int, nullable=False),
        "entry_price": pa.Column(float, nullable=False),
        "exit_price": pa.Column(float, nullable=False),
        "pnl": pa.Column(float, nullable=False),
        "exit_reason": pa.Column(str, nullable=False),
    },
    checks=[
        pa.Check(lambda df: (df["quantity"] > 0).all(), error="cantidad <= 0"),
        pa.Check(
            lambda df: (df["entry_price"] > 0).all() & (df["exit_price"] > 0).all(),
            error="precios <= 0",
        ),
    ],
    strict=False,
    coerce=True,
)


# ---------------------------
# Helper de validación con reporte claro
# ---------------------------
def validate_df(df: pd.DataFrame, schema: pa.DataFrameSchema, name: str = "DF") -> pd.DataFrame:
    """
    Valida un DataFrame con pandera en modo lazy, lanzando un ValueError
    con un resumen de los primeros casos fallidos.
    """
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        fc = err.failure_cases
        head = fc.head(5).to_string(index=False) if isinstance(fc, pd.DataFrame) else str(fc)[:1000]
        raise ValueError(f"[SchemaError:{name}] {err}\n\nTop failure cases:\n{head}") from err
