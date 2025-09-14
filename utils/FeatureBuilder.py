# utils/feature_builder.py
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Pasa-a-través controlado que CONGELA las columnas de entrada tras fit.
    - Si falta 'atrp_14' y se permite, la calcula desde OHLC (Wilder 14) y close.
    - Limpia inf → NaN. No rellena NaNs; eso lo hace el preprocesador.

    Parámetros
    ----------
    keep_cols : Iterable[str] | None
        Si se pasa, se usará EXACTAMENTE ese conjunto de columnas (en ese orden) y se ignorará el resto.
        Si es None, se usan las columnas de X en fit(), en el orden presente.
    atrp_col : str
        Nombre de la columna de ATR% esperada. Por defecto 'atrp_14'.
    derive_atrp_if_missing : bool
        Si True y no existe atrp_col, la derivamos de OHLC (Wilder 14).
    atr_window : int
        Ventana para ATR si se deriva.

    Atributos
    ---------
    feature_cols_ : List[str]
        Lista CONGELADA de columnas tras el fit.
    """

    def __init__(
        self,
        keep_cols: Iterable[str] | None = None,
        atrp_col: str = "atrp_14",
        derive_atrp_if_missing: bool = True,
        atr_window: int = 14,
    ):
        self.keep_cols = list(keep_cols) if keep_cols is not None else None
        self.atrp_col = atrp_col
        self.derive_atrp_if_missing = derive_atrp_if_missing
        self.atr_window = atr_window
        self.feature_cols_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._maybe_add_atrp(X.copy())
        if self.keep_cols is not None:
            # usa exactamente las columnas indicadas (si faltan, lanzará en transform)
            self.feature_cols_ = list(self.keep_cols)
        else:
            # congela las columnas presentes (orden estable)
            self.feature_cols_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame):
        if self.feature_cols_ is None:
            raise RuntimeError(
                "FeatureBuilder no está ajustado (feature_cols_ = None). Llama a fit primero."
            )
        X = self._maybe_add_atrp(X.copy())
        # asegura presencia y orden
        missing = [c for c in self.feature_cols_ if c not in X.columns]
        if missing:
            # si falta alguna columna que estaba en fit: crea como NaN (para que el preproc impute)
            for c in missing:
                X[c] = np.nan
        X = X[self.feature_cols_]
        # limpia inf
        X = X.replace([np.inf, -np.inf], np.nan)
        return X

    # ------------ helpers ------------
    def _maybe_add_atrp(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.atrp_col in X.columns:
            return X
        if not self.derive_atrp_if_missing:
            return X
        needed = {"high", "low", "close"}
        if not needed.issubset(set(map(str.lower, X.columns))):
            return X  # no podemos derivar sin OHLC
        # mapear por nombre real (por si las mayúsculas)
        cols = {c.lower(): c for c in X.columns}
        high = X[cols["high"]].astype(float)
        low = X[cols["low"]].astype(float)
        close = X[cols["close"]].astype(float)
        prev_close = close.shift(1)

        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)

        # Wilder smoothing
        alpha = 1.0 / float(self.atr_window)
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        atrp = (atr / close).replace([np.inf, -np.inf], np.nan)
        X[self.atrp_col] = atrp
        return X
