# utils/tscv.py
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PurgedWalkForwardSplit:
    """
    Walk-forward CV para series temporales con purge + embargo.
    - scheme: "expanding" o "rolling"
    - n_splits: nº de folds (bloques test)
    - test_size: tamaño del bloque test en nº de filas
    - train_min_size: tamaño mínimo de train en filas
    - embargo: nº de barras a embargar alrededor del bloque test
    - purge: nº de barras a purgar para cubrir horizonte de etiquetas (p.ej. time_limit_candles)
    """

    n_splits: int
    test_size: int
    train_min_size: int
    scheme: str = "expanding"
    embargo: int = 0
    purge: int = 0

    def split(self, X: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        if self.test_size <= 0 or self.n_splits <= 0:
            raise ValueError("test_size y n_splits deben ser > 0")
        total_test = self.n_splits * self.test_size
        if total_test + self.train_min_size > n:
            raise ValueError("No hay suficientes filas para los folds solicitados")

        # posiciones de los bloques test, consecutivos al final de la muestra útil
        start_first_test = n - total_test
        for k in range(self.n_splits):
            test_start = start_first_test + k * self.test_size
            test_end = test_start + self.test_size  # exclusivo
            # train indices según esquema
            if self.scheme == "expanding":
                train_end = test_start - self.embargo  # embargo antes del test
                train_start = 0
            elif self.scheme == "rolling":
                train_end = test_start - self.embargo
                train_start = max(0, train_end - max(self.train_min_size, self.test_size))
            else:
                raise ValueError("scheme debe ser 'expanding' o 'rolling'")

            # aplica purge alrededor del test (futuro cercano del test también puede contaminar)
            purged_left = max(0, train_end - self.purge)
            train_idx = np.arange(train_start, purged_left, dtype=int)
            test_idx = np.arange(test_start, test_end, dtype=int)
            yield train_idx, test_idx
