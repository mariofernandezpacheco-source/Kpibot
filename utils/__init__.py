# utils/__init__.py  — versión segura (sin imports pesados)
# Exporta solo utilidades ligeras. No importes el downloader aquí.

from .A_data_loader import load_data  # opcional
from .B_feature_engineering import add_technical_indicators, load_context_data  # opcional
from .C_label_generator import generate_triple_barrier_labels  # opcional

# from .tscv import PurgedWalkForwardSplit  # opcional

__all__ = [
    "load_data",
    "add_technical_indicators",
    "load_context_data",
    "generate_triple_barrier_labels",
]
