import pandas as pd
import pytest
from conftest import safe_import


def test_triple_barrier_basic(tiny_prices_for_labels):
    # Buscar función de etiquetas
    fn = None
    for candidate in [
        ("utils.C_label_generator", "generate_labels"),
        ("utils.C_label_generator", "make_labels"),
        ("utils.C_label_generator", "triple_barrier_labels"),
        ("utils.label_generator", "generate_labels"),
    ]:
        try:
            fn = safe_import(*candidate)
            break
        except pytest.skip.Exception:
            continue
    if fn is None:
        pytest.skip("No se encontró la función de labels en utils.C_label_generator")

    df = tiny_prices_for_labels.copy()
    out = fn(
        df,
        take_profit=0.01,
        stop_loss=0.02,
        max_horizon=3,
        price_col="close",
        high_col="high",
        low_col="low",
        date_col="date",
    )

    if isinstance(out, pd.DataFrame):
        label = None
        for c in ["label", "y", "y_cls", "target"]:
            if c in out.columns:
                label = out[c]
                break
        if label is None:
            raise AssertionError("No encuentro columna 'label' en el resultado de labels.")
    else:
        label = out

    assert len(label) == len(df), "Labels debe alinear 1:1 con filas"
    values = set(pd.Series(label).dropna().unique().tolist())
    assert values.issubset({-1, 0, 1, 2}), f"Valores inesperados de labels: {values}"


def test_no_leakage_label_future(tiny_prices_for_labels):
    fn = safe_import("utils.C_label_generator", "generate_labels")

    df1 = tiny_prices_for_labels.copy()
    df2 = tiny_prices_for_labels.copy()
    df2.loc[1:, "close"] = df2.loc[1:, "close"].values[::-1]  # permuta futuro

    y1 = fn(
        df1,
        take_profit=0.01,
        stop_loss=0.02,
        max_horizon=3,
        price_col="close",
        high_col="high",
        low_col="low",
        date_col="date",
    )
    y2 = fn(
        df2,
        take_profit=0.01,
        stop_loss=0.02,
        max_horizon=3,
        price_col="close",
        high_col="high",
        low_col="low",
        date_col="date",
    )

    n = len(df1) - 3
    pd.testing.assert_series_equal(
        pd.Series(y1).iloc[:n].reset_index(drop=True), pd.Series(y2).iloc[:n].reset_index(drop=True)
    )
