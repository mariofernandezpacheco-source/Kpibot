import numpy as np
import pandas as pd
import pytest
from conftest import safe_import


def test_features_basic(synthetic_ohlcv):
    fn = None
    for candidate in [
        ("utils.B_feature_engineering", "build_features"),
        ("utils.B_feature_engineering", "make_features"),
        ("utils.B_feature_engineering", "transform"),
    ]:
        try:
            fn = safe_import(*candidate)
            break
        except pytest.skip.Exception:
            continue
    if fn is None:
        pytest.skip("No se encontró la función de features en utils.B_feature_engineering")

    df = synthetic_ohlcv.copy()
    feat = fn(df)
    assert isinstance(feat, (pd.DataFrame,)), "Las features deben ser DataFrame"
    assert len(feat) <= len(df)
    tail = feat.tail(50).select_dtypes(include=[np.number])
    assert not tail.isna().any().any(), "No debería haber NaNs numéricos en la cola tras el warmup"


def test_no_future_leakage(synthetic_ohlcv):
    fn = safe_import("utils.B_feature_engineering", "build_features")

    df1 = synthetic_ohlcv.copy()
    df2 = synthetic_ohlcv.copy()
    df2.loc[50:, "close"] = df2.loc[50:, "close"].values[::-1]

    f1 = fn(df1)
    f2 = fn(df2)
    k = 30  # ventana máxima aproximada permitida
    n = min(len(f1), len(f2)) - k
    pd.testing.assert_frame_equal(
        f1.iloc[:n].reset_index(drop=True),
        f2.iloc[:n].reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        atol=1e-8,
        rtol=1e-5,
    )
