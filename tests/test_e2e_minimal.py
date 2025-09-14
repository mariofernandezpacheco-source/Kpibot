import numpy as np
import pandas as pd
import pytest
from conftest import safe_import


@pytest.mark.slow
def test_minimal_pipeline_e2e(synthetic_ohlcv):
    feat_fn = safe_import("utils.B_feature_engineering", "build_features")
    lab_fn = safe_import("utils.C_label_generator", "generate_labels")

    df = synthetic_ohlcv.copy()
    X = feat_fn(df)
    y = lab_fn(
        df,
        take_profit=0.006,
        stop_loss=0.006,
        max_horizon=6,
        price_col="close",
        high_col="high",
        low_col="low",
        date_col="date",
    )

    X = X.iloc[-len(y) :].reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]
    y_bin = (y == 1).astype(int)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X_num = X.select_dtypes(include=[np.number]).fillna(0.0)
    if X_num.shape[0] < 50 or X_num.shape[1] < 1:
        pytest.skip("Muy pocas filas o sin columnas numéricas para el mini e2e")

    Xtr, Xte, ytr, yte = train_test_split(
        X_num, y_bin, test_size=0.25, random_state=42, shuffle=False
    )
    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    assert proba.shape[0] == yte.shape[0]

    thr = 0.55
    preds = (proba >= thr).astype(int)
    assert preds.sum() >= 1 and preds.sum() <= len(preds) - 1
