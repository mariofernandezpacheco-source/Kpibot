import pytest
from conftest import safe_import


def test_time_series_split_is_chronological(synthetic_ohlcv):
    splitter = None
    for candidate in [
        ("utils.A_data_loader", "time_series_split"),
        ("utils.A_data_loader", "build_timeseries_splits"),
        ("utils.A_data_loader", "get_time_series_cv"),
    ]:
        try:
            splitter = safe_import(*candidate)
            break
        except pytest.skip.Exception:
            continue
    if splitter is None:
        pytest.skip("No hay splitter temporal propio definido")

    df = synthetic_ohlcv.copy()
    splits = list(splitter(df, n_splits=3)) if callable(splitter) else splitter
    assert len(splits) >= 2, "Debería devolver al menos 2 folds"
    for tr, te in splits:
        assert max(tr) < min(te), "Train debe acabar antes de Test (orden temporal)"
        assert set(tr).isdisjoint(set(te)), "Train y Test no deben solaparse"
