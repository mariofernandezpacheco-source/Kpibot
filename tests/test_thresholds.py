import pytest
from conftest import safe_import


def test_threshold_clamped():
    fn = None
    for cand in [
        ("engine.strategy", "get_threshold_for"),
        ("engine.strategy", "recommended_threshold"),
        ("engine.strategy", "load_threshold_for_ticker"),
    ]:
        try:
            fn = safe_import(*cand)
            break
        except pytest.skip.Exception:
            continue
    if fn is None:
        pytest.skip("No se localiza función de threshold; ajusta el nombre en test_thresholds.py")

    thr = float(fn("TEST", timeframe="10min"))
    assert 0.3 <= thr <= 0.9, f"Threshold fuera de rango razonable: {thr}"
