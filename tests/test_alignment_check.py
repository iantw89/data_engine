"""Verdicts of the geometric alignment check (pure parts)."""

from data_engine.solve.alignment_check import judge, widen

LAGS = [-2, -1, 0, 1, 2]


def test_lag_zero_best_or_tied_is_verified():
    assert judge([300, 600, 900, 610, 320], LAGS)["verdict"] == "verified"
    assert judge([300, 600, 900, 915, 320], LAGS)["verdict"] == "verified"  # within 3% of the best


def test_too_few_pairs_is_no_evidence():
    assert judge([10, 20, 30, 100, 10], LAGS)["verdict"] == "too_little_evidence"


def test_disagreement_at_the_edge_is_not_a_measurement():
    entry = judge([900, 700, 500, 400, 300], LAGS)
    assert entry["verdict"] == "disagrees" and entry["best_lag"] == -2 and not entry["peak_inside_scan"]
    wide = list(range(-12, 13))
    counts = [1000 - 60 * abs(lag + 6) for lag in wide]  # the real peak: 6 frames early
    measured = widen({"t": [0, 3000], **entry}, counts, wide)
    assert measured["verdict"] == "disagrees" and measured["best_lag"] == -6 and measured["peak_inside_scan"]
    assert measured["lags"] == wide and measured["t"] == [0, 3000]


def test_interior_disagreement_is_a_measurement():
    entry = judge([300, 900, 500, 400, 300], LAGS)
    assert entry["verdict"] == "disagrees" and entry["best_lag"] == -1 and entry["peak_inside_scan"]
