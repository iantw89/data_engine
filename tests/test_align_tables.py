"""Sync-table helpers of workers/align_dm.py (pure numpy; no data_management needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_engine.workers import align_dm as A  # noqa: E402


def clock(n, period, start=0.0):
    return start + np.arange(n) * period


def test_constant_offsets_and_fast_clock():
    # Camera B runs 0.05% fast and its frame 0 is stamped 1.0 s: it shows common time t at its own time t + 1.0
    # (a constant TIME offset of 1.0 s), while the frame-index offset between A and B drifts ~3 frames per 600 s.
    a = clock(6000, 0.1)
    b = clock(6000, 0.1 * (1 - 0.0005)) + 1.0
    table = A.build_sync_table({"A": a, "B": b}, [(10.0, {"A": 0.0, "B": 1.0})], anchor="A")
    assert list(table["cameras"]) == ["A", "B"]
    t, native = table["aligned_time"], table["native"]
    assert native.dtype == np.int32 and native.shape == (len(t), 2)
    ok = (native >= 0).all(axis=1)
    for i, times in enumerate((a, b)):
        off = 0.0 if i == 0 else 1.0
        assert (np.abs(times[native[ok, i]] - (t[ok] + off)) <= 0.05 + 1e-9).all()
    early, late = int(np.argmin(np.abs(t - 10.0))), int(np.argmin(np.abs(t - 590.0)))
    assert native[early, 1] - native[early, 0] == 0
    assert native[late, 1] - native[late, 0] == 3  # time-based mapping follows the faster clock
    assert np.allclose(table["offsets_seconds"][:, 1], 1.0) and (table["offsets_seconds"][:, 0] == 0).all()
    # B has frames from common time -1.0; the span starts where two cameras first have frames.
    assert abs(t[0] - 0.0) < 0.15 and native[0, 0] >= 0 and native[0, 1] >= 0


def test_unresolved_span_gives_minus_one_and_nan():
    a, b, c = clock(300, 0.1), clock(300, 0.1) + 0.5, clock(300, 0.1) + 0.2
    # C joins only at t=10 (not in the first entry) and is lost again from t=20.
    schedule = [(5.0, {"A": 0.0, "B": 0.5}), (10.0, {"A": 0.0, "B": 0.5, "C": 0.2}), (20.0, {"A": 0.0, "B": 0.5})]
    table = A.build_sync_table({"B": b, "A": a, "C": c}, schedule, anchor="A")
    assert list(table["cameras"]) == ["A", "B", "C"]
    t, native, off = table["aligned_time"], table["native"], table["offsets_seconds"]
    ci = 2
    before = t < 10.0 - 1e-9
    during = (t >= 10.0 - 1e-9) & (t < 20.0 - 1e-9) & (t > 0.0)
    after = t >= 20.0 - 1e-9
    assert np.isnan(off[before, ci]).all() and (native[before, ci] == -1).all()
    assert np.isfinite(off[during, ci]).all() and (native[during, ci] >= 0).all()
    assert np.isnan(off[after, ci]).all() and (native[after, ci] == -1).all()
    # Resolved cameras are unaffected, including before the first entry (first entry applies).
    assert (native[t < 5.0][:, 1] >= -1).all()
    assert np.allclose(off[:, 1], 0.5)
    # An unresolved camera is never replaced by 0 or a guess.
    assert not np.isnan(off[:, 0]).any()


def test_dropout_and_out_of_range_give_minus_one():
    a = clock(100, 0.1)
    b = np.concatenate([clock(40, 0.1), clock(40, 0.1, start=6.0)])  # 2 s dropout after frame 39
    table = A.build_sync_table({"A": a, "B": b}, [(0.0, {"A": 0.0, "B": 0.0})], anchor="A")
    t, native = table["aligned_time"], table["native"]
    inside_gap = (t > 4.1) & (t < 5.9)
    assert inside_gap.any() and (native[inside_gap, 1] == -1).all()
    assert (native[inside_gap, 0] >= 0).all()
    assert native[np.argmin(np.abs(t - 7.0)), 1] == 40 + 10


def test_no_schedule_yields_empty_table():
    table = A.build_sync_table({"A": clock(50, 0.1), "B": clock(50, 0.1)}, [], anchor="A")
    assert len(table["aligned_time"]) == 0 and table["native"].shape == (0, 2)


def test_robust_line_and_spans():
    t = np.arange(0.0, 600.0, 10.0)
    y = 1.5 + 4e-4 * t
    y[5] += 3.0  # outlier
    slope, intercept = A.robust_line(t, y)
    assert abs(slope - 4e-4) < 1e-6 and abs(intercept - 1.5) < 1e-3
    assert A.robust_line(t[:2], y[:2]) is None
    unresolved, stale = A.stale_unresolved_spans(np.array([30.0, 40.0, 100.0]), 10.0, 150.0, 30.0)
    assert unresolved == [[10.0, 30.0]] and stale == [[70.0, 100.0], [130.0, 150.0]]
    assert A.stale_unresolved_spans(np.array([]), 10.0, 150.0, 30.0) == ([[10.0, 150.0]], [])
    assert A.coverage_spans({"a": [(0, 10)], "b": [(5, 20)], "c": [(30, 40)]}) == [(5, 10)]
