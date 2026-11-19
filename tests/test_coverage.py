"""Coverage recovery: where to look (hints), and what may be accepted (tiers). No pixels needed."""

from __future__ import annotations

import numpy as np

from data_engine.config import RECIPE
from data_engine.solve import coverage

RECIPE_COVERAGE = RECIPE["coverage"]


def track(times, start=(0.0, 0.0, 36.0), step=(1.0, 0.0, 0.0), identity=3):
    times = np.asarray(times)
    root = np.asarray(start)[None] + times[:, None] * np.asarray(step)[None]
    xyz = np.repeat(root[:, None, :], 17, axis=1) + np.arange(17)[None, :, None] * 0.5
    return np.full(len(times), identity), times, root.astype(np.float32), xyz.astype(np.float32)


def test_missing_instants_inside_and_just_beyond_a_track():
    presence = coverage.build_presence(*track([10, 11, 14, 15]))
    missing = coverage.missing_instants(presence, total=100, end_frames=2)
    assert missing == [(3, 8), (3, 9), (3, 12), (3, 13), (3, 16), (3, 17)]
    assert coverage.missing_instants(presence, total=16, end_frames=5)[-1] == (3, 13)  # never beyond the recording


def test_hint_between_two_measured_poses_is_interpolated_and_only_a_hint():
    presence = coverage.build_presence(*track([10, 20]))
    hint = coverage.make_hint(presence, 3, 14, reach=30)
    assert coverage.HINT_KINDS[hint.kind] == "interpolated" and hint.distance == 4
    assert np.allclose(hint.root, [14.0, 0.0, 36.0])
    assert coverage.make_hint(presence, 3, 10, reach=30) is None  # measured instants need no hint


def test_hint_near_an_end_is_held_and_out_of_reach_is_none():
    presence = coverage.build_presence(*track([10, 20]))
    after = coverage.make_hint(presence, 3, 25, reach=30)
    assert coverage.HINT_KINDS[after.kind] == "held_from_before" and after.distance == 5
    assert np.allclose(after.root, [20.0, 0.0, 36.0])
    before = coverage.make_hint(presence, 3, 4, reach=30)
    assert coverage.HINT_KINDS[before.kind] == "held_from_after" and before.distance == 6
    assert coverage.make_hint(presence, 3, 60, reach=30) is None
    assert coverage.make_hint(presence, 99, 12, reach=30) is None


def test_long_gap_uses_only_the_side_within_reach():
    presence = coverage.build_presence(*track([10, 100]))
    hint = coverage.make_hint(presence, 3, 20, reach=30)
    assert coverage.HINT_KINDS[hint.kind] == "held_from_before"
    assert coverage.make_hint(presence, 3, 55, reach=30) is None  # 45 frames from either side


def test_box_from_skeleton_uses_measured_margins():
    points = np.full((17, 2), np.nan)
    points[[5, 6, 15, 16]] = [[100, 100], [140, 100], [105, 300], [135, 300]]
    box = coverage.keypoint_box(points, RECIPE_COVERAGE["box_margin"])
    assert np.allclose(box, [100 - 20, 100 - 26, 140 + 20, 300 + 28])
    assert coverage.keypoint_box(points[:6], RECIPE_COVERAGE["box_margin"]) is None  # fewer than 4 joints
    assert coverage.share_inside(np.array([-50.0, 0, 50, 100]), 1000, 1000) == 0.5


def good_features(**changes):
    features = {"cameras": 3, "body_joints": 11, "median_residual_px": 4.0, "bad_bones": 0, "hint_distance_in": 5.0,
                "weakest_view_peak": 0.7, "other_identity_in": 60.0, "hint_kind": 0, "hint_frames": 3}
    return {**features, **changes}


def test_three_camera_recovery_is_tier_a():
    assert coverage.tier(good_features(), RECIPE_COVERAGE) == "A"
    assert coverage.tier(good_features(hint_kind=1, hint_frames=25), RECIPE_COVERAGE) == "A"  # near an end is fine with 3 views


def test_two_camera_recovery_needs_strong_image_evidence_and_a_bracketed_gap():
    two = dict(cameras=2, median_residual_px=2.0)
    assert coverage.tier(good_features(**two), RECIPE_COVERAGE) == "B"
    assert coverage.tier(good_features(**{**two, "median_residual_px": 5.0}), RECIPE_COVERAGE) is None
    assert coverage.tier(good_features(**two, weakest_view_peak=0.4), RECIPE_COVERAGE) is None
    assert coverage.tier(good_features(**two, hint_kind=1), RECIPE_COVERAGE) is None  # held hint: not bracketed
    assert coverage.tier(good_features(**two, hint_frames=9), RECIPE_COVERAGE) is None


def test_nothing_is_accepted_on_top_of_another_person_or_far_from_the_hint():
    assert coverage.tier(good_features(other_identity_in=10.0), RECIPE_COVERAGE) is None
    assert coverage.tier(good_features(hint_distance_in=30.0), RECIPE_COVERAGE) is None
    assert coverage.tier(good_features(bad_bones=3), RECIPE_COVERAGE) is None
    assert coverage.tier(good_features(body_joints=6), RECIPE_COVERAGE) is None
    assert coverage.tier(good_features(cameras=1), RECIPE_COVERAGE) is None


def test_bone_ranges_come_from_this_recordings_measured_poses():
    rng = np.random.default_rng(0)
    xyz = rng.normal(0, 1, (2000, 17, 3)) + np.arange(17)[None, :, None] * 10.0
    ranges = coverage.bone_ranges(xyz)
    assert ranges.shape == (12, 2) and (ranges[:, 0] < ranges[:, 1]).all()
    assert np.isinf(coverage.bone_ranges(xyz[:100])[:, 1]).all()  # too few poses: no bone gate rather than a guess


def test_failed_gates_names_what_is_missing():
    two = good_features(cameras=2, median_residual_px=5.0, weakest_view_peak=0.4, hint_kind=1)
    assert coverage.failed_gates(two, RECIPE_COVERAGE, "B") == ["residual", "image confidence", "not bracketed by nearby poses"]
    assert coverage.failed_gates(good_features(), RECIPE_COVERAGE, "A") == []
    crowded = good_features(other_identity_in=5.0, bad_bones=4)
    assert coverage.failed_gates(crowded, RECIPE_COVERAGE, "A") == ["bone lengths", "another identity is measured there"]


def test_missing_inside_tracks_counts_only_between_first_and_last_position():
    presence = coverage.build_presence(*track([10, 11, 14, 20]))
    assert coverage.inside_missing(presence) == 7  # 12, 13 and 15..19
    more = coverage.build_presence(*track([10, 11, 12, 13, 14, 20]))
    assert coverage.inside_missing(more) == 5


def test_a_recovered_pose_becomes_the_hint_for_its_neighbours():
    """What the recursion relies on: adding a position moves the reach."""
    identity, t, root, xyz = track([10])
    presence = coverage.build_presence(identity, t, root, xyz)
    assert coverage.make_hint(presence, 3, 45, reach=30) is None  # 35 frames away: out of reach
    identity, t, root, xyz = track([10, 40])  # a pass recovered t=40
    presence = coverage.build_presence(identity, t, root, xyz)
    hint = coverage.make_hint(presence, 3, 45, reach=30)
    assert hint is not None and hint.distance == 5 and coverage.HINT_KINDS[hint.kind] == "held_from_before"
