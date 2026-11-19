"""Recovered poses as Autofit input: which are new detections, how they accumulate, when a lap is worth it."""

from __future__ import annotations

import numpy as np

from data_engine import autofit_refresh
from data_engine.config import RECIPE

RULES = RECIPE["autofit_refresh"]


def recovered(frames, boxes):
    """One camera column; a pose per row."""
    count = len(frames)
    keypoints = np.zeros((count, 1, 17, 3), np.float32)
    keypoints[..., 2] = 0.7
    return {"pose_frame": np.array(frames, np.int64)[:, None], "pose_box": np.array(boxes, np.float32)[:, None, :],
            "pose_keypoints": keypoints}


def camera(frames, boxes, kept=None):
    return {"name": "c0", "frame_of_row": np.array(frames, np.int64), "boxes": np.array(boxes, np.float32).reshape(-1, 4),
            "kept": np.ones(len(frames), bool) if kept is None else np.array(kept, bool)}


def test_a_pose_on_a_kept_detection_is_not_a_new_detection():
    raw = camera([3, 3, 4], [[0, 0, 100, 200], [300, 0, 400, 200], [0, 0, 100, 200]])
    poses = recovered([3, 3, 5, -1], [[5, 5, 100, 200], [600, 0, 700, 200], [0, 0, 100, 200], [0, 0, 1, 1]])
    found = autofit_refresh.new_detections(poses, [raw])
    assert found["frame"].tolist() == [3, 5] and found["camera"].tolist() == ["c0", "c0"]
    assert np.allclose(found["box"][0], [600, 0, 700, 200]) and np.allclose(found["score"], 0.7)


def test_a_pose_on_a_dropped_detection_is_new_for_autofit():
    """Autofit is fed the kept detections only: a pose where the post-filter dropped the detection is new to it."""
    raw = camera([3], [[0, 0, 100, 200]], kept=[False])
    found = autofit_refresh.new_detections(recovered([3], [[0, 0, 100, 200]]), [raw])
    assert found["frame"].tolist() == [3]


def test_detections_accumulate_and_are_not_added_twice():
    first = autofit_refresh.new_detections(recovered([3, 5], [[0, 0, 100, 200], [0, 0, 100, 200]]), [camera([], [])])
    union, added = autofit_refresh.merge(autofit_refresh.empty(), first)
    assert added == 2
    # the next recovery finds one of them again (slightly different box), one new, and no longer the other
    second = autofit_refresh.new_detections(recovered([5, 9], [[2, 2, 100, 200], [0, 0, 100, 200]]), [camera([], [])])
    union, added = autofit_refresh.merge(union, second)
    assert added == 1 and sorted(union["frame"].tolist()) == [3, 5, 9]
    again, added = autofit_refresh.merge(union, second)
    assert again is union and added == 0


def test_saved_detections_belong_to_one_alignment_and_calibration(tmp_path):
    found = autofit_refresh.new_detections(recovered([3], [[0, 0, 100, 200]]), [camera([], [])])
    signature = {"sync_table": "s1", "calibration": "c1"}
    autofit_refresh.save(tmp_path / "extra.npz", found, signature)
    assert autofit_refresh.load(tmp_path / "extra.npz", signature)["frame"].tolist() == [3]
    assert not len(autofit_refresh.load(tmp_path / "extra.npz", {**signature, "calibration": "c2"})["frame"])
    assert not len(autofit_refresh.load(tmp_path / "missing.npz", signature)["frame"])


def decide(**changes):
    values = {"new": 5000, "fed": 1_000_000, "identities": 300, "history": [], "laps_this_round": 0, "recipe": RULES}
    return autofit_refresh.decide(**{**values, **changes})


def test_a_lap_needs_enough_new_detections():
    assert decide()["refresh"]
    assert not decide(new=100)["refresh"]  # below the count
    assert not decide(new=600, fed=5_000_000)["refresh"]  # below the share of Autofit's input
    assert "too few" in decide(new=100)["reason"]


def test_laps_per_round_are_limited():
    answer = decide(laps_this_round=RULES["max_laps_per_round"])
    assert not answer["refresh"] and "next round" in answer["reason"]


def test_another_lap_only_while_laps_pay_off():
    paid = [{"round": 3, "new": 5000, "identities_before": 300, "identities_after": 270}]
    assert decide(history=paid, new=1500)["refresh"]
    flat = [{"round": 3, "new": 5000, "identities_before": 300, "identities_after": 299}]
    answer = decide(history=flat, new=1500)
    assert not answer["refresh"] and answer["last_lap"]["gain"] < RULES["min_gain"]
    assert decide(history=flat, new=10_000)["refresh"]  # much more evidence than the lap that did not help
    unfinished = [{"round": 3, "new": 5000, "identities_before": 300, "identities_after": None}]
    assert decide(history=unfinished, new=1500)["refresh"]  # nothing known against it yet


def test_disabled_recipe_never_asks():
    assert not decide(recipe={**RULES, "enabled": False})["refresh"]
