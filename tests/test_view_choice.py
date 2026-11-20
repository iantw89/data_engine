"""Which frame of a piece of track a question shows (solve/view_choice.py and its use in the stages).

The clearer frame comes from the same piece of the same camera's tracklet; cameras and pieces are what the
question's own rule chose. One question in five keeps the nearest frames (a control).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from test_link_ranking import SIZE, camera_of, person, scene_of

from data_engine.config import RECIPE
from data_engine.solve import swap_check, view_choice
from data_engine.stages import solve as solve_stage

CLEAR = RECIPE["link_ranking"]


def test_a_fixed_share_of_the_questions_keeps_the_nearest_frames():
    recipe = {"prefer_clear": True, "control_share": 0.2}
    keys = [f"cam-{i % 7}/{i * 13}/{i % 5}" for i in range(2000)]
    share = np.mean([view_choice.in_control(key, recipe) for key in keys])
    assert 0.16 < share < 0.24
    assert [view_choice.in_control(key, recipe) for key in keys[:50]] == [view_choice.in_control(key, recipe) for key in keys[:50]]
    assert not any(view_choice.in_control(key, {"control_share": 0.0}) for key in keys[:200])
    assert all(view_choice.in_control(key, {"control_share": 1.0}) for key in keys[:200])
    control = next(key for key in keys if view_choice.in_control(key, recipe))
    treated = next(key for key in keys if not view_choice.in_control(key, recipe))
    assert view_choice.rule(recipe, CLEAR, control) is None and view_choice.rule(recipe, CLEAR, treated) is CLEAR
    assert view_choice.rule({"prefer_clear": False, "control_share": 0.0}, CLEAR, treated) is None


def test_clearer_takes_the_nearest_clear_view_of_the_pool():
    entries = [(t, *person(joints=17 if t in (3, 9) else 6)) for t in range(12)]
    camera = camera_of(entries)
    pool = np.arange(12)
    assert view_choice.clearer(camera, 5, pool, SIZE, 0.3, CLEAR) == 3, "0.2 s away rather than 0.4 s"
    assert view_choice.clearer(camera, 8, pool, SIZE, 0.3, CLEAR) == 9
    assert view_choice.clearer(camera, 9, pool, SIZE, 0.3, CLEAR) == 9, "already clear"
    assert view_choice.clearer(camera, 5, np.array([4, 5, 6]), SIZE, 0.3, CLEAR) == 5, "no clear view in reach"
    assert view_choice.clearer(camera, 5, pool, SIZE, 0.3, CLEAR, t=11) == 9, "nearest to the question's moment"


def two_pieces():
    """Camera 0: the component in two pieces of the tracklet (atoms 0 and 1); the frame nearest to t=20 is a
    partial view of piece 0, and the only clear views near it are in piece 1. Camera 1: a small clear view."""
    first = camera_of([(t, *person(joints=6 if t < 22 else 17)) for t in range(10, 30)], index=0)
    second = camera_of([(t, *person(x=900.0, height=150.0)) for t in range(10, 30)], index=1)
    scene = scene_of([first, second])
    labels = SimpleNamespace(label=[np.zeros(20, np.int32), np.zeros(20, np.int32)],
                             atom=[np.where(np.arange(10, 30) < 22, 0, 1).astype(np.int32), np.full(20, 2, np.int32)])
    return scene, labels


def test_the_clearer_frame_comes_from_the_same_piece_and_cameras_keep_their_order():
    scene, labels = two_pieces()
    key = [camera.detections.key for camera in scene.cameras]
    nearest = solve_stage.component_keys_at(scene, labels, 0, 20, count=2, tolerance=8)
    assert nearest == [key[0](10), key[1](10)], "larger box first"
    shown = solve_stage.component_keys_at(scene, labels, 0, 20, count=2, tolerance=8, clear=CLEAR)
    assert shown == nearest, "piece 0 has no clear view: a clear frame of piece 1 is another piece, not a better frame"
    # once the clear views are part of the same piece, the nearest of them is shown - same camera, same place in the order
    labels.atom[0][:] = 0
    shown = solve_stage.component_keys_at(scene, labels, 0, 20, count=2, tolerance=8, clear=CLEAR)
    assert shown == [key[0](12), key[1](10)]
    # one camera asked for: still the one with the larger box, although only the other has a clear view at t
    labels.atom[0][:] = np.where(np.arange(10, 30) < 22, 0, 1)
    assert solve_stage.component_keys_at(scene, labels, 0, 20, count=1, tolerance=8, clear=CLEAR) == [key[0](10)]


def test_evenly_spaced_images_of_a_piece_move_to_a_clear_view_nearby():
    entries = [(t, *person(joints=17 if t % 10 == 3 else 6)) for t in range(100)]
    scene = scene_of([camera_of(entries)])
    key = scene.cameras[0].detections.key
    rows = np.arange(100)
    assert solve_stage.representative_keys(scene, 0, rows) == [key(0), key(50), key(99)]
    assert solve_stage.representative_keys(scene, 0, rows, clear=CLEAR) == [key(3), key(53), key(93)]
    assert solve_stage.representative_keys(scene, 0, rows, clear=CLEAR, reach=2) == [key(0), key(50), key(99)]


def test_swap_check_images_stay_on_the_same_tracklet():
    """Just after a crossing the nearest frame is often two people in one box; the clear frame shown instead
    must be of the same single-camera tracklet - another tracklet may be the other person."""
    entries = [(t, *person(joints=6 if t < 4 else 17)) for t in range(12)]
    camera = camera_of(entries)
    camera.tracklet = np.where(np.arange(12) < 6, 0, 1).astype(np.int32)
    scene = scene_of([camera])
    index = SimpleNamespace(scene=scene, rows=lambda who, lo, hi: (np.zeros(12, np.int64), np.arange(12), np.arange(12)))
    key = camera.detections.key
    assert swap_check._keys(index, 0, 0, 11, nearest_to=0) == [key(0)]
    assert swap_check._keys(index, 0, 0, 11, nearest_to=0, clear=CLEAR) == [key(4)], "rows 4 and 5 are clear and of tracklet 0"
    camera.tracklet = np.where(np.arange(12) < 4, 0, 1).astype(np.int32)  # every clear view is on the other tracklet
    assert swap_check._keys(index, 0, 0, 11, nearest_to=0, clear=CLEAR) == [key(0)]
