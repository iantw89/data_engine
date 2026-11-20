"""Ranking the candidates of a re-identification question (solve/link_ranking.py and its use in the stages).

The ranking orders candidates; it never joins or excludes. Appearance counts only on clear views.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from data_engine.config import RECIPE
from data_engine.solve import link_ranking
from data_engine.solve.identity import ComponentSummary
from data_engine.solve.scene import CameraScene, Scene
from data_engine.stages import solve as solve_stage
from data_engine.stages import tasks as stage_tasks
from data_engine.store import CameraDetections

RANKING = RECIPE["link_ranking"]
SIZE = (1920, 1080)


def person(x=500.0, y=300.0, height=300.0, joints=17, peak=0.9):
    """A box and a skeleton of ``joints`` found keypoints inside it."""
    box = np.array([x, y, x + height / 3, y + height, 0.9], np.float32)
    points = np.zeros((17, 3), np.float32)
    points[:, 0], points[:, 1] = x + height / 6, np.linspace(y + 5, y + height - 5, 17)
    points[:joints, 2] = peak
    return box, points


def camera_of(entries, index=0, frames=400):
    """entries: [(t, box, keypoints)] in time order."""
    t = np.array([entry[0] for entry in entries], np.int32)
    boxes = np.array([entry[1] for entry in entries], np.float32).reshape(-1, 5)
    points = np.array([entry[2] for entry in entries], np.float32).reshape(-1, 17, 3)
    start = np.searchsorted(t, np.arange(frames + 1)).astype(np.int64)
    detections = CameraDetections(str(index), frames, start, boxes, points, np.ones(frames, bool))
    return CameraScene(index, str(index), None, detections, np.ones(len(t), bool), np.zeros(len(t), np.int32), t,
                       start.copy(), np.arange(len(t)))


def quality(camera, rows):
    return link_ranking.view_quality(camera, np.asarray(rows), SIZE, 0.3, RANKING["border_px"])


# --------------------------------------------------------------------------- clear views
def test_a_whole_person_alone_in_the_image_is_a_clear_view():
    camera = camera_of([(0, *person())])
    assert link_ranking.clear_view(quality(camera, [0]), RANKING).tolist() == [True]


@pytest.mark.parametrize("entries, why", [
    ([(0, *person(joints=8))], "partial skeleton"),
    ([(0, *person(height=60))], "too small"),
    ([(0, *person(x=0.0))], "cut by the image border"),
    ([(0, *person(y=790.0))], "cut by the lower border"),
    ([(0, *person()), (0, *person(x=540.0))], "half covered by another person's box"),
])
def test_what_is_not_a_clear_view(entries, why):
    camera = camera_of(entries)
    assert not link_ranking.clear_view(quality(camera, [0]), RANKING)[0], why


def test_another_person_counts_only_at_the_same_instant():
    camera = camera_of([(0, *person()), (1, *person(x=540.0))])
    assert link_ranking.clear_view(quality(camera, [0, 1]), RANKING).tolist() == [True, True]
    assert quality(camera_of([(0, *person()), (0, *person(x=540.0))]), [0])["overlap"][0] > 0.5


def scene_of(cameras):
    workspace = SimpleNamespace(dataset=SimpleNamespace(cameras={
        camera.camera: SimpleNamespace(width=SIZE[0], height=SIZE[1]) for camera in cameras}))
    return Scene(workspace, cameras, np.zeros((400, len(cameras)), np.int32), np.arange(400) / 10.0, 0.3)


def test_window_views_takes_clear_views_spread_over_the_window():
    entries = [(t, *person(joints=17 if t % 2 == 0 else 6)) for t in range(40)]
    scene = scene_of([camera_of(entries)])
    index = link_ranking.ComponentRows(scene, [np.zeros(40, np.int32)])
    views, found = link_ranking.window_views(index, 0, 0, 29, RANKING, [SIZE])
    assert found == {"detections": 30, "clear": 15, "cameras": 1}
    assert len(views[0]) == RANKING["crops_per_camera"] and all(row % 2 == 0 for row in views[0])
    assert views[0][0] == 0 and views[0][-1] == 28, "from the whole window, not its first moments"
    nothing, found = link_ranking.window_views(index, 1, 0, 29, RANKING, [SIZE])
    assert nothing == {} and found["clear"] == 0


# --------------------------------------------------------------------------- appearance
def unit(*values):
    vector = np.array(values, np.float64)
    return vector / np.linalg.norm(vector)


def test_appearance_is_compared_camera_by_camera_where_both_tracks_are_seen():
    first = {0: unit(1, 0, 0), 1: unit(0, 1, 0)}
    second = {0: unit(1, 0.1, 0), 2: unit(0, 0, 1)}
    found = link_ranking.similarity(first, second)
    assert found["mode"] == "same_camera" and found["cameras"] == 1 and found["similarity"] > 0.99
    apart = link_ranking.similarity({1: unit(0, 1, 0)}, {2: unit(0, 1, 0.2)})
    assert apart["mode"] == "cross_camera" and apart["similarity"] > 0.95


def test_without_a_clear_view_appearance_is_unknown_not_different():
    assert link_ranking.similarity({}, {0: unit(1, 0)}) is None
    assert link_ranking.appearance_evidence(None, RANKING) == 0.0
    alike = link_ranking.appearance_evidence({"similarity": 0.7, "mode": "same_camera"}, RANKING)
    unlike = link_ranking.appearance_evidence({"similarity": 0.25, "mode": "same_camera"}, RANKING)
    assert alike > 0 > unlike
    extreme = link_ranking.appearance_evidence({"similarity": 1.0, "mode": "same_camera"}, RANKING)
    assert extreme == RANKING["appearance"]["limit"], "appearance alone is never overwhelming"
    # comparing across cameras says less than comparing within one
    assert link_ranking.appearance_evidence({"similarity": 0.7, "mode": "cross_camera"}, RANKING) < alike


# --------------------------------------------------------------------------- motion
def test_motion_someone_who_stayed_is_likelier_than_someone_across_the_room():
    stayed = link_ranking.motion_evidence(60.0, 0.1, RANKING)
    walked = link_ranking.motion_evidence(60.0, 3.0, RANKING)
    far = link_ranking.motion_evidence(60.0, 7.0, RANKING)
    assert stayed > 3 and stayed > walked > far
    assert walked > -3, "after a minute most of the room is plausible: motion says little"
    assert link_ranking.motion_evidence(2.0, 8.0, RANKING) < -8, "8 m in 2 s: practically excluded"
    # 4 m takes time: walking it in 4 s is plausible, in 1 s it is not
    assert link_ranking.motion_evidence(4.0, 4.0, RANKING) > link_ranking.motion_evidence(1.0, 4.0, RANKING) + 3


def test_score_adds_the_two_and_reports_its_parts():
    near_unlike = link_ranking.score(40.0, 1.5, {"similarity": 0.2, "mode": "same_camera"}, RANKING)
    far_alike = link_ranking.score(40.0, 4.0, {"similarity": 0.75, "mode": "same_camera"}, RANKING)
    unseen = link_ranking.score(40.0, 1.5, None, RANKING)
    assert far_alike["score"] > unseen["score"] > near_unlike["score"]
    assert unseen["appearance"] == 0 and unseen["similarity"] is None and unseen["compared"] is None
    assert far_alike["speed_mps"] == 0.1 and far_alike["compared"] == "same_camera"
    assert far_alike["score"] == pytest.approx(far_alike["motion"] + far_alike["appearance"], abs=2e-3)


# --------------------------------------------------------------------------- in the solve stage
def summary(label, t_first, t_last):
    return ComponentSummary(label, t_first, t_last, t_last - t_first + 1, 50, (0,), 50, np.zeros(3), t_first,
                            np.zeros(3), t_last)


def solved_with(candidate_joints):
    """Track 0 appears at t=200; tracks 1 and 2 ended before, 1 nearer than 2. One camera."""
    entries, label = [], []
    for t in range(100, 130):
        entries.append((t, *person(x=300.0, joints=candidate_joints[1]))), label.append(1)
        entries.append((t, *person(x=900.0, joints=candidate_joints[2]))), label.append(2)
    for t in range(200, 230):
        entries.append((t, *person(x=1400.0))), label.append(0)
    scene = scene_of([camera_of(entries)])
    labels = SimpleNamespace(label=[np.array(label, np.int32)])
    open_ends = [{"component": 0, "end": "start", "t": 200, "candidates": [
        {"component": 1, "gap_s": 7.0, "distance_m": 1.0}, {"component": 2, "gap_s": 7.0, "distance_m": 3.0}]}]
    solved = SimpleNamespace(scene=scene, labels=labels, summaries=[summary(0, 200, 229), summary(1, 100, 129),
                                                                    summary(2, 100, 129)],
                             links=SimpleNamespace(open_ends=open_ends))
    look = {0: unit(1, 0, 0), 1: unit(0, 1, 0), 2: unit(1, 0.05, 0)}  # track 2 looks like track 0, track 1 does not

    def embed(rows_by_camera):
        slot = np.full(len(label), -1, np.int32)
        rows = rows_by_camera.get(0, np.empty(0, np.int64))
        slot[rows] = np.arange(len(rows))
        return [(slot, np.array([look[label[row]] for row in rows]).reshape(-1, 3))]

    return solved, embed


def test_the_look_alike_comes_first_when_both_tracks_have_clear_views():
    solved, embed = solved_with({1: 17, 2: 17})
    ranked = solve_stage.rank_link_candidates(solved, embed, log=lambda *_: None)[0]
    assert [option["component"] for option in ranked] == [2, 1], "nearer, but looks different"
    assert ranked[0]["compared"] == "same_camera" and ranked[0]["similarity"] > 0.9 and ranked[1]["similarity"] < 0.1
    assert ranked[0]["clear_views"] == [30, 30]


def test_partial_skeletons_are_not_compared_and_motion_orders():
    solved, embed = solved_with({1: 17, 2: 6})  # the look-alike is only ever seen in part
    ranked = solve_stage.rank_link_candidates(solved, embed, log=lambda *_: None)[0]
    by_component = {option["component"]: option for option in ranked}
    assert by_component[2]["similarity"] is None and by_component[2]["appearance"] == 0
    assert by_component[2]["clear_views"] == [30, 0]
    assert by_component[1]["similarity"] is not None, "the clear candidate is still compared"
    # without an embedder nothing is compared: the order is motion alone (nearer first here)
    ranked = solve_stage.rank_link_candidates(solved, None, log=lambda *_: None)[0]
    assert [option["component"] for option in ranked] == [1, 2] and ranked[0]["similarity"] is None


def test_images_shown_prefer_clear_views_and_avoid_what_could_not_be_judged():
    entries = [(100, *person(joints=5)), (104, *person())]  # nearest to t=100 is the partial one
    scene = scene_of([camera_of(entries)])
    labels = SimpleNamespace(label=[np.zeros(2, np.int32)])
    key = scene.cameras[0].detections.key
    assert solve_stage.component_keys_at(scene, labels, 0, 100, tolerance=40) == [key(0)]
    assert solve_stage.component_keys_at(scene, labels, 0, 100, tolerance=40, clear=RANKING) == [key(1)]
    assert solve_stage.component_keys_at(scene, labels, 0, 100, tolerance=40, avoid={key(0)}) == [key(1)]
    # nothing else to show: the avoided image is better than none
    assert solve_stage.component_keys_at(scene, labels, 0, 100, tolerance=2, avoid={key(0)}) == [key(0)]


def test_the_candidate_card_says_what_its_place_rests_on():
    class Detections:
        def lookup(self, key):
            return np.array([10.0, 20.0, 110.0, 220.0, 0.9]), None

    context = SimpleNamespace(workspace=SimpleNamespace(detections=Detections()), round=1, db=None)
    candidates = [
        {"component": 4, "identity": 1, "gap_s": 30.0, "distance_m": 4.0, "keys": ["1/10/0"],
         **link_ranking.score(30.0, 4.0, {"similarity": 0.71, "mode": "same_camera"}, RANKING), "clear_views": [12, 9]},
        {"component": 5, "identity": 2, "gap_s": 3.0, "distance_m": 1.0, "keys": ["3/11/2"],
         **link_ranking.score(3.0, 1.0, None, RANKING), "clear_views": [12, 0]}]
    unknown = {"detail": {"identity": 3, "t": 300, "reference_keys": ["0/10/0"], "feasible_candidates": 2, "why": "w",
                          "candidates": candidates}}
    _, payload = stage_tasks.identity_task(context, unknown)
    first, second = payload["candidates"]
    assert "appearance alike (0.71, same camera)" in first["caption"]
    assert "appearance not compared (no clear view of both)" in second["caption"]
    assert first["ranking"]["similarity"] == 0.71 and second["ranking"]["clear_views"] == [12, 0]
