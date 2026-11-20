"""An image flagged "someone else" leaves its track - what about its neighbours in the same single-camera tracklet?

``stages/solve.py: left_behind`` asks where the tracklet changes person; ``pipeline.unit_cuts`` makes the
answer's cut land on the piece that holds the detection.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from test_link_ranking import camera_of, person, scene_of

from data_engine.solve.pipeline import Constraints, unit_cuts
from data_engine.stages import solve as solve_stage
from data_engine.stages import tasks as stage_tasks


def situation(labels_0, *, cuts=None, tracklet=None):
    """Camera 0: one tracker tracklet over t=0..119 (one detection per frame). Camera 1: the track's person.
    ``labels_0``: component of every detection of camera 0. Component 0 is the track; row 60 was flagged."""
    first = camera_of([(t, *person()) for t in range(120)], index=0)
    second = camera_of([(t, *person(x=900.0)) for t in range(120)], index=1)
    first.tracklet = np.full(120, 7, np.int32) if tracklet is None else np.asarray(tracklet, np.int32)
    second.tracklet = np.full(120, 8, np.int32)
    scene = scene_of([first, second])
    labels = SimpleNamespace(label=[np.asarray(labels_0, np.int32), np.zeros(120, np.int32)])
    solved = SimpleNamespace(scene=scene, labels=labels, identity_of_label=np.array([0, 1, 2], np.int32))
    # the reviewer: detection 60 of camera 0 is not the person of the group shown with it (camera 1, t=60)
    said = ([(0, 60)], [(1, 60)])
    constraints = Constraints(different=[said], flagged=[said], cuts=dict(cuts or {}))
    return solved, constraints, scene


def piece_left(start, stop):
    labels = np.zeros(120, np.int32)
    labels[start:stop] = 1  # the flagged detection's piece is somebody else's track now
    return labels


def test_neighbours_of_a_flagged_image_that_stay_in_the_track_raise_a_question():
    solved, constraints, scene = situation(piece_left(50, 70))
    [unknown] = solve_stage.left_behind(solved, constraints)
    detail = unknown["detail"]
    key = scene.cameras[0].detections.key
    assert detail["flagged_key"] == key(60) and detail["keys"][detail["flagged_crop"] - 1] == key(60)
    assert len(detail["keys"]) == 10 and detail["keys"][0] == key(0) and detail["keys"][-1] == key(119)
    assert detail["left_in_track"] == 100 and detail["detections"] == 120 and detail["camera"] == "0"
    assert unknown["subject"] == f"tracklet_purity:{key(0)}|{key(119)}"
    # it is part of the stage's unknowns, once per tracklet
    again = Constraints(different=constraints.different * 2, flagged=constraints.flagged * 2)
    assert len(solve_stage.left_behind(solved, again)) == 1
    # a verdict on whole groups ("A is not B") is not a flag on an image: not this question
    assert solve_stage.left_behind(solved, Constraints(different=constraints.different)) == []


def test_no_question_when_the_whole_tracklet_left_or_the_piece_did_not():
    solved, constraints, _ = situation(piece_left(0, 120))
    assert solve_stage.left_behind(solved, constraints) == []
    solved, constraints, _ = situation(piece_left(57, 120))  # 57 detections before it are still in the track: asked
    assert len(solve_stage.left_behind(solved, constraints)) == 1
    labels = piece_left(58, 120)
    labels[:54] = 2  # only 4 neighbours are still in the track: too few to ask about
    solved, constraints, _ = situation(labels)
    assert solve_stage.left_behind(solved, constraints) == []
    # the flagged detection is still in the track (its piece could not be separated): not this question
    solved, constraints, _ = situation(np.zeros(120, np.int32))
    assert solve_stage.left_behind(solved, constraints) == []


def test_neighbours_beyond_a_confirmed_change_of_person_are_not_asked_about_again():
    """After the answer "the track changes person at t=40", what is before t=40 rightly stays in the track."""
    labels = piece_left(40, 120)
    solved, constraints, _ = situation(labels, cuts={7: [2]})  # window 2 = sync frames 40..59
    assert solve_stage.left_behind(solved, constraints) == []
    solved, constraints, _ = situation(labels)  # without that answer the question is open
    assert len(solve_stage.left_behind(solved, constraints)) == 1
    # only the tracker's own tracklet counts: another tracklet of the camera in the track is a different matter
    tracklet = np.where(np.arange(120) < 40, 3, 7)
    solved, constraints, _ = situation(labels, tracklet=tracklet)
    assert solve_stage.left_behind(solved, constraints) == []


def test_the_question_marks_the_flagged_crop():
    solved, constraints, _ = situation(piece_left(50, 70))
    [unknown] = solve_stage.left_behind(solved, constraints)

    class Detections:
        def lookup(self, key):
            return np.array([10.0, 20.0, 110.0, 220.0, 0.9]), None

    context = SimpleNamespace(workspace=SimpleNamespace(detections=Detections()), round=1)
    task_type, payload = stage_tasks.purity_task(context, unknown)
    assert task_type == "continuity" and payload["ui"] == "strip"
    flagged = unknown["detail"]["flagged_crop"]
    assert payload["panels"][flagged - 1]["label"] == f"{flagged} flagged"
    assert payload["evidence"]["crop you flagged as someone else"] == flagged
    assert payload["evidence"]["detections still in the track"] == 100
    assert "all leave the track" in payload["why"]


def test_a_human_cut_lands_on_the_piece_that_holds_the_detection():
    """The solver's pieces are renumbered; a cut names the tracker's tracklet. Applied under that number it
    would cut an unrelated piece."""
    camera = camera_of([(t, *person()) for t in range(120)], index=0)
    camera.source_tracklet = np.full(120, 7, np.int32)  # the tracker: one tracklet
    camera.tracklet = np.where(np.arange(120) < 50, 0, 1).astype(np.int32)  # the solver: two pieces of it
    other = camera_of([(t, *person()) for t in range(120)], index=1)
    other.source_tracklet = np.full(120, 1, np.int32)
    other.tracklet = np.full(120, 7, np.int32)  # a piece that happens to carry the number 7
    scene = scene_of([camera, other])
    assert unit_cuts(scene, {7: [4]}) == {1: [4]}, "window 4 = frames 80..99 lies in the second piece"
    assert unit_cuts(scene, {7: [1, 4]}) == {0: [1], 1: [4]}
    assert unit_cuts(scene, {7: [0]}) == {}, "a cut where the tracklet starts cuts nothing"
    assert unit_cuts(scene, {}) == {}
    # a scene that was not re-cut keeps the tracker's numbers
    plain = scene_of([camera_of([(t, *person()) for t in range(40)])])
    assert unit_cuts(plain, {0: [1]}) == {0: [1]}
