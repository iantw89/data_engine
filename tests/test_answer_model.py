"""Answers of the group tasks (cross-camera match, re-identification, swap check, triage, pose ownership):
flags on single images, a verdict on what remains, and what the engine does with both."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from synthetic_scene import HAVE_CALIBRATION, SceneBuilder, stationary

from data_engine.config import RECIPE
from data_engine.solve.pipeline import Constraints, solve
from data_engine.stages import solve as solve_stage
from data_engine.stages import tasks as stage_tasks
from data_engine.tasks.apply import apply_task_answer
from data_engine.web.validate import AnswerError, validate_answer
from data_engine.workspace import Workspace

REF = ["0/10/0", "0/20/1", "0/30/0"]
CAND_B = ["1/10/0", "2/12/1"]
CAND_C = ["3/11/2"]


class StubDetections:
    def lookup(self, key):
        return np.array([10.0, 20.0, 110.0, 220.0, 0.9]), None


@pytest.fixture()
def workspace(tmp_path):
    ws = Workspace("x", root=tmp_path)
    yield ws
    ws.db.close()


@pytest.fixture()
def context(workspace):
    return SimpleNamespace(workspace=SimpleNamespace(detections=StubDetections()), round=1, db=workspace.db)


def reidentify(context, feasible=2):
    unknown = {"detail": {"identity": 3, "t": 300, "reference_keys": REF, "feasible_candidates": feasible, "why": "w",
                          "candidates": [{"component": 4, "identity": 1, "gap_s": 3.0, "distance_m": 1.0, "keys": CAND_B},
                                         {"component": 5, "identity": 2, "gap_s": 4.0, "distance_m": 2.0, "keys": CAND_C}]}}
    return stage_tasks.identity_task(context, unknown)


def cross_view(context, more=0):
    unknown = {"detail": {"camera": "0", "detections": 30, "t": 100, "span_t": [90, 120], "reference_keys": REF,
                          "more_candidates": more, "candidates": [
                              {"component": 4, "identity": 1, "evidence": "two-view geometry", "samples": 40, "keys": CAND_B},
                              {"component": 5, "identity": 2, "evidence": "projected 3D pose", "frames_matching": 9, "keys": CAND_C}]}}
    return stage_tasks.association_task(context, unknown)


def triage(context):
    unknown = {"detail": {"camera": "0", "detections": 40, "t": 100, "reference_keys": REF, "why": "single camera",
                          "member_keys": REF + ["0/55/0"]}}
    return stage_tasks.unassigned_task(context, unknown)


def ownership(context):
    unknown = {"detail": {"camera": "0", "t": 100, "keys": ["0/10/0", "0/10/1"], "frames_shared": 12,
                          "median_distance_in": 4.0, "why": "w"}}
    return stage_tasks.ownership_task(context, unknown)


def swap(context):
    unknown = {"detail": {"moment": "crossing", "identity": 7, "other": 9, "t": 100, "min_distance_m": 0.4, "why": "w",
                          "cues": {}, "reference_keys": REF, "after_a_keys": CAND_B, "after_b_keys": CAND_C,
                          "before_b_keys": ["5/9/0"]}}
    return stage_tasks.swap_task(context, unknown)


def applied(workspace, built, answer, subject="s"):
    """Validate the answer as the web server does, store it, apply it; returns (result, constraints)."""
    task_type, payload = built
    cleaned = validate_answer(payload["ui"], payload, answer)
    task_id = workspace.db.add_task(task_type, subject, payload, priority=1.0, round_number=1)
    workspace.db.answer_task(task_id, cleaned, "ian")
    result = apply_task_answer(workspace, workspace.db.task(task_id), 2)
    return result, [(row["kind"], row["a"], row["b"], row["detail"]) for row in workspace.db.constraints()]


def kinds(rows):
    return [(kind, a, b) for kind, a, b, _ in rows]


# --------------------------------------------------------------------------- what the payload offers


def test_payloads_say_what_can_be_answered(context):
    _, match = cross_view(context)
    assert (match["mode"], match["multiple"], match["answer_model"]) == ("match", False, 2)
    _, reid = reidentify(context, feasible=5)
    assert (reid["mode"], reid["multiple"], reid["more_candidates"]) == ("match", True, 3)
    _, person = triage(context)
    assert person["mode"] == "person" and person["candidates"] == [] and "single-camera track" in person["texts"]["person"]
    _, duplicate = ownership(context)
    assert duplicate["mode"] == "duplicate" and len(duplicate["candidates"]) == 1
    _, after = swap(context)
    assert after["mode"] == "match" and "No swap" in after["candidates"][0]["if_same"] and "cut" in after["texts"]["none_closed"]
    assert all("allow_none" in p and p["allow_none"] is False for p in (match, reid, person, duplicate, after))


# --------------------------------------------------------------------------- flags on single images


def test_someone_else_takes_the_image_out_and_the_answer_binds_the_rest(workspace, context):
    answer = {"images": {REF[1]: {"exclude": "other_person"}}, "candidates": {"B": "same", "C": "different"}}
    result, rows = applied(workspace, reidentify(context), answer)
    assert result == {"constraints": 3, "decided": True}
    rest = [REF[0], REF[2]]
    assert kinds(rows) == [("different", [REF[1]], rest), ("same", rest, CAND_B), ("different", rest, CAND_C)]
    assert rows[0][3]["image_flag"] == "other_person"


def test_mixed_and_not_a_person_reject_that_detection_only(workspace, context):
    answer = {"images": {CAND_B[0]: {"exclude": "mixed"}, REF[0]: {"exclude": "not_person"}},
              "candidates": {"B": "same", "C": "unsure"}}
    _, rows = applied(workspace, reidentify(context), answer)
    rejected = [(a, detail["problem"], detail["scope"]) for kind, a, _, detail in rows if kind == "not_person"]
    assert rejected == [([REF[0]], "not_person", "detection"), ([CAND_B[0]], "mixed_pose", "detection")]
    assert ("same", REF[1:], [CAND_B[1]]) in kinds(rows)


def test_a_group_flagged_not_a_person_throughout_rejects_its_tracklets(workspace, context):
    answer = {"images": {key: {"exclude": "not_person"} for key in REF}, "candidates": {}}
    result, rows = applied(workspace, cross_view(context), answer)
    assert result["decided"] is True, "nothing is left of the reference: the flags are the whole answer"
    assert [(kind, detail["scope"]) for kind, _, _, detail in rows] == [("not_person", "tracklet")] * 3


def test_bad_pose_is_reported_and_changes_nothing_else(workspace, context):
    answer = {"images": {REF[0]: {"bad_pose": True}, CAND_C[0]: {"exclude": "other_person", "bad_pose": True}},
              "candidates": {"B": "same", "C": "different"}}
    task_type, payload = reidentify(context)
    with pytest.raises(AnswerError, match="needs one image of group C left"):
        validate_answer(payload["ui"], payload, answer)  # C has one image: 'someone else' than what?
    answer["images"].pop(CAND_C[0])
    _, rows = applied(workspace, (task_type, payload), answer)
    assert kinds(rows) == [("bad_pose", [REF[0]], []), ("same", REF, CAND_B), ("different", REF, CAND_C)]


def test_a_candidate_without_images_needs_no_answer(workspace, context):
    task_type, payload = reidentify(context)
    with pytest.raises(AnswerError, match="candidate C needs an answer"):
        validate_answer(payload["ui"], payload, {"candidates": {"B": "same"}})
    answer = {"images": {CAND_C[0]: {"exclude": "mixed"}}, "candidates": {"B": "different"}}
    cleaned = validate_answer(payload["ui"], payload, answer)
    assert cleaned["candidates"] == {"B": "different", "C": None}
    with pytest.raises(AnswerError, match="not an image of this task"):
        validate_answer(payload["ui"], payload, {"images": {"9/9/9": {"exclude": "mixed"}}, "candidates": {"B": "same", "C": "same"}})


# --------------------------------------------------------------------------- none of the candidates


def test_none_of_the_candidates_closes_the_question_only_when_no_others_exist(workspace, context):
    answer = {"candidates": {"B": "different", "C": "different"}}
    result, rows = applied(workspace, reidentify(context, feasible=2), answer, subject="closed")
    assert result == {"constraints": 3, "decided": True}
    assert kinds(rows)[-1] == ("undeterminable", REF, []) and rows[-1][3]["reason"] == "no_predecessor"

    result, rows = applied(workspace, reidentify(context, feasible=6), answer, subject="more")
    assert result == {"constraints": 2, "decided": False}, "four more candidates exist: they are asked next"
    assert [kind for kind, *_ in rows[3:]] == ["different", "different"]


def test_unsure_keeps_the_question_open_and_records_the_differents(workspace, context):
    result, rows = applied(workspace, cross_view(context), {"candidates": {"B": "different", "C": "unsure"}})
    assert result == {"constraints": 1, "decided": False} and kinds(rows) == [("different", REF, CAND_B)]


def test_a_single_camera_piece_that_is_no_tracked_person_stays_a_single_camera_track(workspace, context):
    result, rows = applied(workspace, cross_view(context), {"candidates": {"B": "different", "C": "different"}})
    assert result["decided"] and rows[-1][0] == "undeterminable" and rows[-1][3]["reason"] == "none_of_candidates"


# --------------------------------------------------------------------------- swap check


def test_swap_answers(workspace, context):
    result, rows = applied(workspace, swap(context), {"candidates": {"B": "same", "C": "different"}}, subject="stay")
    assert result["decided"] and kinds(rows) == [("same", REF, CAND_B), ("different", REF, CAND_C)]
    result, rows = applied(workspace, swap(context), {"candidates": {"B": "different", "C": "different"}}, subject="neither")
    assert result["decided"], "neither: the track is cut here"
    assert kinds(rows)[2:] == [("split", REF, [CAND_B[0]]), ("split", REF, [CAND_B[1]]), ("different", REF, CAND_C)]
    task_type, payload = swap(context)
    with pytest.raises(AnswerError, match="only one"):
        validate_answer(payload["ui"], payload, {"candidates": {"B": "same", "C": "same"}})


# --------------------------------------------------------------------------- triage: is it a person


def test_triage_answers(workspace, context):
    task_type, payload = triage(context)
    with pytest.raises(AnswerError, match="verdict"):
        validate_answer(payload["ui"], payload, {})
    result, rows = applied(workspace, (task_type, payload), {"verdict": "person"}, subject="person")
    assert result["decided"] and kinds(rows) == [("undeterminable", REF + ["0/55/0"], [])]
    assert rows[0][3]["reason"] == "single_view_person", "every piece of the chain, not only the crops shown"
    result, rows = applied(workspace, triage(context), {"verdict": "not_person", "images": {REF[0]: {"bad_pose": True}}},
                           subject="not")
    assert kinds(rows)[1:] == [("bad_pose", [REF[0]], []), ("not_person", REF + ["0/55/0"], [])]
    assert "scope" not in rows[-1][3], "the verdict rejects the whole tracklets, as before"
    result, _ = applied(workspace, triage(context), {"verdict": "unsure"}, subject="unsure")
    assert result == {"constraints": 0, "decided": False}


# --------------------------------------------------------------------------- pose ownership: one person or two


def test_one_person_with_two_poses_can_say_which_pose_is_better(workspace, context):
    a, b = "0/10/0", "0/10/1"
    result, rows = applied(workspace, ownership(context), {"candidates": {"B": "same"}, "better": "B"}, subject="b")
    assert result["decided"] and kinds(rows) == [("duplicate", [a], [b])] and rows[0][3]["better"] == "b"
    _, rows = applied(workspace, ownership(context), {"candidates": {"B": "same"}, "better": "equal"}, subject="equal")
    assert rows[-1][3]["better"] is None
    result, rows = applied(workspace, ownership(context), {"candidates": {"B": "different"}}, subject="two")
    assert result["decided"] and kinds(rows)[-1] == ("different", [a], [b])
    result, rows = applied(workspace, ownership(context), {"images": {b: {"exclude": "mixed"}}, "candidates": {}}, subject="mixed")
    assert result["decided"] and kinds(rows)[-1] == ("not_person", [b], []), "the mixed pose is gone: nothing is left to own"
    task_type, payload = ownership(context)
    with pytest.raises(AnswerError, match="better must be"):
        validate_answer(payload["ui"], payload, {"candidates": {"B": "same"}, "better": "C"})


# --------------------------------------------------------------------------- answers given before image flags existed


def test_answers_of_the_first_model_keep_their_meaning(workspace, context):
    answer = {"reference_ok": True, "candidates": {"B": "different", "C": "different"}, "none_of_these": True}
    task_type, payload = reidentify(context)
    payload = {**payload, "allow_none": True}  # as such tasks were stored
    result, rows = applied(workspace, (task_type, payload), answer)
    assert result["decided"] and rows[-1][0] == "undeterminable" and rows[-1][3]["reason"] == "no_predecessor"


# --------------------------------------------------------------------------- continuity after 'someone else'


def test_continuous_supersedes_the_flag_that_asked(workspace):
    db = workspace.db
    keys = ["0/10/0", "0/20/0", "0/30/0", "0/40/0"]
    old = db.add_constraint("different", [keys[0]], [keys[3]], source="human", round_number=1)
    other = db.add_constraint("different", [keys[0]], ["7/1/0"], source="human", round_number=1)
    payload = {"ui": "strip", "panels": [{"highlight": [key]} for key in keys]}
    task_id = db.add_task("continuity", "s", payload, priority=1.0, round_number=1)
    db.answer_task(task_id, {"verdict": "continuous", "unclear": []}, "ian")
    apply_task_answer(workspace, db.task(task_id), 2)
    active = {row["id"] for row in db.constraints("different")}
    assert old not in active and other in active, "only the contradicted flag is superseded; nothing is deleted"
    assert any(row["id"] == old for row in db.constraints("different", active_only=False))


# --------------------------------------------------------------------------- what the solve does with them


needs_calibration = pytest.mark.skipif(not HAVE_CALIBRATION, reason="standard calibration not available")
quiet = dict(log=lambda *args: None)


def one_person(n=80, cameras=("0", "1", "2")):
    sim = SceneBuilder(n, seed=1)
    person = sim.add_person(stationary((-100, -50), n))
    sim.see(person, list(cameras))
    return sim.build(), person


@needs_calibration
def test_a_bad_pose_gives_no_rays_but_keeps_its_place():
    scene, person = one_person()
    camera, row = scene.endpoint(person, "0", 40)
    constraints = Constraints(bad_pose=[(camera, row)])
    solve_stage.apply_image_answers(scene.scene, constraints)
    assert not scene.scene.cameras[camera].joint_ok[row].any()
    solved = solve(scene.scene, RECIPE, constraints=constraints, **quiet)
    label = int(solved.labels.label[camera][row])
    assert label >= 0, "the detection still belongs to its person"
    at = np.flatnonzero((solved.frames.label == label) & (solved.frames.t == 40))[0]
    assert not (solved.frames.used[at] & (1 << camera)).any(), "its keypoints took no part in the 3D pose"
    assert (solved.frames.used[at - 1] & (1 << camera)).any()
    unknowns = solve_stage.human_unknowns(solved, constraints)
    assert [item["subject"] for item in unknowns["pose_quality"]] == [f"pose_quality:{scene.scene.cameras[camera].detections.key(row)}"]


@needs_calibration
def test_two_crops_of_one_tracklet_marked_as_different_people_ask_where_it_changes():
    scene, person = one_person()
    camera, early = scene.endpoint(person, "0", 5)
    _, late = scene.endpoint(person, "0", 70)
    constraints = Constraints(different=[([(camera, early)], [(camera, late)])])
    solved = solve(scene.scene, RECIPE, constraints=constraints, **quiet)
    purity = solve_stage.human_unknowns(solved, constraints)["tracklet_purity"]
    assert len(purity) == 1
    detail = purity[0]["detail"]
    detections = scene.scene.cameras[camera].detections
    assert detail["keys"][0] == detections.key(early) and detail["keys"][-1] == detections.key(late)
    assert len(detail["keys"]) == 10 and detail["detections"] == 66
    context = SimpleNamespace(workspace=SimpleNamespace(detections=StubDetections()), round=1)
    task_type, payload = stage_tasks.purity_task(context, purity[0])
    assert task_type == "continuity" and payload["ui"] == "strip" and len(payload["panels"]) == 10
    # two crops within one 2 s window cannot be separated by a cut: no question
    _, near = scene.endpoint(person, "0", 12)
    close = Constraints(different=[([(camera, early)], [(camera, near)])])
    assert solve_stage.human_unknowns(solve(scene.scene, RECIPE, constraints=close, **quiet), close)["tracklet_purity"] == []


@needs_calibration
def test_the_worse_pose_of_a_duplicate_does_not_win_the_slot():
    from data_engine.solve.tracks3d import member_table

    scene, person = one_person()
    camera, row = scene.endpoint(person, "0", 40)
    other = row + 1 if scene.scene.cameras[camera].t[row + 1] == 41 else row - 1
    labels = [np.where(c.kept, 0, -1).astype(np.int32) for c in scene.scene.cameras]
    scene.scene.cameras[camera].t[other] = 40  # two detections of camera 0 at one instant, both labelled as the person
    scene.scene.cameras[camera].detections.boxes[row, 4] = 0.9
    scene.scene.cameras[camera].detections.boxes[other, 4] = 0.5
    order = scene.scene.cameras[camera].order
    _, ts, table, _ = member_table(scene.scene, labels)
    assert table[ts == 40][0][camera] == row, "the higher score wins"
    solve_stage.apply_image_answers(scene.scene, Constraints(demoted=[(camera, row)]))
    _, ts, table, _ = member_table(scene.scene, labels)
    assert table[ts == 40][0][camera] == other, "unless a person said the other pose is better"
    assert len(order)
