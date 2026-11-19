"""tasks.apply.apply_task_answer: human answers -> durable constraints on the detection keys shown.

Payloads are built with the real builders of ``stages/tasks.py`` (with a stub detection store
behind ``panel``), answers are stored through ``Database.answer_task`` and re-read with
``Database.task`` so the dict handed to ``apply_task_answer`` has exactly the shape the engine
gives it.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from data_engine.db import Database
from data_engine.stages import tasks as stage_tasks
from data_engine.tasks.apply import apply_task_answer
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


def make_task(workspace, task_type, subject, payload, answer, *, unknown_ids=(), answered_by="ian"):
    db = workspace.db
    task_id = db.add_task(task_type, subject, payload, priority=1.0, round_number=1, unknown_ids=unknown_ids)
    db.answer_task(task_id, answer, answered_by)
    return db.task(task_id)


def constraints(workspace, kind=None):
    return workspace.db.constraints(kind)


def association(context, keys=REF):
    unknown = {"detail": {"camera": "0", "detections": 30, "t": 100, "span_t": [90, 120], "reference_keys": keys,
                          "candidates": [
                              {"component": 4, "identity": 1, "evidence": "two-view geometry",
                               "share_within_gate": 0.9, "samples": 40, "keys": CAND_B},
                              {"component": 5, "identity": 2, "evidence": "projected 3D pose",
                               "frames_matching": 9, "frames_compared": 10, "keys": CAND_C}]}}
    return stage_tasks.association_task(context, unknown)


def unassigned(context):
    unknown = {"detail": {"camera": "0", "detections": 40, "t": 100, "reference_keys": REF, "why": "single camera"}}
    return stage_tasks.unassigned_task(context, unknown)


def identity(context):
    unknown = {"detail": {"identity": 3, "t": 300, "reference_keys": REF, "feasible_candidates": 2, "why": "w",
                          "candidates": [{"component": 4, "identity": 1, "gap_s": 3.0, "distance_m": 1.0, "keys": CAND_B},
                                         {"component": 5, "identity": 2, "gap_s": 4.0, "distance_m": 2.0, "keys": CAND_C}]}}
    return stage_tasks.identity_task(context, unknown)


def ownership(context):
    unknown = {"detail": {"camera": "0", "t": 100, "keys": ["0/10/0", "0/10/1"], "frames_shared": 12,
                          "median_distance_in": 4.0, "why": "w"}}
    return stage_tasks.ownership_task(context, unknown)


# --------------------------------------------------------------------------- builders sanity


def test_builder_payloads_carry_the_keys_that_are_shown(context):
    task_type, payload = association(context)
    assert task_type == "cross_view_match" and payload["ui"] == "reference_candidates"
    assert payload["reference_keys"] == REF
    assert [c["id"] for c in payload["candidates"]] == ["B", "C"]
    assert [c["keys"] for c in payload["candidates"]] == [CAND_B, CAND_C]


# --------------------------------------------------------------------------- cross_view_match


def test_same_and_different_bind_to_shown_keys(workspace, context):
    task_type, payload = association(context)
    task = make_task(workspace, task_type, "association:0/10/0", payload, {"reference_ok": True,
                     "candidates": {"B": "same", "C": "different"}})
    result = apply_task_answer(workspace, task, 2)
    assert result == {"constraints": 2, "decided": True}
    rows = constraints(workspace)
    assert [(r["kind"], r["a"], r["b"]) for r in rows] == [("same", REF, CAND_B), ("different", REF, CAND_C)]
    assert all(r["source"] == "human" and r["task_id"] == task["id"] and r["round"] == 2 and r["active"] == 1
               for r in rows)
    assert rows[0]["detail"]["task_type"] == "cross_view_match" and rows[0]["detail"]["answered_by"] == "ian"


def test_unsure_produces_no_constraint(workspace, context):
    task_type, payload = association(context)
    task = make_task(workspace, task_type, "s", payload, {"reference_ok": True,
                     "candidates": {"B": "unsure", "C": "unsure"}})
    assert apply_task_answer(workspace, task, 2) == {"constraints": 0, "decided": False}
    assert constraints(workspace) == []


def test_only_different_is_recorded_but_task_stays_undecided(workspace, context):
    task_type, payload = association(context)
    task = make_task(workspace, task_type, "s", payload, {"candidates": {"B": "different", "C": "unsure"}})
    assert apply_task_answer(workspace, task, 2) == {"constraints": 1, "decided": False}
    assert [(r["kind"], r["b"]) for r in constraints(workspace)] == [("different", CAND_B)]


def test_reference_not_ok_not_person(workspace, context):
    task_type, payload = association(context)
    task = make_task(workspace, task_type, "s", payload, {"reference_ok": False, "problems": {"A": "not_person"},
                     "candidates": {"B": "same"}})
    assert apply_task_answer(workspace, task, 2) == {"constraints": 1, "decided": True}
    [row] = constraints(workspace)
    assert (row["kind"], row["a"], row["b"]) == ("not_person", REF, [])
    assert row["detail"]["problem"] == "not_person"


def test_reference_not_ok_defaults_to_not_person(workspace, context):
    task_type, payload = association(context)
    task = make_task(workspace, task_type, "s", payload, {"reference_ok": False})
    apply_task_answer(workspace, task, 2)
    assert [(r["kind"], r["a"]) for r in constraints(workspace)] == [("not_person", REF)]


@pytest.mark.parametrize("problem", ["mixed_pose", "identity_switch"])
def test_reference_not_ok_mixed_is_undeterminable_not_not_person(workspace, context, problem):
    task_type, payload = association(context)
    task = make_task(workspace, task_type, "s", payload, {"reference_ok": False, "problems": {"A": problem}})
    assert apply_task_answer(workspace, task, 2) == {"constraints": 1, "decided": True}
    [row] = constraints(workspace)
    assert (row["kind"], row["a"], row["detail"]["reason"]) == ("undeterminable", REF, problem)


def test_reference_not_ok_wrong_box_is_not_person(workspace, context):
    task_type, payload = association(context)
    task = make_task(workspace, task_type, "s", payload, {"reference_ok": False, "problems": {"A": "wrong_box"}})
    apply_task_answer(workspace, task, 2)
    assert [(r["kind"], r["detail"]["problem"]) for r in constraints(workspace)] == [("not_person", "wrong_box")]


def test_reference_ok_true_is_not_the_failure_branch(workspace, context):
    task_type, payload = association(context)
    task = make_task(workspace, task_type, "s", payload, {"reference_ok": True, "candidates": {"B": "same"}})
    apply_task_answer(workspace, task, 2)
    assert [r["kind"] for r in constraints(workspace)] == ["same"]


def test_candidate_not_person_problem_marks_candidate_keys(workspace, context):
    task_type, payload = association(context)
    task = make_task(workspace, task_type, "s", payload, {"candidates": {"B": "same", "C": "different"},
                     "problems": {"C": "not_person"}})
    assert apply_task_answer(workspace, task, 2)["constraints"] == 3
    kinds = [(r["kind"], r["a"], r["b"]) for r in constraints(workspace)]
    assert ("same", REF, CAND_B) in kinds and ("different", REF, CAND_C) in kinds
    assert ("not_person", CAND_C, []) in kinds


def test_candidate_duplicate_problem_yields_duplicate(workspace, context):
    task_type, payload = association(context)
    task = make_task(workspace, task_type, "s", payload, {"candidates": {"B": "same"}, "problems": {"B": "duplicate"}})
    assert apply_task_answer(workspace, task, 2) == {"constraints": 1, "decided": True}
    assert [(r["kind"], r["a"], r["b"]) for r in constraints(workspace)] == [("duplicate", REF, CAND_B)]


@pytest.mark.parametrize("task_type,builder,reason", [
    ("cross_view_match", association, "none_of_candidates"),
    ("reidentify", identity, "no_predecessor"),
    ("detection_triage", unassigned, "single_view_person"),
])
def test_none_of_these_is_undeterminable_with_reason(workspace, context, task_type, builder, reason):
    built_type, payload = builder(context)
    assert built_type == task_type
    task = make_task(workspace, task_type, "s", payload, {"none_of_these": True,
                     "candidates": {c["id"]: "different" for c in payload["candidates"]}})
    result = apply_task_answer(workspace, task, 2)
    assert result["decided"] is True
    rows = constraints(workspace)
    assert rows[-1]["kind"] == "undeterminable" and rows[-1]["a"] == REF and rows[-1]["detail"]["reason"] == reason
    assert [r["kind"] for r in rows[:-1]] == ["different"] * len(payload["candidates"])


def test_none_of_these_ignored_when_a_candidate_was_confirmed(workspace, context):
    task_type, payload = association(context)
    task = make_task(workspace, task_type, "s", payload, {"none_of_these": True, "candidates": {"B": "same"}})
    apply_task_answer(workspace, task, 2)
    assert [r["kind"] for r in constraints(workspace)] == ["same"]


def test_unknown_candidate_id_is_ignored(workspace, context):
    task_type, payload = association(context)
    task = make_task(workspace, task_type, "s", payload, {"candidates": {"Z": "same"}})
    assert apply_task_answer(workspace, task, 2) == {"constraints": 0, "decided": False}


# --------------------------------------------------------------------------- reidentify / triage


def test_reidentify_same_links_reference_to_the_chosen_earlier_person(workspace, context):
    task_type, payload = identity(context)
    task = make_task(workspace, task_type, "s", payload, {"candidates": {"B": "different", "C": "same"}})
    apply_task_answer(workspace, task, 2)
    assert [(r["kind"], r["b"]) for r in constraints(workspace)] == [("different", CAND_B), ("same", CAND_C)]


def test_triage_not_a_person_even_for_mixed_pose(workspace, context):
    task_type, payload = unassigned(context)
    task = make_task(workspace, task_type, "s", payload, {"reference_ok": False, "problems": {"A": "mixed_pose"}})
    apply_task_answer(workspace, task, 2)
    assert [r["kind"] for r in constraints(workspace)] == ["not_person"]


# --------------------------------------------------------------------------- pose_ownership


def test_ownership_same_means_duplicate(workspace, context):
    task_type, payload = ownership(context)
    task = make_task(workspace, task_type, "s", payload, {"candidates": {"B": "same"}})
    assert apply_task_answer(workspace, task, 2) == {"constraints": 1, "decided": True}
    [row] = constraints(workspace)
    assert (row["kind"], row["a"], row["b"]) == ("duplicate", ["0/10/0"], ["0/10/1"])


def test_ownership_different_is_decided_with_a_different_constraint(workspace, context):
    task_type, payload = ownership(context)
    task = make_task(workspace, task_type, "s", payload, {"candidates": {"B": "different"}})
    assert apply_task_answer(workspace, task, 2) == {"constraints": 1, "decided": True}
    assert [r["kind"] for r in constraints(workspace)] == ["different"]


def test_ownership_unsure_creates_nothing(workspace, context):
    task_type, payload = ownership(context)
    task = make_task(workspace, task_type, "s", payload, {"candidates": {"B": "unsure"}})
    assert apply_task_answer(workspace, task, 2) == {"constraints": 0, "decided": False}


# --------------------------------------------------------------------------- strip (continuity)


def strip_payload(n=5, camera="0"):
    keys = [f"{camera}/{10 * i}/0" for i in range(n)]
    return ({"ui": "strip", "panels": [{"camera": camera, "frame": 10 * i, "highlight": [k]}
                                       for i, k in enumerate(keys)]}, keys)


def test_strip_continuous_is_same(workspace):
    payload, keys = strip_payload()
    task = make_task(workspace, "continuity", "s", payload, {"verdict": "continuous"})
    assert apply_task_answer(workspace, task, 2) == {"constraints": 1, "decided": True}
    [row] = constraints(workspace)
    assert (row["kind"], row["a"], row["b"]) == ("same", [keys[0]], keys[1:])


def test_strip_continuous_skips_unclear_crops(workspace):
    payload, keys = strip_payload()
    task = make_task(workspace, "continuity", "s", payload, {"verdict": "continuous", "unclear": [0, 3]})
    apply_task_answer(workspace, task, 2)
    [row] = constraints(workspace)
    assert (row["a"], row["b"]) == ([keys[1]], [keys[2], keys[4]])


def test_strip_break_is_split_before_first_different(workspace):
    payload, keys = strip_payload()
    task = make_task(workspace, "continuity", "s", payload, {"verdict": "break", "first_different": 3, "unclear": [1]})
    assert apply_task_answer(workspace, task, 2) == {"constraints": 1, "decided": True}
    [row] = constraints(workspace)
    assert (row["kind"], row["a"], row["b"]) == ("split", [keys[0], keys[2]], keys[3:])
    assert row["detail"]["first_different"] == 3


def test_strip_unclear_verdict_is_open(workspace):
    payload, _ = strip_payload()
    task = make_task(workspace, "continuity", "s", payload, {"verdict": "unclear"})
    assert apply_task_answer(workspace, task, 2) == {"constraints": 0, "decided": False}


def test_strip_break_without_a_before_side_creates_nothing(workspace):
    payload, _ = strip_payload()
    task = make_task(workspace, "continuity", "s", payload, {"verdict": "break", "first_different": 1, "unclear": [0]})
    assert apply_task_answer(workspace, task, 2) == {"constraints": 0, "decided": False}


# --------------------------------------------------------------------------- decisions


def test_calibration_choice_recorded_for_camera(workspace, context):
    task_type, payload = stage_tasks.calibration_task(
        context, {"subject": "calibration:3", "detail": {"camera": "3", "explanation": "moved"}})
    task = make_task(workspace, task_type, "calibration:3", payload, {"choice": "use_corrected"})
    assert apply_task_answer(workspace, task, 2) == {"constraints": 1, "decided": True}
    [row] = constraints(workspace)
    assert (row["kind"], row["a"], row["detail"]["choice"]) == ("calibration", ["3"], "use_corrected")


@pytest.mark.parametrize("choice", ["unsure", None])
def test_calibration_unsure_produces_nothing(workspace, context, choice):
    task_type, payload = stage_tasks.calibration_task(
        context, {"subject": "calibration:3", "detail": {"camera": "3"}})
    task = make_task(workspace, task_type, "calibration:3", payload, {"choice": choice})
    assert apply_task_answer(workspace, task, 2) == {"constraints": 0, "decided": False}
    assert constraints(workspace) == []


def test_alignment_decision(workspace, context):
    task_type, payload = stage_tasks.alignment_task(context, {"subject": "alignment:2", "detail": {"camera": "2"}})
    task = make_task(workspace, task_type, "alignment:2", payload, {"choice": "exclude_camera"})
    assert apply_task_answer(workspace, task, 2) == {"constraints": 1, "decided": True}
    [row] = constraints(workspace)
    assert (row["kind"], row["a"], row["detail"]["choice"]) == ("alignment", ["2"], "exclude_camera")


def short_policy(workspace, context, choice):
    db = workspace.db
    db.report_unknowns("solve", "unassigned", [{"subject": "unassigned:short:0", "severity": 5.0,
                                                "detail": {"camera": "0"}}], 1)
    [unknown] = db.unknowns(kind="unassigned")
    task_type, payload = stage_tasks.unassigned_task(
        context, {"detail": {"camera": "0", "detections": 5, "policy_question": True, "why": "short"}})
    task = make_task(workspace, task_type, "unassigned:short:0", payload, {"choice": choice},
                     unknown_ids=[unknown["id"]])
    return task, unknown


def test_policy_undeterminable_records_constraint_and_closes_unknown(workspace, context):
    task, unknown = short_policy(workspace, context, "undeterminable")
    assert apply_task_answer(workspace, task, 2) == {"constraints": 1, "decided": True}
    [row] = constraints(workspace)
    assert (row["kind"], row["a"]) == ("undeterminable", ["policy:short:0"])
    [closed] = workspace.db.unknowns(status=None, kind="unassigned")
    assert closed["status"] == "undeterminable" and closed["resolution"] == f"human:{task['id']}"


def test_policy_keep_open_leaves_the_unknown_open(workspace, context):
    task, unknown = short_policy(workspace, context, "keep_open")
    apply_task_answer(workspace, task, 2)
    assert constraints(workspace) == []
    [row] = workspace.db.unknowns(status=None, kind="unassigned")
    assert row["status"] == "open"


def test_visibility_check(workspace):
    payload = {"ui": "decision", "reference_keys": REF[:2]}
    task = make_task(workspace, "visibility_check", "v", payload, {"choice": "occluded"})
    assert apply_task_answer(workspace, task, 2) == {"constraints": 1, "decided": True}
    [row] = constraints(workspace)
    assert (row["kind"], row["a"], row["detail"]["reason"]) == ("undeterminable", REF[:2], "occluded")


def test_visibility_visible_and_unsure(workspace):
    payload = {"ui": "decision", "reference_keys": REF[:2]}
    visible = make_task(workspace, "visibility_check", "v1", payload, {"choice": "visible"})
    assert apply_task_answer(workspace, visible, 2) == {"constraints": 0, "decided": True}
    unsure = make_task(workspace, "visibility_check", "v2", payload, {"choice": "unsure"})
    assert apply_task_answer(workspace, unsure, 2) == {"constraints": 0, "decided": False}
    assert constraints(workspace) == []


def test_visibility_falls_back_to_subject_key(workspace):
    task = make_task(workspace, "visibility_check", "2/5/0", {"ui": "decision"}, {"choice": "out_of_view"})
    apply_task_answer(workspace, task, 2)
    [row] = constraints(workspace)
    assert row["a"] == ["2/5/0"]


def test_audit_answers(workspace):
    payload = {"ui": "decision", "reference_keys": REF, "other_keys": CAND_B}
    wrong = make_task(workspace, "audit", "a1", payload, {"choice": "wrong_identity"})
    assert apply_task_answer(workspace, wrong, 2) == {"constraints": 1, "decided": True}
    junk = make_task(workspace, "audit", "a2", payload, {"choice": "not_person"})
    assert apply_task_answer(workspace, junk, 2) == {"constraints": 1, "decided": True}
    correct = make_task(workspace, "audit", "a3", payload, {"choice": "correct"})
    assert apply_task_answer(workspace, correct, 2) == {"constraints": 0, "decided": True}
    assert [(r["kind"], r["a"], r["b"]) for r in constraints(workspace)] == [
        ("different", REF, CAND_B), ("not_person", REF, [])]


def test_audit_wrong_identity_without_other_keys_creates_nothing(workspace):
    task = make_task(workspace, "audit", "a", {"ui": "decision", "reference_keys": REF}, {"choice": "wrong_identity"})
    assert apply_task_answer(workspace, task, 2) == {"constraints": 0, "decided": False}


def test_cvat_ui_reports_done_flag(workspace):
    task = make_task(workspace, "pose_correction", "c", {"ui": "cvat"}, {"done": True})
    assert apply_task_answer(workspace, task, 2) == {"constraints": 0, "decided": True}


def test_apply_logs_each_task(workspace, context):
    task_type, payload = association(context)
    task = make_task(workspace, task_type, "s", payload, {"candidates": {"B": "unsure"}})
    apply_task_answer(workspace, task, 2)
    messages = [row["message"] for row in workspace.db.query("SELECT message FROM events")]
    assert any(f"task {task['id']}" in m and "left open" in m for m in messages)
