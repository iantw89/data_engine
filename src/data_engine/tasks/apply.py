"""Turn a human answer into durable constraints.

Rules carried over from the golden-set work:
* an answer binds to the detections that were actually shown (their detection keys), not
  to track numbers, which change whenever a stage is rerun;
* "unsure" is a valid answer and creates no constraint - unknown is not negative;
* a human negative is never bypassed or weakened by later machine evidence; it can only be
  superseded by a later explicit human answer, which keeps the old one for audit;
* nothing is deleted: "not a person" removes a detection from tracks, not from the store.
"""

from __future__ import annotations

from ..workspace import Workspace


def _add(workspace: Workspace, kind: str, a: list, b: list | None, task: dict, round_number: int, **detail) -> int:
    return workspace.db.add_constraint(kind, a, b, source="human", round_number=round_number, task_id=task["id"],
                                       detail={"task_type": task["type"], "answered_by": task.get("answered_by"),
                                               **detail})


def _reference_candidates(workspace: Workspace, task: dict, round_number: int) -> tuple[int, bool]:
    payload, answer = task["payload"], task["answer"]
    reference = list(payload.get("reference_keys") or [])
    created, decided = 0, False
    problems = answer.get("problems") or {}
    if answer.get("reference_ok") is False:
        kind = problems.get("A", "not_person")
        if kind in ("not_person", "wrong_box") or task["type"] == "detection_triage":
            _add(workspace, "not_person", reference, None, task, round_number, problem=kind)
            return 1, True
        # A mixed pose or identity switch in the reference makes every comparison meaningless.
        _add(workspace, "undeterminable", reference, None, task, round_number, reason=kind)
        return 1, True
    candidates = {c["id"]: c for c in payload.get("candidates", [])}
    for candidate_id, verdict in (answer.get("candidates") or {}).items():
        keys = candidates.get(candidate_id, {}).get("keys") or []
        if not keys:
            continue
        if problems.get(candidate_id) == "duplicate" or (task["type"] == "pose_ownership" and verdict == "same"):
            _add(workspace, "duplicate", reference, keys, task, round_number)
            created, decided = created + 1, True
        elif verdict == "same":
            _add(workspace, "same", reference, keys, task, round_number)
            created, decided = created + 1, True
        elif verdict == "different":
            _add(workspace, "different", reference, keys, task, round_number)
            created += 1
        if problems.get(candidate_id) in ("not_person", "wrong_box"):
            _add(workspace, "not_person", keys, None, task, round_number, problem=problems[candidate_id])
            created += 1
    if answer.get("none_of_these") and not decided:
        # The reference is a real person who is none of the candidates: for a re-identification
        # that closes the question (someone not tracked before); for triage it is a single-view track.
        reason = {"reidentify": "no_predecessor", "detection_triage": "single_view_person",
                  "cross_view_match": "none_of_candidates"}.get(task["type"], "none_of_candidates")
        _add(workspace, "undeterminable", reference, None, task, round_number, reason=reason)
        created, decided = created + 1, True
    if task["type"] == "pose_ownership" and (answer.get("candidates") or {}).get("B") == "different":
        decided = True
    return created, decided


def _strip(workspace: Workspace, task: dict, round_number: int) -> tuple[int, bool]:
    payload, answer = task["payload"], task["answer"]
    keys = [panel["highlight"][0] for panel in payload.get("panels", []) if panel.get("highlight")]
    unclear = set(answer.get("unclear") or [])
    clear = [key for index, key in enumerate(keys) if index not in unclear]
    if answer.get("verdict") == "continuous" and len(clear) >= 2:
        _add(workspace, "same", [clear[0]], clear[1:], task, round_number)
        return 1, True
    if answer.get("verdict") == "break" and answer.get("first_different"):
        index = int(answer["first_different"])
        before = [key for i, key in enumerate(keys[:index]) if i not in unclear]
        after = [key for i, key in enumerate(keys[index:], start=index) if i not in unclear]
        if before and after:
            # 'split' cuts the tracklet at the first different crop and keeps the sides apart.
            _add(workspace, "split", before, after, task, round_number, first_different=index)
            return 1, True
    return 0, False


def _decision(workspace: Workspace, task: dict, round_number: int) -> tuple[int, bool]:
    payload, answer = task["payload"], task["answer"]
    choice = answer.get("choice")
    if choice in (None, "unsure"):
        return 0, False
    if task["type"] == "calibration_review":
        _add(workspace, "calibration", [payload["camera"]], None, task, round_number, choice=choice)
        return 1, True
    if task["type"] == "operator_decision":
        policy = payload.get("policy") or {}
        if policy.get("kind") == "short_identities":
            if choice == "leave_unlinked":
                _add(workspace, "undeterminable", ["policy:short_identities"], None, task, round_number,
                     reason="short 3D tracks left as separate tracks by operator decision")
                return 1, True
            if choice == "review_individually":
                _add(workspace, "undeterminable", ["policy:review_short_identities"], None, task, round_number,
                     reason="operator asked for one question per short 3D track")
                return 1, False
            return 0, False
        if policy.get("kind") == "outside_coverage":
            camera = policy["camera"]
            if choice == "review_individually":
                _add(workspace, "undeterminable", [f"policy:review_individually:{camera}"], None, task, round_number,
                     reason="operator asked for piece-by-piece review of people outside the 3D-covered area")
                return 1, False
            if choice in ("undeterminable", "single_view_tracks"):
                _add(workspace, "undeterminable", [f"policy:outside_coverage:{camera}"], None, task, round_number,
                     reason=("out of scope" if choice == "undeterminable" else "kept as single-camera tracks")
                     + ": people seen only outside the 3D-covered area", choice=choice)
                return 1, True
            return 0, False
        if payload.get("policy") and choice == "undeterminable":
            _add(workspace, "undeterminable", [f"policy:short:{payload['policy']['camera']}"], None, task,
                 round_number, reason="short single-camera fragments left out by operator decision")
            return 1, True
        if payload.get("camera") and choice in ("reviewed", "exclude_camera"):
            _add(workspace, "alignment", [payload["camera"]], None, task, round_number, choice=choice)
            return 1, True
        return 0, False  # "keep open": no constraint, the unknown stays open
    if task["type"] == "visibility_check":
        keys = payload.get("reference_keys") or [payload.get("subject_key", task["subject"])]
        if choice in ("occluded", "out_of_view"):
            _add(workspace, "undeterminable", keys, None, task, round_number, reason=choice)
            return 1, True
        return 0, choice == "visible"
    if task["type"] == "audit":
        keys = payload.get("reference_keys") or []
        if choice == "not_person" and keys:
            _add(workspace, "not_person", keys, None, task, round_number)
            return 1, True
        if choice == "wrong_identity" and keys and payload.get("other_keys"):
            _add(workspace, "different", keys, payload["other_keys"], task, round_number)
            return 1, True
        return 0, choice == "correct"
    return 0, False


def apply_task_answer(workspace: Workspace, task: dict, round_number: int) -> dict:
    """Record the constraints of one answered task. Returns what was created."""
    ui = task["payload"].get("ui")
    if ui == "reference_candidates":
        created, decided = _reference_candidates(workspace, task, round_number)
    elif ui == "strip":
        created, decided = _strip(workspace, task, round_number)
    elif ui in ("decision", "frame_picker"):
        created, decided = _decision(workspace, task, round_number)
    else:
        created, decided = 0, bool((task["answer"] or {}).get("done"))
    # Unknowns that are not reported by a machine stage any more are closed there; policy
    # unknowns answered by an operator are closed here so they are not asked again.
    if decided and task["payload"].get("policy"):
        for unknown_id in workspace.db.task_unknown_ids(task["id"]):
            workspace.db.close_unknown(unknown_id, round_number, f"human:{task['id']}", status="undeterminable")
    workspace.db.log(f"task {task['id']} ({task['type']}): {created} constraints"
                     + ("" if decided else "; left open (unsure or no decision)"))
    return {"constraints": created, "decided": decided}
