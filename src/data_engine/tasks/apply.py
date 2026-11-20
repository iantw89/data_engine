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


CLOSED_AS = {"reidentify": "no_predecessor", "swap_check": "neither_candidate", "cross_view_match": "none_of_candidates"}


def _groups(workspace: Workspace, task: dict, round_number: int) -> tuple[int, bool]:
    """Answer model 2 (taxonomy.IMAGE_FLAGS): flags on single images, then a verdict on what remains.

    * someone else      different(image, rest of its group)
    * mixed             not_person(image, problem=mixed_pose), this detection only
    * not a person      not_person(image): this detection - its whole tracklet when every image of its group is flagged
    * unsure            nothing: the image is left out of the answer (too little visible to tell)
    * bad pose          bad_pose(image); does not change the verdict
    Verdicts bind the images that were not taken out. The question is closed (returns decided) when a
    candidate is the same person, when nothing is left of the reference, or when every candidate that
    could be judged is a different person and the engine holds no further candidates back.
    """
    from ..web.validate import answer_mode, flag_groups, other_before

    payload, answer = task["payload"], task["answer"]
    images = answer.get("images") or {}
    groups = flag_groups(payload)  # with the comparison images: they can be flagged like any other
    reference_id = str((payload.get("reference") or {}).get("id", "A"))

    def excluded(key: str) -> str | None:
        return (images.get(key) or {}).get("exclude")

    kept = {gid: [key for key in keys if not excluded(key)] for gid, keys in groups.items()}
    # The reference answer can cover more detections than the crops shown (every piece of a chain).
    reference = [key for key in (payload.get("reference_keys") or groups.get(reference_id, [])) if not excluded(key)]
    created = 0
    for gid, keys in groups.items():
        all_not_person = bool(keys) and all(excluded(key) == "not_person" for key in keys)
        rest = reference if gid == reference_id else kept[gid]
        for key in keys:
            flag = excluded(key)
            if flag == "other_person" and rest:
                _add(workspace, "different", [key], rest, task, round_number, image_flag="other_person", group=gid)
                created += 1
            elif flag == "mixed":
                _add(workspace, "not_person", [key], None, task, round_number, problem="mixed_pose", scope="detection")
                created += 1
            elif flag == "not_person":
                _add(workspace, "not_person", [key], None, task, round_number, problem="not_person",
                     scope="tracklet" if all_not_person else "detection")
                created += 1
            if (images.get(key) or {}).get("bad_pose"):
                _add(workspace, "bad_pose", [key], None, task, round_number)
                created += 1
    mode = answer_mode(payload)

    def unsure_only(gid: str) -> bool:
        """No image of the group is left to judge, and at least one was left out as 'unsure'."""
        return not kept.get(gid) and any(excluded(key) == "unsure" for key in groups.get(gid, []))

    if not reference:
        # Nothing is left of the reference: the flags are the whole answer - unless an image was only left
        # out as unsure, which says nothing about it: the question stays open.
        return created, not unsure_only(reference_id)
    if mode == "person":
        verdict = answer.get("verdict")
        if verdict == "not_person":
            _add(workspace, "not_person", reference, None, task, round_number, problem="not_person")
            return created + 1, True
        if verdict == "person":
            _add(workspace, "undeterminable", reference, None, task, round_number, reason="single_view_person")
            return created + 1, True
        return created, False
    candidates = {str(c["id"]): c for c in payload.get("candidates", [])}
    values = answer.get("candidates") or {}
    decided, judged, same_as_reference = False, [], []
    for candidate_id, candidate in candidates.items():
        keys, verdict = kept.get(candidate_id) or [], values.get(candidate_id)
        if not keys and unsure_only(candidate_id):
            judged.append("unsure")  # its images could not be judged: that is not "different"
            continue
        if not keys:
            decided = decided or mode == "duplicate"  # one of the two poses is gone: nothing left to own
            continue
        judged.append(verdict)
        if mode == "duplicate":
            if verdict == "same":
                better = answer.get("better")
                _add(workspace, "duplicate", reference, keys, task, round_number,
                     better={reference_id: "a", candidate_id: "b"}.get(str(better)))
                created, decided = created + 1, True
            elif verdict == "different":
                _add(workspace, "different", reference, keys, task, round_number)
                created, decided = created + 1, True
        elif verdict == "same":
            _add(workspace, "same", reference, keys, task, round_number)
            created, decided = created + 1, True
            same_as_reference.append(candidate_id)
        elif verdict == "different" and task["type"] == "swap_check" and candidate.get("continuation"):
            # The track did not stay on this person: cut each camera's tracklet at the later detection
            # (a 'split' also keeps the two sides apart) so the solver can re-link the pieces.
            for key in keys:
                _add(workspace, "split", reference, [key], task, round_number, reason="identity swap confirmed")
                created += 1
            decided = True
        elif verdict == "different":
            _add(workspace, "different", reference, keys, task, round_number)
            created += 1
    other = other_before(payload) if task["type"] == "swap_check" else None
    if other is not None and len(same_as_reference) > 1:
        # The reference is both candidates: after the moment the two tracks show one person. What the other
        # track showed before decides whether it is that person tracked twice, or a track that changed person.
        before = answer.get("other_before")
        before_keys = [key for key in other["keys"] if not excluded(key)]  # images taken out say nothing here
        if before == "same" and before_keys:
            _add(workspace, "same", reference, before_keys, task, round_number, reason="one person tracked twice")
            created += 1
        elif before == "different" and before_keys:
            for key in kept.get(other["id"]) or []:
                _add(workspace, "split", before_keys, [key], task, round_number,
                     reason="the other track moved onto the reference's person")
                created += 1
            _add(workspace, "different", reference, before_keys, task, round_number)
            created += 1
    if mode == "match" and not decided and judged and all(verdict == "different" for verdict in judged) \
            and not payload.get("more_candidates"):
        # None of the candidates, and the engine has no others: the question is closed as it stands.
        _add(workspace, "undeterminable", reference, None, task, round_number,
             reason=CLOSED_AS.get(task["type"], "none_of_candidates"))
        created, decided = created + 1, True
    return created, decided


def _reference_candidates(workspace: Workspace, task: dict, round_number: int) -> tuple[int, bool]:
    payload, answer = task["payload"], task["answer"]
    if not any(key in answer for key in ("reference_ok", "none_of_these", "problems")):
        return _groups(workspace, task, round_number)
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
        elif verdict == "different" and task["type"] == "swap_check" and candidates[candidate_id].get("continuation"):
            # The track did not stay on this person: cut each camera's tracklet at the later detection
            # (a 'split' also keeps the two sides apart) so the solver can re-link the pieces.
            for key in keys:
                _add(workspace, "split", reference, [key], task, round_number, reason="identity swap confirmed")
                created += 1
            decided = True
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
                  "swap_check": "neither_candidate",
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
        new = _add(workspace, "same", [clear[0]], clear[1:], task, round_number)
        # A later explicit answer supersedes an earlier one it contradicts: a 'someone else' flag between two
        # of these crops opened this question, and the person now says they are one person throughout.
        for old in workspace.db.constraints("different"):
            if old["source"] == "human" and old["a"] and old["b"] and set(old["a"]) | set(old["b"]) <= set(clear):
                workspace.db.supersede_constraint(old["id"], new)
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
