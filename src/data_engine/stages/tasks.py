"""Tasks stage: the smallest useful set of human questions for the open unknowns.

One task per unknown, highest label volume first, capped per round: answers are applied as
constraints and the next round usually resolves many of the remaining unknowns by itself
(a confirmed link removes competitors elsewhere), so asking everything up front would
waste human time on questions that are about to answer themselves.

A question that a human already answered "unsure" is not asked again with the same
evidence (golden-set learning: resending an unclear page teaches nothing); it stays an
open unknown, marked as needing different evidence.
"""

from __future__ import annotations

import json

from ..engine import Context, StageResult
from ..store import parse_det_key
from ..taxonomy import KINDS, TASK_TYPES

ROUND_BUDGET = {"cross_view_match": 40, "reidentify": 40, "detection_triage": 25, "pose_ownership": 15,
                "swap_check": 30,
                "calibration_review": 10, "alignment_review": 10, "operator_decision": 10, "continuity": 20}
COLORS = ["magenta", "cyan", "yellow", "orange"]


def panel(context: Context, key: str, *, label: str = "", color: str = "green", context_view: bool = True) -> dict:
    camera, frame, _ = parse_det_key(key)
    box, _ = context.workspace.detections.lookup(key)
    return {"camera": camera, "frame": int(frame), "box": [float(v) for v in box[:4]], "highlight": [key],
            "others": True, "project": [], "label": label, "color": color, "context": context_view}


def time_text(context: Context, t: int | None) -> tuple[float | None, str]:
    if t is None:
        return None, ""
    seconds = t / 10.0
    return seconds, f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def base_payload(task_type: str, title: str, why: str, evidence: dict, seconds: float | None) -> dict:
    spec = TASK_TYPES[task_type]
    return {"ui": spec.ui, "title": title, "question": spec.question, "why": why, "evidence": evidence,
            "time_seconds": seconds}


def answers(payload: dict, mode: str, *, multiple: bool = False, more_candidates: int = 0, **texts: str) -> None:
    """What the person can answer, and what each answer will do (shown in the task's status line).

    mode       "match": is each candidate group the same person as the reference group;
               "person": is the reference a real person (no candidates);
               "duplicate": are the two poses one person (then: which pose is better) or two people.
    multiple   several candidates may be the same person
    more_candidates   candidates the engine holds back: "none of these" then does not close the question
    Every image can be flagged (taxonomy.IMAGE_FLAGS); the answer applies to the images that remain.
    """
    payload.update({"answer_model": 2, "mode": mode, "multiple": multiple, "single_choice": not multiple,
                    "allow_none": False, "more_candidates": int(more_candidates), "texts": texts})


def association_task(context: Context, unknown: dict) -> tuple[str, dict]:
    detail = unknown["detail"]
    seconds, clock = time_text(context, detail.get("t"))
    payload = base_payload(
        "cross_view_match", f"Who is this in {detail['camera']} at {clock}?",
        "Geometry alone could not decide: more than one tracked person is consistent with this detection, "
        "or its only match is not confirmed by the person's other views.",
        {"detections in this piece": detail["detections"], "candidates": len(detail["candidates"])}, seconds)
    payload["reference"] = {"id": "A", "caption": f"{detail['camera']} - the piece in question",
                            "panels": [panel(context, key, label="A") for key in detail["reference_keys"]]}
    payload["candidates"] = []
    for index, candidate in enumerate(detail["candidates"]):
        letter = chr(ord("B") + index)
        facts = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in candidate.items()
                          if k in ("share_within_gate", "samples", "frames_matching", "frames_compared"))
        payload["candidates"].append({
            "id": letter, "track": candidate["identity"],
            "caption": f"person {candidate['identity']} ({candidate['evidence']}; {facts})",
            "panels": [panel(context, key, label=letter, color=COLORS[index % len(COLORS)])
                       for key in candidate["keys"]],
            "keys": candidate["keys"], "component": candidate["component"]})
    payload["reference_keys"] = detail["reference_keys"]
    answers(payload, "match", more_candidates=detail.get("more_candidates", 0),
            same="{ref} is {ids}: this piece joins that person's track.",
            none_closed="{ref} is none of the tracked people shown: it stays a single-camera track of its own.",
            none_more="{ref} is none of these. The engine has {n} more candidate(s) and will ask about them next.")
    return "cross_view_match", payload


def unassigned_task(context: Context, unknown: dict) -> tuple[str, dict]:
    detail = unknown["detail"]
    if detail.get("policy") == "outside_coverage":
        payload = base_payload(
            "operator_decision", f"People outside the 3D-covered area in {detail['camera']}", detail["why"],
            {"pieces": detail["pieces"], "detections": detail["detections"]}, None)
        payload["question"] = ("These people are visible to this camera only, in a region no other camera covers. "
                               "How should the engine treat everyone seen only there?")
        payload["panels"] = [panel(context, key, label=str(index + 1)) for index, key in
                             enumerate(detail["reference_keys"])]
        payload["options"] = [
            {"value": "undeterminable", "label": "Out of scope", "hotkey": "1",
             "hint": "Not part of the labels (kept in the raw store); cannot be measured in 3D"},
            {"value": "single_view_tracks", "label": "Keep as single-camera tracks", "hotkey": "2",
             "hint": "Real people with 2D tracklets only; no 3D positions"},
            {"value": "review_individually", "label": "Ask me about each piece", "hotkey": "3",
             "hint": "Creates one triage question per piece"}]
        payload["policy"] = {"camera": detail["camera"], "kind": "outside_coverage"}
        payload["reference_keys"] = detail["reference_keys"]
        return "operator_decision", payload
    if detail.get("policy_question"):
        payload = base_payload(
            "operator_decision", f"Short single-camera fragments in {detail['camera']}", detail["why"],
            {"detections": detail["detections"]}, None)
        payload["question"] = ("These detections belong to no track and are too short to review one by one. "
                               "How should the engine treat them?")
        payload["panels"] = []
        payload["options"] = [
            {"value": "undeterminable", "label": "Leave out of tracks", "hotkey": "1",
             "hint": "Recorded as undeterminable: kept in the raw store, not part of any track"},
            {"value": "keep_open", "label": "Keep as open unknown", "hotkey": "2",
             "hint": "Do not close; later rounds or retrained models may attach them"}]
        payload["policy"] = {"camera": detail["camera"]}
        return "operator_decision", payload
    seconds, clock = time_text(context, detail.get("t"))
    payload = base_payload(
        "detection_triage", f"Unassigned track in {detail['camera']} at {clock}", detail["why"],
        {"detections in this piece": detail["detections"]}, seconds)
    payload["reference"] = {"id": "A", "caption": f"{detail['camera']} - seen by this camera only",
                            "panels": [panel(context, key, label="A") for key in detail["reference_keys"]]}
    # the answer applies to every piece of the chain, not only the crops shown
    payload.update({"candidates": [], "reference_keys": detail.get("member_keys") or detail["reference_keys"]})
    answers(payload, "person",
            person="A real person seen by this camera only: kept as a single-camera track, without 3D.",
            not_person="Not a person: these detections are taken out of every track (they stay in the raw store).")
    if detail.get("pieces", 1) > 1:
        payload["evidence"]["consecutive pieces at the same place"] = detail["pieces"]
    return "detection_triage", payload


def identity_task(context: Context, unknown: dict) -> tuple[str, dict]:
    detail = unknown["detail"]
    if detail.get("policy") == "short_identities":
        payload = base_payload("operator_decision", "Short 3D tracks without an established person", detail["why"],
                               {"tracks": detail["tracks"], "detections": detail["detections"]}, None)
        payload["question"] = ("These brief 3D tracks could not be tied to an earlier person. "
                               "How should the engine treat them?")
        payload["panels"] = [panel(context, key, label=str(index + 1)) for index, key in
                             enumerate(detail["reference_keys"])]
        payload["options"] = [
            {"value": "leave_unlinked", "label": "Leave as separate short tracks", "hotkey": "1",
             "hint": "Each stays a correct, brief track; no identity is assumed"},
            {"value": "review_individually", "label": "Ask me about each", "hotkey": "2",
             "hint": "One re-identification question per short track"}]
        payload["policy"] = {"kind": "short_identities"}
        return "operator_decision", payload
    seconds, clock = time_text(context, detail.get("t"))
    payload = base_payload(
        "reidentify", f"Person {detail['identity']} appears at {clock} - who are they?", detail["why"],
        {"earlier people who could have reached this spot": detail["feasible_candidates"],
         **({"shown first": "the likeliest: by distance for the time passed, and by appearance where both tracks "
                            "have clear views"} if any("score" in c for c in detail["candidates"]) else {})}, seconds)
    payload["reference"] = {"id": "A", "caption": f"person {detail['identity']}, just after appearing",
                            "panels": [panel(context, key, label="A") for key in detail["reference_keys"]]}
    payload["candidates"] = []
    letter_of = {}
    for index, candidate in enumerate(detail["candidates"]):
        letter = chr(ord("B") + index)
        letter_of[candidate["component"]] = letter
        tracked = ""
        if candidate.get("tracked_t"):
            tracked = f", tracked {time_text(context, candidate['tracked_t'][0])[1]} to {time_text(context, candidate['tracked_t'][1])[1]}"
        looks = ""
        if "score" in candidate or "similarity" in candidate:
            # What the order of the candidates rests on (solve/link_ranking.py): shown, never decisive.
            if candidate.get("similarity") is None:
                looks = "; appearance not compared (no clear view of both)"
            else:
                evidence = candidate.get("appearance", 0)
                word = "alike" if evidence > 1 else "different" if evidence < -1 else "inconclusive"
                where = "same camera" if candidate.get("compared") == "same_camera" else "different cameras"
                looks = f"; appearance {word} ({candidate['similarity']:.2f}, {where})"
        payload["candidates"].append({
            "id": letter, "track": candidate["identity"],
            "ranking": {name: candidate[name] for name in ("score", "motion", "appearance", "similarity", "compared",
                                                           "speed_mps", "clear_views") if name in candidate},
            "caption": f"person {candidate['identity']}, last seen {candidate['gap_s']} s earlier, "
                       f"{candidate['distance_m']} m away{looks}{tracked}"
                       + (" - the only one that fits once recovered positions are known" if candidate.get("proposed_after_recovery") else ""),
            "panels": [panel(context, key, label=letter, color=COLORS[index % len(COLORS)])
                       for key in candidate["keys"]],
            "keys": candidate["keys"], "component": candidate["component"]})
    # Several candidates can be the same person (a track that broke more than once). Only two candidates that
    # are measured at the same time in different places exclude each other.
    payload["cannot_both"] = [
        {"ids": [letter_of[a], letter_of[b]],
         "why": f"they are tracked at the same time for {pair['seconds']} s, {pair['distance_m']} m apart: two people"}
        for pair in detail.get("cannot_both", []) for a, b in [pair["components"]] if a in letter_of and b in letter_of]
    payload["reference_keys"] = detail["reference_keys"]
    answers(payload, "match", multiple=True,
            more_candidates=max(0, int(detail.get("feasible_candidates", 0)) - len(payload["candidates"])),
            same="{ref} is the same person as {ids}: the tracks are joined.",
            none_closed="{ref} is none of the earlier people who could have been here: it stays a separate person.",
            none_more="{ref} is none of these. {n} more earlier people could have reached this spot; the engine "
                      "will ask about them next.")
    return "reidentify", payload


def swap_task(context: Context, unknown: dict) -> tuple[str, dict]:
    detail = unknown["detail"]
    seconds, clock = time_text(context, detail.get("t"))
    a, b = detail["identity"], detail["other"]
    title = {"exchange": f"Tracks {a} and {b} exchanged places at {clock}",
             "ending": f"Track {b} ends next to person {a} at {clock}",
             "crossing": f"People {a} and {b} came very close at {clock}"}[detail["moment"]]
    payload = base_payload("swap_check", title, detail["why"],
                           {"closest distance (m)": detail.get("min_distance_m"), **{
                               key.replace("_", " "): value for key, value in detail["cues"].items() if value is not None}},
                           seconds)
    payload["question"] = (f"Reference A is person {a} just before. Which candidate shows the same person just "
                           "after? Both, if both show this person; neither, if neither does.")
    payload["reference"] = {"id": "A", "caption": f"person {a}, just before",
                            "panels": [panel(context, key, label="A") for key in detail["reference_keys"]]}
    payload["candidates"] = [{
        "id": "B", "track": a, "caption": f"labelled person {a} just after (no swap)", "keys": detail["after_a_keys"],
        "continuation": True, "if_same": f"No swap: track {a} stays on this person.",
        "panels": [panel(context, key, label="B", color=COLORS[0]) for key in detail["after_a_keys"]]}]
    if detail.get("after_b_keys"):
        payload["candidates"].append({
            "id": "C", "track": b, "caption": f"labelled person {b} just after (a swap if this is A)",
            "keys": detail["after_b_keys"],
            "if_same": f"A swap: this person continues under the label {b}; track {a} is cut at this moment.",
            "panels": [panel(context, key, label="C", color=COLORS[1]) for key in detail["after_b_keys"]]})
    if detail.get("before_b_keys"):
        payload["context_panels"] = [panel(context, key, label=f"person {b} before", color=COLORS[2])
                                     for key in detail["before_b_keys"]]
        # Asked only when A is both candidates: did track b show this person before the moment as well?
        payload["other_before"] = {"track": b, "keys": detail["before_b_keys"]}
    payload["reference_keys"] = detail["reference_keys"]
    answers(payload, "match", multiple=True,
            same="{ref} is {ids}.",
            none_closed=f"{{ref}} is neither: track {a} did not stay on this person and is cut at this moment.")
    return "swap_check", payload


def ownership_task(context: Context, unknown: dict) -> tuple[str, dict]:
    detail = unknown["detail"]
    seconds, clock = time_text(context, detail.get("t"))
    first, second = detail["keys"]
    payload = base_payload("pose_ownership", f"One person or two? {detail['camera']} at {clock}", detail["why"],
                           {"frames where both appear": detail["frames_shared"],
                            "3D distance (in)": detail["median_distance_in"]}, seconds)
    payload["question"] = "Is pose B the same person as pose A (a duplicate detection), or someone else?"
    payload["reference"] = {"id": "A", "caption": "pose A", "panels": [panel(context, first, label="A")]}
    payload["candidates"] = [{"id": "B", "caption": "pose B", "keys": [second], "track": None,
                              "panels": [panel(context, second, label="B", color="magenta")]}]
    payload["reference_keys"] = [first]
    answers(payload, "duplicate",
            same="One person with two poses: the better pose is used, the other is kept as its duplicate.",
            different="Two people: the two poses are kept apart.")
    return "pose_ownership", payload


def purity_task(context: Context, unknown: dict) -> tuple[str, dict]:
    """A reviewer said two crops of one single-camera tracklet show different people: where does it change?"""
    detail = unknown["detail"]
    seconds, clock = time_text(context, detail.get("t"))
    payload = base_payload("continuity", f"Where does this track change person? {detail['camera']} at {clock}",
                           detail["why"], {"detections in between": detail["detections"]}, seconds)
    payload["panels"] = [panel(context, key, label=str(index + 1)) for index, key in enumerate(detail["keys"])]
    payload["reference_keys"] = detail["keys"]
    return "continuity", payload


def flat_facts(detail: dict) -> dict:
    """Scalar facts of an unknown's detail (one level deep) for the evidence line."""
    facts = {}
    for key, value in detail.items():
        if isinstance(value, dict):
            facts.update({k: v for k, v in value.items() if isinstance(v, (int, float, str)) and v != ""})
        elif isinstance(value, (int, float, str)) and key not in ("explanation", "what_would_resolve"):
            facts[key] = value
    return facts


def calibration_task(context: Context, unknown: dict) -> tuple[str, dict]:
    detail = unknown["detail"]
    camera = detail.get("camera") or unknown["subject"].split(":", 1)[1]
    payload = base_payload("calibration_review", f"Calibration of {camera}",
                           detail.get("explanation") or "The camera view differs from the calibration capture.",
                           flat_facts(detail), None)
    payload["question"] = (f"{camera} appears to have moved since calibration. Which calibration should this "
                           "dataset use for it?")
    payload["panels"] = []
    hints = {"use_standard": "Accept the small shift as is",
             "use_corrected": "Apply the rotation measured by the mount check",
             "exclude_camera": "Do not use this camera for 3D"}
    options = detail.get("options") or list(hints)
    payload["options"] = []
    for index, option in enumerate(options):
        value = option["value"] if isinstance(option, dict) else option
        label = option.get("label") if isinstance(option, dict) else None
        payload["options"].append({"value": value, "label": label or value.replace("_", " ").capitalize(),
                                   "hint": hints.get(value, ""), "hotkey": str(index + 1)})
    payload["camera"] = camera
    return "calibration_review", payload


def alignment_task(context: Context, unknown: dict) -> tuple[str, dict]:
    detail = unknown["detail"]
    camera = detail.get("camera") or unknown["subject"].split(":")[1]
    payload = base_payload("operator_decision", f"Time alignment of {camera}",
                           detail.get("explanation") or "The machine alignment could not resolve this camera.",
                           {"reasons": "; ".join(detail.get("reasons", []))} if detail.get("reasons") else {}, None)
    payload["question"] = (f"{camera} has no confirmed time offset here. Review it in the data_management alignment "
                           "view (Recorded datasets > deep dive > Save as human-confirmed), or exclude the camera.")
    payload["panels"] = []
    payload["options"] = [
        {"value": "reviewed", "label": "I saved a human-confirmed alignment", "hotkey": "1",
         "hint": "The next round reads the app_review alignment set"},
        {"value": "exclude_camera", "label": "Exclude this camera for now", "hotkey": "2",
         "hint": "3D uses the remaining cameras; its detections stay single-view"}]
    payload["camera"] = camera
    return "operator_decision", payload


BUILDERS = {"association": association_task, "unassigned": unassigned_task, "identity_link": identity_task,
            "pose_ownership": ownership_task, "calibration": calibration_task, "alignment": alignment_task,
            "identity_swap": swap_task, "tracklet_purity": purity_task}


def evidence_signature(payload: dict) -> str:
    keys = list(payload.get("reference_keys", []))
    for candidate in payload.get("candidates", []):
        keys.extend(candidate.get("keys", []))
    return json.dumps(sorted(keys))


def run(context: Context) -> StageResult:
    db = context.db
    # Questions already answered "unsure" with identical evidence are not repeated.
    unsure = set()
    for task in db.tasks(status="applied"):
        answer = task["answer"] or {}
        values = list((answer.get("candidates") or {}).values()) + [answer.get("choice"), answer.get("verdict")]
        values += [flags.get("exclude") for flags in (answer.get("images") or {}).values() if isinstance(flags, dict)]
        if ("unsure" in values or "keep_open" in values) and "same" not in values:
            unsure.add((task["subject"], evidence_signature(task["payload"])))
    created: dict[str, int] = {}
    skipped_unsure = skipped_budget = failed = 0
    open_unknowns = db.unknowns(status="open")
    live_subjects = {unknown["subject"] for unknown in open_unknowns}
    # An open task whose unknown was resolved by the machine in the meantime is withdrawn.
    superseded = 0
    for task in db.tasks(status="open"):
        if task["subject"] not in live_subjects:
            db.set_task_status(task["id"], "superseded")
            superseded += 1
    # An open task shows the question as the last solve states it: candidates, their track numbers and what
    # can be answered may have changed since the task was created. Answered tasks are never touched.
    refreshed = 0
    by_subject = {unknown["subject"]: unknown for unknown in open_unknowns}
    for task in db.tasks(status="open"):
        unknown, builder = by_subject.get(task["subject"]), None
        if unknown is not None:
            builder = BUILDERS.get(unknown["kind"])
        if builder is None:
            continue
        try:
            task_type, payload = builder(context, unknown)
        except (KeyError, IndexError, ValueError):
            continue
        if task_type != task["type"]:
            continue
        payload["unknown"] = {"id": unknown["id"], "kind": unknown["kind"], "subject": unknown["subject"],
                              "meaning": KINDS[unknown["kind"]].meaning}
        if json.dumps(payload, sort_keys=True) != json.dumps(task["payload"], sort_keys=True):
            db.update_task(task["id"], payload, priority=unknown["severity"])
            refreshed += 1
    budget = dict(ROUND_BUDGET)
    for task in db.tasks(status="open"):
        budget[task["type"]] = budget.get(task["type"], 0) - 1
    asked = {task["subject"] for status in ("open", "answered") for task in db.tasks(status=status)}
    for unknown in open_unknowns:  # ordered by severity
        builder = BUILDERS.get(unknown["kind"])
        if builder is None or unknown["subject"] in asked:
            continue
        try:
            task_type, payload = builder(context, unknown)
        except (KeyError, IndexError, ValueError) as exc:
            failed += 1
            context.log(f"could not build a task for {unknown['subject']}: {exc}", level="warning")
            continue
        if (unknown["subject"], evidence_signature(payload)) in unsure:
            skipped_unsure += 1
            continue
        if budget.get(task_type, 0) <= 0:
            skipped_budget += 1
            continue
        payload["unknown"] = {"id": unknown["id"], "kind": unknown["kind"], "subject": unknown["subject"],
                              "meaning": KINDS[unknown["kind"]].meaning}
        task_id = db.add_task(task_type, unknown["subject"], payload, priority=unknown["severity"],
                              round_number=context.round, unknown_ids=[unknown["id"]])
        if task_id is not None:
            created[task_type] = created.get(task_type, 0) + 1
            budget[task_type] -= 1
    total = sum(created.values())
    metrics = {"created": created, "created_total": total, "withdrawn": superseded, "refreshed": refreshed,
               "held_back_by_round_budget": skipped_budget, "not_repeated_after_unsure": skipped_unsure,
               "open_tasks": len(db.tasks(status="open")), "build_failures": failed}
    message = (f"{total} new tasks ({', '.join(f'{n} {t}' for t, n in created.items()) or 'none'}); "
               + (f"{refreshed} open tasks updated to the current solve; " if refreshed else "")
               + f"{skipped_budget} unknowns held back for a later round")
    return StageResult(metrics=metrics, message=message)
