"""Validate a human answer against the task's UI kind and payload.

Returns the cleaned answer or raises ``AnswerError`` whose message names every problem.
"Unsure" is always a legal, recorded answer and is never rejected.
"""

from __future__ import annotations

from typing import Any

VERDICTS = ("continuous", "break", "unclear")
REFERENCE_VALUES = ("same", "different", "unsure")
MAX_NOTES = 5000


class AnswerError(ValueError):
    pass


def _notes(answer: dict, problems: list[str]) -> str:
    notes = answer.get("notes", "")
    if notes is None:
        return ""
    if not isinstance(notes, str):
        problems.append("notes must be a string")
        return ""
    return notes[:MAX_NOTES]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _strip(payload: dict, answer: dict, problems: list[str]) -> dict:
    count = len(payload.get("panels") or [])
    verdict = answer.get("verdict")
    if verdict not in VERDICTS:
        problems.append(f"verdict must be one of {', '.join(VERDICTS)}")
    first = answer.get("first_different")
    if verdict == "break":
        if not _is_int(first) or not 1 <= first < count:
            problems.append(f"a break needs first_different between 1 and {count - 1} (index of the first crop of a different person)")
    elif first is not None:
        problems.append("first_different must be null unless the verdict is 'break'")
    unclear = answer.get("unclear", [])
    if not isinstance(unclear, list) or not all(_is_int(i) and 0 <= i < count for i in unclear):
        problems.append(f"unclear must be a list of crop indices in 0..{count - 1}")
        unclear = []
    return {"verdict": verdict, "first_different": first if verdict == "break" else None,
            "unclear": sorted(set(unclear)), "notes": _notes(answer, problems)}


IMAGE_EXCLUSIONS = ("other_person", "mixed", "not_person", "unsure")
MODE_BY_TYPE_HINT = {"person": "detection_triage", "duplicate": "pose_ownership"}


def group_keys(payload: dict) -> dict[str, list[str]]:
    """Group id -> detection keys of the images shown in it (reference first)."""
    def keys(panels):
        return [panel["highlight"][0] for panel in panels or [] if panel.get("highlight")]
    groups = {}
    reference = payload.get("reference")
    if reference:
        groups[str(reference.get("id", "A"))] = keys(reference.get("panels"))
    for candidate in payload.get("candidates") or []:
        groups[str(candidate["id"])] = keys(candidate.get("panels")) or list(candidate.get("keys") or [])
    return groups


CONTEXT = "context"  # id of the group of comparison images (a swap check's "other person before")


def context_keys(payload: dict) -> list[str]:
    """Images shown for comparison only. They are one track too, so any of them can be flagged."""
    return [panel["highlight"][0] for panel in payload.get("context_panels") or [] if panel.get("highlight")]


def flag_groups(payload: dict) -> dict[str, list[str]]:
    """Every group whose images can be flagged: reference, candidates and the comparison images."""
    groups = group_keys(payload)
    context = context_keys(payload)
    if context:
        groups[CONTEXT] = context
    return groups


def answer_mode(payload: dict) -> str:
    if payload.get("mode"):
        return payload["mode"]
    if not payload.get("candidates"):
        return "person"
    return "match"


def other_before(payload: dict) -> dict | None:
    """Swap check: the other track just before the moment ({"track", "keys"}), shown for comparison. When the
    reference is both candidates, the answer also says whether that track showed this person before."""
    candidates = payload.get("candidates") or []
    if not any(candidate.get("continuation") for candidate in candidates):
        return None
    given = payload.get("other_before") or {}
    keys = list(given.get("keys") or []) or [panel["highlight"][0] for panel in payload.get("context_panels") or []
                                               if panel.get("highlight")]
    others = [candidate for candidate in candidates if not candidate.get("continuation")]
    if not keys or not others:
        return None
    return {"track": given.get("track", others[0].get("track")), "keys": keys, "id": str(others[0]["id"])}


def several_allowed(payload: dict) -> bool:
    """Several candidates can be the reference's person: a track that broke more than once (re-identification),
    or two tracks on one person (swap check; payloads built before that was allowed do not say so)."""
    if any(candidate.get("continuation") for candidate in payload.get("candidates") or []):
        return True
    return bool(payload.get("multiple", not payload.get("single_choice", True)))


def _groups(payload: dict, answer: dict, problems: list[str]) -> dict:
    """Answer model 2: flags on single images, then a verdict on what remains (taxonomy.IMAGE_FLAGS)."""
    groups = group_keys(payload)
    flagged = flag_groups(payload)
    reference_id = str((payload.get("reference") or {}).get("id", "A"))
    group_of = {key: gid for gid, keys in flagged.items() for key in keys}
    images = answer.get("images") or {}
    if not isinstance(images, dict):
        problems.append("images must be an object {detection key: {exclude, bad_pose}}")
        images = {}
    cleaned_images = {}
    for key, flags in images.items():
        if key not in group_of:
            problems.append(f"'{key}' is not an image of this task")
            continue
        flags = flags if isinstance(flags, dict) else {}
        exclude, bad_pose = flags.get("exclude"), flags.get("bad_pose", False)
        if exclude is not None and exclude not in IMAGE_EXCLUSIONS:
            problems.append(f"image flag must be one of {', '.join(IMAGE_EXCLUSIONS)} (got {exclude!r})")
            exclude = None
        if not isinstance(bad_pose, bool):
            problems.append("bad_pose must be true or false")
            bad_pose = False
        if exclude or bad_pose:
            cleaned_images[key] = {"exclude": exclude,
                                   "bad_pose": bool(bad_pose) and exclude in (None, "other_person", "unsure")}
    kept = {gid: [key for key in keys if not cleaned_images.get(key, {}).get("exclude")] for gid, keys in flagged.items()}
    for key, flags in cleaned_images.items():
        if flags["exclude"] == "other_person" and not kept[group_of[key]]:
            name = "the comparison images" if group_of[key] == CONTEXT else f"group {group_of[key]}"
            problems.append(f"'someone else' needs one image of {name} left to be the person the "
                            "others differ from; flag the others, or use 'not a person' / 'two people in one pose'")
            break
    mode = answer_mode(payload)
    reference_alive = bool(kept.get(reference_id)) if reference_id in groups else True
    candidates = [gid for gid in groups if gid != reference_id]
    values = answer.get("candidates") or {}
    if not isinstance(values, dict):
        problems.append("candidates must be an object {candidate id: same|different|unsure}")
        values = {}
    cleaned_values = {}
    for cid, value in values.items():
        if str(cid) not in candidates:
            problems.append(f"unknown candidate '{cid}'")
    for cid in candidates:
        value = values.get(cid)
        if reference_alive and kept[cid]:
            if value not in REFERENCE_VALUES:
                problems.append(f"candidate {cid} needs an answer: same, different or unsure")
            cleaned_values[cid] = value
        else:
            cleaned_values[cid] = None  # nothing left to compare: no answer is needed, and none is recorded
    same = [cid for cid, value in cleaned_values.items() if value == "same"]
    if len(same) > 1 and not several_allowed(payload):
        problems.append(f"only one candidate can be 'same' (got {', '.join(same)})")
    for pair in payload.get("cannot_both") or []:
        if all(str(cid) in same for cid in pair["ids"]):
            problems.append(f"{' and '.join(map(str, pair['ids']))} cannot both be 'same': {pair['why']}")
    cleaned = {"images": cleaned_images, "candidates": cleaned_values, "notes": _notes(answer, problems)}
    other = other_before(payload)
    if other is not None:
        # Both tracks show the reference's person after the moment: was the other track this person before too?
        # ... asked only while an image of that track before the moment is left to judge.
        before = answer.get("other_before")
        asked = len(same) > 1 and any(not cleaned_images.get(key, {}).get("exclude") for key in other["keys"])
        if asked and before not in REFERENCE_VALUES:
            problems.append(f"{' and '.join(same)} are both the reference: say whether track {other['track']} showed "
                            "this person before the moment as well (other_before: same, different or unsure)")
        cleaned["other_before"] = before if asked else None
    if mode == "person":
        verdict = answer.get("verdict")
        if reference_alive and verdict not in ("person", "not_person", "unsure"):
            problems.append("verdict must be person, not_person or unsure")
        cleaned["verdict"] = verdict if reference_alive else None
    if mode == "duplicate":
        better = answer.get("better")
        if better is not None and str(better) not in (*groups, "equal"):
            problems.append(f"better must be one of {', '.join([*groups, 'equal'])}")
            better = None
        cleaned["better"] = str(better) if better is not None and same else None
    return cleaned


def _reference_candidates(payload: dict, answer: dict, problems: list[str]) -> dict:
    # Answers of model 1 (reference_ok / none_of_these / problems) are still understood: tasks answered
    # before the change keep their meaning.
    if not any(key in answer for key in ("reference_ok", "none_of_these", "problems")):
        return _groups(payload, answer, problems)
    candidates = {str(c["id"]): c for c in payload.get("candidates") or []}
    reference_id = str((payload.get("reference") or {}).get("id", "A"))
    reference_ok = answer.get("reference_ok", True)
    if not isinstance(reference_ok, bool):
        problems.append("reference_ok must be true or false")
        reference_ok = True
    values = answer.get("candidates", {})
    if not isinstance(values, dict):
        problems.append("candidates must be an object {candidate id: same|different|unsure}")
        values = {}
    for cid, value in values.items():
        if str(cid) not in candidates:
            problems.append(f"unknown candidate '{cid}'")
        elif value not in REFERENCE_VALUES:
            problems.append(f"candidate '{cid}' must be one of same, different, unsure (got {value!r})")
    none_of = answer.get("none_of_these", False)
    if not isinstance(none_of, bool):
        problems.append("none_of_these must be true or false")
        none_of = False
    if none_of and not payload.get("allow_none", True):
        problems.append("this task does not allow 'none of these'")
    if reference_ok:
        missing = [cid for cid in candidates if cid not in {str(k) for k in values}]
        if missing:
            problems.append(f"every candidate needs an explicit value; missing: {', '.join(missing)}")
        same = [cid for cid, v in values.items() if v == "same"]
        if payload.get("single_choice") and len(same) > 1:
            problems.append(f"only one candidate can be 'same' (got {', '.join(map(str, same))})")
        for pair in payload.get("cannot_both") or []:
            if all(str(cid) in {str(s) for s in same} for cid in pair["ids"]):
                problems.append(f"{' and '.join(map(str, pair['ids']))} cannot both be 'same': {pair['why']}")
        if none_of and same:
            problems.append("'none of these' contradicts a candidate marked 'same'")
    else:
        none_of = False
    problem_values = answer.get("problems", {})
    allowed = payload.get("problem_kinds")
    if not isinstance(problem_values, dict):
        problems.append("problems must be an object {candidate id: kind}")
        problem_values = {}
    for cid, kind in problem_values.items():
        if str(cid) not in candidates and str(cid) != reference_id:
            problems.append(f"problem given for unknown id '{cid}'")
        elif not isinstance(kind, str) or not kind:
            problems.append(f"problem kind for '{cid}' must be a non-empty string")
        elif allowed and kind not in allowed:
            problems.append(f"problem kind '{kind}' not allowed (use {', '.join(allowed)})")
    return {"reference_ok": reference_ok,
            "candidates": {str(k): v for k, v in values.items() if str(k) in candidates},
            "none_of_these": none_of, "problems": {str(k): v for k, v in problem_values.items()},
            "notes": _notes(answer, problems)}


def _decision(payload: dict, answer: dict, problems: list[str]) -> dict:
    allowed = [str(o["value"]) for o in payload.get("options") or []]
    choice = answer.get("choice")
    if choice != "unsure" and str(choice) not in allowed:
        problems.append(f"choice must be one of {', '.join(allowed + ['unsure'])}")
    return {"choice": choice, "notes": _notes(answer, problems)}


def _frame_picker(payload: dict, answer: dict, problems: list[str]) -> dict:
    allowed = [str(c["id"]) for c in payload.get("candidates") or []] + ["none", "unsure"]
    choice = answer.get("choice")
    if str(choice) not in allowed:
        problems.append(f"choice must be one of {', '.join(allowed)}")
    return {"choice": choice, "notes": _notes(answer, problems)}


def _cvat(payload: dict, answer: dict, problems: list[str]) -> dict:
    if answer.get("done") is not True:
        problems.append("a CVAT task is answered with {\"done\": true}")
    return {**{k: v for k, v in answer.items() if k != "notes"}, "done": True, "notes": _notes(answer, problems)}


VALIDATORS = {"strip": _strip, "reference_candidates": _reference_candidates, "decision": _decision,
              "frame_picker": _frame_picker, "cvat": _cvat}


def validate_answer(ui: str, payload: dict, answer: Any) -> dict:
    if ui not in VALIDATORS:
        raise AnswerError(f"unknown ui kind '{ui}'")
    if not isinstance(answer, dict):
        raise AnswerError("answer must be a JSON object")
    problems: list[str] = []
    cleaned = VALIDATORS[ui](payload, answer, problems)
    if problems:
        raise AnswerError("; ".join(problems))
    return cleaned
