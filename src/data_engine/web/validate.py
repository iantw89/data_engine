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


def _reference_candidates(payload: dict, answer: dict, problems: list[str]) -> dict:
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
