"""Feeding recovered poses back to Autofit, and deciding when that is worth a lap.

The recover stage reads poses where tracked people must be. A pose that does not coincide with a
detection the detector made is a NEW detection: a real person read from pixels at that frame, confirmed
by triangulation with other cameras. Autofit never saw it. Given these detections (without identities -
Autofit associates them itself) it loses fewer people and starts fewer new track ids, so the solver gets
longer independent tracks: fewer identities to link, fewer questions, and gaps that were between two
identities become gaps inside one track, which the next recovery can close.

A lap (Autofit -> solve -> recover) costs about as much as a round, so it is only run when
  * the recovery produced enough detections Autofit has not seen (count and share of what it was fed),
  * the previous lap, if any, paid off (it removed identities), and
  * the laps of this round are not used up.
``decide`` is that judgement; everything it used is returned and logged.

Detections accumulate: a pose that passed the recovery gate once stays a detection (for the same
alignment and calibration) even if a later solve no longer asks for it, so Autofit's input never shrinks
from lap to lap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FIELDS = ("camera", "frame", "box", "keypoints", "score")


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of one box with many."""
    if not len(b):
        return np.zeros(0)
    width = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    height = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    both = width * height
    return both / np.maximum((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - both, 1e-9)


def empty() -> dict:
    return {"camera": np.zeros(0, "U1"), "frame": np.zeros(0, np.int64), "box": np.zeros((0, 4), np.float32),
            "keypoints": np.zeros((0, 17, 3), np.float32), "score": np.zeros(0, np.float32)}


def new_detections(recovered: dict, cameras: list[dict], *, overlap: float = 0.5) -> dict:
    """The recovered poses that are not a detection of the raw store.

    ``recovered``: the recover stage's arrays (``pose_frame`` (N, C), ``pose_box``, ``pose_keypoints``).
    ``cameras``: per camera column {"name", "frame_of_row", "boxes" (M, >=4), "kept" (M,) bool}.
    A pose whose box overlaps a kept detection of its frame (IoU >= ``overlap``) is that detection.
    """
    out = {name: [] for name in FIELDS}
    for column, camera in enumerate(cameras):
        frames = recovered["pose_frame"][:, column].astype(np.int64)
        kept = np.flatnonzero(camera["kept"])
        kept = kept[np.argsort(camera["frame_of_row"][kept], kind="stable")]
        frame_sorted = camera["frame_of_row"][kept]
        for index in np.flatnonzero(frames >= 0):
            box = recovered["pose_box"][index, column].astype(np.float64)
            low, high = np.searchsorted(frame_sorted, [frames[index], frames[index] + 1])
            overlaps = box_iou(box, camera["boxes"][kept[low:high], :4].astype(np.float64))
            if len(overlaps) and overlaps.max() >= overlap:
                continue
            keypoints = np.nan_to_num(recovered["pose_keypoints"][index, column].astype(np.float32))
            out["camera"].append(camera["name"])
            out["frame"].append(int(frames[index]))
            out["box"].append(box)
            out["keypoints"].append(keypoints)
            # no detector scored these: the mean heatmap peak of the 12 body joints
            out["score"].append(float(keypoints[5:17, 2].mean()))
    if not out["frame"]:
        return empty()
    return {"camera": np.array(out["camera"]), "frame": np.array(out["frame"], np.int64),
            "box": np.array(out["box"], np.float32), "keypoints": np.array(out["keypoints"], np.float32),
            "score": np.array(out["score"], np.float32)}


def merge(known: dict, found: dict, *, overlap: float = 0.5) -> tuple[dict, int]:
    """``known`` plus the detections of ``found`` that are not in it (same camera and frame, IoU >= overlap).
    Returns the union and how many were added."""
    index: dict[tuple[str, int], list[int]] = {}
    for row, (camera, frame) in enumerate(zip(known["camera"], known["frame"])):
        index.setdefault((str(camera), int(frame)), []).append(row)
    boxes = [box for box in known["box"]]
    add = []
    for row, (camera, frame) in enumerate(zip(found["camera"], found["frame"])):
        key = (str(camera), int(frame))
        rows = index.get(key, [])
        if rows and box_iou(found["box"][row].astype(np.float64), np.array([boxes[r] for r in rows], np.float64)).max() >= overlap:
            continue
        index.setdefault(key, []).append(len(boxes))
        boxes.append(found["box"][row])
        add.append(row)
    if not add:
        return known, 0
    add = np.array(add, np.int64)
    if not len(known["frame"]):
        return {name: found[name][add] for name in FIELDS}, int(len(add))
    return {name: np.concatenate([known[name], found[name][add]]) for name in FIELDS}, int(len(add))


def load(path: Path, signature: dict) -> dict:
    """The detections Autofit was last given, or none when they belong to another alignment or calibration."""
    if not path.is_file():
        return empty()
    with np.load(path) as data:
        if str(data["sync_table"]) != signature["sync_table"] or str(data["calibration"]) != signature["calibration"]:
            return empty()
        return {name: data[name] for name in FIELDS}


def save(path: Path, detections: dict, signature: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **{name: detections[name] for name in FIELDS},
                        sync_table=np.array(signature["sync_table"]), calibration=np.array(signature["calibration"]))
    temporary.replace(path)


def decide(*, new: int, fed: int, identities: int, history: list[dict], laps_this_round: int, recipe: dict) -> dict:
    """Is another Autofit lap worth its cost? Returns {"refresh": bool, "reason": str, ...what it looked at}.

    ``new``: recovered detections Autofit has not seen. ``fed``: detections it was fed in its last run.
    ``identities``: identities of the current solve. ``history``: earlier laps, each
    {"new", "identities_before", "identities_after"} (``identities_after`` None while the lap is unfinished).
    """
    share = new / fed if fed else 0.0
    facts = {"new_detections": int(new), "share_of_autofit_input": round(share, 5), "identities": int(identities),
             "laps_this_round": int(laps_this_round)}
    finished = [lap for lap in history if lap.get("identities_after") is not None]
    last = finished[-1] if finished else None
    gain = None
    if last and last["identities_before"]:
        gain = (last["identities_before"] - last["identities_after"]) / last["identities_before"]
        facts["last_lap"] = {"new_detections": last["new"], "identities": [last["identities_before"], last["identities_after"]],
                             "gain": round(gain, 4)}

    def answer(refresh: bool, reason: str) -> dict:
        return {"refresh": refresh, "reason": reason, **facts}

    if not recipe.get("enabled", True):
        return answer(False, "disabled in the recipe")
    if new < recipe["min_new_detections"] or share < recipe["min_new_share"]:
        return answer(False, f"too few new detections for Autofit ({new}, {share:.2%} of its input; needs "
                             f"{recipe['min_new_detections']} and {recipe['min_new_share']:.2%})")
    if laps_this_round >= recipe["max_laps_per_round"]:
        return answer(False, f"{laps_this_round} laps already run in this round (limit {recipe['max_laps_per_round']}); "
                             "the next round will look again")
    if gain is not None and gain < recipe["min_gain"] and new < 2 * last["new"]:
        return answer(False, f"the last lap removed {gain:.1%} of the identities (needs {recipe['min_gain']:.0%}) and "
                             f"this one has not twice its new detections ({new} vs {last['new']})")
    return answer(True, f"{new} recovered detections Autofit has not seen ({share:.2%} of its input)"
                        + (f"; the last lap removed {gain:.1%} of the identities" if gain is not None else ""))
