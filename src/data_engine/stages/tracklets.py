"""2D tracklets: adopted post-filter + BoT-SORT per camera, then deliberate over-splitting.

The detector output is filtered with cvat_labeling's adopted post-filter
(``detection_postprocess.keep_indices``: score floor, near-identical-pose suppression) and
tracked with the adopted 10 fps BoT-SORT parameters. Both are read from
``prelabel_policy.json`` and must equal the engine recipe, so a policy change is a visible,
deliberate event rather than a silent drift.

Tracker ids are only proposals. A tracklet here is a run of one tracker id that is cut
wherever the tracker bridged a gap or the box teleported: a wrong cut is cheap (the solve
stage re-joins pieces with 3D evidence), a missed identity switch is expensive (it merges
two people). Filtered and untracked detections are kept in the raw store with
``tracklet = -1``; they are evidence for later recovery, never deleted.

Artifact ``tracklets/<camera>.npz`` (rows parallel to the raw detection rows of the camera):
    kept      (M,) bool   survived the post-filter
    tracklet  (M,) int32  global tracklet id, -1 if none
    tracker   (M,) int32  raw BoT-SORT id (provenance only)
and ``tracklets/index.npz``: per tracklet id -> camera index, first/last native frame, size.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..config import RECIPE, load_site
from ..engine import Context, StageResult
from ..store import load_camera
from ..workspace import file_sha256

ID_STRIDE = 1_000_000  # global tracklet id = camera index * stride + local index


def adopted_policy(cvat_labeling: Path) -> tuple[dict, dict, str]:
    """Post-filter and tracker settings from the approved recipe; refuse if they diverge from RECIPE."""
    path = cvat_labeling / "prelabel_policy.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    postfilter, tracking = policy["detection_postprocess"], policy["association_tracking"]["parameters"]
    recipe_filter, recipe_tracking = RECIPE["postfilter"], RECIPE["tracklets"]
    pose = postfilter["pose_suppression"]
    mismatches = [name for name, mine, theirs in [
        ("score_floor", recipe_filter["score_floor"], postfilter["score_floor"]),
        ("pose_tau", recipe_filter["pose_tau"], pose["tau"]),
        ("pair_iou", recipe_filter["pair_iou"], pose["gate_iou"]),
        ("pair_containment", recipe_filter["pair_containment"], pose["gate_containment"]),
        ("joint_peak", recipe_filter["joint_peak"], pose["min_raw_peak"]),
        ("min_common_joints", recipe_filter["min_common_joints"], pose["min_common_body_joints"]),
        *[(key, recipe_tracking[key], tracking[key]) for key in tracking if key != "model"],
    ] if mine != theirs]
    if mismatches:
        raise RuntimeError(f"Engine recipe differs from prelabel_policy.json for: {mismatches}; "
                           "adopt the change explicitly in data_engine.config.RECIPE")
    return postfilter, tracking, file_sha256(path)


def local_assignments(tracked: np.ndarray, proposals: np.ndarray, high: float, low: float) -> dict[int, int]:
    """Map BoT-SORT output rows back to source proposal rows.

    This Ultralytics version returns ordinals within the high/low score subgroups rather
    than source indices (cvat_labeling association_inference.local_assignments). A source
    row claimed by more than one tracker id is left unassigned.
    """
    high_rows = np.flatnonzero(proposals[:, 4] >= high)
    low_rows = np.flatnonzero((proposals[:, 4] > low) & (proposals[:, 4] < high))
    candidates: dict[int, set[int]] = {}
    for row in tracked:
        ordinal, track_id = int(row[-1]), int(row[4])
        source = high_rows if row[5] >= high else low_rows
        if not 0 <= ordinal < len(source):
            raise ValueError("BoT-SORT returned an invalid source detection index")
        index = int(source[ordinal])
        if np.float32(row[5]) != np.float32(proposals[index, 4]):
            raise ValueError("BoT-SORT subgroup index does not identify the original detection")
        candidates.setdefault(index, set()).add(track_id)
    return {index: next(iter(ids)) for index, ids in candidates.items() if len(ids) == 1}


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    return float(intersection / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection))


def track_camera(args: tuple) -> dict:
    """Filter and track one camera. Runs in a worker process."""
    (detections_dir, camera, frame_count, height, width, cvat_labeling, postfilter, tracking, splitting,
     output) = args
    sys.path.insert(0, cvat_labeling)
    from cvat_workflow.detection_postprocess import keep_indices
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.bot_sort import BOTSORT

    detections = load_camera(Path(detections_dir), camera, frame_count)
    tracker = BOTSORT(SimpleNamespace(**{k: v for k, v in tracking.items()}), frame_rate=10)
    kept = np.zeros(len(detections.boxes), dtype=bool)
    tracker_id = np.full(len(detections.boxes), -1, dtype=np.int32)
    high, low = tracking["track_high_thresh"], tracking["track_low_thresh"]
    for frame in range(frame_count):
        if not detections.covered[frame]:
            continue
        rows = detections.rows(frame)
        boxes = detections.boxes[rows]
        keep = keep_indices(boxes, list(detections.keypoints[rows]), postfilter) if len(boxes) else []
        proposals = np.column_stack([boxes[keep, :4], boxes[keep, 4], np.zeros(len(keep))]).astype(np.float32) \
            if len(keep) else np.empty((0, 6), np.float32)
        kept[rows.start + np.asarray(keep, dtype=np.int64)] = True
        tracked = tracker.update(Boxes(proposals, (height, width)))
        for index, track in local_assignments(np.asarray(tracked), proposals, high, low).items():
            tracker_id[rows.start + keep[index]] = track

    # Over-split: cut a tracker id wherever it bridged a gap or the box teleported.
    frame_of_row = detections.frame_of_row
    tracklet = np.full(len(tracker_id), -1, dtype=np.int32)
    next_id = 0
    cuts = {"gap": 0, "jump": 0, "bridged": 0}
    order = np.lexsort((frame_of_row, tracker_id))
    order = order[tracker_id[order] >= 0]
    previous = None
    for row in order:
        box = detections.boxes[row]
        cut = previous is None or tracker_id[row] != tracker_id[previous]
        if not cut:
            gap = int(frame_of_row[row] - frame_of_row[previous])
            last = detections.boxes[previous]
            scale = max(1.0, min(box[3] - box[1], last[3] - last[1]))
            shift = np.hypot((box[0] + box[2] - last[0] - last[2]) / 2, (box[1] + box[3] - last[1] - last[3]) / 2)
            if gap > splitting["split_gap_frames"]:
                # A dropout is bridged only when it is short, the box is where it was, and no
                # other detection could be confused with it (competitor clearance was the
                # best predictor of human-found breaks on the golden set).
                bridged = False
                if gap <= splitting["bridge_gap_frames"] and box_iou(last, box) >= splitting["bridge_iou"]:
                    frame_rows = detections.rows(int(frame_of_row[row]))
                    others = np.flatnonzero(kept[frame_rows]) + frame_rows.start
                    bridged = not any(other != row and box_iou(last, detections.boxes[other])
                                      >= splitting["bridge_competitor_iou"] for other in others)
                if bridged:
                    cuts["bridged"] += 1
                else:
                    cut, cuts["gap"] = True, cuts["gap"] + 1
            elif shift / scale > splitting["split_jump_heights"] * max(1, gap):
                cut, cuts["jump"] = True, cuts["jump"] + 1
        if cut:
            next_id += 1
        tracklet[row] = next_id - 1
        previous = row
    np.savez_compressed(output, kept=kept, tracklet=tracklet, tracker=tracker_id)
    sizes = np.bincount(tracklet[tracklet >= 0], minlength=next_id)
    return {"camera": camera, "detections": int(len(kept)), "kept": int(kept.sum()),
            "tracked": int((tracklet >= 0).sum()), "tracker_ids": int(len(np.unique(tracker_id[tracker_id >= 0]))),
            "tracklets": int(next_id), "cuts": cuts, "short_tracklets": int((sizes < 10).sum())}


def run(context: Context) -> StageResult:
    workspace = context.workspace
    site = workspace.site
    dataset = workspace.dataset
    postfilter, tracking, policy_hash = adopted_policy(site.repo("cvat_labeling"))
    splitting = {key: RECIPE["tracklets"][key] for key in (
        "split_gap_frames", "split_jump_heights", "bridge_gap_frames", "bridge_iou", "bridge_competitor_iou")}
    directory = workspace.path("tracklets")
    directory.mkdir(parents=True, exist_ok=True)
    cameras = sorted(dataset.cameras)
    jobs = [(str(workspace.detections_dir), camera, dataset.cameras[camera].frame_count,
             dataset.cameras[camera].height, dataset.cameras[camera].width, str(site.repo("cvat_labeling")),
             postfilter, tracking, splitting, str(directory / f"{camera}.npz")) for camera in cameras]
    with ProcessPoolExecutor(max_workers=min(len(jobs), 7)) as pool:
        results = list(pool.map(track_camera, jobs))

    # Global tracklet index: ids unique across cameras.
    index_camera, index_first, index_last, index_size, index_id = [], [], [], [], []
    for camera_index, camera in enumerate(cameras):
        path = directory / f"{camera}.npz"
        with np.load(path) as data:
            kept, tracklet, tracker = data["kept"], data["tracklet"], data["tracker"]
        tracklet = np.where(tracklet >= 0, tracklet + camera_index * ID_STRIDE, -1).astype(np.int32)
        np.savez_compressed(path, kept=kept, tracklet=tracklet, tracker=tracker)
        detections = workspace.detections.camera(camera)
        frames = detections.frame_of_row
        for tid in np.unique(tracklet[tracklet >= 0]):
            rows = np.flatnonzero(tracklet == tid)
            index_id.append(tid), index_camera.append(camera_index)
            index_first.append(frames[rows[0]]), index_last.append(frames[rows[-1]]), index_size.append(len(rows))
    np.savez_compressed(directory / "index.npz", cameras=np.array(cameras), tracklet=np.array(index_id, np.int32),
                        camera=np.array(index_camera, np.int16), first=np.array(index_first, np.int32),
                        last=np.array(index_last, np.int32), size=np.array(index_size, np.int32))
    totals = {key: int(sum(r[key] for r in results)) for key in ("detections", "kept", "tracked", "tracklets",
                                                                 "tracker_ids", "short_tracklets")}
    metrics = {**totals, "per_camera": {r["camera"]: r for r in results},
               "filtered_fraction": round(1 - totals["kept"] / max(1, totals["detections"]), 4),
               "untracked_kept": totals["kept"] - totals["tracked"]}
    return StageResult(
        metrics=metrics,
        message=f"{totals['tracklets']} tracklets from {totals['tracker_ids']} tracker ids; "
                f"{totals['kept']} of {totals['detections']} detections kept",
        inputs={"prelabel_policy_sha256": policy_hash, "postfilter": postfilter, "tracking": tracking,
                "splitting": splitting},
        outputs={"tracklets": "tracklets"})
