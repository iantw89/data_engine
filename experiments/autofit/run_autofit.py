"""Drive Autofit (HumanBotSort + GlobalTrackMerger) offline from Data Engine detection artifacts.

Run with the AUTOFIT interpreter from the Autofit repo root (flat imports, .env.* read from cwd):

    cd C:/Users/iantw/git/Autofit
    CUDA_VISIBLE_DEVICES= .venv/Scripts/python.exe C:/Users/iantw/git/data_engine/experiments/autofit/run_autofit.py \
        --workspace C:/data_engine_data/<ws> --calibration <pseudo3d json> --variant kept

Only numpy/json are used to read engine artifacts (no data_engine import). CPU only.

Time axis: the engine's sync table. At sync index t camera c shows native frame native[t, c]. A camera gets its
detections at the FIRST t that shows a native frame (same convention as ``solve.scene``: rows carry one ``t``);
repeated frames and missing frames hand the tracker ``None`` for that camera (no evidence, no reset).

Output: <out-root>/<workspace-name>/<variant>/labels.npz with, per camera name:
    raw_<cam>     int32 (rows,)  Autofit tracker id of the detection (-1 none)  [activated tracks, births back-filled]
    global_<cam>  int32 (rows,)  GlobalTrackMerger id (-1 none); trackers never sent to the merger get 1_000_000+id
    t_<cam>       int32 (rows,)  sync index at which the row was fed (-1 not fed)
and the 3D poses: pose_tid, pose_t, pose_state (0 tracked 1 tentative(back-filled) 2 lost), pose_ncam, pose_xyz (N,12,3)
(inches, z up, Autofit JointIndex order), plus per-frame wall times and a verification of reprojection.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

AUTOFIT = Path.cwd()
if not (AUTOFIT / "trackers" / "human_bot_sort.py").is_file():
    sys.exit("run from the Autofit repo root")
sys.path.insert(0, str(AUTOFIT))

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("AUTOFIT_THREADS", "6")))

from camera_utils.pnp_locations import EXTRINSICS_CACHE, INTRINSICS_CACHE  # noqa: E402
from detectors.detector_base import PersonDetections  # noqa: E402
from streaming.rerun_global_tracking import _MIN_TRACK_LENGTH, _SEND_EVERY, _DirectGlobalTrackMerger  # noqa: E402
import trackers.human_bot_sort as human_bot_sort  # noqa: E402
from trackers.human_bot_sort import HumanBotSort  # noqa: E402
from trackers.utils.human_kalman_state import AngularHumanKalmanState, JointIndex  # noqa: E402
from utils import TRACKER_CFG, IterableSimpleNamespace, yaml_load  # noqa: E402
from utils.types import BoundingBox, CameraMeta, TrackerOutput, TrackerTrack, TrackFrameData  # noqa: E402

human_bot_sort.TORCH_DEVICE = "cpu"  # the CUDA setting in .env.local must not be used (shared GPU)

SCALE = 0.625  # native 2048 -> 1280 (pixel_cvat); canvas = pixel_cvat * 1.0 + [0, 160]
COCO_BODY_JOINTS = [
    JointIndex.LEFT_SHOULDER, JointIndex.RIGHT_SHOULDER, JointIndex.LEFT_ELBOW, JointIndex.RIGHT_ELBOW,
    JointIndex.LEFT_WRIST, JointIndex.RIGHT_WRIST, JointIndex.LEFT_HIP, JointIndex.RIGHT_HIP,
    JointIndex.LEFT_KNEE, JointIndex.RIGHT_KNEE, JointIndex.LEFT_ANKLE, JointIndex.RIGHT_ANKLE,
]
# Autofit joint j  <->  COCO keypoint 5 + KEYPOINT_REMAP[j]
JOINT_TO_COCO = [5 + k for k in [6, 7, 0, 1, 2, 3, 4, 5, 8, 9, 10, 11]]
UNMERGED_BASE = 1_000_000


def install_calibration(path: Path, camera_ids: list[int]) -> tuple[str, np.ndarray, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    padding, scale = np.asarray(data["padding"], float), float(data["scale"])
    if data["world_units"] != "inches" or scale != 1.0 or not np.array_equal(padding, [0, 160]):
        raise ValueError("calibration is not on the Autofit 1280x1280 canvas (padding [0,160], scale 1)")
    INTRINSICS_CACHE.clear()
    EXTRINSICS_CACHE.clear()
    for camera in camera_ids:
        item = data["cameras"][str(camera)]
        K, D = np.asarray(item["K"], float), np.asarray(item["D"], float)
        rvec, tvec = np.asarray(item["rvec"], float), np.asarray(item["tvec"], float)
        assert K.shape == (3, 3) and rvec.shape == (3,) and tvec.shape == (3,) and np.isfinite(K).all()
        INTRINSICS_CACHE[camera] = K, D
        EXTRINSICS_CACHE[camera] = rvec.reshape(3, 1), tvec.reshape(3, 1)
    return hashlib.sha256(path.read_bytes()).hexdigest(), padding, scale


def world_pose(track):
    if track.mean is None:
        if track.is_activated:
            return None
        joints = np.asarray(track.tracked_points, dtype=float)
        if joints.shape != (len(COCO_BODY_JOINTS), 3):
            return None
    else:
        joints = AngularHumanKalmanState.state_to_keypoints(np.asarray(track.mean, dtype=float))[0]
    if not np.isfinite(joints).all():
        return None
    return joints[COCO_BODY_JOINTS].astype(float)


def run_global_merger(track_frames, frame_count):
    merger = _DirectGlobalTrackMerger()
    snapshots, sent = [], {}
    for window, start in enumerate(range(0, frame_count, _SEND_EVERY)):
        end = start + _SEND_EVERY - 1
        tracks = []
        for tracker_id, frames in sorted(track_frames.items()):
            window_frames = [item for item in frames if start <= item.frame_idx <= end]
            if len(window_frames) >= _MIN_TRACK_LENGTH:
                tracks.append(TrackerTrack(tracker_id=tracker_id, frames=window_frames))
                sent.setdefault(tracker_id, {})[window] = (window_frames[0].frame_idx, window_frames[-1].frame_idx)
        if tracks:
            merger._run_internal(TrackerOutput(start_frame=start, end_frame=end, tracks=tracks))
        snapshots.append(dict(merger._tracker_to_global))
    return snapshots, sent


def global_of(tracker_id, frame, snapshots, sent):
    windows = sent.get(tracker_id)
    if not windows:
        return UNMERGED_BASE + tracker_id
    window = min(windows, key=lambda w: (max(windows[w][0] - frame, frame - windows[w][1], 0), w))
    return int(snapshots[window][tracker_id])


def load_camera(root: Path, camera: str):
    counts, boxes, keypoints = [], [], []
    frames_total = 0
    for path in sorted((root / camera).glob("[0-9]*.npz")):
        if ".tmp" in path.name:
            continue
        with np.load(path) as shard:
            frames = shard["frames"]
            counts.append((frames, shard["counts"]))
            boxes.append(shard["boxes"])
            keypoints.append(shard["keypoints"])
            frames_total = max(frames_total, int(frames[-1]) + 1 if len(frames) else 0)
    count = np.zeros(frames_total, np.int64)
    covered = np.zeros(frames_total, bool)
    for frames, c in counts:
        count[frames] = c
        covered[frames] = True
    start = np.concatenate(([0], np.cumsum(count)))
    return start, np.concatenate(boxes), np.concatenate(keypoints), covered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--variant", choices=["kept", "score25", "kept25"], default="kept")
    parser.add_argument("--t-start", type=int, default=0)
    parser.add_argument("--t-end", type=int, default=None, help="exclusive sync index")
    parser.add_argument("--out-root", type=Path, default=Path(r"C:\data_engine_data\_scratch\autofit"))
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    ws = args.workspace
    dataset = json.loads((ws / "dataset.json").read_text(encoding="utf-8"))
    with np.load(ws / "alignment" / "sync_table.npz") as table:
        names = [str(c) for c in table["cameras"]]
        native = table["native"].astype(np.int64)
    T = len(native) if args.t_end is None else min(args.t_end, len(native))
    used = [i for i, n in enumerate(names) if (native[:, i] >= 0).any()]
    cal_ids = [int(dataset["cameras"][names[i]]["calibration_id"]) for i in used]
    order = np.argsort(cal_ids)
    used = [used[i] for i in order]
    cal_ids = [cal_ids[i] for i in order]
    cal_hash, padding, scale = install_calibration(args.calibration, cal_ids)
    print("cameras", [names[i] for i in used], "calibration ids", cal_ids, flush=True)

    cams = {}
    for i in used:
        start, boxes, keypoints, covered = load_camera(ws / "detections" / "raw", names[i])
        with np.load(ws / "tracklets" / f"{names[i]}.npz") as data:
            kept = data["kept"]
        assert len(kept) == len(boxes), (names[i], len(kept), len(boxes))
        frame_of_row = np.repeat(np.arange(len(start) - 1, dtype=np.int32), np.diff(start))
        shown = np.flatnonzero(native[:, i] >= 0)
        sync_of_frame = np.full(len(start) - 1, -1, np.int32)
        frames = native[shown, i]
        keep_first = frames < len(sync_of_frame)
        shown, frames = shown[keep_first], frames[keep_first]
        sync_of_frame[frames[::-1]] = shown[::-1]
        t_of_row = sync_of_frame[frame_of_row]
        score = boxes[:, 4]
        if args.variant == "kept":
            use = kept.copy()
        elif args.variant == "score25":
            use = score > 0.25
        else:
            use = kept & (score > 0.25)
        use &= t_of_row >= 0
        rows = np.flatnonzero(use)
        rows = rows[np.argsort(t_of_row[rows], kind="stable")]
        row_start = np.searchsorted(t_of_row[rows], np.arange(len(native) + 1))
        cams[i] = dict(boxes=boxes, keypoints=keypoints, rows=rows, row_start=row_start, t_of_row=t_of_row,
                       covered=covered, sync_of_frame=sync_of_frame, n_rows=len(boxes), frame_count=len(start) - 1)

    camera_meta = [CameraMeta(cid, f"camera_{cid}", 0, (1280, 960)) for cid in cal_ids]
    config = dict(yaml_load(TRACKER_CFG))
    config["with_reid"] = False  # no appearance embeddings in the cache; never treat missing ReID as a match
    tracker = HumanBotSort(IterableSimpleNamespace(**config), cameras=camera_meta, frame_rate=10.0)

    raw = {i: np.full(cams[i]["n_rows"], -1, np.int32) for i in used}
    tfed = {i: np.full(cams[i]["n_rows"], -1, np.int32) for i in used}
    assignments = []  # (cam, row, t, tracker_id)
    births, birth_poses = {}, {}
    track_frames, poses = {}, []
    conflicts = 0
    seen_ref = {}
    frame_times, det_counts = [], []
    repro = []  # reprojection check samples
    t0_all = time.time()
    for t in range(args.t_start, T):
        tick = time.perf_counter()
        detections, sources = [], []
        n_det = 0
        for i in used:
            c = cams[i]
            f = native[t, i]
            lo, hi = c["row_start"][t], c["row_start"][t + 1]
            # None when the camera shows no (new) frame at t; empty detections when it has a frame but no rows
            fresh = f >= 0 and f < c["frame_count"] and bool(c["covered"][f]) and c["sync_of_frame"][f] == t
            if f < 0 or not fresh:
                detections.append(None)
                sources.append(np.empty(0, np.int64))
                continue
            rows = c["rows"][lo:hi]
            boxes_native = c["boxes"][rows]
            kp = c["keypoints"][rows].astype(np.float64).copy()
            kp[:, :, :2] = kp[:, :, :2] * SCALE * scale + padding
            bx = boxes_native[:, :4].astype(np.float64).reshape(-1, 2, 2) * SCALE * scale + padding
            detections.append(PersonDetections(
                [BoundingBox(tuple(bx[k, 0]), tuple(bx[k, 1]), -1, k) for k in range(len(rows))],
                [float(s) for s in boxes_native[:, 4]], [kp[k] for k in range(len(rows))]))
            sources.append(rows)
            tfed[i][rows] = t
            n_det += len(rows)
        tracker.update(detections)
        for track in tracker.tracked_stracks:
            refs = [(used[k], t, int(sources[k][r.detection_idx]))
                    for k, r in enumerate(track.result.camera_tracks) if r.detection_idx is not None]
            pose = world_pose(track)
            if not track.is_activated:
                births.setdefault(track.uuid, []).extend(refs)
                if pose is not None:
                    birth_poses.setdefault(track.uuid, []).append((t, pose, len(refs)))
                continue
            tid = track.track_id
            track_frames.setdefault(tid, []).append(TrackFrameData(
                frame_idx=t, keypoints_3d=track.tracked_points.copy(),
                kalman_state=track.mean.copy() if track.mean is not None else None, camera_detections={}))
            for cam, tt, row in births.pop(track.uuid, []) + refs:
                key = (cam, row)
                if key in seen_ref and seen_ref[key] != tid:
                    conflicts += 1
                    continue
                seen_ref[key] = tid
                raw[cam][row] = tid
                assignments.append((cam, row, tt, tid))
            for tt, earlier, n in birth_poses.pop(track.uuid, []):
                poses.append((tid, tt, 1, n, earlier))
            if pose is not None:
                poses.append((tid, t, 0, len(refs), pose))
                if t % 10 == 0 and refs:
                    for cam, _, row in refs:
                        cid = cal_ids[used.index(cam)]
                        K, D = INTRINSICS_CACHE[cid]
                        rv, tv = EXTRINSICS_CACHE[cid]
                        depth = (cv2.Rodrigues(rv)[0] @ pose.T + tv)[2]
                        proj = cv2.projectPoints(pose, rv, tv, K, D)[0].reshape(-1, 2)
                        k = cams[cam]["keypoints"][row].astype(np.float64)
                        det_xy = k[JOINT_TO_COCO, :2] * SCALE * scale + padding
                        ok = (k[JOINT_TO_COCO, 2] > 0.3) & (depth > 0)
                        if ok.sum() >= 4:
                            repro.append(float(np.median(np.linalg.norm(proj[ok] - det_xy[ok], axis=1))))
        for track in tracker.lost_stracks:
            pose = world_pose(track)
            if track.is_activated and pose is not None:
                poses.append((track.track_id, t, 2, 0, pose))
        pending = {track.uuid for track in tracker.tracked_stracks}
        births = {k: v for k, v in births.items() if k in pending}
        birth_poses = {k: v for k, v in birth_poses.items() if k in pending}
        frame_times.append(time.perf_counter() - tick)
        det_counts.append(n_det)
        if (t + 1) % 100 == 0:
            print(f"t={t + 1}/{T} mean {np.mean(frame_times[-100:]):.2f}s/frame tracks={len(track_frames)}", flush=True)
    wall_tracking = time.time() - t0_all

    tick = time.time()
    snapshots, sent = run_global_merger(track_frames, T)
    merger_s = time.time() - tick
    glob = {i: np.full(cams[i]["n_rows"], -1, np.int32) for i in used}
    for cam, row, tt, tid in assignments:
        glob[cam][row] = global_of(tid, tt, snapshots, sent)

    out = args.out_root / ws.name / (args.variant + args.tag)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {"cameras": np.array([names[i] for i in used])}
    for i in used:
        arrays[f"raw_{names[i]}"] = raw[i]
        arrays[f"global_{names[i]}"] = glob[i]
        arrays[f"t_{names[i]}"] = tfed[i]
    if poses:
        arrays["pose_tid"] = np.array([p[0] for p in poses], np.int32)
        arrays["pose_t"] = np.array([p[1] for p in poses], np.int32)
        arrays["pose_state"] = np.array([p[2] for p in poses], np.int8)
        arrays["pose_ncam"] = np.array([p[3] for p in poses], np.int8)
        arrays["pose_xyz"] = np.array([p[4] for p in poses], np.float32)
    arrays["frame_seconds"] = np.array(frame_times, np.float32)
    arrays["frame_detections"] = np.array(det_counts, np.int32)
    np.savez_compressed(out / "labels.npz", **arrays)
    final = snapshots[-1] if snapshots else {}
    meta = {
        "workspace": ws.name, "variant": args.variant, "t_range": [args.t_start, T], "cameras": [names[i] for i in used],
        "calibration": str(args.calibration), "calibration_sha256": cal_hash, "tracker_config": config,
        "frames": T - args.t_start, "tracking_wall_s": round(wall_tracking, 1), "merger_wall_s": round(merger_s, 1),
        "mean_s_per_frame": round(float(np.mean(frame_times)), 4), "p95_s_per_frame": round(float(np.percentile(frame_times, 95)), 4),
        "detections_fed": int(sum(det_counts)), "tracker_ids": len(track_frames), "assigned_detections": len(assignments),
        "duplicate_assignment_conflicts": conflicts, "merged_tracker_ids": len(sent),
        "global_ids": len({g for s in snapshots for g in s.values()}), "final_tracker_to_global": {str(k): v for k, v in final.items()},
        "torch_threads": torch.get_num_threads(), "device": human_bot_sort.TORCH_DEVICE,
        "reprojection_px_median_of_medians": round(float(np.median(repro)), 2) if repro else None,
        "reprojection_px_p90": round(float(np.percentile(repro, 90)), 2) if repro else None,
        "reprojection_samples": len(repro),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("tracker_config", "final_tracker_to_global")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
