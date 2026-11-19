"""Coverage recovery: read poses at the places where tracked people must be, and triangulate them.

After the solve, every track has gaps: instants at which its person was measured shortly before and
after (or just before/after an end of the track) but not there. ``solve/coverage.py`` plans one crop
per camera that should see the person, ``workers/recover_trt.py`` reads a pose from each crop with
the production ViTPose path, and positions are accepted only as gated triangulations of those poses.
A known pose only says where to look; nothing is interpolated.

The stage works in passes: each accepted recovery is a measured pose and becomes the hint for its
neighbours in the next pass, so gaps close from both edges inwards and tracks are followed beyond
where they first appear and last disappear, until a pass finds nothing more.

Outputs, next to the solution they belong to (``rounds/rNNN``):
    tracks3d_recovered.npz   recovered person-frames in the layout of tracks3d.npz, plus ``tier``,
                             ``pass`` and the 2D poses each was triangulated from (published as 2D
                             detections by the publish stage)
    recover_rejected.npz     every instant that was looked at and is still missing, with the reason
    recover.json             counts per pass and per reason
Poses read from crops are cached in ``recovered/poses_cache.npz`` (keyed by camera, frame and box),
so a later pass or round only reads what changed.
"""

from __future__ import annotations

import glob
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .. import autofit_refresh
from ..config import RECIPE
from ..engine import Context, StageResult
from ..solve import coverage
from ..solve.scene import load_scene
from ..workspace import file_sha256
from .ingest import WORKERS, log_tail, run_worker

REASONS = ("no pose read in two cameras", "crops follow another person", "no consistent pose in two cameras",
           "rejected by the acceptance gate")
FEATURES = ("cameras", "body_joints", "median_residual_px", "bad_bones", "hint_distance_in", "weakest_view_peak",
            "other_identity_in")


def latest_solution(root: Path) -> Path | None:
    found = sorted(glob.glob(str(root / "rounds" / "*" / "tracks3d.npz")))
    return Path(found[-1]).parent if found else None


def job_keys(plan: coverage.Plan) -> np.ndarray:
    """(J, 6) identity of a crop request: camera, frame, box rounded to whole pixels."""
    if not len(plan.job_case):
        return np.zeros((0, 6), np.int64)
    return np.concatenate([plan.job_camera[:, None].astype(np.int64), plan.job_frame[:, None],
                           np.rint(plan.job_box).astype(np.int64)], axis=1)


class WorkerFailed(RuntimeError):
    pass


class CropReader:
    """Reads the crops of a plan: from the cache where this exact crop was read before, else with the worker."""

    def __init__(self, context: Context, camera_names: list[str], vitpose: Path, model_sha: str, margin: dict):
        self.context, self.names, self.vitpose, self.model_sha, self.margin = context, camera_names, vitpose, model_sha, margin
        self.path = context.workspace.path("recovered", "poses_cache.npz")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keys = np.zeros((0, 6), np.int64)
        self.keypoints = np.zeros((0, 17, 3), np.float32)
        self.box = np.zeros((0, 4), np.float32)
        if self.path.is_file():
            with np.load(self.path, allow_pickle=False) as data:
                if str(data["model_sha256"]) == model_sha:
                    self.keys, self.keypoints, self.box = data["keys"], data["keypoints"], data["box"]
        self.slot = {tuple(int(v) for v in key): index for index, key in enumerate(self.keys)}
        self.read_now = self.worker_seconds = 0
        self.passes = self.recovered = 0
        self.dirty = False

    def progress(self, message: str) -> None:
        """What the stage is doing now, for the dashboard (the number of passes is not known in advance)."""
        path = self.context.workspace.path("logs", "recover_progress.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"message": f"{message}; {self.recovered} positions recovered so far",
                                    "updated": time.time()}), encoding="utf-8")

    def note(self, message: str) -> None:
        """Log callback of the recovery: counts passes and recovered positions for the progress line."""
        self.context.log(message)
        if message.startswith("pass "):
            self.passes += 1
            self.recovered += int(message.rsplit(",", 1)[1].split()[0])
            self.progress(f"pass {self.passes} done")

    def __call__(self, plan: coverage.Plan) -> tuple[np.ndarray, np.ndarray]:
        keys = job_keys(plan)
        slot = np.array([self.slot.get(tuple(int(v) for v in key), -1) for key in keys], np.int64)
        keypoints = np.full((len(keys), 17, 3), np.nan, np.float32)
        boxes = np.full((len(keys), 4), np.nan, np.float32)
        known = slot >= 0
        keypoints[known], boxes[known] = self.keypoints[slot[known]], self.box[slot[known]]
        pending = np.flatnonzero(~known)
        if len(pending):
            workspace = self.context.workspace
            requests, results = workspace.path("recovered", "requests.npz"), workspace.path("recovered", "results.npz")
            results.unlink(missing_ok=True)
            np.savez(requests, cameras=np.array(self.names), camera=plan.job_camera[pending],
                     frame=plan.job_frame[pending], box=plan.job_box[pending])
            command = [workspace.site.interpreter("data_management"), WORKERS / "recover_trt.py", "--dataset-json",
                       workspace.dataset_json, "--requests", requests, "--vitpose", self.vitpose, "--out", results,
                       "--margins", ",".join(str(self.margin[side]) for side in ("left", "right", "top", "bottom"))]
            self.context.log(f"reading {len(pending)} crops ({int(known.sum())} already cached)")
            self.progress(f"pass {self.passes + 1}: reading {len(pending)} crops")
            started = time.time()
            code = run_worker(self.context, command)
            self.worker_seconds += time.time() - started
            if code != 0 or not results.is_file():
                raise WorkerFailed(f"recover worker exited with code {code}: {log_tail(self.context)}")
            with np.load(results) as data:
                keypoints[pending], boxes[pending] = data["keypoints"], data["box"]
            read = pending[np.isfinite(keypoints[pending]).all(axis=(1, 2))]
            base = len(self.keys)
            self.keys = np.concatenate([self.keys, keys[read]])
            self.keypoints = np.concatenate([self.keypoints, keypoints[read]])
            self.box = np.concatenate([self.box, boxes[read]])
            for offset, key in enumerate(keys[read]):
                self.slot[tuple(int(v) for v in key)] = base + offset
            self.read_now += int(len(pending))
            self.dirty = True
            requests.unlink(missing_ok=True)
            results.unlink(missing_ok=True)
        return keypoints, boxes

    def save(self) -> None:
        if self.dirty:
            np.savez_compressed(self.path, model_sha256=np.array(self.model_sha), keys=self.keys,
                                keypoints=self.keypoints, box=self.box)
            self.dirty = False


def save_recovered(directory: Path, scene, arrays: dict, tracks_sha: str) -> int:
    count, cameras = len(arrays["t"]), len(scene.cameras)
    bits = np.zeros((count, 17), np.uint8)
    for camera_index in range(cameras):
        bits |= (arrays["used"][:, camera_index].astype(np.uint8) << camera_index)
    with np.errstate(all="ignore"):
        worst = np.nanmax(np.where(arrays["used"], arrays["residual"], np.nan), axis=1) if count else np.zeros((0, 17))
    np.savez_compressed(
        directory / "tracks3d_recovered.npz",
        component=np.full(count, -1, np.int32), identity=arrays["identity"].astype(np.int32),
        t=arrays["t"].astype(np.int32), rows=np.full((count, cameras), -1, np.int32),
        xyz=arrays["xyz"].astype(np.float32), used=bits, residual=np.asarray(worst, np.float32),
        root=arrays["root"].astype(np.float32), tier=arrays["tier"], hint_kind=arrays["hint_kind"],
        hint_frames=arrays["hint_frames"], **{"pass": arrays["pass"]},
        # the 2D poses each position was triangulated from, per camera (frame -1 = camera not used)
        pose_frame=arrays["pose_frame"].astype(np.int32), pose_keypoints=arrays["pose_keypoints"].astype(np.float32),
        pose_box=arrays["pose_box"].astype(np.float32), cameras=np.array([camera.camera for camera in scene.cameras]),
        source_tracks_sha256=np.array(tracks_sha))
    return count


def save_rejected(directory: Path, rejected: dict) -> dict:
    """Write the still-missing instants that were looked at; returns counts per reason and per failed condition."""
    keys = sorted(rejected)
    records = [rejected[key] for key in keys]
    features = np.full((len(keys), len(FEATURES)), np.nan)
    root = np.full((len(keys), 3), np.nan, np.float32)
    xyz = np.full((len(keys), 17, 3), np.nan, np.float32)
    failed = Counter()
    for index, record in enumerate(records):
        if "features" in record:
            features[index] = [record["features"][name] for name in FEATURES]
            root[index], xyz[index] = record["root"], record["xyz"]
            failed.update(record["failed"])
    np.savez_compressed(
        directory / "recover_rejected.npz",
        identity=np.array([key[0] for key in keys], np.int32), t=np.array([key[1] for key in keys], np.int32),
        reason=np.array([REASONS.index(record["reason"]) for record in records], np.int8), reasons=np.array(REASONS),
        features=features, feature_names=np.array(FEATURES),
        failed=np.array(["; ".join(record.get("failed", [])) for record in records]),
        tier_tried=np.array([record.get("tier_tried", "") for record in records]),
        hint_kind=np.array([record["hint_kind"] for record in records], np.int8),
        hint_frames=np.array([record["hint_frames"] for record in records], np.int32), root=root, xyz=xyz)
    return {"by_reason": dict(Counter(record["reason"] for record in records)), "gate_conditions_failed": dict(failed)}


HISTORY_KEY = "autofit_refresh_history"


def consider_autofit_refresh(context: Context, scene, arrays: dict, identities: int) -> dict:
    """Should Autofit run again with the recovered poses as detections (``autofit_refresh.decide``)?

    When yes, the detections Autofit will get are written to ``evidence/autofit/extra.npz`` and the lap is
    recorded in the workspace's history; the caller asks the engine to rerun Autofit.
    """
    workspace = context.workspace
    evidence = workspace.path("evidence", "autofit")
    if not (evidence / "labels.npz").is_file() or not (evidence / "meta.json").is_file():
        return {"refresh": False, "reason": "no Autofit tracks in this workspace"}
    signature = {"sync_table": file_sha256(workspace.path("alignment", "sync_table.npz")),
                 "calibration": file_sha256(workspace.path("calibration", "calibration.json"))}
    known = autofit_refresh.load(evidence / "extra.npz", signature)
    cameras = [{"name": camera.camera, "frame_of_row": camera.detections.frame_of_row,
                "boxes": camera.detections.boxes, "kept": camera.kept} for camera in scene.cameras]
    found = autofit_refresh.new_detections(arrays, cameras)
    union, added = autofit_refresh.merge(known, found)
    meta = json.loads((evidence / "meta.json").read_text(encoding="utf-8"))
    history = list(context.db.get_meta(HISTORY_KEY, []))
    if history and history[-1].get("identities_after") is None:
        # this recovery follows the solve of the last lap: it shows what the lap achieved
        used = json.loads((evidence / "inputs.json").read_text(encoding="utf-8")) if (evidence / "inputs.json").is_file() else {}
        if (evidence / "extra.npz").is_file() and used.get("extra_detections") == file_sha256(evidence / "extra.npz"):
            history[-1]["identities_after"] = int(identities)
            history[-1]["autofit_seconds"] = meta.get("tracking_wall_s")
    laps = sum(1 for lap in history if lap["round"] == context.round)
    decision = autofit_refresh.decide(new=added, fed=int(meta.get("detections_fed", 0)), identities=identities,
                                      history=history, laps_this_round=laps, recipe=RECIPE["autofit_refresh"])
    decision["detections_autofit_has"] = int(len(known["frame"]))
    decision["autofit_seconds_last_run"] = meta.get("tracking_wall_s")
    if decision["refresh"]:
        autofit_refresh.save(evidence / "extra.npz", union, signature)
        history.append({"round": context.round, "new": added, "total": int(len(union["frame"])),
                        "identities_before": int(identities), "identities_after": None})
    context.db.set_meta(HISTORY_KEY, history)
    decision["history"] = history
    return decision


def run(context: Context) -> StageResult:
    workspace = context.workspace
    site = workspace.site
    recipe, geometry = RECIPE["coverage"], RECIPE["geometry"]
    directory = latest_solution(workspace.root)
    if directory is None:
        return StageResult(status="blocked", message="no solution yet; run the solve stage")
    vitpose = site.model("vitpose_onnx")
    if not vitpose.is_file():
        return StageResult(status="failed", message=f"model file missing: {vitpose}")
    model_sha = file_sha256(vitpose)
    scene = load_scene(workspace, joint_peak=geometry["joint_peak"], rays=False)
    with np.load(directory / "tracks3d.npz") as data:
        tracks = {name: data[name] for name in ("identity", "t", "root", "xyz")}
    tracks_sha = file_sha256(directory / "tracks3d.npz")
    inputs = {"solution": str(directory.relative_to(workspace.root)), "tracks3d_sha256": tracks_sha,
              "vitpose_onnx_sha256": model_sha}
    reader = CropReader(context, [camera.camera for camera in scene.cameras], vitpose, model_sha, recipe["box_margin"])
    started = time.time()
    try:
        result = coverage.recover_iteratively(scene, tracks["identity"].astype(np.int64), tracks["t"].astype(np.int64),
                                              tracks["root"], tracks["xyz"], reader, geometry, recipe, log=reader.note)
    except WorkerFailed as exc:
        reader.save()
        return StageResult(status="failed", inputs=inputs, message=str(exc))
    reader.save()
    arrays = result["arrays"]
    count = save_recovered(directory, scene, arrays, tracks_sha)
    remaining = save_rejected(directory, result["rejected"])
    tiers = {name: int((arrays["tier"] == name).sum()) for name in ("A", "B")}
    first_pass = int((arrays["pass"] == 0).sum()) if count else 0
    summary = {"first_plan": result["first_plan"], "passes": result["passes"], "recovered": count,
               "recovered_in_first_pass": first_pass, "recovered_by_later_passes": count - first_pass,
               "recovered_by_tier": tiers, "positions_before": result["positions_before"],
               "positions_after": result["positions_before"] + count,
               "missing_inside_tracks_before": result["inside_missing_before"],
               "missing_inside_tracks_after": result["inside_missing_after"],
               "still_missing_after_looking": remaining, "stopped_by_pass_limit": result["stopped_by_pass_limit"],
               "crops_read_now": reader.read_now,
               "worker_seconds": round(reader.worker_seconds, 1), "seconds": round(time.time() - started, 1)}
    solved_identity = tracks["identity"][tracks["identity"] >= 0]
    refresh = consider_autofit_refresh(context, scene, arrays, int(len(np.unique(solved_identity))))
    summary["autofit_refresh"] = refresh
    context.log(("another Autofit lap: " if refresh["refresh"] else "no Autofit lap: ") + refresh["reason"])
    (directory / "recover.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    context.log(f"recovered {count} positions in {len(result['passes'])} passes ({tiers['A']} from three or more "
                f"cameras, {tiers['B']} from two); missing inside tracks {result['inside_missing_before']} -> "
                f"{result['inside_missing_after']}")
    if result["stopped_by_pass_limit"]:
        context.log("recovery stopped at the pass limit while still finding positions (coverage.max_passes)", level="warning")
    relative = str(directory.relative_to(workspace.root)).replace("\\", "/")
    return StageResult(
        metrics=summary, inputs=inputs,
        message=f"{count} positions recovered from pixels in {len(result['passes'])} passes; missing inside tracks "
                f"{result['inside_missing_before']} -> {result['inside_missing_after']}",
        outputs={name: f"{relative}/{name}" for name in ("tracks3d_recovered.npz", "recover_rejected.npz", "recover.json")},
        rerun=("autofit",) if refresh["refresh"] else ())
