"""Coverage recovery: read poses at the places where tracked people must be, and triangulate them.

After the solve, every track has gaps: instants at which its person was measured shortly before and
after (or just before/after an end of the track) but not there. ``solve/coverage.py`` plans one crop
per camera that should see the person, ``workers/recover_trt.py`` reads a pose from each crop with
the production ViTPose path, and positions are accepted only as gated triangulations of those poses.
The neighbouring track only says where to look; nothing is interpolated.

Outputs, next to the solution they belong to (``rounds/rNNN``):
    tracks3d_recovered.npz   recovered person-frames in the layout of tracks3d.npz (plus ``tier`` and the
                             2D poses they were triangulated from); the publish stage merges them
    recover.json             counts: missing instants, why each was or was not recovered
Poses read from crops are cached in ``recovered/poses_cache.npz`` (keyed by camera, frame and box),
so a later round only reads what changed.
"""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import numpy as np

from ..config import RECIPE
from ..engine import Context, StageResult
from ..solve import coverage
from ..solve.scene import load_scene
from ..workspace import file_sha256
from .ingest import WORKERS, log_tail, run_worker


def latest_solution(root: Path) -> Path | None:
    found = sorted(glob.glob(str(root / "rounds" / "*" / "tracks3d.npz")))
    return Path(found[-1]).parent if found else None


def job_keys(plan: coverage.Plan) -> np.ndarray:
    """(J, 6) identity of a crop request: camera, frame, box rounded to whole pixels."""
    return np.concatenate([plan.job_camera[:, None].astype(np.int64), plan.job_frame[:, None],
                           np.rint(plan.job_box).astype(np.int64)], axis=1) if len(plan.job_case) else np.zeros((0, 6), np.int64)


def load_cache(path: Path, model_sha: str) -> dict:
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            if str(data["model_sha256"]) == model_sha:
                return {tuple(int(v) for v in key): i for i, key in enumerate(data["keys"])} | {
                    "_keypoints": data["keypoints"], "_box": data["box"], "_keys": data["keys"]}
    return {"_keypoints": np.zeros((0, 17, 3), np.float32), "_box": np.zeros((0, 4), np.float32),
            "_keys": np.zeros((0, 6), np.int64)}


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
    identity, t = tracks["identity"].astype(np.int64), tracks["t"].astype(np.int64)
    presence = coverage.build_presence(identity, t, tracks["root"], tracks["xyz"])
    started = time.time()
    plan = coverage.plan(scene, presence, recipe, geometry["min_ray_angle_deg"])
    context.log(f"{plan.counts['missing_instants']} missing person-instants; {plan.counts['cases']} have a nearby "
                f"measured pose and two cameras that should see them ({len(plan.job_case)} crops)")
    inputs = {"solution": str(directory.relative_to(workspace.root)), "tracks3d_sha256": file_sha256(directory / "tracks3d.npz"),
              "vitpose_onnx_sha256": model_sha}

    # Poses: from the cache where this exact crop was read before, else from the worker.
    cache_path = workspace.path("recovered", "poses_cache.npz")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path, model_sha)
    keys = job_keys(plan)
    slot = np.array([cache.get(tuple(int(v) for v in key), -1) for key in keys], np.int64)
    keypoints = np.full((len(keys), 17, 3), np.nan, np.float32)
    boxes = np.full((len(keys), 4), np.nan, np.float32)
    known = slot >= 0
    keypoints[known], boxes[known] = cache["_keypoints"][slot[known]], cache["_box"][slot[known]]
    pending = np.flatnonzero(~known)
    wall = 0.0
    if len(pending):
        requests, results = workspace.path("recovered", "requests.npz"), workspace.path("recovered", "results.npz")
        np.savez(requests, cameras=np.array([camera.camera for camera in scene.cameras]),
                 camera=plan.job_camera[pending], frame=plan.job_frame[pending], box=plan.job_box[pending])
        margin = recipe["box_margin"]
        command = [site.interpreter("data_management"), WORKERS / "recover_trt.py", "--dataset-json", workspace.dataset_json,
                   "--requests", requests, "--vitpose", vitpose, "--out", results,
                   "--margins", ",".join(str(margin[side]) for side in ("left", "right", "top", "bottom"))]
        context.log(f"reading {len(pending)} crops ({int(known.sum())} already cached)")
        worker_started = time.time()
        code = run_worker(context, command)
        wall = time.time() - worker_started
        if code != 0 or not results.is_file():
            return StageResult(status="failed", inputs=inputs,
                               message=f"recover worker exited with code {code}: {log_tail(context)}")
        with np.load(results) as data:
            keypoints[pending], boxes[pending] = data["keypoints"], data["box"]
        read = pending[np.isfinite(keypoints[pending]).all(axis=(1, 2))]
        np.savez_compressed(cache_path, model_sha256=np.array(model_sha),
                            keys=np.concatenate([cache["_keys"], keys[read]]),
                            keypoints=np.concatenate([cache["_keypoints"], keypoints[read]]),
                            box=np.concatenate([cache["_box"], boxes[read]]))
        requests.unlink(missing_ok=True)
        results.unlink(missing_ok=True)
    else:
        context.log("every crop was read in an earlier round; worker not launched")

    result = coverage.accept(scene, presence, identity, t, plan, keypoints, boxes, geometry, recipe)
    count = len(result.case)
    cases = np.array(result.case, np.int64)
    cameras = len(scene.cameras)
    used = np.array(result.used, bool).reshape(count, cameras, 17)
    residual = np.array(result.residual, np.float64).reshape(count, cameras, 17)
    bits = np.zeros((count, 17), np.uint8)
    for camera_index in range(cameras):
        bits |= (used[:, camera_index].astype(np.uint8) << camera_index)
    with np.errstate(all="ignore"):
        worst = np.nanmax(np.where(used, residual, np.nan), axis=1) if count else np.zeros((0, 17))
    jobs = np.array(result.jobs, np.int64).reshape(count, cameras)
    pose_frame = np.where(jobs >= 0, plan.job_frame[np.clip(jobs, 0, None)] if len(plan.job_frame) else -1, -1)
    pose_keypoints = np.where((jobs >= 0)[:, :, None, None], keypoints[np.clip(jobs, 0, None)] if len(keys) else np.nan, np.nan)
    pose_box = np.where((jobs >= 0)[:, :, None], boxes[np.clip(jobs, 0, None)] if len(keys) else np.nan, np.nan)
    np.savez_compressed(
        directory / "tracks3d_recovered.npz",
        component=np.full(count, -1, np.int32), identity=plan.identity[cases].astype(np.int32),
        t=plan.t[cases].astype(np.int32), rows=np.full((count, cameras), -1, np.int32),
        xyz=np.array(result.xyz, np.float32).reshape(count, 17, 3), used=bits, residual=worst.astype(np.float32),
        root=np.array(result.root, np.float32).reshape(count, 3), tier=np.array(result.tier, dtype="U1"),
        hint_kind=plan.kind[cases], hint_frames=plan.distance[cases],
        # the 2D poses each position was triangulated from, per camera (frame -1 = camera not used)
        pose_frame=pose_frame.astype(np.int32), pose_keypoints=pose_keypoints.astype(np.float32),
        pose_box=pose_box.astype(np.float32), cameras=np.array([camera.camera for camera in scene.cameras]),
        source_tracks_sha256=np.array(inputs["tracks3d_sha256"]))
    tiers = {name: int(sum(1 for value in result.tier if value == name)) for name in ("A", "B")}
    measured = int(sum(len(times) for times in presence.t.values()))
    summary = {**plan.counts, "crops": int(len(plan.job_case)), "crops_read_now": int(len(pending)),
               "outcome": result.reasons, "recovered": count, "recovered_by_tier": tiers,
               "positions_before": measured, "positions_after": measured + count,
               "missing_after": plan.counts["missing_instants"] - count,
               "worker_seconds": round(wall, 1), "seconds": round(time.time() - started, 1)}
    (directory / "recover.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    context.log(f"recovered {count} positions ({tiers['A']} from three or more cameras, {tiers['B']} from two); "
                f"{summary['missing_after']} instants inside or next to tracks remain unmeasured")
    relative = str(directory.relative_to(workspace.root)).replace("\\", "/")
    return StageResult(
        metrics=summary, inputs=inputs,
        message=f"{count} positions recovered from pixels; {summary['missing_after']} of "
                f"{plan.counts['missing_instants']} missing instants remain",
        outputs={"tracks3d_recovered.npz": f"{relative}/tracks3d_recovered.npz", "recover.json": f"{relative}/recover.json"})
