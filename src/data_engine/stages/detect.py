"""2D detection: dense YOLO + ViTPose over every native frame (``workers/detect_trt.py``, TensorRT).

The worker is resumable (finished shards are skipped), so re-running the stage continues where
a previous run stopped; if every shard already exists the worker is not launched at all.
Coverage is verified afterwards: every frame of every camera must have been processed.
"""

from __future__ import annotations

import time

from ..engine import Context, StageResult
from ..store import load_camera
from ..workers.detect_trt import pending_shards
from ..workspace import file_sha256
from .ingest import WORKERS, log_tail, run_worker


def run(context: Context) -> StageResult:
    workspace = context.workspace
    site = workspace.site
    dataset = workspace.dataset
    out = workspace.detections_dir
    yolo, vitpose = site.model("yolo_onnx"), site.model("vitpose_onnx")

    broken = {c: cam.issues for c, cam in sorted(dataset.cameras.items()) if cam.issues}
    if broken:
        return StageResult(status="blocked", message="unresolved stream layout (see stream_layout unknowns) for: "
                           + ", ".join(broken), metrics={"cameras_blocked": sorted(broken)})

    pending = {c: pending_shards(cam, out) for c, cam in dataset.cameras.items()}
    pending_frames = sum(e - s for shards in pending.values() for s, e in shards)
    inputs = {"yolo_onnx": str(yolo), "vitpose_onnx": str(vitpose), "dataset_json_sha256": file_sha256(workspace.dataset_json),
              "interpreter": str(site.interpreter("data_management")), "pending_frames_at_start": pending_frames}
    wall = 0.0
    if pending_frames:
        for model in (yolo, vitpose):
            if not model.is_file():
                return StageResult(status="failed", message=f"model file missing: {model}", inputs=inputs)
        inputs["yolo_onnx_sha256"], inputs["vitpose_onnx_sha256"] = file_sha256(yolo), file_sha256(vitpose)
        command = [site.interpreter("data_management"), WORKERS / "detect_trt.py", "--dataset-json", workspace.dataset_json,
                   "--out", out, "--yolo", yolo, "--vitpose", vitpose,
                   "--progress", workspace.path("logs", "detect_progress.json")]
        context.log(f"{pending_frames} frames pending; launching detector (resumable)")
        started = time.time()
        code = run_worker(context, command)
        wall = time.time() - started
        if code != 0:
            return StageResult(status="failed", inputs=inputs, metrics={"wall_seconds": round(wall, 1)},
                               message=f"detect worker exited with code {code}: {log_tail(context)}")
    else:
        context.log("all detection shards already exist; worker not launched")
        for model in (yolo, vitpose):
            if model.is_file():
                inputs[f"{model.stem}_sha256"] = file_sha256(model)

    per_camera, incomplete = {}, {}
    for camera_id, camera in sorted(dataset.cameras.items()):
        detections = load_camera(out, camera_id, camera.frame_count)
        covered = int(detections.covered.sum())
        per_camera[camera_id] = {"frames": camera.frame_count, "covered": covered, "detections": int(len(detections.boxes)),
                                 "detections_per_frame": round(len(detections.boxes) / max(1, covered), 3)}
        if covered != camera.frame_count:
            incomplete[camera_id] = f"{covered}/{camera.frame_count}"
    if incomplete:
        return StageResult(status="failed", inputs=inputs, metrics={"per_camera": per_camera, "wall_seconds": round(wall, 1)},
                           message="detection coverage below 100%: " + ", ".join(f"{c} {v}" for c, v in incomplete.items()))
    total_frames = sum(v["frames"] for v in per_camera.values())
    total_dets = sum(v["detections"] for v in per_camera.values())
    metrics = {"frames": total_frames, "detections": total_dets, "detections_per_frame": round(total_dets / max(1, total_frames), 3),
               "per_camera": per_camera, "wall_seconds": round(wall, 1), "frames_processed": pending_frames,
               "rate_fps": round(pending_frames / wall, 1) if wall > 0 and pending_frames else None}
    message = f"{total_frames} frames, {total_dets} detections ({metrics['detections_per_frame']}/frame)"
    message += f"; processed {pending_frames} frames in {wall / 60:.1f} min" if pending_frames else "; reused existing shards"
    return StageResult(metrics=metrics, message=message, inputs=inputs, outputs={"detections": "detections/raw"})
