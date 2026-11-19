"""Ingest: resolve the dataset's camera files into verified frame sequences (data_management catalog).

Runs ``workers/ingest_dm.py`` in data_management's interpreter, then reports every camera whose
layout is unresolved as a ``stream_layout`` unknown. Notes (e.g. a stream rebuilt from the main
channel) are informational: logged and kept in the metrics, never unknowns.

This module also holds :func:`run_worker`, the shared way stages shell out to a worker.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from ..engine import Context, StageResult
from ..workspace import file_sha256

WORKERS = Path(__file__).resolve().parents[1] / "workers"
SRC = Path(__file__).resolve().parents[2]


def run_worker(context: Context, command: list[str], *, cwd: Path | None = None) -> int:
    """Run a worker, streaming stdout+stderr to ``<workspace>/logs/<stage>.log``; returns the exit code."""
    log_path = context.workspace.path("logs", f"{context.stage}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": os.pathsep.join(p for p in (str(SRC), os.environ.get("PYTHONPATH", "")) if p)}
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} round {context.round}: "
                  f"{' '.join(str(c) for c in command)}\n")
        log.flush()
        completed = subprocess.run([str(c) for c in command], stdout=log, stderr=subprocess.STDOUT, env=env,
                                   cwd=str(cwd) if cwd else None, check=False)
        log.write(f"=== exit code {completed.returncode}\n")
    return int(completed.returncode)


def log_tail(context: Context, lines: int = 6) -> str:
    path = context.workspace.path("logs", f"{context.stage}.log")
    try:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    tail = [line for line in text if line.strip() and not line.startswith("===")][-lines:]
    return " | ".join(tail)


def stream_layout_unknowns(dataset) -> list[dict]:
    items = []
    for camera_id, camera in sorted(dataset.cameras.items()):
        if not camera.issues:
            continue
        items.append({
            "subject": f"stream_layout:{camera_id}",
            "severity": float(camera.frame_count or 1),
            "detail": {
                "camera": camera_id, "issues": list(camera.issues), "notes": list(camera.notes),
                "frames": camera.frame_count, "parts": [part.path for part in camera.parts],
                "what_would_resolve": ("An operator decision on how this camera's files form one frame sequence "
                                       "(which files, in which order, with what overlap), or exclusion of the camera. "
                                       "Detection and alignment skip this camera until then."),
            }})
    return items


def verify_boundaries(context: Context) -> None:
    """Pixel-check every file boundary the catalog left unverified; record exact repeats as ``overlap``."""
    from ..dataset import Dataset
    from ..overlap import verify_boundary
    from ..video import ExactReader, frame_times

    workspace = context.workspace
    dataset = Dataset.load(workspace.dataset_json)
    index_dir = workspace.path("frame_index")
    changed = False
    for camera_id, camera in sorted(dataset.cameras.items()):
        for number in range(1, len(camera.parts)):
            previous, part = camera.parts[number - 1], camera.parts[number]
            if part.verified or camera.issues:
                continue
            name = os.path.basename(part.path)
            try:
                a = ExactReader(previous.path, frame_times(previous.path, index_dir))
                b = ExactReader(part.path, frame_times(part.path, index_dir))
                try:
                    found = verify_boundary(a.read, previous.trim + previous.count - 1, b.read, part.trim, part.count)
                finally:
                    a.close()
                    b.close()
            except (OSError, RuntimeError, ValueError, IndexError) as exc:
                context.log(f"{camera_id}: boundary check of {name} failed ({exc}); left unverified", level="warning")
                continue
            evidence = (f"difference {found.get('residual')} grey levels, next best {found.get('next_best')}, "
                        f"{found.get('frames_compared')} frames compared")
            if found["overlap"]:
                part.overlap, part.verified, changed = int(found["overlap"]), True, True
                camera.notes = [note for note in camera.notes if name not in note]
                camera.notes.append(f"{name}: first {part.overlap} frames repeat the previous file's last "
                                    f"{part.overlap} frames (pixel-identical; {evidence}); kept in the frame index, "
                                    "taken off the timeline")
            else:
                changed = True
                camera.notes.append(f"{name}: boundary not verified by pixel comparison either "
                                    f"({found['reason']}; {evidence})")
    if changed:
        dataset.save(workspace.dataset_json)
        for cached in ("dataset", "detections", "frames"):
            workspace.__dict__.pop(cached, None)


def run(context: Context) -> StageResult:
    workspace = context.workspace
    site = workspace.site
    interpreter = site.interpreter("data_management")
    video_root = str(site["video_root"])
    command = [interpreter, WORKERS / "ingest_dm.py", "--dataset", workspace.source_dataset,
               "--video-root", video_root, "--out", workspace.dataset_json]
    context.log(f"resolving {workspace.dataset_id} from {video_root}")
    code = run_worker(context, command)
    if code != 0:
        return StageResult(status="failed", message=f"ingest worker exited with code {code}: {log_tail(context)}",
                           inputs={"video_root": video_root, "interpreter": str(interpreter)})
    if not workspace.dataset_json.is_file():
        return StageResult(status="failed", message="ingest worker did not write dataset.json")
    # Later stages (and this process) must see the new file, not a previously cached load.
    for cached in ("dataset", "detections", "frames"):
        workspace.__dict__.pop(cached, None)
    verify_boundaries(context)
    limit = context.db.get_meta("limit_frames")
    if limit:
        # Pilot slice: keep only the first N native frames of every camera.
        from ..dataset import Dataset, Part
        full = Dataset.load(workspace.dataset_json)
        for camera in full.cameras.values():
            remaining, parts = int(limit), []
            for part in camera.parts:
                if remaining <= 0:
                    break
                parts.append(Part(part.path, part.trim, min(part.count, remaining), part.verified,
                                  min(part.overlap, min(part.count, remaining))))
                remaining -= parts[-1].count
            camera.parts = parts
            camera.notes.append(f"pilot slice: first {limit} frames only")
        full.dataset_id = workspace.dataset_id
        full.save(workspace.dataset_json)
        context.log(f"pilot slice: limited every camera to its first {limit} frames")
    dataset = workspace.dataset

    cameras = {}
    for camera_id, camera in sorted(dataset.cameras.items()):
        cameras[camera_id] = {"frames": camera.frame_count, "fps": round(camera.fps, 4), "parts": len(camera.parts),
                              "resolution": [camera.width, camera.height], "calibration_id": camera.calibration_id,
                              "issues": list(camera.issues), "notes": list(camera.notes)}
        for note in camera.notes:
            context.log(f"{camera_id}: {note}")
        for issue in camera.issues:
            context.log(f"{camera_id}: ISSUE {issue}", level="warning")
    no_calibration = [c for c, v in cameras.items() if not v["calibration_id"]]
    if no_calibration:
        context.log(f"cameras without a calibration id: {', '.join(no_calibration)}", level="warning")
    unknowns = stream_layout_unknowns(dataset)
    metrics = {
        "cameras": len(cameras), "total_frames": sum(v["frames"] for v in cameras.values()),
        "frames_per_camera": {c: v["frames"] for c, v in cameras.items()},
        "cameras_with_issues": [c for c, v in cameras.items() if v["issues"]],
        "cameras_without_calibration": no_calibration,
        "notes": {c: v["notes"] for c, v in cameras.items() if v["notes"]},
        "per_camera": cameras,
    }
    message = f"{len(cameras)} cameras, {metrics['total_frames']} frames"
    if unknowns:
        message += f"; {len(unknowns)} camera(s) with unresolved stream layout"
    return StageResult(
        metrics=metrics, message=message, unknowns={"stream_layout": unknowns},
        inputs={"dataset_id": workspace.dataset_id, "video_root": video_root, "interpreter": str(interpreter),
                "dataset_json_sha256": file_sha256(workspace.dataset_json)},
        outputs={"dataset": "dataset.json", "frame_index": "frame_index"})
