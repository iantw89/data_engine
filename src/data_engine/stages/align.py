"""Time alignment: per-camera clock offsets from pose motion (and audio) over the whole recording.

Runs ``workers/align_dm.py`` (data_management's alignment generator). The worker exits with code
3 when fewer than two cameras resolve; its outputs are still written, and the stage is then
``blocked`` (not failed) with the unknowns derived from ``report.json``.

Human alignment checkpoints (active ``alignment`` constraints) are passed through into the stage
inputs and noted in the metrics; they are not applied to the worker yet.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..engine import Context, StageResult
from ..workspace import file_sha256
from .ingest import WORKERS, log_tail, run_worker

SHORT_RECORDING_SECONDS = 300.0
LONG_SPAN_SECONDS = 30.0
SPREAD_LIMIT_FRAMES = 1.0


def covered_frames(detections_dir: Path, camera: str) -> int:
    """Frames with detection records for a camera, reading only the small ``frames`` arrays."""
    total = 0
    for shard in sorted((detections_dir / camera).glob("[0-9]*.npz")):
        if ".tmp" in shard.name:
            continue
        with np.load(shard) as data:
            total += len(data["frames"])
    return total


def recording_seconds(dataset, detections_dir: Path) -> float:
    """Longest detected duration over cameras: the footage the aligner will actually see."""
    return max((covered_frames(detections_dir, c) / (cam.fps or 10.0) for c, cam in dataset.cameras.items()), default=0.0)


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3g}"


def alignment_unknowns(report: dict, dataset) -> list[dict]:
    span = report.get("run", {}).get("span_seconds") or [0.0, 0.0]
    span_seconds = max(0.0, float(span[1]) - float(span[0]))
    items: dict[str, dict] = {}
    for camera, entry in sorted(report.get("cameras", {}).items()):
        fps = float(entry.get("frame_rate") or 10.0)
        stream_frames = dataset.cameras[camera].frame_count if camera in dataset.cameras else None
        numbers = {k: entry.get(k) for k in (
            "resolved", "anchor", "final_offset_seconds", "accepted_windows_including", "candidate_windows",
            "drift_seconds_per_hour", "spread_mad_seconds", "spread_mad_frames", "unresolved_spans", "stale_spans",
            "uncertain", "frame_rate")}
        if entry.get("candidates") is not None:
            numbers["candidates"] = entry["candidates"]
        reasons = []
        if not entry.get("resolved"):
            reasons.append("no confirmed offset: the camera could not be tied to the others from pose motion or audio")
        spread = entry.get("spread_mad_frames")
        if spread is not None and spread > SPREAD_LIMIT_FRAMES:
            reasons.append(f"offset estimates scatter by {spread:.2f} frames around the drift fit (limit {SPREAD_LIMIT_FRAMES:g})")
        if reasons:
            frames = int(min(round(span_seconds * fps), stream_frames) if stream_frames else round(span_seconds * fps))
            items[f"alignment:{camera}"] = {
                "subject": f"alignment:{camera}", "severity": float(max(1, frames)),
                "detail": {"camera": camera, "kind": "camera", "reasons": reasons, "report": numbers,
                           "explanation": (f"{camera}'s time offset against the anchor ({report.get('anchor')}) is not reliable: "
                                           + "; ".join(reasons) + ". Cross-camera geometry cannot be trusted for this camera "
                                           "until a person confirms the matching frame at a few instants."),
                           "what_would_resolve": "alignment_review checkpoints: pick the frame showing the same instant as the reference."}}
        if not entry.get("resolved"):
            continue  # an unresolved camera is one unknown; its spans add nothing
        for label, spans in (("unresolved", entry.get("unresolved_spans") or []), ("stale", entry.get("stale_spans") or [])):
            for start, end in spans:
                if end - start <= LONG_SPAN_SECONDS:
                    continue
                subject = f"alignment:{camera}:{start:.1f}-{end:.1f}"
                items[subject] = {
                    "subject": subject, "severity": float(max(1, round((end - start) * fps))),
                    "detail": {"camera": camera, "kind": f"{label}_span", "start_seconds": start, "end_seconds": end,
                               "report": numbers,
                               "explanation": (f"For {end - start:.0f} s ({start:.1f}-{end:.1f} s) {camera} has no accepted "
                                               f"alignment evidence ({label}); its offset there is carried over from earlier, "
                                               "and may have drifted or jumped."),
                               "what_would_resolve": "An alignment_review checkpoint inside this span."}}
    return list(items.values())


def run(context: Context) -> StageResult:
    workspace = context.workspace
    site = workspace.site
    dataset = workspace.dataset
    calibration = workspace.path("calibration", "calibration.json")
    detections = workspace.detections_dir
    out = workspace.path("alignment")
    if not calibration.is_file():
        return StageResult(status="failed", message="calibration/calibration.json is missing; run the calibration stage")
    constraints = [{"id": c["id"], "camera": c["a"][0] if c["a"] else None, "detail": c["detail"]}
                   for c in context.db.constraints("alignment")]
    seconds = recording_seconds(dataset, detections)
    command = [site.interpreter("data_management"), WORKERS / "align_dm.py", "--dataset-json", workspace.dataset_json,
               "--detections", detections, "--calibration", calibration, "--out", out]
    if seconds < SHORT_RECORDING_SECONDS:
        command += ["--update-seconds", 10]
    inputs = {"dataset_json_sha256": file_sha256(workspace.dataset_json), "calibration_sha256": file_sha256(calibration),
              "detections": str(detections), "detected_seconds": round(seconds, 1),
              "worker_arguments": [str(c) for c in command[2:] if str(c).startswith("--") or str(c).isdigit()],
              "human_alignment_constraints": constraints}
    if constraints:
        context.log(f"{len(constraints)} human alignment constraint(s) recorded; not applied by this stage yet", level="warning")
    context.log(f"aligning {len(dataset.cameras)} cameras over {seconds / 60:.1f} min of detections")
    started = time.time()
    code = run_worker(context, command)
    wall = time.time() - started
    report_path = out / "report.json"
    if code not in (0, 3):
        return StageResult(status="failed", inputs=inputs, metrics={"wall_seconds": round(wall, 1)},
                           message=f"align worker exited with code {code}: {log_tail(context)}")
    if not report_path.is_file():
        return StageResult(status="failed", inputs=inputs, message=f"align worker (exit {code}) wrote no report.json: {log_tail(context)}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    inputs["report_sha256"] = file_sha256(report_path)
    run_info = report.get("run", {})
    per_camera = {}
    for camera, entry in sorted(report["cameras"].items()):
        per_camera[camera] = {
            "offset_seconds": entry.get("final_offset_seconds"), "resolved": entry.get("resolved"),
            "anchor": entry.get("anchor"), "drift_seconds_per_hour": entry.get("drift_seconds_per_hour"),
            "spread_mad_frames": entry.get("spread_mad_frames"),
            "accepted_windows": entry.get("accepted_windows_including"), "total_windows": run_info.get("windows")}
        context.log(f"{camera}: offset {_fmt(entry.get('final_offset_seconds'))} s, drift {_fmt(entry.get('drift_seconds_per_hour'))} s/h, "
                    f"spread {_fmt(entry.get('spread_mad_frames'))} frames, windows {entry.get('accepted_windows_including')}/{run_info.get('windows')}"
                    + ("" if entry.get("resolved") else " UNRESOLVED"))
    unknowns = alignment_unknowns(report, dataset)
    resolved = [c for c, v in per_camera.items() if v["resolved"]]
    metrics = {"per_camera": per_camera, "anchor": report.get("anchor"), "windows": run_info.get("windows"),
               "accepted_windows": run_info.get("accepted"), "resolved_cameras": len(resolved),
               "unresolved_cameras": report.get("unresolved_cameras", []), "wall_seconds": round(wall, 1),
               "human_alignment_constraints": len(constraints),
               "human_constraints_applied": False if constraints else None}
    outputs = {name: f"alignment/{name}" for name in ("alignment.json", "provenance.json", "clocks.npz", "sync_table.npz", "report.json")
               if (out / name).is_file()}
    message = f"{len(resolved)}/{len(per_camera)} cameras resolved (anchor {report.get('anchor')}); {run_info.get('accepted')}/{run_info.get('windows')} windows accepted"
    if constraints:
        message += f"; {len(constraints)} human alignment constraints not yet applied"
    if code == 3:
        return StageResult(status="blocked", metrics=metrics, inputs=inputs, outputs=outputs, unknowns={"alignment": unknowns},
                           message=f"fewer than 2 cameras resolved ({', '.join(resolved) or 'none'}); alignment blocked. " + message)
    return StageResult(metrics=metrics, message=message, unknowns={"alignment": unknowns}, inputs=inputs, outputs=outputs)
