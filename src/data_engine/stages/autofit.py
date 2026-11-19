"""Autofit stage: independent detection-level 3D tracking as evidence for the solver.

Autofit (the user's geometry tracker with a learned human prior) associates detections
frame by frame in 3D instead of trusting single-camera tracklets. On the golden set its raw
tracks were purer and more complete than tracklet-level geometry alone, and the errors of
single-camera trackers (sliding onto a neighbour) are exactly what it does not make. The
solver therefore uses Autofit's raw track ids to cut 2D tracklets where Autofit changes
person and to propose cross-camera links, which the solver's own geometry can veto
(see ``solve.pipeline.with_external_units``). Global (merged) Autofit ids are NOT used:
they made wrong links across gaps on the golden set.

Runs ``workers/autofit_tracks.py`` with Autofit's interpreter from the Autofit repository
root, CPU only. Output: ``evidence/autofit/labels.npz`` (``raw_<camera>`` per raw detection
row) and ``meta.json``. If Autofit is not installed the stage completes without output and
the solver falls back to geometry only (with many more questions for humans).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from ..engine import Context, StageResult
from ..workspace import file_sha256, stable_hash
from .ingest import WORKERS, log_tail


def input_signature(context: Context) -> dict:
    workspace = context.workspace
    return {
        "sync_table": file_sha256(workspace.path("alignment", "sync_table.npz")),
        "calibration": file_sha256(workspace.path("calibration", "calibration.json")),
        "tracklets": file_sha256(workspace.path("tracklets", "index.npz")),
        "detections": "kept",
    }


def run(context: Context) -> StageResult:
    workspace, site = context.workspace, context.workspace.site
    repo, interpreter = site.repo("autofit"), site.interpreter("autofit")
    if not (repo / "trackers" / "human_bot_sort.py").is_file() or not Path(interpreter).is_file():
        context.log("Autofit repository or interpreter not found; the solver will use geometry only", level="warning")
        return StageResult(metrics={"available": False}, message="Autofit not available: geometry-only solve",
                           output_hash="unavailable")
    out = workspace.path("evidence", "autofit")
    out.mkdir(parents=True, exist_ok=True)
    signature = input_signature(context)
    recorded = out / "inputs.json"
    reused = False
    if (out / "labels.npz").is_file() and recorded.is_file() \
            and json.loads(recorded.read_text(encoding="utf-8")) == signature:
        context.log("existing Autofit tracks match the current inputs; reusing them")
        reused = True
    else:
        command = [str(interpreter), str(WORKERS / "autofit_tracks.py"), "--workspace", str(workspace.root),
                   "--calibration", str(workspace.path("calibration", "calibration.json")), "--variant", "kept",
                   "--out", str(out), "--progress", str(workspace.path("logs", "autofit_progress.json"))]
        log_path = workspace.path("logs", "autofit.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        env.pop("PYTHONPATH", None)  # Autofit's flat imports must come from its own repository
        started = time.time()
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} round {context.round}: {' '.join(command)}\n")
            log.flush()
            code = subprocess.run(command, cwd=str(repo), stdout=log, stderr=subprocess.STDOUT, env=env,
                                  check=False).returncode
        if code != 0 or not (out / "labels.npz").is_file():
            return StageResult(status="failed", message=f"Autofit worker exited with code {code}: {log_tail(context)}")
        recorded.write_text(json.dumps(signature, indent=1), encoding="utf-8")
        context.log(f"Autofit tracking finished in {(time.time() - started) / 60:.1f} min")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8")) if (out / "meta.json").is_file() else {}
    metrics = {"available": True, "reused": reused,
               **{key: meta[key] for key in ("frames", "tracking_wall_s", "merger_wall_s", "mean_s_per_frame",
                                             "p95_s_per_frame") if key in meta}}
    seconds = meta.get("tracking_wall_s")
    message = ("Autofit raw tracks " + ("reused" if reused else "computed")
               + (f" ({seconds / 60:.0f} min of tracking)" if seconds else ""))
    return StageResult(metrics=metrics, message=message, output_hash=stable_hash(signature),
                       inputs={**signature, "autofit_repo": str(repo), "worker_sha256": file_sha256(WORKERS / "autofit_tracks.py")},
                       outputs={"labels": "evidence/autofit/labels.npz", "meta": "evidence/autofit/meta.json"})
