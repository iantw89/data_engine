"""Calibration check: has any camera moved relative to the calibration capture?

``workers/calibration_dm.py`` runs data_management's mount check (SIFT background match against the
capture, rotation fit) per camera. This stage turns the result into ``calibration/calibration.json``
(the standard calibration, same schema) and into ``calibration`` unknowns.

A mount-check correction is NEVER applied on its own. It is applied only when an active human
``calibration`` constraint (``a == [camera_id]``) says ``detail["choice"] == "use_corrected"``;
``use_standard`` records that the person accepted the standard calibration for that camera, and
``exclude_camera`` removes the camera from the calibration file. Every decision and the status of
every camera is written to the file's top-level ``data_engine`` note.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ..engine import Context, StageResult
from ..workspace import file_sha256
from .ingest import WORKERS, log_tail, run_worker

SAMPLES = 9
ATTENTION = ("shifted", "moved", "inconclusive")
CHOICES = ("use_standard", "use_corrected", "exclude_camera")


def human_decisions(context: Context) -> dict[str, dict]:
    """Latest active human decision per camera: {camera: {"choice", "constraint_id", "notes"}}."""
    decisions = {}
    for constraint in context.db.constraints("calibration"):  # ordered by id: later answers win
        if not str(constraint.get("source", "")).startswith("human") or len(constraint["a"]) != 1:
            continue
        choice = (constraint.get("detail") or {}).get("choice")
        if choice in CHOICES:
            decisions[str(constraint["a"][0])] = {"choice": choice, "constraint_id": constraint["id"],
                                                  "notes": (constraint.get("detail") or {}).get("notes", "")}
    return decisions


def reusable_mount_check(path: Path, dataset_sha: str, calibration_sha: str, cameras: list[str]) -> dict | None:
    """The previous mount check, if it was made from exactly these inputs (it is the slow part)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    meta = data.get("data_engine", {})
    if (meta.get("dataset_json_sha256") == dataset_sha and meta.get("calibration_sha256") == calibration_sha
            and meta.get("samples") == SAMPLES and all(c in data.get("cameras", {}) for c in cameras)):
        return data
    return None


def describe(camera: str, check: dict, frames: int) -> dict:
    status = check["status"]
    shift, rotation, residual = check.get("median_shift_px"), check.get("rotation_deg"), check.get("residual_px")
    correction = bool(check.get("corrected_rvec") and check.get("corrected_tvec"))
    numbers = {"status": status, "median_shift_px": shift, "rotation_deg": rotation, "residual_px": residual,
               "matches": check.get("matches"), "inliers": check.get("inliers"), "reason": check.get("reason", ""),
               "pan_tilt_roll_deg": check.get("pan_tilt_roll_deg"), "correction_available": correction}
    options = [{"value": "use_standard", "label": "Keep the standard calibration for this camera"}]
    if correction:
        options.append({"value": "use_corrected", "label": "Use the rotation-corrected extrinsics from the mount check"})
    options.append({"value": "exclude_camera", "label": "Leave this camera out of the calibration (and the 3D solve)"})
    if status == "shifted":
        explanation = (f"{camera}'s static background is displaced by about {shift} px (rotation {rotation} deg) from the "
                       "calibration capture, and a pure rotation explains it (residual "
                       f"{residual} px). The camera has probably slipped in its mount. The corrected extrinsics may "
                       "fix triangulation; the standard ones would be off by about that much. Check that the projected "
                       "reference geometry lines up with the image, then choose.")
    elif status == "moved":
        explanation = (f"{camera} differs from the calibration capture by more than a mount slip can explain "
                       f"(shift {shift} px, residual {residual} px): the camera was moved, or the scene changed. "
                       "No correction is offered; either accept the standard calibration knowingly or exclude the camera.")
    else:
        explanation = (f"The mount check could not decide for {camera} ({check.get('reason') or 'too few distributed matches'}). "
                       "The standard calibration may or may not fit. Inspect the camera view against the reference "
                       "geometry and choose.")
    return {"camera": camera, "calibration_id": check.get("calibration_id"), "frames": frames, "mount_check": numbers,
            "options": options, "explanation": explanation,
            "what_would_resolve": "A calibration_review decision: use_standard, use_corrected or exclude_camera."}


def run(context: Context) -> StageResult:
    workspace = context.workspace
    site = workspace.site
    dataset = workspace.dataset
    source = Path(site["calibration"])
    if not source.is_file():
        return StageResult(status="failed", message=f"standard calibration not found: {source}")
    out_dir = workspace.path("calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    check_path = out_dir / "mount_check.json"
    source_sha, dataset_sha = file_sha256(source), file_sha256(workspace.dataset_json)
    inputs = {"calibration_source": str(source), "calibration_source_sha256": source_sha,
              "dataset_json_sha256": dataset_sha, "mount_check_samples": SAMPLES,
              "interpreter": str(site.interpreter("data_management"))}

    check = reusable_mount_check(check_path, dataset_sha, source_sha, list(dataset.cameras))
    if check is not None:
        context.log("mount check inputs unchanged; reusing calibration/mount_check.json")
    else:
        command = [site.interpreter("data_management"), WORKERS / "calibration_dm.py", "--dataset-json", workspace.dataset_json,
                   "--dataset", workspace.source_dataset, "--video-root", site["video_root"], "--calibration", source,
                   "--out", out_dir, "--samples", SAMPLES]
        context.log("running mount check (CPU: ~9 frames per camera + SIFT)")
        code = run_worker(context, command)
        if code != 0:
            return StageResult(status="failed", inputs=inputs,
                               message=f"calibration worker exited with code {code}: {log_tail(context)}")
        check = json.loads(check_path.read_text(encoding="utf-8"))
    inputs["mount_check_sha256"] = file_sha256(check_path)
    inputs["main_channel_rebuilt"] = check.get("data_engine", {}).get("main_channel_rebuilt", [])

    decisions = human_decisions(context)
    inputs["human_decisions"] = {c: {"choice": d["choice"], "constraint_id": d["constraint_id"]} for c, d in decisions.items()}
    calibration = json.loads(source.read_text(encoding="utf-8"))
    note_cameras, per_camera, unknowns, excluded = {}, {}, [], []
    for camera_id, camera in sorted(dataset.cameras.items()):
        result = check["cameras"].get(camera_id) or {"status": "inconclusive", "reason": "camera missing from the mount check",
                                                    "calibration_id": camera.calibration_id}
        result = {**result, "calibration_id": result.get("calibration_id") or camera.calibration_id}
        cid, status = result["calibration_id"], result["status"]
        correction = bool(result.get("corrected_rvec") and result.get("corrected_tvec"))
        decision = decisions.get(camera_id)
        applied = "standard"
        if decision and decision["choice"] == "exclude_camera":
            applied = "excluded"
            calibration["cameras"].pop(str(cid), None)
            excluded.append(camera_id)
        elif decision and decision["choice"] == "use_corrected":
            if correction and str(cid) in calibration["cameras"]:
                applied = "corrected"
                calibration["cameras"][str(cid)]["rvec"] = list(result["corrected_rvec"])
                calibration["cameras"][str(cid)]["tvec"] = list(result["corrected_tvec"])
            else:
                applied = "standard (use_corrected requested but the mount check offers no correction)"
        elif decision:
            applied = "standard (accepted by a human)"
        elif status in ATTENTION:
            applied = "standard (undecided: open calibration unknown)"
        # A decision that cannot be honoured leaves the question open.
        satisfied = decision is not None and not (decision["choice"] == "use_corrected" and applied != "corrected")
        if status in ATTENTION and not satisfied:
            item = describe(camera_id, result, camera.frame_count)
            unknowns.append({"subject": f"calibration:{camera_id}", "severity": float(camera.frame_count or 1), "detail": item})
        note_cameras[camera_id] = {
            "calibration_id": cid, "status": status, "median_shift_px": result.get("median_shift_px"),
            "rotation_deg": result.get("rotation_deg"), "residual_px": result.get("residual_px"),
            "reason": result.get("reason", ""), "correction_available": correction,
            "human_decision": decision["choice"] if decision else None,
            "constraint_id": decision["constraint_id"] if decision else None, "applied": applied}
        per_camera[camera_id] = {"status": status, "median_shift_px": result.get("median_shift_px"),
                                 "rotation_deg": result.get("rotation_deg"), "residual_px": result.get("residual_px"),
                                 "decision": decision["choice"] if decision else None, "applied": applied}
        if status != "unchanged":
            context.log(f"{camera_id}: {status}, shift {result.get('median_shift_px')} px"
                        + (f" - {result.get('reason')}" if result.get("reason") else "")
                        + f"; applied: {applied}")
    calibration["data_engine"] = {
        "source": str(source), "source_sha256": source_sha, "dataset": workspace.dataset_id,
        "mount_check_sha256": inputs["mount_check_sha256"], "cameras": note_cameras, "excluded_cameras": excluded,
        "note": "Corrections are applied only on an explicit human use_corrected decision; see cameras[*].applied."}
    target = out_dir / "calibration.json"
    target.write_text(json.dumps(calibration, indent=1), encoding="utf-8")

    counts = {s: sum(1 for v in per_camera.values() if v["status"] == s) for s in ("unchanged", *ATTENTION)}
    corrected = [c for c, v in per_camera.items() if v["applied"] == "corrected"]
    message = ", ".join(f"{n} {s}" for s, n in counts.items() if n) or "no cameras"
    if unknowns:
        message += f"; {len(unknowns)} camera(s) await a calibration decision"
    if corrected:
        message += f"; corrections applied by human decision: {', '.join(corrected)}"
    from ..workspace import stable_hash
    geometry = {cid: {k: cam[k] for k in ("K", "D", "rvec", "tvec")} for cid, cam in calibration["cameras"].items()}
    return StageResult(
        metrics={"per_camera": per_camera, "counts": counts, "corrected": corrected, "excluded": excluded,
                 "decisions": len(decisions)},
        # Downstream stages depend on the camera geometry only: a decision that keeps the
        # standard calibration must not invalidate alignment and everything after it.
        output_hash=stable_hash(geometry),
        message=message, unknowns={"calibration": unknowns}, inputs=inputs,
        outputs={"calibration": "calibration/calibration.json", "mount_check": "calibration/mount_check.json"})
