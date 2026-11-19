"""Publish stage: hand the solution to data_management as label sets.

Runs ``workers/publish_dm.py`` in data_management's interpreter into ``<workspace>/published/<tag>/``.

Default behaviour is **drafts only**: the worker builds the three label-set drafts (frame_alignment, two_d_tracklets,
three_d_tracks), validates them exactly as the store would (in a throw-away scratch store) and stops. Nothing is
registered or published. The stage publishes to the real label store only when the workspace meta key
``publish_to_store`` is true - an operator decision (set it deliberately, e.g. through the CLI/web UI of the
workspace) - and then only if the data_engine repository has a commit (label rules for machine provenance).

Optional workspace meta keys (all read with ``context.db.get_meta``):
    publish_to_store    false      publish into the label store (true = operator allows it)
    publish_store_root  null       publish into this store root instead of the real store (e.g. a test store)
    publish_stride      1          every Nth sync index (1 = every frame, 10 Hz; 5 = 2 Hz like the golden sets)
    publish_tag         null       set tag (default ``data_engine[_<slice>]_r<round>``)
    publish_replace     false      replace already published sets of the same names
"""

from __future__ import annotations

import json

from ..engine import Context, StageResult
from ..workers import publish_dm
from .ingest import WORKERS, log_tail, run_worker


def _read_json(path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def run(context: Context) -> StageResult:
    workspace = context.workspace
    site = workspace.site
    db = context.db
    interpreter = site.interpreter("data_management")
    try:
        round_number, _ = publish_dm.find_round(workspace.root)
    except publish_dm.PublishError as exc:
        return StageResult(status="blocked", message=str(exc))
    slice_name = publish_dm.split_workspace_id(workspace.dataset_id)[1]
    tag = publish_dm.TAG_CHARS.sub("_", str(db.get_meta("publish_tag", None) or publish_dm.default_tag(round_number, slice_name)))
    stride = max(1, int(db.get_meta("publish_stride", 1) or 1))
    to_store = bool(db.get_meta("publish_to_store", False))
    store_root = db.get_meta("publish_store_root", None)
    replace = bool(db.get_meta("publish_replace", False))
    out = workspace.path("published", tag)
    for stale in ("result.json", "summary.json"):
        (out / stale).unlink(missing_ok=True)

    command = [interpreter, WORKERS / "publish_dm.py", "--workspace", workspace.root, "--dataset", workspace.source_dataset,
               "--tag", tag, "--out", out, "--stride", stride]
    if to_store:
        command.append("--publish")
        if store_root:
            command += ["--store-root", store_root]
        else:
            command.append("--allow-real-store")
        if replace:
            command.append("--replace")
    inputs = {"tag": tag, "round": round_number, "stride": stride, "publish_to_store": to_store,
              "store_root": store_root or ("real label store" if to_store else None), "interpreter": str(interpreter)}
    context.log(f"building label-set drafts '{tag}' (round {round_number}, stride {stride})"
                + ("; publishing" if to_store else "; drafts only (publish_to_store is not enabled)"))
    code = run_worker(context, command)
    result = _read_json(out / "result.json")
    summary = _read_json(out / "summary.json")
    rel = str(out.relative_to(workspace.root))
    if code == 2 and result.get("refused"):
        return StageResult(status="blocked", message=f"publish refused: {result['refused']}", inputs=inputs,
                           metrics={"drafts": rel, "validation_errors": result.get("validation_errors", 0)},
                           outputs={"drafts": rel})
    if code != 0 or not summary:
        errors = (summary.get("check") or {}).get("errors") or []
        reason = result.get("error") or ("; ".join(e.splitlines()[0] for e in errors[:3]) if errors else log_tail(context))
        return StageResult(status="failed", message=f"publish worker exited with code {code}: {reason}", inputs=inputs,
                           metrics={"drafts": rel, "validation_errors": (summary.get("check") or {}).get("error_count")},
                           outputs={"drafts": rel} if summary else {})

    check = summary.get("check") or {}
    detections, tracks, alignment = summary["detections"], summary["tracks"], summary["alignment"]
    per_set = {
        "frame_alignment": {"name": summary["sets"]["frame_alignment"], "checkpoint_mode": alignment["mode"],
                            "sync_indices": summary["published_sync_indices"]},
        "two_d_tracklets": {"name": summary["sets"]["two_d_tracklets"], "detections": detections["published"],
                            "share_in_3d_tracks": detections["share_in_3d_tracks"],
                            "cameras": sum(1 for c in detections["per_camera"].values() if c.get("published_detections"))},
        "three_d_tracks": {"name": summary["sets"]["three_d_tracks"], "tracks": tracks["tracks"],
                           "positions": tracks["positions"], "members": tracks["members"]},
    }
    metrics = {
        "sets": per_set, "validation_errors": check.get("error_count", 0), "validation_warnings": check.get("warning_count", 0),
        "drafts": rel, "published": [s["name"] for s in summary.get("published", [])],
        "alignment_max_frame_diff": alignment["positions"]["max_abs_diff"], "bytes": summary["bytes_total"],
        "open_unknowns": summary["open_unknowns"], "stride": stride,
        "commit_problem": summary["provenance"]["commit_problem"],
    }
    if summary.get("published"):
        message = f"published {len(summary['published'])} label sets to {summary['store']['root']}"
    else:
        message = "drafts validated, not published"
        if to_store:
            message += " (publishing was requested but did not happen)"
        elif summary["provenance"]["commit_problem"]:
            message += "; publishing will be refused until the data_engine repository has a commit"
    message += (f": {detections['published']} detections, {tracks['tracks']} tracks, {tracks['positions']} positions "
                f"at stride {stride}")
    return StageResult(metrics=metrics, message=message, inputs=inputs, outputs={"drafts": rel})
