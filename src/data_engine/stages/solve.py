"""Solve stage: association, 3D tracks and identity, with the unknowns that remain.

Inputs: tracklets, alignment sync table, calibration, and every active human constraint.
Outputs (``rounds/rNNN/``):
    solution.npz   per camera: ``identity_<cam>``, ``component_<cam>``, ``source_<cam>`` per raw row
    tracks3d.npz   one row per (identity component, sync index): xyz (17x3), root, views, residual
    solve.json     log of what was merged/attached/linked and why

Unknown subjects are built from detection keys, so they stay stable when tracklets,
components or identities are renumbered by a later round.
"""

from __future__ import annotations

import json

import numpy as np

from ..config import RECIPE
from ..engine import Context, StageResult
from ..solve import recover
from ..solve.pipeline import Constraints, Solved, coverage_metrics, solve
from ..solve.scene import Scene, load_scene
from ..store import parse_det_key

WINDOW = 20  # sync frames per evidence window (2 s)
MIN_TASK_DETECTIONS = 30  # a free piece shorter than 3 s is not worth an individual question


def load_constraints(context: Context, scene: Scene) -> tuple[Constraints, dict]:
    """Translate active constraints (detection keys) into rows of this scene."""
    index = {camera.camera: camera for camera in scene.cameras}

    def rows(keys: list[str]) -> list[tuple[int, int]]:
        result = []
        for key in keys:
            try:
                camera, frame, det = parse_det_key(key)
                result.append((index[camera].index, index[camera].detections.row(frame, det)))
            except (KeyError, ValueError):
                pass  # not a detection of this scene (e.g. a policy key or an excluded camera)
        return result

    constraints = Constraints()
    used = {"same": 0, "different": 0, "duplicate": 0, "not_person": 0, "split": 0, "undeterminable": 0}
    resolved_keys: set[str] = set()
    for row in context.db.constraints():
        kind = row["kind"]
        a, b = rows(row["a"]) if kind in used else [], rows(row["b"]) if kind in used else []
        if kind == "same" and a and b:
            constraints.same.append((a, b))
        elif kind == "different" and a and b:
            constraints.different.append((a, b))
        elif kind == "duplicate" and a and b:
            constraints.duplicate.append((a, b))
            constraints.same.append((a, b))
        elif kind == "not_person" and a:
            # The human judged the piece from sample crops: reject the whole tracklet of each sample.
            for camera_index, sample in a:
                camera = scene.cameras[camera_index]
                tracklet = int(camera.tracklet[sample])
                members = np.flatnonzero(camera.tracklet == tracklet) if tracklet >= 0 else [sample]
                constraints.not_person.extend((camera_index, int(member)) for member in members)
        elif kind == "split" and b:
            # Cut the tracklet of the first "after" detection at that detection's window.
            camera_index, first = b[0]
            camera = scene.cameras[camera_index]
            tracklet, t = int(camera.tracklet[first]), int(camera.t[first])
            if tracklet >= 0 and t >= 0:
                constraints.cuts.setdefault(tracklet, []).append(t // WINDOW)
            if a:
                constraints.different.append((a, b))
        elif kind == "undeterminable":
            resolved_keys.update(row["a"])
        else:
            continue
        used[kind] += 1
    return constraints, {"used": used, "resolved_keys": resolved_keys}


def representative_keys(scene: Scene, camera_index: int, rows: np.ndarray, count: int = 3) -> list[str]:
    """Evenly spaced detections of a piece, as detection keys."""
    camera = scene.cameras[camera_index]
    rows = rows[camera.t[rows] >= 0]
    if not len(rows):
        return []
    picks = rows[np.unique(np.linspace(0, len(rows) - 1, min(count, len(rows))).round().astype(int))]
    return [camera.detections.key(int(r)) for r in picks]


def component_keys_at(scene: Scene, labels: recover.Labels, label: int, t: int, *, exclude_camera: int | None = None,
                      count: int = 2, tolerance: int = 15) -> list[str]:
    """Up to ``count`` detections of a component near sync index ``t`` (largest boxes, distinct cameras)."""
    options = []
    for camera in scene.cameras:
        if camera.index == exclude_camera:
            continue
        rows = np.flatnonzero(labels.label[camera.index] == label)
        rows = rows[camera.t[rows] >= 0]
        if not len(rows):
            continue
        nearest = rows[np.argmin(np.abs(camera.t[rows] - t))]
        if abs(int(camera.t[nearest]) - t) > tolerance:
            continue
        box = camera.detections.boxes[nearest]
        options.append((float((box[2] - box[0]) * (box[3] - box[1])), camera.detections.key(int(nearest))))
    return [key for _, key in sorted(options, reverse=True)[:count]]


def extract_unknowns(solved: Solved, resolved_keys: set[str]) -> dict[str, list[dict]]:
    scene, labels, association = solved.scene, solved.labels, solved.association
    cams = recover.camera_counts(scene, labels)
    multi = {label for label, cameras in cams.items() if len(cameras) >= 2}
    identity = solved.identity_of_label
    unknowns: dict[str, list[dict]] = {"association": [], "unassigned": [], "identity_link": [], "pose_ownership": [],
                                       "tracklet_purity": [], "coverage": []}

    # Hypotheses per free atom: partners from ambiguous pairwise edges and from projection voting.
    atom_label = {}
    for atom, rows in enumerate(association.atom_rows):
        if len(rows):
            atom_label[atom] = int(labels.label[association.atom_camera[atom]][rows[0]])
    hypotheses: dict[int, dict[int, dict]] = {}
    for edge in association.solution.ambiguous:
        for atom, partner in ((edge["a"], edge["b"]), (edge["b"], edge["a"])):
            partner_label = atom_label.get(partner, -1)
            if partner_label in multi and atom_label.get(atom, -1) not in multi:
                hypotheses.setdefault(atom, {})[partner_label] = {
                    "evidence": "two-view geometry", "share_within_gate": round(edge["f2"], 2), "samples": edge["n"],
                    "why_open": edge["reason"]}
    for event in solved.log:
        for entry in event.get("atoms_ambiguous", []):
            for option in entry["options"]:
                if option["label"] in multi:
                    hypotheses.setdefault(entry["atom"], {})[option["label"]] = {
                        "evidence": "projected 3D pose", "frames_matching": option["matches"],
                        "frames_compared": option["comparable"], "why_open": entry.get("reason", "competing matches")}

    short = {camera.camera: 0 for camera in scene.cameras}
    for atom, rows in enumerate(association.atom_rows):
        camera_index = int(association.atom_camera[atom])
        camera = scene.cameras[camera_index]
        rows = rows[(labels.atom[camera_index][rows] == atom) & (camera.t[rows] >= 0)]
        if not len(rows) or atom_label.get(atom, -1) in multi:
            continue
        keys = representative_keys(scene, camera_index, rows)
        if not keys or resolved_keys & set(keys):
            continue
        middle = int(camera.t[rows[len(rows) // 2]])
        subject_key = camera.detections.key(int(rows[0]))
        options = hypotheses.get(atom, {})
        if options:
            candidates = [{"component": int(label), "identity": int(identity[label]), **info,
                           "keys": component_keys_at(scene, labels, label, middle)}
                          for label, info in sorted(options.items(), key=lambda kv: -kv[1].get("samples", kv[1].get("frames_matching", 0)))]
            candidates = [c for c in candidates if c["keys"]][:4]
            if candidates:
                unknowns["association"].append({
                    "subject": f"association:{subject_key}", "severity": float(len(rows)),
                    "detail": {"camera": camera.camera, "detections": int(len(rows)), "t": middle,
                               "span_t": [int(camera.t[rows[0]]), int(camera.t[rows[-1]])],
                               "reference_keys": keys, "candidates": candidates,
                               "resolve_by": "a human picks which candidate is the same person, or none"}})
                continue
        if len(rows) >= MIN_TASK_DETECTIONS:
            unknowns["unassigned"].append({
                "subject": f"unassigned:{subject_key}", "severity": float(len(rows)),
                "detail": {"camera": camera.camera, "detections": int(len(rows)), "t": middle,
                           "span_t": [int(camera.t[rows[0]]), int(camera.t[rows[-1]])], "reference_keys": keys,
                           "why": "seen by one camera only, with no geometric hypothesis linking it to a tracked person",
                           "resolve_by": "a human says whether it is a person (single-view track) or a false detection"}})
        else:
            short[camera.camera] += int(len(rows))
    for camera_name, count in short.items():
        if count and f"policy:short:{camera_name}" not in resolved_keys:
            unknowns["unassigned"].append({
                "subject": f"unassigned:short:{camera_name}", "severity": float(count),
                "detail": {"camera": camera_name, "detections": count, "policy_question": True,
                           "why": f"pieces shorter than {MIN_TASK_DETECTIONS / 10:.0f} s seen only by this camera; "
                                  "too short to ask about one by one",
                           "resolve_by": "an operator decision: leave them out of the tracks (undeterminable) or review a sample"}})

    # Identity: components that start inside the recording without a predecessor.
    summaries = {s.label: s for s in solved.summaries if s is not None}
    for end in solved.links.open_ends:
        if end["end"] != "start":
            continue
        label = end["component"]
        keys = component_keys_at(scene, labels, label, end["t"] + 5, count=3, tolerance=40)
        if not keys or resolved_keys & set(keys):
            continue
        candidates = []
        for option in end["candidates"][:6]:
            other = summaries.get(option["component"])
            if other is None:
                continue
            other_keys = component_keys_at(scene, labels, other.label, other.t_last - 5, count=3, tolerance=40)
            if other_keys:
                candidates.append({"component": int(other.label), "identity": int(identity[other.label]),
                                   "gap_s": option["gap_s"], "distance_m": option["distance_m"], "keys": other_keys})
        unknowns["identity_link"].append({
            "subject": f"identity_link:{keys[0]}", "severity": float(summaries[label].detections),
            "detail": {"component": int(label), "identity": int(identity[label]), "t": int(end["t"]),
                       "reference_keys": keys, "candidates": candidates[:4],
                       "feasible_candidates": len(end["candidates"]),
                       "why": "this person appears inside the recording; "
                              + ("several earlier people could have reached this spot" if len(end["candidates"]) > 1
                                 else "one earlier person could have reached this spot but not within the automatic gate"
                                 if end["candidates"] else "nobody tracked earlier could have reached this spot"),
                       "resolve_by": "a human picks the earlier person, or says it is someone not tracked before"}})

    # Two components measured at the same place that share a camera at the same time: duplicate poses.
    seen = set()
    for event in solved.log:
        for refused in event.get("merge_refused", []):
            if refused.get("reason") != "same_camera_overlap":
                continue
            a, b = sorted((refused["a"], refused["b"]))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            t = summaries[a].t_first if a in summaries else 0
            for camera in scene.cameras:
                rows_a = np.flatnonzero(labels.label[camera.index] == a)
                rows_b = np.flatnonzero(labels.label[camera.index] == b)
                shared = np.intersect1d(camera.t[rows_a], camera.t[rows_b])
                shared = shared[shared >= 0]
                if not len(shared):
                    continue
                t = int(shared[len(shared) // 2])
                key_a = camera.detections.key(int(rows_a[camera.t[rows_a] == t][0]))
                key_b = camera.detections.key(int(rows_b[camera.t[rows_b] == t][0]))
                if resolved_keys & {key_a, key_b}:
                    break
                unknowns["pose_ownership"].append({
                    "subject": f"pose_ownership:{key_a}|{key_b}", "severity": float(len(shared)),
                    "detail": {"camera": camera.camera, "t": t, "keys": [key_a, key_b],
                               "components": [int(a), int(b)], "frames_shared": int(len(shared)),
                               "median_distance_in": refused["median_in"],
                               "why": "two tracked people are measured at the same place while one camera shows two poses",
                               "resolve_by": "a human says whether the two poses are one person (duplicate) or two"}})
                break
    return unknowns


def save_solution(context: Context, solved: Solved) -> dict[str, str]:
    scene = solved.scene
    directory = context.workspace.round_dir(context.round)
    arrays = {"cameras": np.array([c.camera for c in scene.cameras]), "component_identity": solved.identity_of_label}
    cams = recover.camera_counts(scene, solved.labels)
    multi = np.zeros(solved.labels.next_label() + 1, dtype=bool)
    for label, cameras in cams.items():
        multi[label] = len(cameras) >= 2
    for camera in scene.cameras:
        component = solved.labels.label[camera.index]
        identity = np.where((component >= 0) & multi[np.clip(component, 0, None)],
                            solved.identity_of_label[np.clip(component, 0, None)], -1).astype(np.int32)
        arrays[f"component_{camera.camera}"] = component
        arrays[f"identity_{camera.camera}"] = identity
        arrays[f"source_{camera.camera}"] = solved.labels.source[camera.index]
        arrays[f"atom_{camera.camera}"] = solved.labels.atom[camera.index]  # tracklet piece per row, -1 if none
    np.savez_compressed(directory / "solution.npz", **arrays)
    frames = solved.frames
    keep = multi[frames.label]
    np.savez_compressed(
        directory / "tracks3d.npz", component=frames.label[keep], identity=solved.identity_of_label[frames.label[keep]],
        t=frames.t[keep], rows=frames.rows[keep], xyz=frames.xyz[keep], used=frames.used[keep],
        residual=frames.residual[keep], root=frames.root[keep], aligned_time=scene.aligned_time,
        native=scene.native, cameras=np.array([c.camera for c in scene.cameras]))
    log = {"events": solved.log, "links": solved.links.accepted, "timings": solved.timings,
           "switch_events": solved.association.switch_events,
           "accepted_edges": len(solved.association.solution.accepted)}
    (directory / "solve.json").write_text(json.dumps(log, indent=1, default=int), encoding="utf-8")
    root = context.workspace.root
    return {name: str((directory / name).relative_to(root)) for name in ("solution.npz", "tracks3d.npz", "solve.json")}


def run(context: Context) -> StageResult:
    workspace = context.workspace
    scene = load_scene(workspace, joint_peak=RECIPE["geometry"]["joint_peak"])
    constraints, info = load_constraints(context, scene)
    long_recording = scene.T > 6000
    solved = solve(scene, RECIPE, constraints=constraints, step=2 if long_recording else 1, log=context.log)
    outputs = save_solution(context, solved)
    unknowns = extract_unknowns(solved, info["resolved_keys"])
    coverage = coverage_metrics(solved)
    metrics = {"coverage": coverage, "constraints_used": info["used"],
               "refinement": [{k: v for k, v in e.items() if not isinstance(v, list)} for e in solved.log],
               "timings_s": {k: round(v, 1) for k, v in solved.timings.items()},
               "open_unknowns": {kind: len(items) for kind, items in unknowns.items()}}
    message = (f"{coverage['identities']} identities, {coverage['share_detections_in_3d_tracks']:.0%} of detections "
               f"in 3D tracks, {coverage['person_frames_with_3d_root']} person-frames with 3D")
    return StageResult(metrics=metrics, message=message, outputs=outputs,
                       inputs={"constraints": info["used"], "evidence_step": 2 if long_recording else 1},
                       unknowns=unknowns)
