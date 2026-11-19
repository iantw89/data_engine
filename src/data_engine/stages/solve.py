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
MIN_IDENTITY_DETECTIONS = 150  # a 3D track this small (a few seconds) is asked about as a group


def _iou(a, b) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    return float(intersection / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection))


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
    # Where in each image do 3D-tracked people appear? A single-camera piece that stays in image
    # cells no 3D-tracked person ever enters (a doorway into another room, a mirror, a desk
    # behind glass) cannot be measured in 3D at all: these are grouped into one scope decision
    # per camera instead of one question per piece.
    grid_w, grid_h = 40, 30
    visited = []
    for camera in scene.cameras:
        mask = np.zeros((grid_h, grid_w), dtype=bool)
        lab = labels.label[camera.index]
        in_track = np.flatnonzero((lab >= 0) & np.isin(lab, list(multi)))
        if len(in_track):
            boxes = camera.detections.boxes[in_track]
            width = float(scene.workspace.dataset.cameras[camera.camera].width)
            height = float(scene.workspace.dataset.cameras[camera.camera].height)
            gx = np.clip(((boxes[:, 0] + boxes[:, 2]) / 2 / width * grid_w).astype(int), 0, grid_w - 1)
            gy = np.clip(((boxes[:, 1] + boxes[:, 3]) / 2 / height * grid_h).astype(int), 0, grid_h - 1)
            mask[gy, gx] = True
            grown = mask.copy()  # one cell of tolerance around visited cells
            grown[1:, :] |= mask[:-1, :]
            grown[:-1, :] |= mask[1:, :]
            grown[:, 1:] |= mask[:, :-1]
            grown[:, :-1] |= mask[:, 1:]
            mask = grown
        visited.append(mask)
    outside: dict[str, dict] = {}
    loose: dict[int, list[dict]] = {}
    for atom, rows in enumerate(association.atom_rows):
        camera_index = int(association.atom_camera[atom])
        camera = scene.cameras[camera_index]
        rows = rows[(labels.atom[camera_index][rows] == atom) & (camera.t[rows] >= 0)]
        if not len(rows) or atom_label.get(atom, -1) in multi:
            continue
        if (labels.duplicate_of[camera_index][rows] >= 0).mean() > 0.5:
            continue  # a second detection of an already tracked person: nothing to ask
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
        if len(rows) >= MIN_TASK_DETECTIONS and f"policy:review_individually:{camera.camera}" not in resolved_keys:
            boxes = camera.detections.boxes[rows]
            size = scene.workspace.dataset.cameras[camera.camera]
            gx = np.clip(((boxes[:, 0] + boxes[:, 2]) / 2 / size.width * grid_w).astype(int), 0, grid_w - 1)
            gy = np.clip(((boxes[:, 1] + boxes[:, 3]) / 2 / size.height * grid_h).astype(int), 0, grid_h - 1)
            if visited[camera_index][gy, gx].mean() <= 0.1:
                group = outside.setdefault(camera.camera, {"pieces": 0, "detections": 0, "examples": []})
                group["pieces"] += 1
                group["detections"] += int(len(rows))
                group["examples"].append((int(len(rows)), keys[len(keys) // 2]))
                continue
        loose.setdefault(camera_index, []).append({
            "rows": rows, "keys": keys, "t0": int(camera.t[rows[0]]), "t1": int(camera.t[rows[-1]]),
            "first_box": camera.detections.boxes[rows[0], :4], "last_box": camera.detections.boxes[rows[-1], :4]})
    # Consecutive single-camera pieces of someone who stays in place (a tracker id change or a
    # cut, the box reappearing where it was within 5 s) are one question, not one per piece.
    for camera_index, pieces in loose.items():
        camera = scene.cameras[camera_index]
        pieces.sort(key=lambda piece: piece["t0"])
        chains: list[list[dict]] = []
        for piece in pieces:
            for chain in chains:
                tail = chain[-1]
                if 0 <= piece["t0"] - tail["t1"] <= 50 and _iou(tail["last_box"], piece["first_box"]) >= 0.5:
                    chain.append(piece)
                    break
            else:
                chains.append([piece])
        for chain in chains:
            total = int(sum(len(piece["rows"]) for piece in chain))
            if total < MIN_TASK_DETECTIONS:
                short[camera.camera] += total
                continue
            largest = sorted(chain, key=lambda piece: -len(piece["rows"]))[:4]
            unknowns["unassigned"].append({
                "subject": f"unassigned:{camera.detections.key(int(chain[0]['rows'][0]))}", "severity": float(total),
                "detail": {"camera": camera.camera, "detections": total, "pieces": len(chain),
                           "t": int((chain[0]["t0"] + chain[-1]["t1"]) // 2), "span_t": [chain[0]["t0"], chain[-1]["t1"]],
                           "reference_keys": [piece["keys"][len(piece["keys"]) // 2] for piece in
                                              sorted(largest, key=lambda piece: piece["t0"])],
                           # one key per piece: an answer covers the whole chain
                           "member_keys": [piece["keys"][len(piece["keys"]) // 2] for piece in chain],
                           "why": "seen by one camera only, with no geometric hypothesis linking it to a tracked person",
                           "resolve_by": "a human says whether it is a person (single-view track) or a false detection"}})
    for camera_name, group in outside.items():
        if f"policy:outside_coverage:{camera_name}" in resolved_keys:
            continue
        examples = [key for _, key in sorted(group["examples"], reverse=True)[:4]]
        unknowns["unassigned"].append({
            "subject": f"unassigned:outside_coverage:{camera_name}", "severity": float(group["detections"]),
            "detail": {"camera": camera_name, "detections": group["detections"], "pieces": group["pieces"],
                       "policy_question": True, "policy": "outside_coverage", "reference_keys": examples,
                       "why": "people seen only by this camera, in a part of its image where nobody tracked in 3D "
                              "ever appears (for example through a doorway into another room): no second view "
                              "exists, so they cannot be measured in 3D",
                       "resolve_by": "one operator decision for the whole group: out of scope, keep as "
                                     "single-camera 2D tracks, or review piece by piece"}})
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
    identity_size: dict[int, int] = {}
    for summary in summaries.values():
        identity_size[int(identity[summary.label])] = identity_size.get(int(identity[summary.label]), 0) + summary.detections
    brief = {"count": 0, "detections": 0, "examples": []}
    for end in solved.links.open_ends:
        if end["end"] != "start":
            continue
        label = end["component"]
        keys = component_keys_at(scene, labels, label, end["t"] + 5, count=3, tolerance=40)
        if not keys or resolved_keys & set(keys):
            continue
        size = identity_size.get(int(identity[label]), 0)
        if size < MIN_IDENTITY_DETECTIONS and "policy:review_short_identities" not in resolved_keys:
            brief["count"] += 1
            brief["detections"] += size
            brief["examples"].append((size, keys[0]))
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

    if brief["count"] and "policy:short_identities" not in resolved_keys:
        unknowns["identity_link"].append({
            "subject": "identity_link:short", "severity": float(brief["detections"]),
            "detail": {"policy": "short_identities", "tracks": brief["count"], "detections": brief["detections"],
                       "reference_keys": [key for _, key in sorted(brief["examples"], reverse=True)[:4]],
                       "why": f"{brief['count']} short 3D tracks (under {MIN_IDENTITY_DETECTIONS} detections each) start "
                              "inside the recording without a unique continuation; each is a correct but brief track "
                              "whose person is not established",
                       "resolve_by": "one operator decision: leave them as separate short tracks, or review each"}})

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
        duplicate = solved.labels.duplicate_of[camera.index]
        arrays[f"duplicate_of_{camera.camera}"] = np.where(  # identity this row is a second detection of
            duplicate >= 0, solved.identity_of_label[np.clip(duplicate, 0, None)], -1).astype(np.int32)
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
    evidence = workspace.path("evidence", "autofit", "labels.npz")
    if evidence.is_file():
        with np.load(evidence) as data:
            constraints.external = [
                data[f"raw_{camera.camera}"].astype(np.int64) if f"raw_{camera.camera}" in data.files
                else np.full(len(camera.kept), -1, dtype=np.int64) for camera in scene.cameras]
        info["used"]["autofit_tracks"] = int(sum(len(np.unique(e[e >= 0])) for e in constraints.external))
        context.log(f"using Autofit raw tracks as independent evidence ({info['used']['autofit_tracks']} track ids)")
    else:
        context.log("no Autofit evidence: geometry-only solve (expect many more questions)", level="warning")
    # Evidence at every sync frame up to about an hour (the golden bench was scored that way);
    # only longer recordings subsample to 5 Hz to bound memory.
    long_recording = scene.T > 40000
    solved = solve(scene, RECIPE, constraints=constraints, step=2 if long_recording else 1, log=context.log)
    outputs = save_solution(context, solved)
    unknowns = extract_unknowns(solved, info["resolved_keys"])
    # Alignment: the generator's "stale" spans are replaced by a direct geometric check.
    from ..solve import alignment_check
    checked = alignment_check.verify(scene, RECIPE)
    (workspace.round_dir(context.round) / "alignment_check.json").write_text(json.dumps(checked, indent=1), encoding="utf-8")
    unknowns["alignment"] = []
    decided = {c["a"][0] for c in context.db.constraints("alignment") if c["a"]}
    for camera in scene.cameras:
        entries = checked.get(camera.camera)
        if entries is None:
            if camera.camera not in decided:
                unknowns["alignment"].append({
                    "subject": f"alignment:{camera.camera}", "severity": float(camera.detections.frame_count),
                    "detail": {"camera": camera.camera, "reasons": ["no resolved time offset: the camera has no "
                                                                     "frames on the common timeline"],
                               "explanation": f"{camera.camera} could not be tied to the other cameras in time, so "
                                              "none of its detections take part in 3D."}})
            continue
        for entry in entries:
            if entry["verdict"] == "disagrees" and camera.camera not in decided:
                start, end = entry["t"]
                unknowns["alignment"].append({
                    "subject": f"alignment:{camera.camera}:{start}-{end}", "severity": float(end - start),
                    "detail": {"camera": camera.camera, "block_sync_frames": entry["t"], "best_lag_frames": entry["best_lag"],
                               "agreeing_pairs_by_lag": entry["counts"],
                               "reasons": [f"cross-camera geometry agrees best when this camera is shifted by "
                                           f"{entry['best_lag']} frame(s)"],
                               "explanation": f"Between {start / 10:.0f} s and {end / 10:.0f} s, {camera.camera}'s "
                                              f"detections line up with the other cameras better at a shift of "
                                              f"{entry['best_lag']} frame(s) than at the estimated offset."}})
    verdicts = [e["verdict"] for entries in checked.values() for e in entries]
    alignment_summary = {v: verdicts.count(v) for v in set(verdicts)}
    context.log(f"alignment check by geometry: {alignment_summary}")
    coverage = coverage_metrics(solved)
    metrics = {"coverage": coverage, "constraints_used": info["used"], "alignment_check": alignment_summary,
               "refinement": [{k: v for k, v in e.items() if not isinstance(v, list)} for e in solved.log],
               "timings_s": {k: round(v, 1) for k, v in solved.timings.items()},
               "open_unknowns": {kind: len(items) for kind, items in unknowns.items()}}
    message = (f"{coverage['identities']} identities, {coverage['share_detections_in_3d_tracks']:.0%} of detections "
               f"in 3D tracks, {coverage['person_frames_with_3d_root']} person-frames with 3D")
    return StageResult(metrics=metrics, message=message, outputs=outputs,
                       inputs={"constraints": info["used"], "evidence_step": 2 if long_recording else 1},
                       unknowns=unknowns)
