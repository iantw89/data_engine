"""Step 1: classify every missing person-frame by what the pixels/detections support.

Reference-free: uses only the solved workspace (tracks3d.npz, solution.npz, raw detections,
calibration). Neighbouring measured 3D poses give a search hint; the hint is projected into
every camera and compared with the raw detections (kept, post-filter rejected, assigned to
other identities). Nothing from the hint is stored as a position.

Usage: python experiments/coverage/gap_classify.py [WORKSPACE] [ROUND]
Outputs: C:\\data_engine_data\\_scratch\\coverage\\gap_classify.{json,npz}
"""
import json
import sys
import time
from collections import Counter, defaultdict

from common import *  # noqa
from hints import (MARGIN, build_presence, clip_share, box_iou, keypoint_agreement, keypoint_box, make_hint,
                   pair_angle_deg, project_hint)

REACH = 30           # frames: neighbouring measured pose must be this close to give a hint
END_FRAMES = 10      # look this far beyond first / last measured position (1 s)
MIN_VISIBLE_SHARE = 0.6
MIN_BODY_PROJECTED = 6
NEAR_IOU = 0.3
NEAR_AGREE = 0.6
BUCKETS = [("1", 1, 1), ("2-5", 2, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21-50", 21, 50), ("51+", 51, 10**9)]


def bucket(length: int) -> str:
    for name, a, b in BUCKETS:
        if a <= length <= b:
            return name
    return "?"


def camera_state(scene, solution, c, t, hint, who):
    """Per-camera view of the hint: visibility, the nearest detections and their state."""
    camera = scene.cameras[c]
    out = {"visible": False, "state": "oob", "row": -1, "iou": 0.0, "agree": 0.0, "n_near": 0}
    if camera.cam is None:
        return out
    frame = int(scene.native[t, c])
    if frame < 0:
        out["state"] = "no_frame"
        return out
    xy = project_hint(camera.cam, hint.pose)
    width, height = camera.cam.native_width, camera.cam.native_width * 0.75
    body_ok = np.isfinite(xy[5:17]).all(axis=1) & (xy[5:17, 0] >= 0) & (xy[5:17, 0] < width) & (xy[5:17, 1] >= 0) & (xy[5:17, 1] < height)
    box = keypoint_box(xy, width, height)
    if box is None or body_ok.sum() < MIN_BODY_PROJECTED or clip_share(box, width, height) < MIN_VISIBLE_SHARE:
        return out
    out.update(visible=True, state="none", box=box, xy=xy)
    rows = camera.detections.rows(frame)
    if rows.stop <= rows.start:
        return out
    rr = np.arange(rows.start, rows.stop)
    iou = box_iou(box, camera.detections.boxes[rr, :4].astype(np.float64))
    best = None
    ident = solution[f"identity_{camera.camera}"]
    for k, row in enumerate(rr):
        agree, common = keypoint_agreement(camera.cam, xy, camera.detections.keypoints[row], camera.detections.boxes[row])
        share = agree / common if common >= 5 else 0.0
        if iou[k] >= NEAR_IOU or share >= NEAR_AGREE:
            owner = int(ident[row])
            kept = bool(camera.kept[row])
            if owner == who:
                kind = "own"
            elif owner >= 0:
                kind = "other"
            elif kept:
                kind = "unassigned"
            else:
                kind = "filtered"
            rank = {"own": 3, "unassigned": 2, "filtered": 2, "other": 1}[kind]
            key = (rank, share + iou[k])
            out["n_near"] += 1
            if best is None or key > best[0]:
                best = (key, kind, int(row), float(iou[k]), float(share), owner)
    if best is not None:
        out.update(state=best[1], row=best[2], iou=best[3], agree=best[4], owner=best[5])
    return out


def occluded_share(scene, c, t, hint, box, depth_hint, others_at_t, tracks, who):
    """Largest share of the hint box covered by a NEARER other person's projected skeleton box."""
    camera = scene.cameras[c]
    best = 0.0
    for row in others_at_t:
        if int(tracks["identity"][row]) == who:
            continue
        pose = tracks["xyz"][row].astype(np.float64)
        root = tracks["root"][row].astype(np.float64)
        if camera.cam.depth(root) >= depth_hint:
            continue
        xy = project_hint(camera.cam, pose)
        other = keypoint_box(xy, 0, 0)
        if other is None:
            continue
        x1, y1 = max(box[0], other[0]), max(box[1], other[1])
        x2, y2 = min(box[2], other[2]), min(box[3], other[3])
        share = max(x2 - x1, 0) * max(y2 - y1, 0) / max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
        best = max(best, share)
    return best


def detection_rate_map(scene, tracks, cell=48.0):
    """P(camera has a detection on a measured person | hint visible in camera), per camera and floor cell."""
    root = tracks["root"].astype(np.float64)
    ok = np.isfinite(root).all(axis=1)
    out = []
    for c, camera in enumerate(scene.cameras):
        seen = np.zeros(len(root), bool)
        if camera.cam is not None:
            p = np.full((len(root), 2), np.nan)
            p[ok] = camera.cam.project(root[ok])
            seen = np.isfinite(p).all(axis=1) & (p[:, 0] >= 0) & (p[:, 0] < camera.cam.native_width) & (p[:, 1] >= 0) & (p[:, 1] < camera.cam.native_width * 0.75)
        det = tracks["rows"][:, c] >= 0
        cells = np.floor(root[:, :2] / cell).astype(int)
        total, hit = defaultdict(int), defaultdict(int)
        for (i, j), s, d in zip(cells[ok & seen], np.ones(int((ok & seen).sum())), det[ok & seen]):
            total[(i, j)] += 1
            hit[(i, j)] += int(d)
        out.append({k: (hit[k], total[k]) for k in total})
    return out


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else WORKSPACE
    round_name = sys.argv[2] if len(sys.argv) > 2 else "r002"
    started = time.time()
    ws, scene, tracks, solution = load(name, round_name)
    presence = build_presence(tracks)
    by_t = defaultdict(list)
    for row in np.flatnonzero(np.isfinite(tracks["root"]).all(axis=1)):
        by_t[int(tracks["t"][row])].append(row)
    rate = detection_rate_map(scene, tracks)
    centers = [c.cam.center for c in scene.cameras if c.cam is not None]
    frames = []
    for who, ts in presence.t.items():
        first, last = int(ts[0]), int(ts[-1])
        todo = []
        for a, b in zip(ts[:-1], ts[1:]):
            if b - a > 1:
                gap = int(b - a - 1)
                todo += [(int(t), "interior", gap, k + 1) for k, t in enumerate(range(int(a) + 1, int(b)))]
        todo += [(t, "start", END_FRAMES, first - t) for t in range(max(0, first - END_FRAMES), first)]
        todo += [(t, "end", END_FRAMES, t - last) for t in range(last + 1, min(scene.T, last + END_FRAMES + 1))]
        for t, kind, gap, offset in todo:
            hint = make_hint(presence, who, t, reach=REACH)
            record = {"identity": who, "t": t, "kind": kind, "gap": gap, "offset": offset, "hint": hint is not None}
            if hint is None:
                record["cause"] = "no_hint"
                frames.append(record)
                continue
            record.update(hint_kind=hint.kind, hint_distance=hint.distance, span_in=round(hint.span, 1))
            states = [camera_state(scene, solution, c, t, hint, who) for c in range(len(scene.cameras))]
            visible = [c for c, s in enumerate(states) if s["visible"]]
            record["visible"] = visible
            record["states"] = [s["state"] for s in states]
            angle = pair_angle_deg(hint.root, [scene.cameras[c].cam.center for c in range(len(scene.cameras))])

            def pair_ok(cams):
                return any(angle[a, b] >= 8.0 for i, a in enumerate(cams) for b in cams[i + 1:])

            own = [c for c in visible if states[c]["state"] == "own"]
            usable = [c for c in visible if states[c]["state"] in ("own", "unassigned", "filtered")]
            other = [c for c in visible if states[c]["state"] == "other"]
            none = [c for c in visible if states[c]["state"] == "none"]
            record.update(n_own=len(own), n_usable=len(usable), n_other=len(other), n_none=len(none))
            # Is another identity already measured at the hinted place at this instant (fragmented identity)?
            near_other = any(int(tracks["identity"][r]) != who and
                             np.linalg.norm(tracks["root"][r][:2] - hint.root[:2]) < 24.0 for r in by_t.get(t, []))
            record["covered_by_other_identity"] = bool(near_other)
            # blocking evidence for the cameras that have no detection
            occ = {}
            for c in none + other:
                s = states[c]
                share = occluded_share(scene, c, t, hint, s["box"], scene.cameras[c].cam.depth(hint.root), by_t.get(t, []), tracks, who)
                cell = (int(np.floor(hint.root[0] / 48.0)), int(np.floor(hint.root[1] / 48.0)))
                hits, total = rate[c].get(cell, (0, 0))
                occ[c] = (round(share, 2), round(hits / total, 2) if total >= 20 else None)
            record["occlusion"] = occ
            if not pair_ok(visible):
                cause = "d_unobservable"          # fewer than 2 cameras (8 deg apart) can see the hinted place
            elif len(own) >= 2 and pair_ok(own):
                cause = "a_gates_failed"          # 2+ views carry this identity's detections yet no 3D pose
            elif pair_ok(usable):
                cause = "b_existing_detections"   # own + unassigned + filtered detections reach 2 views
            elif len(usable) + len(other) >= 2 and pair_ok(usable + other):
                cause = "b2_needs_other_label"    # would need a detection that carries another identity
            elif len(usable) >= 1 or len(other) >= 1:
                cause = "c1_one_view_only"        # exactly one view has any detection near the hint
            else:
                cause = "c2_no_detection"         # visible in >=2 views, no detection anywhere near
            record["cause"] = cause
            # plausible-occlusion annotation for the (c) causes
            if cause.startswith("c"):
                blocked = [c for c in none if occ[c][0] >= 0.5 or (occ[c][1] is not None and occ[c][1] < 0.2)]
                record["n_blocked"] = len(blocked)
                record["occluded_plausible"] = bool(len(visible) - len(blocked) < 2 or not pair_ok([c for c in visible if c not in blocked]))
            frames.append(record)
    # ------------------------------------------------------------------ summaries
    def table(selection):
        rows = {}
        for name_, a, b in BUCKETS:
            sel = [f for f in selection if a <= f["gap"] <= b]
            counts = Counter(f["cause"] for f in sel)
            rows[name_] = {"frames": len(sel), **dict(sorted(counts.items()))}
        return rows

    interior = [f for f in frames if f["kind"] == "interior"]
    ends = [f for f in frames if f["kind"] != "interior"]
    gaps = Counter()
    gap_cause = defaultdict(Counter)
    seen = {}
    for f in interior:
        key = (f["identity"], f["t"] - f["offset"])
        seen.setdefault(key, []).append(f["cause"])
    for key, causes in seen.items():
        n = len(causes)
        gaps[bucket(n)] += 1
        gap_cause[bucket(n)]["all_frames_" + "recoverable" if all(c.startswith(("a", "b")) for c in causes) else "mixed_or_unrecoverable"] += 1
    summary = {
        "workspace": name, "round": round_name, "seconds": round(time.time() - started),
        "interior_missing_frames": len(interior), "end_frames_examined": len(ends),
        "interior_by_gap_length": table(interior),
        "ends": {"start": dict(Counter(f["cause"] for f in ends if f["kind"] == "start")),
                 "end": dict(Counter(f["cause"] for f in ends if f["kind"] == "end"))},
        "interior_totals": dict(Counter(f["cause"] for f in interior)),
        "covered_by_other_identity": {"interior": sum(f.get("covered_by_other_identity", False) for f in interior),
                                      "ends": sum(f.get("covered_by_other_identity", False) for f in ends)},
        "c_plausibly_occluded": {"yes": sum(1 for f in interior if f.get("occluded_plausible")),
                                 "no": sum(1 for f in interior if f["cause"].startswith("c") and not f.get("occluded_plausible"))},
        "gap_counts_by_length": dict(gaps),
        "gaps_whole_gap_recoverable_by_existing_or_gates": {k: dict(v) for k, v in gap_cause.items()},
    }
    (SCRATCH / "gap_classify.json").write_text(json.dumps(summary, indent=1))
    (SCRATCH / "gap_classify_frames.json").write_text(json.dumps(frames, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
