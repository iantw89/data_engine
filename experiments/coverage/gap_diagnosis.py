"""Where do 3D tracks lose positions that the footage supports?

Two measures on a solved workspace:
1. Temporal gaps inside each identity: sync frames between its first and last 3D position
   with no position, by gap length (reference-free; works on any dataset).
2. With golden 2D identity labels: every (person, labelled instant) that at least two cameras
   show, and why the engine has no 3D position there.

Usage: python experiments/coverage/gap_diagnosis.py WORKSPACE [GOLDEN_DATASET]
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from data_engine.config import RECIPE, SITE_DEFAULTS  # noqa: E402
from data_engine.evaluation import box_iou_matrix  # noqa: E402
from data_engine.solve.scene import load_scene  # noqa: E402
from data_engine.workspace import Workspace  # noqa: E402

EDGES = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 300), (301, 10**9)]


def latest(workspace: Workspace, name: str) -> Path:
    return Path(sorted(glob.glob(str(workspace.root / "rounds" / "*" / name)))[-1])


def temporal_gaps(tracks) -> dict:
    ok = np.isfinite(tracks["root"]).all(axis=1)
    identity, t = tracks["identity"][ok], tracks["t"][ok]
    lengths = []
    present = 0
    for who in np.unique(identity):
        times = np.unique(t[identity == who])
        present += len(times)
        lengths += [int(g) - 1 for g in np.diff(times) if g > 1]
    lengths = np.array(lengths)
    table = {}
    for a, b in EDGES:
        inside = (lengths >= a) & (lengths <= b)
        table[f"{a}-{b if b < 10**9 else 'inf'}"] = {"gaps": int(inside.sum()), "missing_frames": int(lengths[inside].sum())}
    return {"identities": int(len(np.unique(identity))), "person_frames_with_position": present,
            "gaps": int(len(lengths)), "missing_frames_inside_tracks": int(lengths.sum()), "by_length_frames": table}


def golden_gaps(scene, tracks, solution, dataset: str, tag: str = "cvat_identity") -> dict:
    data = Path(SITE_DEFAULTS["repos"]["data_management"]) / "data" / "labels" / "data"
    owner = defaultdict(list)
    reference = json.loads((data / f"{dataset}.three_d_tracks.{tag}" / "tracks.json").read_text(encoding="utf-8"))
    for track in reference["tracks"]:
        for member in track["members"]:
            owner[(member["camera"], int(member["tracklet_id"]))].append((*member["frame_range"], int(track["track_id"])))
    ok = np.isfinite(tracks["root"]).all(axis=1)
    in_3d = [np.zeros(len(c.kept), bool) for c in scene.cameras]
    for column in range(len(scene.cameras)):
        rows = tracks["rows"][ok, column]
        in_3d[column][rows[rows >= 0]] = True
    views = defaultdict(list)  # (person, t) -> [(camera, state, row)]
    for camera in scene.cameras:
        path = data / f"{dataset}.two_d_tracklets.{tag}" / "cameras" / f"{camera.camera}.json"
        sync_of_frame = np.full(camera.detections.frame_count, -1, np.int64)
        shown = np.flatnonzero(scene.native[:, camera.index] >= 0)
        sync_of_frame[scene.native[shown, camera.index][::-1]] = shown[::-1]
        for entry in json.loads(path.read_text(encoding="utf-8"))["frames"]:
            frame = int(entry["frame"])
            if frame >= camera.detections.frame_count or sync_of_frame[frame] < 0:
                continue
            boxes, people = [], []
            for detection in entry["detections"]:
                who = [p for first, last, p in owner.get((camera.camera, int(detection["tracklet_id"])), ())
                       if first <= frame <= last]
                if len(who) == 1 and detection.get("bbox"):
                    boxes.append(detection["bbox"])
                    people.append(who[0])
            if not boxes:
                continue
            rows = camera.detections.rows(frame)
            if rows.stop > rows.start:
                overlap = box_iou_matrix(np.asarray(boxes, float), camera.detections.boxes[rows, :4].astype(float))
            else:
                overlap = np.zeros((len(boxes), 0))
            for index, who in enumerate(people):
                state, row = "no_detection", -1
                candidates = np.flatnonzero(overlap[index] >= 0.5) if overlap.shape[1] else []
                if len(candidates):
                    kept = [c for c in candidates if camera.kept[rows.start + c]]
                    best = max(kept or list(candidates), key=lambda c: overlap[index, c])
                    row = rows.start + int(best)
                    state = "kept" if camera.kept[row] else "filtered"
                views[(who, int(sync_of_frame[frame]))].append((camera.index, state, row))
    identity = [solution[f"identity_{c.camera}"] for c in scene.cameras]
    reasons = Counter()
    determinable = covered = 0
    per_person = defaultdict(lambda: [0, 0])
    for (who, t), seen in views.items():
        if len(seen) < 2:
            reasons["(one golden view only: not counted)"] += 1
            continue
        determinable += 1
        per_person[who][0] += 1
        if any(row >= 0 and in_3d[c][row] for c, _, row in seen):
            covered += 1
            per_person[who][1] += 1
            continue
        kept = [(c, row) for c, state, row in seen if state == "kept"]
        raw = [(c, row) for c, state, row in seen if state != "no_detection"]
        if len(kept) >= 2:
            assigned = [int(identity[c][row]) for c, row in kept if identity[c][row] >= 0]
            if len(assigned) >= 2 and len(set(assigned)) == 1:
                reasons["2+ kept views on one identity, triangulation gate failed"] += 1
            elif len(assigned) >= 2:
                reasons["2+ kept views split over identities"] += 1
            elif len(assigned) == 1:
                reasons["2+ kept views, only one assigned"] += 1
            else:
                reasons["2+ kept views, none assigned"] += 1
        elif len(raw) >= 2:
            reasons["2+ detections but post-filter left fewer than 2"] += 1
        else:
            reasons["fewer than 2 detections (detector miss in a golden view)"] += 1
    worst = sorted(((n - c, who, n) for who, (n, c) in per_person.items()), reverse=True)[:6]
    return {"person_instants_seen_by_2plus_golden_views": determinable, "with_engine_3d_position": covered,
            "share": round(covered / max(1, determinable), 4), "missing_by_reason": dict(reasons.most_common()),
            "most_missing_people": [{"person": who, "missing": miss, "of": n} for miss, who, n in worst]}


def main() -> None:
    workspace = Workspace(sys.argv[1])
    tracks = dict(np.load(latest(workspace, "tracks3d.npz")))
    out = {"workspace": sys.argv[1], "round_file": str(latest(workspace, "tracks3d.npz")), "temporal": temporal_gaps(tracks)}
    if len(sys.argv) > 2:
        scene = load_scene(workspace, joint_peak=RECIPE["geometry"]["joint_peak"], rays=False)
        out["golden"] = golden_gaps(scene, tracks, dict(np.load(latest(workspace, "solution.npz"))), sys.argv[2])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
