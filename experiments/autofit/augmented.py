"""Does Autofit track better when it also gets the poses the recover stage read?

    python experiments/autofit/augmented.py extras WORKSPACE OUT.npz      # recovered poses as extra detections
    python experiments/autofit/augmented.py compare WORKSPACE BASE_LABELS AUGMENTED_LABELS [--golden DATASET]

``extras`` writes the recovered poses that do not coincide with a kept detection (the same rule as
publishing) in the form ``workers/autofit_tracks.py --extra`` reads. ``compare`` counts, on the raw rows
both runs share: tracker ids, how long they live, how the two runs' ids map onto each other (ids of the
base run that the augmented run joined), and with --golden the fragmentation and purity per labelled person.
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from data_engine.config import RECIPE, SITE_DEFAULTS  # noqa: E402
from data_engine.solve import coverage  # noqa: E402
from data_engine.solve.scene import load_scene  # noqa: E402
from data_engine.workspace import Workspace  # noqa: E402


def extras(scene, recovered: dict, overlap: float = 0.5) -> dict:
    out = defaultdict(list)
    for camera in scene.cameras:
        frames = recovered["pose_frame"][:, camera.index].astype(np.int64)
        order = np.argsort(camera.detections.frame_of_row, kind="stable")
        frame_sorted = camera.detections.frame_of_row[order]
        with np.load(scene.workspace.root / "tracklets" / f"{camera.camera}.npz") as data:
            kept = data["kept"]
        for index in np.flatnonzero(frames >= 0):
            box = recovered["pose_box"][index, camera.index].astype(np.float64)
            lo, hi = np.searchsorted(frame_sorted, [frames[index], frames[index] + 1])
            rows = order[lo:hi]
            rows = rows[kept[rows]]  # Autofit is fed the kept detections only
            if len(rows) and max(coverage.box_iou(box, other) for other in
                                 camera.detections.boxes[rows, :4].astype(np.float64)) >= overlap:
                continue
            keypoints = recovered["pose_keypoints"][index, camera.index].astype(np.float32)
            out["camera"].append(camera.camera)
            out["frame"].append(int(frames[index]))
            out["box"].append(box)
            out["keypoints"].append(keypoints)
            out["score"].append(float(np.nanmean(keypoints[5:17, 2])))
            out["identity"].append(int(recovered["identity"][index]))
            out["t"].append(int(recovered["t"][index]))
    return {"camera": np.array(out["camera"]), "frame": np.array(out["frame"], np.int64),
            "box": np.array(out["box"], np.float32).reshape(-1, 4),
            "keypoints": np.array(out["keypoints"], np.float32).reshape(-1, 17, 3),
            "score": np.array(out["score"], np.float32), "identity": np.array(out["identity"], np.int64),
            "t": np.array(out["t"], np.int64)}


def describe(name: str, labels: list[np.ndarray], times: list[np.ndarray]) -> dict:
    spans = {}
    for label, t in zip(labels, times):
        ok = label >= 0
        for who in np.unique(label[ok]):
            mine = t[ok & (label == who)]
            low, high = spans.get(int(who), (10**9, -1))
            spans[int(who)] = (min(low, int(mine.min())), max(high, int(mine.max())))
    length = np.array([high - low + 1 for low, high in spans.values()])
    assigned = int(sum((label >= 0).sum() for label in labels))
    print(f"{name}: tracker ids {len(spans)}, assigned detections {assigned}, id life (frames) "
          f"median {int(np.median(length))}, under 30 frames: {int((length < 30).sum())}, "
          f"under 100: {int((length < 100).sum())}")
    return spans


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extras")
    p.add_argument("workspace")
    p.add_argument("out")
    p = sub.add_parser("compare")
    p.add_argument("workspace")
    p.add_argument("base")
    p.add_argument("augmented")
    p.add_argument("--golden")
    args = parser.parse_args()
    workspace = Workspace(args.workspace)
    scene = load_scene(workspace, joint_peak=RECIPE["geometry"]["joint_peak"], rays=False)
    if args.command == "extras":
        directory = Path(sorted(glob.glob(str(workspace.root / "rounds" / "*" / "tracks3d_recovered.npz")))[-1]).parent
        recovered = dict(np.load(directory / "tracks3d_recovered.npz"))
        found = extras(scene, recovered)
        np.savez_compressed(args.out, **found)
        views = int((recovered["pose_frame"] >= 0).sum())
        print(f"{directory}: {len(recovered['t'])} recovered positions, {views} pose views, {len(found['frame'])} are new "
              f"detections; score p10/p50/p90 {np.percentile(found['score'], [10, 50, 90]).round(2).tolist()}")
        return
    base, augmented = dict(np.load(args.base)), dict(np.load(args.augmented))
    names = [camera.camera for camera in scene.cameras]
    times = [camera.t for camera in scene.cameras]
    a = [base[f"raw_{name}"].astype(np.int64) for name in names]
    b = [augmented[f"raw_{name}"].astype(np.int64) for name in names]
    describe("base     ", a, times)
    describe("augmented", b, times)
    if "extra_raw" in augmented:
        print(f"extra detections: {len(augmented['extra_raw'])}, assigned to a track: {int((augmented['extra_raw'] >= 0).sum())}")
    # how the base ids map onto the augmented ids (by shared raw rows)
    votes = defaultdict(Counter)
    for x, y in zip(a, b):
        both = (x >= 0) & (y >= 0)
        for old, new in zip(x[both], y[both]):
            votes[int(old)][int(new)] += 1
    main_of = {old: counts.most_common(1)[0][0] for old, counts in votes.items() if sum(counts.values()) >= 20}
    joined = Counter(main_of.values())
    print(f"base ids with 20+ shared detections: {len(main_of)}; they map onto {len(joined)} augmented ids "
          f"({sum(1 for v in joined.values() if v > 1)} augmented ids join 2+ base ids)")
    changed = sum(int(((x >= 0) & (y < 0)).sum()) for x, y in zip(a, b)), sum(int(((x < 0) & (y >= 0)).sum()) for x, y in zip(a, b))
    print(f"raw detections assigned only in base: {changed[0]}, only in augmented: {changed[1]}")
    if not args.golden:
        return
    from data_engine.evaluation import load_reference

    reference = load_reference(scene, Path(SITE_DEFAULTS["repos"]["data_management"]) / "data" / "labels", args.golden)
    for name, labels in (("base", a), ("augmented", b)):
        people, purity_bad, total = defaultdict(Counter), 0, 0
        per_label = defaultdict(Counter)
        for camera in scene.cameras:
            person, label = reference.person[camera.index], labels[camera.index]
            both = (person >= 0) & (label >= 0)
            for who, lab in zip(person[both], label[both]):
                people[int(who)][int(lab)] += 1
                per_label[int(lab)][int(who)] += 1
        for lab, counts in per_label.items():
            total += sum(counts.values())
            purity_bad += sum(counts.values()) - counts.most_common(1)[0][1]
        fragments = [sum(1 for n in counts.values() if n >= 5) for counts in people.values()]
        print(f"{name}: golden people {len(people)}, tracker ids per person (5+ labelled detections) "
              f"mean {np.mean(fragments):.2f} total {sum(fragments)}, labelled detections {total}, "
              f"on a track that is mostly another person {purity_bad} ({purity_bad / max(total, 1):.2%})")


if __name__ == "__main__":
    main()
