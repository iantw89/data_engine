"""Recovered poses that coincide with a detection the detector already made: whose is that detection?

    python experiments/coverage/coinciding_detections.py WORKSPACE [--overlap 0.5] [--golden DATASET]

For every pose the recover stage read (one per camera view of a recovered position), find the raw
detection in the same frame whose box overlaps it most. When the overlap is at least ``--overlap`` the
detection is one of:
  same      already assigned to this identity (it was the only view, so no 3D position existed)
  unassigned  belongs to no identity: the recovery is evidence for whose it is
  other     assigned to another identity: an identity question (same person twice, or a wrong recovery)
Also reports how well the detection's own keypoints agree with the recovered pose and, with --golden,
whether the coinciding detection is labelled as the identity's golden person.
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from data_engine.config import RECIPE  # noqa: E402
from data_engine.solve import coverage  # noqa: E402
from data_engine.solve.scene import load_scene  # noqa: E402
from data_engine.workspace import Workspace  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--golden")
    parser.add_argument("--agree", type=float, default=0.05, help="keypoint distance / box height counted as the same pose")
    args = parser.parse_args()
    workspace = Workspace(args.workspace)
    directory = Path(sorted(glob.glob(str(workspace.root / "rounds" / "*" / "tracks3d_recovered.npz")))[-1]).parent
    recovered = dict(np.load(directory / "tracks3d_recovered.npz"))
    solution = dict(np.load(directory / "solution.npz"))
    scene = load_scene(workspace, joint_peak=RECIPE["geometry"]["joint_peak"], rays=False)
    print(directory)
    person, person_of = None, {}
    if args.golden:
        from data_engine.config import SITE_DEFAULTS
        from data_engine.evaluation import load_reference

        reference = load_reference(scene, Path(SITE_DEFAULTS["repos"]["data_management"]) / "data" / "labels", args.golden)
        person = reference.person
        votes = defaultdict(Counter)
        for camera in scene.cameras:
            identity = solution[f"identity_{camera.camera}"]
            both = (identity >= 0) & (person[camera.index] >= 0)
            for who, labelled in zip(identity[both], person[camera.index][both]):
                votes[int(who)][int(labelled)] += 1
        person_of = {who: counts.most_common(1)[0][0] for who, counts in votes.items()}
    truth = defaultdict(Counter)
    kinds, agreement = Counter(), {"same": [], "unassigned": [], "other": []}
    views = 0
    for camera in scene.cameras:
        identity = solution[f"identity_{camera.camera}"].astype(np.int64)
        frames = recovered["pose_frame"][:, camera.index].astype(np.int64)
        order = np.argsort(camera.detections.frame_of_row, kind="stable")
        frame_sorted = camera.detections.frame_of_row[order]
        for index in np.flatnonzero(frames >= 0):
            views += 1
            lo, hi = np.searchsorted(frame_sorted, [frames[index], frames[index] + 1])
            rows = order[lo:hi]
            if not len(rows):
                kinds["no detection in the frame"] += 1
                continue
            box = recovered["pose_box"][index, camera.index].astype(np.float64)
            overlaps = np.array([coverage.box_iou(box, other) for other in camera.detections.boxes[rows, :4].astype(np.float64)])
            best = int(np.argmax(overlaps))
            if overlaps[best] < args.overlap:
                kinds["no coinciding detection"] += 1
                continue
            owner = int(identity[rows[best]])
            kind = "same" if owner == int(recovered["identity"][index]) else "unassigned" if owner < 0 else "other"
            kinds[kind] += 1
            mine, theirs = recovered["pose_keypoints"][index, camera.index], camera.detections.keypoints[rows[best]]
            both = (mine[5:17, 2] >= 0.3) & (theirs[5:17, 2] >= 0.3)
            distance = None
            if both.sum() >= 4:
                height = max(box[3] - box[1], 1.0)
                distance = float(np.median(np.linalg.norm(mine[5:17, :2][both] - theirs[5:17, :2][both], axis=1)) / height)
                agreement[kind].append(distance)
            if person is not None:
                labelled, expected = int(person[camera.index][rows[best]]), person_of.get(int(recovered["identity"][index]))
                pose = "same pose" if distance is not None and distance <= args.agree else "pose differs or too few joints"
                truth[(kind, pose)]["not labelled" if labelled < 0 or expected is None else
                                    "the identity's golden person" if labelled == expected else "ANOTHER golden person"] += 1
    print(f"recovered positions {len(recovered['t'])}, pose views {views}")
    for kind, count in kinds.most_common():
        print(f"  {kind:28s} {count:7d}  {count / views:.1%}")
    for kind, values in agreement.items():
        if values:
            values = np.array(values)
            print(f"  keypoint distance / box height, {kind}: n={len(values)} p50={np.percentile(values, 50):.3f} "
                  f"p90={np.percentile(values, 90):.3f} within 0.05: {(values <= 0.05).mean():.1%}")
    for key in sorted(truth):
        print(f"  golden label of the coinciding detection, {key[0]}, {key[1]}: {dict(truth[key])}")


if __name__ == "__main__":
    main()
