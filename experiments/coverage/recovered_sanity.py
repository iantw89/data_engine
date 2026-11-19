"""Reference-free checks of a workspace's recovered positions.

1. Gaps inside tracks before and after merging the recovered positions.
2. Continuity: distance from each recovered root to the same identity's measured root one frame
   earlier/later, compared with the same statistic between adjacent measured roots.
3. With --sheets DIR: crops of the poses a few recovered positions were triangulated from, next to
   the identity's nearest measured detection in the same camera.

    python experiments/coverage/recovered_sanity.py WORKSPACE [--sheets DIR --count 8]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

INCH = 0.0254
EDGES = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 300), (301, 10**9)]


def gaps(identity, t):
    lengths = []
    for who in np.unique(identity):
        times = np.unique(t[identity == who])
        lengths += [int(g) - 1 for g in np.diff(times) if g > 1]
    lengths = np.array(lengths)
    table = {f"{a}-{b if b < 10**9 else 'inf'}": (int(((lengths >= a) & (lengths <= b)).sum()),
                                                  int(lengths[(lengths >= a) & (lengths <= b)].sum())) for a, b in EDGES}
    return len(lengths), int(lengths.sum()), table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace")
    parser.add_argument("--sheets")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--pick", help="identity:t,identity:t,... render these recovered positions instead of random ones")
    args = parser.parse_args()
    from data_engine.workspace import Workspace

    workspace = Workspace(args.workspace)
    directory = Path(sorted(glob.glob(str(workspace.root / "rounds" / "*" / "tracks3d_recovered.npz")))[-1]).parent
    tracks, extra = dict(np.load(directory / "tracks3d.npz")), dict(np.load(directory / "tracks3d_recovered.npz"))
    ok = np.isfinite(tracks["root"]).all(axis=1)
    identity, t, root = tracks["identity"][ok].astype(np.int64), tracks["t"][ok].astype(np.int64), tracks["root"][ok]
    print(directory)
    before = gaps(identity, t)
    after = gaps(np.concatenate([identity, extra["identity"]]), np.concatenate([t, extra["t"]]))
    print(f"positions {len(t)} -> {len(t) + len(extra['t'])} (+{len(extra['t'])}, {len(extra['t']) / len(t):.1%})")
    print(f"gaps inside tracks {before[0]} -> {after[0]}; missing frames inside tracks {before[1]} -> {after[1]}")
    for key in before[2]:
        print(f"  gap length {key:>8}: gaps {before[2][key][0]:5d} -> {after[2][key][0]:5d}   missing frames "
              f"{before[2][key][1]:6d} -> {after[2][key][1]:6d}")
    # continuity
    lookup = {}
    order = np.lexsort((t, identity))
    for who, when, where in zip(identity[order], t[order], root[order]):
        lookup.setdefault((int(who), int(when)), where)
    measured_steps = []
    for (who, when), where in list(lookup.items())[::7]:
        nxt = lookup.get((who, when + 1))
        if nxt is not None:
            measured_steps.append(np.linalg.norm(nxt - where) * INCH)
    recovered_steps = []
    for who, when, where in zip(extra["identity"], extra["t"], extra["root"]):
        for other in (lookup.get((int(who), int(when) - 1)), lookup.get((int(who), int(when) + 1))):
            if other is not None:
                recovered_steps.append(np.linalg.norm(other - where) * INCH)
    for name, values in (("measured -> next measured", measured_steps), ("recovered -> adjacent measured", recovered_steps)):
        values = np.array(values)
        print(f"one-frame step, {name}: n={len(values)} p50={np.percentile(values, 50):.3f} m "
              f"p90={np.percentile(values, 90):.3f} p99={np.percentile(values, 99):.3f} max={values.max():.2f} "
              f"over 0.5 m: {(values > 0.5).mean():.3%}")
    if not args.sheets:
        return
    import cv2

    from data_engine.config import RECIPE
    from data_engine.geometry import SKELETON
    from data_engine.solve.scene import load_scene

    scene = load_scene(workspace, joint_peak=RECIPE["geometry"]["joint_peak"], rays=False)
    solution = dict(np.load(directory / "solution.npz"))
    out = Path(args.sheets)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    chosen = rng.choice(len(extra["t"]), size=min(args.count, len(extra["t"])), replace=False)
    if args.pick:
        wanted = [tuple(int(v) for v in item.split(":")) for item in args.pick.split(",")]
        chosen = [int(np.flatnonzero((extra["identity"] == who) & (extra["t"] == when))[0]) for who, when in wanted
                  if ((extra["identity"] == who) & (extra["t"] == when)).any()]
    for number, row in enumerate(chosen):
        who, when = int(extra["identity"][row]), int(extra["t"][row])
        cells = []
        for camera in scene.cameras:
            frame = int(extra["pose_frame"][row, camera.index])
            if frame < 0:
                continue
            image = workspace.frames.read(camera.camera, frame).copy()
            box, points = extra["pose_box"][row, camera.index], extra["pose_keypoints"][row, camera.index]
            for a, b in SKELETON:
                if points[a, 2] >= 0.3 and points[b, 2] >= 0.3:
                    cv2.line(image, tuple(int(v) for v in points[a, :2]), tuple(int(v) for v in points[b, :2]), (0, 255, 255), 3)
            w, h = box[2] - box[0], box[3] - box[1]
            x0, y0 = int(max(0, box[0] - 0.4 * w)), int(max(0, box[1] - 0.2 * h))
            x1, y1 = int(min(image.shape[1], box[2] + 0.4 * w)), int(min(image.shape[0], box[3] + 0.2 * h))
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * 260 / crop.shape[0])), 260))
            cv2.putText(crop, f"recovered cam{camera.index}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            # nearest measured detection of the same identity in this camera
            mine = np.flatnonzero((solution[f"identity_{camera.camera}"] == who) & (camera.t >= 0))
            cells.append(crop)
            if len(mine):
                nearest = mine[np.argmin(np.abs(camera.t[mine] - when))]
                from data_engine.solve import montage
                ref = montage.crop(scene, camera.index, int(nearest), height=260)
                ref = cv2.resize(ref, (max(1, int(ref.shape[1] * 260 / ref.shape[0])), 260))
                cv2.putText(ref, f"measured dt={int(camera.t[nearest]) - when}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cells.append(ref)
        if cells:
            sheet = np.hstack([np.pad(c, ((0, 0), (0, 6), (0, 0))) for c in cells])[:, :1500]
            name = out / f"recovered_{number:02d}_id{who}_t{when}_tier{extra['tier'][row]}.jpg"
            cv2.imwrite(str(name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
            print(name)


if __name__ == "__main__":
    main()
