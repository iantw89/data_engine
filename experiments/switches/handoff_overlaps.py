"""Hand-offs with overlap: a track ends next to another that was born just before at the same spot.

Why were they not merged by the solver's hand-off rule? For every such pair of identities in a saved
solution: overlap length, distance at the common instants, distance between the earlier track's last
and the later track's first position (what the rule compares), camera clashes, 2D tracklets that
continue from the ending track into the surviving one.

    python experiments/switches/handoff_overlaps.py WORKSPACE [GOLDEN_DATASET]
"""
from __future__ import annotations

import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from data_engine.config import RECIPE  # noqa: E402
from data_engine.solve import swap_check  # noqa: E402
from data_engine.solve.scene import load_scene  # noqa: E402
from data_engine.workspace import Workspace  # noqa: E402

INCH = 0.0254


def main() -> None:
    workspace = Workspace(sys.argv[1])
    scene = load_scene(workspace, joint_peak=RECIPE["geometry"]["joint_peak"], rays=False)
    directory = Path(sorted(glob.glob(str(workspace.root / "rounds" / "*" / "tracks3d.npz")))[-1]).parent
    tracks, solution = dict(np.load(directory / "tracks3d.npz")), dict(np.load(directory / "solution.npz"))
    row_identity = [solution[f"identity_{camera.camera}"].astype(np.int64) for camera in scene.cameras]
    recipe = RECIPE["swap_check"]
    index = swap_check.Index.build(scene, row_identity, tracks["identity"].astype(np.int64), tracks["t"], tracks["root"],
                                   tracks["xyz"], window=recipe["window_frames"])
    person_of = {}
    if len(sys.argv) > 2:
        from data_engine.config import SITE_DEFAULTS
        from data_engine.evaluation import load_reference
        reference = load_reference(scene, Path(SITE_DEFAULTS["repos"]["data_management"]) / "data" / "labels", sys.argv[2])
        votes = defaultdict(Counter)
        for camera in scene.cameras:
            both = (row_identity[camera.index] >= 0) & (reference.person[camera.index] >= 0)
            for who, golden in zip(row_identity[camera.index][both], reference.person[camera.index][both]):
                votes[int(who)][int(golden)] += 1
        person_of = {who: counts.most_common(1)[0][0] for who, counts in votes.items()}
    # every identity that ends inside the recording, and the nearest other identity measured at that moment
    order = np.argsort(index.tr_t, kind="stable")
    ident, t, root = index.tr_ident[order], index.tr_t[order], index.tr_root[order]
    first_at = np.searchsorted(t, np.arange(scene.T + 2))
    rows = []
    for who, (s, e) in index.tr_span.items():
        last = int(index.tr_t[e - 1])
        if e - s < 10 or last >= scene.T - 20:
            continue
        a, b = first_at[max(0, last - 3)], first_at[min(scene.T, last + 4)]
        others = ident[a:b] != who
        if not others.any():
            continue
        d = np.linalg.norm(root[a:b][others] - index.tr_root[e - 1], axis=1) * INCH
        nearest = int(np.argmin(d))
        survivor = int(ident[a:b][others][nearest])
        if d[nearest] >= recipe["ending_radius_m"]:
            continue
        history, _, _ = index.track(survivor, last - index.window, last - 1)
        if len(history) >= recipe["established_frames"]:
            continue  # an established neighbour: the swap check's business
        ss, se = index.tr_span[survivor]
        born = int(index.tr_t[ss])
        common, ia, ib = np.intersect1d(index.tr_t[s:e], index.tr_t[ss:se], return_indices=True)
        apart = np.linalg.norm(index.tr_root[s:e][ia] - index.tr_root[ss:se][ib], axis=1) * INCH if len(common) else np.zeros(0)
        rule_distance = float(np.linalg.norm(index.tr_root[ss] - index.tr_root[e - 1]) * INCH)
        clash = 0
        for camera in scene.cameras:
            mine = np.flatnonzero((row_identity[camera.index] == who) & (camera.t >= born) & (camera.t <= last))
            theirs = np.flatnonzero((row_identity[camera.index] == survivor) & (camera.t >= born) & (camera.t <= last))
            clash += len(np.intersect1d(camera.t[mine], camera.t[theirs]))
        tracklet = swap_check.cue_tracklet(index, survivor, who, last, last)
        rows.append({"ending": who, "survivor": survivor, "t": last, "overlap_frames": last - born + 1, "common": int(len(common)),
                     "median_apart_m": round(float(np.median(apart)), 2) if len(apart) else None,
                     "max_apart_m": round(float(apart.max()), 2) if len(apart) else None,
                     "rule_distance_m": round(rule_distance, 2), "camera_clash_frames": clash,
                     "tracklets_continuing": tracklet["contradict"], "tracklets_staying": tracklet["agree"],
                     "same_golden_person": (person_of.get(who) == person_of.get(survivor)) if who in person_of and survivor in person_of else None})
    print(directory, len(rows), "hand-offs with overlap")
    for row in sorted(rows, key=lambda r: r["t"]):
        print(row)
    print("overlap > 15 frames:", sum(r["overlap_frames"] > 15 for r in rows), "| rule distance > 0.5 m:",
          sum(r["rule_distance_m"] > 0.508 for r in rows), "| camera clash:", sum(r["camera_clash_frames"] > 0 for r in rows),
          "| co-located throughout (max apart <= 0.3 m):", sum((r["max_apart_m"] or 9) <= 0.3 for r in rows),
          "| with a continuing 2D tracklet:", sum(r["tracklets_continuing"] > 0 for r in rows))


if __name__ == "__main__":
    main()
