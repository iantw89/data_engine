"""What the coverage recovery says about identity and association - evidence the solve does not use yet.

    python experiments/coverage/recovery_as_identity_evidence.py WORKSPACE [--golden DATASET]

A recovered pose is read at the place where identity A must be. When a raw detection with the same pose
(median joint distance <= 5% of the box height) is already there, that detection is A's. Two cases carry
information for the solve:

  the detection belongs to identity B     A and B are one person, or one of the two is wrong there
  the detection belongs to no identity    the tracklet piece it is part of is a candidate for A

For every pair (A, B) the script counts the instants and cameras of that evidence and looks at the measured
3D positions of both: never measured at the same instant (a link candidate), measured at the same instant
at one place (a duplicate), or measured apart at the same instant (cannot be one person). With --golden it
says whether A and B are the same golden person and whether a piece belongs to A's golden person.
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

INCH = 0.0254


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--agree", type=float, default=0.05)
    parser.add_argument("--golden")
    parser.add_argument("--min-instants", type=int, default=5, help="instants of evidence a pair needs to be listed as strong")
    args = parser.parse_args()
    workspace = Workspace(args.workspace)
    directory = Path(sorted(glob.glob(str(workspace.root / "rounds" / "*" / "tracks3d_recovered.npz")))[-1]).parent
    recovered = dict(np.load(directory / "tracks3d_recovered.npz"))
    solution = dict(np.load(directory / "solution.npz"))
    tracks = dict(np.load(directory / "tracks3d.npz"))
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

    pair_instants = defaultdict(set)   # (A, B) -> instants
    pair_cameras = defaultdict(set)
    piece_claims = defaultdict(Counter)  # (camera, piece) -> identity -> detections claimed
    piece_rows = {}
    for camera in scene.cameras:
        identity = solution[f"identity_{camera.camera}"].astype(np.int64)
        atom = solution[f"atom_{camera.camera}"].astype(np.int64)
        frames = recovered["pose_frame"][:, camera.index].astype(np.int64)
        order = np.argsort(camera.detections.frame_of_row, kind="stable")
        frame_sorted = camera.detections.frame_of_row[order]
        for index in np.flatnonzero(frames >= 0):
            lo, hi = np.searchsorted(frame_sorted, [frames[index], frames[index] + 1])
            rows = order[lo:hi]
            if not len(rows):
                continue
            box = recovered["pose_box"][index, camera.index].astype(np.float64)
            overlaps = np.array([coverage.box_iou(box, other) for other in camera.detections.boxes[rows, :4].astype(np.float64)])
            best = int(np.argmax(overlaps))
            if overlaps[best] < args.overlap:
                continue
            row = int(rows[best])
            who, owner = int(recovered["identity"][index]), int(identity[row])
            if owner == who:
                continue
            mine, theirs = recovered["pose_keypoints"][index, camera.index], camera.detections.keypoints[row]
            both = (mine[5:17, 2] >= 0.3) & (theirs[5:17, 2] >= 0.3)
            if both.sum() < 4:
                continue
            height = max(box[3] - box[1], 1.0)
            if np.median(np.linalg.norm(mine[5:17, :2][both] - theirs[5:17, :2][both], axis=1)) / height > args.agree:
                continue
            if owner >= 0:
                pair = (min(who, owner), max(who, owner))
                pair_instants[pair].add(int(recovered["t"][index]))
                pair_cameras[pair].add(camera.camera)
            elif atom[row] >= 0:
                key = (camera.camera, int(atom[row]))
                piece_claims[key][who] += 1
                if key not in piece_rows:
                    piece_rows[key] = np.flatnonzero(atom == atom[row])

    # ---- pairs of identities
    times = defaultdict(dict)
    for who, t, root in zip(tracks["identity"], tracks["t"], tracks["root"]):
        if who >= 0 and np.isfinite(root).all():
            times[int(who)][int(t)] = root
    print(f"\nidentity pairs with a recovered pose of one on a same-pose detection of the other: {len(pair_instants)} "
          f"({sum(len(v) for v in pair_instants.values())} instants)")
    classes, golden = Counter(), defaultdict(Counter)
    strong = []
    for pair, instants in pair_instants.items():
        a, b = pair
        common = set(times[a]) & set(times[b])
        apart = sum(1 for t in common if np.linalg.norm(times[a][t] - times[b][t]) * INCH > 1.0)
        kind = ("never measured at the same instant" if not common else
                "measured at the same instants, apart (cannot be one person)" if apart > 0.2 * len(common) and apart >= 3 else
                "measured at the same instants, at one place (duplicate)")
        strength = "strong" if len(instants) >= args.min_instants else "weak"
        classes[(kind, strength)] += 1
        if person is not None:
            same = person_of.get(a), person_of.get(b)
            golden[(kind, strength)]["not labelled" if None in same else "same golden person" if same[0] == same[1]
                                     else "DIFFERENT golden people"] += 1
        if strength == "strong":
            strong.append((len(instants), pair, kind, len(pair_cameras[pair]), len(common), apart))
    for key, count in sorted(classes.items()):
        print(f"  {key[0]:62s} {key[1]:6s} {count:5d}" + (f"   {dict(golden[key])}" if person is not None else ""))
    identities = int(len(np.unique(tracks["identity"][tracks["identity"] >= 0])))
    candidates = [pair for n, pair, kind, *_ in strong if kind.startswith("never")]
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in candidates:
        parent[find(a)] = find(b)
    merged = len({x for pair in candidates for x in pair}) - len({find(x) for pair in candidates for x in pair})
    print(f"  identities {identities}; strong pairs never measured together: {len(candidates)} -> at most {merged} fewer identities")
    for n, pair, kind, cameras, common, apart in sorted(strong, reverse=True)[:12]:
        extra = f" golden {person_of.get(pair[0])}/{person_of.get(pair[1])}" if person is not None else ""
        print(f"    {pair}: {n} instants in {cameras} cameras; {kind}; common instants {common}, apart {apart}{extra}")

    # ---- unassigned pieces
    print(f"\nunassigned tracklet pieces with a same-pose recovered pose on them: {len(piece_claims)}")
    single = {key: claims for key, claims in piece_claims.items() if len(claims) == 1}
    detections = sum(len(piece_rows[key]) for key in single)
    touched = sum(sum(claims.values()) for claims in single.values())
    print(f"  claimed by one identity: {len(single)} pieces, {detections} detections in them, {touched} under a recovered pose")
    print(f"  claimed by several identities: {len(piece_claims) - len(single)}")
    sizes = np.array([len(piece_rows[key]) for key in single]) if single else np.zeros(0)
    if len(sizes):
        print(f"  piece length (detections): p50 {np.percentile(sizes, 50):.0f}, p90 {np.percentile(sizes, 90):.0f}, max {sizes.max():.0f}")
    if person is not None:
        verdict, weight = Counter(), Counter()
        for (name, _), claims in single.items():
            camera = next(c for c in scene.cameras if c.camera == name)
            rows = piece_rows[(name, _)]
            labelled = person[camera.index][rows]
            labelled = labelled[labelled >= 0]
            expected = person_of.get(next(iter(claims)))
            if not len(labelled) or expected is None:
                kind = "not labelled"
            else:
                kind = "the identity's golden person" if Counter(labelled.tolist()).most_common(1)[0][0] == expected \
                    else "ANOTHER golden person"
            verdict[kind] += 1
            weight[kind] += len(rows)
        print(f"  golden person of the piece: {dict(verdict)}; detections: {dict(weight)}")


if __name__ == "__main__":
    main()
