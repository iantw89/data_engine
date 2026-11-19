"""What is still missing after recovery, and what would a human answer (or another task) change?

    python experiments/coverage/rejected_impact.py WORKSPACE [GOLDEN_DATASET]

Reads the latest round's tracks3d.npz, tracks3d_recovered.npz and recover_rejected.npz.
With GOLDEN_DATASET: also counts the missing instants at which the same golden person is measured
under ANOTHER engine identity (those disappear when the two identities are linked).
"""
from __future__ import annotations

import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# A pose with implausible bones or too few joints is a poor measurement whoever it shows. The other
# conditions exist because nobody has shown that the crops show the tracked person: with two cameras,
# agreement of two views is weak proof, so the gate demands a tight residual, confident joints and a
# bracketed gap. A person saying "yes, both crops show this person" replaces those conditions; the
# position is then an ordinary two-view triangulation at the engine's normal gates.
QUALITY = {"body joints", "bone lengths", "cameras"}
PROOF = {"residual", "image confidence", "not bracketed by nearby poses", "distance from the hint"}
OTHER = "another identity is measured there"


def runs(identity: np.ndarray, t: np.ndarray) -> int:
    """Number of runs of consecutive instants per identity (one question per run)."""
    order = np.lexsort((t, identity))
    identity, t = identity[order], t[order]
    if not len(t):
        return 0
    return int(1 + ((identity[1:] != identity[:-1]) | (t[1:] - t[:-1] > 1)).sum())


def main() -> None:
    from data_engine.workspace import Workspace

    workspace = Workspace(sys.argv[1])
    directory = Path(sorted(glob.glob(str(workspace.root / "rounds" / "*" / "recover_rejected.npz")))[-1]).parent
    tracks = dict(np.load(directory / "tracks3d.npz"))
    extra = dict(np.load(directory / "tracks3d_recovered.npz"))
    rejected = dict(np.load(directory / "recover_rejected.npz"))
    ok = np.isfinite(tracks["root"]).all(axis=1) & (tracks["identity"] >= 0)
    identity = np.concatenate([tracks["identity"][ok], extra["identity"]]).astype(np.int64)
    t = np.concatenate([tracks["t"][ok], extra["t"]]).astype(np.int64)
    span = {}
    have = defaultdict(set)
    for who, when in zip(identity, t):
        lo, hi = span.get(int(who), (10**9, -1))
        span[int(who)] = (min(lo, int(when)), max(hi, int(when)))
        have[int(who)].add(int(when))
    inside_total = sum(hi - lo + 1 - len(have[who]) for who, (lo, hi) in span.items())
    print(directory)
    print(f"positions: measured {int(ok.sum())} + recovered {len(extra['t'])}; missing inside tracks now: {inside_total}")

    r_identity, r_t = rejected["identity"].astype(np.int64), rejected["t"].astype(np.int64)
    inside = np.array([span.get(int(who), (0, -1))[0] <= when <= span.get(int(who), (0, -1))[1] for who, when in zip(r_identity, r_t)], bool)
    reason = np.array([str(rejected["reasons"][code]) for code in rejected["reason"]])
    print(f"\nlooked at and still missing: {len(r_t)} instants ({int(inside.sum())} inside tracks, {int((~inside).sum())} beyond an end)")
    print(f"missing inside tracks that no pass could look at (no known pose within reach): {inside_total - int(inside.sum())}")
    for name in sorted(set(reason)):
        mine = reason == name
        print(f"  {name:38s} {int(mine.sum()):6d} instants ({int((mine & inside).sum())} inside), "
              f"{runs(r_identity[mine], r_t[mine])} runs")

    gate = reason == "rejected by the acceptance gate"
    failed = [set(filter(None, str(text).split("; "))) for text in rejected["failed"]]
    classes = Counter()
    members = defaultdict(list)
    for index in np.flatnonzero(gate):
        if OTHER in failed[index]:
            label = "another identity is already measured at that place (an identity/association question, not a coverage one)"
        elif failed[index] & QUALITY:
            label = "poor measurement (implausible bones / too few joints): needs pose correction, not a yes/no"
        else:
            label = "measurement fine, proof that it is this person missing: a yes/no would add the position"
        classes[label] += 1
        members[label].append(index)
    print(f"\ngate-rejected: {int(gate.sum())} instants, {runs(r_identity[gate], r_t[gate])} runs")
    for label, count in classes.most_common():
        index = np.array(members[label])
        tried = Counter(str(v) for v in rejected["tier_tried"][index])
        print(f"  {count:6d}  {label}  [{runs(r_identity[index], r_t[index])} runs; inside tracks {int(inside[index].sum())}; "
              f"cameras: {dict(tried)}]")
    conditions = Counter(condition for index in np.flatnonzero(gate) for condition in failed[index])
    print("  conditions failed (an instant can fail several):", dict(conditions.most_common()))
    names = [str(n) for n in rejected["feature_names"]]
    other = rejected["features"][:, names.index("other_identity_in")]
    print(f"  gate-rejected with another identity measured within 24 in: {int((gate & (other < 24)).sum())}; within 40 in: "
          f"{int((gate & (other < 40)).sum())}")

    if len(sys.argv) > 2:
        from data_engine.config import RECIPE, SITE_DEFAULTS
        from data_engine.evaluation import load_reference
        from data_engine.solve.scene import load_scene

        scene = load_scene(workspace, joint_peak=RECIPE["geometry"]["joint_peak"], rays=False)
        solution = dict(np.load(directory / "solution.npz"))
        reference = load_reference(scene, Path(SITE_DEFAULTS["repos"]["data_management"]) / "data" / "labels", sys.argv[2])
        votes = defaultdict(Counter)
        for camera in scene.cameras:
            ident, person = solution[f"identity_{camera.camera}"], reference.person[camera.index]
            both = (ident >= 0) & (person >= 0)
            for who, golden in zip(ident[both], person[both]):
                votes[int(who)][int(golden)] += 1
        person_of = {who: counts.most_common(1)[0][0] for who, counts in votes.items()}
        present = defaultdict(set)  # golden person -> instants with a position under any identity
        for who, when in zip(identity, t):
            if int(who) in person_of:
                present[person_of[int(who)]].add(int(when))
        total = same_person_elsewhere = 0
        for who, (lo, hi) in span.items():
            if who not in person_of:
                continue
            for when in range(lo, hi + 1):
                if when not in have[who]:
                    total += 1
                    same_person_elsewhere += when in present[person_of[who]]
        print(f"\ngolden: of {total} missing instants inside tracks of identities with a golden person, "
              f"{same_person_elsewhere} have that same person measured under another engine identity at that instant "
              "(linking the identities removes them)")
        gate_same = sum(1 for index in np.flatnonzero(gate) if int(r_identity[index]) in person_of
                        and int(r_t[index]) in present[person_of[int(r_identity[index])]])
        print(f"golden: of the {int(gate.sum())} gate-rejected instants, {gate_same} have the same person measured under another identity")


if __name__ == "__main__":
    main()
