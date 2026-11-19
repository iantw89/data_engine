"""Where do the engine's wrong assignments on the golden set come from?

Breaks impure observations down by (a) how the detection got its label and (b) whether its
2D tracklet piece was already impure (a tracker identity switch the solver inherited).
"""
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from data_engine import evaluation
from data_engine.config import RECIPE, load_site
from data_engine.solve import recover
from data_engine.solve.pipeline import solve
from data_engine.solve.scene import load_scene
from data_engine.workspace import Workspace

sys.path.insert(0, str(Path(__file__).parent))
from bench import DATASET, SCRATCH, VARIANTS, WORKSPACE, merged  # noqa: E402

variant = sys.argv[1] if len(sys.argv) > 1 else "baseline"
workspace = Workspace(WORKSPACE)
scene = load_scene(workspace, joint_peak=0.3)
reference = evaluation.load_reference(scene, load_site().repo("data_management") / "data" / "labels", DATASET)

# --- 2D tracklet purity (before any association)
impure_tracklets = total_tracklets = impure_rows = labelled_rows = 0
worst = []
for camera in scene.cameras:
    person = reference.person[camera.index]
    rows = np.flatnonzero((person >= 0) & (camera.tracklet >= 0))
    by_tracklet = defaultdict(Counter)
    for tid, who in zip(camera.tracklet[rows], person[rows]):
        by_tracklet[int(tid)][int(who)] += 1
    for tid, counts in by_tracklet.items():
        total_tracklets += 1
        major = counts.most_common(1)[0][1]
        minor = sum(counts.values()) - major
        labelled_rows += sum(counts.values())
        if minor:
            impure_tracklets += 1
            impure_rows += minor
            worst.append((minor, camera.camera, tid, dict(counts.most_common(3))))
print(f"2D tracklets with reference rows: {total_tracklets}; impure: {impure_tracklets}; "
      f"minority observations inside tracklets: {impure_rows} of {labelled_rows}")
for item in sorted(worst, reverse=True)[:10]:
    print("   ", item)

recipe = merged(RECIPE, VARIANTS[variant])
samples = None
for path in SCRATCH.glob("samples_*.pkl"):
    samples = pickle.loads(path.read_bytes())
    break
solved = solve(scene, recipe, samples=samples, log=lambda *_: None)
cams = recover.camera_counts(scene, solved.labels)
multi = {l for l, c in cams.items() if len(c) >= 2}

# majority person per component
confusion = defaultdict(Counter)
for camera in scene.cameras:
    person, lab = reference.person[camera.index], solved.labels.label[camera.index]
    rows = np.flatnonzero((person >= 0) & np.isin(lab, list(multi)))
    for l, who in zip(lab[rows], person[rows]):
        confusion[int(l)][int(who)] += 1
major = {l: c.most_common(1)[0][0] for l, c in confusion.items()}

by_source = Counter()
total_by_source = Counter()
from_impure_atom = 0
atom_major = {}
for camera in scene.cameras:
    person, lab, atom = reference.person[camera.index], solved.labels.label[camera.index], solved.labels.atom[camera.index]
    counts = defaultdict(Counter)
    rows = np.flatnonzero((person >= 0) & (atom >= 0))
    for a, who in zip(atom[rows], person[rows]):
        counts[int(a)][int(who)] += 1
    atom_major[camera.index] = {a: (c.most_common(1)[0][0], sum(c.values()) - c.most_common(1)[0][1]) for a, c in counts.items()}
names = {v: k for k, v in recover.Labels.SOURCE.items()}
wrong_atoms = Counter()
for camera in scene.cameras:
    person, lab = reference.person[camera.index], solved.labels.label[camera.index]
    atom, source = solved.labels.atom[camera.index], solved.labels.source[camera.index]
    rows = np.flatnonzero((person >= 0) & np.isin(lab, list(multi)))
    for r in rows:
        total_by_source[names[int(source[r])]] += 1
        if person[r] != major[int(lab[r])]:
            by_source[names[int(source[r])]] += 1
            a = int(atom[r])
            info = atom_major[camera.index].get(a)
            if info and info[1] > 0:
                from_impure_atom += 1
            if a >= 0:
                wrong_atoms[(camera.camera, a)] += 1
print("covered observations by source:", dict(total_by_source))
print("impure observations by source:", dict(by_source), "total", sum(by_source.values()))
print("of which inside an internally impure tracklet piece:", from_impure_atom)
print("wrong observations concentrated in pieces (top 12):", wrong_atoms.most_common(12))
whole_wrong = sum(n for (cam, a), n in wrong_atoms.items()
                  if atom_major[next(c.index for c in scene.cameras if c.camera == cam)].get(a, (None, 1))[1] == 0)
print("wrong observations in pieces that are internally pure (the whole piece is on the wrong person):", whole_wrong)
