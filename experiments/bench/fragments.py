"""Timeline of the engine's fragments per golden person: what is still split, and how.

Usage: python experiments/bench/fragments.py [variant]
"""
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench import DATASET, SCRATCH, VARIANTS, WORKSPACE, external_tracks, merged  # noqa: E402

from data_engine import evaluation
from data_engine.config import RECIPE, load_site
from data_engine.solve import recover
from data_engine.solve.pipeline import Constraints, solve
from data_engine.solve.scene import load_scene
from data_engine.stages.solve import extract_unknowns
from data_engine.workspace import Workspace

variant = sys.argv[1] if len(sys.argv) > 1 else "autofit+temporal10"
scene = load_scene(Workspace(WORKSPACE), joint_peak=0.3)
reference = evaluation.load_reference(scene, load_site().repo("data_management") / "data" / "labels", DATASET)
samples = pickle.loads(next(SCRATCH.glob("samples_*.pkl")).read_bytes())
constraints = Constraints(external=external_tracks(scene, VARIANTS[variant]["_external"])) \
    if VARIANTS[variant].get("_external") else None
solved = solve(scene, merged(RECIPE, VARIANTS[variant]), samples=samples, constraints=constraints, log=lambda *_: None)
scene = solved.scene
cams = recover.camera_counts(scene, solved.labels)

info = defaultdict(lambda: {"people": Counter(), "t": [], "cams": set(), "rows": 0})
for camera in scene.cameras:
    lab, person = solved.labels.label[camera.index], reference.person[camera.index]
    rows = camera.order[lab[camera.order] >= 0]
    for l, t, who in zip(lab[rows], camera.t[rows], person[rows]):
        entry = info[int(l)]
        entry["t"].append(int(t))
        entry["cams"].add(camera.index)
        entry["rows"] += 1
        if who >= 0:
            entry["people"][int(who)] += 1
by_person = defaultdict(list)
unlabelled = []
for label, entry in info.items():
    span = (min(entry["t"]), max(entry["t"]))
    record = (span[0], span[1], label, sorted(entry["cams"]), entry["rows"], int(solved.identity_of_label[label]),
              dict(entry["people"].most_common(2)))
    if entry["people"]:
        by_person[entry["people"].most_common(1)[0][0]].append(record)
    else:
        unlabelled.append(record)
print("fragments per golden person (start, end, component, cameras, detections, engine identity, people):")
for person in sorted(by_person):
    parts = sorted(by_person[person])
    multi = [p for p in parts if len(p[3]) >= 2]
    single = [p for p in parts if len(p[3]) < 2]
    print(f"person {person}: {len(multi)} multi-camera fragments, {len(single)} single-camera pieces "
          f"({sum(p[4] for p in single)} detections)")
    for part in multi:
        print("    3D ", part)
    for part in sorted(single, key=lambda p: -p[4])[:4]:
        print("    1cam", part)
print("components with no golden observations:", len(unlabelled), "detections", sum(u[4] for u in unlabelled),
      "multi-camera", sum(len(u[3]) >= 2 for u in unlabelled))
for part in sorted(unlabelled, key=lambda p: -p[4])[:8]:
    print("    ", part)
unknowns = extract_unknowns(solved, set())
print({k: len(v) for k, v in unknowns.items()})
print("links accepted:", len(solved.links.accepted), solved.links.accepted[:6])
for event in solved.log:
    print({k: (v if not isinstance(v, list) else len(v)) for k, v in event.items()})
