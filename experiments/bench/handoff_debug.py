"""Why are back-to-back fragments of one golden person not merged? (diagnostic)"""
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
from data_engine.solve.identity import summarise
from data_engine.solve.pipeline import Constraints, solve
from data_engine.solve.scene import load_scene
from data_engine.workspace import Workspace

variant = sys.argv[1] if len(sys.argv) > 1 else "autofit+handoff+temporal10"
scene = load_scene(Workspace(WORKSPACE), joint_peak=0.3)
reference = evaluation.load_reference(scene, load_site().repo("data_management") / "data" / "labels", DATASET)
samples = pickle.loads(next(SCRATCH.glob("samples_*.pkl")).read_bytes())
constraints = Constraints(external=external_tracks(scene, VARIANTS[variant]["_external"]))
solved = solve(scene, merged(RECIPE, VARIANTS[variant]), samples=samples, constraints=constraints, log=lambda *_: None)
scene, labels, frames = solved.scene, solved.labels, solved.frames
people = defaultdict(Counter)
for camera in scene.cameras:
    lab, person = labels.label[camera.index], reference.person[camera.index]
    rows = np.flatnonzero((lab >= 0) & (person >= 0))
    for l, who in zip(lab[rows], person[rows]):
        people[int(l)][int(who)] += 1
summaries = {s.label: s for s in summarise(frames, labels.next_label()) if s is not None and s.n_measured >= 3}
occupied = recover.occupancy(scene, labels)
by_person = defaultdict(list)
for label, summary in summaries.items():
    if people[label] and len(summary.cameras) >= 2:
        by_person[people[label].most_common(1)[0][0]].append(summary)
reasons = Counter()
for person, parts in sorted(by_person.items()):
    parts.sort(key=lambda s: s.first_root_t)
    for earlier, later in zip(parts, parts[1:]):
        gap = later.first_root_t - earlier.last_root_t
        distance = float(np.linalg.norm(later.first_root - earlier.last_root))
        clash_cameras = [c.camera for c in scene.cameras
                         if (earlier.label, c.index) in occupied and (later.label, c.index) in occupied
                         and len(np.intersect1d(occupied[(earlier.label, c.index)], occupied[(later.label, c.index)]))]
        nearby_starts = [s.label for s in summaries.values() if s.label not in (earlier.label, later.label)
                         and -15 <= s.first_root_t - earlier.last_root_t <= 10
                         and np.linalg.norm(s.first_root - earlier.last_root) <= 40]
        nearby_ends = [s.label for s in summaries.values() if s.label not in (earlier.label, later.label)
                       and -15 <= later.first_root_t - s.last_root_t <= 10
                       and np.linalg.norm(later.first_root - s.last_root) <= 40]
        contained = later.last_root_t <= earlier.last_root_t
        why = ("contained" if contained else "gap>10" if gap > 10 else "overlap>15" if gap < -15 else
               "same-camera clash" if clash_cameras else "far" if distance > 20 else
               "not unique" if nearby_starts or nearby_ends else "should merge?")
        reasons[why] += 1
        print(f"person {person}: {earlier.label}[{earlier.first_root_t}-{earlier.last_root_t}] -> "
              f"{later.label}[{later.first_root_t}-{later.last_root_t}] gap {gap} dist {distance:.0f} in "
              f"clash {clash_cameras} rivals {nearby_starts + nearby_ends} => {why}")
print(dict(reasons))
