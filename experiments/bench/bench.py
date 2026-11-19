"""Golden-set bench: score solver variants against the golden identities.

Usage (engine venv, PYTHONPATH=src):
    python experiments/bench/bench.py [variant ...]         # run named variants (default: all)
    python experiments/bench/bench.py --list

For each variant: how much the machine resolves (coverage), how wrong it is (impure
observations, mixed labels, wrong cross-camera pairs) and how many questions are left for
a human (open unknowns by kind). A lever is worth adopting only if it removes questions
without adding wrong merges.

The golden labels are used for scoring only; nothing from them enters the solver.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from data_engine import evaluation
from data_engine.config import RECIPE, load_site
from data_engine.solve import recover
from data_engine.solve.evidence import pair_samples
from data_engine.solve.pipeline import Constraints, solve
from data_engine.solve.scene import load_scene
from data_engine.stages.solve import extract_unknowns
from data_engine.workspace import Workspace

WORKSPACE = "golden_20250505071549_20250505080000@first600s"
DATASET = "golden_20250505071549_20250505080000"
SCRATCH = Path(r"C:\data_engine_data\_scratch\bench")

VARIANTS: dict[str, dict] = {
    "baseline": {},
    "bones": {"geometry": {"bone_veto": {"min_bones": 4, "max_bad_share": 0.34}}},
    "temporal10": {"refine": {"temporal_reach_frames": 10}},
    "temporal30": {"refine": {"temporal_reach_frames": 30}},
    "bones+temporal10": {"geometry": {"bone_veto": {"min_bones": 4, "max_bad_share": 0.34}},
                         "refine": {"temporal_reach_frames": 10}},
    "bones+temporal30": {"geometry": {"bone_veto": {"min_bones": 4, "max_bad_share": 0.34}},
                         "refine": {"temporal_reach_frames": 30}},
    "temporal10+bridge": {"refine": {"temporal_reach_frames": 10, "bridge_identities": True}},
    "temporal30+bridge": {"refine": {"temporal_reach_frames": 30, "bridge_identities": True}},
    "bones+temporal10+bridge": {"geometry": {"bone_veto": {"min_bones": 4, "max_bad_share": 0.34}},
                                "refine": {"temporal_reach_frames": 10, "bridge_identities": True}},
    "temporal10+bridge_loose": {"refine": {"temporal_reach_frames": 10, "bridge_identities": True,
                                           "bridge_max_competitor_share": 0.1}},
    # Autofit raw tracks as an independent tracker: cut tracklets where it changes, link where it agrees.
    "autofit": {"_external": "raw"},
    "autofit+temporal10": {"_external": "raw", "refine": {"temporal_reach_frames": 10}},
    "autofit+temporal10+bridge": {"_external": "raw", "refine": {"temporal_reach_frames": 10, "bridge_identities": True}},
    "autofit+bones+temporal10+bridge": {"_external": "raw",
                                        "geometry": {"bone_veto": {"min_bones": 4, "max_bad_share": 0.34}},
                                        "refine": {"temporal_reach_frames": 10, "bridge_identities": True}},
    "autofit+temporal30+bridge": {"_external": "raw", "refine": {"temporal_reach_frames": 30, "bridge_identities": True}},
    "autofit+handoff": {"_external": "raw", "refine": {"merge_handoffs": True}},
    "autofit+handoff+temporal10": {"_external": "raw", "refine": {"merge_handoffs": True, "temporal_reach_frames": 10}},
    "autofit+handoff+temporal10+bridge": {"_external": "raw", "refine": {"merge_handoffs": True, "temporal_reach_frames": 10,
                                                                         "bridge_identities": True}},
    "handoff+temporal10": {"refine": {"merge_handoffs": True, "temporal_reach_frames": 10}},
    **{f"autofit+handoff+temporal10+spot{gap}": {"_external": "raw",
                                                "refine": {"merge_handoffs": True, "temporal_reach_frames": 10},
                                                "temporal": {"same_spot_gap_s": float(gap)}}
       for gap in (10, 30, 60, 120)},
    "adopted": {"_external": "raw"},
    "adopted-no-rebirth": {"_external": "raw", "refine": {"external_rebirth": False}},
    **{f"adopted+only{gap}v{speed}": {"_external": "raw", "temporal": {"only_candidate_gap_s": float(gap),
                                                                      "only_candidate_speed_mps": float(speed)}}
       for gap, speed in ((10, 3), (30, 3), (60, 3), (120, 3), (30, 12), (120, 12))},
    "adopted+duplicates": {"_external": "raw", "refine": {"mark_duplicates": True}},
    # hand-offs that position alone cannot single out, singled out by continuing 2D tracklets in N cameras
    **{f"adopted+handoff_tracklets{n}": {"_external": "raw", "refine": {"handoff_tracklets": n}} for n in (1, 2, 3)},
    "autofit+handoff+temporal10+spot60r30": {"_external": "raw",
                                            "refine": {"merge_handoffs": True, "temporal_reach_frames": 10},
                                            "temporal": {"same_spot_gap_s": 60.0, "same_spot_radius_m": 0.3}},
}
AUTOFIT_LABELS = Path("C:/data_engine_data/_scratch/autofit") / WORKSPACE / "kept" / "labels.npz"


def merged(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key.startswith("_"):
            continue
        result[key] = {**result.get(key, {}), **value} if isinstance(value, dict) else value
    return result


def external_tracks(scene, kind: str):
    with np.load(AUTOFIT_LABELS) as data:
        return [data[f"{kind}_{camera.camera}"].astype(np.int64) for camera in scene.cameras]


def evaluate(scene, reference, solved) -> dict:
    cams = recover.camera_counts(scene, solved.labels)
    multi = np.zeros(solved.labels.next_label() + 1, dtype=bool)
    for label, cameras in cams.items():
        multi[label] = len(cameras) >= 2
    components, identities = [], []
    for camera in scene.cameras:
        lab = solved.labels.label[camera.index]
        in_track = (lab >= 0) & multi[np.clip(lab, 0, None)]
        components.append(np.where(in_track, lab, -1))
        identities.append(np.where(in_track, solved.identity_of_label[np.clip(lab, 0, None)], -1))
    by_component = evaluation.score(scene, reference, components)
    by_identity = evaluation.score(scene, reference, identities)
    pairs = evaluation.cross_camera_pairs(scene, reference, components)
    unknowns = extract_unknowns(solved, set())
    counts = {kind: len(items) for kind, items in unknowns.items() if items}
    return {
        "coverage": by_component["coverage"],
        "impure_obs(component)": by_component["impure_observations"],
        "mixed_components": by_component["mixed_labels"],
        "components": by_component["labels"],
        "impure_obs(identity)": by_identity["impure_observations"],
        "mixed_identities": by_identity["mixed_labels"],
        "identities": by_identity["labels"],
        "people_seen": by_identity["reference_people_seen"],
        "pair_precision": pairs["precision"], "pair_recall": pairs["recall"], "wrong_pairs": pairs["wrong_pairs"],
        "questions": counts, "questions_total": sum(counts.values()),
        "duplicates": duplicate_check(scene, reference, solved),
        "mixed_examples": by_identity["mixed_examples"][:5],
    }


def duplicate_check(scene, reference, solved) -> dict:
    """Of the pieces marked as a second detection of person X: does the reference agree?"""
    majority = {}
    counts: dict[int, dict[int, int]] = {}
    for camera in scene.cameras:
        lab, person = solved.labels.label[camera.index], reference.person[camera.index]
        rows = np.flatnonzero((lab >= 0) & (person >= 0))
        for l, who in zip(lab[rows], person[rows]):
            counts.setdefault(int(l), {}).setdefault(int(who), 0)
            counts[int(l)][int(who)] += 1
    majority = {l: max(c, key=c.get) for l, c in counts.items()}
    rows_marked = agree = disagree = unlabelled = 0
    for camera in scene.cameras:
        duplicate, person = solved.labels.duplicate_of[camera.index], reference.person[camera.index]
        rows = np.flatnonzero(duplicate >= 0)
        rows_marked += len(rows)
        for target, who in zip(duplicate[rows], person[rows]):
            if who < 0:
                unlabelled += 1
            elif majority.get(int(target)) == int(who):
                agree += 1
            else:
                disagree += 1
    return {"detections_marked": rows_marked, "reference_same_person": agree, "reference_other_person": disagree,
            "not_in_reference": unlabelled}


def main(argv: list[str]) -> None:
    if "--list" in argv:
        print("\n".join(VARIANTS))
        return
    names = [a for a in argv if not a.startswith("--")] or list(VARIANTS)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    workspace = Workspace(WORKSPACE)
    started = time.time()
    scene = load_scene(workspace, joint_peak=RECIPE["geometry"]["joint_peak"])
    store = load_site().repo("data_management") / "data" / "labels"
    reference = evaluation.load_reference(scene, store, DATASET)
    print(f"scene T={scene.T}, reference: {reference.matched}/{reference.observations} observations matched, "
          f"{reference.people} people ({time.time() - started:.0f} s)", flush=True)
    results = {}
    cache: dict[str, object] = {}
    for name in names:
        recipe = merged(RECIPE, VARIANTS[name])
        key = json.dumps(recipe["geometry"], sort_keys=True) + json.dumps(recipe["association"], sort_keys=True)
        if key not in cache:
            path = SCRATCH / f"samples_{hashlib.sha256(key.encode()).hexdigest()[:12]}.pkl"
            if path.is_file() and "--fresh" not in argv:
                cache[key] = pickle.loads(path.read_bytes())
            else:
                cache[key] = pair_samples(scene, recipe)
                path.write_bytes(pickle.dumps(cache[key]))
        started = time.time()
        constraints = None
        if VARIANTS[name].get("_external"):
            constraints = Constraints(external=external_tracks(scene, VARIANTS[name]["_external"]))
        solved = solve(scene, recipe, samples=cache[key], constraints=constraints, log=lambda *_: None)
        result = evaluate(scene, reference, solved)
        result["seconds"] = round(time.time() - started)
        results[name] = result
        print(f"\n== {name}  ({result['seconds']} s)")
        for metric, value in result.items():
            if metric not in ("seconds",):
                print(f"   {metric:24s} {value}", flush=True)
    (SCRATCH / f"results_{int(time.time())}.json").write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
