"""Shared loaders for the swap-detection research (golden 600 s workspace).

A ``Sol`` is one labelling of the detections plus the 3D track table it implies:
    ident[cam]   (M_cam,) int  identity (or raw track id) of every raw detection row, -1 none
    tr_*         one row per (identity, sync t): identity, t, root (inches), xyz (17,3), rows (C,)
Solutions: ``saved`` (rounds/r002), ``adopted`` / ``baseline`` (re-solved in memory through the
bench, cached under SCRATCH), ``autofit`` (Autofit raw track ids alone, triangulated here).

Golden labels are used for scoring (``gt.py``) and injection tests only.
"""
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "bench"))

DATA = Path(r"C:\data_engine_data")
SCRATCH = DATA / "_scratch" / "switches"
SCRATCH.mkdir(parents=True, exist_ok=True)
GOLDEN = "golden_20250505071549_20250505080000@first600s"
GOLDEN_DS = "golden_20250505071549_20250505080000"
INCH_PER_M = 39.3701
FPS = 10.0


@dataclass
class Sol:
    name: str
    ident: list[np.ndarray]
    tr_ident: np.ndarray
    tr_t: np.ndarray
    tr_root: np.ndarray
    tr_xyz: np.ndarray
    tr_rows: np.ndarray


def load_scene_ref(rays: bool = False):
    from data_engine import evaluation
    from data_engine.config import RECIPE, load_site
    from data_engine.solve.scene import load_scene
    from data_engine.workspace import Workspace
    ws = Workspace(GOLDEN)
    scene = load_scene(ws, joint_peak=RECIPE["geometry"]["joint_peak"], rays=rays)
    ref = evaluation.load_reference(scene, load_site().repo("data_management") / "data" / "labels", GOLDEN_DS)
    return ws, scene, ref


def _save(sol: Sol, path: Path):
    arrays = {f"ident{i}": a for i, a in enumerate(sol.ident)}
    arrays.update(tr_ident=sol.tr_ident, tr_t=sol.tr_t, tr_root=sol.tr_root, tr_xyz=sol.tr_xyz, tr_rows=sol.tr_rows)
    np.savez_compressed(path, **arrays)


def _load(name: str, path: Path) -> Sol:
    with np.load(path) as d:
        n = len([k for k in d.files if k.startswith("ident")])
        return Sol(name, [d[f"ident{i}"] for i in range(n)], d["tr_ident"], d["tr_t"], d["tr_root"], d["tr_xyz"], d["tr_rows"])


def load_saved(ws, scene) -> Sol:
    r = ws.path("rounds", "r002")
    with np.load(r / "solution.npz") as s:
        ident = [s[f"identity_{c.camera}"].astype(np.int64) for c in scene.cameras]
    with np.load(r / "tracks3d.npz") as t:
        return Sol("saved", ident, t["identity"], t["t"], t["root"], t["xyz"], t["rows"])


def bench_variant(name: str) -> Sol:
    path = SCRATCH / f"sol_{name}.npz"
    if path.is_file():
        return _load(name, path)
    import hashlib
    import bench
    from data_engine.config import RECIPE
    from data_engine.solve import recover
    from data_engine.solve.evidence import pair_samples
    from data_engine.solve.pipeline import Constraints, solve
    from data_engine.solve.scene import load_scene
    from data_engine.workspace import Workspace
    ws = Workspace(GOLDEN)
    scene = load_scene(ws, joint_peak=RECIPE["geometry"]["joint_peak"])
    recipe = bench.merged(RECIPE, bench.VARIANTS[name])
    key = json.dumps(recipe["geometry"], sort_keys=True) + json.dumps(recipe["association"], sort_keys=True)
    cache = bench.SCRATCH / f"samples_{hashlib.sha256(key.encode()).hexdigest()[:12]}.pkl"
    samples = pickle.loads(cache.read_bytes()) if cache.is_file() else pair_samples(scene, recipe)
    constraints = None
    if bench.VARIANTS[name].get("_external"):
        constraints = Constraints(external=bench.external_tracks(scene, "raw"))
    solved = solve(scene, recipe, samples=samples, constraints=constraints, log=lambda *_: None)
    cams = recover.camera_counts(solved.scene, solved.labels)
    multi = np.zeros(solved.labels.next_label() + 1, dtype=bool)
    for label, cs in cams.items():
        multi[label] = len(cs) >= 2
    ident = []
    for camera in scene.cameras:
        comp = solved.labels.label[camera.index]
        ok = (comp >= 0) & multi[np.clip(comp, 0, None)]
        ident.append(np.where(ok, solved.identity_of_label[np.clip(comp, 0, None)], -1).astype(np.int64))
    f = solved.frames
    keep = multi[f.label] & (solved.identity_of_label[f.label] >= 0)
    sol = Sol(name, ident, solved.identity_of_label[f.label[keep]], f.t[keep], f.root[keep], f.xyz[keep], f.rows[keep])
    _save(sol, path)
    return sol


def autofit_alone() -> Sol:
    path = SCRATCH / "sol_autofit_raw.npz"
    if path.is_file():
        return _load("autofit", path)
    from data_engine.config import RECIPE
    from data_engine.solve.scene import load_scene
    from data_engine.solve.tracks3d import triangulate_labels
    from data_engine.workspace import Workspace
    ws = Workspace(GOLDEN)
    scene = load_scene(ws, joint_peak=RECIPE["geometry"]["joint_peak"])
    with np.load(ws.path("evidence", "autofit", "labels.npz")) as d:
        raw = [d[f"raw_{c.camera}"].astype(np.int64) for c in scene.cameras]
    frames, _ = triangulate_labels(scene, [r.astype(np.int32) for r in raw], RECIPE)
    # keep only labels that are measured in 3D at least once per row, like the other solutions
    sol = Sol("autofit", raw, frames.label.astype(np.int64), frames.t, frames.root, frames.xyz, frames.rows)
    _save(sol, path)
    return sol


def get_solution(name: str, ws=None, scene=None) -> Sol:
    if name == "saved":
        return load_saved(ws, scene)
    if name == "autofit":
        return autofit_alone()
    return bench_variant(name)


def oracle_tracks(scene, ref):
    """Triangulate reference people directly (label = reference person) on the labelled frames."""
    path = SCRATCH / "oracle_tracks.npz"
    if path.is_file():
        with np.load(path) as d:
            return d["person"], d["t"], d["root"], d["rows"]
    from data_engine.config import RECIPE
    from data_engine.solve.scene import load_scene
    from data_engine.solve.tracks3d import triangulate_labels
    from data_engine.workspace import Workspace
    scene_r = load_scene(Workspace(GOLDEN), joint_peak=RECIPE["geometry"]["joint_peak"])
    frames, _ = triangulate_labels(scene_r, [p.astype(np.int32) for p in ref.person], RECIPE)
    np.savez_compressed(path, person=frames.label, t=frames.t, root=frames.root, rows=frames.rows)
    return frames.label, frames.t, frames.root, frames.rows
