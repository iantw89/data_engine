"""Run cross-camera association for a scene: samples -> atoms -> components (iterated)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import cluster, evidence
from .scene import Scene, tracklet_table


@dataclass
class Association:
    atoms: cluster.Atoms
    atom_camera: np.ndarray  # (A,) camera index
    atom_rows: list[np.ndarray]  # raw rows per atom, time-sorted
    atom_t: list[np.ndarray]  # sync indices per atom
    solution: cluster.Solution
    edges: cluster.Histograms  # pooled atom-pair evidence
    cells: cluster.Histograms  # (tracklet, tracklet, window) evidence
    component: np.ndarray  # (A,) dense component id
    window: int
    passes: int
    switch_events: list[dict] = field(default_factory=list)

    @property
    def n_components(self) -> int:
        return int(self.component.max()) + 1 if len(self.component) else 0

    def row_component(self, scene: Scene) -> list[np.ndarray]:
        """Per camera: component id per raw row (-1 where the row is in no atom)."""
        result = [np.full(len(camera.kept), -1, dtype=np.int32) for camera in scene.cameras]
        for atom, rows in enumerate(self.atom_rows):
            result[self.atom_camera[atom]][rows] = self.component[atom]
        return result

    def row_atom(self, scene: Scene) -> list[np.ndarray]:
        result = [np.full(len(camera.kept), -1, dtype=np.int32) for camera in scene.cameras]
        for atom, rows in enumerate(self.atom_rows):
            result[self.atom_camera[atom]][rows] = atom
        return result


def build_atoms(table: dict[int, dict], cuts: dict[int, list[int]], window: int):
    tracklets = np.array(sorted(table), dtype=np.int64)
    atoms = cluster.make_atoms(tracklets, cuts)
    atom_camera = np.empty(atoms.count, dtype=np.int16)
    atom_rows: list[np.ndarray] = [None] * atoms.count
    atom_t: list[np.ndarray] = [None] * atoms.count
    for position, tid in enumerate(atoms.tracklets):
        entry = table[int(tid)]
        rows, t = entry["rows"], entry["t"]
        tracklet_cuts = np.asarray(atoms.cuts.get(int(tid), []), dtype=np.int64)
        piece = (np.searchsorted(tracklet_cuts, t // window, side="right") if len(tracklet_cuts)
                 else np.zeros(len(t), dtype=np.int64))
        for k in range(len(tracklet_cuts) + 1):
            atom = int(atoms.first_atom[position]) + k
            mask = piece == k
            atom_camera[atom] = entry["camera"]
            atom_rows[atom], atom_t[atom] = rows[mask], t[mask]
    return atoms, atom_camera, atom_rows, atom_t


ConstraintMapper = Callable[[list[np.ndarray], np.ndarray], tuple[list[tuple[int, int]], list[tuple[int, int]],
                                                                  list[tuple[int, int]]]]


def associate(scene: Scene, recipe: dict, *, step: int = 1, window: int = 20, max_passes: int = 4,
              samples: evidence.PairSamples | None = None, cuts: dict[int, list[int]] | None = None,
              constraints: ConstraintMapper | None = None) -> Association:
    """Association to a fixpoint.

    Each pass builds atoms from the current cuts, pools evidence, grows components, then
    looks for identity switches by consensus; new cuts trigger another pass. ``constraints``
    maps human evidence onto the current atoms: ``(atom_rows, atom_camera) -> (must_link,
    cannot_link, duplicates)`` as atom-index pairs. ``cuts`` seeds human-confirmed splits.
    """
    association = recipe["association"]
    samples = samples if samples is not None else evidence.pair_samples(scene, recipe, step=step)
    cells = cluster.tracklet_window_cells(scene, samples, window)
    table = tracklet_table(scene)
    cuts = {int(t): list(c) for t, c in (cuts or {}).items()}
    min_samples = max(3, association["min_shared_frames"] // step)
    events: list[dict] = []
    for iteration in range(1, max_passes + 1):
        atoms, atom_camera, atom_rows, atom_t = build_atoms(table, cuts, window)
        unit_a = atoms.atom_of(cells.a, cells.window.astype(np.int64))
        unit_b = atoms.atom_of(cells.b, cells.window.astype(np.int64))
        edges = cluster.regroup(cells, unit_a, unit_b)
        components = cluster.Components(atoms.count, atom_camera, [np.unique(t) for t in atom_t])
        must_link, cannot_link, duplicates = constraints(atom_rows, atom_camera) if constraints else ([], [], [])
        components.duplicates = {(min(a, b), max(a, b)) for a, b in duplicates}
        solution = cluster.agglomerate(edges, components, min_samples=min_samples,
                                       margin=association["margin_f2"], must_link=must_link,
                                       cannot_link=cannot_link)
        _, dense = np.unique(solution.labels, return_inverse=True)
        added, new_events = cluster.refine_cuts(cells, atoms, dense)
        if not added or iteration == max_passes:
            break
        for tracklet, tracklet_cuts in added.items():
            cuts[tracklet] = tracklet_cuts
        events.extend({**event, "pass": iteration} for event in new_events)
    return Association(atoms, atom_camera, atom_rows, atom_t, solution, edges, cells, dense.astype(np.int32),
                       window, iteration, events)
