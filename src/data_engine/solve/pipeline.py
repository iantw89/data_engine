"""The solve stage's algorithm: association -> 3D -> refinement fixpoint -> identity -> unknowns."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import recover
from .associate import Association, associate
from .evidence import PairSamples, pair_samples
from .identity import ComponentSummary, Links, link_components, summarise
from .scene import Scene
from .tracks3d import PersonFrames, triangulate_labels


@dataclass
class Constraints:
    """Human evidence, expressed on raw rows: lists of (camera index, row) per endpoint."""
    same: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = field(default_factory=list)
    different: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = field(default_factory=list)
    duplicate: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = field(default_factory=list)
    not_person: list[tuple[int, int]] = field(default_factory=list)  # rows a human rejected
    cuts: dict[int, list[int]] = field(default_factory=dict)  # tracklet -> window cuts (human splits)


@dataclass
class Solved:
    scene: Scene
    association: Association
    labels: recover.Labels  # component per row
    frames: PersonFrames  # 3D per (component, t)
    summaries: list[ComponentSummary | None]
    links: Links
    identity_of_label: np.ndarray  # component -> identity (-1 for components without identity)
    duplicates: list[tuple[int, int, int]]
    log: list[dict]
    timings: dict[str, float]


def _atom_pairs(endpoint_pairs, row_atom) -> list[tuple[int, int]]:
    """Translate (rows, rows) endpoint pairs to atom pairs (every atom on one side x the other)."""
    pairs = set()
    for first, second in endpoint_pairs:
        atoms_a = {int(row_atom[c][r]) for c, r in first if row_atom[c][r] >= 0}
        atoms_b = {int(row_atom[c][r]) for c, r in second if row_atom[c][r] >= 0}
        pairs.update((min(a, b), max(a, b)) for a in atoms_a for b in atoms_b if a != b)
    return sorted(pairs)


def solve(scene: Scene, recipe: dict, *, constraints: Constraints | None = None, step: int = 1,
          samples: PairSamples | None = None, max_rounds: int = 4, log=print) -> Solved:
    constraints = constraints or Constraints()
    timings, events = {}, []
    started = time.time()
    samples = samples if samples is not None else pair_samples(scene, recipe, step=step)
    timings["pair_samples"] = time.time() - started

    def mapper(atom_rows, atom_camera):
        row_atom = [np.full(len(camera.kept), -1, dtype=np.int32) for camera in scene.cameras]
        for atom, rows in enumerate(atom_rows):
            row_atom[atom_camera[atom]][rows] = atom
        must = _atom_pairs(constraints.same, row_atom)
        # A human "same" between two atoms of one camera at the same time is a duplicate, handled below.
        return (must, _atom_pairs(constraints.different, row_atom), _atom_pairs(constraints.duplicate, row_atom))

    started = time.time()
    association = associate(scene, recipe, step=step, samples=samples, cuts=constraints.cuts, constraints=mapper)
    timings["associate"] = time.time() - started
    row_atom = association.row_atom(scene)
    labels = recover.Labels(association.row_component(scene), row_atom,
                            forbidden=set(_atom_pairs(constraints.different, row_atom)),
                            duplicates=set(_atom_pairs(constraints.duplicate, row_atom)))
    for c, r in constraints.not_person:
        labels.label[c][r] = -1
        labels.atom[c][r] = -1  # a rejected detection takes no part in any vote ...
        labels.rejected[c][r] = True  # ... and is never re-attached by projection
    events.append({"step": "associate", "passes": association.passes, "atoms": association.atoms.count,
                   "components": association.n_components, "accepted": len(association.solution.accepted),
                   "ambiguous": len(association.solution.ambiguous), "switch_events": len(association.switch_events)})

    started = time.time()
    frames, duplicates = triangulate_labels(scene, labels.label, recipe)
    for round_number in range(1, max_rounds + 1):
        merged = recover.merge_colocated(scene, labels, frames)
        n_merged = sum(m["merged"] for m in merged)
        if n_merged:
            frames, duplicates = triangulate_labels(scene, labels.label, recipe)
        attached = recover.projection_attach(scene, labels, frames)
        changed = n_merged + len(attached["atoms_attached"]) + attached["rows_attached"]
        events.append({"step": "refine", "round": round_number, "components_merged": n_merged,
                       "merge_refused": [m for m in merged if not m["merged"]][:50],
                       "atoms_attached": len(attached["atoms_attached"]), "rows_attached": attached["rows_attached"],
                       "atoms_ambiguous": attached["atoms_ambiguous"][:200]})
        log(f"refine round {round_number}: merged {n_merged} components, attached "
            f"{len(attached['atoms_attached'])} pieces and {attached['rows_attached']} detections")
        if not changed:
            break
        frames, duplicates = triangulate_labels(scene, labels.label, recipe)
    timings["refine"] = time.time() - started

    started = time.time()
    n_labels = labels.next_label()
    summaries = summarise(frames, n_labels)
    row_atom_now = labels.atom
    members = labels.members()
    atom_label = {atom: label for label, atoms in members.items() for atom in atoms}

    def label_pairs(endpoint_pairs):
        pairs = set()
        for a, b in _atom_pairs(endpoint_pairs, row_atom_now):
            if a in atom_label and b in atom_label and atom_label[a] != atom_label[b]:
                pairs.add((atom_label[a], atom_label[b]))
        return sorted(pairs)

    links = link_components(summaries, recipe, T=scene.T, must_link=label_pairs(constraints.same),
                            cannot_link=label_pairs(constraints.different))
    timings["identity"] = time.time() - started
    events.append({"step": "identity", "links": len(links.accepted), "open_ends": len(links.open_ends)})
    return Solved(scene, association, labels, frames, summaries, links, links.identity, duplicates, events, timings)


def coverage_metrics(solved: Solved) -> dict:
    """How much of the observed data is explained, for the dashboard and the round ledger."""
    scene, labels, frames = solved.scene, solved.labels, solved.frames
    cams = recover.camera_counts(scene, labels)
    multi = np.zeros(labels.next_label() + 1, dtype=bool)
    for label, cameras in cams.items():
        multi[label] = len(cameras) >= 2
    kept = tracked = in_multi = on_timeline = 0
    for camera in scene.cameras:
        lab = labels.label[camera.index]
        k = camera.kept
        kept += int(k.sum())
        on_timeline += int((k & (camera.t >= 0)).sum())
        tracked += int((k & (camera.tracklet >= 0)).sum())
        in_multi += int((k & (lab >= 0) & multi[np.clip(lab, 0, None)]).sum())
    views = (frames.rows >= 0).sum(axis=1)
    measured = np.isfinite(frames.root[:, 0])
    multi_frames = multi[frames.label] & (views >= 1)
    joints = np.isfinite(frames.xyz[:, :, 0])
    residual = frames.residual[joints]
    eligible = [s for s in solved.summaries if s is not None and s.n_measured >= 5]
    identities = {int(solved.identity_of_label[s.label]) for s in eligible}
    return {
        "detections_kept": kept, "detections_on_timeline": on_timeline, "detections_tracked": tracked,
        "detections_in_3d_tracks": in_multi,
        "share_detections_in_3d_tracks": round(in_multi / max(1, on_timeline), 4),
        "components_multi_camera": int(multi.sum()), "components_with_3d": len(eligible),
        "identities": len(identities),
        "person_frames": int(multi_frames.sum()), "person_frames_with_3d_root": int((measured & multi_frames).sum()),
        "share_person_frames_with_3d": round(float((measured & multi_frames).sum() / max(1, multi_frames.sum())), 4),
        "joints_measured_3d": int(joints.sum()),
        "reprojection_px_median": round(float(np.median(residual)), 2) if len(residual) else None,
        "reprojection_px_p90": round(float(np.percentile(residual, 90)), 2) if len(residual) else None,
        "identity_links": len(solved.links.accepted),
    }
