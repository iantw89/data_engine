"""3D-first refinement: once a person has a measured 3D pose, use it.

Two-view geometry is ambiguous along epipolar lines; a 3D pose is not. After the pairwise
bootstrap gives some components a triangulated pose, this module

* merges components that are measured at the same place at the same time (two people
  cannot occupy one spot), and
* attaches single-camera pieces and untracked detections to the person whose *projected*
  3D pose they match joint by joint ("projection ranking", golden recall_v2), by voting
  over the piece's lifetime.

Every attachment adds views, which measures more joints and more frames in 3D, which makes
further pieces decidable - the caller repeats triangulate -> merge -> attach to a fixpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import BODY
from .scene import Scene
from .tracks3d import PersonFrames

INCH = 0.0254


@dataclass
class Labels:
    """Mutable assignment of raw rows to components."""
    label: list[np.ndarray]  # per camera (M,) int32 component id, -1 = none
    atom: list[np.ndarray]  # per camera (M,) int32 atom id, -1 = untracked
    forbidden: set[tuple[int, int]] = field(default_factory=set)  # atom pairs that must stay apart (human)
    duplicates: set[tuple[int, int]] = field(default_factory=set)  # atom pairs allowed to coexist in one camera
    source: list[np.ndarray] = None  # per camera (M,) uint8: how the row got its label
    rejected: list[np.ndarray] = None  # per camera (M,) bool: a human said this is not a person

    SOURCE = {"none": 0, "pairwise": 1, "colocated": 2, "projection_atom": 3, "projection_row": 4, "human": 5}

    def __post_init__(self):
        if self.source is None:
            self.source = [np.where(lab >= 0, 1, 0).astype(np.uint8) for lab in self.label]
        if self.rejected is None:
            self.rejected = [np.zeros(len(lab), dtype=bool) for lab in self.label]

    def next_label(self) -> int:
        return max((int(lab.max()) for lab in self.label if len(lab)), default=-1) + 1

    def members(self) -> dict[int, set[int]]:
        """label -> atoms."""
        result: dict[int, set[int]] = {}
        for lab, atom in zip(self.label, self.atom):
            mask = (lab >= 0) & (atom >= 0)
            if not mask.any():
                continue
            pairs = np.unique(np.stack([lab[mask], atom[mask]], axis=1), axis=0)
            for l, a in pairs:
                result.setdefault(int(l), set()).add(int(a))
        return result


def camera_counts(scene: Scene, labels: Labels) -> dict[int, set[int]]:
    """label -> cameras that contribute rows."""
    result: dict[int, set[int]] = {}
    for camera in scene.cameras:
        for l in np.unique(labels.label[camera.index][labels.label[camera.index] >= 0]):
            result.setdefault(int(l), set()).add(camera.index)
    return result


def occupancy(scene: Scene, labels: Labels) -> dict[tuple[int, int], np.ndarray]:
    """(label, camera) -> sorted sync indices occupied."""
    result = {}
    for camera in scene.cameras:
        lab = labels.label[camera.index]
        rows = camera.order[lab[camera.order] >= 0]
        order = np.argsort(lab[rows], kind="stable")
        rows = rows[order]
        ids = lab[rows]
        for chunk in np.split(np.arange(len(ids)), np.flatnonzero(np.diff(ids)) + 1):
            if len(chunk):
                result[(int(ids[chunk[0]]), camera.index)] = np.unique(camera.t[rows[chunk]])
    return result


def merge_colocated(scene: Scene, labels: Labels, frames: PersonFrames, *, radius_in: float = 10.0,
                    p80_in: float = 18.0, min_frames: int = 10) -> list[dict]:
    """Merge components whose measured 3D roots coincide over their common frames."""
    measured = np.isfinite(frames.root[:, 0])
    if not measured.any():
        return []
    t, label, root = frames.t[measured], frames.label[measured], frames.root[measured].astype(np.float64)
    order = np.argsort(t, kind="stable")
    t, label, root = t[order], label[order], root[order]
    starts = np.flatnonzero(np.concatenate(([True], np.diff(t) != 0)))
    ends = np.concatenate((starts[1:], [len(t)]))
    pairs: dict[tuple[int, int], list[float]] = {}
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        distance = np.linalg.norm(root[s:e, None] - root[None, s:e], axis=-1)
        i, j = np.nonzero(np.triu(distance < 3 * p80_in, k=1))
        for a, b, d in zip(label[s:e][i], label[s:e][j], distance[i, j]):
            pairs.setdefault((int(min(a, b)), int(max(a, b))), []).append(float(d))
    occupied = occupancy(scene, labels)
    members = labels.members()
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    merged = []
    for (a, b), distances in sorted(pairs.items(), key=lambda kv: np.median(kv[1])):
        if len(distances) < min_frames:
            continue
        median, p80 = float(np.median(distances)), float(np.percentile(distances, 80))
        if median > radius_in or p80 > p80_in:
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        # Same camera at the same time -> two detections of one spot: a duplicate question, not a merge.
        clash = False
        for camera in scene.cameras:
            first, second = occupied.get((ra, camera.index)), occupied.get((rb, camera.index))
            if first is not None and second is not None and len(np.intersect1d(first, second, assume_unique=True)):
                clash = True
                break
        if clash or any((min(x, y), max(x, y)) in labels.forbidden
                        for x in members.get(ra, ()) for y in members.get(rb, ())):
            merged.append({"a": ra, "b": rb, "median_in": round(median, 1), "frames": len(distances),
                           "merged": False, "reason": "same_camera_overlap" if clash else "human_different"})
            continue
        parent[rb] = ra
        for camera in scene.cameras:
            key_a, key_b = (ra, camera.index), (rb, camera.index)
            if key_b in occupied:
                occupied[key_a] = np.union1d(occupied.get(key_a, np.empty(0, np.int32)), occupied.pop(key_b))
            lab = labels.label[camera.index]
            moved = lab == rb
            lab[moved] = ra
            labels.source[camera.index][moved] = np.maximum(labels.source[camera.index][moved], Labels.SOURCE["colocated"])
        members.setdefault(ra, set()).update(members.pop(rb, set()))
        merged.append({"a": ra, "b": rb, "median_in": round(median, 1), "p80_in": round(p80, 1),
                       "frames": len(distances), "merged": True})
    return merged


def projection_attach(scene: Scene, labels: Labels, frames: PersonFrames, *, min_joints: int = 6,
                      pixel_floor: float = 20.0, diagonal_fraction: float = 0.06, min_common: int = 5,
                      agree_share: float = 0.7, min_votes: int = 5, vote_share: float = 0.7,
                      runner_up_share: float = 0.3, attach_rows: bool = True) -> dict:
    """Attach free atoms / untracked rows to the component whose projected 3D pose they match."""
    multi = {l for l, cams in camera_counts(scene, labels).items() if len(cams) >= 2}
    body_measured = np.isfinite(frames.xyz[:, BODY, 0]).sum(axis=1)
    usable = np.flatnonzero((body_measured >= min_joints) & np.isin(frames.label, list(multi)))
    occupied = occupancy(scene, labels)
    members = labels.members()
    votes: dict[tuple[int, int, int], list[int]] = {}  # (camera, atom, label) -> [matches, comparable]
    row_matches: list[tuple[int, int, int, int]] = []  # (camera, row, label, t) for untracked rows
    for camera in scene.cameras:
        if camera.cam is None or not len(usable):
            continue
        c = camera.index
        candidates = usable[frames.rows[usable, c] < 0]
        if not len(candidates):
            continue
        lab = labels.label[c]
        free = np.array([l not in multi for l in range(labels.next_label() + 1)])
        # Free rows of this camera: kept, on the timeline, not already in a multi-camera component.
        rows_free = camera.order[((lab[camera.order] < 0) | free[np.clip(lab[camera.order], 0, None)])
                                 & ~labels.rejected[c][camera.order]]
        if not len(rows_free):
            continue
        t_free = camera.t[rows_free]
        by_t = np.argsort(t_free, kind="stable")
        rows_free, t_free = rows_free[by_t], t_free[by_t]
        first = np.searchsorted(t_free, frames.t[candidates], side="left")
        last = np.searchsorted(t_free, frames.t[candidates], side="right")
        counts = last - first
        keep = counts > 0
        candidates, first, counts = candidates[keep], first[keep], counts[keep]
        if not len(candidates):
            continue
        projected = camera.cam.project(frames.xyz[candidates][:, BODY].astype(np.float64))  # (N,12,2) native
        inside = (np.isfinite(projected).all(axis=-1) & (projected[..., 0] >= 0) & (projected[..., 0] < camera.cam.native_width)
                  & (projected[..., 1] >= 0))
        pair_frame = np.repeat(np.arange(len(candidates)), counts)
        offsets = np.arange(counts.sum()) - np.repeat(np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
        pair_row = rows_free[np.repeat(first, counts) + offsets]
        points = camera.detections.keypoints[pair_row][:, BODY]
        common = inside[pair_frame] & camera.joint_ok[pair_row][:, BODY]
        distance = np.linalg.norm(np.nan_to_num(projected[pair_frame]) - points[:, :, :2], axis=-1) * camera.cam.to_cvat
        boxes = camera.detections.boxes[pair_row]
        diagonal = np.hypot(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) * camera.cam.to_cvat
        gate = np.maximum(pixel_floor, diagonal_fraction * diagonal)
        agree = (common & (distance <= gate[:, None])).sum(axis=1)
        n_common = common.sum(axis=1)
        comparable = n_common >= min_common
        match = comparable & (agree >= np.maximum(min_common, agree_share * n_common))
        atoms = labels.atom[c][pair_row][comparable]
        pair_label = frames.label[candidates][pair_frame][comparable]
        matched = match[comparable]
        tracked = atoms >= 0
        if tracked.any():
            keys = np.stack([atoms[tracked], pair_label[tracked]], axis=1)
            unique, inverse = np.unique(keys, axis=0, return_inverse=True)
            inverse = inverse.reshape(-1)
            seen = np.bincount(inverse, minlength=len(unique))
            hits = np.bincount(inverse, weights=matched[tracked].astype(np.float64), minlength=len(unique))
            for (a, l), n_match, n_seen in zip(unique, hits, seen):
                entry = votes.setdefault((c, int(a), int(l)), [0, 0])
                entry[0] += int(n_match)
                entry[1] += int(n_seen)
        loose = ~tracked & matched
        if loose.any():
            times = frames.t[candidates][pair_frame][comparable][loose]
            row_matches.extend((c, int(r), int(l), int(tt)) for r, l, tt in
                               zip(pair_row[comparable][loose], pair_label[loose], times))
    # Decide per atom: one clear winner, enough votes, no conflict.
    by_atom: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for (c, a, l), (matches, comparable) in votes.items():
        by_atom.setdefault((c, a), []).append((matches, comparable, l))
    attached, ambiguous = [], []
    for (c, a), options in sorted(by_atom.items(), key=lambda kv: -max(o[0] for o in kv[1])):
        options.sort(reverse=True)
        matches, comparable, best = options[0]
        if matches < min_votes or matches < vote_share * comparable:
            continue
        runner = options[1] if len(options) > 1 else None
        if runner and runner[0] > runner_up_share * matches:
            ambiguous.append({"camera": c, "atom": a, "options": [{"label": o[2], "matches": o[0], "comparable": o[1]}
                                                                    for o in options[:3]]})
            continue
        rows = np.flatnonzero(labels.atom[c] == a)
        rows = rows[scene.cameras[c].t[rows] >= 0]
        times = np.unique(scene.cameras[c].t[rows])
        taken = occupied.get((best, c))
        if taken is not None and len(np.intersect1d(times, taken, assume_unique=True)):
            ambiguous.append({"camera": c, "atom": a, "options": [{"label": best, "matches": matches,
                                                                    "comparable": comparable}],
                              "reason": "same_camera_overlap"})
            continue
        if any((min(a, other), max(a, other)) in labels.forbidden for other in members.get(best, ())):
            continue
        all_rows = np.flatnonzero(labels.atom[c] == a)
        labels.label[c][all_rows] = best
        labels.source[c][all_rows] = Labels.SOURCE["projection_atom"]
        occupied[(best, c)] = np.union1d(taken, times) if taken is not None else times
        members.setdefault(best, set()).add(a)
        attached.append({"camera": c, "atom": a, "label": best, "matches": matches, "comparable": comparable,
                         "rows": int(len(all_rows))})
    rows_attached = 0
    if attach_rows and row_matches:
        # Untracked rows: accept when exactly one label matches the row and that label has no
        # other detection in this camera at that instant.
        per_row: dict[tuple[int, int], set[int]] = {}
        per_slot: dict[tuple[int, int, int], set[int]] = {}
        for c, r, l, tt in row_matches:
            per_row.setdefault((c, r), set()).add(l)
            per_slot.setdefault((c, l, tt), set()).add(r)
        for c, r, l, tt in row_matches:
            if len(per_row[(c, r)]) != 1 or len(per_slot[(c, l, tt)]) != 1:
                continue
            taken = occupied.get((l, c))
            if taken is not None and tt in taken:
                continue
            labels.label[c][r] = l
            labels.source[c][r] = Labels.SOURCE["projection_row"]
            occupied[(l, c)] = np.union1d(taken, [tt]) if taken is not None else np.array([tt])
            rows_attached += 1
    return {"atoms_attached": attached, "atoms_ambiguous": ambiguous, "rows_attached": rows_attached}
