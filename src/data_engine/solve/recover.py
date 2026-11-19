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
    duplicate_of: list[np.ndarray] = None  # per camera (M,) int32: component this row is a second detection of

    SOURCE = {"none": 0, "pairwise": 1, "colocated": 2, "projection_atom": 3, "projection_row": 4, "human": 5}

    def __post_init__(self):
        if self.source is None:
            self.source = [np.where(lab >= 0, 1, 0).astype(np.uint8) for lab in self.label]
        if self.rejected is None:
            self.rejected = [np.zeros(len(lab), dtype=bool) for lab in self.label]
        if self.duplicate_of is None:
            self.duplicate_of = [np.full(len(lab), -1, dtype=np.int32) for lab in self.label]

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


def merge_handoffs(scene: Scene, labels: Labels, frames: PersonFrames, *, max_overlap: int = 15,
                   max_gap: int = 10, radius_in: float = 20.0, clear_radius_in: float = 40.0,
                   tracklet_support: int = 0, support_window: int = 30) -> list[dict]:
    """Merge a component that ends where and when another begins (a tracker hand-off).

    A person tracked in 3D does not vanish and get replaced by someone else at the same spot
    within a second. When component X's last measured position and component Y's first one
    are within ``radius_in`` and Y starts no more than ``max_gap`` sync frames after X ends
    (or the two overlap by at most ``max_overlap`` frames, seen by different cameras), they
    are one person - provided the pairing is unique in both directions within the wider
    ``clear_radius_in``, they never show the same camera at the same time, and no human
    said otherwise.

    In a crowd the pairing is rarely unique by position alone. With ``tracklet_support`` > 0 a
    non-unique hand-off is also merged when the single-camera tracker singles it out: in at
    least that many cameras one 2D tracklet (the tracker's own, before any cut by the 3D
    tracker) holds detections of X before its end and of Y after its start, and no other
    candidate successor of X or predecessor of Y has such a tracklet in any camera.
    """
    from .identity import summarise

    summaries = [s for s in summarise(frames, labels.next_label()) if s is not None and s.n_measured >= 3]
    if len(summaries) < 2:
        return []
    occupied = occupancy(scene, labels)
    members = labels.members()

    def clash(a: int, b: int) -> bool:
        for camera in scene.cameras:
            first, second = occupied.get((a, camera.index)), occupied.get((b, camera.index))
            if first is not None and second is not None and len(np.intersect1d(first, second, assume_unique=True)):
                return True
        return any((min(x, y), max(x, y)) in labels.forbidden for x in members.get(a, ()) for y in members.get(b, ()))

    by_label = {s.label: s for s in summaries}
    # rows of every component per camera, for the tracklet-continuity test
    sorted_rows = []
    for camera in scene.cameras:
        order = np.argsort(labels.label[camera.index], kind="stable")
        sorted_rows.append((order, labels.label[camera.index][order]))
    support_cache: dict[tuple[int, int], int] = {}

    def continuing_tracklets(earlier: int, later: int) -> int:
        """Cameras in which one 2D tracklet holds the earlier component before its end and the later one after its start."""
        key = (earlier, later)
        if key not in support_cache:
            end, start = by_label[earlier].last_root_t, by_label[later].first_root_t
            count = 0
            for camera, (order, sorted_label) in zip(scene.cameras, sorted_rows):
                source = getattr(camera, "source_tracklet", camera.tracklet)
                found = []
                for label, low, high in ((earlier, end - support_window, end), (later, start, start + support_window)):
                    rows = order[np.searchsorted(sorted_label, label):np.searchsorted(sorted_label, label, side="right")]
                    rows = rows[(camera.t[rows] >= low) & (camera.t[rows] <= high)]
                    ids = source[rows]
                    found.append(np.unique(ids[ids >= 0]))
                count += bool(len(np.intersect1d(found[0], found[1], assume_unique=True)))
            support_cache[key] = count
        return support_cache[key]

    by_start = sorted(summaries, key=lambda s: s.first_root_t)
    starts = np.array([s.first_root_t for s in by_start])
    successors: dict[int, list[tuple[int, float]]] = {}
    predecessors: dict[int, list[tuple[int, float]]] = {}
    for earlier in summaries:
        low = np.searchsorted(starts, earlier.last_root_t - max_overlap, side="left")
        high = np.searchsorted(starts, earlier.last_root_t + max_gap, side="right")
        for later in by_start[low:high]:
            if later.label == earlier.label or later.first_root_t <= earlier.first_root_t \
                    or later.last_root_t <= earlier.last_root_t:
                continue
            distance = float(np.linalg.norm(later.first_root - earlier.last_root))
            if distance <= clear_radius_in:
                successors.setdefault(earlier.label, []).append((later.label, distance))
                predecessors.setdefault(later.label, []).append((earlier.label, distance))
    merged = []
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    proposals: list[tuple[int, int, float, str]] = []
    for earlier_label, options in sorted(successors.items()):
        later_label, distance = options[0]
        if len(options) == 1 and len(predecessors[later_label]) == 1 and distance <= radius_in:
            proposals.append((earlier_label, later_label, distance, "unique"))
    if tracklet_support > 0:
        # Hand-offs that position alone does not single out (a crowd), or that the 1 s endpoint medians
        # miss (someone walking): compare the positions AT the hand-off (last / first three measured
        # frames) and let the single-camera trackers single out the successor.
        order = np.lexsort((frames.t, frames.label))
        measured = order[np.isfinite(frames.root[order, 0])]
        cut = np.flatnonzero(np.diff(frames.label[measured])) + 1
        tight = {}
        for chunk in np.split(measured, cut):
            if len(chunk):
                tight[int(frames.label[chunk[0]])] = (np.median(frames.root[chunk[:3]], axis=0),
                                                      np.median(frames.root[chunk[-3:]], axis=0))
        near_after: dict[int, list[tuple[int, float]]] = {}
        near_before: dict[int, list[tuple[int, float]]] = {}
        for earlier in summaries:
            low = np.searchsorted(starts, earlier.last_root_t - max_overlap, side="left")
            high = np.searchsorted(starts, earlier.last_root_t + max_gap, side="right")
            for later in by_start[low:high]:
                if later.label == earlier.label or later.first_root_t <= earlier.first_root_t \
                        or later.last_root_t <= earlier.last_root_t:
                    continue
                distance = float(np.linalg.norm(tight[later.label][0] - tight[earlier.label][1]))
                if distance <= clear_radius_in:
                    near_after.setdefault(earlier.label, []).append((later.label, distance))
                    near_before.setdefault(later.label, []).append((earlier.label, distance))
        already = {(a, b) for a, b, _, _ in proposals}
        for earlier_label, options in sorted(near_after.items()):
            supported = [(label, d, continuing_tracklets(earlier_label, label)) for label, d in options]
            chosen = [(label, d, n) for label, d, n in supported if n >= tracklet_support and d <= radius_in]
            if len(chosen) != 1 or sum(1 for _, _, n in supported if n >= 1) != 1:
                continue
            later_label, distance, count = chosen[0]
            if (earlier_label, later_label) in already:
                continue
            if any(continuing_tracklets(other, later_label) >= 1 for other, _ in near_before[later_label]
                   if other != earlier_label):
                continue
            proposals.append((earlier_label, later_label, distance, f"{count} continuing 2D tracklets"))
    for earlier_label, later_label, distance, evidence in proposals:
        a, b = find(earlier_label), find(later_label)
        if a == b or clash(a, b):
            continue
        parent[b] = a
        for camera in scene.cameras:
            moved = labels.label[camera.index] == b
            if moved.any():
                labels.label[camera.index][moved] = a
                labels.source[camera.index][moved] = np.maximum(labels.source[camera.index][moved],
                                                               Labels.SOURCE["colocated"])
            key_b = (b, camera.index)
            if key_b in occupied:
                occupied[(a, camera.index)] = np.union1d(occupied.get((a, camera.index), np.empty(0, np.int32)),
                                                         occupied.pop(key_b))
        members.setdefault(a, set()).update(members.pop(b, set()))
        merged.append({"from": int(earlier_label), "to": int(later_label), "distance_in": round(distance, 1),
                       "evidence": evidence})
    return merged


def projection_attach(scene: Scene, labels: Labels, frames: PersonFrames, *, min_joints: int = 6,
                      pixel_floor: float = 20.0, diagonal_fraction: float = 0.06, min_common: int = 5,
                      agree_share: float = 0.7, min_votes: int = 5, vote_share: float = 0.7,
                      runner_up_share: float = 0.3, attach_rows: bool = True, temporal_reach: int = 0) -> dict:
    """Attach free atoms / untracked rows to the component whose projected 3D pose they match.

    With ``temporal_reach`` > 0 a person's measured 3D pose is also compared with detections
    up to that many sync frames before or after it, at instants where the person has no
    measured pose of their own. A person who stays put (on a machine, at a rack) and drops
    to a single camera view is still exactly where they were measured a moment ago; the
    piece that continues them matches the pose joint by joint, and a piece of someone
    moving does not. Only tracked pieces are attached this way, never single detections,
    and the same uniqueness rules apply. The attached frames get no 3D position of their
    own (one view cannot measure it); they keep the person's 2D track and identity alive.
    """
    multi = {l for l, cams in camera_counts(scene, labels).items() if len(cams) >= 2}
    body_measured = np.isfinite(frames.xyz[:, BODY, 0]).sum(axis=1)
    usable = np.flatnonzero((body_measured >= min_joints) & np.isin(frames.label, list(multi)))
    cand_index, cand_t = usable, frames.t[usable].astype(np.int64)
    cand_exact = np.ones(len(usable), dtype=bool)
    if temporal_reach > 0 and len(usable):
        stride = scene.T + 1
        filled = np.zeros((int(frames.label.max()) + 1) * stride, dtype=bool)
        filled[frames.label[usable].astype(np.int64) * stride + frames.t[usable]] = True
        extra_index, extra_t = [], []
        for shift in sorted((d for d in range(-temporal_reach, temporal_reach + 1) if d), key=abs):
            target = frames.t[usable].astype(np.int64) + shift
            ok = (target >= 0) & (target < scene.T)
            key = frames.label[usable][ok].astype(np.int64) * stride + target[ok]
            fresh = np.flatnonzero(~filled[key])
            key_fresh, first = np.unique(key[fresh], return_index=True)  # nearest pose wins a slot
            filled[key_fresh] = True
            extra_index.append(usable[ok][fresh][first])
            extra_t.append(target[ok][fresh][first])
        cand_index = np.concatenate([cand_index, *extra_index])
        cand_t = np.concatenate([cand_t, *extra_t])
        cand_exact = np.concatenate([cand_exact, np.zeros(len(cand_index) - len(cand_exact), dtype=bool)])
    occupied = occupancy(scene, labels)
    members = labels.members()
    votes: dict[tuple[int, int, int], list[int]] = {}  # (camera, atom, label) -> [matches, comparable]
    row_matches: list[tuple[int, int, int, int]] = []  # (camera, row, label, t) for untracked rows
    for camera in scene.cameras:
        if camera.cam is None or not len(usable):
            continue
        c = camera.index
        # At the same instant the person must not already have a detection in this camera. A pose
        # borrowed from a nearby instant may come from a moment when this camera did see them
        # (the piece being tested is then the continuation); overlaps are refused at attach time.
        select = (frames.rows[cand_index, c] < 0) | ~cand_exact
        candidates, times_c, exact_c = cand_index[select], cand_t[select], cand_exact[select]
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
        first = np.searchsorted(t_free, times_c, side="left")
        last = np.searchsorted(t_free, times_c, side="right")
        counts = last - first
        keep = counts > 0
        candidates, first, counts = candidates[keep], first[keep], counts[keep]
        times_c, exact_c = times_c[keep], exact_c[keep]
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
        loose = ~tracked & matched & exact_c[pair_frame][comparable]  # single detections: same instant only
        if loose.any():
            times = times_c[pair_frame][comparable][loose]
            row_matches.extend((c, int(r), int(l), int(tt)) for r, l, tt in
                               zip(pair_row[comparable][loose], pair_label[loose], times))
    # Decide per atom: one clear winner, enough votes, no conflict.
    by_atom: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for (c, a, l), (matches, comparable) in votes.items():
        by_atom.setdefault((c, a), []).append((matches, comparable, l))
    attached, ambiguous, bridges = [], [], []
    span_low = np.full(labels.next_label() + 1, np.iinfo(np.int64).max)
    span_high = np.full(labels.next_label() + 1, -1)
    np.minimum.at(span_low, frames.label, frames.t)
    np.maximum.at(span_high, frames.label, frames.t)
    for (c, a), options in sorted(by_atom.items(), key=lambda kv: -max(o[0] for o in kv[1])):
        options.sort(reverse=True)
        matches, comparable, best = options[0]
        if matches < min_votes or matches < vote_share * comparable:
            continue
        runner = options[1] if len(options) > 1 else None
        if runner and runner[0] > runner_up_share * matches:
            # One continuous single-camera piece that matches two people whose measured lifetimes
            # do not overlap is not a conflict: it is the same person, seen by one camera in between.
            second_ok = runner[0] >= min_votes and runner[0] >= vote_share * runner[1]
            third_weak = len(options) < 3 or options[2][0] <= runner_up_share * runner[0]
            disjoint = span_high[best] < span_low[runner[2]] or span_high[runner[2]] < span_low[best]
            if second_ok and third_weak and disjoint:
                bridges.append({"camera": c, "atom": a, "labels": [int(best), int(runner[2])],
                                "matches": [int(matches), int(runner[0])],
                                "comparable": [int(comparable), int(runner[1])]})
                continue
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
    return {"atoms_attached": attached, "atoms_ambiguous": ambiguous, "rows_attached": rows_attached,
            "bridges": bridges}


def mark_duplicates(scene: Scene, labels: Labels, frames: PersonFrames, *, min_joints: int = 6,
                    pixel_floor: float = 20.0, diagonal_fraction: float = 0.06, min_common: int = 5,
                    agree_share: float = 0.7, min_votes: int = 5, vote_share: float = 0.7,
                    runner_up_share: float = 0.3) -> list[dict]:
    """Recognise single-camera pieces that are a second detection of an already tracked person.

    The detector sometimes reports a person twice in one image (typically a partial-body box
    beside the full one, with a different enough pose to survive near-identical-pose
    suppression). Such a piece coincides, in the same camera and at the same instants, with a
    detection that already belongs to a tracked person, and its joints match that person's
    projected 3D pose. It is not a new person and not a missing association: it is marked
    ``duplicate_of`` that person and takes no further part (it stays in the raw store).
    A piece whose joints match nobody's projection is left alone - that is another person.
    """
    multi = {l for l, cams in camera_counts(scene, labels).items() if len(cams) >= 2}
    body_measured = np.isfinite(frames.xyz[:, BODY, 0]).sum(axis=1)
    usable = np.flatnonzero((body_measured >= min_joints) & np.isin(frames.label, list(multi)))
    marked = []
    for camera in scene.cameras:
        if camera.cam is None or not len(usable):
            continue
        c = camera.index
        candidates = usable[frames.rows[usable, c] >= 0]  # the person already has a detection here
        lab = labels.label[c]
        free = np.array([l not in multi for l in range(labels.next_label() + 1)])
        rows_free = camera.order[((lab[camera.order] < 0) | free[np.clip(lab[camera.order], 0, None)])
                                 & ~labels.rejected[c][camera.order] & (labels.atom[c][camera.order] >= 0)
                                 & (labels.duplicate_of[c][camera.order] < 0)]
        if not len(candidates) or not len(rows_free):
            continue
        t_free = camera.t[rows_free]
        by_t = np.argsort(t_free, kind="stable")
        rows_free, t_free = rows_free[by_t], t_free[by_t]
        first = np.searchsorted(t_free, frames.t[candidates], side="left")
        counts = np.searchsorted(t_free, frames.t[candidates], side="right") - first
        keep = counts > 0
        candidates, first, counts = candidates[keep], first[keep], counts[keep]
        if not len(candidates):
            continue
        projected = camera.cam.project(frames.xyz[candidates][:, BODY].astype(np.float64))
        inside = (np.isfinite(projected).all(axis=-1) & (projected[..., 0] >= 0)
                  & (projected[..., 0] < camera.cam.native_width) & (projected[..., 1] >= 0))
        pair_frame = np.repeat(np.arange(len(candidates)), counts)
        offsets = np.arange(counts.sum()) - np.repeat(np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
        pair_row = rows_free[np.repeat(first, counts) + offsets]
        points = camera.detections.keypoints[pair_row][:, BODY]
        common = inside[pair_frame] & camera.joint_ok[pair_row][:, BODY]
        distance = np.linalg.norm(np.nan_to_num(projected[pair_frame]) - points[:, :, :2], axis=-1) * camera.cam.to_cvat
        boxes = camera.detections.boxes[pair_row]
        gate = np.maximum(pixel_floor, diagonal_fraction * np.hypot(boxes[:, 2] - boxes[:, 0],
                                                                   boxes[:, 3] - boxes[:, 1]) * camera.cam.to_cvat)
        n_common = common.sum(axis=1)
        comparable = n_common >= min_common
        match = comparable & ((common & (distance <= gate[:, None])).sum(axis=1)
                              >= np.maximum(min_common, agree_share * n_common))
        keys = np.stack([labels.atom[c][pair_row][comparable], frames.label[candidates][pair_frame][comparable]], axis=1)
        if not len(keys):
            continue
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        seen = np.bincount(inverse, minlength=len(unique))
        hits = np.bincount(inverse, weights=match[comparable].astype(np.float64), minlength=len(unique))
        by_atom: dict[int, list[tuple[int, int, int]]] = {}
        for (atom, label), n_match, n_seen in zip(unique, hits, seen):
            by_atom.setdefault(int(atom), []).append((int(n_match), int(n_seen), int(label)))
        for atom, options in by_atom.items():
            options.sort(reverse=True)
            matches, compared, best = options[0]
            if matches < min_votes or matches < vote_share * compared:
                continue
            if len(options) > 1 and options[1][0] > runner_up_share * matches:
                continue
            rows = np.flatnonzero(labels.atom[c] == atom)
            on_timeline = rows[camera.t[rows] >= 0]
            # Most of the piece must coincide with the person's own detection in this camera.
            if compared < 0.5 * len(on_timeline):
                continue
            labels.duplicate_of[c][rows] = best
            marked.append({"camera": c, "atom": atom, "label": best, "matches": matches, "compared": compared,
                           "rows": int(len(rows))})
    return marked


def competitor_share(scene: Scene, camera_index: int, rows: np.ndarray, *, iou: float = 0.2) -> float:
    """Share of a piece's detections whose box is overlapped by another kept detection in the same frame."""
    camera = scene.cameras[camera_index]
    boxes = camera.detections.boxes
    crowded = 0
    for row in rows:
        frame_rows = camera.detections.rows(int(camera.detections.frame_of_row[row]))
        others = np.flatnonzero(camera.kept[frame_rows]) + frame_rows.start
        others = others[others != row]
        if not len(others):
            continue
        a, b = boxes[row, :4], boxes[others, :4]
        width = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
        height = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - width * height
        crowded += bool((width * height / np.maximum(union, 1e-9) >= iou).any())
    return crowded / max(1, len(rows))


def apply_bridges(scene: Scene, labels: Labels, bridges: list[dict], *, max_competitor_share: float = 0.0) -> list[dict]:
    """Join two people linked by one clean single-camera piece (see ``projection_attach``).

    The piece must be clean - no other detection ever overlaps its box - because inside a
    single view nothing but the 2D tracker vouches that it is one person throughout.
    """
    applied = []
    for bridge in bridges:
        c, atom = bridge["camera"], bridge["atom"]
        first, second = bridge["labels"]
        rows = np.flatnonzero(labels.atom[c] == atom)
        if (labels.label[c][rows] >= 0).any() and not np.isin(labels.label[c][rows], [first, second, -1]).all():
            continue
        current = {int(l) for cam_labels in labels.label for l in np.unique(cam_labels[np.isin(cam_labels, [first, second])])}
        if current != {first, second}:
            continue  # one of the two was merged away earlier in this round
        share = competitor_share(scene, c, rows[scene.cameras[c].t[rows] >= 0])
        occupied = occupancy(scene, labels)
        clash = any(
            (first, camera.index) in occupied and (second, camera.index) in occupied
            and len(np.intersect1d(occupied[(first, camera.index)], occupied[(second, camera.index)], assume_unique=True))
            for camera in scene.cameras)
        members = labels.members()
        vetoed = any((min(x, y), max(x, y)) in labels.forbidden
                     for x in members.get(first, set()) | {atom} for y in members.get(second, set()) | {atom})
        record = {**bridge, "competitor_share": round(share, 3), "applied": False}
        if share > max_competitor_share:
            record["reason"] = "piece is not clean (another detection overlaps it)"
        elif clash or vetoed:
            record["reason"] = "same_camera_overlap" if clash else "human_different"
        else:
            for camera in scene.cameras:
                moved = labels.label[camera.index] == second
                labels.label[camera.index][moved] = first
                labels.source[camera.index][moved] = np.maximum(labels.source[camera.index][moved],
                                                               Labels.SOURCE["colocated"])
            labels.label[c][rows] = first
            labels.source[c][rows] = Labels.SOURCE["projection_atom"]
            record["applied"] = True
        applied.append(record)
    return applied
