"""Identity across time: link components that are the same person before and after a gap.

A component is one person observed continuously (by any mix of cameras). When a person is
lost by every camera - occluded, out of view - a new component starts when they are seen
again. Linking uses physics before appearance:

* a person cannot be in two places: components that overlap in time are different people
  (their cross-camera evidence was already judged by the association pass);
* a person cannot teleport: a successor must start within reach of where the predecessor
  was last measured (golden review filter: 12 m/s x elapsed + 0.5 m);
* a link is accepted only when it is the *unique* feasible continuation in both
  directions, within a short gap and a tight radius. Each accepted link removes a candidate
  from every other open end, so the pass is repeated until nothing changes - the same
  elimination recursion as in association.

Everything else is left open and reported with its remaining candidates. No identity is
assigned by default and "new person" is never assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tracks3d import PersonFrames

INCH = 0.0254


@dataclass
class ComponentSummary:
    label: int
    t_first: int
    t_last: int
    n_frames: int
    n_measured: int  # frames with a 3D root
    cameras: tuple[int, ...]
    detections: int
    first_root: np.ndarray | None  # median 3D root over the first measured second, inches
    first_root_t: int | None
    last_root: np.ndarray | None
    last_root_t: int | None


def summarise(frames: PersonFrames, n_labels: int) -> list[ComponentSummary | None]:
    order = np.lexsort((frames.t, frames.label))
    label, t = frames.label[order], frames.t[order]
    boundaries = np.flatnonzero(np.diff(label)) + 1
    summaries: list[ComponentSummary | None] = [None] * n_labels
    for chunk in np.split(np.arange(len(label)), boundaries):
        if not len(chunk):
            continue
        rows = order[chunk]
        present = frames.rows[rows] >= 0
        measured = np.isfinite(frames.root[rows, 0])
        first_root = last_root = None
        first_t = last_t = None
        if measured.any():
            index = np.flatnonzero(measured)
            times = t[chunk][index]
            head, tail = index[times <= times[0] + 10], index[times >= times[-1] - 10]
            first_root = np.median(frames.root[rows[head]], axis=0)
            last_root = np.median(frames.root[rows[tail]], axis=0)
            first_t, last_t = int(times[0]), int(times[-1])
        summaries[int(label[chunk[0]])] = ComponentSummary(
            int(label[chunk[0]]), int(t[chunk[0]]), int(t[chunk[-1]]), len(chunk), int(measured.sum()),
            tuple(int(c) for c in np.flatnonzero(present.any(axis=0))), int(present.sum()),
            first_root, first_t, last_root, last_t)
    return summaries


@dataclass
class Links:
    identity: np.ndarray  # (n_labels,) identity id per component (dense, by first appearance)
    accepted: list[dict]
    open_ends: list[dict] = field(default_factory=list)  # unexplained starts/ends with candidates


def link_components(summaries: list[ComponentSummary | None], recipe: dict, *, T: int, fps: float = 10.0,
                    edge_margin: int = 20, must_link: list[tuple[int, int]] = (),
                    cannot_link: list[tuple[int, int]] = (), min_measured: int = 5) -> Links:
    temporal = recipe["temporal"]
    reach_speed, slack = temporal["max_speed_mps"], temporal["slack_m"]
    auto_gap, auto_speed, auto_radius = temporal["auto_gap_s"], temporal["auto_speed_mps"], temporal["auto_radius_m"]
    candidate_gap = temporal["candidate_gap_s"]
    n = len(summaries)
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    forbidden = {(min(a, b), max(a, b)) for a, b in cannot_link}
    eligible = sorted((s for s in summaries if s is not None and s.n_measured >= min_measured),
                      key=lambda s: s.first_root_t)
    eligible_labels = {s.label for s in eligible}
    starts = np.array([s.first_root_t for s in eligible])
    chains: dict[int, list[ComponentSummary]] = {s.label: [s] for s in eligible}  # chain root -> members

    def chain(label: int) -> list[ComponentSummary]:
        return chains[find(label)]

    def compatible(a: int, b: int) -> bool:
        """Two chains can be one person only if none of their components overlap in time."""
        if find(a) == find(b):
            return False
        members_a, members_b = chain(a), chain(b)
        if forbidden and any((min(x.label, y.label), max(x.label, y.label)) in forbidden
                             for x in members_a for y in members_b):
            return False
        return not any(x.t_first <= y.t_last and y.t_first <= x.t_last for x in members_a for y in members_b)

    def join(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            chains[ra].extend(chains.pop(rb))

    accepted = []
    for a, b in must_link:
        if a in eligible_labels and b in eligible_labels and find(a) != find(b):
            join(a, b)
            accepted.append({"from": int(a), "to": int(b), "source": "human"})

    def feasible(earlier: ComponentSummary, later: ComponentSummary) -> tuple[float, float] | None:
        """(gap seconds, distance metres) if ``later`` can be ``earlier`` continued, else None."""
        if later.first_root_t <= earlier.last_root_t or later.t_first <= earlier.t_last:
            return None
        gap = (later.first_root_t - earlier.last_root_t) / fps
        distance = float(np.linalg.norm(later.first_root - earlier.last_root)) * INCH
        return (gap, distance) if distance <= reach_speed * gap + slack else None

    def candidates():
        successors: dict[int, list[tuple[int, float, float]]] = {}
        predecessors: dict[int, list[tuple[int, float, float]]] = {}
        for earlier in eligible:
            # Only a chain's open ends are linkable: its last component ...
            if max(chain(earlier.label), key=lambda s: s.t_last) is not earlier:
                continue
            low = np.searchsorted(starts, earlier.last_root_t, side="right")
            high = np.searchsorted(starts, earlier.last_root_t + candidate_gap * fps, side="right")
            for later in eligible[low:high]:
                # ... and the first component of the later chain.
                if min(chain(later.label), key=lambda s: s.t_first) is not later:
                    continue
                reach = feasible(earlier, later)
                if reach is None or not compatible(earlier.label, later.label):
                    continue
                successors.setdefault(earlier.label, []).append((later.label, *reach))
                predecessors.setdefault(later.label, []).append((earlier.label, *reach))
        return successors, predecessors

    def tight(options):
        """Candidates a person could plausibly be within a short gap (walking pace, small radius)."""
        return [o for o in options if o[1] <= auto_gap and o[2] <= auto_radius + auto_speed * o[1]]

    while True:
        successors, predecessors = candidates()
        merged = 0
        for earlier_label, options in successors.items():
            # Unique in both directions among everything within plausible reach. The loose
            # review gate (12 m/s) admits most of the room after a few seconds, so it only
            # defines the candidate list shown to a human, not the automatic rule.
            near = tight(options)
            if len(near) != 1:
                continue
            later_label, gap, distance = near[0]
            if len(tight(predecessors.get(later_label, []))) != 1:
                continue
            if compatible(earlier_label, later_label):
                join(earlier_label, later_label)
                accepted.append({"from": int(earlier_label), "to": int(later_label), "gap_s": round(gap, 2),
                                 "distance_m": round(distance, 2), "source": "machine:unique_motion_continuity"})
                merged += 1
        if not merged:
            break

    # Open ends: starts/ends inside the recording without an accepted link, with their candidates.
    def describe(options):
        return [{"component": int(c), "gap_s": round(g, 2), "distance_m": round(d, 2)}
                for c, g, d in sorted(options, key=lambda o: o[2])]

    open_ends = []
    for summary in eligible:
        members = chain(summary.label)
        if summary.t_first > edge_margin and min(members, key=lambda s: s.t_first) is summary:
            open_ends.append({"component": summary.label, "end": "start", "t": summary.t_first,
                              "candidates": describe(predecessors.get(summary.label, []))})
        if summary.t_last < T - 1 - edge_margin and max(members, key=lambda s: s.t_last) is summary:
            open_ends.append({"component": summary.label, "end": "end", "t": summary.t_last,
                              "candidates": describe(successors.get(summary.label, []))})
    roots = np.array([find(i) for i in range(n)])
    first_seen: dict[int, int] = {}
    for summary in sorted((s for s in summaries if s is not None), key=lambda s: s.t_first):
        first_seen.setdefault(int(roots[summary.label]), len(first_seen))
    identity = np.array([first_seen.get(int(r), -1) for r in roots], dtype=np.int32)
    return Links(identity, accepted, open_ends)
