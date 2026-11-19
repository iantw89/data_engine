"""Cross-camera association: atoms -> components.

Atoms are pieces of single-camera tracklets. Evidence between two atoms of different
cameras is the pooled histogram of their simultaneous two-view residuals. Components are
grown by constrained agglomeration:

* an edge is *supported* when most samples meet the review gate (p80 <= 2 accept units);
* it is *contradicted* when most samples are far outside it (median > 4);
* a component can never contain two atoms of one camera that are visible at the same time,
  a contradicted pair, or a pair a human marked different;
* two components merge only if their *pooled* evidence - every cross pair, not just the
  proposing edge - is supported (N-view consistency: three agreeing views must not hide a
  fourth that is someone else);
* a supported edge is only taken when it is unambiguous: every competing atom (same camera
  as the partner, also supported) is clearly worse - or has been *eliminated*, i.e. now
  belongs to a component that cannot contain this atom.

The last rule is the recursion: each accepted link removes competitors for other links, so
edges that were undecidable in isolation become decidable. The loop runs to a fixpoint and
whatever stays ambiguous is reported as an unknown, with its competing hypotheses.

Two-view geometry cannot separate two people who stay on the same epipolar line, so a
single pair that flips between consistent and inconsistent is NOT treated as an identity
switch. Switches are detected by consensus instead (``refine_cuts``): a tracklet piece that
is consistent with one component for a sustained run of windows and with a different,
incompatible component for another run is cut between the runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .evidence import PairSamples
from .scene import Scene

EDGES = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])  # histogram bin upper edges of u; last bin open
CENTRES = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, 10.0])
BIN_1, BIN_2, BIN_4 = 1, 3, 5  # inclusive bin index for u <= 1, <= 2, <= 4


def frac(hist: np.ndarray, upto: int) -> np.ndarray:
    return hist[..., :upto + 1].sum(axis=-1) / np.maximum(1, hist.sum(axis=-1))


def mean_of(hist: np.ndarray) -> float:
    """Mean residual (accept units) of one histogram, from bin centres."""
    return float((hist * CENTRES).sum() / max(1, hist.sum()))


@dataclass
class Histograms:
    """Pooled residual histograms per (unit a < unit b[, window])."""
    a: np.ndarray
    b: np.ndarray
    window: np.ndarray
    hist: np.ndarray  # (K, 9) int32

    def __len__(self) -> int:
        return len(self.a)

    @property
    def n(self) -> np.ndarray:
        return self.hist.sum(axis=1)

    def frac(self, upto: int) -> np.ndarray:
        return frac(self.hist, upto)

    @property
    def mean_u(self) -> np.ndarray:
        return (self.hist * CENTRES).sum(axis=1) / np.maximum(1, self.n)


def _group_sum(keys: list[np.ndarray], hist: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    if not len(hist):
        return [key[:0] for key in keys], hist[:0]
    order = np.lexsort(keys[::-1])
    keys = [key[order] for key in keys]
    hist = hist[order]
    change = np.zeros(len(hist), dtype=bool)
    for key in keys:
        change[1:] |= key[1:] != key[:-1]
    starts = np.concatenate(([0], np.flatnonzero(change)))
    return [key[starts] for key in keys], np.add.reduceat(hist, starts, axis=0)


def tracklet_window_cells(scene: Scene, samples: PairSamples, window: int) -> Histograms:
    """Histogram cells per (tracklet a, tracklet b, window index)."""
    tracklet_a = np.empty(len(samples), dtype=np.int64)
    tracklet_b = np.empty(len(samples), dtype=np.int64)
    for camera in scene.cameras:
        mask = samples.cam_a == camera.index
        tracklet_a[mask] = camera.tracklet[samples.row_a[mask]]
        mask = samples.cam_b == camera.index
        tracklet_b[mask] = camera.tracklet[samples.row_b[mask]]
    valid = (tracklet_a >= 0) & (tracklet_b >= 0)
    bins = np.where(samples.plausible, np.searchsorted(EDGES, samples.u, side="left"), len(EDGES)).astype(np.int64)
    low = np.minimum(tracklet_a, tracklet_b)[valid]
    high = np.maximum(tracklet_a, tracklet_b)[valid]
    win = (samples.t[valid] // window).astype(np.int64)
    bins = bins[valid]
    order = np.lexsort((win, high, low))
    low, high, win, bins = low[order], high[order], win[order], bins[order]
    change = np.concatenate(([True], (low[1:] != low[:-1]) | (high[1:] != high[:-1]) | (win[1:] != win[:-1]))) \
        if len(low) else np.empty(0, dtype=bool)
    group = np.cumsum(change) - 1
    width = len(EDGES) + 1
    groups = int(group[-1]) + 1 if len(group) else 0
    hist = np.bincount(group * width + bins, minlength=groups * width).reshape(groups, width).astype(np.int32)
    first = np.flatnonzero(change)
    return Histograms(low[first], high[first], win[first].astype(np.int32), hist)


def regroup(cells: Histograms, unit_a: np.ndarray, unit_b: np.ndarray) -> Histograms:
    """Pool cells by unit pair (ignoring windows)."""
    valid = (unit_a >= 0) & (unit_b >= 0) & (unit_a != unit_b)
    low = np.minimum(unit_a, unit_b)[valid].astype(np.int64)
    high = np.maximum(unit_a, unit_b)[valid].astype(np.int64)
    (low, high), hist = _group_sum([low, high], cells.hist[valid])
    return Histograms(low, high, np.zeros(len(low), np.int32), hist)


# --------------------------------------------------------------------------- atoms


@dataclass
class Atoms:
    """Tracklet pieces. ``cuts[tracklet]`` lists the window indices where a new piece starts."""
    tracklets: np.ndarray  # sorted tracklet ids
    cuts: dict[int, list[int]]
    first_atom: np.ndarray  # atom id of each tracklet's first piece
    count: int
    tracklet_of_atom: np.ndarray

    def atom_of(self, tracklet: np.ndarray, window: np.ndarray) -> np.ndarray:
        position = np.clip(np.searchsorted(self.tracklets, tracklet), 0, len(self.tracklets) - 1)
        known = self.tracklets[position] == tracklet
        atom = np.where(known, self.first_atom[position], -1).astype(np.int64)
        for tid, cuts in self.cuts.items():
            mask = tracklet == tid
            if mask.any():
                atom[mask] += np.searchsorted(np.asarray(cuts), window[mask], side="right")
        return atom


def make_atoms(tracklets: np.ndarray, cuts: dict[int, list[int]]) -> Atoms:
    tracklets = np.unique(tracklets)
    cuts = {int(t): sorted(set(int(w) for w in c)) for t, c in cuts.items() if len(c)}
    pieces = np.array([1 + len(cuts.get(int(t), ())) for t in tracklets], dtype=np.int64)
    first = np.concatenate(([0], np.cumsum(pieces)[:-1])) if len(pieces) else np.empty(0, np.int64)
    return Atoms(tracklets, cuts, first, int(pieces.sum()), np.repeat(tracklets, pieces))


def refine_cuts(cells: Histograms, atoms: Atoms, component: np.ndarray, *, min_samples: int = 4,
                support: float = 0.8, contradict: float = 0.5, run: int = 2) -> tuple[dict[int, list[int]], list[dict]]:
    """Find identity switches by consensus and return additional cuts per tracklet.

    For each atom and window, the atom's residuals are pooled against every *component*
    (all of that component's atoms together). A component supports the atom in a window if
    the pooled evidence meets the gate. If the supporting component changes from X to Y
    between two sustained runs of windows - and X and Y are different components - the
    atom changed person at that boundary: it is cut there. A run in which the atom's own
    component contradicts it (and nothing else supports it) is cut out as well.
    """
    atom_a = atoms.atom_of(cells.a, cells.window.astype(np.int64))
    atom_b = atoms.atom_of(cells.b, cells.window.astype(np.int64))
    valid = (atom_a >= 0) & (atom_b >= 0)
    atom_a, atom_b, window, hist = atom_a[valid], atom_b[valid], cells.window[valid].astype(np.int64), cells.hist[valid]
    # Each cell is evidence for both of its atoms against the other atom's component.
    atom = np.concatenate([atom_a, atom_b])
    other = np.concatenate([component[atom_b], component[atom_a]]).astype(np.int64)
    window = np.concatenate([window, window])
    hist = np.concatenate([hist, hist])
    (atom, window, other), hist = _group_sum([atom, window, other], hist)
    n = hist.sum(axis=1)
    state = np.zeros(len(n), dtype=np.int8)
    state[(n >= min_samples) & (frac(hist, BIN_2) >= support)] = 1
    state[(n >= min_samples) & (frac(hist, BIN_4) <= contradict)] = -1

    new_cuts: dict[int, set[int]] = {}
    events = []
    boundaries = np.flatnonzero(np.diff(atom)) + 1
    for chunk in np.split(np.arange(len(atom)), boundaries):
        if not len(chunk):
            continue
        this = int(atom[chunk[0]])
        own = int(component[this])
        # Per window: the unique supporting component, or -1 (none / several).
        per_window: dict[int, int] = {}
        own_contradicted: set[int] = set()
        for w in np.unique(window[chunk]):
            rows = chunk[window[chunk] == w]
            supporters = other[rows][state[rows] == 1]
            per_window[int(w)] = int(supporters[0]) if len(supporters) == 1 else -1
            if ((other[rows] == own) & (state[rows] == -1)).any() and not len(supporters):
                own_contradicted.add(int(w))
        windows = sorted(per_window)
        # Runs of the same supporting component over consecutive informative windows.
        runs: list[list] = []  # [component, first window, last window, count]
        for w in windows:
            label = per_window[w]
            if label < 0:
                continue
            if runs and runs[-1][0] == label:
                runs[-1][2], runs[-1][3] = w, runs[-1][3] + 1
            else:
                runs.append([label, w, w, 1])
        sustained = [r for r in runs if r[3] >= run]
        tracklet = int(atoms.tracklet_of_atom[this])
        for before, after in zip(sustained, sustained[1:]):
            if before[0] != after[0]:
                boundary = int(after[1])
                # Cut after the last window of the old run and before the first of the new one:
                # the windows in between (where the switch happened) become their own piece
                # and stay unassigned rather than carrying the wrong person into either side.
                new_cuts.setdefault(tracklet, set()).update({int(before[2]) + 1, boundary})
                events.append({"tracklet": tracklet, "atom": this, "window": boundary, "kind": "switch",
                               "transition_from_window": int(before[2]) + 1,
                               "from_component": int(before[0]), "to_component": int(after[0])})
        # Sustained contradiction by the atom's own component: detach that stretch.
        bad = sorted(own_contradicted)
        start = None
        for index, w in enumerate(bad + [None]):
            if start is None:
                start, previous, count = w, w, 1
                continue
            if w is not None and w == previous + 1:
                previous, count = w, count + 1
                continue
            if count >= run:
                new_cuts.setdefault(tracklet, set()).update({int(start), int(previous) + 1})
                events.append({"tracklet": tracklet, "atom": this, "window": int(start), "kind": "detach",
                               "until_window": int(previous) + 1})
            start, previous, count = w, w, 1
    merged = {t: sorted(set(atoms.cuts.get(t, [])) | cuts) for t, cuts in new_cuts.items()}
    added = {t: c for t, c in merged.items() if c != atoms.cuts.get(t, [])}
    return added, events


# --------------------------------------------------------------------------- agglomeration


class Components:
    """Union-find over atoms with per-component, per-camera occupancy for conflict checks.

    Occupancy is kept as packed bitmaps over sync indices (one per component and camera), so
    "do these two components show the same camera at the same time" is one AND over a few
    kilobytes regardless of how many pieces the components have.
    """

    def __init__(self, count: int, camera: np.ndarray, occupancy: list[np.ndarray]):
        self.parent = np.arange(count)
        self.members: dict[int, list[int]] = {i: [i] for i in range(count)}
        self.camera = camera
        self.occupancy = occupancy  # per atom: sorted sync indices where it has a detection
        self.by_camera: dict[int, dict[int, list[int]]] = {i: {int(camera[i]): [i]} for i in range(count)}
        self.forbidden: dict[int, set[int]] = {}  # root -> roots it must never join
        self.duplicates: set[tuple[int, int]] = set()  # atom pairs a human confirmed as one person twice
        size = max((int(o[-1]) for o in occupancy if len(o)), default=0) + 1
        self._bytes = (size + 7) // 8
        self.atom_bits = [self._pack(o) for o in occupancy]
        # Component bitmaps start as views of their single atom's bitmap; copied on first merge.
        self.bits: dict[int, dict[int, np.ndarray]] = {i: {int(camera[i]): self.atom_bits[i]} for i in range(count)}

    def _pack(self, indices: np.ndarray) -> np.ndarray:
        bits = np.zeros(self._bytes, dtype=np.uint8)
        if len(indices):
            np.bitwise_or.at(bits, indices >> 3, (1 << (indices & 7)).astype(np.uint8))
        return bits

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return int(root)

    def forbid(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.forbidden.setdefault(rx, set()).add(ry)
            self.forbidden.setdefault(ry, set()).add(rx)

    def atoms_overlap(self, first: int, second: int) -> bool:
        """Do two atoms have detections at a common instant?"""
        return bool(np.bitwise_and(self.atom_bits[first], self.atom_bits[second]).any())

    def conflict(self, x: int, y: int) -> str | None:
        """Why the components of x and y cannot merge, or None."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return None
        if ry in self.forbidden.get(rx, ()):
            return "contradicted"
        bits_x, bits_y = self.bits[rx], self.bits[ry]
        if len(bits_y) < len(bits_x):
            bits_x, bits_y = bits_y, bits_x
        for camera, first in bits_x.items():
            second = bits_y.get(camera)
            if second is None or not np.bitwise_and(first, second).any():
                continue
            if not self.duplicates:
                return "same_camera_overlap"
            # Human-confirmed duplicates may coexist in one camera: check piece by piece.
            for a in self.by_camera[rx].get(camera, ()):
                for b in self.by_camera[ry].get(camera, ()):
                    if (min(a, b), max(a, b)) not in self.duplicates and self.atoms_overlap(a, b):
                        return "same_camera_overlap"
        return None

    def union(self, x: int, y: int) -> int:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return rx
        if len(self.members[rx]) < len(self.members[ry]):
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.members[rx].extend(self.members.pop(ry))
        for camera, atoms in self.by_camera.pop(ry).items():
            self.by_camera[rx].setdefault(camera, []).extend(atoms)
        target = self.bits[rx]
        for camera, bits in self.bits.pop(ry).items():
            target[camera] = np.bitwise_or(target[camera], bits) if camera in target else bits
        moved = self.forbidden.pop(ry, set())
        for other in moved:
            self.forbidden[other].discard(ry)
            self.forbidden[other].add(rx)
        self.forbidden.setdefault(rx, set()).update(moved)
        self.forbidden[rx].discard(rx)
        return rx

    def labels(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.parent))])


@dataclass
class Solution:
    labels: np.ndarray  # component root per atom
    accepted: list[dict]
    ambiguous: list[dict]  # supported edges that stayed undecided, with their competitors
    blocked: list[dict]  # supported edges refused by a conflict or by pooled evidence
    iterations: int


def agglomerate(edges: Histograms, components: Components, *, min_samples: int, support: float = 0.8,
                contradict: float = 0.5, margin: float = 0.25, strong: float = 0.8, corroborate: float = 0.6,
                must_link: list[tuple[int, int]] = (), cannot_link: list[tuple[int, int]] = ()) -> Solution:
    """Grow components from supported, unambiguous edges until nothing changes.

    ``edges`` are pooled atom-pair histograms. Human constraints come first: ``cannot_link``
    pairs are forbidden before anything is merged, ``must_link`` pairs are merged unless
    that would violate a cannot-link (then they are reported as blocked, never forced).
    """
    n = edges.n
    f1, f2, f4 = edges.frac(BIN_1), edges.frac(BIN_2), edges.frac(BIN_4)
    mean_u = edges.mean_u
    enough = n >= min_samples
    supported = enough & (f2 >= support)
    contradicted = enough & (f4 <= contradict)
    lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(edges.a, edges.b))}
    accepted, blocked = [], []
    for x, y in cannot_link:
        components.forbid(x, y)
    for x, y in must_link:
        reason = components.conflict(x, y)
        if reason:
            blocked.append({"a": int(x), "b": int(y), "reason": f"human same vs {reason}", "source": "human"})
        else:
            components.union(x, y)
            accepted.append({"a": int(x), "b": int(y), "source": "human"})
    for index in np.flatnonzero(contradicted):
        components.forbid(int(edges.a[index]), int(edges.b[index]))

    # Pooled histogram between components, kept incrementally: neighbours[root][other root].
    neighbours: dict[int, dict[int, np.ndarray]] = {}
    for index in range(len(edges)):
        a, b = components.find(int(edges.a[index])), components.find(int(edges.b[index]))
        if a == b:
            continue
        hist = edges.hist[index].astype(np.int64)
        for first, second in ((a, b), (b, a)):
            row = neighbours.setdefault(first, {})
            row[second] = row[second] + hist if second in row else hist

    def pooled(x: int, y: int) -> np.ndarray:
        """Pooled histogram over every cross pair of the two components."""
        found = neighbours.get(components.find(x), {}).get(components.find(y))
        return found if found is not None else np.zeros(edges.hist.shape[1], dtype=np.int64)

    def merge(x: int, y: int) -> None:
        """Union two components and fold their neighbour histograms together."""
        rx, ry = components.find(x), components.find(y)
        root = components.union(x, y)
        gone = ry if root == rx else rx
        kept_row, gone_row = neighbours.setdefault(root, {}), neighbours.pop(gone, {})
        kept_row.pop(gone, None)
        gone_row.pop(root, None)
        for other, hist in gone_row.items():
            kept_row[other] = kept_row[other] + hist if other in kept_row else hist
            back = neighbours.setdefault(other, {})
            back.pop(gone, None)
            back[root] = kept_row[other]

    candidates = np.flatnonzero(supported)
    candidates = candidates[np.lexsort((mean_u[candidates], -f1[candidates]))]  # strongest first
    partners: dict[int, dict[int, list[int]]] = {}
    for index in candidates:
        a, b = int(edges.a[index]), int(edges.b[index])
        partners.setdefault(a, {}).setdefault(int(components.camera[b]), []).append(int(index))
        partners.setdefault(b, {}).setdefault(int(components.camera[a]), []).append(int(index))

    def other(index: int, atom: int) -> int:
        return int(edges.b[index]) if int(edges.a[index]) == atom else int(edges.a[index])

    supporters: dict[int, set[int]] = {}
    firm: dict[int, set[int]] = {}  # supporters that also meet the accept gate most of the time
    for index in candidates:
        x, y = int(edges.a[index]), int(edges.b[index])
        supporters.setdefault(x, set()).add(y)
        supporters.setdefault(y, set()).add(x)
        if f1[index] >= corroborate:
            firm.setdefault(x, set()).add(y)
            firm.setdefault(y, set()).add(x)
    corroboration: dict[int, bool] = {}

    def corroborated(index: int) -> bool:
        """Is the edge confirmed by a third view: an atom of another camera supported by both ends?

        Two-view agreement can be an accident of the epipolar geometry; the same accident in a
        third camera at the same time is not plausible. Only firm support counts (most
        samples within the accept gate): a loose review-gate agreement between two people
        standing close together must not corroborate anything. (Evaluated on the initial atoms.)
        """
        if index not in corroboration:
            x, y = int(edges.a[index]), int(edges.b[index])
            cameras = {int(components.camera[x]), int(components.camera[y])}
            corroboration[index] = any(
                int(components.camera[z]) not in cameras
                and components.atoms_overlap(z, x) and components.atoms_overlap(z, y)
                for z in firm.get(x, set()) & firm.get(y, set()))
        return corroboration[index]

    def rivals(atom: int, partner: int, index: int) -> list[int]:
        """Supported edges of ``atom`` that taking ``index`` would rule out, and that are not clearly worse.

        A rival is any other supported neighbour of the atom, in any camera, whose component
        cannot coexist with the partner's (contradicted by it, or visible in the same camera
        at the same time). Accepting the edge would silently discard that alternative, so
        the edge stays open until the alternative is eliminated by other evidence or by a
        human - unless this edge is corroborated by a third view, the rival's is not, and the
        rival has its own counterpart in the atom's camera.
        This covers the two-view trap: P seen by A and C, Q by B and C, with P in A and Q
        in B on one epipolar line. The ghost edge A-B and the true edges rule each other out,
        none is confirmed by a third view, and none is taken on geometry alone.
        """
        result = []
        for rival_edges in partners.get(atom, {}).values():
            for rival_index in rival_edges:
                if rival_index == index:
                    continue
                rival = other(rival_index, atom)
                if components.find(rival) in (components.find(partner), components.find(atom)):
                    continue
                if components.conflict(atom, rival) is not None:
                    continue  # eliminated: the rival's component can no longer contain this atom
                if components.conflict(rival, partner) is None:
                    continue  # compatible with the partner: all three may be one person
                # Compare the two hypotheses on everything known so far: the atom's whole
                # component against the partner's and against the rival's. A ghost that
                # looks perfect for one camera pair is exposed once the component has
                # other views (this is why the comparison improves as components grow).
                mine, theirs = pooled(atom, partner), pooled(atom, rival)
                if not mine.sum() or not theirs.sum():
                    mine, theirs = edges.hist[index], edges.hist[rival_index]
                mine_f1, mine_f2, mine_mean = frac(mine, BIN_1), frac(mine, BIN_2), mean_of(mine)
                theirs_f2, theirs_mean = frac(theirs, BIN_2), mean_of(theirs)
                clearly_worse = theirs_f2 <= mine_f2 - margin or (
                    mine_f1 >= strong and theirs_mean >= 2.0 * mine_mean + 0.25)
                if clearly_worse:
                    continue
                # A third view confirms this edge and not the rival's - but that alone is not
                # enough: when the atom's ray passes through two people, the "ghost" is
                # confirmed by every view of the other person. The rival is only dismissed
                # if it has a home of its own: another detection in the atom's camera, at
                # the same time, that the rival is consistent with (its own person's view).
                if corroborated(index) and not corroborated(rival_index) and any(
                        int(components.camera[z]) == int(components.camera[atom]) and z != atom
                        and components.atoms_overlap(z, atom) for z in supporters.get(rival, ())):
                    continue
                result.append(rival_index)
        return result

    iterations, pending = 0, [int(i) for i in candidates]
    ambiguous_now: dict[int, list[int]] = {}
    while True:
        iterations += 1
        merged, still = 0, []
        ambiguous_now = {}
        for index in pending:
            a, b = int(edges.a[index]), int(edges.b[index])
            if components.find(a) == components.find(b):
                continue
            reason = components.conflict(a, b)
            if reason:
                blocked.append({"a": a, "b": b, "reason": reason, "f2": float(f2[index]), "n": int(n[index])})
                continue
            total = pooled(a, b)
            if frac(total, BIN_2) < support:
                # Other views of the two components disagree with this edge: keep it open; it may
                # be re-tested after the components change, and is reported if it never passes.
                still.append(index)
                ambiguous_now[index] = []
                continue
            competing = rivals(a, b, index) + rivals(b, a, index)
            if competing:
                ambiguous_now[index] = competing
                still.append(index)
                continue
            merge(a, b)
            accepted.append({"a": a, "b": b, "f1": float(f1[index]), "f2": float(f2[index]), "n": int(n[index]),
                             "pooled_n": int(total.sum()), "pooled_f2": float(frac(total, BIN_2)),
                             "strong": bool(f1[index] >= strong), "iteration": iterations, "source": "machine"})
            merged += 1
        pending = still
        if not merged or not pending:
            break
    ambiguous = []
    for index, competing in ambiguous_now.items():
        a, b = int(edges.a[index]), int(edges.b[index])
        if components.find(a) == components.find(b) or components.conflict(a, b):
            continue
        total = pooled(a, b)
        ambiguous.append({"a": a, "b": b, "f2": float(f2[index]), "f1": float(f1[index]), "n": int(n[index]),
                          "pooled_f2": float(frac(total, BIN_2)), "pooled_n": int(total.sum()),
                          "reason": "rivals" if competing else "pooled evidence below support",
                          "rivals": [{"atom": other(r, a) if a in (int(edges.a[r]), int(edges.b[r])) else other(r, b),
                                      "f2": float(f2[r]), "n": int(n[r])} for r in competing]})
    return Solution(components.labels(), accepted, ambiguous, blocked, iterations)
