"""Did two tracked people exchange identities when they came close?

The association backbone is a detection-level 3D tracker (Autofit) cross-checked by two-view
geometry. Both are geometric, so neither can see the one error geometry allows: two people
come close and each track continues on the other person (a *swap*), or one track continues
on the other person while that person's own track ends (a *takeover*).

Where can that have happened? A track is a sequence of measured 3D positions. While two
people are measured continuously and stay apart, exchanging their labels would make both
tracks jump by their separation between two frames. So for every pair of identities and
every step between consecutive instants at which both are measured, this module compares

    stay = |A2 - A1| + |B2 - B1|        swap = |A2 - B1| + |B2 - A1|        (metres)

on short medians either side of the step:

* ``swap - stay`` clearly positive: the step is unambiguous - nothing to check;
* clearly negative: the two tracks *exchanged places* between frames - flagged;
* near zero (the people are within reach of each other, or one was unmeasured for a
  while): a *crossing* - continuity cannot tell who is who afterwards.

Crossings are tested with evidence that does not come from the 3D tracker:

* ``heading``    - do both keep their velocity through the crossing, or each take the other's;
* ``body``       - limb lengths and torso size measured in 3D before and after;
* ``tracklet``   - single-camera BoT-SORT tracklets that span the crossing in a camera where
  the two people stay apart in the image: a 2D tracker has no reason to fail there, so a
  tracklet that changes identity contradicts the 3D tracker;
* ``appearance`` - ReID embeddings before and after, when the workspace has them.

A crossing is ``flagged`` when two cues say swap, or one does and none says stay;
``verified`` when at least two cues say stay and none says swap; ``unverified`` otherwise.
Flagged and unverified crossings become review questions; nothing is relabelled here.

Two identities that sit on the same spot for their whole common life are one person tracked
twice (a duplicate-pose problem with its own unknown kind), not a swap risk; they are counted
and skipped. A track that *ends* within reach of another that continues is checked like a
crossing, one-sided (the survivor may now be following the person whose track ended).

Thresholds of the body and appearance cues come from an injection test on the golden
recording (``experiments/switches``); all are provisional for other rooms. "Verified" means
"these two tracks did not exchange people", not "every detection is on the right track".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene import Scene

INCH = 0.0254
FPS = 10
BONES = {"shoulder": [(5, 6)], "hip": [(11, 12)], "upper_arm": [(5, 7), (6, 8)], "forearm": [(7, 9), (8, 10)],
         "thigh": [(11, 13), (12, 14)], "shin": [(13, 15), (14, 16)]}
FEATURES = list(BONES) + ["torso", "leg_torso"]


def bone_features(xyz: np.ndarray) -> np.ndarray:
    """(n, 17, 3) inches -> (n, 8) lengths (NaN where a joint is unmeasured)."""
    out = np.full((len(xyz), len(FEATURES)), np.nan, np.float32)
    with np.errstate(all="ignore"):
        for k, pairs in enumerate(BONES.values()):
            out[:, k] = np.nanmean(np.stack([np.linalg.norm(xyz[:, a] - xyz[:, b], axis=1) for a, b in pairs]), axis=0)
        shoulders, hips = xyz[:, [5, 6]].mean(axis=1), xyz[:, [11, 12]].mean(axis=1)
        out[:, len(BONES)] = np.linalg.norm(shoulders - hips, axis=1)
        out[:, len(BONES) + 1] = out[:, len(BONES)] + out[:, 4] + out[:, 5]
    return out


def _spans(ident: np.ndarray) -> dict[int, tuple[int, int]]:
    if not len(ident):
        return {}
    starts = np.flatnonzero(np.concatenate(([True], ident[1:] != ident[:-1])))
    ends = np.concatenate((starts[1:], [len(ident)]))
    return {int(ident[s]): (int(s), int(e)) for s, e in zip(starts, ends)}


@dataclass
class Index:
    """Per-identity access to 3D track entries and to detection rows, both sorted by time."""
    scene: Scene
    tr_ident: np.ndarray
    tr_t: np.ndarray
    tr_root: np.ndarray
    tr_xyz: np.ndarray
    tr_span: dict
    r_cam: np.ndarray
    r_row: np.ndarray
    r_t: np.ndarray
    r_ident: np.ndarray
    r_span: dict
    sigma: np.ndarray
    window: int

    @staticmethod
    def build(scene: Scene, row_identity: list[np.ndarray], tr_identity: np.ndarray, tr_t: np.ndarray,
              tr_root: np.ndarray, tr_xyz: np.ndarray, *, window: int = 100) -> "Index":
        ok = (tr_identity >= 0) & np.isfinite(tr_root).all(axis=1)
        ident, t, root, xyz = tr_identity[ok], tr_t[ok], tr_root[ok], tr_xyz[ok]
        order = np.lexsort((t, ident))
        ident, t, root, xyz = ident[order], t[order], root[order], xyz[order]
        # one entry per (identity, t): an identity made of two components at one instant keeps the first
        first = np.concatenate(([True], (ident[1:] != ident[:-1]) | (t[1:] != t[:-1]))) if len(t) else np.zeros(0, bool)
        ident, t, root, xyz = ident[first], t[first], root[first], xyz[first]
        cams, rows, times, who = [], [], [], []
        for camera in scene.cameras:
            use = np.flatnonzero((row_identity[camera.index] >= 0) & (camera.t >= 0))
            cams.append(np.full(len(use), camera.index))
            rows.append(use)
            times.append(camera.t[use])
            who.append(row_identity[camera.index][use])
        cams, rows, times, who = (np.concatenate(x) if x else np.zeros(0, np.int64) for x in (cams, rows, times, who))
        order = np.lexsort((times, who))
        index = Index(scene, ident, t, root, xyz, _spans(ident), cams[order], rows[order], times[order], who[order],
                      {}, np.ones(len(FEATURES), np.float32), window)
        index.r_span = _spans(index.r_ident)
        index.sigma = index._bone_noise()
        return index

    def _bone_noise(self) -> np.ndarray:
        """Within-identity spread of each bone feature's window medians: the unit of the body cue."""
        values: list[dict[int, list[float]]] = [{} for _ in FEATURES]
        for who, (s, e) in self.tr_span.items():
            t, xyz = self.tr_t[s:e], self.tr_xyz[s:e]
            for start in range(int(t[0]), int(t[-1]), self.window):
                a, b = np.searchsorted(t, start), np.searchsorted(t, start + self.window)
                if b - a < 15:
                    continue
                with np.errstate(all="ignore"):
                    median = np.nanmedian(bone_features(xyz[a:b]), axis=0)
                for k, value in enumerate(median):
                    if np.isfinite(value):
                        values[k].setdefault(who, []).append(float(value))
        sigma = np.ones(len(FEATURES), np.float32)
        for k in range(len(FEATURES)):
            residuals = [np.array(v) - np.median(v) for v in values[k].values() if len(v) >= 3]
            if residuals:
                sigma[k] = max(1.4826 * float(np.median(np.abs(np.concatenate(residuals)))), 0.05)
        return sigma

    def track(self, who: int, lo: int, hi: int):
        if who not in self.tr_span:
            return np.zeros(0, np.int64), np.zeros((0, 3), np.float32), np.zeros((0, 17, 3), np.float32)
        s, e = self.tr_span[who]
        t = self.tr_t[s:e]
        a, b = s + np.searchsorted(t, lo), s + np.searchsorted(t, hi, side="right")
        return self.tr_t[a:b], self.tr_root[a:b], self.tr_xyz[a:b]

    def rows(self, who: int, lo: int, hi: int):
        if who not in self.r_span:
            empty = np.zeros(0, np.int64)
            return empty, empty, empty
        s, e = self.r_span[who]
        t = self.r_t[s:e]
        a, b = s + np.searchsorted(t, lo), s + np.searchsorted(t, hi, side="right")
        return self.r_cam[a:b], self.r_row[a:b], self.r_t[a:b]


# --------------------------------------------------------------------------- where a swap is possible


def candidate_pairs(index: Index, radius_m: float) -> list[tuple[int, int]]:
    """Pairs of identities measured within ``radius_m`` of each other at some instant."""
    order = np.argsort(index.tr_t, kind="stable")
    ident, t, root = index.tr_ident[order], index.tr_t[order], index.tr_root[order]
    if not len(t):
        return []
    starts = np.flatnonzero(np.concatenate(([True], t[1:] != t[:-1])))
    ends = np.concatenate((starts[1:], [len(t)]))
    found = set()
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        d = np.linalg.norm(root[s:e, None, :] - root[None, s:e, :], axis=-1) * INCH
        i, j = np.nonzero(np.triu(d < radius_m, 1))
        for a, b in zip(ident[s:e][i], ident[s:e][j]):
            if a != b:
                found.add((int(min(a, b)), int(max(a, b))))
    return sorted(found)


def _side_medians(root: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """For every index i: median of root[i-k+1 .. i] and of root[i .. i+k-1] (shorter at the ends)."""
    n = len(root)
    back, ahead = np.empty_like(root), np.empty_like(root)
    for i in range(n):
        back[i] = np.median(root[max(0, i - k + 1):i + 1], axis=0)
        ahead[i] = np.median(root[i:min(n, i + k)], axis=0)
    return back, ahead


def pair_steps(index: Index, a: int, b: int, recipe: dict) -> dict:
    """Classify every step between consecutive instants at which both identities are measured.

    Returns {"duplicate": bool, "median_distance_m": float, "crossings": [(s, e, kind, margin_m, distance_m)]}
    where kind is "exchange" (the tracks swapped places) or "crossing" (continuity cannot decide).
    """
    sa, ea = index.tr_span[a]
    sb, eb = index.tr_span[b]
    common, ia, ib = np.intersect1d(index.tr_t[sa:ea], index.tr_t[sb:eb], return_indices=True)
    if len(common) < 2:
        return {"duplicate": False, "median_distance_m": None, "crossings": []}
    ra, rb = index.tr_root[sa:ea][ia].astype(np.float64), index.tr_root[sb:eb][ib].astype(np.float64)
    separation = np.linalg.norm(ra - rb, axis=1) * INCH
    median_distance = float(np.median(separation))
    if median_distance < recipe["duplicate_m"]:
        return {"duplicate": True, "median_distance_m": round(median_distance, 3), "crossings": []}
    k = int(recipe["median_frames"])
    back_a, ahead_a = _side_medians(ra, k)
    back_b, ahead_b = _side_medians(rb, k)
    # step i: common[i] -> common[i + 1]
    gap = np.diff(common)
    stay = np.linalg.norm(ahead_a[1:] - back_a[:-1], axis=1) + np.linalg.norm(ahead_b[1:] - back_b[:-1], axis=1)
    swap = np.linalg.norm(ahead_a[1:] - back_b[:-1], axis=1) + np.linalg.norm(ahead_b[1:] - back_a[:-1], axis=1)
    margin = (swap - stay) * INCH
    near = np.minimum(separation[:-1], separation[1:]) < recipe["radius_m"]
    # A longer time without a common measurement lets people move: the bar rises with the gap.
    needed = recipe["margin_m"] + recipe["margin_per_second_m"] * np.maximum(gap - 1, 0) / FPS
    exchange = near & (margin < -recipe["exchange_m"]) & (gap <= recipe["step_gap_frames"])
    unclear = near & ~exchange & (margin < needed) & (gap <= recipe["max_gap_frames"])
    marked = np.flatnonzero(exchange | unclear)
    crossings = []
    start = 0
    while start < len(marked):
        stop = start
        while stop + 1 < len(marked) and common[marked[stop + 1]] - common[marked[stop] + 1] <= recipe["merge_frames"]:
            stop += 1
        steps = marked[start:stop + 1]
        kind = "exchange" if exchange[steps].any() else "crossing"
        crossings.append((int(common[steps[0]]), int(common[steps[-1] + 1]), kind,
                          round(float(margin[steps].min()), 3),
                          round(float(min(separation[steps].min(), separation[steps + 1].min())), 3)))
        start = stop + 1
    return {"duplicate": False, "median_distance_m": round(median_distance, 3), "crossings": crossings}


def endings_near(index: Index, total: int, recipe: dict, duplicates: set[tuple[int, int]],
                 min_frames: int = 10) -> tuple[list[dict], int]:
    """A track that ends within reach of another identity that continues (the takeover pattern).

    The survivor must be an established, separate track: measured for ``established_frames`` in the
    window before the ending. A survivor born just before, at the same spot, is the same person
    tracked again (a hand-off with a short overlap) - an identity-link matter, counted here and not
    asked as a swap.
    """
    handoffs = 0
    order = np.argsort(index.tr_t, kind="stable")
    ident, t, root = index.tr_ident[order], index.tr_t[order], index.tr_root[order]
    first_at = np.searchsorted(t, np.arange(total + 2))
    events = []
    for who, (s, e) in index.tr_span.items():
        last = int(index.tr_t[e - 1])
        if e - s < min_frames or last >= total - 2 * FPS:
            continue
        a, b = first_at[max(0, last - 3)], first_at[min(total, last + 4)]
        others = ident[a:b] != who
        if not others.any():
            continue
        d = np.linalg.norm(root[a:b][others] - index.tr_root[e - 1], axis=1) * INCH
        nearest = int(np.argmin(d))
        survivor = int(ident[a:b][others][nearest])
        if (min(who, survivor), max(who, survivor)) in duplicates:
            continue
        span = index.tr_span[survivor]
        if d[nearest] < recipe["ending_radius_m"] and span[1] - span[0] >= min_frames \
                and int(index.tr_t[span[1] - 1]) > last + FPS:
            history, _, _ = index.track(survivor, last - index.window, last - 1)
            if len(history) < recipe["established_frames"]:
                handoffs += 1
                continue
            events.append({"kind": "ending", "a": survivor, "b": int(who), "t": (last, last),
                           "margin_m": None, "min_distance_m": round(float(d[nearest]), 3)})
    return events, handoffs


# --------------------------------------------------------------------------- cues


def cue_heading(index: Index, a: int, b: int, s: int, e: int, *, frames: int = 15, min_frames: int = 6,
                min_difference: float = 0.5) -> float:
    """m/s: velocity change if each keeps its own track minus if they exchanged. Positive: swap.
    NaN unless the two were moving differently before (``min_difference`` m/s) - two people who
    stand, or walk together, carry no heading evidence."""
    velocity = {}
    for name, who in (("a", a), ("b", b)):
        for side, (lo, hi) in (("before", (s - frames, s - 1)), ("after", (e + 1, e + frames))):
            t, root, _ = index.track(who, lo, hi)
            if len(t) >= min_frames and t[-1] > t[0]:
                velocity[(name, side)] = np.polyfit((t - t[0]) / FPS, root.astype(np.float64) * INCH, 1)[0]
    if ("a", "before") not in velocity or ("b", "before") not in velocity:
        return float("nan")
    if np.linalg.norm(velocity[("a", "before")] - velocity[("b", "before")]) < min_difference:
        return float("nan")
    total, count = 0.0, 0
    for name, other in (("a", "b"), ("b", "a")):
        if (name, "after") in velocity:
            total += float(np.linalg.norm(velocity[(name, "after")] - velocity[(name, "before")])
                           - np.linalg.norm(velocity[(name, "after")] - velocity[(other, "before")]))
            count += 1
    return total / count if count else float("nan")


def _signature(index: Index, who: int, lo: int, hi: int, min_frames: int = 8):
    t, _, xyz = index.track(who, lo, hi)
    if len(t) < min_frames:
        return None
    with np.errstate(all="ignore"):
        median = np.nanmedian(bone_features(xyz), axis=0)
    return median if np.isfinite(median).sum() >= 5 else None


def cue_body(index: Index, a: int, b: int, s: int, e: int) -> float:
    """Bone-signature distance (in units of within-person noise): stay minus swap."""
    signature = {}
    for name, who in (("a", a), ("b", b)):
        signature[(name, "before")] = _signature(index, who, s - index.window, s - 1)
        signature[(name, "after")] = _signature(index, who, e + 1, e + index.window)

    def distance(u, v) -> float:
        if u is None or v is None:
            return float("nan")
        z = (u - v) / index.sigma
        z = z[np.isfinite(z)]
        return float(np.sqrt(np.mean(z ** 2))) if len(z) >= 4 else float("nan")

    total, count = 0.0, 0
    for name, other in (("a", "b"), ("b", "a")):
        stay = distance(signature[(name, "before")], signature[(name, "after")])
        swap = distance(signature[(other, "before")], signature[(name, "after")])
        if np.isfinite(stay) and np.isfinite(swap):
            total += stay - swap
            count += 1
    return total / count if count else float("nan")


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    return float(width * height / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - width * height))


def cue_tracklet(index: Index, a: int, b: int, s: int, e: int, *, separated_iou: float = 0.05) -> dict:
    """2D tracklets with detections of {a, b} both before ``s`` and after ``e``: do they keep one identity?
    Counted only in cameras where the two people's boxes never overlap around the moment."""
    out = {"contradict": 0, "agree": 0, "cameras_separated": 0, "contradicting": []}
    data = {who: index.rows(who, s - index.window, e + index.window) for who in (a, b)}
    for camera in index.scene.cameras:
        rows, who, when = [], [], []
        for label, (cams, rws, ts) in data.items():
            mine = cams == camera.index
            rows.append(rws[mine])
            when.append(ts[mine])
            who.append(np.full(int(mine.sum()), label))
        rows, who, when = np.concatenate(rows), np.concatenate(who), np.concatenate(when)
        if not len(rows):
            continue
        tracklet = camera.tracklet[rows]
        if not (tracklet >= 0).any():
            continue
        near = (when >= s - 10) & (when <= e + 10)
        rows_a, rows_b = rows[near & (who == a)], rows[near & (who == b)]
        t_a, t_b = when[near & (who == a)], when[near & (who == b)]
        separated = True
        position_b = {int(t): int(r) for t, r in zip(t_b, rows_b)}
        boxes = camera.detections.boxes
        for t, row in zip(t_a, rows_a):
            other = position_b.get(int(t))
            if other is not None and _iou(boxes[row, :4], boxes[other, :4]) > separated_iou:
                separated = False
                break
        if not separated:
            continue
        out["cameras_separated"] += 1
        for number in np.unique(tracklet[tracklet >= 0]):
            mine = tracklet == number
            before, after = mine & (when < s), mine & (when > e)
            if before.sum() < 2 or after.sum() < 2:
                continue
            majority_before = a if (who[before] == a).sum() >= (who[before] == b).sum() else b
            majority_after = a if (who[after] == a).sum() >= (who[after] == b).sum() else b
            if majority_before == majority_after:
                out["agree"] += 1
            else:
                out["contradict"] += 1
                out["contradicting"].append({"camera": camera.camera, "tracklet": int(number)})
    return out


def cue_appearance(index: Index, embeddings, a: int, b: int, s: int, e: int) -> float:
    """Cosine similarity of mean embeddings: swap minus stay. ``embeddings[camera index] = (row -> slot, vectors)``."""
    if embeddings is None:
        return float("nan")

    def mean(who: int, lo: int, hi: int):
        cams, rows, _ = index.rows(who, lo, hi)
        vectors = []
        for camera, row in zip(cams, rows):
            slot_of, table = embeddings[int(camera)]
            slot = slot_of[row] if row < len(slot_of) else -1
            if slot >= 0:
                vectors.append(table[slot])
        if len(vectors) < 2:
            return None
        total = np.sum(vectors, axis=0)
        norm = np.linalg.norm(total)
        return total / norm if norm > 1e-9 else None

    windows = {(name, side): mean(who, *(span))
               for name, who in (("a", a), ("b", b))
               for side, span in (("before", (s - index.window, s - 1)), ("after", (e + 1, e + index.window)))}
    total, count = 0.0, 0
    for name, other in (("a", "b"), ("b", "a")):
        after, mine, theirs = windows[(name, "after")], windows[(name, "before")], windows[(other, "before")]
        if after is not None and mine is not None and theirs is not None:
            total += float(theirs @ after - mine @ after)
            count += 1
    return total / count if count else float("nan")


# --------------------------------------------------------------------------- verdict


def judge(cues: dict, recipe: dict, *, exchange: bool = False) -> dict:
    """``cues``: heading, body, appearance (floats, NaN = unavailable), tracklet {contradict, agree}."""
    flag, stay = recipe["flag"], recipe["stay"]
    causes, stays, available = [], [], 0
    for name in ("heading", "body", "appearance"):
        value = cues.get(name, float("nan"))
        if value is None or not np.isfinite(value):
            continue
        available += 1
        if value > flag[name]:
            causes.append(name)
        elif value < stay[name]:
            stays.append(name)
    tracklet = cues.get("tracklet") or {}
    contradict, agree = int(tracklet.get("contradict", 0)), int(tracklet.get("agree", 0))
    if contradict + agree > 0:
        available += 1
        if contradict >= recipe["tracklet_contradictions"] and contradict > agree:
            causes.append("tracklet")
        elif agree >= recipe["tracklet_agreements"] and contradict == 0:
            stays.append("tracklet")
    if exchange:
        causes = ["exchanged places"] + causes
    if exchange or len(causes) >= 2 or (causes and not stays):
        status = "flagged"
    elif len(stays) >= 2 and not causes:
        status = "verified"
    else:
        status = "unverified"
    return {"status": status, "swap_cues": causes, "stay_cues": stays, "cues_available": available}


def _human_verified(index: Index, who: int, s: int, e: int, same_rows: list[tuple[list, list]],
                    row_identity: list[np.ndarray]) -> bool:
    """A person said two detections are the same person, one before and one after this moment, and both carry
    this identity: the identity did not change hands here."""
    for first, second in same_rows:
        times = []
        for camera_index, row in list(first) + list(second):
            if row_identity[camera_index][row] == who:
                times.append(int(index.scene.cameras[camera_index].t[row]))
        if times and min(times) < s and max(times) > e:
            return True
    return False


def _keys(index: Index, who: int, lo: int, hi: int, *, nearest_to: int, count: int = 3,
          prefer: list[int] | None = None) -> list[str]:
    """Up to ``count`` detections of an identity in [lo, hi], one per camera, nearest to ``nearest_to``;
    cameras in ``prefer`` first, then larger boxes."""
    cams, rows, times = index.rows(who, lo, hi)
    options = []
    for camera in index.scene.cameras:
        mine = np.flatnonzero(cams == camera.index)
        if not len(mine):
            continue
        pick = mine[np.argmin(np.abs(times[mine] - nearest_to))]
        box = camera.detections.boxes[rows[pick]]
        rank = prefer.index(camera.index) if prefer and camera.index in prefer else 99
        options.append((rank, -float((box[2] - box[0]) * (box[3] - box[1])), camera.detections.key(int(rows[pick]))))
    return [key for _, _, key in sorted(options)[:count]]


def evaluate(index: Index, moment: dict, recipe: dict, embeddings=None) -> dict:
    """Cues and verdict for one moment {"kind", "a", "b", "t": (s, e)}."""
    a, b = moment["a"], moment["b"]
    s, e = moment["t"]
    tracklet = cue_tracklet(index, a, b, s, e)
    cues = {"heading": cue_heading(index, a, b, s, e), "body": cue_body(index, a, b, s, e),
            "appearance": cue_appearance(index, embeddings, a, b, s, e), "tracklet": tracklet}
    verdict = judge(cues, recipe, exchange=moment["kind"] == "exchange")
    return {"kind": moment["kind"], "a": a, "b": b, "t": [int(s), int(e)],
            "min_distance_m": moment.get("min_distance_m"), "continuity_margin_m": moment.get("margin_m"),
            "cues": {"heading_mps": _round(cues["heading"]), "body": _round(cues["body"]),
                     "appearance": _round(cues["appearance"]),
                     "tracklets_contradicting": tracklet["contradict"], "tracklets_agreeing": tracklet["agree"],
                     "cameras_separated": tracklet["cameras_separated"],
                     "contradicting_tracklets": tracklet["contradicting"]},
            **verdict}


def moments(index: Index, total: int, recipe: dict) -> tuple[list[dict], dict]:
    """Every moment at which two identities could have exchanged people, and what was skipped."""
    found, duplicates = [], set()
    pairs = candidate_pairs(index, recipe["radius_m"])
    for a, b in pairs:
        steps = pair_steps(index, a, b, recipe)
        if steps["duplicate"]:
            duplicates.add((a, b))
            continue
        for s, e, kind, margin, distance in steps["crossings"]:
            found.append({"kind": kind, "a": a, "b": b, "t": (s, e), "margin_m": margin, "min_distance_m": distance})
    endings, handoffs = endings_near(index, total, recipe, duplicates)
    return found + endings, {"pairs_within_radius": len(pairs), "duplicate_pairs_skipped": len(duplicates),
                             "handoff_overlaps_skipped": handoffs}


def check(scene: Scene, row_identity: list[np.ndarray], tr_identity: np.ndarray, tr_t: np.ndarray,
          tr_root: np.ndarray, tr_xyz: np.ndarray, recipe: dict, *, embeddings=None,
          same_rows: list[tuple[list, list]] | None = None) -> dict:
    """Find and test every moment at which two tracks could have exchanged people.

    ``scene`` must carry the single-camera tracker's own tracklets (not pieces cut by the 3D tracker).
    Returns {"entries": [...], "summary": {...}}; an entry has the two identities, the sync span, the cue
    values, the verdict and, unless verified, the detection keys a reviewer needs.
    """
    window = int(recipe["window_frames"])
    index = Index.build(scene, row_identity, tr_identity, tr_t, tr_root, tr_xyz, window=window)
    found, skipped = moments(index, scene.T, recipe)
    entries = []
    for moment in found:
        entry = evaluate(index, moment, recipe, embeddings)
        a, b = entry["a"], entry["b"]
        s, e = entry["t"]
        if entry["status"] != "verified" and entry["kind"] != "exchange" and same_rows and \
                _human_verified(index, a, s, e, same_rows, row_identity) and \
                (entry["kind"] == "ending" or _human_verified(index, b, s, e, same_rows, row_identity)):
            entry.update(status="verified", stay_cues=entry["stay_cues"] + ["human"])
        if entry["status"] != "verified":
            # Reviewer evidence: the reference is identity a just before; candidates are a and b just after.
            prefer = [index.scene.camera_index(c["camera"]) for c in entry["cues"]["contradicting_tracklets"]]
            entry["reference_keys"] = _keys(index, a, s - window, s - 1, nearest_to=s - 1, prefer=prefer)
            entry["after_a_keys"] = _keys(index, a, e + 1, e + window, nearest_to=e + 1, prefer=prefer)
            entry["after_b_keys"] = _keys(index, b, e + 1, e + window, nearest_to=e + 1, prefer=prefer)
            entry["before_b_keys"] = _keys(index, b, s - window, s - 1, nearest_to=s - 1, prefer=prefer)
        entries.append(entry)
    statuses = [entry["status"] for entry in entries]
    kinds = [entry["kind"] for entry in entries]
    summary = {**skipped, "moments": len(entries),
               **{kind: kinds.count(kind) for kind in ("crossing", "exchange", "ending")},
               **{status: statuses.count(status) for status in ("verified", "flagged", "unverified")},
               "appearance_available": embeddings is not None}
    return {"entries": entries, "summary": summary}


def _round(value) -> float | None:
    return round(float(value), 3) if value is not None and np.isfinite(value) else None
