"""Case builders and rule evaluation for the ReID experiments (all scored against the golden reference)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common import Data, reid

WIN = 100            # fragment window: 10 s of sync frames
MIN_FRAG = 5         # embedded labelled detections for a usable fragment / query
MIN_GAL = 3          # detections needed for a gallery descriptor to exist


# ---------------------------------------------------------------- descriptors
def det_weights(D: Data, mode: str) -> np.ndarray:
    return np.ones(len(D.t), np.float32) if mode == "none" else reid.quality_weights(D.box)


class Gallery:
    """Time-prefix sums of weighted embeddings per (person, camera) over labelled detections."""

    def __init__(self, D: Data, w: np.ndarray):
        self.D = D
        self.items: dict[tuple[int, int], tuple] = {}
        lab = np.flatnonzero(D.person >= 0)
        for p in np.unique(D.person[lab]):
            for c in range(len(D.cameras)):
                idx = lab[(D.person[lab] == p) & (D.cam[lab] == c)]
                if len(idx) == 0:
                    continue
                cum = np.concatenate([np.zeros((1, D.emb.shape[1]), np.float64),
                                      np.cumsum(D.emb[idx] * w[idx, None], axis=0, dtype=np.float64)])
                self.items[(int(p), c)] = (D.t[idx], cum, idx)
        self.people = sorted({p for p, _ in self.items})

    def sum_count(self, p: int, cams, lo: float, hi: float):
        """Sum of weighted embeddings and count of detections of person p in cameras ``cams`` with lo <= t < hi."""
        total, n = np.zeros(self.D.emb.shape[1]), 0
        for c in cams:
            it = self.items.get((p, c))
            if it is None:
                continue
            ts, cum, _ = it
            a, b = np.searchsorted(ts, lo, "left"), np.searchsorted(ts, hi, "left")
            total += cum[b] - cum[a]
            n += b - a
        return total, n

    def desc(self, p, cams, lo, hi, min_n=MIN_GAL):
        s, n = self.sum_count(p, cams, lo, hi)
        if n < min_n:
            return None
        norm = np.linalg.norm(s)
        return None if norm < 1e-9 else (s / norm)

    def last_seen(self, p, cams, hi):
        best = -1
        for c in cams:
            it = self.items.get((p, c))
            if it is None:
                continue
            k = np.searchsorted(it[0], hi, "left")
            if k:
                best = max(best, int(it[0][k - 1]))
        return best


# ---------------------------------------------------------------- fragments
@dataclass
class Frags:
    cam: np.ndarray
    trk: np.ndarray
    win: np.ndarray
    t0: np.ndarray
    t1: np.ndarray
    person: np.ndarray
    n: np.ndarray
    desc: np.ndarray
    first: np.ndarray   # index into Data of the first detection
    dropped_ambiguous: int = 0

    def __len__(self):
        return len(self.cam)


def build_fragments(D: Data, w: np.ndarray, win=WIN, min_n=MIN_FRAG) -> Frags:
    lab = np.flatnonzero((D.person >= 0) & (D.tracklet >= 0))
    key = D.tracklet[lab].astype(np.int64) * 100000 + D.t[lab] // win
    order = np.argsort(key, kind="stable")
    lab, key = lab[order], key[order]
    cuts = np.flatnonzero(np.diff(key)) + 1
    out = {k: [] for k in ("cam", "trk", "win", "t0", "t1", "person", "n", "desc", "first")}
    amb = 0
    for g in np.split(lab, cuts):
        if len(g) < min_n:
            continue
        people = np.unique(D.person[g])
        if len(people) != 1:
            amb += 1
            continue
        d = reid.aggregate(D.emb[g], w[g])
        if d is None:
            continue
        out["cam"].append(D.cam[g[0]]); out["trk"].append(D.tracklet[g[0]]); out["win"].append(D.t[g[0]] // win)
        out["t0"].append(D.t[g].min()); out["t1"].append(D.t[g].max()); out["person"].append(people[0])
        out["n"].append(len(g)); out["desc"].append(d); out["first"].append(g[np.argmin(D.t[g])])
    arr = {k: np.array(v) for k, v in out.items()}
    return Frags(arr["cam"], arr["trk"], arr["win"], arr["t0"], arr["t1"], arr["person"], arr["n"],
                 arr["desc"].astype(np.float32), arr["first"], amb)


# ---------------------------------------------------------------- cases
@dataclass
class Cases:
    name: str
    sim: np.ndarray        # (Q, G) similarity to each candidate, NaN = no such candidate
    true: np.ndarray       # (Q,) column of the correct candidate, -1 = correct answer is not a candidate
    mutual: np.ndarray     # (Q,) bool: the best candidate's best query is this query
    t: np.ndarray          # (Q,) time (sync index) used for the train/held-out split
    cam: np.ndarray        # (Q,) query camera
    meta: list = field(default_factory=list)   # per-case dict for the case listing
    cand_person: np.ndarray | None = None      # (Q, G) person ids of the candidates (for the listing)

    def __len__(self):
        return len(self.sim)


def _finish(name, sim, true, t, cam, meta, comp_groups, cand_person=None):
    """Mutual-best flag: the winning column's top similarity among the competing queries (comp_groups[q]) is q's."""
    Q = len(sim)
    filled = np.where(np.isfinite(sim), sim, -np.inf)
    best = filled.argmax(1) if sim.shape[1] else np.zeros(Q, int)
    mutual = np.zeros(Q, bool)
    for q in range(Q):
        comp = comp_groups[q]
        mutual[q] = bool(len(comp) == 0 or filled[q, best[q]] >= filled[comp, best[q]].max())
    return Cases(name, sim, np.asarray(true), mutual, np.asarray(t), np.asarray(cam), meta, cand_person)


def build_b1(D: Data, G: Gallery, *, scope="all", recency=None, qwin=WIN, name="B1") -> Cases:
    """Identity across a gap: first qwin of a tracklet vs per-person descriptors built only from t < tracklet start."""
    people = G.people
    cols = {p: i for i, p in enumerate(people)}
    lab_trk = np.flatnonzero(D.tracklet >= 0)
    order = np.argsort(D.tracklet[lab_trk], kind="stable")
    lab_trk = lab_trk[order]
    cuts = np.flatnonzero(np.diff(D.tracklet[lab_trk])) + 1
    sims, true, tt, cc, meta, starts = [], [], [], [], [], []
    for g in np.split(lab_trk, cuts):
        t0 = int(D.t[g].min())
        q = g[(D.t[g] < t0 + qwin) & (D.person[g] >= 0)]
        if len(q) < MIN_FRAG or len(np.unique(D.person[q])) != 1:
            continue
        person, cam = int(D.person[q[0]]), int(D.cam[q[0]])
        qd = reid.aggregate(D.emb[q], G_w[id(G)][q])
        cams = [cam] if scope == "cam" else range(len(D.cameras))
        lo = 0 if recency is None else t0 - recency
        row = np.full(len(people), np.nan)
        for p in people:
            d = G.desc(p, cams, lo, t0)
            if d is not None:
                row[cols[p]] = float(qd @ d)
        sims.append(row)
        true.append(cols[person] if np.isfinite(row[cols[person]]) else -1)
        tt.append(t0); cc.append(cam); starts.append(t0)
        last = G.last_seen(person, range(len(D.cameras)), t0)
        meta.append({"query": int(q[np.argmin(D.t[q])]), "tracklet": int(D.tracklet[q[0]]), "person": person, "t0": t0,
                     "cam": cam, "n": len(q), "gap": (t0 - last) if last >= 0 else None})
    sim = np.array(sims)
    starts = np.array(starts)
    comp = [np.flatnonzero((np.abs(starts - s) <= WIN) & (np.arange(len(starts)) != i)) for i, s in enumerate(starts)]
    c = _finish(name, sim, true, tt, cc, meta, comp)
    c.cand_person = np.tile(np.array(people), (len(sim), 1))
    return c


def build_b3(D: Data, G: Gallery, *, radius=600, name="B3", single=None) -> Cases:
    """Single-camera pieces (by reference: the person has no labelled detection in another camera within the
    tracklet's time span +-1 s): whole tracklet vs per-person descriptors of the OTHER detections within +-radius."""
    people = G.people
    cols = {p: i for i, p in enumerate(people)}
    lab = np.flatnonzero((D.tracklet >= 0) & (D.person >= 0))
    order = np.argsort(D.tracklet[lab], kind="stable")
    lab = lab[order]
    cuts = np.flatnonzero(np.diff(D.tracklet[lab])) + 1
    w = G_w[id(G)]
    sims, true, tt, cc, meta, mids = [], [], [], [], [], []
    for g in np.split(lab, cuts):
        if len(g) < MIN_FRAG or len(np.unique(D.person[g])) != 1:
            continue
        person, cam = int(D.person[g[0]]), int(D.cam[g[0]])
        t0, t1 = int(D.t[g].min()), int(D.t[g].max())
        others = [c for c in range(len(D.cameras)) if c != cam]
        if single is not None:
            if single[g].mean() < 0.8:
                continue  # the engine linked this piece to a multi-camera identity
        elif G.sum_count(person, others, t0 - 10, t1 + 11)[1] > 0:
            continue  # seen by another camera at the same time: not a single-camera piece
        qd = reid.aggregate(D.emb[g], w[g])
        mid = (t0 + t1) // 2
        own = (person, cam)
        row = np.full(len(people), np.nan)
        for p in people:
            s, n = G.sum_count(p, range(len(D.cameras)), mid - radius, mid + radius)
            if p == person:   # the tracklet itself is not part of its own gallery
                s = s - (D.emb[g] * w[g, None]).sum(0)
                n -= len(g)
            if n >= MIN_GAL and np.linalg.norm(s) > 1e-9:
                row[cols[p]] = float(qd @ (s / np.linalg.norm(s)))
        sims.append(row)
        true.append(cols[person] if np.isfinite(row[cols[person]]) else -1)
        tt.append(mid); cc.append(cam); mids.append(mid)
        meta.append({"query": int(g[0]), "tracklet": int(D.tracklet[g[0]]), "person": person, "t0": t0, "t1": t1,
                     "cam": cam, "n": len(g)})
    sim = np.array(sims)
    mids = np.array(mids)
    comp = [np.flatnonzero((np.abs(mids - s) <= WIN) & (np.arange(len(mids)) != i)) for i, s in enumerate(mids)]
    c = _finish(name, sim, true, tt, cc, meta, comp)
    c.cand_person = np.tile(np.array(people), (len(sim), 1))
    return c


def build_b2(D: Data, F: Frags, name="B2") -> Cases:
    """Rival tie-break: fragment A (camera X) vs all simultaneous fragments of camera Y (same 10 s window) with exactly
    one of them the same person and at least two candidates."""
    rows, true, tt, cc, meta, comp, cand_people = [], [], [], [], [], [], []
    cmax = 0
    for w in np.unique(F.win):
        in_w = np.flatnonzero(F.win == w)
        for x in np.unique(F.cam[in_w]):
            A = in_w[F.cam[in_w] == x]
            for y in np.unique(F.cam[in_w]):
                if y == x:
                    continue
                B = in_w[F.cam[in_w] == y]
                if len(B) < 2:
                    continue
                S = F.desc[A] @ F.desc[B].T
                colbest = S.argmax(0)
                for i, a in enumerate(A):
                    match = np.flatnonzero(F.person[B] == F.person[a])
                    if len(match) != 1:
                        continue
                    rows.append(S[i]); true.append(int(match[0])); tt.append(int(w) * WIN); cc.append(int(x))
                    cand_people.append(F.person[B])
                    cmax = max(cmax, len(B))
                    meta.append({"query": int(F.first[a]), "tracklet": int(F.trk[a]), "person": int(F.person[a]),
                                 "t0": int(F.t0[a]), "cam": int(x), "camB": int(y),
                                 "cand_first": [int(F.first[b]) for b in B], "cand_person": [int(F.person[b]) for b in B],
                                 "mutual": bool(colbest[S[i].argmax()] == i)})
    sim = np.full((len(rows), cmax), np.nan)
    cp = np.full((len(rows), cmax), -1)
    for i, r in enumerate(rows):
        sim[i, :len(r)] = r
        cp[i, :len(r)] = cand_people[i]
    mutual = np.array([m["mutual"] for m in meta])
    return Cases(name, sim, np.array(true), mutual, np.array(tt), np.array(cc), meta, cp)


G_w: dict[int, np.ndarray] = {}   # weights per Gallery (set by make_gallery)


def make_gallery(D: Data, weights="none") -> Gallery:
    w = det_weights(D, weights)
    G = Gallery(D, w)
    G_w[id(G)] = w
    return G


# ---------------------------------------------------------------- rules
S_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)
M_GRID = np.round(np.arange(0.0, 0.4001, 0.01), 2)


def top2(C: Cases):
    filled = np.where(np.isfinite(C.sim), C.sim, -np.inf)
    order = np.argsort(-filled, axis=1)
    rows = np.arange(len(C))
    best, bcol = filled[rows, order[:, 0]], order[:, 0]
    second = filled[rows, order[:, 1]] if C.sim.shape[1] > 1 else np.full(len(C), -np.inf)
    with np.errstate(invalid='ignore'):
        gap = np.where(np.isfinite(second), best - second, np.inf)
    gap = np.where(np.isfinite(best), gap, -np.inf)
    return best, gap, bcol


def grid_counts(C: Cases, mask: np.ndarray, mutual: bool):
    """decided / correct / wrong over the (s_min, margin) grid for the queries in ``mask``."""
    best, gap, bcol = top2(C)
    ok = (bcol == C.true)[mask]
    b, g = best[mask], gap[mask]
    keep = (b[:, None, None] >= S_GRID[None, :, None]) & (g[:, None, None] >= M_GRID[None, None, :])
    if mutual:
        keep &= C.mutual[mask][:, None, None]
    decided = keep.sum(0)
    correct = (keep & ok[:, None, None]).sum(0)
    return decided, correct, decided - correct


def apply_rule(C: Cases, mask: np.ndarray, s_min: float, margin: float, mutual: bool) -> dict:
    best, gap, bcol = top2(C)
    dec = (best >= s_min) & (gap >= margin) & mask
    if mutual:
        dec &= C.mutual
    corr = dec & (bcol == C.true)
    wrong = dec & ~corr
    return {"queries": int(mask.sum()), "decided": int(dec.sum()), "correct": int(corr.sum()), "wrong": int(wrong.sum()),
            "open": int(mask.sum() - dec.sum()), "wrong_idx": np.flatnonzero(wrong)}


def choose_rule(C: Cases, train: np.ndarray, *, guard=0.02):
    """Zero wrong on the training queries; among those maximise decided. 'robust' additionally requires zero wrong
    after loosening both thresholds by ``guard``. Returns {mutual: (strict, robust)} with (s, m, decided_train)."""
    out = {}
    for mutual in (False, True):
        dec, cor, wr = grid_counts(C, train, mutual)
        gs = int(round(guard / 0.01))
        padded = np.pad(wr, ((gs, 0), (gs, 0)), constant_values=0)  # index (i-gs, j-gs); out of range = more strict
        loosened = padded[:wr.shape[0], :wr.shape[1]]
        picks = {}
        for label, ok in (("strict", wr == 0), ("robust", (wr == 0) & (loosened == 0))):
            score = np.where(ok, dec, -1)
            if score.max() < 0:
                picks[label] = None
                continue
            # most decided; ties -> larger margin then larger s_min (more conservative)
            best = score.max()
            cand = np.argwhere(score == best)
            i, j = max(cand, key=lambda ij: (M_GRID[ij[1]], S_GRID[ij[0]]))
            picks[label] = (float(S_GRID[i]), float(M_GRID[j]), int(dec[i, j]))
        out[mutual] = picks
    return out


def is_heldout(C: Cases, T: int):
    return C.t >= T - 3000


def is_train(C: Cases):
    return C.t < 3000
