"""Swap-vs-stay cues for an encounter (pair of identities A, B around a chunk [s, e] of sync frames).

Every cue returns a dict of floats where *larger = more evidence that the labels were exchanged*
("swap evidence"; NaN when the data to compute it is missing). Data before ([s-W, s)) and after
((e, e+W]) the chunk is compared:  stay = A_b~A_a and B_b~B_a;  swap = A_b~B_a and B_b~A_a.

A hypothetical swap is *injected* by passing ``swap=(A, B, tau)``: for sync t >= tau every row and
every 3D track entry labelled A is read as B's and vice versa (physical data unchanged).

Cues: motion_pos, motion_cv (3D root continuity), body (bone-length signature), app, app_samecam
(fast-reid embeddings, golden labelled frames only), trk_* (BoT-SORT tracklets that span the chunk
and change engine identity, in cameras where A and B stay apart in the image), raw_* (Autofit raw
track ids: churn / changes inside the window), own_* (per-camera pose ownership: a row fits the
other identity's 3D pose better than its own).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import common
from common import FPS, INCH_PER_M

W_DEFAULT = 100  # frames (10 s)
# COCO bones (pairs) used for the signature, left/right averaged
BONES = {"shoulder": [(5, 6)], "hip": [(11, 12)], "upper_arm": [(5, 7), (6, 8)], "forearm": [(7, 9), (8, 10)],
         "thigh": [(11, 13), (12, 14)], "shin": [(13, 15), (14, 16)]}
FEATURES = list(BONES) + ["torso", "leg_torso"]


def bone_features(xyz: np.ndarray) -> np.ndarray:
    """(n, 17, 3) -> (n, 8) lengths in inches (NaN where a joint is missing)."""
    out = np.full((len(xyz), len(FEATURES)), np.nan, np.float32)
    for k, (name, pairs) in enumerate(BONES.items()):
        vals = np.stack([np.linalg.norm(xyz[:, a] - xyz[:, b], axis=1) for a, b in pairs])
        with np.errstate(all="ignore"):
            out[:, k] = np.nanmean(vals, axis=0)
    sh, hp = xyz[:, [5, 6]].mean(axis=1), xyz[:, [11, 12]].mean(axis=1)
    out[:, len(BONES)] = np.linalg.norm(sh - hp, axis=1)
    out[:, len(BONES) + 1] = out[:, len(BONES)] + out[:, list(BONES).index("thigh")] + out[:, list(BONES).index("shin")]
    return out


class Ctx:
    """Everything the cues read, loaded once per process."""

    def __init__(self, with_appearance: bool = True, workspace: str = common.GOLDEN):
        if workspace == common.GOLDEN:
            self.ws, self.scene, self.ref = common.load_scene_ref()
        else:
            from data_engine.config import RECIPE
            from data_engine.solve.scene import load_scene
            from data_engine.workspace import Workspace
            self.ws = Workspace(workspace)
            self.scene = load_scene(self.ws, joint_peak=RECIPE["geometry"]["joint_peak"], rays=False)
            self.ref = None
            with_appearance = False
        sc = self.scene
        self.T = sc.T
        self.boxes = [c.detections.boxes for c in sc.cameras]
        self.kps = [c.detections.keypoints for c in sc.cameras]
        self.track = [c.tracklet for c in sc.cameras]
        self.t_of = [c.t for c in sc.cameras]
        with np.load(self.ws.path("evidence", "autofit", "labels.npz")) as d:
            self.raw = [d[f"raw_{c.camera}"] for c in sc.cameras]
            self.pose_tid, self.pose_t, self.pose_state = d["pose_tid"], d["pose_t"], d["pose_state"]
        self.raw_life = {}
        for c in range(len(sc.cameras)):
            ok = (self.raw[c] >= 0) & (self.t_of[c] >= 0)
            for tid in np.unique(self.raw[c][ok]):
                tt = self.t_of[c][ok & (self.raw[c] == tid)]
                lo, hi = self.raw_life.get(int(tid), (10 ** 9, -1))
                self.raw_life[int(tid)] = (min(lo, int(tt.min())), max(hi, int(tt.max())))
        self.emb = None
        if with_appearance:
            from data_engine import reid
            from data_engine.reid import l2_normalize
            self.emb = []
            for c in sc.cameras:
                rows, e = reid.load_embeddings(common.DATA / "_scratch" / "reid" / "golden_fastreid", c.camera)
                idx = np.full(len(c.kept), -1, np.int32)
                idx[rows] = np.arange(len(rows), dtype=np.int32)
                self.emb.append((idx, l2_normalize(e)))


class Index:
    """Per-identity access to a Sol's rows and 3D track entries (sorted by t)."""

    def __init__(self, ctx: Ctx, sol: common.Sol):
        self.ctx, self.sol = ctx, sol
        o = np.lexsort((sol.tr_t, sol.tr_ident))
        self.tr_ident, self.tr_t = sol.tr_ident[o], sol.tr_t[o]
        self.tr_root, self.tr_xyz = sol.tr_root[o], sol.tr_xyz[o]
        self.tr_span = self._spans(self.tr_ident)
        cam, row, t, ident = [], [], [], []
        for c in ctx.scene.cameras:
            r = np.flatnonzero((sol.ident[c.index] >= 0) & (c.t >= 0))
            cam.append(np.full(len(r), c.index)); row.append(r); t.append(c.t[r]); ident.append(sol.ident[c.index][r])
        cam, row, t, ident = map(np.concatenate, (cam, row, t, ident))
        o = np.lexsort((t, ident))
        self.r_cam, self.r_row, self.r_t, self.r_ident = cam[o], row[o], t[o], ident[o]
        self.r_span = self._spans(self.r_ident)
        self.sigma = self._bone_noise()

    @staticmethod
    def _spans(ident):
        spans = {}
        if len(ident):
            starts = np.flatnonzero(np.concatenate(([True], ident[1:] != ident[:-1])))
            ends = np.concatenate((starts[1:], [len(ident)]))
            for s, e in zip(starts, ends):
                spans[int(ident[s])] = (int(s), int(e))
        return spans

    def _bone_noise(self):
        """Within-identity spread of 10 s window medians of each bone feature (inches)."""
        devs = [[] for _ in FEATURES]
        for x, (s, e) in self.tr_span.items():
            t, xyz = self.tr_t[s:e], self.tr_xyz[s:e]
            for w0 in range(int(t[0]), int(t[-1]), W_DEFAULT):
                m = (t >= w0) & (t < w0 + W_DEFAULT)
                if m.sum() >= 15:
                    f = np.nanmedian(bone_features(xyz[m]), axis=0) if np.isfinite(xyz[m]).any() else None
                    if f is not None:
                        for k, v in enumerate(f):
                            devs[k].append((x, v))
        sig = np.ones(len(FEATURES), np.float32)
        for k in range(len(FEATURES)):
            by = {}
            for x, v in devs[k]:
                if np.isfinite(v):
                    by.setdefault(x, []).append(v)
            res = np.concatenate([np.array(v) - np.median(v) for v in by.values() if len(v) >= 3] or [np.array([1.0])])
            sig[k] = max(1.4826 * np.median(np.abs(res)), 0.05)
        return sig

    # ------------------------------------------------------------------ windows
    def segments(self, label, lo, hi, swap):
        if swap is None or label not in swap[:2]:
            return [(label, lo, hi)]
        a, b, tau = swap[:3]
        takeover = len(swap) > 3 and swap[3] == "take"  # a continues on b's person, b is lost after tau
        other = b if label == a else a
        out = []
        if takeover and label == b:
            return [(label, lo, min(hi, tau - 1))] if lo < tau else []
        if lo < tau:
            out.append((label, lo, min(hi, tau - 1)))
        if hi >= tau:
            out.append((other, max(lo, tau), hi))
        return out

    def tr(self, label, lo, hi, swap=None):
        ts, roots, xyzs = [], [], []
        for who, a, b in self.segments(label, lo, hi, swap):
            if who not in self.tr_span:
                continue
            s, e = self.tr_span[who]
            t = self.tr_t[s:e]
            i, j = s + np.searchsorted(t, a), s + np.searchsorted(t, b, side="right")
            ts.append(self.tr_t[i:j]); roots.append(self.tr_root[i:j]); xyzs.append(self.tr_xyz[i:j])
        if not ts:
            return np.empty(0, int), np.empty((0, 3)), np.empty((0, 17, 3))
        return np.concatenate(ts), np.concatenate(roots), np.concatenate(xyzs)

    def rows(self, label, lo, hi, swap=None):
        cs, rs, ts = [], [], []
        for who, a, b in self.segments(label, lo, hi, swap):
            if who not in self.r_span:
                continue
            s, e = self.r_span[who]
            t = self.r_t[s:e]
            i, j = s + np.searchsorted(t, a), s + np.searchsorted(t, b, side="right")
            cs.append(self.r_cam[i:j]); rs.append(self.r_row[i:j]); ts.append(self.r_t[i:j])
        if not cs:
            return np.empty(0, int), np.empty(0, int), np.empty(0, int)
        return np.concatenate(cs), np.concatenate(rs), np.concatenate(ts)


# ====================================================================== cues
def _swap_score(d_ab: dict) -> float:
    """d_ab[(X, 'b'), (Y, 'a')] distances; swap evidence = sum(d_stay) - sum(d_swap) over available terms."""
    total, n = 0.0, 0
    for X, Y in (("A", "B"), ("B", "A")):
        stay, swp = d_ab.get((X, X)), d_ab.get((Y, X))  # after-window of X compared with X_b (stay) vs Y_b (swap)
        if stay is not None and swp is not None and np.isfinite(stay) and np.isfinite(swp):
            total += stay - swp
            n += 1
    return total / n if n else np.nan


def cue_motion(ix: Index, A, B, s, e, swap, W=W_DEFAULT, k=5, max_gap=50):
    out = {"motion_pos": np.nan, "motion_cv": np.nan}
    before, after = {}, {}
    for name, label in (("A", A), ("B", B)):
        tb, rb, _ = ix.tr(label, s - W, s - 1, swap)
        ta, ra, _ = ix.tr(label, e + 1, e + W, swap)
        okb, oka = np.isfinite(rb[:, 0]), np.isfinite(ra[:, 0])
        tb, rb, ta, ra = tb[okb], rb[okb], ta[oka], ra[oka]
        if len(tb) >= 3 and s - tb[-1] <= max_gap:
            last = rb[-k:]; tl = tb[-k:]
            v = np.zeros(3)
            if len(last) >= 3 and tl[-1] > tl[0]:
                v = np.polyfit(tl - tl[-1], last, 1)[0]
            before[name] = (np.median(last, axis=0), v, int(tl[-1]))
        if len(ta) >= 3 and ta[0] - e <= max_gap:
            after[name] = (np.median(ra[:k], axis=0), int(ta[0]))
    for key, use_v in (("motion_pos", False), ("motion_cv", True)):
        d = {}
        for X in after:
            for Y in before:
                p_b, v, tbl = before[Y]
                p_a, ta0 = after[X]
                pred = p_b + (np.clip(np.linalg.norm(v), 0, 6.0) * (v / max(np.linalg.norm(v), 1e-9)) * min(ta0 - tbl, 20) if use_v else 0)
                d[(Y, X)] = float(np.linalg.norm(p_a - pred)) / INCH_PER_M
        out[key] = _swap_score_motion(d, before, after)
    return out


def _swap_score_motion(d, before, after):
    total, n = 0.0, 0
    for X in after:
        Y = "B" if X == "A" else "A"
        if (X, X) in d and (Y, X) in d:
            total += d[(X, X)] - d[(Y, X)]
            n += 1
    return total / n if n else np.nan


def _sig(ix, label, lo, hi, swap, min_frames=8):
    t, _, xyz = ix.tr(label, lo, hi, swap)
    if len(t) < min_frames:
        return None
    f = bone_features(xyz)
    with np.errstate(all="ignore"):
        med = np.nanmedian(f, axis=0)
    return med if np.isfinite(med).sum() >= 5 else None


def cue_body(ix, A, B, s, e, swap, W=W_DEFAULT):
    sig = {}
    for name, label in (("A", A), ("B", B)):
        sig[(name, "b")] = _sig(ix, label, s - W, s - 1, swap)
        sig[(name, "a")] = _sig(ix, label, e + 1, e + W, swap)

    def dist(u, v):
        if u is None or v is None:
            return np.nan
        z = (u - v) / ix.sigma
        z = z[np.isfinite(z)]
        return float(np.sqrt(np.mean(z ** 2))) if len(z) >= 4 else np.nan
    total, n = 0.0, 0
    for X, Y in (("A", "B"), ("B", "A")):
        st, sw = dist(sig[(X, "b")], sig[(X, "a")]), dist(sig[(Y, "b")], sig[(X, "a")])
        if np.isfinite(st) and np.isfinite(sw):
            total += st - sw; n += 1
    return {"body": total / n if n else np.nan}


def _emb_mean(ctx, cams, rows, only_cam=None):
    vecs, w = [], []
    for c, r in zip(cams, rows):
        if only_cam is not None and c != only_cam:
            continue
        idx, E = ctx.emb[c]
        i = idx[r]
        if i >= 0:
            h = ctx.boxes[c][r, 3] - ctx.boxes[c][r, 1]
            vecs.append(E[i]); w.append(np.clip((h - 32) / 224, 0.05, 1.0) * np.clip(ctx.boxes[c][r, 4], 0.05, 1))
    if len(vecs) < 2:
        return None
    m = (np.asarray(vecs) * np.asarray(w)[:, None]).sum(axis=0)
    n = np.linalg.norm(m)
    return m / n if n > 1e-9 else None


def cue_appearance(ctx, ix, A, B, s, e, swap, W=W_DEFAULT):
    if ctx.emb is None:
        return {"app": np.nan, "app_samecam": np.nan}
    wins = {}
    for name, label in (("A", A), ("B", B)):
        wins[(name, "b")] = ix.rows(label, s - W, s - 1, swap)
        wins[(name, "a")] = ix.rows(label, e + 1, e + W, swap)

    def score(only_cam=None):
        mean = {k: _emb_mean(ctx, v[0], v[1], only_cam) for k, v in wins.items()}
        total, n = 0.0, 0
        for X, Y in (("A", "B"), ("B", "A")):
            xa, xb, yb = mean[(X, "a")], mean[(X, "b")], mean[(Y, "b")]
            if xa is not None and xb is not None and yb is not None:
                total += float(yb @ xa - xb @ xa); n += 1       # swap similarity - stay similarity
        return total / n if n else np.nan
    allc = score()
    per = [score(c) for c in range(len(ctx.emb))]
    per = [p for p in per if np.isfinite(p)]
    return {"app": allc, "app_samecam": float(np.mean(per)) if per else np.nan}


def _box_iou(a, b):
    x1, y1 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / np.maximum((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter, 1e-9)


def cue_tracklet(ctx, ix, A, B, s, e, swap, W=W_DEFAULT, iou_sep=0.05):
    """BoT-SORT tracklets with rows labelled {A,B} both before s and after e: do they keep one label?"""
    out = {"trk_contra_sep": 0.0, "trk_agree_sep": 0.0, "trk_contra_all": 0.0, "trk_agree_all": 0.0}
    la = {"A": A, "B": B}
    data = {}
    for name, label in la.items():
        data[name] = ix.rows(label, s - W, e + W, swap)
    for c in range(len(ctx.scene.cameras)):
        rows, who, tt = [], [], []
        for name in "AB":
            cm, rw, t = data[name]
            m = cm == c
            rows.append(rw[m]); tt.append(t[m]); who.append(np.full(m.sum(), 0 if name == "A" else 1))
        rows, who, tt = map(np.concatenate, (rows, who, tt))
        if not len(rows):
            continue
        trk = ctx.track[c][rows]
        ok = trk >= 0
        if not ok.any():
            continue
        # separation of the two identities in this camera around the chunk
        sep = True
        za = (tt >= s - 10) & (tt <= e + 10)
        ra, rb_ = rows[za & (who == 0)], rows[za & (who == 1)]
        ta_, tb_ = tt[za & (who == 0)], tt[za & (who == 1)]
        if len(ra) and len(rb_):
            common_t = np.intersect1d(ta_, tb_)
            for t0 in common_t:
                ia, ib = ra[ta_ == t0][0], rb_[tb_ == t0][0]
                if _box_iou(ctx.boxes[c][[ia]], ctx.boxes[c][[ib]])[0] > iou_sep:
                    sep = False
                    break
        for k in np.unique(trk[ok]):
            m = ok & (trk == k)
            before, after = m & (tt < s), m & (tt > e)
            if before.sum() < 2 or after.sum() < 2:
                continue
            vb = np.bincount(who[before], minlength=2).argmax()
            va = np.bincount(who[after], minlength=2).argmax()
            kind = "contra" if vb != va else "agree"
            out[f"trk_{kind}_all"] += 1
            if sep:
                out[f"trk_{kind}_sep"] += 1
    out["trk_swap_score"] = out["trk_contra_sep"] - out["trk_agree_sep"] if (out["trk_contra_sep"] + out["trk_agree_sep"]) else np.nan
    return out


def cue_autofit_raw(ctx, ix, A, B, s, e, swap, W=W_DEFAULT, margin=20):
    """Autofit-internal signs around the chunk (no swap/stay direction: raw ids are not changed by an injected relabel
    except raw_cross, which compares engine identity with raw-id continuity)."""
    out = {"raw_changes": np.nan, "raw_birthdeath": np.nan, "raw_lost_frames": np.nan, "raw_cross": np.nan}
    def dom(label, lo, hi):
        cm, rw, t = ix.rows(label, lo, hi, swap)
        ids = np.array([ctx.raw[c][r] for c, r in zip(cm, rw)])
        ids = ids[ids >= 0]
        return (np.bincount(ids).argmax() if len(ids) else None), set(ids.tolist())
    changes, n = 0, 0
    sets = {}
    for name, label in (("A", A), ("B", B)):
        db, sb = dom(label, s - W, s - 1)
        da, sa = dom(label, e + 1, e + W)
        sets[(name, "b")], sets[(name, "a")] = sb, sa
        if db is not None and da is not None:
            n += 1
            changes += int(db != da)
    out["raw_changes"] = float(changes) if n else np.nan
    zone = set()
    for label in (A, B):
        _, z = dom(label, s - margin, e + margin)
        zone |= z
    lo, hi = s - margin, e + margin
    out["raw_birthdeath"] = float(sum(1 for r in zone if lo <= ctx.raw_life[r][0] <= hi or lo <= ctx.raw_life[r][1] <= hi))
    if len(ctx.pose_tid):
        m = np.isin(ctx.pose_tid, list(zone)) & (ctx.pose_t >= lo) & (ctx.pose_t <= hi) & (ctx.pose_state > 0)
        out["raw_lost_frames"] = float(m.sum())
    if all(sets[k] for k in sets):
        stay = len(sets[("A", "b")] & sets[("A", "a")]) + len(sets[("B", "b")] & sets[("B", "a")])
        cross = len(sets[("A", "b")] & sets[("B", "a")]) + len(sets[("B", "b")] & sets[("A", "a")])
        out["raw_cross"] = float(cross - stay)
    return out


def row_ownership(ctx, ix, label, other, cams, rows, ts, swap=None):
    """Per row: normalised reprojection error of the label's own 3D pose and of the other identity's pose."""
    own, oth = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    if not len(rows):
        return own, oth
    lo, hi = int(ts.min()), int(ts.max())
    poses = {}
    for name, lab in (("own", label), ("other", other)):
        t, _, xyz = ix.tr(lab, lo, hi, swap)
        poses[name] = (t, xyz)
    for c in range(len(ctx.scene.cameras)):
        cam = ctx.scene.cameras[c].cam
        m = np.flatnonzero(cams == c)
        if not len(m) or cam is None:
            continue
        kp = ctx.kps[c][rows[m]][:, 5:17]
        h = np.maximum(ctx.boxes[c][rows[m], 3] - ctx.boxes[c][rows[m], 1], 1.0)
        for name, store in (("own", own), ("other", oth)):
            t, xyz = poses[name]
            if not len(t):
                continue
            pos = np.searchsorted(t, ts[m])
            found = (pos < len(t)) & (t[np.clip(pos, 0, len(t) - 1)] == ts[m])
            if not found.any():
                continue
            pts = xyz[pos[found]][:, 5:17]
            proj = cam.project(pts, verify=False)
            d = np.linalg.norm(proj - kp[found][:, :, :2], axis=-1)
            good = (kp[found][:, :, 2] >= 0.3) & np.isfinite(d)
            d = np.where(good, d, np.nan)
            enough = good.sum(axis=1) >= 4
            with np.errstate(all="ignore"):
                med = np.nanmedian(d, axis=1) / h[found]
            vals = np.where(enough, med, np.nan)
            store[m[found]] = vals
    return own, oth


def cue_ownership(ctx, ix, A, B, s, e, swap, flip_margin=0.04):
    """Rows inside [s, e] whose own 3D pose fits worse than the other identity's by more than flip_margin box heights."""
    flips, rows_n, worst = 0, 0, 0.0
    for label, other in ((A, B), (B, A)):
        cm, rw, t = ix.rows(label, s, e, swap)
        own, oth = row_ownership(ctx, ix, label, other, cm, rw, t, swap)
        ok = np.isfinite(own) & np.isfinite(oth)
        rows_n += int(ok.sum())
        diff = (own - oth)[ok]
        flips += int((diff > flip_margin).sum())
        if len(diff):
            worst = max(worst, float(diff.max()))
    return {"own_flips": float(flips), "own_rows": float(rows_n), "own_flip_share": flips / rows_n if rows_n else np.nan,
            "own_worst": worst if rows_n else np.nan}


ALL_CUES = ("motion", "body", "app", "trk", "raw", "own")


def all_cues(ctx, ix, A, B, s, e, swap=None, W=W_DEFAULT, which=ALL_CUES):
    out = {}
    if "motion" in which:
        out.update(cue_motion(ix, A, B, s, e, swap, W))
    if "body" in which:
        out.update(cue_body(ix, A, B, s, e, swap, W))
    if "app" in which:
        out.update(cue_appearance(ctx, ix, A, B, s, e, swap, W))
    if "trk" in which:
        out.update(cue_tracklet(ctx, ix, A, B, s, e, swap, W))
    if "raw" in which:
        out.update(cue_autofit_raw(ctx, ix, A, B, s, e, swap, W))
    if "own" in which:
        out.update(cue_ownership(ctx, ix, A, B, s, e, swap))
    return out
