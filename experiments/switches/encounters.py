"""Close encounters between identities of a solution, without golden labels.

An *encounter* is a pair of identities measured (3D root) within R metres at >= k sync frames, runs
closer than ``gap_s`` apart merged. *Edge events* are the takeover pattern: an identity whose last
(first) measured root lies within R of another identity measured at that moment (``death_near`` /
``birth_near``), and ``handoff``: a track ending and another starting within ``handoff_s`` and
``handoff_m`` of each other.

    python experiments/switches/encounters.py              # golden counts + felix counts
"""
from __future__ import annotations

import sys

import numpy as np

import common
from common import FPS, INCH_PER_M


def close_pairs(tr_ident, tr_t, tr_root, R_m, min_ident_frames=0):
    """All (a, b, t, d_m) with both roots measured at sync t and distance < R. One root per (identity, t)."""
    ok = np.isfinite(tr_root[:, 0])
    ident, t, root = tr_ident[ok], tr_t[ok], tr_root[ok]
    order = np.argsort(t, kind="stable")
    ident, t, root = ident[order], t[order], root[order]
    starts = np.flatnonzero(np.concatenate(([True], t[1:] != t[:-1])))
    ends = np.concatenate((starts[1:], [len(t)]))
    out_a, out_b, out_t, out_d = [], [], [], []
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        r = root[s:e]
        d = np.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1) / INCH_PER_M
        i, j = np.nonzero(np.triu(d < R_m, 1))
        if len(i):
            ids = ident[s:e]
            a, b = ids[i], ids[j]
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            keep = lo != hi
            out_a.append(lo[keep]); out_b.append(hi[keep]); out_t.append(np.full(keep.sum(), t[s])); out_d.append(d[i, j][keep])
    if not out_a:
        return [np.empty(0, int)] * 3 + [np.empty(0)]
    return [np.concatenate(x) for x in (out_a, out_b, out_t, out_d)]


def pair_encounters(tr_ident, tr_t, tr_root, R_m=1.0, k=5, gap_s=3.0):
    a, b, t, d = close_pairs(tr_ident, tr_t, tr_root, R_m)
    key = a.astype(np.int64) * 10_000_000 + b
    order = np.lexsort((t, key))
    a, b, t, d, key = a[order], b[order], t[order], d[order], key[order]
    enc = []
    if not len(t):
        return enc
    new = np.ones(len(t), bool)
    new[1:] = (key[1:] != key[:-1]) | (t[1:] - t[:-1] > gap_s * FPS)
    starts = np.flatnonzero(new)
    ends = np.concatenate((starts[1:], [len(t)]))
    for s, e in zip(starts, ends):
        if e - s >= k:
            enc.append({"type": "pair", "a": int(a[s]), "b": int(b[s]), "t0": int(t[s]), "t1": int(t[e - 1]),
                        "n_close": int(e - s), "d_min": float(d[s:e].min()), "d_med": float(np.median(d[s:e]))})
    return enc


def lifetimes(tr_ident, tr_t, tr_root):
    ok = np.isfinite(tr_root[:, 0])
    ident, t, root = tr_ident[ok], tr_t[ok], tr_root[ok]
    life = {}
    order = np.lexsort((t, ident))
    ident, t, root = ident[order], t[order], root[order]
    starts = np.flatnonzero(np.concatenate(([True], ident[1:] != ident[:-1])))
    ends = np.concatenate((starts[1:], [len(t)]))
    for s, e in zip(starts, ends):
        life[int(ident[s])] = {"t_first": int(t[s]), "t_last": int(t[e - 1]), "n": int(e - s),
                               "p_first": root[s], "p_last": root[e - 1]}
    return life


def edge_events(tr_ident, tr_t, tr_root, T, R_m=1.0, handoff_s=5.0, handoff_m=1.5, min_frames=10, margin_s=2.0):
    life = lifetimes(tr_ident, tr_t, tr_root)
    life = {k: v for k, v in life.items() if v["n"] >= min_frames}
    ok = np.isfinite(tr_root[:, 0])
    ident, t, root = tr_ident[ok], tr_t[ok], tr_root[ok]
    order = np.argsort(t, kind="stable")
    ident, t, root = ident[order], t[order], root[order]
    lo = np.searchsorted(t, np.arange(T + 2))
    events = []

    def neighbours(x, tt, pos):
        a, b = lo[max(0, tt - 3)], lo[min(T, tt + 4)]
        m = (ident[a:b] != x)
        if not m.any():
            return None
        d = np.linalg.norm(root[a:b][m] - pos, axis=1) / INCH_PER_M
        i = int(np.argmin(d))
        return (int(ident[a:b][m][i]), float(d[i])) if d[i] < R_m else None

    for x, v in life.items():
        if v["t_last"] < T - 20:
            n = neighbours(x, v["t_last"], v["p_last"])
            if n:
                events.append({"type": "death_near", "a": x, "b": n[0], "t0": v["t_last"], "t1": v["t_last"], "d_min": n[1]})
        if v["t_first"] > 20:
            n = neighbours(x, v["t_first"], v["p_first"])
            if n:
                events.append({"type": "birth_near", "a": x, "b": n[0], "t0": v["t_first"], "t1": v["t_first"], "d_min": n[1]})
    ends = sorted(((v["t_last"], x) for x, v in life.items() if v["t_last"] < T - 20))
    births = sorted(((v["t_first"], x) for x, v in life.items() if v["t_first"] > 20))
    for te, x in ends:
        for tb, y in births:
            if y == x or tb < te - 5 or tb > te + handoff_s * FPS:
                continue
            d = np.linalg.norm(life[y]["p_first"] - life[x]["p_last"]) / INCH_PER_M
            if d < handoff_m:
                events.append({"type": "handoff", "a": x, "b": y, "t0": te, "t1": tb, "d_min": float(d)})
    return events


def enumerate_all(sol_tr, T, R_m=1.0, k=5, gap_s=3.0):
    ident, t, root = sol_tr
    return pair_encounters(ident, t, root, R_m, k, gap_s) + edge_events(ident, t, root, T, R_m)


def felix_tracks():
    import shutil
    root = common.DATA / "felix_20251223173052_20251223183103" / "rounds"
    latest = sorted(p for p in root.iterdir() if (p / "tracks3d.npz").is_file())[-1]
    dst = common.SCRATCH / f"felix_{latest.name}_tracks3d.npz"
    if not dst.is_file():
        shutil.copyfile(latest / "tracks3d.npz", dst)
    with np.load(dst) as d:
        return latest.name, d["identity"], d["t"], d["root"], len(d["native"]), d["aligned_time"]


if __name__ == "__main__":
    ws, scene, ref = common.load_scene_ref()
    print("golden T =", scene.T, "minutes", scene.T / FPS / 60)
    for name in ("saved", "autofit", "baseline"):
        sol = common.get_solution(name, ws, scene)
        print(f"\n[{name}] identities with 3D: {len(np.unique(sol.tr_ident))}")
        for R in (0.5, 0.75, 1.0):
            enc = pair_encounters(sol.tr_ident, sol.tr_t, sol.tr_root, R, 5, 3.0)
            dur = sum((e["t1"] - e["t0"] + 1) for e in enc) / FPS
            print(f"  R={R}: {len(enc)} pair encounters, {dur:.0f} s total, longest {max([e['t1']-e['t0'] for e in enc] or [0])/FPS:.0f} s")
        ev = edge_events(sol.tr_ident, sol.tr_t, sol.tr_root, scene.T, 1.0)
        for typ in ("death_near", "birth_near", "handoff"):
            print(f"  {typ}: {sum(e['type']==typ for e in ev)}")
    rname, ident, t, root, T, at = felix_tracks()
    print(f"\nfelix {rname}: T={T} ({T/FPS/60:.1f} min), identities {len(np.unique(ident))}")
    for R in (0.5, 0.75, 1.0):
        enc = pair_encounters(ident, t, root, R, 5, 3.0)
        dur = sum((e["t1"] - e["t0"] + 1) for e in enc) / FPS
        print(f"  R={R}: {len(enc)} pair encounters ({len(enc)/(T/FPS/3600):.0f}/h), {dur:.0f} s total, longest {max([e['t1']-e['t0'] for e in enc] or [0])/FPS:.0f} s")
    ev = edge_events(ident, t, root, T, 1.0)
    for typ in ("death_near", "birth_near", "handoff"):
        print(f"  {typ}: {sum(e['type']==typ for e in ev)}")


def pair_encounters_2d(ctx, sol, iou_thr=0.1, k=5, gap_s=3.0):
    """Pairs of identities whose boxes overlap (IoU > iou_thr) in some camera at the same sync frame."""
    keys, ts = [], []
    for c in ctx.scene.cameras:
        ident = sol.ident[c.index]
        rows = np.flatnonzero((ident >= 0) & (c.t >= 0))
        if not len(rows):
            continue
        o = rows[np.argsort(c.t[rows], kind="stable")]
        t = c.t[o]
        starts = np.flatnonzero(np.concatenate(([True], t[1:] != t[:-1])))
        ends = np.concatenate((starts[1:], [len(o)]))
        boxes = ctx.boxes[c.index]
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            r = o[s:e]
            b = boxes[r, :4].astype(np.float64)
            x1, y1 = np.maximum(b[:, None, 0], b[None, :, 0]), np.maximum(b[:, None, 1], b[None, :, 1])
            x2, y2 = np.minimum(b[:, None, 2], b[None, :, 2]), np.minimum(b[:, None, 3], b[None, :, 3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            iou = inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-9)
            i, j = np.nonzero(np.triu(iou > iou_thr, 1))
            if len(i):
                ids = ident[r]
                a_, b_ = ids[i], ids[j]
                ok = a_ != b_
                keys.append(np.minimum(a_, b_)[ok].astype(np.int64) * 10_000_000 + np.maximum(a_, b_)[ok])
                ts.append(np.full(ok.sum(), t[s]))
    if not keys:
        return []
    key, t = np.concatenate(keys), np.concatenate(ts)
    u = np.unique(np.stack([key, t], 1), axis=0)
    key, t = u[:, 0], u[:, 1]
    order = np.lexsort((t, key))
    key, t = key[order], t[order]
    new = np.ones(len(t), bool)
    new[1:] = (key[1:] != key[:-1]) | (t[1:] - t[:-1] > gap_s * FPS)
    starts = np.flatnonzero(new)
    ends = np.concatenate((starts[1:], [len(t)]))
    out = []
    for s, e in zip(starts, ends):
        if e - s >= k:
            out.append({"type": "pair2d", "a": int(key[s] // 10_000_000), "b": int(key[s] % 10_000_000), "t0": int(t[s]),
                        "t1": int(t[e - 1]), "n_close": int(e - s), "d_min": float("nan"), "d_med": float("nan")})
    return out
