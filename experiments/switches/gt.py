"""Ground truth: every identity error of a solution against the golden labels, with closeness.

    python experiments/switches/gt.py saved adopted baseline autofit

An *episode* = a run of labelled observations of identity X whose reference person P is not X's
majority person M, split where consecutive wrong observations are > GAP_S apart.
Types (decided from the labels only):
    swap          identity Y whose majority is P carries M at an overlapping time (A<->B exchanged)
    takeover      no exchange; the episode touches the start or end of X's labelled life
                  (X continues on the other person / is born on the other person)
    contamination no exchange; X has correct observations before and after the episode
Closeness uses the root distance between P and M over the window (episode +- MARGIN_S) from the
*saved* solution's 3D tracks (person trajectory = tracks of identities whose majority is that person),
and 2D box IoU between P and M boxes in every camera that labels both.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict

import numpy as np

import common
from common import INCH_PER_M

GAP_S = 5.0
MARGIN_S = 3.0


def obs_table(scene, ref, sol):
    cam, row, t, person, ident = [], [], [], [], []
    for c in scene.cameras:
        r = np.flatnonzero((ref.person[c.index] >= 0) & (sol.ident[c.index] >= 0) & (c.t >= 0))
        cam.append(np.full(len(r), c.index)); row.append(r); t.append(c.t[r])
        person.append(ref.person[c.index][r]); ident.append(sol.ident[c.index][r])
    return [np.concatenate(a) for a in (cam, row, t, person, ident)]


def majority(person, ident):
    maj, counts = {}, {}
    for x in np.unique(ident):
        c = Counter(person[ident == x].tolist())
        maj[int(x)] = c.most_common(1)[0][0]
        counts[int(x)] = c
    return maj, counts


def trajectories(scene, ref, sol_ref, maj_ref, T):
    """person -> (T,3) root (inches, NaN) from the reference solution's tracks."""
    traj = {}
    for p in range(ref.people + 1):
        traj[p] = np.full((T, 3), np.nan, np.float32)
    ok = np.isfinite(sol_ref.tr_root[:, 0])
    for i in np.flatnonzero(ok):
        p = maj_ref.get(int(sol_ref.tr_ident[i]))
        if p is not None:
            traj[p][sol_ref.tr_t[i]] = sol_ref.tr_root[i]
    return traj


def build_ref_maj(scene, ref, sol):
    o = obs_table(scene, ref, sol)
    maj, _ = majority(o[3], o[4])
    return maj


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def pair_geometry(scene, ref, traj, p, q, t0, t1):
    """closeness of reference people p and q over [t0, t1] sync indices."""
    a, b = max(0, t0), min(scene.T - 1, t1)
    d = np.linalg.norm(traj[p][a:b + 1] - traj[q][a:b + 1], axis=1) / INCH_PER_M
    dmin = float(np.nanmin(d)) if np.isfinite(d).any() else np.nan
    iou = 0.0
    seen = set()
    for c in scene.cameras:
        rp = np.flatnonzero(ref.person[c.index] == p)
        rq = np.flatnonzero(ref.person[c.index] == q)
        if not len(rp) or not len(rq):
            continue
        tp = c.t[rp]; tq = c.t[rq]
        fp = c.detections.frame_of_row[rp]; fq = c.detections.frame_of_row[rq]
        common_f = np.intersect1d(fp, fq)
        for f in common_f:
            ip, iq = rp[fp == f][0], rq[fq == f][0]
            tt = int(c.t[ip])
            if a <= tt <= b:
                iou = max(iou, box_iou(c.detections.boxes[ip, :4], c.detections.boxes[iq, :4]))
                seen.add(c.index)
    return dmin, iou, len(seen)


def episodes(scene, ref, sol, traj, gap_s=GAP_S):
    cam, row, t, person, ident = obs_table(scene, ref, sol)
    maj, counts = majority(person, ident)
    life = {x: (int(t[ident == x].min()), int(t[ident == x].max())) for x in maj}
    wrong = np.array([person[i] != maj[int(ident[i])] for i in range(len(t))])
    groups = defaultdict(list)
    for i in np.flatnonzero(wrong):
        groups[(int(ident[i]), int(person[i]))].append(i)
    eps = []
    for (x, p), idx in groups.items():
        idx = np.array(idx)
        ts = t[idx]
        order = np.argsort(ts)
        idx, ts = idx[order], ts[order]
        cut = np.flatnonzero(np.diff(ts) > gap_s * common.FPS) + 1
        for chunk in np.split(np.arange(len(idx)), cut):
            sel = idx[chunk]
            eps.append({"identity": x, "true_person": p, "majority": maj[x], "t0": int(t[sel].min()), "t1": int(t[sel].max()),
                        "n_obs": len(sel), "n_cams": len(set(cam[sel].tolist())), "rows": sel})
    # classify
    for e in eps:
        x, p, m, t0, t1 = e["identity"], e["true_person"], e["majority"], e["t0"], e["t1"]
        partner = None
        for f in eps:
            if f is e or f["majority"] != p or f["true_person"] != m:
                continue
            if f["t0"] <= t1 + 20 and f["t1"] >= t0 - 20:
                partner = f["identity"]
                break
        lo, hi = life[x]
        if partner is not None:
            e["type"] = "swap"
        elif t0 - lo <= 20 or hi - t1 <= 20:
            e["type"] = "takeover"
        else:
            e["type"] = "contamination"
        e["partner_identity"] = partner
        w0, w1 = t0 - int(MARGIN_S * common.FPS), t1 + int(MARGIN_S * common.FPS)
        e["min_dist_m"], e["max_iou"], e["cams_both"] = pair_geometry(scene, ref, traj, p, m, w0, w1)
        # distance of the pair at the episode start (+-0.5 s)
        d0, _, _ = pair_geometry(scene, ref, traj, p, m, t0 - 5, t0 + 5)
        e["dist_at_start_m"] = d0
        # nearest other person to the victim around the episode (explains non-pair errors)
        best = (np.inf, None)
        for q in traj:
            if q in (p, m):
                continue
            a, b = max(0, t0 - 5), min(scene.T - 1, t1 + 5)
            for who in (p, m):
                d = np.linalg.norm(traj[q][a:b + 1] - traj[who][a:b + 1], axis=1) / INCH_PER_M
                if np.isfinite(d).any() and np.nanmin(d) < best[0]:
                    best = (float(np.nanmin(d)), q)
        e["third_party_dist_m"], e["third_party"] = best
        # whole-track swap vs one-camera detection error: how many cameras are wrong at the same instant
        by_t = defaultdict(set)
        for i in e["rows"]:
            by_t[int(t[i]) // 10].add(int(cam[i]))
        e["max_cams_same_second"] = max(len(v) for v in by_t.values())
        w = (t >= w0) & (t <= w1)
        e["cams_see_victim"] = len(set(cam[w & (person == p)].tolist()))
        e["cams_see_majority"] = len(set(cam[w & (person == m)].tolist()))
    return eps, maj, life


def summarise(name, eps, n_obs_wrong):
    print(f"\n=== {name}: {len(eps)} error episodes, {sum(e['n_obs'] for e in eps)} wrong observations")
    print("type            episodes  obs   min<0.5m  <0.75m  <1.0m   iou>0   iou>0.1   multi-cam(>=3 wrong at once)")
    for typ in ("swap", "takeover", "contamination", "ALL"):
        sub = [e for e in eps if typ == "ALL" or e["type"] == typ]
        if not sub:
            continue
        n = len(sub)
        f = lambda R: sum(1 for e in sub if e["min_dist_m"] < R)
        print(f"{typ:15s} {n:5d} {sum(e['n_obs'] for e in sub):6d}   {f(.5):4d}   {f(.75):4d}   {f(1.0):4d}"
              f"   {sum(1 for e in sub if e['max_iou'] > 0):4d}  {sum(1 for e in sub if e['max_iou'] > .1):4d}   {sum(1 for e in sub if e['max_cams_same_second'] >= 3):4d}")


def describe(eps, fps=common.FPS):
    for e in sorted(eps, key=lambda e: e["t0"]):
        print(f"  t={e['t0']/fps:6.1f}-{e['t1']/fps:6.1f}s id{e['identity']:>4d}(maj P{e['majority']}) carries P{e['true_person']} "
              f"{e['type']:13s} obs={e['n_obs']:3d} cams={e['n_cams']} d_min={e['min_dist_m']:.2f}m d0={e['dist_at_start_m']:.2f} "
              f"iou={e['max_iou']:.2f} both_cams={e['cams_both']} cams_wrong_at_once={e['max_cams_same_second']} see(P/M)={e['cams_see_victim']}/{e['cams_see_majority']} 3rd={e['third_party_dist_m']:.2f}m(P{e['third_party']})")


if __name__ == "__main__":
    ws, scene, ref = common.load_scene_ref()
    saved = common.load_saved(ws, scene)
    maj_ref = build_ref_maj(scene, ref, saved)
    traj = trajectories(scene, ref, saved, maj_ref, scene.T)
    for name in sys.argv[1:] or ["saved"]:
        sol = common.get_solution(name, ws, scene)
        eps, maj, life = episodes(scene, ref, sol, traj)
        summarise(name, eps, None)
        describe(eps)
        import json
        out = [{k: (v.item() if isinstance(v, np.generic) else v) for k, v in e.items() if k != "rows"} for e in eps]
        (common.SCRATCH / f"gt_episodes_{name}.json").write_text(json.dumps(out, default=float, indent=1))
