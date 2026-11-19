"""Breakdowns of the injection test: per person pair, per separation at the swap, own-cue and tracklet-cue thresholds.

    python experiments/switches/analysis2.py
"""
from __future__ import annotations

import pickle
from collections import Counter, defaultdict

import numpy as np

import common
import eval_cues as ec
import gt
import rule


def pair_table():
    rows = rule.load("saved")
    ctx_pairs = {}
    # map identity -> person via golden majority
    import cues
    ctx = cues.Ctx(with_appearance=False)
    sol = common.load_saved(ctx.ws, ctx.scene)
    maj = gt.build_ref_maj(ctx.scene, ctx.ref, sol)
    neg = [r for r in rows if r["kind"] == "neg"]
    thr = rule.thresholds(rows)
    print("thresholds (flag above / stay below):", {k: tuple(round(x, 3) for x in v) for k, v in thr.items()})
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[tuple(sorted((maj.get(r["a"], -1), maj.get(r["b"], -1))))].append(r)
    print("\nper person pair (golden persons): chunks, injected swaps missed by the OR rule (motion|body|app), worst margins")
    print(f"{'pair':10s} {'chunks':>6s} {'inj':>4s} {'miss':>5s}  min over injected of: motion_pos  body  app   (negatives' max: motion body app)")
    for pair, rs in sorted(by_pair.items()):
        inj = [r for r in rs if r["kind"] == "inj"]
        ng = [r for r in rs if r["kind"] == "neg"]
        miss = 0
        for r in inj:
            f, _, _, avail = rule.verdict(r, thr, use_own=False, use_trk=False)
            miss += (not f)
        def mm(key, rr, fn):
            v = np.array([x[key] for x in rr], float)
            v = v[np.isfinite(v)]
            return fn(v) if len(v) else np.nan
        print(f"{str(pair):10s} {len(ng):6d} {len(inj):4d} {miss:5d}        "
              f"{mm('motion_pos', inj, np.min):7.2f} {mm('body', inj, np.min):6.2f} {mm('app', inj, np.min):6.2f}"
              f"      ({mm('motion_pos', ng, np.max):6.2f} {mm('body', ng, np.max):6.2f} {mm('app', ng, np.max):6.2f})")


def by_distance():
    rows = rule.load("saved")
    thr = rule.thresholds(rows)
    print("\nOR-rule (motion|body|app) recall of injected swaps / takeovers by pair separation at the swap instant (thresholds from all negatives)")
    bins = [(0, 0.5), (0.5, 0.75), (0.75, 1.01)]
    for kind in ("inj", "tak"):
        for lo, hi in bins:
            sub = [r for r in rows if r["kind"] == kind and lo <= r.get("d_tau", np.nan) < hi]
            if not sub:
                continue
            flags = [rule.verdict(r, thr, use_own=False, use_trk=False)[0] for r in sub]
            per = {k: np.mean([(np.isfinite(r[k]) and r[k] > thr[k][0]) for r in sub]) for k in ("motion_pos", "body", "app")}
            print(f"  {kind} d_tau in [{lo},{hi}) m: n={len(sub):3d} recall(OR)={np.mean(flags):.3f}  per cue {({k: round(v, 2) for k, v in per.items()})}")
    sub = [r for r in rows if r["kind"] == "inj"]
    print(f"  injected swaps with d_tau unknown: {sum(not np.isfinite(r.get('d_tau', np.nan)) for r in sub)} of {len(sub)}")


def own_and_trk():
    for name in ("saved", "autofit", "baseline"):
        rows = rule.load(name)
        neg = [r for r in rows if r["kind"] == "neg"]
        print(f"\n[{name}] per-camera ownership cue (own_flips = rows fitting the neighbour's 3D pose better by > 0.04 box heights), {len(neg)} chunks")
        print("  k   flagged_chunks  share   real_any_recall  real_boundary_recall  flags/10min")
        for k in (1, 2, 3, 5, 8):
            f = [r for r in neg if r["own_flips"] >= k]
            ra = [r for r in neg if r["real_any"]]
            rb = [r for r in neg if r["real_boundary"]]
            print(f"  {k:2d}  {len(f):8d}  {len(f) / len(neg):6.2f}  {np.mean([r['own_flips'] >= k for r in ra]) if ra else np.nan:10.2f}  "
                  f"{np.mean([r['own_flips'] >= k for r in rb]) if rb else np.nan:12.2f}  {len(f) / 10.18 * 10:8.1f}  "
                  f"(precision for real error: {np.mean([r['real_any'] for r in f]) if f else np.nan:.2f})")
        print(f"  tracklet cue (BoT-SORT tracklet in a separated camera changes identity across the chunk): negatives with contra_sep>=1: "
              f"{np.mean([r['trk_contra_sep'] >= 1 for r in neg]):.2f}, >=2: {np.mean([r['trk_contra_sep'] >= 2 for r in neg]):.2f}, "
              f">=3: {np.mean([r['trk_contra_sep'] >= 3 for r in neg]):.2f}; injected swaps: "
              f">=1 {np.mean([r['trk_contra_sep'] >= 1 for r in rows if r['kind'] == 'inj']):.2f}, "
              f">=2 {np.mean([r['trk_contra_sep'] >= 2 for r in rows if r['kind'] == 'inj']):.2f}, "
              f">=3 {np.mean([r['trk_contra_sep'] >= 3 for r in rows if r['kind'] == 'inj']):.2f}")


if __name__ == "__main__":
    pair_table()
    by_distance()
    own_and_trk()
