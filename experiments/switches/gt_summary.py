"""Summary tables of gt_episodes_<solution>.json (written by gt.py)."""
import json
import numpy as np
import common
for name in ("saved", "autofit", "baseline"):
    eps = json.loads((common.SCRATCH / f"gt_episodes_{name}.json").read_text())
    n = len(eps); obs = sum(e["n_obs"] for e in eps)
    print(f"\n[{name}] {n} episodes, {obs} wrong obs")
    for typ in ("swap", "takeover", "contamination", "ALL"):
        sub = [e for e in eps if typ == "ALL" or e["type"] == typ]
        if not sub: continue
        d = np.array([e["min_dist_m"] if e["min_dist_m"] is not None else np.nan for e in sub], float)
        iou = np.array([e["max_iou"] for e in sub])
        print(f"  {typ:14s} n={len(sub):3d} obs={sum(e['n_obs'] for e in sub):5d} "
              f"d<0.5:{np.sum(d<.5):3d} <0.75:{np.sum(d<.75):3d} <1.0:{np.sum(d<1.0):3d} nan:{np.sum(np.isnan(d)):2d} | "
              f"obs in d<1.0: {sum(e['n_obs'] for e, x in zip(sub, d) if x < 1.0):5d} | IoU>0.1: {np.sum(iou>.1):3d} | "
              f"1-camera episodes: {sum(e['max_cams_same_second']==1 for e in sub):3d} | >=3 cams at once: {sum(e['max_cams_same_second']>=3 for e in sub):2d} | "
              f"median cams labelled both people: {np.median([e['cams_both'] for e in sub]):.0f} | median duration {np.median([(e['t1']-e['t0'])/10 for e in sub]):.1f}s, max {max((e['t1']-e['t0'])/10 for e in sub):.0f}s")
    far = [e for e in eps if e["min_dist_m"] is None or not (e["min_dist_m"] < 1.0)]
    print(f"  errors NOT within 1 m: {len(far)} episodes, {sum(e['n_obs'] for e in far)} obs; IoU>0.1: {sum(e['max_iou']>0.1 for e in far)}; "
          f"types {dict(zip(*np.unique([e['type'] for e in far], return_counts=True)))}")
    trio = [e for e in eps if {e['majority'], e['true_person']} <= {1, 4, 10}]
    print(f"  episodes between persons {{1,4,10}} (the standing trio): {len(trio)} of {n}, obs {sum(e['n_obs'] for e in trio)} of {obs}")
