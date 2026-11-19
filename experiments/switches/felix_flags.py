"""How many chunks would the flagging rule raise on the felix hour? (read-only; copies of round files in scratch)

Thresholds are learned on golden (rule.thresholds, appearance cue off because felix has no embeddings).
No labels exist for felix: this only measures review load.

    python experiments/switches/felix_flags.py [3d|union]
"""
from __future__ import annotations

import pickle
import shutil
import sys
import time
import warnings

import numpy as np

import common
import cues
import encounters
import eval_cues
import rule
from common import FPS

warnings.filterwarnings("ignore")
FELIX = "felix_20251223173052_20251223183103"


def felix_solution(ctx):
    root = common.DATA / FELIX / "rounds"
    latest = sorted(p for p in root.iterdir() if (p / "tracks3d.npz").is_file() and (p / "solution.npz").is_file())[-1]
    out = {}
    for fname in ("solution.npz", "tracks3d.npz"):
        dst = common.SCRATCH / f"felix_{latest.name}_{fname}"
        if not dst.is_file():
            shutil.copyfile(latest / fname, dst)
        out[fname] = dst
    with np.load(out["solution.npz"]) as s:
        ident = [s[f"identity_{c.camera}"].astype(np.int64) for c in ctx.scene.cameras]
    with np.load(out["tracks3d.npz"]) as t:
        sol = common.Sol("felix", ident, t["identity"], t["t"], t["root"], t["xyz"], t["rows"])
    return latest.name, sol


def main(mode="3d"):
    ctx = cues.Ctx(workspace=FELIX)
    rname, sol = felix_solution(ctx)
    ix = cues.Index(ctx, sol)
    extra = encounters.pair_encounters_2d(ctx, sol, 0.1, 5, 3.0) if mode == "union" else None
    events = eval_cues.make_events(sol, extra=extra)
    hours = ctx.T / FPS / 3600
    if (common.SCRATCH / f"felix_cues_{mode}.pkl").is_file():
        events = events
    print(f"felix {rname}: T={ctx.T} ({hours * 60:.1f} min); {len(events)} chunk events "
          f"({sum(e['src'] == '3d' for e in events)} from 3D<1m encounters, {sum(e['src'] == '2d' for e in events)} from 2D-only overlaps)", flush=True)
    t0 = time.time()
    rows = []
    cache = common.SCRATCH / f"felix_cues_{mode}.pkl"
    if cache.is_file() and "--fresh" not in sys.argv:
        rows = pickle.loads(cache.read_bytes())
        events = []
    for i, ev in enumerate(events):
        rows.append({**ev, "kind": "neg", **cues.all_cues(ctx, ix, ev["a"], ev["b"], ev["s"], ev["e"], which=("motion", "body", "trk", "raw", "own"))})
        if i % 100 == 0:
            print(f"  {i}/{len(events)} {time.time() - t0:.0f}s", flush=True)
    pickle.dump(rows, open(cache, "wb"))
    gold = rule.load("saved")
    thr = rule.thresholds(gold, use_app=False)
    print("golden-learned thresholds:", {k: tuple(round(x, 3) for x in v) for k, v in thr.items()})
    for label, kw in (("motion|body|trk>=1", dict(use_own=False)), ("motion|body|trk>=2", dict(use_own=False, trk_k=2)),
                      ("motion|body", dict(use_own=False, use_trk=False)),
                      ("2-of-3 agree (motion,body,trk>=1)", dict(use_own=False, min_causes=2)),
                      ("2-of-3 agree, trk>=2", dict(use_own=False, min_causes=2, trk_k=2)),
                      ("motion|body|trk>=1|own>=3", dict(own_k=3)), ("motion|body|trk>=2|own>=5", dict(own_k=5, trk_k=2))):
        flagged, causes, certified, unver = 0, {}, 0, 0
        for r in rows:
            f, c, stay, avail = rule.verdict(r, thr, use_app=False, **kw)
            flagged += f
            for k in c:
                causes[k] = causes.get(k, 0) + 1
            certified += (not f) and stay >= 2
            unver += (not f) and avail < 2
        print(f"  rule {label:28s}: flagged chunks {flagged:4d} ({flagged / hours:6.1f}/h, {flagged / len(rows):.1%} of chunks)  "
              f"certified-stay {certified / len(rows):.1%}  unverifiable {unver / len(rows):.1%}  causes {causes}")
    # 10-s chunks collapsed to encounters: how many distinct encounters have >= 1 flagged chunk
    fl_enc = {r["enc"] for r in rows if rule.verdict(r, thr, use_app=False, use_own=False)[0]}
    print(f"  encounters with >=1 flagged chunk (motion|body|trk>=1): {len(fl_enc)} of {len({r['enc'] for r in rows})} ({len(fl_enc) / hours:.1f}/h)")


if __name__ == "__main__":
    main(*([a for a in sys.argv[1:] if not a.startswith("--")] or ["3d"]))
