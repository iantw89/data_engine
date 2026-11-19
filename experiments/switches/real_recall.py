"""Recall of the enumeration and of the flagging rule on the REAL golden error episodes of each solution,
and the list of flagged chunks of the saved solution with what the golden labels say about them.

    python experiments/switches/real_recall.py
"""
from __future__ import annotations

import contextlib
import io
import pickle

import numpy as np

import common
import cues
import encounters
import gt
import rule
from common import FPS


def covered(ep, encs, pad=30):
    return any((e["a"] == ep["identity"] or e["b"] == ep["identity"]) and e["t1"] + pad >= ep["t0"] and e["t0"] - pad <= ep["t1"]
               for e in encs)


def main():
    ctx = cues.Ctx(with_appearance=False)
    saved = common.load_saved(ctx.ws, ctx.scene)
    maj_ref = gt.build_ref_maj(ctx.scene, ctx.ref, saved)
    traj = gt.trajectories(ctx.scene, ctx.ref, saved, maj_ref, ctx.T)
    for name in ("saved", "autofit", "baseline"):
        sol = common.get_solution(name, ctx.ws, ctx.scene)
        eps, maj, life = gt.episodes(ctx.scene, ctx.ref, sol, traj)
        edge = encounters.edge_events(sol.tr_ident, sol.tr_t, sol.tr_root, ctx.T, 1.0)
        ed = [{"a": e["a"], "b": e["b"], "t0": e["t0"], "t1": e["t1"]} for e in edge]
        print(f"\n[{name}] {len(eps)} error episodes ({sum(e['n_obs'] for e in eps)} wrong obs)")
        print("  covered by a pair encounter of the solution (either identity involved, +-3 s):")
        for R in (0.5, 0.75, 1.0):
            encs = encounters.pair_encounters(sol.tr_ident, sol.tr_t, sol.tr_root, R, 5, 3.0)
            c = [covered(e, encs) for e in eps]
            cw = sum(e["n_obs"] for e, k in zip(eps, c) if k)
            print(f"    R={R}: episodes {sum(c)}/{len(eps)}, wrong observations {cw}/{sum(e['n_obs'] for e in eps)}")
        encs1 = encounters.pair_encounters(sol.tr_ident, sol.tr_t, sol.tr_root, 1.0, 5, 3.0)
        c = [covered(e, encs1 + ed) for e in eps]
        print(f"    R=1.0 + edge events (death/birth near another identity, hand-off): episodes {sum(c)}/{len(eps)}")
        rest = [e for e, k in zip(eps, c) if not k]
        for e in rest:
            print(f"      not covered: t={e['t0']/FPS:.0f}-{e['t1']/FPS:.0f}s id{e['identity']} (maj P{e['majority']}) carries P{e['true_person']} "
                  f"{e['type']} obs={e['n_obs']} d_min={e['min_dist_m']:.2f} iou={e['max_iou']:.2f}")
        # 2D overlap in a camera as alternative closeness definition
        iou_close = [e for e in eps if e["max_iou"] > 0.1]
        print(f"    box overlap (IoU>0.1 in some camera, +-3 s): episodes {len(iou_close)}/{len(eps)}")
        # rule flags
        rows = rule.load(name)
        with contextlib.redirect_stdout(io.StringIO()):
            out = rule.evaluate(rows, name, use_own=True)
            out2 = rule.evaluate(rows, name, use_own=False)
        for label, o in (("rule with ownership cue", out), ("rule without ownership cue", out2)):
            chunks = [r for r in o if r["kind"] == "neg"]
            hit = []
            for e in eps:
                hit.append(any(c["flag"] and (c["a"] == e["identity"] or c["b"] == e["identity"]) and c["e"] + 10 >= e["t0"] and c["s"] - 10 <= e["t1"]
                               for c in chunks))
            hw = sum(e["n_obs"] for e, k in zip(eps, hit) if k)
            print(f"  {label}: episodes flagged {sum(hit)}/{len(eps)}; by type " +
                  ", ".join(f"{t}: {sum(k for e, k in zip(eps, hit) if e['type'] == t)}/{sum(e['type'] == t for e in eps)}"
                            for t in ("swap", "takeover", "contamination")) + f"; wrong obs inside flagged chunks {hw}/{sum(e['n_obs'] for e in eps)}")
        if name == "saved":
            print("\n  flagged chunks of the saved solution (rule without ownership cue), with golden truth:")
            for c in sorted([r for r in out2 if r["kind"] == "neg" and r["flag"]], key=lambda r: r["s"]):
                pa, pb = maj_ref.get(c["a"], -1), maj_ref.get(c["b"], -1)
                print(f"    t={c['s']/FPS:6.1f}-{c['e']/FPS:6.1f}s ids {c['a']}/{c['b']} (P{pa}/P{pb}) causes={c['causes']} "
                      f"real_error_in_chunk={c['real_any']} ({c['real_obs']} obs, {c['real_types']}) motion={c['motion_pos']:.2f} body={c['body']:.2f} app={c['app']:.2f} "
                      f"trk_contra={c['trk_contra_sep']:.0f}")


if __name__ == "__main__":
    main()
