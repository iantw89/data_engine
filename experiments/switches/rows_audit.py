"""Row-level audits on the golden saved solution (golden labels used for scoring only).

 1. pose ownership: for every labelled detection of identity X with another identity within 1.5 m, compare the
    normalised reprojection error of X's 3D pose (own) with that of the nearest neighbour's pose (other).
    Score = own - other (box heights). Wrong = reference person differs from X's majority person.
 2. tracklet/engine disagreement: BoT-SORT tracklets whose detections change engine identity. Who is right?

    python experiments/switches/rows_audit.py saved
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter, defaultdict

import numpy as np

import common
import cues
import eval_cues
import gt
from common import INCH_PER_M

warnings.filterwarnings("ignore")


def ownership(ctx, ix, sol, maj, neighbour_m=1.5):
    ok = np.isfinite(ix.tr_root[:, 0])
    t_all, id_all, root_all = ix.tr_t[ok], ix.tr_ident[ok], ix.tr_root[ok]
    o = np.argsort(t_all, kind="stable")
    t_all, id_all, root_all = t_all[o], id_all[o], root_all[o]
    lo = np.searchsorted(t_all, np.arange(ctx.T + 2))
    by_pair = defaultdict(list)
    for c in ctx.scene.cameras:
        rows = np.flatnonzero((ctx.ref.person[c.index] >= 0) & (sol.ident[c.index] >= 0) & (c.t >= 0))
        for r in rows:
            x, t = int(sol.ident[c.index][r]), int(c.t[r])
            a, b = lo[t], lo[t + 1]
            if b - a < 2:
                continue
            ids, roots = id_all[a:b], root_all[a:b]
            mine = np.flatnonzero(ids == x)
            if not len(mine):
                continue
            d = np.linalg.norm(roots - roots[mine[0]], axis=1) / INCH_PER_M
            d[mine[0]] = 9
            d[ids == x] = 9
            j = int(np.argmin(d))
            if d[j] < neighbour_m:
                by_pair[(x, int(ids[j]))].append((c.index, int(r), t, d[j], int(ctx.ref.person[c.index][r] != maj[x])))
    out = []
    for (x, y), items in by_pair.items():
        arr = np.array(items, float)
        own, oth = cues.row_ownership(ctx, ix, x, y, arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2].astype(int))
        for k in range(len(items)):
            out.append((own[k], oth[k], arr[k, 3], arr[k, 4], x, y, int(arr[k, 0]), int(arr[k, 1]), int(arr[k, 2])))
    return np.array(out, float)


def tracklet_audit(ctx, sol, maj, ix, near_m=1.5):
    res = Counter()
    examples = []
    for c in ctx.scene.cameras:
        ident, trk, t = sol.ident[c.index], c.tracklet, c.t
        rows = np.flatnonzero((trk >= 0) & (ident >= 0) & (t >= 0))
        order = rows[np.lexsort((t[rows], trk[rows]))]
        starts = np.flatnonzero(np.concatenate(([True], trk[order][1:] != trk[order][:-1])))
        ends = np.concatenate((starts[1:], [len(order)]))
        for s, e in zip(starts, ends):
            seq = order[s:e]
            ids = ident[seq]
            if len(set(ids.tolist())) < 2:
                continue
            res["tracklets_with_identity_change"] += 1
            # runs
            change = np.flatnonzero(ids[1:] != ids[:-1]) + 1
            res["identity_change_points"] += len(change)
            persons = ctx.ref.person[c.index][seq]
            for k in change:
                before = [p for p in persons[max(0, k - 20):k] if p >= 0]
                after = [p for p in persons[k:k + 20] if p >= 0]
                if not before or not after:
                    res["change_unlabelled"] += 1
                    continue
                pb, pa = Counter(before).most_common(1)[0][0], Counter(after).most_common(1)[0][0]
                xb, xa = int(ids[k - 1]), int(ids[k])
                same_person = pb == pa
                # engine says different identities; truth says same person -> identity fragmentation/relink
                if same_person:
                    res["truth_same_person (engine fragments one person or hand-off)"] += 1
                else:
                    res["truth_different_people (2D tracker switched person)"] += 1
                    engine_follows_truth = maj.get(xb) == pb and maj.get(xa) == pa
                    res["  of which engine identities follow the truth"] += int(engine_follows_truth)
                    examples.append((c.camera, int(trk[seq[k]]), int(t[seq[k]]), xb, xa, pb, pa))
    return res, examples


if __name__ == "__main__":
    name = (sys.argv[1:] or ["saved"])[0]
    ctx = cues.Ctx(with_appearance=False)
    sol = common.get_solution(name, ctx.ws, ctx.scene)
    ix = cues.Index(ctx, sol)
    maj = gt.build_ref_maj(ctx.scene, ctx.ref, sol)
    arr = ownership(ctx, ix, sol, maj)
    own, oth, d, wrong = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    ok = np.isfinite(own) & np.isfinite(oth)
    print(f"[{name}] rows near a neighbour (<1.5 m): {len(arr)}; with both poses: {ok.sum()}; wrong: {int(wrong[ok].sum())}")
    score = (own - oth)
    print("score = own - other error (box heights); AUC row-level:", round(eval_cues.auc(score[ok & (wrong == 1)], score[ok & (wrong == 0)]), 3))
    pos, neg = score[ok & (wrong == 1)], score[ok & (wrong == 0)]
    print(f"wrong rows: median score {np.median(pos):.3f}, clean rows: median {np.median(neg):.3f}, p99 clean {np.quantile(neg, .99):.3f}")
    print("margin  flagged_rows  flagged_wrong  precision  recall_of_wrong  flagged_per_10min")
    minutes = ctx.T / 10 / 60
    for m in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.2, 0.3):
        f = ok & (score > m)
        tp = int((f & (wrong == 1)).sum())
        print(f"{m:5.2f}  {int(f.sum()):8d}  {tp:8d}  {tp / max(1, f.sum()):8.3f}  {tp / max(1, wrong[ok].sum()):8.2f}  {f.sum() / minutes * 10:10.1f}")
    # distance breakdown for wrong rows
    print("\nwrong rows without both poses available (cannot be judged by this cue):", int(((wrong == 1) & ~ok).sum()))
    res, ex = tracklet_audit(ctx, sol, maj, ix)
    print("\nBoT-SORT tracklets whose detections change engine identity (golden labels judge who is right):")
    for k, v in res.items():
        print(f"  {k}: {v}")
    for e in ex[:15]:
        print("   ", e)
