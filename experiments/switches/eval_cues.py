"""Compute every cue on every chunk of every close encounter of a golden solution, four ways:

    neg       as is (labels untouched)
    inj       after a label SWAP injected at a random instant inside the chunk (N_INJ per chunk)
    tak       after a TAKEOVER injected (one identity continues on the other person, the other is lost)
    real      the golden label tells whether the chunk holds a real error of that solution (gt.py episodes)

    python experiments/switches/eval_cues.py run saved        # saved | autofit | baseline
    python experiments/switches/eval_cues.py report saved     # AUC / recall tables
Outputs: C:\\data_engine_data\\_scratch\\switches\\cues_<solution>.pkl
"""
from __future__ import annotations

import pickle
import sys
import time
import warnings

import numpy as np

import common
import cues
import encounters
import gt

warnings.filterwarnings("ignore")
N_INJ = 4
CHUNK = 100
R_DEFAULT = 1.0


def make_events(sol, R=R_DEFAULT, k=5, gap_s=3.0, chunk=CHUNK, extra=None):
    out = []
    encs = encounters.pair_encounters(sol.tr_ident, sol.tr_t, sol.tr_root, R, k, gap_s)
    for e in encs:
        e["kind2"] = "3d"
    for e in (extra or []):  # 2D-overlap encounters that are not already a 3D one (same pair, within 3 s)
        if not any({e["a"], e["b"]} == {f["a"], f["b"]} and f["t1"] + 30 >= e["t0"] and f["t0"] - 30 <= e["t1"] for f in encs):
            encs.append({**e, "kind2": "2d"})
    for n, e in enumerate(encs):
        s = e["t0"]
        while s <= e["t1"]:
            en = min(s + chunk - 1, e["t1"])
            if e["t1"] - en < 20:
                en = e["t1"]
            out.append({"enc": n, "a": e["a"], "b": e["b"], "s": s, "e": en, "enc_len": e["t1"] - e["t0"] + 1,
                        "d_min": e["d_min"], "src": e["kind2"]})
            s = en + 1
    return out


def pair_distance(ix, a, b, tau):
    ta, ra, _ = ix.tr(a, tau - 5, tau + 5)
    tb, rb, _ = ix.tr(b, tau - 5, tau + 5)
    common_t = np.intersect1d(ta, tb)
    if not len(common_t):
        return np.nan
    ia, ib = np.searchsorted(ta, common_t), np.searchsorted(tb, common_t)
    return float(np.nanmin(np.linalg.norm(ra[ia] - rb[ib], axis=1)) / common.INCH_PER_M)


def run(name: str, with_real=True, mode="3d"):
    rng = np.random.default_rng(0)
    ctx = cues.Ctx()
    sol = common.get_solution(name, ctx.ws, ctx.scene)
    ix = cues.Index(ctx, sol)
    extra = encounters.pair_encounters_2d(ctx, sol, 0.1, 5, 3.0) if mode == "union" else None
    events = make_events(sol, extra=extra)
    print(f"[{name}] {len(events)} chunk events from {len({e['enc'] for e in events})} encounters", flush=True)
    real = []
    if with_real:
        saved = common.load_saved(ctx.ws, ctx.scene)
        maj = gt.build_ref_maj(ctx.scene, ctx.ref, saved)
        traj = gt.trajectories(ctx.scene, ctx.ref, saved, maj, ctx.T)
        real, _, _ = gt.episodes(ctx.scene, ctx.ref, sol, traj)
    t0 = time.time()
    rows = []
    for i, ev in enumerate(events):
        a, b, s, e = ev["a"], ev["b"], ev["s"], ev["e"]
        base = dict(ev)
        hit = [r for r in real if r["identity"] in (a, b) and r["t1"] >= s - 10 and r["t0"] <= e + 10]
        base["real_any"] = len(hit) > 0
        base["real_obs"] = sum(r["n_obs"] for r in hit)
        base["real_multicam"] = any(r["max_cams_same_second"] >= 3 for r in hit)
        base["real_types"] = sorted({r["type"] for r in hit})
        sw = [r for r in real if r["identity"] in (a, b) and (r["t1"] - r["t0"] >= 30 or r["n_obs"] >= 10)
              and (s - 15 <= r["t0"] <= e + 15 or s - 15 <= r["t1"] <= e + 15)]
        base["real_boundary"] = len(sw) > 0  # a sustained wrong-person run starts or ends in this chunk
        rows.append({**base, "kind": "neg", "tau": -1, **cues.all_cues(ctx, ix, a, b, s, e)})
        for _ in range(N_INJ):
            tau = int(rng.integers(s, e + 1))
            d_tau = pair_distance(ix, a, b, tau)
            rows.append({**base, "kind": "inj", "tau": tau, "d_tau": d_tau,
                         **cues.all_cues(ctx, ix, a, b, s, e, swap=(a, b, tau))})
            who = (a, b) if rng.random() < 0.5 else (b, a)  # who continues on the other person
            rows.append({**base, "kind": "tak", "tau": tau, "d_tau": d_tau,
                         **cues.all_cues(ctx, ix, who[0], who[1], s, e, swap=(who[0], who[1], tau, "take"))})
        if i % 20 == 0:
            print(f"  {i}/{len(events)}  {time.time() - t0:.0f}s", flush=True)
    path = common.SCRATCH / (f"cues_{name}.pkl" if mode == "3d" else f"cues_{name}_{mode}.pkl")
    path.write_bytes(pickle.dumps({"rows": rows, "events": events,
                                   "episodes": [{k: v for k, v in r.items() if k != "rows"} for r in real]}))
    print("saved", path)


# ---------------------------------------------------------------------- statistics
def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="stable")
    sv = allv[order]
    r = np.empty(len(allv))
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def recall_at_fpr(pos, neg, fpr):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan, np.nan
    thr = np.quantile(neg, 1 - fpr)
    return float((pos > thr).mean()), float(thr)


SCORES = ["motion_pos", "motion_cv", "body", "app", "app_samecam", "trk_swap_score", "trk_contra_sep", "raw_cross",
          "raw_changes", "raw_birthdeath", "raw_lost_frames", "own_flips", "own_flip_share", "own_worst"]


def col(rows, key, kind=None, mask=None):
    return np.array([r.get(key, np.nan) for r in rows
                     if (kind is None or r["kind"] == kind) and (mask is None or mask(r))], float)


def report(name: str):
    d = pickle.loads((common.SCRATCH / f"cues_{name.replace(':', '_')}.pkl").read_bytes())
    rows = d["rows"]
    neg_rows = [r for r in rows if r["kind"] == "neg"]
    print(f"\n##### {name}: {len(neg_rows)} chunk events; injected per kind: {sum(r['kind'] == 'inj' for r in rows)}")
    print(f"chunks with a real error: any {sum(r['real_any'] for r in neg_rows)}, multi-camera "
          f"{sum(r['real_multicam'] for r in neg_rows)}, sustained-run boundary {sum(r['real_boundary'] for r in neg_rows)}")
    strata = (("ALL chunks", lambda r: True),
              ("short encounters (<10 s)", lambda r: r["enc_len"] < 100),
              ("long encounters (>=10 s)", lambda r: r["enc_len"] >= 100),
              ("injected while pair < 0.6 m apart", lambda r: r.get("d_tau", 9) < 0.6),
              ("injected while pair 0.6-1.0 m apart", lambda r: r.get("d_tau", 0) >= 0.6))
    for kind_name, kind in (("SWAP (A<->B exchanged)", "inj"), ("TAKEOVER (A continues on B's person, B lost)", "tak")):
        for label, mask in strata:
            print(f"\n--- INJECTED {kind_name}; {label}  (score: larger = swap evidence)")
            print(f"{'cue':16s} {'avail_pos':>9s} {'avail_neg':>9s} {'AUC':>6s} {'rec@2%':>7s} {'rec@5%':>7s} {'rec@10%':>8s}  thr@5%")
            for key in SCORES:
                pos, neg = col(rows, key, kind, mask), col(rows, key, "neg")
                r2, _ = recall_at_fpr(pos, neg, 0.02)
                r5, t5 = recall_at_fpr(pos, neg, 0.05)
                r10, _ = recall_at_fpr(pos, neg, 0.10)
                print(f"{key:16s} {np.isfinite(pos).mean():9.2f} {np.isfinite(neg).mean():9.2f} {auc(pos, neg):6.3f}"
                      f" {r2:7.2f} {r5:7.2f} {r10:8.2f}  {t5:.3f}")
    print("\n--- REAL errors of this solution (golden labels)")
    for label, key in (("any wrong obs in chunk", "real_any"), ("multi-camera wrong at once", "real_multicam"),
                       ("start/end of a sustained wrong run", "real_boundary")):
        npos = sum(r[key] for r in neg_rows)
        if npos == 0:
            print(f"{label}: no positive chunks")
            continue
        print(f"{label}: {npos} positive chunks vs {sum(not r['real_any'] for r in neg_rows)} clean chunks")
        print(f"{'cue':16s} {'AUC':>6s} {'rec@5%':>7s} {'rec@10%':>8s}")
        for key2 in SCORES:
            pos = col(neg_rows, key2, None, lambda r: r[key])
            neg = col(neg_rows, key2, None, lambda r: not r["real_any"])
            r5, _ = recall_at_fpr(pos, neg, 0.05)
            r10, _ = recall_at_fpr(pos, neg, 0.10)
            print(f"{key2:16s} {auc(pos, neg):6.3f} {r5:7.2f} {r10:8.2f}")


if __name__ == "__main__":
    cmd, names = sys.argv[1], sys.argv[2:] or ["saved"]
    for n in names:
        if cmd == "run":
            run(n.split(":")[0], mode=(n.split(":") + ["3d"])[1])
        else:
            report(n)
