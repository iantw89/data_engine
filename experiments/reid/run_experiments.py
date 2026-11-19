"""Experiments A-C against the golden reference. Writes results.json, cases/cases.json and prints tables."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from lib import *     # noqa

OUT = Path(__file__).parent
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
EMB = Path(sys.argv[1]) if len(sys.argv) > 1 else EMB_DIR
D = load_data(emb_dir=EMB)
T = D.T
R = {}
print(f"embedded detections {len(D.t)}, labelled {int((D.person >= 0).sum())}, people {len(np.unique(D.person[D.person >= 0]))}, T={T}")
for c, name in enumerate(D.cameras):
    m = D.cam == c
    print(f"  {name}: {m.sum()} dets, {(D.person[m] >= 0).sum()} labelled")
R["data"] = {"embedded": int(len(D.t)), "labelled": int((D.person >= 0).sum()), "T": int(T)}


def fmt(x):
    return f"{x:.4f}"


# ------------------------------------------------------------------ A. separability
def separability(F: Frags, tag):
    sim = F.desc @ F.desc.T
    iu = np.triu_indices(len(F), 1)
    s = sim[iu]
    same_p = F.person[iu[0]] == F.person[iu[1]]
    same_c = F.cam[iu[0]] == F.cam[iu[1]]
    disjoint = (F.t1[iu[0]] < F.t0[iu[1]]) | (F.t1[iu[1]] < F.t0[iu[0]])
    diff_trk = F.trk[iu[0]] != F.trk[iu[1]]
    simult = F.win[iu[0]] == F.win[iu[1]]
    groups = {
        "within-camera (same: other tracklet, time-disjoint; diff: all pairs)":
            (same_p & same_c & diff_trk & disjoint, ~same_p & same_c),
        "cross-camera, all (same: any time)": (same_p & ~same_c, ~same_p & ~same_c),
        "cross-camera, simultaneous 10 s window": (same_p & ~same_c & simult, ~same_p & ~same_c & simult),
        "cross-camera, non-simultaneous": (same_p & ~same_c & ~simult, ~same_p & ~same_c & ~simult),
    }
    rows = []
    for name, (ms, md) in groups.items():
        a, b = s[ms], s[md]
        th0 = reid.threshold_for_false_accepts(a, b, 0.0)
        th1 = reid.threshold_for_false_accepts(a, b, 0.001)
        rows.append({"group": name, "n_same": int(ms.sum()), "n_diff": int(md.sum()), "auc": reid.roc_auc(a, b),
                     "same_mean": float(a.mean()), "diff_mean": float(b.mean()), "diff_p99": float(np.quantile(b, .99)),
                     "diff_max": float(b.max()), "th_far0": th0, "tar_far0": float((a >= th0).mean()),
                     "th_far0.1pct": th1, "tar_far0.1pct": float((a >= th1).mean())})
    print(f"\n== A. separability [{tag}]  fragments={len(F)} (10 s windows of tracklets, >= {MIN_FRAG} labelled dets; "
          f"ambiguous dropped={F.dropped_ambiguous})")
    for r in rows:
        print(f"  {r['group']}\n    same={r['n_same']} diff={r['n_diff']} AUC={fmt(r['auc'])} same_mean={fmt(r['same_mean'])} "
              f"diff_mean={fmt(r['diff_mean'])} diff_p99={fmt(r['diff_p99'])} diff_max={fmt(r['diff_max'])}\n"
              f"    FAR=0: s>={fmt(r['th_far0'])} accepts {r['tar_far0']:.1%} of same;  FAR<=0.1%: s>={fmt(r['th_far0.1pct'])} "
              f"accepts {r['tar_far0.1pct']:.1%}")
    return rows


R["A"] = {}
for wmode in ("none", "quality"):
    F = build_fragments(D, det_weights(D, wmode))
    R["A"][wmode] = separability(F, f"weights={wmode}")
    if wmode == "none":
        F0 = F


# train / held-out separability (thresholds frozen from the first half) for the pairwise score
def sep_split(F):
    sim = F.desc @ F.desc.T
    iu = np.triu_indices(len(F), 1)
    s = sim[iu]
    same_p = F.person[iu[0]] == F.person[iu[1]]
    cross = F.cam[iu[0]] != F.cam[iu[1]]
    tr = (F.t1[iu[0]] < 3000) & (F.t1[iu[1]] < 3000)
    ho = (F.t0[iu[0]] >= T - 3000) & (F.t0[iu[1]] >= T - 3000)
    out = {}
    for name, sel in (("within-camera", ~cross), ("cross-camera", cross)):
        a, b = s[tr & sel & same_p], s[tr & sel & ~same_p]
        th = reid.threshold_for_false_accepts(a, b, 0.0)
        a2, b2 = s[ho & sel & same_p], s[ho & sel & ~same_p]
        out[name] = {"th_far0_train": th, "train_tar": float((a >= th).mean()), "heldout_false_accepts": int((b2 >= th).sum()),
                     "heldout_diff_pairs": int(len(b2)), "heldout_tar": float((a2 >= th).mean()), "heldout_auc": reid.roc_auc(a2, b2)}
    print("  A. frozen pair threshold (FAR=0 on train fragments), held-out fragments:", json.dumps(out))
    return out


R["A_split"] = sep_split(F0)

# ------------------------------------------------------------------ B. use cases
sol = np.load(D.ws.path("rounds", "r000", "solution.npz"))
SINGLE = np.zeros(len(D.t), bool)
for ci, cname in enumerate(D.cameras):
    m = D.cam == ci
    SINGLE[m] = sol["identity_" + cname][D.row[m]] < 0
G = make_gallery(D, "none")
Gq = make_gallery(D, "quality")
cases = {
    "B1_all": build_b1(D, G, scope="all", name="B1 identity across gap, multi-camera descriptors"),
    "B1_cam": build_b1(D, G, scope="cam", name="B1 identity across gap, same-camera descriptors"),
    "B1_all_rec60": build_b1(D, G, scope="all", recency=600, name="B1 multi-camera, last 60 s only"),
    "B2": build_b2(D, F0, name="B2 rival tie-break (simultaneous candidates in other camera)"),
    "B3": build_b3(D, G, name="B3 single-camera pieces (engine r000: no multi-camera identity) vs gallery of +-60 s",
                   single=SINGLE),
    "B3_ref": build_b3(D, G, name="B3 single-camera pieces (reference: no other camera sees the person) vs +-60 s"),
}
G_w[id(Gq)] = det_weights(D, "quality")
cases["B1_all_q"] = build_b1(D, Gq, scope="all", name="B1 multi-camera, quality-weighted")


def basic(C: Cases):
    best, gap, bcol = top2(C)
    has = C.true >= 0
    tr, ho = is_train(C), is_heldout(C, T)
    r1 = lambda m: (float((bcol[m & has] == C.true[m & has]).mean()), int((m & has).sum())) if (m & has).any() else (None, 0)
    out = {"queries": len(C), "true_in_candidates": int(has.sum()), "new_person_cases": int((~has).sum()),
           "rank1_all": r1(np.ones(len(C), bool)), "rank1_train": r1(tr), "rank1_heldout": r1(ho),
           "median_candidates": float(np.median(np.isfinite(C.sim).sum(1)))}
    out["rank1_per_camera"] = {D.cameras[c]: r1(C.cam == c) for c in range(len(D.cameras))}
    if C.meta and C.meta[0].get("gap", 0) is not None and "gap" in C.meta[0]:
        gaps = np.array([m["gap"] if m["gap"] is not None else 10 ** 9 for m in C.meta])
        for lo, hi, nm in ((0, 50, "<5s"), (50, 600, "5-60s"), (600, 10 ** 8, ">60s")):
            out[f"rank1_gap_{nm}"] = r1((gaps >= lo) & (gaps < hi))
    print(f"\n== {C.name}: queries={out['queries']} (truth is a candidate in {out['true_in_candidates']}, new/unseen "
          f"person in {out['new_person_cases']}); median candidates {out['median_candidates']}")
    print(f"   rank-1: all {out['rank1_all']}  train {out['rank1_train']}  held-out {out['rank1_heldout']}")
    print("   per camera:", {k: (None if v[0] is None else round(v[0], 3), v[1]) for k, v in out['rank1_per_camera'].items()})
    for k in out:
        if k.startswith("rank1_gap"):
            print(f"   {k}: {out[k]}")
    return out


R["B"] = {}
for k, C in cases.items():
    R["B"][k] = basic(C)

# ------------------------------------------------------------------ C. calibration: freeze on train, report held-out
PRIOR = (0.908, 0.05)   # earlier manual-effort rule (own-score floor, margin); descriptors differ, shown for reference
R["C"] = {}
listing = []
for k, C in cases.items():
    tr, ho = is_train(C), is_heldout(C, T)
    rules = choose_rule(C, tr)
    print(f"\n== C. {C.name}: train queries={int(tr.sum())}, held-out queries={int(ho.sum())}")
    R["C"][k] = {"train_n": int(tr.sum()), "heldout_n": int(ho.sum()), "rules": {}}
    for mutual in (False, True):
        for label in ("strict", "robust"):
            pick = rules[mutual][label]
            tag = f"{label}{'+mutual' if mutual else ''}"
            if pick is None:
                print(f"   {tag}: no zero-wrong rule on train")
                R["C"][k]["rules"][tag] = None
                continue
            s, m, _ = pick
            a, b = apply_rule(C, tr, s, m, mutual), apply_rule(C, ho, s, m, mutual)
            print(f"   {tag:16s} s_min={s:.2f} margin={m:.2f} | train decided {a['decided']}/{a['queries']} wrong {a['wrong']} "
                  f"| HELD-OUT decided {b['decided']}/{b['queries']} correct {b['correct']} wrong {b['wrong']} open {b['open']}")
            R["C"][k]["rules"][tag] = {"s_min": s, "margin": m, "train": {x: y for x, y in a.items() if x != 'wrong_idx'},
                                       "heldout": {x: y for x, y in b.items() if x != 'wrong_idx'}}
            for split, res in (("train", a), ("heldout", b)):
                for i in res["wrong_idx"]:
                    listing.append({"usecase": k, "rule": tag, "split": split, "case": int(i)})
    a, b = apply_rule(C, tr, *PRIOR, False), apply_rule(C, ho, *PRIOR, False)
    print(f"   prior rule floor .908/margin .05: train decided {a['decided']} wrong {a['wrong']} | held-out decided "
          f"{b['decided']} wrong {b['wrong']}")
    R["C"][k]["prior_rule"] = {"train": {x: y for x, y in a.items() if x != 'wrong_idx'},
                               "heldout": {x: y for x, y in b.items() if x != 'wrong_idx'}}
    # oracle: best zero-wrong rule chosen on held-out itself (upper bound of what any frozen rule could reach)
    dec, cor, wr = grid_counts(C, ho, False)
    ok = np.where(wr == 0, dec, -1)
    print(f"   oracle zero-wrong on held-out (no freezing, optimistic): decided {int(ok.max())}/{int(ho.sum())}")
    R["C"][k]["oracle_heldout_decided"] = int(ok.max())
    # decided / wrong curve over margin at selected s_min on the held-out, no mutual (for the report)
    curve = []
    for s in (0.5, 0.6, 0.7, 0.8, 0.9):
        for m in (0.0, 0.05, 0.1, 0.2):
            x, y = apply_rule(C, tr, s, m, False), apply_rule(C, ho, s, m, False)
            curve.append({"s_min": s, "margin": m, "train_decided": x["decided"], "train_wrong": x["wrong"],
                          "heldout_decided": y["decided"], "heldout_wrong": y["wrong"]})
    R["C"][k]["curve"] = curve

# 3-fold across cameras (query camera held out): thresholds from queries of the other cameras, evaluated on the fold
ncam = len(D.cameras)
folds = [[0, 3, 6], [1, 4], [2, 5]]
R["C3"] = {}
print("\n== C. 3-fold across query cameras (folds", [[D.cameras[c] for c in f] for f in folds], ")")
for k in ("B1_all", "B1_cam", "B2", "B3"):
    if len(cases[k]) < 20: continue
    C = cases[k]
    tot = {}
    for mutual in (False,):
        for label in ("strict", "robust"):
            dec = cor = wrong = open_ = 0
            picks = []
            for f in folds:
                test = np.isin(C.cam, f)
                pick = choose_rule(C, ~test)[mutual][label]
                if pick is None:
                    open_ += int(test.sum()); picks.append(None); continue
                r = apply_rule(C, test, pick[0], pick[1], mutual)
                dec += r["decided"]; cor += r["correct"]; wrong += r["wrong"]; open_ += r["open"]
                picks.append(pick[:2])
            tot[label] = {"decided": dec, "correct": cor, "wrong": wrong, "open": open_, "picks": picks}
            print(f"   {k:7s} {label:7s}: decided {dec} correct {cor} wrong {wrong} open {open_}  rules {picks}")
    R["C3"][k] = tot

# ------------------------------------------------------------------ wrong-case listing
cases_out = []
for item in listing:
    C = cases[item["usecase"]]
    i = item["case"]
    m = C.meta[i]
    best, gap, bcol = top2(C)
    pred = int(C.cand_person[i, bcol[i]]) if C.cand_person is not None else None
    q = m["query"]
    cases_out.append({**item, "query_key": D.key(q), "camera": D.cameras[m["cam"]], "tracklet": m["tracklet"],
                      "time_s": round(m["t0"] / SYNC_HZ, 1), "true_person": m["person"], "predicted_person": pred,
                      "truth_was_candidate": bool(C.true[i] >= 0), "best_sim": round(float(best[i]), 4),
                      "gap_to_second": None if not np.isfinite(gap[i]) else round(float(gap[i]), 4),
                      "query_det_index": int(q), "pred_col": int(bcol[i]), "true_col": int(C.true[i])})
(OUT / "cases").mkdir(exist_ok=True)
(OUT / "cases" / f"cases{TAG}.json").write_text(json.dumps(cases_out, indent=1), encoding="utf-8")
(OUT / f"results{TAG}.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")
np.savez_compressed(SCRATCH / "case_arrays.npz", **{f"{k}_{n}": getattr(C, n) for k, C in cases.items()
                                                     for n in ("sim", "true", "mutual", "t", "cam")})
print("\nwrong-case entries:", len(cases_out))
