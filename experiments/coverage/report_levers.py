"""Sweep acceptance gates over eval_<name>.pkl: yield on gaps / withhold, false accepts on negatives.

Usage: python experiments/coverage/report_levers.py [eval_name]
"""
import itertools
import json
import pickle
import sys
from collections import Counter, defaultdict

from common import *  # noqa
from gap_classify import BUCKETS, bucket
from recover_lib import root_of

name = sys.argv[1] if len(sys.argv) > 1 else "vit"
ws, scene, tracks, solution = load()
cases = pickle.loads((SCRATCH / "cases.pkl").read_bytes())
ev = pickle.loads((SCRATCH / f"eval_{name}.pkl").read_bytes())
evmap = {e["id"]: e for e in ev}
pairs = [(c, evmap[c["id"]]) for c in cases if c["id"] in evmap]
by_t = defaultdict(list)
for row in np.flatnonzero(np.isfinite(tracks["root"]).all(axis=1)):
    by_t[int(tracks["t"][row])].append(row)


def nearest_person(t, root):
    best = np.inf
    for r in by_t.get(t, []):
        best = min(best, float(np.linalg.norm(tracks["root"][r][:2] - root[:2])))
    return best


def make_gate(min_body=6, max_resid=12.0, min_pair=6, radius=100.0, min_peak=0.0, max_comp=9.0, min_hint_agree=0.0):
    def gate(r):
        if r is None:
            return False
        return (r["n_body"] >= min_body and r["median_residual"] <= max_resid and r["pair_support"] >= min_pair
                and r["hint_dist"] <= radius and r["min_peak"] >= min_peak and r["comp_iou"] <= max_comp
                and np.nanmin(r["agree_hint"] + [1.0]) >= min_hint_agree)
    return gate


def evaluate(key, gate, verbose=False):
    out = {}
    wh = [(c, e) for c, e in pairs if c["tag"] == "withhold"]
    acc = [(c, e[key]) for c, e in wh if gate(e.get(key))]
    errs = np.array([np.linalg.norm(r["root"] - root_of(c["truth"])) for c, r in acc]) if acc else np.array([])
    jerr = []
    for c, r in acc:
        both = np.isfinite(c["truth"][:, 0]) & np.isfinite(r["xyz"][:, 0])
        if both.any():
            jerr.append(np.median(np.linalg.norm(c["truth"][both] - r["xyz"][both], axis=1)))
    out["withhold"] = {"cases": len(wh), "accepted": len(acc), "share": round(len(acc) / max(1, len(wh)), 3),
                       "root_err_p50_p90_p99": np.percentile(errs, [50, 90, 99]).round(2).tolist() if len(errs) else None,
                       "wrong_gt12in": int((errs > 12).sum()), "wrong_gt24in": int((errs > 24).sum()),
                       "joint_err_med_p50_p90": np.percentile(jerr, [50, 90]).round(2).tolist() if jerr else None}
    for tag in ("neg_shift", "neg_absent", "neg_empty"):
        sel = [(c, e) for c, e in pairs if c["tag"] == tag]
        fa = [(c, e[key]) for c, e in sel if gate(e.get(key))]
        real = hall = 0
        for c, r in fa:
            d = nearest_person(c["t"], r["root"])
            if d < 24.0:
                real += 1
            else:
                hall += 1
        out[tag] = {"cases": len(sel), "accepted": len(fa), "rate": round(len(fa) / max(1, len(sel)), 4),
                    "accepted_on_a_real_measured_person": real, "accepted_unexplained(hallucination)": hall}
    gp = [(c, e) for c, e in pairs if c["tag"] == "gap"]
    table = {}
    for bname, a, b in BUCKETS:
        s = [(c, e) for c, e in gp if c["gap_kind"] == "interior" and a <= c["gap"] <= b]
        table[bname] = {"frames_with_hint": len(s), "recovered": sum(1 for c, e in s if gate(e.get(key)))}
    ends = [(c, e) for c, e in gp if c["gap_kind"] != "interior"]
    table["ends"] = {"frames_with_hint": len(ends), "recovered": sum(1 for c, e in ends if gate(e.get(key)))}
    out["gap"] = table
    return out


if __name__ == "__main__":
    results = {}
    gates = {"loose": make_gate(), "n8_r6": make_gate(min_body=8, max_resid=6.0, radius=60.0),
             "n9_r4_h40": make_gate(min_body=9, max_resid=4.0, radius=40.0, min_peak=0.5),
             "n10_r4_h40_p6": make_gate(min_body=10, max_resid=4.0, radius=40.0, min_peak=0.6)}
    keys = sys.argv[2:] or ["R0/crop", "R1/crop", "R2/crop", "R0/raw+crop", "R1/raw+crop"]
    for key in keys:
        for gname, gate in gates.items():
            results[f"{key} | {gname}"] = evaluate(key, gate)
    (SCRATCH / f"report_{name}.json").write_text(json.dumps(results, indent=1, default=float))
    for k, v in results.items():
        w = v["withhold"]
        print(f"\n{k}\n  withhold share {w['share']} wrong>12in {w['wrong_gt12in']} err {w['root_err_p50_p90_p99']}"
              f" | neg_shift {v['neg_shift']['accepted']}/{v['neg_shift']['cases']} (real {v['neg_shift']['accepted_on_a_real_measured_person']})"
              f" neg_absent {v['neg_absent']['accepted']}/{v['neg_absent']['cases']} (real {v['neg_absent']['accepted_on_a_real_measured_person']})"
              f" neg_empty {v['neg_empty']['accepted']}/{v['neg_empty']['cases']} (real {v['neg_empty']['accepted_on_a_real_measured_person']})"
              f"\n  gap recovered " + " ".join(f"{b}:{d['recovered']}/{d['frames_with_hint']}" for b, d in v["gap"].items()))
