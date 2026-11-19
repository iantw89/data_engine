"""Which acceptance features separate real recoveries (withhold, correct) from false accepts (negatives)?

Usage: python experiments/coverage/gate_analysis.py [eval_name] [key]
Prints feature percentiles per class and a greedy gate search (zero false accepts on the negatives
that are not explained by a real person, maximum yield on correct withhold recoveries).
"""
import itertools
import json
import pickle
import sys
from collections import defaultdict

from common import *  # noqa
from recover_lib import root_of

name = sys.argv[1] if len(sys.argv) > 1 else "vit"
key = sys.argv[2] if len(sys.argv) > 2 else "R1/crop"
ws, scene, tracks, solution = load()
cases = pickle.loads((SCRATCH / "cases.pkl").read_bytes())
ev = {e["id"]: e for e in pickle.loads((SCRATCH / f"eval_{name}.pkl").read_bytes())}
by_t = defaultdict(list)
for row in np.flatnonzero(np.isfinite(tracks["root"]).all(axis=1)):
    by_t[int(tracks["t"][row])].append(row)

BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 11), (6, 12), (5, 6), (11, 12)]
ok = np.isfinite(tracks["xyz"][:, 5:17, 0]).all(axis=1)
sample = tracks["xyz"][np.flatnonzero(ok)[::10]].astype(np.float64)
RANGE = []
for a, b in BONES:
    length = np.linalg.norm(sample[:, a] - sample[:, b], axis=1)
    RANGE.append(np.percentile(length, [0.5, 99.5]))
RANGE = np.array(RANGE)


def bone_features(xyz):
    bad = tot = 0
    for (a, b), (lo, hi) in zip(BONES, RANGE):
        if np.isfinite(xyz[a, 0]) and np.isfinite(xyz[b, 0]):
            tot += 1
            length = np.linalg.norm(xyz[a] - xyz[b])
            bad += not (lo <= length <= hi)
    return tot, bad


def nearest_other(case, rec):
    """Distance to the nearest OTHER identity measured at t (3D rule: the pose already belongs to someone)."""
    best = np.inf
    for r in by_t.get(case["t"], []):
        if int(tracks["identity"][r]) != case["who"]:
            best = min(best, float(np.linalg.norm(tracks["root"][r][:2] - rec["root"][:2])))
    return best


rows = []
for case in cases:
    e = ev.get(case["id"])
    if e is None or e.get(key) is None:
        continue
    r = e[key]
    tot, bad = bone_features(r["xyz"])
    feat = {"id": case["id"], "tag": case["tag"], "n_body": r["n_body"], "resid": r["median_residual"], "pair": r["pair_support"],
            "min_peak": r["min_peak"], "mean_peak": r["mean_peak"], "hint": r["hint_dist"], "ncam": len(r["cams"]),
            "comp": r["comp_iou"], "agree": float(np.nanmin(r["agree_hint"] + [1.0])), "bones": tot, "bad_bones": bad,
            "other_dist": nearest_other(case, r)}
    if case["tag"] == "withhold":
        err = float(np.linalg.norm(r["root"] - root_of(case["truth"])))
        feat["class"] = "pos" if err <= 12 else "wrong"
        feat["err"] = err
    elif case["tag"] in ("neg_empty", "neg_absent", "neg_shift"):
        if case["tag"] == "neg_shift" and np.linalg.norm(r["root"][:2] - case["true_root"][:2]) < 24.0:
            feat["class"] = "benign_same_person"
        elif feat["other_dist"] < 24.0:
            feat["class"] = "other_person"
        else:
            feat["class"] = "false_" + case["tag"]
    else:
        continue
    rows.append(feat)

classes = defaultdict(list)
for f in rows:
    classes[f["class"]].append(f)
print({k: len(v) for k, v in classes.items()})
for feature in ("n_body", "resid", "pair", "min_peak", "mean_peak", "hint", "ncam", "comp", "agree", "bad_bones", "other_dist"):
    print(f"{feature:10s}", {k: np.percentile([f[feature] for f in v], [5, 50, 95]).round(2).tolist() for k, v in classes.items() if len(v) > 3})

neg = [f for f in rows if f["class"].startswith("false_")]
pos = [f for f in rows if f["class"] == "pos"]
wrong = [f for f in rows if f["class"] == "wrong"]
other = [f for f in rows if f["class"] == "other_person"]


def accepts(f, g):
    return (f["n_body"] >= g["n_body"] and f["resid"] <= g["resid"] and f["min_peak"] >= g["min_peak"] and f["hint"] <= g["hint"]
            and f["bad_bones"] <= g["bad_bones"] and f["agree"] >= g["agree"] and f["other_dist"] >= g["other_dist"] and f["ncam"] >= g["ncam"])


results = []
for n_body, resid, min_peak, hint, bad, agree, ncam in itertools.product((6, 8, 9, 10, 11), (12, 6, 4, 3), (0.0, 0.4, 0.5, 0.6), (100, 60, 40, 24), (99, 1, 0), (0.0, 0.3, 0.5), (2, 3)):
    g = {"n_body": n_body, "resid": resid, "min_peak": min_peak, "hint": hint, "bad_bones": bad, "agree": agree, "other_dist": 24.0, "ncam": ncam}
    fa = sum(accepts(f, g) for f in neg)
    results.append((fa, -sum(accepts(f, g) for f in pos), sum(accepts(f, g) for f in wrong), g))
results.sort(key=lambda x: (x[0], x[1]))
print("\nbest gates (false accepts on unexplained negatives, then max positives):")
shown = 0
for fa, negpos, w, g in results:
    if fa == 0 and shown < 8:
        print(f"  false={fa} positives={-negpos}/{len(pos)} wrong_withhold={w}/{len(wrong)} gate={g}")
        shown += 1
fa_min = results[0][0]
print("min false accepts reachable:", fa_min, "of", len(neg), "(negatives accepted at loosest gates)")
for fa, negpos, w, g in results[:3]:
    print(f"  false={fa} pos={-negpos}/{len(pos)} wrong={w}/{len(wrong)} {g}")
(SCRATCH / f"gate_analysis_{name}_{key.replace('/', '_')}.json").write_text(json.dumps(
    {"classes": {k: len(v) for k, v in classes.items()}, "top": [(fa, -np_, w, g) for fa, np_, w, g in results[:20] if fa == 0][:10]}, indent=1, default=float))


# ---- second search: allow two-camera recoveries, find the strictest-needed gate
print("\nwrong withhold errors (in):", sorted(round(f["err"], 1) for f in wrong))
print("pos err pcts 50/90/99/max:", np.percentile([f["err"] for f in pos], [50, 90, 99, 100]).round(2).tolist())
res2 = []
for n_body, resid, min_peak, bad, agree, ncam, pair in itertools.product((8, 9, 10, 11, 12), (6, 4, 3), (0.0, 0.4, 0.5, 0.6, 0.7), (1, 0), (0.0, 0.3, 0.5, 0.7), (2, 3), (6, 9, 11)):
    g = {"n_body": n_body, "resid": resid, "min_peak": min_peak, "hint": 100, "bad_bones": bad, "agree": agree, "other_dist": 24.0, "ncam": ncam}
    acc = lambda f: accepts(f, g) and f["pair"] >= pair
    res2.append((sum(acc(f) for f in neg), -sum(acc(f) for f in pos), sum(acc(f) for f in wrong), {**g, "pair": pair}))
res2.sort(key=lambda x: (x[0], x[1]))
print("best zero-false gates, any camera count:")
shown = 0
for fa, np_, w, g in res2:
    if fa == 0 and shown < 5:
        print(f"  pos={-np_}/{len(pos)} wrong={w} {g}")
        shown += 1
print("best zero-false gates that still accept 2-camera poses:")
shown = 0
for fa, np_, w, g in res2:
    if fa == 0 and g["ncam"] == 2 and shown < 5:
        print(f"  pos={-np_}/{len(pos)} wrong={w} {g}")
        shown += 1
print("with <=1 false accept (of", len(neg), ") any cam:")
shown = 0
for fa, np_, w, g in res2:
    if fa <= 1 and g["ncam"] == 2 and shown < 3:
        print(f"  false={fa} pos={-np_}/{len(pos)} wrong={w} {g}")
        shown += 1

print("\nTwo-camera subset: positives", sum(f["ncam"] == 2 for f in pos), "negatives", sum(f["ncam"] == 2 for f in neg),
      "wrong", sum(f["ncam"] == 2 for f in wrong))
two = [r for r in res2 if r[3]["ncam"] == 2]
two.sort(key=lambda x: (x[0], x[1]))
for fa, np_, w, g in two[:3]:
    print(f"  best 2-cam gate: false={fa} pos={-np_}/{len(pos)} wrong={w} {g}")
# alternatives: false-accept budget
for budget in (2, 5, 10):
    c = [r for r in two if r[0] <= budget]
    c.sort(key=lambda x: x[1])
    if c:
        fa, np_, w, g = c[0]
        print(f"  budget {budget}: false={fa} pos={-np_} wrong={w} {g}")
for feature in ("n_body", "resid", "min_peak", "bad_bones", "agree", "hint"):
    print("  2-cam", feature, {k: np.percentile([f[feature] for f in v if f["ncam"] == 2], [5, 50, 95]).round(2).tolist()
                               for k, v in classes.items() if sum(f["ncam"] == 2 for f in v) > 3 and k in ("pos", "false_neg_shift", "false_neg_absent", "false_neg_empty")})

(SCRATCH / f"gate_rows_{name}_{key.replace('/', '_')}.json").write_text(json.dumps(rows, default=float))
