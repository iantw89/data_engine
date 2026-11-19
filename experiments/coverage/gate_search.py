"""Search per-tier gates: maximise recovered gap frames subject to a false-accept budget on the negatives.

Usage: python experiments/coverage/gate_search.py [eval_name] [key]
"""
import itertools
import json
import pickle
import sys

from common import *  # noqa
from gates import Context, features
from recover_lib import root_of

name = sys.argv[1] if len(sys.argv) > 1 else "vit"
key = sys.argv[2] if len(sys.argv) > 2 else "R1/crop"
ws, scene, tracks, solution = load()
ctx = Context(tracks)
cases = pickle.loads((SCRATCH / "cases.pkl").read_bytes())
ev = {e["id"]: e for e in pickle.loads((SCRATCH / f"eval_{name}.pkl").read_bytes())}
FEATS = ["cams", "n_body", "resid", "min_peak", "agree", "bad_bones", "hint", "other"]
rows, meta = [], []
for case in cases:
    e = ev.get(case["id"])
    rec = None if e is None else e.get(key)
    if rec is None:
        continue
    f = features(ctx, case["who"], case["t"], rec)
    cls = None
    if case["tag"] == "withhold":
        err = float(np.linalg.norm(rec["root"] - root_of(case["truth"])))
        cls = "wh_ok" if err <= 24 else "wh_wrong"
    elif case["tag"] in ("neg_shift", "neg_absent", "neg_empty"):
        if case["tag"] == "neg_shift" and np.linalg.norm(rec["root"][:2] - case["true_root"][:2]) < 24.0:
            continue  # benign: it is the identity's own real person
        cls = "neg"
    elif case["tag"] == "gap":
        cls = "gap"
    else:
        continue
    rows.append([f[k] for k in FEATS])
    meta.append((cls, case["tag"], case.get("gap_kind"), case.get("gap"), case["who"], case["t"]))
X = np.array(rows, float)
cls = np.array([m[0] for m in meta])
N_WH = sum(1 for c in cases if c["tag"] == "withhold")
print("records", len(X), {c: int((cls == c).sum()) for c in set(cls)})


def mask(g, X):
    return ((X[:, 0] >= g[0]) & (X[:, 0] <= g[1]) & (X[:, 1] >= g[2]) & (X[:, 2] <= g[3]) & (X[:, 5] <= g[4]) & (X[:, 6] <= g[5])
            & (X[:, 7] >= 24.0) & (X[:, 3] >= g[6]) & (X[:, 4] >= g[7]))


best = {}
for tier, (cmin, cmax) in {"A (3+ cams)": (3, 9), "B (2 cams)": (2, 2)}.items():
    results = []
    for n_body, resid, bones, hint, peak, agree in itertools.product((8, 9, 10, 11, 12), (3, 4, 6, 8, 12), (0, 1, 2, 99), (12, 15, 24, 40, 100), (0, 0.4, 0.5, 0.6), (0, 0.3, 0.5)):
        g = (cmin, cmax, n_body, resid, bones, hint, peak, agree)
        m = mask(g, X)
        results.append((int((m & (cls == "neg")).sum()), int((m & (cls == "wh_wrong")).sum()), -int((m & (cls == "gap")).sum()), -int((m & (cls == "wh_ok")).sum()), g))
    results.sort(key=lambda r: (r[0], r[1], r[2]))
    print(f"\n== tier {tier}: top by (false accepts, wrong, -gap yield)")
    for fa, wrong, gap, wh, g in results[:4]:
        print(f"  false={fa} wh_wrong={wrong} gap_frames={-gap} wh_ok={-wh}/{N_WH} gate n_body>={g[2]} resid<={g[3]} bones<={g[4]} hint<={g[5]} peak>={g[6]} agree>={g[7]}")
    for budget in (1, 2, 5):
        c = [r for r in results if r[0] <= budget and r[1] == 0]
        c.sort(key=lambda r: r[2])
        if c:
            fa, wrong, gap, wh, g = c[0]
            print(f"  FA budget {budget}: false={fa} gap_frames={-gap} wh_ok={-wh} gate n_body>={g[2]} resid<={g[3]} bones<={g[4]} hint<={g[5]} peak>={g[6]} agree>={g[7]}")
    best[tier] = results[:10]
n_neg = int((cls == "neg").sum())
print("negatives considered (excl. benign):", n_neg, "hypotheses incl. benign: 6500")
