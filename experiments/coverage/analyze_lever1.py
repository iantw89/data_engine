import json
import pickle
from collections import Counter, defaultdict

from common import *  # noqa
from gap_classify import BUCKETS, bucket

gaps = pickle.loads((SCRATCH / "lever1_gaps.pkl").read_bytes())
wh = pickle.loads((SCRATCH / "lever1_withhold.pkl").read_bytes())["results"]
frames = json.loads((SCRATCH / "gap_classify_frames.json").read_text())

out = {}
# ---- hint distance distribution, withhold: recovered root vs truth root
for variant in ("L1", "L1+other"):
    w = [r for r in wh if r["variant"] == variant]
    rec = [r for r in w if r["ok"]]
    err = np.array([r["root_err"] for r in rec])
    jerr = np.array([r["joint_err_median"] for r in rec if r.get("joint_err_median") is not None])
    out[f"withhold_{variant}"] = {
        "frames": len(w), "recovered": len(rec), "share": round(len(rec) / len(w), 3),
        "root_err_in_pcts_50_90_99_max": np.percentile(err, [50, 90, 99, 100]).round(2).tolist(),
        "joint_err_median_pcts_50_90_99": np.percentile(jerr, [50, 90, 99]).round(2).tolist(),
        "n_root_err_gt_6in": int((err > 6).sum()), "n_root_err_gt_12in": int((err > 12).sum()),
        "hint_dist_xy_pcts_50_90_99": np.percentile([r["hint_dist_xy"] for r in rec], [50, 90, 99]).round(1).tolist(),
        "by_length": {str(L): {"frames": sum(1 for r in w if r["length"] == L), "recovered": sum(1 for r in rec if r["length"] == L)}
                      for L in sorted({r["length"] for r in w})},
    }
    # hint distance of WRONG picks (root err > 12 in)
    wrong = [r for r in rec if r["root_err"] > 12]
    out[f"withhold_{variant}"]["wrong_picks_detail"] = [
        {"len": r["length"], "hint_dist": round(r["hint_dist_xy"], 1), "root_err": round(r["root_err"], 1), "src": r["sources"],
         "resid": round(r["median_residual"], 1), "n_body": r["n_body"]} for r in wrong[:10]]

# ---- real gaps
for variant in ("L1", "L1+other"):
    g = [r for r in gaps if r["variant"] == variant]
    interior = [r for r in g if r["kind"] == "interior"]
    table = {}
    for name, a, b in BUCKETS:
        sel = [r for r in interior if a <= r["gap"] <= b]
        table[name] = {"frames_with_hint": len(sel), "recovered": sum(r["ok"] for r in sel)}
    ends = [r for r in g if r["kind"] != "interior"]
    out[f"gaps_{variant}"] = {"interior_by_gap_length": table,
                              "ends": {"frames": len(ends), "recovered": sum(r["ok"] for r in ends)},
                              "by_cause": {c: {"n": sum(1 for r in interior if r["cause"] == c), "recovered": sum(1 for r in interior if r["cause"] == c and r["ok"])}
                                           for c in sorted({r["cause"] for r in interior})}}
    rec = [r for r in interior if r["ok"]]
    out[f"gaps_{variant}"]["recovered_hint_dist_xy_pcts_50_90_99"] = np.percentile([r["hint_dist_xy"] for r in rec], [50, 90, 99]).round(1).tolist()
    out[f"gaps_{variant}"]["recovered_n_body_hist"] = dict(Counter(r["n_body"] for r in rec))
    out[f"gaps_{variant}"]["recovered_median_resid_pcts"] = np.percentile([r["median_residual"] for r in rec], [50, 90, 99]).round(2).tolist()
    # whole gaps closed
    gap_ok = defaultdict(list)
    for r in interior:
        gap_ok[(r["identity"], r["t"] - r["offset"])].append(r["ok"])
    # frames without hint inside gap can't be recovered by hint search
    closed = Counter()
    total = Counter()
    for (who, a), v in gap_ok.items():
        pass
print(json.dumps(out, indent=1))
(SCRATCH / "lever1_summary.json").write_text(json.dumps(out, indent=1))
