"""Yield per lever on the real gaps (frames and whole gaps), union across levers, at a chosen gate.

Usage: python experiments/coverage/final_tables.py [gate_name]
"""
import json
import pickle
import sys
from collections import defaultdict

from common import *  # noqa
from gap_classify import BUCKETS
from report_levers import make_gate

GATES = {"loose": dict(), "n8_r6": dict(min_body=8, max_resid=6.0, radius=60.0),
         "n9_r4_h40": dict(min_body=9, max_resid=4.0, radius=40.0, min_peak=0.5),
         "n8_r6_h40_p5": dict(min_body=8, max_resid=6.0, radius=40.0, min_peak=0.5)}
gate_name = sys.argv[1] if len(sys.argv) > 1 else "n8_r6"
G = GATES[gate_name]
gate = make_gate(**G)

cases = pickle.loads((SCRATCH / "cases.pkl").read_bytes())
frames = json.loads((SCRATCH / "gap_classify_frames.json").read_text())
gap_cases = [c for c in cases if c["tag"] == "gap"]
sets = {}


def from_eval(name, key):
    ev = {e["id"]: e for e in pickle.loads((SCRATCH / f"eval_{name}.pkl").read_bytes())}
    return {(c["who"], c["t"]) for c in gap_cases if c["id"] in ev and gate(ev[c["id"]].get(key))}


l1 = pickle.loads((SCRATCH / "lever1_gaps.pkl").read_bytes())
sets["L1 existing detections"] = {(r["identity"], r["t"]) for r in l1 if r["variant"] == "L1" and r["ok"] and
                                  r["n_body"] >= G.get("min_body", 6) and r["median_residual"] <= G.get("max_resid", 12.0)
                                  and r["hint_dist_xy"] <= G.get("radius", 100.0) and r["pair_support"] >= 6}
sets["L2 ViTPose crop (competitor rule)"] = from_eval("vit", "R1/crop")
for det in ("yolov8x", "yolo11l"):
    try:
        sets[f"L3 {det} ROI + ViTPose"] = from_eval(det, "R1/roi")
    except FileNotFoundError:
        pass
sets["L1 + L2"] = sets["L1 existing detections"] | sets["L2 ViTPose crop (competitor rule)"]
all_l3 = set().union(*[v for k, v in sets.items() if k.startswith("L3")]) if any(k.startswith("L3") for k in sets) else set()
sets["L1 + L2 + L3"] = sets["L1 + L2"] | all_l3

# missing frames per gap bucket (all missing frames, incl. those without a hint)
by_bucket = defaultdict(list)
for f in frames:
    if f["kind"] == "interior":
        for name, a, b in BUCKETS:
            if a <= f["gap"] <= b:
                by_bucket[name].append((f["identity"], f["t"], f["identity"], f["t"] - f["offset"]))
table = {}
for label, s in sets.items():
    row = {}
    for name, a, b in BUCKETS:
        items = by_bucket[name]
        rec = sum(1 for who, t, _, _ in items if (who, t) in s)
        # whole gaps closed / shortened
        gaps = defaultdict(list)
        for who, t, _, start in items:
            gaps[(who, start)].append((who, t) in s)
        closed = sum(all(v) for v in gaps.values())
        partial = sum(any(v) and not all(v) for v in gaps.values())
        row[name] = {"missing_frames": len(items), "recovered": rec, "gaps": len(gaps), "gaps_closed": closed, "gaps_shortened": partial}
    ends = [f for f in frames if f["kind"] != "interior"]
    row["ends(1s beyond first/last)"] = {"examined": len(ends), "recovered": sum(1 for f in ends if (f["identity"], f["t"]) in s)}
    row["total_interior_recovered"] = sum(row[n]["recovered"] for n, _, _ in BUCKETS)
    table[label] = row
print(gate_name, G)
print(json.dumps(table, indent=1))
(SCRATCH / f"final_tables_{gate_name}.json").write_text(json.dumps({"gate": G, "table": table}, indent=1))
