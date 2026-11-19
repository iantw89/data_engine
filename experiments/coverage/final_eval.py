"""Final numbers at the proposed gate (gates.py): withhold, negatives, real-gap yield per lever.

Usage: python experiments/coverage/final_eval.py
"""
import json
import pickle
from collections import Counter, defaultdict

from common import *  # noqa
from gap_classify import BUCKETS
from gates import TIERS, Context, accept, features
from recover_lib import root_of

ws, scene, tracks, solution = load()
ctx = Context(tracks)
cases = pickle.loads((SCRATCH / "cases.pkl").read_bytes())
case_by_id = {c["id"]: c for c in cases}


def load_eval(name):
    try:
        return {e["id"]: e for e in pickle.loads((SCRATCH / f"eval_{name}.pkl").read_bytes())}
    except FileNotFoundError:
        return {}


SOURCES = {
    "L2 ViTPose crop": ("vit", "R1/crop"),
    "L1+L2 (raw preferred, else crop)": ("vit", "R1/raw+crop"),
    "L3 YOLOv8x ROI ROI side 1.6h (~5x zoom)": ("yolov8x", "R1/roi"),
    "L3 YOLO11l ROI ROI side 1.6h (~5x zoom)": ("yolo11l", "R1/roi"),
    "L3 YOLOv8x ROI ROI side 4h (~3x zoom)": ("yolov8x_z4gap", "R1/roi"),
}
evals = {k: load_eval(v[0]) for k, v in SOURCES.items()}
out = {"tiers": TIERS, "withhold": {}, "negatives": {}, "gap": {}}
accepted_gap = {}  # source -> {(who,t): tier}


def per_case(source):
    name, key = SOURCES[source]
    for cid, e in evals[source].items():
        yield case_by_id[cid], e.get(key)


# --------------------------------------------------------------------------- L1 (separate store format)
l1_gap = pickle.loads((SCRATCH / "lever1_gaps.pkl").read_bytes())
l1_wh = pickle.loads((SCRATCH / "lever1_withhold.pkl").read_bytes())["results"]


def withhold_table(rows):
    """rows: (err or None if no recovery, tier)"""
    n = len(rows)
    by = {}
    for tier in ("A", "B", "any"):
        sel = [e for e, t in rows if t is not None and (tier == "any" or t == tier) and e is not None]
        errs = np.array(sel)
        by[tier] = {"accepted": len(sel), "share": round(len(sel) / max(1, n), 3),
                    "root_err_p50_p90_p99": np.percentile(errs, [50, 90, 99]).round(2).tolist() if len(errs) else None,
                    "gt12in": int((errs > 12).sum()), "gt24in(wrong person?)": int((errs > 24).sum())}
    return {"cases": n, **by}


rows = []
for r in l1_wh:
    if r["variant"] != "L1":
        continue
    if not r["ok"]:
        rows.append((None, None))
        continue
    f = features(ctx, r["identity"], r["t"], {**r, "hint_dist": r["hint_dist_xy"]})
    rows.append((r["root_err"], accept(f)))
out["withhold"]["L1 existing detections (rows unassigned, identity hidden)"] = withhold_table(rows)
for source in SOURCES:
    rows = []
    for case, rec in per_case(source):
        if case["tag"] != "withhold":
            continue
        if rec is None:
            rows.append((None, None))
            continue
        f = features(ctx, case["who"], case["t"], rec)
        rows.append((float(np.linalg.norm(rec["root"] - root_of(case["truth"]))), accept(f)))
    if rows:
        out["withhold"][source] = withhold_table(rows)

# --------------------------------------------------------------------------- negatives
for source in SOURCES:
    table = {}
    for tag in ("neg_shift", "neg_absent", "neg_empty"):
        total = acc = benign = other_blocked = 0
        loose = 0
        for case, rec in per_case(source):
            if case["tag"] != tag:
                continue
            total += 1
            if rec is None:
                continue
            f = features(ctx, case["who"], case["t"], rec)
            loose_f = {**f, "other": 1e9}
            tier = accept(loose_f)
            if tier is None:
                continue
            if tag == "neg_shift" and np.linalg.norm(rec["root"][:2] - case["true_root"][:2]) < 24.0:
                benign += 1  # the pose of the identity's real person, found 20-79 in from the hint
            elif f["other"] < 24.0:
                other_blocked += 1  # a measured person: stopped by the "someone else already there" rule
            else:
                acc += 1
        if total:
            table[tag] = {"cases": total, "accepted_unexplained": acc, "accepted_benign_same_person": benign,
                          "stopped_by_other_identity_rule": other_blocked}
    if table:
        out["negatives"][source] = table

# --------------------------------------------------------------------------- real gaps
frames = json.loads((SCRATCH / "gap_classify_frames.json").read_text())
by_bucket = defaultdict(list)
for f in frames:
    if f["kind"] == "interior":
        for name, a, b in BUCKETS:
            if a <= f["gap"] <= b:
                by_bucket[name].append((f["identity"], f["t"], f["t"] - f["offset"]))
ends = [f for f in frames if f["kind"] != "interior"]
sets = {}
s = {}
for r in l1_gap:
    if r["variant"] == "L1" and r["ok"]:
        tier = accept(features(ctx, r["identity"], r["t"], {**r, "hint_dist": r["hint_dist_xy"]}))
        if tier:
            s[(r["identity"], r["t"])] = tier
sets["L1 existing detections"] = s
for source in SOURCES:
    s = {}
    for case, rec in per_case(source):
        if case["tag"] == "gap" and rec is not None:
            tier = accept(features(ctx, case["who"], case["t"], rec))
            if tier:
                s[(case["who"], case["t"])] = tier
    if s:
        sets[source] = s
sets["L1 + L2"] = {**sets["L2 ViTPose crop"], **sets["L1 existing detections"]}
l3 = {}
for k, v in sets.items():
    if k.startswith("L3"):
        l3.update(v)
sets["L1 + L2 + L3 (any detector)"] = {**sets["L1 + L2"], **l3}
for label, s in sets.items():
    table = {}
    for name, a, b in BUCKETS:
        items = by_bucket[name]
        rec = sum(1 for who, t, _ in items if (who, t) in s)
        gaps = defaultdict(list)
        for who, t, start in items:
            gaps[(who, start)].append((who, t) in s)
        table[name] = {"missing_frames": len(items), "recovered": rec, "tier_B_only": sum(1 for who, t, _ in items if s.get((who, t)) == "B"),
                       "gaps": len(gaps), "closed": sum(all(v) for v in gaps.values()), "shortened": sum(any(v) and not all(v) for v in gaps.values())}
    table["ends"] = {"examined": len(ends), "recovered": sum(1 for f in ends if (f["identity"], f["t"]) in s)}
    table["total_interior_recovered"] = sum(table[n]["recovered"] for n, _, _ in BUCKETS)
    out["gap"][label] = table
    accepted_gap[label] = s
(SCRATCH / "final_eval.json").write_text(json.dumps(out, indent=1, default=float))
pickle.dump(accepted_gap, open(SCRATCH / "accepted_gap.pkl", "wb"))
print(json.dumps(out, indent=1, default=float))
