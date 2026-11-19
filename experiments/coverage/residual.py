"""Why are frames still missing after levers 1-3 at the proposed gate? (interior gaps and ends)"""
import json
import pickle
from collections import Counter, defaultdict

from common import *  # noqa
from gap_classify import BUCKETS
from gates import Context, accept, features

ws, scene, tracks, solution = load()
ctx = Context(tracks)
cases = pickle.loads((SCRATCH / "cases.pkl").read_bytes())
frames = json.loads((SCRATCH / "gap_classify_frames.json").read_text())
accepted = pickle.loads((SCRATCH / "accepted_gap.pkl").read_bytes())["L1 + L2 + L3 (any detector)"]
vit = {e["id"]: e for e in pickle.loads((SCRATCH / "eval_vit.pkl").read_bytes())}
case_of = {(c["who"], c["t"]): c for c in cases if c["tag"] == "gap"}
table = defaultdict(Counter)
detail = defaultdict(Counter)
for f in frames:
    key = (f["identity"], f["t"])
    bucket = "ends" if f["kind"] != "interior" else next(n for n, a, b in BUCKETS if a <= f["gap"] <= b)
    if key in accepted:
        reason = "recovered"
    elif not f["hint"]:
        reason = "no hint (nearest measured pose > 3 s away)"
    elif f["cause"] == "d_unobservable":
        reason = "fewer than 2 cameras can see the hinted place"
    else:
        case = case_of.get(key)
        rec = None if case is None else vit[case["id"]].get("R1/crop")
        rec2 = None if case is None else vit[case["id"]].get("R1/raw+crop")
        rec = rec or rec2
        if case is None:
            reason = "visible in <2 cameras after crop planning"
        elif rec is None:
            reason = "pixels: no pose pair consistent in 2+ views (ViTPose finds no triangulable person)"
        else:
            fe = features(ctx, case["who"], case["t"], rec)
            if fe["other"] < 24.0:
                reason = "pose is on someone already measured at that instant (other identity)"
            else:
                reason = "pose found, rejected by confidence gate (tier A/B)"
                if f["cause"].startswith("c") and f.get("occluded_plausible"):
                    reason += " [occlusion plausible]"
    table[bucket][reason] += 1
out = {b: dict(c) for b, c in table.items()}
tot = Counter()
for c in table.values():
    tot.update(c)
out["ALL"] = dict(tot)
print(json.dumps(out, indent=1))
(SCRATCH / "residual.json").write_text(json.dumps(out, indent=1))
