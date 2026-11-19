"""Competitor rule comparison at the proposed tier gates (R0 none, R1 skip a crop that a competitor explains, R2 skip any crop overlapping a competitor)."""
import json
import pickle
from common import *
from gates import Context, accept, features
from recover_lib import root_of
ws, scene, tracks, solution = load()
ctx = Context(tracks)
cases = pickle.loads((SCRATCH / "cases.pkl").read_bytes())
ev = {e["id"]: e for e in pickle.loads((SCRATCH / "eval_vit.pkl").read_bytes())}
out = {}
for key in ("R0/crop", "R1/crop", "R2/crop", "R0/raw+crop", "R1/raw+crop"):
    row = {"withhold_accepted": 0, "withhold_cases": 0, "withhold_gt12": 0, "withhold_gt24": 0, "gap_frames": 0,
           "neg_unexplained": 0, "neg_other_identity_stopped": 0}
    for c in cases:
        rec = ev[c["id"]].get(key)
        row["withhold_cases"] += c["tag"] == "withhold"
        if rec is None:
            continue
        f = features(ctx, c["who"], c["t"], rec)
        loose = accept({**f, "other": 1e9})
        tier = accept(f)
        if c["tag"] == "withhold" and tier:
            err = float(np.linalg.norm(rec["root"] - root_of(c["truth"])))
            row["withhold_accepted"] += 1; row["withhold_gt12"] += err > 12; row["withhold_gt24"] += err > 24
        elif c["tag"] == "gap" and tier:
            row["gap_frames"] += 1
        elif c["tag"].startswith("neg") and loose:
            if c["tag"] == "neg_shift" and np.linalg.norm(rec["root"][:2] - c["true_root"][:2]) < 24: continue
            if tier is None: row["neg_other_identity_stopped"] += 1
            else: row["neg_unexplained"] += 1
    out[key] = row
    print(key, row)
(SCRATCH / "rule_compare.json").write_text(json.dumps(out, indent=1, default=float))
