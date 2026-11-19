"""One ViTPose pass on the hint box (kp1) vs the second pass on the re-derived box (kp2), at the tier gates."""
import pickle
from common import *
from gates import Context, accept, features
from recover_lib import root_of
ws, scene, tracks, solution = load()
ctx = Context(tracks)
cases = pickle.loads((SCRATCH / "cases.pkl").read_bytes())
for name in ("vit_kp1", "vit"):
    ev = {e["id"]: e for e in pickle.loads((SCRATCH / f"eval_{name}.pkl").read_bytes())}
    wh = acc = gap = 0; errs = []
    for c in cases:
        if c["tag"] not in ("withhold", "gap") or c["id"] not in ev: continue
        rec = ev[c["id"]].get("R1/crop")
        wh += c["tag"] == "withhold"
        if rec is None: continue
        tier = accept(features(ctx, c["who"], c["t"], rec))
        if not tier: continue
        if c["tag"] == "withhold":
            acc += 1; errs.append(np.linalg.norm(rec["root"] - root_of(c["truth"])))
        else: gap += 1
    print(name, "withhold accepted", acc, "/", wh, "err p50/p90/p99", np.percentile(errs, [50, 90, 99]).round(2).tolist(), "gap frames", gap)
