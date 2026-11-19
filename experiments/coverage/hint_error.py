"""How far is the neighbour-interpolated hint from the truth (withhold test), by distance to the nearest measured pose?"""
import json
import pickle

from common import *  # noqa
from recover_lib import root_of

cases = pickle.loads((SCRATCH / "cases.pkl").read_bytes())
out = {}
for kind in ("interp", "hold_before", "hold_after"):
    sel = [c for c in cases if c["tag"] == "withhold" and c["hint_kind"] == kind]
    if not sel:
        continue
    err = np.array([np.linalg.norm(c["root"][:2] - root_of(c["truth"])[:2]) for c in sel])
    out[kind] = {"n": len(sel), "xy_err_in_p50_p90_p99_max": np.percentile(err, [50, 90, 99, 100]).round(1).tolist()}
by = {}
for c in cases:
    if c["tag"] == "withhold":
        by.setdefault(min(c["hint_distance"], 10), []).append(np.linalg.norm(c["root"][:2] - root_of(c["truth"])[:2]))
out["by_distance_to_nearest_measured_frame"] = {str(k): {"n": len(v), "p50": round(float(np.percentile(v, 50)), 1), "p90": round(float(np.percentile(v, 90)), 1),
                                                         "p99": round(float(np.percentile(v, 99)), 1)} for k, v in sorted(by.items())}
gap = [c for c in cases if c["tag"] == "gap"]
out["gap_hint_distance_frames"] = {str(k): sum(1 for c in gap if min(c["hint_distance"], 11) == k) for k in range(1, 12)}
print(json.dumps(out, indent=1))
(SCRATCH / "hint_error.json").write_text(json.dumps(out, indent=1))
