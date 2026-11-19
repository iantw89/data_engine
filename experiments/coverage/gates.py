"""The proposed acceptance gate for a recovered person-frame, as one function.

Every number is backed by gate_analysis.py / final_eval.py (see the report). A recovery is a gated
triangulation of >= 2 pixel-read 2D poses; the hint only selects WHICH 2D poses; it never enters
the position. Features are computed from the recovered pose and the measured poses of OTHER
identities at the same instant.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 11), (6, 12), (5, 6), (11, 12)]

# proposed tiers (see report): A = three or more cameras, B = two cameras (stricter)
TIERS = {
    # three or more cameras: engine gates + body completeness + plausible bones + near the hint
    "A": dict(min_cams=3, min_body=9, max_resid=12.0, max_bad_bones=1, max_hint=15.0, min_other=24.0, min_peak=0.0, min_agree=0.0),
    # two cameras: much stricter on image evidence (tight residual, confident heatmaps) and hint
    "B": dict(min_cams=2, min_body=8, max_resid=3.0, max_bad_bones=2, max_hint=12.0, min_other=24.0, min_peak=0.6, min_agree=0.0),
}


class Context:
    """Per-workspace lookups for the gate (bone ranges from measured poses, measured roots per instant)."""

    def __init__(self, tracks: dict):
        ok = np.isfinite(tracks["xyz"][:, 5:17, 0]).all(axis=1)
        sample = tracks["xyz"][np.flatnonzero(ok)[::10]].astype(np.float64)
        self.range = np.array([np.percentile(np.linalg.norm(sample[:, a] - sample[:, b], axis=1), [0.5, 99.5]) for a, b in BONES])
        self.by_t = defaultdict(list)
        for row in np.flatnonzero(np.isfinite(tracks["root"]).all(axis=1)):
            self.by_t[int(tracks["t"][row])].append(row)
        self.tracks = tracks

    def bad_bones(self, xyz: np.ndarray) -> int:
        bad = 0
        for (a, b), (lo, hi) in zip(BONES, self.range):
            if np.isfinite(xyz[a, 0]) and np.isfinite(xyz[b, 0]):
                bad += not (lo <= np.linalg.norm(xyz[a] - xyz[b]) <= hi)
        return bad

    def other_distance(self, who: int, t: int, root: np.ndarray) -> float:
        best = np.inf
        for r in self.by_t.get(t, []):
            if int(self.tracks["identity"][r]) != who:
                best = min(best, float(np.linalg.norm(self.tracks["root"][r][:2] - root[:2])))
        return best


def features(ctx: Context, who: int, t: int, rec: dict) -> dict:
    agree = rec.get("agree_hint") or []
    agree = [a for a in agree if a == a]
    return {"cams": len(rec["cams"]), "n_body": rec["n_body"], "resid": rec["median_residual"], "min_peak": rec["min_peak"],
            "agree": min(agree) if agree else 1.0, "bad_bones": ctx.bad_bones(rec["xyz"]),
            "hint": rec["hint_dist"] if "hint_dist" in rec else rec.get("hint_dist_xy"),
            "other": ctx.other_distance(who, t, np.asarray(rec["root"]))}


def accept(f: dict, tiers=TIERS) -> str | None:
    """Tier name that accepts the recovery, else None."""
    for name, g in tiers.items():
        if (f["cams"] >= g["min_cams"] and f["n_body"] >= g["min_body"] and f["resid"] <= g["max_resid"]
                and f["bad_bones"] <= g["max_bad_bones"] and f["hint"] <= g["max_hint"] and f["other"] >= g["min_other"]
                and f["min_peak"] >= g["min_peak"] and f["agree"] >= g["min_agree"]):
            return name
    return None
