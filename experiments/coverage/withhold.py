"""Withhold test sampling: hide measured person-frames, keep their neighbours as the hint source."""
from __future__ import annotations

from common import *  # noqa: F401,F403
from hints import Presence

LENGTHS = [1, 1, 2, 3, 5, 8, 10, 15, 20]  # mix like the real gap distribution (318 of 375 gaps <= 10)


def sample_runs(presence: Presence, tracks: dict, rng, target_frames: int = 3000, min_separation: int = 3):
    """List of runs {identity, ts, kind}: consecutive measured frames bracketed by measured frames on both sides."""
    runs, taken, total = [], {}, 0
    identities = list(presence.t)
    attempts = 0
    while total < target_frames and attempts < 200000:
        attempts += 1
        who = identities[rng.integers(len(identities))]
        ts = presence.t[who]
        if len(ts) < 60:
            continue
        length = int(LENGTHS[rng.integers(len(LENGTHS))])
        start = int(rng.integers(1, len(ts) - length - 1))
        run = ts[start:start + length]
        if run[-1] - run[0] != length - 1:  # must be consecutive frames
            continue
        before, after = ts[start - 1], ts[start + length]
        kind = "interior"
        if before != run[0] - 1 or after != run[-1] + 1:
            continue
        used = taken.setdefault(who, [])
        if any(not (run[-1] + min_separation < a or b + min_separation < run[0]) for a, b in used):
            continue
        used.append((int(run[0]), int(run[-1])))
        runs.append({"identity": int(who), "ts": [int(x) for x in run], "kind": kind, "length": length})
        total += length
    return runs


def withheld_sets(runs):
    out = {}
    for r in runs:
        out.setdefault(r["identity"], set()).update(r["ts"])
    return out
