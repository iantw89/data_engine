"""Run the engine's swap check (src/data_engine/solve/swap_check.py) on a saved solution.

    python experiments/switches/run_swap_check.py WORKSPACE [--inject N] [--out FILE]

With --inject N two recall tests that need no golden labels (labels exchanged, physical data unchanged):
  apart    - at N random instants where two identities are within the radius but NOT at a crossing,
             exchange their labels from then on (and, separately, a takeover): the check must find a
             moment there and must not verify it.
  crossing - at N of the crossings the check found, exchange the labels from the middle of the
             crossing on: the verdict must not be "verified" (flagged or unverified are both reviewed).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from data_engine.config import RECIPE  # noqa: E402
from data_engine.solve import swap_check  # noqa: E402
from data_engine.solve.scene import load_scene  # noqa: E402
from data_engine.workspace import Workspace  # noqa: E402


def load(workspace_id: str):
    workspace = Workspace(workspace_id)
    scene = load_scene(workspace, joint_peak=RECIPE["geometry"]["joint_peak"], rays=False)
    directory = Path(sorted(glob.glob(str(workspace.root / "rounds" / "*" / "tracks3d.npz")))[-1]).parent
    tracks = dict(np.load(directory / "tracks3d.npz"))
    solution = dict(np.load(directory / "solution.npz"))
    row_identity = [solution[f"identity_{camera.camera}"].astype(np.int64) for camera in scene.cameras]
    return scene, row_identity, tracks, directory


def exchanged_index(scene, row_identity, tracks, a: int, b: int, tau: int, takeover: bool, full: swap_check.Index):
    """Index of only a and b, with their labels exchanged from sync ``tau`` on (takeover: a's person is lost)."""
    rows = []
    for camera in scene.cameras:
        identity = row_identity[camera.index]
        out = np.full(len(identity), -1, np.int64)
        later = camera.t >= tau
        is_a, is_b = identity == a, identity == b
        out[is_a & ~later] = a
        out[is_b & ~later] = b
        out[is_b & later] = a
        if not takeover:
            out[is_a & later] = b
        rows.append(out)
    identity = tracks["identity"].astype(np.int64)
    out = np.full(len(identity), -1, np.int64)
    later = tracks["t"] >= tau
    is_a, is_b = identity == a, identity == b
    out[is_a & ~later] = a
    out[is_b & ~later] = b
    out[is_b & later] = a
    if not takeover:
        out[is_a & later] = b
    index = swap_check.Index.build(scene, rows, out, tracks["t"], tracks["root"], tracks["xyz"], window=full.window)
    index.sigma = full.sigma
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace")
    parser.add_argument("--inject", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()
    scene, row_identity, tracks, directory = load(args.workspace)
    recipe = RECIPE["swap_check"]
    identity = tracks["identity"].astype(np.int64)
    result = swap_check.check(scene, row_identity, identity, tracks["t"], tracks["root"], tracks["xyz"], recipe)
    minutes = scene.T / 600
    entries = result["entries"]
    print(directory)
    print(json.dumps(result["summary"], indent=1))
    review = [e for e in entries if e["status"] != "verified"]
    print(f"review questions: {len(review)} ({len(review) / minutes * 10:.1f} per 10 min)")
    print("by kind/status:", dict(Counter((e["kind"], e["status"]) for e in entries)))
    print("flag causes:", dict(Counter("+".join(e["swap_cues"]) for e in entries if e["status"] == "flagged")))
    print("stay cues of verified:", dict(Counter("+".join(e["stay_cues"]) for e in entries if e["status"] == "verified")))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    if not args.inject:
        return
    rng = np.random.default_rng(0)
    full = swap_check.Index.build(scene, row_identity, identity, tracks["t"], tracks["root"], tracks["xyz"],
                                  window=recipe["window_frames"])
    # ---- apart: exchange labels where the pair is close but continuity is unambiguous
    pairs = swap_check.candidate_pairs(full, recipe["radius_m"])
    busy = {(e["a"], e["b"]): [] for e in entries}
    for e in entries:
        busy[(e["a"], e["b"])].append(e["t"])
    candidates = []
    for a, b in pairs:
        if swap_check.pair_steps(full, a, b, recipe)["duplicate"]:
            continue
        ta, ra, _ = full.track(a, 0, scene.T)
        tb, rb, _ = full.track(b, 0, scene.T)
        common, ia, ib = np.intersect1d(ta, tb, return_indices=True)
        near = np.linalg.norm(ra[ia] - rb[ib], axis=1) * swap_check.INCH < recipe["radius_m"]
        for t in common[near][5:-5]:
            if all(not (s - 30 <= t <= e + 30) for s, e in busy.get((a, b), [])):
                candidates.append((a, b, int(t)))
    picks = [candidates[i] for i in rng.choice(len(candidates), size=min(args.inject, len(candidates)), replace=False)]
    for takeover in (False, True):
        outcome = Counter()
        for a, b, tau in picks:
            index = exchanged_index(scene, row_identity, tracks, a, b, tau, takeover, full)
            found, _ = swap_check.moments(index, scene.T, recipe)
            hits = [m for m in found if m["t"][0] - 15 <= tau <= m["t"][1] + 15]
            if not hits:
                outcome["no moment found (MISSED)"] += 1
                continue
            statuses = {swap_check.evaluate(index, m, recipe)["status"] for m in hits}
            outcome["flagged" if "flagged" in statuses else "unverified (reviewed)" if "unverified" in statuses
                    else "verified (MISSED)"] += 1
        print(f"injected {'takeover' if takeover else 'swap'} while apart ({len(picks)} of {len(candidates)} instants):",
              dict(outcome))
    # ---- crossing: exchange labels in the middle of crossings the check found
    crossings = [e for e in entries if e["kind"] == "crossing"]
    picks = [crossings[i] for i in rng.choice(len(crossings), size=min(args.inject, len(crossings)), replace=False)]
    for takeover in (False, True):
        outcome, was = Counter(), Counter()
        for entry in picks:
            s, e = entry["t"]
            index = exchanged_index(scene, row_identity, tracks, entry["a"], entry["b"], (s + e + 1) // 2, takeover, full)
            verdict = swap_check.evaluate(index, {"kind": "crossing", "a": entry["a"], "b": entry["b"], "t": (s, e)}, recipe)
            outcome[verdict["status"] if verdict["status"] != "verified" else "verified (MISSED)"] += 1
            if verdict["status"] == "verified":
                was[entry["status"]] += 1
        print(f"injected {'takeover' if takeover else 'swap'} at a crossing ({len(picks)}):", dict(outcome),
              "| missed ones were originally:", dict(was))


if __name__ == "__main__":
    main()
