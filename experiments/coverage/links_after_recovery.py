"""Does the coverage recovery make re-identification possible that was not possible before?

    python experiments/coverage/links_after_recovery.py WORKSPACE [--golden DATASET]

The solve links two tracks across a gap by physics (`solve/identity.py`): the later one must start within
reach of where the earlier one was last measured, and be the unique such continuation in both directions.
The recovery measures positions before a track's first and after its last measured position and inside its
gaps, so the "last measured" place and time of a track move. The solve does not see that.

This runs the solve's own linking rule on the final identities twice: with the measured positions only
(nothing new may come out - the solve already ran it), and with the recovered positions added. It reports
the links that become possible, how the open ends and their candidate lists change and, with --golden,
whether each new link joins one golden person.
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from data_engine.config import RECIPE  # noqa: E402
from data_engine.solve.identity import link_components, summarise  # noqa: E402
from data_engine.solve.scene import load_scene  # noqa: E402
from data_engine.workspace import Workspace  # noqa: E402


def frames_of(parts: list[dict], index_of: dict[int, int]) -> SimpleNamespace:
    keep = [part["identity"] >= 0 for part in parts]
    label = np.concatenate([np.array([index_of[int(who)] for who in part["identity"][mask]], np.int64)
                            for part, mask in zip(parts, keep)])
    return SimpleNamespace(
        label=label, t=np.concatenate([part["t"][mask].astype(np.int64) for part, mask in zip(parts, keep)]),
        rows=np.concatenate([part["rows"][mask] for part, mask in zip(parts, keep)]),
        root=np.concatenate([part["root"][mask].astype(np.float64) for part, mask in zip(parts, keep)]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace")
    parser.add_argument("--golden")
    args = parser.parse_args()
    workspace = Workspace(args.workspace)
    directory = Path(sorted(glob.glob(str(workspace.root / "rounds" / "*" / "tracks3d_recovered.npz")))[-1]).parent
    tracks = dict(np.load(directory / "tracks3d.npz"))
    recovered = dict(np.load(directory / "tracks3d_recovered.npz"))
    T = len(tracks["aligned_time"])
    print(directory)
    identities = sorted(int(who) for who in np.unique(tracks["identity"]) if who >= 0)
    index_of = {who: index for index, who in enumerate(identities)}
    recovered["identity"] = np.where(np.isin(recovered["identity"], identities), recovered["identity"], -1)

    person_of = {}
    if args.golden:
        from data_engine.config import SITE_DEFAULTS
        from data_engine.evaluation import load_reference

        scene = load_scene(workspace, joint_peak=RECIPE["geometry"]["joint_peak"], rays=False)
        solution = dict(np.load(directory / "solution.npz"))
        reference = load_reference(scene, Path(SITE_DEFAULTS["repos"]["data_management"]) / "data" / "labels", args.golden)
        votes = defaultdict(Counter)
        for camera in scene.cameras:
            identity = solution[f"identity_{camera.camera}"]
            both = (identity >= 0) & (reference.person[camera.index] >= 0)
            for who, labelled in zip(identity[both], reference.person[camera.index][both]):
                votes[int(who)][int(labelled)] += 1
        person_of = {who: counts.most_common(1)[0][0] for who, counts in votes.items()}

    results = {}
    for name, parts in (("measured only", [tracks]), ("measured + recovered", [tracks, recovered])):
        frames = frames_of(parts, index_of)
        summaries = summarise(frames, len(identities))
        links = link_components(summaries, RECIPE, T=T, frames=frames)
        results[name] = (summaries, links)
        candidates = [len(end["candidates"]) for end in links.open_ends]
        tight = sum(1 for end in links.open_ends
                    if any(c["gap_s"] <= 5.0 and c["distance_m"] <= 0.75 + 2.0 * c["gap_s"] for c in end["candidates"]))
        print(f"\n{name}: {len(identities)} identities, {len(links.accepted)} links accepted by the solve's rule, "
              f"{len(links.open_ends)} open ends")
        print(f"  open ends without any candidate {sum(1 for c in candidates if c == 0)}, with exactly one "
              f"{sum(1 for c in candidates if c == 1)}, with several {sum(1 for c in candidates if c > 1)} "
              f"(median {np.median(candidates) if candidates else 0:.0f}); with a candidate within the automatic gap and radius {tight}")
        verdict = Counter()
        for link in links.accepted:
            a, b = identities[link["from"]], identities[link["to"]]
            truth = ""
            if person_of:
                pa, pb = person_of.get(a), person_of.get(b)
                kind = "not labelled" if pa is None or pb is None else "same golden person" if pa == pb else "DIFFERENT golden people"
                verdict[kind] += 1
                truth = f"  [{kind}]"
            sa, sb = summaries[link["from"]], summaries[link["to"]]
            print(f"    {a} -> {b}: gap {link['gap_s']} s, {link['distance_m']} m, {link['source'].split(':')[-1]}; "
                  f"end of the first moved by {(sa.last_root_t - results['measured only'][0][link['from']].last_root_t) / 10:.1f} s, "
                  f"start of the second by {(results['measured only'][0][link['to']].first_root_t - sb.first_root_t) / 10:.1f} s{truth}")
        if person_of and links.accepted:
            print(f"  golden: {dict(verdict)}")

    before, after = results["measured only"][0], results["measured + recovered"][0]
    moved_end = [(a.last_root_t - b.last_root_t) / 10 for a, b in zip(after, before) if a and b and a.last_root_t != b.last_root_t]
    moved_start = [(b.first_root_t - a.first_root_t) / 10 for a, b in zip(after, before) if a and b and a.first_root_t != b.first_root_t]
    for what, moved in (("ends moved later", moved_end), ("starts moved earlier", moved_start)):
        if moved:
            moved = np.array(moved)
            print(f"\ntrack {what}: {len(moved)} tracks, median {np.median(moved):.1f} s, over 5 s: {(moved > 5).sum()}, max {moved.max():.1f} s")


if __name__ == "__main__":
    main()
