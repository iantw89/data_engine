"""Sheets of the identity links the solve accepted only with recovered positions, to check them by eye.

    python experiments/coverage/link_sheets.py WORKSPACE OUT_DIR [--round rNNN] [--measured LARGEST,RANDOM]

Without --measured: the links accepted (or proposed) only with recovered positions. With --measured 6,6: of
the links accepted on measured positions, the 6 with the largest distance and 6 drawn at random.
One PNG per link: the top row shows the earlier track at its last detections, the bottom row the later
track at its first detections, one column per camera. Both rows must show the same person.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from data_engine.config import RECIPE  # noqa: E402
from data_engine.solve.montage import crop  # noqa: E402
from data_engine.solve.scene import load_scene  # noqa: E402
from data_engine.workspace import Workspace  # noqa: E402

HEIGHT, CELL = 300, 330


def strip(scene, solution, component: int, last: bool, label: str) -> np.ndarray:
    out = np.zeros((HEIGHT + 30, 20 + CELL * len(scene.cameras), 3), np.uint8)
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    for camera in scene.cameras:
        rows = np.flatnonzero((solution[f"component_{camera.camera}"] == component) & (camera.t >= 0))
        if not len(rows):
            continue
        row = int(rows[np.argmax(camera.t[rows])] if last else rows[np.argmin(camera.t[rows])])
        patch = crop(scene, camera.index, row, height=HEIGHT, tint=(0, 255, 0) if last else (0, 200, 255))
        x = 10 + CELL * camera.index
        out[30:, x:x + min(CELL, patch.shape[1])] = patch[:, :CELL]
        cv2.putText(out, f"{camera.camera} t={int(camera.t[row]) / 10:.1f}s", (x, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1)
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("workspace")
    parser.add_argument("out_dir")
    parser.add_argument("--round")
    parser.add_argument("--measured")
    args = parser.parse_args()
    workspace = Workspace(args.workspace)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    directory = workspace.root / "rounds" / args.round if args.round else         Path(sorted(glob.glob(str(workspace.root / "rounds" / "*" / "solve.json")))[-1]).parent
    log = json.loads((directory / "solve.json").read_text(encoding="utf-8"))
    if args.measured:
        largest, random = (int(part) for part in args.measured.split(","))
        measured = [link for link in log["links"] if not link.get("after_recovery") and "distance_m" in link]
        measured.sort(key=lambda link: -link["distance_m"])
        rest = measured[largest:]
        picks = np.random.default_rng(0).choice(len(rest), size=min(random, len(rest)), replace=False) if rest else []
        links = measured[:largest] + [rest[int(i)] for i in sorted(picks)]
        print(f"{len(measured)} links on measured positions; distance p50 {np.median([l['distance_m'] for l in measured]):.2f} m, "
              f"over 0.75 m: {sum(l['distance_m'] > 0.75 for l in measured)}, over 2 m: {sum(l['distance_m'] > 2 for l in measured)}")
    else:
        links = [link for link in log["links"] if link.get("after_recovery")] +             [link for event in log["events"] for link in event.get("links_proposed_after_recovery", [])]
    scene = load_scene(workspace, joint_peak=RECIPE["geometry"]["joint_peak"], rays=False)
    solution = dict(np.load(directory / "solution.npz"))
    for link in links:
        text = f"gap {link['gap_s']} s, {link['distance_m']} m, {link['source'].split(':')[-1]}"
        sheet = np.vstack([strip(scene, solution, link["from"], True, f"earlier track {link['from']} at its end   ({text})"),
                           strip(scene, solution, link["to"], False, f"later track {link['to']} at its start")])
        path = out_dir / f"link_{link['from']}_{link['to']}.png"
        cv2.imwrite(str(path), sheet)
        print(path)


if __name__ == "__main__":
    main()
