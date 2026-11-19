"""Evidence sheets for swap-check entries: rows = A before, A after, B before, B after; columns = cameras.

    python experiments/switches/sheets.py WORKSPACE RESULT.json OUT_DIR [--status flagged] [--count 8] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from data_engine.config import RECIPE  # noqa: E402
from data_engine.solve import montage  # noqa: E402
from data_engine.solve.scene import load_scene  # noqa: E402
from data_engine.store import parse_det_key  # noqa: E402
from data_engine.workspace import Workspace  # noqa: E402

HEIGHT = 230


def strip(scene, keys, label, colour, columns=3):
    cells = []
    index = {camera.camera: camera for camera in scene.cameras}
    for key in keys[:columns]:
        camera, frame, det = parse_det_key(key)
        row = index[camera].detections.row(frame, det)
        patch = montage.crop(scene, index[camera].index, row, height=HEIGHT, tint=colour)
        scale = HEIGHT / patch.shape[0]
        patch = cv2.resize(patch, (max(1, int(patch.shape[1] * scale)), HEIGHT))
        cv2.putText(patch, f"{camera[-1]} t={int(index[camera].t[row])}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cells.append(patch)
    width = 3 * 230
    canvas = np.zeros((HEIGHT, width + 90, 3), np.uint8)
    cv2.putText(canvas, label, (4, HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    x = 90
    for cell in cells:
        cell = cell[:, :max(0, min(cell.shape[1], width + 90 - x))]
        canvas[:, x:x + cell.shape[1]] = cell
        x += cell.shape[1] + 4
        if x >= width + 90:
            break
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace")
    parser.add_argument("result")
    parser.add_argument("out")
    parser.add_argument("--status", default="flagged")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cause")
    parser.add_argument("--kind")
    args = parser.parse_args()
    workspace = Workspace(args.workspace)
    scene = load_scene(workspace, joint_peak=RECIPE["geometry"]["joint_peak"], rays=False)
    entries = [e for e in json.loads(Path(args.result).read_text(encoding="utf-8"))["entries"]
               if e["status"] == args.status and e.get("reference_keys")]
    if args.kind:
        entries = [e for e in entries if e["kind"] == args.kind]
    if args.cause:
        entries = [e for e in entries if "+".join(e["swap_cues"]) == args.cause]
    rng = np.random.default_rng(args.seed)
    picks = [entries[i] for i in rng.choice(len(entries), size=min(args.count, len(entries)), replace=False)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for number, entry in enumerate(picks):
        rows = [strip(scene, entry["reference_keys"], f"A={entry['a']} pre", (0, 255, 0)),
                strip(scene, entry["after_a_keys"], f"A={entry['a']} post", (0, 255, 0)),
                strip(scene, entry["before_b_keys"], f"B={entry['b']} pre", (255, 0, 255)),
                strip(scene, entry["after_b_keys"], f"B={entry['b']} post", (255, 0, 255))]
        sheet = np.vstack(rows)
        text = (f"{entry['kind']} t={entry['t']} d={entry['min_distance_m']}m swap={entry['swap_cues']} stay={entry['stay_cues']} "
                f"head={entry['cues'].get('heading_mps')} body={entry['cues']['body']} "
                f"trk -{entry['cues']['tracklets_contradicting']}/+{entry['cues']['tracklets_agreeing']}")
        header = np.zeros((22, sheet.shape[1], 3), np.uint8)
        cv2.putText(header, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        name = out / f"{args.status}_{args.kind or 'any'}_{number:02d}.jpg"
        cv2.imwrite(str(name), np.vstack([header, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(name, text)


if __name__ == "__main__":
    main()
