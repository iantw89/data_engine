"""Render the reference crops of the largest open unknowns of a kind (contact sheet).

Usage: python experiments/bench/show_unknowns.py WORKSPACE KIND [count] [out.jpg]
"""
import sys

import cv2
import numpy as np

from data_engine.render import render_panel
from data_engine.workspace import Workspace

workspace = Workspace(sys.argv[1])
kind = sys.argv[2]
count = int(sys.argv[3]) if len(sys.argv) > 3 else 8
out = sys.argv[4] if len(sys.argv) > 4 else str(workspace.path(f"unknowns_{kind}.jpg"))
rows = []
for unknown in workspace.db.unknowns(status="open", kind=kind):
    detail = unknown["detail"]
    keys = detail.get("reference_keys") or detail.get("keys") or []
    if not keys:
        continue
    tiles = []
    for key in keys[:3]:
        camera, frame, _ = key.rsplit("/", 2)
        box, _ = workspace.detections.lookup(key)
        panel = {"camera": camera, "frame": int(frame), "box": [float(v) for v in box[:4]], "highlight": [key],
                 "others": True, "project": [], "label": "", "color": "green", "context": False}
        image = cv2.imdecode(np.frombuffer(render_panel(workspace, panel, max_side=360), np.uint8), cv2.IMREAD_COLOR)
        tile = np.zeros((360, 360, 3), np.uint8)
        tile[:image.shape[0], :image.shape[1]] = image[:360, :360]
        tiles.append(tile)
    while len(tiles) < 3:
        tiles.append(np.zeros((360, 360, 3), np.uint8))
    # one full-frame view with the box, for room context
    panel["context"] = True
    full = cv2.imdecode(np.frombuffer(render_panel(workspace, panel, max_side=480, full=True), np.uint8), cv2.IMREAD_COLOR)
    wide = np.zeros((360, 480, 3), np.uint8)
    wide[:min(360, full.shape[0]), :min(480, full.shape[1])] = full[:360, :480]
    strip = np.hstack(tiles + [wide])
    text = f"{unknown['subject'][:60]}  dets={detail.get('detections')}  t={detail.get('span_t')}"
    cv2.putText(strip, text, (6, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
    cv2.putText(strip, text, (6, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    rows.append(strip)
    if len(rows) >= count:
        break
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(out, len(rows))
