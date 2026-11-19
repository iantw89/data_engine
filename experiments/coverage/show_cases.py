"""Montage of recovered poses for chosen cases: crop around the box per used camera, skeleton + hint box.

Usage: python experiments/coverage/show_cases.py eval_name key out.jpg id [id ...]
"""
import pickle
import sys

import cv2

from common import *  # noqa
from data_engine.geometry import SKELETON

name, key, out = sys.argv[1], sys.argv[2], sys.argv[3]
ids = [int(x) for x in sys.argv[4:]]
ws, scene, tracks, solution = load()
cases = pickle.loads((SCRATCH / "cases.pkl").read_bytes())
ev = {e["id"]: e for e in pickle.loads((SCRATCH / f"eval_{name}.pkl").read_bytes())}
jobs = np.load(SCRATCH / "jobs.npz")
vit = np.load(SCRATCH / "pass_vit.npz")
job_of = {(int(c), int(k)): j for j, (c, k) in enumerate(zip(jobs["case"], jobs["cam"]))}
tiles = []
for cid in ids:
    case, rec = cases[cid], ev[cid][key]
    row_tiles = []
    for cam in rec["cams"][:4]:
        j = job_of[(cid, cam)]
        frame = ws.frames.read(scene.cameras[cam].camera, int(scene.native[case["t"], cam])).copy()
        kp = vit["kp2"][j]
        box = case["views"][cam].astype(int)
        cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 255, 255), 2)
        for a, b in SKELETON:
            if kp[a, 2] >= 0.3 and kp[b, 2] >= 0.3:
                cv2.line(frame, tuple(int(v) for v in kp[a, :2]), tuple(int(v) for v in kp[b, :2]), (0, 255, 0), 2)
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        side = int(max(box[3] - box[1], box[2] - box[0]) * 1.6) + 40
        x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
        tile = frame[y0:y0 + side, x0:x0 + side]
        tile = cv2.resize(tile, (320, 320))
        cv2.putText(tile, f"#{cid} {case['tag']} c{cam} n{rec['n_body']} r{rec['median_residual']:.1f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        row_tiles.append(tile)
    while len(row_tiles) < 4:
        row_tiles.append(np.zeros((320, 320, 3), np.uint8))
    tiles.append(np.hstack(row_tiles))
cv2.imwrite(out, np.vstack(tiles))
print("wrote", out)
