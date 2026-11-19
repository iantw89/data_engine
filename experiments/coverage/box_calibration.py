"""How far does the detector's YOLO box extend beyond the bbox of its 17 keypoints?

Used to turn a projected skeleton (hint) into a detector-style box. Measured on detections that
are already members of 3D tracks (tracks3d rows), fractions of the keypoint-bbox height.
"""
import json
from common import *  # noqa

ws, scene, tracks, solution = load()
rows = tracks["rows"]
stats = []
for c, cam in enumerate(scene.cameras):
    r = rows[:, c]
    r = np.unique(r[r >= 0])
    b = cam.detections.boxes[r]
    k = cam.detections.keypoints[r]
    ok = k[:, :, 2] >= 0.3
    for bi, ki, oi in zip(b, k, ok):
        if oi.sum() < 8:
            continue
        p = ki[oi, :2]
        x1, y1 = p.min(0)
        x2, y2 = p.max(0)
        h = y2 - y1
        if h < 80:
            continue
        stats.append([(x1 - bi[0]) / h, (y1 - bi[1]) / h, (bi[2] - x2) / h, (bi[3] - y2) / h, h, (x2 - x1) / h])
s = np.array(stats)
q = {n: np.percentile(s[:, i], [5, 25, 50, 75, 95]).round(3).tolist() for i, n in
     enumerate(["left_gap/h", "top_gap/h", "right_gap/h", "bottom_gap/h", "kp_height_px", "kp_w/h"])}
print(len(s), json.dumps(q, indent=1))
(SCRATCH / "box_calibration.json").write_text(json.dumps({"n": len(s), "quantiles_5_25_50_75_95": q}, indent=1))
