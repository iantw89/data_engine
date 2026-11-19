"""Does our ViTPose pass reproduce the stored detections when fed the stored YOLO boxes?

Also: how stable are keypoints under the second (re-boxed) iteration?
"""
import json
import time

from common import *  # noqa
from pixel_pass import PixelPass

ws, scene, tracks, solution = load()
rng = np.random.default_rng(1)
cam, frame, box, ref, rows_ = [], [], [], [], []
for c, camera in enumerate(scene.cameras):
    pool = camera.order[rng.permutation(len(camera.order))[:300]]
    for r in pool:
        cam.append(c)
        frame.append(int(camera.detections.frame_of_row[r]))
        box.append(camera.detections.boxes[r, :4])
        ref.append(camera.detections.keypoints[r])
cam, frame, box, ref = np.array(cam), np.array(frame), np.array(box, np.float32), np.array(ref)
t = time.time()
p = PixelPass(ws, scene)
print("model ready", time.time() - t)
out = p.run(cam, frame, box)
print("pass", out["seconds"], "s for", out["frames"], "frames", out["crops"], "crops")
res = {}
for name in ("kp1", "kp2"):
    kp = out[name]
    d = np.linalg.norm(kp[:, :, :2] - ref[:, :, :2], axis=-1)
    ok = ref[:, :, 2] >= 0.3
    res[name] = {"px_diff_p50_p90_p99": np.percentile(d[ok], [50, 90, 99]).round(2).tolist(),
                 "peak_diff_p50_p90_p99": np.percentile(np.abs(kp[:, :, 2] - ref[:, :, 2])[ok], [50, 90, 99]).round(3).tolist()}
print(json.dumps(res, indent=1))
(SCRATCH / "fidelity_check.json").write_text(json.dumps(res, indent=1))
