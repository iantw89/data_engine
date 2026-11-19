"""Clean timings (nothing else of mine running): frame decode and ViTPose crops, for the cost extrapolation."""
import json
import threading
import time

from common import *  # noqa
from pixel_pass import PixelPass

ws, scene, tracks, solution = load()
out = {}
# --- decode: one camera, then seven in parallel
t = time.time()
for f in range(2000, 2200):
    ws.frames.read("golden-0", f)
out["decode_fps_1_camera"] = round(200 / (time.time() - t), 1)


def run(cam, start, n):
    for f in range(start, start + n):
        ws.frames.read(cam, f)


t = time.time()
threads = [threading.Thread(target=run, args=(c.camera, 3000, 300)) for c in scene.cameras]
[x.start() for x in threads]
[x.join() for x in threads]
out["decode_fps_7_cameras_parallel_total"] = round(7 * 300 / (time.time() - t), 1)

# --- ViTPose on preloaded frames: crops per second (2 passes per crop, as in pixel_pass)
p = PixelPass(ws, scene)
images = [ws.frames.read(scene.cameras[c].camera, 3100).copy() for c in range(7)]
boxes = []
for c in range(7):
    k = scene.cameras[c].detections.rows(int(3100))
    boxes.append(np.array([[600, 400, 760, 760]] * 12, np.float32))
for repeat in range(2):
    t = time.time()
    n = 0
    for _ in range(20):
        for start in range(0, 7, 4):
            part = images[start:start + 4]
            rows = [boxes[start + i] for i in range(len(part))]
            sources = p.vit.upload(part)
            shapes = [(im.shape[1], im.shape[0]) for im in part]
            first = p._decode(sources, rows, shapes)
            p._decode(sources, rows, shapes)
            n += sum(len(r) for r in rows)
    dt = time.time() - t
out["vitpose_crops_per_second_incl_upload_and_second_pass"] = round(n / dt, 1)
print(json.dumps(out, indent=1))
(SCRATCH / "benchmark.json").write_text(json.dumps(out, indent=1))
