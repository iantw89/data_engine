"""Lever 3: run a full detector (YOLO pose) on a zoomed ROI crop around the hint, then ViTPose on its boxes.

    python experiments/coverage/pass_detect_roi.py {yolov8x|yolo11l} [max_cases]

The ROI is a square of side max(1.6 x hint box height, 384 px) centred on the hint box, cut from the
native frame (grey padding outside the image); the detector letterboxes it to 1280, i.e. the person
is seen at 3-4x the scale of the production pass. Detections come back in native pixels (crop
offset added). Output: pass_<detector>.npz with ragged detections per job.
"""
import queue
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2

from common import *  # noqa
import pickle

MODELS = {"yolov8x": r"C:\Users\iantw\git\data_management\models\yolov8x-pose-p6-curated-v6-1280.onnx",
          "yolo11l": r"C:\Users\iantw\git\data_management\models\yolo11l-pose-1280.onnx"}
VIT = r"C:\Users\iantw\git\data_management\models\vitpose-curated-v6.onnx"


FACTOR, MIN_SIDE = 1.6, 384.0


def roi_for(box, w, h):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = max(FACTOR * (box[3] - box[1]), MIN_SIDE)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    return x0, y0, int(round(side))


def crop(image, x0, y0, side):
    h, w = image.shape[:2]
    out = np.full((side, side, 3), 114, np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return out


def main():
    global FACTOR, MIN_SIDE
    name = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    suffix = ""
    if len(sys.argv) > 4:  # zoom test: factor, min side, suffix; subset = withhold + gap only
        FACTOR, MIN_SIDE, suffix = float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    from data_management.live_detection import RollingPoseDetector
    from data_management.live_vitpose import RefinedPoseDetector
    from data_management.media import VideoFrame

    ws, scene, tracks, solution = load()
    jobs = np.load(SCRATCH / "jobs.npz")
    cases = pickle.loads((SCRATCH / "cases.pkl").read_bytes())
    # subset of cases: all gaps, a sample of the others (cost: one YOLO call per crop)
    rng = np.random.default_rng(5)
    keep = set()
    quota = (("gap", 10**9), ("withhold", 800), ("neg_shift", 500), ("neg_absent", 500), ("neg_empty", 500))
    if suffix:
        quota = (("gap", 10**9),) if suffix.endswith("gap") else (("withhold", 400), ("neg_shift", 200), ("neg_absent", 200))
    for tag, n in quota:
        ids = [c["id"] for c in cases if c["tag"] == tag]
        keep.update(int(i) for i in rng.permutation(ids)[:n])
    sel = np.flatnonzero(np.isin(jobs["case"], list(keep)))
    if limit:
        sel = sel[:limit]
    cam, frame, box = jobs["cam"][sel], jobs["frame"][sel], jobs["box"][sel]
    detector = RefinedPoseDetector(RollingPoseDetector(Path(MODELS[name]), imgsz=1280), Path(VIT))
    groups = defaultdict(list)
    for i in range(len(sel)):
        groups[(int(cam[i]), int(frame[i]))].append(i)
    per_cam = defaultdict(list)
    for (c, f) in groups:
        per_cam[c].append(f)
    sink: queue.Queue = queue.Queue(maxsize=16)

    def decode(c):
        for f in sorted(per_cam[c]):
            sink.put((c, f, ws.frames.read(scene.cameras[c].camera, f).copy()))
        sink.put((c, None, None))

    for c in per_cam:
        threading.Thread(target=decode, args=(c,), daemon=True).start()
    alive = len(per_cam)
    results = {}
    pending: list = []  # (job index, crop image, offset)
    started, done = time.time(), 0

    def flush(items):
        nonlocal done
        for k in range(0, len(items), 8):
            part = items[k:k + 8]
            frames = [VideoFrame(float(i), b"", pixels=img) for i, img, _ in part]
            poses = detector.infer(frames)
            for (i, img, (x0, y0)), pose in zip(part, poses):
                boxes = np.asarray(pose.boxes).copy()
                kps = np.asarray(pose.keypoints).copy()
                boxes[:, [0, 2]] += x0
                boxes[:, [1, 3]] += y0
                kps[:, :, 0] += x0
                kps[:, :, 1] += y0
                results[i] = (boxes, kps)
        done += len(items)
        if done % 400 < len(items):
            print(f"  roi: {done}/{len(sel)} crops {time.time() - started:.0f} s", flush=True)

    while alive:
        c, f, image = sink.get()
        if f is None:
            alive -= 1
            continue
        for i in groups[(c, f)]:
            x0, y0, side = roi_for(box[i], image.shape[1], image.shape[0])
            pending.append((i, crop(image, x0, y0, side), (x0, y0)))
        if len(pending) >= 32:
            flush(pending)
            pending = []
    if pending:
        flush(pending)
    counts = np.array([len(results[i][0]) for i in range(len(sel))])
    np.savez(SCRATCH / f"pass_{name}{suffix}.npz", sel=sel, counts=counts,
             boxes=np.concatenate([results[i][0] for i in range(len(sel))]),
             keypoints=np.concatenate([results[i][1] for i in range(len(sel))]), seconds=time.time() - started)
    print("done", time.time() - started, "s for", len(sel), "crops")


if __name__ == "__main__":
    main()
