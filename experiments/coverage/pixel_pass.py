"""GPU pass: ViTPose (production TensorRT FP16 path of data_management) on arbitrary boxes.

Frames are decoded once, sequentially per camera (one thread each: ~75 fps per camera; random
access would cost ~0.75 s per frame). ``run`` takes jobs (camera, native frame, box) and returns
the decoded keypoints exactly as the detector stage would (UDP/DARK decode, x/y clipped to the
image, heatmap peak as confidence). Each job is run twice: on the given box, then on the box
re-derived from the first result's keypoints (calibrated margins), the usual top-down refinement.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict

import cv2

from common import *  # noqa: F401,F403
from hints import keypoint_box

VIT = r"C:\Users\iantw\git\data_management\models\vitpose-curated-v6.onnx"


class PixelPass:
    def __init__(self, ws, scene):
        from pathlib import Path
        from data_management.live_vitpose import GpuViTPose
        self.ws, self.scene = ws, scene
        self.vit = GpuViTPose(Path(VIT))
        self.size = {c.camera: (ws.dataset.cameras[c.camera].width, ws.dataset.cameras[c.camera].height) for c in scene.cameras}

    def _decode(self, sources, rows, shapes):
        from data_management.live_vitpose import decode_peaks
        peaks = self.vit.peaks_from_sources(sources, rows)
        boxes = np.concatenate(rows)
        kp = decode_peaks(peaks, boxes)
        out, offset = [], 0
        for r, (w, h) in zip(rows, shapes):
            part = kp[offset:offset + len(r)].copy()
            part[:, :, 0] = part[:, :, 0].clip(0, w)
            part[:, :, 1] = part[:, :, 1].clip(0, h)
            part[:, :, 2] = part[:, :, 2].clip(0, 1)
            out.append(part)
            offset += len(r)
        return out

    @staticmethod
    def refine_box(kp, box, peak=0.3):
        ok = kp[:, 2] >= peak
        if ok.sum() < 8:
            return box
        refined = keypoint_box(kp[ok, :2], 0, 0)
        return box if refined is None else refined

    def run(self, cam, frame, box, *, log_every=2000):
        """cam (n,), frame (n,) native frame, box (n,4) -> dict(kp1, kp2, box2) arrays."""
        n = len(cam)
        kp1 = np.full((n, 17, 3), np.nan, np.float32)
        kp2 = np.full((n, 17, 3), np.nan, np.float32)
        box2 = np.full((n, 4), np.nan, np.float32)
        groups = defaultdict(list)
        for i in range(n):
            groups[(int(cam[i]), int(frame[i]))].append(i)
        per_cam = defaultdict(list)
        for (c, f) in groups:
            per_cam[c].append(f)
        sink: queue.Queue = queue.Queue(maxsize=24)
        stop = threading.Event()
        errors = []

        def decode(c):
            name = self.scene.cameras[c].camera
            try:
                for f in sorted(per_cam[c]):
                    if stop.is_set():
                        return
                    sink.put((c, f, self.ws.frames.read(name, f).copy()))
            except Exception as exc:  # surfaced below
                errors.append(f"{name}: {exc!r}")
                stop.set()
            finally:
                sink.put((c, None, None))

        threads = [threading.Thread(target=decode, args=(c,), daemon=True) for c in per_cam]
        for t in threads:
            t.start()
        alive, done, started = len(threads), 0, time.time()
        batch = []

        def flush(batch):
            images = [b[2] for b in batch]
            idx = [groups[(b[0], b[1])] for b in batch]
            rows = [np.asarray(box[i], np.float32).reshape(-1, 4) for i in idx]
            shapes = [(im.shape[1], im.shape[0]) for im in images]
            sources = self.vit.upload(images)
            first = self._decode(sources, rows, shapes)
            second_boxes = []
            for r, k in zip(rows, first):
                second_boxes.append(np.stack([self.refine_box(k[j], r[j]) for j in range(len(r))]).astype(np.float32))
            second = self._decode(sources, second_boxes, shapes)
            for i, a, b, bx in zip(idx, first, second, second_boxes):
                kp1[i], kp2[i], box2[i] = a, b, bx

        while alive:
            c, f, image = sink.get()
            if f is None:
                alive -= 1
                continue
            batch.append((c, f, image))
            if len(batch) == 8:
                flush(batch)
                done += len(batch)
                batch = []
                if (done // 8) % 100 == 0:
                    print(f"  pass: {done}/{len(groups)} frames, {time.time() - started:.0f} s", flush=True)
        if batch:
            flush(batch)
        if errors:
            raise RuntimeError("; ".join(errors))
        return {"kp1": kp1, "kp2": kp2, "box2": box2, "seconds": time.time() - started, "frames": len(groups), "crops": n}
