"""Dense 2D pose detection over every native frame. Runs in data_management's interpreter.

Uses the production TensorRT path (fine-tuned YOLOv8x-pose-p6 + fine-tuned ViTPose refiner)
from data_management, which reproduces the adopted cvat_labeling single-view recipe at
roughly 20x the speed of the torch reference. Detection is independent of time alignment:
results are indexed by (camera, native frame) and stored in native pixels.

Output: ``<out>/<camera>/<first_frame:06d>.npz`` shards with
  frames   (n,)        native frame indices of the shard
  counts   (n,)        detections per frame
  boxes    (M,5)       xyxy + YOLO person score, native pixels
  keypoints(M,17,3)    COCO17 x, y, raw ViTPose heatmap peak (NOT multiplied by person score)
Shards are written atomically; rerunning resumes after the last complete shard.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_engine.dataset import Camera, Dataset  # noqa: E402

SHARD = 2000


def pending_shards(camera: Camera, out: Path) -> list[tuple[int, int]]:
    shards = []
    for start in range(0, camera.frame_count, SHARD):
        if not (out / camera.camera_id / f"{start:06d}.npz").is_file():
            shards.append((start, min(start + SHARD, camera.frame_count)))
    return shards


def decode_camera(camera: Camera, shards, sink: queue.Queue, stop: threading.Event, errors: list) -> None:
    from data_management.frame_index import ExactFrameReader

    reader = None
    try:
        for start, end in shards:
            for frame in range(start, end):
                if stop.is_set():
                    return
                path, local = camera.locate(frame)
                if reader is None or reader.path != path:
                    if reader is not None:
                        reader.close()
                    reader = ExactFrameReader(path)
                image = reader.read(local)
                if image.shape[1] != camera.width or image.shape[0] != camera.height:
                    raise RuntimeError(f"{camera.camera_id} frame {frame}: unexpected size {image.shape}")
                sink.put((camera.camera_id, frame, image))
    except Exception as exc:  # surfaced by the main thread; never swallowed
        errors.append(f"{camera.camera_id}: {type(exc).__name__}: {exc}")
        stop.set()
    finally:
        if reader is not None:
            reader.close()
        sink.put((camera.camera_id, None, None))


def write_shard(out: Path, camera_id: str, start: int, rows: dict) -> None:
    frames = np.array(sorted(rows), dtype=np.int32)
    boxes = [rows[int(frame)][0] for frame in frames]
    keypoints = [rows[int(frame)][1] for frame in frames]
    directory = out / camera_id
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f"{start:06d}.tmp.npz"
    np.savez_compressed(
        temporary, frames=frames, counts=np.array([len(b) for b in boxes], dtype=np.int32),
        boxes=np.concatenate(boxes).astype(np.float32) if boxes else np.empty((0, 5), np.float32),
        keypoints=np.concatenate(keypoints).astype(np.float32) if keypoints else np.empty((0, 17, 3), np.float32),
    )
    temporary.replace(directory / f"{start:06d}.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--yolo", required=True)
    parser.add_argument("--vitpose", required=True)
    parser.add_argument("--cameras", nargs="*")
    parser.add_argument("--limit", type=int, help="process only the first N frames per camera (smoke test)")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--progress", help="JSON progress file updated while running")
    args = parser.parse_args()

    from data_management.live_detection import RollingPoseDetector
    from data_management.live_vitpose import RefinedPoseDetector
    from data_management.media import VideoFrame

    dataset = Dataset.load(Path(args.dataset_json))
    out = Path(args.out)
    cameras = [dataset.cameras[c] for c in (args.cameras or sorted(dataset.cameras))]
    for camera in cameras:
        if camera.issues:
            raise SystemExit(f"{camera.camera_id}: unresolved stream layout: {camera.issues}")

    plan = {}
    for camera in cameras:
        shards = pending_shards(camera, out)
        if args.limit:
            shards = [(s, min(e, args.limit)) for s, e in shards if s < args.limit]
        plan[camera.camera_id] = shards
    total = sum(e - s for shards in plan.values() for s, e in shards)
    print(f"pending frames: {total}", flush=True)
    if not total:
        return

    detector = RefinedPoseDetector(RollingPoseDetector(Path(args.yolo), imgsz=1280), Path(args.vitpose))
    print(f"detector ready: {detector.device}", flush=True)

    stop = threading.Event()
    errors: list[str] = []
    sink: queue.Queue = queue.Queue(maxsize=8 * len(cameras))
    threads = [
        threading.Thread(target=decode_camera, args=(camera, plan[camera.camera_id], sink, stop, errors), daemon=True)
        for camera in cameras if plan[camera.camera_id]
    ]
    for thread in threads:
        thread.start()

    meta: deque = deque()  # per submitted batch: [(camera_id, frame), ...]

    def batches():
        alive, counter, current, ids = len(threads), 0, [], []
        while alive:
            camera_id, frame, image = sink.get()
            if frame is None:
                alive -= 1
                continue
            counter += 1
            current.append(VideoFrame(float(counter), b"", pixels=image))
            ids.append((camera_id, frame))
            if len(current) == args.batch:
                meta.append(ids)
                yield current
                current, ids = [], []
        if current:
            meta.append(ids)
            yield current

    shard_end = {(cid, s): e for cid, shards in plan.items() for s, e in shards}
    buffers: dict[tuple[str, int], dict] = {}
    done, started, last_report = 0, time.time(), time.time()
    people = 0
    for poses in detector.infer_batches(batches(), stop):
        ids = meta.popleft()
        if len(ids) != len(poses):
            raise RuntimeError("Detector changed the batch size")
        for (camera_id, frame), pose in zip(ids, poses):
            camera = dataset.cameras[camera_id]
            if pose.resolution != (camera.width, camera.height):
                raise RuntimeError(f"{camera_id} frame {frame}: pose resolution {pose.resolution}")
            start = frame // SHARD * SHARD
            rows = buffers.setdefault((camera_id, start), {})
            rows[frame] = (np.asarray(pose.boxes), np.asarray(pose.keypoints))
            people += len(pose.boxes)
            if len(rows) == shard_end[(camera_id, start)] - start:
                write_shard(out, camera_id, start, buffers.pop((camera_id, start)))
        done += len(ids)
        now = time.time()
        if now - last_report > 15 or done == total:
            last_report = now
            rate = done / (now - started)
            status = {"done": done, "total": total, "rate_fps": round(rate, 1), "people": people,
                      "eta_minutes": round((total - done) / max(rate, 1e-6) / 60, 1), "updated": now}
            print(json.dumps(status), flush=True)
            if args.progress:
                Path(args.progress).write_text(json.dumps(status), encoding="utf-8")
    if errors:
        raise SystemExit("; ".join(errors))
    if buffers:
        raise SystemExit(f"Incomplete shards left unwritten: {sorted(buffers)}")
    print("complete", flush=True)


if __name__ == "__main__":
    main()
