"""Read poses from given boxes with the production ViTPose path. Runs in data_management's interpreter.

Input ``--requests``: npz with ``cameras`` (names), ``camera`` (J,) index into ``cameras``,
``frame`` (J,) native frame, ``box`` (J, 4) native pixels (may extend beyond the image).
Every request is read twice, as usual for a top-down pose model: on the given box, then on the box
derived from the first result's confident keypoints (``--margins``: left,right,top,bottom as fractions
of the keypoint-box height - the same margins the engine uses to turn a skeleton into a box).

Output ``--out``: npz with ``keypoints_first`` and ``keypoints`` (J, 17, 3: x, y native pixels clipped to
the image, heatmap peak - exactly as the detector stage stores them) and ``box`` (J, 4) the second box.
Requests whose frame could not be read keep NaN.

Frames are decoded once, in order, one thread per camera (random access costs about 0.75 s a frame).
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_engine.dataset import Dataset  # noqa: E402

BATCH = 8


def second_box(keypoints: np.ndarray, box: np.ndarray, margins: tuple[float, float, float, float],
               peak: float = 0.3) -> np.ndarray:
    """Box around the confident keypoints of the first read; the given box if fewer than 8 are confident."""
    confident = keypoints[:, 2] >= peak
    if confident.sum() < 8:
        return box
    x1, y1 = keypoints[confident, :2].min(axis=0)
    x2, y2 = keypoints[confident, :2].max(axis=0)
    left, right, top, bottom = margins
    height = y2 - y1
    return np.array([x1 - left * height, y1 - top * height, x2 + right * height, y2 + bottom * height], np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--vitpose", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--margins", default="0.10,0.10,0.13,0.14")
    args = parser.parse_args()

    from data_management.frame_index import ExactFrameReader
    from data_management.live_vitpose import GpuViTPose, decode_peaks

    dataset = Dataset.load(Path(args.dataset_json))
    margins = tuple(float(v) for v in args.margins.split(","))
    with np.load(args.requests) as data:
        names = [str(name) for name in data["cameras"]]
        camera, frame, box = data["camera"].astype(np.int64), data["frame"].astype(np.int64), data["box"].astype(np.float32)
    count = len(camera)
    first = np.full((count, 17, 3), np.nan, np.float32)
    second = np.full((count, 17, 3), np.nan, np.float32)
    boxes = np.full((count, 4), np.nan, np.float32)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for job in range(count):
        groups[(int(camera[job]), int(frame[job]))].append(job)
    frames_of: dict[int, list[int]] = defaultdict(list)
    for index, native in groups:
        frames_of[index].append(native)
    print(f"{count} boxes on {len(groups)} frames of {len(frames_of)} cameras", flush=True)
    if not count:
        np.savez_compressed(args.out, keypoints_first=first, keypoints=second, box=boxes)
        return 0

    model = GpuViTPose(Path(args.vitpose))
    sink: queue.Queue = queue.Queue(maxsize=3 * BATCH)
    stop = threading.Event()
    errors: list[str] = []

    def decode(index: int) -> None:
        described = dataset.cameras[names[index]]
        reader = None
        try:
            for native in sorted(frames_of[index]):
                if stop.is_set():
                    return
                path, local = described.locate(native)
                if reader is None or reader.path != path:
                    if reader is not None:
                        reader.close()
                    reader = ExactFrameReader(path)
                sink.put((index, native, reader.read(local)))
        except Exception as exc:  # surfaced by the main thread; never swallowed
            errors.append(f"{names[index]}: {type(exc).__name__}: {exc}")
            stop.set()
        finally:
            if reader is not None:
                reader.close()
            sink.put((index, None, None))

    def read(sources, rows: list[np.ndarray], shapes: list[tuple[int, int]]) -> list[np.ndarray]:
        peaks = model.peaks_from_sources(sources, rows)
        decoded = decode_peaks(peaks, np.concatenate(rows))
        out, offset = [], 0
        for part, (width, height) in zip(rows, shapes):
            poses = decoded[offset:offset + len(part)].copy()
            poses[:, :, 0] = poses[:, :, 0].clip(0, width)
            poses[:, :, 1] = poses[:, :, 1].clip(0, height)
            poses[:, :, 2] = poses[:, :, 2].clip(0, 1)
            out.append(poses)
            offset += len(part)
        return out

    def flush(batch: list) -> None:
        images = [item[2] for item in batch]
        jobs = [groups[(item[0], item[1])] for item in batch]
        rows = [box[job_list].reshape(-1, 4) for job_list in jobs]
        shapes = [(image.shape[1], image.shape[0]) for image in images]
        sources = model.upload(images)
        once = read(sources, rows, shapes)
        again_boxes = [np.stack([second_box(poses[j], part[j], margins) for j in range(len(part))]).astype(np.float32)
                       for part, poses in zip(rows, once)]
        twice = read(sources, again_boxes, shapes)
        for job_list, a, b, new_boxes in zip(jobs, once, twice, again_boxes):
            first[job_list], second[job_list], boxes[job_list] = a, b, new_boxes

    threads = [threading.Thread(target=decode, args=(index,), daemon=True) for index in frames_of]
    for thread in threads:
        thread.start()
    alive, done, started, batch = len(threads), 0, time.time(), []
    while alive:
        index, native, image = sink.get()
        if native is None:
            alive -= 1
            continue
        batch.append((index, native, image))
        if len(batch) == BATCH:
            flush(batch)
            done += len(batch)
            batch = []
            if done % (BATCH * 250) == 0:
                print(f"{done}/{len(groups)} frames, {time.time() - started:.0f} s", flush=True)
    if batch:
        flush(batch)
    if errors:
        print("decode errors: " + "; ".join(errors), flush=True)
        return 2
    np.savez_compressed(args.out, keypoints_first=first, keypoints=second, box=boxes)
    print(f"read {count} boxes on {len(groups)} frames in {time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
