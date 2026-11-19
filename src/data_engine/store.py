"""Read access to the immutable raw detection store written by the detect stage.

A detection is identified by its *detection key* ``"<camera>/<native frame>/<index>"``,
where index is its row within that frame in the raw store. Raw detections are never
deleted or renumbered, so keys stay valid across every rerun of later stages; all human
evidence is recorded against them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import Dataset


def det_key(camera: str, frame: int, index: int) -> str:
    return f"{camera}/{int(frame)}/{int(index)}"


def parse_det_key(key: str) -> tuple[str, int, int]:
    camera, frame, index = key.rsplit("/", 2)
    return camera, int(frame), int(index)


@dataclass
class CameraDetections:
    camera: str
    frame_count: int
    start: np.ndarray  # (frame_count + 1,) row offsets: detections of frame f are rows start[f]:start[f+1]
    boxes: np.ndarray  # (M, 5) xyxy + person score, native pixels
    keypoints: np.ndarray  # (M, 17, 3) x, y, raw heatmap peak, native pixels
    covered: np.ndarray  # (frame_count,) bool: frame was processed by the detector

    def rows(self, frame: int) -> slice:
        return slice(int(self.start[frame]), int(self.start[frame + 1]))

    @property
    def frame_of_row(self) -> np.ndarray:
        if not hasattr(self, "_frame_of_row"):
            self._frame_of_row = np.repeat(np.arange(self.frame_count, dtype=np.int32), np.diff(self.start))
        return self._frame_of_row

    def row(self, frame: int, index: int) -> int:
        row = int(self.start[frame]) + int(index)
        if not self.start[frame] <= row < self.start[frame + 1]:
            raise KeyError(det_key(self.camera, frame, index))
        return row

    def key(self, row: int) -> str:
        frame = int(self.frame_of_row[row])
        return det_key(self.camera, frame, row - int(self.start[frame]))

    def row_of_key(self, key: str) -> int:
        camera, frame, index = parse_det_key(key)
        if camera != self.camera:
            raise KeyError(key)
        return self.row(frame, index)


def load_camera(root: Path, camera: str, frame_count: int) -> CameraDetections:
    directory = Path(root) / camera
    counts = np.zeros(frame_count, dtype=np.int64)
    covered = np.zeros(frame_count, dtype=bool)
    boxes, keypoints = [], []
    for path in sorted(directory.glob("[0-9]*.npz")):
        if ".tmp" in path.name:
            continue
        with np.load(path) as shard:
            frames = shard["frames"]
            if len(frames) and not np.array_equal(frames, np.arange(frames[0], frames[0] + len(frames))):
                raise ValueError(f"{path}: frames are not contiguous")
            counts[frames] = shard["counts"]
            covered[frames] = True
            boxes.append(shard["boxes"])
            keypoints.append(shard["keypoints"])
    start = np.concatenate(([0], np.cumsum(counts)))
    all_boxes = np.concatenate(boxes) if boxes else np.empty((0, 5), np.float32)
    all_keypoints = np.concatenate(keypoints) if keypoints else np.empty((0, 17, 3), np.float32)
    if len(all_boxes) != start[-1]:
        raise ValueError(f"{camera}: detection rows do not match per-frame counts")
    return CameraDetections(camera, frame_count, start, all_boxes, all_keypoints, covered)


class DetectionStore:
    def __init__(self, root: Path, dataset: Dataset):
        self.root, self.dataset = Path(root), dataset
        self._cameras: dict[str, CameraDetections] = {}

    def camera(self, camera: str) -> CameraDetections:
        if camera not in self._cameras:
            self._cameras[camera] = load_camera(self.root, camera, self.dataset.cameras[camera].frame_count)
        return self._cameras[camera]

    def lookup(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        camera, frame, index = parse_det_key(key)
        detections = self.camera(camera)
        row = detections.row(frame, index)
        return detections.boxes[row], detections.keypoints[row]

    def coverage(self) -> dict[str, float]:
        return {camera: float(self.camera(camera).covered.mean()) for camera in self.dataset.cameras}
