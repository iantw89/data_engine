"""Identity montage: one row per person, one column per camera, at one instant.

The quickest way to see whether cross-camera association is right: every crop in a row
must show the same person.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..geometry import SKELETON
from .debug_view import colour
from .scene import Scene


def crop(scene: Scene, camera_index: int, row: int, *, height: int = 240, pad: float = 0.2,
         tint=(0, 255, 0), joint_peak: float = 0.3) -> np.ndarray:
    camera = scene.cameras[camera_index]
    frame = int(camera.detections.frame_of_row[row])
    image = scene.workspace.frames.read(camera.camera, frame)
    box = camera.detections.boxes[row, :4]
    w, h = box[2] - box[0], box[3] - box[1]
    x0, y0 = int(max(0, box[0] - pad * w)), int(max(0, box[1] - pad * h))
    x1, y1 = int(min(image.shape[1], box[2] + pad * w)), int(min(image.shape[0], box[3] + pad * h))
    patch = image[y0:y1, x0:x1].copy()
    points = camera.detections.keypoints[row]
    for i, j in SKELETON:
        if points[i, 2] >= joint_peak and points[j, 2] >= joint_peak:
            a = (int(points[i, 0] - x0), int(points[i, 1] - y0))
            b = (int(points[j, 0] - x0), int(points[j, 1] - y0))
            cv2.line(patch, a, b, tint, 2)
    scale = height / max(1, patch.shape[0])
    width = max(40, min(int(patch.shape[1] * scale), int(height * 1.4)))
    return cv2.resize(patch, (width, height), interpolation=cv2.INTER_AREA)


def identity_montage(scene: Scene, row_identity: list[np.ndarray], t: int, *, identities: list[int] | None = None,
                     height: int = 240, cell: int = 300) -> np.ndarray:
    cameras = scene.cameras
    present: dict[int, dict[int, int]] = {}
    for camera in cameras:
        for row in camera.rows_at(t):
            identity = int(row_identity[camera.index][row])
            if identity >= 0:
                present.setdefault(identity, {})[camera.index] = int(row)
    order = identities if identities is not None else sorted(present)
    rows = []
    header = np.zeros((40, 90 + cell * len(cameras), 3), np.uint8)
    for camera in cameras:
        cv2.putText(header, camera.camera, (100 + cell * camera.index, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
    rows.append(header)
    for identity in order:
        if identity not in present:
            continue
        strip = np.zeros((height, 90 + cell * len(cameras), 3), np.uint8)
        cv2.putText(strip, str(identity), (8, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour(identity), 3)
        for camera_index, row in present[identity].items():
            patch = crop(scene, camera_index, row, height=height, tint=colour(identity))
            x = 90 + cell * camera_index
            strip[:, x:x + min(cell, patch.shape[1])] = patch[:, :cell]
        rows.append(strip)
    return np.vstack(rows)
