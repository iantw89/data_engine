"""Contact sheets for eyeballing a solution: every camera at one instant, people coloured by label."""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

from ..geometry import SKELETON
from .scene import Scene


def colour(label: int) -> tuple[int, int, int]:
    if label < 0:
        return (160, 160, 160)
    hue = (label * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_camera(scene: Scene, camera_index: int, t: int, labels: np.ndarray, *, width: int = 960,
                projections: dict[int, np.ndarray] | None = None, joint_peak: float = 0.3) -> np.ndarray:
    camera = scene.cameras[camera_index]
    frame = int(scene.native[t, camera_index])
    if frame < 0:
        return np.zeros((width * 3 // 4, width, 3), np.uint8)
    image = scene.workspace.frames.read(camera.camera, frame).copy()
    for row in camera.rows_at(t):
        label = int(labels[row])
        tint = colour(label)
        box = camera.detections.boxes[row]
        points = camera.detections.keypoints[row]
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), tint, 3 if label >= 0 else 1)
        for i, j in SKELETON:
            if points[i, 2] >= joint_peak and points[j, 2] >= joint_peak:
                cv2.line(image, tuple(points[i, :2].astype(int)), tuple(points[j, :2].astype(int)), tint, 3)
        text = str(label) if label >= 0 else "-"
        cv2.putText(image, text, (int(box[0]), max(30, int(box[1]) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 7)
        cv2.putText(image, text, (int(box[0]), max(30, int(box[1]) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, tint, 3)
    for label, points in (projections or {}).items():
        tint = colour(label)
        for i, j in SKELETON:
            if np.isfinite(points[[i, j]]).all():
                a, b = points[i].astype(int), points[j].astype(int)
                cv2.line(image, tuple(a), tuple(b), (255, 255, 255), 5)
                cv2.line(image, tuple(a), tuple(b), tint, 2)
    cv2.putText(image, f"{camera.camera} f{frame}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 8)
    cv2.putText(image, f"{camera.camera} f{frame}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
    return cv2.resize(image, (width, width * 3 // 4), interpolation=cv2.INTER_AREA)


def contact_sheet(scene: Scene, t: int, labels: list[np.ndarray], *, columns: int = 4, width: int = 960,
                  projections: list[dict[int, np.ndarray]] | None = None) -> np.ndarray:
    tiles = [draw_camera(scene, c.index, t, labels[c.index], width=width,
                         projections=projections[c.index] if projections else None) for c in scene.cameras]
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    return np.vstack(rows)
