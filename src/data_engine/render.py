"""Render evidence panels (JPEG) for the review web UI.

A panel is one image: the camera frame (or a crop around a person's box), the measured
keypoints of the highlighted detections as a thin COCO-17 skeleton, optionally the other
detections of that frame as faint grey boxes, and optionally dashed projections of 3D
tracks. Boxes alone are ambiguous and heavy overlays hide the person, so the skeleton is
drawn with thin lines and small dots, and only for joints whose raw peak is >= 0.3.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Any

import cv2
import numpy as np

JOINT_PEAK = 0.3
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
COLORS_BGR = {  # bright, readable on gym footage
    "green": (90, 230, 90), "magenta": (230, 60, 230), "cyan": (240, 230, 40),
    "yellow": (40, 225, 250), "orange": (30, 150, 255),
}
GREY = (150, 150, 150)
PADDING = 0.25
MAX_UPSCALE = 4.0
CACHE_ITEMS = 256

_cache: "OrderedDict[str, tuple[bytes, dict]]" = OrderedDict()
_cache_lock = threading.Lock()
_inflight: dict[str, threading.Lock] = {}


# ----------------------------------------------------------------------------- geometry
def crop_window(box, width: int, height: int, *, pad: float = PADDING, min_side: float = 96.0,
                aspect: tuple[float, float] = (0.6, 1.7)) -> tuple[int, int, int, int]:
    """Window (x0, y0, x1, y1) around ``box`` with ``pad`` margin, aspect kept sane, inside the frame."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = bw * (1 + 2 * pad), bh * (1 + 2 * pad)
    w, h = max(w, min_side), max(h, min_side)
    if w / h < aspect[0]:
        w = h * aspect[0]
    elif w / h > aspect[1]:
        h = w / aspect[1]
    w, h = min(w, float(width)), min(h, float(height))
    left = min(max(cx - w / 2, 0.0), width - w)
    top = min(max(cy - h / 2, 0.0), height - h)
    return int(round(left)), int(round(top)), int(round(left + w)), int(round(top + h))


# ----------------------------------------------------------------------------- drawing helpers
def _ipt(point) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def _dashed_line(image, p, q, color, thickness, dash=9, gap=6):
    p, q = np.asarray(p, float), np.asarray(q, float)
    length = float(np.linalg.norm(q - p))
    if length < 1:
        return
    direction = (q - p) / length
    position = 0.0
    while position < length:
        end = min(position + dash, length)
        cv2.line(image, _ipt(p + direction * position), _ipt(p + direction * end), color, thickness, cv2.LINE_AA)
        position += dash + gap


def draw_skeleton(image, points: np.ndarray, color, *, min_conf: float = JOINT_PEAK, thickness: int = 1,
                  radius: int = 2, dashed: bool = False) -> int:
    """Draw a COCO-17 skeleton; ``points`` is (17, 2|3). Joints below ``min_conf`` (col 3) are skipped.

    Returns the number of joints drawn.
    """
    points = np.asarray(points, float)
    confident = np.isfinite(points[:, :2]).all(axis=1)
    if points.shape[1] >= 3:
        confident &= points[:, 2] >= min_conf
    halo = (20, 20, 20)
    for a, b in SKELETON:
        if confident[a] and confident[b]:
            if dashed:
                _dashed_line(image, points[a, :2], points[b, :2], halo, thickness + 2)
                _dashed_line(image, points[a, :2], points[b, :2], color, thickness)
            else:
                cv2.line(image, _ipt(points[a]), _ipt(points[b]), halo, thickness + 2, cv2.LINE_AA)
                cv2.line(image, _ipt(points[a]), _ipt(points[b]), color, thickness, cv2.LINE_AA)
    for index in np.flatnonzero(confident):
        centre = _ipt(points[index])
        if dashed:
            cv2.circle(image, centre, radius + 1, halo, 1, cv2.LINE_AA)
            cv2.circle(image, centre, radius, color, 1, cv2.LINE_AA)
        else:
            cv2.circle(image, centre, radius + 1, halo, -1, cv2.LINE_AA)
            cv2.circle(image, centre, radius, color, -1, cv2.LINE_AA)
    return int(confident.sum())


def _tag(image, text: str, origin: tuple[int, int], color, scale: float, *, anchor_bottom: bool = False,
         filled: bool = True) -> None:
    font, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1 if scale < 0.8 else 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    pad = max(3, int(scale * 6))
    x, y = origin
    box_h = th + base + 2 * pad - 2
    top = y - box_h if anchor_bottom else y
    cv2.rectangle(image, (x, top), (x + tw + 2 * pad, top + box_h), color if filled else (25, 25, 25), -1)
    ink = (20, 20, 20) if filled else (235, 235, 235)
    cv2.putText(image, text, (x + pad, top + th + pad - 1), font, scale, ink, thickness, cv2.LINE_AA)


def error_image(message: str, size: tuple[int, int] = (480, 360)) -> bytes:
    """Neutral placeholder so a failed panel never shows as a broken image."""
    image = np.full((size[1], size[0], 3), 38, np.uint8)
    cv2.putText(image, "panel unavailable", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 160, 255), 2, cv2.LINE_AA)
    line, y = "", 66
    for word in message.split():
        if len(line) + len(word) > 44:
            cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            line, y = "", y + 22
        line = f"{line} {word}".strip()
    cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", image)[1].tobytes()


# ----------------------------------------------------------------------------- rendering
def _project_pose(workspace, track_id: int, camera: str, frame: int):
    try:
        from data_engine.tracks import projected_pose  # type: ignore
    except ImportError:
        return None
    try:
        pose = projected_pose(workspace, track_id, camera, frame)
    except Exception:
        return None
    return None if pose is None else np.asarray(pose, float)


def _rect(canvas, corners, color, thickness) -> None:
    cv2.rectangle(canvas, _ipt(corners[0]), _ipt(corners[1]), color, thickness, cv2.LINE_AA)


def _skeleton_geometry(points: np.ndarray, out_w: int, out_h: int, *, dashed: bool,
                       min_conf: float = JOINT_PEAK) -> dict | None:
    """A skeleton for the web overlay: joints as [x, y] fractions of the image, ``None`` where not drawn."""
    points = np.asarray(points, float)
    ok = np.isfinite(points[:, :2]).all(axis=1)
    if points.shape[1] >= 3:
        ok &= points[:, 2] >= min_conf
    if not ok.any():
        return None
    joints = [[round(float(p[0]) / out_w, 5), round(float(p[1]) / out_h, 5)] if good else None
              for p, good in zip(points, ok)]
    return {"joints": joints, "dashed": bool(dashed)}


def _render(workspace, panel: dict, max_side: int, full: bool, skeletons: bool = True) -> tuple[bytes, dict]:
    """JPEG bytes plus the skeleton overlay geometry; ``skeletons=False`` leaves them off the JPEG."""
    camera, frame = panel["camera"], int(panel["frame"])
    color = COLORS_BGR.get(panel.get("color", "green"), COLORS_BGR["green"])
    image = np.asarray(workspace.frames.read(camera, frame))
    height, width = image.shape[:2]
    box = panel.get("box")
    whole = full or box is None
    x0, y0, x1, y1 = (0, 0, width, height) if whole else crop_window(box, width, height)
    crop = image[y0:y1, x0:x1]
    scale = min(max_side / max(crop.shape[:2]), MAX_UPSCALE)
    out_w, out_h = max(2, round(crop.shape[1] * scale)), max(2, round(crop.shape[0] * scale))
    canvas = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    side = max(out_w, out_h)
    thickness, radius = max(1, round(side / 520)), max(2, round(side / 230))

    def to_out(points):
        return (np.asarray(points, float) - [x0, y0]) * scale

    blue, green, red = color
    overlay = {"size": [out_w, out_h], "color": f"#{red:02x}{green:02x}{blue:02x}", "thickness": thickness,
               "radius": radius, "bones": SKELETON, "skeletons": []}

    def skeleton(points, dashed=False):
        geometry = _skeleton_geometry(points, out_w, out_h, dashed=dashed)
        if geometry is not None:
            overlay["skeletons"].append(geometry)
        if skeletons:
            draw_skeleton(canvas, points, color, thickness=thickness, radius=radius, dashed=dashed)

    highlight = list(panel.get("highlight") or [])
    own_rows: set[int] = set()
    for key in highlight:  # measured keypoints of the highlighted detections
        try:
            det_box, keypoints = workspace.detections.lookup(key)
            cam, f, index = key.rsplit("/", 2)
        except (KeyError, FileNotFoundError, ValueError, IndexError):
            continue  # missing detection: the image alone is still useful
        if cam == camera and int(f) == frame:
            own_rows.add(int(index))
        if whole:  # a thin own-colour outline locates the person in the full frame
            _rect(canvas, to_out([[det_box[0], det_box[1]], [det_box[2], det_box[3]]]), color, 1)
        skeleton(np.concatenate([to_out(keypoints[:, :2]), keypoints[:, 2:3]], axis=1))

    if panel.get("others", True):
        try:
            cam_store = workspace.detections.camera(camera)
            rows = cam_store.rows(frame)
            for offset, det_box in enumerate(cam_store.boxes[rows]):
                if offset in own_rows:
                    continue
                corners = to_out([[det_box[0], det_box[1]], [det_box[2], det_box[3]]])
                if corners[1][0] < 0 or corners[1][1] < 0 or corners[0][0] > out_w or corners[0][1] > out_h:
                    continue
                _rect(canvas, corners, GREY, 1)
        except (KeyError, FileNotFoundError, ValueError, IndexError):
            pass

    if whole and box is not None:
        _rect(canvas, to_out([[box[0], box[1]], [box[2], box[3]]]), color, max(2, thickness + 1))

    for track_id in panel.get("project") or []:
        pose = _project_pose(workspace, int(track_id), camera, frame)
        if pose is not None and pose.ndim == 2 and pose.shape[0] == 17:
            skeleton(to_out(pose[:, :2]), dashed=True)

    font_scale = 0.45 + side / 1800
    label = str(panel.get("label") or "")
    if label:
        _tag(canvas, label, (0, 0), color, font_scale + 0.1)
    _tag(canvas, f"{camera}  f{frame}", (0, out_h), color, font_scale * 0.8, anchor_bottom=True, filled=False)
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 86])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes(), overlay


def render_panel(workspace, panel: dict[str, Any], *, max_side: int = 720, full: bool = False,
                 skeletons: bool = True) -> bytes:
    """JPEG bytes for ``panel``; identical requests are served from an in-memory LRU (thread safe).

    The web UI asks for ``skeletons=False`` and draws them itself from :func:`panel_overlay`, so it can
    hide them while the reviewer looks at the person.
    """
    return _render_cached(workspace, panel, max_side, full, skeletons)[0]


def panel_overlay(workspace, panel: dict[str, Any], *, max_side: int = 720, full: bool = False) -> dict:
    """Skeletons of ``panel`` as JSON-able geometry: joints in image fractions plus the stroke style
    (``thickness`` / ``radius`` in pixels of an image ``size`` wide) that :func:`draw_skeleton` uses."""
    return _render_cached(workspace, panel, max_side, full, False)[1]


def _render_cached(workspace, panel: dict, max_side: int, full: bool, skeletons: bool) -> tuple[bytes, dict]:
    max_side = int(max(64, min(max_side, 4096)))
    key = json.dumps([panel, max_side, bool(full), bool(skeletons), getattr(workspace, "dataset_id", "")],
                     sort_keys=True, default=str)
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
        gate = _inflight.setdefault(key, threading.Lock())
    with gate:  # concurrent identical requests render once
        with _cache_lock:
            hit = _cache.get(key)
        if hit is None:
            try:
                hit = _render(workspace, panel, max_side, bool(full), bool(skeletons))
            finally:
                with _cache_lock:
                    _inflight.pop(key, None)
            with _cache_lock:
                _cache[key] = hit
                while len(_cache) > CACHE_ITEMS:
                    _cache.popitem(last=False)
    return hit


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
