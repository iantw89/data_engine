"""Read-side helpers over the solved artifacts of a workspace (3D tracks projected back, summaries).

Used by the web renderer (dashed projections of a track's 3D pose into a camera) and by the
dashboards (one row per identity). Everything is read from the newest round that holds the
artifacts and cached per workspace and artifact modification time, so repeated panel renders
do not reload ``tracks3d.npz`` or the calibration.

Conventions (see ``stages/solve.py``): ``tracks3d.npz`` has one row per (component, sync index
``t``) with ``identity``, ``xyz`` (17x3 world inches, NaN = not measured, never interpolated),
``root``, and ``native`` (T, C) = native frame of camera C at sync index ``t`` (-1 none).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_LOCK = threading.Lock()
_CACHE: dict[tuple, "TrackData"] = {}
_MAX_CACHED = 4


@dataclass
class TrackData:
    cameras: list[str]
    identity: np.ndarray  # (N,)
    t: np.ndarray  # (N,)
    xyz: np.ndarray  # (N, 17, 3) float32, NaN = not measured
    root: np.ndarray  # (N, 3)
    native: np.ndarray  # (T, C)
    aligned_time: np.ndarray  # (T,)
    row_of: dict[tuple[int, int], int]  # (identity, t) -> row (the one with most measured joints)
    t_of_frame: dict[str, dict[int, int]]  # camera -> native frame -> first sync index showing it
    cams: dict[str, Any] = field(default_factory=dict)  # camera id -> geometry.Cam (lazy)
    cams_loaded: bool = False


def _artifact(workspace, name: str) -> Path | None:
    return workspace.latest_round_file(name)


def _signature(workspace) -> tuple | None:
    path = _artifact(workspace, "tracks3d.npz")
    if path is None:
        return None
    stat = path.stat()
    calibration = Path(workspace.path("calibration", "calibration.json"))
    calibration_mtime = calibration.stat().st_mtime_ns if calibration.is_file() else 0
    return (str(workspace.root), str(path), stat.st_mtime_ns, stat.st_size, calibration_mtime)


def _load(workspace) -> TrackData | None:
    signature = _signature(workspace)
    if signature is None:
        return None
    with _LOCK:
        cached = _CACHE.get(signature)
        if cached is not None:
            return cached
    with np.load(signature[1]) as data:
        cameras = [str(c) for c in data["cameras"]]
        identity, t = data["identity"].astype(np.int64), data["t"].astype(np.int64)
        xyz = data["xyz"]
        root = data["root"]
        native = data["native"].astype(np.int64)
        aligned_time = data["aligned_time"]
    measured = np.isfinite(xyz).all(axis=2).sum(axis=1) if len(xyz) else np.zeros(0, int)
    row_of: dict[tuple[int, int], int] = {}
    for row in range(len(identity)):
        key = (int(identity[row]), int(t[row]))
        best = row_of.get(key)
        if best is None or measured[row] > measured[best]:
            row_of[key] = row
    t_of_frame = {}
    for index, camera in enumerate(cameras):
        column = native[:, index]
        shown = np.flatnonzero(column >= 0)
        # dict() keeps the last assignment: iterate backwards so the first sync index showing a frame wins
        t_of_frame[camera] = dict(zip(column[shown][::-1].tolist(), shown[::-1].tolist()))
    entry = TrackData(cameras, identity, t, xyz, root, native, aligned_time, row_of, t_of_frame)
    with _LOCK:
        if len(_CACHE) >= _MAX_CACHED:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[signature] = entry
    return entry


def _cams(workspace, data: TrackData) -> dict[str, Any]:
    """Calibrated cameras by engine camera id (loaded once per cache entry)."""
    if data.cams_loaded:
        return data.cams
    from .geometry import load_calibration

    path = Path(workspace.path("calibration", "calibration.json"))
    if not path.is_file():
        path = Path(workspace.site["calibration"])
    dataset_cameras = {}
    try:
        dataset_cameras = workspace.dataset.cameras
    except Exception:  # no dataset.json: fall back to calibration ids equal to camera ids, default width
        dataset_cameras = {}
    widths = {cam.calibration_id: float(cam.width) for cam in dataset_cameras.values() if cam.calibration_id is not None}
    by_calibration = load_calibration(path, widths)
    cams = {}
    for camera in data.cameras:
        info = dataset_cameras.get(camera)
        calibration_id = info.calibration_id if info is not None else camera
        if calibration_id in by_calibration:
            cams[camera] = by_calibration[calibration_id]
    data.cams, data.cams_loaded = cams, True
    return cams


def projected_pose(workspace, track_id: int, camera: str, frame: int) -> np.ndarray | None:
    """Identity ``track_id``'s measured 3D joints projected into ``camera`` at native ``frame``: (17, 2) pixels.

    The sync index is the first one at which ``camera`` shows native ``frame``; joints that were not
    measured in 3D (or are behind the camera / outside the invertible lens model) are NaN. ``None``
    when the track has no 3D pose at that frame, the frame is not on the common timeline, or the
    camera is unknown or uncalibrated.
    """
    data = _load(workspace)
    if data is None:
        return None
    t = data.t_of_frame.get(camera, {}).get(int(frame))
    if t is None:
        return None
    row = data.row_of.get((int(track_id), int(t)))
    if row is None:
        return None
    cam = _cams(workspace, data).get(camera)
    if cam is None:
        return None
    world = data.xyz[row].astype(np.float64)
    if not np.isfinite(world).any():
        return None
    pose = cam.project(world)
    return pose if np.isfinite(pose).any() else None


def track_summary(workspace) -> list[dict]:
    """One row per identity of the newest solved round, for dashboards.

    Keys: ``identity``; ``first_t``/``last_t`` (sync indices) and ``first_time_s``/``last_time_s`` (seconds
    on the engine's common timeline, ``aligned_time``); ``cameras`` (those with detections of the identity);
    ``detections`` and ``detections_by_camera`` (raw detection rows assigned to it); ``frames`` (sync
    indices of its 3D track), ``frames_with_3d`` (measured root) and ``frames_with_keypoints`` (at least
    one measured joint). Empty when the workspace has no solved round.
    """
    data = _load(workspace)
    solution = _artifact(workspace, "solution.npz")
    if data is None or solution is None:
        return []
    detections: dict[int, dict[str, int]] = {}
    with np.load(solution) as arrays:
        for camera in [str(c) for c in arrays["cameras"]]:
            ids = arrays[f"identity_{camera}"]
            ids = ids[ids >= 0]
            if not len(ids):
                continue
            values, counts = np.unique(ids, return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist()):
                detections.setdefault(int(value), {})[camera] = int(count)
    rows = []
    keypoint_frames = np.isfinite(data.xyz).any(axis=(1, 2)) if len(data.xyz) else np.zeros(0, bool)
    root_frames = np.isfinite(data.root).all(axis=1) if len(data.root) else np.zeros(0, bool)
    for identity in sorted(set(np.unique(data.identity).tolist()) | set(detections)):
        mask = data.identity == identity
        ts = np.unique(data.t[mask])
        by_camera = detections.get(int(identity), {})
        row = {"identity": int(identity), "cameras": sorted(by_camera), "detections": int(sum(by_camera.values())),
               "detections_by_camera": dict(sorted(by_camera.items())), "frames": int(len(ts)),
               "frames_with_3d": int(len(np.unique(data.t[mask & root_frames]))),
               "frames_with_keypoints": int(len(np.unique(data.t[mask & keypoint_frames])))}
        if len(ts):
            row.update(first_t=int(ts[0]), last_t=int(ts[-1]), first_time_s=round(float(data.aligned_time[ts[0]]), 3),
                       last_time_s=round(float(data.aligned_time[ts[-1]]), 3))
        else:
            row.update(first_t=None, last_t=None, first_time_s=None, last_time_s=None)
        rows.append(row)
    return rows


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
