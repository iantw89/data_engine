"""Everything the solver needs in one in-memory structure.

Rows are raw detection rows per camera (immutable order of the detection store). Time is
the *sync index* ``t`` of the alignment's 10 Hz common timeline; ``native[t, c]`` is the
native frame of camera ``c`` shown at ``t`` (-1 when the camera has no frame or no
resolved offset there - such detections simply carry no cross-camera evidence).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..geometry import BODY, Cam, load_calibration
from ..store import CameraDetections
from ..workspace import Workspace


@dataclass
class CameraScene:
    index: int
    camera: str
    cam: Cam | None
    detections: CameraDetections
    kept: np.ndarray  # (M,) bool
    tracklet: np.ndarray  # (M,) int32 global tracklet id or -1
    t: np.ndarray  # (M,) int32 sync index of each row, -1 if its frame is not on the timeline
    row_start: np.ndarray  # (T+1,) rows of sync index t are order[row_start[t]:row_start[t+1]]
    order: np.ndarray  # kept rows with t >= 0, sorted by t
    rays: np.ndarray = field(default=None, repr=False)  # (M,17,3) float32 unit world rays (NaN if unusable)
    joint_ok: np.ndarray = field(default=None, repr=False)  # (M,17) bool: peak >= gate and ray valid

    @property
    def height_cvat(self) -> np.ndarray:
        boxes = self.detections.boxes
        return (boxes[:, 3] - boxes[:, 1]) * (self.cam.to_cvat if self.cam else 0.625)

    def rows_at(self, t: int) -> np.ndarray:
        return self.order[self.row_start[t]:self.row_start[t + 1]]


@dataclass
class Scene:
    workspace: Workspace
    cameras: list[CameraScene]
    native: np.ndarray  # (T, C) int32
    aligned_time: np.ndarray  # (T,) seconds
    joint_peak: float

    @property
    def T(self) -> int:
        return len(self.native)

    def camera_index(self, camera: str) -> int:
        return next(c.index for c in self.cameras if c.camera == camera)


def load_sync(workspace: Workspace) -> tuple[list[str], np.ndarray, np.ndarray]:
    with np.load(workspace.path("alignment", "sync_table.npz")) as table:
        return [str(c) for c in table["cameras"]], table["native"].astype(np.int32), table["aligned_time"]


def load_scene(workspace: Workspace, *, joint_peak: float, calibration: Path | None = None,
               rays: bool = True) -> Scene:
    dataset = workspace.dataset
    camera_names, native, aligned_time = load_sync(workspace)
    calibration_path = calibration or workspace.path("calibration", "calibration.json")
    if not Path(calibration_path).is_file():
        calibration_path = Path(workspace.site["calibration"])
    widths = {dataset.cameras[c].calibration_id: float(dataset.cameras[c].width) for c in camera_names
              if dataset.cameras[c].calibration_id is not None}
    cams = load_calibration(Path(calibration_path), widths)
    scenes = []
    for index, camera in enumerate(camera_names):
        detections = workspace.detections.camera(camera)
        with np.load(workspace.path("tracklets", f"{camera}.npz")) as data:
            kept, tracklet = data["kept"], data["tracklet"]
        # Sync index per native frame (first index showing the frame), then per row.
        frames = native[:, index]
        sync_of_frame = np.full(detections.frame_count, -1, dtype=np.int32)
        shown = np.flatnonzero(frames >= 0)
        sync_of_frame[frames[shown][::-1]] = shown[::-1].astype(np.int32)
        t = sync_of_frame[detections.frame_of_row] if len(detections.boxes) else np.empty(0, np.int32)
        usable = np.flatnonzero(kept & (t >= 0))
        order = usable[np.argsort(t[usable], kind="stable")]
        row_start = np.searchsorted(t[order], np.arange(len(native) + 1)).astype(np.int64)
        cam = cams.get(dataset.cameras[camera].calibration_id)
        scene = CameraScene(index, camera, cam, detections, kept, tracklet, t, row_start, order)
        if rays and cam is not None:
            scene.rays = np.full((len(kept), 17, 3), np.nan, dtype=np.float32)
            scene.joint_ok = np.zeros((len(kept), 17), dtype=bool)
            if len(order):
                points = detections.keypoints[order]
                directions = cam.rays(points[:, :, :2])
                valid = np.isfinite(directions).all(axis=-1) & (points[:, :, 2] >= joint_peak)
                scene.rays[order] = directions.astype(np.float32)
                scene.joint_ok[order] = valid
        scenes.append(scene)
    return Scene(workspace, scenes, native, aligned_time, joint_peak)


def tracklet_table(scene: Scene) -> dict[int, dict]:
    """tracklet id -> camera index, rows (sorted by time), sync indices."""
    table: dict[int, dict] = {}
    for camera in scene.cameras:
        rows = camera.order[camera.tracklet[camera.order] >= 0]
        ids = camera.tracklet[rows]
        grouping = np.argsort(ids, kind="stable")
        rows, ids = rows[grouping], ids[grouping]
        boundaries = np.flatnonzero(np.diff(ids)) + 1
        for chunk in np.split(np.arange(len(ids)), boundaries):
            if len(chunk):
                table[int(ids[chunk[0]])] = {"camera": camera.index, "rows": rows[chunk], "t": camera.t[rows[chunk]]}
    return table


BODY_JOINTS = BODY
