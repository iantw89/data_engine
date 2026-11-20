"""3D poses per (component, sync frame) by gated multi-view triangulation.

Policy is the golden identity_triangulation contract: a joint is measured only from views
whose heatmap peak passes the gate, with at least two views, a ray angle of at least 8
degrees and every used view within 12 px (1280x960 scale); the least consistent view is
dropped while more than two remain. Nothing is interpolated or smoothed: a joint that
cannot be measured is NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import triangulate_views
from .scene import Scene

HIPS, SHOULDERS = (11, 12), (5, 6)


@dataclass
class PersonFrames:
    """One row per (label, sync index) that has at least one member detection."""
    label: np.ndarray  # (N,) int32
    t: np.ndarray  # (N,) int32
    rows: np.ndarray  # (N, C) int32 raw row per camera, -1 if that camera has no member
    xyz: np.ndarray  # (N, 17, 3) float32, NaN where unmeasured
    used: np.ndarray  # (N, 17) uint8 bitmask of cameras used per joint
    residual: np.ndarray  # (N, 17) float32 worst used-view reprojection residual (1280-scale px)
    root: np.ndarray  # (N, 3) float32 hip centre, else mean of >= 2 torso joints, else NaN

    def __len__(self) -> int:
        return len(self.t)

    def index(self) -> dict[tuple[int, int], int]:
        return {(int(label), int(t)): i for i, (label, t) in enumerate(zip(self.label, self.t))}


def member_table(scene: Scene, row_label: list[np.ndarray]):
    """(label, t) -> row per camera. Returns label, t, rows (N, C) and duplicates per camera.

    If one camera contributes two detections to a label at one instant (only possible after
    a human 'duplicate' answer or projection recovery), the higher person score is used and
    the pair is reported so it can be reviewed as a pose-ownership question.
    """
    labels, ts, cams, rows = [], [], [], []
    for camera in scene.cameras:
        candidate = camera.order[row_label[camera.index][camera.order] >= 0]
        labels.append(row_label[camera.index][candidate])
        ts.append(camera.t[candidate])
        cams.append(np.full(len(candidate), camera.index))
        rows.append(candidate)
    labels, ts, cams, rows = map(np.concatenate, (labels, ts, cams, rows))
    if not len(labels):
        count = len(scene.cameras)
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty((0, count), np.int32), []
    key = labels.astype(np.int64) * (scene.T + 1) + ts
    unique, inverse = np.unique(key, return_inverse=True)
    table = np.full((len(unique), len(scene.cameras)), -1, dtype=np.int32)
    # Highest person score wins a (label, t, camera) slot; other claimants are reported.
    score = np.empty(len(rows), dtype=np.float32)
    for camera in scene.cameras:
        mask = cams == camera.index
        score[mask] = camera.detections.boxes[rows[mask], 4]
        if camera.demoted is not None:  # a human preference between two poses of one person beats the score
            score[mask] -= camera.demoted[rows[mask]].astype(np.float32)
    slot = inverse.astype(np.int64) * len(scene.cameras) + cams
    order = np.lexsort((-score, slot))
    first = np.concatenate(([True], slot[order][1:] != slot[order][:-1]))
    winners = order[first]
    table[inverse[winners], cams[winners]] = rows[winners]
    duplicates = [(int(cams[i]), int(table[inverse[i], cams[i]]), int(rows[i])) for i in order[~first]]
    return (unique // (scene.T + 1)).astype(np.int32), (unique % (scene.T + 1)).astype(np.int32), table, duplicates


def triangulate_labels(scene: Scene, row_label: list[np.ndarray], recipe: dict, *, chunk: int = 40_000):
    geometry = recipe["geometry"]
    label, t, table, duplicates = member_table(scene, row_label)
    count, views = len(label), len(scene.cameras)
    xyz = np.full((count, 17, 3), np.nan, dtype=np.float32)
    used = np.zeros((count, 17), dtype=np.uint8)
    residual = np.full((count, 17), np.nan, dtype=np.float32)
    calibrated = [c for c in scene.cameras if c.cam is not None]
    cams = [c.cam for c in calibrated]
    multi = np.flatnonzero((table[:, [c.index for c in calibrated]] >= 0).sum(axis=1) >= 2)
    for start in range(0, len(multi), chunk):
        block = multi[start:start + chunk]
        native = np.zeros((len(cams), len(block) * 17, 2))
        rays = np.full((len(cams), len(block) * 17, 3), np.nan)
        usable = np.zeros((len(cams), len(block) * 17), dtype=bool)
        for v, camera in enumerate(calibrated):
            rows = table[block, camera.index]
            present = rows >= 0
            if not present.any():
                continue
            points = camera.detections.keypoints[rows[present]]
            mask = np.zeros((len(block), 17), dtype=bool)
            mask[present] = camera.joint_ok[rows[present]]
            pixel = np.zeros((len(block), 17, 2))
            pixel[present] = points[:, :, :2]
            ray = np.full((len(block), 17, 3), np.nan)
            ray[present] = camera.rays[rows[present]]
            native[v], rays[v], usable[v] = pixel.reshape(-1, 2), ray.reshape(-1, 3), mask.reshape(-1)
        points, kept, errors = triangulate_views(
            cams, native, usable, max_reprojection_px=geometry["max_reprojection_px"],
            min_ray_angle_deg=geometry["min_ray_angle_deg"], rays=rays)
        xyz[block] = points.reshape(len(block), 17, 3)
        bits = np.zeros(len(block) * 17, dtype=np.uint8)
        for v, camera in enumerate(calibrated):
            bits |= (kept[v].astype(np.uint8) << camera.index)
        used[block] = bits.reshape(len(block), 17)
        worst = np.max(np.where(kept, np.nan_to_num(errors, nan=0.0), 0.0), axis=0)
        residual[block] = np.where(kept.any(axis=0), worst, np.nan).reshape(len(block), 17)
    root = np.full((count, 3), np.nan, dtype=np.float32)
    hips_ok = np.isfinite(xyz[:, HIPS, 0]).all(axis=1)
    root[hips_ok] = xyz[hips_ok][:, HIPS].mean(axis=1)
    torso = xyz[:, HIPS + SHOULDERS]
    torso_count = np.isfinite(torso[:, :, 0]).sum(axis=1)
    fallback = ~hips_ok & (torso_count >= 2)
    root[fallback] = np.nanmean(torso[fallback], axis=1)
    # A root below the floor or implausibly high is a geometry failure, not a measurement.
    low, high = geometry["root_z_range_in"]
    return PersonFrames(label, t, table, xyz, used, residual, root), duplicates


def root_plausibility(frames: PersonFrames, recipe: dict) -> np.ndarray:
    """Share of each label's measured roots inside the physically plausible height band."""
    low, high = recipe["geometry"]["floor_veto_in"], 110.0
    measured = np.isfinite(frames.root[:, 2])
    inside = measured & (frames.root[:, 2] > low) & (frames.root[:, 2] < high)
    labels = int(frames.label.max()) + 1 if len(frames) else 0
    total = np.bincount(frames.label[measured], minlength=labels)
    good = np.bincount(frames.label[inside], minlength=labels)
    return np.where(total > 0, good / np.maximum(1, total), np.nan)
