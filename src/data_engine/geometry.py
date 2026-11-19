"""Vectorized multi-view geometry on the standard calibration.

Conventions are those of cvat_labeling ``pseudo3d.geometry`` (the golden-set reference):
``X_cam = R(rvec) X_world + tvec``; OpenCV distortion on a letterboxed canvas where
``canvas = pixel_1280x960 * scale + padding``; world units inches, z up, floor at z = 0.
Residuals are reported in 1280x960-scale pixels so thresholds carry over from the golden
set (12 px reprojection gate, 8 degree ray angle). Inputs are native pixels; each camera
knows its native width and scales internally.

``tests/test_geometry.py`` checks these functions against ``pseudo3d.geometry`` when the
cvat_labeling repository is present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
CALIBRATION_WIDTH = 1280.0  # pixel_cvat width; native pixels are scaled to it

COCO17 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
          "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle"]
BODY = np.arange(5, 17)  # 12 body joints; head points are too close together to add geometry
TORSO = np.array([5, 6, 11, 12])
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)]


@dataclass
class Cam:
    calibration_id: str
    K: np.ndarray
    D: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    padding: np.ndarray
    scale: float
    native_width: float = 2048.0

    def __post_init__(self):
        self.R = cv2.Rodrigues(self.rvec)[0]
        self.center = -self.R.T @ self.tvec
        self.focal = float(self.K[0, 0]) / self.scale  # focal length in 1280x960-scale pixels

    @property
    def to_cvat(self) -> float:
        return CALIBRATION_WIDTH / self.native_width

    def normalized(self, native: np.ndarray) -> np.ndarray:
        """Undistorted normalized image coordinates for native-pixel points, shape (..., 2).

        NaN where the distortion model cannot be inverted (image corners of some cameras):
        re-distorting the result must return the input pixel, otherwise the ray is wrong
        by tens of pixels and must not be used as geometry evidence.
        """
        native = np.asarray(native, dtype=np.float64)
        flat = native.reshape(-1, 1, 2) * self.to_cvat * self.scale + self.padding
        if not len(flat):
            return np.empty(native.shape, dtype=np.float64)
        undistort = getattr(cv2, "undistortPointsIter", cv2.undistortPoints)
        xy = undistort(flat, self.K, self.D, criteria=UNDISTORT_CRITERIA).reshape(-1, 2)
        homogeneous = np.column_stack([xy, np.ones(len(xy))])
        again = cv2.projectPoints(homogeneous, np.zeros(3), np.zeros(3), self.K, self.D)[0].reshape(-1, 2)
        invalid = ~(np.linalg.norm(again - flat.reshape(-1, 2), axis=1) < 0.5 * self.scale)
        xy[invalid] = np.nan
        return xy.reshape(native.shape)

    def rays(self, native: np.ndarray) -> np.ndarray:
        """Unit ray directions in world coordinates for native-pixel points, shape (..., 3)."""
        xy = self.normalized(native)
        directions = np.concatenate([xy, np.ones(xy.shape[:-1] + (1,))], axis=-1) @ self.R
        return directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    def depth(self, world: np.ndarray) -> np.ndarray:
        return (np.asarray(world, dtype=np.float64) @ self.R.T + self.tvec)[..., 2]

    def project(self, world: np.ndarray, *, verify: bool = True) -> np.ndarray:
        """Project world points to native pixels (with distortion), shape (..., 2).

        NaN where the point is behind the camera or, with ``verify``, where the distortion
        model is not invertible there: the rational model folds rays far outside the
        calibrated field of view back into the image, which would place people who are out
        of view inside the frame. A projection is kept only if undistorting it returns the
        original viewing direction (within about a pixel).
        """
        world = np.asarray(world, dtype=np.float64)
        flat = world.reshape(-1, 3)
        result = np.full((len(flat), 2), np.nan)
        valid = np.isfinite(flat).all(axis=1)
        if valid.any():
            camera_points = flat[valid] @ self.R.T + self.tvec
            front = camera_points[:, 2] > 1e-6
            rows = np.flatnonzero(valid)[front]
            if len(rows):
                canvas = cv2.projectPoints(flat[rows], self.rvec, self.tvec, self.K, self.D)[0].reshape(-1, 2)
                native = (canvas - self.padding) / self.scale / self.to_cvat
                if verify:
                    ideal = camera_points[front, :2] / camera_points[front, 2:3]
                    back = self.normalized(native)
                    consistent = np.linalg.norm(back - ideal, axis=1) * self.focal < 1.5
                    native[~consistent] = np.nan
                result[rows] = native
        return result.reshape(world.shape[:-1] + (2,))


def load_calibration(path: Path, native_widths: dict[str, float] | None = None) -> dict[str, Cam]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    padding, scale = np.asarray(data["padding"], float), float(data["scale"])
    return {
        str(cid): Cam(str(cid), np.asarray(v["K"], float), np.asarray(v["D"], float), np.asarray(v["rvec"], float),
                      np.asarray(v["tvec"], float), padding, scale,
                      float((native_widths or {}).get(str(cid), 2048.0)))
        for cid, v in data["cameras"].items()
    }


# --------------------------------------------------------------------------- two-view pair evidence


def pair_ray_evidence(origin_a: np.ndarray, rays_a: np.ndarray, focal_a: float,
                      origin_b: np.ndarray, rays_b: np.ndarray, focal_b: float):
    """Closest approach of ray pairs; all arrays broadcast over leading dimensions.

    Returns (midpoint (...,3), residual_px (...), angle_deg (...), in_front (...) bool).
    ``residual_px`` is each ray's distance to the midpoint expressed in 1280-scale pixels in
    its own view (the larger of the two), i.e. the two-view reprojection residual.
    """
    w = origin_a - origin_b
    b = np.sum(rays_a * rays_b, axis=-1)
    d = np.sum(rays_a * w, axis=-1)
    e = np.sum(rays_b * w, axis=-1)
    denominator = np.maximum(1.0 - b * b, 1e-9)
    s = (b * e - d) / denominator  # distance along ray a
    t = (e - b * d) / denominator  # distance along ray b
    point_a = origin_a + s[..., None] * rays_a
    point_b = origin_b + t[..., None] * rays_b
    midpoint = 0.5 * (point_a + point_b)
    half_gap = 0.5 * np.linalg.norm(point_a - point_b, axis=-1)
    in_front = (s > 1.0) & (t > 1.0)
    residual = np.maximum(half_gap / np.maximum(s, 1.0) * focal_a, half_gap / np.maximum(t, 1.0) * focal_b)
    angle = np.degrees(np.arccos(np.clip(b, -1.0, 1.0)))
    return midpoint, residual, angle, in_front


# --------------------------------------------------------------------------- N-view triangulation


def triangulate_rays(origins: np.ndarray, rays: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Least-squares point nearest to weighted rays.

    origins (V,3); rays (V,N,3) unit; weights (V,N) >= 0 (0 = view unused). Returns (N,3),
    NaN where fewer than two views have weight or the rays are near-parallel.
    """
    eye = np.eye(3)
    projector = eye[None, None] - rays[..., :, None] * rays[..., None, :]  # (V,N,3,3)
    weighted = projector * weights[..., None, None]
    lhs = weighted.sum(axis=0)  # (N,3,3)
    rhs = np.einsum("vnij,vj->ni", weighted, origins)
    count = (weights > 0).sum(axis=0)
    result = np.full((rays.shape[1], 3), np.nan)
    good = count >= 2
    if good.any():
        determinant = np.linalg.det(lhs[good])
        solvable = np.abs(determinant) > 1e-9
        rows = np.flatnonzero(good)[solvable]
        if len(rows):
            result[rows] = np.linalg.solve(lhs[rows], rhs[rows][..., None])[..., 0]
    return result


def max_ray_angle(rays: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Largest pairwise angle (degrees) between used rays per point: rays (V,N,3), weights (V,N)."""
    views = rays.shape[0]
    best = np.zeros(rays.shape[1])
    for i in range(views):
        for j in range(i + 1, views):
            both = (weights[i] > 0) & (weights[j] > 0)
            if both.any():
                cosine = np.clip(np.sum(rays[i] * rays[j], axis=-1), -1.0, 1.0)
                best = np.where(both, np.maximum(best, np.degrees(np.arccos(cosine))), best)
    return best


def triangulate_views(cams: list[Cam], native: np.ndarray, usable: np.ndarray, *,
                      max_reprojection_px: float = 12.0, min_ray_angle_deg: float = 8.0,
                      refine: bool = True, rays: np.ndarray | None = None):
    """Gated N-view triangulation of N points seen by V cameras.

    native (V,N,2) native pixels; usable (V,N) bool. Policy (identity_triangulation, golden
    set): at least two views, positive depth in every used view, largest pairwise ray angle
    >= ``min_ray_angle_deg``, every used view within ``max_reprojection_px`` (1280x960 scale);
    the worst view is dropped and the point retried while more than two views remain.
    Nothing is interpolated: points that fail are NaN.

    Returns xyz (N,3), used (V,N) bool, residual (V,N) px in 1280 scale (NaN if unused).
    """
    views, count = native.shape[:2]
    origins = np.stack([cam.center for cam in cams])
    if rays is None:
        rays = np.stack([cam.rays(native[v]) for v, cam in enumerate(cams)])
    scale = np.array([cam.to_cvat for cam in cams])
    usable = usable & np.isfinite(rays).all(axis=-1)  # un-invertible image corners carry no geometry
    rays = np.nan_to_num(rays).astype(np.float64)
    # Undistorted normalized coordinates, recovered from the rays (camera-frame direction x/z, y/z).
    observed = np.zeros(rays.shape[:2] + (2,))
    for v, cam in enumerate(cams):
        direction = rays[v] @ cam.R.T
        observed[v] = direction[:, :2] / np.where(np.abs(direction[:, 2:3]) > 1e-9, direction[:, 2:3], 1.0)

    def solve(index: np.ndarray, weights: np.ndarray):
        """Triangulate points ``index`` with view weights (V, len(index))."""
        points = triangulate_rays(origins, rays[:, index], weights)
        if refine:
            points = _refine_reprojection(cams, observed[:, index], weights, points)
        errors = np.full((views, len(index)), np.nan)
        behind = np.zeros(len(index), dtype=bool)
        for v, cam in enumerate(cams):
            mask = weights[v] > 0
            if mask.any():
                projected = cam.project(points[mask], verify=False)
                errors[v, mask] = np.linalg.norm(projected - native[v, index][mask], axis=-1) * scale[v]
                behind[mask] |= ~np.isfinite(projected).all(axis=-1)
        worst = np.max(np.where(weights > 0, np.nan_to_num(errors, nan=np.inf), -np.inf), axis=0)
        angle = max_ray_angle(rays[:, index], weights)
        ok = (np.isfinite(points).all(axis=1) & ~behind & ((weights > 0).sum(axis=0) >= 2)
              & (worst <= max_reprojection_px) & (angle >= min_ray_angle_deg))
        return points, errors, worst, ok

    used = usable.copy()
    xyz = np.full((count, 3), np.nan)
    residual = np.full((views, count), np.nan)
    used[:, used.sum(axis=0) < 2] = False
    pending = np.flatnonzero(used.sum(axis=0) >= 2)
    while len(pending):
        weights = used[:, pending].astype(np.float64)
        points, errors, _, ok = solve(pending, weights)
        accepted = pending[ok]
        xyz[accepted] = points[ok]
        residual[:, accepted] = np.where(weights[:, ok] > 0, errors[:, ok], np.nan)
        failed = pending[~ok]
        more = used[:, failed].sum(axis=0) > 2
        used[:, failed[~more]] = False
        failed = failed[more]
        if not len(failed):
            break
        # Leave-one-out: drop the view whose removal leaves the most consistent set. Dropping
        # the largest residual is not enough - a least-squares fit spreads one wrong view's
        # error over the good ones.
        best_worst = np.full(len(failed), np.inf)
        best_view = np.full(len(failed), -1)
        for v in range(views):
            candidates = np.flatnonzero(used[v, failed])
            if not len(candidates):
                continue
            weights = used[:, failed[candidates]].astype(np.float64)
            weights[v] = 0
            _, _, worst, _ = solve(failed[candidates], weights)
            better = worst < best_worst[candidates]
            best_worst[candidates[better]] = worst[better]
            best_view[candidates[better]] = v
        hopeless = best_view < 0
        used[:, failed[hopeless]] = False
        used[best_view[~hopeless], failed[~hopeless]] = False
        pending = failed[~hopeless]
    used &= np.isfinite(xyz).all(axis=1)[None, :]
    return xyz, used, residual



def _refine_reprojection(cams: list[Cam], observed: np.ndarray, weights: np.ndarray, points: np.ndarray,
                         iterations: int = 3) -> np.ndarray:
    """Gauss-Newton on undistorted normalized reprojection error (pixel-equivalent), batched.

    ``observed`` (V,N,2) are undistorted normalized image coordinates.
    """
    points = points.copy()
    valid = np.isfinite(points).all(axis=1)
    if not valid.any():
        return points
    for _ in range(iterations):
        jtj = np.zeros((len(points), 3, 3))
        jtr = np.zeros((len(points), 3))
        for v, cam in enumerate(cams):
            mask = valid & (weights[v] > 0)
            if not mask.any():
                continue
            camera_points = points[mask] @ cam.R.T + cam.tvec
            z = np.maximum(camera_points[:, 2], 1e-6)
            predicted = camera_points[:, :2] / z[:, None]
            error = (predicted - observed[v, mask]) * cam.focal
            # d(x/z, y/z)/dX_world = (1/z) [R0 - x R2 ; R1 - y R2]
            jac = np.stack([cam.R[0][None] - predicted[:, :1] * cam.R[2][None],
                            cam.R[1][None] - predicted[:, 1:] * cam.R[2][None]], axis=1) * (cam.focal / z)[:, None, None]
            w = weights[v, mask][:, None, None]
            jtj[mask] += np.einsum("nki,nkj->nij", jac, jac) * w
            jtr[mask] += np.einsum("nki,nk->ni", jac, error) * weights[v, mask][:, None]
        solvable = valid & (np.abs(np.linalg.det(jtj)) > 1e-12)
        if not solvable.any():
            break
        step = np.linalg.solve(jtj[solvable] + 1e-9 * np.eye(3), jtr[solvable][..., None])[..., 0]
        points[solvable] -= step
    return points
