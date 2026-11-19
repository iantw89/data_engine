"""Candidate selection + gated triangulation for recovered person-frames.

A recovered position is a triangulation of 2D poses read from pixels in >= 2 cameras, with the
engine's gates (peak >= .3 per joint, >= 8 degree ray angle, <= 12 px reprojection, leave-one-out).
The hint only decides WHICH 2D poses belong to the person; it never contributes to the position.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from common import *  # noqa: F401,F403
from data_engine.geometry import BODY, pair_ray_evidence, triangulate_views

HIPS, SHOULDERS = (11, 12), (5, 6)


@dataclass
class Cand:
    camera: int
    kp: np.ndarray            # (17,3) native px + peak
    box: np.ndarray           # (4,)
    source: str               # 'raw_kept', 'raw_filtered', 'raw_other', 'vitpose_crop', ...
    row: int = -1
    score: float = 0.0        # selection prior (agreement with hint)
    meta: dict = field(default_factory=dict)
    _rays: np.ndarray | None = None


@dataclass
class Recovered:
    xyz: np.ndarray           # (17,3)
    used: np.ndarray          # (17, V) bool
    residual: np.ndarray      # (17,) worst used-view residual, px (1280 scale)
    root: np.ndarray          # (3,)
    cams: list                # camera indices contributing
    cands: list               # the Cand objects used
    n_body: int
    median_residual: float
    pair_support: int


def cand_rays(scene, c: Cand) -> np.ndarray:
    if c._rays is None:
        cam = scene.cameras[c.camera].cam
        c._rays = cam.rays(c.kp[:, :2])
    return c._rays


def good_joints(scene, c: Cand, peak: float) -> np.ndarray:
    return (c.kp[:, 2] >= peak) & np.isfinite(cand_rays(scene, c)).all(axis=1)


def root_of(xyz: np.ndarray) -> np.ndarray:
    hips = xyz[list(HIPS)]
    if np.isfinite(hips[:, 0]).all():
        return hips.mean(axis=0)
    torso = xyz[[11, 12, 5, 6]]
    ok = np.isfinite(torso[:, 0])
    if ok.sum() >= 2:
        return torso[ok].mean(axis=0)
    return np.full(3, np.nan)


def pair_stats(scene, a: Cand, b: Cand, peak: float, gate_px: float = 12.0, min_angle: float = 8.0):
    """(supported joints, median residual of supported joints) for the BODY joints of two candidates."""
    ca, cb = scene.cameras[a.camera].cam, scene.cameras[b.camera].cam
    ok = good_joints(scene, a, peak) & good_joints(scene, b, peak)
    ok[:5] = False
    if ok.sum() < 3:
        return 0, np.inf
    _, residual, angle, front = pair_ray_evidence(ca.center, cand_rays(scene, a)[ok], ca.focal,
                                                  cb.center, cand_rays(scene, b)[ok], cb.focal)
    good = front & (residual <= gate_px) & (angle >= min_angle)
    if not good.any():
        return 0, np.inf
    return int(good.sum()), float(np.median(residual[good]))


def triangulate_cands(scene, cands: list[Cand], recipe: dict, peak: float | None = None):
    """Gated triangulation of the 17 joints from the chosen candidates (one per camera)."""
    geometry = recipe["geometry"]
    peak = geometry["joint_peak"] if peak is None else peak
    cams = [c.cam for c in scene.cameras if c.cam is not None]
    index = {c.index: i for i, c in enumerate([c for c in scene.cameras if c.cam is not None])}
    V = len(cams)
    native = np.zeros((V, 17, 2))
    rays = np.full((V, 17, 3), np.nan)
    usable = np.zeros((V, 17), bool)
    for cand in cands:
        v = index[cand.camera]
        native[v] = cand.kp[:, :2]
        rays[v] = cand_rays(scene, cand)
        usable[v] = good_joints(scene, cand, peak)
    xyz, used, residual = triangulate_views(cams, native, usable, max_reprojection_px=geometry["max_reprojection_px"],
                                            min_ray_angle_deg=geometry["min_ray_angle_deg"], rays=rays)
    # Engine rule: a root below the floor veto / implausible height is a failure, handled by the caller.
    worst = np.max(np.where(used, np.nan_to_num(residual, nan=0.0), 0.0), axis=0)
    worst = np.where(used.any(axis=0), worst, np.nan)
    return xyz, used, worst, index


def recover(scene, cands: list[Cand], recipe: dict, *, peak: float | None = None, pair_gate_px: float = 12.0,
            min_pair_joints: int = 6, hint_root: np.ndarray | None = None, hint_radius_in: float | None = None) -> Recovered | None:
    """Pick the mutually consistent set of candidates (at most one per camera) and triangulate it.

    Pairs are scored by supported BODY joints (<= pair_gate_px, >= 8 degrees); the best pair that
    also lands near the hint (if given) seeds the set; other cameras join if consistent with it.
    """
    geometry = recipe["geometry"]
    peak = geometry["joint_peak"] if peak is None else peak
    best = None
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            if a.camera == b.camera:
                continue
            n, med = pair_stats(scene, a, b, peak, pair_gate_px)
            if n < min_pair_joints:
                continue
            if hint_root is not None and hint_radius_in is not None:
                xyz, used, worst, _ = triangulate_cands(scene, [a, b], recipe, peak)
                root = root_of(xyz)
                if not np.isfinite(root).all() or np.linalg.norm(root[:2] - hint_root[:2]) > hint_radius_in:
                    continue
            key = (n, -med)
            if best is None or key > best[0]:
                best = (key, a, b, n)
    if best is None:
        return None
    _, a, b, n = best
    chosen = [a, b]
    for c in cands:
        if c is a or c is b or any(c.camera == x.camera for x in chosen):
            continue
        # joins when consistent with the seed pair (>= min_pair_joints supported with either seed view)
        support = max(pair_stats(scene, c, a, peak, pair_gate_px)[0], pair_stats(scene, c, b, peak, pair_gate_px)[0])
        if support >= min_pair_joints:
            chosen.append(c)
    xyz, used, worst, index = triangulate_cands(scene, chosen, recipe, peak)
    root = root_of(xyz)
    if not np.isfinite(root).all():
        return None
    low, high = geometry["root_z_range_in"]
    if not (low <= root[2] <= high):
        return None
    n_body = int(np.isfinite(xyz[5:17, 0]).sum())
    contributing = sorted({c.camera for c in chosen if used[index[c.camera]].any()})
    if len(contributing) < 2:
        return None
    return Recovered(xyz, used, worst, root, contributing, [c for c in chosen if c.camera in contributing], n_body,
                     float(np.nanmedian(worst)), n)
