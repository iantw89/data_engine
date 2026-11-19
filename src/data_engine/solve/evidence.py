"""Cross-camera pair evidence.

For every pair of simultaneous detections in two different cameras, the two-view residual
of their common reliable body joints says how well they agree as one 3D person. A single
frame is weak evidence (another person on the same epipolar line agrees too), so residuals
are accumulated per (tracklet A, tracklet B, time window) cell: people move, and only the
true partner stays consistent over time.

Residuals are normalised to *accept units*: ``u = residual_px / max(accept_px,
accept_rel * box_height)``, so ``u <= 1`` meets the golden accept gate (5 px / .025 box
height) and ``u <= 2`` the review gate (10 px / .05).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import BODY, TORSO, pair_ray_evidence
from .scene import Scene

U_CLIP = 12.0  # residuals beyond this are all "certainly different" and carry no more information

# Limb segments between body joints (indices into BODY: shoulders 0-1, elbows 2-3, wrists 4-5,
# hips 6-7, knees 8-9, ankles 10-11) with generous adult length limits in inches. Pairing the
# views of two different people triangulates joints at unrelated depths, so the "skeleton"
# has impossible proportions; a real person's does not, whatever the pose.
BONES = np.array([(0, 1), (6, 7), (0, 2), (1, 3), (2, 4), (3, 5), (6, 8), (7, 9), (8, 10), (9, 11), (0, 6), (1, 7)])
BONE_LIMITS = np.array([(9, 21), (4, 15), (7, 16), (7, 16), (6, 14), (6, 14), (11, 23), (11, 23), (10, 22), (10, 22),
                        (13, 28), (13, 28)], dtype=np.float64)


def bone_statistics(midpoint: np.ndarray, usable: np.ndarray, tolerance: float = 1.25):
    """Per sample: (measurable limb segments, segments outside the limits widened by ``tolerance``)."""
    first, second = BONES[:, 0], BONES[:, 1]
    measurable = usable[:, first] & usable[:, second]
    length = np.linalg.norm(midpoint[:, first] - midpoint[:, second], axis=-1)
    bad = measurable & ((length < BONE_LIMITS[:, 0] / tolerance) | (length > BONE_LIMITS[:, 1] * tolerance))
    return measurable.sum(axis=1).astype(np.uint8), bad.sum(axis=1).astype(np.uint8)


@dataclass
class PairSamples:
    """One record per simultaneous cross-camera detection pair with enough common joints."""
    cam_a: np.ndarray  # (P,) int8
    cam_b: np.ndarray
    row_a: np.ndarray  # (P,) int32 raw rows in camera a
    row_b: np.ndarray
    t: np.ndarray  # (P,) int32 sync index
    u: np.ndarray  # (P,) float32 normalised residual (clipped)
    plausible: np.ndarray  # (P,) bool: in front of both cameras, ray angle and floor checks passed
    bones: np.ndarray = None  # (P,) uint8 limb segments measurable from the two-view triangulation
    bones_bad: np.ndarray = None  # (P,) uint8 of those, how many have an impossible length

    def __len__(self) -> int:
        return len(self.t)


def _cross_pairs(scene: Scene, a: int, b: int, ts: np.ndarray):
    """All (row in a, row in b, t) combinations at sync indices ``ts``."""
    cam_a, cam_b = scene.cameras[a], scene.cameras[b]
    count_a = (cam_a.row_start[ts + 1] - cam_a.row_start[ts]).astype(np.int64)
    count_b = (cam_b.row_start[ts + 1] - cam_b.row_start[ts]).astype(np.int64)
    pairs = count_a * count_b
    keep = pairs > 0
    ts, count_a, count_b, pairs = ts[keep], count_a[keep], count_b[keep], pairs[keep]
    if not len(ts):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    first = np.concatenate(([0], np.cumsum(pairs)[:-1]))
    local = np.arange(pairs.sum()) - np.repeat(first, pairs)
    width = np.repeat(count_b, pairs)
    index_a = np.repeat(cam_a.row_start[ts], pairs) + local // width
    index_b = np.repeat(cam_b.row_start[ts], pairs) + local % width
    return cam_a.order[index_a], cam_b.order[index_b], np.repeat(ts, pairs)


def pair_samples(scene: Scene, recipe: dict, *, step: int = 1, min_common: int = 5,
                 chunk: int = 250_000, only_camera: int | None = None) -> PairSamples:
    """Two-view evidence for every simultaneous cross-camera detection pair.

    ``step`` subsamples the timeline (2 = 5 Hz), which halves the cost when the pairing
    evidence is needed only per time window.
    """
    geometry, association = recipe["geometry"], recipe["association"]
    accept_px, accept_rel = association["accept_p80_px"], association["accept_p80_rel"]
    min_angle, floor = geometry["min_ray_angle_deg"], geometry["floor_veto_in"]
    torso_in_body = np.isin(BODY, TORSO)
    out: dict[str, list] = {key: [] for key in ("cam_a", "cam_b", "row_a", "row_b", "t", "u", "plausible", "bones",
                                                 "bones_bad")}
    bone_veto = geometry.get("bone_veto")  # None = measure only; else {"min_bones": n, "max_bad_share": x}
    ts_all = np.arange(0, scene.T, step)
    cameras = [c for c in scene.cameras if c.cam is not None and len(c.order)]
    centres = np.array([c.cam.center for c in scene.cameras if c.cam is not None])
    ceiling = float(centres[:, 2].max())  # nobody's torso is above the cameras
    room_low = centres[:, :2].min(axis=0) - geometry["room_margin_in"]
    room_high = centres[:, :2].max(axis=0) + geometry["room_margin_in"]
    for i, cam_a in enumerate(cameras):
        for cam_b in cameras[i + 1:]:
            if only_camera is not None and only_camera not in (cam_a.index, cam_b.index):
                continue
            rows_a, rows_b, ts = _cross_pairs(scene, cam_a.index, cam_b.index, ts_all)
            height = np.minimum(cam_a.height_cvat[rows_a], cam_b.height_cvat[rows_b])
            for start in range(0, len(ts), chunk):
                block = slice(start, start + chunk)
                ra, rb = rows_a[block], rows_b[block]
                rays_a = cam_a.rays[ra][:, BODY].astype(np.float64)
                rays_b = cam_b.rays[rb][:, BODY].astype(np.float64)
                common = cam_a.joint_ok[ra][:, BODY] & cam_b.joint_ok[rb][:, BODY]
                midpoint, residual, angle, front = pair_ray_evidence(
                    cam_a.cam.center, np.nan_to_num(rays_a), cam_a.cam.focal,
                    cam_b.cam.center, np.nan_to_num(rays_b), cam_b.cam.focal)
                enough = common.sum(axis=1) >= min_common
                if not enough.any():
                    continue
                common, residual, angle, front, midpoint = (
                    common[enough], residual[enough], angle[enough], front[enough], midpoint[enough])
                masked = np.where(common, residual, np.nan)
                cost = np.nanmedian(masked, axis=1)
                gate = np.maximum(accept_px, accept_rel * height[block][enough])
                u = np.minimum(cost / gate, U_CLIP)
                # Plausibility (golden physical veto): joints in front of both cameras, a usable
                # ray angle, and the torso not below the floor.
                angle_ok = np.nanmedian(np.where(common, angle, np.nan), axis=1) >= min_angle
                front_ok = np.where(common, front, True).mean(axis=1) >= 0.75
                torso = common & torso_in_body[None, :]
                torso_z = np.where(torso, midpoint[..., 2], np.nan)
                has_torso = torso.any(axis=1)
                below = np.zeros(len(u), dtype=bool)
                below[has_torso] = np.nanmax(torso_z[has_torso], axis=1) < floor
                # Outside the room: a wrong pairing puts the body at a wrong depth along the
                # ray, which for ceiling cameras means far above head height or outside the
                # footprint spanned by the cameras.
                outside = np.zeros(len(u), dtype=bool)
                centre = np.nanmean(np.where(common[..., None], midpoint, np.nan), axis=1)
                outside |= centre[:, 2] > ceiling
                outside |= ((centre[:, :2] < room_low) | (centre[:, :2] > room_high)).any(axis=1)
                plausible = angle_ok & front_ok & ~below & ~outside
                bones, bones_bad = bone_statistics(midpoint, common & front)
                if bone_veto:
                    plausible &= ~((bones >= bone_veto["min_bones"])
                                   & (bones_bad > bone_veto["max_bad_share"] * bones))
                out["bones"].append(bones)
                out["bones_bad"].append(bones_bad)
                out["cam_a"].append(np.full(len(u), cam_a.index, np.int8))
                out["cam_b"].append(np.full(len(u), cam_b.index, np.int8))
                out["row_a"].append(ra[enough].astype(np.int32))
                out["row_b"].append(rb[enough].astype(np.int32))
                out["t"].append(ts[block][enough].astype(np.int32))
                out["u"].append(u.astype(np.float32))
                out["plausible"].append(plausible)
    if not out["t"]:
        return PairSamples(*[np.empty(0, dtype) for dtype in (np.int8, np.int8, np.int32, np.int32, np.int32,
                                                              np.float32, bool, np.uint8, np.uint8)])
    return PairSamples(**{key: np.concatenate(values) for key, values in out.items()})


@dataclass
class Cells:
    """Pair evidence accumulated per (unit a, unit b, window); units are tracklets or fragments."""
    a: np.ndarray  # (K,) int64 unit ids, a < b
    b: np.ndarray
    window: np.ndarray  # (K,) int32
    n: np.ndarray  # (K,) samples
    n_plausible: np.ndarray
    median: np.ndarray  # (K,) float32 median u
    p80: np.ndarray  # (K,) float32 80th percentile of u
    frac_accept: np.ndarray  # (K,) float32 share of samples with u <= 1 and plausible

    def __len__(self) -> int:
        return len(self.a)


def accumulate(unit_a: np.ndarray, unit_b: np.ndarray, window: np.ndarray, u: np.ndarray,
               plausible: np.ndarray) -> Cells:
    """Group samples by (min unit, max unit, window) and summarise each group."""
    valid = (unit_a >= 0) & (unit_b >= 0) & (unit_a != unit_b)
    unit_a, unit_b, window, u, plausible = unit_a[valid], unit_b[valid], window[valid], u[valid], plausible[valid]
    low, high = np.minimum(unit_a, unit_b).astype(np.int64), np.maximum(unit_a, unit_b).astype(np.int64)
    # Implausible samples count as maximally inconsistent: they are evidence against the pair.
    value = np.where(plausible, u, U_CLIP).astype(np.float32)
    order = np.lexsort((value, window, high, low))
    low, high, window, value, plausible = low[order], high[order], window[order], value[order], plausible[order]
    change = np.flatnonzero((np.diff(low) != 0) | (np.diff(high) != 0) | (np.diff(window) != 0)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(low)]))
    counts = ends - starts
    if not len(low):
        empty = np.empty(0)
        return Cells(empty.astype(np.int64), empty.astype(np.int64), empty.astype(np.int32), empty.astype(np.int32),
                     empty.astype(np.int32), empty.astype(np.float32), empty.astype(np.float32),
                     empty.astype(np.float32))
    # Values are sorted within each group, so order statistics are direct lookups.
    median = value[starts + (counts - 1) // 2]
    p80 = value[starts + np.minimum(counts - 1, np.ceil(0.8 * counts).astype(np.int64) - 1 + (counts == 1))]
    p80 = value[np.minimum(ends - 1, starts + np.maximum(0, np.ceil(0.8 * counts).astype(np.int64) - 1))]
    accept = np.add.reduceat((value <= 1.0).astype(np.int32), starts)
    return Cells(low[starts], high[starts], window[starts].astype(np.int32), counts.astype(np.int32),
                 np.add.reduceat(plausible.astype(np.int32), starts).astype(np.int32), median, p80,
                 (accept / counts).astype(np.float32))
