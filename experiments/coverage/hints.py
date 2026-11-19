"""Search hints: where should a person be, from neighbouring MEASURED 3D poses.

A hint is only ever used to decide where to look in an image. It is never an output: no
position derived from a hint is stored. Everything here is reference-free (no golden labels).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common import *  # noqa: F401,F403
from data_engine.geometry import BODY

# Margins from YOLO box vs bbox of the 17 detected keypoints, measured by calibrate_box_margin()
# (see box_calibration.json); fractions of the keypoint-bbox height / width.
MARGIN = {"left": 0.10, "right": 0.10, "top": 0.13, "bottom": 0.14}


@dataclass
class Presence:
    """Per identity: sorted sync indices with a measured root and their row in tracks3d."""
    t: dict[int, np.ndarray]
    index: dict[int, np.ndarray]
    tracks: dict


def build_presence(tracks: dict) -> Presence:
    ok = np.isfinite(tracks["root"]).all(axis=1)
    identity, t = tracks["identity"], tracks["t"]
    order = np.flatnonzero(ok)
    order = order[np.lexsort((t[order], identity[order]))]
    ts, idx = {}, {}
    for who in np.unique(identity[order]):
        sel = order[identity[order] == who]
        # one row per (identity, t): if duplicated keep the first
        _, first = np.unique(t[sel], return_index=True)
        ts[int(who)], idx[int(who)] = t[sel][first], sel[first]
    return Presence(ts, idx, tracks)


@dataclass
class Hint:
    identity: int
    t: int
    pose: np.ndarray  # (17,3) world inches, NaN where unknown
    root: np.ndarray  # (3,)
    before: int  # sync index of the measured pose before (or -1)
    after: int
    kind: str  # 'interp' | 'hold_before' | 'hold_after'
    distance: int  # frames to the nearest measured pose
    span: float  # inches the root moved between the two measured poses (0 for hold)


def make_hint(presence: Presence, who: int, t: int, reach: int = 30, max_gap: int | None = None,
              exclude: set | None = None) -> Hint | None:
    """Hint for identity ``who`` at instant ``t`` from measured poses within ``reach`` frames.

    ``exclude``: set of sync indices (for this identity) to treat as not measured (withhold test).
    """
    ts = presence.t.get(who)
    if ts is None:
        return None
    k = np.searchsorted(ts, t)
    candidates = ts
    before = after = None
    i = k - 1
    while i >= 0 and (exclude is not None and int(ts[i]) in exclude or ts[i] == t):
        i -= 1
    if i >= 0 and t - ts[i] <= reach:
        before = i
    j = k
    while j < len(ts) and (exclude is not None and int(ts[j]) in exclude or ts[j] == t):
        j += 1
    if j < len(ts) and ts[j] - t <= reach:
        after = j
    if before is None and after is None:
        return None
    tr = presence.tracks

    def pose_of(position):
        row = presence.index[who][position]
        return tr["xyz"][row].astype(np.float64), tr["root"][row].astype(np.float64)

    if before is not None and after is not None:
        pb, rb = pose_of(before)
        pa, ra = pose_of(after)
        tb, ta = int(ts[before]), int(ts[after])
        w = (t - tb) / (ta - tb)
        root = (1 - w) * rb + w * ra
        pose = np.full((17, 3), np.nan)
        fb, fa = np.isfinite(pb[:, 0]), np.isfinite(pa[:, 0])
        both = fb & fa
        pose[both] = (1 - w) * pb[both] + w * pa[both]
        only_b, only_a = fb & ~fa, fa & ~fb
        pose[only_b] = pb[only_b] + (root - rb)
        pose[only_a] = pa[only_a] + (root - ra)
        return Hint(who, t, pose, root, tb, ta, "interp", int(min(t - tb, ta - t)), float(np.linalg.norm(ra - rb)))
    if before is not None:
        pb, rb = pose_of(before)
        return Hint(who, t, pb + 0.0, rb, int(ts[before]), -1, "hold_before", int(t - ts[before]), 0.0)
    pa, ra = pose_of(after)
    return Hint(who, t, pa + 0.0, ra, -1, int(ts[after]), "hold_after", int(ts[after] - t), 0.0)


def project_hint(cam, pose: np.ndarray) -> np.ndarray:
    """(17,2) native pixels, NaN where behind the camera / not invertible / unknown joint."""
    out = np.full((17, 2), np.nan)
    ok = np.isfinite(pose[:, 0])
    if ok.any():
        out[ok] = cam.project(pose[ok])
    return out


def keypoint_box(points: np.ndarray, width: float, height: float, margin=MARGIN, extra_px: float = 0.0):
    """Detector-style box (x1,y1,x2,y2) from projected joints; None if fewer than 4 finite points.

    Not clipped to the image so that partly out-of-frame people keep their true extent; callers clip.
    """
    ok = np.isfinite(points).all(axis=1)
    if ok.sum() < 4:
        return None
    x1, y1 = points[ok].min(axis=0)
    x2, y2 = points[ok].max(axis=0)
    w, h = x2 - x1, y2 - y1
    return np.array([x1 - margin["left"] * h - extra_px, y1 - margin["top"] * h - extra_px,
                     x2 + margin["right"] * h + extra_px, y2 + margin["bottom"] * h + extra_px])


def clip_share(box, width, height) -> float:
    """Share of the box area inside the image."""
    area = max(box[2] - box[0], 1e-6) * max(box[3] - box[1], 1e-6)
    x1, y1, x2, y2 = max(box[0], 0), max(box[1], 0), min(box[2], width), min(box[3], height)
    return max(x2 - x1, 0) * max(y2 - y1, 0) / area


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (4,), b (N,4) -> (N,)"""
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def keypoint_agreement(cam, hint_xy: np.ndarray, det_kp: np.ndarray, det_box: np.ndarray,
                       joint_peak: float = 0.3, floor_px: float = 20.0, diag_frac: float = 0.06):
    """(agree, common) over BODY joints: projected hint vs detected keypoints (the engine's
    projection_attach test: gate = max(20 px, 6% of box diagonal), 1280-scale pixels)."""
    scale = cam.to_cvat
    pts = det_kp[BODY]
    hint = hint_xy[BODY]
    common = np.isfinite(hint).all(axis=1) & (pts[:, 2] >= joint_peak)
    if not common.any():
        return 0, 0
    dist = np.linalg.norm(hint[common] - pts[common, :2], axis=1) * scale
    diag = np.hypot(det_box[2] - det_box[0], det_box[3] - det_box[1]) * scale
    gate = max(floor_px, diag_frac * diag)
    return int((dist <= gate).sum()), int(common.sum())


def pair_angle_deg(root: np.ndarray, centers: list[np.ndarray]) -> np.ndarray:
    """Pairwise angle between directions root->camera centre, (V,V) degrees."""
    d = np.stack([c - root for c in centers])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(d @ d.T, -1, 1)))
