"""Verify the time alignment with cross-camera geometry.

The alignment generator confirms offsets from short windows and often cannot re-confirm
them later, which says nothing about whether they are right. Geometry can: if a camera's
offset is correct, shifting it by a frame in either direction makes fewer simultaneous
detection pairs agree across cameras. For every camera and block of time this module
counts the well-agreeing two-view pairs at lags -L..+L (in sync frames) and reports whether
lag 0 is the best (or statistically tied with the best).

This verifies; it does not correct. A block whose best lag is clearly not 0 is reported as
an alignment unknown with the evidence, for a person (or a later sub-frame refinement) to
resolve.
"""

from __future__ import annotations

import copy

import numpy as np

from . import evidence
from .scene import Scene


def _shifted(scene: Scene, index: int, lag: int) -> Scene:
    out = copy.copy(scene)
    out.cameras = list(scene.cameras)
    camera = copy.copy(scene.cameras[index])
    t = np.where(camera.t >= 0, camera.t + lag, -1)
    t[(t < 0) | (t >= scene.T)] = -1
    usable = np.flatnonzero(camera.kept & (t >= 0))
    order = usable[np.argsort(t[usable], kind="stable")]
    camera.t, camera.order = t, order
    camera.row_start = np.searchsorted(t[order], np.arange(scene.T + 1)).astype(np.int64)
    out.cameras[index] = camera
    return out


def verify(scene: Scene, recipe: dict, *, max_lag: int = 2, block: int = 3000, step: int = 10,
           tie: float = 0.03, min_pairs: int = 150) -> dict:
    """Per camera and time block: is lag 0 the best-agreeing offset?

    Returns {camera: [{"t": [start, end], "counts": [...], "best_lag": int, "verdict": str}]}
    with verdict "verified" (lag 0 best or within ``tie`` of the best), "too_little_evidence"
    (fewer than ``min_pairs`` agreeing pairs at the best lag) or "disagrees".
    """
    lags = list(range(-max_lag, max_lag + 1))
    blocks = [(start, min(start + block, scene.T)) for start in range(0, scene.T, block)]
    result: dict[str, list[dict]] = {}
    for camera in scene.cameras:
        if camera.cam is None or not len(camera.order):
            continue
        counts = np.zeros((len(blocks), len(lags)), dtype=np.int64)
        for j, lag in enumerate(lags):
            samples = evidence.pair_samples(_shifted(scene, camera.index, lag), recipe, step=step,
                                            only_camera=camera.index)
            good = samples.plausible & (samples.u <= 0.5)
            counts[:, j] = np.bincount(samples.t[good] // block, minlength=len(blocks))[:len(blocks)]
        entries = []
        zero = lags.index(0)
        for (start, end), row in zip(blocks, counts):
            best = int(np.argmax(row))
            if row[best] < min_pairs:
                verdict = "too_little_evidence"
            elif best == zero or row[zero] >= (1 - tie) * row[best]:
                verdict = "verified"
            else:
                verdict = "disagrees"
            entries.append({"t": [int(start), int(end)], "counts": [int(v) for v in row],
                            "best_lag": lags[best], "verdict": verdict})
        result[camera.camera] = entries
    return result
