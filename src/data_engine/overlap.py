"""Pixel verification of a file boundary the catalog could not verify from encoded packets.

Reolink exports are split into numbered files. data_management removes an overlap between two
files only when the encoded packets match; when they do not, the continuation is placed right
after the previous file ("continuation without overlap evidence") and every later frame of
that camera is late by the unknown overlap. Seen on felix-5, 2025-12-23: the second file
repeats the first file's last 6 frames, which shifted the camera by 0.6 s for half an hour.

This module compares decoded frames instead. It decides only when the evidence is exact and
unique: the continuation's first ``k`` frames are pixel-identical to the previous file's last
``k`` frames, and no other ``k`` comes close. Anything else stays unverified (a static scene,
a real gap between the files, a longer overlap than the search window).

The repeated frames stay in the camera's frame index (it equals data_management's index, and
the detection store is keyed by it); ``Part.overlap`` only removes them from the timeline.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

SEARCH_FRAMES = 60  # frames compared on each side of the boundary (6 s at 10 fps)
MIN_OVERLAP = 2  # one repeated frame is not distinguishable from a still scene
IDENTICAL = 0.05  # mean absolute grey-level difference of a repeated frame (identical packets decode to ~0)
SEPARATION = 4.0  # every other candidate must be at least this many times worse (and above IDENTICAL)


def reduce(image: np.ndarray, width: int = 480) -> np.ndarray:
    """Small float32 grey image for comparison."""
    import cv2

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height = max(1, round(grey.shape[0] * width / grey.shape[1]))
    return cv2.resize(grey, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)


def find_overlap(tail: Sequence[np.ndarray], head: Sequence[np.ndarray]) -> dict:
    """How many leading frames of ``head`` repeat the last frames of ``tail``.

    ``tail`` are the previous file's last frames in order, ``head`` the continuation's first
    frames. Returns ``{"overlap": k or None, "reason": ..., "residual": ..., "next_best": ...}``.
    ``overlap`` is ``None`` unless exactly one ``k >= MIN_OVERLAP`` is an exact repeat.
    """
    limit = min(len(tail), len(head))
    if limit < MIN_OVERLAP + 1:
        return {"overlap": None, "reason": "too few frames to compare", "residual": None, "next_best": None}
    residual = np.full(limit + 1, np.inf)
    for k in range(1, limit + 1):
        worst = 0.0
        for j in range(k):
            worst = max(worst, float(np.abs(head[j] - tail[len(tail) - k + j]).mean()))
            if worst > IDENTICAL * SEPARATION * 4:
                break  # far from a repeat already; the exact value does not matter
        residual[k] = worst
    best = int(np.argmin(residual[1:])) + 1
    others = np.delete(residual[1:], best - 1)
    next_best = float(others.min()) if len(others) else float("inf")
    result = {"residual": round(float(residual[best]), 4), "next_best": round(next_best, 4), "candidate": best}
    if residual[best] > IDENTICAL:
        return {**result, "overlap": None, "reason": "no exact repeat of the previous file's last frames"}
    if next_best < max(IDENTICAL, residual[best]) * SEPARATION:
        return {**result, "overlap": None, "reason": "repeat is not unique (still scene)"}
    if best < MIN_OVERLAP:
        return {**result, "overlap": None, "reason": "a single repeated frame is not evidence"}
    if best == limit:
        return {**result, "overlap": None, "reason": "overlap reaches the search window; not measured"}
    return {**result, "overlap": best, "reason": "exact repeat"}


def verify_boundary(read_previous: Callable[[int], np.ndarray], previous_last: int,
                    read_next: Callable[[int], np.ndarray], next_first: int, next_frames: int,
                    search: int = SEARCH_FRAMES) -> dict:
    """Compare the frames around a boundary. ``previous_last`` is the last used file-local frame
    of the previous file; ``next_first`` the first used file-local frame of the continuation."""
    count = min(search, previous_last + 1, next_frames)
    tail = [reduce(read_previous(i)) for i in range(previous_last - count + 1, previous_last + 1)]
    head = [reduce(read_next(next_first + j)) for j in range(count)]
    return {**find_overlap(tail, head), "frames_compared": count}
