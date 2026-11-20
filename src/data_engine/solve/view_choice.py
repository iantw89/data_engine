"""Which frame of a piece of track to show a reviewer.

A question shows a few detections of each track: one per camera, taken where the question is (a track's
start or end, the middle of a piece, just before or after a crossing). Those moments are where views are
worst - the track ended *because* the person got hidden - so most detections shown were partial, cut or
overlapped although a clear view of the same person existed a fraction of a second away
(``experiments/images/REPORT.md``).

The rule here changes only *which frame*: among the detections of the **same piece of the same camera's
tracklet** near the chosen one, the nearest clear view is shown instead. The camera and the piece stay what
the question's own rule chose, so what the question can reveal about the grouping (a piece that belongs to
somebody else) is still in front of the reviewer; what is no longer shown as often are problems of a single
detection (a pose built from two people, a sliver, bad keypoints), which a question about tracks finds
only by chance anyway.

A share of the questions (``control_share``) keeps the frames nearest to the moment, so that answers can
keep telling whether the clearer frames hide anything.
"""

from __future__ import annotations

import hashlib

import numpy as np

from . import link_ranking
from .scene import CameraScene


def in_control(anchor: str, recipe: dict) -> bool:
    """Is this question one of those that keep the nearest frames? Decided by the question's anchor detection,
    so it is the same in every round."""
    share = float(recipe.get("control_share", 0.0))
    if share <= 0:
        return False
    return hashlib.sha1(anchor.encode()).digest()[0] < round(share * 256)


def rule(recipe: dict, clear_rule: dict, anchor: str) -> dict | None:
    """The clear-view rule to apply to a question's images, or None (off, or a control question)."""
    if not recipe.get("prefer_clear") or in_control(anchor, recipe):
        return None
    return clear_rule


def clearer(camera: CameraScene, row: int, pool: np.ndarray, size: tuple[int, int], joint_peak: float,
            clear_rule: dict, *, t: int | None = None) -> int:
    """The detection to show instead of ``row``: ``row`` itself if it is a clear view, else the clear view of
    ``pool`` (detections of the same piece, already limited to the time span a choice is allowed in) nearest
    in time to ``t`` (default: the time of ``row``); ``row`` when the pool has none."""
    pool = np.asarray(pool, dtype=np.int64)
    if not len(pool):
        return int(row)
    quality = link_ranking.view_quality(camera, pool, size, joint_peak, clear_rule["border_px"])
    good = pool[link_ranking.clear_view(quality, clear_rule)]
    if not len(good) or row in good:
        return int(row)
    at = int(camera.t[row]) if t is None else int(t)
    return int(good[np.argmin(np.abs(camera.t[good] - at))])
