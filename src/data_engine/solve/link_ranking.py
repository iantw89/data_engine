"""Ranking the candidates of an identity link: which earlier track is the person who appears here?

Nothing in this module decides anything. Its scores only order the candidates a person is shown in a
re-identification question (``stages/solve.py``), so that the likeliest earlier tracks come first and
fewer rounds of "none of these" are needed. Two independent kinds of evidence are added up:

* **motion** - how far the candidate's last known position is from where the new track starts, for
  the time that passed. People in a gym mostly stay where they are; when they move, they walk.
* **appearance** - ReID similarity of the two tracks around the gap. It is used only where both
  tracks have *clear views*: a detection with most of the skeleton found, not cut by the image
  border and not overlapped by another person's box. A partial skeleton or a crop shared with
  somebody else carries no usable appearance; then the evidence is "unknown" (zero), never
  "different" (``experiments/links/REPORT.md``).

Each kind is expressed as a log-likelihood ratio (same person against a different person), so
"unknown" is zero, and the two can be added. The parameters are in ``RECIPE["link_ranking"]``.
"""

from __future__ import annotations

import math

import numpy as np

from .scene import CameraScene, Scene


# --------------------------------------------------------------------------- clear views

def view_quality(camera: CameraScene, rows: np.ndarray, size: tuple[int, int], joint_peak: float,
                 border_px: float = 4.0) -> dict[str, np.ndarray]:
    """Per detection: ``joints`` found (of 17), ``overlap`` (largest share of its box covered by another kept
    box of the same instant), ``cut`` (the box touches the image border) and box ``height`` in pixels."""
    rows = np.asarray(rows, dtype=np.int64)
    boxes = camera.detections.boxes[rows, :4].astype(np.float64)
    points = camera.detections.keypoints[rows]
    width, height = size
    inside = (points[:, :, 0] >= 0) & (points[:, :, 0] < width) & (points[:, :, 1] >= 0) & (points[:, :, 1] < height)
    joints = ((points[:, :, 2] >= joint_peak) & inside).sum(axis=1)
    cut = (boxes[:, 0] <= border_px) | (boxes[:, 1] <= border_px) | (boxes[:, 2] >= width - border_px) \
        | (boxes[:, 3] >= height - border_px)
    overlap = np.zeros(len(rows))
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-9)
    for i, row in enumerate(rows):
        t = int(camera.t[row])
        if t < 0:
            continue
        others = camera.rows_at(t)
        others = others[others != row]
        if not len(others):
            continue
        other = camera.detections.boxes[others, :4]
        w = np.minimum(boxes[i, 2], other[:, 2]) - np.maximum(boxes[i, 0], other[:, 0])
        h = np.minimum(boxes[i, 3], other[:, 3]) - np.maximum(boxes[i, 1], other[:, 1])
        overlap[i] = float(np.max(np.clip(w, 0, None) * np.clip(h, 0, None)) / area[i])
    return {"joints": joints, "overlap": overlap, "cut": cut, "height": boxes[:, 3] - boxes[:, 1]}


def clear_view(quality: dict[str, np.ndarray], recipe: dict) -> np.ndarray:
    """Which detections show one whole person well enough for appearance to mean anything."""
    return (quality["joints"] >= recipe["min_joints"]) & (quality["overlap"] <= recipe["max_overlap"]) \
        & ~quality["cut"] & (quality["height"] >= recipe["min_height_px"])


class ComponentRows:
    """Rows of each component per camera, without scanning every label array for every question."""

    def __init__(self, scene: Scene, label: list[np.ndarray]):
        self.scene = scene
        self.order, self.sorted = [], []
        for camera in scene.cameras:
            lab = np.asarray(label[camera.index])
            order = np.flatnonzero((lab >= 0) & (camera.t >= 0))
            order = order[np.lexsort((camera.t[order], lab[order]))]
            self.order.append(order)
            self.sorted.append(lab[order])

    def rows(self, camera_index: int, label: int, lo: int, hi: int) -> np.ndarray:
        """Rows of ``label`` in one camera with sync index in [lo, hi], in time order."""
        low = np.searchsorted(self.sorted[camera_index], label, side="left")
        high = np.searchsorted(self.sorted[camera_index], label, side="right")
        rows = self.order[camera_index][low:high]
        t = self.scene.cameras[camera_index].t[rows]
        return rows[(t >= lo) & (t <= hi)]


def window_views(index: ComponentRows, label: int, lo: int, hi: int, recipe: dict, sizes: list[tuple[int, int]],
                 available: list[np.ndarray] | None = None) -> tuple[dict[int, np.ndarray], dict]:
    """Clear views of a component between two sync indices: ``{camera index: rows}`` (at most
    ``crops_per_camera`` each, spread over the window) and what was found
    (``{"detections", "clear", "cameras"}``). ``available[camera]`` (bool per row) restricts the choice to
    detections that have an embedding already (experiments on a fixed set of embeddings)."""
    scene = index.scene
    chosen, total, clear_total = {}, 0, 0
    for camera in scene.cameras:
        rows = index.rows(camera.index, label, lo, hi)
        if available is not None and len(rows):
            rows = rows[available[camera.index][rows]]
        total += len(rows)
        if not len(rows):
            continue
        quality = view_quality(camera, rows, sizes[camera.index], scene.joint_peak, recipe["border_px"])
        rows = rows[clear_view(quality, recipe)]
        clear_total += len(rows)
        if not len(rows):
            continue
        per_camera = int(recipe["crops_per_camera"])
        if len(rows) > per_camera:
            rows = rows[np.unique(np.linspace(0, len(rows) - 1, per_camera).round().astype(int))]
        chosen[camera.index] = rows
    return chosen, {"detections": int(total), "clear": int(clear_total), "cameras": len(chosen)}


# --------------------------------------------------------------------------- appearance

def descriptors(views: dict[int, np.ndarray], embeddings: list[tuple[np.ndarray, np.ndarray]]) -> dict[int, np.ndarray]:
    """One unit vector per camera: the mean embedding of the track's clear views in that camera.
    ``embeddings[camera index] = (row -> slot or -1, unit vectors)`` as ``appearance.embedder`` returns them."""
    result = {}
    for camera_index, rows in views.items():
        slot_of, table = embeddings[camera_index]
        slots = slot_of[rows[rows < len(slot_of)]]
        slots = slots[slots >= 0]
        if not len(slots):
            continue
        total = table[slots].astype(np.float64).sum(axis=0)
        norm = float(np.linalg.norm(total))
        if norm > 1e-9:
            result[camera_index] = total / norm
    return result


def similarity(first: dict[int, np.ndarray], second: dict[int, np.ndarray]) -> dict | None:
    """Cosine similarity of two tracks' descriptors. Where both have clear views in the same camera, those
    cameras are compared one by one and averaged (``same_camera``: same viewpoint, light and background,
    the more reliable comparison); otherwise all views of each are pooled (``cross_camera``). None when
    either track has no clear view."""
    if not first or not second:
        return None
    shared = sorted(set(first) & set(second))
    if shared:
        value = float(np.mean([first[camera] @ second[camera] for camera in shared]))
        return {"similarity": value, "mode": "same_camera", "cameras": len(shared)}
    a, b = np.sum(list(first.values()), axis=0), np.sum(list(second.values()), axis=0)
    value = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))
    return {"similarity": value, "mode": "cross_camera", "cameras": 0}


def appearance_evidence(found: dict | None, recipe: dict) -> float:
    """Log-likelihood ratio (same person : different person) of an appearance similarity; 0 when unknown.
    A logistic curve per comparison mode, ``slope * (similarity - midpoint)``, limited to +-``limit``:
    appearance alone was measured not to be safe, so it may reorder candidates but never outweigh a
    physical impossibility."""
    if found is None:
        return 0.0
    slope, midpoint = recipe["appearance"][found["mode"]]
    limit = float(recipe["appearance"]["limit"])
    return float(np.clip(slope * (found["similarity"] - midpoint), -limit, limit))


# --------------------------------------------------------------------------- motion

def motion_evidence(gap_s: float, distance_m: float, recipe: dict) -> float:
    """Log-likelihood ratio (same person : different person) of ending ``distance_m`` from where the new
    track starts, ``gap_s`` earlier.

    Same person: either they stayed where they were (displacement ~ ``stay_m``, with probability ``stay``)
    or they walked (displacement ~ ``walk_m + walk_mps * gap``). A different person's track ended anywhere
    in the room (uniform over a disc of ``room_m`` radius). Displacements are 2D Gaussians, so the ratio
    of the two densities at distance d is ``(R^2 / 2 sigma^2) exp(-d^2 / 2 sigma^2)`` per component."""
    motion = recipe["motion"]
    room = float(motion["room_m"])

    def ratio(sigma: float) -> float:
        sigma = min(max(sigma, 1e-3), room)
        return room * room / (2.0 * sigma * sigma) * math.exp(-distance_m * distance_m / (2.0 * sigma * sigma))

    stay = float(motion["stay"])
    value = stay * ratio(float(motion["stay_m"])) + (1.0 - stay) * ratio(motion["walk_m"] + motion["walk_mps"] * gap_s)
    return float(math.log(max(value, 1e-12)))


def score(gap_s: float, distance_m: float, found: dict | None, recipe: dict) -> dict:
    """The ranking score of one candidate and its parts (higher = more likely the same person)."""
    motion = motion_evidence(gap_s, distance_m, recipe)
    appearance = appearance_evidence(found, recipe)
    return {"score": round(motion + appearance, 3), "motion": round(motion, 3), "appearance": round(appearance, 3),
            "similarity": None if found is None else round(found["similarity"], 3),
            "compared": None if found is None else found["mode"],
            "speed_mps": round(distance_m / gap_s, 2) if gap_s > 0 else None}
