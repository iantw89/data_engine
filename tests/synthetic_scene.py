"""Synthetic multi-camera scenes for solver tests.

Builds a real ``data_engine.solve.scene.Scene`` directly (no files): simulated people with
articulated 17-joint COCO skeletons (inches, z up) are projected through the standard
calibration, detections are made per camera and grouped into 2D tracklets, and the ground
truth person of every detection row is kept so tests can measure purity / completeness.

Usage::

    sim = SceneBuilder(T=60, seed=1)
    p = sim.add_person(stationary((100, 50)))
    q = sim.add_person(walking((0, 0), (200, 100)))
    sim.see(p, ["0", "1", "3"])               # whole clip
    sim.see(q, ["0", "1"], 10, 40)            # frames [10, 40)
    s = sim.build()                           # -> SynthScene (.scene, .gt, ...)

Tracklets: by default one tracklet per (person, camera, contiguous visible run).
``fragment=n`` splits a run every n frames; ``sim.tracklet(cam, [(pid, t0, t1), ...])``
builds an explicit tracklet that follows several people in turn (identity switch).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from data_engine.config import load_site
from data_engine.geometry import Cam, load_calibration
from data_engine.solve.scene import CameraScene, Scene
from data_engine.store import CameraDetections

CALIBRATION = Path(load_site()["calibration"])
HAVE_CALIBRATION = CALIBRATION.is_file()
CAM_IDS = [str(i) for i in range(7)]
IMG_W, IMG_H = 2048, 1536
MIN_VISIBLE_JOINTS = 10


def load_cams() -> dict[str, Cam]:
    return load_calibration(CALIBRATION)


# --------------------------------------------------------------------------- skeletons


def skeleton(xy: np.ndarray, heading: np.ndarray, phase: float, *, pose: str = "stand", swing: float = 6.0,
             walk: bool = False) -> np.ndarray:
    """(T,17,3) joints in inches for root positions xy (T,2) and heading angles (T,) in radians.

    ``phase`` (T,) drives arm/leg swing so that successive frames differ slightly.
    """
    T = len(xy)
    f = np.stack([np.cos(heading), np.sin(heading)], axis=1)  # forward
    l = np.stack([-np.sin(heading), np.cos(heading)], axis=1)  # left
    sw = swing * np.sin(phase)

    def at(forward, left, z):
        forward, left, z = (np.broadcast_to(np.asarray(v, float), (T,)) for v in (forward, left, z))
        p = xy + f * forward[:, None] + l * left[:, None]
        return np.column_stack([p, z])

    j = np.zeros((T, 17, 3))
    if pose == "stand":
        hip, sho, head = 36.0, 56.0, 64.0
        leg = sw if walk else 0.0 * sw
        j[:, 11], j[:, 12] = at(0, 6, hip), at(0, -6, hip)
        j[:, 13], j[:, 14] = at(leg * 0.6, 6, 19), at(-leg * 0.6, -6, 19)
        j[:, 15], j[:, 16] = at(leg, 7, 3), at(-leg, -7, 3)
    else:  # seated: thighs forward, shins down
        hip, sho, head = 19.0, 39.0, 47.0
        j[:, 11], j[:, 12] = at(0, 6, hip), at(0, -6, hip)
        j[:, 13], j[:, 14] = at(15, 6, 21), at(15, -6, 21)
        j[:, 15], j[:, 16] = at(16, 7, 4), at(16, -7, 4)
    j[:, 5], j[:, 6] = at(0, 8, sho), at(0, -8, sho)
    j[:, 7], j[:, 8] = at(sw * 0.5, 10, sho - 11), at(-sw * 0.5, -10, sho - 11)
    j[:, 9], j[:, 10] = at(sw + 2, 11, sho - 22), at(-sw + 2, -11, sho - 22)
    j[:, 0] = at(3, 0, head)
    j[:, 1], j[:, 2] = at(2.5, 1.5, head + 1.5), at(2.5, -1.5, head + 1.5)
    j[:, 3], j[:, 4] = at(0, 3.5, head), at(0, -3.5, head)
    return j


@dataclass
class Trajectory:
    xyz: np.ndarray  # (T,17,3)

    def __len__(self):
        return len(self.xyz)


def stationary(xy, T: int, heading: float = 0.0, pose: str = "stand", phase0: float = 0.0, swing: float = 3.0) -> Trajectory:
    t = np.arange(T)
    pos = np.tile(np.asarray(xy, float), (T, 1))
    return Trajectory(skeleton(pos, np.full(T, heading), 0.35 * t + phase0, pose=pose, swing=swing))


def walking(start, end, T: int, pose: str = "stand", phase0: float = 0.0) -> Trajectory:
    t = np.arange(T)
    s = np.linspace(0, 1, T)[:, None]
    pos = np.asarray(start, float) * (1 - s) + np.asarray(end, float) * s
    d = np.asarray(end, float) - np.asarray(start, float)
    heading = np.full(T, np.arctan2(d[1], d[0]))
    return Trajectory(skeleton(pos, heading, 0.5 * t + phase0, pose=pose, swing=8.0, walk=True))


# --------------------------------------------------------------------------- builder


@dataclass
class SynthScene:
    scene: Scene
    gt: list[np.ndarray]  # per camera (M,) simulated person of each raw row
    truth: dict[int, np.ndarray]  # pid -> (T,17,3) world joints
    tracklet_ids: dict[tuple, int]  # (cam id, local key) -> global tracklet id
    cam_ids: list[str]
    seen: dict[tuple[int, int], np.ndarray]  # (pid, camera index) -> (T,) bool detections present
    joint_ok_truth: dict[tuple[int, int], np.ndarray]  # (pid, camera index) -> (T,17) bool joint inside image

    # ---- row helpers
    def cam_index(self, cam: str) -> int:
        return self.cam_ids.index(cam)

    def rows(self, pid: int, cam: str | int, t0: int = 0, t1: int | None = None) -> np.ndarray:
        c = cam if isinstance(cam, int) else self.cam_index(cam)
        camera = self.scene.cameras[c]
        sel = (self.gt[c] == pid) & camera.kept & (camera.t >= t0) & (camera.t < (t1 if t1 is not None else 10 ** 9))
        return np.flatnonzero(sel)

    def tracklet_of(self, pid: int, cam: str, t: int | None = None) -> int:
        rows = self.rows(pid, cam)
        camera = self.scene.cameras[self.cam_index(cam)]
        if t is not None:
            rows = rows[camera.t[rows] == t]
        return int(camera.tracklet[rows[0]])

    def endpoint(self, pid: int, cam: str, t: int) -> tuple[int, int]:
        """(camera index, row) of the detection of ``pid`` in ``cam`` at sync index ``t``."""
        c = self.cam_index(cam)
        rows = self.rows(pid, c, t, t + 1)
        assert len(rows), (pid, cam, t)
        return c, int(rows[0])

    def pid_rows(self) -> list[tuple[int, int, int]]:
        """All (camera index, row, pid)."""
        out = []
        for camera in self.scene.cameras:
            out.extend((camera.index, int(r), int(self.gt[camera.index][r])) for r in camera.order)
        return out

    # ---- purity
    def composition(self, row_label: list[np.ndarray]) -> dict[int, Counter]:
        """component label -> Counter of simulated persons over all its rows."""
        comp: dict[int, Counter] = {}
        for camera in self.scene.cameras:
            lab = row_label[camera.index]
            for row in camera.order:
                if lab[row] >= 0:
                    comp.setdefault(int(lab[row]), Counter())[int(self.gt[camera.index][row])] += 1
        return comp

    def mixed(self, row_label: list[np.ndarray], min_share: float = 0.0) -> dict[int, Counter]:
        """Components holding detections of more than one person."""
        return {l: c for l, c in self.composition(row_label).items() if len(c) > 1}

    def components_of(self, row_label: list[np.ndarray], pid: int, min_rows: int = 1) -> list[int]:
        return sorted(l for l, c in self.composition(row_label).items() if c.get(pid, 0) >= min_rows)

    def report(self, row_label: list[np.ndarray]) -> dict:
        comp = self.composition(row_label)
        pids = sorted(self.truth)
        per_person = {p: sorted(l for l, c in comp.items() if c.get(p, 0)) for p in pids}
        return {"components": len(comp), "mixed": {l: dict(c) for l, c in comp.items() if len(c) > 1},
                "per_person": per_person, "fragments": {p: len(v) for p, v in per_person.items()}}


class SceneBuilder:
    def __init__(self, T: int, seed: int = 0, noise_px: float = 1.0, peak: float = 0.9):
        self.T, self.rng, self.noise, self.peak = T, np.random.default_rng(seed), noise_px, peak
        self.cams = load_cams()
        self.cam_ids = CAM_IDS
        self.persons: dict[int, Trajectory] = {}
        self.visible: dict[tuple[int, str], np.ndarray] = {}  # (pid, cam) -> (T,) bool
        self.explicit: dict[str, list[list[tuple[int, int, int]]]] = {}  # cam -> list of tracklets of (pid,t0,t1)
        self.fragment: dict[tuple[int, str], int] = {}
        self.low_joints: dict[tuple[int, str], tuple[list[int], float]] = {}
        self.dropped_joints: dict[tuple[int, str], list[int]] = {}

    # ---- people
    def add_person(self, trajectory: Trajectory) -> int:
        assert len(trajectory) == self.T
        pid = len(self.persons)
        self.persons[pid] = trajectory
        return pid

    def see(self, pid: int, cams, t0: int = 0, t1: int | None = None, *, fragment: int | None = None,
            drop=()) -> None:
        """Camera(s) detect ``pid`` over frames [t0, t1); ``drop`` lists frames with no detection (occlusion)."""
        for cam in ([cams] if isinstance(cams, str) else cams):
            mask = self.visible.setdefault((pid, cam), np.zeros(self.T, bool))
            mask[t0:(t1 if t1 is not None else self.T)] = True
            for t in drop:
                mask[t] = False
            if fragment:
                self.fragment[(pid, cam)] = fragment

    def tracklet(self, cam: str, parts: list[tuple[int, int, int]]) -> None:
        """Explicit tracklet in ``cam`` following person ``pid`` over [t0, t1) for each part, in order."""
        self.explicit.setdefault(cam, []).append(parts)

    def low_peak(self, pid: int, cam: str, joints: list[int], value: float = 0.05) -> None:
        self.low_joints[(pid, cam)] = (list(joints), value)

    # ---- build
    def build(self) -> SynthScene:
        T = self.T
        truth = {pid: tr.xyz for pid, tr in self.persons.items()}
        cameras, gts = [], []
        tracklet_ids: dict[tuple, int] = {}
        seen: dict[tuple[int, int], np.ndarray] = {}
        joint_in: dict[tuple[int, int], np.ndarray] = {}
        next_tracklet = 0
        for index, cid in enumerate(self.cam_ids):
            cam = self.cams[cid]
            records = []  # (t, pid, tracklet global id)
            claimed: dict[int, np.ndarray] = {}
            # explicit tracklets first
            for k, parts in enumerate(self.explicit.get(cid, [])):
                tid = next_tracklet
                next_tracklet += 1
                tracklet_ids[(cid, f"explicit{k}")] = tid
                for pid, t0, t1 in parts:
                    mask = self.visible.get((pid, cid), np.zeros(T, bool)).copy()
                    sel = np.zeros(T, bool)
                    sel[t0:t1] = True
                    sel &= mask
                    claimed.setdefault(pid, np.zeros(T, bool))[sel] = True
                    records.extend((int(t), pid, tid) for t in np.flatnonzero(sel))
            for pid in self.persons:
                mask = self.visible.get((pid, cid))
                if mask is None:
                    continue
                rest = mask & ~claimed.get(pid, np.zeros(T, bool))
                frag = self.fragment.get((pid, cid))
                ts = np.flatnonzero(rest)
                if not len(ts):
                    continue
                runs = np.split(ts, np.flatnonzero(np.diff(ts) > 1) + 1)
                for r, run in enumerate(runs):
                    pieces = [run] if not frag else [run[i:i + frag] for i in range(0, len(run), frag)]
                    for q, piece in enumerate(pieces):
                        tid = next_tracklet
                        next_tracklet += 1
                        tracklet_ids[(cid, f"p{pid}r{r}q{q}")] = tid
                        records.extend((int(t), pid, tid) for t in piece)
            records.sort(key=lambda x: (x[0], x[1]))
            M = len(records)
            kp = np.zeros((M, 17, 3), np.float32)
            boxes = np.zeros((M, 5), np.float32)
            tracklet = np.full(M, -1, np.int32)
            t_arr = np.zeros(M, np.int32)
            gt = np.zeros(M, np.int32)
            drop_rows = []
            for m, (t, pid, tid) in enumerate(records):
                world = truth[pid][t]
                pix = cam.project(world)
                inside = (np.isfinite(pix).all(axis=1) & (pix[:, 0] >= 0) & (pix[:, 0] < IMG_W)
                          & (pix[:, 1] >= 0) & (pix[:, 1] < IMG_H))
                noisy = pix + self.rng.normal(0, self.noise, pix.shape)
                peak = np.where(inside, self.peak, 0.0)
                if (pid, cid) in self.low_joints:
                    js, v = self.low_joints[(pid, cid)]
                    peak[js] = np.minimum(peak[js], v)
                xy = np.where(inside[:, None], noisy, 0.0)
                if inside.sum() < MIN_VISIBLE_JOINTS:
                    drop_rows.append(m)
                    continue
                kp[m, :, :2], kp[m, :, 2] = xy, peak
                lo, hi = xy[inside].min(axis=0), xy[inside].max(axis=0)
                pad = 0.05 * (hi - lo)
                boxes[m] = [lo[0] - pad[0], lo[1] - pad[1], hi[0] + pad[0], hi[1] + pad[1], 0.9]
                tracklet[m], t_arr[m], gt[m] = tid, t, pid
                seen.setdefault((pid, index), np.zeros(T, bool))[t] = True
                joint_in.setdefault((pid, index), np.zeros((T, 17), bool))[t] = inside
            keep = np.ones(M, bool)
            keep[drop_rows] = False
            kp, boxes, tracklet, t_arr, gt = kp[keep], boxes[keep], tracklet[keep], t_arr[keep], gt[keep]
            M = len(t_arr)
            start = np.searchsorted(t_arr, np.arange(T + 1)).astype(np.int64)
            dets = CameraDetections(cid, T, start, boxes, kp, np.ones(T, bool))
            kept = np.ones(M, bool)
            order = np.arange(M)  # already sorted by t
            camera = CameraScene(index, cid, cam, dets, kept, tracklet, t_arr, start.copy(), order)
            gate = 0.3
            camera.rays = np.full((M, 17, 3), np.nan, np.float32)
            camera.joint_ok = np.zeros((M, 17), bool)
            if M:
                directions = cam.rays(kp[:, :, :2])
                valid = np.isfinite(directions).all(axis=-1) & (kp[:, :, 2] >= gate)
                camera.rays[:] = directions.astype(np.float32)
                camera.joint_ok[:] = valid
            cameras.append(camera)
            gts.append(gt)
        native = np.tile(np.arange(T, dtype=np.int32)[:, None], (1, len(self.cam_ids)))
        scene = Scene(None, cameras, native, np.arange(T) / 10.0, 0.3)  # type: ignore[arg-type]
        return SynthScene(scene, gts, truth, tracklet_ids, self.cam_ids, seen, joint_in)


# --------------------------------------------------------------------------- helpers for tests


def visible_cameras(cams: dict[str, Cam], world: np.ndarray, min_joints: int = MIN_VISIBLE_JOINTS) -> list[str]:
    """Cameras whose image contains at least ``min_joints`` of the (17,3) joints."""
    out = []
    for cid, cam in cams.items():
        pix = cam.project(world)
        inside = (np.isfinite(pix).all(axis=1) & (pix[:, 0] >= 0) & (pix[:, 0] < IMG_W)
                  & (pix[:, 1] >= 0) & (pix[:, 1] < IMG_H))
        if inside.sum() >= min_joints:
            out.append(cid)
    return out


TORSO_JOINTS = [5, 6, 11, 12]


def trap_position(cams: dict[str, Cam], cam_id: str, p_xy, *, offset_in: float = 0.0, seated_torso_z: float = 29.0):
    """World xy for a seated person Q whose torso lies on camera ``cam_id``'s ray through standing P's torso.

    A person on that ray is two-view consistent with P in camera A for *any* second camera (the epipolar
    ambiguity); a seated person is an approximately homothetic copy of a standing one seen from a ceiling
    camera, so the whole skeleton nearly agrees. ``offset_in`` moves Q sideways (horizontally, perpendicular
    to the ray) to make the trap imperfect.
    """
    cam = cams[cam_id]
    p_torso = stationary(p_xy, 1).xyz[0][TORSO_JOINTS].mean(axis=0)
    direction = (p_torso - cam.center) / np.linalg.norm(p_torso - cam.center)
    q_xy = (cam.center + (seated_torso_z - cam.center[2]) / direction[2] * direction)[:2]
    flat = np.asarray(p_xy, float) - cam.center[:2]
    flat /= np.linalg.norm(flat)
    return q_xy + offset_in * np.array([-flat[1], flat[0]])


def assert_pure(sim: SynthScene, row_label: list[np.ndarray], what: str = "") -> None:
    mixed = sim.mixed(row_label)
    assert not mixed, f"{what} components mix people: {({l: dict(c) for l, c in mixed.items()})}"
