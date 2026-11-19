"""Recover 3D positions that the footage supports but the detector pass did not deliver.

A track has a *gap* where its person was measured before and after (or just before / just
after an end of the track) but not at an instant in between: the detector missed the person
in the views that see them, or the 2D poses did not pass the triangulation gates. The pixels
are still there. For every such instant this module

1. builds a *hint* - the neighbouring measured 3D poses, interpolated (or held near an end) -
   and projects it into every camera that should see the person, giving a box per camera;
2. has the top-down pose model (ViTPose) read a pose from the image inside each box
   (``workers/recover_trt.py``; twice: on the hint box, then on the box of its own result);
3. accepts a position only if the poses read from at least two cameras triangulate with the
   engine's own gates and pass a stricter acceptance gate.

The hint only says where to look. It never enters a position: a recovered position is a
gated triangulation of poses read from pixels, exactly like every other position. What the
hint could have biased is *which* person a crop shows, so a crop is dropped when another
measured person overlaps the hint box and the pose follows that person at least as well
(the competitor rule), and a recovery is refused when another identity is already measured
at that place.

Measured on the golden recording (``experiments/coverage``): 816 of 5,065 missing
person-frames recovered with no false accept in 6,500 negative controls for three-camera
recoveries (empty floor, displaced hints, absent people) and a median error of 0.5 in
against withheld measured poses. The largest remaining block (about 43%) are gaps longer
than 3 s, where no neighbouring pose is close enough to say where to look.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import BODY, pair_ray_evidence, triangulate_views
from .scene import Scene

BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 11), (6, 12), (5, 6), (11, 12)]
HINT_KINDS = ("interpolated", "held_from_before", "held_from_after")


# --------------------------------------------------------------------------- hints


@dataclass
class Presence:
    """Per identity: sorted sync indices with a measured root, and their row in the track table."""
    t: dict[int, np.ndarray]
    row: dict[int, np.ndarray]
    xyz: np.ndarray
    root: np.ndarray


def build_presence(identity: np.ndarray, t: np.ndarray, root: np.ndarray, xyz: np.ndarray) -> Presence:
    ok = np.flatnonzero((identity >= 0) & np.isfinite(root).all(axis=1))
    ok = ok[np.lexsort((t[ok], identity[ok]))]
    times, rows = {}, {}
    for who in np.unique(identity[ok]):
        mine = ok[identity[ok] == who]
        _, first = np.unique(t[mine], return_index=True)  # one entry per instant
        times[int(who)], rows[int(who)] = t[mine][first].astype(np.int64), mine[first]
    return Presence(times, rows, xyz, root)


@dataclass
class Hint:
    identity: int
    t: int
    pose: np.ndarray  # (17, 3) world inches, NaN where unknown - where to LOOK, never an output
    root: np.ndarray
    kind: int  # index into HINT_KINDS
    distance: int  # frames to the nearest measured pose


def make_hint(presence: Presence, who: int, t: int, reach: int) -> Hint | None:
    times = presence.t.get(who)
    if times is None:
        return None
    after = int(np.searchsorted(times, t))
    if after < len(times) and times[after] == t:
        return None  # measured: nothing to look for
    before = after - 1
    has_before = before >= 0 and t - times[before] <= reach
    has_after = after < len(times) and times[after] - t <= reach
    if not has_before and not has_after:
        return None

    def pose_at(position: int):
        row = presence.row[who][position]
        return presence.xyz[row].astype(np.float64), presence.root[row].astype(np.float64)

    if has_before and has_after:
        pose_b, root_b = pose_at(before)
        pose_a, root_a = pose_at(after)
        weight = (t - times[before]) / (times[after] - times[before])
        root = (1 - weight) * root_b + weight * root_a
        pose = np.full((17, 3), np.nan)
        known_b, known_a = np.isfinite(pose_b[:, 0]), np.isfinite(pose_a[:, 0])
        both = known_b & known_a
        pose[both] = (1 - weight) * pose_b[both] + weight * pose_a[both]
        pose[known_b & ~known_a] = pose_b[known_b & ~known_a] + (root - root_b)
        pose[known_a & ~known_b] = pose_a[known_a & ~known_b] + (root - root_a)
        return Hint(who, t, pose, root, 0, int(min(t - times[before], times[after] - t)))
    if has_before:
        pose, root = pose_at(before)
        return Hint(who, t, pose, root, 1, int(t - times[before]))
    pose, root = pose_at(after)
    return Hint(who, t, pose, root, 2, int(times[after] - t))


def missing_instants(presence: Presence, total: int, end_frames: int) -> list[tuple[int, int]]:
    """(identity, t) for every unmeasured instant inside a track and up to ``end_frames`` beyond its ends."""
    out = []
    for who, times in presence.t.items():
        have = np.zeros(total, bool)
        have[times] = True
        low, high = max(0, int(times[0]) - end_frames), min(total - 1, int(times[-1]) + end_frames)
        out += [(who, int(t)) for t in np.flatnonzero(~have[low:high + 1]) + low]
    return out


# --------------------------------------------------------------------------- image side of a hint


def project(cam, pose: np.ndarray) -> np.ndarray:
    out = np.full((17, 2), np.nan)
    known = np.isfinite(pose[:, 0])
    if known.any():
        out[known] = cam.project(pose[known])
    return out


def keypoint_box(points: np.ndarray, margin: dict) -> np.ndarray | None:
    """Detector-style box around projected joints (margins measured on 297k detections); not clipped."""
    known = np.isfinite(points).all(axis=1)
    if known.sum() < 4:
        return None
    x1, y1 = points[known].min(axis=0)
    x2, y2 = points[known].max(axis=0)
    height = y2 - y1
    return np.array([x1 - margin["left"] * height, y1 - margin["top"] * height,
                     x2 + margin["right"] * height, y2 + margin["bottom"] * height])


def share_inside(box: np.ndarray, width: float, height: float) -> float:
    area = max(box[2] - box[0], 1e-6) * max(box[3] - box[1], 1e-6)
    return max(min(box[2], width) - max(box[0], 0), 0) * max(min(box[3], height) - max(box[1], 0), 0) / area


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    width, height = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    return float(width * height / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - width * height))


def keypoint_agreement(cam, expected: np.ndarray, keypoints: np.ndarray, box: np.ndarray, joint_peak: float) -> float:
    """Share of confident body joints within max(20 px, 6% of the box diagonal) of the expected skeleton
    (1280-scale pixels; the engine's projection-attach test). 0 when nothing is comparable."""
    points, wanted = keypoints[BODY], expected[BODY]
    common = np.isfinite(wanted).all(axis=1) & (points[:, 2] >= joint_peak)
    if not common.any():
        return 0.0
    distance = np.linalg.norm(wanted[common] - points[common, :2], axis=1) * cam.to_cvat
    gate = max(20.0, 0.06 * np.hypot(box[2] - box[0], box[3] - box[1]) * cam.to_cvat)
    return float((distance <= gate).mean())


def view_of(scene: Scene, camera_index: int, t: int, pose: np.ndarray, recipe: dict):
    """(box, projected joints) of a hinted pose in one camera, or None if the person should not be visible."""
    camera = scene.cameras[camera_index]
    if camera.cam is None or scene.native[t, camera_index] < 0:
        return None
    described = scene.workspace.dataset.cameras[camera.camera]
    width, height = float(described.width), float(described.height)
    points = project(camera.cam, pose)
    body = points[5:17]
    inside = np.isfinite(body).all(axis=1) & (body[:, 0] >= 0) & (body[:, 0] < width) & (body[:, 1] >= 0) & (body[:, 1] < height)
    box = keypoint_box(points, recipe["box_margin"])
    if box is None or inside.sum() < recipe["min_body_in_view"] or share_inside(box, width, height) < recipe["min_box_in_view"]:
        return None
    return box, points


# --------------------------------------------------------------------------- plan


@dataclass
class Plan:
    """Cases (one hinted person-instant each) and the crops to read (one per case and camera)."""
    identity: np.ndarray  # (N,)
    t: np.ndarray  # (N,) sync index
    pose: np.ndarray  # (N, 17, 3) hint pose
    root: np.ndarray  # (N, 3) hint root
    kind: np.ndarray  # (N,) index into HINT_KINDS
    distance: np.ndarray  # (N,) frames to the nearest measured pose
    job_case: np.ndarray  # (J,)
    job_camera: np.ndarray  # (J,) camera index
    job_frame: np.ndarray  # (J,) native frame
    job_box: np.ndarray  # (J, 4) native pixels
    counts: dict = field(default_factory=dict)


def plan(scene: Scene, presence: Presence, recipe: dict, min_angle_deg: float,
         only: set[tuple[int, int]] | None = None) -> Plan:
    """Cases and crops for the missing instants (``only``: restrict to these (identity, t) - used when a
    later pass looks again around positions that were just recovered)."""
    cases, jobs = [], []
    counts = {"missing_instants": 0, "without_hint": 0, "seen_by_fewer_than_2_cameras": 0}
    centers = [camera.cam.center if camera.cam is not None else None for camera in scene.cameras]
    for who, t in missing_instants(presence, scene.T, int(recipe["end_frames"])):
        if only is not None and (who, t) not in only:
            continue
        counts["missing_instants"] += 1
        hint = make_hint(presence, who, t, int(recipe["reach_frames"]))
        if hint is None:
            counts["without_hint"] += 1
            continue
        views = {}
        for camera_index in range(len(scene.cameras)):
            view = view_of(scene, camera_index, t, hint.pose, recipe)
            if view is not None:
                views[camera_index] = view[0]
        cameras = sorted(views)
        directions = {c: (centers[c] - hint.root) / np.linalg.norm(centers[c] - hint.root) for c in cameras}
        wide = any(np.degrees(np.arccos(np.clip(directions[a] @ directions[b], -1, 1))) >= min_angle_deg
                   for i, a in enumerate(cameras) for b in cameras[i + 1:])
        if len(cameras) < 2 or not wide:
            counts["seen_by_fewer_than_2_cameras"] += 1
            continue
        case = len(cases)
        cases.append(hint)
        jobs += [(case, c, int(scene.native[t, c]), views[c]) for c in cameras]
    counts["cases"] = len(cases)
    return Plan(np.array([h.identity for h in cases], np.int64), np.array([h.t for h in cases], np.int64),
                np.array([h.pose for h in cases], np.float64).reshape(-1, 17, 3),
                np.array([h.root for h in cases], np.float64).reshape(-1, 3),
                np.array([h.kind for h in cases], np.int8), np.array([h.distance for h in cases], np.int32),
                np.array([j[0] for j in jobs], np.int64), np.array([j[1] for j in jobs], np.int16),
                np.array([j[2] for j in jobs], np.int64), np.array([j[3] for j in jobs], np.float32).reshape(-1, 4), counts)


# --------------------------------------------------------------------------- triangulating what was read


@dataclass
class Read:
    """A pose read from the image of one camera."""
    camera: int
    keypoints: np.ndarray  # (17, 3) native pixels + heatmap peak
    box: np.ndarray
    job: int
    rays: np.ndarray | None = None


def _rays(scene: Scene, read: Read) -> np.ndarray:
    if read.rays is None:
        read.rays = scene.cameras[read.camera].cam.rays(read.keypoints[:, :2])
    return read.rays


def _good(scene: Scene, read: Read, peak: float) -> np.ndarray:
    return (read.keypoints[:, 2] >= peak) & np.isfinite(_rays(scene, read)).all(axis=1)


def pair_support(scene: Scene, a: Read, b: Read, peak: float, gate_px: float, min_angle: float) -> tuple[int, float]:
    """(body joints the two poses agree on, their median residual)."""
    cam_a, cam_b = scene.cameras[a.camera].cam, scene.cameras[b.camera].cam
    usable = _good(scene, a, peak) & _good(scene, b, peak)
    usable[:5] = False
    if usable.sum() < 3:
        return 0, float("inf")
    _, residual, angle, front = pair_ray_evidence(cam_a.center, _rays(scene, a)[usable], cam_a.focal,
                                                  cam_b.center, _rays(scene, b)[usable], cam_b.focal)
    good = front & (residual <= gate_px) & (angle >= min_angle)
    return (int(good.sum()), float(np.median(residual[good]))) if good.any() else (0, float("inf"))


def root_of(xyz: np.ndarray) -> np.ndarray:
    if np.isfinite(xyz[[11, 12], 0]).all():
        return xyz[[11, 12]].mean(axis=0)
    torso = xyz[[11, 12, 5, 6]]
    known = np.isfinite(torso[:, 0])
    return torso[known].mean(axis=0) if known.sum() >= 2 else np.full(3, np.nan)


def triangulate(scene: Scene, reads: list[Read], geometry: dict):
    """Gated triangulation (the engine's own) of one pose per camera. Returns xyz, used (C, 17), residual (C, 17)
    indexed by scene camera index."""
    with_cam = [camera.index for camera in scene.cameras if camera.cam is not None]
    slot = {camera_index: i for i, camera_index in enumerate(with_cam)}
    cams = [scene.cameras[c].cam for c in with_cam]
    native = np.zeros((len(cams), 17, 2))
    rays = np.full((len(cams), 17, 3), np.nan)
    usable = np.zeros((len(cams), 17), bool)
    for read in reads:
        v = slot[read.camera]
        native[v], rays[v], usable[v] = read.keypoints[:, :2], _rays(scene, read), _good(scene, read, geometry["joint_peak"])
    xyz, used, residual = triangulate_views(cams, native, usable, max_reprojection_px=geometry["max_reprojection_px"],
                                            min_ray_angle_deg=geometry["min_ray_angle_deg"], rays=rays)
    full_used = np.zeros((len(scene.cameras), 17), bool)
    full_residual = np.full((len(scene.cameras), 17), np.nan)
    for camera_index, v in slot.items():
        full_used[camera_index], full_residual[camera_index] = used[v], residual[v]
    return xyz, full_used, full_residual


def consistent_set(scene: Scene, reads: list[Read], geometry: dict, recipe: dict, hint_root: np.ndarray):
    """The mutually consistent poses (one per camera) and their triangulation, or None.

    The pair agreeing on most body joints that also lands within ``search_radius_in`` of the hint seeds the
    set; other cameras join when they agree with either seed view.
    """
    peak, gate, angle = geometry["joint_peak"], geometry["max_reprojection_px"], geometry["min_ray_angle_deg"]
    best = None
    for i, a in enumerate(reads):
        for b in reads[i + 1:]:
            if a.camera == b.camera:
                continue
            joints, median = pair_support(scene, a, b, peak, gate, angle)
            if joints < recipe["min_pair_joints"]:
                continue
            root = root_of(triangulate(scene, [a, b], geometry)[0])
            if not np.isfinite(root).all() or np.linalg.norm(root[:2] - hint_root[:2]) > recipe["search_radius_in"]:
                continue
            if best is None or (joints, -median) > best[0]:
                best = ((joints, -median), a, b)
    if best is None:
        return None
    chosen = [best[1], best[2]]
    for read in reads:
        if any(read.camera == other.camera for other in chosen):
            continue
        if max(pair_support(scene, read, chosen[0], peak, gate, angle)[0],
               pair_support(scene, read, chosen[1], peak, gate, angle)[0]) >= recipe["min_pair_joints"]:
            chosen.append(read)
    xyz, used, residual = triangulate(scene, chosen, geometry)
    root = root_of(xyz)
    low, high = geometry["root_z_range_in"]
    contributing = [read for read in chosen if used[read.camera].any()]
    if not np.isfinite(root).all() or not low <= root[2] <= high or len(contributing) < 2:
        return None
    return {"xyz": xyz, "used": used, "residual": residual, "root": root, "reads": contributing}


# --------------------------------------------------------------------------- acceptance


def bone_ranges(xyz: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    """(12, 2) plausible length range of each bone, from the measured poses of this recording."""
    complete = np.isfinite(xyz[:, 5:17, 0]).all(axis=1)
    sample = xyz[np.flatnonzero(complete)[::10]].astype(np.float64)
    if len(sample) < 50:
        return np.tile(np.array([0.0, np.inf]), (len(BONES), 1))
    return np.array([np.percentile(np.linalg.norm(sample[:, a] - sample[:, b], axis=1), [low, high]) for a, b in BONES])


def failed_gates(features: dict, recipe: dict, name: str) -> list[str]:
    """Which conditions of tier ``name`` a recovery does not meet (empty: it would be accepted)."""
    gate = recipe["tiers"][name]
    checks = [("cameras", features["cameras"] >= gate["min_cameras"]),
              ("body joints", features["body_joints"] >= gate["min_body_joints"]),
              ("residual", features["median_residual_px"] <= gate["max_median_residual_px"]),
              ("bone lengths", features["bad_bones"] <= gate["max_bad_bones"]),
              ("distance from the hint", features["hint_distance_in"] <= gate["max_hint_distance_in"]),
              ("image confidence", features["weakest_view_peak"] >= gate["min_view_peak"]),
              ("another identity is measured there", features["other_identity_in"] >= recipe["other_identity_clear_in"])]
    if name == "B":
        checks.append(("not bracketed by nearby poses", features["hint_kind"] == 0 and features["hint_frames"] <= gate["max_hint_frames"]))
    return [label for label, passed in checks if not passed]


def tier(features: dict, recipe: dict) -> str | None:
    """Name of the tier that accepts a recovery ("A": three or more cameras, "B": two cameras), else None."""
    for name in ("A", "B"):
        gate = recipe["tiers"][name]
        if (features["cameras"] >= gate["min_cameras"] and features["body_joints"] >= gate["min_body_joints"]
                and features["median_residual_px"] <= gate["max_median_residual_px"]
                and features["bad_bones"] <= gate["max_bad_bones"] and features["hint_distance_in"] <= gate["max_hint_distance_in"]
                and features["weakest_view_peak"] >= gate["min_view_peak"]
                and features["other_identity_in"] >= recipe["other_identity_clear_in"]
                and (name == "A" or (features["hint_kind"] == 0 and features["hint_frames"] <= gate["max_hint_frames"]))):
            return name
    return None


@dataclass
class Recovered:
    case: list = field(default_factory=list)
    xyz: list = field(default_factory=list)
    used: list = field(default_factory=list)  # (C, 17) bool
    residual: list = field(default_factory=list)  # (C, 17)
    root: list = field(default_factory=list)
    tier: list = field(default_factory=list)
    jobs: list = field(default_factory=list)  # job index per camera (-1 = camera not used)
    reasons: dict = field(default_factory=dict)
    # Every case that was not accepted: {"case", "reason", and for gate rejections "features", "failed", "root", "xyz"}
    rejected: list = field(default_factory=list)


def accept(scene: Scene, presence: Presence, tr_identity: np.ndarray, tr_t: np.ndarray, the_plan: Plan,
           keypoints: np.ndarray, boxes: np.ndarray, geometry: dict, recipe: dict,
           ranges: np.ndarray | None = None) -> Recovered:
    """Triangulate and gate every case. ``keypoints`` (J, 17, 3) and ``boxes`` (J, 4) are what the pose model
    read for each job (NaN where it did not run). ``ranges``: plausible bone lengths (default: from ``presence``)."""
    measured_at: dict[int, list[int]] = {}
    for row in np.flatnonzero((tr_identity >= 0) & np.isfinite(presence.root).all(axis=1)):
        measured_at.setdefault(int(tr_t[row]), []).append(int(row))
    ranges = bone_ranges(presence.xyz) if ranges is None else ranges
    jobs_of: dict[int, list[int]] = {}
    for job, case in enumerate(the_plan.job_case):
        jobs_of.setdefault(int(case), []).append(job)
    out = Recovered()
    reasons = {"no pose read in two cameras": 0, "crops follow another person": 0, "no consistent pose in two cameras": 0,
               "rejected by the acceptance gate": 0, "accepted": 0}
    for case in range(len(the_plan.t)):
        who, t = int(the_plan.identity[case]), int(the_plan.t[case])
        others = [row for row in measured_at.get(t, []) if int(tr_identity[row]) != who]
        reads, dropped = [], 0
        for job in jobs_of.get(case, []):
            pose = keypoints[job].astype(np.float64)
            if not np.isfinite(pose).all():
                continue
            camera = scene.cameras[int(the_plan.job_camera[job])]
            hint_box = the_plan.job_box[job].astype(np.float64)
            follows_hint = keypoint_agreement(camera.cam, project(camera.cam, the_plan.pose[case]), pose, hint_box,
                                              geometry["joint_peak"])
            # Competitor rule: another measured person overlaps the hint box and the pose follows them at least as well.
            overlap, follows_other = 0.0, 0.0
            for row in others:
                other_points = project(camera.cam, presence.xyz[row].astype(np.float64))
                other_box = keypoint_box(other_points, recipe["box_margin"])
                if other_box is None:
                    continue
                iou = box_iou(hint_box, other_box)
                overlap = max(overlap, iou)
                if iou > 0.1:
                    follows_other = max(follows_other, keypoint_agreement(camera.cam, other_points, pose, hint_box,
                                                                          geometry["joint_peak"]))
            if overlap >= recipe["competitor_iou"] and follows_other >= follows_hint:
                dropped += 1
                continue
            reads.append(Read(camera.index, pose, boxes[job].astype(np.float64), job))
        if len({read.camera for read in reads}) < 2:
            reason = "crops follow another person" if dropped else "no pose read in two cameras"
            reasons[reason] += 1
            out.rejected.append({"case": case, "reason": reason})
            continue
        found = consistent_set(scene, reads, geometry, recipe, the_plan.root[case])
        if found is None:
            reasons["no consistent pose in two cameras"] += 1
            out.rejected.append({"case": case, "reason": "no consistent pose in two cameras"})
            continue
        xyz, root = found["xyz"], found["root"]
        worst = np.nanmax(np.where(found["used"], found["residual"], np.nan), axis=0) if found["used"].any() else np.zeros(17)
        bad = sum(1 for (a, b), (low, high) in zip(BONES, ranges)
                  if np.isfinite(xyz[a, 0]) and np.isfinite(xyz[b, 0]) and not low <= np.linalg.norm(xyz[a] - xyz[b]) <= high)
        other_distance = min((float(np.linalg.norm(presence.root[row][:2] - root[:2])) for row in others), default=np.inf)
        features = {"cameras": len(found["reads"]), "body_joints": int(np.isfinite(xyz[5:17, 0]).sum()),
                    "median_residual_px": float(np.nanmedian(worst)) if np.isfinite(worst).any() else np.inf,
                    "bad_bones": bad, "hint_distance_in": float(np.linalg.norm(root[:2] - the_plan.root[case][:2])),
                    "weakest_view_peak": float(min(read.keypoints[5:17, 2].mean() for read in found["reads"])),
                    "other_identity_in": other_distance, "hint_kind": int(the_plan.kind[case]),
                    "hint_frames": int(the_plan.distance[case])}
        accepted = tier(features, recipe)
        if accepted is None:
            reasons["rejected by the acceptance gate"] += 1
            name = "A" if features["cameras"] >= recipe["tiers"]["A"]["min_cameras"] else "B"
            out.rejected.append({"case": case, "reason": "rejected by the acceptance gate", "features": features,
                                 "failed": failed_gates(features, recipe, name), "tier_tried": name,
                                 "root": root, "xyz": xyz})
            continue
        reasons["accepted"] += 1
        per_camera = np.full(len(scene.cameras), -1, np.int64)
        for read in found["reads"]:
            per_camera[read.camera] = read.job
        out.case.append(case)
        out.xyz.append(xyz)
        out.used.append(found["used"])
        out.residual.append(found["residual"])
        out.root.append(root)
        out.tier.append(accepted)
        out.jobs.append(per_camera)
    out.reasons = reasons
    return out


# --------------------------------------------------------------------------- recursion from the edges


def recover_iteratively(scene: Scene, identity: np.ndarray, t: np.ndarray, root: np.ndarray, xyz: np.ndarray,
                        read, geometry: dict, recipe: dict, log=print) -> dict:
    """Recover positions pass after pass, each pass looking around what the previous one found.

    A pass can only look where a known pose is at most ``reach_frames`` away. Every accepted recovery is
    itself a measured pose, so the next pass uses it as the hint for its neighbours: gaps are closed from
    both edges inwards, and tracks are followed beyond the instants where they first appear and last
    disappear, for as long as poses keep triangulating and passing the acceptance gate. It stops when a
    pass accepts nothing (or after ``max_passes``).

    ``read(plan) -> (keypoints (J, 17, 3), boxes (J, 4))`` reads the crops of a plan.
    Returns the recovered person-frames (arrays), the per-pass log and, for every instant that was looked
    at and is still missing, why (``rejected``: {(identity, t): record}).
    """
    measured = (identity >= 0) & np.isfinite(root).all(axis=1)
    known_identity, known_t = identity[measured].astype(np.int64), t[measured].astype(np.int64)
    known_root, known_xyz = root[measured].astype(np.float64), xyz[measured].astype(np.float64)
    ranges = bone_ranges(known_xyz)  # from measured poses only: recoveries must not widen their own gate
    cameras = len(scene.cameras)
    out = {name: [] for name in ("identity", "t", "xyz", "used", "residual", "root", "tier", "pass", "hint_kind",
                                 "hint_frames", "pose_frame", "pose_keypoints", "pose_box")}
    rejected: dict[tuple[int, int], dict] = {}
    passes, only, first_counts, exhausted = [], None, None, False
    reach = int(recipe["reach_frames"])
    for number in range(int(recipe["max_passes"])):
        presence = build_presence(known_identity, known_t, known_root, known_xyz)
        the_plan = plan(scene, presence, recipe, geometry["min_ray_angle_deg"], only=only)
        if first_counts is None:
            first_counts = dict(the_plan.counts)
        if not len(the_plan.t):
            exhausted = True
            break
        keypoints, boxes = read(the_plan)
        result = accept(scene, presence, known_identity, known_t, the_plan, keypoints, boxes, geometry, recipe, ranges)
        for record in result.rejected:
            case = record["case"]
            key = (int(the_plan.identity[case]), int(the_plan.t[case]))
            rejected[key] = {**{k: v for k, v in record.items() if k != "case"}, "pass": number,
                             "hint_kind": int(the_plan.kind[case]), "hint_frames": int(the_plan.distance[case])}
        cases = np.array(result.case, np.int64)
        passes.append({"pass": number, "cases": int(len(the_plan.t)), "crops": int(len(the_plan.job_case)),
                       "accepted": int(len(cases)), "outcome": dict(result.reasons)})
        log(f"pass {number + 1}: {len(the_plan.t)} instants looked at, {len(cases)} positions recovered")
        if not len(cases):
            exhausted = True
            break
        used = np.array(result.used, bool).reshape(len(cases), cameras, 17)
        residual = np.array(result.residual, np.float64).reshape(len(cases), cameras, 17)
        jobs = np.array(result.jobs, np.int64).reshape(len(cases), cameras)
        has = jobs >= 0
        safe = np.clip(jobs, 0, None)
        out["identity"].append(the_plan.identity[cases])
        out["t"].append(the_plan.t[cases])
        out["xyz"].append(np.array(result.xyz, np.float64).reshape(len(cases), 17, 3))
        out["used"].append(used)
        out["residual"].append(residual)
        out["root"].append(np.array(result.root, np.float64).reshape(len(cases), 3))
        out["tier"].append(np.array(result.tier, dtype="U1"))
        out["pass"].append(np.full(len(cases), number, np.int16))
        out["hint_kind"].append(the_plan.kind[cases])
        out["hint_frames"].append(the_plan.distance[cases])
        out["pose_frame"].append(np.where(has, the_plan.job_frame[safe], -1))
        out["pose_keypoints"].append(np.where(has[:, :, None, None], keypoints[safe], np.nan))
        out["pose_box"].append(np.where(has[:, :, None], boxes[safe], np.nan))
        known_identity = np.concatenate([known_identity, the_plan.identity[cases]])
        known_t = np.concatenate([known_t, the_plan.t[cases]])
        known_root = np.concatenate([known_root, out["root"][-1]])
        known_xyz = np.concatenate([known_xyz, out["xyz"][-1]])
        only = set()
        for who, when in zip(the_plan.identity[cases], the_plan.t[cases]):
            rejected.pop((int(who), int(when)), None)
            only.update((int(who), int(when) + delta) for delta in range(-reach, reach + 1) if delta)
    empty = {"identity": np.zeros(0, np.int64), "t": np.zeros(0, np.int64), "xyz": np.zeros((0, 17, 3)),
             "used": np.zeros((0, cameras, 17), bool), "residual": np.zeros((0, cameras, 17)), "root": np.zeros((0, 3)),
             "tier": np.zeros(0, "U1"), "pass": np.zeros(0, np.int16), "hint_kind": np.zeros(0, np.int8),
             "hint_frames": np.zeros(0, np.int32), "pose_frame": np.zeros((0, cameras), np.int64),
             "pose_keypoints": np.zeros((0, cameras, 17, 3)), "pose_box": np.zeros((0, cameras, 4))}
    arrays = {name: (np.concatenate(parts) if parts else empty[name]) for name, parts in out.items()}
    final = build_presence(known_identity, known_t, known_root, known_xyz)
    return {"arrays": arrays, "passes": passes, "rejected": rejected, "first_plan": first_counts or {},
            "stopped_by_pass_limit": not exhausted,
            "inside_missing_before": inside_missing(build_presence(identity[measured], t[measured], root[measured], xyz[measured])),
            "inside_missing_after": inside_missing(final), "positions_before": int(measured.sum())}


def inside_missing(presence: Presence) -> int:
    """Instants without a position between each identity's first and last known position."""
    return int(sum(int(times[-1]) - int(times[0]) + 1 - len(times) for times in presence.t.values()))
