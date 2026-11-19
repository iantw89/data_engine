"""Golden person-instants that only ONE golden-labelled camera shows: could a second camera see them?

For each such instant the person's floor position is estimated from the single view (the ankle /
box-bottom ray meets the floor; hips otherwise), a template standing pose is placed there and
projected into the other cameras. We count: inside calibrated space (>= 1 other camera sees the
spot with >= 8 degrees of ray angle), whether that camera has a raw detection near it (any score),
and whether the engine already has a 3D position for the instant.

The template is only a SEARCH HINT used to count candidates; nothing here is an output position.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

from candidates import hint_view
from common import *  # noqa
from data_engine.config import SITE_DEFAULTS
from data_engine.evaluation import box_iou_matrix
from hints import box_iou, build_presence, pair_angle_deg
from recover_lib import root_of


def golden_views(scene, dataset, tag="cvat_identity"):
    data = Path(SITE_DEFAULTS["repos"]["data_management"]) / "data" / "labels" / "data"
    owner = defaultdict(list)
    reference = json.loads((data / f"{dataset}.three_d_tracks.{tag}" / "tracks.json").read_text(encoding="utf-8"))
    for track in reference["tracks"]:
        for member in track["members"]:
            owner[(member["camera"], int(member["tracklet_id"]))].append((*member["frame_range"], int(track["track_id"])))
    views = defaultdict(list)  # (person, t) -> [(camera, state, row, gbox)]
    for camera in scene.cameras:
        path = data / f"{dataset}.two_d_tracklets.{tag}" / "cameras" / f"{camera.camera}.json"
        sync_of_frame = np.full(camera.detections.frame_count, -1, np.int64)
        shown = np.flatnonzero(scene.native[:, camera.index] >= 0)
        sync_of_frame[scene.native[shown, camera.index][::-1]] = shown[::-1]
        for entry in json.loads(path.read_text(encoding="utf-8"))["frames"]:
            frame = int(entry["frame"])
            if frame >= camera.detections.frame_count or sync_of_frame[frame] < 0:
                continue
            boxes, people = [], []
            for detection in entry["detections"]:
                who = [p for first, last, p in owner.get((camera.camera, int(detection["tracklet_id"])), ()) if first <= frame <= last]
                if len(who) == 1 and detection.get("bbox"):
                    boxes.append(detection["bbox"])
                    people.append(who[0])
            if not boxes:
                continue
            rows = camera.detections.rows(frame)
            overlap = (box_iou_matrix(np.asarray(boxes, float), camera.detections.boxes[rows, :4].astype(float))
                       if rows.stop > rows.start else np.zeros((len(boxes), 0)))
            for index, who in enumerate(people):
                state, row = "no_detection", -1
                candidates = np.flatnonzero(overlap[index] >= 0.5) if overlap.shape[1] else []
                if len(candidates):
                    kept = [c for c in candidates if camera.kept[rows.start + c]]
                    best = max(kept or list(candidates), key=lambda c: overlap[index, c])
                    row = rows.start + int(best)
                    state = "kept" if camera.kept[row] else "filtered"
                views[(who, int(sync_of_frame[frame]))].append((camera.index, state, row, boxes[index]))
    return views


def ray_plane(origin, direction, z):
    if abs(direction[2]) < 1e-6:
        return None
    s = (z - origin[2]) / direction[2]
    return None if s <= 0 else origin + s * direction


def main():
    ws, scene, tracks, solution = load()
    views = golden_views(scene, DATASET)
    ok = np.isfinite(tracks["root"]).all(axis=1)
    in3d = [np.zeros(len(c.kept), bool) for c in scene.cameras]
    for c in range(len(scene.cameras)):
        rr = tracks["rows"][ok, c]
        in3d[c][rr[rr >= 0]] = True
    full = np.flatnonzero(np.isfinite(tracks["xyz"][:, 5:17, 0]).all(axis=1) & ok)
    template = np.nanmedian(tracks["xyz"][full] - tracks["root"][full][:, None, :], axis=0)  # (17,3) relative to root
    centers = [c.cam.center for c in scene.cameras]
    count = Counter()
    detail = Counter()
    ident = {c.camera: solution[f"identity_{c.camera}"] for c in scene.cameras}
    for (who, t), seen in views.items():
        if len({s[0] for s in seen}) != 1:
            continue
        c1, state, row, gbox = seen[0]
        count["one_golden_view"] += 1
        if row >= 0 and in3d[c1][row]:
            count["engine_has_3d"] += 1
            continue
        camera = scene.cameras[c1]
        # floor / root estimate from this single view
        P = None
        if row >= 0:
            kp = camera.detections.keypoints[row]
            ankles = [j for j in (15, 16) if kp[j, 2] >= 0.3]
            hips = [j for j in (11, 12) if kp[j, 2] >= 0.3]
            if len(hips) == 2:
                r = camera.cam.rays(kp[hips, :2]).mean(axis=0)
                P = ray_plane(camera.cam.center, r / np.linalg.norm(r), 36.0)
            if P is None and ankles:
                r = camera.cam.rays(kp[ankles, :2]).mean(axis=0)
                foot = ray_plane(camera.cam.center, r / np.linalg.norm(r), 0.0)
                P = None if foot is None else foot + np.array([0, 0, 36.0])
        if P is None:
            foot_px = np.array([(gbox[0] + gbox[2]) / 2, gbox[3]])
            r = camera.cam.rays(foot_px[None])[0]
            foot = ray_plane(camera.cam.center, r, 0.0) if np.isfinite(r).all() else None
            P = None if foot is None else foot + np.array([0, 0, 36.0])
        if P is None:
            count["no_floor_estimate"] += 1
            continue
        pose = template + P
        visible = {}
        for c2 in range(len(scene.cameras)):
            if c2 == c1:
                continue
            v = hint_view(scene, c2, t, pose, min_visible=0.5)
            if v is not None:
                visible[c2] = v
        angle = pair_angle_deg(P, centers)
        second = [c2 for c2 in visible if angle[c1, c2] >= 8.0]
        if not second:
            count["no_second_camera_can_see (outside calibrated space / out of view)"] += 1
            continue
        count["second_camera_can_see"] += 1
        # detection near the template projection in any such camera?
        found = []
        for c2 in second:
            box = visible[c2][0]
            frame = int(scene.native[t, c2])
            rows = scene.cameras[c2].detections.rows(frame)
            if rows.stop > rows.start:
                iou = box_iou(box, scene.cameras[c2].detections.boxes[rows, :4].astype(float))
                if iou.max() >= 0.25:
                    k = rows.start + int(iou.argmax())
                    found.append((c2, "kept" if scene.cameras[c2].kept[k] else "filtered", int(ident[scene.cameras[c2].camera][k])))
        if not found:
            count["visible_second_camera_no_detection"] += 1
        else:
            count["visible_second_camera_has_detection"] += 1
            kinds = Counter(("assigned_other" if f[2] >= 0 else f[1]) for f in found)
            detail[json.dumps(sorted(kinds))] += 1
    out = {"counts": dict(count), "second_camera_detection_kinds": dict(detail)}
    print(json.dumps(out, indent=1))
    (SCRATCH / "golden_single_view.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
