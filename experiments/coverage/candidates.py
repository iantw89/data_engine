"""Raw detections near a hint, as candidates for recovery (lever 1)."""
from __future__ import annotations

from common import *  # noqa: F401,F403
from hints import MARGIN, box_iou, clip_share, keypoint_agreement, keypoint_box, project_hint
from recover_lib import Cand

NEAR_IOU = 0.3
NEAR_AGREE = 0.6


def hint_view(scene, c: int, t: int, pose: np.ndarray, min_visible: float = 0.6, min_body: int = 6):
    """(box, xy) of the hinted skeleton in camera c at sync index t, or None if not visible there."""
    camera = scene.cameras[c]
    if camera.cam is None or scene.native[t, c] < 0:
        return None
    xy = project_hint(camera.cam, pose)
    width, height = camera.cam.native_width, camera.cam.native_width * 0.75
    body = xy[5:17]
    body_ok = np.isfinite(body).all(axis=1) & (body[:, 0] >= 0) & (body[:, 0] < width) & (body[:, 1] >= 0) & (body[:, 1] < height)
    box = keypoint_box(xy, width, height)
    if box is None or body_ok.sum() < min_body or clip_share(box, width, height) < min_visible:
        return None
    return box, xy


def near_candidates(scene, identity_arrays: dict, c: int, t: int, box, xy, who: int, *, max_n: int = 3,
                    kinds=("own", "unassigned", "filtered"), hide_rows: set | None = None) -> list[Cand]:
    """Raw rows of camera c at t that match the hinted skeleton, best first.

    kind: own (assigned to ``who``), unassigned (kept, no identity), filtered (post-filter rejected),
    other (assigned to someone else). ``hide_rows`` rows are reported as unassigned (withhold test).
    """
    camera = scene.cameras[c]
    frame = int(scene.native[t, c])
    rows = camera.detections.rows(frame)
    if rows.stop <= rows.start:
        return []
    rr = np.arange(rows.start, rows.stop)
    iou = box_iou(box, camera.detections.boxes[rr, :4].astype(np.float64))
    ident = identity_arrays[camera.camera]
    out = []
    for k, row in enumerate(rr):
        agree, common = keypoint_agreement(camera.cam, xy, camera.detections.keypoints[row], camera.detections.boxes[row])
        share = agree / common if common >= 5 else 0.0
        if iou[k] < NEAR_IOU and share < NEAR_AGREE:
            continue
        owner = -1 if (hide_rows and (c, int(row)) in hide_rows) else int(ident[row])
        if owner == who:
            kind = "own"
        elif owner >= 0:
            kind = "other"
        elif camera.kept[row]:
            kind = "unassigned"
        else:
            kind = "filtered"
        if kind not in kinds:
            continue
        out.append(Cand(c, camera.detections.keypoints[row].astype(np.float64), camera.detections.boxes[row, :4].astype(np.float64),
                        f"raw_{kind}", int(row), float(share + iou[k]),
                        {"kind": kind, "iou": float(iou[k]), "share": float(share), "owner": owner,
                         "person_score": float(camera.detections.boxes[row, 4])}))
    out.sort(key=lambda x: -x.score)
    return out[:max_n]
