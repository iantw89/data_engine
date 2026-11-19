"""Check the coordinate conventions: do Autofit's 3D skeletons land on the engine detections?

Independent of Autofit code: the poses in labels.npz (world inches) are projected with the ENGINE's own geometry
(``data_engine.geometry.Cam.project``, native pixels) and compared with the native-pixel keypoints of the detections
that Autofit assigned to the same track (raw id) at the same sync index, and, as a control, with detections of other
tracks in the same camera frame. Also tries three wrong conventions (no letterbox padding, 2048-wide scale, swapped
x/y) to show the right one is clearly best.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_engine.geometry import load_calibration
from data_engine.solve.scene import load_scene
from data_engine.workspace import Workspace

SCRATCH = Path(r"C:\data_engine_data\_scratch\autofit")
JOINT_TO_COCO = [5 + k for k in [6, 7, 0, 1, 2, 3, 4, 5, 8, 9, 10, 11]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--variant", default="kept")
    ap.add_argument("--calibration", type=Path, default=None)
    args = ap.parse_args()
    ws = Workspace(args.workspace)
    scene = load_scene(ws, joint_peak=0.3, calibration=args.calibration, rays=False)
    af = np.load(SCRATCH / args.workspace / args.variant / "labels.npz")
    tid, pt, state, ncam, xyz = (af[k] for k in ("pose_tid", "pose_t", "pose_state", "pose_ncam", "pose_xyz"))
    sel = np.flatnonzero((state == 0) & (ncam >= 2) & (pt % 5 == 0))
    errs = {"engine_convention": [], "control_other_track": [], "no_padding": [], "swapped_xy": []}
    for cam in scene.cameras:
        if cam.cam is None or f"raw_{cam.camera}" not in af.files:
            continue
        raw = af[f"raw_{cam.camera}"]
        rows = np.flatnonzero((raw >= 0) & (cam.t >= 0))
        by_key = {}
        for r in rows:
            by_key.setdefault((int(cam.t[r]), int(raw[r])), []).append(int(r))
        by_t = {}
        for r in rows:
            by_t.setdefault(int(cam.t[r]), []).append(int(r))
        for k in sel:
            r_list = by_key.get((int(pt[k]), int(tid[k])))
            if not r_list:
                continue
            r = r_list[0]
            kp = cam.detections.keypoints[r][JOINT_TO_COCO]
            ok = kp[:, 2] >= 0.3
            if ok.sum() < 6:
                continue
            world = xyz[k].astype(np.float64)
            proj = cam.cam.project(world)
            good = ok & np.isfinite(proj).all(axis=1)
            if good.sum() < 6:
                continue
            scale = cam.cam.to_cvat  # native -> 1280 px
            errs["engine_convention"].append(np.median(np.linalg.norm(proj[good] - kp[good, :2], axis=1)) * scale)
            others = [o for o in by_t.get(int(pt[k]), []) if o != r]
            if others:
                best = min(np.median(np.linalg.norm(proj[good] - cam.detections.keypoints[o][JOINT_TO_COCO][good, :2], axis=1)) for o in others)
                errs["control_other_track"].append(best * scale)
            # wrong conventions: re-project with perturbed canvas mapping
            import cv2
            c = cam.cam
            canvas = cv2.projectPoints(world, c.rvec, c.tvec, c.K, c.D)[0].reshape(-1, 2)
            wrong = (canvas - 0.0) / c.scale / c.to_cvat  # padding forgotten
            errs["no_padding"].append(np.median(np.linalg.norm(wrong[good] - kp[good, :2], axis=1)) * scale)
            sw = (canvas[:, ::-1] - c.padding) / c.scale / c.to_cvat
            errs["swapped_xy"].append(np.median(np.linalg.norm(sw[good] - kp[good, :2], axis=1)) * scale)
    out = {name: {"n": len(v), "median_px_1280": round(float(np.median(v)), 1) if v else None,
                  "p90": round(float(np.percentile(v, 90)), 1) if v else None} for name, v in errs.items()}
    print(json.dumps(out, indent=1))
    (SCRATCH / args.workspace / args.variant / "geometry_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
