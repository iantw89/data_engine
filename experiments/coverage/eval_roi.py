"""Evaluate lever 3 (detector on a zoomed ROI crop, ViTPose on its boxes) on the cases it was run for.

Usage: python experiments/coverage/eval_roi.py {yolov8x|yolo11l}
Writes eval_<detector>.pkl with keys R0/roi (detector candidates only), R1/roi (competitor rule),
R0/vit+roi (lever-2 crop poses plus detector poses, one per camera consistent with the seed pair).
"""
import pickle
import sys
import time
from collections import defaultdict

from candidates import NEAR_AGREE, NEAR_IOU
from common import *  # noqa
from eval_levers import HINT_RADIUS, competitor_info
from hints import box_iou, keypoint_agreement, project_hint
from recover_lib import Cand, recover


def main():
    name = sys.argv[1]
    started = time.time()
    ws, scene, tracks, solution = load()
    cases = pickle.loads((SCRATCH / "cases.pkl").read_bytes())
    jobs = np.load(SCRATCH / "jobs.npz")
    roi = np.load(SCRATCH / f"pass_{name}.npz")
    vit = np.load(SCRATCH / "pass_vit.npz")
    by_t = defaultdict(list)
    for row in np.flatnonzero(np.isfinite(tracks["root"]).all(axis=1)):
        by_t[int(tracks["t"][row])].append(row)
    offsets = np.concatenate(([0], np.cumsum(roi["counts"])))
    sel = roi["sel"]
    job_of = defaultdict(dict)
    for k, j in enumerate(sel):
        job_of[int(jobs["case"][j])][int(jobs["cam"][j])] = (k, int(j))
    out = []
    for n, case in enumerate(cases):
        if case["id"] not in job_of:
            continue
        if n % 2000 == 0:
            print(n, round(time.time() - started), flush=True)
        roi_c, vit_c = {}, {}
        for c, (k, j) in job_of[case["id"]].items():
            cam = scene.cameras[c].cam
            xy = project_hint(cam, case["pose"])
            box = case["views"][c]
            lo, hi = offsets[k], offsets[k + 1]
            best = None
            if hi > lo:
                boxes, kps = roi["boxes"][lo:hi], roi["keypoints"][lo:hi]
                iou = box_iou(box, boxes[:, :4].astype(np.float64))
                for m in range(hi - lo):
                    agree, common = keypoint_agreement(cam, xy, kps[m], boxes[m])
                    share = agree / common if common >= 5 else 0.0
                    if iou[m] >= NEAR_IOU or share >= NEAR_AGREE:
                        if best is None or share + iou[m] > best[0]:
                            best = (share + iou[m], m)
            if best is not None:
                m = best[1]
                cand = Cand(c, kps[m].astype(np.float64), boxes[m, :4].astype(np.float64), f"{name}_roi", -1)
                cand.meta["comp_iou"], cand.meta["agree_hint"], cand.meta["agree_comp"] = competitor_info(scene, tracks, by_t, case, c, cand.kp)
                roi_c[c] = cand
            kpv = vit["kp2"][j].astype(np.float64)
            if np.isfinite(kpv).all():
                cand = Cand(c, kpv, vit["box2"][j].astype(np.float64), "vitpose_crop", -1)
                cand.meta["comp_iou"], cand.meta["agree_hint"], cand.meta["agree_comp"] = competitor_info(scene, tracks, by_t, case, c, kpv)
                vit_c[c] = cand
        record = {"id": case["id"], "tag": case["tag"], "n_views": len(case["views"]), "n_roi_dets": len(roi_c)}
        rules = {"R0": lambda cd: True,
                 "R1": lambda cd: not (cd.meta["comp_iou"] >= 0.4 and cd.meta["agree_comp"] >= cd.meta["agree_hint"])}
        for rule, allow in rules.items():
            for variant in ("roi", "vit+roi"):
                cands = [cd for cd in roi_c.values() if allow(cd)]
                if variant == "vit+roi":
                    cands += [cd for cd in vit_c.values() if allow(cd)]
                rec = recover(scene, cands, RECIPE, hint_root=case["root"], hint_radius_in=HINT_RADIUS) if len({x.camera for x in cands}) >= 2 else None
                if rec is None:
                    record[f"{rule}/{variant}"] = None
                    continue
                record[f"{rule}/{variant}"] = {
                    "root": rec.root, "xyz": rec.xyz, "n_body": rec.n_body, "median_residual": rec.median_residual,
                    "pair_support": rec.pair_support, "cams": rec.cams, "sources": [c.source for c in rec.cands],
                    "hint_dist": float(np.linalg.norm(rec.root[:2] - case["root"][:2])),
                    "mean_peak": float(np.mean([c.kp[5:17, 2].mean() for c in rec.cands])),
                    "min_peak": float(np.min([c.kp[5:17, 2].mean() for c in rec.cands])),
                    "comp_iou": max([c.meta.get("comp_iou", 0.0) for c in rec.cands]),
                    "agree_hint": [c.meta.get("agree_hint", np.nan) for c in rec.cands],
                    "agree_comp": [c.meta.get("agree_comp", np.nan) for c in rec.cands],
                    "rows": [(c.camera, c.row) for c in rec.cands]}
        out.append(record)
    (SCRATCH / f"eval_{name}.pkl").write_bytes(pickle.dumps(out))
    print("done", round(time.time() - started))


if __name__ == "__main__":
    main()
