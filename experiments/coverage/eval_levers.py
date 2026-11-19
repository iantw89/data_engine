"""Evaluate lever 2 (guided ViTPose crops) and lever 1+2 on every planned case.

Reads cases.pkl, jobs.npz, pass_vit.npz (+ optional pass_*.npz for other detectors).
Writes eval_<name>.pkl: per case and per variant the loosest-gate recovery with the attributes
needed to sweep gates offline (n_body, residual, hint distance, peaks, competitor overlap ...).

Usage: python experiments/coverage/eval_levers.py [pass_file_stem] [output_name]
"""
import os
import pickle
import sys
import time
from collections import defaultdict

from candidates import near_candidates
from common import *  # noqa
from hints import box_iou, keypoint_agreement, keypoint_box, project_hint
from recover_lib import Cand, recover

HINT_RADIUS = 100.0  # loose; stricter radii are applied when sweeping


def competitor_info(scene, tracks, by_t, case, c, cand_kp):
    """Overlap of the hint box with other measured people and which skeleton the crop pose follows."""
    cam = scene.cameras[c].cam
    box = case["views"][c]
    hint_xy = project_hint(cam, case["pose"])
    ah, nh = keypoint_agreement(cam, hint_xy, cand_kp, box)
    best_iou, best_agree = 0.0, 0.0
    for r in by_t.get(case["t"], []):
        if int(tracks["identity"][r]) == case["who"]:
            continue
        xy = project_hint(cam, tracks["xyz"][r].astype(np.float64))
        ob = keypoint_box(xy, 0, 0)
        if ob is None:
            continue
        iou = float(box_iou(box, ob[None])[0])
        if iou > best_iou:
            best_iou = iou
        if iou > 0.1:
            a, n = keypoint_agreement(cam, xy, cand_kp, box)
            best_agree = max(best_agree, a / max(n, 1))
    return best_iou, ah / max(nh, 1), best_agree


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else "pass_vit"
    name = sys.argv[2] if len(sys.argv) > 2 else "vit"
    iteration = os.environ.get("ITER", "kp2")
    only_tags = os.environ.get("TAGS")
    started = time.time()
    ws, scene, tracks, solution = load()
    cases = pickle.loads((SCRATCH / "cases.pkl").read_bytes())
    jobs = np.load(SCRATCH / "jobs.npz")
    res = np.load(SCRATCH / f"{stem}.npz")
    ident = {c.camera: solution[f"identity_{c.camera}"] for c in scene.cameras}
    by_t = defaultdict(list)
    for row in np.flatnonzero(np.isfinite(tracks["root"]).all(axis=1)):
        by_t[int(tracks["t"][row])].append(row)
    hide = set()
    for case in cases:
        if case["tag"] == "withhold":
            for c, r in enumerate(case["truth_rows"]):
                if r >= 0:
                    hide.add((c, int(r)))
    job_of = defaultdict(dict)
    for j, (c, case) in enumerate(zip(jobs["cam"], jobs["case"])):
        job_of[int(case)][int(c)] = j
    out = []
    for n, case in enumerate(cases):
        if only_tags and case["tag"] not in only_tags.split(","):
            continue
        if n % 1500 == 0:
            print(n, len(cases), round(time.time() - started), flush=True)
        crops = {}
        for c, j in job_of[case["id"]].items():
            kp = res[iteration][j].astype(np.float64)
            if not np.isfinite(kp).all():
                continue
            cand = Cand(c, kp, res["box2"][j].astype(np.float64), "vitpose_crop", -1)
            cand.meta["comp_iou"], cand.meta["agree_hint"], cand.meta["agree_comp"] = competitor_info(scene, tracks, by_t, case, c, kp)
            crops[c] = cand
        raws = {}
        for c in case["views"]:
            xy = project_hint(scene.cameras[c].cam, case["pose"])
            found = near_candidates(scene, ident, c, case["t"], case["views"][c], xy, case["who"], max_n=2,
                                    kinds=("own", "unassigned", "filtered"), hide_rows=None)
            found = [f for f in found if (c, f.row) not in hide]
            if found:
                raws[c] = found
        record = {"id": case["id"], "tag": case["tag"], "n_views": len(case["views"]), "n_crops": len(crops), "n_raw_cams": len(raws)}
        rules = {"R0": lambda cd: True,
                 "R1": lambda cd: not (cd.meta["comp_iou"] >= 0.4 and cd.meta["agree_comp"] >= cd.meta["agree_hint"]),
                 "R2": lambda cd: cd.meta["comp_iou"] < 0.3}
        for rule, allow in rules.items():
            if case["tag"] not in ("gap", "withhold", "neg_shift", "neg_absent", "neg_empty"):
                continue
            for variant in ("crop", "raw+crop"):
                cands = []
                for c in case["views"]:
                    if variant == "raw+crop" and c in raws:
                        cands += raws[c]
                    elif c in crops and allow(crops[c]):
                        cands.append(crops[c])
                rec = recover(scene, cands, RECIPE, hint_root=case["root"], hint_radius_in=HINT_RADIUS) if len({x.camera for x in cands}) >= 2 else None
                key = f"{rule}/{variant}"
                if rec is None:
                    record[key] = None
                    continue
                used_peaks = np.array([np.mean(c.kp[rec.used[[i for i in range(len(scene.cameras))][0]].any() * 0 + 0:0 + 17, 2]) for c in rec.cands]) if False else None
                record[key] = {
                    "root": rec.root, "xyz": rec.xyz, "n_body": rec.n_body, "median_residual": rec.median_residual,
                    "pair_support": rec.pair_support, "cams": rec.cams, "sources": [c.source for c in rec.cands],
                    "hint_dist": float(np.linalg.norm(rec.root[:2] - case["root"][:2])),
                    "mean_peak": float(np.mean([c.kp[5:17, 2].mean() for c in rec.cands])),
                    "min_peak": float(np.min([c.kp[5:17, 2].mean() for c in rec.cands])),
                    "comp_iou": max([c.meta.get("comp_iou", 0.0) for c in rec.cands]),
                    "agree_hint": [c.meta.get("agree_hint", np.nan) for c in rec.cands],
                    "agree_comp": [c.meta.get("agree_comp", np.nan) for c in rec.cands],
                    "rows": [(c.camera, c.row) for c in rec.cands],
                }
        out.append(record)
    (SCRATCH / f"eval_{name}.pkl").write_bytes(pickle.dumps(out))
    print("done", round(time.time() - started), "s")


if __name__ == "__main__":
    main()
