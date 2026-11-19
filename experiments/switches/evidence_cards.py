"""Evidence cards for flagged chunks: which cameras and which detections (camera/frame/index keys) a reviewer should look at.

For every flagged chunk of a solution, per camera: the four detections that matter (A just before the chunk, A just after,
B just before, B just after), the box overlap of A and B through the chunk, and a camera rank
(separated first = lowest max IoU, then both people visible before and after, then largest boxes).
No crops are rendered here (decode-bound); keys are the engine's detection keys so the existing strip/crop renderer can show them.

    python experiments/switches/evidence_cards.py            # golden saved, 2-of-n rule -> evidence_cards_saved.json
"""
from __future__ import annotations

import json

import numpy as np

import common
import cues
import rule
from common import FPS


def box_iou1(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def card(ctx, ix, chunk, W=100):
    a, b, s, e = chunk["a"], chunk["b"], chunk["s"], chunk["e"]
    cams = []
    for c in ctx.scene.cameras:
        rec = {"camera": c.camera, "iou_max": 0.0, "dets": {}}
        got = 0
        for name, label in (("A", a), ("B", b)):
            for when, lo, hi, pick in (("before", s - W, s - 1, -1), ("after", e + 1, e + W, 0)):
                cm, rw, t = ix.rows(label, lo, hi)
                m = cm == c.index
                if m.any():
                    k = np.flatnonzero(m)[pick]
                    row, tt = int(rw[k]), int(t[k])
                    frame = int(ctx.scene.native[tt, c.index])
                    box = ctx.boxes[c.index][row]
                    rec["dets"][f"{name}_{when}"] = {"key": f"{c.camera}/{frame}/{int(row - c.detections.rows(frame).start)}",
                                                     "t": tt, "height_px": float(box[3] - box[1])}
                    got += 1
        ra, rb = ix.rows(a, s, e)[:2], ix.rows(b, s, e)[:2]
        ta, tb = ix.rows(a, s, e)[2], ix.rows(b, s, e)[2]
        ma, mb = ra[0] == c.index, rb[0] == c.index
        for t0 in np.intersect1d(ta[ma], tb[mb]):
            ia, ib = ra[1][ma][ta[ma] == t0][0], rb[1][mb][tb[mb] == t0][0]
            rec["iou_max"] = max(rec["iou_max"], float(box_iou1(ctx.boxes[c.index][ia], ctx.boxes[c.index][ib])))
        rec["n_dets"] = got
        rec["rank_key"] = (rec["iou_max"] > 0.05, -got, -np.mean([d["height_px"] for d in rec["dets"].values()] or [0]))
        cams.append(rec)
    cams.sort(key=lambda r: r["rank_key"])
    for r in cams:
        r.pop("rank_key")
    cues_out = {k: (None if not np.isfinite(chunk.get(k, np.nan)) else round(float(chunk[k]), 3))
                for k in ("motion_pos", "body", "app", "app_samecam", "trk_contra_sep", "trk_agree_sep", "own_flips", "raw_changes")}
    return {"identities": [a, b], "t_start_s": s / FPS, "t_end_s": e / FPS, "causes": chunk.get("causes"), "cue_values": cues_out,
            "show_first": [r for r in cams if r["n_dets"] >= 3][:3], "all_cameras": cams}


def main():
    ctx = cues.Ctx(with_appearance=False)
    sol = common.load_saved(ctx.ws, ctx.scene)
    ix = cues.Index(ctx, sol)
    import contextlib, io
    rows = rule.load("saved")
    with contextlib.redirect_stdout(io.StringIO()):
        out = rule.evaluate(rows, "saved", use_own=False, min_causes=2)
    flagged = [r for r in out if r["kind"] == "neg" and r["flag"]]
    cards = [card(ctx, ix, r) for r in flagged]
    (common.SCRATCH / "evidence_cards_saved.json").write_text(json.dumps(cards, indent=1, default=float))
    print(f"{len(cards)} cards -> {common.SCRATCH / 'evidence_cards_saved.json'}")
    if cards:
        c = cards[0]
        print(json.dumps({k: c[k] for k in ("identities", "t_start_s", "t_end_s", "causes", "cue_values")}, default=float))
        for r in c["show_first"]:
            print("  ", r["camera"], "iou_max", round(r["iou_max"], 2), {k: v["key"] for k, v in r["dets"].items()})


if __name__ == "__main__":
    main()
