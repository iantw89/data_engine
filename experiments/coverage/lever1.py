"""Lever 1: recover positions from detections that already exist (unassigned / post-filter rejected / own).

For every gap frame with a hint: raw rows near the projected hint skeleton become candidates; the
mutually consistent set is triangulated with the engine gates. Also runs the withhold test.

Usage: python experiments/coverage/lever1.py
Outputs: C:\\data_engine_data\\_scratch\\coverage\\lever1_gaps.pkl, lever1_withhold.pkl
"""
import json
import pickle
import time

from candidates import hint_view, near_candidates
from common import *  # noqa
from hints import build_presence, make_hint
from recover_lib import recover
from withhold import sample_runs, withheld_sets

REACH = 30


def run_frame(scene, ident, recipe, hint, who, t, kinds, hide_rows=None, hint_radius=None):
    cands = []
    for c in range(len(scene.cameras)):
        view = hint_view(scene, c, t, hint.pose)
        if view is None:
            continue
        cands += near_candidates(scene, ident, c, t, view[0], view[1], who, kinds=kinds, hide_rows=hide_rows)
    if len({c.camera for c in cands}) < 2:
        return None, len(cands)
    rec = recover(scene, cands, recipe, hint_root=hint.root if hint_radius else None, hint_radius_in=hint_radius)
    return rec, len(cands)


def summarize(rec, hint, truth=None):
    out = {"ok": rec is not None}
    if rec is None:
        return out
    out.update(cams=rec.cams, n_body=rec.n_body, median_residual=rec.median_residual, pair_support=rec.pair_support,
               hint_dist_xy=float(np.linalg.norm(rec.root[:2] - hint.root[:2])), root=rec.root.tolist(),
               sources=[c.source for c in rec.cands], rows=[(c.camera, c.row) for c in rec.cands], xyz=rec.xyz,
               min_peak=float(np.min([c.kp[5:17, 2].mean() for c in rec.cands])),
               mean_peak=float(np.mean([c.kp[5:17, 2].mean() for c in rec.cands])),
               hint_kind=hint.kind, hint_distance=hint.distance, span=hint.span)
    if truth is not None:
        both = np.isfinite(truth[:, 0]) & np.isfinite(rec.xyz[:, 0])
        err = np.linalg.norm(truth[both] - rec.xyz[both], axis=1)
        out.update(joint_err_median=float(np.median(err)) if len(err) else None, joint_err_max=float(err.max()) if len(err) else None,
                   n_joint_compared=int(both.sum()), root_err=float(np.linalg.norm(truth_root(truth) - rec.root)))
    return out


def truth_root(xyz):
    from recover_lib import root_of
    return root_of(xyz)


def main():
    started = time.time()
    ws, scene, tracks, solution = load()
    recipe = RECIPE
    presence = build_presence(tracks)
    ident = {c.camera: solution[f"identity_{c.camera}"] for c in scene.cameras}
    frames = json.loads((SCRATCH / "gap_classify_frames.json").read_text())
    results = []
    for k, f in enumerate(frames):
        if not f["hint"]:
            continue
        hint = make_hint(presence, f["identity"], f["t"], reach=REACH)
        for variant, kinds in (("L1", ("own", "unassigned", "filtered")), ("L1+other", ("own", "unassigned", "filtered", "other"))):
            rec, n = run_frame(scene, ident, recipe, hint, f["identity"], f["t"], kinds)
            results.append({**{x: f[x] for x in ("identity", "t", "kind", "gap", "offset", "cause")}, "variant": variant,
                            "n_cands": n, **summarize(rec, hint)})
        if k % 1000 == 0:
            print(k, len(frames), time.time() - started, flush=True)
    (SCRATCH / "lever1_gaps.pkl").write_bytes(pickle.dumps(results))

    # ---------------------------------------------------------------- withhold test
    rng = np.random.default_rng(7)
    runs = sample_runs(presence, tracks, rng, target_frames=3000)
    hidden = withheld_sets(runs)
    hide_rows = set()
    for r in runs:
        for t in r["ts"]:
            row = presence.index[r["identity"]][np.searchsorted(presence.t[r["identity"]], t)]
            for c in range(len(scene.cameras)):
                if tracks["rows"][row, c] >= 0:
                    hide_rows.add((c, int(tracks["rows"][row, c])))
    out = []
    for r in runs:
        who = r["identity"]
        for t in r["ts"]:
            hint = make_hint(presence, who, t, reach=REACH, exclude=hidden[who])
            row = presence.index[who][np.searchsorted(presence.t[who], t)]
            truth = tracks["xyz"][row].astype(np.float64)
            for variant, kinds in (("L1", ("own", "unassigned", "filtered")), ("L1+other", ("own", "unassigned", "filtered", "other"))):
                rec, n = run_frame(scene, ident, recipe, hint, who, t, kinds, hide_rows=hide_rows)
                out.append({"identity": who, "t": t, "length": r["length"], "variant": variant, "n_cands": n,
                            "n_truth_views": int((tracks["rows"][row] >= 0).sum()),
                            "truth_root": truth_root(truth).tolist(), **summarize(rec, hint, truth)})
    (SCRATCH / "lever1_withhold.pkl").write_bytes(pickle.dumps({"runs": runs, "results": out}))
    print("done", time.time() - started)


if __name__ == "__main__":
    main()
