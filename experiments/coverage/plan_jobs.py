"""Plan every pixel job for levers 2/3 in one go (so frames are decoded once).

Cases (each = one person-frame hypothesis at sync index t, with a hint pose used ONLY to place boxes):
  gap          every missing person-frame of the saved solution that has a neighbouring measured pose
  withhold     measured person-frames hidden together with their detections (truth = engine 3D pose)
  neg_shift    hint displaced 20 / 39 / 79 in from a measured pose (the person is NOT there)
  neg_absent   hint = pose held at the first/last appearance, applied 10-60 s outside the lifetime
  neg_empty    standing pose planted at a floor spot with nobody measured within 3 m and no raw detection overlapping

Output: cases.pkl, jobs.npz in C:\\data_engine_data\\_scratch\\coverage
"""
import json
import pickle

from candidates import hint_view
from common import *  # noqa
from hints import build_presence, make_hint, pair_angle_deg
from withhold import sample_runs, withheld_sets

REACH = 30


def case_cams(scene, t, pose):
    """Visible cameras of a hinted pose (>=2 with a usable ray pair), their boxes."""
    views = {}
    for c in range(len(scene.cameras)):
        v = hint_view(scene, c, t, pose)
        if v is not None:
            views[c] = v[0]
    centers = [c.cam.center for c in scene.cameras]
    root = np.nanmean(pose[[11, 12]], axis=0) if np.isfinite(pose[[11, 12], 0]).all() else np.nanmean(pose[5:13], axis=0)
    cams = sorted(views)
    angle = pair_angle_deg(root, centers)
    ok = any(angle[a, b] >= 8 for i, a in enumerate(cams) for b in cams[i + 1:])
    return (views if ok else {}), root


def main():
    ws, scene, tracks, solution = load()
    presence = build_presence(tracks)
    rng = np.random.default_rng(11)
    by_t = {}
    for row in np.flatnonzero(np.isfinite(tracks["root"]).all(axis=1)):
        by_t.setdefault(int(tracks["t"][row]), []).append(row)

    def nearest_other(t, root, who):
        best = np.inf
        for r in by_t.get(t, []):
            if int(tracks["identity"][r]) != who:
                best = min(best, float(np.linalg.norm(tracks["root"][r][:2] - root[:2])))
        return best

    cases = []

    def add(tag, who, t, pose, hint=None, truth=None, **meta):
        views, root = case_cams(scene, t, pose)
        if len(views) < 2:
            return False
        cases.append({"id": len(cases), "tag": tag, "who": who, "t": int(t), "pose": pose.copy(), "root": root,
                      "views": {int(c): b for c, b in views.items()}, "truth": truth, "nearest_other_in": nearest_other(t, root, who),
                      "hint_kind": hint.kind if hint else None, "hint_distance": hint.distance if hint else None,
                      "span": hint.span if hint else None, **meta})
        return True

    # gaps
    frames = json.loads((SCRATCH / "gap_classify_frames.json").read_text())
    for f in frames:
        if not f["hint"]:
            continue
        hint = make_hint(presence, f["identity"], f["t"], reach=REACH)
        add("gap", f["identity"], f["t"], hint.pose, hint, gap=f["gap"], gap_kind=f["kind"], cause=f["cause"])
    # withhold (same seed as lever1.py)
    runs = sample_runs(presence, tracks, np.random.default_rng(7), target_frames=3000)
    hidden = withheld_sets(runs)
    for r in runs:
        who = r["identity"]
        for t in r["ts"]:
            hint = make_hint(presence, who, t, reach=REACH, exclude=hidden[who])
            row = presence.index[who][np.searchsorted(presence.t[who], t)]
            add("withhold", who, t, hint.pose, hint, truth=tracks["xyz"][row].astype(np.float64), truth_rows=tracks["rows"][row].tolist(),
                length=r["length"])
    # negative: displaced hints
    identities = list(presence.t)
    n = 0
    while n < 2000:
        who = identities[rng.integers(len(identities))]
        ts = presence.t[who]
        k = int(rng.integers(len(ts)))
        row = presence.index[who][k]
        pose = tracks["xyz"][row].astype(np.float64)
        shift = [20.0, 39.4, 78.7][n % 3]
        theta = rng.uniform(0, 2 * np.pi)
        moved = pose + np.array([np.cos(theta) * shift, np.sin(theta) * shift, 0.0])
        n += add("neg_shift", who, int(ts[k]), moved, shift_in=shift, true_root=tracks["root"][row].astype(np.float64))
    # negative: absent person, hint = pose held at first / last appearance
    n = 0
    while n < 1500:
        who = identities[rng.integers(len(identities))]
        ts = presence.t[who]
        side = rng.integers(2)
        away = int(rng.integers(100, 600))
        t = int(ts[0] - away) if side == 0 else int(ts[-1] + away)
        if not 0 <= t < scene.T:
            continue
        row = presence.index[who][0 if side == 0 else -1]
        pose = tracks["xyz"][row].astype(np.float64)
        n += add("neg_absent", who, t, pose, away=away, side=int(side), alive_elsewhere=False)
    # negative: empty floor
    pool = [r for r in np.flatnonzero(np.isfinite(tracks["xyz"][:, 5:17, 0]).all(axis=1))]
    roots_all = tracks["root"][np.isfinite(tracks["root"]).all(axis=1)]
    n = tries = 0
    while n < 3000 and tries < 100000:
        tries += 1
        t = int(rng.integers(scene.T))
        base = roots_all[rng.integers(len(roots_all))][:2]
        theta, rad = rng.uniform(0, 2 * np.pi), rng.uniform(100, 250)
        spot = base + rad * np.array([np.cos(theta), np.sin(theta)])
        if nearest_other(t, np.array([*spot, 0]), -1) < 120.0:
            continue
        template = tracks["xyz"][pool[rng.integers(len(pool))]].astype(np.float64)
        hips = template[[11, 12]].mean(axis=0)
        pose = template.copy()
        pose[:, :2] += spot - hips[:2]
        views, _ = case_cams(scene, t, pose)
        if len(views) < 2:
            continue
        # no raw detection (any score) may overlap the planted box in any camera
        crowded = False
        for c, box in views.items():
            frame = int(scene.native[t, c])
            rows = scene.cameras[c].detections.rows(frame)
            if rows.stop > rows.start:
                from hints import box_iou
                if box_iou(box, scene.cameras[c].detections.boxes[rows, :4].astype(np.float64)).max() > 0.05:
                    crowded = True
                    break
        if crowded:
            continue
        n += add("neg_empty", -1, t, pose)
    print({tag: sum(1 for c in cases if c["tag"] == tag) for tag in ("gap", "withhold", "neg_shift", "neg_absent", "neg_empty")})
    cam, frame, box, case = [], [], [], []
    for c in cases:
        for cam_i, b in c["views"].items():
            cam.append(cam_i)
            frame.append(int(scene.native[c["t"], cam_i]))
            box.append(b)
            case.append(c["id"])
    np.savez(SCRATCH / "jobs.npz", cam=np.array(cam), frame=np.array(frame), box=np.array(box, np.float32), case=np.array(case))
    (SCRATCH / "cases.pkl").write_bytes(pickle.dumps(cases))
    print("jobs", len(cam), "frames", len(set(zip(cam, frame))))


if __name__ == "__main__":
    main()
