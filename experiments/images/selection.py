"""Which images should a group show? What the reviewer's answers say about hard-to-judge images.

    python experiments/images/selection.py table [--sharpness]   -> shown_images.json
    python experiments/images/selection.py report                -> tables

Every image shown in an answered group task (re-identification, cross-view match, swap check) of the felix
hour, with what can be measured about it (joints found, overlap with other boxes, cut by the border, size,
sharpness) and what the reviewer said about it (flag, verdict of its group). Then: what would another choice
of images have shown, and what would it have hidden? Read-only on the workspace.
"""
from __future__ import annotations

import json
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data_engine.config import RECIPE  # noqa: E402
from data_engine.solve import link_ranking  # noqa: E402
from data_engine.solve.scene import load_scene  # noqa: E402
from data_engine.store import parse_det_key  # noqa: E402
from data_engine.web.validate import CONTEXT, flag_groups  # noqa: E402
from data_engine.workspace import Workspace  # noqa: E402

OUT = Path(__file__).parent
FELIX = "felix_20251223173052_20251223183103"
CLEAR = RECIPE["link_ranking"]
TOLERANCE = {"reidentify": 40, "cross_view_match": 15, "swap_check": 40}  # frames around the image a choice exists in
# Answers up to here (2026-11-19 22:11, task 404) undercount single-image flags: the reviewer tired of marking
# every mixed pose, sliver and bad pose part-way through. Later answers mark every problem.
FLAGS_EXHAUSTIVE_AFTER = 1795155100.0


def table(with_sharpness: bool, dry_run: Path | None) -> None:
    ws = Workspace(FELIX)
    scene = load_scene(ws, joint_peak=RECIPE["geometry"]["joint_peak"], rays=False)
    with open(ws.path("solve_state", "associated.pkl"), "rb") as handle:
        labels = pickle.load(handle)["associated"].labels
    after = None
    if dry_run is not None and (dry_run / "result.pkl").is_file():
        with open(dry_run / "result.pkl", "rb") as handle:
            after = pickle.load(handle)  # the solve with the answers applied (scratch copy)
    sizes = [(int(ws.dataset.cameras[c.camera].width), int(ws.dataset.cameras[c.camera].height)) for c in scene.cameras]
    by_name = {camera.camera: camera for camera in scene.cameras}

    def measured(camera, rows):
        quality = link_ranking.view_quality(camera, np.asarray(rows), sizes[camera.index], scene.joint_peak, CLEAR["border_px"])
        quality["clear"] = link_ranking.clear_view(quality, CLEAR)
        peaks = camera.detections.keypoints[np.asarray(rows), :, 2]
        quality["peak"] = peaks.mean(axis=1)
        return quality

    def identity_after(camera_index, row):
        if after is None:
            return None
        label = int(after["label"][camera_index][row])
        return int(after["identity_of_label"][label]) if label >= 0 else -1

    db = sqlite3.connect(f"file:{ws.path('engine.sqlite3')}?mode=ro", uri=True)
    rows_out, tasks_out = [], []
    for task_id, task_type, payload, answer, answered in db.execute(
            "select id, type, payload, answer, answered from tasks where status in ('answered', 'applied') "
            "and answer is not null order by answered"):
        payload, answer = json.loads(payload), json.loads(answer)
        if payload.get("ui") != "reference_candidates" or "images" not in answer:
            continue
        groups = flag_groups(payload)
        reference_id = str((payload.get("reference") or {}).get("id", "A"))
        tolerance = TOLERANCE.get(task_type, 15)
        tasks_out.append({"task": task_id, "type": task_type, "answered": answered,
                          "verdicts": answer.get("candidates") or {}, "images": sum(len(keys) for keys in groups.values())})
        for gid, keys in groups.items():
            role = "reference" if gid == reference_id else "context" if gid == CONTEXT else "candidate"
            flags = {key: answer["images"].get(key) or {} for key in keys}
            kept = [key for key in keys if not flags[key].get("exclude")]
            for key in keys:
                name, frame, det = parse_det_key(key)
                camera = by_name[name]
                row = camera.detections.row(frame, det)
                t = int(camera.t[row])
                quality = measured(camera, [row])
                label, atom = int(labels.label[camera.index][row]), int(labels.atom[camera.index][row])
                entry = {"task": task_id, "type": task_type, "role": role, "group": gid, "key": key,
                         "group_size": len(keys), "verdict": (answer.get("candidates") or {}).get(gid),
                         "exclude": flags[key].get("exclude"), "bad_pose": bool(flags[key].get("bad_pose")),
                         "joints": int(quality["joints"][0]), "overlap": round(float(quality["overlap"][0]), 3),
                         "cut": bool(quality["cut"][0]), "height": round(float(quality["height"][0]), 1),
                         "peak": round(float(quality["peak"][0]), 3), "clear": bool(quality["clear"][0]),
                         "camera": camera.index, "row": int(row), "t": t, "label": label, "atom": atom}
                # What else could have been shown from this camera, close in time?
                for scope, mask in (("piece", labels.atom[camera.index] == atom if atom >= 0 else None),
                                    ("camera", labels.label[camera.index] == label if label >= 0 else None)):
                    if mask is None:
                        entry[f"alt_{scope}"] = None
                        continue
                    options = np.flatnonzero(mask & (np.abs(camera.t - t) <= tolerance) & (camera.t >= 0))
                    q = measured(camera, options)
                    good = options[q["clear"]]
                    entry[f"n_{scope}"], entry[f"clear_{scope}"] = int(len(options)), int(len(good))
                    if len(good):
                        best = int(good[np.argmin(np.abs(camera.t[good] - t))])
                        entry[f"alt_{scope}"] = {"row": best, "dt": int(camera.t[best] - t),
                                                 "same_piece": bool(labels.atom[camera.index][best] == atom),
                                                 "identity_after": identity_after(camera.index, best)}
                    else:
                        entry[f"alt_{scope}"] = None
                entry["identity_after"] = identity_after(camera.index, row)
                if after is not None and flags[key].get("exclude") == "other_person":
                    rest = [k for k in kept if k != key]
                    ids = set()
                    for other in rest:
                        n2, f2, d2 = parse_det_key(other)
                        ids.add(identity_after(by_name[n2].index, by_name[n2].detections.row(f2, d2)))
                    entry["track_after"] = sorted(i for i in ids if i is not None and i >= 0)
                if with_sharpness:
                    image = ws.frames.read(name, frame)
                    box = camera.detections.boxes[row, :4]
                    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
                    crop = image[y0:max(y0 + 2, int(box[3])), x0:max(x0 + 2, int(box[2]))].astype(np.float32).mean(axis=2)
                    if crop.shape[0] > 2 and crop.shape[1] > 2:
                        lap = crop[1:-1, 1:-1] * 4 - crop[:-2, 1:-1] - crop[2:, 1:-1] - crop[1:-1, :-2] - crop[1:-1, 2:]
                        entry["sharpness"] = round(float(lap.var()), 1)
                        entry["brightness"] = round(float(crop.mean()), 1)
                rows_out.append(entry)
        print(task_id, task_type, len(rows_out), flush=True)
    # The population the shown images are drawn from: a sample of all detections in 3D tracks.
    rng = np.random.default_rng(0)
    population = {"n": 0, "clear": 0, "joints_lt13": 0, "overlap": 0, "cut": 0, "small": 0}
    for camera in scene.cameras:
        tracked = np.flatnonzero((labels.label[camera.index] >= 0) & (camera.t >= 0))
        sample = rng.choice(tracked, size=min(4000, len(tracked)), replace=False)
        q = measured(camera, sample)
        population["n"] += len(sample)
        population["clear"] += int(q["clear"].sum())
        population["joints_lt13"] += int((q["joints"] < CLEAR["min_joints"]).sum())
        population["overlap"] += int((q["overlap"] > CLEAR["max_overlap"]).sum())
        population["cut"] += int(q["cut"].sum())
        population["small"] += int((q["height"] < CLEAR["min_height_px"]).sum())
    population["tracked_detections"] = int(sum(int(((labels.label[c.index] >= 0) & (c.t >= 0)).sum()) for c in scene.cameras))
    (OUT / "shown_images.json").write_text(json.dumps({"images": rows_out, "tasks": tasks_out, "population": population}),
                                           encoding="utf-8")
    print("images", len(rows_out), "tasks", len(tasks_out), population)


if __name__ == "__main__" and sys.argv[1] == "table":
    extra = [a for a in sys.argv[2:] if not a.startswith("--")]
    table("--sharpness" in sys.argv, Path(extra[0]) if extra else None)


# --------------------------------------------------------------------------- the engine's own choice, before and after
def engine() -> None:
    """The questions the solve stage would ask on the felix state as it is, with the images chosen the old way
    (nearest frame) and the new way (clearer frame of the same piece, one question in five kept as a control).
    Read-only: constraints come from a copy of the database, nothing is written to the workspace."""
    import copy

    from data_engine.db import Database
    from data_engine.engine import Context
    from data_engine.solve.pipeline import identify
    from data_engine.stages import solve as stage

    ws = Workspace(FELIX)
    scene = load_scene(ws, joint_peak=RECIPE["geometry"]["joint_peak"], rays=False)
    with open(ws.path("solve_state", "associated.pkl"), "rb") as handle:
        associated = pickle.load(handle)["associated"]
    copy_path = OUT / "_felix_db_copy.sqlite3"
    copy_path.unlink(missing_ok=True)
    source = sqlite3.connect(f"file:{ws.path('engine.sqlite3')}?mode=ro", uri=True)
    target = sqlite3.connect(str(copy_path))
    source.backup(target)
    target.close(), source.close()
    ws.__dict__["db"] = Database(copy_path)
    constraints, info = stage.load_constraints(Context(ws, 0, "solve"), scene)
    solved = identify(scene, associated, RECIPE, constraints, log=lambda *_: None)
    sizes = [(int(ws.dataset.cameras[c.camera].width), int(ws.dataset.cameras[c.camera].height)) for c in scene.cameras]
    by_name = {camera.camera: camera for camera in scene.cameras}

    def is_clear(key: str) -> bool:
        name, frame, det = parse_det_key(key)
        camera = by_name[name]
        quality = link_ranking.view_quality(camera, np.array([camera.detections.row(frame, det)]), sizes[camera.index],
                                            scene.joint_peak, CLEAR["border_px"])
        return bool(link_ranking.clear_view(quality, CLEAR)[0])

    def groups_of(unknown: dict) -> list[list[str]]:
        detail = unknown["detail"]
        return [list(detail.get("reference_keys") or [])] + [list(c["keys"]) for c in detail.get("candidates") or []]

    results = {}
    original = copy.deepcopy(RECIPE["task_images"])
    try:
        for name, settings in (("nearest", {"prefer_clear": False, "control_share": 0.0}), ("adopted", original),
                               ("clear, no control", {"prefer_clear": True, "control_share": 0.0})):
            RECIPE["task_images"].clear()
            RECIPE["task_images"].update(settings)
            unknowns = stage.extract_unknowns(solved, info["resolved_keys"])
            results[name] = {kind: {item["subject"]: item for item in unknowns[kind]}
                             for kind in ("identity_link", "association", "unassigned")}
    finally:
        RECIPE["task_images"].clear()
        RECIPE["task_images"].update(original)
        ws.db.connection.close()
        copy_path.unlink(missing_ok=True)
    summary = {}
    for kind in ("identity_link", "association", "unassigned"):
        base = results["nearest"][kind]
        for name, found in results.items():
            images = [key for item in found[kind].values() for group in groups_of(item) for key in group]
            groups = [group for item in found[kind].values() for group in groups_of(item) if group]
            clear = [is_clear(key) for key in images]
            flags = {key: value for key, value in zip(images, clear)}
            control = sum(1 for item in found[kind].values() if item["detail"].get("image_choice") == "nearest")
            summary[f"{kind} / {name}"] = {
                "questions": len(found[kind]), "same subjects as before": set(found[kind]) == set(base),
                "control questions": control, "images": len(images), "clear": round(float(np.mean(clear)), 3) if clear else None,
                "groups without a clear image": round(float(np.mean([not any(flags[k] for k in g) for g in groups])), 3) if groups else None,
                "images changed": sum(1 for subject, item in found[kind].items() if subject in base
                                      for a, b in zip(groups_of(item), groups_of(base[subject])) for x, y in zip(a, b) if x != y)}
            print(f"{kind} / {name}:", summary[f"{kind} / {name}"], flush=True)
    (OUT / "engine_choice.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__" and sys.argv[1] == "engine":
    engine()


# --------------------------------------------------------------------------- report
def report() -> None:
    import collections

    from scipy.stats import fisher_exact

    data = json.loads((OUT / "shown_images.json").read_text(encoding="utf-8"))
    images, tasks, population = data["images"], data["tasks"], data["population"]
    answered = {task["task"]: task["answered"] for task in tasks}
    n_clear = sum(e["clear"] for e in images)
    n_unclear = len(images) - n_clear
    print(f"images shown {len(images)} in {len(tasks)} answered tasks; clear {n_clear / len(images):.0%} "
          f"(all tracked detections: {population['clear'] / population['n']:.0%})")
    # The two batches of answers, and the two ways the images were chosen, kept apart.
    choice = {}
    try:
        db = sqlite3.connect(f"file:{Workspace(FELIX).path('engine.sqlite3')}?mode=ro", uri=True)
        for task_id, payload in db.execute("select id, payload from tasks where answer is not null"):
            choice[task_id] = (json.loads(payload).get("unknown") or {}).get("image_choice") or "nearest"
    except sqlite3.Error:
        pass
    for label, part in (("answers with undercounted flags", [e for e in images if answered[e["task"]] <= FLAGS_EXHAUSTIVE_AFTER]),
                        ("exhaustive answers, nearest frames (control)",
                         [e for e in images if answered[e["task"]] > FLAGS_EXHAUSTIVE_AFTER and choice.get(e["task"], "nearest") == "nearest"]),
                        ("exhaustive answers, clearer frames",
                         [e for e in images if answered[e["task"]] > FLAGS_EXHAUSTIVE_AFTER and choice.get(e["task"]) == "clear"])):
        if part:
            single = sum(1 for e in part if e["bad_pose"] or e["exclude"] in ("mixed", "not_person", "unsure"))
            other = sum(1 for e in part if e["exclude"] == "other_person")
            unsure = sum(1 for e in part if e["verdict"] == "unsure")
            print(f"  {label}: {len(part)} images, clear {np.mean([e['clear'] for e in part]):.0%}, single-image flags "
                  f"{single} ({single / len(part):.1%}), someone else {other} ({other / len(part):.1%}), in groups judged "
                  f"unsure {unsure}")
        else:
            print(f"  {label}: none yet")

    def has(e, flag):
        return e["bad_pose"] if flag == "bad_pose" else e["exclude"] == flag

    print("\nflag                 on clear        on unclear      p     | unclear: no clear frame in the piece / would be exchanged")
    for flag in ("other_person", "mixed", "not_person", "unsure", "bad_pose"):
        a = sum(1 for e in images if e["clear"] and has(e, flag))
        b = sum(1 for e in images if not e["clear"] and has(e, flag))
        p = fisher_exact([[a, n_clear - a], [b, n_unclear - b]])[1]
        swapped = sum(1 for e in images if not e["clear"] and has(e, flag) and e.get("alt_piece"))
        print(f"{flag:13s} {a:3d} ({a / n_clear:5.1%})   {b:3d} ({b / n_unclear:5.1%})   {p:.3f} | {b - swapped:2d} / {swapped:2d}")
    unclear = [e for e in images if not e["clear"]]
    exchanged = [e for e in unclear if e.get("alt_piece")]
    shift = [abs(e["alt_piece"]["dt"]) for e in exchanged]
    print(f"\nunclear images {len(unclear)}: a clear view of the same piece within the tolerance for {len(exchanged)} "
          f"({len(exchanged) / len(unclear):.0%}); shift median {np.median(shift):.0f} frames, 90% within {np.quantile(shift, 0.9):.0f}")
    print(f"clear share: {n_clear / len(images):.0%} -> {(n_clear + len(exchanged)) / len(images):.0%}")
    hidden = kept = 0
    for e in images:
        if e["exclude"] == "other_person" and not e["clear"] and e.get("alt_piece"):
            if e["alt_piece"]["identity_after"] in set(e.get("track_after") or []):
                hidden += 1
            else:
                kept += 1
    total = sum(1 for e in images if e["exclude"] == "other_person")
    # (``hidden`` counts flags whose image itself stays in the track in that solve: given after it was run.)
    print(f"'someone else' flags {total}: exchanged frame leaves the track with the flagged image {kept}; "
          f"not checkable (flag given after the solve with the answers applied) {hidden}")
    groups = collections.defaultdict(list)
    for e in images:
        groups[(e["task"], e["group"], e["role"])].append(e)
    none = [key for key, es in groups.items() if not any(e["clear"] for e in es)]
    fixed = [key for key in none if any(e.get("alt_piece") for e in groups[key])]
    print(f"groups {len(groups)}: without a clear image {len(none)} ({len(none) / len(groups):.0%}) -> {len(none) - len(fixed)} "
          f"({(len(none) - len(fixed)) / len(groups):.0%})")
    verdicts = collections.Counter(("no clear image" if key in none else "a clear image", es[0]["verdict"])
                                   for key, es in groups.items() if key[2] == "candidate")
    print("candidate verdicts:", dict(verdicts))
    by_key = collections.defaultdict(list)
    for e in images:
        by_key[e["key"]].append(e)
    again = collections.Counter()
    for es in by_key.values():
        flags = {(e["exclude"], e["bad_pose"]) for e in es}
        if len(es) > 1 and any(f[0] or f[1] for f in flags):
            again[("clear" if es[0]["clear"] else "unclear", "same flag each time" if len(flags) == 1 else "flag changed")] += 1
    # (The reviewer: a flag dropped at a later showing is fatigue from flagging one image again and again.)
    print("images shown in two or more tasks and flagged at least once:", dict(again))
    first = {}
    for e in sorted(images, key=lambda e: answered[e["task"]]):
        first.setdefault(e["key"], e)
    once = list(first.values())
    for name, part in (("clear", [e for e in once if e["clear"]]), ("unclear", [e for e in once if not e["clear"]])):
        flagged = sum(1 for e in part if e["exclude"] or e["bad_pose"])
        print(f"first showing only, {name}: {len(part)} images, flagged {flagged} ({flagged / len(part):.1%})")
    order = sorted(tasks, key=lambda task: task["answered"])
    unclear_count = collections.Counter(e["task"] for e in unclear)
    flag_count = collections.Counter(e["task"] for e in images if e["exclude"] or e["bad_pose"])
    x, y = [], []
    for before, task in zip(order, order[1:]):
        seconds = task["answered"] - before["answered"]
        if 0 < seconds < 300:
            x.append([1, task["images"], flag_count[task["task"]], unclear_count[task["task"]]])
            y.append(np.log(seconds))
    x, y = np.array(x, float), np.array(y)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    error = np.sqrt(np.diag(residual @ residual / (len(y) - x.shape[1]) * np.linalg.inv(x.T @ x)))
    print(f"time per task (n={len(y)}, median {np.exp(np.median(y)):.0f} s): per flag set {np.exp(beta[2]) - 1:+.0%}, per unclear image "
          f"{np.exp(beta[3]) - 1:+.1%} (95% {np.exp(beta[3] - 1.96 * error[3]) - 1:+.0%} .. {np.exp(beta[3] + 1.96 * error[3]) - 1:+.0%})")
    share, total_tracked = population["clear"] / population["n"], population["tracked_detections"]
    for flag in ("mixed", "bad_pose", "not_person"):
        a = sum(1 for e in images if e["clear"] and has(e, flag))
        b = sum(1 for e in images if not e["clear"] and has(e, flag))
        rate = share * a / n_clear + (1 - share) * b / n_unclear
        print(f"{flag}: {a + b} flagged; weighted to all tracked detections {rate:.1%} = about {rate * total_tracked:,.0f}")


if __name__ == "__main__" and sys.argv[1] == "report":
    report()
