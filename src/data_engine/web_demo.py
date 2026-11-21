"""Build a development demo workspace so the web UI can be exercised without a real run.

    python -m data_engine.web_demo [--running]

Creates/refreshes ``<data_root>/webdev_demo`` from the smoke detections (first 400 native
frames of the 7 felix cameras) and fills the state database with plausible *fake* rounds,
stage runs, unknowns, events and tasks whose panels are built from real detections, so
panels render real crops with skeletons. For development only: nothing here is a result.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np

from .config import Site, load_site
from .engine import STAGE, STAGES, Engine
from .workspace import Workspace

DEMO_ID = "webdev_demo"
SOURCE_DATASET = Path(r"C:\data_engine_data\felix_20251223173052_20251223183103")
SMOKE_DETECT = Path(r"C:\data_engine_data\smoke_detect")
FPS = 10.0
COLORS = ["cyan", "yellow", "orange", "magenta", "green"]


# ----------------------------------------------------------------------------- detection helpers
def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Dets:
    def __init__(self, workspace: Workspace):
        self.store, self.cameras = workspace.detections, list(workspace.dataset.cameras)

    def at(self, camera: str, frame: int, min_score: float = 0.5, min_joints: int = 10):
        data = self.store.camera(camera)
        rows = data.rows(frame)
        out = []
        for offset, (box, keypoints) in enumerate(zip(data.boxes[rows], data.keypoints[rows])):
            if box[4] >= min_score and int((keypoints[:, 2] >= 0.3).sum()) >= min_joints:
                out.append({"key": f"{camera}/{frame}/{offset}", "box": [float(v) for v in box[:4]], "score": float(box[4]),
                            "camera": camera, "frame": frame, "area": float((box[2] - box[0]) * (box[3] - box[1]))})
        return sorted(out, key=lambda d: -d["area"])


def panel(det: dict, label: str = "", color: str = "green", *, others: bool = True, context: bool = True) -> dict:
    return {"camera": det["camera"], "frame": det["frame"], "box": det["box"], "highlight": [det["key"]],
            "others": others, "project": [], "label": label, "color": color, "context": context}


def frame_panel(camera: str, frame: int, label: str = "", color: str = "green", box=None, highlight=()) -> dict:
    return {"camera": camera, "frame": frame, "box": box, "highlight": list(highlight), "others": True,
            "project": [], "label": label, "color": color, "context": False}


def follow(dets: Dets, start: dict, step: int, count: int) -> list[dict]:
    """Follow a detection forward by IoU; returns the matched detections (including the start)."""
    chain = [start]
    frame = start["frame"]
    while len(chain) < count and frame + step < 395:
        frame += step
        best = max(dets.at(start["camera"], frame, 0.3, 6), key=lambda d: iou(d["box"], chain[-1]["box"]), default=None)
        if best is None or iou(best["box"], chain[-1]["box"]) < 0.2:
            break
        chain.append(best)
    return chain


def common(ui: str, title: str, question: str, why: str, evidence: dict, seconds: float | None) -> dict:
    return {"ui": ui, "title": title, "question": question, "why": why, "evidence": evidence, "time_seconds": seconds}


# ----------------------------------------------------------------------------- tasks
def make_tasks(workspace: Workspace, round_number: int, rng: random.Random) -> list[int]:
    db, dets = workspace.db, Dets(workspace)
    cams = dets.cameras
    created: list[int] = []

    def add(task_type, subject, payload, priority, unknown_ids=()):
        task_id = db.add_task(task_type, subject, payload, priority=priority, round_number=round_number, unknown_ids=unknown_ids)
        if task_id:
            created.append(task_id)
        return task_id

    def pick(camera_pool=None, min_frame=20, max_frame=380, need=1, **kw):
        for _ in range(400):
            camera = rng.choice(camera_pool or cams)
            frame = rng.randrange(min_frame, max_frame)
            found = dets.at(camera, frame, **kw)
            if len(found) >= need:
                return found
        raise RuntimeError("no suitable detections in the smoke data")

    # continuity (strip)
    for n in range(3):
        start = rng.choice(pick(min_score=0.7)[:2])
        chain = follow(dets, start, 5, 8)
        panels = [panel(d, str(i + 1), "green") for i, d in enumerate(chain)]
        if n == 1 and len(chain) >= 5:  # one strip with a real identity switch: swap in another person
            for i in range(4, len(chain)):
                others = [d for d in dets.at(start["camera"], chain[i]["frame"], 0.5) if iou(d["box"], chain[i]["box"]) < 0.1]
                if others:
                    panels[i] = panel(others[0], str(i + 1), "green")
        add("continuity", f"demo/continuity/{n}", {**common(
            "strip", "Same person across this tracklet?", "Do all crops show the same person? If not, click the first crop showing a different person.",
            "The tracker over-split this tracklet at a possible identity switch (two boxes overlapped).",
            {"overlap IoU": round(rng.uniform(0.3, 0.6), 2), "tracklet length": len(chain) * 5, "camera": start["camera"]},
            start["frame"] / FPS), "panels": panels}, priority=8 - n)

    # cross-view match
    for n in range(3):
        ref = pick(need=1, min_score=0.7)[0]
        pool = [c for c in cams if c != ref["camera"]]
        cands = []
        for camera in rng.sample(pool, 4):
            found = dets.at(camera, ref["frame"], 0.5)
            if found:
                cands.append(found[0])
        labels = "BCDEF"
        add("cross_view_match", f"demo/cross/{n}", {**common(
            "reference_candidates", "Who is A?", "Which tracked person, if any, is the person in reference A?",
            "Two candidates fit the reference's geometry almost equally well (margin below the accept gate).",
            {"reprojection p80 px": round(rng.uniform(4, 11), 1), "runner-up margin": round(rng.uniform(0.01, 0.04), 3), "shared frames": rng.randrange(8, 40)},
            ref["frame"] / FPS),
            "reference": {"id": "A", "caption": f"{ref['camera']} f{ref['frame']}", "panels": [panel(ref, "A", "green")]},
            "candidates": [{"id": labels[i], "caption": f"{c['camera']} f{c['frame']}", "panels": [panel(c, labels[i], COLORS[i % 4])], "track": 100 + i}
                           for i, c in enumerate(cands)],
            "answer_model": 2, "mode": "match", "multiple": False, "single_choice": True, "more_candidates": 0,
            "texts": {"same": "{ref} is {ids}: this piece joins that person's track.",
                      "none_closed": "{ref} is none of the tracked people shown: it stays a single-camera track of its own."}},
            priority=9 - n)

    # re-identification across a gap
    for n in range(2):
        ref = pick(min_frame=200, min_score=0.7)[0]
        cands = []
        for k, camera in enumerate(rng.sample(cams, 3)):
            found = dets.at(camera, max(5, ref["frame"] - 120 - 25 * k), 0.5)
            if found:
                cands.append(found[0])
        add("reidentify", f"demo/reid/{n}", {**common(
            "reference_candidates", "Who is A?", "Which of these earlier tracks show the same person as reference A? More than one can be right.",
            "A track ended inside the room and a new one started 12 s later; several people could continue it.",
            {"gap seconds": 12.0, "endpoint distance m": round(rng.uniform(1.0, 6.0), 1), "ReID margin": round(rng.uniform(0.01, 0.07), 3)}, ref["frame"] / FPS),
            "reference": {"id": "A", "caption": f"later: {ref['camera']} f{ref['frame']}", "panels": [panel(ref, "A")]},
            "candidates": [{"id": "BCD"[i], "caption": f"earlier: {c['camera']} f{c['frame']}", "panels": [panel(c, "BCD"[i], COLORS[i])], "track": 20 + i} for i, c in enumerate(cands)],
            "answer_model": 2, "mode": "match", "multiple": True, "single_choice": False, "more_candidates": 2,
            "texts": {"same": "{ref} is the same person as {ids}: the tracks are joined.",
                      "none_more": "{ref} is none of these. {n} more earlier people could have reached this spot; the engine will ask about them next."}},
            priority=6 - n)

    # detection triage
    for n in range(2):
        found = pick(need=2, min_score=0.4, min_joints=6)
        subject, others = found[-1], found[:-1]
        add("detection_triage", f"demo/triage/{n}", {**common(
            "reference_candidates", "Is this a real person?", "Is this a real person? (Seen by one camera only, and matching no tracked person.)",
            "A tracklet of confident detections belongs to no 3D track (seen by one camera only).",
            {"tracklet frames": rng.randrange(12, 90), "mean score": round(subject["score"], 2)}, subject["frame"] / FPS),
            "reference": {"id": "A", "caption": f"{subject['camera']} f{subject['frame']}", "panels": [panel(subject, "A", "orange")]},
            "candidates": [], "answer_model": 2, "mode": "person",
            "texts": {"person": "A real person seen by this camera only: kept as a single-camera track, without 3D.",
                      "not_person": "Not a person: these detections are taken out of every track."}}, priority=4 - n)

    # pose ownership
    for n in range(2):
        found = pick(need=2, min_score=0.5)
        a, b = found[0], found[1]
        add("pose_ownership", f"demo/pose/{n}", {**common(
            "reference_candidates", "One person or two?", "Two poses in one camera at the same place. One person detected twice, or two people?",
            "Two detections in one frame overlap and their poses are near-identical on 9 joints.",
            {"box IoU": round(iou(a["box"], b["box"]), 2), "common joints": 9}, a["frame"] / FPS),
            "reference": {"id": "A", "caption": "pose A", "panels": [panel(a, "A", "green")]},
            "candidates": [{"id": "B", "caption": "pose B", "panels": [panel(b, "B", "magenta")], "track": None}],
            "answer_model": 2, "mode": "duplicate", "single_choice": True,
            "texts": {"same": "One person with two poses: the better one is used, the other is kept as its duplicate.",
                      "different": "Two people: the two poses are kept apart."}}, priority=3 - n)

    # visibility check (decision)
    options = [{"value": "visible", "label": "Visible", "hint": "I can see them here", "hotkey": "1"},
               {"value": "occluded", "label": "Occluded", "hint": "behind equipment or another person", "hotkey": "2"},
               {"value": "out_of_view", "label": "Out of view", "hint": "outside the camera frame", "hotkey": "3"},
               {"value": "wrong_projection", "label": "Wrong projection", "hint": "skeleton is not where the person is", "hotkey": "4"}]
    for n in range(2):
        d = pick(min_score=0.5)[0]
        add("visibility_check", f"demo/vis/{n}", {**common(
            "decision", "Is the tracked person visible here?", "The dashed skeleton is where this person should be. Are they visible in this view?",
            "A tracked person projects inside this view for 3 s but has no detection.", {"expected px": 1200 + 40 * n, "span frames": 30}, d["frame"] / FPS),
            "panels": [panel(d, "expected", "cyan", others=True), frame_panel(d["camera"], d["frame"], "room", "cyan", box=d["box"])], "options": options}, priority=2 - n * 0.1)

    # calibration review (decision on full frames)
    for n in range(2):
        camera = rng.choice(cams)
        add("calibration_review", f"demo/calib/{n}", {**common(
            "decision", f"Calibration of {camera}", "Does the projected reference geometry line up with this camera's image?",
            "The camera view appears shifted against the calibration capture (rotation-only correction found).",
            {"median shift px": round(rng.uniform(6, 30), 1), "inliers": 41}, None),
            "panels": [frame_panel(camera, 30, "standard", "orange"), frame_panel(camera, 90, "corrected", "green")],
            "options": [{"value": "use_standard", "label": "Use standard", "hotkey": "1"}, {"value": "use_corrected", "label": "Use corrected", "hotkey": "2"},
                        {"value": "exclude_camera", "label": "Exclude camera", "hint": "its labels cannot be trusted", "hotkey": "3"}]}, priority=7 - n)

    # audit
    for n in range(2):
        d = pick(min_score=0.6)[0]
        add("audit", f"demo/audit/{n}", {**common("decision", "Audit: is this label correct?", "Is this label correct (right person, plausible pose)?",
                                                  "Random sample of published-candidate labels.", {"track": 7 + n, "views": 4}, d["frame"] / FPS),
                                         "panels": [panel(d, "label", "green")],
                                         "options": [{"value": v, "label": v.replace("_", " ").capitalize(), "hotkey": str(i + 1)}
                                                     for i, v in enumerate(["correct", "wrong_identity", "wrong_pose", "not_person"])]}, priority=1)

    # alignment review (frame picker)
    for n in range(3):
        ref = pick(min_frame=40, max_frame=360, need=1, min_score=0.7)[0]
        target = rng.choice([c for c in cams if c != ref["camera"]])
        cands = []
        for offset in range(-4, 5):
            found = dets.at(target, ref["frame"] + offset, 0.4, 6)
            cands.append({"id": str(offset), "caption": f"{offset / FPS:+.1f}s (f{ref['frame'] + offset})",
                          "panels": [panel(found[0], f"{offset:+d}", "cyan") if found else frame_panel(target, ref["frame"] + offset, f"{offset:+d}", "cyan")]})
        add("alignment_review", f"demo/align/{n}", {**common(
            "frame_picker", f"Which {target} frame matches?", "Which frame of the target camera shows the same instant as the reference frame?",
            "The audio and motion lag estimates disagree by 0.3 s in this span.", {"audio lag s": 0.2, "motion lag s": -0.1}, ref["frame"] / FPS),
            "reference": {"caption": f"{ref['camera']} f{ref['frame']}", "panels": [panel(ref, "ref", "green")]}, "candidates": cands}, priority=5 - n)

    # CVAT pose correction
    for n in range(2):
        add("pose_correction", f"demo/cvat/{n}", {**common("cvat", "Correct pose in CVAT", "Correct or draw this person's 13-point skeleton.",
                                                          "Keypoints disagree with the 3D pose beyond the reprojection gate.", {"max reprojection px": 18.5}, None),
                                                 "cvat_url": f"http://localhost:8080/tasks/{312 + n}/jobs/{540 + n}",
                                                 "instructions": "Open the job, fix the highlighted skeletons, save, and mark the job complete."}, priority=0.5)

    # operator decision (no images)
    add("operator_decision", "demo/operator/0", {**common("decision", "Camera felix-5 file layout", "The engine needs a decision about the input data or configuration.",
                                                          "Two overlapping files for felix-5 differ in 3 frames at the seam.", {"seam frames": 3}, None),
                                                 "panels": [], "options": [{"value": "trust_main", "label": "Trust main channel", "hotkey": "1"},
                                                                          {"value": "trim_overlap", "label": "Trim overlap", "hotkey": "2"}]}, priority=10)
    return created


# ----------------------------------------------------------------------------- state
def _insert_round(db, number, start, end, trigger, summary):
    db.execute("INSERT INTO rounds(round, started, finished, trigger, summary) VALUES (?, ?, ?, ?, ?)",
               (number, start, end, trigger, json.dumps(summary) if summary is not None else "{}"))


def fake_state(workspace: Workspace, *, running: bool = False) -> None:
    db, engine, now = workspace.db, Engine(workspace), time.time()
    cams = len(workspace.dataset.cameras)
    base = now - 6 * 3600
    metrics = {
        "ingest": {"cameras": cams, "parts": cams * 2, "frames": 35000},
        "detect": {"frames": 35099 * cams, "detections": 1_204_331, "fps": 41.7},
        "calibration": {"cameras_checked": cams, "shifted_cameras": 2, "max_shift_px": 17.4},
        "align": {"cameras": cams, "offset_span_s": 0.82, "unstable_cameras": 1},
        "tracklets": {"tracklets": 412, "mean_length_frames": 38.4},
        "solve": {"tracks": 14, "association_pairs": 96, "coverage": {
            "triangulated_fraction": 0.71, "assigned_fraction": 0.83, "persons_tracked": 14, "mean_views_per_pose": 3.4, "frames_covered_ratio": 0.97}},
        "tasks": {"tasks_created": 52, "dedupe_skipped": 7},
    }
    durations = {"ingest": 41, "detect": 2480, "calibration": 95, "align": 310, "tracklets": 188, "solve": 420, "tasks": 6}

    def run_stage(name, round_number, at):
        run_id = db.start_stage(name, round_number, engine.input_hash(STAGE[name]))
        db.finish_stage(run_id, "done", metrics=metrics[name], message="ok" if name != "calibration" else "2 cameras shifted; rotation correction applied")
        db.execute("UPDATE stage_runs SET started = ?, finished = ? WHERE id = ?", (at, at + durations[name], run_id))
        return at + durations[name]

    # human constraints from earlier answers (added first so every stage run below is up to date with them)
    db.add_constraint("same", ["felix-0/12/0"], ["felix-1/12/0"], source="human", round_number=1)
    db.add_constraint("different", ["felix-0/30/1"], ["felix-2/30/0"], source="human", round_number=1)
    db.add_constraint("not_person", ["felix-3/44/2"], None, source="human", round_number=1)
    db.add_constraint("alignment", ["felix-4"], [], source="human", round_number=1, detail={"offset_frames": -2})
    db.add_constraint("split", ["felix-0/80/0"], [], source="human", round_number=2)
    db.add_constraint("same", ["felix-5/60/0"], ["felix-6/60/1"], source="human", round_number=2)
    # round 0: everything once
    at = base
    _insert_round(db, 0, base, base + 3700, "init", {
        "stages_run": ["ingest", "detect", "calibration", "align", "tracklets", "solve", "tasks"], "answers_applied": 0, "stopped": None,
        "open_by_kind": {"calibration": 3, "alignment": 4, "tracklet_purity": 40, "association": 62, "identity_link": 25, "unassigned": 33, "pose_ownership": 18, "coverage": 21},
        "open_total": 206, "open_severity_total": 912.0, "shrinking": None, "tasks_open": 71})
    for name in ("ingest", "detect", "calibration", "align", "tracklets", "solve", "tasks"):
        at = run_stage(name, 0, at)
    _insert_round(db, 1, base + 7200, base + 7800, "human_answers", {
        "stages_run": ["solve", "tasks"], "answers_applied": 48, "stopped": None,
        "open_by_kind": {"calibration": 1, "alignment": 2, "tracklet_purity": 22, "association": 31, "identity_link": 14, "unassigned": 20, "pose_ownership": 7, "coverage": 16},
        "open_total": 113, "open_severity_total": 540.0, "shrinking": True, "tasks_open": 38})
    at = base + 7200
    for name in ("solve", "tasks"):
        at = run_stage(name, 1, at)
    # a human 'break' answer re-opens identity questions: this round did not shrink the set
    _insert_round(db, 2, base + 14400, base + 15000, "human_answers", {
        "stages_run": ["solve", "tasks"], "answers_applied": 31, "stopped": None,
        "open_by_kind": {"calibration": 1, "alignment": 2, "tracklet_purity": 30, "association": 28, "identity_link": 17, "unassigned": 19, "pose_ownership": 6, "coverage": 17},
        "open_total": 120, "open_severity_total": 561.0, "shrinking": False, "tasks_open": 37})
    db.log("round 2 did not shrink the unknown set (113 -> 120); more human input or a different lever is needed", level="warning", stage="solve")
    at = base + 14400
    for name in ("solve", "tasks"):
        at = run_stage(name, 2, at)

    # unknowns consistent with the last round
    final = {"calibration": 1, "alignment": 2, "tracklet_purity": 30, "association": 28, "identity_link": 17, "unassigned": 19, "pose_ownership": 6, "coverage": 17}
    whys = {"association": "two candidates within margin", "tracklet_purity": "boxes overlapped at frame 212", "identity_link": "track ends inside the room",
            "unassigned": "single-view tracklet", "pose_ownership": "near-identical poses", "coverage": "no detection inside projected view",
            "calibration": "mount shift 17 px", "alignment": "lag unstable over 10 min"}
    resolved_extra = {"association": 40, "tracklet_purity": 12, "identity_link": 9, "unassigned": 14, "pose_ownership": 12, "coverage": 4, "calibration": 2, "alignment": 2}
    for kind, open_n in final.items():
        items = [{"subject": f"{kind}/{i}", "severity": float(5 + (i * 7) % 40), "detail": {"why": whys[kind], "would_resolve": "human task"}}
                 for i in range(open_n + resolved_extra[kind])]
        db.report_unknowns("solve", kind, items, 0)
        for row in db.unknowns("open", kind)[open_n:]:
            if row["id"] % 5 == 0:
                db.close_unknown(row["id"], 1, "undeterminable:occluded in all views", "undeterminable")
            else:
                db.close_unknown(row["id"], 1 + row["id"] % 2, "human:3" if row["id"] % 2 else "machine:solve")

    # earlier answered/applied tasks (history)
    for n in range(6):
        task_id = db.add_task("cross_view_match", f"demo/old/{n}", {"ui": "reference_candidates", "title": "old", "question": "q", "why": "", "evidence": {},
                                                                 "time_seconds": 1.0, "reference": {"id": "A", "caption": "", "panels": []}, "candidates": []},
                              priority=1, round_number=0)
        db.answer_task(task_id, {"reference_ok": True, "candidates": {}, "none_of_these": False, "problems": {}, "notes": ""}, "ian")
        db.set_task_status(task_id, "applied", applied_round=1 + n % 2)
    for kind, text in [("info", "round 2 started (answers applied: 31)"), ("info", "[solve] unknowns[association]: 28 open (+9 / -14)"),
                       ("info", "[tasks] 52 tasks created, 7 deduplicated"), ("warning", "[align] felix-4 offset unstable over 10 min"),
                       ("info", "[detect] 35099 frames x 7 cameras in 41m")]:
        db.execute("INSERT INTO events(time, level, stage, message) VALUES (?, ?, ?, ?)", (now - 600 - 60 * len(text) % 500, kind, "", text))
    if running:
        run_id = db.start_stage("solve", 3, "running-demo")
        _insert_round(db, 3, now - 400, None, "human_answers", {})
        db.execute("UPDATE rounds SET finished = NULL WHERE round = 3")
        db.execute("UPDATE stage_runs SET started = ? WHERE id = ?", (now - 400, run_id))
        progress = workspace.path("logs", "solve_progress.json")
        progress.parent.mkdir(parents=True, exist_ok=True)
        progress.write_text(json.dumps({"done": 1800, "total": 4200, "updated": now, "message": "associating window 18/42"}), encoding="utf-8")


def build(site: Site | None = None, *, running: bool = False) -> Workspace:
    site = site or load_site()
    root = site.data_root / DEMO_ID
    if root.resolve() == SOURCE_DATASET.resolve():
        raise RuntimeError("refusing to touch the real dataset")
    root.mkdir(parents=True, exist_ok=True)
    for name in ("engine.sqlite3", "engine.sqlite3-wal", "engine.sqlite3-shm"):
        (root / name).unlink(missing_ok=True)
    for sub in ("detections", "logs", "manifests"):  # frame_index is kept: it caches ffprobe results
        shutil.rmtree(root / sub, ignore_errors=True)
    dataset = json.loads((SOURCE_DATASET / "dataset.json").read_text(encoding="utf-8"))
    dataset["dataset_id"] = DEMO_ID
    (root / "dataset.json").write_text(json.dumps(dataset, indent=1), encoding="utf-8")
    for camera in dataset["cameras"]:
        source = SMOKE_DETECT / camera / "000000.npz"
        if source.is_file():
            target = root / "detections" / "raw" / camera
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / "000000.npz")
    workspace = Workspace(DEMO_ID, site)
    workspace.db  # create schema
    fake_state(workspace, running=running)
    make_tasks(workspace, round_number=2, rng=random.Random(11))
    workspace.db.close()
    return workspace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--running", action="store_true", help="also fake a solve stage that is currently running")
    arguments = parser.parse_args()
    workspace = build(running=arguments.running)
    print(f"demo workspace ready: {workspace.root}")
    print("serve it with: python -m data_engine.web --port 8610   then open http://127.0.0.1:8610/d/" + DEMO_ID)


if __name__ == "__main__":
    main()
