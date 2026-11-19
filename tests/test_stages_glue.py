"""Stage glue (ingest/detect/calibration/align), run lock and CLI, with every worker subprocess faked."""

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from data_engine import cli, engine as engine_module, runlock
from data_engine.config import Site
from data_engine.dataset import Camera, Dataset, Part
from data_engine.engine import Context, Engine, Stage, StageResult
from data_engine.stages import align, calibration, detect, ingest
from data_engine.workspace import Workspace, file_sha256

DATASET_ID = "unit_20250101000000_20250101010000"


# --------------------------------------------------------------------------- fixtures
def make_dataset(cameras=("c0", "c1", "c2"), frames=5, **per_camera):
    dataset = Dataset(DATASET_ID, "unit", "20250101000000", "20250101010000")
    for index, camera_id in enumerate(cameras):
        extra = per_camera.get(camera_id, {})
        dataset.cameras[camera_id] = Camera(
            camera_id=camera_id, calibration_id=extra.get("calibration_id", str(index)), width=64, height=48, fps=10.0,
            parts=[Part(f"{camera_id}.mp4", 0, extra.get("frames", frames))], issues=extra.get("issues", []),
            notes=extra.get("notes", []))
    return dataset


def standard_calibration(ids=("0", "1", "2")):
    return {"padding": [0, 160], "scale": 0.625, "output_resolution": [1280, 960], "canvas": "x",
            "cameras": {i: {"K": [[900, 0, 640], [0, 900, 480], [0, 0, 1]], "D": [0, 0, 0, 0], "rvec": [0.1, 0.2, 0.3],
                            "tvec": [1.0, 2.0, 3.0], "solver": "s"} for i in ids}}


class FakeSubprocess:
    """Stands in for the `subprocess` module inside data_engine.stages.ingest."""
    STDOUT = subprocess.STDOUT

    def __init__(self):
        self.commands = []
        self.handlers = {}  # worker script name -> fn(argv dict, stdout) -> returncode

    def run(self, command, stdout=None, stderr=None, env=None, cwd=None, check=False):
        command = [str(c) for c in command]
        self.commands.append(command)
        script = Path(command[1]).name
        args = {a: (command[i + 1] if i + 1 < len(command) and not command[i + 1].startswith("--") else True)
                for i, a in enumerate(command) if a.startswith("--")}
        stdout.write(f"fake {script} output\n")
        return types.SimpleNamespace(returncode=self.handlers[script](args, stdout))

    def calls(self, script):
        return [c for c in self.commands if Path(c[1]).name == script]


@pytest.fixture()
def env(tmp_path, monkeypatch):
    calibration_file = tmp_path / "standard.json"
    calibration_file.write_text(json.dumps(standard_calibration()), encoding="utf-8")
    models = {name: tmp_path / f"{name}.onnx" for name in ("yolo_onnx", "vitpose_onnx")}
    for path in models.values():
        path.write_bytes(path.name.encode())
    site = Site({"data_root": str(tmp_path / "root"), "repos": {}, "interpreters": {"data_management": "dm-python"},
                 "video_root": "V:/videos", "calibration": str(calibration_file),
                 "models": {k: str(v) for k, v in models.items()}})
    workspace = Workspace(DATASET_ID, site)
    fake = FakeSubprocess()
    monkeypatch.setattr(ingest, "subprocess", fake)  # stages share ingest.run_worker
    return types.SimpleNamespace(workspace=workspace, site=site, fake=fake, tmp=tmp_path, calibration_file=calibration_file,
                                 db=workspace.db)


def context(env, stage, round_number=0):
    return Context(env.workspace, round_number, stage)


def install_dataset(env, dataset):
    dataset.save(env.workspace.dataset_json)
    env.workspace.__dict__.pop("dataset", None)


def write_shard(out, camera, start, frames, per_frame=1):
    directory = out / camera
    directory.mkdir(parents=True, exist_ok=True)
    n = len(frames)
    np.savez_compressed(directory / f"{start:06d}.npz", frames=np.asarray(frames, np.int32),
                        counts=np.full(n, per_frame, np.int32), boxes=np.zeros((n * per_frame, 5), np.float32),
                        keypoints=np.zeros((n * per_frame, 17, 3), np.float32))


# --------------------------------------------------------------------------- ingest
def test_ingest_reports_stream_layout_unknowns_and_metrics(env):
    cameras = make_dataset(("c0", "c1", "c2"), frames=100,
                           c1={"issues": ["parts overlap not verified"], "frames": 40,
                               "notes": ["continuation without overlap evidence: b.mp4"]},
                           c2={"calibration_id": None, "notes": ["catalog stream mixed main and sub channel files; rebuilt from channel 0 only"]})
    stale = make_dataset(("old",))
    install_dataset(env, stale)
    assert list(env.workspace.dataset.cameras) == ["old"]  # cached_property loaded

    def worker(args, stdout):
        assert Path(args["--out"]) == env.workspace.dataset_json
        cameras.save(Path(args["--out"]))
        return 0
    env.fake.handlers["ingest_dm.py"] = worker
    result = ingest.run(context(env, "ingest"))

    assert result.status == "done"
    command = env.fake.calls("ingest_dm.py")[0]
    assert command[0] == "dm-python" and command[command.index("--dataset") + 1] == DATASET_ID
    assert command[command.index("--video-root") + 1] == "V:/videos"
    assert list(env.workspace.dataset.cameras) == ["c0", "c1", "c2"]  # cache invalidated
    assert result.metrics["cameras"] == 3 and result.metrics["total_frames"] == 240
    assert result.metrics["frames_per_camera"] == {"c0": 100, "c1": 40, "c2": 100}
    assert result.metrics["cameras_without_calibration"] == ["c2"]
    assert set(result.metrics["notes"]) == {"c1", "c2"}
    [unknown] = result.unknowns["stream_layout"]
    assert unknown["subject"] == "stream_layout:c1" and unknown["severity"] == 40.0
    assert unknown["detail"]["issues"] == ["parts overlap not verified"] and "what_would_resolve" in unknown["detail"]
    assert result.inputs["dataset_json_sha256"] == file_sha256(env.workspace.dataset_json)
    assert env.workspace.path("logs", "ingest.log").read_text(encoding="utf-8").count("fake ingest_dm.py output") == 1


def test_ingest_issue_camera_without_frames_gets_severity_one(env):
    ds = make_dataset(("c0",), c0={"issues": ["no resolved chunk layout"], "frames": 0})
    env.fake.handlers["ingest_dm.py"] = lambda args, out: ds.save(Path(args["--out"])) or 0
    [unknown] = ingest.run(context(env, "ingest")).unknowns["stream_layout"]
    assert unknown["severity"] == 1.0


def test_ingest_clean_dataset_reports_empty_open_set(env):
    ds = make_dataset()
    env.fake.handlers["ingest_dm.py"] = lambda args, out: ds.save(Path(args["--out"])) or 0
    result = ingest.run(context(env, "ingest"))
    assert result.unknowns == {"stream_layout": []}  # complete (empty) set closes earlier unknowns


def test_ingest_failure_includes_log_tail(env):
    def worker(args, stdout):
        stdout.write("Traceback...\nKeyError: 'catalog'\n")
        return 1
    env.fake.handlers["ingest_dm.py"] = worker
    result = ingest.run(context(env, "ingest"))
    assert result.status == "failed" and "code 1" in result.message and "KeyError: 'catalog'" in result.message


# --------------------------------------------------------------------------- detect
def test_detect_runs_worker_and_verifies_coverage(env):
    install_dataset(env, make_dataset(("c0", "c1"), frames=6))

    def worker(args, stdout):
        assert Path(args["--dataset-json"]) == env.workspace.dataset_json
        for camera in ("c0", "c1"):
            write_shard(Path(args["--out"]), camera, 0, range(6), per_frame=2)
        return 0
    env.fake.handlers["detect_trt.py"] = worker
    result = detect.run(context(env, "detect"))
    assert result.status == "done", result.message
    command = env.fake.calls("detect_trt.py")[0]
    assert Path(command[command.index("--yolo") + 1]).name == "yolo_onnx.onnx"
    assert Path(command[command.index("--progress") + 1]) == env.workspace.path("logs", "detect_progress.json")
    assert result.metrics["frames"] == 12 and result.metrics["detections"] == 24
    assert result.metrics["detections_per_frame"] == 2.0
    assert result.metrics["per_camera"]["c0"] == {"frames": 6, "covered": 6, "detections": 12, "detections_per_frame": 2.0}
    assert result.metrics["frames_processed"] == 12 and result.metrics["wall_seconds"] >= 0
    assert result.inputs["yolo_onnx_sha256"] == file_sha256(env.site.model("yolo_onnx"))
    assert result.outputs == {"detections": "detections/raw"}


def test_detect_skips_worker_when_every_shard_exists(env):
    install_dataset(env, make_dataset(("c0", "c1"), frames=6))
    for camera in ("c0", "c1"):
        write_shard(env.workspace.detections_dir, camera, 0, range(6))
    env.fake.handlers["detect_trt.py"] = lambda a, o: pytest.fail("worker must not run")
    result = detect.run(context(env, "detect"))
    assert result.status == "done" and env.fake.calls("detect_trt.py") == []
    assert result.metrics["wall_seconds"] == 0 and "reused" in result.message


def test_detect_resumes_with_partial_shards(env):
    install_dataset(env, make_dataset(("c0", "c1"), frames=6))
    write_shard(env.workspace.detections_dir, "c0", 0, range(6))  # c0 finished earlier, c1 not
    seen = {}

    def worker(args, stdout):
        seen["called"] = True
        write_shard(Path(args["--out"]), "c1", 0, range(6))
        return 0
    env.fake.handlers["detect_trt.py"] = worker
    result = detect.run(context(env, "detect"))
    assert seen and result.status == "done" and result.metrics["frames_processed"] == 6


def test_detect_fails_when_coverage_incomplete(env):
    install_dataset(env, make_dataset(("c0", "c1"), frames=6))

    def worker(args, stdout):
        write_shard(Path(args["--out"]), "c0", 0, range(6))
        write_shard(Path(args["--out"]), "c1", 0, range(4))
        return 0
    env.fake.handlers["detect_trt.py"] = worker
    result = detect.run(context(env, "detect"))
    assert result.status == "failed" and "coverage below 100%" in result.message and "c1 4/6" in result.message


def test_detect_fails_on_worker_error_and_missing_model(env):
    install_dataset(env, make_dataset(("c0",), frames=6))
    env.fake.handlers["detect_trt.py"] = lambda a, o: 1
    result = detect.run(context(env, "detect"))
    assert result.status == "failed" and "code 1" in result.message
    env.site.model("vitpose_onnx").unlink()
    assert "model file missing" in detect.run(context(env, "detect")).message


def test_detect_blocked_by_unresolved_stream_layout(env):
    install_dataset(env, make_dataset(("c0", "c1"), c1={"issues": ["x"]}))
    env.fake.handlers["detect_trt.py"] = lambda a, o: pytest.fail("worker must not run")
    result = detect.run(context(env, "detect"))
    assert result.status == "blocked" and "c1" in result.message


# --------------------------------------------------------------------------- calibration
def mount_check_worker(env, cameras, dataset=None):
    """Fake calibration_dm: writes mount_check.json (same shape as the real worker)."""
    def worker(args, stdout):
        out = Path(args["--out"])
        out.mkdir(parents=True, exist_ok=True)
        result = {"time_range": ["unit"], "calibration": args["--calibration"], "cameras": cameras,
                  "data_engine": {"dataset": DATASET_ID, "main_channel_rebuilt": [],
                                  "dataset_json_sha256": file_sha256(Path(args["--dataset-json"])),
                                  "calibration_sha256": file_sha256(Path(args["--calibration"])),
                                  "samples": int(args["--samples"])}}
        (out / "mount_check.json").write_text(json.dumps(result), encoding="utf-8")
        return 0
    env.fake.handlers["calibration_dm.py"] = worker


CHECKS = {
    "c0": {"camera": "c0", "calibration_id": "0", "status": "unchanged", "median_shift_px": 0.4},
    "c1": {"camera": "c1", "calibration_id": "1", "status": "shifted", "median_shift_px": 2.75, "rotation_deg": 0.12,
           "residual_px": 0.6, "corrected_rvec": [0.11, 0.21, 0.31], "corrected_tvec": [1.1, 2.1, 3.1]},
    "c2": {"camera": "c2", "calibration_id": "2", "status": "inconclusive", "reason": "too few matches"},
}


def run_calibration(env, round_number=0):
    return calibration.run(context(env, "calibration", round_number))


def written(env):
    return json.loads(env.workspace.path("calibration", "calibration.json").read_text(encoding="utf-8"))


def test_calibration_without_decisions_applies_nothing_and_asks(env):
    install_dataset(env, make_dataset(frames=50))
    mount_check_worker(env, CHECKS)
    result = run_calibration(env)
    assert result.status == "done"
    subjects = {u["subject"]: u for u in result.unknowns["calibration"]}
    assert set(subjects) == {"calibration:c1", "calibration:c2"}  # unchanged camera asks nothing
    assert subjects["calibration:c1"]["severity"] == 50.0
    c1 = subjects["calibration:c1"]["detail"]
    assert [o["value"] for o in c1["options"]] == ["use_standard", "use_corrected", "exclude_camera"]
    assert c1["mount_check"]["median_shift_px"] == 2.75 and "slipped" in c1["explanation"]
    assert [o["value"] for o in subjects["calibration:c2"]["detail"]["options"]] == ["use_standard", "exclude_camera"]
    out = written(env)
    assert out["cameras"]["1"]["rvec"] == [0.1, 0.2, 0.3]  # NEVER silently corrected
    assert {k: v for k, v in out.items() if k != "data_engine"} == standard_calibration()
    note = out["data_engine"]
    assert note["source_sha256"] == file_sha256(env.calibration_file)
    assert note["cameras"]["c1"]["status"] == "shifted" and note["cameras"]["c1"]["applied"].startswith("standard")
    assert result.metrics["per_camera"]["c1"]["median_shift_px"] == 2.75 and result.metrics["counts"]["unchanged"] == 1
    assert result.inputs["calibration_source_sha256"] == file_sha256(env.calibration_file)
    assert result.outputs["calibration"] == "calibration/calibration.json"


def test_calibration_use_corrected_applies_only_on_human_decision(env):
    install_dataset(env, make_dataset())
    mount_check_worker(env, CHECKS)
    env.db.add_constraint("calibration", ["c1"], None, source="human", round_number=0, detail={"choice": "use_corrected"})
    result = run_calibration(env)
    assert [u["subject"] for u in result.unknowns["calibration"]] == ["calibration:c2"]
    out = written(env)
    assert out["cameras"]["1"]["rvec"] == [0.11, 0.21, 0.31] and out["cameras"]["1"]["tvec"] == [1.1, 2.1, 3.1]
    assert out["cameras"]["0"]["rvec"] == [0.1, 0.2, 0.3] and out["cameras"]["1"]["K"] == standard_calibration()["cameras"]["1"]["K"]
    assert out["data_engine"]["cameras"]["c1"]["applied"] == "corrected" and result.metrics["corrected"] == ["c1"]
    assert result.inputs["human_decisions"]["c1"]["choice"] == "use_corrected"


def test_calibration_machine_constraint_is_not_a_human_decision(env):
    install_dataset(env, make_dataset())
    mount_check_worker(env, CHECKS)
    env.db.add_constraint("calibration", ["c1"], None, source="machine:auto", round_number=0, detail={"choice": "use_corrected"})
    result = run_calibration(env)
    assert "calibration:c1" in {u["subject"] for u in result.unknowns["calibration"]}
    assert written(env)["cameras"]["1"]["rvec"] == [0.1, 0.2, 0.3]


def test_calibration_use_standard_and_exclude_camera(env):
    install_dataset(env, make_dataset())
    mount_check_worker(env, CHECKS)
    env.db.add_constraint("calibration", ["c1"], None, source="human", round_number=0, detail={"choice": "use_standard"})
    env.db.add_constraint("calibration", ["c2"], None, source="human", round_number=0, detail={"choice": "exclude_camera"})
    result = run_calibration(env)
    assert result.unknowns["calibration"] == []
    out = written(env)
    assert sorted(out["cameras"]) == ["0", "1"] and out["cameras"]["1"]["rvec"] == [0.1, 0.2, 0.3]
    assert out["data_engine"]["excluded_cameras"] == ["c2"]
    assert out["data_engine"]["cameras"]["c1"]["applied"] == "standard (accepted by a human)"
    assert out["data_engine"]["cameras"]["c2"]["applied"] == "excluded" and result.metrics["excluded"] == ["c2"]


def test_calibration_latest_human_decision_wins(env):
    install_dataset(env, make_dataset())
    mount_check_worker(env, CHECKS)
    env.db.add_constraint("calibration", ["c1"], None, source="human", round_number=0, detail={"choice": "use_corrected"})
    env.db.add_constraint("calibration", ["c1"], None, source="human", round_number=1, detail={"choice": "use_standard"})
    run_calibration(env)
    assert written(env)["cameras"]["1"]["rvec"] == [0.1, 0.2, 0.3]


def test_calibration_use_corrected_without_correction_keeps_question_open(env):
    install_dataset(env, make_dataset())
    mount_check_worker(env, CHECKS)
    env.db.add_constraint("calibration", ["c2"], None, source="human", round_number=0, detail={"choice": "use_corrected"})
    result = run_calibration(env)
    assert "calibration:c2" in {u["subject"] for u in result.unknowns["calibration"]}
    assert "no correction" in written(env)["data_engine"]["cameras"]["c2"]["applied"]


def test_calibration_reuses_mount_check_when_inputs_unchanged(env):
    install_dataset(env, make_dataset())
    mount_check_worker(env, CHECKS)
    run_calibration(env)
    env.db.add_constraint("calibration", ["c1"], None, source="human", round_number=0, detail={"choice": "use_corrected"})
    run_calibration(env, 1)
    assert len(env.fake.calls("calibration_dm.py")) == 1
    command = env.fake.calls("calibration_dm.py")[0]
    assert command[command.index("--calibration") + 1] == str(env.calibration_file)
    assert command[command.index("--dataset") + 1] == DATASET_ID
    install_dataset(env, make_dataset(frames=6))  # a different dataset.json invalidates the reuse
    run_calibration(env, 2)
    assert len(env.fake.calls("calibration_dm.py")) == 2


def test_calibration_worker_failure_and_missing_source(env):
    install_dataset(env, make_dataset())
    env.fake.handlers["calibration_dm.py"] = lambda a, o: 1
    assert run_calibration(env).status == "failed"
    env.calibration_file.unlink()
    assert "not found" in run_calibration(env).message


# --------------------------------------------------------------------------- align
def report_for(cameras, *, anchor="c0", windows=10, accepted=8, span=(10.0, 300.0)):
    return {"cameras": cameras, "anchor": anchor, "unresolved_cameras": [c for c, e in cameras.items() if not e["resolved"]],
            "run": {"windows": windows, "accepted": accepted, "span_seconds": list(span)}}


def cam_entry(offset=0.0, **extra):
    base = {"resolved": offset is not None, "anchor": False, "final_offset_seconds": offset, "accepted_windows_including": 8,
            "unresolved_spans": [], "stale_spans": [], "frame_rate": 10.0, "uncertain": [], "candidate_windows": 8,
            "drift_seconds_per_hour": 0.5, "spread_mad_seconds": 0.02, "spread_mad_frames": 0.2}
    base.update(extra)
    return base


def align_worker(env, report, code=0):
    def worker(args, stdout):
        out = Path(args["--out"])
        out.mkdir(parents=True, exist_ok=True)
        for name in ("alignment.json", "provenance.json"):
            (out / name).write_text("{}", encoding="utf-8")
        (out / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return code
    env.fake.handlers["align_dm.py"] = worker


def align_env(env, frames=100, covered=100):
    install_dataset(env, make_dataset(frames=frames))
    for camera in ("c0", "c1", "c2"):
        write_shard(env.workspace.detections_dir, camera, 0, range(covered), per_frame=0)
    cal = env.workspace.path("calibration")
    cal.mkdir(parents=True, exist_ok=True)
    (cal / "calibration.json").write_text(json.dumps(standard_calibration()), encoding="utf-8")


def test_align_builds_unknowns_from_report(env):
    align_env(env)
    report = report_for({
        "c0": cam_entry(0.0, anchor=True),
        "c1": cam_entry(1.5, stale_spans=[[100.0, 145.0], [200.0, 220.0]], unresolved_spans=[[10.0, 50.0]]),
        "c2": cam_entry(None, resolved=False, unresolved_spans=[[10.0, 300.0]], accepted_windows_including=0,
                        drift_seconds_per_hour=None, spread_mad_seconds=None, spread_mad_frames=None),
        })
    report["cameras"]["c0"].update(spread_mad_frames=1.6, spread_mad_seconds=0.16)  # anchor with a wide spread
    align_worker(env, report)
    result = align.run(context(env, "align"))
    assert result.status == "done", result.message
    command = env.fake.calls("align_dm.py")[0]
    assert command[command.index("--update-seconds") + 1] == "10"  # 10 s of detections: short recording
    assert Path(command[command.index("--calibration") + 1]) == env.workspace.path("calibration", "calibration.json")
    subjects = {u["subject"]: u for u in result.unknowns["alignment"]}
    assert set(subjects) == {"alignment:c0", "alignment:c2", "alignment:c1:10.0-50.0", "alignment:c1:100.0-145.0"}
    assert subjects["alignment:c1:100.0-145.0"]["severity"] == 450.0  # 45 s * 10 fps; the 20 s span is ignored
    assert subjects["alignment:c2"]["severity"] == 100.0  # whole span, capped at the stream's 100 frames
    assert "no confirmed offset" in subjects["alignment:c2"]["detail"]["reasons"][0]
    assert "1.60 frames" in subjects["alignment:c0"]["detail"]["reasons"][0]
    assert result.metrics["per_camera"]["c1"] == {"offset_seconds": 1.5, "resolved": True, "anchor": False,
                                                  "drift_seconds_per_hour": 0.5, "spread_mad_frames": 0.2,
                                                  "accepted_windows": 8, "total_windows": 10}
    assert result.metrics["anchor"] == "c0" and result.metrics["unresolved_cameras"] == ["c2"]
    assert set(result.outputs) >= {"report.json", "alignment.json"} and result.outputs["report.json"] == "alignment/report.json"


def test_align_default_update_for_long_recordings(env):
    align_env(env, frames=4000, covered=3100)  # 310 s of detections
    align_worker(env, report_for({"c0": cam_entry(0.0, anchor=True), "c1": cam_entry(1.0)}))
    result = align.run(context(env, "align"))
    assert "--update-seconds" not in env.fake.calls("align_dm.py")[0]
    assert result.unknowns["alignment"] == []


def test_align_exit_3_is_blocked_but_keeps_outputs_and_unknowns(env):
    align_env(env)
    align_worker(env, report_for({"c0": cam_entry(0.0, anchor=True), "c1": cam_entry(None, resolved=False),
                                  "c2": cam_entry(None, resolved=False)}, accepted=0), code=3)
    result = align.run(context(env, "align"))
    assert result.status == "blocked" and "fewer than 2 cameras resolved" in result.message
    assert "report.json" in result.outputs
    assert {u["subject"] for u in result.unknowns["alignment"]} == {"alignment:c1", "alignment:c2"}


def test_align_worker_crash_and_missing_calibration_fail(env):
    align_env(env)
    env.fake.handlers["align_dm.py"] = lambda a, o: 1
    assert align.run(context(env, "align")).status == "failed"
    env.workspace.path("calibration", "calibration.json").unlink()
    assert "calibration" in align.run(context(env, "align")).message


def test_align_passes_human_constraints_through_without_applying(env):
    align_env(env)
    env.db.add_constraint("alignment", ["c1"], None, source="human", round_number=0,
                          detail={"offset_seconds": 1.4, "aligned_time": 120.0})
    align_worker(env, report_for({"c0": cam_entry(0.0, anchor=True), "c1": cam_entry(1.0)}))
    result = align.run(context(env, "align"))
    [constraint] = result.inputs["human_alignment_constraints"]
    assert constraint["camera"] == "c1" and constraint["detail"]["offset_seconds"] == 1.4
    assert result.metrics["human_alignment_constraints"] == 1 and result.metrics["human_constraints_applied"] is False
    assert "not yet applied" in result.message
    assert not any("1.4" in part for part in env.fake.calls("align_dm.py")[0])  # nothing invented


def test_alignment_unknowns_ignores_short_spans_and_clean_report():
    dataset = make_dataset(("c0", "c1"))
    report = report_for({"c0": cam_entry(0.0, anchor=True), "c1": cam_entry(1.0, stale_spans=[[10.0, 40.0]])})
    assert align.alignment_unknowns(report, dataset) == []  # exactly 30 s is not "longer than 30 s"


# --------------------------------------------------------------------------- run lock
def test_runlock_acquire_holder_release(tmp_path):
    workspace = Workspace(DATASET_ID, Site({"data_root": str(tmp_path)}))
    assert runlock.holder(workspace) is None
    assert runlock.acquire(workspace, "data-engine run x") is True
    holder = runlock.holder(workspace)
    assert holder["pid"] == os.getpid() and holder["command"] == "data-engine run x" and holder["started"] > 0
    assert runlock.acquire(workspace) is False  # held (also against this very process)
    runlock.release(workspace)
    assert runlock.holder(workspace) is None and not workspace.path("logs", "run.lock").exists()
    assert runlock.acquire(workspace) is True
    runlock.release(workspace)


def dead_pid():
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def test_runlock_takes_over_stale_lock(tmp_path):
    workspace = Workspace(DATASET_ID, Site({"data_root": str(tmp_path)}))
    path = workspace.path("logs", "run.lock")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"pid": dead_pid(), "started": 1.0, "command": "old"}), encoding="utf-8")
    assert runlock.holder(workspace) is None  # stale locks are not holders
    assert runlock.acquire(workspace, "new") is True
    assert runlock.holder(workspace)["command"] == "new"
    runlock.release(workspace)


def test_runlock_takes_over_corrupt_lock_and_respects_live_holder(tmp_path):
    workspace = Workspace(DATASET_ID, Site({"data_root": str(tmp_path)}))
    path = workspace.path("logs", "run.lock")
    path.parent.mkdir(parents=True)
    path.write_text("not json", encoding="utf-8")
    assert runlock.acquire(workspace) is True
    runlock.release(workspace)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        path.write_text(json.dumps({"pid": other.pid, "started": 1.0, "command": "other"}), encoding="utf-8")
        assert runlock.holder(workspace)["pid"] == other.pid
        assert runlock.acquire(workspace) is False
        runlock.release(workspace)  # not ours: must not remove another live process's lock
        assert path.exists()
    finally:
        other.kill()
        other.wait()
    assert runlock.pid_alive(other.pid) is False and runlock.pid_alive(os.getpid()) is True


# --------------------------------------------------------------------------- CLI
@pytest.fixture()
def cli_env(tmp_path, monkeypatch):
    root = tmp_path / "data"
    config = tmp_path / "engine.json"
    config.write_text(json.dumps({"data_root": str(root)}), encoding="utf-8")
    monkeypatch.setenv("DATA_ENGINE_CONFIG", str(config))
    return types.SimpleNamespace(root=root)


def test_cli_init_is_idempotent_and_records_created(cli_env, capsys):
    assert cli.main(["init", DATASET_ID]) == 0
    first = capsys.readouterr().out.strip()
    assert Path(first) == cli_env.root / DATASET_ID and (Path(first) / "engine.sqlite3").is_file()
    created = Workspace(DATASET_ID, Site({"data_root": str(cli_env.root)})).db.get_meta("created")
    assert created
    assert cli.main(["init", DATASET_ID]) == 0
    assert capsys.readouterr().out.strip() == first
    assert Workspace(DATASET_ID, Site({"data_root": str(cli_env.root)})).db.get_meta("created") == created


def test_cli_init_rejects_malformed_id(cli_env, capsys):
    assert cli.main(["init", "nounderscores"]) == 2


def test_cli_data_root_can_come_from_site_defaults(tmp_path, monkeypatch):
    from data_engine import config
    monkeypatch.delenv("DATA_ENGINE_CONFIG", raising=False)
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)  # no engine.local.json here
    monkeypatch.setitem(config.SITE_DEFAULTS, "data_root", str(tmp_path / "sd"))
    assert cli.main(["init", DATASET_ID]) == 0
    assert (tmp_path / "sd" / DATASET_ID / "engine.sqlite3").is_file()


def test_cli_unknown_dataset_is_refused_everywhere(cli_env, capsys):
    for command in (["status", "nope_1_2"], ["run", "nope_1_2"], ["tasks", "nope_1_2"], ["publish", "nope_1_2"]):
        assert cli.main(command) == 2, command
    assert not (cli_env.root / "nope_1_2").exists()  # refusing must not create a workspace
    assert "unknown dataset" in capsys.readouterr().err


def test_cli_list_and_status(cli_env, capsys):
    assert cli.main(["list"]) == 0 and "no workspaces" in capsys.readouterr().out
    cli.main(["init", DATASET_ID])
    capsys.readouterr()
    assert cli.main(["list"]) == 0
    line = [l for l in capsys.readouterr().out.splitlines() if DATASET_ID in l][0]
    assert "new" in line and line.split()[2] == "-1"

    assert cli.main(["status", DATASET_ID]) == 0
    text = capsys.readouterr().out
    assert "phase: new" in text and "ingest" in text and "publish" in text and "next actions" in text
    assert cli.main(["status", DATASET_ID, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["dataset_id"] == DATASET_ID and data["phase"] == "new" and len(data["stages"]) == len(engine_module.STAGES)


def test_cli_tasks_listing(cli_env, capsys):
    cli.main(["init", DATASET_ID])
    db = Workspace(DATASET_ID, Site({"data_root": str(cli_env.root)})).db
    first = db.add_task("continuity", "s1", {"title": "Same person?"}, priority=3, round_number=0)
    db.add_task("cross_view_match", "s2", {}, priority=9, round_number=0)
    done = db.add_task("continuity", "s3", {}, priority=1, round_number=0)
    db.set_task_status(done, "applied", applied_round=0)
    capsys.readouterr()
    assert cli.main(["tasks", DATASET_ID]) == 0
    out = capsys.readouterr().out
    assert out.index("cross_view_match") < out.index("continuity")  # priority order
    assert "Same person?" in out and "Cross-camera match" in out and "s3" not in out
    cli.main(["tasks", DATASET_ID, "--status", "all", "--type", "continuity", "--limit", "5"])
    out = capsys.readouterr().out
    assert "s1" in out and "s3" in out and "s2" not in out
    cli.main(["tasks", DATASET_ID, "--status", "answered"])
    assert "no tasks" in capsys.readouterr().out and first


@pytest.fixture()
def fake_engine(monkeypatch):
    stages = [Stage("ingest", "Ingest", "", "x", owns=("stream_layout",)),
              Stage("solve", "Solve", "", "x", requires=("ingest",), owns=("association",), every_round=True),
              Stage("publish", "Publish", "", "x", requires=("solve",))]
    monkeypatch.setattr(engine_module, "STAGES", stages)
    monkeypatch.setattr(engine_module, "STAGE", {s.name: s for s in stages})
    behavior = {}
    monkeypatch.setattr(Stage, "run", lambda self, ctx: behavior.get(self.name, lambda c: StageResult(message="ok"))(ctx))
    module = types.ModuleType("data_engine.tasks.apply")
    module.apply_task_answer = lambda workspace, task, round_number: None
    monkeypatch.setitem(sys.modules, "data_engine.tasks.apply", module)
    return behavior


def test_cli_run_prints_round_summary_and_releases_lock(cli_env, fake_engine, capsys):
    cli.main(["init", DATASET_ID])
    fake_engine["solve"] = lambda c: StageResult(unknowns={"association": [{"subject": "u1", "severity": 4, "detail": {}}]})
    capsys.readouterr()
    assert cli.main(["run", DATASET_ID]) == 0
    out = capsys.readouterr().out
    assert "stages run:      ingest, solve" in out and "association 1" in out and "open tasks:" in out and "round stopped:" in out
    workspace = Workspace(DATASET_ID, Site({"data_root": str(cli_env.root)}))
    assert runlock.holder(workspace) is None
    assert cli.main(["run", DATASET_ID, "--until", "ingest", "--force", "ingest"]) == 0
    assert "stages run:      ingest" in capsys.readouterr().out
    assert cli.main(["run", DATASET_ID, "--until", "bogus"]) == 2


def test_cli_run_refused_while_lock_held(cli_env, fake_engine, capsys):
    cli.main(["init", DATASET_ID])
    workspace = Workspace(DATASET_ID, Site({"data_root": str(cli_env.root)}))
    assert runlock.acquire(workspace, "other run")
    assert cli.main(["run", DATASET_ID]) == 2
    assert "already in progress" in capsys.readouterr().err
    assert runlock.holder(workspace)["command"] == "other run"  # the refused run did not steal or free the lock
    assert cli.main(["status", DATASET_ID]) == 0 and "run in progress" in capsys.readouterr().out
    runlock.release(workspace)


def test_cli_run_failure_exits_1_and_releases_lock(cli_env, fake_engine, capsys):
    cli.main(["init", DATASET_ID])

    def boom(context):
        raise RuntimeError("worker died")
    fake_engine["ingest"] = boom
    assert cli.main(["run", DATASET_ID]) == 1
    assert "ingest: failed - worker died" in capsys.readouterr().out
    assert runlock.holder(Workspace(DATASET_ID, Site({"data_root": str(cli_env.root)}))) is None


def test_cli_publish_runs_only_publish_stage(cli_env, fake_engine, capsys):
    cli.main(["init", DATASET_ID])
    ran = []
    fake_engine["publish"] = lambda c: ran.append(c.stage) or StageResult(message="3 label sets")
    assert cli.main(["publish", DATASET_ID]) == 2  # nothing has run yet
    cli.main(["run", DATASET_ID])
    capsys.readouterr()
    assert cli.main(["publish", DATASET_ID]) == 0
    assert ran == ["publish"] and "publish: done - 3 label sets" in capsys.readouterr().out
    fake_engine["publish"] = lambda c: StageResult(status="failed", message="no space")
    assert cli.main(["publish", DATASET_ID]) == 1


def test_cli_serve_imports_web_lazily(monkeypatch):
    calls = []
    module = types.ModuleType("data_engine.web.app")
    module.serve = lambda host, port: calls.append((host, port))
    monkeypatch.setitem(sys.modules, "data_engine.web.app", module)
    assert cli.main(["serve"]) == 0 and cli.main(["serve", "--host", "0.0.0.0", "--port", "9000"]) == 0
    assert calls == [("127.0.0.1", 8600), ("0.0.0.0", 9000)]
