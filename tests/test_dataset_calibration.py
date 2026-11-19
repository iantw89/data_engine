"""Dataset-scoped calibration in the engine: the stage starts from data_management's resolution, and an explicit
operator command stores a decided workspace calibration. Every worker subprocess is faked."""

import json
from pathlib import Path

import pytest

from data_engine import cli, runlock
from data_engine.engine import STAGE, Engine
from data_engine.stages import calibration
from data_engine.workspace import file_sha256

from test_stages_glue import (  # noqa: F401  (fixtures and helpers shared with the stage glue tests)
    CHECKS, DATASET_ID, cli_env, env, install_dataset, make_dataset, mount_check_worker, resolve_worker, run_calibration,
    standard_calibration, written)


def stored_calibration(env, camera="1", rvec=(0.9, 0.8, 0.7)):
    data = standard_calibration()
    data["cameras"][camera]["rvec"] = list(rvec)
    path = env.tmp / "dataset_cal.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def use_dataset_calibration(env, path, overridden=("1",)):
    env.fake.handlers["calibration_store_dm.py"] = resolve_worker(
        source="dataset", path=str(path), sha256=file_sha256(path), cameras_overridden=list(overridden))


def test_stage_starts_from_the_dataset_calibration(env):
    install_dataset(env, make_dataset())
    stored = stored_calibration(env)
    use_dataset_calibration(env, stored)
    mount_check_worker(env, {"c0": CHECKS["c0"], "c1": {**CHECKS["c1"], "status": "unchanged"}, "c2": {**CHECKS["c2"], "status": "unchanged"}})
    result = run_calibration(env)
    assert result.status == "done"
    lookup = env.fake.calls("calibration_store_dm.py")[0]
    assert lookup[lookup.index("--dataset") + 1] == DATASET_ID and lookup[lookup.index("--mode") + 1] == "resolve"
    command = env.fake.calls("calibration_dm.py")[0]
    assert command[command.index("--calibration") + 1] == str(stored)  # the mount check measures against it
    out = written(env)
    assert out["cameras"]["1"]["rvec"] == [0.9, 0.8, 0.7] and out["cameras"]["0"]["rvec"] == [0.1, 0.2, 0.3]
    assert out["data_engine"]["source_sha256"] == file_sha256(stored)
    assert out["data_engine"]["base"] == {"source": "dataset", "cameras_overridden": ["1"]}
    assert result.inputs["calibration_base"] == "dataset" and result.inputs["calibration_source_sha256"] == file_sha256(stored)
    assert result.unknowns["calibration"] == []


def test_stage_without_a_dataset_calibration_uses_the_site_default(env):
    install_dataset(env, make_dataset())
    mount_check_worker(env, CHECKS)
    result = run_calibration(env)
    assert result.inputs["calibration_base"] == "default" and result.inputs["calibration_source"] == str(env.calibration_file)
    assert written(env)["data_engine"]["base"] == {"source": "default", "cameras_overridden": []}
    # an older data_management without the store behaves the same
    env.fake.handlers["calibration_store_dm.py"] = resolve_worker(source="unavailable", reason="ImportError")
    assert run_calibration(env, 1).inputs["calibration_base"] == "default"


def test_stage_fails_when_the_store_refuses_a_stale_dataset_calibration(env):
    install_dataset(env, make_dataset())
    mount_check_worker(env, CHECKS)
    env.fake.handlers["calibration_store_dm.py"] = resolve_worker(error="the default calibration changed since ...", stale=True)
    result = run_calibration(env)
    assert result.status == "failed" and "default calibration changed" in result.message
    assert env.fake.calls("calibration_dm.py") == []  # nothing guessed, no mount check on a doubtful base


def test_use_corrected_already_carried_by_the_dataset_calibration_is_satisfied(env):
    install_dataset(env, make_dataset())
    use_dataset_calibration(env, stored_calibration(env))
    # measured against the stored (corrected) calibration the camera is no longer shifted
    mount_check_worker(env, {"c0": CHECKS["c0"], "c1": {"camera": "c1", "calibration_id": "1", "status": "unchanged", "median_shift_px": 0.3},
                             "c2": {**CHECKS["c2"], "status": "unchanged"}})
    env.db.add_constraint("calibration", ["c1"], None, source="human", round_number=0, detail={"choice": "use_corrected"})
    result = run_calibration(env)
    assert result.unknowns["calibration"] == []
    assert written(env)["data_engine"]["cameras"]["c1"]["applied"] == calibration.ALREADY_CARRIED
    assert written(env)["cameras"]["1"]["rvec"] == [0.9, 0.8, 0.7]


# ------------------------------------------------------------------ store-calibration
def decided(env, **choices):
    """Run the calibration stage through the engine with human decisions (default: use_corrected c1, accept c2)."""
    install_dataset(env, make_dataset())
    mount_check_worker(env, CHECKS)
    for camera, choice in (choices or {"c1": "use_corrected", "c2": "use_standard"}).items():
        env.db.add_constraint("calibration", [camera], None, source="human", round_number=0, detail={"choice": choice})
    return Engine(env.workspace).run_stage(STAGE["calibration"], 0)


def store_worker(env, code=0):
    seen = {}

    def worker(args, stdout):
        if args["--mode"] == "resolve":
            return resolve_worker()(args, stdout)
        seen["args"] = args
        seen["calibration"] = json.loads(Path(args["--calibration"]).read_text(encoding="utf-8"))
        seen["plan"] = json.loads(Path(args["--plan"]).read_text(encoding="utf-8"))
        return code
    env.fake.handlers["calibration_store_dm.py"] = worker
    return seen


def store_calls(env):
    return [c for c in env.fake.calls("calibration_store_dm.py") if "store" in c[c.index("--mode") + 1:c.index("--mode") + 2]]


def test_store_calibration_records_decisions_and_provenance(env):
    assert decided(env).unknowns["calibration"] == []
    seen = store_worker(env)
    message = calibration.store_calibration(env.workspace)
    assert message
    assert "--replace" not in seen["args"] and seen["args"]["--dataset"] == DATASET_ID
    assert "data_engine" not in seen["calibration"] and seen["calibration"]["cameras"]["1"]["rvec"] == [0.11, 0.21, 0.31]
    plan = seen["plan"]
    assert plan["expected_base_sha256"] == file_sha256(env.calibration_file)
    assert list(plan["reasons"]) == ["1"]
    reason = plan["reasons"]["1"]
    assert (reason["camera"], reason["decision"], reason["median_shift_px"], reason["rotation_deg"]) == ("c1", "use_corrected", 2.75, 0.12)
    provenance = plan["provenance"]
    assert provenance["engine_workspace"] == str(env.workspace.root) and provenance["round"] == 0
    assert provenance["workspace_calibration_sha256"] == file_sha256(env.workspace.path("calibration", "calibration.json"))
    assert provenance["data_engine_note"]["cameras"]["c2"]["applied"] == "standard (accepted by a human)"
    calibration.store_calibration(env.workspace, replace=True)
    assert "--replace" in store_calls(env)[-1]


def test_store_calibration_refuses_undecided_cameras(env):
    result = decided(env, c1="use_corrected")  # c2 (inconclusive) is still open
    assert [u["subject"] for u in result.unknowns["calibration"]] == ["calibration:c2"]
    store_worker(env)
    with pytest.raises(calibration.StoreRefused, match="calibration:c2"):
        calibration.store_calibration(env.workspace)
    assert store_calls(env) == []


def test_store_calibration_refuses_before_a_run_and_after_a_new_decision(env):
    install_dataset(env, make_dataset())
    store_worker(env)
    with pytest.raises(calibration.StoreRefused, match="no calibration"):
        calibration.store_calibration(env.workspace)
    decided(env)
    env.db.add_constraint("calibration", ["c1"], None, source="human", round_number=1, detail={"choice": "use_standard"})
    with pytest.raises(calibration.StoreRefused, match="stale"):
        calibration.store_calibration(env.workspace)  # the decision is not applied to calibration.json yet
    assert store_calls(env) == []


def test_store_calibration_refuses_excluded_cameras_and_pilot_slices(env):
    decided(env, c1="use_corrected", c2="exclude_camera")
    store_worker(env)
    with pytest.raises(calibration.StoreRefused, match="excluded"):
        calibration.store_calibration(env.workspace)
    from data_engine.workspace import Workspace
    with pytest.raises(calibration.StoreRefused, match="pilot"):
        calibration.store_calibration(Workspace(DATASET_ID + "@pilot", env.site))
    assert store_calls(env) == []


def test_store_calibration_reports_a_refusal_from_data_management(env):
    decided(env)
    store_worker(env, code=3)
    with pytest.raises(calibration.StoreRefused, match="exit 3"):
        calibration.store_calibration(env.workspace)


def test_nothing_stores_automatically(env):
    decided(env)
    Engine(env.workspace).run_stage(STAGE["calibration"], 1)
    assert [c[c.index("--mode") + 1] for c in env.fake.calls("calibration_store_dm.py")] == ["resolve", "resolve"]


def test_cli_store_calibration(cli_env, capsys, monkeypatch):
    assert cli.main(["store-calibration", "nope_1_2"]) == 2 and "unknown dataset" in capsys.readouterr().err
    assert cli.main(["init", DATASET_ID]) == 0
    capsys.readouterr()
    assert cli.main(["store-calibration", DATASET_ID]) == 2  # no calibration stage has run
    assert "refused" in capsys.readouterr().err
    calls = []
    monkeypatch.setattr(calibration, "store_calibration", lambda workspace, replace=False: calls.append(replace) or "stored X")
    assert cli.main(["store-calibration", DATASET_ID, "--replace"]) == 0 and calls == [True]
    assert "stored X" in capsys.readouterr().out
    workspace = cli.__dict__["_existing"](DATASET_ID)
    assert runlock.acquire(workspace, "other run")
    assert cli.main(["store-calibration", DATASET_ID]) == 2 and calls == [True]  # refused while a run holds the lock
    runlock.release(workspace)
