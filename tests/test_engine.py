"""State machine tests with a small fake stage registry (no real stage runs)."""

import sys
import types

import pytest

from data_engine import engine as engine_module
from data_engine.config import Site
from data_engine.engine import Engine, Stage, StageResult
from data_engine.workspace import Workspace


def fake_stages():
    return [
        Stage("ingest", "Ingest", "", "x", owns=("stream_layout",)),
        Stage("detect", "Detect", "", "x", requires=("ingest",)),
        Stage("calibration", "Calibration", "", "x", requires=("ingest",), constraints=("calibration",),
              owns=("calibration",)),
        Stage("solve", "Solve", "", "x", requires=("detect", "calibration"), constraints=("same", "different"),
              owns=("association",), every_round=True),
        Stage("tasks", "Tasks", "", "x", requires=("solve",), every_round=True),
        Stage("publish", "Publish", "", "x", requires=("solve",)),
    ]


class Harness:
    """Records calls; per-stage behaviour is a callable(context) -> StageResult."""

    def __init__(self):
        self.calls = []
        self.behavior = {}

    def run(self, stage, context):
        self.calls.append(stage.name)
        fn = self.behavior.get(stage.name)
        return fn(context) if fn else StageResult(message=f"{stage.name} ok")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    stages = fake_stages()
    monkeypatch.setattr(engine_module, "STAGES", stages)
    monkeypatch.setattr(engine_module, "STAGE", {s.name: s for s in stages})
    harness = Harness()
    monkeypatch.setattr(Stage, "run", lambda self, context: harness.run(self, context))

    # Answers become 'same' constraints (the real apply module belongs to the tasks package).
    def apply_task_answer(workspace, task, round_number):
        workspace.db.add_constraint("same", ["c0/1/0"], ["c1/1/0"], source="human", round_number=round_number, task_id=task["id"])
        for unknown_id in workspace.db.task_unknown_ids(task["id"]):
            workspace.db.close_unknown(unknown_id, round_number, f"human:{task['id']}")
    module = types.ModuleType("data_engine.tasks.apply")
    module.apply_task_answer = apply_task_answer
    monkeypatch.setitem(sys.modules, "data_engine.tasks.apply", module)

    workspace = Workspace("unit_20250101000000_20250101010000", Site({"data_root": str(tmp_path)}))
    return types.SimpleNamespace(engine=Engine(workspace), workspace=workspace, db=workspace.db, harness=harness, stages=stages)


def unknowns(subjects, kind="association"):
    return StageResult(unknowns={kind: [{"subject": s, "severity": 2.0, "detail": {}} for s in subjects]})


def make_tasks_for_open_unknowns(context):
    for unknown in context.db.unknowns(kind="association"):
        context.db.add_task("cross_view_match", unknown["subject"], {"title": "t"}, priority=1,
                            round_number=context.round, unknown_ids=[unknown["id"]])
    return StageResult()


def states(env):
    return {s["name"]: s["status"] for s in env.engine.status()["stages"]}


# --------------------------------------------------------------------------- freshness
def test_first_round_runs_everything_but_publish(env):
    summary = env.engine.advance()
    assert env.harness.calls == ["ingest", "detect", "calibration", "solve", "tasks"]
    assert summary["stages_run"] == env.harness.calls and summary["stopped"] is None
    assert states(env) == {"ingest": "done", "detect": "done", "calibration": "done", "solve": "done",
                           "tasks": "done", "publish": "pending"}
    assert env.db.rounds()[0]["trigger"] == "init"


def test_done_stage_is_not_rerun(env):
    env.engine.advance()
    env.harness.calls.clear()
    summary = env.engine.advance()
    assert env.harness.calls == [] and summary["stages_run"] == []
    assert env.db.rounds()[1]["trigger"] == "manual"


def test_force_reruns_stage_and_makes_dependants_stale(env):
    env.engine.advance()
    env.harness.calls.clear()
    env.engine.advance(force=("detect",))
    # detect reran; its dependants (solve, then tasks) saw a new upstream run id; calibration did not.
    assert env.harness.calls == ["detect", "solve", "tasks"]


def test_upstream_rerun_marks_dependants_stale_in_status(env):
    env.engine.advance()
    env.engine.run_stage(env.stages[1], 1)  # rerun detect outside a round
    assert states(env)["solve"] == "stale" and states(env)["calibration"] == "done" and states(env)["detect"] == "done"


def test_constraint_of_listed_kind_makes_stage_and_dependants_stale(env):
    env.engine.advance()
    env.harness.calls.clear()
    env.db.add_constraint("calibration", ["c1"], None, source="human", round_number=0, detail={"choice": "use_standard"})
    assert states(env)["calibration"] == "stale" and states(env)["detect"] == "done"
    env.engine.advance()
    assert env.harness.calls == ["calibration", "solve", "tasks"]


def test_constraint_of_unlisted_kind_changes_nothing(env):
    env.engine.advance()
    env.harness.calls.clear()
    env.db.add_constraint("alignment", ["c1"], None, source="human", round_number=0, detail={"offset_seconds": 1.0})
    assert set(states(env).values()) == {"done", "pending"}
    env.engine.advance()
    assert env.harness.calls == []


def test_superseding_a_constraint_changes_the_digest(env):
    env.engine.advance()
    old = env.db.add_constraint("same", ["a"], ["b"], source="human", round_number=0)
    env.engine.advance()
    env.harness.calls.clear()
    new = env.db.add_constraint("different", ["a"], ["b"], source="human", round_number=1)
    env.db.supersede_constraint(old, new)
    env.engine.advance()
    assert "solve" in env.harness.calls


def test_every_round_stage_reruns_after_answers_applied(env):
    env.harness.behavior["solve"] = lambda c: unknowns(["u1"])
    env.harness.behavior["tasks"] = make_tasks_for_open_unknowns
    env.engine.advance()
    [task] = env.db.tasks(status="open")
    env.db.answer_task(task["id"], {"reference_ok": True}, "ian")
    env.harness.calls.clear()
    summary = env.engine.advance()
    assert summary["answers_applied"] == 1
    assert env.harness.calls == ["solve", "tasks"]  # the answer's 'same' constraint also made solve stale
    assert env.db.task(task["id"])["status"] == "applied" and env.db.task(task["id"])["applied_round"] == 1


def test_until_stops_after_stage(env):
    summary = env.engine.advance(until="detect")
    assert env.harness.calls == ["ingest", "detect"] and summary["stopped"] is None
    assert env.engine.status()["phase"] == "machine_pending"


def test_until_stage_already_done_still_stops(env):
    env.engine.advance(until="ingest")
    env.harness.calls.clear()
    env.engine.advance(until="ingest")
    assert env.harness.calls == []


def test_stage_waits_for_unfinished_requirement(env):
    env.harness.behavior["ingest"] = lambda c: StageResult(status="blocked", message="layout unresolved")
    summary = env.engine.advance()
    assert env.harness.calls == ["ingest"]
    assert summary["stopped"] == "ingest: blocked - layout unresolved"


# --------------------------------------------------------------------------- failure
def test_failing_stage_stops_round_and_shows_failed(env):
    def boom(context):
        raise ValueError("model file missing")
    env.harness.behavior["detect"] = boom
    summary = env.engine.advance()
    assert env.harness.calls == ["ingest", "detect"]
    assert summary["stopped"] == "detect: failed - model file missing"
    status = env.engine.status()
    detect = next(s for s in status["stages"] if s["name"] == "detect")
    assert detect["status"] == "failed" and "ValueError: model file missing" in detect["message"]
    assert status["phase"] == "blocked" and "Detect" in status["phase_detail"]
    assert any(e["level"] == "error" for e in status["events"])


def test_failed_stage_is_retried_next_round(env):
    attempts = []

    def flaky(context):
        attempts.append(1)
        if len(attempts) == 1:
            return StageResult(status="failed", message="gpu busy")
        return StageResult()
    env.harness.behavior["detect"] = flaky
    env.engine.advance()
    assert states(env)["detect"] == "failed" and "solve" not in env.harness.calls
    env.engine.advance()
    assert states(env)["detect"] == "done" and states(env)["solve"] == "done"


def test_stage_reporting_unowned_unknown_kind_raises(env):
    env.harness.behavior["detect"] = lambda c: unknowns(["x"], kind="association")  # detect owns nothing
    with pytest.raises(RuntimeError, match="does not own"):
        env.engine.advance()


def test_unowned_kind_does_not_leave_stage_running(env):
    env.harness.behavior["detect"] = lambda c: unknowns(["x"], kind="association")
    with pytest.raises(RuntimeError):
        env.engine.advance()
    assert env.db.latest_stage_run("detect")["status"] != "running"


def test_unknowns_of_owned_kind_are_recorded_and_closed(env):
    env.harness.behavior["calibration"] = lambda c: StageResult(unknowns={"calibration": [
        {"subject": "calibration:c1", "severity": 5, "detail": {"x": 1}}]})
    env.engine.advance()
    assert [u["subject"] for u in env.db.unknowns(kind="calibration")] == ["calibration:c1"]
    env.harness.behavior["calibration"] = lambda c: StageResult(unknowns={"calibration": []})
    env.engine.advance(force=("calibration",))
    assert env.db.unknowns(kind="calibration") == []
    assert env.db.unknowns(status="resolved", kind="calibration")[0]["resolution"] == "machine:calibration"


def test_manifest_written_with_inputs_and_outputs(env):
    env.harness.behavior["ingest"] = lambda c: StageResult(inputs={"sha": "abc"}, outputs={"dataset": "dataset.json"})
    env.engine.advance(until="ingest")
    run = env.db.latest_stage_run("ingest")
    import json
    manifest = json.loads(env.workspace.path(run["manifest"]).read_text(encoding="utf-8"))
    assert manifest["inputs"] == {"sha": "abc"} and manifest["outputs"] == {"dataset": "dataset.json"}
    assert manifest["stage"] == "ingest" and manifest["status"] == "done"


# --------------------------------------------------------------------------- round summaries
def test_round_summary_records_open_counts(env):
    env.harness.behavior["solve"] = lambda c: unknowns(["u1", "u2", "u3"])
    summary = env.engine.advance()
    assert summary["open_by_kind"] == {"association": 3}
    assert summary["open_total"] == 3 and summary["open_severity_total"] == 6.0
    assert summary["shrinking"] is None  # first round has nothing to compare with
    assert env.db.rounds()[-1]["summary"]["open_total"] == 3


def test_shrinking_flag_follows_the_open_set(env):
    env.harness.behavior["solve"] = lambda c: unknowns(["u1", "u2", "u3"])
    env.engine.advance()
    summary = env.engine.advance(force=("solve",))  # same unknowns again: no progress
    assert summary["shrinking"] is False
    assert any(e["level"] == "warning" and "did not shrink" in e["message"] for e in env.engine.status()["events"])
    env.harness.behavior["solve"] = lambda c: unknowns(["u1"])
    assert env.engine.advance(force=("solve",))["shrinking"] is True
    env.harness.behavior["solve"] = lambda c: unknowns([])
    assert env.engine.advance(force=("solve",))["shrinking"] is True


def test_shrinking_not_judged_when_solve_did_not_run(env):
    env.harness.behavior["solve"] = lambda c: unknowns(["u1"])
    env.engine.advance()
    assert env.engine.advance()["shrinking"] is None


def test_summary_counts_open_tasks(env):
    env.harness.behavior["solve"] = lambda c: unknowns(["u1", "u2"])
    env.harness.behavior["tasks"] = make_tasks_for_open_unknowns
    assert env.engine.advance()["tasks_open"] == 2


# --------------------------------------------------------------------------- phases
def test_phase_transitions_end_to_end(env):
    env.harness.behavior["solve"] = lambda c: unknowns(["u1"]) if not c.db.constraints("same") else unknowns([])
    env.harness.behavior["tasks"] = make_tasks_for_open_unknowns

    status = env.engine.status()
    assert status["phase"] == "new" and status["round"] == -1
    assert status["next_actions"][0]["text"].startswith("Run Ingest")

    env.engine.advance(until="ingest")
    assert env.engine.status()["phase"] == "machine_pending"

    env.engine.advance()
    status = env.engine.status()
    assert status["phase"] == "waiting_for_human" and status["tasks"]["open_total"] == 1
    assert status["unknowns"]["open_total"] == 1
    assert any(a["kind"] == "human" for a in status["next_actions"])

    [task] = env.db.tasks(status="open")
    env.db.answer_task(task["id"], {"reference_ok": True, "candidates": {}}, "ian")
    status = env.engine.status()
    assert status["phase"] == "answers_pending" and status["tasks"]["answered_pending"] == 1

    env.engine.advance()
    status = env.engine.status()
    assert status["phase"] == "ready_to_publish" and status["unknowns"]["open_total"] == 0
    assert [a["text"] for a in status["next_actions"]] == ["Publish label sets"]

    env.engine.run_stage(engine_module.STAGE["publish"], env.db.current_round())
    assert env.engine.status()["phase"] == "published"


def test_phase_machine_pending_when_unknowns_open_without_tasks(env):
    env.harness.behavior["solve"] = lambda c: unknowns(["u1"])
    env.engine.advance()
    status = env.engine.status()
    assert status["phase"] == "machine_pending" and "without tasks" in status["phase_detail"]


def test_phase_running_while_a_stage_run_is_open(env):
    env.engine.advance()
    env.db.start_stage("detect", 1, "hash")
    status = env.engine.status()
    assert status["phase"] == "running" and "Detect" in status["phase_detail"]


def test_new_constraint_after_publish_makes_solve_stale(env):
    env.engine.advance()
    env.engine.run_stage(engine_module.STAGE["publish"], 0)
    assert env.engine.status()["phase"] == "published"
    env.db.add_constraint("same", ["a"], ["b"], source="human", round_number=1)
    status = env.engine.status()
    assert states(env)["solve"] == "stale" and status["phase"] == "machine_pending"
