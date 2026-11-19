"""Unknown ledger, task and constraint semantics of data_engine.db.Database."""

import pytest

from data_engine.db import Database


@pytest.fixture()
def db(tmp_path):
    database = Database(tmp_path / "engine.sqlite3")
    yield database
    database.close()


def item(subject, severity=1.0, **detail):
    return {"subject": subject, "severity": severity, "detail": detail}


def status_of(db, kind, subject):
    rows = [u for u in db.unknowns(status=None, kind=kind) if u["subject"] == subject]
    assert len(rows) == 1
    return rows[0]


# --------------------------------------------------------------------------- unknown ledger
def test_report_opens_new_unknowns(db):
    change = db.report_unknowns("solve", "association", [item("a", 3), item("b", 5, why="x")], 0)
    assert change == {"opened": 2, "closed": 0, "kept": 0}
    rows = db.unknowns(kind="association")
    assert [r["subject"] for r in rows] == ["b", "a"]  # ordered by severity
    assert rows[0]["detail"] == {"why": "x"} and rows[0]["opened_round"] == 0 and rows[0]["stage"] == "solve"


def test_report_keeps_open_and_updates_severity_and_detail(db):
    db.report_unknowns("solve", "association", [item("a", 3, n=1)], 0)
    change = db.report_unknowns("solve", "association", [item("a", 7, n=2)], 1)
    assert change == {"opened": 0, "closed": 0, "kept": 1}
    row = status_of(db, "association", "a")
    assert (row["severity"], row["detail"], row["opened_round"], row["status"]) == (7.0, {"n": 2}, 0, "open")


def test_report_closes_vanished_unknowns_as_machine_resolved(db):
    db.report_unknowns("solve", "association", [item("a"), item("b")], 0)
    change = db.report_unknowns("solve", "association", [item("b")], 2)
    assert change == {"opened": 0, "closed": 1, "kept": 1}
    row = status_of(db, "association", "a")
    assert (row["status"], row["closed_round"], row["resolution"]) == ("resolved", 2, "machine:solve")


def test_report_reopens_machine_resolved_unknown(db):
    db.report_unknowns("solve", "association", [item("a", 2)], 0)
    db.report_unknowns("solve", "association", [], 1)
    assert status_of(db, "association", "a")["status"] == "resolved"
    change = db.report_unknowns("solve", "association", [item("a", 9, back=True)], 2)
    assert change["opened"] == 1
    row = status_of(db, "association", "a")
    assert (row["status"], row["severity"], row["closed_round"], row["resolution"]) == ("open", 9.0, None, None)
    assert row["detail"] == {"back": True}


def test_report_never_reopens_human_resolved(db):
    db.report_unknowns("solve", "association", [item("a")], 0)
    unknown = status_of(db, "association", "a")
    db.close_unknown(unknown["id"], 1, "human:7")
    change = db.report_unknowns("solve", "association", [item("a", 50)], 2)
    assert change == {"opened": 0, "closed": 0, "kept": 0}
    row = status_of(db, "association", "a")
    assert (row["status"], row["resolution"], row["severity"]) == ("resolved", "human:7", 1.0)


def test_report_never_reopens_undeterminable(db):
    db.report_unknowns("solve", "coverage", [item("a")], 0)
    unknown = status_of(db, "coverage", "a")
    db.close_unknown(unknown["id"], 1, "undeterminable:occluded", status="undeterminable")
    assert db.report_unknowns("solve", "coverage", [item("a")], 2)["opened"] == 0
    assert db.report_unknowns("solve", "coverage", [], 3)["closed"] == 0
    assert status_of(db, "coverage", "a")["status"] == "undeterminable"


def test_report_is_scoped_to_kind(db):
    db.report_unknowns("solve", "association", [item("a")], 0)
    db.report_unknowns("solve", "coverage", [item("a")], 0)  # same subject in another kind is a different unknown
    db.report_unknowns("solve", "coverage", [], 1)
    assert status_of(db, "association", "a")["status"] == "open"
    assert status_of(db, "coverage", "a")["status"] == "resolved"


def test_unknown_counts_and_severity(db):
    db.report_unknowns("solve", "association", [item("a", 2), item("b", 3), item("c", 4)], 0)
    db.report_unknowns("solve", "association", [item("a", 2), item("b", 3)], 1)
    counts = db.unknown_counts()["association"]
    assert counts["open"] == 2 and counts["open_severity"] == 5.0
    assert counts["resolved"] == 1 and counts["resolved_severity"] == 4.0


# --------------------------------------------------------------------------- tasks
def test_task_dedupe_while_open_or_answered(db):
    first = db.add_task("continuity", "t1", {"title": "x"}, priority=1, round_number=0)
    assert first is not None
    assert db.add_task("continuity", "t1", {"title": "y"}, priority=9, round_number=1) is None
    db.answer_task(first, {"verdict": "continuous"}, "tester")
    assert db.task(first)["status"] == "answered"
    assert db.add_task("continuity", "t1", {}, priority=1, round_number=1) is None  # answered still blocks duplicates
    assert len(db.tasks()) == 1


def test_task_dedupe_is_per_type_and_subject(db):
    assert db.add_task("continuity", "t1", {}, priority=1, round_number=0)
    assert db.add_task("cross_view_match", "t1", {}, priority=1, round_number=0)
    assert db.add_task("continuity", "t2", {}, priority=1, round_number=0)


@pytest.mark.parametrize("status", ["applied", "skipped", "superseded"])
def test_task_can_be_recreated_once_closed(db, status):
    first = db.add_task("continuity", "t1", {}, priority=1, round_number=0)
    db.set_task_status(first, status, applied_round=1 if status == "applied" else None)
    second = db.add_task("continuity", "t1", {}, priority=1, round_number=2)
    assert second is not None and second != first


def test_task_links_unknowns_and_orders_by_priority(db):
    db.report_unknowns("solve", "association", [item("a")], 0)
    unknown_id = status_of(db, "association", "a")["id"]
    low = db.add_task("continuity", "low", {}, priority=1, round_number=0)
    high = db.add_task("continuity", "high", {"k": 1}, priority=8, round_number=0, unknown_ids=[unknown_id])
    assert [t["id"] for t in db.tasks(status="open")] == [high, low]
    assert db.task_unknown_ids(high) == [unknown_id]
    assert db.task(high)["payload"] == {"k": 1}
    assert db.tasks(task_type="continuity", limit=1)[0]["id"] == high
    assert db.task_counts()["continuity"] == {"open": 2}


def test_answer_task_records_answer(db):
    task = db.add_task("continuity", "t1", {}, priority=1, round_number=0)
    db.answer_task(task, {"verdict": "break", "first_different": 3}, "ian")
    row = db.task(task)
    assert (row["status"], row["answer"], row["answered_by"]) == ("answered", {"verdict": "break", "first_different": 3}, "ian")
    assert row["answered"] is not None


def test_answering_a_closed_task_is_ignored(db):
    task = db.add_task("continuity", "t1", {}, priority=1, round_number=0)
    db.set_task_status(task, "applied", applied_round=1)
    db.answer_task(task, {"verdict": "continuous"}, "ian")
    assert db.task(task)["status"] == "applied" and db.task(task)["answer"] is None


# --------------------------------------------------------------------------- constraints
def test_constraints_supersede_keeps_audit_trail(db):
    old = db.add_constraint("same", ["c0/1/0"], ["c1/1/0"], source="human", round_number=0)
    new = db.add_constraint("different", ["c0/1/0"], ["c1/1/0"], source="human", round_number=1)
    db.supersede_constraint(old, new)
    assert [c["id"] for c in db.constraints("same")] == []
    assert [c["id"] for c in db.constraints()] == [new]
    everything = {c["id"]: c for c in db.constraints(active_only=False)}
    assert everything[old]["active"] == 0 and everything[old]["superseded_by"] == new
    assert db.constraints("different")[0]["a"] == ["c0/1/0"]


def test_constraint_detail_round_trips_and_filters_by_kind(db):
    db.add_constraint("calibration", ["felix-4"], None, source="human", round_number=0, detail={"choice": "use_corrected"})
    db.add_constraint("alignment", ["felix-1"], None, source="human", round_number=0, detail={"offset_seconds": 1.5})
    [calibration] = db.constraints("calibration")
    assert calibration["detail"] == {"choice": "use_corrected"} and calibration["b"] == []
    assert len(db.constraints()) == 2


def test_rounds_and_stage_runs(db):
    assert db.current_round() == -1
    assert db.open_round("init") == 0 and db.open_round("manual") == 1
    db.close_round(1, {"open_total": 3})
    assert db.rounds()[1]["summary"] == {"open_total": 3} and db.rounds()[1]["finished"] is not None
    run = db.start_stage("ingest", 0, "h1")
    assert db.latest_stage_run("ingest")["status"] == "running"
    db.finish_stage(run, "done", metrics={"n": 1}, message="ok")
    assert db.latest_stage_run("ingest", "done")["metrics"] == {"n": 1}
    assert db.latest_stage_run("ingest", "failed") is None


def test_meta_overwrite_semantics(db):
    db.set_meta("created", "t0", overwrite=False)
    db.set_meta("created", "t1", overwrite=False)
    assert db.get_meta("created") == "t0"
    db.set_meta("created", "t2")
    assert db.get_meta("created") == "t2" and db.get_meta("missing", 5) == 5
