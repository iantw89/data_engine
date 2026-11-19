import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

from data_engine.config import Site
from data_engine.web import app as web_app
from data_engine.workspace import Workspace

DATASET = "unit_20250101000000_20250101010000"


def panel(camera="c0", frame=0, **extra):
    return {"camera": camera, "frame": frame, "box": [10, 10, 60, 120], "highlight": [], "others": True,
            "label": "A", "color": "green", "context": False, **extra}


def payload(ui, **extra):
    base = {"ui": ui, "title": f"{ui} task", "question": "q?", "why": "because", "evidence": {"p80 px": 7.2},
            "time_seconds": 12.5}
    return {**base, **extra}


@pytest.fixture()
def env(tmp_path):
    site = Site({"data_root": str(tmp_path)})
    workspace = Workspace(DATASET, site)
    db = workspace.db
    round_number = db.open_round("init")
    ids = {}
    ids["strip"] = db.add_task("continuity", "s1", payload("strip", panels=[panel(frame=i) for i in range(5)]),
                               priority=5, round_number=round_number)
    ids["rc"] = db.add_task("cross_view_match", "rc1", payload(
        "reference_candidates", reference={"id": "A", "caption": "ref", "panels": [panel()]},
        candidates=[{"id": "B", "caption": "b", "panels": [panel()], "track": 1},
                    {"id": "C", "caption": "c", "panels": [panel()], "track": 2}],
        allow_none=True, single_choice=True, problem_kinds=["mixed_pose", "not_person"]),
        priority=9, round_number=round_number)
    ids["decision"] = db.add_task("visibility_check", "d1", payload(
        "decision", panels=[panel()], options=[{"value": "visible", "label": "Visible"},
                                               {"value": "occluded", "label": "Occluded"}]),
        priority=1, round_number=round_number)
    ids["picker"] = db.add_task("alignment_review", "f1", payload(
        "frame_picker", reference={"caption": "ref", "panels": [panel()]},
        candidates=[{"id": "0", "caption": "x", "panels": [panel()]}, {"id": "1", "caption": "y", "panels": [panel()]}]),
        priority=3, round_number=round_number)
    ids["cvat"] = db.add_task("pose_correction", "v1", payload("cvat", cvat_url="http://x", instructions="fix"),
                              priority=0, round_number=round_number)
    db.close_round(round_number, {})
    workspace.db.close()
    client = TestClient(web_app.create_app(site))
    yield client, ids, site
    client.close()


def url(path):
    return f"/api/datasets/{DATASET}{path}"


def test_datasets_and_status(env):
    client, ids, _ = env
    rows = client.get("/api/datasets").json()
    assert [r["id"] for r in rows] == [DATASET]
    assert rows[0]["open_tasks"] == 5
    status = client.get(url("/status")).json()
    assert status["dataset_id"] == DATASET and status["tasks"]["open_total"] == 5
    assert status["run"]["running"] is False
    assert {s["name"] for s in status["stages"]} >= {"ingest", "solve", "tasks"}
    assert client.get("/api/datasets/nope/status").status_code == 404
    assert client.get("/api/datasets/..%2f..%2fx/status").status_code == 404


def test_taxonomy(env):
    client, *_ = env
    taxonomy = client.get("/api/taxonomy").json()
    assert {t["name"] for t in taxonomy["task_types"]} >= {"continuity", "cross_view_match", "alignment_review"}
    assert "association" in {k["name"] for k in taxonomy["kinds"]}


def test_task_listing_filters_and_paging(env):
    client, ids, _ = env
    listing = client.get(url("/tasks")).json()
    assert listing["total"] == 5
    assert [t["id"] for t in listing["tasks"]][0] == ids["rc"]  # highest priority first
    assert listing["tasks"][0]["ui"] == "reference_candidates"
    only = client.get(url("/tasks?type=continuity&status=open")).json()
    assert [t["id"] for t in only["tasks"]] == [ids["strip"]]
    page = client.get(url("/tasks?limit=2&offset=2")).json()
    assert len(page["tasks"]) == 2 and page["total"] == 5
    detail = client.get(url(f"/tasks/{ids['strip']}")).json()
    assert detail["payload"]["panels"][0]["camera"] == "c0" and detail["ui"] == "strip"
    assert client.get(url("/tasks/9999")).status_code == 404


def test_next_task_order_and_after(env):
    client, ids, _ = env
    first = client.get(url("/tasks/next")).json()
    assert first["task"]["id"] == ids["rc"] and first["remaining"] == 5
    second = client.get(url(f"/tasks/next?after={ids['rc']}")).json()
    assert second["task"]["id"] == ids["strip"]  # priority 5 follows 9
    last = client.get(url(f"/tasks/next?after={ids['cvat']}")).json()
    assert last["task"]["id"] == ids["rc"]  # wraps to the first open task
    typed = client.get(url("/tasks/next?type=decision")).json()
    assert typed["task"] is None
    typed = client.get(url("/tasks/next?type=visibility_check")).json()
    assert typed["task"]["id"] == ids["decision"]


def answer(client, task_id, body, user="tester"):
    return client.post(url(f"/tasks/{task_id}/answer"), json={"answer": body, "user": user})


def test_answer_strip_validation_and_storage(env):
    client, ids, site = env
    task = ids["strip"]
    assert answer(client, task, {"verdict": "bogus"}).status_code == 422
    bad = answer(client, task, {"verdict": "break", "first_different": None})
    assert bad.status_code == 422 and "first_different" in bad.json()["detail"]
    assert answer(client, task, {"verdict": "break", "first_different": 9}).status_code == 422
    assert answer(client, task, {"verdict": "continuous", "first_different": 2}).status_code == 422
    ok = answer(client, task, {"verdict": "break", "first_different": 3, "unclear": [1], "notes": "hat"})
    assert ok.status_code == 200
    stored = Workspace(DATASET, site).db.task(task)
    assert stored["status"] == "answered" and stored["answered_by"] == "tester"
    assert stored["answer"]["first_different"] == 3 and stored["answer"]["unclear"] == [1]
    assert answer(client, ids["cvat"], {"verdict": "continuous"}).status_code == 422
    assert answer(client, task, {"verdict": "unclear", "unclear": [0, 1]}).status_code == 200  # re-answer allowed


def test_answer_reference_candidates_rules(env):
    client, ids, _ = env
    task = ids["rc"]
    incomplete = answer(client, task, {"reference_ok": True, "candidates": {"B": "same"}})
    assert incomplete.status_code == 422 and "missing: C" in incomplete.json()["detail"]
    both = answer(client, task, {"reference_ok": True, "candidates": {"B": "same", "C": "same"}})
    assert both.status_code == 422 and "only one" in both.json()["detail"]
    ghost = answer(client, task, {"reference_ok": True, "candidates": {"B": "same", "C": "different", "Z": "same"}})
    assert ghost.status_code == 422
    none_and_same = answer(client, task, {"reference_ok": True, "candidates": {"B": "same", "C": "different"},
                                          "none_of_these": True})
    assert none_and_same.status_code == 422
    bad_problem = answer(client, task, {"reference_ok": True, "candidates": {"B": "different", "C": "unsure"},
                                        "problems": {"B": "weird"}})
    assert bad_problem.status_code == 422
    assert answer(client, task, {"reference_ok": True, "candidates": {"B": "unsure", "C": "unsure"}}).status_code == 200
    # a broken reference needs no candidate verdicts
    assert answer(client, task, {"reference_ok": False, "candidates": {}, "problems": {"A": "mixed_pose"}}).status_code == 200
    assert answer(client, task, {"reference_ok": True, "candidates": {"B": "different", "C": "different"},
                                 "none_of_these": True}).status_code == 200


def test_answer_decision_picker_cvat(env):
    client, ids, _ = env
    assert answer(client, ids["decision"], {"choice": "maybe"}).status_code == 422
    assert answer(client, ids["decision"], {"choice": "occluded"}).status_code == 200
    assert answer(client, ids["picker"], {"choice": "7"}).status_code == 422
    assert answer(client, ids["picker"], {"choice": "1"}).status_code == 200
    assert answer(client, ids["picker"], {"choice": "none"}).status_code == 200
    assert answer(client, ids["cvat"], {"done": False}).status_code == 422
    assert answer(client, ids["cvat"], {"done": True}).status_code == 200
    unsure = client.get(url("/tasks/next?type=visibility_check")).json()
    assert unsure["task"] is None


def test_unsure_is_always_accepted(env):
    client, ids, site = env
    assert answer(client, ids["decision"], {"choice": "unsure"}).status_code == 200
    assert answer(client, ids["picker"], {"choice": "unsure"}).status_code == 200
    assert Workspace(DATASET, site).db.task(ids["decision"])["answer"]["choice"] == "unsure"


def test_skip_and_reopen(env):
    client, ids, site = env
    task = ids["decision"]
    assert client.post(url(f"/tasks/{task}/reopen")).status_code == 409  # still open
    assert client.post(url(f"/tasks/{task}/skip")).json()["task"]["status"] == "skipped"
    assert client.post(url(f"/tasks/{task}/skip")).status_code == 409
    assert answer(client, task, {"choice": "visible"}).status_code == 409  # skipped tasks are reopened first
    assert client.post(url(f"/tasks/{task}/reopen")).json()["task"]["status"] == "open"
    answer(client, task, {"choice": "visible"})
    reopened = client.post(url(f"/tasks/{task}/reopen"), json={"user": "tester"}).json()["task"]
    assert reopened["status"] == "open" and reopened["answer"] is None
    db = Workspace(DATASET, site).db
    db.set_task_status(task, "applied", applied_round=1)
    assert client.post(url(f"/tasks/{task}/reopen")).status_code == 409
    assert answer(client, task, {"choice": "visible"}).status_code == 409
    assert client.post(url(f"/tasks/{task}/skip")).status_code == 409


def test_unknowns_and_constraints(env):
    client, ids, site = env
    db = Workspace(DATASET, site).db
    db.report_unknowns("solve", "association", [{"subject": "a/1", "severity": 3, "detail": {"why": "x"}}], 0)
    db.add_constraint("same", ["c0/1/0"], ["c1/1/0"], source="human", round_number=0, task_id=ids["rc"])
    unknowns = client.get(url("/unknowns?kind=association")).json()
    assert unknowns["total"] == 1 and unknowns["unknowns"][0]["detail"] == {"why": "x"}
    assert client.get(url("/unknowns?status=resolved")).json()["total"] == 0
    constraints = client.get(url("/constraints")).json()
    assert constraints["total"] == 1 and constraints["constraints"][0]["a"] == ["c0/1/0"]


def test_advance_refuses_second_run(env, monkeypatch):
    client, ids, site = env
    calls = []

    def fake_spawn(command, log_path, environment, cwd):
        calls.append(command)
        log_path.write_text("starting round\n", encoding="utf-8")
        return os.getpid()  # a pid that is certainly alive

    monkeypatch.setattr(web_app, "spawn_process", fake_spawn)
    first = client.post(url("/advance"))
    assert first.status_code == 200 and first.json()["started"] is True
    assert calls[0][-3:] == ["data_engine.cli", "run", DATASET] or calls[0][-2:] == ["run", DATASET]
    second = client.post(url("/advance"))
    assert second.status_code == 409 and "already in progress" in second.json()["detail"]
    run = client.get(url("/run")).json()
    assert run["running"] is True and "starting round" in run["tail"]
    assert len(calls) == 1
    assert client.get(url("/status")).json()["run"]["running"] is True


def test_advance_refuses_when_a_stage_is_running(env, monkeypatch):
    client, ids, site = env
    db = Workspace(DATASET, site).db
    db.start_stage("detect", 0, "h")
    monkeypatch.setattr(web_app, "spawn_process", lambda *a, **k: pytest.fail("must not spawn"))
    response = client.post(url("/advance"))
    assert response.status_code == 409 and "detect" in response.json()["detail"]


def test_advance_after_finished_process_is_allowed(env, monkeypatch):
    client, ids, site = env
    workspace = Workspace(DATASET, site)
    lock = workspace.path("logs", "run.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"pid": 2 ** 22 + 12345, "started": 0, "log": "logs/run_old.log"}))
    monkeypatch.setattr(web_app, "spawn_process", lambda command, log, env_, cwd: os.getpid())
    assert client.post(url("/advance")).status_code == 200


def test_panel_endpoint_returns_jpeg_or_placeholder(env):
    client, ids, site = env
    spec = json.dumps(panel())
    response = client.get(url("/panel"), params={"spec": spec, "w": 300})
    # no dataset.json in this workspace: the render fails and a placeholder JPEG is served, never a 5xx
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    assert response.headers.get("x-render-error")
    assert client.get(url("/panel"), params={"spec": "{not json"}).status_code == 422


def test_pages_and_static(env):
    client, *_ = env
    for path in ("/", f"/d/{DATASET}", f"/d/{DATASET}/tasks"):
        response = client.get(path)
        assert response.status_code == 200 and "text/html" in response.headers["content-type"]
    assert client.get("/static/style.css").status_code == 200
