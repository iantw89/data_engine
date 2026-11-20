"""A flag put on an image once is pre-set wherever another open task shows the same image (tasks/carry.py)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from data_engine.config import Site
from data_engine.tasks.carry import carried_flags, flagged_unsure
from data_engine.web import app as web_app
from data_engine.workspace import Workspace

DATASET = "unit_20250101000000_20250101010000"


def panel(key):
    camera, frame, index = key.split("/")
    return {"camera": camera, "frame": int(frame), "box": [10, 10, 60, 120], "highlight": [key]}


def task_payload(reference, **candidates):
    return {"ui": "reference_candidates", "mode": "match", "multiple": True,
            "reference": {"id": "A", "panels": [panel(key) for key in reference]}, "reference_keys": list(reference),
            "candidates": [{"id": gid, "track": i, "panels": [panel(key) for key in keys], "keys": list(keys)}
                           for i, (gid, keys) in enumerate(candidates.items())]}


TRACK = ["1/10/0", "2/12/1", "3/14/0"]  # one earlier track: the same three images in every question about it


def answered(task_id, payload, images, when=1.0):
    return {"id": task_id, "payload": payload, "answer": {"images": images, "candidates": {}}, "answered": when}


def test_someone_else_carries_to_another_question_showing_the_same_group():
    first = answered(1, task_payload(["0/10/0"], B=TRACK), {TRACK[1]: {"exclude": "other_person", "bad_pose": False}})
    second = task_payload(["0/99/0"], B=["5/1/0"], C=TRACK)
    assert carried_flags(second, [first]) == {TRACK[1]: {"exclude": "other_person", "bad_pose": False, "task": 1}}


def test_someone_else_does_not_carry_next_to_other_images():
    """'Not the person the rest of this group shows' says nothing about a group of other images."""
    first = answered(1, task_payload(["0/10/0"], B=TRACK), {TRACK[1]: {"exclude": "other_person", "bad_pose": False}})
    elsewhere = task_payload(["0/99/0"], B=[TRACK[1], "6/50/0", "6/60/0"])
    assert carried_flags(elsewhere, [first]) == {}
    # ... nor when the only common image was itself taken out there
    both = answered(2, task_payload(["0/10/0"], B=TRACK), {TRACK[1]: {"exclude": "other_person", "bad_pose": False},
                                                            TRACK[0]: {"exclude": "unsure", "bad_pose": False}})
    assert set(carried_flags(task_payload(["0/99/0"], B=[TRACK[1], TRACK[0]]), [both])) == {TRACK[0]}


def test_statements_about_the_image_itself_carry_wherever_it_is_shown():
    flags = {TRACK[0]: {"exclude": "mixed", "bad_pose": False}, TRACK[1]: {"exclude": "not_person", "bad_pose": False},
             TRACK[2]: {"exclude": None, "bad_pose": True}, "0/10/0": {"exclude": "unsure", "bad_pose": True}}
    first = answered(1, task_payload(["0/10/0"], B=TRACK), flags)
    elsewhere = task_payload([TRACK[0]], B=[TRACK[1]], C=[TRACK[2], "0/10/0"])
    carried = carried_flags(elsewhere, [first])
    assert {key: (value["exclude"], value["bad_pose"]) for key, value in carried.items()} == {
        TRACK[0]: ("mixed", False), TRACK[1]: ("not_person", False), TRACK[2]: (None, True), "0/10/0": ("unsure", True)}
    assert flagged_unsure([first]) == {"0/10/0"}


def test_the_latest_answer_about_an_image_wins():
    payload = task_payload(["0/10/0"], B=TRACK)
    first = answered(1, payload, {TRACK[1]: {"exclude": "other_person", "bad_pose": False}}, when=1.0)
    changed = answered(2, payload, {TRACK[1]: {"exclude": "unsure", "bad_pose": False}}, when=2.0)
    cleared = answered(3, payload, {}, when=3.0)  # shown again, the pre-set flag was undone
    target = task_payload(["0/99/0"], B=TRACK)
    assert carried_flags(target, [changed, first])[TRACK[1]]["exclude"] == "unsure"
    assert carried_flags(target, [first, changed, cleared]) == {}
    assert carried_flags(target, [first, changed, cleared], skip_task=3)[TRACK[1]]["task"] == 2


def test_the_server_sends_the_carried_flags_with_an_open_task(tmp_path):
    site = Site({"data_root": str(tmp_path)})
    workspace = Workspace(DATASET, site)
    db = workspace.db
    round_number = db.open_round("init")
    one = db.add_task("reidentify", "r1", task_payload(["0/10/0"], B=TRACK), priority=2, round_number=round_number)
    two = db.add_task("reidentify", "r2", task_payload(["0/99/0"], B=TRACK), priority=1, round_number=round_number)
    db.close_round(round_number, {})
    db.close()
    client = TestClient(web_app.create_app(site))
    try:
        assert client.get(f"/api/datasets/{DATASET}/tasks/{two}").json()["known_flags"] == {}
        body = {"answer": {"images": {TRACK[1]: {"exclude": "other_person"}, TRACK[2]: {"bad_pose": True}},
                           "candidates": {"B": "different"}}, "user": "t"}
        assert client.post(f"/api/datasets/{DATASET}/tasks/{one}/answer", json=body).status_code == 200
        known = client.get(f"/api/datasets/{DATASET}/tasks/{two}").json()["known_flags"]
        assert known == {TRACK[1]: {"exclude": "other_person", "bad_pose": False, "task": one},
                         TRACK[2]: {"exclude": None, "bad_pose": True, "task": one}}
        # the answered task itself is restored from its own answer, not pre-set
        assert client.get(f"/api/datasets/{DATASET}/tasks/{one}").json()["known_flags"] == {}
        # and the next task the runner asks for carries them too
        following = client.get(f"/api/datasets/{DATASET}/tasks/next").json()
        task = following.get("task") or following
        assert task["id"] == two and task["known_flags"][TRACK[1]]["exclude"] == "other_person"
    finally:
        client.close()
