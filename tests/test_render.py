import numpy as np
import pytest

from data_engine import render
from data_engine.render import COLORS_BGR, crop_window, render_panel

import cv2


class FakeFrames:
    def __init__(self, width=800, height=600):
        self.image = np.full((height, width, 3), 60, np.uint8)
        self.reads = 0

    def read(self, camera, frame):
        self.reads += 1
        return self.image


class FakeCamera:
    def __init__(self, boxes):
        self.boxes = np.array(boxes, np.float32)
        self.start = np.array([0, len(boxes)])

    def rows(self, frame):
        return slice(0, len(self.boxes)) if frame == 0 else slice(0, 0)


class FakeDetections:
    def __init__(self, boxes, keypoints):
        self.cam = FakeCamera(boxes)
        self.keypoints = np.array(keypoints, np.float32)

    def camera(self, name):
        return self.cam

    def lookup(self, key):
        index = int(key.rsplit("/", 1)[1])
        return self.cam.boxes[index], self.keypoints[index]


class FakeWorkspace:
    dataset_id = "fake"

    def __init__(self, boxes=((300, 100, 400, 400, 0.9),), keypoints=None, width=800, height=600):
        self.frames = FakeFrames(width, height)
        if keypoints is None:
            keypoints = [person_keypoints(350, 250, 0.9)]
        self.detections = FakeDetections(boxes, keypoints)


def person_keypoints(cx, cy, conf):
    points = np.zeros((17, 3), np.float32)
    for i in range(17):
        points[i] = (cx + (i % 5 - 2) * 12, cy - 120 + i * 14, conf)
    return points


def decode(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def coloured_pixels(image, colour, tolerance=40):
    return int((np.abs(image.astype(int) - np.array(colour)).max(axis=2) < tolerance).sum())


@pytest.fixture(autouse=True)
def fresh_cache():
    render.clear_cache()


def test_crop_window_pads_and_stays_inside():
    x0, y0, x1, y1 = crop_window([300, 100, 400, 400], 800, 600)
    assert 0 <= x0 < x1 <= 800 and 0 <= y0 < y1 <= 600
    assert x0 <= 300 and x1 >= 400 and y0 <= 100 and y1 >= 400  # whole box included
    assert (y1 - y0) >= 300 * 1.5 - 2  # 25% padding on each side


def test_crop_window_shifts_at_frame_edge_and_keeps_aspect():
    x0, y0, x1, y1 = crop_window([0, 0, 40, 200], 800, 600)
    assert x0 == 0 and y0 == 0
    assert 0.55 <= (x1 - x0) / (y1 - y0) <= 1.75
    x0, y0, x1, y1 = crop_window([790, 590, 800, 600], 800, 600)  # tiny box in the corner: minimum size
    assert x1 == 800 and y1 == 600 and (x1 - x0) >= 90


def test_panel_is_jpeg_with_requested_max_side():
    workspace = FakeWorkspace()
    data = render_panel(workspace, {"camera": "c", "frame": 0, "box": [300, 100, 400, 400], "highlight": []}, max_side=300)
    assert data[:2] == b"\xff\xd8"
    image = decode(data)
    assert max(image.shape[:2]) == 300


def test_skeleton_drawn_only_for_confident_joints():
    green = COLORS_BGR["green"]
    panel = {"camera": "c", "frame": 0, "box": [300, 100, 400, 400], "highlight": ["c/0/0"], "others": False,
             "label": ""}
    confident = decode(render_panel(FakeWorkspace(keypoints=[person_keypoints(350, 250, 0.9)]), panel, max_side=600))
    render.clear_cache()
    weak = decode(render_panel(FakeWorkspace(keypoints=[person_keypoints(350, 250, 0.29)]), panel, max_side=600))
    assert coloured_pixels(confident[:-40], green) > 150
    assert coloured_pixels(weak[:-40], green) == 0  # nothing drawn below the 0.3 peak gate


def test_web_mode_leaves_skeleton_to_the_overlay():
    green = COLORS_BGR["green"]
    workspace = FakeWorkspace(keypoints=[person_keypoints(350, 250, 0.9)])
    panel = {"camera": "c", "frame": 0, "box": [300, 100, 400, 400], "highlight": ["c/0/0"], "others": False,
             "label": ""}
    bare = decode(render_panel(workspace, panel, max_side=600, skeletons=False))
    assert coloured_pixels(bare[:-40], green) == 0
    overlay = render.panel_overlay(workspace, panel, max_side=600)
    assert overlay["color"] == "#5ae65a" and overlay["bones"] == render.SKELETON
    assert overlay["size"] == [bare.shape[1], bare.shape[0]]
    (skeleton,) = overlay["skeletons"]
    assert not skeleton["dashed"] and len(skeleton["joints"]) == 17
    assert all(0 <= v <= 1 for joint in skeleton["joints"] for v in joint)
    weak = FakeWorkspace(keypoints=[person_keypoints(350, 250, 0.29)])
    render.clear_cache()
    assert render.panel_overlay(weak, panel, max_side=600)["skeletons"] == []  # same 0.3 peak gate


def test_others_drawn_faintly_only_when_requested():
    boxes = [(300, 100, 400, 400, 0.9), (500, 120, 600, 420, 0.8)]
    keypoints = [person_keypoints(350, 250, 0.9), person_keypoints(550, 270, 0.9)]
    workspace = FakeWorkspace(boxes, keypoints)
    base = {"camera": "c", "frame": 0, "box": None, "highlight": ["c/0/0"]}
    with_others = decode(render_panel(workspace, {**base, "others": True}, max_side=800))
    without = decode(render_panel(workspace, {**base, "others": False}, max_side=800))
    assert coloured_pixels(with_others[:-40], render.GREY, 8) > coloured_pixels(without[:-40], render.GREY, 8)


def test_full_frame_mode_shows_whole_frame_with_box_outline():
    workspace = FakeWorkspace()
    panel = {"camera": "c", "frame": 0, "box": [300, 100, 400, 400], "highlight": [], "others": False, "color": "orange"}
    crop = decode(render_panel(workspace, panel, max_side=400))
    full = decode(render_panel(workspace, panel, max_side=400, full=True))
    assert full.shape[1] / full.shape[0] == pytest.approx(800 / 600, abs=0.02)  # whole 4:3 frame
    assert crop.shape[1] / crop.shape[0] < 1.0  # portrait crop around the person
    assert coloured_pixels(full[:-40], COLORS_BGR["orange"]) > 100  # outline
    assert coloured_pixels(crop[:-40], COLORS_BGR["orange"]) < 40  # crop has no outline


def test_label_is_drawn_and_cache_avoids_rereading_frames():
    workspace = FakeWorkspace()
    panel = {"camera": "c", "frame": 0, "box": [300, 100, 400, 400], "label": "B2", "color": "magenta"}
    first = render_panel(workspace, panel, max_side=300)
    second = render_panel(workspace, panel, max_side=300)
    assert first is second and workspace.frames.reads == 1
    assert coloured_pixels(decode(first)[:40, :80], COLORS_BGR["magenta"]) > 100  # label chip top-left
    render_panel(workspace, panel, max_side=301)
    assert workspace.frames.reads == 2


def test_project_uses_tracks_module_when_present(monkeypatch):
    import sys
    import types

    pose = np.array([[200 + i * 4, 150 + i * 12] for i in range(17)], float)
    pose[3] = np.nan
    module = types.ModuleType("data_engine.tracks")
    module.projected_pose = lambda ws, track, camera, frame: pose
    monkeypatch.setitem(sys.modules, "data_engine.tracks", module)
    workspace = FakeWorkspace()
    panel = {"camera": "c", "frame": 0, "box": None, "others": False, "project": [7], "color": "cyan"}
    drawn = decode(render_panel(workspace, panel, max_side=800))
    assert coloured_pixels(drawn[:-40], COLORS_BGR["cyan"]) > 100
    render.clear_cache()
    monkeypatch.setitem(sys.modules, "data_engine.tracks", None)  # import fails -> silently skipped
    plain = decode(render_panel(workspace, panel, max_side=800))
    assert coloured_pixels(plain[:-40], COLORS_BGR["cyan"]) == 0


def test_concurrent_requests_render_once():
    import threading
    workspace = FakeWorkspace()
    panel = {"camera": "c", "frame": 0, "box": [300, 100, 400, 400]}
    results = []
    threads = [threading.Thread(target=lambda: results.append(render_panel(workspace, panel, max_side=256)))
               for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(results) == 8 and len({id(r) for r in results}) == 1 and workspace.frames.reads == 1
