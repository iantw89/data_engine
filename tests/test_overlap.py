"""File-boundary verification by pixel comparison, and the timeline without repeated frames."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_engine.dataset import Camera, Part  # noqa: E402
from data_engine.overlap import find_overlap, verify_boundary  # noqa: E402
from data_engine.workers import align_dm  # noqa: E402


def moving_scene(count: int, seed: int = 0) -> list[np.ndarray]:
    """Frames that all differ: a block moving over sensor noise."""
    rng = np.random.default_rng(seed)
    frames = []
    for index in range(count):
        image = rng.normal(100.0, 2.0, (48, 64)).astype(np.float32)
        image[10:20, index % 50:index % 50 + 8] += 80.0
        frames.append(image)
    return frames


def test_exact_repeat_is_measured():
    scene = moving_scene(80)
    tail, head = scene[10:40], scene[34:64]  # the continuation starts 6 frames before the previous file ends
    found = find_overlap(tail, head)
    assert found["overlap"] == 6 and found["residual"] == 0.0 and found["next_best"] > 0.2


def test_no_repeat_means_no_decision():
    scene = moving_scene(80)
    found = find_overlap(scene[10:40], scene[40:70])  # a clean continuation: nothing repeats
    assert found["overlap"] is None and "no exact repeat" in found["reason"]


def test_still_scene_is_not_evidence():
    frame = np.full((48, 64), 90.0, np.float32)
    found = find_overlap([frame] * 30, [frame] * 30)  # every shift looks identical
    assert found["overlap"] is None and "not unique" in found["reason"]


def test_single_repeated_frame_is_not_evidence():
    scene = moving_scene(80)
    found = find_overlap(scene[10:40], scene[39:69])
    assert found["overlap"] is None and "single" in found["reason"]


def test_overlap_reaching_the_window_is_not_measured():
    scene = moving_scene(80)
    found = find_overlap(scene[10:40], scene[10:40])
    assert found["overlap"] is None and "search window" in found["reason"]


def test_noisy_near_repeat_is_rejected():
    scene = moving_scene(80)
    rng = np.random.default_rng(5)
    head = [image + rng.normal(0, 1.0, image.shape).astype(np.float32) for image in scene[34:64]]
    assert find_overlap(scene[10:40], head)["overlap"] is None  # re-encoded, not identical: no decision


def test_verify_boundary_reads_the_right_frames():
    scene = moving_scene(200)
    first, second = scene[:120], scene[112:200]  # 8 repeated frames
    grey = lambda image: np.repeat(image[:, :, None], 3, axis=2).clip(0, 255).astype(np.uint8)  # noqa: E731
    found = verify_boundary(lambda i: grey(first[i]), 119, lambda i: grey(second[i]), 0, len(second), search=40)
    assert found["overlap"] == 8 and found["frames_compared"] == 40


def camera(overlap: int) -> Camera:
    return Camera("c", "0", 64, 48, 10.0, [Part("a.mp4", 0, 10), Part("b.mp4", 2, 8, True, overlap)])


def test_repeated_frames_and_timeline():
    cam = camera(3)
    assert cam.frame_count == 18
    assert cam.repeated_frames() == [(10, 7), (11, 8), (12, 9)]
    assert cam.locate(10) == ("b.mp4", 2)  # still addressable: the frame index is unchanged
    keep = align_dm.timeline_frames(cam.parts)
    assert keep.tolist() == list(range(10)) + list(range(13, 18))
    assert align_dm.timeline_frames(camera(0).parts).tolist() == list(range(18))


def test_detections_follow_the_timeline():
    cam = camera(3)
    keep = align_dm.timeline_frames(cam.parts)
    frames = np.arange(18)
    counts = np.ones(18, np.int64)
    counts[11] = 2
    boxes = np.repeat(frames, counts)[:, None].astype(np.float32) * np.ones((1, 5), np.float32)
    keypoints = np.zeros((len(boxes), 17, 3), np.float32)
    detections = align_dm.CameraDetections("c", frames, np.concatenate(([0], np.cumsum(counts)[:-1])), counts, boxes, keypoints)
    moved = detections.on_timeline(keep)
    assert moved.frames.tolist() == list(range(15))
    assert moved.boxes[:, 0].tolist() == [float(f) for f in keep]  # rows of repeated frames are gone
    assert moved.record(10)[0][0, 0] == 13.0  # timeline position 10 is frame 13


def test_sync_table_and_clock_return_to_frame_indices():
    cam = camera(3)
    keep = align_dm.timeline_frames(cam.parts)
    native = np.array([-1, 0, 9, 10, 14], np.int32)
    assert align_dm.to_frame_index(native, keep).tolist() == [-1, 0, 9, 13, 17]
    times = np.arange(15) * 0.1
    clock = align_dm.stream_clock(times, keep, cam.repeated_frames(), cam.frame_count)
    assert np.allclose(clock[:10], times[:10]) and np.allclose(clock[13:], times[10:])
    assert np.allclose(clock[10:13], clock[7:10])  # a repeated frame shows the instant of the frame it repeats
