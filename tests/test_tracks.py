"""tracks.py: projected 3D poses and per-identity summaries over a synthetic solved workspace."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from data_engine import tracks
from data_engine.config import Site
from data_engine.workspace import Workspace
from test_publish import CAMERAS, T, make_workspace


@pytest.fixture()
def workspace(tmp_path):
    tracks.clear_cache()
    root, site, native = make_workspace(tmp_path)
    return Workspace(root.name, site, root=root), native


def expected_pixels(world, camera_index):
    """Pinhole (no distortion, rvec 0) projection used by the synthetic calibration, to native pixels."""
    x, y, z = np.asarray(world, float)
    depth = z + 100.0 + 10 * camera_index
    canvas = np.array([900 * x / depth + 640, 900 * y / depth + 480])
    return (canvas - [0, 160]) / 0.625 / (1280 / 2048)


def test_projected_pose_projects_measured_joints_only(workspace):
    ws, native = workspace
    t = 1  # aligned 0: all three cameras show a frame
    for index, camera in enumerate(CAMERAS):
        frame = int(native[t, index])
        pose = tracks.projected_pose(ws, 0, camera, frame)
        assert pose.shape == (17, 2)
        assert np.isnan(pose[:2]).all()  # head joints were not measured: NaN, never invented
        for joint in range(2, 17):
            world = np.array([50 + t, 20, 40]) + (joint - 2) * 0.5
            assert np.allclose(pose[joint], expected_pixels(world, index), atol=1e-3)


def test_projected_pose_none_when_there_is_no_3d_pose(workspace):
    ws, native = workspace
    assert tracks.projected_pose(ws, 7, "c0", 1) is None  # unknown track
    assert tracks.projected_pose(ws, 0, "nope", 1) is None  # unknown camera
    assert tracks.projected_pose(ws, 0, "c1", 0) is None  # c1 shows frame 0 at no sync index (native = t + 2)
    assert tracks.projected_pose(ws, 0, "c0", 400) is None  # frame outside the recording
    assert tracks.projected_pose(ws, 0, "c0", int(native[5, 0])) is None  # nothing measured at that sync index


def test_projected_pose_cache_reloads_when_the_artifact_changes(workspace):
    ws, native = workspace
    frame = int(native[1, 0])
    before = tracks.projected_pose(ws, 0, "c0", frame)
    assert tracks.projected_pose(ws, 0, "c0", frame) is not None
    assert len(tracks._CACHE) == 1  # one entry per workspace/artifact version
    path = ws.latest_round_file("tracks3d.npz")
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["xyz"] = arrays["xyz"] + np.float32(5.0)
    np.savez_compressed(path, **arrays)
    stamp = time.time() + 5
    os.utime(path, (stamp, stamp))
    after = tracks.projected_pose(ws, 0, "c0", frame)
    assert not np.allclose(before[2:], after[2:])


def test_track_summary_counts(workspace):
    ws, native = workspace
    rows = tracks.track_summary(ws)
    assert [r["identity"] for r in rows] == [0]
    row = rows[0]
    assert row["cameras"] == CAMERAS and row["detections"] == 36 and row["detections_by_camera"] == {c: 12 for c in CAMERAS}
    # sync indices with at least two cameras showing a frame: t = 0..11 (see make_workspace)
    assert row["frames"] == 12 and row["first_t"] == 0 and row["last_t"] == 11
    assert row["first_time_s"] == -0.1 and row["last_time_s"] == 1.0
    assert row["frames_with_3d"] == 10  # t = 5 and 6 have no measured root
    assert row["frames_with_keypoints"] == 11  # t = 5 has no measured joint at all


def test_nothing_solved_gives_empty_results(tmp_path):
    tracks.clear_cache()
    site = Site({"data_root": str(tmp_path), "calibration": str(tmp_path / "none.json")})
    empty = Workspace("empty_20250101000000_20250101010000", site, root=tmp_path / "empty")
    assert tracks.track_summary(empty) == []
    assert tracks.projected_pose(empty, 0, "c0", 0) is None
    assert T > 0
