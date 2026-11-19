"""Publish: conversion of a solved workspace into label-set drafts (pure builders, no data_management needed).

A tiny synthetic workspace (3 cameras, 12 native frames, one 3-camera person plus a second person) is written
to ``tmp_path``; the builders of ``workers/publish_dm.py`` run on it in the engine venv. Tests that need
data_management (scratch-store validation, publishing into a temp store) skip when it is not importable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data_engine.config import Site
from data_engine.dataset import Camera, Dataset, Part
from data_engine.db import Database
from data_engine.engine import Context, StageResult
from data_engine.workers import publish_dm as pm
from data_engine.workspace import Workspace

CAMERAS = ["c0", "c1", "c2"]
NATIVE_FRAMES = 12
T = 14
OFFSET_D = {"c0": 0, "c1": 2, "c2": -1}  # native = t + d (sync index t)
LABEL_OFFSETS = {"c0": 1, "c1": 3, "c2": 0}  # aligned_index = t - 1  =>  native = aligned + offset
DATASET = "unit_20250101000000_20250101010000"
GOOD_CODE = {"repository": "data_engine", "git_commit": "a" * 40, "git_dirty": False}
NO_COMMIT = {"repository": "data_engine", "git_commit": None, "git_dirty": True, "diff_sha256": "d" * 64}


def standard_calibration():
    cameras = {str(i): {"K": [[900, 0, 640], [0, 900, 480], [0, 0, 1]], "D": [0, 0, 0, 0], "rvec": [0, 0, 0],
                        "tvec": [0.0, 0.0, 100.0 + 10 * i], "solver": "s"} for i in range(3)}
    return {"padding": [0.0, 160.0], "scale": 0.625, "output_resolution": [1280, 960], "canvas": "x",
            "world_units": "inches", "convention": "X_cam = R X + t", "cameras": cameras,
            "data_engine": {"source": "C:\\cal\\v5_tracks_ransac.json", "source_sha256": "abc123",
                            "cameras": {"c0": {"status": "unchanged", "applied": "standard", "median_shift_px": 0.2},
                                        "c1": {"status": "shifted", "applied": "standard", "median_shift_px": 3.0},
                                        "c2": {"status": "unchanged", "applied": "standard", "median_shift_px": 0.1}}}}


def sync_native(drift: bool = False) -> np.ndarray:
    native = np.full((T, 3), -1, np.int32)
    for t in range(T):
        for column, camera in enumerate(CAMERAS):
            d = OFFSET_D[camera] + (1 if (drift and camera == "c1" and t >= 7) else 0)
            if 0 <= t + d < NATIVE_FRAMES:
                native[t, column] = t + d
    return native


def make_workspace(tmp_path: Path, *, drift: bool = False, name: str = "unit_20250101000000_20250101010000@pilot12"):
    root = tmp_path / name
    for sub in ("alignment", "calibration", "detections/raw", "tracklets", "rounds/r001", "manifests"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    dataset = Dataset(DATASET + "@pilot12", "unit", "20250101000000", "20250101010000")
    for index, camera in enumerate(CAMERAS):
        dataset.cameras[camera] = Camera(camera, str(index), 2048, 1536, 10.0, [Part(f"{camera}.mp4", 0, NATIVE_FRAMES)])
    dataset.save(root / "dataset.json")

    kept, identity, atom, source = {}, {}, {}, {}
    for camera in CAMERAS:
        rows = 2 * NATIVE_FRAMES
        (root / "detections/raw" / camera).mkdir(parents=True, exist_ok=True)
        frames = np.arange(NATIVE_FRAMES)
        boxes = np.zeros((rows, 5), np.float32)
        keypoints = np.zeros((rows, 17, 3), np.float32)
        for r in range(rows):
            boxes[r] = [10 * (r % 2) + r // 2 + 0.123456, 20.0, 60.0 + r // 2, 120.0, 0.9 - 0.01 * (r // 2)]
            keypoints[r, :, 0] = 100 + np.arange(17) + r * 0.123456
            keypoints[r, :, 1] = 200 + np.arange(17) * 1.55555
            keypoints[r, :, 2] = 0.5 + 0.01 * np.arange(17)
        np.savez_compressed(root / "detections/raw" / camera / "000000.npz", frames=frames,
                            counts=np.full(NATIVE_FRAMES, 2), boxes=boxes, keypoints=keypoints)
        k = np.ones(rows, bool)
        ident = np.full(rows, -1, np.int32)
        at = np.full(rows, -1, np.int32)
        src = np.zeros(rows, np.uint8)
        for f in range(NATIVE_FRAMES):
            a, b = 2 * f, 2 * f + 1
            ident[a], src[a] = 0, 1
            if camera == "c0":
                at[a] = 0
            elif camera == "c1":
                at[a], at[b] = (5 if f < 6 else -1), 3
                src[a] = 1 if f < 6 else 4
            else:
                at[a] = -1 if f in (2, 3, 4, 8, 9) else 0
                src[a] = 4 if at[a] < 0 else 1
                if f < 4:
                    k[b] = False
        kept[camera], identity[camera], atom[camera], source[camera] = k, ident, at, src
        np.savez_compressed(root / "tracklets" / f"{camera}.npz", kept=k, tracklet=np.full(rows, -1, np.int32),
                            tracker=np.full(rows, -1, np.int32))

    native = sync_native(drift)
    aligned_time = 0.1 * (np.arange(T) - 1.0)
    np.savez_compressed(root / "alignment/sync_table.npz", cameras=np.array(CAMERAS), aligned_time=aligned_time,
                        native=native, offsets_seconds=np.zeros((T, 3)))
    checkpoint = {"aligned_index": 0, "frames": {c: LABEL_OFFSETS[c] for c in CAMERAS},
                  "offsets": dict(LABEL_OFFSETS), "provenance": "data_engine.alignment.test"}
    alignment = {"schema_version": 2, "dataset": DATASET, "cameras": CAMERAS, "clock": "pts", "nominal_fps": 10.0,
                 "offsets": dict(LABEL_OFFSETS), "offsets_seconds": {c: 0.1 * v for c, v in LABEL_OFFSETS.items()},
                 "provenance": "data_engine.alignment.test", "checkpoints": [checkpoint],
                 "source": {"scene": "unit", "start": "20250101000000", "end": "20250101010000", "anchor": "c0"}}
    (root / "alignment/alignment.json").write_text(json.dumps(alignment), encoding="utf-8")
    (root / "alignment/provenance.json").write_text(json.dumps({
        "schema_version": 1, "name": "data_engine.alignment.test", "kind": "machine", "description": "test alignment",
        "registered_at": "2026-11-18T00:00:00+00:00",
        "reproduction": {"steps": ["align_dm.py"], "config": {},
                         "code": {"data_engine": {"git_commit": None, "note": "uncommitted"},
                                  "data_management": {"git_commit": "b" * 40, "git_dirty": False}}}}), encoding="utf-8")
    (root / "calibration/calibration.json").write_text(json.dumps(standard_calibration()), encoding="utf-8")
    (root / "manifests/detect_r000_1.json").write_text(json.dumps({
        "stage": "detect", "config": {"detect": {"imgsz": 1280}}, "code": NO_COMMIT,
        "inputs": {"yolo_onnx": "yolo.onnx", "yolo_sha256": "y" * 64, "dataset_json_sha256": "ignored"}}), encoding="utf-8")

    # solution: identity 0 = person A seen by all cameras at every frame; person B has no identity
    arrays = {"cameras": np.array(CAMERAS), "component_identity": np.array([0], np.int32)}
    for camera in CAMERAS:
        arrays[f"identity_{camera}"] = identity[camera]
        arrays[f"component_{camera}"] = np.where(identity[camera] >= 0, 0, -1).astype(np.int32)
        arrays[f"atom_{camera}"] = atom[camera]
        arrays[f"source_{camera}"] = source[camera]
    np.savez_compressed(root / "rounds/r001/solution.npz", **arrays)

    rows_t, xyz, used, rows = [], [], [], []
    for t in range(T):
        valid = [c for c in range(3) if native[t, c] >= 0]
        if len(valid) < 2:
            continue
        joints = np.full((17, 3), np.nan, np.float32)
        joints[2:] = np.array([50 + t, 20, 40], np.float32) + np.arange(15)[:, None] * 0.5
        if t == 5:
            joints[:] = np.nan  # nothing measured
        used_row = np.zeros(17, np.uint8)
        used_row[np.isfinite(joints).all(axis=1)] = sum(1 << c for c in valid)
        r = np.full(3, -1, np.int32)
        for c in valid:
            r[c] = 2 * native[t, c]
        rows_t.append(t), xyz.append(joints), used.append(used_row), rows.append(r)
    root_xyz = np.array([[50 + t, 20, 40] for t in rows_t], np.float32)
    root_xyz[[i for i, t in enumerate(rows_t) if t in (5, 6)]] = np.nan  # t=6: joints measured but no root
    xyz = np.array(xyz)
    xyz[[i for i, t in enumerate(rows_t) if t == 6]] = np.array([60, 20, 40], np.float32) + np.arange(17)[:, None] * 0.5
    np.savez_compressed(root / "rounds/r001/tracks3d.npz", component=np.zeros(len(rows_t), np.int32),
                        identity=np.zeros(len(rows_t), np.int32), t=np.array(rows_t, np.int32), rows=np.array(rows),
                        xyz=xyz, used=np.array(used), residual=np.zeros((len(rows_t), 17), np.float32), root=root_xyz,
                        aligned_time=aligned_time, native=native, cameras=np.array(CAMERAS))

    db = Database(root / "engine.sqlite3")
    db.add_constraint("same", ["c0/1/0"], ["c1/1/0"], source="human", round_number=0)
    db.add_constraint("same", ["c0/2/0"], ["c1/2/0"], source="human:ui", round_number=0)
    db.add_constraint("different", ["c0/3/0"], ["c0/3/1"], source="human", round_number=0)
    db.add_constraint("split", ["c0/4/0"], ["c0/5/0"], source="machine:rule", round_number=0)
    superseded = db.add_constraint("same", ["c0/6/0"], ["c1/6/0"], source="human", round_number=0)
    db.supersede_constraint(superseded, superseded)
    db.report_unknowns("solve", "association", [{"subject": "a:1", "severity": 1}, {"subject": "a:2", "severity": 1}], 1)
    db.report_unknowns("solve", "unassigned", [{"subject": "u:1", "severity": 1}], 1)
    db.set_meta("limit_frames", NATIVE_FRAMES)
    db.close()
    site = Site({"data_root": str(tmp_path), "calibration": str(root / "calibration/calibration.json"),
                 "interpreters": {"data_management": "dm-python"}})
    return root, site, native


@pytest.fixture()
def drafted(tmp_path, monkeypatch):
    root, site, native = make_workspace(tmp_path)
    monkeypatch.setattr(pm, "code_identity", lambda: dict(GOOD_CODE))
    drafts = pm.build_drafts(root, DATASET, "tagx", tmp_path / "out", stride=5)
    return root, native, drafts, tmp_path / "out"


def load(out: Path, *parts):
    return json.loads(out.joinpath(*parts).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- aligned index and lookup rule


def test_aligned_index_follows_payload_offsets(tmp_path):
    root, _, native = make_workspace(tmp_path)
    solved = pm.load_solved(root)
    aligned, shift = pm.aligned_indices(solved.aligned_time, solved.native, solved.cameras, solved.alignment)
    assert shift == 0 and aligned.tolist() == list(range(-1, T - 1))
    # a payload whose offsets are all 5 smaller implies aligned indices 5 larger: native = aligned + offset
    shifted = json.loads(json.dumps(solved.alignment))
    shifted["offsets"] = {c: v - 5 for c, v in shifted["offsets"].items()}
    aligned5, shift5 = pm.aligned_indices(solved.aligned_time, solved.native, solved.cameras, shifted)
    assert shift5 == 5 and np.array_equal(aligned5, aligned + 5)


def reference_native_frames(data, aligned_index):
    """data_management.label_sets.frame_alignment.native_frames / frame_offsets_near, verbatim."""
    offsets = {c: int(v) for c, v in (data.get("offsets") or {}).items() if v is not None}
    checkpoints = list(data.get("checkpoints") or [])
    if checkpoints:
        nearest = min(checkpoints, key=lambda c: abs(int(c["aligned_index"]) - aligned_index))
        offsets.update({c: int(v) for c, v in nearest["offsets"].items()})
    return {c: aligned_index + v for c, v in offsets.items()}


def test_lookup_matches_store_rule_including_ties_and_missing():
    rng = np.random.default_rng(3)
    cameras = ["a", "b", "c"]
    for _ in range(25):
        indices = sorted(rng.choice(np.arange(-20, 60), size=rng.integers(0, 8), replace=False).tolist())
        checkpoints = []
        for index in indices:
            offsets = {c: int(rng.integers(-5, 6)) for c in cameras if rng.random() > 0.2}
            checkpoints.append({"aligned_index": index, "frames": {c: index + o for c, o in offsets.items()},
                                "offsets": offsets, "provenance": "p"})
        top = {c: (None if rng.random() < 0.3 else int(rng.integers(-5, 6))) for c in cameras}
        data = {"offsets": top, "checkpoints": checkpoints}
        lookup = pm.AlignmentLookup(data, cameras)
        aligned = np.arange(-25, 65)
        got = lookup.frames(aligned)
        for row, a in enumerate(aligned):
            expected = reference_native_frames(data, int(a))
            for column, camera in enumerate(cameras):
                assert got[row, column] == expected.get(camera, -1), (a, camera)


def test_exact_alignment_is_kept_and_agreement_measured(tmp_path):
    root, _, native = make_workspace(tmp_path)
    solved = pm.load_solved(root)
    aligned, shift = pm.aligned_indices(solved.aligned_time, solved.native, solved.cameras, solved.alignment)
    published = aligned % 5 == 0
    payload, report = pm.build_alignment(solved.alignment, DATASET, aligned, native, published, CAMERAS, "auto", shift)
    assert report["mode"] == "engine" and payload["checkpoints"] == solved.alignment["checkpoints"]
    assert report["published_payload"]["max_abs_diff"] == 0 and report["published_payload"]["compared"] > 0
    assert payload["source"]["data_engine"]["shift"] == 0


def test_drifting_alignment_is_rederived_from_the_sync_table(tmp_path):
    root, _, native = make_workspace(tmp_path, drift=True)
    solved = pm.load_solved(root)
    aligned, shift = pm.aligned_indices(solved.aligned_time, solved.native, solved.cameras, solved.alignment)
    published = np.ones(len(aligned), bool)
    payload, report = pm.build_alignment(solved.alignment, DATASET, aligned, native, published, CAMERAS, "auto", shift)
    # the engine's single checkpoint cannot reproduce c1 after its frame step ...
    assert report["engine_payload"]["max_abs_diff"] == 1 and report["engine_payload"]["exact"] < report["engine_payload"]["compared"]
    # ... checkpoints derived from the sync table do, exactly, with few checkpoints (run ends only)
    assert report["mode"] == "dense" and report["published_payload"]["max_abs_diff"] == 0
    assert report["published_payload"]["missing"] == 0
    assert len(payload["checkpoints"]) < int(published.sum())
    assert payload["source"]["engine_checkpoints"] == solved.alignment["checkpoints"]
    indices = [c["aligned_index"] for c in payload["checkpoints"]]
    assert indices == sorted(set(indices))
    for checkpoint in payload["checkpoints"]:  # label-store invariants
        assert set(checkpoint["frames"]) == set(checkpoint["offsets"])
        assert all(f - checkpoint["offsets"][c] == checkpoint["aligned_index"] for c, f in checkpoint["frames"].items())
    # and the label store's own rule (nearest checkpoint) gives the sync table's frames at every published index
    for t in range(T):
        frames = reference_native_frames(payload, int(aligned[t]))
        for column, camera in enumerate(CAMERAS):
            if native[t, column] >= 0:
                assert frames[camera] == native[t, column]
    forced, _ = pm.build_alignment(solved.alignment, DATASET, aligned, native, published, CAMERAS, "engine", shift)
    assert forced["checkpoints"] == solved.alignment["checkpoints"]  # 'engine' never rewrites


# --------------------------------------------------------------------------- tracklet ids


def test_raw_tracklet_ids_pieces_runs_and_singles():
    frame = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], np.int64)
    atom = np.array([4, 4, -1, -1, -1, -1, -1, -1, 7, -1], np.int64)
    identity = np.array([0, 0, 2, 2, 2, 2, 3, 3, 3, -1], np.int64)
    kept = np.ones(10, bool)
    kept[5] = False
    ids, stats = pm.raw_tracklet_ids(frame, kept, atom, identity)
    assert ids[0] == ids[1] == 4 and ids[8] == 7 and ids[5] == -1
    assert ids[2] == ids[3] == ids[4] and ids[2] > 7  # one run of identity 2 (frame 5 not kept; broken run is fine)
    assert ids[6] == ids[7] != ids[2]  # identity 3, frames 6-7 consecutive
    assert ids[9] not in (ids[2], ids[6]) and ids[9] > 7
    assert stats["identity_runs"] == 2 and stats["single_detections"] == 1
    assert len(set(ids[ids >= 0])) == len(set(ids[ids >= 0].tolist()))


def test_duplicate_row_of_a_piece_in_one_frame_is_split():
    frame = np.array([0, 0, 1], np.int64)
    ids, stats = pm.raw_tracklet_ids(frame, np.ones(3, bool), np.array([1, 1, 1]), np.array([-1, -1, -1]))
    assert stats["duplicate_in_frame_split"] == 1
    assert len({(int(i), int(f)) for i, f in zip(ids, frame)}) == 3


def test_camera_tracklets_dense_ids_frames_and_values(tmp_path):
    root, _, _ = make_workspace(tmp_path)
    solved = pm.load_solved(root)
    from data_engine.store import load_camera
    detections = load_camera(root / "detections/raw", "c1", NATIVE_FRAMES)
    mask = np.zeros(NATIVE_FRAMES, bool)
    mask[[0, 5, 10]] = True
    payload, dense, stats = pm.build_camera_tracklets(
        "c1", DATASET, detections, solved.kept["c1"], solved.atom["c1"], solved.identity["c1"], mask, (2048, 1536), "pos", "trk")
    assert [f["frame"] for f in payload["frames"]] == [0, 5, 10] and payload["frame_range"] == [0, 10]
    ids = {d["tracklet_id"] for f in payload["frames"] for d in f["detections"]}
    assert ids == set(range(len(ids))) and stats["tracklets"] == len(ids)  # dense, camera-local, from 0
    for frame in payload["frames"]:
        frame_ids = [d["tracklet_id"] for d in frame["detections"]]
        assert len(frame_ids) == len(set(frame_ids))
    first = payload["frames"][0]["detections"]
    assert {d["source_id"] for d in first} == {"c1/0/0", "c1/0/1"}
    row = next(d for d in first if d["source_id"] == "c1/0/0")
    assert row["class"] == "person" and len(row["keypoints"]) == 17 and row["bbox"][0] == 0.1
    assert row["keypoints"][0] == [100.0, 200.0, 0.5] and row["score"] == 0.9
    assert payload["position_provenance"] == "pos" and payload["tracklet_provenance"] == "trk"
    assert payload["image_size"] == [2048, 1536] and (dense >= 0).sum() == stats["published_detections"] == 6
    assert (dense[[2 * 1, 2 * 1 + 1]] == -1).all()  # frame 1 not published


def test_nonfinite_keypoints_become_absent_joints(tmp_path):
    root, _, _ = make_workspace(tmp_path)
    solved = pm.load_solved(root)
    from data_engine.store import load_camera
    detections = load_camera(root / "detections/raw", "c0", NATIVE_FRAMES)
    detections.keypoints[0, 3] = np.nan
    payload, _, stats = pm.build_camera_tracklets(
        "c0", DATASET, detections, solved.kept["c0"], solved.atom["c0"], solved.identity["c0"], None, None, "p", "t")
    assert stats["nonfinite_values"] == 1
    json.dumps(payload, allow_nan=False)
    assert "image_size" not in payload


# --------------------------------------------------------------------------- drafts end to end


def test_drafts_contents_and_cross_references(drafted):
    root, native, drafts, out = drafted
    assert drafts.names == {k: f"{DATASET}.{k}.tagx" for k in ("frame_alignment", "two_d_tracklets", "three_d_tracks")}
    assert sorted(p.name for p in (out / "two_d_tracklets/cameras").iterdir()) == ["c0.json", "c1.json", "c2.json"]
    alignment = load(out, "frame_alignment", "alignment.json")
    assert alignment["dataset"] == DATASET
    tracks = load(out, "three_d_tracks", "tracks.json")
    assert tracks["units"] == "in" and tracks["keypoint_format"] == "coco17"
    calibration = tracks["calibration"]
    assert calibration["path"] == "calibration/calibration.json" and len(calibration["sha256"]) == 64
    assert calibration["standard_calibration"] == "v5_tracks_ransac" and calibration["world"]["up_axis"] == "z"
    assert calibration["cameras"]["c1"]["status"] == "shifted"

    two_d = {c: load(out, "two_d_tracklets", "cameras", f"{c}.json") for c in CAMERAS}
    frames_of = {}
    for camera, data in two_d.items():
        previous = -1
        for frame in data["frames"]:
            assert frame["frame"] > previous and data["frame_range"][0] <= frame["frame"] <= data["frame_range"][1]
            previous = frame["frame"]
            ids = [d["tracklet_id"] for d in frame["detections"]]
            assert len(ids) == len(set(ids)) and min(ids) >= 0
            for detection in frame["detections"]:
                frames_of.setdefault((camera, detection["tracklet_id"]), []).append(frame["frame"])
    # stride 5 on aligned = t - 1: published t = 1, 6, 11 -> native frames native[t, c]
    for column, camera in enumerate(CAMERAS):
        expected = sorted(int(native[t, column]) for t in (1, 6, 11) if native[t, column] >= 0)
        assert [f["frame"] for f in two_d[camera]["frames"]] == expected
    # c2 person B rows before frame 4 were post-filtered: not published; kept rows are
    c2_first = {d["source_id"] for d in two_d["c2"]["frames"][0]["detections"]}
    assert "c2/0/1" not in c2_first and "c2/0/0" in c2_first

    assert [t["track_id"] for t in tracks["tracks"]] == [0]
    track = tracks["tracks"][0]
    assert track["class"] == "person" and track["provenance"].startswith("data_engine.solve.")
    for member in track["members"]:  # every member exists in the published tracklets, inside its frames
        frames = frames_of[(member["camera"], member["tracklet_id"])]
        assert min(frames) <= member["frame_range"][0] <= member["frame_range"][1] <= max(frames)
    # positions: aligned 0 (t=1) and 10 (t=11); t=6 has joints but no root -> omitted; nothing interpolated
    assert [p["aligned_index"] for p in track["positions"]] == [0, 10]
    first = track["positions"][0]
    assert first["xyz"] == [51.0, 20.0, 40.0] and len(first["keypoints"]) == 17
    assert first["keypoints"][0] is None and first["keypoints"][1] is None and first["keypoints"][2] is not None
    assert drafts.summary["tracks"]["rows_with_joints_but_no_root_omitted"] == 1


def test_aligned_index_agreement_with_sync_table_is_exact(drafted):
    root, native, drafts, out = drafted
    report = drafts.summary["alignment"]
    assert report["mode"] == "engine" and report["positions"]["max_abs_diff"] == 0 and report["positions"]["missing"] == 0
    assert report["positions"]["camera_observations"] > 0
    alignment = load(out, "frame_alignment", "alignment.json")
    tracks = load(out, "three_d_tracks", "tracks.json")
    t_of_aligned = {a: t for t, a in enumerate(range(-1, T - 1))}
    for position in tracks["tracks"][0]["positions"]:
        frames = reference_native_frames(alignment, position["aligned_index"])
        t = t_of_aligned[position["aligned_index"]]
        for column, camera in enumerate(CAMERAS):
            if native[t, column] >= 0:
                assert abs(frames[camera] - native[t, column]) <= 1


def test_stride_one_publishes_every_kept_detection_including_off_timeline(tmp_path, monkeypatch):
    root, _, native = make_workspace(tmp_path)
    monkeypatch.setattr(pm, "code_identity", lambda: dict(GOOD_CODE))
    drafts = pm.build_drafts(root, DATASET, "all", tmp_path / "all", stride=1)
    summary = drafts.summary
    kept = sum(int(np.load(root / "tracklets" / f"{c}.npz")["kept"].sum()) for c in CAMERAS)
    assert summary["detections"]["published"] == summary["detections"]["kept_total"] == kept
    assert summary["detections"]["kept_off_common_timeline"] > 0  # frames the common timeline does not show
    assert summary["published_sync_indices"] == T
    tracks = load(tmp_path / "all", "three_d_tracks", "tracks.json")
    # t=5 has no measurement and t=6 no root: omitted; all other sync indices with a root are published
    assert len(tracks["tracks"][0]["positions"]) == len([t for t in range(T) if (native[t] >= 0).sum() >= 2]) - 2


def test_drift_workspace_uses_dense_checkpoints_and_agrees(tmp_path, monkeypatch):
    root, _, native = make_workspace(tmp_path, drift=True)
    monkeypatch.setattr(pm, "code_identity", lambda: dict(GOOD_CODE))
    drafts = pm.build_drafts(root, DATASET, "drift", tmp_path / "d", stride=1)
    report = drafts.summary["alignment"]
    assert report["mode"] == "dense" and report["engine_payload"]["max_abs_diff"] == 1
    assert report["published_payload"]["max_abs_diff"] == 0 and report["positions"]["max_abs_diff"] == 0


def test_provenance_notes_and_commit_policy(tmp_path, monkeypatch):
    root, _, _ = make_workspace(tmp_path)
    monkeypatch.setattr(pm, "code_identity", lambda: dict(NO_COMMIT))
    drafts = pm.build_drafts(root, DATASET, "nc", tmp_path / "nc")
    summary = drafts.summary
    assert "no git commit" in summary["provenance"]["commit_problem"]  # publishing would be refused, drafts still built
    assert (tmp_path / "nc/three_d_tracks/tracks.json").is_file()
    names = summary["provenance"]["names"]
    assert [n.split(".")[1] for n in names] == ["alignment", "detect", "solve"]
    solve = next(p for n, p in drafts.provenances.items() if ".solve." in n)
    assert solve["kind"] == "machine" and solve["reproduction"]["steps"] == [
        "data-engine run unit_20250101000000_20250101010000@pilot12",
        "data-engine publish unit_20250101000000_20250101010000@pilot12"]
    config = solve["reproduction"]["config"]
    assert config["round"] == 1 and config["constraints"]["human_active"] == {"different": 1, "same": 2}
    assert config["constraints"]["machine_active"] == {"split": 1} and config["constraints"]["superseded_inactive"] == 1
    assert set(config["recipe"]) == {"postfilter", "tracklets", "geometry", "association", "temporal"}
    assert solve["reproduction"]["code"] == NO_COMMIT
    detect = next(p for n, p in drafts.provenances.items() if ".detect." in n)
    assert detect["reproduction"]["config"]["detect"] == {"imgsz": 1280}
    assert detect["reproduction"]["config"]["models_sha256"] == {"yolo_sha256": "y" * 64}
    # the alignment provenance's mapping of repositories is flattened to the code identity the store requires
    align = drafts.provenances["data_engine.alignment.test"]
    assert align["reproduction"]["code"] == NO_COMMIT and align["reproduction"]["code_components"]["data_management"]
    # same inputs -> same names; a different commit -> different names (provenances are immutable)
    monkeypatch.setattr(pm, "code_identity", lambda: dict(GOOD_CODE))
    again = pm.build_drafts(root, DATASET, "nc", tmp_path / "nc2")
    assert again.summary["provenance"]["commit_problem"] is None
    assert set(again.summary["provenance"]["names"]) & set(names) == {"data_engine.alignment.test"}
    notes = again.notes["three_d_tracks"]
    for phrase in ("Machine prelabels", "round 1", "Human answers applied: 3 (different 1, same 2)", "provisional unless human-confirmed",
                   "unmeasured joints are null", "nothing is interpolated", "association 2, unassigned 1 (total 3)",
                   "pilot slice 'pilot12'"):
        assert phrase in notes, phrase
    assert "%) belong to multi-camera 3D tracks" in notes


def test_default_tag_and_dataset_split():
    assert pm.default_tag(1, "pilot40s") == "data_engine_pilot40s_r001"
    assert pm.default_tag(12) == "data_engine_r012"
    assert pm.split_workspace_id("a_b_c@pilot40s") == ("a_b_c", "pilot40s")
    assert pm.split_workspace_id("a_b_c") == ("a_b_c", None)
    assert pm.commit_problem({"git_commit": "a" * 40, "git_dirty": True}) is not None  # dirty needs a diff digest
    assert pm.commit_problem({"git_commit": "a" * 40, "git_dirty": False}) is None


def test_estimate_sizes_scales_with_stride():
    measured = {"bytes_per_detection": 500.0, "bytes_per_position": 400.0, "positions_per_sync_index": 10.0}
    five, one = pm.estimate_sizes(measured, stride=5), pm.estimate_sizes(measured, stride=1)
    assert one["detections"] == 5 * five["detections"] and one["total_mb"] > 4.9 * five["total_mb"]


# --------------------------------------------------------------------------- command line policy (engine venv)


def test_publish_without_store_root_is_refused_before_anything_is_built(tmp_path, capsys):
    root, _, _ = make_workspace(tmp_path)
    code = pm.main(["--workspace", str(root), "--dataset", DATASET, "--out", str(tmp_path / "o"), "--publish"])
    assert code == 2 and "real label store" in capsys.readouterr().out
    result = load(tmp_path / "o", "result.json")
    assert result["ok"] is False and "real label store" in result["refused"]
    assert not (tmp_path / "o/three_d_tracks").exists()


def test_uncommitted_test_flag_needs_a_store_root(tmp_path):
    root, _, _ = make_workspace(tmp_path)
    code = pm.main(["--workspace", str(root), "--dataset", DATASET, "--out", str(tmp_path / "o"), "--publish",
                    "--allow-real-store", "--allow-uncommitted-test"])
    assert code == 2


# --------------------------------------------------------------------------- publish stage (worker faked)


def test_stage_runs_worker_drafts_only_by_default(tmp_path, monkeypatch):
    from data_engine.stages import ingest, publish

    root, site, _ = make_workspace(tmp_path)
    workspace = Workspace(root.name, site, root=root)
    context = Context(workspace, 1, "publish")
    calls = []

    def fake_worker(ctx, command, **kwargs):
        calls.append([str(c) for c in command])
        argv = calls[-1]
        out = Path(argv[argv.index("--out") + 1])
        monkeypatch.setattr(pm, "code_identity", lambda: dict(NO_COMMIT))
        drafts = pm.build_drafts(root, DATASET, argv[argv.index("--tag") + 1], out, stride=int(argv[argv.index("--stride") + 1]))
        drafts.summary["check"] = {"ok": True, "error_count": 0, "warning_count": 0}
        (out / "summary.json").write_text(json.dumps(drafts.summary), encoding="utf-8")
        (out / "result.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        return 0

    monkeypatch.setattr(publish, "run_worker", fake_worker)
    result = publish.run(context)
    assert isinstance(result, StageResult) and result.status == "done"
    assert "drafts validated, not published" in result.message
    command = calls[0]
    assert "--publish" not in command and command[command.index("--stride") + 1] == "1"
    assert command[command.index("--tag") + 1] == "data_engine_pilot12_r001"
    assert command[command.index("--dataset") + 1] == DATASET
    assert result.metrics["sets"]["three_d_tracks"]["positions"] == 10 and result.metrics["validation_errors"] == 0
    assert result.metrics["published"] == [] and Path(result.outputs["drafts"]) == Path("published/data_engine_pilot12_r001")
    assert ingest.WORKERS.name == "workers"


def test_stage_publishes_only_when_the_operator_enabled_it(tmp_path, monkeypatch):
    from data_engine.stages import publish

    root, site, _ = make_workspace(tmp_path)
    workspace = Workspace(root.name, site, root=root)
    workspace.db.set_meta("publish_to_store", True)
    workspace.db.set_meta("publish_stride", 1)
    seen = []

    def refusing_worker(ctx, command, **kwargs):
        seen.append([str(c) for c in command])
        out = Path(seen[-1][seen[-1].index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps({"ok": False, "refused": "publishing refused: no git commit",
                                                     "validation_errors": 0}), encoding="utf-8")
        return 2

    monkeypatch.setattr(publish, "run_worker", refusing_worker)
    result = publish.run(Context(workspace, 1, "publish"))
    assert result.status == "blocked" and "no git commit" in result.message
    assert "--publish" in seen[0] and "--allow-real-store" in seen[0] and "--store-root" not in seen[0]
    assert seen[0][seen[0].index("--stride") + 1] == "1"
    workspace.db.set_meta("publish_store_root", str(tmp_path / "teststore"))
    publish.run(Context(workspace, 1, "publish"))
    assert "--allow-real-store" not in seen[1] and seen[1][seen[1].index("--store-root") + 1] == str(tmp_path / "teststore")


# --------------------------------------------------------------------------- needs data_management (skipped in the engine venv)


def test_drafts_validate_and_publish_into_a_temp_store(drafted, tmp_path):
    pytest.importorskip("data_management.label_sets")
    root, native, drafts, out = drafted
    check = pm.check_drafts(drafts)
    assert check["ok"], check
    store_root = tmp_path / "store"
    published = pm.publish_into(store_root, drafts, replace=False, placeholder="TEST ONLY - placeholder commit")
    assert [p["label_type"] for p in published] == ["frame_alignment", "two_d_tracklets", "three_d_tracks"]
    assert pm.validate_root(store_root)["ok"]
    rule = pm.verify_with_store_rule(drafts, root)
    assert rule["compared"] > 0 and rule["max_abs_diff"] == 0
    with pytest.raises(Exception):  # republishing an existing name needs replace
        pm.publish_into(store_root, drafts, replace=False, placeholder="TEST ONLY - placeholder commit")
    pm.publish_into(store_root, drafts, replace=True, placeholder="TEST ONLY - placeholder commit")
