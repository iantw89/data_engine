"""Publish worker: convert the engine's solution into data_management label sets.

Runs in data_management's interpreter (``python publish_dm.py --workspace DIR --dataset ID --tag TAG
--out DRAFT_DIR [--stride 1] [--store-root DIR] [--publish] [--replace]``). Three drafts are built under
``--out``:

    frame_alignment/alignment.json
    two_d_tracklets/cameras/<camera>.json        one file per camera
    three_d_tracks/tracks.json

then validated exactly as the store would (in a throw-away scratch store, so dependencies and member
checks run too), and only with ``--publish`` registered/published into the label store.

Time and frames (the contract this worker keeps)
------------------------------------------------
The engine's sync index ``t`` (10 Hz common timeline) maps to the label store's ``aligned_index`` by
``aligned_index = round(aligned_time[t] * nominal_fps) + K``, K being the integer for which the frame_alignment
payload's offsets reproduce the sync table (``native[t, c] = aligned_index + offset[c]``). Every published
frame_alignment payload is measured against the sync table with the label store's own lookup rule (nearest
checkpoint, else top-level offsets); when the engine's checkpoints (sparse, one per update window) do not
reproduce ``native[t, c]`` exactly at every published ``t``, checkpoints are re-derived from the sync table
(``--alignment auto|engine|dense``). Nothing is rounded to fit: the agreement is reported.

Sampling: ``--stride N`` publishes the sync indices with ``aligned_index % N == 0``. A 2D detection is published
when its native frame is ``native[t, camera]`` of a published ``t`` (stride > 1). With stride 1 every kept
detection of every native frame is published, including frames outside the common timeline and cameras without
an alignment (they simply have no aligned index). 3D positions are published at the published ``t`` only.

The pure builders below need only numpy; ``data_management`` is imported lazily (check/publish), so the builders
are importable and testable in the engine venv.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_engine.config import RECIPE  # noqa: E402
from data_engine.dataset import Dataset  # noqa: E402
from data_engine.provenance import code_identity  # noqa: E402
from data_engine.store import load_camera  # noqa: E402

PRODUCER = "data_engine publish_dm"
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}([0-9a-f]{24})?")
COMPACT_DATA_BYTES = 256 * 1024  # data_management.label_sets.store.COMPACT_DATA_BYTES
TAG_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


class PublishError(RuntimeError):
    """The solution cannot be converted (stale artifacts, missing inputs)."""


class Refused(RuntimeError):
    """Publishing is refused by policy (no commit, real store not allowed, existing set)."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"[publish_dm] {message}", file=sys.stderr, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def short_digest(value: Any, length: int = 10) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:length]


def dump_data_json(value: Any) -> str:
    """Same layout rule as the label store: indented while small, compact one-liner when large."""
    compact = json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
    if len(compact) > COMPACT_DATA_BYTES:
        return compact + "\n"
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n"


def write_json(path: Path, value: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = dump_data_json(value)
    path.write_text(text, encoding="utf-8", newline="\n")
    return len(text.encode("utf-8"))


def commit_problem(code: dict | None) -> str | None:
    """Why ``code`` cannot identify machine-produced labels (label rules), or None."""
    if not isinstance(code, dict) or not isinstance(code.get("git_commit"), str) \
            or not COMMIT_PATTERN.fullmatch(code["git_commit"]):
        return ("the data_engine repository has no git commit (or it cannot be read); machine labels need the exact "
                "code identity - commit the code, then publish again")
    if code.get("git_dirty") and not code.get("diff_sha256"):
        return "uncommitted changes without a diff digest"
    return None


def default_tag(round_number: int, slice_name: str | None = None) -> str:
    stem = f"data_engine_{slice_name}_" if slice_name else "data_engine_"
    return TAG_CHARS.sub("_", f"{stem}r{round_number:03d}")


def split_workspace_id(value: str) -> tuple[str, str | None]:
    """``<dataset>@<slice>`` -> (dataset, slice)."""
    dataset, _, slice_name = value.partition("@")
    return dataset, (slice_name or None)


# ---------------------------------------------------------------------------
# Loading the solved round
# ---------------------------------------------------------------------------


@dataclass
class Solved:
    root: Path
    round: int
    round_dir: Path
    cameras: list[str]  # sync-table order
    native: np.ndarray  # (T, C) int
    aligned_time: np.ndarray  # (T,)
    tracks: dict[str, np.ndarray]
    identity: dict[str, np.ndarray]
    atom: dict[str, np.ndarray]
    source: dict[str, np.ndarray]
    kept: dict[str, np.ndarray]
    alignment: dict
    alignment_provenance: dict | None
    calibration_path: Path | None
    artifact_sha256: dict[str, str] = field(default_factory=dict)


def find_round(root: Path) -> tuple[int, Path]:
    rounds = sorted((p for p in (root / "rounds").glob("r[0-9]*") if p.is_dir()), key=lambda p: p.name, reverse=True)
    for directory in rounds:
        if (directory / "solution.npz").is_file() and (directory / "tracks3d.npz").is_file():
            return int(directory.name[1:]), directory
    raise PublishError(f"{root}: no round with solution.npz and tracks3d.npz; run the solve stage first")


def load_solved(root: Path) -> Solved:
    root = Path(root)
    round_number, directory = find_round(root)
    with np.load(directory / "tracks3d.npz") as data:
        tracks = {name: data[name] for name in data.files}
    cameras = [str(c) for c in tracks["cameras"]]
    sync_path = root / "alignment" / "sync_table.npz"
    if not sync_path.is_file():
        raise PublishError(f"{sync_path} missing; run the align stage")
    with np.load(sync_path) as sync:
        sync_cameras = [str(c) for c in sync["cameras"]]
        if sync_cameras != cameras or not np.array_equal(sync["native"], tracks["native"]) \
                or not np.allclose(sync["aligned_time"], tracks["aligned_time"]):
            raise PublishError("alignment/sync_table.npz differs from the sync table embedded in tracks3d.npz: the "
                               "solution is stale relative to the alignment; run the solve stage again")
    identity, atom, source, kept = {}, {}, {}, {}
    with np.load(directory / "solution.npz") as solution:
        if [str(c) for c in solution["cameras"]] != cameras:
            raise PublishError("solution.npz and tracks3d.npz list different cameras")
        for camera in cameras:
            identity[camera] = solution[f"identity_{camera}"].astype(np.int64)
            atom[camera] = solution[f"atom_{camera}"].astype(np.int64) if f"atom_{camera}" in solution.files \
                else np.full(len(identity[camera]), -1, np.int64)
            source[camera] = solution[f"source_{camera}"]
    for camera in cameras:
        with np.load(root / "tracklets" / f"{camera}.npz") as tracklets:
            kept[camera] = tracklets["kept"].astype(bool)
        if len(kept[camera]) != len(identity[camera]):
            raise PublishError(f"{camera}: tracklets/ and solution.npz disagree on the number of detections")
    alignment = json.loads((root / "alignment" / "alignment.json").read_text(encoding="utf-8"))
    provenance_path = root / "alignment" / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else None
    calibration = root / "calibration" / "calibration.json"
    return Solved(root, round_number, directory, cameras, tracks["native"].astype(np.int64), tracks["aligned_time"],
                  tracks, identity, atom, source, kept, alignment, provenance,
                  calibration if calibration.is_file() else None,
                  {"solution.npz": sha256_file(directory / "solution.npz"),
                   "tracks3d.npz": sha256_file(directory / "tracks3d.npz"),
                   "sync_table.npz": sha256_file(sync_path)})


# ---------------------------------------------------------------------------
# Alignment: aligned index of a sync index, lookup rule, agreement, dense checkpoints
# ---------------------------------------------------------------------------


def aligned_indices(aligned_time: np.ndarray, native: np.ndarray, cameras: list[str],
                    alignment: dict) -> tuple[np.ndarray, int]:
    """``aligned_index`` of every sync index: ``round(aligned_time * fps) + K`` with K from the payload's offsets."""
    fps = float(alignment.get("nominal_fps") or 10.0)
    base = np.rint(np.asarray(aligned_time, float) * fps).astype(np.int64)
    if len(base) > 1 and not np.all(np.diff(base) > 0):
        raise PublishError("aligned_time * nominal_fps is not strictly increasing; cannot define aligned indices")
    votes = []
    for column, camera in enumerate(cameras):
        offset = (alignment.get("offsets") or {}).get(camera)
        ok = native[:, column] >= 0
        if offset is not None and ok.any():
            votes.append(native[ok, column] - int(offset) - base[ok])
    if not votes:
        raise PublishError("no camera has both an offset in alignment.json and frames in the sync table")
    values, counts = np.unique(np.concatenate(votes), return_counts=True)
    shift = int(values[np.argmax(counts)])
    return base + shift, shift


class AlignmentLookup:
    """``frame_alignment.native_frames`` for many aligned indices at once (nearest checkpoint, else top-level)."""

    def __init__(self, alignment: dict, cameras: list[str]):
        self.cameras = list(cameras)
        top = alignment.get("offsets") or {}
        self.top = np.array([np.nan if top.get(c) is None else float(top[c]) for c in cameras])
        checkpoints = sorted(alignment.get("checkpoints") or [], key=lambda c: int(c["aligned_index"]))
        self.index = np.array([int(c["aligned_index"]) for c in checkpoints], dtype=np.int64)
        self.offsets = np.full((len(checkpoints), len(cameras)), np.nan)
        for row, checkpoint in enumerate(checkpoints):
            for column, camera in enumerate(cameras):
                if camera in checkpoint["offsets"]:
                    self.offsets[row, column] = float(checkpoint["offsets"][camera])

    def frames(self, aligned: np.ndarray) -> np.ndarray:
        """(n, C) native frame per camera, -1 where the camera has no offset."""
        aligned = np.asarray(aligned, dtype=np.int64)
        offsets = np.tile(self.top, (len(aligned), 1))
        if len(self.index):
            at = np.searchsorted(self.index, aligned)
            left, right = np.clip(at - 1, 0, len(self.index) - 1), np.clip(at, 0, len(self.index) - 1)
            nearest = np.where(np.abs(aligned - self.index[left]) <= np.abs(self.index[right] - aligned), left, right)
            chosen = self.offsets[nearest]
            offsets = np.where(np.isnan(chosen), offsets, chosen)
        frames = aligned[:, None] + offsets
        return np.where(np.isnan(offsets), -1, np.rint(np.nan_to_num(frames))).astype(np.int64)


def alignment_agreement(lookup: AlignmentLookup, aligned: np.ndarray, native: np.ndarray, published: np.ndarray,
                        cameras: list[str]) -> dict:
    """Compare the lookup's native frames with the sync table at every published sync index and resolved camera."""
    selected = np.flatnonzero(published)
    got, want = lookup.frames(aligned[selected]), native[selected]
    valid = want >= 0
    difference = np.where(valid & (got >= 0), np.abs(got - want), 0)
    missing = valid & (got < 0)
    compared = int(valid.sum())
    per_camera = {}
    for column, camera in enumerate(cameras):
        v = valid[:, column]
        if v.any():
            per_camera[camera] = {"compared": int(v.sum()), "exact": int(((difference[:, column] == 0) & ~missing[:, column] & v).sum()),
                                  "max_abs_diff": int(difference[:, column][v].max()), "missing": int(missing[:, column].sum())}
    exact = int(((difference == 0) & valid & ~missing).sum())
    return {"compared": compared, "exact": exact, "within_1_frame": int(((difference <= 1) & valid & ~missing).sum()),
            "max_abs_diff": int(difference.max()) if difference.size else 0, "missing": int(missing.sum()),
            "published_sync_indices": int(len(selected)), "per_camera": per_camera}


def positions_agreement(lookup: AlignmentLookup, aligned: np.ndarray, native: np.ndarray, t: np.ndarray,
                        used: np.ndarray, rows: np.ndarray, published: np.ndarray) -> dict:
    """Agreement for the cameras that actually measured each published 3D position (``used`` bitmask)."""
    sel = np.flatnonzero(published[t])
    bits = np.bitwise_or.reduce(used[sel], axis=1) if len(sel) else np.zeros(0, np.uint8)
    got, want = lookup.frames(aligned[t[sel]]), native[t[sel]]
    compared = exact = within = missing = 0
    worst = 0
    for column in range(native.shape[1]):
        measured = ((bits >> column) & 1).astype(bool)
        if not measured.any():
            continue
        g, w = got[measured, column], want[measured, column]
        compared += int(len(g))
        missing += int(((g < 0) | (w < 0)).sum())
        ok = (g >= 0) & (w >= 0)
        d = np.abs(g[ok] - w[ok])
        exact += int((d == 0).sum())
        within += int((d <= 1).sum())
        worst = max(worst, int(d.max()) if d.size else 0)
    return {"positions": int(len(sel)), "camera_observations": compared, "exact": exact, "within_1_frame": within,
            "max_abs_diff": worst, "missing": missing}


def dense_alignment(alignment: dict, aligned: np.ndarray, native: np.ndarray, published: np.ndarray,
                    cameras: list[str]) -> dict:
    """Re-derive checkpoints from the sync table: the first and last published index of every run of constant offsets.

    The label lookup uses the *nearest* checkpoint, so a run of identical offsets is represented exactly by its
    two ends (every index inside the run is nearer to an end of its own run than to any other run).
    """
    payload = copy.deepcopy(alignment)
    allowed = set(payload["cameras"])
    entries = []
    for t in np.flatnonzero(published):
        frames = {camera: int(native[t, column]) for column, camera in enumerate(cameras)
                  if native[t, column] >= 0 and camera in allowed}
        if frames:
            entries.append((int(aligned[t]), int(t), frames))
    entries.sort()
    keep: list[tuple[int, int, dict]] = []
    run: list[tuple[int, int, dict]] = []

    def signature(entry):
        index, _, frames = entry
        return tuple(sorted((camera, frame - index) for camera, frame in frames.items()))

    def flush():
        if run:
            keep.append(run[0])
            if len(run) > 1:
                keep.append(run[-1])

    for entry in entries:
        if run and signature(run[-1]) != signature(entry):
            flush()
            run = []
        run.append(entry)
    flush()
    provenance = payload["provenance"]
    payload["checkpoints"] = [
        {"aligned_index": index, "frames": frames, "offsets": {c: f - index for c, f in frames.items()},
         "provenance": provenance,
         "evidence": {"derived_from": "alignment/sync_table.npz", "sync_index": t}}
        for index, t, frames in keep]
    source = dict(payload.get("source") or {})
    source["engine_checkpoints"] = alignment.get("checkpoints") or []
    source["checkpoints_note"] = ("checkpoints re-derived from the engine sync table (first and last published index "
                                  "of each run of constant offsets) because the engine's own checkpoints do not "
                                  "reproduce the frames the 3D positions were triangulated from")
    payload["source"] = source
    return payload


def build_alignment(engine_payload: dict, dataset: str, aligned: np.ndarray, native: np.ndarray, published: np.ndarray,
                    cameras: list[str], mode: str = "auto", shift: int | None = None, extra_source: dict | None = None
                    ) -> tuple[dict, dict]:
    """The frame_alignment payload to publish and the measured agreement with the sync table."""
    if mode not in ("auto", "engine", "dense"):
        raise ValueError(mode)
    payload = copy.deepcopy(engine_payload)
    payload["dataset"] = dataset
    before = alignment_agreement(AlignmentLookup(payload, cameras), aligned, native, published, cameras)
    exact = before["max_abs_diff"] == 0 and before["missing"] == 0
    used = "engine"
    if mode == "dense" or (mode == "auto" and not exact):
        payload, used = dense_alignment(payload, aligned, native, published, cameras), "dense"
    source = dict(payload.get("source") or {})
    source["data_engine"] = {"aligned_index_rule": "aligned_index = round(aligned_time[t] * nominal_fps) + shift",
                             "shift": shift, **(extra_source or {})}
    payload["source"] = source
    after = alignment_agreement(AlignmentLookup(payload, cameras), aligned, native, published, cameras)
    return payload, {"mode": used, "engine_payload": before, "published_payload": after, "shift": shift}


# ---------------------------------------------------------------------------
# 2D tracklets
# ---------------------------------------------------------------------------


def raw_tracklet_ids(frame_of_row: np.ndarray, kept: np.ndarray, atom: np.ndarray, identity: np.ndarray
                     ) -> tuple[np.ndarray, dict]:
    """Camera-local tracklet id per kept row (-1 otherwise), independent of the published subset.

    Rows of a piece (``atom`` >= 0) share its id. A kept row without piece gets a fresh id: one per run of
    consecutive frames among its camera's rows attached to the same identity, or one per detection when it is
    attached to no identity. If two rows of one id fall in the same frame the later ones get fresh ids.
    """
    rows_total = len(kept)
    ids = np.full(rows_total, -1, dtype=np.int64)
    piece = kept & (atom >= 0)
    ids[piece] = atom[piece]
    nxt = int(atom.max()) + 1 if rows_total and atom.max() >= 0 else 0
    runs = 0
    attached = np.flatnonzero(kept & (atom < 0) & (identity >= 0))
    if len(attached):
        attached = attached[np.lexsort((attached, frame_of_row[attached], identity[attached]))]
        who, frame = identity[attached], frame_of_row[attached]
        start = np.ones(len(attached), bool)
        start[1:] = (who[1:] != who[:-1]) | (frame[1:] != frame[:-1] + 1)
        run = np.cumsum(start) - 1
        ids[attached] = nxt + run
        runs = int(run[-1]) + 1
        nxt += runs
    single = np.flatnonzero(kept & (atom < 0) & (identity < 0))
    ids[single] = nxt + np.arange(len(single))
    nxt += len(single)
    present = np.flatnonzero(ids >= 0)
    split = 0
    if len(present):
        order = present[np.lexsort((present, frame_of_row[present], ids[present]))]
        same = np.zeros(len(order), bool)
        same[1:] = (ids[order][1:] == ids[order][:-1]) & (frame_of_row[order][1:] == frame_of_row[order][:-1])
        duplicates = order[same]
        ids[duplicates] = nxt + np.arange(len(duplicates))
        split = int(len(duplicates))
    return ids, {"pieces": int(len(np.unique(atom[piece]))), "identity_runs": runs, "single_detections": int(len(single)),
                 "duplicate_in_frame_split": split}


def published_frame_mask(frame_count: int, native: np.ndarray, column: int, published: np.ndarray, stride: int
                         ) -> np.ndarray | None:
    """Native frames of a camera whose detections are published, or None for 'all frames' (stride 1)."""
    if stride == 1:
        return None
    mask = np.zeros(frame_count, dtype=bool)
    frames = native[published, column]
    frames = frames[(frames >= 0) & (frames < frame_count)]
    mask[frames] = True
    return mask


def build_camera_tracklets(camera: str, dataset: str, detections, kept: np.ndarray, atom: np.ndarray, identity: np.ndarray,
                           frame_mask: np.ndarray | None, image_size: tuple[int, int] | None,
                           position_provenance: str, tracklet_provenance: str) -> tuple[dict | None, np.ndarray, dict]:
    """One camera's two_d_tracklets file, the dense tracklet id of every raw row (-1 = unpublished) and counts."""
    frame_of_row = detections.frame_of_row.astype(np.int64)
    raw_ids, id_stats = raw_tracklet_ids(frame_of_row, kept, atom, identity)
    rows = np.flatnonzero(raw_ids >= 0)
    if frame_mask is not None:
        rows = rows[frame_mask[frame_of_row[rows]]]
    dense = np.full(len(kept), -1, dtype=np.int64)
    stats = {**id_stats, "published_detections": int(len(rows)), "nonfinite_values": 0}
    if not len(rows):
        return None, dense, stats
    frames = frame_of_row[rows]
    # dense ids ordered by first published frame, then raw id
    ordered = rows[np.lexsort((rows, frames))]
    raw_sorted = raw_ids[ordered]
    uniq, first = np.unique(raw_sorted, return_index=True)
    rank = np.lexsort((uniq, frame_of_row[ordered][first]))
    mapping = np.empty(len(uniq), dtype=np.int64)
    mapping[rank] = np.arange(len(uniq))
    dense[rows] = mapping[np.searchsorted(uniq, raw_ids[rows])]
    tids = dense[rows]

    boxes = np.asarray(detections.boxes[rows], dtype=np.float64)
    keypoints = np.asarray(detections.keypoints[rows], dtype=np.float64)
    bad = ~np.isfinite(boxes).all(axis=1) | ~np.isfinite(keypoints).all(axis=(1, 2))
    stats["nonfinite_values"] = int(bad.sum())
    keypoints = np.nan_to_num(keypoints, nan=0.0, posinf=0.0, neginf=0.0)  # absent joint: [0, 0, 0] (COCO convention)
    boxes = np.nan_to_num(boxes, nan=0.0, posinf=0.0, neginf=0.0)
    box_list = np.round(boxes[:, :4], 1).tolist()
    score_list = np.round(boxes[:, 4], 4).tolist()
    xy = np.round(keypoints[:, :, :2], 1)
    conf = np.round(keypoints[:, :, 2], 3)
    keypoint_list = np.concatenate([xy, conf[:, :, None]], axis=2).tolist()
    start = detections.start
    order = np.lexsort((tids, frames))
    out_frames = []
    current, bucket = None, None
    for i in order:
        frame = int(frames[i])
        if frame != current:
            bucket = []
            out_frames.append({"frame": frame, "detections": bucket})
            current = frame
        row = int(rows[i])
        bucket.append({"class": "person", "tracklet_id": int(tids[i]), "bbox": box_list[i],
                       "keypoints": keypoint_list[i], "score": score_list[i],
                       "source_id": f"{camera}/{frame}/{row - int(start[frame])}"})
    payload = {"schema_version": 1, "dataset": dataset, "camera": camera, "keypoint_format": "coco17",
               "frame_range": [out_frames[0]["frame"], out_frames[-1]["frame"]],
               "position_provenance": position_provenance, "tracklet_provenance": tracklet_provenance,
               "frames": out_frames}
    if image_size:
        payload["image_size"] = [int(image_size[0]), int(image_size[1])]
    stats["published_frames"] = len(out_frames)
    stats["tracklets"] = int(len(uniq))
    return payload, dense, stats


# ---------------------------------------------------------------------------
# 3D tracks
# ---------------------------------------------------------------------------


def _members(identity: np.ndarray, dense: np.ndarray, frame_of_row: np.ndarray) -> dict[int, list[tuple[int, int, int]]]:
    """identity -> [(tracklet_id, first frame, last frame)] over published rows attached to an identity."""
    rows = np.flatnonzero((identity >= 0) & (dense >= 0))
    if not len(rows):
        return {}
    key = identity[rows] * (int(dense[rows].max()) + 1) + dense[rows]
    uniq, inverse = np.unique(key, return_inverse=True)
    low = np.full(len(uniq), np.iinfo(np.int64).max)
    high = np.full(len(uniq), -1)
    np.minimum.at(low, inverse, frame_of_row[rows])
    np.maximum.at(high, inverse, frame_of_row[rows])
    first_row = np.zeros(len(uniq), dtype=np.int64)
    first_row[inverse[::-1]] = rows[::-1]
    out: dict[int, list[tuple[int, int, int]]] = {}
    for i in range(len(uniq)):
        out.setdefault(int(identity[first_row[i]]), []).append((int(dense[first_row[i]]), int(low[i]), int(high[i])))
    return out


def build_tracks(dataset: str, solved: Solved, dense: dict[str, np.ndarray], frame_of_row: dict[str, np.ndarray],
                 aligned: np.ndarray, published: np.ndarray, provenance: str, calibration: dict | None
                 ) -> tuple[dict, dict]:
    """The three_d_tracks payload and its counts: one track per identity, positions at published sync indices."""
    members: dict[int, list[dict]] = {}
    for camera in solved.cameras:
        for identity, pieces in _members(solved.identity[camera], dense[camera], frame_of_row[camera]).items():
            for tracklet, low, high in pieces:
                members.setdefault(identity, []).append({"camera": camera, "tracklet_id": tracklet, "frame_range": [low, high]})
    data = solved.tracks
    t, identity = data["t"].astype(np.int64), data["identity"].astype(np.int64)
    root, xyz = data["root"], data["xyz"]
    candidates = np.flatnonzero(published[t])
    measured_root = np.isfinite(root[candidates]).all(axis=1)
    joints = np.isfinite(xyz[candidates]).all(axis=2).sum(axis=1)
    by_identity: dict[int, dict[int, int]] = {}
    duplicates = 0
    for position, row in enumerate(candidates):
        if not measured_root[position]:
            continue
        slot = by_identity.setdefault(int(identity[row]), {})
        key = int(t[row])
        if key in slot:  # two components of one identity at the same sync index: keep the better measured
            duplicates += 1
            if joints[position] <= int(np.isfinite(xyz[slot[key]]).all(axis=1).sum()):
                continue
        slot[key] = int(row)
    omitted_with_joints = int((~measured_root & (joints > 0)).sum())
    tracks, dropped, position_count = [], [], 0
    for track_id in sorted(set(members) | set(by_identity)):
        pieces = sorted(members.get(track_id, []), key=lambda m: (m["frame_range"][0], m["camera"], m["tracklet_id"]))
        if not pieces:
            dropped.append(track_id)
            continue
        positions = []
        for key, row in sorted(by_identity.get(track_id, {}).items(), key=lambda kv: int(aligned[kv[0]])):
            position = {"aligned_index": int(aligned[key]), "xyz": [round(float(v), 2) for v in root[row]]}
            joints_xyz = [None if not np.isfinite(p).all() else [round(float(v), 2) for v in p] for p in xyz[row]]
            if any(j is not None for j in joints_xyz):
                position["keypoints"] = joints_xyz
            positions.append(position)
        position_count += len(positions)
        track = {"track_id": int(track_id), "class": "person", "members": pieces, "provenance": provenance}
        if positions:
            track["positions"] = positions
        tracks.append(track)
    payload: dict[str, Any] = {"schema_version": 1, "dataset": dataset, "tracks": tracks, "units": "in",
                               "keypoint_format": "coco17"}
    if calibration is not None:
        payload["calibration"] = calibration
    stats = {"tracks": len(tracks), "positions": position_count, "members": sum(len(t["members"]) for t in tracks),
             "tracks_without_members_dropped": dropped, "duplicate_identity_t_rows": duplicates,
             "rows_with_joints_but_no_root_omitted": omitted_with_joints,
             "rows_without_any_measurement": int((~measured_root & (joints == 0)).sum())}
    return payload, stats


def calibration_descriptor(solved: Solved, workspace_id: str) -> dict | None:
    path = solved.calibration_path
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    note = data.get("data_engine") or {}
    source = str(note.get("source") or "")
    name = Path(source.replace("\\", "/")).stem if source else "unknown"
    return {
        "path": "calibration/calibration.json", "workspace": workspace_id, "sha256": sha256_file(path),
        "standard_calibration": name, "source_path": source or None, "source_sha256": note.get("source_sha256"),
        "world": {"units": "in", "up_axis": "z", "floor_z": 0.0,
                  "frame": f"the standard {name} world frame (inches, z up, floor at z = 0)"},
        "convention": data.get("convention"), "pixel_space": "native camera pixels",
        "cameras": {camera: {"status": info.get("status"), "applied": info.get("applied"),
                             "median_shift_px": info.get("median_shift_px")}
                    for camera, info in (note.get("cameras") or {}).items()},
    }


# ---------------------------------------------------------------------------
# Provenance and notes
# ---------------------------------------------------------------------------


def read_engine_db(root: Path) -> dict:
    """Constraints and open unknowns, opened read-only (never creates or migrates the database)."""
    path = root / "engine.sqlite3"
    out: dict[str, Any] = {"constraints": [], "unknowns_open": {}, "meta": {}}
    if not path.is_file():
        return out
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=30)
    try:
        for query, key in (("SELECT kind, source, active, COUNT(*) FROM constraints GROUP BY 1, 2, 3", "constraints"),
                           ("SELECT kind, COUNT(*) FROM unknowns WHERE status = 'open' GROUP BY 1", "unknowns_open"),
                           ("SELECT key, value FROM meta", "meta")):
            try:
                rows = connection.execute(query).fetchall()
            except sqlite3.Error:
                continue
            out[key] = ([list(r) for r in rows] if key == "constraints"
                        else {r[0]: (r[1] if key == "unknowns_open" else _loads(r[1])) for r in rows})
    finally:
        connection.close()
    return out


def _loads(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def constraint_counts(db: dict) -> dict:
    human: dict[str, int] = {}
    machine: dict[str, int] = {}
    inactive = 0
    for kind, source, active, count in db["constraints"]:
        if not active:
            inactive += int(count)
        elif str(source).startswith("human"):
            human[kind] = human.get(kind, 0) + int(count)
        else:
            machine[kind] = machine.get(kind, 0) + int(count)
    return {"human_active": dict(sorted(human.items())), "human_active_total": sum(human.values()),
            "machine_active": dict(sorted(machine.items())), "superseded_inactive": inactive}


def latest_manifest(root: Path, stage: str, round_number: int | None = None) -> dict | None:
    pattern = f"{stage}_r{round_number:03d}_*.json" if round_number is not None else f"{stage}_r*.json"
    found = sorted((root / "manifests").glob(pattern), key=lambda p: (p.name.split("_")[1], p.stat().st_mtime))
    return json.loads(found[-1].read_text(encoding="utf-8")) | {"_file": found[-1].name} if found else None


def finish_provenance(name_prefix: str, body: dict) -> dict:
    """Name = prefix + digest of the whole content (provenances are immutable: any change must change the name)."""
    return {"name": f"{name_prefix}.{short_digest(body)}", **body}


def detect_provenance(root: Path, code: dict) -> dict:
    manifest = latest_manifest(root, "detect")
    config = (manifest or {}).get("config", {}).get("detect") or RECIPE["detect"]
    inputs = (manifest or {}).get("inputs", {})
    models = {key: inputs[key] for key in sorted(inputs) if key.endswith("_sha256") and not key.startswith("dataset")}
    paths = {key: inputs[key] for key in ("yolo_onnx", "vitpose_onnx") if key in inputs}
    body = {
        "kind": "machine",
        "description": ("Per-frame 2D person detections and COCO17 keypoints of the Data Engine detect stage: YOLOv8x-pose-p6 "
                        "(curated v6, 1280) boxes with ViTPose (curated v6) keypoints, run with TensorRT on every native "
                        "frame. Keypoint confidence is the raw ViTPose heatmap peak. Positions are measured, not corrected."),
        "reproduction": {
            "steps": ["data-engine run <workspace>   (detect stage: python src/data_engine/workers/detect_trt.py "
                      "--dataset-json <dataset.json> --out <detections/raw> --yolo <yolo onnx> --vitpose <vitpose onnx>)"],
            "code": code,
            "config": {"detect": config, "models_sha256": models, "model_paths": paths,
                       "detect_manifest": (manifest or {}).get("_file"),
                       "code_at_detection": (manifest or {}).get("code")},
        },
    }
    return finish_provenance("data_engine.detect", body)


def solve_provenance(root: Path, solved: Solved, workspace_id: str, db: dict, code: dict, calibration: dict | None) -> dict:
    sections: dict[str, Any] = {}
    manifests = {}
    for stage in ("tracklets", "solve"):
        manifest = latest_manifest(root, stage, solved.round) if stage == "solve" else latest_manifest(root, stage)
        if manifest:
            manifests[stage] = manifest["_file"]
            sections.update(manifest.get("config") or {})
    for section in ("postfilter", "tracklets", "geometry", "association", "temporal"):
        sections.setdefault(section, RECIPE[section])
    recipe = {section: sections[section] for section in ("postfilter", "tracklets", "geometry", "association", "temporal")}
    counts = constraint_counts(db)
    body = {
        "kind": "machine",
        "description": (f"Data Engine solve (round {solved.round}): post-filtered detections, over-split single-camera tracklets "
                        "(BoT-SORT), cross-camera association, identity across gaps and multi-view triangulated 3D poses. "
                        f"Machine output; {counts['human_active_total']} human answer(s) applied as constraints. Cross-view membership "
                        "and identities are provisional unless a person confirmed them."),
        "reproduction": {
            "steps": [f"data-engine run {workspace_id}", f"data-engine publish {workspace_id}"],
            "code": code,
            "config": {"round": solved.round, "recipe": recipe,
                       "recipe_equals_current_config": recipe == {k: RECIPE[k] for k in recipe},
                       "constraints": counts, "manifests": manifests, "artifact_sha256": solved.artifact_sha256,
                       "calibration": None if calibration is None else {k: calibration[k] for k in
                                                                         ("sha256", "standard_calibration", "cameras")},
                       "alignment_provenance": (solved.alignment or {}).get("provenance")},
        },
    }
    return finish_provenance("data_engine.solve", body)


def alignment_provenance_record(solved: Solved, code: dict) -> dict | None:
    """The engine's alignment provenance as the label store can register it.

    The align worker records ``code`` as a mapping of repositories (data_engine and data_management); the store
    needs a flat ``code`` with the full ``git_commit`` of the producing code. The flat identity of this repository
    is used and the original mapping is kept as ``code_components``.
    """
    record = solved.alignment_provenance
    if record is None:
        return None
    reproduction = copy.deepcopy(record.get("reproduction") or {})
    original = reproduction.get("code")
    if not (isinstance(original, dict) and isinstance(original.get("git_commit"), str)):
        reproduction["code_components"] = original
        reproduction["code"] = code
    return {"name": record["name"], "kind": record["kind"], "description": record["description"],
            "reproduction": reproduction}


def build_notes(*, stats: dict, solved: Solved, workspace_id: str, slice_note: str | None, stride: int, db: dict,
                calibration: dict | None, cameras_unaligned: list[str], what: str) -> str:
    counts = constraint_counts(db)
    open_unknowns = db["unknowns_open"]
    human = ", ".join(f"{k} {v}" for k, v in counts["human_active"].items())
    unresolved = f" Cameras without an alignment (no aligned frames at stride > 1): {', '.join(cameras_unaligned)}." \
        if cameras_unaligned else ""
    shifted = sorted(c for c, info in ((calibration or {}).get("cameras") or {}).items()
                     if info.get("status") not in (None, "unchanged"))
    mount = f" Calibration mount check not 'unchanged' for: {', '.join(shifted)} (standard calibration applied)." if shifted else ""
    sample = ("every sync index" if stride == 1 else f"every {stride}th sync index ({10 / stride:g} Hz)")
    return (
        f"{what} Machine prelabels from the Data Engine, round {solved.round} of workspace {workspace_id}"
        f"{f' ({slice_note})' if slice_note else ''}; not human-reviewed except as stated. "
        f"Human answers applied: {counts['human_active_total']}" + (f" ({human})" if human else "") + ". "
        f"Published detections: {stats['published_detections']}, of which {stats['detections_in_3d_tracks']} "
        f"({stats['share_in_3d_tracks']:.0%}) belong to multi-camera 3D tracks (identity assigned); the others are "
        "single-view or unassigned. Cross-view membership and identities are provisional unless human-confirmed "
        f"({stats['human_sourced_rows']} published detections carry a human-sourced label). "
        "3D keypoints are measured joints only: unmeasured joints are null and nothing is interpolated or extrapolated; "
        "positions whose root is unmeasured are omitted. "
        f"Sampling: {sample}; time as aligned indices of the published frame_alignment set.{unresolved}{mount} "
        "Open unknowns at publication (engine ledger): "
        + (", ".join(f"{k} {v}" for k, v in sorted(open_unknowns.items())) + f" (total {sum(open_unknowns.values())})"
           if open_unknowns else "none") + ".")


# ---------------------------------------------------------------------------
# Drafts
# ---------------------------------------------------------------------------


@dataclass
class Drafts:
    out: Path
    dataset: str
    tag: str
    names: dict[str, str]
    provenances: dict[str, dict]  # name -> record (name, kind, description, reproduction)
    dependencies: dict[str, list[str]]
    notes: dict[str, str]
    summary: dict


def estimate_sizes(measured: dict, *, sync_indices: int = 36000, cameras: int = 7, kept_per_camera_frame: float = 7.55,
                   stride: int = 5) -> dict:
    """Extrapolate output sizes from measured bytes per item (pilot) to a long recording."""
    detections = sync_indices / stride * cameras * kept_per_camera_frame
    positions = measured["positions_per_sync_index"] * sync_indices / stride
    two_d = detections * measured["bytes_per_detection"]
    three_d = positions * measured["bytes_per_position"]
    return {"stride": stride, "detections": int(detections), "positions": int(positions),
            "two_d_tracklets_mb": round(two_d / 1e6, 1), "three_d_tracks_mb": round(three_d / 1e6, 1),
            "total_mb": round((two_d + three_d) / 1e6, 1)}


def build_drafts(workspace: Path, dataset: str, tag: str, out: Path, *, stride: int = 5, alignment_mode: str = "auto"
                 ) -> Drafts:
    root = Path(workspace)
    workspace_id = root.name
    started = time.time()
    solved = load_solved(root)
    ds = Dataset.load(root / "dataset.json")
    cameras = solved.cameras
    db = read_engine_db(root)
    code = code_identity()
    log(f"round {solved.round}: {len(cameras)} cameras, {len(solved.aligned_time)} sync indices, stride {stride}")

    aligned, shift = aligned_indices(solved.aligned_time, solved.native, cameras, solved.alignment)
    published = (aligned % stride == 0) if stride > 1 else np.ones(len(aligned), bool)
    names = {kind: f"{dataset}.{kind}.{tag}" for kind in ("frame_alignment", "two_d_tracklets", "three_d_tracks")}

    calibration = calibration_descriptor(solved, workspace_id)
    detect = detect_provenance(root, code)
    solve = solve_provenance(root, solved, workspace_id, db, code, calibration)
    align_prov = alignment_provenance_record(solved, code)
    if align_prov is None:
        raise PublishError("alignment/provenance.json is missing; cannot reference the alignment provenance")
    provenances = {p["name"]: p for p in (align_prov, detect, solve)}

    alignment_payload, agreement = build_alignment(
        solved.alignment, dataset, aligned, solved.native, published, cameras, alignment_mode, shift,
        {"workspace": workspace_id, "round": solved.round, "stride": stride})
    positions_check = positions_agreement(AlignmentLookup(alignment_payload, cameras), aligned, solved.native,
                                          solved.tracks["t"].astype(np.int64), solved.tracks["used"],
                                          solved.tracks["rows"], published)
    agreement["positions"] = positions_check
    sizes: dict[str, int] = {}
    sizes["frame_alignment/alignment.json"] = write_json(out / "frame_alignment" / "alignment.json", alignment_payload)

    dense: dict[str, np.ndarray] = {}
    frame_of_row: dict[str, np.ndarray] = {}
    per_camera: dict[str, dict] = {}
    unaligned = [c for c, column in zip(cameras, range(len(cameras))) if not (solved.native[:, column] >= 0).any()]
    total_published = in_3d = human_rows = kept_total = off_timeline = 0
    for column, camera in enumerate(cameras):
        detections = load_camera(root / "detections" / "raw", camera, ds.cameras[camera].frame_count)
        frame_of_row[camera] = detections.frame_of_row.astype(np.int64)
        mask = published_frame_mask(detections.frame_count, solved.native, column, published, stride)
        info = ds.cameras[camera]
        payload, dense[camera], stats = build_camera_tracklets(
            camera, dataset, detections, solved.kept[camera], solved.atom[camera], solved.identity[camera], mask,
            (info.width, info.height), detect["name"], solve["name"])
        per_camera[camera] = stats
        kept_total += int(solved.kept[camera].sum())
        on_timeline = np.zeros(detections.frame_count, bool)
        shown = solved.native[:, column]
        on_timeline[shown[(shown >= 0) & (shown < detections.frame_count)]] = True
        off_timeline += int((solved.kept[camera] & ~on_timeline[frame_of_row[camera]]).sum())
        published_rows = dense[camera] >= 0
        total_published += int(published_rows.sum())
        in_3d += int((published_rows & (solved.identity[camera] >= 0)).sum())
        human_rows += int((published_rows & (solved.source[camera] == 5)).sum())
        if payload is not None:
            sizes[f"two_d_tracklets/cameras/{camera}.json"] = write_json(
                out / "two_d_tracklets" / "cameras" / f"{camera}.json", payload)
        del payload
        log(f"{camera}: {stats['published_detections']} detections, {stats.get('tracklets', 0)} tracklets")

    tracks_payload, track_stats = build_tracks(dataset, solved, dense, frame_of_row, aligned, published, solve["name"], calibration)
    sizes["three_d_tracks/tracks.json"] = write_json(out / "three_d_tracks" / "tracks.json", tracks_payload)
    stats = {"published_detections": total_published, "detections_in_3d_tracks": in_3d,
             "share_in_3d_tracks": in_3d / total_published if total_published else 0.0, "human_sourced_rows": human_rows}
    slice_name = split_workspace_id(workspace_id)[1]
    limit = db["meta"].get("limit_frames")
    slice_note = f"pilot slice '{slice_name}': first {limit} native frames of every camera" if limit else \
        (f"slice '{slice_name}'" if slice_name else None)
    notes = {}
    labels = {"frame_alignment": "Time alignment of the cameras on the engine's common 10 Hz timeline, re-expressed as aligned indices.",
              "two_d_tracklets": "Per-camera person detections (boxes, COCO17 keypoints) with camera-local tracklet ids "
                                 "(tracklet pieces of the solve).",
              "three_d_tracks": "One 3D track per solved identity: cross-camera tracklet membership plus world positions "
                                "(inches, z up) and 17-joint skeletons."}
    for kind, text in labels.items():
        notes[kind] = build_notes(stats=stats, solved=solved, workspace_id=workspace_id, slice_note=slice_note, stride=stride,
                                  db=db, calibration=calibration, cameras_unaligned=unaligned, what=text)
    dependencies = {"frame_alignment": [], "two_d_tracklets": [names["frame_alignment"]],
                    "three_d_tracks": [names["two_d_tracklets"], names["frame_alignment"]]}
    measured = {"bytes_per_detection": sum(v for k, v in sizes.items() if k.startswith("two_d")) / max(1, total_published),
                "bytes_per_position": sizes["three_d_tracks/tracks.json"] / max(1, track_stats["positions"]),
                "positions_per_sync_index": track_stats["positions"] / max(1, int(published.sum()))}
    drafts_dir = out
    for name, record in provenances.items():
        write_json(drafts_dir / "provenance" / f"{name}.json", record)
    summary = {
        "dataset": dataset, "tag": tag, "workspace": str(root), "workspace_id": workspace_id, "round": solved.round,
        "stride": stride, "shift": shift, "sets": names, "draft_dir": str(out),
        "sync_indices": int(len(aligned)), "published_sync_indices": int(published.sum()),
        "detections": {"kept_total": kept_total, "published": total_published, "in_3d_tracks": in_3d,
                       "share_in_3d_tracks": round(stats["share_in_3d_tracks"], 4),
                       "kept_off_common_timeline": off_timeline, "human_sourced": human_rows,
                       "per_camera": per_camera},
        "tracks": track_stats, "alignment": agreement, "unaligned_cameras": unaligned,
        "constraints": constraint_counts(db), "open_unknowns": db["unknowns_open"],
        "provenance": {"names": list(provenances), "code": code, "commit_problem": commit_problem(code)},
        "bytes": sizes, "bytes_total": sum(sizes.values()), "measured_per_item": measured,
        "estimate_60min": {f"stride_{s}": estimate_sizes(measured, stride=s, cameras=len(cameras)) for s in (5, 1)},
        "notes": notes, "seconds_to_build": round(time.time() - started, 1), "produced_at": utc_now(),
    }
    write_json(out / "summary.json", summary)
    return Drafts(out, dataset, tag, names, provenances, dependencies, notes, summary)


# ---------------------------------------------------------------------------
# Store interaction (data_management is imported lazily here only)
# ---------------------------------------------------------------------------

SET_DIRS = {"frame_alignment": "frame_alignment", "two_d_tracklets": "two_d_tracklets", "three_d_tracks": "three_d_tracks"}


def _placeholder(record: dict, label: str) -> dict:
    """A registrable copy of a machine provenance with a placeholder commit, marked as not real."""
    copy_ = copy.deepcopy(record)
    if copy_["kind"] == "machine":
        reproduction = copy_["reproduction"]
        reproduction["code_unverified"] = reproduction.get("code")
        reproduction["code"] = {"repository": "data_engine", "git_commit": "0" * 40, "git_dirty": True,
                                "diff_sha256": "0" * 64}
        copy_["description"] = f"[{label}] " + copy_["description"]
    return copy_


def publish_into(root: Path, drafts: Drafts, *, replace: bool, placeholder: str | None) -> list[dict]:
    """Register provenances and publish the three sets into the store at ``root`` (in dependency order)."""
    from data_management.label_sets import LabelSetDraft, LabelStore, Provenance

    store = LabelStore(root)
    store.init()
    for record in drafts.provenances.values():
        record = _placeholder(record, placeholder) if placeholder else record
        store.register_provenance(Provenance(name=record["name"], kind=record["kind"],
                                             description=record["description"], reproduction=record["reproduction"]))
    results = []
    for kind in ("frame_alignment", "two_d_tracklets", "three_d_tracks"):
        label_set = store.publish(LabelSetDraft(
            name=drafts.names[kind], label_type=kind, dataset=drafts.dataset, production_notes=drafts.notes[kind],
            producer=PRODUCER, dependencies=list(drafts.dependencies[kind]),
            source_dir=drafts.out / SET_DIRS[kind]), replace=replace)
        results.append({"name": label_set.name, "label_type": kind, "files": len(label_set.data_files),
                        "provenances": label_set.provenances, "dependencies": label_set.dependencies})
    return results


def validate_root(root: Path) -> dict:
    from data_management.label_sets import LabelStore, validate_store

    report = validate_store(LabelStore(root))
    return {"ok": report.ok, "checked_sets": report.checked_sets, "errors": [str(p) for p in report.errors][:50],
            "error_count": len(report.errors), "warnings": [str(p) for p in report.warnings][:50],
            "warning_count": len(report.warnings)}


def check_drafts(drafts: Drafts) -> dict:
    """Validate the drafts as the store would (schema, semantics, provenance, dependencies, member existence).

    Done in a throw-away scratch store with placeholder machine provenances (the scratch store never leaves the
    temp directory), so it works before the repository has a commit.
    """
    scratch = Path(tempfile.mkdtemp(prefix="publish_check_"))
    try:
        try:
            published = publish_into(scratch, drafts, replace=False, placeholder="SCRATCH CHECK ONLY - placeholder commit")
        except Exception as exc:  # LabelStoreError carries the validator's messages
            return {"ok": False, "error_count": 1, "errors": [str(exc)[:6000]], "sets": []}
        report = validate_root(scratch)
        report["sets"] = [s["name"] for s in published]
        return report
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def verify_with_store_rule(drafts: Drafts, solved_root: Path) -> dict:
    """Re-measure the aligned-index agreement with data_management's own ``native_frames`` on the written payload."""
    from data_management.label_sets import frame_alignment as fa

    payload = json.loads((drafts.out / "frame_alignment" / "alignment.json").read_text(encoding="utf-8"))
    solved = load_solved(solved_root)
    aligned, _ = aligned_indices(solved.aligned_time, solved.native, solved.cameras, solved.alignment)
    stride = drafts.summary["stride"]
    published = (aligned % stride == 0) if stride > 1 else np.ones(len(aligned), bool)
    sample = np.flatnonzero(published)
    step = max(1, len(sample) // 400)
    compared = exact = within = worst = 0
    for t in sample[::step]:
        frames = fa.native_frames(payload, int(aligned[t]))
        for column, camera in enumerate(solved.cameras):
            want = int(solved.native[t, column])
            if want < 0 or camera not in frames:
                continue
            d = abs(frames[camera] - want)
            compared += 1
            exact += d == 0
            within += d <= 1
            worst = max(worst, d)
    return {"sampled_sync_indices": int(len(sample[::step])), "compared": compared, "exact": int(exact),
            "within_1_frame": int(within), "max_abs_diff": int(worst)}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def real_store_root() -> Path:
    from data_management.label_sets.store import DEFAULT_ROOT

    return Path(DEFAULT_ROOT).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workspace", required=True, type=Path, help="engine workspace directory")
    parser.add_argument("--dataset", required=True, help="label dataset id (a '@slice' suffix is dropped and names the tag)")
    parser.add_argument("--tag", help="set tag (default data_engine[_<slice>]_r<round>)")
    parser.add_argument("--out", required=True, type=Path, help="draft directory")
    parser.add_argument("--stride", type=int, default=1, help="publish every Nth sync index (default 1 = all, 10 Hz; 5 = 2 Hz)")
    parser.add_argument("--alignment", choices=("auto", "engine", "dense"), default="auto",
                        help="frame_alignment checkpoints: engine's own, re-derived from the sync table, or auto (dense "
                             "only when the engine's do not reproduce the sync table exactly)")
    parser.add_argument("--store-root", type=Path, help="label store root (never defaults to the real store for publishing)")
    parser.add_argument("--publish", action="store_true", help="register provenance and publish into the store")
    parser.add_argument("--replace", action="store_true", help="replace existing sets of the same names")
    parser.add_argument("--allow-real-store", action="store_true",
                        help="permit publishing into the real label store (set by the publish stage on operator request)")
    parser.add_argument("--allow-uncommitted-test", action="store_true",
                        help="TEST ONLY: register placeholder-commit provenances; needs --store-root away from the real store")
    parser.add_argument("--skip-check", action="store_true", help="do not validate in a scratch store (not with --publish)")
    return parser.parse_args(argv)


def write_result(out: Path, **values: Any) -> None:
    """``result.json`` in the draft directory: how this run ended (read by the publish stage)."""
    try:
        write_json(out / "result.json", {"finished_at": utc_now(), **values})
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stride < 1:
        print("--stride must be >= 1", file=sys.stderr)
        return 2
    dataset, flag_slice = split_workspace_id(args.dataset)
    slice_name = flag_slice or split_workspace_id(args.workspace.name)[1]
    summary: dict[str, Any] = {"ok": False}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").unlink(missing_ok=True)
    try:
        round_number, _ = find_round(args.workspace)
        tag = args.tag or default_tag(round_number, slice_name)
        if args.publish:
            if args.store_root is None and not args.allow_real_store:
                raise Refused("publishing without --store-root targets the real label store; pass --allow-real-store "
                              "(the publish stage does so only when the operator enabled publish_to_store)")
            target = Path(args.store_root) if args.store_root else None
            is_real = target is None or target.resolve() == real_store_root()
            if is_real and not args.allow_real_store:
                raise Refused(f"{target} is the real label store; pass --allow-real-store to publish there")
            if args.allow_uncommitted_test and is_real:
                raise Refused("--allow-uncommitted-test is only for a store root away from the real label store")
            if args.skip_check:
                raise Refused("--skip-check cannot be combined with --publish")
        drafts = build_drafts(args.workspace, dataset, tag, args.out, stride=args.stride, alignment_mode=args.alignment)
        summary = drafts.summary
        summary["alignment"]["store_rule_check"] = None
        try:
            summary["alignment"]["store_rule_check"] = verify_with_store_rule(drafts, args.workspace)
        except ImportError:
            pass
        summary["check"] = None if args.skip_check else check_drafts(drafts)
        summary["ok"] = bool(summary["check"] is None or summary["check"]["ok"])
        summary["published"] = []
        if args.publish:
            if not summary["ok"]:
                raise Refused("drafts did not validate; nothing published")
            problem = summary["provenance"]["commit_problem"]
            if problem and not args.allow_uncommitted_test:
                raise Refused(problem)
            root = Path(args.store_root) if args.store_root else real_store_root()
            if not args.replace:
                existing = [n for n in drafts.names.values() if (root / "sets" / f"{n}.json").exists()]
                if existing:
                    raise Refused(f"label set(s) already exist in {root}: {existing}; use --replace or another --tag")
            placeholder = "TEST ONLY - placeholder commit, not a real code identity" if problem else None
            summary["published"] = publish_into(root, drafts, replace=args.replace, placeholder=placeholder)
            summary["store"] = {"root": str(root), "placeholder_provenance": bool(placeholder), **validate_root(root)}
            summary["ok"] = bool(summary["store"]["ok"])
        write_json(args.out / "summary.json", summary)
        write_result(args.out, ok=summary["ok"], published=bool(summary["published"]), exit_code=0 if summary["ok"] else 1,
                     validation_errors=(summary.get("check") or {}).get("error_count", 0))
        print(json.dumps(summary, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
        return 0 if summary["ok"] else 1
    except Refused as exc:
        write_result(args.out, ok=False, published=False, exit_code=2, refused=str(exc),
                     validation_errors=(summary.get("check") or {}).get("error_count", 0))
        print(json.dumps({"ok": False, "refused": str(exc), **{k: v for k, v in summary.items()
                                                               if k in ("sets", "draft_dir", "check", "provenance")}}))
        log(f"REFUSED: {exc}")
        return 2
    except PublishError as exc:
        write_result(args.out, ok=False, published=False, exit_code=3, error=str(exc))
        print(json.dumps({"ok": False, "error": str(exc)}))
        log(f"ERROR: {exc}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
