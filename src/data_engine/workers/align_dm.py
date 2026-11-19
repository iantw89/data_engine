"""Camera time alignment for a dataset, by driving data_management's recorded alignment generator.

Runs in data_management's interpreter. No alignment logic lives here (repository rule): this
script builds the evidence (the engine's dense 2D pose detections, plus optional audio, placed
on each camera's recorded PTS clock exactly as ``label_sets.alignment_labels.cached_evidence``
does), calls ``alignment_generator.run_offline`` over the whole recording, and converts the
output (``frame_alignment.from_alignment_run``, sync table, report).

Coordinates: detections are native pixels (e.g. 2048x1536) and ``PoseFrame.resolution`` says so.
``motion_sync`` letterboxes keypoints from ``src_resolution`` into the calibration's
``calib_resolution`` (1280x1280), which is geometrically identical for 2048x1536 and for the
1280x960 resize the cached sync poses used (same 4:3 scale + vertical padding), so no conversion
is needed. Keypoint confidences are the raw ViTPose heatmap peaks, which is what the live path
feeds the same code (the ``>= .15`` weights in motion_sync assume them); they are only clipped
to [0, 1] because ``PersonDetections`` validation requires that.

Clock caveat: ``dataset.json`` carries no chunk start times, so a camera's parts are joined
consecutively on the PTS clock (each part honouring its trim and count). A recording dropout
exactly at a file boundary is therefore invisible, and a part without overlap evidence
(``verified`` false) is joined with an unknown gap; both are reported as uncertainty.

Outputs in ``--out``: alignment.json, provenance.json, clocks.npz, sync_table.npz, report.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_engine.dataset import Camera, Dataset  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
SYNC_RATE_HZ = 10.0
MIN_FIT_WINDOWS = 6  # drift/spread need at least this many candidate windows ...
MIN_FIT_SECONDS = 60.0  # ... spanning at least this long; shorter runs report None, not a noisy slope


# ---------------------------------------------------------------------------
# Pure helpers (numpy only; unit-tested without data_management)
# ---------------------------------------------------------------------------


def offsets_at(times: np.ndarray, schedule: list[tuple[float, dict[str, float]]],
               cameras: list[str]) -> np.ndarray:
    """Piecewise-constant offsets (len(times), n_cameras); nan where a camera is unresolved.

    ``schedule`` is ``[(effective_time, {camera: offset_seconds})]``. The latest entry with
    ``effective_time <= t`` applies; before the first entry the first entry applies.
    """
    out = np.full((len(times), len(cameras)), np.nan)
    if not schedule:
        return out
    order = sorted(range(len(schedule)), key=lambda i: (schedule[i][0], i))
    effective = np.array([schedule[i][0] for i in order], dtype=float)
    table = np.array([[schedule[i][1].get(c, np.nan) for c in cameras] for i in order], dtype=float)
    index = np.clip(np.searchsorted(effective, times, side="right") - 1, 0, len(order) - 1)
    return table[index]


def frame_period(times: np.ndarray) -> float:
    return float(np.median(np.diff(times))) if len(times) > 1 else float("nan")


def nearest_frames(times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Frame whose clock time is nearest each target, -1 if farther than half a period or nan."""
    period = frame_period(times)
    out = np.full(len(targets), -1, dtype=np.int32)
    ok = np.isfinite(targets)
    if not ok.any():
        return out
    t = targets[ok]
    right = np.clip(np.searchsorted(times, t), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    pick = np.where(np.abs(times[left] - t) <= np.abs(times[right] - t), left, right)
    good = np.abs(times[pick] - t) <= period / 2 + 1e-9
    result = np.where(good, pick, -1)
    out[ok] = result
    return out


def build_sync_table(clock_times: dict[str, np.ndarray], schedule: list[tuple[float, dict[str, float]]],
                     *, anchor: str | None = None, rate_hz: float = SYNC_RATE_HZ) -> dict[str, np.ndarray]:
    """Common-time sync table.

    Common time = camera time - camera offset (offsets relative to the anchor, which is listed
    first). For every sample time ``t`` the native frame of each camera is the one nearest to
    camera time ``t + offset`` (within half a frame period, else -1; also -1 where the offset is
    unresolved). The grid covers the span where at least two cameras have frames.
    """
    cameras = list(clock_times)
    if anchor is not None:
        cameras = [anchor] + [c for c in cameras if c != anchor]
    step = 1.0 / rate_hz
    first = schedule[0][1] if schedule else {}
    lows = [clock_times[c][0] - first.get(c, 0.0) for c in cameras]
    highs = [clock_times[c][-1] - first.get(c, 0.0) for c in cameras]
    # Offsets may change over time; pad the broad grid by the largest offset excursion.
    pad = max([abs(v) for _, e in schedule for v in e.values() if np.isfinite(v)] + [0.0]) * 2 + 1.0
    lo, hi = min(lows) - pad, max(highs) + pad
    grid = np.arange(np.ceil(lo * rate_hz), np.floor(hi * rate_hz) + 1) * step
    offsets = offsets_at(grid, schedule, cameras)
    native = np.stack([nearest_frames(clock_times[c], grid + offsets[:, i]) for i, c in enumerate(cameras)], axis=1)
    populated = np.flatnonzero((native >= 0).sum(axis=1) >= 2)
    if not len(populated):
        return {"cameras": np.array(cameras), "aligned_time": np.zeros(0), "native": np.zeros((0, len(cameras)), np.int32),
                "offsets_seconds": np.zeros((0, len(cameras)))}
    a, b = populated[0], populated[-1] + 1
    return {"cameras": np.array(cameras), "aligned_time": grid[a:b], "native": native[a:b].astype(np.int32),
            "offsets_seconds": offsets[a:b]}


def robust_line(t: np.ndarray, y: np.ndarray, max_points: int = 800) -> tuple[float, float] | None:
    """Theil-Sen line (slope, intercept) of y over t; None with fewer than 3 distinct points."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    if len(t) < 3 or np.ptp(t) <= 0:
        return None
    if len(t) > max_points:
        pick = np.linspace(0, len(t) - 1, max_points).astype(int)
        t, y = t[pick], y[pick]
    i, j = np.triu_indices(len(t), 1)
    dt = t[j] - t[i]
    ok = dt > 0
    slope = float(np.median((y[j] - y[i])[ok] / dt[ok]))
    return slope, float(np.median(y - slope * t))


def mask_spans(times: np.ndarray, mask: np.ndarray, end: float) -> list[list[float]]:
    """[start, end] runs of ``mask`` true at ``times`` (a run ends at the next false sample / ``end``)."""
    spans, begin = [], None
    for t, flag in zip(times, mask):
        if flag and begin is None:
            begin = float(t)
        elif not flag and begin is not None:
            spans.append([begin, float(t)])
            begin = None
    if begin is not None:
        spans.append([begin, float(end)])
    return spans


def coverage_spans(intervals: dict[str, list[tuple[float, float]]], minimum: int = 2) -> list[tuple[float, float]]:
    """Time intervals covered by at least ``minimum`` cameras."""
    events = sorted([(a, 1) for v in intervals.values() for a, _ in v] + [(b, -1) for v in intervals.values() for _, b in v],
                    key=lambda e: (e[0], -e[1]))
    spans, depth, begin = [], 0, None
    for time_, delta in events:
        before = depth
        depth += delta
        if before < minimum <= depth:
            begin = time_
        elif before >= minimum > depth and begin is not None:
            if time_ > begin:
                spans.append((begin, time_))
            begin = None
    return spans


def stale_unresolved_spans(included: np.ndarray, first_end: float, last_end: float,
                           stale_seconds: float) -> tuple[list[list[float]], list[list[float]]]:
    """(unresolved spans, stale spans) for one camera from the window ends of accepted steps that included it."""
    included = np.sort(np.asarray(included, float))
    if not len(included):
        return [[first_end, last_end]], []
    unresolved = [[first_end, float(included[0])]] if included[0] > first_end else []
    stale = []
    for a, b in zip(included[:-1], included[1:]):
        if b - a > stale_seconds:
            stale.append([float(a + stale_seconds), float(b)])
    if last_end - included[-1] > stale_seconds:
        stale.append([float(included[-1] + stale_seconds), float(last_end)])
    return unresolved, stale


# ---------------------------------------------------------------------------
# Detections and clocks
# ---------------------------------------------------------------------------


@dataclass
class CameraDetections:
    camera_id: str
    frames: np.ndarray  # sorted unique native frame indices that have a (possibly empty) detection record
    starts: np.ndarray  # row offset of each frame into boxes / keypoints
    counts: np.ndarray
    boxes: np.ndarray
    keypoints: np.ndarray
    clipped_confidences: int = 0
    dropped_people: int = 0

    def record(self, frame: int):
        i = int(np.searchsorted(self.frames, frame))
        if i >= len(self.frames) or self.frames[i] != frame:
            return None
        a = int(self.starts[i])
        return self.boxes[a:a + int(self.counts[i])], self.keypoints[a:a + int(self.counts[i])]

    def runs(self) -> list[tuple[int, int]]:
        """Contiguous (first, last) frame runs."""
        if not len(self.frames):
            return []
        cuts = np.flatnonzero(np.diff(self.frames) != 1) + 1
        firsts = np.concatenate(([0], cuts))
        lasts = np.concatenate((cuts - 1, [len(self.frames) - 1]))
        return [(int(self.frames[a]), int(self.frames[b])) for a, b in zip(firsts, lasts)]


def load_detections(directory: Path, camera_id: str) -> CameraDetections | None:
    shards = sorted((directory / camera_id).glob("[0-9]*.npz")) if (directory / camera_id).is_dir() else []
    shards = [s for s in shards if not s.name.endswith(".tmp.npz")]
    if not shards:
        return None
    frames, counts, boxes, keypoints = [], [], [], []
    for shard in shards:
        with np.load(shard) as z:
            f, c = z["frames"].astype(np.int64), z["counts"].astype(np.int64)
            b, k = z["boxes"].astype(np.float32), z["keypoints"].astype(np.float32)
        if int(c.sum()) != len(b) or len(b) != len(k) or len(f) != len(c):
            raise ValueError(f"{shard}: inconsistent shard arrays")
        frames.append(f)
        counts.append(c)
        boxes.append(b)
        keypoints.append(k)
    frames, counts = np.concatenate(frames), np.concatenate(counts)
    boxes, keypoints = np.concatenate(boxes), np.concatenate(keypoints)
    if len(frames) > 1 and np.any(np.diff(frames) <= 0):
        raise ValueError(f"{camera_id}: detection frames are not strictly increasing across shards")
    # Drop people with non-finite values; clip confidences into [0, 1] (PersonDetections requires it).
    person_ok = np.isfinite(boxes).all(1) & np.isfinite(keypoints).reshape(len(keypoints), -1).all(1)
    dropped = int((~person_ok).sum())
    if dropped:
        owner = np.repeat(np.arange(len(frames)), counts)
        counts = np.bincount(owner[person_ok], minlength=len(frames)).astype(np.int64)
        boxes, keypoints = boxes[person_ok], keypoints[person_ok]
    conf = keypoints[..., 2]
    clipped = int(((conf < 0) | (conf > 1)).sum())
    keypoints[..., 2] = np.clip(conf, 0.0, 1.0)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return CameraDetections(camera_id, frames, starts, counts, boxes, keypoints, clipped, dropped)


def index_dir_for(camera_set: dict[str, Camera], fallback: Path):
    """Reuse data_management's frame-index cache read-only when it is complete; otherwise use ``fallback``
    and redirect the library default there so nothing is ever written under data_management/data."""
    from data_management import frame_index

    paths = [p.path for cam in camera_set.values() for p in cam.parts]
    if all(frame_index._cache_path(p, frame_index.DEFAULT_INDEX_DIR).is_file() for p in paths):
        return None
    fallback.mkdir(parents=True, exist_ok=True)
    frame_index.DEFAULT_INDEX_DIR = fallback
    return fallback


def build_clock(camera: Camera, index_dir):
    """PTS clock of the camera's stream: each part's smoothed clock cut to [trim, trim+count), joined consecutively."""
    from dataclasses import replace

    from data_management.time_model import ClockModel, clock_for, concatenate

    pieces = []
    for part in camera.parts:
        model = clock_for(part.path, index_dir)
        stop = part.trim + part.count
        if stop > model.frame_count:
            raise ValueError(f"{camera.camera_id}: {part.path} has {model.frame_count} frames, need {stop}")
        times = model.times[part.trim:stop]
        gaps = tuple(replace(g, index=g.index - part.trim) for g in model.gaps if part.trim < g.index < stop)
        pieces.append(ClockModel(times - times[0], model.period, gaps))
    clock = pieces[0] if len(pieces) == 1 else concatenate(pieces)
    if clock.frame_count != camera.frame_count:
        raise ValueError(f"{camera.camera_id}: clock has {clock.frame_count} frames, dataset says {camera.frame_count}")
    return clock


# ---------------------------------------------------------------------------
# Evidence provider
# ---------------------------------------------------------------------------


class AudioSource:
    """5-second audio chunks on the camera clock, loaded on demand and cached.

    Mirrors ``RecordedCamera.audio``: chunks never cross a clock dropout or a part boundary. Each
    chunk is read one frame longer than its span so that clock jitter cannot leave a one-sample hole
    (``sample_audio`` requires gap-free coverage; later chunks overwrite the overlap).
    """

    def __init__(self, camera: Camera, clock, log, workers: int):
        self.camera, self.clock, self.log = camera, clock, log
        self.enabled = True
        self.reason = ""
        self.cache: dict[int, object] = {}
        self.failures = 0
        self.loaded = 0
        self.seconds = 0.0
        self.statuses: dict[str, int] = {}
        boundaries = sorted({g.index for g in clock.gaps} | set(np.cumsum([p.count for p in camera.parts]).tolist()))
        step = max(1, round(5 / clock.period))
        self.chunks: list[tuple[int, int, int]] = []  # (first frame, frame count, frames readable incl. overlap)
        index, bounds = 0, [b for b in boundaries if 0 < b < clock.frame_count] + [clock.frame_count]
        for bound in bounds:
            while index < bound:
                count = min(step, bound - index)
                self.chunks.append((index, count, min(count + 1, bound - index)))
                index += count
        self.firsts = np.array([c[0] for c in self.chunks])

    def _load(self, number: int):
        from data_management.media import AudioChunk
        from data_management.recorded_capture import load_window_audio

        first, count, read = self.chunks[number]
        path, local = self.camera.locate(first)
        started = time.time()
        try:
            samples, status = load_window_audio(path, local, read, min_samples=1)
        except (OSError, RuntimeError, ValueError) as exc:
            samples, status = None, f"error: {exc}"
        self.seconds += time.time() - started
        self.statuses[status.split(":")[0]] = self.statuses.get(status.split(":")[0], 0) + 1
        if samples is None or len(samples) < 16000:
            self.failures += 1
            if status.startswith("error") or (self.loaded == 0 and self.failures >= 3):
                self.enabled, self.reason = False, status
                self.log(f"{self.camera.camera_id}: audio disabled ({status}); continuing with motion only")
            return None
        self.loaded += 1
        return AudioChunk(float(self.clock.time_at(first)), samples.copy())

    def window(self, t0: float, t1: float, pool) -> list:
        if not self.enabled:
            return []
        a = max(0, self.clock.index_at(t0))
        b = min(self.clock.frame_count - 1, self.clock.index_at(t1))
        if b < a:
            return []
        first = max(0, int(np.searchsorted(self.firsts, a, side="right")) - 1)
        last = int(np.searchsorted(self.firsts, b, side="right")) - 1
        wanted = [n for n in range(first, last + 1) if n not in self.cache]
        for n, chunk in zip(wanted, pool.map(self._load, wanted)):
            self.cache[n] = chunk
            if not self.enabled:
                return []
        return [c for n in range(first, last + 1) if (c := self.cache.get(n)) is not None]

    def trim_cache(self, before: float) -> None:
        keep = int(np.searchsorted(self.firsts, self.clock.index_at(before), side="right")) - 1
        for n in [n for n in self.cache if n < keep]:
            del self.cache[n]


class EvidenceProvider:
    """``run_offline`` provider: just the media a window needs (plus the search margin), per camera."""

    def __init__(self, cameras: dict[str, Camera], clocks, detections: dict[str, CameraDetections],
                 calibration_ids: dict, margin: float, audio: dict[str, AudioSource] | None, workers: int):
        from concurrent.futures import ThreadPoolExecutor

        self.cameras, self.clocks, self.detections = cameras, clocks, detections
        self.calibration_ids, self.margin, self.audio = calibration_ids, margin, audio or {}
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.calls = 0
        self.call_starts: list[float] = []
        self.seconds = 0.0
        self.audio_seconds = 0.0

    def __call__(self, start: float, end: float):
        from data_management.alignment_generator import WindowEvidence
        from data_management.media import PoseFrame, Snapshot, VideoFrame

        started = time.time()
        self.calls += 1
        self.call_starts.append(started)
        snapshots = {}
        for camera_id, clock in self.clocks.items():
            det = self.detections[camera_id]
            cam = self.cameras[camera_id]
            a = max(0, clock.index_at(start - self.margin))
            b = min(clock.frame_count - 1, clock.index_at(end + self.margin))
            video, poses = [], []
            for frame in range(a, b + 1):
                record = det.record(frame)
                if record is None:
                    continue  # no detector record: a source-image gap, never an empty detection
                stamp = float(clock.time_at(frame))
                video.append(VideoFrame(stamp, b"", source_time=stamp))
                poses.append(PoseFrame(stamp, (cam.width, cam.height), record[0], record[1]))
            audio = []
            if camera_id in self.audio:
                t = time.time()
                audio = self.audio[camera_id].window(start - self.margin, end + self.margin, self.pool)
                self.audio[camera_id].trim_cache(start - 3 * self.margin)
                self.audio_seconds += time.time() - t
            last = video[-1].time if video else 0.0
            snapshots[camera_id] = Snapshot(1, tuple(video), tuple(audio), last, "", tuple(poses))
        self.seconds += time.time() - started
        return WindowEvidence(snapshots, dict(self.calibration_ids))


# ---------------------------------------------------------------------------
# Run + conversion
# ---------------------------------------------------------------------------


def git_identity(path: Path) -> dict:
    import hashlib

    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=30, check=True).stdout

    try:
        commit = git("rev-parse", "--verify", "HEAD").strip()
    except (OSError, subprocess.SubprocessError):
        return {"repository": "data_engine", "git_commit": None, "note": "uncommitted"}
    diff = git("diff", "HEAD", "--binary")
    out = {"repository": "data_engine", "git_commit": commit, "git_dirty": bool(diff)}
    if diff:
        out["diff_sha256"] = hashlib.sha256(diff.encode("utf-8", "surrogateescape")).hexdigest()
    return out


def dm_identity() -> dict:
    try:
        from data_management.label_sets.provenance import code_identity

        return code_identity()
    except Exception as exc:  # no commit / no git: record, do not fail the run
        return {"repository": "data_management", "git_commit": None, "note": f"unidentified: {exc}"}


def candidate_series(run, final: dict[str, float], anchor: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per camera (window_end, candidate offset on the published scale, relative to the anchor).

    Each window's candidate is solved against an arbitrary reference; like ``confirm`` it is shifted
    by the median difference to the final published offsets of the cameras both contain (needs two).
    """
    rows: dict[str, list[tuple[float, float]]] = {}
    for step in run.steps:
        cand = step.candidate or {}
        common = [c for c in cand if c in final]
        if len(common) < 2:
            continue
        shift = float(np.median([final[c] - cand[c] for c in common]))
        for camera, value in cand.items():
            rows.setdefault(camera, []).append((step.window_end, value + shift - final[anchor]))
    return {c: (np.array([r[0] for r in v]), np.array([r[1] for r in v])) for c, v in rows.items()}


def make_report(run, cameras: list[str], clocks, anchor: str | None, settings, span: tuple[float, float],
                notes: dict[str, list[str]], extra: dict) -> tuple[dict, list[tuple[float, dict[str, float]]]]:
    """Report + the relative-offset schedule used for the sync table."""
    final = dict(run.final.offsets) if run.final is not None else {}
    scale_anchor = anchor if anchor in final else None
    schedule = []
    for index, step in enumerate(run.accepted):
        offs = step.offsets or {}
        base = offs.get(anchor, final.get(anchor)) if anchor else None
        if base is None:
            continue
        schedule.append((step.window_end, {c: v - base for c, v in offs.items() if c in clocks}))
    series = candidate_series(run, final, anchor) if scale_anchor else {}
    first_end, last_end = span
    stale = float(settings.stale_seconds)
    per_camera = {}
    for camera in cameras:
        included = np.array([s.window_end for s in run.accepted if camera in (s.offsets or {})])
        values = np.array([(s.offsets[camera] - (s.offsets.get(anchor, final.get(anchor, 0.)) if anchor else 0.))
                           for s in run.accepted if camera in (s.offsets or {})]) if anchor else np.array([])
        unresolved, stale_spans = stale_unresolved_spans(included, first_end, last_end, stale)
        fps = 1.0 / clocks[camera].period
        entry = {
            "resolved": camera in final, "anchor": camera == anchor,
            "final_offset_seconds": None if camera not in final or anchor is None else float(final[camera] - final[anchor]),
            "offset_median_seconds": float(np.median(values)) if len(values) else None,
            "offset_min_seconds": float(values.min()) if len(values) else None,
            "offset_max_seconds": float(values.max()) if len(values) else None,
            "accepted_windows_including": int(len(included)),
            "unresolved_spans": unresolved, "stale_spans": stale_spans,
            "frame_rate": fps, "uncertain": list(notes.get(camera, [])),
        }
        t, y = series.get(camera, (np.zeros(0), np.zeros(0)))
        entry["candidate_windows"] = int(len(t))
        fit = robust_line(t, y) if len(t) >= MIN_FIT_WINDOWS and np.ptp(t) >= MIN_FIT_SECONDS else None
        if fit is None:
            entry.update(drift_seconds_per_hour=None, spread_mad_seconds=None, spread_mad_frames=None)
            if camera in final:
                entry["uncertain"].append("too few candidate windows to estimate drift/spread")
        else:
            slope, intercept = fit
            mad = float(np.median(np.abs(y - (slope * t + intercept))))
            entry.update(drift_seconds_per_hour=slope * 3600.0, drift_frames_per_hour=slope * 3600.0 * fps,
                         spread_mad_seconds=mad, spread_mad_frames=mad * fps, fit_intercept_seconds=intercept)
            if mad * fps > 1.0:
                entry["uncertain"].append(f"candidate spread {mad * fps:.2f} frames exceeds 1 frame")
        if camera not in final:
            entry["uncertain"].append("UNRESOLVED: no confirmed offset; none is reported (not 0)")
        if stale_spans:
            entry["uncertain"].append("stale: no accepted evidence for more than stale_seconds in some spans")
        per_camera[camera] = entry
    report = {
        "cameras": per_camera, "anchor": anchor,
        "run": {"windows": len(run.steps), "accepted": len(run.accepted), "confirmed_at_seconds": run.confirmed_at,
                "span_seconds": list(span)},
        "settings": settings_dict(settings), **extra,
    }
    return report, schedule


def settings_dict(settings) -> dict:
    from dataclasses import asdict

    return asdict(settings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--update-seconds", type=float, default=20.0,
                        help="seconds between windows (default 20: with audio a 60 min run is ~12-15 min; 10 doubles it)")
    parser.add_argument("--max-offset-seconds", type=float, default=3.0)
    parser.add_argument("--start-seconds", type=float)
    parser.add_argument("--end-seconds", type=float)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--prior", help="accepted for compatibility; the generator has no seeding hook, so it is ignored")
    parser.add_argument("--audio-workers", type=int, default=6)
    args = parser.parse_args()

    def log(message: str) -> None:
        print(message, flush=True)

    from data_management import alignment_generator as ag
    from data_management.calibration import load_calibrations_json
    from data_management.label_sets import frame_alignment as fa
    from data_management.label_sets.alignment_labels import GeneratorSettings, window_ends_for
    from data_management.label_sets.provenance import file_sha256, short_digest
    from data_management.label_sets.store import Provenance, utc_now

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.load(Path(args.dataset_json))
    if args.prior:
        log("--prior ignored: run_offline/RecordedAlignmentGenerator has no hook to seed the search centre "
            "(discovery always starts at 0 with a 3 s radius); not changing policy.")

    detections: dict[str, CameraDetections] = {}
    skipped = {}
    for camera_id, camera in sorted(dataset.cameras.items()):
        det = load_detections(Path(args.detections), camera_id)
        if det is None or not len(det.frames):
            skipped[camera_id] = "no detection shards"
        elif camera.issues:
            skipped[camera_id] = f"unresolved stream layout: {camera.issues}"
        else:
            detections[camera_id] = det
    if len(detections) < 2:
        raise SystemExit(f"Need detections for at least 2 cameras; have {sorted(detections)} (skipped: {skipped}). "
                         f"Run detect_trt first.")
    for camera_id, why in skipped.items():
        log(f"{camera_id}: excluded ({why})")
    cameras = sorted(detections)

    started = time.time()
    selected = {c: dataset.cameras[c] for c in cameras}
    index_dir = index_dir_for(selected, out / "frame_index")
    clocks = {c: build_clock(selected[c], index_dir) for c in cameras}
    for c in cameras:
        if int(detections[c].frames[-1]) >= clocks[c].frame_count:
            raise SystemExit(f"{c}: detections reach frame {int(detections[c].frames[-1])} but stream has "
                             f"{clocks[c].frame_count}")
    log(f"clocks built in {time.time() - started:.1f}s: "
        + ", ".join(f"{c} {clocks[c].frame_count}f/{clocks[c].times[-1]:.1f}s" for c in cameras))
    notes: dict[str, list[str]] = {c: [] for c in cameras}
    for c in cameras:
        cam = selected[c]
        for part in cam.parts[1:]:
            if not part.verified:
                notes[c].append(f"part {Path(part.path).name} joined without overlap evidence: unknown gap at the "
                                "file boundary shifts later time offsets by that gap")
        notes[c] += cam.notes and [f"dataset note: {n}" for n in cam.notes if "overlap evidence" not in n] or []
        if detections[c].clipped_confidences:
            notes[c].append(f"{detections[c].clipped_confidences} keypoint confidences clipped into [0,1]")
        if detections[c].dropped_people:
            notes[c].append(f"{detections[c].dropped_people} non-finite detections dropped")

    settings = GeneratorSettings(
        window_seconds=args.window_seconds, max_offset_seconds=args.max_offset_seconds,
        update_seconds=args.update_seconds, use_audio=not args.no_audio, verify_calibration=False,
        calibration=str(args.calibration))
    calibrations = load_calibrations_json(Path(args.calibration))
    calibration_ids = {}
    for c in cameras:
        cid = selected[c].calibration_id
        if cid in calibrations:
            calibration_ids[c] = cid
        else:
            calibration_ids[c] = None
            notes[c].append(f"calibration id {cid!r} not in {Path(args.calibration).name}: no visual evidence")
            log(f"{c}: calibration {cid!r} missing; camera can only be aligned by audio")
    cfg = settings.live_config(cameras, calibration_ids)

    # Windows: discovery-length windows every update_seconds where >= 2 cameras have detections.
    intervals = {c: [(float(clocks[c].time_at(a)), float(clocks[c].time_at(b))) for a, b in detections[c].runs()]
                 for c in cameras}
    spans = []
    for lo, hi in coverage_spans(intervals):
        lo = max(lo, args.start_seconds if args.start_seconds is not None else lo)
        hi = min(hi, args.end_seconds if args.end_seconds is not None else hi)
        if hi > lo:
            spans.append((lo, hi))
    ends = window_ends_for(spans, settings)
    if not ends:
        raise SystemExit("No span with >= 2 cameras of detections is long enough for a discovery window "
                         f"({max(settings.window_seconds, ag.INITIAL_WINDOW_SECONDS):g} s).")
    log(f"{len(ends)} windows over spans {[(round(a, 1), round(b, 1)) for a, b in spans]}")

    audio = None
    if settings.use_audio:
        audio = {c: AudioSource(selected[c], clocks[c], log, args.audio_workers) for c in cameras}
    margin = max(args.max_offset_seconds, ag.INITIAL_SEARCH_SECONDS) + 1.0
    provider = EvidenceProvider(selected, clocks, detections, calibration_ids, margin, audio, args.audio_workers)

    started = time.time()
    run = ag.run_offline(cfg, provider, calibrations, start=min(ends) - ag.INITIAL_WINDOW_SECONDS, end=max(ends),
                         window_ends=ends)
    elapsed = time.time() - started
    log(f"run_offline: {len(run.steps)} steps, {len(run.accepted)} accepted, confirmed_at={run.confirmed_at}, "
        f"{elapsed:.1f}s ({elapsed / max(1, len(run.steps)):.2f}s/step; evidence {provider.seconds:.1f}s of which "
        f"audio {provider.audio_seconds:.1f}s); median step {np.median(np.diff(provider.call_starts + [started + elapsed])):.2f}s")

    final = dict(run.final.offsets) if run.final is not None else {}
    anchor = next((c for c in cameras if c in final), None)
    span = (min(ends), max(ends))
    audio_info = {}
    if audio is not None:
        for c, source in audio.items():
            audio_info[c] = {"enabled": source.enabled, "disabled_reason": source.reason, "chunks_loaded": source.loaded,
                             "chunk_failures": source.failures, "statuses": source.statuses,
                             "load_seconds": round(source.seconds, 2)}
            if not source.enabled or source.failures:
                notes[c].append(f"audio unavailable/partial ({source.statuses})")
    gaps = np.diff(provider.call_starts + [started + elapsed])
    timing = {"step_seconds_median": float(np.median(gaps)) if len(gaps) else None,
              "step_seconds_p90": float(np.percentile(gaps, 90)) if len(gaps) else None,
              "run_offline_seconds": elapsed, "steps": len(run.steps), "seconds_per_step": elapsed / max(1, len(run.steps)),
              "evidence_seconds": provider.seconds, "audio_seconds": provider.audio_seconds,
              "update_seconds": args.update_seconds}
    extra = {"dataset": dataset.dataset_id, "excluded_cameras": skipped, "audio": audio_info, "timing": timing,
             "prior": {"given": args.prior, "used": False},
             "notes": ["Offsets are seconds on each camera's PTS clock; common time = camera time - offset; "
                       "relative to the anchor camera.",
                       "Parts are joined consecutively on the PTS clock (no chunk start times in dataset.json)."]}
    report, schedule = make_report(run, cameras, clocks, anchor, settings, span, notes, extra)
    report["unresolved_cameras"] = [c for c in cameras if c not in final]

    # Provenance.
    engine, dm = git_identity(REPO), dm_identity()
    config = {k: v for k, v in settings_dict(settings).items() if k != "calibration"}
    config.update({
        "dataset": dataset.dataset_id, "cameras": cameras, "calibration": str(args.calibration),
        "calibration_sha256": file_sha256(Path(args.calibration)), "calibration_ids": calibration_ids,
        "calibration_ids_verified_by": "caller (engine mount check); passed through unverified here",
        "initial_window_seconds": ag.INITIAL_WINDOW_SECONDS, "initial_search_seconds": ag.INITIAL_SEARCH_SECONDS,
        "start_seconds": args.start_seconds, "end_seconds": args.end_seconds,
        "evidence": "engine dense detections (native pixels) + recorded audio on smoothed PTS clocks",
        "detections_dir": str(args.detections), "window_ends": [ends[0], ends[-1], len(ends)],
    })
    name = f"data_engine.alignment.{short_digest({'engine': engine, 'dm': dm, 'config': config})}"
    flags = ["--window-seconds", f"{args.window_seconds:g}", "--update-seconds", f"{args.update_seconds:g}",
             "--max-offset-seconds", f"{args.max_offset_seconds:g}"] + (["--no-audio"] if args.no_audio else [])
    provenance = Provenance(
        name=name, kind="machine",
        description=("Frame alignment from data_management's recorded alignment generator "
                     "(alignment_generator.run_offline) driven by the Data Engine over the whole recording: dense 2D "
                     "pose detections (YOLOv8x-pose + ViTPose, native pixels) and recorded audio placed on smoothed "
                     "PTS clocks; policy and estimation are unmodified data_management code."),
        registered_at=utc_now(),
        reproduction={"steps": ["python src/data_engine/workers/detect_trt.py ... (dense detections)",
                                "python src/data_engine/workers/align_dm.py --dataset-json <dataset.json> "
                                "--detections <dir> --calibration <json> --out <dir> " + " ".join(flags)],
                      "code": {"data_engine": engine, "data_management": dm}, "config": config},
    )
    data = fa.from_alignment_run(run, key=dataset.key, clocks=clocks, nominal_fps=settings.sample_fps,
                                 provenance=name, cameras=cameras)
    data["source"]["tool"] = "data_engine.workers.align_dm"
    data["source"]["anchor"] = anchor

    clock_times = {c: np.asarray(clocks[c].times, dtype=np.float64) for c in cameras}
    table = build_sync_table(clock_times, schedule, anchor=anchor) if anchor else {
        "cameras": np.array(cameras), "aligned_time": np.zeros(0), "native": np.zeros((0, len(cameras)), np.int32),
        "offsets_seconds": np.zeros((0, len(cameras)))}

    (out / "alignment.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps(provenance.to_dict(), indent=1, default=str), encoding="utf-8")
    np.savez_compressed(out / "clocks.npz", **clock_times)
    np.savez_compressed(out / "sync_table.npz", **table)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")

    log("\ncamera      resolved  offset_vs_anchor  windows  drift_s/h  spread_frames")
    for c in cameras:
        e = report["cameras"][c]
        offset = "None" if e["final_offset_seconds"] is None else f"{e['final_offset_seconds']:.3f}"
        drift = "-" if e.get("drift_seconds_per_hour") is None else f"{e['drift_seconds_per_hour']:.2f}"
        spread = "-" if e.get("spread_mad_frames") is None else f"{e['spread_mad_frames']:.2f}"
        log(f"{c:10s}  {str(e['resolved']):8s}  {offset:>16s}  {e['accepted_windows_including']:7d}  {drift:>9s}  {spread:>13s}")
    log(f"wrote {out}")
    return 0 if len([c for c in cameras if c in final]) >= 2 else 3


if __name__ == "__main__":
    raise SystemExit(main())
