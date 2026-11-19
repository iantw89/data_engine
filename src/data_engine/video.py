"""Exact frame access by native frame index.

Naive ``CAP_PROP_POS_FRAMES`` seeks can return the wrong frame on these recordings
(data_management README), so every returned frame is verified by its presentation time
against the file's frame-time index, the same rule as data_management's ExactFrameReader.
The frame-time index per file is computed once with ffprobe and cached in the workspace.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from .dataset import Dataset


def index_cache_path(path: str, cache_dir: Path) -> Path:
    source = Path(path)
    stat = source.stat()
    digest = hashlib.sha256(f"{source.name}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:20]
    return Path(cache_dir) / f"{digest}.npy"


def frame_times(path: str, cache_dir: Path | None = None) -> np.ndarray:
    """Presentation times (seconds from the first frame) of every video frame, in display order.

    The ingest stage pre-fills the cache from data_management's MP4 index (it parses the
    moov box, which takes milliseconds); ffprobe is the fallback and reads the whole file.
    """
    cache = index_cache_path(path, cache_dir) if cache_dir is not None else None
    if cache is not None and cache.is_file():
        return np.load(cache)
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    times = np.sort(np.array([float(p["pts_time"]) for p in json.loads(output)["packets"] if "pts_time" in p]))
    times = times - times[0]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, times)
    return times


class ExactReader:
    """Thread-safe reader for one file; frames are checked against the time index."""

    def __init__(self, path: str, times: np.ndarray):
        self.path, self.times = path, times
        self._lock = threading.Lock()
        self._capture: cv2.VideoCapture | None = None
        self._position = -1
        spacing = np.diff(times)
        self._tolerance_ms = 0.4 * 1000 * float(spacing[spacing > 0].min()) if (spacing > 0).any() else 1.0

    def close(self) -> None:
        with self._lock:
            if self._capture is not None:
                self._capture.release()
            self._capture, self._position = None, -1

    def _open(self) -> cv2.VideoCapture:
        if self._capture is None:
            capture = cv2.VideoCapture(self.path)
            if not capture.isOpened():
                raise OSError(f"Cannot open {self.path}")
            self._capture, self._position = capture, -1
        return self._capture

    def _next(self, capture: cv2.VideoCapture) -> np.ndarray:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise OSError(f"Decode failed in {self.path} after frame {self._position}")
        milliseconds = capture.get(cv2.CAP_PROP_POS_MSEC)
        index = int(np.searchsorted(self.times * 1000, milliseconds - self._tolerance_ms))
        if index >= len(self.times) or abs(self.times[index] * 1000 - milliseconds) > self._tolerance_ms:
            raise RuntimeError(f"Decoded timestamp {milliseconds:.3f} ms has no exact match in {self.path}")
        self._position = index
        return frame

    def read(self, index: int) -> np.ndarray:
        if not 0 <= index < len(self.times):
            raise IndexError(f"Frame {index} outside 0..{len(self.times) - 1}")
        with self._lock:
            capture = self._open()
            if not (self._position < index <= self._position + 120):
                back = 40
                while True:
                    start = max(0, index - back)
                    if start == 0:
                        capture.release()
                        self._capture = None
                        capture = self._open()
                    else:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
                    frame = self._next(capture)
                    if self._position <= index:
                        break
                    if start == 0:
                        raise RuntimeError(f"Cannot reach frame {index} in {self.path}")
                    back *= 4
                if self._position == index:
                    return frame
            while True:
                frame = self._next(capture)
                if self._position == index:
                    return frame
                if self._position > index:
                    raise RuntimeError(f"Frame {index} was skipped while decoding {self.path}")


class FrameSource:
    """Read any (camera, native frame) of a dataset as a BGR image, with a small LRU cache."""

    def __init__(self, dataset: Dataset, index_dir: Path, *, readers: int = 16, cache_frames: int = 48):
        self.dataset, self.index_dir = dataset, Path(index_dir)
        self._readers: OrderedDict[tuple[str, int], ExactReader] = OrderedDict()
        self._frames: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._max_readers, self._max_frames = readers, cache_frames

    def _reader(self, path: str, lane: int) -> ExactReader:
        # A few readers per file ("lanes") keep sequential access cheap when several
        # requests walk different parts of the same file.
        with self._lock:
            key = (path, lane)
            reader = self._readers.pop(key, None)
            if reader is None:
                reader = ExactReader(path, frame_times(path, self.index_dir))
            self._readers[key] = reader
            while len(self._readers) > self._max_readers:
                _, evicted = self._readers.popitem(last=False)
                evicted.close()
            return reader

    def read(self, camera: str, frame: int) -> np.ndarray:
        key = (camera, int(frame))
        with self._lock:
            cached = self._frames.get(key)
            if cached is not None:
                self._frames.move_to_end(key)
                return cached
        path, local = self.dataset.cameras[camera].locate(int(frame))
        image = self._reader(path, lane=(local // 600) % 4).read(local)
        with self._lock:
            self._frames[key] = image
            while len(self._frames) > self._max_frames:
                self._frames.popitem(last=False)
        return image

    def close(self) -> None:
        with self._lock:
            for reader in self._readers.values():
                reader.close()
            self._readers.clear()
            self._frames.clear()
