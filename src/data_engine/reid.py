"""Appearance re-identification (ReID): embedding extraction plus pure helpers and a decision rule.

Embeddings are advisory evidence for unknowns the geometry cannot decide (identity across a gap,
rival tie-breaks). Nothing here changes identities by itself; ``decide_best`` is the calibrated
accept rule the experiments in ``experiments/reid`` evaluate against the golden reference.

Model: official OSNet x1_0 (Market-1501) from deep-person-reid, loaded with plain torch from a
directory that holds ``osnet.py`` and ``osnet_market1501.pth`` (sha256 pinned below). Preprocessing
is identical to the earlier cvat_labeling work: crop the detection box (clamped to the image), PIL
bilinear resize to 128x256 (WxH), ImageNet mean/std normalisation, 512-d output, L2-normalised.

Extraction writes ``<out>/<camera>.npz`` with ``rows`` (raw detection row), ``embedding``
(float16, L2-normalised), ``model`` (json metadata) and ``frames_hash`` (resume key).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

MODEL_NAME = "osnet_x1_0_market1501"
EXPECTED_SHA256 = {
    "osnet_market1501.pth": "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154",
    "osnet.py": "c7c1c29187d6330f859c91da229271531920464c7011aec13842a086b2263cae",
}
DEFAULT_MODEL_DIR = Path(os.environ.get(
    "DATA_ENGINE_REID_DIR", r"C:\cvat_labeling_data\20250505071549\reid_osnet_20261117"))
INPUT_WH = (128, 256)
FALLBACK_SCORE = 0.2


# ====================================================================== pure helpers
def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def quality_weights(boxes: np.ndarray, *, min_height: float = 64.0, full_height: float = 256.0) -> np.ndarray:
    """Weight per detection in (0, 1]: taller boxes (more pixels) are more reliable. boxes: (N, >=4) xyxy
    (a fifth column, if present, is the person score and scales the weight)."""
    boxes = np.asarray(boxes, dtype=np.float32)
    height = boxes[:, 3] - boxes[:, 1]
    weight = np.clip((height - min_height / 2) / max(full_height - min_height / 2, 1e-6), 0.05, 1.0)
    if boxes.shape[1] > 4:
        weight = weight * np.clip(boxes[:, 4], 0.05, 1.0)
    return weight.astype(np.float32)


def aggregate(embeddings: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray | None:
    """Descriptor of a set of detections: (weighted) mean of the L2-normalised embeddings, re-normalised.
    Returns None for an empty set."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or len(embeddings) == 0:
        return None
    e = l2_normalize(embeddings)
    if weights is None:
        mean = e.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float32)
        if w.shape != (len(e),) or not np.all(w >= 0) or w.sum() <= 0:
            raise ValueError("weights must be non-negative, one per embedding, with a positive sum")
        mean = (e * w[:, None]).sum(axis=0) / w.sum()
    n = np.linalg.norm(mean)
    return None if n < 1e-9 else (mean / n).astype(np.float32)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(Na, Nb) cosine similarities; inputs need not be normalised."""
    return l2_normalize(np.atleast_2d(a)) @ l2_normalize(np.atleast_2d(b)).T


def decide_best(sims: np.ndarray, *, s_min: float, margin: float, mutual: np.ndarray | None = None) -> int:
    """Accept rule over one query's similarities to its candidates (1-D). Returns the winning candidate index
    or -1 (leave open) when: fewer than one candidate, best < s_min, (best - second) < margin when a second
    exists, or ``mutual`` (a boolean per candidate: this candidate's best query is the query) is given and the
    best candidate is not mutual. NaN similarities are never winners."""
    sims = np.asarray(sims, dtype=np.float64)
    valid = np.isfinite(sims)
    if sims.ndim != 1 or not valid.any():
        return -1
    masked = np.where(valid, sims, -np.inf)
    order = np.argsort(-masked, kind="stable")
    best = int(order[0])
    if masked[best] < s_min:
        return -1
    if valid.sum() > 1 and masked[best] - masked[order[1]] < margin:
        return -1
    if mutual is not None and not bool(np.asarray(mutual)[best]):
        return -1
    return best


def decide_matrix(sim: np.ndarray, *, s_min: float, margin: float, mutual: bool = False) -> np.ndarray:
    """``decide_best`` for every row of a (Q, G) matrix; with ``mutual`` the winner must also have the query as
    its own best row (column-wise argmax; NaN ignored). Returns (Q,) winner indices or -1."""
    sim = np.asarray(sim, dtype=np.float64)
    out = np.full(len(sim), -1, dtype=np.int64)
    column_best = np.argmax(np.where(np.isfinite(sim), sim, -np.inf), axis=0) if mutual and sim.size else None
    for q in range(len(sim)):
        m = None if column_best is None else (column_best == q)
        out[q] = decide_best(sim[q], s_min=s_min, margin=margin, mutual=m)
    return out


def threshold_for_false_accepts(same: np.ndarray, different: np.ndarray, max_rate: float) -> float:
    """Smallest similarity threshold s such that the share of ``different`` scores >= s is <= max_rate.
    (Accept when sim >= s; with max_rate 0 it lies just above the largest different-person score.)"""
    different = np.sort(np.asarray(different, dtype=np.float64))
    if len(different) == 0:
        return -1.0
    allowed = int(np.floor(max_rate * len(different)))
    if allowed >= len(different):
        return float(different[0])
    cut = different[len(different) - 1 - allowed]  # largest score that must be rejected
    return float(np.nextafter(cut, np.inf))


def roc_auc(same: np.ndarray, different: np.ndarray) -> float:
    """P(same > different) with ties counted half (Mann-Whitney)."""
    same, different = np.asarray(same, dtype=np.float64), np.asarray(different, dtype=np.float64)
    if len(same) == 0 or len(different) == 0:
        return float("nan")
    allv = np.concatenate([same, different])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u = ranks[:len(same)].sum() - len(same) * (len(same) + 1) / 2
    return float(u / (len(same) * len(different)))


# ====================================================================== model
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(model_dir: Path | None = None, device: str = "cuda"):
    """Load the pinned OSNet; returns (model in eval mode, metadata dict). Hash mismatch raises."""
    import torch
    model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
    for name, digest in EXPECTED_SHA256.items():
        if sha256_file(model_dir / name) != digest:
            raise RuntimeError(f"ReID artifact {name} does not match its pinned hash")
    spec = importlib.util.spec_from_file_location("official_osnet_engine", model_dir / "osnet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.osnet_x1_0(num_classes=751, pretrained=False)
    state = torch.load(model_dir / "osnet_market1501.pth", map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.get("state_dict", state).items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    meta = {"name": MODEL_NAME, "weights_sha256": EXPECTED_SHA256["osnet_market1501.pth"],
            "arch_sha256": EXPECTED_SHA256["osnet.py"], "input_wh": list(INPUT_WH), "preprocess": "box crop, PIL bilinear "
            "resize 128x256, imagenet norm, L2"}
    return model, meta


def prepare_crop(image_bgr: np.ndarray, box: Sequence[float]) -> np.ndarray | None:
    """(3, 256, 128) float32 normalised tensor array for one box of a BGR frame, or None if degenerate."""
    from PIL import Image
    h, w = image_bgr.shape[:2]
    x0, y0 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x1, y1 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
    if x1 - x0 < 4 or y1 - y0 < 8:
        return None
    crop = Image.fromarray(np.ascontiguousarray(image_bgr[y0:y1, x0:x1, ::-1]))
    arr = np.asarray(crop.resize(INPUT_WH, Image.BILINEAR), dtype=np.float32) / 255.0
    arr = (arr - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


# ====================================================================== extraction
def kept_mask(workspace, camera: str, boxes: np.ndarray) -> np.ndarray:
    path = workspace.path("tracklets", f"{camera}.npz")
    if path.is_file():
        with np.load(path) as data:
            return data["kept"].astype(bool)
    return boxes[:, 4] >= FALLBACK_SCORE


def _frames_hash(frames: np.ndarray, rows: np.ndarray, model_hash: str) -> str:
    h = hashlib.sha256()
    h.update(model_hash.encode())
    h.update(np.ascontiguousarray(rows, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def _frame_iterator(workspace, camera: str, frames: Sequence[int]):
    """Yield (native frame, BGR image) for the sorted ``frames`` with sequential, time-verified reads."""
    from .video import ExactReader, frame_times
    cam = workspace.dataset.cameras[camera]
    readers: dict[str, ExactReader] = {}
    try:
        for frame in frames:
            path, local = cam.locate(int(frame))
            if path not in readers:
                for r in readers.values():
                    r.close()
                readers = {path: ExactReader(path, frame_times(path, workspace.path("frame_index")))}
            yield int(frame), readers[path].read(local)
    finally:
        for r in readers.values():
            r.close()


def extract_embeddings(workspace, *, stride: int = 5, cameras: Iterable[str] | None = None, batch: int = 128,
                       device: str = "cuda", out_dir: Path | None = None, model_dir: Path | None = None,
                       frames: dict[str, np.ndarray] | None = None, row_filter: Callable | None = None,
                       backend: tuple | None = None, log: Callable[[str], None] = print) -> Path:
    """Embed every kept detection on every ``stride``-th native frame (or on ``frames[camera]`` when given).

    One sequential decoding pass per camera; resumable per camera (an existing file with the same model and
    row set is kept). ``out_dir`` defaults to ``<workspace>/reid``. ``backend`` optionally replaces the default
    OSNet with ``(model, meta, prepare(image_bgr, box) -> array | None, dim)``; ``meta`` needs ``weights_sha256``.
    Returns the output directory.
    """
    import torch
    out_dir = Path(out_dir) if out_dir is not None else workspace.path("reid")
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(cameras) if cameras is not None else list(workspace.dataset.cameras)
    batch = min(int(batch), 256)
    model = meta = None
    prepare, dim = prepare_crop, 512
    if backend is not None:
        model, meta, prepare, dim = backend
    for camera in names:
        det = workspace.detections.camera(camera)
        keep = kept_mask(workspace, camera, det.boxes)
        frame_of_row = det.frame_of_row
        if frames is not None:
            wanted = np.isin(frame_of_row, np.asarray(frames[camera]))
        else:
            wanted = (frame_of_row % stride) == 0
        rows = np.flatnonzero(keep & wanted & det.covered[frame_of_row])
        if row_filter is not None:
            rows = rows[row_filter(camera, rows)]
        target = out_dir / f"{camera}.npz"
        if model is None:
            model, meta = load_model(model_dir, device)
        key = _frames_hash(frame_of_row[rows], rows, meta["weights_sha256"])
        if target.is_file():
            with np.load(target) as old:
                if str(old["frames_hash"]) == key:
                    log(f"{camera}: up to date ({len(rows)} rows)")
                    continue
        order_frames = np.unique(frame_of_row[rows])
        by_frame = {int(f): rows[frame_of_row[rows] == f] for f in order_frames}
        embeddings = np.zeros((len(rows), dim), dtype=np.float16)
        position = {int(r): i for i, r in enumerate(rows)}
        valid = np.zeros(len(rows), dtype=bool)
        q: queue.Queue = queue.Queue(maxsize=3)
        timings = {"decode_s": 0.0}

        def producer():
            try:
                t0 = time.time()
                chunk_rows, chunk_x = [], []
                for frame, image in _frame_iterator(workspace, camera, order_frames):
                    timings["decode_s"] = time.time() - t0
                    for r in by_frame[frame]:
                        x = prepare(image, det.boxes[r, :4])
                        if x is not None:
                            chunk_rows.append(int(r)); chunk_x.append(x)
                    while len(chunk_rows) >= batch:
                        q.put((chunk_rows[:batch], np.stack(chunk_x[:batch])))
                        chunk_rows, chunk_x = chunk_rows[batch:], chunk_x[batch:]
                if chunk_rows:
                    q.put((chunk_rows, np.stack(chunk_x)))
                q.put(None)
            except BaseException as error:  # surfaced in the consumer
                q.put(error)

        t_start = time.time()
        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        done = 0
        with torch.inference_mode():
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                chunk_rows, x = item
                e = model(torch.from_numpy(x).to(device)).float().cpu().numpy()
                e = l2_normalize(e)
                if not np.isfinite(e).all():
                    raise ValueError("non-finite embedding")
                for r, v in zip(chunk_rows, e):
                    i = position[r]
                    embeddings[i] = v.astype(np.float16)
                    valid[i] = True
                done += len(chunk_rows)
        thread.join()
        elapsed = time.time() - t_start
        tmp = out_dir / f"{camera}.tmp.npz"
        np.savez(tmp, rows=rows[valid].astype(np.int32), embedding=embeddings[valid], frames_hash=key,
                 model=json.dumps({**meta, "stride": stride, "device": device}))
        os.replace(tmp, target)
        log(f"{camera}: {done} crops from {len(order_frames)} frames in {elapsed:.1f}s "
            f"({done / max(elapsed, 1e-9):.0f} crops/s, {len(order_frames) / max(elapsed, 1e-9):.1f} frames/s)")
    return out_dir


def load_embeddings(directory: Path, camera: str) -> tuple[np.ndarray, np.ndarray]:
    """(rows int32, embedding float32 (N, 512)) of one camera."""
    with np.load(Path(directory) / f"{camera}.npz") as data:
        return data["rows"], data["embedding"].astype(np.float32)
