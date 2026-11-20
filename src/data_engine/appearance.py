"""Appearance embeddings for a chosen set of detections (fast-reid SBS R50-NL, MSMT17).

Used as one cue of the swap check (``solve/swap_check.py``): do the people on two tracks look the
same before and after they came close? Appearance was measured not to be reliable enough to decide
identities on this footage (``experiments/reid/REPORT.md``); as a before/after comparison of two
specific people around one moment it was the strongest cue at close range
(``experiments/switches/REPORT.md``). It is optional: without the model, the weights or a GPU the
swap check simply has one cue fewer.

The fast-reid repository is read from disk without installing it; config libraries it imports but
does not need for a forward pass are stubbed.
"""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path
from typing import Callable

import numpy as np

MODEL_NAME = "fastreid_sbs_R50_NL_msmt17"
INPUT_HW = (384, 128)
DIM = 2048


class _Any:
    """Permissive stand-in for config libraries fast-reid imports (never used for real work here)."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return _Any()

    def __call__(self, *args, **kwargs):
        return _Any()


_STUBBED: list[str] = []


def _stub_package(name: str) -> None:
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except ImportError:
        pass
    _STUBBED.append(name)
    module = types.ModuleType(name)
    module.__path__ = []

    def _get(attribute):
        if attribute.startswith("__"):
            raise AttributeError(attribute)
        return _Any

    module.__getattr__ = _get
    sys.modules[name] = module


def build_model(repo: Path, weights: Path, device: str = "cuda"):
    import torch
    import torchvision  # noqa: F401  (import before the stubs)
    from torch import nn

    if "termcolor" not in sys.modules:
        try:
            __import__("termcolor")
        except ImportError:
            module = types.ModuleType("termcolor")
            module.colored = lambda text, *a, **k: text
            sys.modules["termcolor"] = module
            _STUBBED.append("termcolor")
    for package in ("yacs", "yacs.config", "fvcore", "fvcore.common", "fvcore.common.checkpoint", "fvcore.common.file_io",
                    "fvcore.nn", "tabulate", "faiss", "iopath", "iopath.common", "iopath.common.file_io", "cloudpickle"):
        _stub_package(package)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from fastreid.layers import pooling
    from fastreid.modeling.backbones.resnet import Bottleneck, ResNet

    # The stand-ins were only needed to import the two modules above; leaving them in sys.modules would
    # hand them to unrelated code (torch looks for ``tabulate`` at exit).
    while _STUBBED:
        sys.modules.pop(_STUBBED.pop(), None)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = ResNet(1, "BN", False, False, True, Bottleneck, [3, 4, 6, 3], [0, 2, 3, 0])
            self.heads = nn.Module()
            self.heads.pool_layer = pooling.GeneralizedMeanPoolingP()
            self.heads.bnneck = nn.BatchNorm2d(2048)
            self.register_buffer("pixel_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1), False)
            self.register_buffer("pixel_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1), False)

        def forward(self, x):  # (B, 3, 384, 128) RGB 0..255
            x = (x - self.pixel_mean) / self.pixel_std
            return self.heads.bnneck(self.heads.pool_layer(self.backbone(x)))[..., 0, 0]

    net = Net()
    state = torch.load(weights, map_location="cpu", weights_only=False)["model"]
    state = {k: v for k, v in state.items() if not k.startswith("heads.classifier") and k not in ("pixel_mean", "pixel_std")}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if [k for k in missing if k not in ("pixel_mean", "pixel_std")] or unexpected:
        raise RuntimeError(f"fast-reid weights do not match the model: missing {missing}, unexpected {unexpected}")
    return net.to(device).eval()


def prepare(image_bgr: np.ndarray, box) -> np.ndarray | None:
    import cv2

    height, width = image_bgr.shape[:2]
    x0, y0 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x1, y1 = min(width, int(np.ceil(box[2]))), min(height, int(np.ceil(box[3])))
    if x1 - x0 < 4 or y1 - y0 < 8:
        return None
    crop = cv2.resize(image_bgr[y0:y1, x0:x1, ::-1], (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(crop.transpose(2, 0, 1).astype(np.float32))


_MODELS: dict[str, tuple] = {}  # weights path -> (model, meta): one copy on the GPU however many embedders ask


def embedder(workspace, camera_names: list[str], log: Callable[[str], None] = print, *,
             directory: str = "appearance", incremental: bool = False) -> Callable | None:
    """``embed(rows_by_camera_index) -> [(row -> slot, unit vectors)] per camera`` for a workspace, or None when
    the model, its weights or a GPU are not available (the reason is logged). ``camera_names`` gives the camera
    of every index (the scene's order).

    Embeddings are kept in ``evidence/<directory>``. By default a camera's file holds exactly the rows last
    asked for (the swap check: its cue values then depend on its own choice of crops only). With
    ``incremental`` the file is a cache that grows: only rows it does not hold are embedded."""
    site = workspace.site
    try:
        repo, weights = Path(site.repo("fast_reid")), Path(site.model("reid_fastreid"))
    except (KeyError, AttributeError):
        log("appearance: no fast-reid repository or weights configured; cue not used")
        return None
    if not repo.is_dir() or not weights.is_file():
        log(f"appearance: {repo} or {weights} not found; cue not used")
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            log("appearance: no GPU; cue not used")
            return None
    except ImportError:
        log("appearance: torch is not installed; cue not used")
        return None
    state = {}

    def embed(rows_by_camera: dict[int, np.ndarray]):
        from . import reid

        import json
        import os

        scene_names = list(camera_names)
        wanted = {scene_names[index]: np.unique(np.asarray(rows, np.int64)) for index, rows in rows_by_camera.items()
                  if len(rows)}
        out_dir = workspace.path("evidence", directory)
        held: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if incremental and "sha" not in state:
            state["sha"] = hashlib.sha256(weights.read_bytes()).hexdigest()
        if incremental:
            for camera in list(wanted):
                path = out_dir / f"{camera}.npz"
                if not path.is_file():
                    continue
                with np.load(path) as old:
                    if json.loads(str(old["model"])).get("weights_sha256") != state["sha"]:
                        continue  # another model: the cache starts again
                    held[camera] = (old["rows"].astype(np.int64), old["embedding"])
                wanted[camera] = np.setdiff1d(wanted[camera], held[camera][0])
                if not len(wanted[camera]):
                    del wanted[camera]
        if wanted and "model" not in state:
            if str(weights) not in _MODELS:
                _MODELS[str(weights)] = (build_model(repo, weights), {
                    "name": MODEL_NAME, "input_hw": list(INPUT_HW),
                    "weights_sha256": state.get("sha") or hashlib.sha256(weights.read_bytes()).hexdigest()})
            state["model"], state["meta"] = _MODELS[str(weights)]
        frames = {}
        for camera, rows in wanted.items():
            frames[camera] = np.unique(workspace.detections.camera(camera).frame_of_row[rows])
        if wanted:
            target = out_dir / "_new" if incremental else out_dir
            reid.extract_embeddings(workspace, cameras=list(wanted), frames=frames, batch=64, out_dir=target,
                                    row_filter=lambda camera, rows: np.isin(rows, wanted[camera]),
                                    backend=(state["model"], state["meta"], prepare, DIM), log=log)
            if incremental:
                for camera in wanted:
                    with np.load(target / f"{camera}.npz") as new:
                        rows, vectors, model = new["rows"].astype(np.int64), new["embedding"], str(new["model"])
                    if camera in held:
                        rows = np.concatenate([held[camera][0], rows])
                        vectors = np.concatenate([held[camera][1], vectors])
                    order = np.argsort(rows, kind="stable")
                    temporary = out_dir / f"{camera}.tmp.npz"
                    np.savez(temporary, rows=rows[order].astype(np.int32), embedding=vectors[order], frames_hash="cache",
                             model=model)
                    os.replace(temporary, out_dir / f"{camera}.npz")
                    (target / f"{camera}.npz").unlink()
        asked = set(wanted) | set(held)
        result = []
        for index, camera in enumerate(scene_names):
            count = len(workspace.detections.camera(camera).boxes)
            slot = np.full(count, -1, np.int32)
            vectors = np.zeros((0, DIM), np.float32)
            if camera in asked and (out_dir / f"{camera}.npz").is_file():
                rows, vectors = reid.load_embeddings(out_dir, camera)
                slot[rows] = np.arange(len(rows), dtype=np.int32)
                vectors = reid.l2_normalize(vectors)
            result.append((slot, vectors))
        return result

    return embed
