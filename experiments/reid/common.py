"""Shared loading for the ReID experiments: embeddings + golden reference + sync time, as flat arrays."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data_engine import reid  # noqa: E402
from data_engine.evaluation import load_reference  # noqa: E402
from data_engine.solve.scene import load_scene  # noqa: E402
from data_engine.store import det_key  # noqa: E402
from data_engine.workspace import Workspace  # noqa: E402

SCRATCH = Path(r"C:\data_engine_data\_scratch\reid")
WS_ID = "golden_20250505071549_20250505080000@first600s"
STORE = Path(r"C:\Users\iantw\git\data_management\data\labels")
DATASET = "golden_20250505071549_20250505080000"
EMB_DIR = SCRATCH / "golden_osnet"
SYNC_HZ = 10.0


@dataclass
class Data:
    ws: Workspace
    scene: object
    cameras: list[str]
    cam: np.ndarray        # (N,) camera index
    row: np.ndarray        # (N,) raw detection row in its camera
    t: np.ndarray          # (N,) sync index (10 Hz)
    person: np.ndarray     # (N,) golden person id or -1
    tracklet: np.ndarray   # (N,) engine tracklet id (global) or -1
    box: np.ndarray        # (N,5)
    emb: np.ndarray        # (N,512) float32 L2-normalised
    T: int

    def key(self, i: int) -> str:
        c = self.scene.cameras[self.cam[i]]
        return c.detections.key(int(self.row[i]))

    def frame(self, i: int) -> int:
        c = self.scene.cameras[self.cam[i]]
        return int(c.detections.frame_of_row[self.row[i]])


def load_scene_and_reference(ws_id: str = WS_ID):
    ws = Workspace(ws_id)
    scene = load_scene(ws, joint_peak=0.3, rays=False)
    ref = load_reference(scene, STORE, DATASET)
    return ws, scene, ref


def labelled_frames(ref):
    return {i: f for i, f in enumerate(ref.labelled_frames)}


def load_data(ws_id: str = WS_ID, emb_dir: Path = EMB_DIR) -> Data:
    ws, scene, ref = load_scene_and_reference(ws_id)
    parts = {k: [] for k in ("cam", "row", "t", "person", "tracklet", "box", "emb")}
    for c in scene.cameras:
        rows, emb = reid.load_embeddings(emb_dir, c.camera)
        rows = rows.astype(np.int64)
        parts["cam"].append(np.full(len(rows), c.index, np.int16))
        parts["row"].append(rows)
        parts["t"].append(c.t[rows])
        parts["person"].append(ref.person[c.index][rows])
        parts["tracklet"].append(c.tracklet[rows])
        parts["box"].append(c.detections.boxes[rows])
        parts["emb"].append(emb)
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    ok = cat["t"] >= 0
    cat = {k: v[ok] for k, v in cat.items()}
    order = np.lexsort((cat["row"], cat["cam"], cat["t"]))
    cat = {k: v[order] for k, v in cat.items()}
    return Data(ws, scene, [c.camera for c in scene.cameras], T=scene.T, **cat)
