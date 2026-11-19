"""Shared setup for the coverage experiments.

Run everything with data_management's interpreter (it has TensorRT, CuPy, onnxruntime and
ultralytics), from the engine repo root:

    ../data_management/.venv/Scripts/python.exe experiments/coverage/<script>.py

The engine's own ``src`` and data_management's ``src`` are put on ``sys.path`` here.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT.parent / "data_management" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# data_management's interpreter ships OpenCV 4.13, where undistortPointsIter needs explicit R/P;
# the engine (OpenCV 5.1) calls it without them. Same maths, so wrap it rather than touch src/.
if cv2.__version__.startswith("4."):
    _undistort_iter = cv2.undistortPointsIter

    def _undistort_compat(src, camera_matrix, dist_coeffs, *args, criteria=None, **kw):
        return _undistort_iter(src, camera_matrix, dist_coeffs, None, None, criteria)

    cv2.undistortPointsIter = _undistort_compat

from data_engine.config import RECIPE  # noqa: E402
from data_engine.solve.scene import load_scene  # noqa: E402
from data_engine.workspace import Workspace  # noqa: E402

WORKSPACE = "golden_20250505071549_20250505080000@first600s"
DATASET = "golden_20250505071549_20250505080000"
SCRATCH = Path(r"C:\data_engine_data\_scratch\coverage")
SCRATCH.mkdir(parents=True, exist_ok=True)


def load(workspace: str = WORKSPACE, round_name: str = "r002"):
    ws = Workspace(workspace)
    scene = load_scene(ws, joint_peak=RECIPE["geometry"]["joint_peak"])
    tracks = dict(np.load(ws.root / "rounds" / round_name / "tracks3d.npz"))
    solution = dict(np.load(ws.root / "rounds" / round_name / "solution.npz"))
    return ws, scene, tracks, solution
