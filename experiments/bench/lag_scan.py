"""Which per-camera frame lag maximises cross-camera geometric agreement? (alignment check)

Usage: python experiments/bench/lag_scan.py WORKSPACE [max_lag] [t_start t_end]
For each camera, shifts it by -max_lag..+max_lag sync frames against all others and counts
the two-view pairs within half the accept gate. The best lag should be 0.
"""
import copy
import sys

import numpy as np

from data_engine.config import RECIPE
from data_engine.solve import evidence
from data_engine.solve.scene import load_scene
from data_engine.workspace import Workspace


def shifted(scene, index, lag):
    out = copy.copy(scene)
    out.cameras = list(scene.cameras)
    camera = copy.copy(scene.cameras[index])
    t = np.where(camera.t >= 0, camera.t + lag, -1)
    t[(t < 0) | (t >= scene.T)] = -1
    usable = np.flatnonzero(camera.kept & (t >= 0))
    order = usable[np.argsort(t[usable], kind="stable")]
    camera.t, camera.order = t, order
    camera.row_start = np.searchsorted(t[order], np.arange(scene.T + 1)).astype(np.int64)
    out.cameras[index] = camera
    return out


def restrict(scene, t0, t1):
    out = copy.copy(scene)
    out.cameras = []
    for camera in scene.cameras:
        camera = copy.copy(camera)
        keep = (camera.t[camera.order] >= t0) & (camera.t[camera.order] < t1)
        camera.order = camera.order[keep]
        camera.row_start = np.searchsorted(camera.t[camera.order], np.arange(scene.T + 1)).astype(np.int64)
        out.cameras.append(camera)
    return out


def main():
    workspace = Workspace(sys.argv[1])
    max_lag = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    scene = load_scene(workspace, joint_peak=0.3)
    if len(sys.argv) > 4:
        scene = restrict(scene, int(sys.argv[3]), int(sys.argv[4]))
    for camera in scene.cameras:
        if camera.cam is None or not len(camera.order):
            continue
        row = []
        for lag in range(-max_lag, max_lag + 1):
            samples = evidence.pair_samples(shifted(scene, camera.index, lag), RECIPE, step=5)
            mine = ((samples.cam_a == camera.index) | (samples.cam_b == camera.index)) & samples.plausible
            row.append(int((samples.u[mine] <= 0.5).sum()))
        best = int(np.argmax(row)) - max_lag
        print(f"{camera.camera}: best lag {best:+d}  counts {row}", flush=True)


if __name__ == "__main__":
    main()
