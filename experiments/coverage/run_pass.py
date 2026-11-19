"""Run ViTPose (production path) over every planned job. Output: pass_vit.npz"""
import time

from common import *  # noqa
from pixel_pass import PixelPass

ws, scene, tracks, solution = load()
jobs = np.load(SCRATCH / "jobs.npz")
p = PixelPass(ws, scene)
t = time.time()
out = p.run(jobs["cam"], jobs["frame"], jobs["box"])
print("pass seconds", out["seconds"], "frames", out["frames"], "crops", out["crops"], "total", time.time() - t)
np.savez(SCRATCH / "pass_vit.npz", kp1=out["kp1"], kp2=out["kp2"], box2=out["box2"], seconds=out["seconds"])
