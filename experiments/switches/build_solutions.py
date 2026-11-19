"""Build and cache the solutions and oracle tracks used by the swap research (CPU, ~3 min)."""
import time

import common

t = time.time()
ws, scene, ref = common.load_scene_ref()
for name in ("adopted", "baseline"):
    sol = common.bench_variant(name)
    print(name, len(sol.tr_t), round(time.time() - t), "s", flush=True)
sol = common.autofit_alone()
print("autofit", len(sol.tr_t), round(time.time() - t), "s", flush=True)
p, tt, root, rows = common.oracle_tracks(scene, ref)
print("oracle", len(tt), round(time.time() - t), "s")
