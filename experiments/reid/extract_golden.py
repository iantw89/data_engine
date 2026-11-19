"""Embed every kept detection on the golden labelled frames (every 5th session frame) into _scratch/reid/golden_osnet."""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *

ws, scene, ref = load_scene_and_reference()
frames = {c.camera: ref.labelled_frames[c.index] for c in scene.cameras}
print({k: len(v) for k, v in frames.items()}, flush=True)
t = time.time()
reid.extract_embeddings(ws, cameras=[c.camera for c in scene.cameras], frames=frames, batch=128, out_dir=EMB_DIR)
print("total s", time.time() - t)
