"""Second candidate model (fast-reid SBS R50-NL, MSMT17): embeds the same golden rows into _scratch/reid/golden_fastreid."""
import sys, time, hashlib
import cv2
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import fastreid_model


def prepare(image_bgr, box):
    h, w = image_bgr.shape[:2]
    x0, y0, x1, y1 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1]))), min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
    if x1 - x0 < 4 or y1 - y0 < 8:
        return None
    crop = cv2.resize(image_bgr[y0:y1, x0:x1, ::-1], (128, 384), interpolation=cv2.INTER_LINEAR)  # RGB, 0..255
    return np.ascontiguousarray(crop.transpose(2, 0, 1).astype(np.float32))


if __name__ == "__main__":
    ws, scene, ref = load_scene_and_reference()
    frames = {c.camera: ref.labelled_frames[c.index] for c in scene.cameras}
    model = fastreid_model.build("cuda")
    meta = {"name": "fastreid_sbs_R50_NL_msmt17", "weights_sha256": hashlib.sha256(open(fastreid_model.WEIGHTS, "rb").read()).hexdigest(),
            "input_hw": [384, 128]}
    t = time.time()
    reid.extract_embeddings(ws, cameras=[c.camera for c in scene.cameras], frames=frames, batch=64,
                            out_dir=SCRATCH / "golden_fastreid", backend=(model, meta, prepare, 2048))
    print("total s", time.time() - t)
