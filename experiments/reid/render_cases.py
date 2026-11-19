"""Render query | predicted | true crops for held-out wrong decisions (primary model = fast-reid) to cases/*.jpg."""
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from lib import *

EMB = SCRATCH / "golden_fastreid"
D = load_data(emb_dir=EMB)
meta = json.loads((Path(__file__).parent / "cases" / "cases_fastreid.json").read_text())
G = make_gallery(D, "none")
cases = {"B1_all": build_b1(D, G, scope="all"), "B1_cam": build_b1(D, G, scope="cam"), "B2": build_b2(D, build_fragments(D, det_weights(D, "none")))}
cases["B3"] = None


def crop(i, label):
    c = D.scene.cameras[D.cam[i]]
    img = D.ws.frames.read(D.cameras[D.cam[i]], D.frame(i))
    x0, y0, x1, y1 = [int(v) for v in D.box[i, :4]]
    h, w = img.shape[:2]
    c_ = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    c_ = cv2.resize(c_, (128, 256))
    cv2.putText(c_, label, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return c_


def exemplar(person, q, t0):
    m = np.flatnonzero((D.person == person) & (D.t < t0))
    return m[np.argmax(D.emb[m] @ D.emb[q])] if len(m) else None


done = set()
for c in meta:
    if c["split"] != "heldout" or c["rule"] != "strict" or c["usecase"] not in ("B1_all", "B1_cam", "B2"):
        continue
    key = (c["usecase"], c["query_key"])
    if key in done:
        continue
    done.add(key)
    C = cases[c["usecase"]]
    q = c["query_det_index"]
    t0 = D.t[q]
    tiles = [crop(q, f"Q P{c['true_person']}")]
    if c["usecase"] == "B2":
        m = next(x for x in C.meta if x["query"] == q)
        tiles += [crop(m["cand_first"][c["pred_col"]], f"pred P{c['predicted_person']}"), crop(m["cand_first"][c["true_col"]], f"true P{c['true_person']}")]
    else:
        for p, tag in ((c["predicted_person"], "pred"), (c["true_person"], "true")):
            e = exemplar(p, q, t0)
            tiles.append(crop(e, f"{tag} P{p}") if e is not None else np.zeros((256, 128, 3), np.uint8))
    name = f"{c['usecase']}_{c['query_key'].replace('/', '_')}.jpg"
    cv2.imwrite(str(Path(__file__).parent / "cases" / name), np.hstack(tiles))
    print(name, c["time_s"], c["best_sim"], c["gap_to_second"])
