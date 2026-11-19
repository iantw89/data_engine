"""Golden check: does a recovered 2D box overlap the RIGHT person's golden box, and not another person's?

For every accepted recovery (gap and withhold cases) and every camera it used, at instants where the
golden set labels people in that camera: IoU of the recovered detection box with the golden box of the
person the engine identity corresponds to (majority of the identity's golden-matched detections) and
with every other golden person's box.

Usage: python experiments/coverage/golden_check.py [eval_name] [key] [gate_name]
"""
import json
import pickle
import sys
from collections import Counter, defaultdict

from common import *  # noqa
from data_engine import evaluation
from data_engine.config import load_site
from golden_single_view import golden_views
from hints import box_iou
from gates import Context, accept, features


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "vit"
    key = sys.argv[2] if len(sys.argv) > 2 else "R1/crop"
    ws, scene, tracks, solution = load()
    ctx = Context(tracks)
    cases = pickle.loads((SCRATCH / "cases.pkl").read_bytes())
    ev = {e["id"]: e for e in pickle.loads((SCRATCH / f"eval_{name}.pkl").read_bytes())}
    jobs = np.load(SCRATCH / "jobs.npz")
    vit = np.load(SCRATCH / "pass_vit.npz")
    job_of = {(int(c), int(k)): j for j, (c, k) in enumerate(zip(jobs["case"], jobs["cam"]))}
    reference = evaluation.load_reference(scene, load_site().repo("data_management") / "data" / "labels", DATASET)
    # engine identity -> golden person (majority over the identity's measured detections)
    votes = defaultdict(Counter)
    for row in range(len(tracks["identity"])):
        for c in range(len(scene.cameras)):
            r = tracks["rows"][row, c]
            if r >= 0 and reference.person[c][r] >= 0:
                votes[int(tracks["identity"][row])][int(reference.person[c][r])] += 1
    person_of = {i: v.most_common(1)[0][0] for i, v in votes.items()}
    gviews = golden_views(scene, DATASET)
    gbox = defaultdict(list)  # (t, camera) -> [(person, box)]
    for (p, t), seen in gviews.items():
        for c, state, row, box in seen:
            gbox[(t, c)].append((p, np.asarray(box, float)))
    result = defaultdict(Counter)
    for case in cases:
        if case["tag"] not in ("gap", "withhold") or case["id"] not in ev:
            continue
        rec = ev[case["id"]].get(key)
        if rec is None or accept(features(ctx, case["who"], case["t"], rec)) is None:
            continue
        person = person_of.get(case["who"])
        if person is None:
            result[case["tag"]]["identity_without_golden_person"] += 1
            continue
        for (c, row), source in zip(rec["rows"], rec["sources"]):
            labelled = gbox.get((case["t"], c))
            if not labelled:
                continue
            if row >= 0:
                box = scene.cameras[c].detections.boxes[row, :4].astype(float)
            else:
                box = vit["box2"][job_of[(case["id"], c)]].astype(float)
            ious = [(p, float(box_iou(box, g[None])[0])) for p, g in labelled]
            right = max([i for p, i in ious if p == person], default=0.0)
            other = max([i for p, i in ious if p != person], default=0.0)
            has_person = any(p == person for p, _ in labelled)
            if right >= 0.5 and right >= other:
                verdict = "right_person (IoU>=.5 with the golden box of the identity's person)"
            elif other >= 0.5:
                verdict = ("OTHER_person_overlapping_both (IoU>=.3 with the identity's person too)" if right >= 0.3
                           else "OTHER_person (IoU>=.5 with another golden person, <.3 with the identity's)")
            elif has_person:
                verdict = "identity's person labelled, IoU .3-.5 (box convention)" if right >= 0.3 else "identity's person labelled, IoU<.3"
            else:
                verdict = "identity's person not labelled in this view: unverifiable"
            if verdict.startswith("OTHER"):
                result[case["tag"] + "_other_ids"][str(case["id"])] += 1
            result[case["tag"]][f"{source}:{verdict}"] += 1
            result[case["tag"]][verdict] += 1
    out = {tag: dict(c) for tag, c in result.items()}
    print(json.dumps(out, indent=1))
    (SCRATCH / f"golden_check_{name}_{key.replace('/', '_')}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
