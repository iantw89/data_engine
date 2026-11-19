"""Check the recover stage's output of a golden workspace against the golden 2D identity labels.

For every recovered person-frame and every camera it was triangulated from, at golden-labelled
frames: does the box of the pose that was read overlap the golden box of the right person
(the identity's majority golden person), another person's, or nobody's?

    python experiments/coverage/check_recovered.py WORKSPACE GOLDEN_DATASET
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from data_engine.config import RECIPE, SITE_DEFAULTS  # noqa: E402
from data_engine.evaluation import box_iou_matrix, load_reference  # noqa: E402
from data_engine.solve.scene import load_scene  # noqa: E402
from data_engine.workspace import Workspace  # noqa: E402


def main() -> None:
    workspace, dataset = Workspace(sys.argv[1]), sys.argv[2]
    directory = Path(sorted(glob.glob(str(workspace.root / "rounds" / "*" / "tracks3d_recovered.npz")))[-1]).parent
    recovered = dict(np.load(directory / "tracks3d_recovered.npz"))
    solution = dict(np.load(directory / "solution.npz"))
    scene = load_scene(workspace, joint_peak=RECIPE["geometry"]["joint_peak"], rays=False)
    store = Path(SITE_DEFAULTS["repos"]["data_management"]) / "data" / "labels"
    reference = load_reference(scene, store, dataset)
    # majority golden person of every engine identity
    votes = defaultdict(Counter)
    for camera in scene.cameras:
        identity, person = solution[f"identity_{camera.camera}"], reference.person[camera.index]
        both = (identity >= 0) & (person >= 0)
        for who, golden in zip(identity[both], person[both]):
            votes[int(who)][int(golden)] += 1
    person_of = {who: counts.most_common(1)[0][0] for who, counts in votes.items()}
    # golden boxes per (camera, frame)
    data = store / "data"
    owner = defaultdict(list)
    for track in json.loads((data / f"{dataset}.three_d_tracks.cvat_identity" / "tracks.json").read_text(encoding="utf-8"))["tracks"]:
        for member in track["members"]:
            owner[(member["camera"], int(member["tracklet_id"]))].append((*member["frame_range"], int(track["track_id"])))
    golden = {}
    for camera in scene.cameras:
        path = data / f"{dataset}.two_d_tracklets.cvat_identity" / "cameras" / f"{camera.camera}.json"
        for entry in json.loads(path.read_text(encoding="utf-8"))["frames"]:
            boxes, people = [], []
            for detection in entry["detections"]:
                who = [p for first, last, p in owner.get((camera.camera, int(detection["tracklet_id"])), ())
                       if first <= int(entry["frame"]) <= last]
                if len(who) == 1 and detection.get("bbox"):
                    boxes.append(detection["bbox"])
                    people.append(who[0])
            if boxes:
                golden[(camera.index, int(entry["frame"]))] = (np.asarray(boxes, float), people)
    verdicts, by_tier = Counter(), defaultdict(Counter)
    for row in range(len(recovered["t"])):
        who = int(recovered["identity"][row])
        expected = person_of.get(who)
        for camera_index in range(len(scene.cameras)):
            frame = int(recovered["pose_frame"][row, camera_index])
            if frame < 0 or (camera_index, frame) not in golden:
                continue
            boxes, people = golden[(camera_index, frame)]
            overlap = box_iou_matrix(recovered["pose_box"][row, camera_index][None].astype(float), boxes)[0]
            mine = [overlap[i] for i, p in enumerate(people) if p == expected]
            others = [overlap[i] for i, p in enumerate(people) if p != expected]
            if expected is None:
                verdict = "identity has no golden person"
            elif mine and max(mine) >= 0.5:
                verdict = "right person (IoU >= 0.5)"
            elif mine and max(mine) >= 0.3:
                verdict = "right person (IoU 0.3-0.5)"
            elif others and max(others) >= 0.5:
                verdict = "ANOTHER golden person"
            elif mine:
                verdict = "right person labelled elsewhere in the frame (IoU < 0.3)"
            else:
                verdict = "person not labelled in this view (unverifiable)"
            verdicts[verdict] += 1
            by_tier[str(recovered["tier"][row])][verdict] += 1
    print(directory)
    print("recovered positions:", len(recovered["t"]), dict(Counter(str(v) for v in recovered["tier"])))
    for verdict, count in verdicts.most_common():
        print(f"  {count:5d}  {verdict}")
    for name, counts in sorted(by_tier.items()):
        print(f"tier {name}:", dict(counts))


if __name__ == "__main__":
    main()
