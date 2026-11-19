"""Score a solution against reference identity labels (the golden set).

The reference is a data_management label-set pair: ``two_d_tracklets`` (detections with a
camera-local tracklet id on labelled native frames) and ``three_d_tracks`` (which tracklets
belong to which person). Engine detections are matched to reference detections by box
overlap on the same native frame, which gives every matched engine detection a reference
person. From that:

* **purity** - engine labels must not mix reference people (a mixed label is a wrong merge:
  the costly error);
* **fragmentation** - how many engine labels one reference person is spread over (each
  extra fragment is a question a human or a better machine rule still has to answer);
* **coverage** - share of reference observations that the engine put into any 3D track;
* **human load** - the unknowns the engine reports (what it would ask).

The reference is used only for scoring. Nothing from it enters the solver.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .solve.scene import Scene


@dataclass
class Reference:
    """Per camera: reference person of every engine detection row (-1 = not a reference observation)."""
    person: list[np.ndarray]
    labelled_frames: list[np.ndarray]  # per camera: native frames that carry reference labels
    observations: int  # reference observations in scope
    matched: int  # of which an engine detection overlaps
    people: int


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-9)


def load_reference(scene: Scene, store_root: Path, dataset: str, tag: str = "cvat_identity", *,
                   min_iou: float = 0.5) -> Reference:
    data = Path(store_root) / "data"
    tracks = json.loads((data / f"{dataset}.three_d_tracks.{tag}" / "tracks.json").read_text(encoding="utf-8"))
    owner: dict[tuple[str, int], list[tuple[int, int, int]]] = defaultdict(list)
    for track in tracks["tracks"]:
        for member in track["members"]:
            first, last = member["frame_range"]
            owner[(member["camera"], int(member["tracklet_id"]))].append((int(first), int(last), int(track["track_id"])))
    person, labelled = [], []
    observations = matched = 0
    for camera in scene.cameras:
        result = np.full(len(camera.kept), -1, dtype=np.int32)
        path = data / f"{dataset}.two_d_tracklets.{tag}" / "cameras" / f"{camera.camera}.json"
        frames = []
        if path.is_file():
            for entry in json.loads(path.read_text(encoding="utf-8"))["frames"]:
                frame = int(entry["frame"])
                if frame >= camera.detections.frame_count or not camera.detections.covered[frame]:
                    continue
                frames.append(frame)
                boxes, people = [], []
                for detection in entry["detections"]:
                    who = [t for first, last, t in owner.get((camera.camera, int(detection["tracklet_id"])), ())
                           if first <= frame <= last]
                    if len(who) == 1 and detection.get("bbox"):
                        boxes.append(detection["bbox"])
                        people.append(who[0])
                if not boxes:
                    continue
                observations += len(boxes)
                rows = camera.detections.rows(frame)
                if rows.stop == rows.start:
                    continue
                overlap = box_iou_matrix(np.asarray(boxes, dtype=np.float64),
                                         camera.detections.boxes[rows, :4].astype(np.float64))
                # One-to-one, best overlap first.
                while True:
                    index = np.unravel_index(np.argmax(overlap), overlap.shape)
                    if overlap[index] < min_iou:
                        break
                    result[rows.start + index[1]] = people[index[0]]
                    matched += 1
                    overlap[index[0], :] = -1
                    overlap[:, index[1]] = -1
        person.append(result)
        labelled.append(np.array(sorted(frames), dtype=np.int32))
    return Reference(person, labelled, observations, matched, len(tracks["tracks"]))


def score(scene: Scene, reference: Reference, labels: list[np.ndarray], *, min_minor: int = 3) -> dict:
    """Compare engine labels (per camera, per row; -1 = unlabelled) with the reference people."""
    confusion: dict[int, Counter] = defaultdict(Counter)
    covered = total = 0
    for camera in scene.cameras:
        person = reference.person[camera.index]
        rows = np.flatnonzero(person >= 0)
        total += len(rows)
        mine = labels[camera.index][rows]
        covered += int((mine >= 0).sum())
        for label, who in zip(mine[mine >= 0], person[rows][mine >= 0]):
            confusion[int(label)][int(who)] += 1
    impure_rows = 0
    mixed_labels = []
    fragments: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for label, counts in confusion.items():
        (major, major_n), *minor = counts.most_common()
        wrong = sum(n for _, n in minor)
        impure_rows += wrong
        if any(n >= min_minor for _, n in minor):
            mixed_labels.append({"label": label, "people": dict(counts.most_common(4))})
        fragments[major].append((label, major_n))
    per_person = {}
    for who, parts in fragments.items():
        size = sum(n for _, n in parts)
        per_person[who] = {"fragments": len(parts), "largest_share": round(max(n for _, n in parts) / size, 3),
                           "observations": size}
    labelled_people = len(per_person)
    return {
        "reference_observations_matched": total,
        "coverage": round(covered / max(1, total), 4),
        "impure_observations": impure_rows,
        "purity": round(1 - impure_rows / max(1, covered), 5),
        "mixed_labels": len(mixed_labels), "mixed_examples": mixed_labels[:8],
        "labels": len(confusion), "reference_people_seen": labelled_people,
        "extra_fragments": len(confusion) - labelled_people,
        "largest_fragment_share_mean": round(float(np.mean([p["largest_share"] for p in per_person.values()])), 3)
        if per_person else None,
        "per_person": per_person,
    }


def cross_camera_pairs(scene: Scene, reference: Reference, labels: list[np.ndarray]) -> dict:
    """Precision / recall of 'same person' over simultaneous cross-camera pairs of reference observations."""
    true_same = said_same = both = 0
    by_t: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for camera in scene.cameras:
        rows = np.flatnonzero((reference.person[camera.index] >= 0) & (camera.t >= 0))
        for t, who, label in zip(camera.t[rows], reference.person[camera.index][rows], labels[camera.index][rows]):
            by_t[int(t)].append((camera.index, int(who), int(label)))
    for items in by_t.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i][0] == items[j][0]:
                    continue
                same_person = items[i][1] == items[j][1]
                same_label = items[i][2] >= 0 and items[i][2] == items[j][2]
                true_same += same_person
                said_same += same_label
                both += same_person and same_label
    return {"pairs_same_in_reference": true_same, "pairs_same_in_engine": said_same,
            "precision": round(both / max(1, said_same), 5), "recall": round(both / max(1, true_same), 4),
            "wrong_pairs": said_same - both}
