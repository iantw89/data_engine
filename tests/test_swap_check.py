"""Identity swap check: where a swap is possible, what the cues say, what gets reviewed."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from data_engine.config import RECIPE
from data_engine.solve import swap_check

RECIPE_SWAP = RECIPE["swap_check"]
M = 1 / 0.0254  # inches per metre
T = 400


def skeleton(root_m: np.ndarray, scale: float) -> np.ndarray:
    """A standing COCO skeleton (inches) at ``root_m`` (metres, hip centre), limb lengths times ``scale``."""
    points = np.full((17, 3), np.nan)
    hip_z = root_m[2]
    offsets = {5: (-0.20, 0, 0.55), 6: (0.20, 0, 0.55), 7: (-0.22, 0, 0.27), 8: (0.22, 0, 0.27),
               9: (-0.22, 0, 0.02), 10: (0.22, 0, 0.02), 11: (-0.13, 0, 0), 12: (0.13, 0, 0),
               13: (-0.13, 0, -0.45), 14: (0.13, 0, -0.45), 15: (-0.13, 0, -0.88), 16: (0.13, 0, -0.88)}
    for joint, (dx, dy, dz) in offsets.items():
        points[joint] = (root_m[0] + dx * scale, root_m[1] + dy * scale, hip_z + dz * scale)
    return points * M


class World:
    """Two (or more) people with given paths; one camera row per person per frame."""

    def __init__(self, paths: dict[int, np.ndarray], scales: dict[int, float], cameras: int = 2,
                 present: dict[int, np.ndarray] | None = None):
        self.paths, self.scales = paths, scales
        self.present = present or {who: np.ones(T, bool) for who in paths}
        people = sorted(paths)
        self.tr_identity, self.tr_t, self.tr_root, self.tr_xyz = [], [], [], []
        for who in people:
            for t in np.flatnonzero(self.present[who]):
                self.tr_identity.append(who)
                self.tr_t.append(t)
                self.tr_root.append(paths[who][t] * M)
                self.tr_xyz.append(skeleton(paths[who][t], scales[who]))
        self.tr_identity, self.tr_t = np.array(self.tr_identity), np.array(self.tr_t)
        self.tr_root, self.tr_xyz = np.array(self.tr_root, np.float32), np.array(self.tr_xyz, np.float32)
        self.cameras, self.row_identity = [], []
        for index in range(cameras):
            times, who_of_row, boxes, tracklet = [], [], [], []
            for slot, who in enumerate(people):
                for t in np.flatnonzero(self.present[who]):
                    times.append(t)
                    who_of_row.append(who)
                    # people are far apart in the image (separated view); the 2D tracker follows the person
                    boxes.append([400 * slot, 100, 400 * slot + 100, 400, 0.9])
                    tracklet.append(slot)
            boxes = np.array(boxes, np.float32)
            detections = SimpleNamespace(boxes=boxes, key=lambda row, c=index: f"cam{c}/{row}/0")
            self.cameras.append(SimpleNamespace(index=index, camera=f"cam{index}", t=np.array(times),
                                                tracklet=np.array(tracklet), detections=detections))
            self.row_identity.append(np.array(who_of_row))
        self.scene = SimpleNamespace(cameras=self.cameras, T=T,
                                     camera_index=lambda name: int(name[3:]))

    def exchange_labels(self, a: int, b: int, tau: int) -> None:
        """Relabel as if the tracker had exchanged a and b from ``tau`` on (the 2D tracklets keep following people)."""
        later = self.tr_t >= tau
        is_a, is_b = (self.tr_identity == a) & later, (self.tr_identity == b) & later
        self.tr_identity[is_a], self.tr_identity[is_b] = b, a
        for camera, identity in zip(self.cameras, self.row_identity):
            later = camera.t >= tau
            is_a, is_b = (identity == a) & later, (identity == b) & later
            identity[is_a], identity[is_b] = b, a

    def check(self) -> dict:
        return swap_check.check(self.scene, self.row_identity, self.tr_identity, self.tr_t, self.tr_root, self.tr_xyz,
                                RECIPE_SWAP)


def line(start, end) -> np.ndarray:
    return np.linspace(np.array(start, float), np.array(end, float), T)


def passing_world() -> World:
    """1 walks +x, 2 walks -x on lanes 0.2 m apart: they pass each other in the middle."""
    return World({1: line((-10, 0.0, 1.0), (10, 0.0, 1.0)), 2: line((10, 0.2, 1.0), (-10, 0.2, 1.0))}, {1: 1.0, 2: 1.25})


def test_people_who_stay_apart_need_no_check():
    world = World({1: line((0, 0, 1), (0.3, 0, 1)), 2: line((0.8, 0, 1), (1.1, 0, 1))}, {1: 1.0, 2: 1.2})
    result = world.check()
    assert result["summary"]["moments"] == 0 and result["summary"]["pairs_within_radius"] == 1


def test_a_clean_pass_is_verified_by_independent_cues():
    result = passing_world().check()
    assert result["summary"]["crossing"] == 1
    entry = result["entries"][0]
    assert entry["status"] == "verified" and {"heading", "body"} <= set(entry["stay_cues"])
    assert entry["min_distance_m"] < 0.3 and "reference_keys" not in entry


def test_a_swap_at_the_pass_is_flagged_with_evidence_for_review():
    world = passing_world()
    world.exchange_labels(1, 2, T // 2)  # the tracker exchanged them exactly where they passed
    entry = world.check()["entries"][0]
    assert entry["kind"] == "crossing" and entry["status"] == "flagged"
    assert {"heading", "body", "tracklet"} <= set(entry["swap_cues"])
    assert entry["reference_keys"] and entry["after_a_keys"] and entry["after_b_keys"]


def test_labels_exchanged_while_apart_show_as_an_exchange_of_places():
    world = World({1: line((0, 0, 1), (0.2, 0, 1)), 2: line((0.8, 0, 1), (0.9, 0, 1))}, {1: 1.0, 2: 1.0})
    world.exchange_labels(1, 2, 150)
    result = world.check()
    assert result["summary"]["exchange"] == 1
    entry = result["entries"][0]
    assert entry["status"] == "flagged" and entry["swap_cues"][0] == "exchanged places"
    assert entry["t"][0] <= 150 <= entry["t"][1] + 1


def test_one_person_tracked_twice_is_not_a_swap_risk():
    path = line((0, 0, 1), (2, 0, 1))
    world = World({1: path, 2: path + np.array([0.05, 0, 0])}, {1: 1.0, 2: 1.0})
    result = world.check()
    assert result["summary"]["duplicate_pairs_skipped"] == 1 and result["summary"]["moments"] == 0


def test_a_track_ending_next_to_an_established_one_is_a_moment():
    present = {1: np.ones(T, bool), 2: np.arange(T) < 200}
    world = World({1: line((0, 0, 1), (0, 0, 1)), 2: line((3, 0, 1), (-2.6, 0, 1))}, {1: 1.0, 2: 1.2}, present=present)
    result = world.check()
    endings = [e for e in result["entries"] if e["kind"] == "ending"]
    assert len(endings) == 1 and endings[0]["a"] == 1 and endings[0]["b"] == 2


def test_a_survivor_born_at_the_ending_is_a_handoff_not_a_swap():
    present = {1: np.arange(T) >= 190, 2: np.arange(T) < 200}
    path = line((0, 0, 1), (4, 0, 1))
    world = World({1: path + np.array([0.35, 0, 0]), 2: path}, {1: 1.0, 2: 1.0}, present=present)
    result = world.check()
    assert result["summary"]["handoff_overlaps_skipped"] == 1
    assert not [e for e in result["entries"] if e["kind"] == "ending"]


def test_a_human_confirmation_closes_the_moment():
    world = passing_world()
    world.exchange_labels(1, 2, T // 2)
    camera = world.cameras[0]
    before = int(np.flatnonzero((world.row_identity[0] == 1) & (camera.t == 20))[0])
    after = int(np.flatnonzero((world.row_identity[0] == 1) & (camera.t == 380))[0])
    before_b = int(np.flatnonzero((world.row_identity[0] == 2) & (camera.t == 20))[0])
    after_b = int(np.flatnonzero((world.row_identity[0] == 2) & (camera.t == 380))[0])
    same = [([(0, before)], [(0, after)]), ([(0, before_b)], [(0, after_b)])]
    result = swap_check.check(world.scene, world.row_identity, world.tr_identity, world.tr_t, world.tr_root,
                              world.tr_xyz, RECIPE_SWAP, same_rows=same)
    assert result["entries"][0]["status"] == "verified" and "human" in result["entries"][0]["stay_cues"]


def test_verdict_rules():
    quiet = {"tracklet": {"contradict": 0, "agree": 0}}
    judge = lambda **cues: swap_check.judge({**quiet, **cues}, RECIPE_SWAP)["status"]  # noqa: E731
    assert judge(heading=-2.0, body=-2.0) == "verified"  # two independent cues say stay
    assert judge(body=-2.0) == "unverified"  # one cue is not a certificate
    assert judge() == "unverified"
    assert judge(body=2.0) == "flagged"  # a swap cue with nothing against it
    assert judge(body=2.0, heading=-2.0) == "unverified"  # one for, one against: a person decides
    assert judge(body=2.0, heading=2.0) == "flagged"
    assert swap_check.judge({"heading": -2.0, "body": -2.0, **quiet}, RECIPE_SWAP, exchange=True)["status"] == "flagged"
    assert swap_check.judge({"body": -2.0, "tracklet": {"contradict": 0, "agree": 2}}, RECIPE_SWAP)["status"] == "verified"
    assert swap_check.judge({"tracklet": {"contradict": 2, "agree": 0}}, RECIPE_SWAP)["status"] == "flagged"
