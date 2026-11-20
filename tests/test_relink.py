"""Identity after the coverage recovery: recovered positions move track ends, the solve decides identity
again on its kept association, and the recovery continues from its previous result."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from synthetic_scene import HAVE_CALIBRATION, SceneBuilder, walking

from data_engine.config import RECIPE
from data_engine.solve import coverage
from data_engine.solve.identity import link_components, links_between_identities, summarise
from data_engine.solve.pipeline import frames_with_recovered, solve
from data_engine.solve.tracks3d import PersonFrames
from data_engine.stages import recover as recover_stage
from data_engine.stages import solve as solve_stage

METRE = 1 / 0.0254
T = 400


def frames_of(spec: dict[int, dict]) -> PersonFrames:
    """label -> {t0, t1 (exclusive), xy (metres)}: a person standing at xy, seen by two cameras."""
    label, t, rows, root = [], [], [], []
    for lab, s in spec.items():
        for frame in range(s["t0"], s["t1"]):
            label.append(lab), t.append(frame)
            rows.append([frame, frame, -1])
            root.append([s["xy"][0] * METRE, s["xy"][1] * METRE, 36.0])
    n = len(label)
    return PersonFrames(np.array(label, np.int32), np.array(t, np.int32), np.array(rows, np.int32),
                        np.full((n, 17, 3), np.nan, np.float32), np.zeros((n, 17), np.uint8),
                        np.full((n, 17), np.nan, np.float32), np.array(root, np.float32))


# Someone stands at x = 0, is not measured for 10 s and is then measured 3 m away: too long a gap for the
# automatic rule, and not the same spot.
LOST_WHILE_WALKING = {0: {"t0": 20, "t1": 100, "xy": (0.0, 0.0)}, 1: {"t0": 200, "t1": T, "xy": (3.0, 0.0)}}


def walk(first: int = 100, last: int = 170, component: int = 0) -> dict:
    """Positions the recovery measured along the walk (3 m in 10 s from t = 100)."""
    t = np.arange(first, last)
    root = np.stack([3.0 * (t - 100) / 100.0 * METRE, np.zeros(len(t)), np.full(len(t), 36.0)], axis=1)
    return {"component": np.full(len(t), component), "identity": np.full(len(t), component), "t": t, "root": root}


def pairs(links) -> set[tuple[int, int]]:
    return {(link["from"], link["to"]) for link in links.accepted}


# --------------------------------------------------------------------------- the rule with recovered positions


def test_recovered_positions_make_a_continuation_decidable():
    frames = frames_of(LOST_WHILE_WALKING)
    measured = link_components(summarise(frames, 2), RECIPE, T=T, frames=frames)
    assert not measured.accepted and len(measured.open_ends) == 2
    known = frames_with_recovered(frames, walk(), 2)
    summaries = summarise(known, 2)
    assert summaries[0].last_root_t == 169 and summaries[0].t_last == 169
    links = link_components(summaries, RECIPE, T=T, frames=known)
    assert pairs(links) == {(0, 1)} and links.accepted[0]["gap_s"] == 3.1
    assert links.identity[0] == links.identity[1] and not links.open_ends


def test_a_short_extension_changes_nothing():
    frames = frames_of(LOST_WHILE_WALKING)
    known = frames_with_recovered(frames, walk(last=120), 2)  # 2 s further: still 8 s away
    assert not link_components(summarise(known, 2), RECIPE, T=T, frames=known).accepted


def test_recovered_positions_of_unknown_components_are_ignored():
    frames = frames_of(LOST_WHILE_WALKING)
    known = frames_with_recovered(frames, walk(component=7), 2)
    assert len(known.t) == len(frames.t)
    nowhere = walk()
    nowhere["root"][:, 0] = np.nan
    assert len(frames_with_recovered(frames, nowhere, 2).t) == len(frames.t)


def test_links_accepted_on_less_information_are_kept():
    """More known positions can add a second candidate; a link accepted before stays."""
    spec = {0: {"t0": 20, "t1": 100, "xy": (0.0, 0.0)}, 1: {"t0": 120, "t1": 200, "xy": (0.2, 0.0)}}
    first = link_components(summarise(frames_of(spec), 3), RECIPE, T=T)
    assert pairs(first) == {(0, 1)}
    spec[2] = {"t0": 125, "t1": 200, "xy": (0.3, 0.3)}  # e.g. a start the recovery moved to within reach
    frames = frames_of(spec)
    assert not link_components(summarise(frames, 3), RECIPE, T=T).accepted, "two candidates: ambiguous on its own"
    again = link_components(summarise(frames, 3), RECIPE, T=T, keep=first.accepted)
    assert pairs(again) == {(0, 1)} and again.accepted[0] == first.accepted[0]
    assert again.identity[2] != again.identity[0]


def test_the_recover_stage_predicts_the_links_of_identities():
    frames = frames_of(LOST_WHILE_WALKING)
    identity = np.where(frames.label == 0, 7, 9)
    recovered = walk()
    recovered["identity"] = np.full(len(recovered["t"]), 7)
    links = links_between_identities(identity, frames.t, frames.rows, frames.root, recovered, RECIPE, T=T)
    assert [(link["from"], link["to"]) for link in links] == [(7, 9)]
    nothing = {name: values[:0] for name, values in recovered.items()}
    assert links_between_identities(identity, frames.t, frames.rows, frames.root, nothing, RECIPE, T=T) == []


# --------------------------------------------------------------------------- signatures and components


def test_signatures():
    component, t = np.array([1, 1, 2]), np.array([10, 11, 10])
    root = np.array([[0, 0, 36], [1, 0, 36], [np.nan] * 3], np.float32)
    signature = coverage.measured_signature(component, t, root)
    assert signature == coverage.measured_signature(component.astype(np.int64), t.astype(np.int32), root.copy())
    assert signature != coverage.measured_signature(component, t, root + 1)
    positions = coverage.positions_signature(t, root[:, :3])
    assert positions == coverage.positions_signature(t[::-1], root[::-1]), "the order of the rows does not matter"
    assert positions != coverage.positions_signature(t[:2], root[:2])


def test_nearest_component_of_a_person_in_time():
    identity, t, component = np.array([3, 3, 3, 5]), np.array([10, 20, 30, 10]), np.array([1, 1, 2, 9])
    found = coverage.nearest_component(identity, t, component, np.array([3, 3, 3, 5, 4]), np.array([12, 28, 99, 50, 1]))
    assert found.tolist() == [1, 2, 2, 9, -1]


# --------------------------------------------------------------------------- continuing a recovery


def recovered_arrays(identity, t, cameras=3) -> dict:
    n = len(t)
    used = np.zeros((n, cameras, 17), bool)
    used[:, :2, :5] = True
    return {"identity": np.array(identity, np.int64), "t": np.array(t, np.int64),
            "xyz": np.random.default_rng(0).normal(size=(n, 17, 3)).astype(np.float32).astype(np.float64), "used": used,
            "residual": np.where(used, 1.5, np.nan), "root": np.arange(n * 3, dtype=np.float64).reshape(n, 3),
            "tier": np.array(["A"] * n), "pass": np.arange(n, dtype=np.int16), "hint_kind": np.zeros(n, np.int8),
            "hint_frames": np.ones(n, np.int32), "pose_frame": np.full((n, cameras), 5, np.int64),
            "pose_keypoints": np.zeros((n, cameras, 17, 3)), "pose_box": np.zeros((n, cameras, 4))}


def tracks_of(identity_of_component: dict[int, int]) -> dict:
    """Component 1 measured at t 0..9, component 2 at t 30..39, component 3 at t 0..39 (somebody else)."""
    component = np.array([1] * 10 + [2] * 10 + [3] * 40, np.int32)
    t = np.concatenate([np.arange(10), np.arange(30, 40), np.arange(40)]).astype(np.int32)
    root = np.stack([t * 1.0, component * 100.0, np.full(len(t), 36.0)], axis=1).astype(np.float32)
    return {"component": component, "identity": np.array([identity_of_component[c] for c in component], np.int32),
            "t": t, "root": root, "xyz": np.repeat(root[:, None, :], 17, axis=1), "rows": np.zeros((len(t), 3), np.int32)}


def scene_of(cameras=3):
    return SimpleNamespace(cameras=[SimpleNamespace(camera=f"c{index}") for index in range(cameras)])


def write_recovery(tmp_path, tracks, arrays, rejected=None):
    directory = tmp_path / "rounds" / "r001"
    directory.mkdir(parents=True)
    signatures = {"measured": coverage.measured_signature(tracks["component"], tracks["t"], tracks["root"]),
                  "recipe": "recipe", "positions": coverage.positions_signature(arrays["t"], arrays["root"])}
    recover_stage.save_recovered(directory, scene_of(), arrays, "tracks-sha", tracks, signatures)
    recover_stage.save_rejected(directory, rejected or {}, tracks)
    return signatures


def test_a_recovery_is_continued_when_tracks_were_only_joined(tmp_path):
    then = tracks_of({1: 10, 2: 11, 3: 12})
    arrays = recovered_arrays(identity=[10, 10, 11, 12], t=[10, 11, 29, 50])
    rejected = {(10, 12): {"reason": recover_stage.REASONS[2], "hint_kind": 1, "hint_frames": 3},
                (11, 28): {"reason": recover_stage.REASONS[3], "hint_kind": 2, "hint_frames": 2, "tier_tried": "B",
                           "features": {name: 1.0 for name in recover_stage.FEATURES}, "failed": ["residual", "bone lengths"],
                           "root": np.zeros(3, np.float32), "xyz": np.zeros((17, 3), np.float32)}}
    signatures = write_recovery(tmp_path, then, arrays, rejected)
    with np.load(tmp_path / "rounds" / "r001" / "tracks3d_recovered.npz") as data:
        assert data["component"].tolist() == [1, 1, 2, 3], "each position is tied to its person's nearest component"

    now = tracks_of({1: 4, 2: 4, 3: 5})  # the solve joined components 1 and 2 and renumbered everybody
    seed = recover_stage.load_seed(tmp_path, now, signatures["measured"], "recipe", 3)
    assert seed["changed"] == {4} and seed["dropped"] == 0
    assert seed["arrays"]["identity"].tolist() == [4, 4, 4, 5]
    for name in ("t", "xyz", "used", "root", "tier", "pass", "pose_frame"):
        assert np.array_equal(seed["arrays"][name], arrays[name]), name
    assert np.array_equal(np.isnan(seed["arrays"]["residual"]), np.isnan(arrays["residual"]))
    assert set(seed["rejected"]) == {(4, 12), (4, 28)}
    assert seed["rejected"][(4, 28)]["failed"] == ["residual", "bone lengths"] and seed["rejected"][(4, 28)]["tier_tried"] == "B"
    assert "features" not in seed["rejected"][(4, 12)]

    same = recover_stage.load_seed(tmp_path, then, signatures["measured"], "recipe", 3)
    assert same["changed"] == set() and same["arrays"]["identity"].tolist() == [10, 10, 11, 12]


def test_a_recovery_is_not_continued_on_other_positions_another_recipe_or_split_identities(tmp_path):
    then = tracks_of({1: 10, 2: 10, 3: 12})
    signatures = write_recovery(tmp_path, then, recovered_arrays(identity=[10, 12], t=[15, 50]))
    assert recover_stage.load_seed(tmp_path, then, signatures["measured"], "recipe", 3) is not None
    assert recover_stage.load_seed(tmp_path, then, "other positions", "recipe", 3) is None
    assert recover_stage.load_seed(tmp_path, then, signatures["measured"], "other recipe", 3) is None
    split = tracks_of({1: 10, 2: 11, 3: 12})  # identity 10 of then is two identities now
    assert recover_stage.load_seed(tmp_path, split, signatures["measured"], "recipe", 3) is None
    assert recover_stage.load_seed(tmp_path / "nothing here", then, signatures["measured"], "recipe", 3) is None


def test_a_recovered_position_gives_way_to_a_measured_one_of_the_joined_person(tmp_path):
    then = tracks_of({1: 10, 2: 11, 3: 12})
    # identity 11 was followed back to t = 5 and 20; component 1 (identity 10) is measured at t = 5
    signatures = write_recovery(tmp_path, then, recovered_arrays(identity=[11, 11, 10], t=[5, 20, 20]))
    seed = recover_stage.load_seed(tmp_path, tracks_of({1: 4, 2: 4, 3: 5}), signatures["measured"], "recipe", 3)
    assert seed["arrays"]["t"].tolist() == [20] and seed["dropped"] == 2, "one position per person and instant"


def test_continuing_without_changed_people_reads_nothing():
    tracks = tracks_of({1: 4, 2: 4, 3: 5})
    seed = recovered_arrays(identity=[4, 4], t=[10, 11])
    rejected = {(4, 12): {"reason": recover_stage.REASONS[2], "hint_kind": 1, "hint_frames": 1}}

    def read(plan):
        raise AssertionError("nobody changed: no crop may be read")
    result = coverage.recover_iteratively(
        scene_of(), tracks["identity"].astype(np.int64), tracks["t"].astype(np.int64), tracks["root"], tracks["xyz"],
        read, RECIPE["geometry"], RECIPE["coverage"], log=lambda message: None, seed=seed, seed_rejected=rejected,
        only_identities=set())
    assert result["seeded"] == 2 and result["passes"] == [] and not result["stopped_by_pass_limit"]
    assert np.array_equal(result["arrays"]["t"], seed["t"]) and result["rejected"] == rejected
    assert result["inside_missing_before"] == 20 and result["inside_missing_after"] == 18


# --------------------------------------------------------------------------- the solve's kept association


def test_the_kept_association_is_only_used_for_the_inputs_it_was_made_from(tmp_path):
    messages = []
    context = SimpleNamespace(workspace=SimpleNamespace(path=lambda *parts: tmp_path.joinpath(*parts)),
                              log=lambda message, level="info": messages.append(message))
    assert solve_stage.load_state(context, "key") is None and not messages
    solve_stage.save_state(context, "key", {"anything": np.arange(3)}, {"c0": []})
    associated, alignment = solve_stage.load_state(context, "key")
    assert associated["anything"].tolist() == [0, 1, 2] and alignment == {"c0": []}
    assert solve_stage.load_state(context, "other inputs") is None
    (tmp_path / "solve_state" / "associated.pkl").write_bytes(b"not a pickle")
    assert solve_stage.load_state(context, "key") is None and "unusable" in messages[-1]


# --------------------------------------------------------------------------- applied or proposed (whole solve)


@pytest.mark.skipif(not HAVE_CALIBRATION, reason="standard calibration not available")
def test_links_through_recovered_positions_are_proposed_unless_the_recipe_applies_them():
    """Someone walks 0.8 m while no camera has them for 11 s: no automatic link on the measured positions."""
    n = 200
    sim = SceneBuilder(n, seed=1)
    person = sim.add_person(walking((-100, -50), (-40, -50), n))
    sim.see(person, ["0", "1", "2"], 0, 40)
    sim.see(person, ["0", "1", "2"], 150, n)
    scene = sim.build()
    quiet = dict(log=lambda *args: None)

    def label_at(solved, t):
        camera, row = scene.endpoint(person, "0", t)
        return int(solved.labels.label[camera][row])

    measured = solve(scene.scene, RECIPE, **quiet)
    first, second = label_at(measured, 10), label_at(measured, 170)
    assert first != second and not measured.links.accepted
    assert measured.identity_of_label[first] != measured.identity_of_label[second]

    t = np.arange(40, 120)  # the recovery followed the walk for 8 s: 3 s and 0.2 m short of the second track
    hips = sim.persons[person].xyz[t][:, [11, 12]].mean(axis=1)
    recovered = {"component": np.full(len(t), first), "t": t, "root": hips}

    proposed = solve(scene.scene, {**RECIPE, "relink": {**RECIPE["relink"], "apply": False}}, recovered=recovered, **quiet)
    event = proposed.log[-1]
    assert [(link["from"], link["to"]) for link in event["links_proposed_after_recovery"]] == [(first, second)]
    assert not event["links_after_recovery"] and not proposed.links.accepted
    assert proposed.identity_of_label[first] != proposed.identity_of_label[second], "proposed, not joined"
    assert [end["t"] for end in proposed.links.open_ends] == [end["t"] for end in measured.links.open_ends]

    applied = solve(scene.scene, {**RECIPE, "relink": {**RECIPE["relink"], "apply": True}}, recovered=recovered, **quiet)
    assert applied.identity_of_label[first] == applied.identity_of_label[second] >= 0
    link = applied.links.accepted[0]
    assert link["after_recovery"] and link["end_moved_s"] == 8.0 and link["gap_s"] == 3.1
    assert not applied.log[-1]["links_proposed_after_recovery"]


# --------------------------------------------------------------------------- which candidates exclude each other


def test_two_tracks_measured_at_the_same_time_in_different_places_are_two_people():
    t = np.arange(100, 160)
    here = np.tile([0.0, 0.0, 36.0], (len(t), 1))
    there = here + [3.0 * METRE, 0.0, 0.0]
    apart = solve_stage.measured_apart((t, here), (t[20:], there[20:]))
    assert apart == {"seconds": 4.0, "distance_m": 3.0}
    assert solve_stage.measured_apart((t, here), (t + 100, there)) is None, "one after the other: could be one person"
    assert solve_stage.measured_apart((t, here), (t, here + [0.2 * METRE, 0, 0])) is None, "one place: one person tracked twice"
    assert solve_stage.measured_apart((t, here), (t[:3], there[:3])) is None, "three common instants prove nothing"
