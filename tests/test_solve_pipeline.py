"""The solve pipeline end to end on synthetic scenes (solve/pipeline.py, tracks3d.py, recover.py).

Covers 3D quality (inch-level, NaN where unmeasured, nothing invented for single-camera frames),
human constraints (``Constraints``), the 3D-first refinement (merge_colocated / projection_attach),
and identity linking across a gap. Ground truth comes from the simulation.
"""

import numpy as np
import pytest

from synthetic_scene import (HAVE_CALIBRATION, SceneBuilder, assert_pure, load_cams, stationary, visible_cameras,
                             walking)

from data_engine.config import RECIPE
from data_engine.solve import recover
from data_engine.solve.associate import associate
from data_engine.solve.evidence import PairSamples, pair_samples
from data_engine.solve.pipeline import Constraints, coverage_metrics, solve
from data_engine.solve.tracks3d import member_table, triangulate_labels

pytestmark = pytest.mark.skipif(not HAVE_CALIBRATION, reason="standard calibration not available")
T = 60
quiet = dict(log=lambda *args: None)


def keep_pairs(samples: PairSamples, allowed: set[tuple[int, int]]) -> PairSamples:
    """Evidence restricted to some camera pairs (simulates people only linkable through those pairs)."""
    keep = np.array([(int(a), int(b)) in allowed or (int(b), int(a)) in allowed
                     for a, b in zip(samples.cam_a, samples.cam_b)], dtype=bool)
    return PairSamples(*[getattr(samples, name)[keep] for name in
                         ("cam_a", "cam_b", "row_a", "row_b", "t", "u", "plausible")])


def label_of(solved, sim, pid, cam, t):
    c, r = sim.endpoint(pid, cam, t)
    return int(solved.labels.label[c][r])


def ground_truth_labels(sim):
    """Labels = simulated person, atoms = tracklets (bypasses association to set up recover.py cases exactly)."""
    label = [np.where(c.kept, sim.gt[c.index], -1).astype(np.int32) for c in sim.scene.cameras]
    atom = [c.tracklet.copy() for c in sim.scene.cameras]
    return recover.Labels(label, atom)


def recover_labels(sim, association):
    return recover.Labels(association.row_component(sim.scene), association.row_atom(sim.scene))


# --------------------------------------------------------------------------- 3D quality


def test_3d_pose_matches_simulation_unmeasured_joints_are_nan_and_single_view_frames_are_empty():
    sim = SceneBuilder(T, seed=3, noise_px=1.0)
    p = sim.add_person(walking((-100, -50), (60, 40), T))
    sim.see(p, ["0"], 0, 20)  # frames 0-19: only camera 0 sees the person
    sim.see(p, ["0", "1", "2"], 20, T)
    sim.low_peak(p, "1", [15, 16]), sim.low_peak(p, "2", [15, 16])  # ankles reliable in camera 0 only
    scene = sim.build()
    solved = solve(scene.scene, RECIPE, **quiet)
    frames = solved.frames
    assert len(frames) == T and not scene.mixed(solved.labels.label)
    truth = scene.truth[p]
    errors = []
    for i, t in enumerate(frames.t):
        measured = np.isfinite(frames.xyz[i, :, 0])
        if t < 20:  # one camera: no invention
            assert not measured.any() and not np.isfinite(frames.root[i]).any() and (frames.used[i] == 0).all()
            continue
        assert measured[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]].all()
        assert not measured[[15, 16]].any(), "ankles seen reliably by one camera only must stay NaN"
        errors.append(np.linalg.norm(frames.xyz[i][measured] - truth[t][measured], axis=1))
        hips = truth[t][[11, 12]].mean(axis=0)
        assert np.linalg.norm(frames.root[i] - hips) < 3.0
        assert (frames.residual[i][measured] < 3.0).all()
    errors = np.concatenate(errors)
    assert np.median(errors) < 1.5 and errors.max() < 4.0


def test_single_camera_person_gets_no_3d():
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(stationary((-100, -50), T))
    sim.see(p, ["0"])
    scene = sim.build()
    solved = solve(scene.scene, RECIPE, **quiet)
    assert np.isnan(solved.frames.xyz).all() and np.isnan(solved.frames.root).all()
    assert coverage_metrics(solved)["person_frames_with_3d_root"] == 0


def test_ray_angle_gate_of_the_recipe_is_honoured():
    """Same detections, two cameras: measured at the recipe's 8 degrees, NaN when the gate is above the actual angle."""
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(stationary((-100, -50), T))
    sim.see(p, ["0", "1"])
    scene = sim.build()
    labels = [np.where(c.kept, 0, -1).astype(np.int32) for c in scene.scene.cameras]
    frames, _ = triangulate_labels(scene.scene, labels, RECIPE)
    assert np.isfinite(frames.xyz[:, 5:, 0]).all()
    ray0 = scene.scene.cameras[0].rays[frames.rows[0, 0]][5]
    ray1 = scene.scene.cameras[1].rays[frames.rows[0, 1]][5]
    angle = float(np.degrees(np.arccos(ray0 @ ray1)))
    assert 8.0 < angle < 175.0
    strict = {**RECIPE, "geometry": {**RECIPE["geometry"], "min_ray_angle_deg": angle + 5.0}}
    frames, _ = triangulate_labels(scene.scene, labels, strict)
    assert np.isnan(frames.xyz[:, 5:, 0]).all() and np.isnan(frames.root).all()


def test_member_table_reports_two_detections_of_one_camera_as_duplicate_and_keeps_the_stronger():
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(stationary((-100, -50), T))
    d = sim.add_person(stationary((-98, -50), T, phase0=0.1))
    sim.see(p, ["0", "1"]), sim.see(d, ["0"])
    scene = sim.build()
    labels = [np.where(c.kept, 0, -1).astype(np.int32) for c in scene.scene.cameras]  # everything one label
    label, t, table, duplicates = member_table(scene.scene, labels)
    assert len(duplicates) == T and all(camera == 0 for camera, _, _ in duplicates)
    assert (table[:, 0] >= 0).all()


# --------------------------------------------------------------------------- human constraints


def one_person_three_cameras():
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(stationary((-100, -50), T))
    sim.see(p, ["0", "1", "2"])
    return sim.build(), p


def test_different_keeps_two_detections_the_machine_would_merge_apart():
    scene, p = one_person_three_cameras()
    machine = solve(scene.scene, RECIPE, **quiet)
    assert len({label_of(machine, scene, p, c, 30) for c in "012"}) == 1, "setup: machine merges all three"
    a, b = scene.endpoint(p, "0", 30), scene.endpoint(p, "1", 30)
    solved = solve(scene.scene, RECIPE, constraints=Constraints(different=[([a], [b])]), **quiet)
    assert label_of(solved, scene, p, "0", 30) != label_of(solved, scene, p, "1", 30)
    # the third view may join one side, never both
    # The third view is consistent with both sides, which a human said are different people:
    # choosing one would be arbitrary, so it joins at most one (it may stay open).
    both = {label_of(solved, scene, p, "2", 30)} & {label_of(solved, scene, p, "0", 30), label_of(solved, scene, p, "1", 30)}
    assert len(both) <= 1
    for component in solved.labels.members().values():
        assert not ({0, 1} <= component), "no component may hold both atoms of the cannot-link pair"


def test_same_joins_pieces_the_machine_left_apart():
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(stationary((-100, -50), T))
    sim.see(p, ["0"]), sim.see(p, ["1"], 10, 16)  # only 6 shared frames: below the 10-sample minimum
    scene = sim.build()
    machine = solve(scene.scene, RECIPE, **quiet)
    assert label_of(machine, scene, p, "0", 12) != label_of(machine, scene, p, "1", 12)
    a, b = scene.endpoint(p, "0", 12), scene.endpoint(p, "1", 12)
    solved = solve(scene.scene, RECIPE, constraints=Constraints(same=[([a], [b])]), **quiet)
    assert label_of(solved, scene, p, "0", 12) == label_of(solved, scene, p, "1", 12)
    assert {"a": 0, "b": 1, "source": "human"} in solved.association.solution.accepted


def test_same_that_contradicts_different_is_not_forced():
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(stationary((-100, -50), T))
    sim.see(p, ["0"]), sim.see(p, ["1"], 10, 16)
    scene = sim.build()
    a, b = scene.endpoint(p, "0", 12), scene.endpoint(p, "1", 12)
    solved = solve(scene.scene, RECIPE, constraints=Constraints(same=[([a], [b])], different=[([a], [b])]), **quiet)
    assert label_of(solved, scene, p, "0", 12) != label_of(solved, scene, p, "1", 12)
    assert [b["reason"] for b in solved.association.solution.blocked if b.get("source") == "human"] == [
        "human same vs contradicted"]


def test_different_people_forced_same_by_a_human_are_joined_only_if_no_conflict():
    """A human 'same' between two atoms of one camera that are visible together is refused (duplicate is the way)."""
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(stationary((-100, -50), T))
    q = sim.add_person(stationary((150, 0), T, phase0=2))
    sim.see(p, ["0", "1", "2"]), sim.see(q, ["0", "1", "2"])
    scene = sim.build()
    a, b = scene.endpoint(p, "0", 30), scene.endpoint(q, "0", 30)
    solved = solve(scene.scene, RECIPE, constraints=Constraints(same=[([a], [b])]), **quiet)
    assert label_of(solved, scene, p, "0", 30) != label_of(solved, scene, q, "0", 30)
    assert_pure(scene, solved.labels.label)
    assert any(e.get("reason") == "human same vs same_camera_overlap" for e in solved.association.solution.blocked)


def test_duplicate_answer_lets_two_poses_of_one_camera_share_a_person_and_reports_them():
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(stationary((-100, -50), T))
    d = sim.add_person(stationary((-98, -50), T, phase0=0.1))  # a second, near identical pose in camera 0
    sim.see(p, ["0", "1", "2"]), sim.see(d, ["0"])
    scene = sim.build()
    machine = solve(scene.scene, RECIPE, **quiet)
    assert label_of(machine, scene, p, "0", 30) != label_of(machine, scene, d, "0", 30)
    a, b = scene.endpoint(p, "0", 30), scene.endpoint(d, "0", 30)
    answer = Constraints(same=[([a], [b])], duplicate=[([a], [b])])  # what stages/solve.load_constraints builds
    solved = solve(scene.scene, RECIPE, constraints=answer, **quiet)
    assert label_of(solved, scene, p, "0", 30) == label_of(solved, scene, d, "0", 30) == label_of(solved, scene, p, "1", 30)
    assert len(solved.duplicates) == T and {c for c, _, _ in solved.duplicates} == {0}


def test_not_person_rows_that_match_nobody_end_up_in_no_component():
    """A false detection (a different, single-view 'person') rejected by a human stays out of every component."""
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(stationary((-100, -50), T))
    ghost = sim.add_person(stationary((150, 100), T, phase0=2))  # far from P: no projection of P matches it
    sim.see(p, ["0", "1", "2"]), sim.see(ghost, ["3"])
    scene = sim.build()
    rows = scene.rows(ghost, "3")
    assert (solve(scene.scene, RECIPE, **quiet).labels.label[3][rows] >= 0).all(), "setup: it is a track by default"
    solved = solve(scene.scene, RECIPE, constraints=Constraints(not_person=[(3, int(r)) for r in rows]), **quiet)
    assert (solved.labels.label[3][rows] == -1).all() and (solved.labels.atom[3][rows] == -1).all()
    assert (solved.frames.rows[:, 3] < 0).all()
    assert label_of(solved, scene, p, "0", 30) == label_of(solved, scene, p, "2", 30)


def test_not_person_rows_are_not_re_attached_by_projection():
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(stationary((-100, -50), T))
    sim.see(p, ["0", "1", "2", "3"])
    scene = sim.build()
    rows = scene.rows(p, "3")
    solved = solve(scene.scene, RECIPE, constraints=Constraints(not_person=[(3, int(r)) for r in rows]), **quiet)
    assert (solved.labels.label[3][rows] == -1).all()


def test_human_cuts_split_a_tracklet_in_solve():
    n = 120
    sim = SceneBuilder(n, seed=1)
    p = sim.add_person(stationary((-100, -50), n))
    q = sim.add_person(stationary((150, 0), n, phase0=2))
    sim.see(p, ["1", "2"]), sim.see(q, ["3", "5"])
    sim.see(p, ["0"], 0, 60), sim.see(q, ["0"], 60, n)
    sim.tracklet("0", [(p, 0, 60), (q, 60, n)])
    scene = sim.build()
    tracklet = scene.tracklet_of(p, "0")
    solved = solve(scene.scene, RECIPE, constraints=Constraints(cuts={tracklet: [3]}), **quiet)
    assert_pure(scene, solved.labels.label)
    assert solved.association.atoms.count == len(scene.tracklet_ids) + 1
    assert scene.report(solved.labels.label)["fragments"] == {0: 1, 1: 1}


# --------------------------------------------------------------------------- refinement (recover.py)


def test_merge_colocated_merges_two_components_of_one_person_seen_by_disjoint_camera_pairs():
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(stationary((-100, -50), T))
    sim.see(p, ["0", "1", "2", "3"])
    scene = sim.build()
    samples = keep_pairs(pair_samples(scene.scene, RECIPE), {(0, 1), (2, 3)})
    association = associate(scene.scene, RECIPE, samples=samples)
    assert association.n_components == 2, "setup: evidence links only (0,1) and (2,3)"
    labels = recover_labels(scene, association)
    frames, _ = triangulate_labels(scene.scene, labels.label, RECIPE)
    merged = recover.merge_colocated(scene.scene, labels, frames)
    assert [m["merged"] for m in merged] == [True] and merged[0]["median_in"] < 3.0
    assert len({int(l) for c in scene.scene.cameras for l in labels.label[c.index][c.order]}) == 1


def test_merge_colocated_leaves_people_who_are_apart_alone():
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(stationary((-100, -50), T))
    q = sim.add_person(stationary((-100, 10), T, phase0=2))  # 60 in apart: beyond radius 10 / 3 * p80 18
    for pid in (p, q):
        sim.see(pid, ["0", "1", "2", "3"])
    scene = sim.build()
    samples = keep_pairs(pair_samples(scene.scene, RECIPE), {(0, 1), (2, 3)})
    association = associate(scene.scene, RECIPE, samples=samples)
    labels = recover_labels(scene, association)
    frames, _ = triangulate_labels(scene.scene, labels.label, RECIPE)
    before = scene.report(labels.label)["components"]
    assert recover.merge_colocated(scene.scene, labels, frames) == []
    assert scene.report(labels.label)["components"] == before
    assert_pure(scene, labels.label)


def test_merge_colocated_refuses_when_one_camera_shows_both_at_the_same_time():
    """Two comps measured at one spot but sharing a camera at the same instant: a duplicate-pose question, not a merge."""
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(stationary((-100, -50), T))
    d = sim.add_person(stationary((-99, -50), T, phase0=0.1))
    sim.see(p, ["0", "1", "2"]), sim.see(d, ["0", "3", "5"])
    scene = sim.build()
    labels = ground_truth_labels(scene)
    frames, _ = triangulate_labels(scene.scene, labels.label, RECIPE)
    merged = recover.merge_colocated(scene.scene, labels, frames)
    assert len(merged) == 1 and merged[0]["merged"] is False and merged[0]["reason"] == "same_camera_overlap"
    assert scene.report(labels.label)["components"] == 2


def test_merge_colocated_honours_human_different():
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(stationary((-100, -50), T))
    sim.see(p, ["0", "1", "2", "3"])
    scene = sim.build()
    samples = keep_pairs(pair_samples(scene.scene, RECIPE), {(0, 1), (2, 3)})
    association = associate(scene.scene, RECIPE, samples=samples)
    labels = recover_labels(scene, association)
    atoms = sorted({int(a) for a in np.concatenate([x[x >= 0] for x in labels.atom])})
    labels.forbidden = {(atoms[0], atoms[-1])}  # one atom of each component
    frames, _ = triangulate_labels(scene.scene, labels.label, RECIPE)
    [refused] = recover.merge_colocated(scene.scene, labels, frames)
    assert refused["merged"] is False and refused["reason"] == "human_different"


def test_merge_colocated_ignores_too_short_coincidence():
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(stationary((-100, -50), T))
    sim.see(p, ["0", "1"], 0, 8), sim.see(p, ["2", "3"], 0, 8)
    scene = sim.build()
    samples = keep_pairs(pair_samples(scene.scene, RECIPE), {(0, 1), (2, 3)})
    association = associate(scene.scene, RECIPE, samples=samples, )
    labels = recover_labels(scene, association)
    frames, _ = triangulate_labels(scene.scene, labels.label, RECIPE)
    # Fewer than 10 coinciding frames: min_frames guards against a coincidence.
    assert recover.merge_colocated(scene.scene, labels, frames) == []


def test_projection_attach_attaches_a_single_camera_tracklet_to_the_right_person():
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(stationary((-100, -50), T))
    q = sim.add_person(stationary((150, 0), T, phase0=2))
    sim.see(p, ["0", "1", "2", "3"]), sim.see(q, ["0", "1", "2"])
    scene = sim.build()
    samples = keep_pairs(pair_samples(scene.scene, RECIPE), {(0, 1), (0, 2), (1, 2)})  # camera 3 stays free
    association = associate(scene.scene, RECIPE, samples=samples)
    labels = recover_labels(scene, association)
    free_rows = scene.rows(p, "3")
    assert (labels.label[3][free_rows] != labels.label[0][scene.rows(p, "0")][0]).all()
    frames, _ = triangulate_labels(scene.scene, labels.label, RECIPE)
    result = recover.projection_attach(scene.scene, labels, frames)
    assert [a["camera"] for a in result["atoms_attached"]] == [3] and not result["atoms_ambiguous"]
    assert set(labels.label[3][free_rows]) == {labels.label[0][scene.rows(p, "0")][0]}
    assert (labels.source[3][free_rows] == recover.Labels.SOURCE["projection_atom"]).all()
    assert_pure(scene, labels.label)


def test_projection_attach_refuses_when_two_people_projections_both_match():
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(stationary((-100, -50), T))
    q = sim.add_person(stationary((-92, -50), T, phase0=2))  # 8 in away: inside the 20 px / 6 % gate from camera 3
    sim.see(p, ["0", "1", "2", "3"]), sim.see(q, ["0", "1", "2"])
    scene = sim.build()
    samples = keep_pairs(pair_samples(scene.scene, RECIPE), {(0, 1), (0, 2), (1, 2)})
    association = associate(scene.scene, RECIPE, samples=samples)
    labels = recover_labels(scene, association)
    frames, _ = triangulate_labels(scene.scene, labels.label, RECIPE)
    before = labels.label[3].copy()
    result = recover.projection_attach(scene.scene, labels, frames)
    assert not result["atoms_attached"]
    [ambiguous] = [a for a in result["atoms_ambiguous"] if a["camera"] == 3]
    assert len(ambiguous["options"]) >= 2 and all(o["matches"] >= 5 for o in ambiguous["options"][:2])
    assert (labels.label[3] == before).all()


def test_projection_attach_never_puts_two_simultaneous_camera_pieces_in_one_person():
    """P and a near-identical duplicate pose D are both free pieces of camera 0: at most one may join P's view set."""
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(stationary((-100, -50), T))
    d = sim.add_person(stationary((-100, -49), T, phase0=0.1))
    sim.see(p, ["0", "1", "2"]), sim.see(d, ["0"])
    scene = sim.build()
    samples = keep_pairs(pair_samples(scene.scene, RECIPE), {(0, 1), (0, 2), (1, 2)})
    association = associate(scene.scene, RECIPE, samples=samples)
    labels = recover_labels(scene, association)
    frames, _ = triangulate_labels(scene.scene, labels.label, RECIPE)
    result = recover.projection_attach(scene.scene, labels, frames)
    assert len(result["atoms_attached"]) == 1
    assert [a["reason"] for a in result["atoms_ambiguous"]] == ["same_camera_overlap"]
    camera0 = labels.label[0]
    both = [camera0[scene.rows(pid, "0")[0]] for pid in (p, d)]
    assert both[0] != both[1], "the two poses of camera 0 must not share a component"


def test_full_solve_attaches_short_fragments_that_pairwise_evidence_cannot_link():
    """Pieces shorter than min_shared_frames (10) cannot be linked pairwise; 3D-first refinement attaches them."""
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(walking((-100, -50), (50, 50), T))
    sim.see(p, ["0"], fragment=6)  # camera 0 is cut into 6-frame tracklets
    sim.see(p, ["1", "2", "3"])
    scene = sim.build()
    association = associate(scene.scene, RECIPE)
    assert scene.report(association.row_component(scene.scene))["fragments"][0] > 1, "setup: pairwise leaves pieces"
    solved = solve(scene.scene, RECIPE, **quiet)
    assert scene.report(solved.labels.label)["fragments"] == {0: 1}
    assert not scene.mixed(solved.labels.label)
    assert sum(e.get("atoms_attached", 0) for e in solved.log) >= 1


# --------------------------------------------------------------------------- identity across a gap (end to end)


def gap_scene(second_person=None, gap=(30, 50), n=80):
    sim = SceneBuilder(n, seed=1)
    p = sim.add_person(stationary((-100, -50), n))
    people = [p]
    if second_person is not None:
        people.append(sim.add_person(stationary(second_person, n, phase0=2)))
    for pid in people:
        sim.see(pid, ["0", "1", "2"], 0, gap[0])
        sim.see(pid, ["0", "1", "2"], gap[1], n)
    return sim.build(), people


def test_person_lost_by_every_camera_for_two_seconds_gets_one_identity():
    scene, [p] = gap_scene()
    solved = solve(scene.scene, RECIPE, **quiet)
    first, second = label_of(solved, scene, p, "0", 10), label_of(solved, scene, p, "0", 70)
    assert first != second, "two components: the person was lost in between"
    assert solved.identity_of_label[first] == solved.identity_of_label[second] >= 0
    assert [(l["from"], l["to"], l["source"]) for l in solved.links.accepted] == [
        (first, second, "machine:unique_motion_continuity")]
    assert not solved.links.open_ends


def test_two_people_within_the_automatic_radius_are_not_linked_across_the_gap():
    scene, [p, q] = gap_scene(second_person=(-80, -50))  # 20 in = 0.5 m apart
    solved = solve(scene.scene, RECIPE, **quiet)
    assert_pure(scene, solved.labels.label)
    assert not solved.links.accepted, "four candidates inside 0.75 m: nothing may be linked automatically"
    ids = {solved.identity_of_label[label_of(solved, scene, pid, "0", t)] for pid in (p, q) for t in (10, 70)}
    assert len(ids) == 4
    starts = [e for e in solved.links.open_ends if e["end"] == "start"]
    assert len(starts) == 2 and all(len(e["candidates"]) == 2 for e in starts)


# --------------------------------------------------------------------------- bookkeeping


def test_coverage_metrics_on_a_clean_scene():
    scene, p = one_person_three_cameras()
    solved = solve(scene.scene, RECIPE, **quiet)
    metrics = coverage_metrics(solved)
    assert metrics["share_detections_in_3d_tracks"] == 1.0 and metrics["identities"] == 1
    assert metrics["person_frames"] == metrics["person_frames_with_3d_root"] == T
    assert metrics["reprojection_px_p90"] < 3.0
