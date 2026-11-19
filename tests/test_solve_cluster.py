"""Cross-camera association (solve/evidence.py, solve/cluster.py, solve/associate.py).

Unit tests drive ``cluster.agglomerate`` with hand-made residual histograms (exact control of
supported / ambiguous / eliminated edges); scene tests simulate people, project them through the
standard calibration and check *purity* (a component never holds two simulated people) and
*completeness* (a person ends up in one component) against the simulation's ground truth.
"""

import numpy as np
import pytest

from synthetic_scene import (HAVE_CALIBRATION, SceneBuilder, Trajectory, assert_pure, load_cams, stationary,
                             trap_position, visible_cameras, walking)

from data_engine.config import RECIPE
from data_engine.solve import cluster
from data_engine.solve.associate import associate
from data_engine.solve.evidence import pair_samples

needs_calibration = pytest.mark.skipif(not HAVE_CALIBRATION, reason="standard calibration not available")
T = 60


# --------------------------------------------------------------------------- histogram helpers


def hist(**bins) -> np.ndarray:
    """Residual histogram from ``b0=16, b4=4`` (bin index -> count); bins: u<.5,1,1.5,2,3,4,6,8,more."""
    row = np.zeros(len(cluster.EDGES) + 1, np.int32)
    for name, count in bins.items():
        row[int(name[1:])] = count
    return row


def edges_of(rows: dict[tuple[int, int], np.ndarray]) -> cluster.Histograms:
    keys = sorted(rows)
    return cluster.Histograms(np.array([k[0] for k in keys]), np.array([k[1] for k in keys]),
                              np.zeros(len(keys), np.int32), np.stack([rows[k] for k in keys]))


def components_of(cameras, occupancy=None) -> cluster.Components:
    occupancy = occupancy if occupancy is not None else [np.arange(20)] * len(cameras)
    return cluster.Components(len(cameras), np.asarray(cameras), [np.asarray(o) for o in occupancy])


EMPTY = cluster.Histograms(np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int32),
                           np.empty((0, 9), np.int32))
GOOD = hist(b0=20)  # f1 = f2 = 1, mean_u .25
BAD = hist(b8=20)  # u far outside everything: contradicted
UNSUPPORTED = hist(b5=20)  # not supported (u 3-4) but not contradicted either


def same_component(solution, *atoms):
    return len({int(solution.labels[a]) for a in atoms}) == 1


# --------------------------------------------------------------------------- agglomerate: rules


def test_supported_unambiguous_edges_merge():
    edges = edges_of({(0, 1): GOOD, (2, 3): GOOD})
    solution = cluster.agglomerate(edges, components_of([0, 1, 0, 1]), min_samples=10)
    assert same_component(solution, 0, 1) and same_component(solution, 2, 3) and not same_component(solution, 0, 2)
    assert [e["source"] for e in solution.accepted] == ["machine", "machine"] and not solution.ambiguous


def test_too_few_samples_never_merge():
    edges = edges_of({(0, 1): hist(b0=9)})
    solution = cluster.agglomerate(edges, components_of([0, 1]), min_samples=10)
    assert not same_component(solution, 0, 1) and not solution.accepted


def test_contradicted_edge_forbids_merge_through_a_third_atom():
    # 0-1 contradicted, 0-2 and 1-2 equally supported: 2 can never join both, and choosing one
    # would be arbitrary - both edges stay open and name each other as rivals.
    edges = edges_of({(0, 1): BAD, (0, 2): GOOD, (1, 2): GOOD})
    solution = cluster.agglomerate(edges, components_of([0, 1, 2]), min_samples=10)
    assert not same_component(solution, 0, 1)
    assert not (same_component(solution, 0, 2) and same_component(solution, 1, 2))
    assert {(a["a"], a["b"]) for a in solution.ambiguous if a["reason"] == "rivals"} == {(0, 2), (1, 2)}


def test_pooled_evidence_of_other_views_vetoes_a_supported_edge():
    """Three agreeing views must not hide a fourth that is someone else (N-view consistency)."""
    edges = edges_of({(0, 1): GOOD, (0, 2): GOOD, (1, 2): GOOD,
                      (0, 3): hist(b0=16, b3=4),  # supported but weaker, so it is considered last
                      (1, 3): UNSUPPORTED, (2, 3): UNSUPPORTED})
    solution = cluster.agglomerate(edges, components_of([0, 1, 2, 3]), min_samples=10)
    assert same_component(solution, 0, 1, 2)
    assert not same_component(solution, 0, 3)
    [entry] = [a for a in solution.ambiguous if (a["a"], a["b"]) == (0, 3)]
    assert entry["reason"] == "pooled evidence below support" and entry["pooled_f2"] < 0.8


def test_ambiguous_edge_is_left_open_with_its_rivals_listed():
    # atom 0 (camera 0) is equally consistent with atoms 1 and 2 (both camera 1, visible together).
    edges = edges_of({(0, 1): GOOD, (0, 2): GOOD})
    solution = cluster.agglomerate(edges, components_of([0, 1, 1]), min_samples=10)
    assert not solution.accepted
    assert {(a["a"], a["b"]) for a in solution.ambiguous} == {(0, 1), (0, 2)}
    assert all(a["reason"] == "rivals" and len(a["rivals"]) == 1 for a in solution.ambiguous)


def test_rival_visible_at_other_times_is_no_rival():
    # Atoms 1 and 2 never appear together, so one person can own both: no ambiguity.
    edges = edges_of({(0, 1): GOOD, (0, 2): GOOD})
    comp = components_of([0, 1, 1], [np.arange(0, 40), np.arange(0, 20), np.arange(20, 40)])
    solution = cluster.agglomerate(edges, comp, min_samples=10)
    assert same_component(solution, 0, 1, 2) and not solution.ambiguous


def test_clearly_better_edge_beats_its_rival():
    edges = edges_of({(0, 1): GOOD, (0, 2): hist(b1=20)})  # rival: mean u .75 >= 2 * .25 + .25
    solution = cluster.agglomerate(edges, components_of([0, 1, 1]), min_samples=10)
    assert same_component(solution, 0, 1) and not same_component(solution, 0, 2)
    assert not solution.ambiguous


# --------------------------------------------------------------------------- agglomerate: elimination


def elimination_edges():
    # X=0 (cam 0), R=1 (cam 1, the rival), Y=2 (cam 1, X's true partner), Z=3 (cam 0, overlaps X, R's true partner).
    # X is equally consistent with R and with Y -> ambiguous in isolation. R is clearly better explained by Z.
    # Z-R is ranked last (lower f1), so in pass 1 X's edges are looked at before Z-R is accepted.
    weak = hist(b0=17, b4=3)  # f1 = f2 = .85
    return edges_of({(0, 1): weak, (0, 2): weak, (1, 3): hist(b0=16, b3=4)})  # Z-R: f1 .8, f2 1.0


def test_elimination_resolves_an_ambiguous_edge_once_a_rival_is_taken():
    comp = components_of([0, 1, 1, 0])
    solution = cluster.agglomerate(elimination_edges(), comp, min_samples=10, margin=0.1)
    pairs = {(e["a"], e["b"]): e for e in solution.accepted}
    assert set(pairs) == {(1, 3), (0, 2)}
    assert pairs[(1, 3)]["iteration"] == 1
    assert pairs[(0, 2)]["iteration"] == 2, "X-Y is only decidable after Z-R eliminated R as X's rival"
    assert solution.iterations == 2 and not solution.ambiguous
    assert same_component(solution, 0, 2) and same_component(solution, 1, 3)
    assert not same_component(solution, 0, 1) and not same_component(solution, 0, 3)
    assert any(b["a"] == 0 and b["b"] == 1 and b["reason"] == "same_camera_overlap" for b in solution.blocked)


def test_without_the_eliminating_link_the_same_edge_stays_ambiguous():
    edges = edges_of({(0, 1): hist(b0=17, b4=3), (0, 2): hist(b0=17, b4=3)})
    solution = cluster.agglomerate(edges, components_of([0, 1, 1]), min_samples=10, margin=0.1)
    assert not solution.accepted
    assert {(a["a"], a["b"]) for a in solution.ambiguous} == {(0, 1), (0, 2)}


# --------------------------------------------------------------------------- agglomerate: human constraints


def test_cannot_link_keeps_atoms_apart_even_when_supported():
    edges = edges_of({(0, 1): GOOD})
    solution = cluster.agglomerate(edges, components_of([0, 1]), min_samples=10, cannot_link=[(0, 1)])
    assert not same_component(solution, 0, 1)
    assert any(b["reason"] == "contradicted" for b in solution.blocked)


def test_cannot_link_propagates_through_merged_components():
    edges = edges_of({(0, 2): GOOD, (1, 2): GOOD})
    solution = cluster.agglomerate(edges, components_of([0, 1, 2]), min_samples=10, cannot_link=[(0, 1)])
    assert not same_component(solution, 0, 1)


def test_must_link_joins_atoms_without_machine_evidence():
    solution = cluster.agglomerate(EMPTY, components_of([0, 1]), min_samples=10, must_link=[(0, 1)])
    assert same_component(solution, 0, 1)
    assert solution.accepted == [{"a": 0, "b": 1, "source": "human"}]


def test_must_link_that_contradicts_cannot_link_is_not_forced():
    edges = edges_of({(0, 1): GOOD})
    solution = cluster.agglomerate(edges, components_of([0, 1]), min_samples=10,
                                   must_link=[(0, 1)], cannot_link=[(0, 1)])
    assert not same_component(solution, 0, 1)
    [blocked] = [b for b in solution.blocked if b.get("source") == "human"]
    assert blocked["reason"] == "human same vs contradicted"


def test_must_link_across_simultaneous_same_camera_atoms_is_blocked_unless_duplicate():
    empty = EMPTY
    blocked = cluster.agglomerate(empty, components_of([0, 0]), min_samples=10, must_link=[(0, 1)])
    assert not same_component(blocked, 0, 1)
    assert blocked.blocked[0]["reason"] == "human same vs same_camera_overlap"
    comp = components_of([0, 0])
    comp.duplicates = {(0, 1)}
    allowed = cluster.agglomerate(empty, comp, min_samples=10, must_link=[(0, 1)])
    assert same_component(allowed, 0, 1)


# --------------------------------------------------------------------------- atoms / cuts


def test_atoms_split_tracklets_at_cut_windows():
    atoms = cluster.make_atoms(np.array([5, 7, 5, 9]), {5: [3, 6], 9: []})
    assert list(atoms.tracklets) == [5, 7, 9] and atoms.count == 5
    windows = np.array([0, 2, 3, 5, 6, 9, 0, 4])
    tracklets = np.array([5, 5, 5, 5, 5, 5, 7, 9])
    assert list(atoms.atom_of(tracklets, windows)) == [0, 0, 1, 1, 2, 2, 3, 4]
    assert list(atoms.tracklet_of_atom) == [5, 5, 5, 7, 9]
    assert atoms.atom_of(np.array([42]), np.array([0]))[0] == -1


# --------------------------------------------------------------------------- scene: evidence


@needs_calibration
def test_pair_evidence_is_tight_for_one_person_and_clipped_for_two():
    cams = load_cams()
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(stationary((-100, -50), T))
    q = sim.add_person(stationary((150, 0), T, phase0=2))
    sim.see(p, ["0"]), sim.see(p, ["1"]), sim.see(q, ["2"])
    scene = sim.build()
    samples = pair_samples(scene.scene, RECIPE)
    same = (samples.cam_a == 0) & (samples.cam_b == 1)
    different = (samples.cam_a == 0) & (samples.cam_b == 2)
    assert same.sum() == T and np.median(samples.u[same]) < 0.5 and samples.plausible[same].all()
    assert np.median(samples.u[different]) >= 8.0


@needs_calibration
def test_floor_veto_a_body_far_below_the_floor_is_never_associated():
    """Two-view agreement is not enough: the triangulated torso must not be below the floor (-30 in)."""
    cams = load_cams()
    normal = stationary((-100, -50), T)
    sunk = Trajectory(normal.xyz - np.array([0, 0, 90.0]))
    outcome = {}
    for name, trajectory in (("normal", normal), ("sunk", sunk)):
        sim = SceneBuilder(T, seed=1)
        pid = sim.add_person(trajectory)
        sim.see(pid, ["0", "1", "2"])
        scene = sim.build()
        samples = pair_samples(scene.scene, RECIPE)
        outcome[name] = (samples.plausible.mean(), associate(scene.scene, RECIPE).n_components)
    assert outcome["normal"] == (1.0, 1)
    assert outcome["sunk"][0] == 0.0 and outcome["sunk"][1] == 3


# --------------------------------------------------------------------------- scene: association


def two_people(pos_p, pos_q, cameras, seed=1, heading=(0.0, 0.0), n=T):
    cams = load_cams()
    sim = SceneBuilder(n, seed=seed)
    p = sim.add_person(stationary(pos_p, n, heading=heading[0]))
    q = sim.add_person(stationary(pos_q, n, heading=heading[1], phase0=2))
    for pid in (p, q):
        visible = visible_cameras(cams, sim.persons[pid].xyz[0], 12)
        assert set(cameras) <= set(visible), (pid, visible)
        sim.see(pid, cameras)
    return sim, p, q


@needs_calibration
def test_well_separated_people_seen_by_three_or_more_cameras_get_one_component_each():
    cams = load_cams()
    sim = SceneBuilder(T, seed=1)
    for xy in [(-100, -50), (150, 0), (0, 200), (200, -200)]:
        pid = sim.add_person(stationary(xy, T))
        visible = visible_cameras(cams, sim.persons[pid].xyz[0], 14)
        assert len(visible) >= 3
        sim.see(pid, visible)
    scene = sim.build()
    association = associate(scene.scene, RECIPE)
    labels = association.row_component(scene.scene)
    assert_pure(scene, labels)
    report = scene.report(labels)
    assert report["components"] == 4 and set(report["fragments"].values()) == {1}
    assert not association.solution.ambiguous
    for camera in scene.scene.cameras:  # complete: every detection is in some component
        assert (labels[camera.index][camera.order] >= 0).all()


@needs_calibration
def test_well_separated_people_walking_are_associated():
    cams = load_cams()
    sim = SceneBuilder(100, seed=2)
    a = sim.add_person(walking((-120, -60), (-60, 120), 100))
    b = sim.add_person(walking((200, 150), (150, -150), 100, phase0=1))
    for pid in (a, b):
        sim.see(pid, ["0", "1", "2", "3", "5"])
    scene = sim.build()
    labels = associate(scene.scene, RECIPE).row_component(scene.scene)
    assert_pure(scene, labels)
    assert scene.report(labels)["fragments"] == {0: 1, 1: 1}


@needs_calibration
def test_two_walkers_crossing_paths_stay_pure_and_complete():
    sim = SceneBuilder(100, seed=1)
    p = sim.add_person(walking((-100, 0), (150, 50), 100))
    q = sim.add_person(walking((150, -50), (-100, 50), 100, phase0=2))
    for pid in (p, q):
        sim.see(pid, ["0", "1", "2", "3", "5"])
    scene = sim.build()
    labels = associate(scene.scene, RECIPE).row_component(scene.scene)
    assert_pure(scene, labels)
    assert scene.report(labels)["fragments"] == {0: 1, 1: 1}


@needs_calibration
@pytest.mark.parametrize("distance", [30, 40])
@pytest.mark.parametrize("cameras", [["0", "1", "2"], ["0", "1", "2", "3", "5"], ["1", "2", "3"], ["0", "1"]])
def test_two_close_stationary_people_are_never_mixed(distance, cameras):
    """People 30-40 in apart seen by the same cameras: unmerged/ambiguous is allowed, mixing never."""
    sim, p, q = two_people((0, 0), (distance, 0), cameras)
    scene = sim.build()
    association = associate(scene.scene, RECIPE)
    labels = association.row_component(scene.scene)
    assert_pure(scene, labels, f"d={distance} cams={cameras}")
    if len(cameras) >= 3:
        assert scene.report(labels)["fragments"] == {0: 1, 1: 1}, "three views should also be complete"


@needs_calibration
def test_person_split_into_tracklets_and_occluded_is_still_one_component():
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(walking((-100, -50), (50, 50), T))
    q = sim.add_person(stationary((150, 0), T, phase0=2))
    sim.see(p, ["0"], fragment=20, drop=range(20, 25))  # cam 0: three tracklets (20, 20, 15 frames), a short occlusion
    sim.see(p, ["1", "2", "3"])
    sim.see(q, ["0", "1", "2", "3"])
    scene = sim.build()
    labels = associate(scene.scene, RECIPE).row_component(scene.scene)
    assert_pure(scene, labels)
    assert scene.report(labels)["fragments"] == {0: 1, 1: 1}
    assert len(set(scene.scene.cameras[0].tracklet[scene.rows(p, "0")])) == 3


# --------------------------------------------------------------------------- the epipolar trap


def trap_scene(cam_a, cam_b, cam_c, p_xy, offset_in, *, swap_order=False, seed=1):
    """P (standing) seen by A (and C); Q (seated, on A's ray through P's torso) seen by B (and C)."""
    cams = load_cams()
    q_xy = trap_position(cams, cam_a, p_xy, offset_in=offset_in)
    sim = SceneBuilder(T, seed=seed)
    trajectories = [("p", stationary(p_xy, T)), ("q", stationary(q_xy, T, pose="sit", phase0=2))]
    ids = {}
    for name, trajectory in (reversed(trajectories) if swap_order else trajectories):
        ids[name] = sim.add_person(trajectory)
    p, q = ids["p"], ids["q"]
    cams_p = [cam_a] + ([cam_c] if cam_c else [])
    cams_q = [cam_b] + ([cam_c] if cam_c else [])
    for pid, wanted in ((p, cams_p), (q, cams_q)):
        visible = visible_cameras(cams, sim.persons[pid].xyz[0], 12)
        assert set(wanted) <= set(visible), (pid, wanted, visible)
        sim.see(pid, wanted)
    return sim.build(), p, q


ORDER_BUG = ("cluster.py:398 (candidate order) + 408-430 (rivals) + 448 / agglomerate: edges are taken strongest-first with ties broken by atom "
             "index, and the strength key (-f1, mean u from 9 coarse bin centres, EDGES[0] = 0.5 accept units) cannot "
             "tell a true pair (~0.3 u) from a ghost pair (~0.5 u). The ghost edge P_A-Q_B is accepted first (its "
             "competitors are only looked for in camera B, rivals() cluster.py:408-430), which eliminates the true "
             "partners P_C and Q_C (they are contradicted by the merged component), leaving a mixed component. "
             "Suggested fix: (1) in rivals(), also count supported neighbours of the atom that the merge would "
             "eliminate (supported edge to r, and r contradicted/overlapping with the partner's component) as "
             "rivals unless clearly worse; (2) rank edges by the raw sum of u (store it next to the histogram) "
             "instead of bin-centre means so true pairs sort before ghosts.")


@needs_calibration
@pytest.mark.parametrize("swap_order", [
    False,
    pytest.param(True),
])
def test_trap_with_third_view_is_pure_for_imperfect_ghost_regardless_of_person_order(swap_order):
    """Same scene, only the numbering of the two people differs: the result must not depend on it."""
    scene, p, q = trap_scene("2", "4", "3", (0, 100), 10, swap_order=swap_order)
    association = associate(scene.scene, RECIPE)
    assert_pure(scene, association.row_component(scene.scene), f"swap={swap_order}")


@needs_calibration
@pytest.mark.parametrize("swap_order", [
    False,
    True,
])
def test_trap_with_third_view_exact_ghost_must_not_mix(swap_order):
    """Q sits exactly on camera 0's ray through P (a seated body is a homothetic copy of P): P_0 is as
    consistent with Q_3/Q_5 as with its own P_5 (all < 2.5 px). Geometry cannot decide, so the solver must
    leave it ambiguous - it must not silently merge P_0 with Q (and orphan P_5)."""
    scene, p, q = trap_scene("0", "3", "5", (50, 50), 0, swap_order=swap_order)
    association = associate(scene.scene, RECIPE)
    assert_pure(scene, association.row_component(scene.scene), f"swap={swap_order}")


@needs_calibration
@pytest.mark.parametrize("swap_order", [False, True])
@pytest.mark.parametrize("offset", [25, 40])
def test_trap_with_third_view_and_a_distinguishable_ghost_is_pure_and_complete(offset, swap_order):
    scene, p, q = trap_scene("0", "3", "5", (50, 50), offset, swap_order=swap_order)
    labels = associate(scene.scene, RECIPE).row_component(scene.scene)
    assert_pure(scene, labels)
    assert scene.report(labels)["fragments"] == {0: 1, 1: 1}


@needs_calibration
@pytest.mark.parametrize("config", [("0", "3", (50, 50)), ("0", "1", (150, -50)), ("2", "4", (0, 100))])
def test_trap_with_both_people_visible_in_both_cameras_never_mixes_and_reports_ambiguity(config):
    cam_a, cam_b, p_xy = config
    cams = load_cams()
    q_xy = trap_position(cams, cam_a, p_xy)
    sim = SceneBuilder(T, seed=1)
    p = sim.add_person(stationary(p_xy, T))
    q = sim.add_person(stationary(q_xy, T, pose="sit", phase0=2))
    for pid in (p, q):
        assert {cam_a, cam_b} <= set(visible_cameras(cams, sim.persons[pid].xyz[0], 12))
        sim.see(pid, [cam_a, cam_b])
    scene = sim.build()
    association = associate(scene.scene, RECIPE)
    assert_pure(scene, association.row_component(scene.scene))
    assert association.solution.ambiguous, "the K2,2 of consistent pairs must be reported, not silently resolved"


@needs_calibration
def test_trap_with_only_two_views_is_inherently_ambiguous_and_documented():
    """P seen only by camera 0, Q (on its ray) only by camera 3: P_0-Q_3 is a perfectly valid two-view
    person (the reconstruction IS Q), so no physical-plausibility rule can reject it. The solver may leave
    them apart (fine) or merge them; if it merges, the merged body must be exactly Q's true place and
    plausible, i.e. nothing a veto could have caught."""
    scene, p, q = trap_scene("0", "3", None, (50, 50), 0)
    association = associate(scene.scene, RECIPE)
    labels = association.row_component(scene.scene)
    mixed = scene.mixed(labels)
    if not mixed:
        return
    from data_engine.solve.tracks3d import triangulate_labels
    frames, _ = triangulate_labels(scene.scene, labels, RECIPE)
    low, high = RECIPE["geometry"]["root_z_range_in"]
    measured = np.isfinite(frames.root[:, 0])
    assert measured.any()
    assert ((frames.root[measured, 2] > low) & (frames.root[measured, 2] < high)).all()
    ghost_vs_q = np.linalg.norm(frames.root[measured][:, :2] - scene.truth[q][frames.t[measured]][:, [11, 12]].mean(axis=1)[:, :2], axis=1)
    assert np.median(ghost_vs_q) < 6.0, "the ghost is Q's real position: geometrically indistinguishable"


# --------------------------------------------------------------------------- identity switch inside a tracklet


def switched_scene(switch, n=120, p_xy=(-100, -50), q_xy=(150, 0), seed=1):
    sim = SceneBuilder(n, seed=seed)
    p = sim.add_person(stationary(p_xy, n))
    q = sim.add_person(stationary(q_xy, n, phase0=2))
    sim.see(p, ["1", "2"])
    sim.see(q, ["3", "5"])
    sim.see(p, ["0"], 0, switch)
    sim.see(q, ["0"], switch, n)
    sim.tracklet("0", [(p, 0, switch), (q, switch, n)])  # camera 0: ONE tracklet follows P, then Q
    scene = sim.build()
    return scene, p, q


@needs_calibration
@pytest.mark.parametrize("switch", [60])
def test_identity_switch_at_a_window_boundary_is_split_and_reassigned(switch):
    scene, p, q = switched_scene(switch)
    tracklet = scene.tracklet_of(p, "0")
    assert tracklet == scene.tracklet_of(q, "0"), "setup: one tracklet carries both people"
    association = associate(scene.scene, RECIPE)
    labels = association.row_component(scene.scene)
    assert_pure(scene, labels)
    # split-and-reassign happened: the first half is with P's component, the second with Q's.
    assert [e for e in association.switch_events if e["kind"] == "switch" and e["tracklet"] == tracklet]
    assert association.passes == 2
    first = labels[0][scene.rows(p, "0")]
    second = labels[0][scene.rows(q, "0")]
    p_other = labels[1][scene.rows(p, "1")]
    q_other = labels[3][scene.rows(q, "3")]
    assert set(first) == set(p_other) and set(second) == set(q_other) and set(first) != set(second)
    assert scene.report(labels)["fragments"] == {0: 1, 1: 1}


@needs_calibration
def test_identity_switch_is_split_in_the_full_solve():
    from data_engine.solve.pipeline import solve
    scene, p, q = switched_scene(60)
    solved = solve(scene.scene, RECIPE, log=lambda *args: None)
    assert_pure(scene, solved.labels.label)
    assert scene.report(solved.labels.label)["fragments"] == {0: 1, 1: 1}


@needs_calibration
@pytest.mark.xfail(strict=True, reason=(
    "cluster.refine_cuts (cluster.py:206-209) cuts at the first window of the *next* sustained run, so a "
    "switch inside a 20-frame window leaves up to 19 detections of the new person (here frames 70-79 of Q) "
    "in the old person's component: with switch=70 the cut lands at frame 80 and P's component keeps Q's frames "
    "70-79; with switch=50 the cut lands at frame 60 and P's component keeps Q's frames 50-59. The mixed 'transition' windows carry no "
    "support for either person, so they should not be attached to either. Suggested fix: when a switch is "
    "found, also cut at the end of the last window supported by the 'from' component "
    "(cuts {before_last + 1, after_first}) so the transition piece becomes its own atom that stays unassigned "
    "(or is bisected at frame level with per-frame residuals), instead of inheriting the 'from' label."))
@pytest.mark.parametrize("switch", [70, 50])
def test_identity_switch_inside_a_window_must_not_leave_wrong_person_detections(switch):
    scene, p, q = switched_scene(switch)
    labels = associate(scene.scene, RECIPE).row_component(scene.scene)
    assert_pure(scene, labels, f"switch at t={switch}")


@needs_calibration
def test_human_cut_splits_the_tracklet_into_two_atoms_that_find_their_own_people():
    scene, p, q = switched_scene(60)
    tracklet = scene.tracklet_of(p, "0")
    plain = associate(scene.scene, RECIPE, max_passes=1)  # without refine: the switched tracklet cannot be placed
    cut = associate(scene.scene, RECIPE, max_passes=1, cuts={tracklet: [3]})
    assert cut.atoms.count == plain.atoms.count + 1
    labels = cut.row_component(scene.scene)
    assert_pure(scene, labels)
    assert scene.report(labels)["fragments"] == {0: 1, 1: 1}
    assert not scene.mixed(labels)
