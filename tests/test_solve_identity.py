"""Identity across time (solve/identity.py): summarise() and link_components().

Components are described by hand (a start, an end and a 3D root track in inches), so each rule of the
module docstring is exercised in isolation: unique feasible continuation, ambiguity inside the automatic
radius, no linking across overlapping lifetimes, human must/cannot links, and open ends at the recording edges.
"""

import numpy as np
import pytest

from data_engine.config import RECIPE
from data_engine.solve.identity import link_components, summarise
from data_engine.solve.tracks3d import PersonFrames

METRE = 1 / 0.0254
T = 400


def frames_of(spec: dict[int, dict]) -> PersonFrames:
    """spec: label -> {t0, t1 (exclusive), xy (metres, constant or callable t -> (x, y)), [unmeasured: frames], [cams]}."""
    label, t, rows, root = [], [], [], []
    for lab, s in spec.items():
        position = s["xy"] if callable(s["xy"]) else (lambda _t, xy=s["xy"]: xy)
        cams = s.get("cams", (0, 1))
        for frame in range(s["t0"], s["t1"]):
            label.append(lab), t.append(frame)
            row = np.full(3, -1, np.int32)
            row[list(cams)] = frame
            rows.append(row)
            x, y = position(frame)
            measured = frame not in s.get("unmeasured", ())
            root.append([x * METRE, y * METRE, 36.0] if measured else [np.nan] * 3)
    n = len(label)
    return PersonFrames(np.array(label, np.int32), np.array(t, np.int32), np.array(rows, np.int32),
                        np.full((n, 17, 3), np.nan, np.float32), np.zeros((n, 17), np.uint8),
                        np.full((n, 17), np.nan, np.float32), np.array(root, np.float32))


def run(spec, n_labels=None, **kwargs):
    frames = frames_of(spec)
    summaries = summarise(frames, n_labels if n_labels is not None else max(spec) + 1)
    return summaries, link_components(summaries, RECIPE, T=T, **kwargs)


def identity_pairs(links):
    return {(a["from"], a["to"]) for a in links.accepted}


# --------------------------------------------------------------------------- summarise


def test_summarise_describes_each_component():
    moving = {"t0": 10, "t1": 60, "xy": lambda t: (0.01 * t, 0.0), "unmeasured": range(10, 14), "cams": (0, 2)}
    still = {"t0": 100, "t1": 130, "xy": (3.0, 1.0)}
    summaries = summarise(frames_of({0: moving, 2: still}), 4)
    assert summaries[1] is None and summaries[3] is None
    s = summaries[0]
    assert (s.label, s.t_first, s.t_last, s.n_frames, s.n_measured) == (0, 10, 59, 50, 46)
    assert s.cameras == (0, 2) and s.detections == 100
    assert (s.first_root_t, s.last_root_t) == (14, 59)
    # median over the first / last second (10 sync frames) of measured roots, inches
    assert s.first_root[0] == pytest.approx(np.median([0.01 * t for t in range(14, 25)]) * METRE, abs=0.1)
    assert s.last_root[0] == pytest.approx(np.median([0.01 * t for t in range(49, 60)]) * METRE, abs=0.1)
    assert summaries[2].cameras == (0, 1) and summaries[2].first_root[1] == pytest.approx(1.0 * METRE, abs=0.1)


def test_summarise_component_without_any_3d_root_has_no_position():
    summaries = summarise(frames_of({0: {"t0": 0, "t1": 20, "xy": (0, 0), "unmeasured": range(0, 20)}}), 1)
    s = summaries[0]
    assert s.n_measured == 0 and s.first_root is None and s.last_root is None and s.first_root_t is None


# --------------------------------------------------------------------------- linking


def test_unique_reappearance_nearby_is_linked():
    spec = {0: {"t0": 100, "t1": 150, "xy": (2.0, 2.0)},
            1: {"t0": 170, "t1": 230, "xy": (2.7, 2.0)}}  # 2 s later, 0.7 m away
    summaries, links = run(spec)
    assert identity_pairs(links) == {(0, 1)}
    [link] = links.accepted
    assert link["source"] == "machine:unique_motion_continuity" and link["gap_s"] == pytest.approx(2.0, abs=0.15)
    assert 0.6 < link["distance_m"] < 0.8
    assert links.identity[0] == links.identity[1] == 0
    # the joined chain starts/ends inside the recording (reported, nothing to choose from); the join itself is no open end
    assert {(e["component"], e["end"], len(e["candidates"])) for e in links.open_ends} == {(0, "start", 0), (1, "end", 0)}


def test_far_away_reappearance_is_not_linked_automatically_but_reported_with_its_candidate():
    spec = {0: {"t0": 100, "t1": 150, "xy": (0.0, 0.0)},
            1: {"t0": 170, "t1": 230, "xy": (6.0, 0.0)}}  # reachable at 12 m/s, but not at the automatic walking pace
    summaries, links = run(spec)
    assert not links.accepted and links.identity[0] != links.identity[1]
    start = {e["component"]: [c["component"] for c in e["candidates"]] for e in links.open_ends if e["end"] == "start"}
    assert start == {0: [], 1: [0]}
    end = [e for e in links.open_ends if e["end"] == "end" and e["component"] == 0]
    assert [c["component"] for c in end[0]["candidates"]] == [1]


def test_teleport_is_never_a_candidate():
    spec = {0: {"t0": 100, "t1": 150, "xy": (0.0, 0.0)},
            1: {"t0": 160, "t1": 230, "xy": (60.0, 0.0)}}  # 60 m in 1 s
    summaries, links = run(spec)
    assert not links.accepted
    assert all(e["candidates"] == [] for e in links.open_ends)


def test_two_people_inside_the_automatic_radius_are_ambiguous_and_listed():
    spec = {0: {"t0": 100, "t1": 150, "xy": (2.0, 2.0)},
            1: {"t0": 100, "t1": 150, "xy": (2.4, 2.0)},  # 0.4 m from component 0, same time
            2: {"t0": 170, "t1": 230, "xy": (2.2, 2.0)}}  # reappears between them
    summaries, links = run(spec)
    assert not links.accepted
    assert len({links.identity[0], links.identity[1], links.identity[2]}) == 3
    [start] = [e for e in links.open_ends if e["end"] == "start" and e["component"] == 2]
    assert {c["component"] for c in start["candidates"]} == {0, 1}
    assert all(c["distance_m"] < 0.75 for c in start["candidates"])
    for lab in (0, 1):  # each predecessor's open end lists the same single continuation
        [end] = [e for e in links.open_ends if e["end"] == "end" and e["component"] == lab]
        assert [c["component"] for c in end["candidates"]] == [2]


def test_two_successors_for_one_predecessor_are_ambiguous():
    spec = {0: {"t0": 100, "t1": 150, "xy": (2.0, 2.0)},
            1: {"t0": 170, "t1": 230, "xy": (2.3, 2.0)},
            2: {"t0": 170, "t1": 230, "xy": (1.7, 2.0)}}
    _, links = run(spec)
    assert not links.accepted
    [end] = [e for e in links.open_ends if e["end"] == "end" and e["component"] == 0]
    assert {c["component"] for c in end["candidates"]} == {1, 2}


def test_gap_longer_than_the_automatic_gap_is_not_linked():
    spec = {0: {"t0": 20, "t1": 60, "xy": (2.0, 2.0)},
            1: {"t0": 130, "t1": 200, "xy": (2.0, 2.1)}}  # 7 s gap: > auto_gap_s (5 s)
    _, links = run(spec)
    assert not links.accepted
    assert links.open_ends  # reported for a human instead


def test_unique_chain_is_followed_transitively():
    spec = {0: {"t0": 50, "t1": 80, "xy": (1.0, 1.0)},
            1: {"t0": 100, "t1": 130, "xy": (1.3, 1.0)},
            2: {"t0": 150, "t1": 180, "xy": (1.6, 1.0)}}
    _, links = run(spec)
    assert identity_pairs(links) == {(0, 1), (1, 2)}
    assert links.identity[0] == links.identity[1] == links.identity[2]


def test_components_overlapping_in_time_are_never_linked():
    # Same place, but the successor starts while the predecessor is still alive: two different people.
    spec = {0: {"t0": 100, "t1": 200, "xy": (2.0, 2.0)},
            1: {"t0": 190, "t1": 260, "xy": (2.1, 2.0)}}
    _, links = run(spec)
    assert not links.accepted and links.identity[0] != links.identity[1]
    assert all(e["candidates"] == [] for e in links.open_ends)


def test_overlap_is_checked_against_every_member_of_a_chain():
    # 0 -> 1 is linked. Component 2 overlaps 1 (but not 0) and would be reachable from 0's end; it must not join.
    spec = {0: {"t0": 50, "t1": 100, "xy": (1.0, 1.0)},
            1: {"t0": 120, "t1": 200, "xy": (1.2, 1.0)},
            2: {"t0": 105, "t1": 140, "xy": (4.0, 1.0)}}  # starts 0.5 s after 0 ended, 3 m away, overlaps 1
    _, links = run(spec)
    assert identity_pairs(links) == {(0, 1)}
    assert links.identity[2] != links.identity[0]
    [start] = [e for e in links.open_ends if e["end"] == "start" and e["component"] == 2]
    assert start["candidates"] == []


def test_cannot_link_blocks_an_otherwise_unique_link():
    spec = {0: {"t0": 100, "t1": 150, "xy": (2.0, 2.0)},
            1: {"t0": 170, "t1": 230, "xy": (2.3, 2.0)}}
    _, links = run(spec, cannot_link=[(0, 1)])
    assert not links.accepted and links.identity[0] != links.identity[1]
    assert all(e["candidates"] == [] for e in links.open_ends) and len(links.open_ends) == 4


def test_cannot_link_applies_to_whole_chains():
    spec = {0: {"t0": 50, "t1": 80, "xy": (1.0, 1.0)},
            1: {"t0": 100, "t1": 130, "xy": (1.3, 1.0)},
            2: {"t0": 150, "t1": 180, "xy": (1.6, 1.0)}}
    _, links = run(spec, cannot_link=[(0, 2)])
    assert identity_pairs(links) == {(0, 1)} and links.identity[2] != links.identity[0]


def test_must_link_forces_a_link_the_machine_would_not_make():
    spec = {0: {"t0": 20, "t1": 60, "xy": (2.0, 2.0)},
            1: {"t0": 300, "t1": 340, "xy": (-5.0, 8.0)}}  # 24 s and 11 m apart
    _, machine = run(spec)
    assert machine.identity[0] != machine.identity[1]
    _, links = run(spec, must_link=[(0, 1)])
    assert links.identity[0] == links.identity[1]
    assert links.accepted == [{"from": 0, "to": 1, "source": "human"}]
    assert not [e for e in links.open_ends if e["end"] == "end" and e["component"] == 0]


def test_must_link_removes_the_pair_from_the_open_ends():
    spec = {0: {"t0": 100, "t1": 150, "xy": (2.0, 2.0)},
            1: {"t0": 170, "t1": 230, "xy": (2.4, 2.0)},
            2: {"t0": 170, "t1": 230, "xy": (1.6, 2.0)}}
    _, auto = run(spec)
    assert not auto.accepted
    _, links = run(spec, must_link=[(0, 2)])
    assert links.identity[0] == links.identity[2] != links.identity[1]
    assert {e["component"] for e in links.open_ends if e["end"] == "start"} == {0, 1}  # chain start (0, 2) and 1


def test_open_ends_at_the_start_and_end_of_the_recording_are_not_reported():
    margin = 20
    spec = {0: {"t0": 5, "t1": 60, "xy": (0.0, 0.0)},  # present from the start of the recording
            1: {"t0": 300, "t1": T - 5, "xy": (4.0, 0.0)}}  # present until the end
    _, links = run(spec)
    assert [e for e in links.open_ends if e["end"] == "start" and e["component"] == 0] == []
    assert [e for e in links.open_ends if e["end"] == "end" and e["component"] == 1] == []
    # a start well inside the recording, and an end well inside it, are reported
    assert {(e["component"], e["end"]) for e in links.open_ends} == {(0, "end"), (1, "start")}
    for e in links.open_ends:
        assert e["t"] > margin if e["end"] == "start" else e["t"] < T - 1 - margin


def test_edge_margin_boundary():
    spec = {0: {"t0": 21, "t1": 60, "xy": (0.0, 0.0)},  # starts at 21 > margin 20 -> reported
            1: {"t0": 100, "t1": T - 1 - 20 + 1, "xy": (5.0, 5.0)}}  # ends at T-1-20 -> not reported
    _, links = run(spec, edge_margin=20)
    assert (0, "start") in {(e["component"], e["end"]) for e in links.open_ends}
    assert (1, "end") not in {(e["component"], e["end"]) for e in links.open_ends}


def test_components_with_too_few_3d_frames_are_not_linkable_but_keep_an_identity():
    spec = {0: {"t0": 100, "t1": 150, "xy": (2.0, 2.0)},
            1: {"t0": 160, "t1": 190, "xy": (2.3, 2.0), "unmeasured": range(162, 190)},  # only 2 measured frames
            2: {"t0": 195, "t1": 260, "xy": (2.4, 2.0)}}
    summaries, links = run(spec)
    assert summaries[1].n_measured == 2
    assert identity_pairs(links) == {(0, 2)}, "the weak component neither blocks nor takes part in the link"
    assert 1 not in {e["component"] for e in links.open_ends}
    assert links.identity[1] >= 0 and links.identity[1] not in (links.identity[0], links.identity[2])


def test_identity_ids_are_dense_by_first_appearance_and_missing_labels_get_minus_one():
    spec = {3: {"t0": 10, "t1": 40, "xy": (0.0, 0.0)},
            1: {"t0": 100, "t1": 140, "xy": (10.0, 0.0)},
            0: {"t0": 200, "t1": 240, "xy": (-10.0, 5.0)}}
    summaries, links = run(spec, n_labels=5)
    assert list(links.identity[[3, 1, 0]]) == [0, 1, 2] and links.identity[2] == -1 and links.identity[4] == -1


def test_a_unique_continuation_far_away_is_a_candidate_not_a_link():
    """The automatic radius grows with the gap, but 'the only track born nearby' is no proof far from where the
    earlier track ended: the earlier person may still be there, untracked (felix hour, 2026-11-19)."""
    def scene(x):
        return {0: {"t0": 20, "t1": 100, "xy": (0.0, 0.0)}, 1: {"t0": 140, "t1": 220, "xy": (x, 0.0)}}
    _, near = run(scene(1.4))
    assert identity_pairs(near) == {(0, 1)}
    _, far = run(scene(3.0))
    assert not far.accepted
    start = next(end for end in far.open_ends if end["component"] == 1 and end["end"] == "start")
    assert [candidate["component"] for candidate in start["candidates"]] == [0]
    relaxed = {**RECIPE, "temporal": {**RECIPE["temporal"], "auto_max_distance_m": 10.0}}
    frames = frames_of(scene(3.0))
    assert identity_pairs(link_components(summarise(frames, 2), relaxed, T=T)) == {(0, 1)}
