"""Solve stage: association, 3D tracks and identity, with the unknowns that remain.

Inputs: tracklets, alignment sync table, calibration, and every active human constraint.
Outputs (``rounds/rNNN/``):
    solution.npz   per camera: ``identity_<cam>``, ``component_<cam>``, ``source_<cam>`` per raw row
    tracks3d.npz   one row per (identity component, sync index): xyz (17x3), root, views, residual
    solve.json     log of what was merged/attached/linked and why

Unknown subjects are built from detection keys, so they stay stable when tracklets,
components or identities are renumbered by a later round.

The stage has an expensive part (association and 3D positions) and a cheap one (identity across gaps, the
swap check, the unknowns). The expensive part is kept in ``solve_state/`` with the hash of everything it
depends on; when the stage runs again with those inputs unchanged - the lap the recover stage asks for
after it measured positions beyond track ends - only the cheap part runs, now with the recovered positions.
"""

from __future__ import annotations

import glob
import json
import pickle
import time

import numpy as np

from ..config import RECIPE
from ..engine import Context, StageResult
from ..solve import coverage as coverage_module
from ..solve import link_ranking, recover, view_choice
from ..solve.pipeline import Associated, Constraints, Solved, associate_and_refine, coverage_metrics, identify, unit_scene
from ..solve.scene import Scene, load_scene
from ..store import parse_det_key
from ..workspace import stable_hash

STATE_VERSION = 1  # layout of the kept association; a change makes every kept state unusable
ASSOCIATION_RECIPE = ("geometry", "association", "refine")  # what associate_and_refine and the alignment check read


def association_key(context: Context, step: int) -> str:
    """Hash of everything the association depends on: the stage's inputs as the engine sees them."""
    from ..engine import STAGE, Engine

    # Only the recipe sections the association reads: a change to the identity or swap-check rules leaves it valid.
    return stable_hash({"inputs": Engine(context.workspace).input_hash(STAGE["solve"], ASSOCIATION_RECIPE),
                        "step": step, "version": STATE_VERSION})


def load_state(context: Context, key: str):
    directory = context.workspace.path("solve_state")
    try:
        if json.loads((directory / "key.json").read_text(encoding="utf-8")).get("key") != key:
            return None
        with open(directory / "associated.pkl", "rb") as handle:
            state = pickle.load(handle)
        return state["associated"], state["alignment_check"]
    except (OSError, ValueError, KeyError, pickle.UnpicklingError, AttributeError, EOFError) as exc:
        if (directory / "key.json").is_file():
            context.log(f"kept association unusable ({type(exc).__name__}: {exc}); solving again", level="warning")
        return None


def save_state(context: Context, key: str, associated: Associated, alignment: dict) -> None:
    directory = context.workspace.path("solve_state")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "key.json").unlink(missing_ok=True)  # never a key next to another state
    temporary = directory / "associated.pkl.tmp"
    with open(temporary, "wb") as handle:
        pickle.dump({"associated": associated, "alignment_check": alignment}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(directory / "associated.pkl")
    (directory / "key.json").write_text(json.dumps({"key": key, "saved": time.time()}), encoding="utf-8")


def multi_camera(scene: Scene, labels: recover.Labels) -> np.ndarray:
    """Per component: is it seen by at least two cameras (only those are 3D tracks)."""
    multi = np.zeros(labels.next_label() + 1, dtype=bool)
    for label, cameras in recover.camera_counts(scene, labels).items():
        multi[label] = len(cameras) >= 2
    return multi


def recovered_positions(context: Context, scene: Scene, associated: Associated) -> dict | None:
    """The positions the recover stage measured for exactly these person-frames, if any."""
    frames = associated.frames
    keep = multi_camera(scene, associated.labels)[frames.label]
    signature = coverage_module.measured_signature(frames.label[keep], frames.t[keep], frames.root[keep])
    found = sorted(glob.glob(str(context.workspace.root / "rounds" / "*" / "tracks3d_recovered.npz")))
    if not found:
        return None
    with np.load(found[-1]) as data:
        if "source_measured_sha256" not in data.files or str(data["source_measured_sha256"]) != signature \
                or not len(data["t"]):
            return None
        return {"component": data["component"].astype(np.int64), "t": data["t"].astype(np.int64),
                "root": data["root"].astype(np.float64), "xyz": data["xyz"].astype(np.float32),
                "positions_sha256": str(data["positions_sha256"]),
                "count": int(len(data["t"]))}

WINDOW = 20  # sync frames per evidence window (2 s)
def swap_unknowns(context: Context, scene: Scene, solved: Solved, constraints: Constraints,
                  resolved_keys: set[str], recovered: dict | None = None) -> dict:
    """Run ``solve/swap_check`` on the solution; flagged and unverified moments become unknowns.

    ``recovered`` (positions of the coverage recovery for these components) are added to the measured
    positions when ``RECIPE["swap_check"]["use_recovered"]`` is set: two people measured through a
    crossing leave less room for an exchange than two people lost in it.
    """
    from ..solve import swap_check

    cams = recover.camera_counts(solved.scene, solved.labels)
    multi = np.zeros(solved.labels.next_label() + 1, dtype=bool)
    for label, cameras in cams.items():
        multi[label] = len(cameras) >= 2
    row_identity = []
    for camera in scene.cameras:
        label = solved.labels.label[camera.index]
        row_identity.append(np.where((label >= 0) & multi[np.clip(label, 0, None)],
                                     solved.identity_of_label[np.clip(label, 0, None)], -1).astype(np.int64))
    frames = solved.frames
    keep = multi[frames.label]
    from .. import appearance

    try:
        embed = appearance.embedder(context.workspace, [camera.camera for camera in scene.cameras], context.log)
    except Exception as exc:  # the cue is optional: never fail the solve over it
        context.log(f"appearance cue unavailable ({exc})", level="warning")
        embed = None
    tr_identity = solved.identity_of_label[frames.label[keep]].astype(np.int64)
    tr_t, tr_root, tr_xyz = frames.t[keep], frames.root[keep], frames.xyz[keep]
    added = 0
    if recovered is not None and RECIPE["swap_check"].get("use_recovered"):
        component = recovered["component"]
        use = (component >= 0) & (component < len(multi)) & multi[np.clip(component, 0, len(multi) - 1)]
        added = int(use.sum())
        tr_identity = np.concatenate([tr_identity, solved.identity_of_label[component[use]].astype(np.int64)])
        tr_t = np.concatenate([tr_t, recovered["t"][use].astype(tr_t.dtype)])
        tr_root = np.concatenate([tr_root, recovered["root"][use].astype(tr_root.dtype)])
        tr_xyz = np.concatenate([tr_xyz, recovered["xyz"][use].astype(tr_xyz.dtype)])
    result = swap_check.check(scene, row_identity, tr_identity, tr_t, tr_root, tr_xyz, RECIPE["swap_check"],
                              same_rows=constraints.same, embed=embed,
                              images=(RECIPE["task_images"], RECIPE["link_ranking"]))
    result["summary"]["recovered_positions_used"] = added
    (context.workspace.round_dir(context.round) / "swap_check.json").write_text(
        json.dumps(result, indent=1), encoding="utf-8")
    unknowns, subjects = [], set()
    for entry in result["entries"]:
        if entry["status"] == "verified" or not entry.get("reference_keys") or not entry.get("after_a_keys"):
            continue
        anchor = entry.get("anchor_keys") or entry["reference_keys"]
        if resolved_keys & (set(entry["reference_keys"]) | set(anchor)):
            continue  # a person already said this moment cannot be judged
        start, end = entry["t"]
        cues = entry["cues"]
        if entry["kind"] == "exchange":
            why = (f"people {entry['a']} and {entry['b']} exchanged places between two frames "
                   f"({-entry['continuity_margin_m']:.2f} m cheaper exchanged than kept)")
        elif entry["kind"] == "ending":
            why = (f"the track of person {entry['b']} ends {entry['min_distance_m']:.2f} m from person {entry['a']}, "
                   "who continues: the surviving track may now be following the other person")
        else:
            why = (f"people {entry['a']} and {entry['b']} came within {entry['min_distance_m']:.2f} m; position "
                   "continuity cannot tell who is who afterwards")
        said = "; ".join(filter(None, [
            "cues for a swap: " + ", ".join(entry["swap_cues"]) if entry["swap_cues"] else "",
            "cues against: " + ", ".join(entry["stay_cues"]) if entry["stay_cues"] else "",
            "" if entry["swap_cues"] or entry["stay_cues"] else "no independent cue could decide"]))
        # The subject is the identity's reference detection before the moment. One identity can meet two
        # others within the same seconds: those are separate questions, told apart by the other party.
        subject = f"identity_swap:{anchor[0]}"
        if subject in subjects:
            other = (entry.get("before_b_keys") or entry.get("after_b_keys") or [f"identity {entry['b']}"])[0]
            subject = f"{subject}|{other}"
        if subject in subjects:
            continue  # the same pair at the same reference twice: one question
        subjects.add(subject)
        unknowns.append({
            "subject": subject,
            # exchanges and flagged crossings first, then the closest unverified ones
            "severity": float({"flagged": 2000, "unverified": 1000}[entry["status"]]
                              + 500 * max(0.0, 1.0 - (entry["min_distance_m"] or 0.0))),
            "detail": {"moment": entry["kind"], "status": entry["status"], "identity": entry["a"], "other": entry["b"],
                       "t": int(start), "span_t": [int(start), int(end)], "min_distance_m": entry["min_distance_m"],
                       "cues": {k: v for k, v in cues.items() if k != "contradicting_tracklets"},
                       "swap_cues": entry["swap_cues"], "stay_cues": entry["stay_cues"],
                       "reference_keys": entry["reference_keys"], "after_a_keys": entry["after_a_keys"],
                       "after_b_keys": entry["after_b_keys"], "before_b_keys": entry["before_b_keys"],
                       "image_choice": entry.get("image_choice", "nearest"),
                       "why": f"{why} ({said})",
                       "resolve_by": "a person compares each one before and after the moment"}})
    return {"unknowns": unknowns, "summary": result["summary"]}


def _file_boundaries(scene, camera) -> list[tuple[int, str]]:
    """(first sync index showing a later file of this camera, file name) for every file after the first."""
    import os

    out, cumulative = [], 0
    shown = scene.native[:, camera.index]
    for part in scene.workspace.dataset.cameras[camera.camera].parts[:-1]:
        cumulative += part.count
        at = np.flatnonzero(shown >= cumulative)
        if len(at):
            following = scene.workspace.dataset.cameras[camera.camera].parts[len(out) + 1]
            out.append((int(at[0]), os.path.basename(following.path)))
    return out


MIN_TASK_DETECTIONS = 30  # a free piece shorter than 3 s is not worth an individual question
MIN_IDENTITY_DETECTIONS = 150  # a 3D track this small (a few seconds) is asked about as a group


def _iou(a, b) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    return float(intersection / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection))


def load_constraints(context: Context, scene: Scene) -> tuple[Constraints, dict]:
    """Translate active constraints (detection keys) into rows of this scene."""
    index = {camera.camera: camera for camera in scene.cameras}

    def rows(keys: list[str]) -> list[tuple[int, int]]:
        result = []
        for key in keys:
            try:
                camera, frame, det = parse_det_key(key)
                result.append((index[camera].index, index[camera].detections.row(frame, det)))
            except (KeyError, ValueError):
                pass  # not a detection of this scene (e.g. a policy key or an excluded camera)
        return result

    constraints = Constraints()
    used = {"same": 0, "different": 0, "duplicate": 0, "not_person": 0, "split": 0, "undeterminable": 0, "bad_pose": 0}
    resolved_keys: set[str] = set()
    for row in context.db.constraints():
        kind = row["kind"]
        a, b = rows(row["a"]) if kind in used else [], rows(row["b"]) if kind in used else []
        if kind == "same" and a and b:
            constraints.same.append((a, b))
        elif kind == "different" and a and b:
            constraints.different.append((a, b))
            if row["detail"].get("image_flag") == "other_person":
                constraints.flagged.append((a, b))
        elif kind == "duplicate" and a and b:
            constraints.duplicate.append((a, b))
            constraints.same.append((a, b))
            # One person, two poses: the person may have said which pose is the better one.
            better = row["detail"].get("better")
            if better in ("a", "b"):
                constraints.demoted.extend(b if better == "a" else a)
        elif kind == "not_person" and a and row["detail"].get("scope") == "detection":
            # One image was flagged (mixed pose, or not a person next to good crops of the same track).
            constraints.not_person.extend(a)
        elif kind == "not_person" and a:
            # The human judged the piece from sample crops: reject the whole tracklet of each sample.
            for camera_index, sample in a:
                camera = scene.cameras[camera_index]
                tracklet = int(camera.tracklet[sample])
                members = np.flatnonzero(camera.tracklet == tracklet) if tracklet >= 0 else [sample]
                constraints.not_person.extend((camera_index, int(member)) for member in members)
        elif kind == "bad_pose" and a:
            constraints.bad_pose.extend(a)
        elif kind == "split" and b:
            # Cut the tracklet of the first "after" detection at that detection's window.
            camera_index, first = b[0]
            camera = scene.cameras[camera_index]
            tracklet, t = int(camera.tracklet[first]), int(camera.t[first])
            if tracklet >= 0 and t >= 0:
                constraints.cuts.setdefault(tracklet, []).append(t // WINDOW)
            if a:
                constraints.different.append((a, b))
        elif kind == "undeterminable":
            resolved_keys.update(row["a"])
        else:
            continue
        used[kind] += 1
    return constraints, {"used": used, "resolved_keys": resolved_keys}


def representative_keys(scene: Scene, camera_index: int, rows: np.ndarray, count: int = 3, *,
                        clear: dict | None = None, reach: int = 15) -> list[str]:
    """Evenly spaced detections of a piece, as detection keys. With ``clear`` (the clear-view rule), each is
    exchanged for the nearest clear view of the piece within ``reach`` frames, where there is one."""
    camera = scene.cameras[camera_index]
    rows = rows[camera.t[rows] >= 0]
    if not len(rows):
        return []
    picks = rows[np.unique(np.linspace(0, len(rows) - 1, min(count, len(rows))).round().astype(int))]
    if clear is not None:
        size = image_sizes(scene)[camera_index]
        shown = []
        for pick in picks:
            pool = rows[np.abs(camera.t[rows] - camera.t[pick]) <= reach]
            best = view_choice.clearer(camera, int(pick), pool, size, scene.joint_peak, clear)
            shown.append(best if best not in shown else int(pick))
        picks = shown
    return [camera.detections.key(int(r)) for r in picks]


def image_sizes(scene: Scene) -> list[tuple[int, int]]:
    cameras = scene.workspace.dataset.cameras
    return [(int(cameras[camera.camera].width), int(cameras[camera.camera].height)) for camera in scene.cameras]


def component_keys_at(scene: Scene, labels: recover.Labels, label: int, t: int, *, exclude_camera: int | None = None,
                      count: int = 2, tolerance: int = 15, avoid: frozenset[str] | set[str] = frozenset(),
                      clear: dict | None = None) -> list[str]:
    """Up to ``count`` detections of a component near sync index ``t`` (largest boxes, distinct cameras).

    ``avoid``: detections a person could not judge ("unsure": too little visible); another detection of the
    camera within the tolerance is taken instead where there is one. ``clear`` (the clear-view rule of
    ``RECIPE["link_ranking"]``): the detection of each camera is exchanged for the nearest clear view of the
    same piece of that camera's tracklet within the tolerance (``solve/view_choice.py``). Which cameras are
    shown, and which piece of each, does not depend on it."""
    options = []
    sizes = image_sizes(scene) if clear is not None else None
    atoms = getattr(labels, "atom", None)
    for camera in scene.cameras:
        if camera.index == exclude_camera:
            continue
        rows = np.flatnonzero(labels.label[camera.index] == label)
        rows = rows[(camera.t[rows] >= 0) & (np.abs(camera.t[rows] - t) <= tolerance)]
        if not len(rows):
            continue
        if avoid:
            judged = np.array([camera.detections.key(int(row)) not in avoid for row in rows])
            rows = rows[judged] if judged.any() else rows
        nearest = rows[np.argmin(np.abs(camera.t[rows] - t))]
        box = camera.detections.boxes[nearest]
        shown = int(nearest)
        if clear is not None:
            piece = rows if atoms is None else rows[atoms[camera.index][rows] == atoms[camera.index][nearest]]
            shown = view_choice.clearer(camera, shown, piece, sizes[camera.index], scene.joint_peak, clear, t=t)
        options.append((float((box[2] - box[0]) * (box[3] - box[1])), camera.detections.key(shown)))
    return [key for _, key in sorted(options, reverse=True)[:count]]


def rank_link_candidates(solved: Solved, embed=None, log=print) -> dict[int, list[dict]]:
    """For every track that starts inside the recording: its feasible earlier tracks, likeliest first.

    ``{later component: [{"component", "gap_s", "distance_m", "score", "motion", "appearance", "similarity",
    "compared", "speed_mps", "clear_views"}, ...]}``. The order is by ``solve/link_ranking.py``: motion, plus
    appearance where both tracks have clear views next to the gap (``embed``: ``appearance.embedder``; without
    it, or without clear views, appearance is unknown and motion alone orders). Nothing is decided here."""
    recipe = RECIPE["link_ranking"]
    scene, labels = solved.scene, solved.labels
    summaries = {s.label: s for s in solved.summaries if s is not None}
    starts = [end for end in solved.links.open_ends if end["end"] == "start" and end["candidates"]]
    window = int(round(recipe["window_s"] * 10))
    views: dict[tuple[int, str], tuple[dict, dict]] = {}
    if embed is not None and starts:
        index, sizes = link_ranking.ComponentRows(scene, labels.label), image_sizes(scene)
        for end in starts:
            wanted = [(end["component"], "start")] + [(option["component"], "end") for option in end["candidates"]]
            for label, side in wanted:
                if (label, side) in views or label not in summaries:
                    continue
                summary = summaries[label]
                lo, hi = (summary.t_first, summary.t_first + window) if side == "start" \
                    else (summary.t_last - window, summary.t_last)
                views[(label, side)] = link_ranking.window_views(index, label, lo, hi, recipe, sizes)
    embeddings = None
    if views:
        rows: dict[int, list] = {}
        for chosen, _ in views.values():
            for camera_index, mine in chosen.items():
                rows.setdefault(camera_index, []).append(mine)
        try:
            embeddings = embed({camera: np.unique(np.concatenate(parts)) for camera, parts in rows.items()})
        except Exception as exc:  # appearance is optional: never fail the solve over it
            log(f"appearance for ranking link candidates unavailable ({type(exc).__name__}: {exc})")
    described = {}
    if embeddings is not None:
        described = {which: link_ranking.descriptors(chosen, embeddings) for which, (chosen, _) in views.items()}
    ranked, compared = {}, 0
    for end in starts:
        later = end["component"]
        options = []
        for option in end["candidates"]:
            earlier = (option["component"], "end")
            found = link_ranking.similarity(described.get((later, "start")) or {}, described.get(earlier) or {})
            compared += found is not None
            clear = [views[which][1]["clear"] if which in views else 0 for which in ((later, "start"), earlier)]
            options.append({**option, **link_ranking.score(option["gap_s"], option["distance_m"], found, recipe),
                            "clear_views": clear})
        ranked[later] = sorted(options, key=lambda option: -option["score"])
    pairs = sum(len(options) for options in ranked.values())
    log(f"link candidates ranked: {len(ranked)} track starts, {pairs} candidates, appearance compared for {compared} "
        f"(both tracks have clear views)")
    return ranked


def measured_apart(first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray], *,
                   min_common: int = 5, apart_m: float = 0.5) -> dict | None:
    """Two tracks, each (sync indices, roots in inches), measured at the same instants in different places:
    {"seconds", "distance_m"} - they are two people - or None when that cannot be said (too few common
    instants, or they are at one place: one person tracked twice)."""
    common, ia, ib = np.intersect1d(first[0], second[0], return_indices=True)
    if len(common) < min_common:
        return None
    distance = float(np.median(np.linalg.norm(first[1][ia] - second[1][ib], axis=1))) * 0.0254
    if distance <= apart_m:
        return None
    return {"seconds": round(len(common) / 10.0, 1), "distance_m": round(distance, 2)}


def apply_image_answers(scene: Scene, constraints: Constraints) -> None:
    """What reviewers said about single detections, applied to the scene the solver works on.

    * bad pose: the keypoints are wrong, so they give no rays - the detection keeps its place in its
      tracklet and its identity, but takes no part in triangulation or in cross-camera geometry;
    * the worse pose of a duplicate never wins the camera's slot of that person and instant.
    """
    for camera_index, row in constraints.bad_pose:
        camera = scene.cameras[camera_index]
        if camera.joint_ok is not None:
            camera.joint_ok[row] = False
            camera.rays[row] = np.nan
    for camera in scene.cameras:
        camera.demoted = None
    for camera_index, row in constraints.demoted:
        camera = scene.cameras[camera_index]
        if camera.demoted is None:
            camera.demoted = np.zeros(len(camera.kept), dtype=bool)
        camera.demoted[row] = True


LEFT_BEHIND = {"reach_frames": 150, "min_detections": 5}  # which questions are asked, not what a label is


def left_behind(solved: Solved, constraints: Constraints, seen: set | None = None) -> list[dict]:
    """A person said a detection is not the person of a track ("someone else"), and its piece left that track -
    but other detections of the same single-camera tracklet, right next to it, are still in the track.

    The single-camera tracker followed one box through all of them. Either it changed person on the way
    (where is a continuity question; its answer cuts the tracklet), or all of them show the flagged
    detection's person and belong out of the track with it. Looked at: the tracker's own tracklet (not the
    solver's pieces of it), ``reach_frames`` around the detection, and never across a cut a person already
    confirmed - beyond it the tracklet is known to show somebody else."""
    scene, labels, identity = solved.scene, solved.labels, solved.identity_of_label
    seen = set() if seen is None else seen
    reach, minimum = LEFT_BEHIND["reach_frames"], LEFT_BEHIND["min_detections"]
    found = []

    def identities(rows) -> set[int]:
        result = set()
        for camera_index, row in rows:
            label = int(labels.label[camera_index][row])
            if label >= 0:
                result.add(int(identity[label]))
        return result

    # Only flags on single images. A verdict "group A is not group B" whose images share a tracklet with the
    # other track is a contradiction too, but a different question (which of the two statements gives way).
    for said, track_rows in constraints.flagged:
        track = identities(track_rows)
        if not track:
            continue
        for camera_index, row in said:
            camera = scene.cameras[camera_index]
            source = getattr(camera, "source_tracklet", camera.tracklet)
            tracklet, t = int(source[row]), int(camera.t[row])
            if tracklet < 0 or t < 0 or (camera_index, tracklet) in seen or identities([(camera_index, row)]) & track:
                continue  # (still in the track: its piece could not be separated; the rule above asks)
            rows = np.flatnonzero((source == tracklet) & (camera.t >= 0))
            lo, hi = t - reach, t + reach
            for cut in constraints.cuts.get(tracklet, []):  # a confirmed change of person bounds the search
                if cut * WINDOW > t:
                    hi = min(hi, cut * WINDOW - 1)
                else:
                    lo = max(lo, cut * WINDOW)
            rows = rows[(camera.t[rows] >= lo) & (camera.t[rows] <= hi)]
            label = labels.label[camera_index][rows]
            still = rows[(label >= 0) & np.isin(identity[np.clip(label, 0, None)], sorted(track))]
            if len(still) < minimum:
                continue
            seen.add((camera_index, tracklet))
            first_t, last_t = min(t, int(camera.t[still].min())), max(t, int(camera.t[still].max()))
            span = rows[(camera.t[rows] >= first_t) & (camera.t[rows] <= last_t)]
            span = span[np.argsort(camera.t[span], kind="stable")]
            picks = span[np.unique(np.linspace(0, len(span) - 1, min(10, len(span))).round().astype(int))]
            if row not in picks:  # the flagged detection is always one of the crops
                picks[np.argmin(np.abs(camera.t[picks] - t))] = row
                picks = picks[np.argsort(camera.t[picks], kind="stable")]
            keys = [camera.detections.key(int(pick)) for pick in picks]
            flagged = camera.detections.key(int(row))
            found.append({
                "subject": f"tracklet_purity:{keys[0]}|{keys[-1]}", "severity": float(len(still)),
                "detail": {"camera": camera.camera, "t": int((first_t + last_t) // 2), "span_t": [first_t, last_t],
                           "detections": int(len(span)), "left_in_track": int(len(still)), "keys": keys,
                           "flagged_key": flagged, "flagged_crop": keys.index(flagged) + 1,
                           "why": f"a reviewer said crop {keys.index(flagged) + 1} is not the person of the track it "
                                  f"was shown in, and its piece left that track - but {len(still)} detections of the "
                                  "same single-camera track next to it are still in it. Either the single-camera "
                                  "track changes person here, or all of these crops show the flagged crop's "
                                  "person and they all leave the track",
                           "resolve_by": "a human clicks the first crop that shows a different person than crop 1, "
                                         "or says they all show one person"}})
    return found


def human_unknowns(solved: Solved, constraints: Constraints) -> dict[str, list[dict]]:
    """Questions that follow from what reviewers flagged on single images.

    tracklet_purity   a person said two detections are different people, and both lie in one piece of a
                      single-camera tracklet: the tracklet changes person somewhere between them. Where is a
                      continuity question; its answer cuts the tracklet. Also (``left_behind``): the flagged
                      detection's piece left the track, but its neighbours in the same tracklet did not.
    pose_quality      a person reported a pose as wrong: open until it is corrected.
    """
    scene, labels = solved.scene, solved.labels
    purity, seen = [], set()
    for first, second in constraints.different:
        for camera_a, row_a in first:
            for camera_b, row_b in second:
                if camera_a != camera_b or row_a == row_b:
                    continue
                camera = scene.cameras[camera_a]
                atom = int(labels.atom[camera_a][row_a])
                if atom < 0 or atom != int(labels.atom[camera_b][row_b]) or (camera_a, atom) in seen:
                    continue
                t_a, t_b = int(camera.t[row_a]), int(camera.t[row_b])
                if t_a < 0 or t_b < 0 or abs(t_a - t_b) < WINDOW:
                    continue  # a cut works on 2 s windows: these two cannot be separated by one
                seen.add((camera_a, atom))
                rows = np.flatnonzero(labels.atom[camera_a] == atom)
                rows = rows[(camera.t[rows] >= min(t_a, t_b)) & (camera.t[rows] <= max(t_a, t_b))]
                rows = rows[np.argsort(camera.t[rows], kind="stable")]
                picks = rows[np.unique(np.linspace(0, len(rows) - 1, min(10, len(rows))).round().astype(int))]
                keys = [camera.detections.key(int(row)) for row in picks]
                purity.append({
                    "subject": f"tracklet_purity:{keys[0]}|{keys[-1]}", "severity": float(len(rows)),
                    "detail": {"camera": camera.camera, "t": int((t_a + t_b) // 2), "span_t": [min(t_a, t_b), max(t_a, t_b)],
                               "detections": int(len(rows)), "keys": keys,
                               "why": "a reviewer marked the first and the last of these crops as different people, "
                                      "and both belong to one single-camera track: the track changes person "
                                      "somewhere in between",
                               "resolve_by": "a human clicks the first crop that shows the other person"}})
    purity.extend(left_behind(solved, constraints))
    quality = []
    for camera_index, row in sorted(set(constraints.bad_pose)):
        camera = scene.cameras[camera_index]
        key = camera.detections.key(int(row))
        quality.append({
            "subject": f"pose_quality:{key}", "severity": 1.0,
            "detail": {"camera": camera.camera, "key": key, "t": int(camera.t[row]),
                       "why": "a reviewer reported this pose as wrong; its keypoints are left out of 3D",
                       "resolve_by": "pose correction (the CVAT hand-off is not built yet: the pose stays reported)"}})
    return {"tracklet_purity": purity, "pose_quality": quality}


def extract_unknowns(solved: Solved, resolved_keys: set[str], *, ranked: dict[int, list[dict]] | None = None,
                     avoid: frozenset[str] | set[str] = frozenset()) -> dict[str, list[dict]]:
    """``ranked``: ``rank_link_candidates`` (without it, candidates are ordered by distance). ``avoid``:
    detections a person left out as "unsure"; other images are shown where there are any."""
    scene, labels, association = solved.scene, solved.labels, solved.association
    ranked = ranked or {}

    def clear_for(anchor: str) -> dict | None:
        """The clear-view rule for the images of the question anchored on this detection (None: a control)."""
        return view_choice.rule(RECIPE["task_images"], RECIPE["link_ranking"], anchor)
    cams = recover.camera_counts(scene, labels)
    multi = {label for label, cameras in cams.items() if len(cameras) >= 2}
    identity = solved.identity_of_label
    unknowns: dict[str, list[dict]] = {"association": [], "unassigned": [], "identity_link": [], "pose_ownership": [],
                                       "tracklet_purity": [], "coverage": []}

    # Hypotheses per free atom: partners from ambiguous pairwise edges and from projection voting.
    atom_label = {}
    for atom, rows in enumerate(association.atom_rows):
        if len(rows):
            atom_label[atom] = int(labels.label[association.atom_camera[atom]][rows[0]])
    hypotheses: dict[int, dict[int, dict]] = {}
    for edge in association.solution.ambiguous:
        for atom, partner in ((edge["a"], edge["b"]), (edge["b"], edge["a"])):
            partner_label = atom_label.get(partner, -1)
            if partner_label in multi and atom_label.get(atom, -1) not in multi:
                hypotheses.setdefault(atom, {})[partner_label] = {
                    "evidence": "two-view geometry", "share_within_gate": round(edge["f2"], 2), "samples": edge["n"],
                    "why_open": edge["reason"]}
    for event in solved.log:
        for entry in event.get("atoms_ambiguous", []):
            for option in entry["options"]:
                if option["label"] in multi:
                    hypotheses.setdefault(entry["atom"], {})[option["label"]] = {
                        "evidence": "projected 3D pose", "frames_matching": option["matches"],
                        "frames_compared": option["comparable"], "why_open": entry.get("reason", "competing matches")}

    short = {camera.camera: 0 for camera in scene.cameras}
    # Where in each image do 3D-tracked people appear? A single-camera piece that stays in image
    # cells no 3D-tracked person ever enters (a doorway into another room, a mirror, a desk
    # behind glass) cannot be measured in 3D at all: these are grouped into one scope decision
    # per camera instead of one question per piece.
    grid_w, grid_h = 40, 30
    visited = []
    for camera in scene.cameras:
        mask = np.zeros((grid_h, grid_w), dtype=bool)
        lab = labels.label[camera.index]
        in_track = np.flatnonzero((lab >= 0) & np.isin(lab, list(multi)))
        if len(in_track):
            boxes = camera.detections.boxes[in_track]
            width = float(scene.workspace.dataset.cameras[camera.camera].width)
            height = float(scene.workspace.dataset.cameras[camera.camera].height)
            gx = np.clip(((boxes[:, 0] + boxes[:, 2]) / 2 / width * grid_w).astype(int), 0, grid_w - 1)
            gy = np.clip(((boxes[:, 1] + boxes[:, 3]) / 2 / height * grid_h).astype(int), 0, grid_h - 1)
            mask[gy, gx] = True
            grown = mask.copy()  # one cell of tolerance around visited cells
            grown[1:, :] |= mask[:-1, :]
            grown[:-1, :] |= mask[1:, :]
            grown[:, 1:] |= mask[:, :-1]
            grown[:, :-1] |= mask[:, 1:]
            mask = grown
        visited.append(mask)
    outside: dict[str, dict] = {}
    loose: dict[int, list[dict]] = {}
    for atom, rows in enumerate(association.atom_rows):
        camera_index = int(association.atom_camera[atom])
        camera = scene.cameras[camera_index]
        rows = rows[(labels.atom[camera_index][rows] == atom) & (camera.t[rows] >= 0)]
        if not len(rows) or atom_label.get(atom, -1) in multi:
            continue
        if (labels.duplicate_of[camera_index][rows] >= 0).mean() > 0.5:
            continue  # a second detection of an already tracked person: nothing to ask
        # ``keys``: the detections the question is anchored on (its closure is recognised by them);
        # ``shown``: the clearer frames of the same piece a person is shown and answers about.
        keys = representative_keys(scene, camera_index, rows)
        subject_key = camera.detections.key(int(rows[0]))
        clear = clear_for(subject_key)
        shown = representative_keys(scene, camera_index, rows, clear=clear) if clear else keys
        if not keys or resolved_keys & (set(keys) | set(shown)):
            continue
        middle = int(camera.t[rows[len(rows) // 2]])
        options = hypotheses.get(atom, {})
        if options:
            candidates = [{"component": int(label), "identity": int(identity[label]), **info,
                           "keys": component_keys_at(scene, labels, label, middle, clear=clear)}
                          for label, info in sorted(options.items(), key=lambda kv: -kv[1].get("samples", kv[1].get("frames_matching", 0)))]
            candidates = [c for c in candidates if c["keys"]]
            held_back, candidates = max(0, len(candidates) - 4), candidates[:4]
            if candidates:
                unknowns["association"].append({
                    "subject": f"association:{subject_key}", "severity": float(len(rows)),
                    "detail": {"camera": camera.camera, "detections": int(len(rows)), "t": middle,
                               "span_t": [int(camera.t[rows[0]]), int(camera.t[rows[-1]])],
                               "reference_keys": shown, "candidates": candidates, "more_candidates": held_back,
                               "image_choice": "clear" if clear else "nearest",
                               "resolve_by": "a human picks which candidate is the same person, or none"}})
                continue
        if len(rows) >= MIN_TASK_DETECTIONS and f"policy:review_individually:{camera.camera}" not in resolved_keys:
            boxes = camera.detections.boxes[rows]
            size = scene.workspace.dataset.cameras[camera.camera]
            gx = np.clip(((boxes[:, 0] + boxes[:, 2]) / 2 / size.width * grid_w).astype(int), 0, grid_w - 1)
            gy = np.clip(((boxes[:, 1] + boxes[:, 3]) / 2 / size.height * grid_h).astype(int), 0, grid_h - 1)
            if visited[camera_index][gy, gx].mean() <= 0.1:
                group = outside.setdefault(camera.camera, {"pieces": 0, "detections": 0, "examples": []})
                group["pieces"] += 1
                group["detections"] += int(len(rows))
                group["examples"].append((int(len(rows)), keys[len(keys) // 2]))
                continue
        loose.setdefault(camera_index, []).append({
            "rows": rows, "keys": keys, "shown": shown, "t0": int(camera.t[rows[0]]), "t1": int(camera.t[rows[-1]]),
            "first_box": camera.detections.boxes[rows[0], :4], "last_box": camera.detections.boxes[rows[-1], :4]})
    # Consecutive single-camera pieces of someone who stays in place (a tracker id change or a
    # cut, the box reappearing where it was within 5 s) are one question, not one per piece.
    for camera_index, pieces in loose.items():
        camera = scene.cameras[camera_index]
        pieces.sort(key=lambda piece: piece["t0"])
        chains: list[list[dict]] = []
        for piece in pieces:
            for chain in chains:
                tail = chain[-1]
                if 0 <= piece["t0"] - tail["t1"] <= 50 and _iou(tail["last_box"], piece["first_box"]) >= 0.5:
                    chain.append(piece)
                    break
            else:
                chains.append([piece])
        for chain in chains:
            total = int(sum(len(piece["rows"]) for piece in chain))
            if total < MIN_TASK_DETECTIONS:
                short[camera.camera] += total
                continue
            largest = sorted(chain, key=lambda piece: -len(piece["rows"]))[:4]
            unknowns["unassigned"].append({
                "subject": f"unassigned:{camera.detections.key(int(chain[0]['rows'][0]))}", "severity": float(total),
                "detail": {"camera": camera.camera, "detections": total, "pieces": len(chain),
                           "t": int((chain[0]["t0"] + chain[-1]["t1"]) // 2), "span_t": [chain[0]["t0"], chain[-1]["t1"]],
                           "reference_keys": [piece["shown"][len(piece["shown"]) // 2] for piece in
                                              sorted(largest, key=lambda piece: piece["t0"])],
                           # one key per piece: an answer covers the whole chain
                           "member_keys": [piece["keys"][len(piece["keys"]) // 2] for piece in chain],
                           "why": "seen by one camera only, with no geometric hypothesis linking it to a tracked person",
                           "resolve_by": "a human says whether it is a person (single-view track) or a false detection"}})
    for camera_name, group in outside.items():
        if f"policy:outside_coverage:{camera_name}" in resolved_keys:
            continue
        examples = [key for _, key in sorted(group["examples"], reverse=True)[:4]]
        unknowns["unassigned"].append({
            "subject": f"unassigned:outside_coverage:{camera_name}", "severity": float(group["detections"]),
            "detail": {"camera": camera_name, "detections": group["detections"], "pieces": group["pieces"],
                       "policy_question": True, "policy": "outside_coverage", "reference_keys": examples,
                       "why": "people seen only by this camera, in a part of its image where nobody tracked in 3D "
                              "ever appears (for example through a doorway into another room): no second view "
                              "exists, so they cannot be measured in 3D",
                       "resolve_by": "one operator decision for the whole group: out of scope, keep as "
                                     "single-camera 2D tracks, or review piece by piece"}})
    for camera_name, count in short.items():
        if count and f"policy:short:{camera_name}" not in resolved_keys:
            unknowns["unassigned"].append({
                "subject": f"unassigned:short:{camera_name}", "severity": float(count),
                "detail": {"camera": camera_name, "detections": count, "policy_question": True,
                           "why": f"pieces shorter than {MIN_TASK_DETECTIONS / 10:.0f} s seen only by this camera; "
                                  "too short to ask about one by one",
                           "resolve_by": "an operator decision: leave them out of the tracks (undeterminable) or review a sample"}})

    # Identity: components that start inside the recording without a predecessor.
    summaries = {s.label: s for s in solved.summaries if s is not None}
    identity_size: dict[int, int] = {}
    identity_span: dict[int, list[int]] = {}  # first and last sync index at which the identity has a detection
    for summary in summaries.values():
        who = int(identity[summary.label])
        identity_size[who] = identity_size.get(who, 0) + summary.detections
        span = identity_span.setdefault(who, [summary.t_first, summary.t_last])
        span[0], span[1] = min(span[0], summary.t_first), max(span[1], summary.t_last)
    frame_identity = identity[solved.frames.label]
    frame_measured = np.isfinite(solved.frames.root[:, 0])
    tracks: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def track_of(who: int) -> tuple[np.ndarray, np.ndarray]:
        """Measured positions of an identity: (sync indices, roots), one per instant."""
        if who not in tracks:
            rows = np.flatnonzero(frame_measured & (frame_identity == who))
            times, first = np.unique(solved.frames.t[rows], return_index=True)
            tracks[who] = (times, solved.frames.root[rows][first].astype(np.float64))
        return tracks[who]
    brief = {"count": 0, "detections": 0, "examples": []}
    # Links only the recovered positions make decidable, not applied (RECIPE["relink"]["apply"]): later component -> link
    proposed = {link["to"]: link for event in solved.log for link in event.get("links_proposed_after_recovery", [])}
    for end in solved.links.open_ends:
        if end["end"] != "start":
            continue
        label = end["component"]
        # ``anchor``: what the question is recognised by; ``keys``: what is shown and answered about.
        anchor = component_keys_at(scene, labels, label, end["t"] + 5, count=3, tolerance=40)
        if not anchor:
            continue
        clear = clear_for(anchor[0])
        keys = anchor
        if clear or avoid & set(anchor):
            keys = component_keys_at(scene, labels, label, end["t"] + 5, count=3, tolerance=40, avoid=avoid, clear=clear)
        if resolved_keys & (set(anchor) | set(keys)):
            continue
        size = identity_size.get(int(identity[label]), 0)
        if size < MIN_IDENTITY_DETECTIONS and "policy:review_short_identities" not in resolved_keys:
            brief["count"] += 1
            brief["detections"] += size
            brief["examples"].append((size, anchor[0]))
            continue
        candidates = []
        # Likeliest first (motion, and appearance where both tracks have clear views); the four best are shown.
        for option in ranked.get(label) or end["candidates"]:
            if len(candidates) >= 4:
                break
            other = summaries.get(option["component"])
            if other is None:
                continue
            other_keys = component_keys_at(scene, labels, other.label, other.t_last - 5, count=3, tolerance=40,
                                           avoid=avoid, clear=clear)
            if other_keys:
                candidates.append({"component": int(other.label), "identity": int(identity[other.label]),
                                   "gap_s": option["gap_s"], "distance_m": option["distance_m"], "keys": other_keys,
                                   "tracked_t": identity_span.get(int(identity[other.label])),
                                   **{name: option[name] for name in ("score", "motion", "appearance", "similarity",
                                                                      "compared", "speed_mps", "clear_views")
                                      if name in option}})
        proposal = proposed.get(label)
        if proposal is not None and proposal["from"] in summaries:
            other = summaries[proposal["from"]]
            other_keys = component_keys_at(scene, labels, other.label, other.t_last - 5, count=3, tolerance=40,
                                           avoid=avoid, clear=clear)
            if other_keys:
                scored = next((option for option in ranked.get(label) or [] if option["component"] == other.label), {})
                candidates = [c for c in candidates if c["component"] != other.label]
                candidates.insert(0, {**{name: scored[name] for name in ("appearance", "similarity", "compared", "clear_views")
                                         if name in scored},
                                      "component": int(other.label), "identity": int(identity[other.label]),
                                      "gap_s": proposal["gap_s"], "distance_m": proposal["distance_m"],
                                      "keys": other_keys, "proposed_after_recovery": True,
                                      "rule": proposal["source"].split(":")[-1],
                                      "tracked_t": identity_span.get(int(identity[other.label]))})
            else:
                proposal = None
        candidates = candidates[:4]
        # A person whose track broke several times has several earlier tracks: more than one candidate can be
        # the same person. Two candidates measured at the same instants in different places cannot.
        cannot_both = []
        for i, first in enumerate(candidates):
            for second in candidates[i + 1:]:
                apart = measured_apart(track_of(first["identity"]), track_of(second["identity"]))
                if apart is not None:
                    cannot_both.append({"components": [first["component"], second["component"]], **apart})
        unknowns["identity_link"].append({
            "subject": f"identity_link:{anchor[0]}", "severity": float(summaries[label].detections),
            "detail": {"component": int(label), "identity": int(identity[label]), "t": int(end["t"]),
                       "image_choice": "clear" if clear else "nearest",
                       "reference_keys": keys, "candidates": candidates, "cannot_both": cannot_both,
                       "feasible_candidates": len(end["candidates"]),
                       "why": "this person appears inside the recording; "
                              + ("several earlier people could have reached this spot" if len(end["candidates"]) > 1
                                 else "one earlier person could have reached this spot but not within the automatic gate"
                                 if end["candidates"] else "nobody tracked earlier could have reached this spot")
                              + ("; with the positions recovered around both tracks, the first candidate is the only "
                                 "one that fits (gap and distance are measured from those positions) - proposed, not "
                                 "applied: such links have joined two people when the earlier person was still "
                                 "there, untracked" if proposal is not None else ""),
                       "proposed_after_recovery": proposal is not None,
                       "resolve_by": "a human picks the earlier person, or says it is someone not tracked before"}})

    if brief["count"] and "policy:short_identities" not in resolved_keys:
        unknowns["identity_link"].append({
            "subject": "identity_link:short", "severity": float(brief["detections"]),
            "detail": {"policy": "short_identities", "tracks": brief["count"], "detections": brief["detections"],
                       "reference_keys": [key for _, key in sorted(brief["examples"], reverse=True)[:4]],
                       "why": f"{brief['count']} short 3D tracks (under {MIN_IDENTITY_DETECTIONS} detections each) start "
                              "inside the recording without a unique continuation; each is a correct but brief track "
                              "whose person is not established",
                       "resolve_by": "one operator decision: leave them as separate short tracks, or review each"}})

    # Two components measured at the same place that share a camera at the same time: duplicate poses.
    seen = set()
    for event in solved.log:
        for refused in event.get("merge_refused", []):
            if refused.get("reason") != "same_camera_overlap":
                continue
            a, b = sorted((refused["a"], refused["b"]))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            t = summaries[a].t_first if a in summaries else 0
            for camera in scene.cameras:
                rows_a = np.flatnonzero(labels.label[camera.index] == a)
                rows_b = np.flatnonzero(labels.label[camera.index] == b)
                shared = np.intersect1d(camera.t[rows_a], camera.t[rows_b])
                shared = shared[shared >= 0]
                if not len(shared):
                    continue
                t = int(shared[len(shared) // 2])
                key_a = camera.detections.key(int(rows_a[camera.t[rows_a] == t][0]))
                key_b = camera.detections.key(int(rows_b[camera.t[rows_b] == t][0]))
                if resolved_keys & {key_a, key_b}:
                    break
                unknowns["pose_ownership"].append({
                    "subject": f"pose_ownership:{key_a}|{key_b}", "severity": float(len(shared)),
                    "detail": {"camera": camera.camera, "t": t, "keys": [key_a, key_b],
                               "components": [int(a), int(b)], "frames_shared": int(len(shared)),
                               "median_distance_in": refused["median_in"],
                               "why": "two tracked people are measured at the same place while one camera shows two poses",
                               "resolve_by": "a human says whether the two poses are one person (duplicate) or two"}})
                break
    return unknowns


def save_solution(context: Context, solved: Solved, notes: dict | None = None) -> dict[str, str]:
    scene = solved.scene
    directory = context.workspace.round_dir(context.round)
    arrays = {"cameras": np.array([c.camera for c in scene.cameras]), "component_identity": solved.identity_of_label}
    multi = multi_camera(scene, solved.labels)
    for camera in scene.cameras:
        component = solved.labels.label[camera.index]
        identity = np.where((component >= 0) & multi[np.clip(component, 0, None)],
                            solved.identity_of_label[np.clip(component, 0, None)], -1).astype(np.int32)
        arrays[f"component_{camera.camera}"] = component
        arrays[f"identity_{camera.camera}"] = identity
        arrays[f"source_{camera.camera}"] = solved.labels.source[camera.index]
        arrays[f"atom_{camera.camera}"] = solved.labels.atom[camera.index]  # tracklet piece per row, -1 if none
        duplicate = solved.labels.duplicate_of[camera.index]
        arrays[f"duplicate_of_{camera.camera}"] = np.where(  # identity this row is a second detection of
            duplicate >= 0, solved.identity_of_label[np.clip(duplicate, 0, None)], -1).astype(np.int32)
    np.savez_compressed(directory / "solution.npz", **arrays)
    frames = solved.frames
    keep = multi[frames.label]
    np.savez_compressed(
        directory / "tracks3d.npz", component=frames.label[keep], identity=solved.identity_of_label[frames.label[keep]],
        t=frames.t[keep], rows=frames.rows[keep], xyz=frames.xyz[keep], used=frames.used[keep],
        residual=frames.residual[keep], root=frames.root[keep], aligned_time=scene.aligned_time,
        native=scene.native, cameras=np.array([c.camera for c in scene.cameras]))
    log = {"events": solved.log, "links": solved.links.accepted, "timings": solved.timings,
           "switch_events": solved.association.switch_events,
           "accepted_edges": len(solved.association.solution.accepted), **(notes or {})}
    (directory / "solve.json").write_text(json.dumps(log, indent=1, default=int), encoding="utf-8")
    root = context.workspace.root
    return {name: str((directory / name).relative_to(root)) for name in ("solution.npz", "tracks3d.npz", "solve.json")}


def run(context: Context) -> StageResult:
    workspace = context.workspace
    scene = load_scene(workspace, joint_peak=RECIPE["geometry"]["joint_peak"])
    constraints, info = load_constraints(context, scene)
    evidence = workspace.path("evidence", "autofit", "labels.npz")
    if evidence.is_file():
        with np.load(evidence) as data:
            constraints.external = [
                data[f"raw_{camera.camera}"].astype(np.int64) if f"raw_{camera.camera}" in data.files
                else np.full(len(camera.kept), -1, dtype=np.int64) for camera in scene.cameras]
        info["used"]["autofit_tracks"] = int(sum(len(np.unique(e[e >= 0])) for e in constraints.external))
        context.log(f"using Autofit raw tracks as independent evidence ({info['used']['autofit_tracks']} track ids)")
    else:
        context.log("no Autofit evidence: geometry-only solve (expect many more questions)", level="warning")
    # Evidence at every sync frame up to about an hour (the golden bench was scored that way);
    # only longer recordings subsample to 5 Hz to bound memory.
    long_recording = scene.T > 40000
    step = 2 if long_recording else 1
    from ..solve import alignment_check

    apply_image_answers(scene, constraints)
    units, unit_stats = unit_scene(scene, RECIPE, constraints)
    key = association_key(context, step)
    state = load_state(context, key)
    if state is None:
        associated = associate_and_refine(units, RECIPE, constraints, step=step, log=context.log, unit_stats=unit_stats)
        # Alignment: the generator's "stale" spans are replaced by a direct geometric check.
        checked = alignment_check.verify(scene, RECIPE)
        save_state(context, key, associated, checked)
    else:
        associated, checked = state
        context.log("detections, calibration, alignment and constraints are unchanged since the last solve: "
                    "its association is reused; identity, the swap check and the open questions are decided again")
    recovered = recovered_positions(context, units, associated)
    solved = identify(units, associated, RECIPE, constraints, recovered=recovered, log=context.log)
    identity_event = next(event for event in reversed(solved.log) if event.get("step") == "identity")
    after_recovery = identity_event.get("links_after_recovery", [])
    proposed_links = identity_event.get("links_proposed_after_recovery", [])
    outputs = save_solution(context, solved, {
        "association_reused": state is not None,
        "recovered": {"positions_sha256": recovered["positions_sha256"], "count": recovered["count"]} if recovered else None})
    from .. import appearance
    from ..tasks.carry import flagged_unsure

    try:
        embed_links = appearance.embedder(workspace, [camera.camera for camera in scene.cameras], context.log,
                                          directory="appearance_links", incremental=True)
    except Exception as exc:  # optional evidence: never fail the solve over it
        context.log(f"appearance for ranking link candidates unavailable ({exc})", level="warning")
        embed_links = None
    ranked = rank_link_candidates(solved, embed_links, context.log)
    answered = context.db.tasks(status="answered") + context.db.tasks(status="applied")
    unknowns = extract_unknowns(solved, info["resolved_keys"], ranked=ranked, avoid=flagged_unsure(answered))
    unknowns.update(human_unknowns(solved, constraints))
    (workspace.round_dir(context.round) / "alignment_check.json").write_text(json.dumps(checked, indent=1), encoding="utf-8")
    unknowns["alignment"] = []
    decided = {c["a"][0] for c in context.db.constraints("alignment") if c["a"]}
    for camera in scene.cameras:
        entries = checked.get(camera.camera)
        if entries is None:
            if camera.camera not in decided:
                unknowns["alignment"].append({
                    "subject": f"alignment:{camera.camera}", "severity": float(camera.detections.frame_count),
                    "detail": {"camera": camera.camera, "reasons": ["no resolved time offset: the camera has no "
                                                                     "frames on the common timeline"],
                               "explanation": f"{camera.camera} could not be tied to the other cameras in time, so "
                                              "none of its detections take part in 3D."}})
            continue
        for entry in entries:
            if entry["verdict"] == "disagrees" and camera.camera not in decided:
                start, end = entry["t"]
                shift = f"{entry['best_lag']} frame(s)" if entry.get("peak_inside_scan", True) else \
                    f"{entry['best_lag']} frame(s) or more (the edge of the scan)"
                boundaries = _file_boundaries(scene, camera)
                near = [name for sync, name in boundaries if sync <= end]
                hint = (f" The camera's file {near[-1]} starts at or before this block; a shift that begins at a file "
                        "boundary is a gap or overlap between the two files.") if near else ""
                unknowns["alignment"].append({
                    "subject": f"alignment:{camera.camera}:{start}-{end}", "severity": float(end - start),
                    "detail": {"camera": camera.camera, "block_sync_frames": entry["t"], "best_lag_frames": entry["best_lag"],
                               "agreeing_pairs_by_lag": entry["counts"], "lags": entry.get("lags"),
                               "shift_measured": bool(entry.get("peak_inside_scan", True)),
                               "file_boundaries_sync": [[sync, name] for sync, name in boundaries],
                               "reasons": [f"cross-camera geometry agrees best when this camera is shifted by {shift}"],
                               "explanation": f"Between {start / 10:.0f} s and {end / 10:.0f} s, {camera.camera}'s "
                                              f"detections line up with the other cameras better at a shift of "
                                              f"{shift} than at the estimated offset.{hint}"}})
    verdicts = [e["verdict"] for entries in checked.values() for e in entries]
    alignment_summary = {v: verdicts.count(v) for v in set(verdicts)}
    context.log(f"alignment check by geometry: {alignment_summary}")
    # Identity swaps: every moment two tracks could have exchanged people, tested with cues that do
    # not come from the 3D tracker. ``scene`` still has the 2D tracker's own tracklets (the solver
    # works on a copy cut at Autofit id changes).
    swaps = swap_unknowns(context, scene, solved, constraints, info["resolved_keys"], recovered)
    unknowns["identity_swap"] = swaps["unknowns"]
    context.log(f"swap check: {swaps['summary']}")
    coverage = coverage_metrics(solved)
    metrics = {"coverage": coverage, "constraints_used": info["used"], "alignment_check": alignment_summary,
               "swap_check": swaps["summary"], "association_reused": state is not None,
               "identity_after_recovery": {
                   "recovered_positions_used": identity_event.get("recovered_positions_used", 0),
                   "links": len(after_recovery), "links_proposed": len(proposed_links),
                   "open_ends": [identity_event.get("open_ends_on_measured_positions", identity_event["open_ends"]),
                                 identity_event["open_ends"]]},
               "refinement": [{k: v for k, v in e.items() if not isinstance(v, list)} for e in solved.log],
               "timings_s": {k: round(v, 1) for k, v in solved.timings.items()},
               "open_unknowns": {kind: len(items) for kind, items in unknowns.items()}}
    message = (f"{coverage['identities']} identities, {coverage['share_detections_in_3d_tracks']:.0%} of detections "
               f"in 3D tracks, {coverage['person_frames_with_3d_root']} person-frames with 3D")
    if state is not None:
        message += "; association reused, identity decided again"
    if recovered and RECIPE["relink"].get("apply"):
        message += f"; {len(after_recovery)} tracks linked through recovered positions"
    elif recovered:
        message += f"; {len(proposed_links)} links proposed for review through recovered positions"
    return StageResult(metrics=metrics, message=message, outputs=outputs,
                       inputs={"constraints": info["used"], "evidence_step": 2 if long_recording else 1},
                       unknowns=unknowns)
