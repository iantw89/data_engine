"""Ranking the candidates of a re-identification question: motion, appearance on clear views, both.

    python experiments/links/ranking.py pairs golden   -> pairs_golden.json (truth: golden reference)
    python experiments/links/ranking.py pairs felix    -> pairs_felix.json  (truth: the answered tasks)
    python experiments/links/ranking.py report         -> tables from the pair files

Read-only on the workspaces except for ``pairs felix --embed``, which adds the embeddings it needs to
the workspace's appearance cache (the same rows a round would embed).
"""
from __future__ import annotations

import json
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data_engine import reid  # noqa: E402
from data_engine.config import RECIPE  # noqa: E402
from data_engine.evaluation import load_reference  # noqa: E402
from data_engine.solve import link_ranking  # noqa: E402
from data_engine.solve.identity import INCH, summarise  # noqa: E402
from data_engine.solve.pipeline import Constraints, identify  # noqa: E402
from data_engine.solve.scene import load_scene  # noqa: E402
from data_engine.store import parse_det_key  # noqa: E402
from data_engine.workspace import Workspace  # noqa: E402

OUT = Path(__file__).parent
GOLDEN = "golden_20250505071549_20250505080000@first600s"
FELIX = "felix_20251223173052_20251223183103"
STORE = Path(r"C:\Users\iantw\git\data_management\data\labels")
GOLDEN_EMBEDDINGS = Path(r"C:\data_engine_data\_scratch\reid\golden_fastreid")
WINDOW = 100  # sync frames (10 s) of a track next to the gap

# What counts as a usable crop. "any" is what the swap check uses (a box at least 48 px tall).
GATES = {
    "any": {"min_joints": 0, "max_overlap": 9.0, "min_height_px": 48, "border_px": -1.0, "crops_per_camera": 6},
    "joints13": {"min_joints": 13, "max_overlap": 9.0, "min_height_px": 48, "border_px": -1.0, "crops_per_camera": 6},
    "alone": {"min_joints": 0, "max_overlap": 0.2, "min_height_px": 48, "border_px": -1.0, "crops_per_camera": 6},
    "clear": {"min_joints": 13, "max_overlap": 0.2, "min_height_px": 96, "border_px": 4.0, "crops_per_camera": 6},
    "clear15": {"min_joints": 15, "max_overlap": 0.1, "min_height_px": 128, "border_px": 4.0, "crops_per_camera": 6},
}
# The complement of "clear": only crops with a partial skeleton, cut by the border or overlapped by someone.
GATES["unclear"] = dict(GATES["clear"], invert=True)


def window_views(index, label, lo, hi, gate, sizes, available=None):
    if not gate.get("invert"):
        return link_ranking.window_views(index, label, lo, hi, gate, sizes, available)
    scene, chosen, total = index.scene, {}, 0
    for camera in scene.cameras:
        rows = index.rows(camera.index, label, lo, hi)
        if available is not None and len(rows):
            rows = rows[available[camera.index][rows]]
        if not len(rows):
            continue
        quality = link_ranking.view_quality(camera, rows, sizes[camera.index], scene.joint_peak, gate["border_px"])
        rows = rows[~link_ranking.clear_view(quality, gate) & (quality["height"] >= 48)]
        if len(rows) > 6:
            rows = rows[np.unique(np.linspace(0, len(rows) - 1, 6).round().astype(int))]
        if len(rows):
            chosen[camera.index] = rows
            total += len(rows)
    return chosen, {"detections": total, "clear": total, "cameras": len(chosen)}


def load(ws_id: str):
    ws = Workspace(ws_id)
    scene = load_scene(ws, joint_peak=RECIPE["geometry"]["joint_peak"], rays=False)
    with open(ws.path("solve_state", "associated.pkl"), "rb") as handle:
        associated = pickle.load(handle)["associated"]
    sizes = [(int(ws.dataset.cameras[c.camera].width), int(ws.dataset.cameras[c.camera].height)) for c in scene.cameras]
    return ws, scene, associated, sizes


def table(directory: Path, scene):
    result = []
    for camera in scene.cameras:
        slot = np.full(len(camera.kept), -1, np.int32)
        vectors = np.zeros((0, 1), np.float32)
        if (directory / f"{camera.camera}.npz").is_file():
            rows, vectors = reid.load_embeddings(directory, camera.camera)
            slot[rows] = np.arange(len(rows), dtype=np.int32)
            vectors = reid.l2_normalize(vectors)
        result.append((slot, vectors))
    return result


def pair_features(index, sizes, embeddings, later, earlier, available=None) -> dict:
    """Appearance of (start of ``later``, end of ``earlier``) under every gate."""
    out = {}
    for name, gate in GATES.items():
        a, found_a = window_views(index, later.label, later.t_first, later.t_first + WINDOW, gate, sizes, available)
        b, found_b = window_views(index, earlier.label, earlier.t_last - WINDOW, earlier.t_last, gate, sizes, available)
        found = link_ranking.similarity(link_ranking.descriptors(a, embeddings), link_ranking.descriptors(b, embeddings))
        out[name] = {"views": [found_a["clear"], found_b["clear"]], "cameras": [found_a["cameras"], found_b["cameras"]],
                     "similarity": None if found is None else round(found["similarity"], 4),
                     "mode": None if found is None else found["mode"]}
    return out


def gap_distance(later, earlier) -> tuple[float, float]:
    return ((later.first_root_t - earlier.last_root_t) / 10.0,
            float(np.linalg.norm(later.first_root - earlier.last_root)) * INCH)


def pairs_golden() -> None:
    ws, scene, associated, sizes = load(GOLDEN)
    reference = load_reference(scene, STORE, GOLDEN.split("@")[0])
    embeddings = table(GOLDEN_EMBEDDINGS, scene)
    available = [slot >= 0 for slot, _ in embeddings]
    labels = associated.labels
    n_labels = labels.next_label()
    person = {}
    for label in range(n_labels):
        votes = np.concatenate([reference.person[c.index][labels.label[c.index] == label] for c in scene.cameras])
        votes = votes[votes >= 0]
        if len(votes) >= 5:
            values, counts = np.unique(votes, return_counts=True)
            if counts.max() / len(votes) >= 0.8:
                person[label] = int(values[np.argmax(counts)])
    index = link_ranking.ComponentRows(scene, labels.label)
    cases = []
    for window in (120.0, 600.0):
        recipe = json.loads(json.dumps(RECIPE))
        recipe["temporal"]["candidate_gap_s"] = window
        solved = identify(scene, associated, recipe, Constraints(), log=lambda *_: None)
        summaries = {s.label: s for s in solved.summaries if s is not None}
        for end in solved.links.open_ends:
            if end["end"] != "start" or end["component"] not in person:
                continue
            later = summaries[end["component"]]
            rows = []
            for option in end["candidates"]:
                earlier = summaries[option["component"]]
                if earlier.label not in person:
                    continue
                rows.append({"component": int(earlier.label), "gap_s": option["gap_s"], "distance_m": option["distance_m"],
                             "same": person[earlier.label] == person[later.label],
                             "appearance": pair_features(index, sizes, embeddings, later, earlier, available)})
            cases.append({"window_s": window, "component": int(later.label), "t": int(later.t_first),
                          "feasible": len(end["candidates"]), "candidates": rows})
        print(window, "open starts with a known person:", sum(1 for c in cases if c["window_s"] == window), flush=True)
    (OUT / "pairs_golden.json").write_text(json.dumps(cases), encoding="utf-8")


def pairs_felix(embed: bool) -> None:
    ws, scene, associated, sizes = load(FELIX)
    labels = associated.labels
    summaries = {s.label: s for s in summarise(associated.frames, labels.next_label()) if s is not None}
    index = link_ranking.ComponentRows(scene, labels.label)
    by_camera = {camera.camera: camera for camera in scene.cameras}

    def component(keys: list[str]) -> int | None:
        found = set()
        for key in keys:
            camera, frame, det = parse_det_key(key)
            found.add(int(labels.label[by_camera[camera].index][by_camera[camera].detections.row(frame, det)]))
        found.discard(-1)
        return found.pop() if len(found) == 1 else None

    db = sqlite3.connect(f"file:{ws.path('engine.sqlite3')}?mode=ro", uri=True)
    judged = []
    for task_id, payload, answer in db.execute("select id, payload, answer from tasks where type='reidentify' and "
                                               "status in ('answered', 'applied') and answer is not null"):
        payload, answer = json.loads(payload), json.loads(answer)
        if "images" not in answer:
            continue  # an answer of the first model: no flags on images
        out = {key for key, flags in answer["images"].items() if flags.get("exclude")}
        later = component([key for key in payload["reference_keys"] if key not in out])
        for candidate in payload["candidates"]:
            verdict = (answer.get("candidates") or {}).get(candidate["id"])
            earlier = component([key for key in candidate["keys"] if key not in out])
            if verdict not in ("same", "different") or later is None or earlier is None or later == earlier:
                continue
            if later not in summaries or earlier not in summaries or summaries[later].first_root is None \
                    or summaries[earlier].last_root is None:
                continue
            judged.append((task_id, summaries[later], summaries[earlier], verdict == "same"))
    print("judged pairs:", len(judged), "same:", sum(1 for *_, same in judged if same), flush=True)
    wanted: dict[int, list] = {}
    for _, later, earlier, _ in judged:
        for label, lo, hi in ((later.label, later.t_first, later.t_first + WINDOW), (earlier.label, earlier.t_last - WINDOW, earlier.t_last)):
            for gate in GATES.values():
                views, _ = window_views(index, label, lo, hi, gate, sizes)
                for camera_index, rows in views.items():
                    wanted.setdefault(camera_index, []).append(rows)
    wanted = {camera: np.unique(np.concatenate(parts)) for camera, parts in wanted.items()}
    print("crops needed:", {scene.cameras[c].camera: len(rows) for c, rows in wanted.items()}, flush=True)
    if embed:
        from data_engine import appearance
        embeddings = appearance.embedder(ws, [camera.camera for camera in scene.cameras], directory="appearance_links",
                                         incremental=True)(wanted)
    else:
        embeddings = table(ws.path("evidence", "appearance_links"), scene)
    cases: dict[int, dict] = {}
    for task_id, later, earlier, same in judged:
        gap, distance = gap_distance(later, earlier)
        cases.setdefault(task_id, {"task": task_id, "component": int(later.label), "candidates": []})["candidates"].append(
            {"component": int(earlier.label), "gap_s": round(gap, 2), "distance_m": round(distance, 2), "same": same,
             "appearance": pair_features(index, sizes, embeddings, later, earlier)})
    (OUT / "pairs_felix.json").write_text(json.dumps(list(cases.values())), encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1] == "pairs" and sys.argv[2] == "golden":
        pairs_golden()
    elif sys.argv[1] == "pairs" and sys.argv[2] == "felix":
        pairs_felix("--embed" in sys.argv)


# --------------------------------------------------------------------------- golden, at scale
def synthetic() -> None:
    """Appearance across a gap on the golden reference itself: every labelled person in 10 s windows, each
    window against every window at least 10 s earlier, with the crops chosen by each gate - and by the
    complement of the clear-view gate ("unclear": only partial or overlapped crops)."""
    from scipy.optimize import minimize

    ws, scene, _, sizes = load(GOLDEN)
    reference = load_reference(scene, STORE, GOLDEN.split("@")[0])
    embeddings = table(GOLDEN_EMBEDDINGS, scene)
    gates = GATES
    rows_of, quality, fragment_of = [], [], []
    for camera in scene.cameras:
        slot, _ = embeddings[camera.index]
        rows = np.flatnonzero((slot >= 0) & (reference.person[camera.index] >= 0) & (camera.t >= 0))
        rows_of.append(rows)
        quality.append(link_ranking.view_quality(camera, rows, sizes[camera.index], scene.joint_peak, 4.0))
        fragment_of.append(reference.person[camera.index][rows].astype(np.int64) * 10000 + camera.t[rows] // WINDOW)
    fragments = np.unique(np.concatenate(fragment_of))
    person, window = fragments // 10000, fragments % 10000
    later, earlier = np.nonzero(window[:, None] - window[None, :] >= 2)
    same = person[later] == person[earlier]
    print(f"fragments {len(fragments)} of {len(np.unique(person))} people; pairs {len(later)} ({int(same.sum())} same)")
    report = {}
    for name, gate in gates.items():
        per_camera, pooled, shared, n_views = [], np.zeros((len(fragments), embeddings[0][1].shape[1])), None, 0
        total = np.zeros((len(fragments), len(fragments)))
        count = np.zeros((len(fragments), len(fragments)))
        for camera in scene.cameras:
            rows, q = rows_of[camera.index], quality[camera.index]
            g = dict(gate)
            if name in ("any", "joints13", "alone"):
                q = dict(q, cut=np.zeros(len(rows), bool))
            ok = link_ranking.clear_view(q, g)
            if gate.get("invert"):
                ok = ~ok & (q["height"] >= 48)
            slot, vectors = embeddings[camera.index]
            desc = np.zeros((len(fragments), vectors.shape[1]))
            has = np.zeros(len(fragments), bool)
            ids = fragment_of[camera.index]
            for f in np.unique(ids[ok]):
                mine = rows[ok & (ids == f)]
                if len(mine) > 6:
                    mine = mine[np.unique(np.linspace(0, len(mine) - 1, 6).round().astype(int))]
                v = vectors[slot[mine]].astype(np.float64).sum(axis=0)
                i = int(np.searchsorted(fragments, f))
                desc[i], has[i] = v / max(np.linalg.norm(v), 1e-9), True
                n_views += len(mine)
            pooled += desc
            both = has[:, None] & has[None, :]
            total += np.where(both, desc @ desc.T, 0.0)
            count += both
        norm = np.linalg.norm(pooled, axis=1)
        known = norm > 1e-9
        unit = pooled / np.maximum(norm, 1e-9)[:, None]
        cross = unit @ unit.T
        sim_same = (total / np.maximum(count, 1))[later, earlier]
        is_same_camera = count[later, earlier] > 0
        is_cross = ~is_same_camera & known[later] & known[earlier]
        sim = np.where(is_same_camera, sim_same, cross[later, earlier])
        entry = {"fragments_with_views": int(known.sum()), "share_pairs_known": round(float((is_same_camera | is_cross).mean()), 3)}
        for mode, mask in (("same_camera", is_same_camera), ("cross_camera", is_cross)):
            a, b = sim[mask & same], sim[mask & ~same]
            if len(a) < 20 or len(b) < 20:
                continue
            x = np.concatenate([a, b])
            y = np.concatenate([np.ones(len(a)), np.zeros(len(b))])
            w = np.concatenate([np.full(len(a), 0.5 / len(a)), np.full(len(b), 0.5 / len(b))])  # prior odds 1: log-odds = LLR

            def loss(p):
                z = p[0] * (x - p[1])
                return float(np.sum(w * (np.logaddexp(0, z) - y * z)))
            fit = minimize(loss, [10.0, 0.5], method="Nelder-Mead").x
            entry[mode] = {"same": int(len(a)), "different": int(len(b)), "auc": round(reid.roc_auc(a, b), 4),
                           "same_mean": round(float(a.mean()), 3), "different_mean": round(float(b.mean()), 3),
                           "different_p99": round(float(np.quantile(b, 0.99)), 3), "same_p10": round(float(np.quantile(a, 0.1)), 3),
                           "slope": round(float(fit[0]), 2), "midpoint": round(float(fit[1]), 3)}
        # Gallery test: a window against all people's windows of one earlier time; is the right one first?
        hits = asked = decidable = 0
        full = np.full((len(fragments), len(fragments)), np.nan)
        full[later, earlier] = np.where(is_same_camera | is_cross, sim, np.nan)
        by_window = {w: np.flatnonzero(window == w) for w in np.unique(window)}
        for i in range(len(fragments)):
            for w, members in by_window.items():
                if window[i] - w < 2 or len(members) < 4:
                    continue
                true = members[person[members] == person[i]]
                if not len(true):
                    continue
                asked += 1
                scores = full[i, members]
                if np.isnan(full[i, true[0]]):
                    continue
                decidable += 1
                hits += int(np.nanargmax(scores) == np.flatnonzero(members == true[0])[0])
        entry["gallery"] = {"asked": asked, "true_has_appearance": round(decidable / max(asked, 1), 3),
                            "rank1_when_it_has": round(hits / max(decidable, 1), 3)}
        report[name] = entry
        print(name, json.dumps(entry), flush=True)
    (OUT / "synthetic_golden.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__" and sys.argv[1] == "synthetic":
    synthetic()


# --------------------------------------------------------------------------- report
def report() -> None:
    """How each ordering would have ranked the candidates of the questions whose answer is known."""
    recipe = RECIPE["link_ranking"]
    for name in ("felix", "golden"):
        path = OUT / f"pairs_{name}.json"
        if not path.is_file():
            continue
        cases = json.loads(path.read_text(encoding="utf-8"))
        if name == "golden":
            cases = [case for case in cases if case["window_s"] == 120.0]
        pairs = [pair for case in cases for pair in case["candidates"]]
        print(f"\n== {name}: {len(cases)} questions, {len(pairs)} judged pairs, {sum(p['same'] for p in pairs)} same")
        print("  appearance by which crops are used (same / different person):")
        for gate in GATES:
            if pairs and gate not in pairs[0]["appearance"]:
                continue
            for mode in ("same_camera", "cross_camera"):
                a = [p["appearance"][gate]["similarity"] for p in pairs if p["same"] and p["appearance"][gate]["mode"] == mode]
                b = [p["appearance"][gate]["similarity"] for p in pairs if not p["same"] and p["appearance"][gate]["mode"] == mode]
                if len(a) >= 3 and len(b) >= 3:
                    print(f"    {gate:9s} {mode:13s} n={len(a):3d}/{len(b):3d}  mean {np.mean(a):.2f} / {np.mean(b):.2f}  "
                          f"AUC {reid.roc_auc(a, b):.3f}")
            unknown = sum(1 for p in pairs if p["appearance"][gate]["mode"] is None)
            print(f"    {gate:9s} not comparable: {unknown} of {len(pairs)} pairs")

        def orderings(pair):
            found = pair["appearance"]["clear"]
            found = None if found["mode"] is None else {"similarity": found["similarity"], "mode": found["mode"]}
            anyone = pair["appearance"]["any"]
            anyone = None if anyone["mode"] is None else {"similarity": anyone["similarity"], "mode": anyone["mode"]}
            motion = link_ranking.motion_evidence(pair["gap_s"], pair["distance_m"], recipe)
            return {"distance (today)": -pair["distance_m"], "motion": motion,
                    "appearance, clear views": link_ranking.appearance_evidence(found, recipe),
                    "motion + appearance, any crop": motion + link_ranking.appearance_evidence(anyone, recipe),
                    "motion + appearance, clear views": motion + link_ranking.appearance_evidence(found, recipe)}

        scored = [[(orderings(pair), pair["same"]) for pair in case["candidates"]] for case in cases]
        mixed = [case for case in scored if any(same for _, same in case) and any(not same for _, same in case)]
        print(f"  ordering within a question ({len(mixed)} questions with both a 'same' and a 'different'):")
        for key in scored[0][0][0] if scored and scored[0] else []:
            first = sum(1 for case in mixed if max(case, key=lambda entry: entry[0][key])[1])
            wins = total = 0
            for case in mixed:
                for sa, same_a in case:
                    for sb, same_b in case:
                        if same_a and not same_b:
                            total += 1
                            wins += 1.0 if sa[key] > sb[key] else 0.5 if sa[key] == sb[key] else 0.0
            everything = [(entry[0][key], entry[1]) for case in scored for entry in case]
            auc = reid.roc_auc([s for s, same in everything if same], [s for s, same in everything if not same])
            print(f"    {key:34s} first is a 'same' in {first:2d}/{len(mixed)}; a 'same' above a 'different' "
                  f"{wins / max(total, 1):.0%} of {total} pairs; over all pairs AUC {auc:.3f}")


if __name__ == "__main__" and sys.argv[1] == "report":
    report()


# --------------------------------------------------------------------------- felix, end to end
def felix_shown() -> None:
    """The engine's own ranking (``stages/solve.py: rank_link_candidates``) over ALL feasible candidates of the
    felix questions that were answered: where do the tracks the person confirmed, and those they rejected, land?
    Embeds what it needs into the workspace's ``evidence/appearance_links`` cache."""
    from data_engine import appearance
    from data_engine.db import Database
    from data_engine.engine import Context
    from data_engine.stages import solve as stage

    ws, scene, associated, _ = load(FELIX)
    copy = OUT / "_felix_db_copy.sqlite3"
    copy.unlink(missing_ok=True)
    source = sqlite3.connect(f"file:{ws.path('engine.sqlite3')}?mode=ro", uri=True)
    target = sqlite3.connect(str(copy))
    source.backup(target)
    target.close(), source.close()
    ws.__dict__["db"] = Database(copy)  # reads (and the log lines of this run) go to a copy
    context = Context(ws, 0, "solve")
    constraints, _ = stage.load_constraints(context, scene)
    solved = identify(scene, associated, RECIPE, constraints, log=print)
    embed = appearance.embedder(ws, [camera.camera for camera in scene.cameras], print, directory="appearance_links",
                                incremental=True)
    ranked = stage.rank_link_candidates(solved, embed, print)
    labels = solved.labels
    by_camera = {camera.camera: camera for camera in scene.cameras}

    def component(keys):
        found = set()
        for key in keys:
            camera, frame, det = parse_det_key(key)
            found.add(int(labels.label[by_camera[camera].index][by_camera[camera].detections.row(frame, det)]))
        found.discard(-1)
        return found.pop() if len(found) == 1 else None

    rows = []
    for task in ws.db.tasks(task_type="reidentify"):
        answer, payload = task.get("answer") or {}, task["payload"]
        if task["status"] not in ("answered", "applied") or "images" not in answer:
            continue
        out = {key for key, flags in answer["images"].items() if flags.get("exclude")}
        later = component([key for key in payload["reference_keys"] if key not in out])
        if later is None or later not in ranked:
            continue
        order = [option["component"] for option in ranked[later]]
        by_distance = [option["component"] for option in sorted(ranked[later], key=lambda option: option["distance_m"])]
        entry = {"task": task["id"], "feasible": len(order), "same": [], "different": [], "same_by_distance": [],
                 "ranked": [{name: option.get(name) for name in ("component", "gap_s", "distance_m", "score", "motion",
                                                                 "appearance", "similarity", "compared", "clear_views")}
                            for option in ranked[later]], "verdicts": {}}
        for candidate in payload["candidates"]:
            verdict = (answer.get("candidates") or {}).get(candidate["id"])
            earlier = component([key for key in candidate["keys"] if key not in out])
            if verdict in ("same", "different") and earlier in order:
                entry[verdict].append(order.index(earlier) + 1)
                entry["verdicts"][str(earlier)] = verdict
                if verdict == "same":
                    entry["same_by_distance"].append(by_distance.index(earlier) + 1)
        entry["never_judged_in_top4"] = sum(1 for c in order[:4] if order.index(c) + 1 not in entry["same"] + entry["different"])
        rows.append(entry)
        print({name: value for name, value in entry.items() if name not in ("ranked", "verdicts")}, flush=True)
    (OUT / "felix_shown.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    ws.db.connection.close()
    copy.unlink(missing_ok=True)
    same = [rank for entry in rows for rank in entry["same"]]
    old = [rank for entry in rows for rank in entry["same_by_distance"]]
    different = [rank for entry in rows for rank in entry["different"]]
    print(f"questions {len(rows)}; confirmed 'same' tracks: {len(same)}, in the first four {sum(r <= 4 for r in same)} "
          f"(by distance {sum(r <= 4 for r in old)}), first {sum(r == 1 for r in same)} (by distance {sum(r == 1 for r in old)})")
    print(f"rejected tracks: {len(different)}, still in the first four {sum(r <= 4 for r in different)}")
    print("candidates in the new first four that were never judged:", sum(entry["never_judged_in_top4"] for entry in rows))


if __name__ == "__main__" and sys.argv[1] == "felix-shown":
    felix_shown()
