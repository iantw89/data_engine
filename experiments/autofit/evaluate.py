"""Score Autofit against the golden identity labels and test it as corroborating evidence for the engine.

Engine interpreter, PYTHONPATH=src:

    python experiments/autofit/evaluate.py --workspace golden_...@first600s --variant kept --round r000

Parts (all numbers go to <scratch>/<ws>/<variant>/evaluation.json and are printed):
  A  score(): Autofit raw ids / global ids / engine components / identities on the same reference.
  B  piece-pair co-membership: engine atoms of different cameras that overlap in time; Autofit raw/global
     co-membership vs golden truth; atom-pair level and engine-component-pair level; threshold grid chosen on the
     first half, frozen on the second.
  C  identity across gaps: engine components linked through shared Autofit raw/global ids (no simultaneity).
  D  negative evidence: overlapping atom pairs Autofit puts in different tracks.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from data_engine.evaluation import cross_camera_pairs, load_reference, score, Reference
from data_engine.solve.scene import load_scene
from data_engine.workspace import Workspace

LABELS = Path(r"C:\Users\iantw\git\data_management\data\labels")
SCRATCH = Path(r"C:\data_engine_data\_scratch\autofit")


# ----------------------------------------------------------------------------------------------- helpers
def majority(values: np.ndarray, min_rows: int, min_share: float):
    """Majority person among labelled rows; None if too few rows or not pure enough."""
    values = values[values >= 0]
    if len(values) < min_rows:
        return None, len(values), 0.0
    people, counts = np.unique(values, return_counts=True)
    share = counts.max() / len(values)
    return (int(people[counts.argmax()]) if share >= min_share else None), len(values), float(share)


def table_score(scene, reference, labels, name):
    s = score(scene, reference, labels)
    p = cross_camera_pairs(scene, reference, labels)
    per = s["per_person"]
    row = {"name": name, "coverage": s["coverage"], "impure_obs": s["impure_observations"], "purity": s["purity"],
           "mixed_labels": s["mixed_labels"], "labels": s["labels"], "extra_fragments": s["extra_fragments"],
           "fragments_per_person_mean": round(float(np.mean([v["fragments"] for v in per.values()])), 2),
           "fragments_per_person_median": float(np.median([v["fragments"] for v in per.values()])),
           "largest_share_mean": s["largest_fragment_share_mean"], "people_seen": s["reference_people_seen"],
           "pair_precision": p["precision"], "pair_recall": p["recall"], "wrong_pairs": p["wrong_pairs"],
           "pairs_same_ref": p["pairs_same_in_reference"], "pairs_said_same": p["pairs_same_in_engine"]}
    return row, s["mixed_examples"]


class Units:
    """A set of labelled units (atoms or components) with majority truth, built from per-row labels."""

    def __init__(self, scene, reference, labels, min_rows, min_share):
        self.rows = defaultdict(list)  # label -> list of (camera, row)
        persons = defaultdict(list)
        self.t = defaultdict(list)
        for cam in scene.cameras:
            lab = labels[cam.index]
            sel = np.flatnonzero(lab >= 0)
            for r in sel:
                persons[int(lab[r])].append(int(reference.person[cam.index][r]))
                self.t[int(lab[r])].append(int(cam.t[r]))
        self.truth, self.nref, self.share = {}, {}, {}
        for label, who in persons.items():
            m, n, sh = majority(np.asarray(who), min_rows, min_share)
            self.truth[label], self.nref[label], self.share[label] = m, n, sh
        self.t = {k: np.asarray(v) for k, v in self.t.items()}


# ----------------------------------------------------------------------------------------------- Part B
def pair_evidence(scene, atom_lab, raw, glob, usable_atoms, split_t):
    """Aggregate per (atom a < atom b, half): common simultaneous frames, same/diff raw, same/diff global."""
    entries = []  # t, cam, atom, raw, glob
    for cam in scene.cameras:
        a = atom_lab[cam.index]
        rows = np.flatnonzero((a >= 0) & (cam.t >= 0))
        rows = rows[np.isin(a[rows], usable_atoms)]
        entries.append(np.column_stack([cam.t[rows], np.full(len(rows), cam.index), a[rows],
                                        raw[cam.index][rows], glob[cam.index][rows]]).astype(np.int64))
    e = np.concatenate(entries)
    e = e[np.argsort(e[:, 0], kind="stable")]
    bounds = np.flatnonzero(np.diff(e[:, 0])) + 1
    parts = []
    A = int(e[:, 2].max()) + 2
    chunk_keys, chunk_vals = [], []
    for g in np.split(np.arange(len(e)), bounds):
        if len(g) < 2:
            continue
        sub = e[g]
        i, j = np.triu_indices(len(g), 1)
        ok = sub[i, 1] != sub[j, 1]
        i, j = i[ok], j[ok]
        if not len(i):
            continue
        a, b = sub[i, 2], sub[j, 2]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        ri, rj, gi, gj = sub[i, 3], sub[j, 3], sub[i, 4], sub[j, 4]
        half = np.full(len(i), int(sub[0, 0] >= split_t), np.int64)
        key = (lo * A + hi) * 2 + half
        vals = np.column_stack([np.ones(len(i)), (ri >= 0) & (ri == rj), (ri >= 0) & (rj >= 0) & (ri != rj),
                                (gi >= 0) & (gi == gj), (gi >= 0) & (gj >= 0) & (gi != gj)]).astype(np.int32)
        chunk_keys.append(key)
        chunk_vals.append(vals)
    keys = np.concatenate(chunk_keys)
    vals = np.concatenate(chunk_vals)
    uniq, inv = np.unique(keys, return_inverse=True)
    sums = np.zeros((len(uniq), 5), np.int64)
    for c in range(5):
        sums[:, c] = np.bincount(inv, weights=vals[:, c], minlength=len(uniq))
    half = uniq % 2
    pair = uniq // 2
    return {"a": pair // A, "b": pair % A, "half": half, "common": sums[:, 0], "sr": sums[:, 1], "dr": sums[:, 2],
            "sg": sums[:, 3], "dg": sums[:, 4]}


def rule_mask(ev, kind, x, n, y):
    same, diff = (ev["sr"], ev["dr"]) if kind == "raw" else (ev["sg"], ev["dg"])
    common = np.maximum(ev["common"], 1)
    return (same >= n) & (same / common >= x) & (diff / common <= y)


GRID_X = (0.5, 0.7, 0.8, 0.9, 0.95)
GRID_N = (3, 5, 10, 20, 50)
GRID_Y = (0.0, 0.05, 0.1, 0.2)


def grid(ev, kind, truth_same, pool_mask, half):
    """For every (x, n, y): among pairs in `pool_mask` & `half`, correct / wrong confirmations."""
    sel = pool_mask & (ev["half"] == half)
    n_same = int((sel & truth_same).sum())
    n_diff = int((sel & ~truth_same).sum())
    rows = []
    for x, n, y in itertools.product(GRID_X, GRID_N, GRID_Y):
        m = rule_mask(ev, kind, x, n, y) & sel
        rows.append({"x": x, "n": n, "y": y, "correct": int((m & truth_same).sum()), "wrong": int((m & ~truth_same).sum()),
                     "pool_same": n_same, "pool_diff": n_diff})
    return rows


def choose(dev_rows, max_wrong=0):
    """Most productive rule with at most max_wrong errors on the dev half; ties -> most conservative."""
    ok = [r for r in dev_rows if r["wrong"] <= max_wrong and r["correct"] > 0]
    if not ok:
        return None
    return sorted(ok, key=lambda r: (-r["correct"], -r["x"], -r["n"], r["y"]))[0]


def apply_rule(rows, rule):
    for r in rows:
        if (r["x"], r["n"], r["y"]) == (rule["x"], rule["n"], rule["y"]):
            return r
    return None


def aggregate_units(ev, unit_of_atom, truth_of_unit):
    """Sum atom-pair evidence over engine-component pairs (per half)."""
    ua, ub = unit_of_atom[ev["a"]], unit_of_atom[ev["b"]]
    lo, hi = np.minimum(ua, ub), np.maximum(ua, ub)
    ok = lo != hi
    N = int(hi.max()) + 2
    key = ((lo.astype(np.int64) + 1) * N + (hi + 1)) * 2 + ev["half"]
    key = key[ok]
    uniq, inv = np.unique(key, return_inverse=True)
    out = {"half": uniq % 2}
    pair = uniq // 2
    out["a"], out["b"] = pair // N - 1, pair % N - 1
    for name in ("common", "sr", "dr", "sg", "dg"):
        out[name] = np.bincount(inv, weights=ev[name][ok], minlength=len(uniq)).astype(np.int64)
    return out


# ----------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--variant", default="kept")
    ap.add_argument("--round", default="r000")
    ap.add_argument("--dataset", default=None, help="label dataset id (default: workspace id before @)")
    ap.add_argument("--split-t", type=int, default=None)
    ap.add_argument("--fake-reference", action="store_true", help="smoke test only: use engine identities as truth")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ws = Workspace(args.workspace)
    scene = load_scene(ws, joint_peak=0.3, rays=False)
    C = len(scene.cameras)
    af = np.load(SCRATCH / args.workspace / args.variant / "labels.npz")
    sol = np.load(ws.path("rounds", args.round, "solution.npz"))
    dataset = args.dataset or ws.source_dataset
    if args.fake_reference:
        person = []
        for cam in scene.cameras:
            ident = sol[f"identity_{cam.camera}"]
            person.append(np.where(ident >= 0, ident, -1).astype(np.int32))
        ref = Reference(person, [np.array([], np.int32)] * C, int(sum((p >= 0).sum() for p in person)), 0, 0)
    else:
        ref = load_reference(scene, LABELS, dataset)
    print("reference people", ref.people, "observations", ref.observations, "matched", ref.matched, flush=True)

    def per_cam(source, prefix):
        return [source[f"{prefix}_{cam.camera}"] if f"{prefix}_{cam.camera}" in source.files else
                np.full(len(cam.kept), -1, np.int32) for cam in scene.cameras]

    raw, glob = per_cam(af, "raw"), per_cam(af, "global")
    comp, ident, atom = per_cam(sol, "component"), per_cam(sol, "identity"), per_cam(sol, "atom")
    T = scene.T
    split_t = args.split_t if args.split_t is not None else T // 2
    result = {"workspace": args.workspace, "variant": args.variant, "round": args.round, "T": T, "split_t": split_t,
              "reference": {"people": ref.people, "observations": ref.observations, "matched": ref.matched}}

    # ---------------------------------------------------------------- A
    table, mixed = [], {}
    for name, lab in (("autofit_raw", raw), ("autofit_global", glob), ("engine_component", comp), ("engine_identity", ident)):
        row, ex = table_score(scene, ref, lab, name)
        table.append(row)
        mixed[name] = ex
    # Autofit restricted to detections the engine kept (fair vs engine that never labels filtered rows)
    kept_only = lambda lab: [np.where(cam.kept, lab[cam.index], -1) for cam in scene.cameras]
    for name, lab in (("autofit_raw|kept", kept_only(raw)), ("autofit_global|kept", kept_only(glob))):
        row, ex = table_score(scene, ref, lab, name)
        table.append(row)
    # halves of time
    result["A_scores"], result["A_mixed_examples"] = table, mixed
    print(json.dumps(table, indent=1), flush=True)

    # ---------------------------------------------------------------- B: piece pairs
    atoms = Units(scene, ref, atom, min_rows=1, min_share=1.0)  # impure atoms (labelled rows disagree) get None
    comps_units = Units(scene, ref, comp, min_rows=3, min_share=0.9)
    # atom -> engine component (majority over rows), atom -> camera
    atom_comp, atom_cam = {}, {}
    for cam in scene.cameras:
        for r in np.flatnonzero(atom[cam.index] >= 0):
            a = int(atom[cam.index][r])
            atom_cam[a] = cam.index
            atom_comp.setdefault(a, Counter())[int(comp[cam.index][r])] += 1
    atom_comp = {a: c.most_common(1)[0][0] for a, c in atom_comp.items()}
    usable = np.array(sorted(a for a, w in atoms.truth.items() if w is not None))
    skipped_atoms = {"atoms_with_reference_rows": len(atoms.truth), "pure": len(usable),
                     "impure_or_unlabelled": len(atoms.truth) - len(usable)}
    result["B_atoms"] = skipped_atoms
    print(skipped_atoms, flush=True)
    ev = pair_evidence(scene, atom, raw, glob, usable, split_t)
    truth_arr = np.full(int(max(usable.max(), ev["b"].max())) + 2, -1, np.int64)
    comp_arr = np.full(len(truth_arr), -(10 ** 9), np.int64)
    for a in usable:
        truth_arr[a] = atoms.truth[int(a)]
        c = atom_comp.get(int(a), -1)
        comp_arr[a] = c if c >= 0 else -(int(a) + 2)  # unassigned atoms are singleton "components"
    truth_same = truth_arr[ev["a"]] == truth_arr[ev["b"]]
    diff_comp = comp_arr[ev["a"]] != comp_arr[ev["b"]]
    result["B_pair_counts"] = {"overlapping_cross_camera_atom_pairs": int(len(ev["a"])),
                               "truly_same": int(truth_same.sum()), "truly_same_engine_undecided": int((truth_same & diff_comp).sum()),
                               "truly_same_engine_same_component": int((truth_same & ~diff_comp).sum()),
                               "truly_different_engine_same_component(engine wrong merges)": int((~truth_same & ~diff_comp).sum())}
    print(result["B_pair_counts"], flush=True)

    out_B = {}
    for kind in ("raw", "global"):
        dev = grid(ev, kind, truth_same, diff_comp, 0)
        test = grid(ev, kind, truth_same, diff_comp, 1)
        rule = choose(dev)
        entry = {"dev_pool": {"same": dev[0]["pool_same"], "diff": dev[0]["pool_diff"]},
                 "test_pool": {"same": test[0]["pool_same"], "diff": test[0]["pool_diff"]},
                 "rule": rule, "dev": apply_rule(dev, rule) if rule else None, "test": apply_rule(test, rule) if rule else None,
                 "dev_grid": dev, "test_grid": test}
        # all rules with zero wrong on dev, and their test outcome
        entry["zero_wrong_on_dev"] = [{**r, "test_correct": t["correct"], "test_wrong": t["wrong"]}
                                      for r, t in zip(dev, test) if r["wrong"] == 0 and r["correct"] > 0]
        out_B[f"atom_pairs_{kind}"] = entry

    # component-pair level (human-question granularity)
    comp_label_truth = {}
    for a in usable:
        pass
    # component truth: use component Units (>=3 labelled rows, 90% pure); unassigned atoms are singleton units
    unit_of_atom = np.full(len(truth_arr), -1, np.int64)
    unit_truth = {}
    for a in usable:
        c = atom_comp.get(int(a), -1)
        if c >= 0:
            unit_of_atom[a] = c
            unit_truth[c] = comps_units.truth.get(c)
        else:
            unit_of_atom[a] = 10 ** 6 + int(a)
            unit_truth[10 ** 6 + int(a)] = atoms.truth[int(a)]
    cev = aggregate_units(ev, unit_of_atom, unit_truth)
    ct_a = np.array([unit_truth.get(int(u)) if unit_truth.get(int(u)) is not None else -2 for u in cev["a"]])
    ct_b = np.array([unit_truth.get(int(u)) if unit_truth.get(int(u)) is not None else -3 for u in cev["b"]])
    known = (ct_a >= 0) & (ct_b >= 0)
    c_same = ct_a == ct_b
    for kind in ("raw", "global"):
        pool = known  # all component pairs are different engine components by construction
        dev = grid(cev, kind, c_same, pool, 0)
        test = grid(cev, kind, c_same, pool, 1)
        rule = choose(dev)
        out_B[f"component_pairs_{kind}"] = {
            "dev_pool": {"same": dev[0]["pool_same"], "diff": dev[0]["pool_diff"]},
            "test_pool": {"same": test[0]["pool_same"], "diff": test[0]["pool_diff"]},
            "rule": rule, "dev": apply_rule(dev, rule) if rule else None, "test": apply_rule(test, rule) if rule else None,
            "zero_wrong_on_dev": [{**r, "test_correct": t["correct"], "test_wrong": t["wrong"]}
                                  for r, t in zip(dev, test) if r["wrong"] == 0 and r["correct"] > 0],
            "dev_grid": dev, "test_grid": test}
    result["B"] = out_B

    # ---------------------------------------------------------------- D: negative evidence
    neg = []
    for kind in ("raw", "global"):
        d = ev["dr"] if kind == "raw" else ev["dg"]
        s = ev["sr"] if kind == "raw" else ev["sg"]
        for n in (1, 3, 5, 10, 20):
            m = (d >= n) & (s == 0) & (ev["common"] > 0)
            for half in (0, 1):
                sel = m & (ev["half"] == half)
                neg.append({"kind": kind, "min_diff_frames": n, "half": half, "pairs": int(sel.sum()),
                            "truly_different": int((sel & ~truth_same).sum()), "truly_same(wrong elimination)": int((sel & truth_same).sum())})
    result["D_negative_atom_pairs"] = neg

    # ---------------------------------------------------------------- C: identity across gaps
    comp_ids = sorted(comps_units.truth)
    comp_truth = {c: comps_units.truth[c] for c in comp_ids if comps_units.truth[c] is not None}
    # engine identity per component
    ident_of_comp = {}
    for cam in scene.cameras:
        for r in np.flatnonzero(comp[cam.index] >= 0):
            ident_of_comp[int(comp[cam.index][r])] = int(sol["component_identity"][comp[cam.index][r]]) if "component_identity" in sol.files else -1
    # per-component histograms of Autofit ids
    hist = {"raw": defaultdict(Counter), "global": defaultdict(Counter)}
    span = {}
    for cam in scene.cameras:
        for kind, lab in (("raw", raw), ("global", glob)):
            rows = np.flatnonzero((comp[cam.index] >= 0) & (lab[cam.index] >= 0) & (cam.t >= 0))
            for r in rows:
                hist[kind][int(comp[cam.index][r])][int(lab[cam.index][r])] += 1
        rows = np.flatnonzero((comp[cam.index] >= 0) & (cam.t >= 0))
        for r in rows:
            c = int(comp[cam.index][r])
            lo, hi, n = span.get(c, (10 ** 9, -1, 0))
            span[c] = (min(lo, int(cam.t[r])), max(hi, int(cam.t[r])), n + 1)
    comp_t = {c: comps_units.t[c] for c in comp_truth}
    comp_half = {c: int(np.median(comp_t[c]) >= split_t) for c in comp_truth}
    cids = sorted(comp_truth)
    tset = {c: set(comp_t[c].tolist()) for c in cids}
    pairs = []
    for i, a in enumerate(cids):
        for b in cids[i + 1:]:
            same = comp_truth[a] == comp_truth[b]
            overlap = len(tset[a] & tset[b])
            gap = (min(comps_units.t[b].min() - comps_units.t[a].max(), comps_units.t[a].min() - comps_units.t[b].max()))
            row = {"a": a, "b": b, "same": same, "overlap": overlap, "gap": int(max(gap, 0)),
                   "same_identity": ident_of_comp.get(a, -1) == ident_of_comp.get(b, -2) and ident_of_comp.get(a, -1) >= 0,
                   "half": (comp_half[a] if comp_half[a] == comp_half[b] else -1)}
            for kind in ("raw", "global"):
                ha, hb = hist[kind].get(a, Counter()), hist[kind].get(b, Counter())
                shared = [(min(ha[g], hb[g]), ha[g], hb[g]) for g in ha.keys() & hb.keys()]
                best = max(shared) if shared else (0, 0, 0)
                na, nb = sum(ha.values()), sum(hb.values())
                row[kind] = {"n": best[0], "share_a": best[1] / max(na, 1), "share_b": best[2] / max(nb, 1)}
            pairs.append(row)
    result["C_pairs_total"] = len(pairs)
    result["C_pool"] = {"same_not_same_identity": sum(p["same"] and not p["same_identity"] for p in pairs),
                        "same_total": sum(p["same"] for p in pairs),
                        "different_not_same_identity": sum((not p["same"]) and not p["same_identity"] for p in pairs),
                        "engine_wrong_identity_merges": sum((not p["same"]) and p["same_identity"] for p in pairs)}
    cgrid = []
    for kind in ("raw", "global"):
        for n, x in itertools.product((1, 3, 5, 10, 20, 50), (0.0, 0.3, 0.5, 0.7, 0.9)):
            for mode in ("gap", "overlap", "any"):
                for half in (0, 1, -1):
                    sel = [p for p in pairs if not p["same_identity"] and p["half"] == half
                           and (mode == "any" or (mode == "gap" and p["overlap"] == 0) or (mode == "overlap" and p["overlap"] > 0))]
                    fired = [p for p in sel if p[kind]["n"] >= n and min(p[kind]["share_a"], p[kind]["share_b"]) >= x]
                    cgrid.append({"kind": kind, "n": n, "x": x, "mode": mode, "half": half, "pool": len(sel),
                                  "pool_same": sum(p["same"] for p in sel), "fired": len(fired),
                                  "correct": sum(p["same"] for p in fired), "wrong": sum(not p["same"] for p in fired)})
    result["C_grid"] = cgrid
    result["C_wrong_examples"] = [{k: p[k] for k in ("a", "b", "overlap", "gap", "raw", "global")} | {"truth": (comp_truth[p["a"]], comp_truth[p["b"]])}
                                  for p in pairs if not p["same"] and not p["same_identity"] and p["global"]["n"] >= 5][:20]

    result["C_wrong_at_n5_x0.9"] = [
        {"a": p["a"], "b": p["b"], "half": p["half"], "overlap": p["overlap"], "gap": p["gap"], "global": p["global"], "raw": p["raw"],
         "truth": (comp_truth[p["a"]], comp_truth[p["b"]]), "t_a": [int(comps_units.t[p["a"]].min()), int(comps_units.t[p["a"]].max())],
         "t_b": [int(comps_units.t[p["b"]].min()), int(comps_units.t[p["b"]].max())],
         "ref_rows": [comps_units.nref[p["a"]], comps_units.nref[p["b"]]], "purity": [round(comps_units.share[p["a"]], 3), round(comps_units.share[p["b"]], 3)]}
        for p in pairs if not p["same"] and not p["same_identity"] and p["global"]["n"] >= 5
        and min(p["global"]["share_a"], p["global"]["share_b"]) >= 0.9]
    # worst truly-different atom pairs by raw/global agreement (how close do wrong pairs come to the rule?)
    worst = []
    for kind in ("sr", "sg"):
        idx = np.flatnonzero(~truth_same & diff_comp)
        order_ = idx[np.argsort(-ev[kind][idx])][:12]
        worst.append({kind: [{"atoms": [int(ev["a"][i]), int(ev["b"][i])], "half": int(ev["half"][i]), "common": int(ev["common"][i]),
                              "same": int(ev[kind][i]), "diff": int(ev["dr" if kind == "sr" else "dg"][i]),
                              "persons": [int(truth_arr[ev["a"][i]]), int(truth_arr[ev["b"][i]])]} for i in order_]})
    result["B_worst_wrong_atom_pairs"] = worst
    out = args.out or (SCRATCH / args.workspace / args.variant / "evaluation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=int), encoding="utf-8")
    print("wrote", out)
    summarize(result)
    return 0


def summarize(result):
    print("\n== A ==")
    for r in result["A_scores"]:
        print({k: r[k] for k in ("name", "coverage", "impure_obs", "mixed_labels", "labels", "fragments_per_person_mean", "pair_precision", "pair_recall", "wrong_pairs")})
    print("\n== B ==", result["B_pair_counts"])
    for name, e in result["B"].items():
        print(name, "dev pool", e["dev_pool"], "test pool", e["test_pool"], "rule", e["rule"] and (e["rule"]["x"], e["rule"]["n"], e["rule"]["y"]),
              "dev", e["dev"] and (e["dev"]["correct"], e["dev"]["wrong"]), "test", e["test"] and (e["test"]["correct"], e["test"]["wrong"]))
    print("\n== D ==")
    for r in result["D_negative_atom_pairs"]:
        if r["min_diff_frames"] in (3, 10):
            print(r)
    print("\n== C pool ==", result["C_pool"])


if __name__ == "__main__":
    sys.exit(main())
