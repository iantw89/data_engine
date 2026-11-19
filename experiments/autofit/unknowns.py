"""Would Autofit have answered the engine's ACTUAL open questions (engine.sqlite3 unknowns, read-only) correctly?

association unknown : a single-camera piece with candidate engine components ("which one is the same person, or none")
identity_link unknown: a component that starts inside the recording with candidate earlier components
Autofit answer: the unique candidate whose rows co-occur in one Autofit raw/global id with the piece/component under a rule
(x = share of evidence frames, n = frames, y = conflicting share). Truth = golden majority person (pure pieces only).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import LABELS, SCRATCH, Units  # noqa: E402
from data_engine.evaluation import load_reference  # noqa: E402
from data_engine.solve.scene import load_scene  # noqa: E402
from data_engine.workspace import Workspace  # noqa: E402

RULES = {"dev_chosen(x.9,n3,y.1)": (0.9, 3, 0.1), "conservative(x.95,n5,y.05)": (0.95, 5, 0.05),
         "loose(x.5,n3,y.2)": (0.5, 3, 0.2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--variant", default="kept")
    ap.add_argument("--round", default="r000")
    args = ap.parse_args()
    ws = Workspace(args.workspace)
    scene = load_scene(ws, joint_peak=0.3, rays=False)
    ref = load_reference(scene, LABELS, ws.source_dataset)
    af = np.load(SCRATCH / args.workspace / args.variant / "labels.npz")
    sol = np.load(ws.path("rounds", args.round, "solution.npz"))
    cams = {c.camera: c for c in scene.cameras}
    raw = [af[f"raw_{c.camera}"] for c in scene.cameras]
    glob = [af[f"global_{c.camera}"] for c in scene.cameras]
    comp = [sol[f"component_{c.camera}"] for c in scene.cameras]
    atom = [sol[f"atom_{c.camera}"] for c in scene.cameras]
    split_t = scene.T // 2
    comps = Units(scene, ref, comp, 3, 0.9)
    atoms = Units(scene, ref, atom, 1, 1.0)

    # per component: {t: [(cam, raw, glob)]}
    by_comp = defaultdict(lambda: defaultdict(list))
    for c in scene.cameras:
        rows = np.flatnonzero((comp[c.index] >= 0) & (c.t >= 0))
        for r in rows:
            by_comp[int(comp[c.index][r])][int(c.t[r])].append((c.index, int(raw[c.index][r]), int(glob[c.index][r])))
    atom_rows = defaultdict(list)
    for c in scene.cameras:
        for r in np.flatnonzero((atom[c.index] >= 0) & (c.t >= 0)):
            atom_rows[int(atom[c.index][r])].append((c.index, int(r)))

    db = sqlite3.connect(f"file:{ws.path('engine.sqlite3').as_posix()}?mode=ro", uri=True)
    unknowns = [(k, json.loads(d)) for k, d in db.execute("select kind, detail from unknowns where status='open'")]
    counts = Counter(k for k, _ in unknowns)
    out = {"open_unknowns": dict(counts), "rules": {}}

    def piece_vs_comp(rows, cidx_of_t, comp_id, kind):
        same = diff = common = 0
        for (cam, r) in rows:
            t = int(scene.cameras[cam].t[r])
            mine = (raw if kind == "raw" else glob)[cam][r]
            for (cc, rr, gg) in by_comp[comp_id].get(t, ()):
                if cc == cam:
                    continue
                other = rr if kind == "raw" else gg
                common += 1
                if mine >= 0 and other >= 0:
                    same += mine == other
                    diff += mine != other
        return same, diff, common

    results = []
    for kind_u, d in unknowns:
        if kind_u != "association":
            continue
        cam = cams[d["camera"]]
        key = d["reference_keys"][0]
        row = cam.detections.row_of_key(key)
        a = int(atom[cam.index][row])
        if a < 0:
            results.append({"skip": "subject row not in an atom"})
            continue
        rows = atom_rows[a]
        truth = atoms.truth.get(a)
        t_mid = d["t"]
        cands = []
        for cnd in d["candidates"]:
            cid = cnd["component"]
            row_ev = {}
            for kind in ("raw", "global"):
                s, df, cm = piece_vs_comp(rows, None, cid, kind)
                row_ev[kind] = (s, df, cm)
            cands.append({"component": cid, "truth": comps.truth.get(cid), "ev": row_ev, "engine_frames_matching": cnd.get("frames_matching", cnd.get("samples"))})
        results.append({"subject_truth": truth, "half": int(t_mid >= split_t), "cands": cands, "atom": a, "n_rows": len(rows)})

    summary = {}
    for rule_name, (x, n, y) in RULES.items():
        for kind in ("raw", "global"):
            tab = defaultdict(Counter)
            for r in results:
                if "skip" in r:
                    tab["all"]["skipped"] += 1
                    continue
                half = r["half"]
                if r["subject_truth"] is None:
                    tab[half]["truth_unknown"] += 1
                    continue
                has_true = any(c["truth"] == r["subject_truth"] for c in r["cands"])
                fired = [c for c in r["cands"] if c["ev"][kind][0] >= n and c["ev"][kind][0] / max(c["ev"][kind][2], 1) >= x
                         and c["ev"][kind][1] / max(c["ev"][kind][2], 1) <= y]
                tab[half]["questions"] += 1
                tab[half]["has_true_candidate"] += has_true
                if len(fired) == 1:
                    if fired[0]["truth"] is None:
                        tab[half]["answered_candidate_truth_unknown"] += 1
                    else:
                        ok = fired[0]["truth"] == r["subject_truth"]
                        tab[half]["answered_correct" if ok else "answered_WRONG"] += 1
                elif len(fired) > 1:
                    tab[half]["ambiguous(>1 fired)"] += 1
                    tab[half]["ambiguous_contains_wrong"] += any(c["truth"] is not None and c["truth"] != r["subject_truth"] for c in fired)
                else:
                    tab[half]["not_answered"] += 1
            summary[f"association|{rule_name}|{kind}"] = {str(h): dict(c) for h, c in tab.items()}
    out["association"] = summary
    elim = defaultdict(Counter)
    for r in results:
        if "skip" in r or r["subject_truth"] is None:
            continue
        for c in r["cands"]:
            if c["truth"] is None:
                continue
            s_, d_, cm_ = c["ev"]["global"]
            for n_ in (3, 10):
                if d_ >= n_ and s_ == 0:
                    elim[f"diff>={n_},same=0"]["eliminated"] += 1
                    elim[f"diff>={n_},same=0"]["wrongly_eliminated_true_person"] += c["truth"] == r["subject_truth"]
            elim["all_scorable_candidates"]["n"] += 1
            elim["all_scorable_candidates"]["true"] += c["truth"] == r["subject_truth"]
    out["association_negative"] = {k: dict(v) for k, v in elim.items()}
    wrong = []
    for r in results:
        if "skip" in r or r["subject_truth"] is None:
            continue
        for c in r["cands"]:
            e = c["ev"]["raw"]
            if e[0] >= 3 and e[0] / max(e[2], 1) >= 0.9 and e[1] / max(e[2], 1) <= 0.1 and c["truth"] is not None and c["truth"] != r["subject_truth"]:
                wrong.append({"half": r["half"], "subject_truth": r["subject_truth"], "n_rows": r["n_rows"], "cand": c["component"],
                              "cand_truth": c["truth"], "raw_ev(same,diff,common)": e, "global_ev": c["ev"]["global"],
                              "n_cands": len(r["cands"]), "cands_truth": [k["truth"] for k in r["cands"]]})
    out["association_wrong_dev_chosen_rule"] = wrong
    out["association_n"] = len(results)

    # ---- identity_link
    res2 = []
    ghist = defaultdict(Counter)
    rhist = defaultdict(Counter)
    for c in scene.cameras:
        for r in np.flatnonzero((comp[c.index] >= 0) & (c.t >= 0)):
            if glob[c.index][r] >= 0:
                ghist[int(comp[c.index][r])][int(glob[c.index][r])] += 1
            if raw[c.index][r] >= 0:
                rhist[int(comp[c.index][r])][int(raw[c.index][r])] += 1
    for kind_u, d in unknowns:
        if kind_u != "identity_link":
            continue
        cid = d["component"]
        truth = comps.truth.get(cid)
        cands = []
        for cnd in d["candidates"]:
            o = cnd["component"]
            ev = {}
            for kind, H in (("raw", rhist), ("global", ghist)):
                ha, hb = H.get(cid, Counter()), H.get(o, Counter())
                sh = [(min(ha[g], hb[g]), ha[g], hb[g]) for g in ha.keys() & hb.keys()]
                b = max(sh) if sh else (0, 0, 0)
                ev[kind] = (b[0], min(b[1] / max(sum(ha.values()), 1), b[2] / max(sum(hb.values()), 1)))
            cands.append({"component": o, "truth": comps.truth.get(o), "ev": ev})
        res2.append({"truth": truth, "half": int(d["t"] >= split_t), "cands": cands})
    summ2 = {}
    for kind in ("raw", "global"):
        for n, x in ((1, 0.5), (5, 0.9), (20, 0.9)):
            tab = defaultdict(Counter)
            for r in res2:
                half = r["half"]
                if r["truth"] is None:
                    tab[half]["truth_unknown"] += 1
                    continue
                tab[half]["questions"] += 1
                tab[half]["has_true_candidate"] += any(c["truth"] == r["truth"] for c in r["cands"])
                fired = [c for c in r["cands"] if c["ev"][kind][0] >= n and c["ev"][kind][1] >= x]
                if len(fired) == 1:
                    if fired[0]["truth"] is None:
                        tab[half]["answered_candidate_truth_unknown"] += 1
                    else:
                        tab[half]["answered_correct" if fired[0]["truth"] == r["truth"] else "answered_WRONG"] += 1
                elif len(fired) > 1:
                    tab[half]["ambiguous"] += 1
                else:
                    tab[half]["not_answered"] += 1
            summ2[f"identity_link|{kind}|n{n}|x{x}"] = {str(h): dict(c) for h, c in tab.items()}
    out["identity_link"] = summ2
    out["identity_link_n"] = len(res2)
    path = SCRATCH / args.workspace / args.variant / "unknowns_eval.json"
    path.write_text(json.dumps(out, indent=1, default=int), encoding="utf-8")
    print(json.dumps(out, indent=1, default=int))


if __name__ == "__main__":
    main()
