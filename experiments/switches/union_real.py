"""Real-error recall of the rule on baseline / autofit-alone, chunks from 3D<1m + 2D-overlap encounters.
    python experiments/switches/union_real.py"""
import contextlib, io, pickle
import numpy as np
import common, rule

for name in ("saved", "autofit", "baseline"):
    d = pickle.loads((common.SCRATCH / f"cues_{name}_union.pkl").read_bytes())
    rows, eps = d["rows"], d["episodes"]
    print(f"\n[{name}] {len(eps)} real episodes, {sum(e['n_obs'] for e in eps)} wrong obs")
    for label, kw in (("2-of-n agree", dict(use_own=False, min_causes=2)), ("any cue", dict(use_own=False)),
                      ("2-of-n agree + own>=3", dict(min_causes=2, own_k=3)), ("any cue + own>=3", dict(own_k=3))):
        with contextlib.redirect_stdout(io.StringIO()):
            out = rule.evaluate(rows, name, **kw)
        neg = [r for r in out if r["kind"] == "neg"]
        def hit(e):
            return any(c["flag"] and (c["a"] == e["identity"] or c["b"] == e["identity"]) and c["e"] + 10 >= e["t0"] and c["s"] - 10 <= e["t1"] for c in neg)
        def boundary(e):  # flagged chunk contains the start or end of the episode
            return any(c["flag"] and (c["a"] == e["identity"] or c["b"] == e["identity"]) and
                       (c["s"] - 15 <= e["t0"] <= c["e"] + 15 or c["s"] - 15 <= e["t1"] <= c["e"] + 15) for c in neg)
        res = []
        for typ, pred in (("all", lambda e: True), ("swap/takeover >=10 obs", lambda e: e["type"] != "contamination" and e["n_obs"] >= 10),
                          ("multi-camera (>=3 cams at once)", lambda e: e["max_cams_same_second"] >= 3), ("contamination", lambda e: e["type"] == "contamination")):
            sub = [e for e in eps if pred(e)]
            if sub:
                res.append(f"{typ}: any-overlap {sum(hit(e) for e in sub)}/{len(sub)}, at boundary {sum(boundary(e) for e in sub)}/{len(sub)}")
        n_fl = sum(c["flag"] for c in neg)
        print(f"  rule {label:24s} flags/10min {n_fl / 10.18 * 10:5.1f} | " + " | ".join(res))
