"""Rule on chunks from 3D<1 m encounters PLUS 2D-overlap encounters (IoU>0.1): recall of real golden errors, flags per 10 min.

    python experiments/switches/eval_cues.py run saved:union     # first
    python experiments/switches/union_eval.py
"""
from __future__ import annotations

import contextlib
import io
import pickle

import numpy as np

import common
import rule


def main():
    d = pickle.loads((common.SCRATCH / "cues_saved_union.pkl").read_bytes())
    rows, eps = d["rows"], d["episodes"]
    print(f"union events: {len({r['enc'] for r in rows if r['kind'] == 'neg'})} encounters, "
          f"{sum(r['kind'] == 'neg' for r in rows)} chunks ({sum(r['kind'] == 'neg' and r['src'] == '2d' for r in rows)} from 2D-only overlaps); "
          f"real episodes {len(eps)}")
    for label, kw in (("2-of-n agree: motion|body|app|trk>=1 (no ownership)", dict(use_own=False, min_causes=2)),
                      ("any of motion|body|app|trk>=1 (no ownership)", dict(use_own=False)),
                      ("2-of-n agree + ownership >= 3 flips", dict(min_causes=2, own_k=3)),
                      ("2-of-n agree + ownership >= 5 flips", dict(min_causes=2, own_k=5)),
                      ("2-of-n agree (no app) + ownership >= 3", dict(min_causes=2, own_k=3, use_app=False))):
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            out = rule.evaluate(rows, "saved_union", **kw)
        print("\n" + label)
        print("\n".join(l for l in buf.getvalue().splitlines() if "injected" in l or "flags on" in l))
        neg = [r for r in out if r["kind"] == "neg"]
        hit = [any(c["flag"] and (c["a"] == e["identity"] or c["b"] == e["identity"]) and c["e"] + 10 >= e["t0"] and c["s"] - 10 <= e["t1"] for c in neg)
               for e in eps]
        print(f"  real golden episodes inside a flagged chunk: {sum(hit)}/{len(eps)}, wrong obs {sum(e['n_obs'] for e, k in zip(eps, hit) if k)}/{sum(e['n_obs'] for e in eps)}"
              f"; flagged chunks by source: 3D {sum(c['flag'] and c['src'] == '3d' for c in neg)}, 2D-only {sum(c['flag'] and c['src'] == '2d' for c in neg)}"
              f" of {sum(c['src'] == '3d' for c in neg)} / {sum(c['src'] == '2d' for c in neg)}")
        by_t = [e for e, k in zip(eps, hit) if not k and e["n_obs"] >= 3]
        print(f"  not flagged with >= 3 obs: {[(round(e['t0'] / 10), e['n_obs']) for e in by_t]}")


if __name__ == "__main__":
    main()
