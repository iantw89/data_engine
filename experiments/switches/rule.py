"""Flagging rule ("review this encounter") and stay-certificate, with cross-validation by encounter.

    python experiments/switches/rule.py saved [baseline autofit]

Thresholds are learned on the unswapped chunks (negatives) and the injected ones of the training folds
(encounters with enc % 2 != fold) and measured on the other fold.
"""
from __future__ import annotations

import pickle
import sys

import numpy as np

import common
import eval_cues as ec

NEG_FPR = 0.02          # per-cue false-flag target on unswapped chunks
STAY_MISS = 0.02        # a cue may call "stay" for at most this share of injected swaps


def load(name):
    return pickle.loads((common.SCRATCH / f"cues_{name}.pkl").read_bytes())["rows"]


def thresholds(train_rows, use_app=True):
    neg = [r for r in train_rows if r["kind"] == "neg"]
    inj = [r for r in train_rows if r["kind"] in ("inj", "tak")]
    t = {}
    for key in ("motion_pos", "body") + (("app",) if use_app else ()):
        v = np.array([r[key] for r in neg], float)
        v = v[np.isfinite(v)]
        t[key] = (float(np.quantile(v, 1 - NEG_FPR)), None)
        w = np.array([r[key] for r in inj], float)
        w = w[np.isfinite(w)]
        t[key] = (t[key][0], float(np.quantile(w, STAY_MISS)))   # (flag above, call "stay" below)
    return t


def verdict(r, thr, trk_k=1, own_k=1, use_app=True, use_own=True, use_trk=True, min_causes=1):
    """-> (flag, flag_causes, stay_votes, n_available).  min_causes=2: flag only when two cues agree, or when
    fewer than 3 swap-cues are available and one of them flags (a single available cue must not be ignored)."""
    causes, stay, avail = [], 0, 0
    for key in ("motion_pos", "body") + (("app",) if use_app else ()):
        v = r.get(key, np.nan)
        if np.isfinite(v):
            avail += 1
            if v > thr[key][0]:
                causes.append(key)
            elif v < thr[key][1]:
                stay += 1
    if use_trk:
        c, a = r.get("trk_contra_sep", 0.0), r.get("trk_agree_sep", 0.0)
        if c + a > 0:
            avail += 1
            if c >= trk_k:
                causes.append("trk")
            elif a >= 2 and c == 0:
                stay += 1
    flag = bool(causes)
    if min_causes >= 2 and flag:
        flag = len(causes) >= 2 or avail < 3
    if use_own and r.get("own_flips", 0) >= own_k:
        causes.append("own")
        flag = True
    return flag, causes, stay, avail


def evaluate(rows, name, use_app=True, trk_k=1, own_k=1, use_own=True, use_trk=True, minutes=10.18, min_causes=1):
    res = {"neg": [], "inj": [], "tak": []}
    folds = (0, 1)
    out = []
    for fold in folds:
        train = [r for r in rows if r["enc"] % 2 != fold]
        test = [r for r in rows if r["enc"] % 2 == fold]
        thr = thresholds(train, use_app)
        for r in test:
            f, causes, stay, avail = verdict(r, thr, trk_k, own_k, use_app, use_own, use_trk, min_causes)
            out.append({**r, "flag": f, "causes": causes, "stay": stay, "avail": avail})
    neg = [r for r in out if r["kind"] == "neg"]
    inj = [r for r in out if r["kind"] == "inj"]
    tak = [r for r in out if r["kind"] == "tak"]
    # certified-stay: >= 2 stay votes and no flag. Unverifiable: < 2 cues available
    cert = lambda r: (not r["flag"]) and r["stay"] >= 2
    unver = lambda r: r["avail"] < 2 and not r["flag"]
    n = len(neg)
    print(f"\n[{name}] rule: cues motion/body{'/app' if use_app else ''}{'/trk>=%d' % trk_k if use_trk else ''}{'/own>=%d' % own_k if use_own else ''}"
          f" (CV by encounter)")
    print(f"  chunks (unswapped): {n}   flagged {sum(r['flag'] for r in neg)} ({np.mean([r['flag'] for r in neg]):.1%});"
          f" certified-stay {sum(cert(r) for r in neg)} ({np.mean([cert(r) for r in neg]):.1%});"
          f" unverifiable {sum(unver(r) for r in neg)}; other (uncertified, unflagged) "
          f"{sum((not r['flag']) and not cert(r) and not unver(r) for r in neg)}")
    print(f"  injected SWAP     : flagged {np.mean([r['flag'] for r in inj]):.3f}  certified-stay (MISSED) {np.mean([cert(r) for r in inj]):.3f}"
          f"  unverifiable {np.mean([unver(r) for r in inj]):.3f}   n={len(inj)}")
    print(f"  injected TAKEOVER : flagged {np.mean([r['flag'] for r in tak]):.3f}  certified-stay (MISSED) {np.mean([cert(r) for r in tak]):.3f}"
          f"  unverifiable {np.mean([unver(r) for r in tak]):.3f}   n={len(tak)}")
    for key, label in (("real_any", "any wrong obs"), ("real_boundary", "start/end of sustained wrong run"),
                       ("real_multicam", "multi-camera wrong")):
        pos = [r for r in neg if r[key]]
        if pos:
            print(f"  real {label:34s}: chunks {len(pos):3d}  flagged {np.mean([r['flag'] for r in pos]):.2f}"
                  f"  certified (MISSED) {np.mean([cert(r) for r in pos]):.2f}  unverifiable {np.mean([unver(r) for r in pos]):.2f}")
    fl = [r for r in neg if r["flag"]]
    print(f"  flags on unswapped chunks per 10 min: {len(fl) / minutes * 10:.1f}  (of which chunk holds a real error: "
          f"{sum(r['real_any'] for r in fl)}/{len(fl)});  causes: "
          f"{ {k: sum(k in r['causes'] for r in fl) for k in ('motion_pos', 'body', 'app', 'trk', 'own')} }")
    return out


if __name__ == "__main__":
    for name in sys.argv[1:] or ["saved"]:
        rows = load(name)
        for kwargs in ({"own_k": 3}, {"use_own": False}, {"use_own": False, "use_trk": False}, {"use_own": False, "min_causes": 2},
                       {"use_own": False, "use_app": False}, {"use_own": False, "use_app": False, "min_causes": 2},
                       {"use_own": False, "use_app": False, "min_causes": 2, "trk_k": 2}):
            evaluate(rows, name, **kwargs)
