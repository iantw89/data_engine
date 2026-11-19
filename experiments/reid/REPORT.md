# Appearance ReID as a lever (golden 600 s, 2026-11-18)

Condensed from the research run that produced `src/data_engine/reid.py`
(`extract_embeddings`, descriptor and decision helpers), `tests/test_reid.py` and the scripts
in this folder. Embeddings: `C:\data_engine_data\_scratch\reid\{golden_osnet,golden_fastreid}`.
65,830 crops on the 1,200 labelled frames per camera; train = queries before 300 s, held-out
= the last 300 s; thresholds chosen on train and frozen.

**Verdict.** Not safe to decide anything automatically on this footage. Useful for ranking
the candidates shown to a human in "identity across a gap" questions.

| Use case | OSNet x1_0 (Market-1501) rank-1 | fast-reid SBS R50-NL (MSMT17) rank-1 |
|---|---|---|
| Identity across a gap, multi-camera gallery | 61% | 76% |
| Identity across a gap, same-camera gallery | 78% | 87% (held-out 90%) |
| Rival tie-break across cameras (median 10 candidates) | 44% | 63% |
| Single-camera pieces | 43% | 70% (30 cases) |

Frozen "zero wrong on train" rules, fast-reid, held-out:

| Use case | Rule | Decided / correct / wrong / open |
|---|---|---|
| Gap, same-camera gallery | s >= 0.81, margin >= 0.08 | 112 / 112 / 0 / 108 |
| Gap, multi-camera gallery | s >= 0.67, margin >= 0.09 | 102 / 101 / 1 / 118 |
| Cross-camera tie-break | s >= 0.70, margin >= 0.25 | 315 / 311 / 4 / 4,599 |

Across camera folds the same-camera gap rule still made 1-2 wrong decisions per 184-241
(about 0.5-1%). Failures are crowded or occluded crops (a box holding two people) and
several people in similar dark clothes. Separability: within-camera AUC 0.91, cross-camera
0.82 (fast-reid); OSNet 0.83 / 0.69.

Cost: about 40 minutes for a 60-minute, 7-camera recording at one crop pass every 5th frame
(decode-bound, under 2.5 GB VRAM).

Most promising untested improvement: fine-tune on confirmed identities of this gym;
camera-specific galleries; upper-body or keypoint-masked crops; exemplar galleries.

Caveats: 17 labelled people, one clip, temporal (not independent) hold-out; the fast-reid
feature path was reconstructed from the repository config with stubbed imports.
