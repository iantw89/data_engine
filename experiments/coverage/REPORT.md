# Recovering 3D positions the footage supports (2026-11-18)

Goal: a track may lack a position only where the person is outside calibrated space or
genuinely unobserved. Engine code: `src/data_engine/solve/coverage.py`,
`src/data_engine/workers/recover_trt.py`, `src/data_engine/stages/recover.py`. This folder
holds the research scripts; outputs are in `C:\data_engine_data\_scratch\coverage\`.
Everything except `golden_single_view.py`, `golden_check.py` and `check_recovered.py` is
reference-free (no golden labels).

## 1. Where positions are missing (golden 600 s, engine solution)

375 gaps inside identities, 4,561 missing person-frames (79,560 present); 318 gaps are one
second or shorter. With the neighbouring measured pose as a hint (interpolated if both sides
are within 3 s, held otherwise):

| Cause (interior missing frames) | Frames |
|---|---|
| No measured pose within 3 s (8 long gaps) | 2,190 |
| The place is visible in 2+ cameras but only one has a detection near it | 1,224 |
| Detections of this identity, unassigned or post-filter-rejected reach 2+ views | 494 |
| Reaching two views would need a detection that sits on another identity | 215 |
| Fewer than two cameras (8 degrees apart) see the place | 180 |
| Visible in 2+ cameras, no detection anywhere near | 152 |
| 2+ views carry this identity's detection but the triangulation gates failed | 106 |

Only 53 of the "visible but undetected" frames look occluded. Of 10,418 golden
person-instants with a single labelled view, 8,862 already have a 3D position.

## 2. Levers

| Lever | Yield at the acceptance gate | Cost | Decision |
|---|---|---|---|
| 2: ViTPose on the hint's projected box, read twice | 816 of 5,065 missing frames | about 45 s of GPU per recorded hour plus one decode pass | **Adopted** |
| 1: existing raw detections near the hint (incl. post-filter-rejected) | +41 beyond lever 2 | none | Not adopted |
| 3: detector on a zoomed crop (YOLOv8x 3x / 5x, YOLO11l 5x), ViTPose on its boxes | +46 beyond levers 1+2 | about 14 min per recorded hour | Rejected (5x zoom halves recall) |

Fidelity: fed the stored detector boxes, the crop pass reproduces the stored keypoints
exactly. The second read (on the box of the first result) adds about 4%.

## 3. Acceptance gate and its evidence

A recovery is the engine's gated triangulation of the poses read (confident joints, 8
degrees, 12 px, leave-one-out), then:

* root height 15-55 in, no other identity measured within 24 in;
* **tier A** (3+ cameras): 9+ body joints, at most one implausible bone, within 15 in of the hint;
* **tier B** (2 cameras): 8+ body joints, median residual 3 px, every view's mean joint
  peak 0.6+, at most two implausible bones, within 12 in of the hint, and only in a gap
  bracketed by measured poses at most 5 frames away;
* competitor rule: a crop is dropped when another measured person's box overlaps the hint
  box (IoU 0.4+) and the pose follows that person at least as well - this removed 22 of 23
  false accepts for 3 frames of yield.

| Test | Result |
|---|---|
| Withheld measured poses (3,005 hidden in runs of 1-20) | 91.0% recovered; root error p50 0.5 in, p90 2.0, p99 8.9; none beyond 24 in |
| Negative: hint displaced 20 / 39 / 79 in (2,000) | 0 unexplained accepts |
| Negative: person absent, hint held 10-60 s outside the lifetime (1,500) | 1 accept (two-camera, held hint; a real person the engine never tracked) |
| Negative: empty floor (3,000) | 0 accepts (29 with the plain 12 px gate: stick figures on the pull-up rack) |
| Golden boxes, research run (340 checkable views) | 331 right person, 6 another person, 3 low overlap |
| Golden boxes, engine stage (318 checkable views) | 309 right person, 6 another person (4 follow a track that is itself on the wrong person at that time), 3 low overlap |

Caveats: the withheld truth is the engine's own pose; gates were tuned on the same
negatives (no held-out set); one 10-minute recording whose detector was fine-tuned on it.

## 4. What remains missing (golden, after the engine stage)

749 recovered (622 tier A, 127 tier B); 4,316 remain: 2,190 without a hint, 245 seen by
fewer than two cameras, 145 where the crops follow another person, 810 with no consistent
pose in two cameras, 926 with a pose that the acceptance gate rejects.

The negative controls also exposed real people the engine never tracked at all - a
coverage problem that gap recovery does not touch.

## 5. Felix hour (no labels; `recovered_sanity.py`)

56,498 missing person-instants inside or next to tracks; 40,825 had a hint and two cameras
(177,540 crops, 19 min of GPU worker time including decode). 17,078 positions recovered
(15,471 tier A, 1,607 tier B): positions 376,511 -> 393,589 (+4.5%); missing frames inside
tracks 49,247 -> 33,493; gaps 5,337 -> 4,482. Not recovered: 14,389 without a hint, 1,284
seen by fewer than two cameras, 1,109 crops following another person, 9,286 without a
consistent pose in two cameras, 13,352 rejected by the acceptance gate.

Continuity check: the distance from a recovered root to the same person's measured root one
frame earlier or later is 0.07 m at the median and 0.46 m at p99 (maximum 0.72 m);
between adjacent measured roots it is 0.03 m and 0.28 m. Rendered examples
(`--sheets`) show the recovered pose on the same person as the neighbouring detections.

## Reproduce

Research: run the scripts with data_management's interpreter in the order given in each
docstring (`gap_classify.py`, `lever1.py`, `plan_jobs.py`, `run_pass.py`,
`eval_levers.py`, `final_eval.py`, ...). Engine stage on a golden workspace and its check:

```
python -m data_engine.cli run "golden_20250505071549_20250505080000@first600s"
python experiments/coverage/check_recovered.py "golden_20250505071549_20250505080000@first600s" golden_20250505071549_20250505080000
python experiments/coverage/gap_diagnosis.py WORKSPACE [GOLDEN_DATASET]
```
