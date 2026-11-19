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

Correction (2026-11-19): the research report called two accepted negative controls "real
people the engine never tracked". Checked against the tracks, that is wrong. One is identity
9 inside a 1.2 s gap of its own track (measured 0.9 s before and 0.4 s after at that spot).
The other is a person with no position within 5 s at that spot while two tracks are
unmeasured there (one starts 6.6 s later, 6 m away; one has a 100 s gap). Both are people
who are tracked at other times and unmeasured at that instant - which is what recovery is
for. Everyone who is ever seen by two cameras has a track at some point.

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

## 6. Recursion from the edges (2026-11-19)

The first version only looked where a measured pose was at most 3 s away. That limit is a
property of one pass, not of the method: every accepted recovery is itself a measured pose,
so the stage now runs in passes and each pass uses what the previous one found as the hint
for its neighbours. Gaps close from both edges inwards, and tracks are followed beyond where
they first appear and last disappear, until a pass accepts nothing.

Golden: 749 positions in one pass -> 1,125 in 7 passes (931 from three or more cameras,
194 from two). Of 422 camera views checkable against golden boxes, 407 are on the right
person, 9 on another person (2.1%; it was 1.9% for a single pass), 6 low overlap. Missing
inside tracks 4,561 -> 3,483.

## 7. What is still missing, and what would change it (`rejected_impact.py`, golden)

Of the 3,483 instants still missing inside tracks:

* 2,210 were never within reach of a known pose: the recursion stopped earlier because the
  frames at the frontier did not yield an accepted pose.
* 1,273 were looked at: 622 gave no consistent pose in two cameras (the person is not
  readable in two views there), 67 had crops that follow another person, 584 triangulated
  but failed the acceptance gate.
* Independently of the cause: at 1,051 of 3,401 missing instants (31%) the same golden
  person IS measured at that instant under another engine identity. Linking those
  identities (the open re-identification and association questions) removes them without
  any recovery.

Gate-rejected recoveries (714 instants in 329 runs, inside tracks and beyond ends):

| Why the gate rejected it | Instants | Runs | What resolves it |
|---|---|---|---|
| Measurement fine, proof that it is this person missing (two cameras: residual over 3 px, low joint confidence, hint from one side only or far away) | 411 | 204 | A person saying "yes, both crops show this person" - the position is then an ordinary two-view triangulation. Or a third camera. |
| Poor measurement (implausible bone lengths, too few joints) | 238 | 175 | Pose correction (CVAT), not a yes/no. Otherwise it stays unmeasured. |
| Another identity is already measured at that place | 65 | 28 | An identity / association answer; 63 of them have the same golden person under another identity. |

So a yes/no review would add at most about 58% of the gate-rejected instants, one question
per run (a run is consecutive frames of one person). None of it is needed for a track to
exist or to carry its identity: these are isolated frames inside tracks that already have
positions around them.

## Reproduce

Research: run the scripts with data_management's interpreter in the order given in each
docstring (`gap_classify.py`, `lever1.py`, `plan_jobs.py`, `run_pass.py`,
`eval_levers.py`, `final_eval.py`, ...). Engine stage on a golden workspace and its check:

```
python -m data_engine.cli run "golden_20250505071549_20250505080000@first600s"
python experiments/coverage/check_recovered.py "golden_20250505071549_20250505080000@first600s" golden_20250505071549_20250505080000
python experiments/coverage/gap_diagnosis.py WORKSPACE [GOLDEN_DATASET]
```
