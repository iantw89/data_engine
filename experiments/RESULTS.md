# Reducing human review: levers tested on the golden set (2026-11-18)

Test bench: the engine run on the first 600 s of the golden recording
(`golden_20250505071549_20250505080000@first600s`, 7 cameras), scored against the golden
identities (17 people, 64,385 observations; 61,467 matched to engine detections by box
overlap). The golden labels are used for scoring only - nothing from them enters the solver.
Scripts: `experiments/bench/bench.py` (variants), `diagnose.py`, `fragments.py`,
`handoff_debug.py`, `lag_scan.py`, `show_unknowns.py`.

Caveats: one recording, one session; the golden labels are provisional (machine evidence
plus recorded human reviews); the standard calibration was refit on these identities; the
detector was fine-tuned on this recording. Numbers are diagnostics, not an independent test.

## Result

| Solver | Coverage | Wrongly assigned obs. | Mixed identities | Identity fragments (17 people) | Wrong cross-camera pairs | Open questions |
|---|---|---|---|---|---|---|
| Geometry on 2D tracklets (first version) | 89.4% | 1,246 | 18 | 66 | 454 | 194 |
| + Autofit raw tracks as backbone | 97.0% | 94 | 6 | 38 | 128 | 87 |
| + pose reach 1 s, hand-off merge, same-spot link (adopted) | 97.0% | 84 | 5 | 32 | 128 | 74 |

"Open questions" for the adopted row is the engine's own count on that workspace
(association 12, identity across a gap 20, unassigned 40, pose ownership 1, alignment 1).
On the full felix hour the same change took open questions from 1,973 to 729 and
detections in 3D tracks from 88% to 95% (see the workspace's round history).

## Why the first version was impure

1,016 of its 1,246 wrong observations sat inside 2D tracklets that themselves contained two
people: the adopted BoT-SORT settings (match .95, 60 s buffer) trade fragmentation for
merges, which the golden workflow repaired with human continuity review. 74 of 808
tracklets were impure. A detection-level 3D tracker does not inherit that error.

## Levers

| Lever | What it does | Golden-bench effect | Decision |
|---|---|---|---|
| **Autofit raw tracks** (`stages/autofit.py`) | Cut every 2D tracklet where Autofit's track id changes; link pieces that one Autofit track holds together for >= 5 instants; the solver's geometry can veto | wrong obs. 1,246 -> 94, coverage +7.6 pts, questions 194 -> 87 | **Adopted** |
| Pose reach 1 s (`refine.temporal_reach_frames`) | A tracked person's measured 3D pose is compared with single-camera pieces up to 1 s away | questions 87 -> 85, wrong obs. 94 -> 84 | **Adopted** |
| Hand-off merge (`refine.merge_handoffs`) | Merge a component that ends where/when another begins (<= 1 s, <= 20 in, unique, no camera clash) | components 49 -> 42 | **Adopted** |
| Same-spot link (`temporal.same_spot_gap_s` = 60) | Same person if measured again within 0.4 m up to 60 s later and nobody else was measured within 1 m in between | identities 37 -> 32, no added error (10/30/60 s: 35/33/32; 120 s: 35) | **Adopted at 60 s** |
| Geometric alignment check (`solve/alignment_check.py`) | Lag 0 must be the best-agreeing shift per camera and 5-minute block | alignment questions 14 -> 1 (the one flagged block is golden camera 6, where the human-confirmed offset also differs from the generator's) | **Adopted** |
| Dropout bridging in tracklets | Do not cut a tracker gap <= 5 s when the box reappears in place and nothing overlaps it | tracklets 283 -> 211 on the pilot | Adopted earlier (not separately scored) |
| Limb-length plausibility (`geometry.bone_veto`) | Two-view pairings with impossible bone lengths count as contradicting | no gain with Autofit (questions 87 -> 89) | Rejected (code kept, off) |
| Single-camera identity bridge (`refine.bridge_identities`) | One clean single-camera piece matching two non-overlapping people joins them | no effect | Rejected (off) |
| Duplicate recognition (`refine.mark_duplicates`) | A piece matching the projected pose of a person who already has a detection in that camera | 23 detections, no fewer questions | Rejected (off) |
| Keep tracklets across Autofit track re-creation (`refine.external_rebirth`) | Do not cut when the two Autofit tracks never coexist | identities 32 -> 31 but one more mixed identity | Rejected (off) |
| "Only candidate" identity elimination (`temporal.only_candidate_gap_s`) | Link when exactly one unaccounted person could have reached the start | up to 400 wrongly assigned observations for 1-2 fewer fragments | **Rejected: unsafe** (confirms the golden rule: no identity by elimination) |
| Appearance ReID (see `experiments/reid`) | OSNet (Market-1501) and fast-reid SBS R50-NL (MSMT17) embeddings | rank-1 across a gap 78% / 90% (multi-camera / same-camera gallery); frozen zero-error-on-train rules still made 0.5-1% wrong decisions across camera folds; cross-camera tie-break 62% rank-1 | **Not used to decide.** Suitable for ranking candidates shown to a human |
| Autofit global (merged) ids | Autofit's own cross-gap re-identification | 2-4 wrong links on the first half of the data at every threshold | **Rejected for identity** |
| Autofit "different tracks" as negative evidence | Eliminate candidates Autofit separates | up to 2.8% of true pairs wrongly eliminated | Rejected as a hard rule |

## Added after the lever study (2026-11-18)

| Mechanism | What it does | Measured | Decision |
|---|---|---|---|
| File-boundary check (`overlap.py`) | Compares decoded frames where the catalog could not verify a join between two files | felix-5: 6 repeated frames found (difference 0.0 vs 0.43 next best); the geometric alignment check then went from 6 disagreeing blocks to 0, identities 408 -> 366, open unknowns 654 -> 585 | **Adopted** |
| Swap check (`solve/swap_check.py`, `experiments/switches/REPORT.md`) | Finds every step where two tracks could have exchanged people; tests crossings with heading, body proportions and 2D tracklets | golden: 2 questions per 10 min, 100/100 injected swaps flagged; felix: 81 questions per hour, a real swap found among the first examples viewed | **Adopted** (review only; nothing is relabelled) |
| Coverage recovery (`stages/recover.py`, `experiments/coverage/REPORT.md`) | Reads poses at each track's expected place and triangulates them | golden: 749 of 5,065 missing person-frames recovered; 0 false accepts in 6,500 controls for three-camera recoveries. felix: 17,078 positions (+4.5%), missing frames inside tracks 49,247 -> 33,493 | **Adopted** |
| Recursive recovery (`coverage.max_passes`) | Each recovered pose is the hint for its neighbours in the next pass | golden: 749 -> 1,125 positions, right-person share of checkable views 96% | **Adopted** |
| Hand-offs singled out by 2D tracklets (`refine.handoff_tracklets` = 2) | Merge a track that ends where another starts when tracklets in two cameras continue into it and into no other | golden: identities 32 -> 31, no added error | **Adopted** |
| Appearance cue in the swap check (`appearance.py`) | fast-reid embeddings before/after a crossing, as one of the two independent cues a verification needs | felix: swap questions 81 -> 51 (engine ledger 75 -> 43 open); injected swaps at crossings still never verified (0 / 86) | **Adopted** |
| Detection under a recovered pose takes the identity (publication) | An unassigned detection with the same pose as a recovered one is that person's detection | golden: 206 detections; where labelled, right person 50 of 52 | **Adopted** |
| Autofit lap (`autofit_refresh.py`, `experiments/autofit/augmented.py`) | Recovered poses that are not kept detections go to Autofit as detections; Autofit, solve and recovery run again when the recovery produced enough of them | golden: Autofit track ids per labelled person 45 -> 39; solve identities 31 -> 27 (bench), impure observations 84 -> 50; engine 33 -> 29 identities, second lap declined (23 new detections); felix: lap with 83,262 detections (5.9% of Autofit's input) Autofit ids 550 -> 394, identities 317 -> 270, identity-link questions 226 -> 193; a second lap with 4,172 (0.28%) changed nothing in its favour (270 -> 279) for 66 min, so the share threshold went from 0.1% to 0.5% | **Adopted**, conditional |
| Identity lap (`stages/recover.py`, `pipeline.identify`, `experiments/coverage/links_after_recovery.py`) | Recovered positions move track ends; the solve decides identity again on its kept association and the recovery continues from its previous result | felix: 8 more links in a 2-minute lap, but 2 of the 8 join two people when viewed; golden: none | **Adopted as proposals** (shown first in the re-identification question, not applied) |
| Ranking re-identification candidates (`solve/link_ranking.py`, `experiments/links/REPORT.md`) | Order the earlier tracks a new track could be by motion plus ReID appearance on clear views only (most joints found, not cut, not overlapped); before: nearest first | felix answers: a confirmed match above a rejected candidate in 88% of pairs (distance: 74%), first candidate a match in 18 / 21 questions (15 / 21); golden: with only partial or overlapped crops the right person is first in 66% of galleries, with clear views in 80% | **Adopted** (order of the candidates only; nothing is joined or closed by it) |
| Frames shown in questions (`solve/view_choice.py`, `experiments/images/REPORT.md`) | Exchange each image of a group for the nearest clear view of the same piece of the same camera's tracklet; before: the frame nearest to the track end or crossing | felix answers: clear images 41% -> 77%, groups without a clear image 35% -> 8%; no wrong-grouping flag known to be hidden (20 of 31 untouched, 10 of the other 11 checked); 85% of unclear images drew no flag | **Adopted**, with one question in five kept on the old frames as a control |
| Cap on automatic identity links (`temporal.auto_max_distance_m` = 1.5 m) | A unique continuation farther away stays a candidate | felix: 6 of 7 links viewed beyond 3.5 m join two people; links 115 -> 95, identities 279 -> 299; golden unchanged (farthest link 1.39 m, all right) | **Adopted** |
| Recovered positions in the swap check (`swap_check.use_recovered`) | Measured positions through a crossing leave less room for an exchange | felix: 19 crossings stop being ambiguous, all of them already verified by cues; open questions 38 -> 42 | Not adopted |
| Existing low-score detections near the hint | Attach raw candidates the post-filter dropped | 41 frames beyond guided crops at the same gate | Not adopted (too little) |
| Other detectors on a zoomed crop (YOLOv8x, YOLO11l) | Detector on a 3-5x ROI, ViTPose on its boxes | +46 frames (5%) for about 14 min per recorded hour; 5x zoom halves recall | Rejected |

## What the remaining questions are (golden)

* **Unassigned (40)**: real people that only one camera can see - a receptionist at a desk
  behind a doorway, people in an adjoining room - plus one policy question per camera for
  short fragments. No machine rule can give them 3D; they need a scope decision.
* **Identity across a gap (20)**: a person who walks while unmeasured, or is unmeasured for
  more than a minute. Appearance is not reliable enough here to decide.
* **Association (12)**, pose ownership (1), alignment (1).

## Not done

* A second detector (GroundingDINO boxes + ViTPose, or the high-recall YOLO variant) was not
  added: on the golden set 97% of reference observations are already in 3D tracks and the
  remaining questions are not caused by missed detections. It would matter for the
  `coverage` unknown kind (projected-but-undetected people), which has no generator yet.
* Other trackers (ByteTrack, OC-SORT) were not tried: Autofit already removes the dominant
  2D-tracker error, and tracklets are now only cut finer, never trusted across its changes.
* ReID fine-tuned on confirmed identities of this gym is the most promising untested lever
  for identity across gaps (the ReID report's top recommendation).
