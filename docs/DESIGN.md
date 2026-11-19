# Data Engine design

Status: first version, 2026-11-18. Working example: `felix_20251223173052_20251223183103`
(7 cameras, 2048x1536, 10 fps, about 60 minutes, a class of roughly 20 people).

## 1. Principles (from the golden-set work)

| Learning from `cvat_labeling` / `data_management` | What the engine does |
|---|---|
| Iteration worked: resolved assignments made undecidable ones decidable. | The solve stage is a fixpoint loop; a human round adds constraints and reruns it. Progress is the shrinking unknown ledger. |
| Unknown is not negative; "unresolved" is a valid state; no identity by elimination, no new-person default. | Unknowns are first-class records. "Unsure" is a valid answer and creates no constraint. Elimination only removes *competitors*; the remaining candidate still has to pass its own evidence gate. |
| Human negatives are never bypassed; evidence is never deleted. | Constraints are append-only; a later human answer supersedes with provenance. `different` pairs are forbidden before any machine merge. Rejected detections stay in the raw store. |
| Answers bind to the detections actually shown, not to track numbers. | Every constraint and unknown subject is made of detection keys `camera/frame/index` into the immutable raw store. |
| Tracker ids are proposals; a missed switch is expensive, a wrong cut is cheap. | Tracklets are cut at tracker gaps and box jumps; pieces are re-joined only by evidence. |
| Two-view geometry has epipolar ambiguity; low reprojection error is not plausibility. | Pairwise evidence only bootstraps. Merges need pooled N-view support; plausibility vetoes (floor, ceiling, room); then 3D-first refinement by projection. |
| Never interpolate or invent; absent means unmeasured. | Triangulation gates (2+ views, 8 degrees, 12 px, positive depth); failed joints are NaN. |
| Never silently change models, thresholds, calibration or timing. | The recipe is hashed into every stage's input hash and manifest; the post-filter and tracker settings are read from `prelabel_policy.json` and must equal the recipe. Mount corrections are applied only on a human decision. |
| Resending an identical unclear page teaches nothing. | A question answered "unsure" is not re-asked with the same evidence. |
| Evidence images must show measured keypoints next to an unobscured person, plus room context. | The renderer draws thin skeletons on crops and offers the full frame. |
| Recording-specific offsets and calibration must not be reused blindly. | Alignment is estimated per dataset over the whole recording; a mount check runs per dataset. |

## 2. State machine

Stages are functions of (recipe section, upstream outputs, active constraints). Each run
stores an input hash; a stage is *fresh* while its hash is unchanged, *stale* otherwise.
`Engine.advance()` applies answered tasks, runs every stale stage in order, turns the
remaining unknowns into tasks and closes the round with a summary.

| Stage | Runs | Produces | Owns unknowns |
|---|---|---|---|
| `ingest` | once | `dataset.json`: per camera the ordered files with verified overlap trims; frame index | `stream_layout` |
| `detect` | once | detection shards per camera (boxes, 17 keypoints, raw heatmap peaks), every native frame | - |
| `calibration` | once, again on a human decision | `calibration/calibration.json`, mount check | `calibration` |
| `align` | once, again when calibration geometry changes | `alignment.json` (label payload), `sync_table.npz`, report | `alignment` |
| `tracklets` | once | post-filter + BoT-SORT + over-split, per camera | - |
| `autofit` | once | `evidence/autofit/labels.npz`: Autofit raw track id per detection (CPU, about a third of real time) | - |
| `solve` | every round | `solution.npz`, `tracks3d.npz`, `solve.json`, `alignment_check.json` | `association`, `identity_link`, `unassigned`, `pose_ownership`, `tracklet_purity`, `coverage`, `alignment` |
| `tasks` | every round | human tasks | - |
| `publish` | on request | label-set drafts / published sets | - |

Dataset phases (dashboard banner): `new` → `machine_pending` → `running` →
`waiting_for_human` → `answers_pending` → … → `ready_to_publish` → `published`, or
`blocked` when a stage failed.

Data structures (SQLite, `engine.sqlite3`): `rounds` (with the open-unknown summary per
round), `stage_runs` (status, input hash, metrics, manifest), `unknowns`, `tasks`,
`constraints`, `events`. Every stage run also writes a manifest under `manifests/` with the
recipe, input identities, outputs and code identity.

## 3. Taxonomy

### Unknown kinds (`taxonomy.KINDS`)

| Kind | Meaning | Machine steps tried first | Human task |
|---|---|---|---|
| `stream_layout` | Files cannot be ordered/trimmed into one verified sequence | main-channel rebuild, verified overlap trim | operator decision |
| `calibration` | View moved since calibration, or check inconclusive | rotation-only mount fit, residual audit | calibration review |
| `alignment` | A camera's offset is unresolved, unstable or stale | audio + pose-motion lag over the whole recording | alignment review |
| `tracklet_purity` | A tracklet may hold two people | gap/jump cuts, consensus switch detection | continuity |
| `association` | Which simultaneous tracklets are the same person is ambiguous | pairwise geometry with margin, pooled N-view support, elimination, projection | cross-view match |
| `identity_link` | A track starts inside the recording without a unique predecessor | unique motion continuity | re-identification |
| `unassigned` | A piece of confident detections belongs to no track | projection onto 3D tracks | detection triage / policy decision |
| `pose_ownership` | Two poses in one camera claim one person | pose suppression, 3D co-location | pose ownership |
| `coverage` | A tracked person projects into a view without a detection | (planned) high-recall re-detection | visibility check |
| `pose_quality` | Keypoints disagree with the multi-view pose | leave-one-out view rejection | pose correction (CVAT) |
| `audit` | Random sample awaiting spot check | - | audit |

### Human task types (`taxonomy.TASK_TYPES`)

Four UI kinds render all types: **decision** (pick an option), **strip** (a time-ordered row
of crops; click the first different one), **reference + candidates** (same / different /
unsure per candidate, "none of these", "reference unusable"), **frame picker**. Pose
correction is carried out in CVAT because it needs a skeleton editor.

| Task type | Question | Answer becomes |
|---|---|---|
| `operator_decision` | A policy or input decision | `undeterminable` / `alignment` constraint |
| `calibration_review` | Which calibration should this camera use? | `calibration` constraint → calibration stage |
| `alignment_review` | Which frame shows the same instant? | `alignment` constraint |
| `continuity` | Do all crops show one person? | `same` or `split` (+ `different`) |
| `cross_view_match` | Which candidates are reference A? | `same` / `different` / `not_person` |
| `reidentify` | Which earlier person is A? | `same` / `different`, or "not tracked before" |
| `detection_triage` | Is this a person? Which one? | `not_person`, `same`, or single-view person |
| `pose_ownership` | Is pose B the same person as A? | `duplicate` or `different` |
| `visibility_check` | Is the projected person visible here? | `undeterminable` (occluded / out of view) or a pose task |
| `pose_correction` | Fix the skeleton (CVAT) | corrected keypoints, training data |
| `audit` | Is this label correct? | `different` / `not_person`; precision estimate |

Tasks are generated one per unknown, largest label volume first, with a per-round budget:
the next round usually resolves many remaining unknowns by itself.

## 4. Solve stage

0. **Independent tracker** (`stages/autofit.py`, `solve/pipeline.with_external_units`).
   Autofit tracks detections frame by frame in 3D with its learned human prior, so it does
   not inherit single-camera tracker switches. Every 2D tracklet is cut where Autofit's raw
   track id changes, and pieces that one Autofit track holds together for at least five
   instants are linked - below human evidence, and only if the solver's own geometry does
   not contradict the pair or put two simultaneous detections of one camera together.
   Autofit's merged (global) ids are not used. Evidence: `experiments/RESULTS.md`.
1. **Pair evidence** (`solve/evidence.py`). For every pair of simultaneous detections in
   two cameras, the two-view residual of the common reliable body joints, normalised to
   accept units (`u <= 1`: 5 px / .025 box height; `u <= 2`: review gate). Samples that are
   behind a camera, below the floor, above the cameras or outside the room count as
   contradicting. Residual histograms are pooled per (tracklet, tracklet, 2 s window).
2. **Agglomeration** (`solve/cluster.py`). Atoms (tracklet pieces) merge along *supported*
   edges (80% of samples within the review gate) when: no same-camera overlap, no
   contradicted pair, no human `different`; the *pooled* evidence between the two components
   is supported (N-view consistency); and no competing atom is comparably good. Competitors
   whose component can no longer contain the atom are **eliminated**, and the loop repeats -
   this is the recursion that makes previously undecidable links decidable.
3. **Consensus switch detection** (`refine_cuts`). A tracklet piece supported by component
   X for a sustained run of windows and by a different component Y for another run is cut
   between the runs; a run contradicted by its own component is detached. (A single pair
   flipping is *not* a switch: two people on one epipolar line stay consistent for seconds.)
4. **3D** (`solve/tracks3d.py`): gated N-view triangulation with leave-one-out view
   rejection; nothing interpolated.
5. **3D-first refinement** (`solve/recover.py`), repeated to a fixpoint: components measured
   at the same place at the same time merge (unless one camera shows both - then it is a
   pose-ownership question); a component that ends where and when another begins is merged
   (tracker hand-off); single-camera pieces and untracked detections attach to the person
   whose *projected* 3D pose they match, by voting over the piece's lifetime, including the
   pose measured up to 1 s earlier or later for a person who stays put.
6. **Identity** (`solve/identity.py`): components link across gaps when the continuation is
   unique in both directions within walking reach, or when the person is measured again at
   the same spot (0.4 m) within 60 s and nobody else was measured within 1 m in between;
   lifetimes that overlap are never linked. Other starts inside the recording are reported
   with their candidates.
8. **Alignment check** (`solve/alignment_check.py`): per camera and 5-minute block, the
   estimated offset must be the best-agreeing frame shift; blocks that disagree are the only
   alignment unknowns.
7. **Unknowns** (`stages/solve.py`): free pieces with hypotheses → `association`; free
   pieces without → `unassigned` (short ones are grouped into one policy question per
   camera); unexplained starts → `identity_link`; co-located duplicates → `pose_ownership`.

## 5. Keeping what humans teach

* Constraints are durable and reapplied on every solve; they also survive detector or
  tracker changes because they reference raw detections.
* `same`/`different` pairs come with the geometry statistics that failed to decide them:
  they are the calibration set for the association gates (accept/review/margin).
* `not_person` crops are detector hard negatives; single-view people and `visible` answers
  are recall positives; corrected skeletons (CVAT) extend the curated pose dataset. The
  fine-tuning itself stays in `cvat_labeling/finetuning` (its rule); a new checkpoint is
  adopted by changing the site model path, which invalidates `detect` and everything after
  it through the input hash.
* Re-identification answers are positive/negative fragment pairs for a ReID model of this
  camera set (planned: use it to pre-rank and to auto-accept above a validated margin).

## 6. Known limits of this version

* Alignment offsets are solved on a 10 Hz grid (0.1 s steps); sub-frame refinement from
  linked 3D tracks is planned. One-frame shifts barely change residuals in this scene.
* `coverage`, `pose_quality`, `continuity` and `audit` tasks have UI and answer handling
  but no generator yet; the CVAT pose-correction hand-off is not implemented.
* Appearance ReID was tested and is not reliable enough to decide (0.5-1% wrong in its best
  case, `experiments/reid/REPORT.md`): identity across long gaps goes to a human. Using it
  to rank the candidates shown is implemented in `reid.py` but not wired into tasks yet.
* People that only one camera can see (outside the 3D-covered floor) need a scope decision
  per piece today; grouping them into one decision per camera region is implemented but only
  triggers where no 3D-tracked person ever appears in that part of the image.
* Thresholds marked NEW in `config.RECIPE` are provisional and unvalidated on other rooms.
* A person seen by only two cameras that lie on the same epipolar plane as another person
  can be mis-associated by geometry alone; such two-view-only components should be
  over-sampled by audit tasks.

## 7. Decision log

* **2026-11-18 - catalog mixes main and sub stream for camera 8-7.** The catalog groups
  both channels under one camera id, the overlap check fails and the mount check was
  inconclusive. Ingest rebuilds such cameras from channel 0; with that, felix-6's layout
  verifies (15-frame overlap) and its mount check reads 0.85 px (unchanged).
* **2026-11-18 - pairwise flip is not an identity switch.** Splitting tracklets where one
  pair flipped between consistent and contradicted produced 65 "switches" in 40 s and
  manufactured consistent-looking pieces. Replaced by consensus detection (2 events).
* **2026-11-18 - merges require pooled N-view support and physical plausibility**
  (ceiling and room bounds added to the golden floor veto).
* **2026-11-18 - distortion model is not invertible in some image corners** (camera 0,
  camera 6): rays from those pixels are discarded, and projections are verified by a
  round trip so people outside a view are not folded back into it.
* **2026-11-18 - dependants stay fresh when a stage's output is unchanged** (a calibration
  decision that keeps the standard calibration does not re-run alignment).
* **2026-11-18 - a link must not silently rule out an equally good alternative.** Unit
  tests on synthetic scenes showed that taking the strongest two-view edge first could merge
  two people on one camera ray even when a third camera disagreed. An edge now stays open
  while any supported neighbour it would exclude is comparably good on the *pooled*
  component evidence; an alternative is dismissed only if this edge is firmly corroborated
  by a third view, the alternative is not, and the alternative has its own counterpart in
  the same camera. Cost on the 40 s pilot: automatic 3D coverage of detections fell from
  74% to 62% and association questions rose from 17 to 39 - the price of not guessing.
* **2026-11-18 - bridge short dropouts in place.** Cutting every tracker gap over 1 s
  doubled the number of pieces. A gap of up to 5 s is now bridged when the box reappears
  in place (IoU >= .5) and nothing else overlaps the old box.
* **Known limit - switch inside a window.** Consensus switch detection works on 2 s
  windows; a switch inside a window can leave up to about 2 s of 2D detections on the
  wrong track (their 3D joints are still gated per frame). Kept as a strict expected
  failure in `tests/test_solve_cluster.py`.
* **2026-11-18 - Autofit raw tracks become the association backbone.** On the golden bench
  the geometry-on-tracklets solver assigned 1,246 observations to the wrong person, 1,016 of
  them inside 2D tracklets that contained two people (74 of 808 tracklets). Autofit alone
  made 192. Cutting tracklets at Autofit id changes and linking on shared Autofit tracks
  (geometry may veto) gives 94 wrong observations, coverage 89% -> 97% and questions
  194 -> 87. Autofit global ids and "different tracks" as negative evidence were rejected.
* **2026-11-18 - adopted on the bench:** pose reach of 1 s, hand-off merge, same-spot link
  up to 60 s, geometric alignment check (questions 87 -> 74, wrong observations 94 -> 84).
* **2026-11-18 - rejected on the bench:** limb-length veto (no gain), single-camera identity
  bridge (no effect), duplicate recognition (no fewer questions), keeping tracklets across
  Autofit track re-creation (one more mixed identity), "only candidate" identity
  elimination (unsafe: about 400 wrongly assigned observations), ReID decisions (0.5-1%
  wrong), Autofit global ids for identity (2-4 wrong links). All remain as off-by-default
  options with their measurements in `experiments/RESULTS.md`.
