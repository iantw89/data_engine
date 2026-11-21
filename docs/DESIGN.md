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
remaining unknowns into tasks and closes the round with a summary. A stage may ask for a
*lap*: earlier stages run again within the same round because it produced something they
consume (`StageResult.rerun`). The stage judges whether a lap is worth its cost; the engine
bounds the laps of a round and keeps an unserved request for the next one. Tasks are built
once, after the last lap. The recover stage asks for two kinds of lap: an *Autofit lap*
(Autofit, the whole solve, the recovery: about an hour per recorded hour) and an *identity
lap* (the solve decides identity again on its kept association, the recovery continues
from its previous result: seconds to minutes).

| Stage | Runs | Produces | Owns unknowns |
|---|---|---|---|
| `ingest` | once | `dataset.json`: per camera the ordered files with verified overlap trims; file boundaries the catalog could not verify are compared pixel by pixel (`overlap.py`); frame index | `stream_layout` |
| `detect` | once | detection shards per camera (boxes, 17 keypoints, raw heatmap peaks), every native frame | - |
| `calibration` | once, again on a human decision (base: the dataset's stored calibration in data_management, else the site default) | `calibration/calibration.json`, mount check | `calibration` |
| `align` | once, again when calibration geometry changes | `alignment.json` (label payload), `sync_table.npz`, report | `alignment` |
| `tracklets` | once | post-filter + BoT-SORT + over-split, per camera | - |
| `autofit` | once; again as a lap when `recover` produced enough detections it has not seen | `evidence/autofit/labels.npz`: Autofit raw track id per detection (CPU; 13 min for the felix hour, 2.5 min for golden's 10 min); input: the kept detections plus `evidence/autofit/extra.npz` (recovered poses) | - |
| `solve` | every round; again as an identity lap when `recover` measured positions that make more tracks linkable (association reused from `solve_state/`) | `solution.npz`, `tracks3d.npz`, `solve.json`, `alignment_check.json`, `swap_check.json` | `association`, `identity_link`, `unassigned`, `pose_ownership`, `tracklet_purity`, `coverage`, `alignment`, `identity_swap`, `pose_quality` |
| `recover` | every round, after `solve` | `tracks3d_recovered.npz` (positions triangulated from poses read at each track's expected place), `recover_rejected.npz`, `recover.json` (incl. the decisions about an Autofit lap and an identity lap); crop poses cached in `recovered/`. After an identity lap it keeps its previous positions and looks again only around the people whose tracks were joined | - |
| `tasks` | every round | human tasks | - |
| `publish` | on request | label-set drafts / published sets | - |

The dashboard shows the round as it happens ("This round"): every stage run in the order
it ran, grouped into the first pass and the laps, with the running stage, its progress and
the stages expected next (`Engine.timeline()`; the expectation is a prediction - a stage
whose output did not change leaves its dependants alone, and the recovery can add a lap).
Earlier rounds are listed with what ran and how long each stage took. The stage cards
below it show only the latest run of each stage.

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
| `tracklet_purity` | A tracklet holds two people: a reviewer marked two of its crops as different people | gap/jump cuts, consensus switch detection (before any question) | continuity |
| `association` | Which simultaneous tracklets are the same person is ambiguous | pairwise geometry with margin, pooled N-view support, elimination, projection | cross-view match |
| `identity_link` | A track starts inside the recording without a unique predecessor | unique motion continuity | re-identification |
| `identity_swap` | Two tracks came within reach (or exchanged places) and nothing independent of the 3D tracker confirms who is who afterwards | step-by-step 3D continuity, heading, body proportions, 2D tracklets in separated views, (appearance) | swap check |
| `unassigned` | A piece of confident detections belongs to no track | projection onto 3D tracks | detection triage / policy decision |
| `pose_ownership` | Two poses in one camera claim one person | pose suppression, 3D co-location | pose ownership |
| `coverage` | A tracked person projects into a view without a detection | guided re-detection (`recover` stage) fills what triangulates; the rest is counted, not yet asked | visibility check (no generator yet) |
| `pose_quality` | A reviewer reported a pose as wrong ("bad pose" flag); its keypoints are left out of 3D | - | pose correction (CVAT; hand-off not built, the unknown stays open) |
| `audit` | Random sample awaiting spot check | - | audit |

### Human task types (`taxonomy.TASK_TYPES`)

Four UI kinds render all types: **decision** (pick an option), **strip** (a time-ordered row
of crops; click the first different one), **groups of images** (a reference group and
candidate groups; below), **frame picker**. Pose correction is carried out in CVAT because
it needs a skeleton editor.

| Task type | Question | Answer becomes |
|---|---|---|
| `operator_decision` | A policy or input decision | `undeterminable` / `alignment` constraint |
| `calibration_review` | Which calibration should this camera use? | `calibration` constraint → calibration stage |
| `alignment_review` | Which frame shows the same instant? | `alignment` constraint |
| `continuity` | Do all crops show one person? If not, where does a different person start (several places possible)? | `same`; per switch `split` (+ `different`), `same` within each run and for a person who comes back; `not_person` for marked crops |
| `cross_view_match` | Which tracked person, if any, is the person in A? (one) | `same` / `different`; none and no further candidate: single-camera track |
| `reidentify` | Which earlier tracks are A's person? (several possible) | `same` / `different`; none and no further candidate: separate person |
| `swap_check` | Which candidate (after) is A (before)? (one, neither, or both) | no swap: `same`; swap: `same` + `split`; neither: `split`; both: `same` twice, then for the other track before the moment `same` (one person tracked twice) or `split` (it moved onto this person) |
| `detection_triage` | Is this a real person? | `not_person`, or single-view person |
| `pose_ownership` | One person detected twice, or two people? | `duplicate` (with the better pose) or `different` |
| `visibility_check` | Is the projected person visible here? | `undeterminable` (occluded / out of view) or a pose task |
| `pose_correction` | Fix the skeleton (CVAT) | corrected keypoints, training data |
| `audit` | Is this label correct? | `different` / `not_person`; precision estimate |

Tasks are generated one per unknown, largest label volume first, with a per-round budget:
the next round usually resolves many remaining unknowns by itself. Open tasks are rebuilt
from the current solve at the end of every round; answered ones are never touched.

**Groups of images** (`cross_view_match`, `reidentify`, `swap_check`, `detection_triage`,
`pose_ownership`; answer model 2, `taxonomy.IMAGE_FLAGS`, `tasks/apply.py: _groups`). Each
group is a set of crops the engine believes to be one person. The person answers in two
layers, and the screen says in words what the answer will do before it is sent.

1. *Any single image can be flagged.* Four flags take the image out of its group, and the
   question is then answered with the images that remain; the fifth leaves it in.

   | Flag | Means | Becomes | Follow-up |
   |---|---|---|---|
   | Someone else | a clean detection of another person: the grouping is wrong here | `different(image, rest of its group)` | the solve regroups; if image and rest lie in one single-camera tracklet, a `tracklet_purity` unknown and a `continuity` task ask where the tracklet changes person |
   | Two people in one pose | the skeleton takes joints from two people | `not_person(image, mixed_pose)`, that detection only | - |
   | Not a person | reflection, poster, equipment | `not_person(image)`: that detection, or its tracklet when every image of the group is flagged | - |
   | Unsure | too little of the person is visible to tell whether the image belongs | nothing: the answer does not cover the image, and it stays where the engine put it | a group with no other image left counts as "unsure": the question stays open and is not asked again with the same images |
   | Bad pose | the right person, keypoints wrong | `bad_pose(image)`: its keypoints give no rays (no part in 3D), the detection keeps its identity | a `pose_quality` unknown, open until pose correction exists |

2. *A verdict on what remains.* Per candidate: same, different or unsure
   (`pose_ownership`: one person / two people / unsure, and for one person which pose is
   better; `detection_triage`: real person / not a person / unsure). A candidate with no
   image left needs no answer. With no image left of the reference the flags are the whole
   answer.

There is no "reference unusable" and no "none of these". The first is what flagging the
reference's images says, image by image and with a reason. The second is what "different"
for every candidate says; what follows from it is the engine's business: if it holds
further candidates back (`more_candidates`), the question stays open and they are asked
next; if not, the question is closed as it stands (the piece stays a single-camera track,
the person stays a separate identity, the track is cut at the swap moment). "Unsure" for
any candidate keeps the question open and still records every "different".

Several candidates can be the same person in `reidentify` (a track that broke more than
once); two candidates measured at the same instants in different places exclude each other
(`cannot_both`). `cross_view_match` takes one.

**Candidates of a re-identification are ranked** (`solve/link_ranking.py`,
`RECIPE["link_ranking"]`, `experiments/links/REPORT.md`). Every earlier track that had ended
and could physically have reached the spot (12 m/s, within 120 s) is a candidate; the four
likeliest are shown, the rest follow when those are rejected. The order adds two
log-likelihood ratios (same person against a different person):

* *motion* - the distance between the candidate's last and the new track's first position
  for the time that passed: people mostly stay where they are, or walk;
* *appearance* - ReID similarity of the 10 s of each track next to the gap, **only on clear
  views**: at least 13 of 17 joints found, the box not cut by the image border, not
  overlapped by another person's box, at least 96 px tall. Tracks seen in the same camera are
  compared there (the reliable comparison); otherwise their views are pooled. Without a
  clear view of both tracks appearance is *unknown* (zero), never "different".

The card of each candidate says what its place rests on ("appearance alike (0.71, same
camera)", "appearance not compared (no clear view of both)"). The ranking joins, excludes
and closes nothing; a link proposed after the coverage recovery still comes first.

**Which frame of a track a question shows** (`solve/view_choice.py`,
`RECIPE["task_images"]`, `experiments/images/REPORT.md`). A group shows one detection per
camera, picked where the question is: a track's start or end, the middle of a piece, just
before or after a crossing. Those are the moments with the worst views; 41% of the images
shown were clear views although 68% of all tracked detections are. The detection is
therefore exchanged for the nearest clear view of the *same piece of the same camera's
tracklet* within the time the rule may choose in (0.4 s away in the median). Cameras and
pieces stay what the question chose, so a wrong grouping is still shown; single-detection
problems (a mixed pose, a sliver) are shown less often, which is intended: questions about
tracks find those by chance only. A question is still recognised by the nearest frames
(its subject and its closure), so the change re-asks nothing. One question in five keeps
the nearest frames (`control_share`, fixed per question; `image_choice` in the task
payload), so that answers keep measuring what the clearer frames hide.

**What a person said about an image is used at once and again later.** One track is a
candidate in many questions and every question shows the same few images of it. A flag is
therefore pre-set on the same image in every other open question, marked "you said so in
#N" and undoable (`tasks/carry.py`; "someone else" only next to at least one image it was
said about). At the next round the answer is applied: an image flagged "someone else" may
not share a track with the rest of its group (cannot-link in the association), so its piece
leaves the track; an image left out as "unsure" is not shown again where the track has
another one.

**Strips** (`continuity`; `validate.strip_segments`, `tasks/apply.py: _strip`). A strip is
a time-ordered row of crops of one single-camera tracklet. The person marks every crop at
which a different person starts; the crops between two marks are one run. A run is one
person (`same`), neighbouring runs are different people (`split`: the tracklet is cut at
the first judged crop of the later run, and the two are kept apart). The tracker can hop
onto a passer-by and back, so a run from the third on can be said to show an earlier
person again (`same` across the stretch in between) or nobody seen before in the strip
(`different` from every earlier one); when neither is said, nothing is recorded between
runs that are not neighbours. Single crops can be taken out, as images can in a group:
unclear (nothing recorded), not a person, two people in one pose (`not_person` for that
detection; for the tracklet when every crop is marked not a person). Marked crops are on
neither side of a switch.

The piece is the part of the camera's tracklet the solver handles as a unit; the
single-camera tracker may have followed the same box for longer. When, after the solve,
at least 5 detections of that tracker's tracklet within 15 s of the flagged image are
still in the track, a `tracklet_purity` unknown asks a `continuity` question on that
stretch, with the flagged image marked among the crops (`stages/solve.py: left_behind`;
felix dry run: 4 of 23 flagged images, 9 to 78 detections each). "The track changes person
at crop n" cuts the tracklet there and keeps each side together; "all one person" joins
the neighbours to the flagged image, and they leave the track with it. Detections beyond a
cut a person already confirmed are not asked about again.

`swap_check` takes both: after the moment the two tracks then show one person. The reference
is joined to both, and one more answer is asked for on the same screen, about the images of
the other track just before the moment (shown for comparison in every swap check): the same
person (one person tracked twice: joined), someone else (the other track moved onto this
person: it is cut at the moment, `split`), or unsure (nothing recorded about its earlier
part). Without such images the follow-up is not asked.

The comparison images are a group like the others (`validate.flag_groups`): any of them can
be flagged (someone else, two people in one pose, not a person, unsure, bad pose), with the
same effect as elsewhere. The follow-up binds only the images left; with none left it is
not asked and nothing is recorded about the other track's earlier part.

Not yet: flags on the images of the strip and decision screens; the CVAT hand-off that
would close `pose_quality`.

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
   unique in both directions within walking reach and at most 1.5 m away
   (`temporal.auto_max_distance_m`), or when the person is measured again at
   the same spot (0.4 m) within 60 s and nobody else was measured within 1 m in between;
   lifetimes that overlap are never linked. Other starts inside the recording are reported
   with their candidates. Where a track was "last measured" includes the positions of the
   coverage recovery once they exist (item 12).
8. **Alignment check** (`solve/alignment_check.py`): per camera and 5-minute block, the
   estimated offset must be the best-agreeing frame shift; blocks that disagree are the only
   alignment unknowns.
9. **Swap check** (`solve/swap_check.py`). Geometry cannot see two people exchanging
   tracks where they meet. For every pair of identities and every step between instants at
   which both are measured, keeping the labels is compared with exchanging them (summed
   movement, on 5-frame medians). A step where exchanging is clearly cheaper is an
   *exchange of places* (flagged); a step where neither is clearly cheaper - the two are
   within reach, or one was unmeasured for a while - is a *crossing*, tested with evidence
   that does not come from the 3D tracker: heading through the crossing, 3D body
   proportions before and after, single-camera tracklets in views where the two stay apart,
   and appearance (`appearance.py`: fast-reid embeddings of up to six crops per camera
   before and after, computed only for the moments the other cues did not verify). Two cues
   for "stay" and none for "swap" verify a crossing; otherwise a person compares before and
   after. A track that ends within reach
   of an established one is checked the same way, one-sided (takeover). Pairs that share a
   spot for their whole common life are one person tracked twice and are skipped.
10. **Coverage recovery** (`stages/recover.py`, `solve/coverage.py`), in passes. For every
   instant missing inside a track (and 1 s beyond its ends) with a known pose at most 3 s
   away - measured, or recovered by an earlier pass - that pose says where to look: it is
   projected into each camera that should see the
   person, the top-down pose model reads a pose from each box (twice), and a position is
   accepted only as a gated triangulation of those poses - three or more cameras (tier A),
   or two cameras with strong image evidence inside a bracketed gap (tier B). A crop is
   dropped when another measured person overlaps the box and the pose follows them. The
   hint never enters a position. Passes repeat until one accepts nothing.
11. **Autofit lap** (`autofit_refresh.py`). A recovered pose that does not coincide with a
   kept detection is a new detection. Autofit gets these as ordinary detections - without
   identities; it associates them itself - and tracks again; the solve then works from
   longer independent tracks and the recovery from fewer, longer identities. The recover
   stage asks for the lap only when it is worth about a round of compute: enough
   detections Autofit has not seen (count and share of its input), the previous lap
   removed identities, and the round's laps are not used up. The detections accumulate
   across laps (valid for one alignment and calibration), so Autofit's input never
   shrinks.
12. **Identity lap** (`stages/recover.py: consider_relink`, `pipeline.identify`). The
   recovery measures positions before a track's first and after its last measured
   position. Two tracks that were too far apart in time or space for the linking rule of
   item 6 can then be a unique continuation of each other. After a recovery that does not
   warrant an Autofit lap, the recover stage runs the linking rule on the identities with
   and without the recovered positions; if the recovered positions add a link, it asks for
   an identity lap. The solve then reuses its association (`solve_state/`, valid for the
   hash of what the association reads), adds the recovered positions to the person-frames
   the identity step summarises and keeps every link it accepted on the measured positions.
   What it does with the additional links is `RECIPE["relink"]["apply"]`:
   * `False` (current): they are **proposals**. The earlier track is shown as the first
     candidate of the later track's re-identification question, with the gap and distance
     measured from the recovered positions; identities are unchanged.
   * `True`: they are accepted (`after_recovery` in `solve.json`); the recovery that
     follows starts from its previous result - positions are tied to components, not
     identity numbers - and looks again only around the joined people, where the gap
     between the two tracks is now inside one track. Laps repeat while they join tracks.
   Proposals only, because on the felix hour two of the eight links join two different
   people (decision log, 2026-11-19).
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
* Coverage: recovery stops where the frames at the frontier of what is known do not yield
  an accepted pose; poses that triangulate but fail the acceptance gate are recorded
  (`recover_rejected.npz`) and not shown to a person; recovered positions feed back into
  identity (item 12) but not into association: they do not let the solve attach more
  detections (only the detection directly under a recovered pose takes the identity, at
  publication), and a recovered pose that lies on another identity's detection is counted,
  not used (`experiments/coverage/REPORT.md` section 10). A track followed
  beyond its end has no measured pose on the far side to anchor it: nothing but frame-to-
  frame continuity and the absence of another measured identity ties a 30 s extension to
  the same person (the nine longest on felix were viewed and are the same person).
* Autofit lap: the recovered detections reach the solver only through Autofit's track ids
  on the raw detections around them. They are not solver rows, so the single-camera
  tracker does not see them and the solve does not triangulate from them (their positions
  come from the recover stage). Making them first-class detections of the store would
  touch every row-indexed artifact; not done. Autofit's own assignment of each recovered
  pose (`extra_raw`) is written but not yet used to cross-check the recovery's identity.
* Identity links are decided by physics alone: "the only track that begins within reach of
  where this one ended" is taken for the same person. It fails when the earlier person is
  still there but no longer tracked and somebody else's track begins nearby. The 1.5 m cap
  removes the cases seen on felix beyond 3.5 m; one wrong link was seen at 0.12 m (two
  people side by side, through recovered positions), so short links can be wrong too and
  nothing independent (appearance, body proportions) checks an automatic link today
  (appearance only orders the candidates of the questions). Of the felix links at most
  0.41 m apart, the three viewed are right.
* Re-identification candidates end 120 s before the new track at most
  (`temporal.candidate_gap_s`). Someone away for longer is never offered, and "none of
  these" then closes the question as a separate person. Felix answers confirm matches up to
  118 s, so the limit cuts real matches off; it was left alone until the ranking has been
  used for a round (a wider window multiplies the candidates of every question).
* A verdict "group A is not group B" can contradict the single-camera tracker in the same
  way a flag on an image can: an image of A lies in a tracklet whose neighbouring detections
  are in B's track (felix dry run: 14 cases). It is not asked about yet; unlike a flag on
  one image, "all one person" would there contradict the verdict itself.
* The ranking's motion model uses round numbers and its appearance curve was fitted on
  the golden recording; on the felix answers appearance orders well and motion adds little
  beyond what distance already gave (`experiments/links/REPORT.md`).
* Swap check: its body and appearance thresholds were fitted on one recording, and its
  recall is known only for injected swaps. A swap of two people who pass within about
  0.3 m with similar builds and similar clothes rests on the 2D tracklet cue or a person.
  A confirmed swap cuts the track at the reviewed
  detection (within about 10 s of the swap), not at the exact frame.
* Tracks that hand off with a short overlap (one person tracked twice for a moment) are
  recognised and skipped by the swap check but are not yet offered to identity linking as
  the obvious candidate (38 such cases on the felix hour).
* A file boundary with a *gap* (frames missing between two files) or a re-encoded overlap
  is not measurable by pixel comparison. The geometric alignment check then reports the
  camera's later blocks as `alignment` unknowns, but a human alignment checkpoint is
  recorded only: applying it (or the lag the check measured) to the camera's clock is not
  implemented.
* `coverage`, `pose_quality`, `continuity` and `audit` tasks have UI and answer handling
  but no generator yet; the CVAT pose-correction hand-off is not implemented.
* Appearance ReID was tested and is not reliable enough to decide (0.5-1% wrong in its best
  case, `experiments/reid/REPORT.md`): identity across long gaps goes to a human. It orders
  the candidates that person is shown (above), on clear views only.
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
* **2026-11-18 - file boundaries are verified by pixels when packets do not match.** The
  geometric alignment check flagged felix-5 for the whole second half-hour of the felix
  recording; a lag scan gave a clear best shift of -6 frames in every 5-minute block after
  the camera's second file starts and 0 before. The catalog had placed that file right
  after the first ("continuation without overlap evidence"). Decoded frames show its first
  6 frames are identical to the first file's last 6 (difference 0.0 grey levels, next best
  alignment 0.43). Ingest now measures such repeats (`Part.overlap`; decided only for an
  exact, unique repeat of at least 2 frames), the frames stay in the frame index (it must
  equal data_management's, and the detection store is keyed by it) and the alignment
  worker takes them off the timeline. Two independent measurements (geometry, pixels)
  agree; neither is a guess.
* **2026-11-18 - swap check added; the first rule was rejected after looking at felix.**
  Research on golden (`experiments/switches`): the engine solution has no track-level swap
  there (its 26 error episodes are 25 single-camera contaminations and one identity-link
  takeover), so recall was measured by exchanging labels after real encounters. A rule
  "two of motion / body / tracklet cues over 10 s chunks of every encounter under 1 m"
  flagged 97% of injected swaps at 3 questions per 10 min on golden - and 210 per hour on
  felix, where rendered examples showed people walking together (position-only motion cue)
  and duplicate tracks of one person. Replaced by step-wise continuity: only steps where
  exchanging the labels is not clearly costlier are candidates. Golden: 6 moments, 2
  questions per 10 min; felix: 107 moments, 81 questions per hour (10 exchanges of places -
  at least one a real swap between two people at adjacent rowers); injected swaps while
  apart 100/100 flagged on both, none verified; at crossings none of 86 injected swaps
  and 1 of 86 takeovers verified on felix.
* **2026-11-18 - coverage recovery by guided top-down pose.** Research on golden
  (`experiments/coverage`): of 5,065 missing person-frames, 2,190 are in gaps longer than
  3 s; of the rest most have a detection in only one of the cameras that see the place.
  ViTPose on the projected box recovered 816 at a gate with 0 false accepts in 6,500
  negative controls for three-camera recoveries (one for two-camera, on a held hint - hence
  tier B requires a bracketed gap), median error 0.5 in against withheld poses. Existing
  low-score detections add about 5%, zoomed-ROI detectors (YOLOv8x, YOLO11l) another 5% at
  14 min per recorded hour: not adopted. In the engine: 749 positions on golden, 97% of
  318 checkable camera views on the right golden person (4 of the 6 others follow a track
  that is itself on the wrong person).
* **2026-11-19 - recovery recurses from the edges; recovered 2D poses are published.** The
  3 s reach of a single pass is not a limit of the method: each accepted recovery is a
  measured pose and is the hint for its neighbours in the next pass (`max_passes`). Golden:
  749 -> 1,125 positions over 7 passes, 96% of checkable views on the right person (2.1% on
  another person, 1.9% before). The 2D poses the positions were triangulated from are now
  published as 2D detections with their own provenance (`data_engine.recover.*`), their own
  tracklets and membership in the 3D track; a pose that coincides with an already published
  detection (IoU >= 0.5) is not published twice (that detection keeps its own assignment).
  Every instant that was looked at and is still missing is written with its reason
  (`recover_rejected.npz`); see `experiments/coverage/REPORT.md` for what a human answer
  would and would not change.
* **2026-11-19 - appearance cue in the swap check.** Appearance alone is not reliable
  enough to decide identity, but as one of two required independent cues at a known
  moment between two known tracks it removes questions: on the felix hour 81 -> 51 swap
  questions (verified 26 -> 56 of 107 moments). Injected swaps with the cue: 100/100
  flagged while apart; at crossings 0 of 86 swaps verified, 3 of 86 takeovers verified
  (all three at moments the check had flagged on the real data, i.e. where the real
  track already looks swapped). Embeddings are computed only for unverified moments.
* **2026-11-19 - a detection under a recovered pose takes the identity.** 37% of the
  recovered pose views on golden coincide with a detection the detector had made: 1,043
  the track's own single-view detection (the two pose readings agree to 0.6% of the box
  height at the median), 215 belonging to no identity. Where labelled, those unassigned
  detections are the track's golden person 50 of 52 times - the same rate as recovery
  itself - so at publication an unassigned detection whose own keypoints show the same
  pose (median joint distance <= 5% of box height) takes the track's identity, with the
  recover provenance as its tracklet provenance (206 on golden). Detections of another
  identity under a recovered pose (24) are left alone and counted.
* **2026-11-19 - the pass limit is a safety bound.** With a limit of 40 passes felix was
  still recovering about 30 positions per pass (two tracks being followed 1 s per pass
  beyond their ends). The limit is now 400 and the summary says when it is hit.
* **2026-11-19 - Autofit runs again with the recovered poses (a lap), when worth it.**
  Experiment on golden (`experiments/autofit/augmented.py`): 2,240 recovered poses that are
  not kept detections (0.7% of Autofit's input) took Autofit from 45 to 39 track ids over
  the 17 labelled people (labelled detections on a track that is mostly someone else:
  0.32% -> 0.29%) and the bench solve from 31 to 27 identities, impure observations
  84 -> 50, identity questions 17 -> 15. In the engine the lap ran end to end on golden:
  33 -> 29 identities; recovered poses on another golden person 9 of 422 -> 2 of 416
  checkable views; the second recovery found 23 new detections and correctly declined a
  further lap. Positions did not increase on golden (79,558 -> 79,552 measured, 1,125 ->
  1,108 recovered): the gain there is consolidation and purity, not coverage. Autofit is
  cheap here (2.5 min golden, 13 min per felix hour, CPU); the cost of a lap is the solve
  and recovery that follow, hence the conditions in `RECIPE["autofit_refresh"]`.
* **2026-11-19 - the lap on the felix hour; share threshold 0.1% -> 0.5%.** Lap 1: 83,262
  new detections (5.9% of Autofit's input), Autofit track ids 550 -> 394, identities
  317 -> 270, identity-link questions 226 -> 193, about an hour (Autofit 12 min, solve
  11, recovery 40). Lap 2: 4,172 new detections (0.28%), Autofit track ids unchanged,
  identities 270 -> 279, identity-link 193 -> 197, 66 min. The rule had let lap 2 run
  because lap 1 paid; it could only have been stopped by its size. `min_new_share` is now
  0.5%: it keeps golden's lap (0.7%) and felix lap 1 and drops felix lap 2. Positions
  barely move in any lap (felix 408,991 -> 408,748 in total); the lap buys consolidation.
  Table in `experiments/coverage/REPORT.md` section 9.
* **2026-11-19 - answers of the group tasks redesigned (answer model 2).** Before: per
  candidate same / different / unsure, a "reference unusable" switch, "none of these", and a
  problem menu per group taken over from the exploratory CVAT tasks. Now: flags on single
  images (someone else, two people in one pose, not a person, bad pose) and a verdict on
  what remains; closing a question when no candidate fits is derived from the answers and
  from whether the engine has more candidates, not asked. Reasons: a wrong image in a
  group does not make the group unusable; "none of these" and "every candidate is
  different" were two ways to say one thing with different effects; a mixed pose and a bad
  pose are different facts with different consequences. Re-identification takes several
  "same" answers. Pose ownership asks which pose is better. Answers given under model 1
  keep their meaning (`apply._reference_candidates`).
* **2026-11-19 - re-identification candidates ranked by motion and clear-view appearance;
  image flags carried between open tasks.** Before: candidates within reach, nearest
  first, four at a time, appearance unused. The reviewer's answers on felix showed that
  distance orders well only for people who barely moved (11 of 23 confirmed matches within
  0.25 m, 8 between 1.9 and 8.8 m among the rejected ones) and that 9 of 23 questions had no
  match among the four shown. Now the order is motion + appearance (`link_ranking`,
  recipe section added). Appearance counts only on clear views - the reviewer's
  observation that ReID is useless on partial skeletons and heavy occlusion, measured on
  the golden reference: with only partial or overlapped crops the right person is first in
  66% of galleries, with clear views in 80%. A new section of the recipe, no label
  threshold: nothing is joined, excluded or closed by it. Its embeddings are a cache of
  their own (`evidence/appearance_links`), so the swap check's cue values are untouched.
  Flags on images are pre-set in the other open tasks that show the same image, and
  images left out as "unsure" are not shown again where the track has others: the
  reviewer was flagging the same image of a candidate track in question after question
  (9 of 58 flags on felix), because answers take effect only at the next round.
* **2026-11-21 - a strip takes several switches and marks on single crops (answer model 5).**
  One "first different crop" could not describe a tracker that hops onto someone and back,
  or two changes in a row, and a crop that was a mixed pose or not a person could only be
  called unclear. Non-neighbouring runs are left unrelated unless the person says who it
  is: "different from the run before" does not say "different from the run before that".
* **2026-11-20 - questions show the clearest frame of the same piece; a control share keeps
  the nearest.** The reviewer found many images hard (slivers, cut or overlapped poses) and
  asked whether easier images would hide the problems being caught. Measured on the 720
  images of 84 answered tasks (`experiments/images/REPORT.md`): 62% of the unclear images had
  a clear view of the same piece 0.4 s away; three quarters or more of the unclear images
  drew no flag (the reviewer tired of flagging part-way through, so single-image flags of
  this batch are undercounts; exhaustive from the next answers on); of 31 "someone else"
  flags, 20
  were on images the rule leaves alone and 10 of the other 11 could be checked: the exchanged
  frame leaves the track with the flagged one every time;
  the single-detection flags that would be lost reach under 0.1% of such detections anyway.
  Adopted in the narrowest form (same camera, same piece) for re-identification, cross-view,
  triage and swap questions, with 20% of the questions as a control. The broader choice
  made for re-identification candidates the day before (any piece of the camera, cameras
  with clear views first) is withdrawn: it would have hidden at least 2 of the 31.
* **2026-11-20 - neighbours of a flagged image; human cuts land on the right piece.** A
  "someone else" flag takes the image's piece out of its track; the rest of the
  single-camera tracklet around it could stay. It is now asked about (continuity question,
  above). The answer "breaks here" records, besides the cut, that each side is one person
  (`same` per side): the screen shows the answer that way, and without it the pieces between
  the flagged image and the cut were bound to nothing. Found on the way and fixed
  (`pipeline.unit_cuts`): a human cut names the tracker's tracklet, but the solver's pieces
  are renumbered when Autofit evidence is used, so the cut was applied to an unrelated
  piece and the intended one stayed whole. No cut had been applied in a real round yet
  (felix: 2 waiting from swap checks).
* **2026-11-19 - the comparison images of a swap check can be flagged (answer model 4).**
  The "other person before" images were display only; a mixed pose or another person among
  them could not be reported, and the "both" follow-up would have joined or cut on them.
* **2026-11-19 - image flag "unsure"; a swap check can be answered "both".** A sliver of a
  person can be impossible to assign; "unsure" on the image leaves it out of the answer and
  records nothing (unknown is not negative, and not positive either). In a swap check both
  candidates can show the reference's person (two tracks on one person); the single-choice
  rule forced a false "different". "Both" asks one follow-up about the other track before the
  moment, because joining it without that answer would assume its earlier part is the same
  person. Also fixed: a triage answered "unsure" was asked again with the same images.
* **2026-11-19 - identity after the recovery (identity lap); links through recovered
  positions are proposals.** The recovery moves track ends (felix: 78 ends and 115 starts,
  median 0.6 s, at most 67 s), which made the linking rule accept 8 more links on the felix
  hour and none on golden. The lap that applies them costs about 2 minutes on felix because
  the solve keeps its association (`solve_state/`, 384 MB for the hour) and the recovery
  continues from its previous result. Viewed (`experiments/coverage/link_sheets.py`), two
  of the eight join two different people, so `relink.apply` is `False`: the links are the
  first candidate of the re-identification question. Using the recovered positions in the
  swap check was evaluated and not adopted (`experiments/switches/REPORT.md` 3c).
* **2026-11-19 - automatic identity links capped at 1.5 m (`temporal.auto_max_distance_m`).**
  Found while viewing the links above: of the links the solve had made on measured
  positions on the felix hour (115; 18 beyond 2 m), six of the seven viewed beyond 3.5 m
  join two different people; the three viewed up to 0.41 m are right. The rule ("the only
  track that begins within reach") assumes the earlier person left; in a busy room they
  are often still there, untracked. Golden's ten links are right and span at most 1.39 m,
  so the bench never exercised the long range. With the cap felix has 95 links (farthest
  1.26 m), 299 identities instead of 279 and 203 open re-identification questions instead
  of 197; the six links still proposed through recovered positions are the six that looked
  right. Golden is unchanged (29 identities, 10 links). Felix links between 0.41 and
  1.26 m were not viewed, and no independent cue checks a link.
* **2026-11-19 - hand-offs singled out by the 2D trackers.** 38 track endings on the felix
  hour had another track born at the same spot within a few frames - one person tracked
  twice - that neither the hand-off merge nor identity linking joined: in a crowd the
  pairing is not unique by position, and the rule compared 1 s endpoint medians, which are
  about a metre apart for someone walking. The rule now also merges when the positions at
  the hand-off (last / first three frames) are within 20 in and single-camera tracklets in
  at least two cameras continue from the ending component into the starting one and into no
  other candidate (`refine.handoff_tracklets`). Golden bench: identities 32 -> 31, questions
  78 -> 77, wrongly assigned observations unchanged (84); the second golden case is blocked
  by a same-camera clash and stays a question. (The earlier, rejected `external_rebirth`
  option accepted one camera and no 3D proximity.)
* **2026-11-18 - rejected on the bench:** limb-length veto (no gain), single-camera identity
  bridge (no effect), duplicate recognition (no fewer questions), keeping tracklets across
  Autofit track re-creation (one more mixed identity), "only candidate" identity
  elimination (unsafe: about 400 wrongly assigned observations), ReID decisions (0.5-1%
  wrong), Autofit global ids for identity (2-4 wrong links). All remain as off-by-default
  options with their measurements in `experiments/RESULTS.md`.
* **2026-11-19 - calibration is stored with the dataset, in data_management.** A recording
  can have its own calibration (`data_management/data/calibrations/<dataset>/`, per-camera
  overrides over the default, derived-from sha256 recorded; see data_management's README).
  The calibration stage now takes its base from data_management's resolver (worker
  `calibration_store_dm.py --mode resolve`, since the engine core does not import
  data_management): a dataset with a stored calibration starts from it and the mount check
  measures against it; every other dataset starts from the site default as before. If the
  default changed after the stored calibration was derived the resolver refuses and the
  stage fails rather than mixing geometries. A decided workspace calibration becomes the
  dataset's only through the explicit `data-engine store-calibration DATASET [--replace]`
  (worker `--mode store`): it refuses while any `calibration` unknown is open, while the
  stage is stale, for excluded cameras (a per-run decision, not geometry), for pilot slices,
  and if the workspace calibration was derived from something other than what the dataset
  resolves to now; provenance (workspace, round, per-camera status/decision/constraint,
  mount-check numbers, base sha256) goes into the record. Nothing stores automatically, and
  the stage does not rerun by itself afterwards (`run --force calibration`). A `use_corrected`
  decision whose correction the stored calibration already carries is satisfied
  (`applied: stored ...`), since measured against it the camera is no longer shifted.
