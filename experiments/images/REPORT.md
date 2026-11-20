# Which frames should a question show? (2026-11-20)

Script: `selection.py` (`table --sharpness <dry-run dir>`, `report`, `engine`). Code:
`src/data_engine/solve/view_choice.py`, `RECIPE["task_images"]`.

**Question.** Many images in re-identification, cross-view and swap questions are slivers
of a person or hard to read. They let the reviewer catch problems (a pose built from two
people, a wrong grouping), but they make the question hard. Should the engine prefer images
that are easy to judge, or would that hide what the reviewer is catching?

**Data.** The felix hour: every image shown in the 84 answered group tasks with flags on
single images (40 re-identification, 40 cross-view, 4 swap check): 720 images, 293 groups,
one reviewer. For each image: joints found, overlap with other boxes, cut by the image
border, height, sharpness; the reviewer's flag; the verdict of its group; which other
frames of the same piece of track were available; and, from a solve with the answers applied
to a scratch copy, where the image and its alternatives end up. "Clear" is the rule already
used for appearance (13+ joints, overlap <= 0.2, not cut, 96+ px).

**Decision.** Adopted, in the narrowest form: the frame is exchanged for the nearest clear
view of the *same piece of the same camera's tracklet*; cameras and pieces stay as chosen.
One question in five keeps the nearest frames as a control.

## 1. The images shown were worse than the tracks they come from

| | clear | few joints | overlapped | cut by the border |
|---|---|---|---|---|
| all tracked detections (sample of 28,000) | 68% | 7% | 20% | 7% |
| images shown in the answered tasks | 41% | 19% | 40% | 19% |

Questions are asked where a track starts or ends, or at a crossing - where the person is
being hidden - and the camera with the largest box is preferred, which favours people close
to the lens and cut by the frame. Re-identification candidates were clear in 32% of cases.
35% of the groups had no clear image at all.

## 2. A clear frame was usually a fraction of a second away

Of the 428 unclear images, 265 (62%) had a clear view in the same piece within the time the
question's rule may choose in; the median shift is 4 frames (0.4 s), 90% within 1.8 s.

| | before | same-piece choice |
|---|---|---|
| images that are clear views | 41% | 77% |
| groups without any clear image | 35% | 8% |
| groups in which every image is clear | 22% | 59% |

## 3. What the reviewer's flags say

| Flag | on clear images (292) | on unclear images (428) | p |
|---|---|---|---|
| someone else | 9 (3.1%) | 22 (5.1%) | 0.20 |
| two people in one pose | 2 (0.7%) | 5 (1.2%) | 0.71 |
| not a person | 0 | 5 (1.2%) | 0.08 |
| unsure (added late) | 2 | 2 | 1.0 |
| bad pose | 11 (3.8%) | 29 (6.8%) | 0.10 |

* **85% of the unclear images were not flagged.** They cost attention and reported nothing.
  The same holds on each image's first showing alone (614 distinct images: 15.0% of the
  unclear and 7.5% of the clear ones flagged), so it is not an effect of repetition.
* **Flags changed when an image came back - because of fatigue, not because the image was
  hard.** Of the images shown in two or more tasks and flagged at least once, the flag
  differed between showings for 8 of 12 unclear images and 1 of 4 clear ones. The reviewer's
  explanation: flagging the same image over and over is tiring, and at some point the task
  just gets finished. This is therefore no evidence about how reliable a judgement on an
  unclear image is, and it is not used as an argument here. It is an argument for carrying
  flags between tasks (`tasks/carry.py`): 106 of the 720 showings were repeats.
* **The four "unsure" verdicts on a candidate were all on groups without a clear image**
  (4 of 79 such groups, 0 of 126 groups with one; p = 0.02). Two of the four would have had
  clear images under the rule.
* **Time per task does not show the difficulty.** Median 41 s; each flag set adds about 49%;
  an unclear image by itself adds nothing measurable (-3%, 95% interval -11% to +6%, 81
  tasks). Unclear images cost time through the flags they draw.

## 4. What the exchange would have hidden

**Wrong groupings ("someone else", 31 flags).** A wrong grouping is a property of the piece,
not of the frame, and the exchange never leaves the piece. 9 flags were on clear images and
11 on unclear images with no clear frame in their piece: unchanged. For the other 11 the
exchanged frame was checked in the solve with the answers applied: in 10 it leaves the
track together with the flagged image (it shows the same wrong person, more clearly); the
eleventh flag was given after that solve was run and could not be checked. So no flagged
grouping is known to be hidden, and 30 of 31 are known to stay in front of the reviewer. A
camera-level choice (any piece of the camera, cameras with clear views first - what commit
`8e3356e` did for re-identification candidates) would have hidden at least 2, where the
clear frame came from another piece; it is replaced by the same-piece rule.

**Problems of one detection (56 flags: mixed pose, not a person, unsure, bad pose).** 22
would not have been shown. Each such flag changes one detection. Weighted to all tracked
detections the reviewer's own rates give about 12,000 mixed poses, 65,000 bad poses and
5,000 non-persons among 1.38 million; the flags found 7, 40 and 5. Questions about tracks
find these by chance, at well under 0.1%; they need their own pass once tracks are settled.

## 5. On the engine's own questions (`engine`)

The solve stage's question builder on the felix state, nearest frames against the adopted
rule (about one question in five is a control and keeps the nearest frames):

| Questions | images clear | groups without a clear image |
|---|---|---|
| re-identification (202) | 38% -> 66% (74% without the control) | 38% -> 16% (10%) |
| cross-view match (108) | 45% -> 65% (73%) | 32% -> 20% (14%) |
| single-camera triage (180) | 17% -> 23% | 65% -> 55% |

The questions keep their subjects (they are anchored on the nearest frames, as before), so
closed questions stay closed and nothing is asked twice because of the change. Triage barely
moves: pieces nobody could associate rarely have a clear view.

## 6. What this does not show

* One reviewer, one recording, 84 tasks; 31 "someone else" flags, 10 exchanged frames
  checked. "0 of 10 hidden" still allows up to about 26% at 95%.
* Whether wrong groupings are caught *more* often on clear frames: plausible, not measured.
* "Fuzzy" images: sharpness (Laplacian variance of the crop) did not separate flagged from
  unflagged images and is not part of the rule; small boxes are (96 px).
* The control questions are what will answer the first two points: compare flag rates and
  "unsure" verdicts between `image_choice` "clear" and "nearest" in the task payloads.
