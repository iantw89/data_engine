# Ranking the candidates of a re-identification question (2026-11-19)

Scripts: `ranking.py` (`pairs golden|felix`, `synthetic`, `report`, `felix-shown`). Code under
test: `src/data_engine/solve/link_ranking.py`, `stages/solve.py: rank_link_candidates`.
Truth: the golden reference (`golden ...@first600s`), and on felix the reviewer's answers to
40 re-identification tasks (148 judged pairs: 41 "same", 107 "different").

**Before.** Candidates were the earlier tracks that had ended and could have reached the spot
(12 m/s, within 120 s), ordered by distance alone, four shown per round. Appearance was not
used. On the felix answers distance orders well only for people who barely moved: 11 of 23
confirmed matches within 0.25 m, 8 between 1.9 and 8.8 m among rejected candidates (median
2.6 m, closest 0.11 m); 9 of 23 questions had no match among the four shown.

**Now.** Score = motion + appearance, both as log-likelihood ratios; appearance only on clear
views, unknown (0) otherwise. Nothing is joined, excluded or closed by the score.

## 1. Clear views (golden reference, 350,109 pairs of 10 s windows, 17 people)

Each window of a labelled person against every window at least 10 s earlier; up to 6 crops per
camera; same-camera comparison. "Gallery": the window against all people at one earlier time.

| Crops used | AUC | Right person first in the gallery | Pairs comparable |
|---|---|---|---|
| any box at least 48 px tall | 0.956 | 82% | 100% |
| clear views only (13+ joints, not cut, overlap <= 0.2, 96+ px) | 0.955 | 80% | 99% |
| only crops that are NOT clear (partial, cut or overlapped) | 0.916 | 66% | 84% |

Partial or overlapped crops are clearly worse (66% against 80%), which supports leaving them
out. Where a track has clear views, mixing unclear ones in does not hurt on this recording
(82% against 80%, labelled detections only); the gate matters when a track has nothing else.
Cross-camera comparison of clear views is weak (AUC 0.69 on 137 same pairs) and gets a flat
curve in the recipe.

Fitted curves (log-odds at prior odds 1): same camera `16.6 * (s - 0.46)`, cross camera
`4.4 * (s - 0.34)`; limited to +-6.

## 2. Felix: the judged pairs

Appearance separates the reviewer's "same" from "different" with every choice of crops
(same-camera AUC 0.92-0.94; clear views: 36 same / 85 different comparable, 12 of 148 pairs
not comparable). The judged pairs cannot show the effect of the gate: pairs a person could not
judge were answered "unsure" and are not in this sample.

Ordering inside a question (21 questions with both a "same" and a "different"):

| Order by | First candidate is a "same" | A "same" above a "different" |
|---|---|---|
| distance (before) | 15 / 21 | 74% of 65 pairs |
| motion model | 15 / 21 | 74% |
| appearance, clear views | 19 / 21 | 89% |
| motion + appearance, clear views (adopted) | 18 / 21 | 88% |

Motion adds nothing over distance on these pairs - expected, because only the four nearest
candidates were ever shown, so the sample is selected by distance.

## 3. Felix: the engine's ranking over every feasible candidate (`felix-shown`)

40 questions, 503 feasible candidates, appearance comparable for 450 (49 across cameras).

| | by distance (before) | adopted ranking |
|---|---|---|
| confirmed "same" tracks (41) placed first | 17 | 20 |
| confirmed "same" tracks in the first four | 41 (by construction) | 37 |
| rejected tracks (107) in the first four | 107 (by construction) | 69 |
| candidates in the first four never shown before | 0 | 47 |

Four confirmed matches leave the first four (they come in the next batch): three have an
appearance comparison built on 3, 9 and 3 clear views that came out low (0.32, 0.16, 0.43);
in the fourth question the new track has no clear view at all, so motion alone orders.
Requiring a minimum number of clear views (5 to 30) or limiting negative appearance evidence
more tightly did not improve the ordering of the judged pairs (88% -> 83-88%), so neither was
added. Of the 47 newly shown candidates the truth is unknown; 4 have a similarity of 0.6 or
more.

## 4. What this does not show

* Whether the 47 newly shown candidates contain matches: that needs the next round's answers.
* Motion parameters are round numbers, not fitted; the appearance curve comes from one
  recording of another day.
* The 120 s candidate window was not changed. Confirmed matches reach 118 s, so it cuts
  matches off; widening it multiplies the candidates of every question.

## 5. Flags on images, dry run (felix, 87 unapplied answers, scratch copy of the database)

The reviewer flagged the same image of a candidate track in several questions, because an
answer takes effect at the next round. Of 58 image flags in re-identification answers, 9 were
on an image flagged before; carrying the flag (`tasks/carry.py`) pre-sets those.

Solving in memory with the answers applied: all 30 "someone else" flags (23 distinct images)
take the image out of the track it was shown in; in 19 of 23 nothing of that camera's tracklet
within 15 s stays in the track, in 4 between 9 and 78 detections of neighbouring pieces stay
(not known whether rightly). Open questions after the dry run: 178 identity links (202
before), 90 associations (108) - without the coverage recovery, so only roughly comparable.
