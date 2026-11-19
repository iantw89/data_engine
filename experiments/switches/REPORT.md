# Detecting identity swaps at close encounters (2026-11-18)

Question: the 3D tracker (Autofit) can, rarely, exchange two people's tracks when they come
very close. How do we find the moments where that may have happened and put them in front of
a person? Engine code: `src/data_engine/solve/swap_check.py`; this folder holds the research
(`gt.py`, `encounters.py`, `cues.py`, `eval_cues.py`, `rule.py`, ...) and the tools used to
check the engine rule (`run_swap_check.py`, `sheets.py`). Outputs:
`C:\data_engine_data\_scratch\switches\`.

## 1. What errors exist on golden (600 s, 17 labelled people)

| Solution | Error episodes (wrong obs.) | swap / takeover / contamination | Cameras wrong at once |
|---|---|---|---|
| Engine (bench `adopted`) | 26 (84) | 0 / 1 / 25 | 1 camera in 25; 3 in the takeover |
| Autofit raw tracks alone | 43 (192) | 4 / 3 / 36 | always 1 camera |
| Geometry only (`baseline`) | 60 (1,417) | 18 / 11 / 31 | one 6-camera swap of 247 obs. |

The engine solution contains no track-level swap on golden: its errors are single rows of
one camera attached to a neighbour (median 0.2 s) and one identity-link takeover. Most
happen in one standing trio 0.65-1.0 m apart. So swap recall cannot be measured on real
engine errors there; it is measured by exchanging labels after real encounters (physical
data unchanged) and on `baseline`'s real swaps.

## 2. First rule (research), and why it was replaced

Cues compared 10 s before and after a 10 s chunk of every encounter under 1 m: position
continuity, body proportions, appearance (fast-reid, golden only), BoT-SORT tracklets in
views where the two stay apart, Autofit id churn, pose ownership. Under injection the
"two of n cues" rule flagged 97-100% of swaps and 95-98% of takeovers at 2-3 questions per
10 minutes on golden. Autofit id churn and pose ownership carried no signal.

On the felix hour the same rule asked 210 questions plus 199 unverified chunks. Rendered
examples (`sheets.py`) showed why: the position cue fires for two people walking in file
(each ends up where the other was), and many "encounters" are one person tracked twice.
Thresholds fitted on a standing trio do not transfer.

## 3. Engine rule: where is a swap physically possible?

While two people are measured continuously and stay apart, exchanging labels makes both
tracks jump by their separation. For every step between consecutive common instants of a
pair, with 5-frame medians either side:

    stay = |A2 - A1| + |B2 - B1|      swap = |A2 - B1| + |B2 - A1|

* `swap - stay >= 0.3 m` (plus 0.5 m per second without a common measurement): unambiguous;
* `swap - stay < -0.3 m`: the tracks exchanged places - flagged;
* in between: a crossing, tested with heading (velocities through the crossing), body
  proportions, 2D tracklets in separated views and, when available, appearance. Two "stay"
  cues and no "swap" cue verify it; anything else goes to a person.

Skipped: pairs whose common median distance is under 0.3 m (one person tracked twice) and
endings whose survivor was born just before at the same spot (hand-off with overlap). A
track ending within 0.5 m of an established one is checked one-sided (takeover).

| | golden 600 s | felix 60 min |
|---|---|---|
| Pairs ever within 1 m | 20 | 408 |
| Duplicate pairs / hand-off overlaps skipped | 4 / 1 | 43 / 38 |
| Moments (crossing / exchange / ending) | 6 (5 / 0 / 1) | 107 (86 / 10 / 11) |
| Verified by two independent cues | 4 | 26 |
| Review questions | 2 | 81 (23 flagged, 58 unverified) |
| Injected swap while apart: flagged | 100 / 100 | 100 / 100 |
| Injected takeover while apart: flagged or reviewed | 100 / 100 | 99 / 100 |
| Injected swap at a crossing: verified (missed) | 0 / 5 | 0 / 86 |
| Injected takeover at a crossing: verified (missed) | 0 / 5 | 1 / 86 |

Of the first felix exchanges viewed, one is a real swap (two people at adjacent rowing
machines: teal shirt and red shirt change tracks at 37.6-38.7 s). Flagged endings before the
hand-off exclusion were the same person tracked twice.

## 3b. Appearance cue (2026-11-19)

`src/data_engine/appearance.py` embeds up to six crops per camera of each of the two identities
before and after the moment (fast-reid SBS R50-NL, MSMT17; boxes at least 48 px tall) and
compares "labels kept" with "labels exchanged". The solve asks for embeddings only at
moments the other cues did not verify. Felix hour (solution r006), `--appearance`:

| | without | with appearance |
|---|---|---|
| Verified by two independent cues | 26 | 56 |
| Review questions | 81 | 51 (29 flagged, 22 unverified) |
| Injected swap while apart: flagged | 100 / 100 | 100 / 100 |
| Injected takeover while apart: flagged or reviewed | 99 / 100 | 99 / 100 |
| Injected swap at a crossing: verified (missed) | 0 / 86 | 0 / 86 |
| Injected takeover at a crossing: verified (missed) | 1 / 86 | 3 / 86 |

The three missed takeovers were injected at moments the check flags on the real data: the
track is already inconsistent there, so exchanging the labels makes it look consistent.
Golden: all 6 moments verified, 0 questions; every injected swap still flagged.

## 3c. Recovered positions in the swap check (2026-11-19): not adopted

The coverage recovery measures positions where the tracker had none, often inside the very
crossings the check examines. `run_swap_check.py --recovered` adds them to the measured
positions (felix hour, solution r012 after the Autofit laps, appearance on):

| | measured positions | + 31,831 recovered |
|---|---|---|
| Pairs ever within 1 m | 341 | 407 |
| Duplicate pairs / hand-off overlaps skipped | 20 / 13 | 37 / 29 |
| Moments (crossing / exchange / ending) | 89 (77 / 6 / 6) | 70 (58 / 4 / 8) |
| Verified by two independent cues | 51 | 28 |
| Review questions | 38 (15 flagged, 23 unverified) | 42 (20 flagged, 22 unverified) |
| Injected swap while apart: flagged | 100 / 100 | 100 / 100 |
| Injected takeover while apart: flagged or reviewed | 100 / 100 | 99 / 100 |
| Injected swap at a crossing: verified (missed) | 0 / 77 | 0 / 58 |
| Injected takeover at a crossing: verified (missed) | 2 / 77 | 3 / 58 |

With positions through the gap, 19 crossings stop being ambiguous by continuity alone. Those
were crossings the cues already verified, so no question goes away: the open questions go
from 38 to 42. And a crossing that continuity declares unambiguous is no longer tested by
the independent cues, while the recovered positions inside it were themselves placed by
continuity (each is read where the track was expected): that is the weaker test of the two.
The recovered positions also bring more pairs close together, half of them one person
tracked twice (duplicates 20 -> 37). `RECIPE["swap_check"]["use_recovered"]` exists and is
off. The five additional flags were not viewed.

## 4. Limits

* Injection exchanges labels, so it validates the continuity part and the before/after
  cues; it cannot reproduce what a real swap inside Autofit's filter looks like.
* For two people of similar build who pass within about 0.3 m, heading and body carry
  little; the decision rests on tracklets in a separated view, appearance (two people in
  similar clothes defeat it) or the reviewer.
* Body and appearance thresholds come from 16 golden encounters; heading thresholds are
  round numbers. All are in `RECIPE["swap_check"]` and provisional.
* Single-row contamination (a neighbour's detection attached for a frame or two) is not a
  swap and is not caught by this check; per-row pose ownership had row-level precision 0.92
  at recall 0.23 and could become its own lever.

## Reproduce

```
python experiments/switches/run_swap_check.py "golden_20250505071549_20250505080000@first600s" --inject 100
python experiments/switches/run_swap_check.py felix_20251223173052_20251223183103 --inject 100 --out felix.json
python experiments/switches/run_swap_check.py felix_20251223173052_20251223183103 --inject 100 --appearance
python experiments/switches/sheets.py felix_20251223173052_20251223183103 felix.json OUT_DIR --kind exchange
```
The research scripts are listed in their docstrings (`build_solutions.py` first).
