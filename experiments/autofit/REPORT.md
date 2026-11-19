# Autofit as evidence for the Data Engine (golden 600 s, 2026-11-18)

Condensed from the research run that produced `run_autofit.py`, `evaluate.py`,
`unknowns.py` and `verify_geometry.py`. Outputs:
`C:\data_engine_data\_scratch\autofit\golden_20250505071549_20250505080000@first600s\{kept,score25}`.
Dev half = sync t < 3054; thresholds chosen there and frozen on the other half.

## Autofit alone vs. the first engine solver (same reference)

| Labelling | Coverage | Impure obs. | Mixed labels | Labels | Pair precision | Pair recall |
|---|---|---|---|---|---|---|
| Autofit raw ids, engine-filtered detections | 0.981 | 192 | 4 | 45 | 0.9954 | 0.9706 |
| Autofit global ids | 0.981 | 201 | 5 | 38 | 0.9954 | 0.9706 |
| Autofit raw ids, score > 0.25 only (old bridge input) | 0.975 | 522 | 14 | 106 | 0.9887 | 0.9299 |
| Engine components (geometry on 2D tracklets) | 0.986 | 2,581 | 36 | 219 | 0.9695 | 0.7415 |

Feeding Autofit the engine's post-filtered detections matters (53 vs 129 tracker ids).
Both systems confuse the same neighbours ({1,4,10}, {2,11}, {14,16}, {12,13}), so their
errors are correlated.

## As corroborating evidence for simultaneous cross-camera pieces

Rule: same Autofit raw id on >= n common frames, share >= x, conflicting share <= y.

| Rule | Dev correct / wrong | Held-out correct / wrong |
|---|---|---|
| x .9, n 3, y .1 | 196 / 0 | 188 / 0 |
| **x .95, n 5, y .1** | 141 / 0 | 167 / 0 |
| x .5, n 3, y .1 (loose) | 207 / 1 | 200 / 0 |

Near miss: two people (14, 16) shared one Autofit track for 333 of 375 common frames
(share 0.888). The engine uses the stricter form: pieces are first cut wherever the
Autofit id changes, so a link means "same single Autofit id on >= 5 common instants".

## Identity across gaps: not safe

Autofit's global ids (GlobalTrackMerger / ReIDMatcher at 30 fps defaults, no appearance)
linked 79-111 pairs correctly on the dev half but with 2-4 wrong links at every threshold
(the 14/16 track merge and a 12/13 link across a 39-step gap). Not used.

## Negative evidence

"Different Autofit tracks, never the same" was wrong for up to 2.8% of truly-same piece
pairs: useful for ordering candidates, not for discarding them.

## Runtime

CPU only, 5 threads: 0.068 s/frame on golden (0.71x real time); the full felix hour
(36,124 sync frames) took 18 minutes of tracking plus 29 s for the merger.

## Proposals for Autofit (not applied)

1. Export per-assignment ambiguity (margin to the second-best candidate) from
   `byte_tracker.update` so swap-prone frames can be discounted.
2. Make `GlobalTrackMerger` / `ReIDConfig` fps-aware and conservative: best-vs-second
   margin, refuse a match whose global id is observed elsewhere at the same time.
3. Must-link from single-camera tracklets as a cost bonus in `centroid_bot_sort.get_dists`;
   cannot-link as a hard gate (cost-matrix mask keyed by detection).
4. Single-camera births with a floor-plane depth prior; choose the ray net's 4 views by
   ray-angle diversity instead of score; a backward pass; an offline `track_offline` API to
   replace the duplicated driver.

## Caveats

The calibration was refit on these identities; Autofit shares the engine's detections,
filter and calibration (only the association logic is independent); one recording.
