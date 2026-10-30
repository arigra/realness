# Does detector AP respond to controlled defects in synthetic data? (pre-registered 2026-10-27)

Step 1 of the evaluation-metric proposal: before building any realism metric,
check that the thing it must predict -- real-test detector AP -- actually moves
when synthetic data gets worse. Defects, levels, settings, seeds and the rule
below were proposed by Claude and accepted by Ari ("go"), 2026-10-27. Written
before any result.

## Data (src/damage.py, scripts/damage_study.py)
The first 200 sequences of the CARRADA engineer simulator cache (the sim-only
pool of the CARRADA detector study), in 19 versions (clean + 18 defects):

| family | levels | how |
|---|---|---|
| clean | - | as cached |
| missing labels | 10, 25, 50, 75% | each (object, frame) label removed; map unchanged |
| range shift | 1, 2, 3 bins (0.2 m each) | every label moved away from the radar; map unchanged |
| Doppler shift | 1, 2, 3 bins (0.42 m/s each) | every label moved, circular; map unchanged |
| blinking | 10, 25, 50% | each (object, frame) removed from map and label (same scene re-rendered) |
| target contrast | -3, -6, -10 dB | moving objects' RCS lowered (same scene re-rendered) |
| clutter | +6, -6 dB | ground and pole RCS changed (same scene re-rendered) |

Re-rendering uses the cache's seeds; with no defect it reproduces the cache
exactly (tests/test_damage.py). Each version is normalised by its own
statistics, as every source in the detector studies.

Checks before training (hit = peak 3 dB over its ring within +/-2 cells):
labels 1381 clean, 1266/1059/719/352 missing, 1241/1032/667 blinking; hit rate
0.99 clean, contrast 0.98/0.97/0.95; range and Doppler shifts of 1-2 bins still
"hit" (inside the window), 3 bins 0.97/0.95. Clutter changes map std 2.80 ->
3.57 (+6) / 2.33 (-6).

## Training
The frame detector and training loop of scripts/radial_detector_study.py
(width 32, lr 1e-3, early stopping on 32 evenly spaced real val sequences,
patience 4 x 250 steps), seeds 1-3, two settings:
- **only**: the version alone (synthetic-only).
- **mix**: half of each batch from the 50 real sequences (data/carrada_seq_n50),
  half from the version (the user's scenario).
Test: CARRADA official test, 264 sequences, class-agnostic AP within 2 bins.

## Rule (applied to "only"; "mix" computed the same way and reported)
- noise = median over the 19 versions of the AP range across the 3 seeds.
- **Spread**: best minus worst version mean AP >= 3 x noise.
- **Ordering**: a family is ordered if its means from clean through each level
  never rise by more than noise, and clean minus worst level >= noise. Five
  ordered families are possible (clutter is two-sided, reported only).
- **Pass** = spread and >= 3 ordered families.

## What each outcome means
- Pass: AP responds to defects in a graded way, so there is something for a
  realism metric to predict; go to step 2 (compute candidate metrics on these
  19 versions).
- Spread but < 3 ordered families: AP responds only to some defect types; step 2
  is still possible but restricted to those, and the metric must separate
  harmful from harmless defects.
- No spread: at this test size AP cannot rank synthetic data; rethink the target
  (more test recordings, another detector) before building a metric.
- Caveat: AP matches detections within 2 bins, so a 3-bin label shift defeats
  scoring by construction; range/Doppler 1-2 are the informative levels.

## Results (2026-10-27 23:00; samples/damage_study.json, samples/damage_verdict.json)
**Both settings FAIL the rule.**

| | gap | noise | gap / noise | ordered families |
|---|---|---|---|---|
| only | 0.367 | 0.055 | 6.7 | 2/5 (range, Doppler) |
| mix | 0.059 | 0.052 | 1.1 | 2/5 (missing labels, Doppler) |

Mean test AP, 3 seeds:

| version | only | mix |
|---|---|---|
| clean | 0.295 | 0.441 |
| missing 10/25/50/75% | 0.330 / 0.315 / 0.322 / 0.260 | 0.433 / 0.415 / 0.400 / 0.383 |
| range 1/2/3 | 0.276 / 0.055 / 0.007 | 0.423 / 0.412 / 0.406 |
| Doppler 1/2/3 | 0.292 / 0.115 / 0.028 | 0.417 / 0.389 / 0.384 |
| blink 10/25/50% | 0.338 / 0.338 / 0.355 | 0.407 / 0.408 / 0.433 |
| contrast -3/-6/-10 dB | 0.314 / 0.301 / 0.270 | 0.427 / 0.401 / 0.399 |
| clutter +6 / -6 dB | 0.337 / 0.375 | 0.414 / 0.425 |

- **only**: the spread comes entirely from misplaced labels (range or Doppler
  >= 2 bins). Missing labels, blinking, weaker targets and clutter changes do
  not lower AP beyond seed noise (several sit above clean, also within noise).
  Outcome by the rule: "AP responds only to some defect types".
- **mix** (50 real + synthetic): no spread. Even labels 3 bins off everywhere
  (range3), which make the synthetic-only detector useless, cost 0.035 -- less
  than seed noise. Outcome by the rule: "no spread" -- with 50 real sequences the
  synthetic half's quality is not measurable at this test size.
