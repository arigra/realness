# Realness score (pre-registered 2026-11-01, before relating any measure to AP)

## Question
Can a cheap measurement of how synthetic maps differ from real ones, with no
detector trained, predict the real-test AP a detector trained on them reaches?
If it can, it is the objective to tune a simulator with (Ari, 2026-11-01:
simulators may be tuned to real data in this repo, the aim is SOTA).
Plan proposed by ChatGPT, adopted by Ari, with two additions (the threshold-shaped
score, and tuning as the real test).

## 1. Smoothing, whole map
Study 1's smoothing left +/-3 x +/-2 cells around each label unsmoothed (a cue).
New suite `smoothall`: 25% and 50%, whole map, CARRADA and RADIal, synthetic
only, seeds 1-3 (12 runs); clean runs reused from the fair suite (same pool,
seeds, training). Results: results/smoothall_<dataset>_runs.json, kept apart
from the fair suite's smoothing.

## 2. Measurements (src/measures.py)
On the maps the detector sees (normalised per source). Real reference = the 50
real training sequences of each dataset.
- texture (per frame): spread at 1/2/4/8 cells (std of differences of box
  means), lag-1 correlation of the 3x3-mean residual in range and in Doppler;
- contrast (per labelled object): peak (max within +/-2) over the frame's
  background median (floor) and over its 99th percentile (strong clutter);
  background excludes +/-3 cells around every label;
- label agreement (per labelled object): distance label -> brightest cell
  within +/-3; prominence = max within +/-1 over the median of the +/-6 ring
  outside +/-3.
Distance per measure = Wasserstein-1(version, real) / std(real); group
distance = mean over its measures.
Already seen (no AP involved): clean CARRADA sim vs real differs most in
texture correlation (range 0.05 vs 0.68) and label displacement (0.5 vs 2.0
cells).

## 3. The contrast puzzle (RADIal)
contrast-12 AP 0.152 vs noise+12 AP 0.314 at equal target-over-ring. Expected
if the measures matter: the two differ in over_clutter (targets sink below the
clutter in contrast-12 only), not in over_floor/prominence. Reported as the
measure values of the two versions next to clean.

## 4. Scores (fitted on CARRADA only)
Points: the 21 fair versions + 2 smoothall versions, AP@2 synthetic-only mean
of 3 seeds. Three forms, in order:
a. sum of the three group distances (nothing fitted);
b. non-negative weighted sum of the groups (least squares to AP);
c. threshold-shaped: sum_g w_g max(0, d_g - tau_g), w >= 0, tau_g on a grid
   (a gap inside the tolerance costs nothing).
CARRADA: Spearman rho with AP, and leave-one-version-out for b and c.

## 5. Frozen test on RADIal
The CARRADA-fitted forms, unchanged, on RADIal's 23 versions: Spearman rho
with AP. Deliverable: score vs AP plots for both datasets and a list of the
versions where score and AP disagree most (rank residuals), which name the
factors for deeper experiments. No pass/fail threshold; rho with a bootstrap
interval over versions is reported.

## 6. Tuning (after 1-5)
Tune the RADDet simulator to lower the score against real RADDet maps, train
synthetic-only, test on RADDet's real test split. The score is useful if the
lower-score simulator gives the higher real-test AP.

## Results (2026-11-02)

### 1. Whole-map smoothing: no harm
AP@2 synthetic only, mean of 3 seeds (seed range):

| | clean | smoothall 25 | smoothall 50 | Study 1 smooth 25 / 50 |
|---|---|---|---|---|
| CARRADA | 0.295 (0.055) | 0.373 (0.069) | 0.339 (0.046) | 0.120 / 0.008 |
| RADIal | 0.312 (0.041) | 0.273 (0.123) | 0.273 (0.196) | 0.124 / 0.007 |

Study 1's smoothing damage came from the unsmoothed box around each label (a
cue the detector learned), not from lost texture. Smoothing is no longer a
finding; Study 1's smooth points are cue artifacts.

### 2. Measurements, clean simulator vs real (medians)
- CARRADA: real range cells strongly correlated (residual corr 0.68 vs sim
  0.05; Doppler -0.53 vs -0.11); real labels sit 2.0 cells from the peak (sim
  0.5), scattered, not shifted (signed mean -0.7 range, 40% >= 2 cells off).
- RADIal: real targets stand well above strong clutter (2.40 vs sim 0.54).

### 3. Puzzle: explained by target vs strong clutter
RADIal medians (normalised maps): over_clutter real 2.40, clean 0.54,
contrast-12 -1.44, noise+12 0.70; over_floor 4.86 / 4.46 / 2.41 / 3.57.
contrast-12 puts targets below the strong clutter; noise+12 buries the clutter
with them. Normalisation also separates their over_floor although their dB
target-over-ring is equal.

### 4-5. Score vs AP: weak
Spearman rho of -score with AP@2 (90% bootstrap interval over versions):

| score | CARRADA (fitted) | CARRADA leave-one-out | RADIal (frozen) |
|---|---|---|---|
| sum | -0.07 | - | 0.28 [-0.10, 0.60] |
| weighted (texture 0.39, contrast 0.07, label 0) | 0.26 | -0.23 | 0.33 [-0.02, 0.63] |
| threshold | 0.26 | 0.02 | 0.21 [-0.22, 0.63] |

Largest RADIal disagreements: noise+12 and clutter +/-6/12 (far from real, AP
fine), Study 1 smoothing (AP collapses, score barely moves: the cue is
invisible to map measures), extra25 (AP lower than the score says). Distance
to real, as measured, does not rank AP; label/map cues do the damage.
Results: results/score_frozen.json, results/distances_<dataset>.json.

### 6. RADDet tuned to the real maps (Ari: fix the specific gaps)
Gaps, clean RADDet sim vs real (50 train sequences): static ridge spread over
+/-1 Doppler bin (+12.7 dB) vs one bin; no receiver filter roll-off (real -8 dB
near, -9 dB dip at row 243); targets more prominent (over_floor 6.1 vs 4.8);
over_clutter equal (0.79). Fixes (src/raddet_tuned.py): processing = RADDet's
windows (Hann range, none Doppler); rangegain = the real per-row gain
(results/tuning/raddet_range_gain.json); contrast = target and static scene
weaker, fitted to over_floor and over_clutter (target strength alone cannot
match both; results/tuning/raddet_contrast_fit.json). The processing fix also
closes most of the texture gap (range corr 0.27 -> 0.54, real 0.70; Doppler
-0.07 -> -0.46, real -0.52). Suite `tune`: clean, each fix alone, all fixes;
synthetic only, 3 seeds, real RADDet test (1,032 frames).

Results, AP@2 / AP@4 / recall@2 on real RADDet test, mean of 3 seeds (seed
range of AP@2):

| version | AP@2 | AP@4 | recall@2 |
|---|---|---|---|
| clean | 0.126 (0.024) | 0.276 | 0.365 |
| processing | 0.142 (0.026) | 0.295 | 0.382 |
| rangegain | 0.125 (0.018) | 0.267 | 0.385 |
| contrast | 0.138 (0.023) | 0.283 | 0.397 |
| all fixes | 0.135 (0.024) | 0.274 | 0.464 (0.028) |

AP does not move (gap 0.018 < noise 0.024); recall@2 with all fixes rises by
0.10 (beyond seed range): the tuned simulator teaches the detector to find more
real targets but not to rank them above false alarms.

### Reference: the same detector trained on the 50 real sequences alone
`--settings real --versions clean` (results/tune_raddet_runs.json,
results/reference_real_<dataset>_runs.json):

| AP@2 | simulator only (clean) | real only (50 sequences) |
|---|---|---|
| CARRADA | 0.295 | 0.417 (0.021) |
| RADDet | 0.126 | 0.379 (0.031) |
| RADIal | 0.312 | 0.833 (0.005) |

The sim-to-real gap is large on RADDet and RADIal and the fixed gaps close
almost none of it (RADDet: 0.009 of 0.253). What the simulators lack is
elsewhere than texture, contrast, Doppler spread or range gain.
