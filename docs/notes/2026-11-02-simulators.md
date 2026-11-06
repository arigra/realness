# Three simulators as good as possible (2026-11-02)

Ari's order of work: (1) the best simulator per dataset, finding what is missing;
(2) damage experiments on them, across datasets (docs/notes/2026-11-02-study2b.md);
(3) then a conversation about the metric. Tuning to real data is allowed in this repo.

## Roles of the real data
- the 50 real training sequences: everything a simulator is tuned on (scripts/tune_simulators.py);
- 32 val sequences: early stopping; the whole val split: every choice (which fixes to keep);
- test: reported, never chosen on.
RADDet moved to a scene-disjoint split (scripts/split_raddet.py, results/raddet_split.json):
its official split interleaves train and test frames of the same scenes, and the tuned
simulator draws its scenes from real training labels. 26 scenes from cuts in the static
zero-Doppler ridge; train/val/test 5,356 / 1,292 / 2,510 frames, dealt in time order.

## Diagnosis: why simulator-trained detectors fail on real test
Simulator-only (spec simulators) vs real-only (50 sequences) detectors, real test, 3 seeds
(dets in results/fair_*_dets, results/reference_real_*_dets, results/tune_raddet_dets):

| AP at match radius 2 / 4 / 8 / 16 cells | simulator | real |
|---|---|---|
| CARRADA | 0.30 / 0.46 / 0.55 / 0.59 | 0.42 / 0.61 / 0.66 / 0.69 |
| RADDet (official split) | 0.13 / 0.28 / 0.59 / 0.71 | 0.38 / 0.60 / 0.71 / 0.75 |
| RADIal | 0.31 / 0.44 / 0.45 / 0.45 | 0.83 / 0.87 / 0.89 / 0.89 |

- **RADDet: mostly where the label sits.** The gap nearly closes at 8-16 cells. RADDet labels the
  centre of the object's box in the RAD cube; a car's RD energy sits on its near face, up to ~9
  cells nearer than the label (mean patch around labels). The simulator labelled the power centroid.
- **RADIal: what targets exist.** The gap stays at any radius. At score > 0.5 the simulator-trained
  detector makes 39 false alarms but misses 1,163 labelled vehicles: far (47-65 m) traffic moving
  with the ego car. Real labels sit a median 8 Doppler bins from zero relative velocity (sim 33);
  15% within 2 bins (sim 3%). Real targets also trail 2-10 cells behind the label in range (the
  body); simulated ones are +/-1 cell. Real strong clutter reaches ~60 m, simulated ~20 m.
- **CARRADA: detection, not localisation** (gap constant with radius). Real targets are fewer per
  frame and nearer zero Doppler than simulated.
- **Processing.** CARRADA and RADDet (same TI chip) both show a one-bin zero-Doppler ridge (no
  Doppler window; the simulators used Hamming, spreading it over +/-1 bin) and the same receiver
  range-gain curve (roll-off in the first ~10 rows, a dip near row 243). RADIal's far roll-off
  starts earlier (~row 430) than its simulator's.

## Fixes (src/tuned.py; each a switch, all measured on the 50 real training sequences)
processing (Hann range / no Doppler window), range_gain, scenes (each sequence starts from a real
training frame's labelled objects), label (centre of the box around the object's returns over
0 dB SNR, fitted on the mean range profile around labels; objects under an annotation SNR are
not labelled), contrast (target and static offsets matching the quartiles of target over floor
and over strong clutter), body (RADIal: scatterers over the whole vehicle footprint, level fitted
on the mean range cut), clutter_profile (static strength per range matching the real ridge /
99th-percentile profile).

### First round (medians only, no annotation rule): CARRADA got worse
CARRADA, whole val AP@2 not yet recorded then; 32-sequence val / test AP@2, 3 seeds
(results/sim2_carrada_runs.json, first round): clean 0.53 / 0.30, all fixes 0.39 / 0.22; leaving
out contrast gave 0.43 / 0.34, leaving out scenes 0.34 / 0.21. Matching the median target strength
pushed a tail of labelled objects into the noise (target over floor 10th percentile 1.6 vs real
3.0): every real labelled object is visible, because CARRADA and RADDet annotate what their
detector found. Hence the annotation rule and the quartile fit (at most 10% of scene objects may
lose their label: the scenes are real labelled objects, all detected).

## Second round and the choice (rule fixed before the runs)
Suite `sim2`: clean (the spec simulator), all fixes, and all fixes minus one (CARRADA and RADDet
without body, RADIal without processing and label), 3 seeds, synthetic only. **The best simulator of
a dataset is the version with the highest mean whole-val AP@2** (scripts/choose_best.py,
results/tuning/best.json); Study 2 and 2b run on it.

## Results (results/sim2_<dataset>_runs.json; 3 seeds, synthetic only)
Whole-val AP@2 (decides) | test AP@2, AP@4, recall@2 (reported). "all-X" = every fix but X.

| CARRADA | val | test AP@2 | AP@4 | recall@2 |
|---|---|---|---|---|
| **all-contrast (chosen)** | 0.375 | 0.340 | 0.588 | 0.595 |
| all-scenes | 0.366 | 0.379 | 0.615 | 0.661 |
| all-label | 0.354 | 0.316 | 0.558 | 0.608 |
| all | 0.349 | 0.287 | 0.460 | 0.587 |
| spec simulator | 0.327 | 0.295 | 0.463 | 0.677 |
| all-clutter_profile | 0.276 | 0.252 | 0.415 | 0.573 |
| all-range_gain | 0.276 | 0.224 | 0.397 | 0.439 |
| all-processing | 0.263 | 0.232 | 0.405 | 0.546 |

| RADDet (scene split) | val | test AP@2 | AP@4 | recall@2 |
|---|---|---|---|---|
| **all-clutter_profile (chosen)** | 0.264 | 0.292 | 0.544 | 0.580 |
| all | 0.256 | 0.305 | 0.559 | 0.593 |
| all-contrast | 0.225 | 0.283 | 0.520 | 0.561 |
| all-processing | 0.225 | 0.274 | 0.561 | 0.553 |
| all-scenes | 0.218 | 0.253 | 0.440 | 0.568 |
| all-range_gain | 0.213 | 0.265 | 0.523 | 0.546 |
| all-label | 0.193 | 0.204 | 0.439 | 0.502 |
| spec simulator | 0.140 | 0.153 | 0.316 | 0.384 |

| RADIal | val | test AP@2 | AP@4 | recall@2 |
|---|---|---|---|---|
| **all-scenes (chosen)** | 0.379 | 0.429 | 0.587 | 0.651 |
| all-contrast | 0.369 | 0.324 | 0.498 | 0.644 |
| spec simulator | 0.351 | 0.312 | 0.438 | 0.677 |
| all | 0.263 | 0.424 | 0.588 | 0.700 |
| all-body | 0.216 | 0.329 | 0.427 | 0.663 |
| all-clutter_profile | 0.205 | 0.362 | 0.494 | 0.624 |
| all-range_gain | 0.136 | 0.247 | 0.381 | 0.508 |

- **What matters everywhere: processing and range gain** (every dataset loses most when they go).
- **RADDet: the simulator doubles** (test 0.15 -> 0.29); every fix but the clutter profile helps, the
  label convention most (without it 0.19 val / 0.20 test).
- **CARRADA: matching target strength still hurts**, even as a per-object distribution; the best
  simulator leaves it out. Test 0.30 -> 0.34 (real-only reference 0.42).
- **RADIal: val and test disagree.** With every fix, val falls (0.35 -> 0.26) while test rises
  (0.31 -> 0.42); the rule picks all-scenes (val 0.38, test 0.43). Real scenes from the 50 training
  sequences may not represent the val recordings' traffic. Body, clutter profile and range gain help
  on val.
- The recorded best lists carry "body" for CARRADA/RADDet, where it is a no-op (it only acts on RADIal),
  and "processing"/"label" for RADIal, likewise.
