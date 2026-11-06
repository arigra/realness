# Few-label utility metric for synthetic radar data: preregistration (2026-11-06)

Status: **skeleton**, committed before any new GPU run. Sections marked *TBD (before ...)* are filled
and committed before the runs they govern; nothing in them may be chosen after seeing those runs.

## Question
Rank candidate synthetic radar datasets by their usefulness for real-data **detection** when only a
few labelled real sequences are available, and say *why* a candidate is risky (object coverage,
labels, target response, hard negatives, radar processing). Exact AP prediction is not the goal.
Classification is a contrast result only (docs/notes/2026-11-05-classification.md: erased-target
crops still score 0.45-0.50, so no signature claim until Phase 8's controls pass).

## Audit (Phase 0.1)
- Code: e67c42c plus one fix: `DamagedSceneSimulator._vehicles` (spec RADIal simulator) read
  `self.scenes`, which only the tuned simulators have, so `tests/test_damage.py::test_radial_undamaged_and_damaged_scenes`
  failed. No cached version was affected: the fair-suite RADIal caches predate that code. Guarded with getattr.
- Environment: hpc-node3, one RTX 3090; Python 3.10.12, torch 2.12.1+cu130, numpy 2.2.6, scipy 1.15.3,
  matplotlib 3.10.9.
- `PYTHONPATH=. python3 -m pytest tests -q` -> 60 passed (59 + 1 failed before the fix).

## Data roles (unchanged; not to change silently)
- 50 real training sequences: simulator fitting, and metric reference/calibration.
  **Caveat:** the tuned simulators drew their scenes, range gain, contrast offsets and label threshold
  from these same 50 sequences, so for tuned candidates the coverage and target-strength components are
  in-sample. Metric references are drawn from these 50 by sequence; this overlap is reported, not hidden.
- Real val split: detector early stopping (32 sequences) and predefined model selection only.
- Real test split: final reporting only. Never used to choose features, thresholds, weights,
  candidates or checkpoints in a prospective experiment.
- RADDet: scene-disjoint split only (`data/raddet_scenesplit`). `tune_raddet` (official interleaved
  split) is in the registry for the record and flagged unusable.

## Candidates (Phase 0.3)
`results/metric/candidates.json` (scripts/build_candidate_registry.py): one row per distinct synthetic
pool, every existing result entry under exactly one row. 288 candidates, 381 result entries;
synthetic-only test AP usable for CARRADA 111, RADDet 76, RADIal 97. Bases: spec simulator (v1, fair,
smoothall, tune act on it), best tuned simulator (study2, study2b, budget), other tuned variants (sim2).
Families are mechanisms (contrast, rangeshift, ...); severity levels of one family are not
independent samples.

## Label budget (Phase 0.2)
`results/metric/figures/label_budget_ap2.{pdf,png}`, `results/metric/label_budget_summary.json`
(scripts/plot_label_budget.py), from the pushed budget and study2 JSON. Test AP@2, mean of 3 seeds:

| budget (sequences) | 0 | 1 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|---|
| CARRADA real only | - | 0.00 | 0.23 | 0.26 | 0.32 | 0.42 |
| CARRADA tuned sim + real | 0.34 | 0.22 | 0.35 | 0.34 | 0.40 | 0.44 |
| RADDet real only | - | 0.01 | 0.10 | 0.21 | 0.34 | 0.41 |
| RADDet tuned sim + real | 0.29 | 0.26 | 0.31 | 0.40 | 0.42 | 0.45 |

Note for Phase 5: at 1 real sequence the mixed detector is *below* the simulator alone (half of each
batch is one sequence's 8 frames), so Delta_AP at b = 1 depends on the mixing rule as much as on the
candidate. The rule stays fixed (4 real + 4 synthetic frames) as preregistered; this is read out, not tuned.

## Metric: the five-part detection adequacy vector (Phase 1)
Per (dataset, candidate, real reference subset); raw per-object / per-frame / per-peak values are saved
so every summary can be recomputed. All on the maps the detector sees (each source normalised by its
own statistics); Doppler wraps everywhere; the same peak extraction for real and synthetic.
1. **Object coverage**: objects per frame; range (m) and radial speed (m/s), marginal and joint; share
   within one Doppler bin of zero; class when the cache stores it (real caches do; tuned caches do not).
2. **Label agreement**: per label, offset (range, Doppler, total) to the strongest cell and to the
   power centroid nearby; target over local floor and over strong clutter; whether a target-like peak
   lies within 2, 4, 8 cells. Per frame: target-like peaks with no label nearby, labels with no peak.
3. **Target response**: label-centred patches; widths, tail, peak count, spectral and autocorrelation
   shape; conditioned on range and speed strata.
4. **Hard negatives**: real false positives of the fixed tuned-simulator detector on real *training*
   sequences vs the candidate's unlabelled strong peaks / false positives.
5. **Processing / background**: zero-Doppler ridge, per-range-row level profile, residual correlations,
   multi-scale spread. Diagnostic; not allowed to dominate.

Threshold rule for "target-like peak" (fixed here): a local maximum (3x3, Doppler wrapping) whose
prominence over its range row's median is at least the q-th percentile of the prominence of the real
reference's labelled objects; q = 25 primary, 10 and 50 reported for sensitivity.

Reference budgets: 5 and 10 sequences first (1, 25, 50 later). Subset 0 = evenly spaced (the budget
study's choice), subsets 1.. = sequence-level random draws, seeds fixed in the feature builder.

## Metric variants (Phase 2) — fixed 2026-11-06, before any Phase 3 evaluation
All computed by src/utility_score.py from the adequacy vector of a candidate against a real reference.

**Probes** (x_j, signed where a direction exists; "shift" = (candidate median - reference median) /
reference sd, from src/utility_metric.feature_distance; "diff" = candidate rate - reference rate):
- coverage: joint range-speed energy distance; shift of objects per frame; diff of near-zero-Doppler share;
- label: shifts of peak_dr, peak_dd, cen_dr, cen_dd, label_prom, over_clutter; diff of labels with a
  target-like peak within 2 cells; shifts of unlabelled target-like peaks per frame (all; off the ridge);
- target: shifts of width_r2, width_d2, area2, near_ext2, far_ext2, tail_level, n_peaks2, entropy_r,
  entropy_d;
- negatives: real-FP coverage ratio - 1 (nearest-neighbour, Phase 1.4); shift of the fixed detector's
  false positives per frame; shift of the prominence of unlabelled peaks;
- processing: shifts of spread1, spread4, corr_range, corr_doppler, frame_q99, ridge_width; mean
  W1 of the row-median and row-q99 profiles; mean shift of the ridge profile.
A probe the dataset cannot measure (no ridge on RADIal, too few reference objects) is NaN and gets no
penalty, and is reported as unmeasured.

**Tolerance score (M1).** Calibration data: development candidates with whole-val AP@2 (study2,
study2b, sim2; fair/v1/smoothall have no val AP and are only evaluated). A development candidate is
*undamaged* if its mean val AP@2 is within the dataset's seed noise (median val seed range) of its
base (tuned for study2/2b and sim2; spec for spec-based families). For each probe, the tolerance
interval [lower_j, upper_j] = min..max of x_j over undamaged development candidates; scale_j = MAD of x_j
over all development candidates (floor 1e-3). penalty_j = distance of x_j outside the interval / scale_j.
M1 = sum of penalties (equal weights). Component diagnosis = sum of its probes' penalties.
**M2** = non-negative least squares of development val AP on the penalties (AP ~ a - sum w_j p_j); risk =
sum w_j p_j. No other readout (no random forest, no neural regressor).

**Baselines:** random choice; the 2026-11-01 global score ("sum"); target-response energy distance
(all object target features, standardised by the reference); conditional target distance (mean of
range x near-zero strata scores); coverage only; label only; the five component scores unweighted
(mean); domain classifier (logistic regression real vs candidate on per-object label + target
features, 5-fold grouped by sequence, balanced accuracy; diagnostic).

## Validation (Phase 3) — fixed 2026-11-06
Outcome: synthetic-only real-test AP@2, mean over every usable "only" seed of the candidate (e.g. the
tuned simulator pools sim2, study2, study2b seeds). Seed noise per dataset = median test seed range.
Reference subsets: budget 5 (subsets 0, 1, 2), budget 10 (0, 1, 2), and all 50; budget 5 subset 0 primary.
- **Leave-one-family-out** (within a dataset): calibrate on every other family, rank the held-out family's
  levels together with its base; families with >= 2 levels.
- **Leave-one-radar-out**: calibrate on the two other radars, rank (a) every candidate of the held-out
  radar, (b) its natural candidates only (spec, tuned, sim2 variants): the simulator-choice problem.
- Primary: top-1 regret (oracle AP@2 - selected AP@2); P(selected within seed noise of the oracle);
  pairwise ordering accuracy over pairs whose AP differs by more than the noise. Secondary: Kendall
  tau, Spearman rho, top-3 recall, runtime; stability across reference subsets. Uncertainty: bootstrap
  over families (LOFO summaries, 2,000 draws).
- Decision gate: M1 must beat both random and the global score on LOFO mean regret (bootstrap
  interval of the difference excludes 0) before M2 or anything learned is reported as the metric.

## Proxy detector (Phase 4) — configuration fixed here
FrameDetector(width=8), the candidate's first 50 synthetic sequences, 1,000 steps, batch 8, no early
stopping, seed 1; scored (AP@2) on the budget-5 reference subsets 0, 1, 2 of the real training
sequences (their mean is the proxy score). The 5 best candidates per dataset by proxy score get seeds
2 and 3. Records wall-clock and GPU time. results/metric/proxy/<dataset>.json.

## Augmentation benefit (Phase 5) — candidates fixed: results/metric/augmentation_candidates.json
Stage A: CARRADA and RADDet, budget 5 (the evenly spaced 5 real training sequences = reference subset 0),
3 seeds, each batch 4 real + 4 synthetic frames, early stopping on the 32 real val sequences (the budget
study's training). Candidates: spec, tuned, study2:extra50, study2:unrender90, study2:contrast-12,
study2:rangeshift1, study2b:tail2. Delta_AP = AP(candidate + 5 real) - AP(5 real only); the real-only
and tuned + real runs are the budget study's (same code, seeds and sequences). Stage B (budgets 1 and
10, sim2 variants, noise+12, blur2) only if Stage A's Delta_AP spread across candidates exceeds 2x the
seed noise.

## Prospective selection (Phase 6) — *TBD (freeze point, before generating final candidates)*

## Stopping rules
- Phase 1 stops if a controlled intervention moves none of its intended probes: that probe is fixed
  or the factor is marked unmeasured; it is never declared irrelevant.
- Phase 3 decision gate above.
- No classification-metric work until Phase 8's gate passes.

## Phase 1 progress (2026-11-06): coverage, label agreement, target response on CARRADA and RADDet

Code: src/utility_metric.py (probes), scripts/build_metric_features.py (raw tables, vectors per
reference subset, `--check`), scripts/plot_metric_probes.py; tests/test_utility_metric.py (8 tests:
Doppler wrap, label shift -> offset, weaker target -> lower contrast, unlabelled object -> more
unlabelled peaks, wider target -> wider response, identical input -> zero distance, determinism,
the builder reads only training splits). Hard negatives (1.4) and processing (1.5) are not built
yet; RADIal has no radar entry yet (its zero-Doppler bin and resolutions are still to be added).

Outputs: results/metric/features/<dataset>/<candidate>/raw.npz (+ .json meta), reference_<b>_<s>.json
for b = 5, 10 (subsets 0-2) and 50; probe_check.json; figures/probes.{pdf,png}; probes_summary.json.

**Every controlled intervention moves its intended probe** (paired bootstrap over the 200 shared-seed
sequences vs the tuned simulator, 95% interval excludes 0), on both datasets:

| intervention | probe | CARRADA | RADDet |
|---|---|---|---|
| rangeshift1 | strongest cell - label, range (median) | -0.75 | -0.59 |
| dopplershift1 | same, Doppler | -0.99 | -0.98 |
| extra50 | unlabelled target-like peaks off the ridge / frame | +1.47 | +0.94 |
| unrender90 | labels with a target-like peak within 2 cells | -0.54 | -0.43 |
| contrast-12 | target prominence (z) | -2.6 | -2.3 |
| blur2 | range / Doppler width at 2 z (cells) | +3 / +2 | +5 / +2 |
| tail2 | extent behind the peak at 2 z (cells) | +2 | +3 |

Changes made while validating the probes (recorded, not hidden):
- Power centroid is now taken around the return's strongest cell, not the label (a window on the label
  pulled the centroid towards it; found by the label-shift unit test).
- The total count of unlabelled target-like peaks *fell* under extra50 on CARRADA (-4.8 / frame): extra
  objects raise the map's std, every other peak shrinks in z (each source is normalised by its own
  statistics), and most strong unlabelled peaks of a static radar sit on the zero-Doppler ridge.
  Off the ridge the count tracks the extra objects; it is added as its own feature (config version 2).
  The total is kept and reported.

Observations for Phase 2-3 (not acted on; the plan forbids choosing probes with AP):
1. **Distance to real is not monotone in harm.** Unweighted W1 distances say contrast-12, blur2 and
   extra50 are *closer* to real CARRADA than the tuned simulator (its targets are more prominent than
   real: median 8.0 vs 5.3 z), yet Study 2 found contrast-12 harmful. Signed, per-probe tolerance
   intervals are needed, not symmetric distances or component means.
2. **RADDet label offset:** the real label sits 1.5 cells behind the strongest cell (median), the tuned
   simulator's 0.5. rangeshift1 moves the simulator *towards* real on this probe (-1.1), and on the
   centroid and near-edge versions, yet Study 2's AP@2 falls 0.29 -> 0.20 (val 0.26 -> 0.19). The probe
   detects the shift; its distance to real does not explain the harm. Open question for Phase 3.
3. **Few-label resolution.** With 5 or 10 real sequences the reference bands are wide (CARRADA: ~27
   labels in 5 sequences; label-offset band -2.5 to +0.3 cells), so only large defects (unrender90,
   Doppler shift, RADDet width / extent / prominence) fall outside the 10-sequence band. RADDet's tuned
   targets are wider (6.7 vs 5.5 cells) and longer behind the peak (2.6 vs 1.9) than real, outside the
   10-sequence band.
4. Normalisation couples defects: anything that changes a map's std (extra objects, clutter) moves every
   z-unit probe. Phase 2 should read probes as the detector sees them (z), and report this coupling.

## Phase 1.4-1.5 definitions (fixed before the runs)
- **Fixed negative detector**: FrameDetector(width=32) trained on the dataset's best tuned simulator
  alone, seed 1, Study 2's training (early stopping on the 32 real val sequences); weights saved to
  results/metric/negative_detector_<dataset>.pt. It is run on the 50 real *training* sequences and on
  every candidate. A false positive = a detection (local maximum, threshold 0.05, top 10 per frame)
  with score >= 0.3 and no label within 4 cells.
- **Negatives** of a pool: its detector false positives, and, detector-free, the 3 most prominent
  unlabelled local maxima of each frame (no label within 4 cells). Each gets the target-response
  features at its strongest cell (within +/-1 of the detection), its prominence, range, ridge flag.
- **Hard-negative coverage**: for each real false positive, the distance (features standardised by the
  real negatives' spread) to its nearest negative of the candidate, vs to the nearest real negative of
  another sequence (reference). Reported overall and by stratum: on the ridge, beyond the real FPs'
  median range, above their median prominence. Also: the fixed detector's false positives per frame on
  the candidate vs on real.
- **Processing**: per frame, multi-scale spread and residual correlations at lags 1, 2, 4 (range,
  Doppler); range-row median and 99th percentile profiles in 32 bands; frame median and 99th
  percentile; zero-Doppler ridge excess over the row median per band and its width (static radars);
  share of cells at the frame's minimum / maximum (clipping).
- **Existing global score** (baseline): the 2026-11-01 realness score's "sum" (texture, contrast, label
  group distances of src/measures.py, nothing fitted), recomputed from these tables.
