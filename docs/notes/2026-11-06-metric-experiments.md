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
- Ties (equal risk, e.g. several candidates with zero penalty): the selection is uniform over the tied
  best set, so regret and P(within noise) are their expectations over that set; a tied pair counts 0.5
  in pairwise accuracy. A NaN risk (baseline not computable) ranks last.
- Family groups for LOFO are mechanisms (all bases held out together); a held-out family is evaluated
  per base (its levels on the tuned simulator with the tuned simulator; on the spec simulator with
  the spec simulator) and summarised per family as the mean over its bases.
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

## Prospective selection (Phase 6) — freeze point, 2026-11-06, before any bank candidate is generated
- **Radars:** CARRADA and RADDet, each held out in turn.
- **Bank** (src/natural.py BANK, on the best tuned simulator and its scene seeds): dr_narrow, dr_broad,
  dr_weak, dr_strong (per-sequence blur, range tail, target dB offset, confuser density, micro-motion
  drawn uniformly from the ranges in BANK); hybrid_x1, hybrid_x1.5 (targets unrendered, real target
  templates pasted at labels at 1x / 1.5x their excess; templates from real training sequences outside
  the metric's reference subset).
- **Metric:** M1 of src/utility_score.py with the probe definitions of this note, calibrated on the two
  other radars' development candidates (val AP), reference = budget 5 subset 0 of the held-out radar's
  real training sequences (scripts/prospective.py select; the tolerance intervals and NNLS weights are
  written into the selection file).
- **Selection rule:** the bank candidate with the lowest M1; ties by lower M2, then by name.
- The ranking and the selection are committed before any bank candidate is trained. Then every bank
  candidate trains synthetic-only (Study 2's training, 3 seeds); the proxy detector (seed 1) runs on
  the bank too.
- **Success:** the selected candidate's test AP@2 is within the bank's seed noise of the test oracle, and
  its regret is lower than random choice (expected), the global score's pick and the proxy's pick.
  Val-oracle and test-oracle picks are reported as upper references.

## Architecture and learned features (Phase 7) — fixed 2026-11-06, before its runs
- 7.1 (scripts/architecture_check.py): src/transformer_detector.py (0.84M parameters vs the U-Net's
  0.84M; 4 x 4 patches on 256 x 64 maps, 8 x 8 on 512 x 256; circular Doppler positions: the lowest 4
  harmonics, amplitude 1/k), Study 2's training and early stopping, 3 seeds, synthetic-only, on CARRADA
  and RADDet: spec, tuned, study2:rangeshift1, study2:blur2, study2b:tail2, and the metric-selected
  and oracle-best candidates of leave-one-radar-out (all candidates). Readout: AP@2 and seed range;
  Kendall tau between the transformer's and the U-Net's candidate ordering; whether M1's choice is
  within the transformer's seed noise of its best.
- 7.2 (scripts/learned_features.py): encoders = U-Net and transformer detectors trained real-only on
  one half of the 50 real training sequences (even / odd), early-stopped on the 32 val sequences,
  frozen; cross-fitted embeddings at label cells; raw multiscale patches as the unlearned comparison;
  energy distances, global and per stratum, evaluated as extra baselines in the Phase 3 framework.

## Classification redesign (Phase 8) — fixed 2026-11-06, before its runs
scripts/classify_controls_v2.py. Designs: current crops (25 x 17, label-centred, ridge filled with
noise); small (13 x 9); small + mask to the target's connected component (cells >= max(peak - 2, 1)
crop-sd, around the strongest cell within 1 of the centre) with N(0,1) background; the same
peak-centred; the same with range/speed-matched class sampling (5 x 5 quantile bins). Controls per
design: target erased (40 rows away), ridge only. Real-trained, 3 seeds, 95% interval over test
sequences. Gate: erased and ridge-only intervals include 1/3, and the real-trained interval lies above
the speed/range logistic baseline. Only a passing design is trained on the tuned simulator.

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

## Results: Phase 3-5 (2026-11-06 15:30)
All usable candidates: CARRADA 111, RADDet 76, RADIal 97. Reference = 5 real sequences (subset 0).
results/metric/scores/evaluation.json, figures/validation.png. Regret = test AP@2 of the oracle minus
that of the pick (lower is better); seed noise (median test seed range): ~0.05-0.07.

**Leave one family out (within a radar)**, mean regret / pairwise ordering accuracy:

| | CARRADA | RADDet | RADIal |
|---|---|---|---|
| M1 (tolerance) | 0.022 / 0.86 | 0.013 / 0.82 | 0.027 / 0.81 |
| proxy detector | 0.020 / 0.85 | 0.014 / 0.75 | 0.037 / 0.71 |
| global score | 0.065 / 0.55 | 0.062 / 0.64 | 0.027 / 0.79 |
| random | 0.077 / 0.50 | 0.077 / 0.50 | 0.102 / 0.50 |

M1 - global: CARRADA -0.043 [-0.080, -0.015], RADDet -0.049 [-0.091, -0.013], RADIal -0.001 [-0.018, 0.019].
M1 - proxy: +0.003, -0.001, -0.010, all intervals include 0. **Decision gate: passed on CARRADA and RADDet,
not on RADIal** (M1 ties the global score there). Caveat: within a radar the tolerance intervals are
anchored on the same simulator's undamaged versions, so M1 hardly depends on the real reference
(regret 0.022-0.024 over all 7 references): this test mostly measures "unlike a simulator known to work".

**Leave one radar out** (calibrated on the two other radars; the realistic test):

| held out | set | M1 | proxy | global | random | Kendall M1 / proxy / global |
|---|---|---|---|---|---|---|
| CARRADA | all 111 | 0.073 | 0.138 | 0.141 | 0.138 | 0.17 / 0.55 / 0.00 |
| CARRADA | natural 8 | 0.000 | 0.000 | 0.126 | 0.088 | 0.64 / 0.71 / - |
| RADDet | all 76 | 0.032 | 0.032 | 0.257 | 0.133 | 0.41 / 0.62 / 0.01 |
| RADDet | natural 8 | 0.000 | 0.000 | 0.022 | 0.052 | 0.57 / 0.43 / 0.36 |
| RADIal | all 97 | 0.172 | 0.136 | 0.128 | 0.283 | 0.43 / 0.63 / 0.15 |
| RADIal | natural 7 | 0.005 | 0.005 | 0.117 | 0.082 | 0.62 / 0.71 / -0.05 |

Reading:
1. **Choosing among simulators (natural candidates):** M1 and the proxy both pick the best (or within
   0.005) on all three radars; the global realism score does not (regret 0.12 on CARRADA and RADIal).
2. **M1 does not beat the proxy.** Top-1 among all candidates: M1 better on CARRADA (0.073 vs 0.138, about
   one noise width), equal on RADDet, worse on RADIal; the proxy orders all candidates better (Kendall
   0.55-0.63 vs 0.17-0.43). One selection per held-out radar: no interval.
3. **No cost advantage:** probes take 22 / 29 / 87 s CPU per candidate (CARRADA / RADDet / RADIal), the
   proxy 16 / 17 / 31 s GPU.
4. Map-only scores (global, coverage, domain classifier) are near random on their own.
5. RADIal: the best of all 97 candidates is study2:rangeshift1 (labels 1 range cell further than the tuned
   simulator's convention); "label only" picks it, M1 does not. The tuned RADIal label convention is
   probably about a cell off.

**Augmentation, 5 real sequences (Phase 5 Stage A)**, gain in test AP@2 over the same 5 sequences alone
(CARRADA 0.225, RADDet 0.101):

| | CARRADA | RADDet |
|---|---|---|
| tuned | +0.12 | +0.21 |
| rangeshift1 | +0.11 | +0.18 |
| extra50 | +0.14 | +0.17 |
| contrast-12 | +0.13 | +0.13 |
| unrender90 | +0.10 | +0.10 |
| spec | +0.10 | +0.08 |
| tail2 | +0.13 | +0.05 |

CARRADA: the spread (0.04) is within seed noise (0.05): with 5 real sequences, the candidates are
interchangeable. RADDet: spread 0.17 > 2 x noise, so Stage B runs (budgets 1 and 10). With real labels in
the batch the label shift costs little (synthetic-only it was the most harmful defect); target shape
(tail2) costs most on RADDet. Which simulator is "best" depends on the use.

## Results: Phase 8 classification redesign (2026-11-06)
results/metric/classification/controls_v2.json. Real-trained, test balanced accuracy, 95% interval
over test sequences (chance 0.33):

| design | CARRADA target / erased / ridge only | RADDet target / erased / ridge only |
|---|---|---|
| current (25 x 17) | 0.61 / 0.46 / 0.68 | 0.66 / 0.50 / 0.60 |
| small (13 x 9) | 0.63 / 0.42 / 0.65 | 0.64 / 0.43 / 0.57 |
| small, masked, noise background | 0.56 / 0.56 / 0.66 | 0.54 / 0.46 / 0.58 |
| same, peak-centred | 0.60 / 0.57 / 0.62 | 0.51 / 0.45 / 0.56 |
| same, range/speed matched | 0.32 / 0.35 / 0.53 (n 201) | 0.49 / 0.44 / 0.45 (n 777) |

Speed/range logistic baseline: 0.62 (CARRADA), 0.60 (RADDet); 0.37 on matched samples.
**No design passes the gate**; the classification metric stays blocked, as preregistered.
- The zero-Doppler ridge alone classifies as well as or better than the target (CARRADA 0.68): where the
  ridge falls inside a slow target's crop gives its speed away.
- Masking does not remove context: masked crops score about the same with the target erased (CARRADA
  0.56 vs 0.56).
- **With range and speed matched across classes, CARRADA classification is at chance (0.32)**: on CARRADA
  RD maps the classes are told apart by kinematics, not by target signature. RADDet keeps a small
  residual (0.49 vs 0.44 erased), still with erased and ridge-only above chance.

## Phase 6 selections (committed 03de8c3, before any bank candidate was trained)
M1 frozen on the two other radars, reference = 5 real sequences (subset 0):
- **CARRADA: bank:dr_broad** (bank order dr_broad, dr_strong, dr_narrow, dr_weak, hybrid_x1, hybrid_x1.5);
  global score picks hybrid_x1.
- **RADDet: bank:dr_weak** (dr_weak, dr_narrow, dr_broad, hybrid_x1.5, dr_strong, hybrid_x1); global picks dr_broad.
Weakness seen in the CARRADA scores (not acted on; the rule is fixed for this test): probes that barely vary
across the two calibration radars get the scale floor (1e-3), so their penalties are huge (coverage ~1,305
for every CARRADA candidate). Constant across the bank, it does not change the bank's order, which comes
from the target and label penalties; a cross-radar scale should not be allowed to collapse like this.

## Results: Phase 6 prospective test (bank trained after the selections were committed)
results/metric/prospective/<dataset>_report.json. Synthetic-only real-test AP@2, 3 seeds:

| bank candidate | CARRADA | RADDet |
|---|---|---|
| dr_weak | **0.376** | **0.263** |
| dr_narrow | 0.351 | 0.227 |
| dr_broad | 0.338 | 0.163 |
| dr_strong | 0.316 | 0.226 |
| hybrid_x1.5 | 0.320 | 0.061 |
| hybrid_x1 | 0.251 | 0.058 |
| seed noise | 0.080 | 0.044 |

| pick (regret) | CARRADA | RADDet |
|---|---|---|
| **metric M1** | dr_broad (0.039, within noise) | **dr_weak (0.000)** |
| global score | hybrid_x1 (0.126) | dr_broad (0.100) |
| proxy | dr_weak (0.000) | dr_narrow (0.036, within noise) |
| random (expected) | 0.051 | 0.097 |
| val oracle | dr_weak | dr_weak |

Preregistered success (within noise of the test oracle and lower regret than random, global and proxy):
**RADDet: met. CARRADA: within noise and better than random and global, but not better than the proxy.**
Also: domain-randomised targets with weaker strength (dr_weak) beat the tuned simulator on CARRADA
(0.376 vs 0.340); pasting real target templates (hybrid_*) is the worst on both radars (RADDet 0.06):
template pasting creates local discontinuities, the same kind of cue as the Study 1 smoothing box.

## Results: Phase 7.1 transformer detector (0.84M parameters, as the U-Net)
| | CARRADA transformer / U-Net | RADDet transformer / U-Net |
|---|---|---|
| spec | 0.296 / 0.295 | 0.111 / 0.153 |
| tuned | 0.275 / 0.340 | 0.132 / 0.292 |
| rangeshift1 | 0.130 / 0.129 | 0.122 / 0.200 |
| blur2 | 0.300 / 0.311 | 0.112 / 0.210 |
| tail2 | 0.365 / 0.343 | 0.105 / 0.134 |
| M1's LORO pick | 0.272 / 0.306 (fair:noise+6) | 0.170 / 0.300 (study2:noise+6) |
| oracle | 0.287 / 0.379 (sim2:all-scenes) | 0.173 / 0.333 (study2:clutter-12) |
| Kendall tau vs U-Net | 0.43 | 0.90 |

The 1-cell label shift is as damaging for the transformer on CARRADA (0.13 / 0.13): the label-geometry
finding is not a U-Net property. On RADDet the transformer is much weaker overall but orders candidates
like the U-Net (tau 0.90); on CARRADA the orders agree less (0.43). M1's pick is within the transformer's
noise of its best on RADDet, not on CARRADA.

## Results: Phase 5 Stage B (RADDet, 1 and 10 real sequences)
Gain in test AP@2 over the same real sequences alone (real only: 0.014 / 0.101 / 0.206 at 1 / 5 / 10):

| | 1 | 5 | 10 |
|---|---|---|---|
| tuned | +0.25 | +0.21 | +0.19 |
| sim2:all | +0.28 | - | +0.18 |
| rangeshift1 | +0.20 | +0.18 | +0.15 |
| extra50 | +0.21 | +0.17 | +0.14 |
| contrast-12 | +0.20 | +0.13 | +0.13 |
| blur2 | +0.17 | - | +0.11 |
| tail2 | +0.14 | +0.05 | +0.08 |
| unrender90 | +0.10 | +0.10 | +0.10 |
| sim2:all-range_gain | +0.10 | - | +0.14 |
| spec | +0.07 | +0.08 | +0.07 |
| seed noise | 0.044 | 0.067 | 0.031 |

On RADDet the order of candidates is stable across budgets (tuned / all fixes on top; spec, unrender90 and
tail2 at the bottom), the gains shrink slowly (tuned +0.25 -> +0.19), and the label shift stays a mild
defect once real labels are present (it was the most harmful one synthetic-only).

## Results: Phase 7.2 learned-feature distances (CARRADA, RADDet; final evaluation 2026-11-06 21:19)
Energy distances on cross-fitted embeddings of real-trained detectors (U-Net, transformer) and on raw
multiscale patches, as rankers (uncalibrated). Regret, reference 5 sequences:

| | CARRADA LOFO | CARRADA LORO all / natural | RADDet LOFO | RADDet LORO all / natural |
|---|---|---|---|---|
| M1 | 0.022 | 0.073 / 0.000 | 0.013 | 0.032 / 0.000 |
| proxy | 0.020 | 0.138 / 0.000 | 0.014 | 0.032 / 0.000 |
| U-Net features | 0.046 | 0.110 / 0.063 | 0.038 | 0.226 / 0.022 |
| U-Net features, conditional | 0.029 | 0.000 / 0.000 | 0.028 | 0.226 / 0.022 |
| transformer features | 0.030 | 0.086 / 0.084 | 0.024 | 0.135 / 0.022 |
| transformer features, conditional | 0.033 | 0.258 / 0.039 | 0.035 | 0.135 / 0.052 |
| raw patches | 0.045 | 0.110 / 0.063 | 0.014 | 0.049 / 0.022 |
| global score | 0.065 | 0.141 / 0.126 | 0.062 | 0.257 / 0.022 |
| random | 0.077 | 0.138 / 0.088 | 0.077 | 0.133 / 0.052 |

Learned features sit between the realism score and M1 / the proxy and are inconsistent (the same encoder
is best on one test and worst on another); transformer features are not better than the U-Net's or than
raw patches. Ordering of all candidates (Kendall, leave one radar out): proxy 0.55 / 0.62, M1 0.17 / 0.41,
learned 0.24-0.42. No learned feature space replaces the interpretable probes or the proxy.
