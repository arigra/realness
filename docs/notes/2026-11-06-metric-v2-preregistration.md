# Detection Metric v2: preregistration (2026-11-06)

Ari's plan (verbatim at the end) asks for four new selectors (M3-M6), a finalist chosen on development data
with validation AP only, and a second prospective bank whose rankings and selections are committed before any
full-detector training. This note fixes every choice the plan leaves open. Nothing under results/metric/ is
modified; new outputs go to results/metric_v2/.

## Audit (Phase A)
- Commit at start: 16184742821826cbb3a2c7040f51365842d537cf (hpc-node3, RTX 3090, driver 580.95.05).
- Python 3.10.12, torch 2.12.1+cu130, numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2, matplotlib 3.10.9.
- `PYTHONPATH=. python3 -m pytest tests -q`: 70 passed.

## Real references (real *training* sequences only; scripts/build_metric_features.subsets)
| budget | subset 0 (evenly spaced) | subset 1 | subset 2 |
|---|---|---|---|
| 5 | 0 12 24 37 49 | 3 19 25 45 48 | 14 23 26 33 48 |
| 10 | 0 5 11 16 22 27 33 38 44 49 | 1 3 5 23 28 40 42 45 47 49 | 10 11 14 24 25 31 32 38 42 43 |
| 20 | 0 3 5 8 10 13 15 18 21 23 26 28 31 34 36 39 41 44 46 49 | 2 3 9 12 16 18 20 21 22 24 26 28 31 32 33 37 38 41 47 48 | 1 5 13 14 17 18 20 22 23 24 27 31 34 37 42 43 44 45 48 49 |
| 50 | all | - | - |

**Primary: 10 sequences, subset 0.** Labels in it: CARRADA 34, RADDet 196, RADIal 64, so CARRADA and RADIal
are *low confidence* at the primary budget (< 100 labels): reported, not dropped.

## Development data and outcomes
- Development candidates: every usable candidate of results/metric/candidates.json plus the first bank
  (results/metric/bank.json, its synthetic-only runs). Outcome for development and the finalist: **whole-val
  AP@2** (mean of seeds). Candidates without val AP (fair, v1, smoothall) are scored but take no part in
  calibration, development evaluation or the finalist choice. Test AP is not read in Phases B-G.
- Undamaged (as in v1): mean val AP within the dataset's val seed noise (median val seed range) of the
  dataset's tuned simulator.
- Natural candidates (primary set of leave-one-radar-out): spec, tuned, sim2 variants, and the first bank.
- Families: the registry's mechanism families; the first bank is one family ("bank"), base tuned.

## Selectors (exactly these; no M7)
Probes are values of a candidate relative to the reference (shift = (candidate median - reference median) /
reference sd; rates and shares as differences), as in v1.

**M3_RTW.** The v1 probes (src/utility_score.PROBES). Safe interval = min..max over undamaged development
candidates. scale_j = max(MAD over development candidates, reference instability_j); reference instability_j =
median over development candidates of |x_j(subset s) - median_s x_j| over the three subsets of the evaluated
budget (50-sequence reference: the 20-sequence value). Both zero or missing: probe excluded (non-informative).
p_j = distance outside / scale_j, clipped to [0, 5]. Component risk = 80th percentile (numpy, linear) of the
component's measured penalties; secondary = mean of its three largest. Order: max component risk, then mean
component risk, then name.

**M4_RDA.** New raw features (src/utility_metric_v2.py), on raw dB and on the detector's normalised map:
- A detectability: strongest cell within Euclidean radius 2 of the label; background = the 25 x 17 patch around
  it minus radius 3 around it, radius 2 around every label, and the zero-Doppler ridge +/-1 (static radars);
  local CFAR margin = peak - q95(background), dB and z. Probes: shifts of the margin's median, q10, q25 (dB, z);
  visibility (target-like peak within radius 2; v1 threshold = 25th percentile of the reference's label
  prominence) as a difference.
- B label-task: v1 offsets (peak_dr, peak_dd, cen_dr, cen_dd); AP@2 energy capture = excess over the local
  range-row median, clipped at 0, within radius 2 (and 4) of the label over the excess in the 25 x 17 patch
  (missing when the patch has none); target-like peak within 2 / 4; unlabelled target-like peaks per frame,
  all and off the ridge.
- C morphology (raw dB relative to the local peak; physical units via RADARS): range width at -3 / -6 dB (m),
  Doppler width at -3 / -6 dB (m/s, wrapping), near / far range extent at -6 dB (m), asymmetry, peak count,
  range / Doppler entropy, autocorrelation lengths.
- D support: strata from the reference: near / far split at its median range x near-zero (|Doppler offset| <=
  1 bin) / moving. Probes: undercoverage max(0, p_real - p_cand) per stratum, its mean, the worst hole; joint
  range-speed energy distance; per-stratum visibility difference and q10-margin shift. A stratum with < 20
  reference labels is flagged low-confidence (kept; no merging).
- E hard negatives: real hard negatives = the fixed negative detector's false positives on the reference
  sequences + their 3 most prominent unlabelled peaks per frame; candidate negatives likewise; interpretable
  features standardised by the real hard negatives, clipped to [-5, 5]; nearest-neighbour distance real ->
  candidate vs real -> real of another reference sequence. Probes: median and q90 log ratio, uncovered share
  (beyond the real-real 95th percentile), worst stratum's uncovered share (ridge / far / strong, >= 5
  examples); FPs per frame shift.
- F processing: v1 processing probes (row-median and q99 profiles, ridge width / excess, residual
  correlations, clipping).
Risk per probe = min(p_j / 5, 1), p_j as in M3 (same interval and scale rules). Block risk = max of its measured
probe risks; secondary = mean of its top three. **M4 = 1 - prod(1 - block risk).** Ties: max block risk, then
label-task, then detectability, then name. Diagnosis: the two largest blocks and the probe that sets each.

**M5_REC.** Fixed negative detector (v1) on the reference's real training sequences: false negatives = labels
with no detection (threshold 0.05, top 10) within radius 2; false positives = detections with score >= 0.3 and no
label within 4; low-score true positives (best score within 2 below 0.3) reported only. Candidate banks:
positives = its labelled targets; negatives = as in M4-E. Features: positives: range (m), radial speed (m/s), CFAR
margin (dB), AP@2 capture, label-to-peak offsets, range / Doppler widths (-6 dB), near / far extents, prominence,
ridge flag; negatives: the same without capture and offsets. Standardised by the real errors of that type,
clipped to [-5, 5]. Per real error: nearest candidate example of the matching type and nearest same-type real
example of another reference sequence; ratio = candidate / real-real distance. Uncovered = candidate distance
beyond the 95th percentile of the real-real distances. FN risk = uncovered share of false negatives; FP risk =
of false positives; stratum risk = the worst uncovered share over near / far x zero / moving strata with >= 5
errors. **M5 = max(FN, FP, stratum risk).** Fewer than 20 real errors of a type: that side low-confidence, and M5
is also reported without it. No calibration.

**M6_MRR.** Inputs: the six M4 block risks + M5's FN and FP risks, z-scaled on development data. Non-negative
linear pairwise logistic ranker: pairs of development candidates of the same radar whose val AP differs by more
than that radar's val noise; input = risk difference; loss = weighted logistic loss + 1.0 ||w||^2; w >= 0
(L-BFGS-B bounds); pair weights such that each family (of the worse candidate) contributes equal total weight.
Score = w . z(risks); lower is better.

**Calibration for the bank-v2 selections:** M1, M2, M3, M4, M6 calibrated on all development candidates of all
three radars (deployment: the radar's earlier simulators have val AP); leave-one-radar-out calibration reported
as a secondary analysis. M5 and global need none. Proxy as v1 (width 8, 50 sequences, 1,000 steps), **three seeds**
for every bank-v2 candidate, scored on the primary 10-sequence reference.

## Development evaluation and finalist (Phase F; validation AP only)
Leave-one-family-out within each radar (family and base held out together; each family summarised once);
leave-one-radar-out (primary: natural candidates; stress test: all candidates with val AP). Regret, within-noise,
pairwise accuracy beyond noise, Kendall / Spearman, top-3 recall as v1's evaluate_set; ties at the top = uniform
choice. Finalist among M3-M6: (1) lowest mean leave-one-radar-out natural regret over the three radars; (2) within
0.01 AP: higher pairwise accuracy; (3) lower leave-one-family-out regret; (4) order M4, M3, M5, M6. Written to
results/metric_v2/finalist.json and committed before bank-v2 scoring.

## Bank v2 (CARRADA, scene-disjoint RADDet; RADIal only if compute permits)
On the best tuned simulator, scene seeds SEED0 + i (paired across candidates), 200 sequences each; per-sequence
draws (seed 9,000,000 + i) saved in each sequence's meta.
| candidate | definition |
|---|---|
| tuned_control | the best tuned simulator, unchanged (its cache) |
| scnr_narrow | target_db ~ U(-3, 3); static_db = target_db + U(-2, 2) |
| scnr_wide | target_db ~ U(-9, 6); static_db ~ U(-9, 6), independent |
| shape_range | range blur ~ U(1, 2.5) (Doppler 1); range tail ~ U(0, 4) cells |
| shape_doppler | Doppler blur ~ U(1, 2.5) (range 1); micro-motion scale ~ U(0, 1) |
| label_stochastic | each (object, frame) label moved to its strongest cell within +/-4 cells plus an offset drawn from the real training labels' (label - strongest cell) offsets, from real sequences outside the primary reference |
| support_stratified | scenes resampled so the four reference strata (near / far at the primary reference's median range x zero / moving) are equally likely: a scene's weight = mean over its objects of 1 / (stratum share among all training scenes' objects); generated shares saved |
| combined_moderate | scnr_narrow + range blur ~ U(1, 1.5) and tail ~ U(0, 2) + support_stratified scenes |
No pasted templates. A candidate is never replaced because of its score.

## Freeze, training, evaluation (Phases G-J)
As in the plan. The freeze commit holds the bank spec, cache metadata, all scores, rankings, one selection per
method and proxy outputs, and is checked to contain no bank-v2 full-detector result. Then every bank-v2
candidate trains synthetic-only (Study 2 training, early stopping on 32 val sequences, 3 seeds; tuned_control
too). Finalist success on a radar: within one median seed-noise width of the test oracle, regret below random
(expected) and below global, and the modified mechanism among its two largest risk blocks (scnr_* ->
detectability; shape_* -> morphology; label_stochastic -> label-task; support_stratified -> support;
combined_moderate -> any of detectability / morphology / support; tuned_control -> no requirement). Strong
result: success on both radars and mean regret no worse than the proxy's by more than half a seed-noise width.
Sample budgets: every selector at 5 / 10 / 20 (three subsets) and 50. Augmentation: RADDet bank-v2 mixed with the
fixed 5 and 10 real sequences of the budget study, 3 seeds. CARRADA's gate is Stage A's 5-sequence spread (0.040
< 2 x 0.051 noise): CARRADA mixed bank-v2 training is not run, and the report says candidate choice is
practically unimportant there at 5 sequences.

## Causal feature checks (Phase C) — fixed 2026-11-07, before the check is run
scripts/build_metric_v2_features.py --check: each controlled candidate vs the tuned simulator (same scene seeds),
paired bootstrap over the 200 synthetic sequences (300 draws), 95% interval must exclude 0 in the stated
direction. Reference for thresholds / strata / negatives: all 50 real training sequences.
| candidate | feature (median over objects unless stated) | expected |
|---|---|---|
| study2:contrast-12 | cfar_db, cfar_z, visibility (share of labels whose strongest cell within 2 reaches the reference's q25 prominence) | down |
| study2:noise+12 | cfar_db, cfar_z | down |
| study2:rangeshift1 | off_r (strongest cell - label, range), capture_r2 | down |
| study2:dopplershift1 | off_d, capture_r2 | down |
| study2:blur2 | width_r6_m, width_d6_mps | up |
| study2b:tail2 | far6_m, asym6_m | up |
| study2:unrender90 | visibility, capture_r2 | down |
| study2:extra50 | unlabelled target-like peaks off the ridge per frame (v1 tables) | up |
| study2b:support100 | worst stratum undercoverage vs the reference | up |
| study2b:confusers4 | fixed-detector false positives per frame | up |
| study2b:confusers0 | share of real hard negatives uncovered by the candidate's negatives | up |
A feature that fails is fixed before retrospective scoring or marked unmeasured (its probes are then dropped
from M4 / M5), never declared irrelevant.

## Deviation log
Every entry was made before any retrospective scoring of M3-M6 unless stated.
1. Draft review (2026-11-07): coverage features standardised NaN-aware (a morphology value cut at a range
   edge sits at the reference mean, instead of 0 before standardising); a detector false positive is measured
   at the strongest cell within +/-1 of the detection (as v1); M3 / M4 orders are exact lexicographic ranks
   (the draft's "max + 1e-4 mean" scalar could break ties wrongly); the tail unit test used a tail that
   overtook the peak and is now src/damage.range_tail.
2. **AP@2 energy capture leaves the zero-Doppler ridge +/-1 out of the 25 x 17 patch** on static radars,
   unless the label lies within 2 bins of zero Doppler (features v3). Found by the CARRADA Doppler-shift causal
   check (capture did not move: the ridge's static excess dominated the denominator, median capture 0.19).
   Re-checked with features v3: the range-shift response grew (-0.042 -> -0.054) but the Doppler shift still does
   not move it (-0.012 [-0.032, +0.022]), so `B.capture_r2` is **unmeasured on CARRADA** (the fix is kept).
3. `E.fp_per_frame` is **unmeasured on CARRADA**: the confusers x4 check failed (+0.05 FPs / frame, interval
   [-0.04, +0.14]); the fixed detector does not fire on more poles there. Its hard-negative coverage check
   (confusers 0 -> +0.29 uncovered real negatives) passed.
4. Further failed checks, no implementation fault found -> unmeasured on that radar (scripts/evaluate_metric_v2.UNMEASURED):
   RADIal `B.capture_r2` (range shift 1: -0.008 [-0.016, +0.002]; its targets are long in range); RADDet
   `E.uncovered` (confusers 0 *lowers* the uncovered share, -0.054 [-0.071, -0.020]; without poles the strongest
   unlabelled peaks look more like real hard negatives). RADDet's hard-negative block keeps a passing check
   (confusers x4: +0.15 FPs / frame). RADIal support100 is identical to its tuned simulator (RADIal's best
   simulator has no real-scene switch), so the support check cannot be run there; it passed on CARRADA and RADDet.
5. Diagnosis for the success criterion when the finalist is not M4 (fixed 2026-11-07 before Phase F's result):
   M6 reads M4's blocks, so it uses M4's two largest blocks; M3's components map as scnr_* -> label (it holds
   label_prom / over_clutter), shape_* -> target, label_stochastic -> label, support_stratified -> coverage,
   combined_moderate -> any of label / target / coverage; M5's sides map as support_stratified -> stratum risk,
   every other mechanism -> FN or FP risk (its two largest of FN / FP / stratum).
6a. After the freeze (bugs only, outputs affected: the *_report.json files, regenerated): (i) the report's
   "modified mechanism among the two largest risks" counted blocks at zero risk, so it was vacuously true when
   every risk was 0; only risks above 0 now count. (ii) A sequence bootstrap draw of the CARRADA reference could
   consist of label-free sequences only (no reference); such draws are skipped and counted (n_valid) - fixed
   before the freeze commit, during scoring.
6. M2's NNLS (unchanged v1 code) hits SciPy's iteration cap in some leave-one-family-out calibrations (v1's 1e-3
   scale floor makes penalties ~1e5); the identical problem is then solved by bounded-variable least squares
   (scipy lsq_linear, bvls). Implementation only; M2 is not redefined.
7. Clarifications of the text, not changes: visibility = the strongest cell within radius 2 reaches the 25th
   percentile of the reference labels' strongest-cell prominence; M5's real-real baseline = the nearest real
   example of the same kind (labelled targets for false negatives, negatives for false positives) in another
   reference sequence; leave-one-family-out calibrates on the same radar's other families (as v1), the held-out
   family's base excluded; support_stratified: scenes without objects keep their natural share of draws
   (CARRADA: 45% of its scene table), the others are reweighted.

## Progress
- 2026-11-07 06:30 (hpc-node3) Phase B-C started from 88d6516. Draft reviewed and fixed (06eb739).
- Phase B: M3 implemented (src/utility_score_v2.RobustTolerance, m3, lex_rank); unit tests; M1 / M2 through the
  v2 code path reproduce v1's Phase 6 bank values to 1e-15 (`scripts/evaluate_metric_v2.py --verify-v1`,
  results/metric_v2/scores/verify_v1.json).
- Phase C: causal checks run (results/metric_v2/features/<dataset>/causal_check.json); everything moved except
  the two CARRADA entries of the deviation log; capture re-checked after the fix (below).

## Results: Phase F development (validation AP only; DEVELOPMENT evidence, not the test)
Development candidates with val AP: CARRADA 82, RADDet 82, RADIal 75 (natural: 14, 14, 7); val seed noise
0.075 / 0.031 / 0.055. Primary reference (10 sequences, subset 0), regret in val AP@2 / pairwise accuracy:

| | M1 | M2 | global | M3 | M4 | M5 | M6 | proxy (v1, dev only) | random |
|---|---|---|---|---|---|---|---|---|---|
| LORO natural CARRADA | 0.000/0.93 | 0.043/0.82 | 0.193/0.45 | **0.000**/0.65 | 0.070/0.59 | 0.048/0.80 | 0.048/0.72 | 0.000/0.90 | 0.122 |
| LORO natural RADDet | 0.028/0.65 | 0.028/0.79 | 0.039/0.45 | 0.045/0.52 | 0.207/0.23 | 0.028/0.30 | 0.207/0.34 | 0.008/0.87 | 0.077 |
| LORO natural RADIal | 0.117/0.38 | 0.117/0.38 | 0.174/0.38 | **0.000**/0.38 | 0.164/0.38 | 0.028/0.91 | 0.000/0.56 | 0.117/0.44 | 0.105 |
| LORO all CARRADA | 0.353 | 0.162 | 0.193 | 0.468 | 0.206 | 0.224 | 0.140 | 0.000 | 0.207 |
| LORO all RADDet | 0.066 | 0.066 | 0.077 | 0.083 | 0.298 | 0.223 | 0.298 | 0.018 | 0.127 |
| LORO all RADIal | 0.294 | 0.312 | 0.058 | 0.399 | 0.085 | 0.354 | 0.112 | 0.060 | 0.274 |
| LOFO CARRADA | 0.021 | 0.026 | 0.081 | 0.030 | 0.024 | 0.100 | 0.024 | 0.016 | 0.105 |
| LOFO RADDet | 0.010 | 0.018 | 0.059 | 0.011 | 0.007 | 0.071 | 0.012 | 0.018 | 0.073 |
| LOFO RADIal | 0.041 | 0.038 | 0.062 | 0.045 | 0.040 | 0.086 | 0.030 | 0.033 | 0.113 |

**Finalist (frozen rule): M3_RTW**: mean LORO natural regret M3 0.015, M5 0.035, M6 0.085, M4 0.147 (rule step 1
decides; no other step needed). Caveats written before bank v2: (a) at the primary reference M3 wins, but over all
10 references its mean LORO natural regret is 0.083 vs random 0.101 and the v1 proxy 0.042: the development
signal is weak and reference-dependent; (b) M4, the radar-physics score, generalises worst across radars (it
picks RADDet's worst first-bank candidate, hybrid_x1), while within a radar (LOFO) M1, M3, M4 and M6 are all
close to each other and to the proxy; (c) on the all-candidate stress test every metric except the proxy is
near or worse than random on CARRADA and RADIal.

## Bank-v2 freeze (Phase G) — 2026-11-07, before any bank-v2 full-detector training
Rendered: results/metric_v2/bank_v2.json (caches data/tuned/<dataset>/bank2_*, per-sequence draws in each
sequence's meta; tuned_control = data/tuned/<dataset>/best). Features: results/metric_v2/features/<dataset>/bank2__*.
Proxy: 3 seeds per candidate, results/metric_v2/prospective/<dataset>_proxy.json. Scores, complete rankings at
every reference, calibration (all development candidates of the three radars; leave-one-radar-out calibration as
secondary), sequence-bootstrap intervals (100 draws at the primary reference, 20 elsewhere; CARRADA 99 valid at
10/0: one draw had no labels): results/metric_v2/prospective/<dataset>_selection.json.

**Selections at the primary reference (10 real sequences, subset 0)**; "tied" = candidates with exactly the
selected one's score, the pick then made by the method's preregistered tie rule (finally the name); bootstrap =
share of reference resamples in which the method picks the same candidate:

| method | CARRADA | RADDet |
|---|---|---|
| global | shape_range (1.00) | shape_range (0.98) |
| M1 | scnr_narrow (tied 5; 0.08) | combined_moderate (tied 7; 0.62) |
| M2 | scnr_narrow (tied 5; 0.11) | combined_moderate (tied 7; 0.58) |
| **M3 (finalist)** | **label_stochastic (tied 6; 0.44)** | **combined_moderate (tied 8; 0.73)** |
| M4 | scnr_narrow (tied 5; 0.04) | scnr_narrow (tied 5; 0.06) |
| M5 | shape_doppler (0.33) | label_stochastic (0.04) |
| M6 | shape_doppler (0.19) | combined_moderate (0.11) |
| proxy (3 seeds) | shape_range | combined_moderate |

Written before any outcome is known: **the tolerance scores barely separate this bank.** On RADDet every candidate
has M3 = 0 (all probes' 80th-percentile component risks inside the safe ranges), so M3's pick is the
alphabetically first name; on CARRADA six of eight tie at 0, broken by name. M1 and M4 tie 5-7 candidates at 0
as well. The prospective test of M3 is therefore close to a test of its tie rule. The 80th-percentile aggregation
also hides single-probe failures (CARRADA label_stochastic: M1 1.3, M4 label-task block 0.26 - M3 0). The report
gives the frozen picks as primary and, labelled secondary, each method's expected regret under a uniform choice
among its tied best (the development rule). The proxy rates label_stochastic lowest on both radars (0.08 / 0.03).

## Results: Phases H-I (prospective, test AP; bank v2 trained after the freeze commit 3281151)
results/metric_v2/prospective/<dataset>_{runs,report}.json, figures in results/metric_v2/figures/. Synthetic-only
U-Net, Study 2's training and early stopping, 3 seeds. Real-test AP@2 (seed range in brackets):

| candidate | CARRADA | RADDet |
|---|---|---|
| tuned_control | 0.324 [0.030] | 0.284 [0.037] |
| scnr_narrow | 0.314 [0.035] | 0.292 [0.036] |
| scnr_wide | 0.348 [0.035] | **0.296** [0.057] |
| shape_range | **0.392** [0.019] | 0.142 [0.022] |
| shape_doppler | 0.324 [0.049] | 0.254 [0.050] |
| label_stochastic | 0.136 [0.072] | 0.063 [0.039] |
| support_stratified | 0.320 [0.054] | 0.293 [0.047] |
| combined_moderate | 0.345 [0.055] | 0.215 [0.035] |
| seed noise (median range) | 0.042 | 0.038 |

The val oracle equals the test oracle on both radars (shape_range, scnr_wide). The same mechanism has opposite
effects on the two radars (range blur / tail: best on CARRADA, second worst on RADDet); stochastic real label
offsets are the worst candidate on both.

**Selections at the primary reference (10 sequences, subset 0): test regret (AP@2) / within noise / pairwise
accuracy beyond noise / Kendall tau over the bank; secondary: expected regret if the tied best were chosen
uniformly.**

| method | CARRADA pick | regret | within | pairwise / tau | ties-uniform | RADDet pick | regret | within | pairwise / tau | ties-uniform |
|---|---|---|---|---|---|---|---|---|---|---|
| global | shape_range | **0.000** | yes | 0.92 / 0.43 | 0.000 | shape_range | 0.154 | no | 0.24 / -0.57 | 0.154 |
| M1 | scnr_narrow | 0.078 | no | 0.77 / 0.21 | 0.051 | combined_moderate | 0.081 | no | 0.43 / -0.07 | 0.054 |
| M2 | scnr_narrow | 0.078 | no | 0.62 / 0.07 | 0.051 | combined_moderate | 0.081 | no | 0.43 / -0.07 | 0.054 |
| **M3 (finalist)** | label_stochastic | **0.255** | no | 0.23 / -0.29 | 0.085 | combined_moderate | 0.081 | no | 0.33 / -0.21 | 0.066 |
| M4 | scnr_narrow | 0.078 | no | 0.77 / 0.21 | 0.051 | scnr_narrow | **0.004** | yes | 0.90 / 0.64 | 0.012 |
| M5 | shape_doppler | 0.068 | no | 0.69 / 0.29 | 0.068 | label_stochastic | 0.233 | no | 0.14 / -0.36 | 0.233 |
| M6 | shape_doppler | 0.068 | no | 0.92 / 0.43 | 0.068 | combined_moderate | 0.081 | no | 0.29 / -0.14 | 0.081 |
| proxy (3 seeds) | shape_range | **0.000** | yes | 1.00 / 0.29 | 0.000 | combined_moderate | 0.081 | no | 0.67 / 0.14 | 0.081 |
| random (expected) | | 0.079 | 12% | 0.50 | | | 0.066 | 50% | 0.50 | |
| val oracle | shape_range | 0.000 | yes | | | scnr_wide | 0.000 | yes | | |

Mean regret over the two radars: proxy 0.040, M4 0.041, M6 0.075, global 0.077, M1 / M2 0.080, random 0.072,
M5 0.150, **M3 0.168**.

**Preregistered success of the finalist (M3): not met on either radar.** CARRADA: not within noise (0.255 >
0.042), worse than random (0.079) and global (0.000); its pick (label_stochastic) has the label-task risk only in
M1 / M4, not in M3's top components (all M3 risks 0 except coverage). RADDet: not within noise (0.081 > 0.038),
worse than random (0.066), better than global (0.154); no positive risk names the mechanism. **Strong result: no.**
Proxy comparison: the finalist is worse than the proxy by 0.255 on CARRADA and equal on RADDet (same pick) -
mean 0.128 worse, three seed-noise widths, far beyond the half-width allowed.

Diagnosis (M4's blocks above zero risk vs the modified mechanism): CARRADA 3 of 7 modified candidates
(label_stochastic -> label-task; support_stratified, combined_moderate -> support); RADDet 1 of 7 (shape_range ->
morphology). Every other candidate had no positive M4 risk: the bank v2 mechanisms mostly stay inside the
development candidates' safe ranges, which is why the tolerance scores tie.

Leave-one-radar-out calibration (secondary, not the frozen selection): M3 would have picked scnr_wide on CARRADA
(regret 0.044) and support_stratified on RADDet (0.003); M4 label_stochastic (0.255) and shape_range (0.154). The
calibration set changes the tolerance metrics' picks completely: their selections are not robust.

**Sample budgets (Phase I; selections at 5 / 10 / 20 sequences x 3 subsets and 50; mean test regret per budget):**

| method | CARRADA 5 / 10 / 20 / 50 | stable from | RADDet 5 / 10 / 20 / 50 | stable from |
|---|---|---|---|---|
| global | 0.000 / 0.000 / 0.000 / 0.000 | 5 | 0.154 / 0.154 / 0.154 / 0.154 | 5 |
| M1 | 0.196 / 0.196 / 0.255 / 0.255 | 20 | 0.182 / 0.182 / 0.081 / 0.081 | 20 |
| M3 | 0.196 / 0.255 / 0.255 / 0.255 | 10 | 0.131 / 0.182 / 0.081 / 0.081 | 20 |
| M4 | 0.031 / 0.127 / 0.101 / 0.047 | never | 0.029 / 0.080 / 0.029 / 0.004 | never |
| M5 | 0.101 / 0.023 / 0.016 / 0.044 | never | 0.105 / 0.233 / 0.154 / 0.233 | 10 |
| M6 | 0.016 / 0.023 / 0.170 / 0.255 | never | 0.027 / 0.156 / 0.103 / 0.081 | never |
| proxy | 0.038 / 0.039 / 0.054 / 0.047 | never | 0.008 / 0.029 / 0.006 / 0.003 | never |

Reference sizes (labels / real FNs / real FPs of the fixed detector): CARRADA 23-32 / 6-14 / 8-31 at 5 sequences,
34-55 / 15-27 / 36-49 at 10, 104-128 at 20, 270 / 127 / 221 at 50 (so CARRADA is low-confidence below 20 sequences);
RADDet ~100 / ~45 / ~175 at 5, 196-277 at 10, 1,155 / 457 / 1,712 at 50. "Stable" = the same pick on all three
subsets; stability is not correctness: M3 is stable on CARRADA from 10 sequences on the worst candidate. No metric
reaches a correct, stable pick at any budget; the proxy is near-correct at every budget without being stable
(its picks change among near-equivalent candidates). Bootstrap selection frequency of the frozen picks at 10/0: M3
0.44 (CARRADA) / 0.73 (RADDet), M4 0.04 / 0.06, global 1.00 / 0.98.

**Augmentation (RADDet, + the budget study's fixed 5 / 10 real sequences, 3 seeds).** Real only: 0.101 / 0.206.
tuned_control + real reproduces the budget study's tuned mix (0.290 vs 0.314 at 5; 0.394 vs 0.398 at 10).
Delta_AP_5: support_stratified +0.247, scnr_narrow +0.238, scnr_wide +0.212, shape_doppler +0.211, tuned +0.190,
combined_moderate +0.170, shape_range +0.112, label_stochastic +0.084 (spread 0.163, noise 0.051). Delta_AP_10:
scnr_narrow +0.190, tuned +0.188, support_stratified +0.186, shape_doppler +0.180, scnr_wide +0.154,
combined_moderate +0.142, shape_range +0.097, label_stochastic +0.094 (spread 0.096, noise 0.028). Regret of each
method's (synthetic-only) pick on Delta_AP 5 / 10: M4 0.009 / 0.000 (pairwise 1.00 / 0.89, tau 0.71 / 0.50), proxy
0.076 / 0.048 (0.87 / 0.79), M1 / M2 / M3 / M6 0.076 / 0.048, global 0.135 / 0.093, M5 0.163 / 0.096, random
0.064 / 0.036. With real labels in the batch the ordering is the synthetic-only one compressed (label noise and
range shape still cost most). CARRADA mixed training was not run (preregistered gate: Stage A spread 0.040 <
2 x 0.051): candidate choice is practically unimportant there at 5 real sequences.

Cost per candidate (hpc-node3, under load): v2 features 9-15 s (CPU, plus the fixed detector on GPU), v1 features
18-22 s, scoring < 1 s per reference; proxy 12-23 s per seed on the 3090 (x3 seeds); a full training 160-180 s.

## Verdict (Phase J)
- **Which metric won:** none of the four new metrics passed its preregistered test. The frozen finalist, **M3_RTW,
  failed on both radars** (mean regret 0.168, the worst of all methods, worse than random 0.072). Its failure is
  structural, not noise: with robust scales and an 80th-percentile weakest link, nearly every bank-v2 candidate
  sits at zero risk (6 / 8 on CARRADA, 8 / 8 on RADDet), so the pick is the preregistered name tie-break; even
  uniform choice among its ties is only random-level (0.085 / 0.066).
- **M4_RDA** (the radar-physics score, not chosen by the development rule because it generalised worst across
  radars on val AP) was the best metric prospectively: within noise on RADDet (synthetic-only and both
  augmentation budgets, tau 0.50-0.71) but 0.078 regret on CARRADA, and its pick is not stable across reference
  subsets (bootstrap frequency 0.04-0.06). One radar out of two; post hoc; not a preregistered success.
- **Metric vs proxy: the proxy won.** Synthetic-only mean regret proxy 0.040 vs finalist 0.168 (M4 0.041, global
  0.077, random 0.072); the proxy picked the oracle on CARRADA and orders the bank best there (pairwise 1.00), and is
  within noise on RADDet at the other reference subsets (mean regret 0.003-0.029) though not at the primary one
  (0.081, same pick as M3). On augmentation utility the proxy is behind M4 but ahead of the tolerance scores. Per the
  plan's stopping rule this is the "proxy wins clearly" outcome: per-candidate task probing is the best selector;
  M4 / M5 are kept for diagnosis only, and their diagnoses named the modified mechanism for 4 of 14 modified
  candidates, so even that role is weak.
- **Global realism score:** right on CARRADA (it picks shape_range at every budget on both radars, which
  happened to be CARRADA's best) and the worst selector on RADDet (0.154): a radar-dependent coin, as in v1.
- **How many real sequences were needed:** for the metrics, no budget up to 50 sequences gave a correct and stable
  pick on either radar, so no sample requirement can be stated for them; more sequences did not fix the tolerance
  scores (on CARRADA M1 and M3 pick label_stochastic, the worst candidate, at 50 sequences). For the proxy, 5 sequences already gave near-oracle
  picks on both radars (mean regret 0.038 CARRADA, 0.008 RADDet), with no gain from 50.
- **What this means for the study:** the development data (damaged versions of known simulators) teach the
  metrics what "unlike a working simulator" looks like, and the safe ranges become wide enough that a new bank of
  plausible variations looks equally safe. Utility differences inside that space (range shape helps CARRADA,
  hurts RADDet) are not visible to distribution-matching probes at 10-50 real sequences, but a 50-sequence,
  1,000-step detector run sees them. Honest framing: an analysis result (which defects matter, and that cheap task
  probing beats realism metrics), not a metric that replaces training.

---

## The plan (verbatim, as given 2026-11-06)

# Agent TODO: Detection Metric v2 Experiments

## 0. Scope and objective

This is the next experiment plan for the `realness` repository.

The objective is to test whether a more scientific, radar-specific metric can select useful synthetic range-Doppler training data better than:

- the existing global realism score;
- M1, the equal-weight tolerance score;
- M2, the AP-weighted version of M1;
- random selection;
- a small proxy detector.

The target is **candidate selection**, not exact AP prediction.

Primary question:

> Given several candidate synthetic RD datasets and a small labelled real reference set, which metric selects the candidate that gives the highest real-test AP@2 after full detector training?

Secondary question:

> Can the metric explain whether the main risk comes from detectability, labels, target morphology, operating-region support, or hard negatives?

### Explicit exclusions

- **Do not implement IQ, ADC, inverse-FFT, complex-RD, or waveform-level experiments.**
- **Do not resume classification experiments in this phase.** Current classification controls still contain kinematic/context shortcuts.
- Do not create another detector architecture until after the prospective Metric-v2 result is complete.
- Do not modify the already completed M1 prospective experiment.

This phase is detection-only and operates on the RD maps seen by the current detector.

---

## 1. Read and preserve the current experiment

Before editing code, read:

- `docs/notes/2026-11-06-metric-experiments.md`
- `src/utility_metric.py`
- `src/utility_score.py`
- `scripts/build_metric_features.py`
- `scripts/evaluate_utility_metric.py`
- `scripts/prospective.py`
- `src/natural.py`
- `results/metric/scores/evaluation.json`
- `results/metric/prospective/*_selection.json`
- `results/metric/prospective/*_report.json`
- `results/metric/augmentation_candidates.json`

The following results are completed and are development evidence, not work to rerun by default:

- the five-part adequacy vector;
- M1 and M2;
- leave-one-family-out and leave-one-radar-out evaluation;
- the first prospective candidate bank;
- the small proxy detector;
- RADDet augmentation budgets 1, 5, and 10;
- the transformer subset;
- classification shortcut controls.

Do not overwrite any file under `results/metric/`.

---

## 2. Non-negotiable experimental rules

- [ ] Record `git rev-parse HEAD`, environment, package versions, GPU, and test status.
- [ ] Run the existing test suite before changes.
- [ ] Create a new preregistration note before evaluating the new metrics.
- [ ] Put new code and outputs under `metric_v2` names and `results/metric_v2/`.
- [ ] Use only real **training** sequences to compute metric inputs.
- [ ] Never read real validation or test maps while computing candidate metric scores.
- [ ] Use validation AP for metric development and finalist selection.
- [ ] Do not use test AP to choose features, thresholds, weights, or the finalist metric.
- [ ] Treat all severity levels from one intervention family as one dependent group.
- [ ] Do not randomly split candidate rows from the same family.
- [ ] Freeze the Metric-v2 finalist, candidate-bank-v2 specification, rankings, and selections before full bank-v2 training.
- [ ] After the freeze, only implementation bugs may be fixed. Record every bug and affected output.
- [ ] Compare all metrics with the same real-reference subset and the same synthetic sequence count.

Create:

`docs/notes/2026-11-XX-metric-v2-preregistration.md`

Copy this plan into that note, then add the exact commit, reference-sequence indices, candidate parameter ranges, and commands.

---

## 3. Existing baselines: keep unchanged

The comparison must include:

- `global`: existing whole-map realism score;
- `M1`: existing equal-weight tolerance score;
- `M2`: existing non-negative weighted M1 score;
- `proxy`: existing small detector procedure;
- `random`: expected random choice;
- `val_oracle`: validation upper reference only;
- `test_oracle`: reported only after prospective selections are frozen.

Do not redefine these methods.

---

# 4. New metric candidates

Implement exactly four new candidates. Do not add M7 after seeing results.

## M3: Robust weakest-link tolerance score

Code name: `M3_RTW`

### Purpose

Test whether the current probe vector is useful and the main problem is M1's scaling and equal-sum aggregation.

### Inputs

Use the existing M1 probes from `src/utility_score.py`. Do not add new raw features for M3.

### Calibration

For each probe `j`, using development candidates only:

1. Define `undamaged` exactly as in the existing experiment: mean validation AP is within validation seed noise of the candidate's base.
2. Safe interval:
   - `lower_j = min(x_j)` over undamaged development candidates;
   - `upper_j = max(x_j)` over undamaged development candidates.
3. Replace M1's fixed `1e-3` scale floor with:

   `scale_j = max(MAD across development candidates, reference instability)`

   `reference instability` is the median absolute change in the same candidate's probe across the fixed real-reference subsets at the same budget.
4. If both scale terms are zero or unavailable, mark the probe non-informative and exclude it.
5. Penalty:

   `p_j = distance outside safe interval / scale_j`

6. Clip `p_j` to `[0, 5]`.

### Aggregation

For each of the five existing components:

- component risk = 80th percentile of measured probe penalties;
- secondary summary = mean of the three largest measured penalties.

Overall order:

1. lower maximum component risk;
2. lower mean component risk;
3. candidate name.

Do not sum every probe. One severe failure must not be cancelled by many harmless probes.

### Output

For every candidate, store:

- each probe value;
- safe interval;
- scale;
- penalty;
- component risk;
- dominant component;
- low-confidence/unmeasured flags.

---

## M4: Radar Detection Adequacy score

Code name: `M4_RDA`

This is the primary radar-specific metric.

### Scientific idea

A synthetic dataset is useful for RD detection when it provides:

1. targets that are detectable against local clutter and noise;
2. labels that agree with the returns under the AP@2 task;
3. target morphology appropriate for the radar resolution and processing;
4. coverage of important range-velocity operating regions;
5. negative structures similar to real detector false positives.

### M4-A: local detectability

For each labelled object, calculate on both the raw dB map and the detector-normalised map:

1. strongest cell within circular Euclidean radius 2 of the label;
2. local background in a `25 x 17` patch around that cell;
3. exclude:
   - radius 3 around the strongest cell;
   - radius 2 around every label in the patch;
   - the zero-Doppler ridge `+/-1` bin on static radars.

Compute:

- `cfar_margin_db = peak_db - q95(local_background_db)`;
- `cfar_margin_z = peak_z - q95(local_background_z)`;
- visibility at the real-reference target-like threshold;
- median, 10th percentile, and 25th percentile margin;
- visibility rate;
- the same summaries by range/velocity stratum.

Call this a **local CFAR margin**, not exact SCNR.

### M4-B: label-task agreement

For every label, compute:

- strongest-cell offset in range and circular Doppler;
- centroid offset;
- fraction of target excess energy within AP radius 2;
- fraction within radius 4;
- target-like peak within radius 2 and 4;
- label with no visible peak;
- unlabelled target-like peaks per frame;
- unlabelled off-ridge peaks per frame.

Energy capture:

1. subtract the local range-row median;
2. clip negative excess to zero;
3. numerator = excess energy within circular Euclidean radius 2 of the label;
4. denominator = excess energy in the full target patch;
5. if denominator is zero, mark the object missing rather than assigning zero.

### M4-C: target morphology in physical units

Measure relative to the local target peak:

- range width at -3 dB and -6 dB, in metres;
- Doppler width at -3 dB and -6 dB, in m/s, with Doppler wrapping;
- near-side and far-side range extent at -6 dB;
- range-tail asymmetry;
- peak count;
- range and Doppler entropy;
- existing autocorrelation/shape features.

Use raw dB relative to the local peak where possible. Do not rely only on source-normalised z units.

Use radar metadata from `src.utility_metric.RADARS` for physical conversion.

### M4-D: operating-region support

Primary strata are defined from the real reference:

- near versus far: split at the real-reference median range;
- near-zero versus moving: absolute relative Doppler <= 1 bin versus > 1 bin.

For each stratum, calculate:

- real share and candidate share;
- undercoverage `max(0, p_real - p_candidate)`;
- target count;
- visibility rate;
- 10th-percentile local CFAR margin;
- morphology summaries.

Also keep the existing joint range-speed energy distance.

If a real stratum contains fewer than 20 labelled objects, mark it low-confidence. Merge only according to a prespecified rule; do not invent a value.

Include both:

- average support discrepancy;
- worst important support hole.

### M4-E: hard-negative adequacy

Reuse the existing fixed negative detector.

Real hard negatives are:

- false positives on real training sequences;
- high-prominence unlabelled peaks;
- ridge, long-range, and high-prominence subsets where sample counts allow.

Candidate negatives are:

- fixed-detector false positives on the candidate;
- strong unlabelled synthetic peaks.

Compute:

- median and 90th-percentile nearest-neighbour distance from each real hard negative to candidate negatives;
- real-to-real cross-sequence nearest-neighbour baseline;
- candidate/real distance ratio;
- uncovered share beyond the real-real 95th percentile;
- worst stratum uncovered share.

### M4-F: processing diagnostics

Retain:

- range-row median and q99 profiles;
- zero-Doppler ridge width/excess on static radars;
- residual range and Doppler correlations;
- clipping fractions;
- known processing-window mismatch from candidate metadata when available.

Processing is one block only. It must not dominate because it contains many probes.

### Risk conversion

For each scalar probe:

1. learn a safe interval from undamaged development candidates using validation AP;
2. use the M3 data-derived scale rule;
3. convert distance outside the interval to a capped risk in `[0, 1]`.

For each block:

- block risk = maximum measured probe risk;
- secondary = mean of top three measured probe risks.

Overall M4 risk:

`M4_RDA = 1 - product(1 - block_risk)`

Lower is better.

Tie-break by:

1. lower maximum block risk;
2. lower label-task risk;
3. lower detectability risk;
4. candidate name.

Store a human-readable diagnosis such as:

- `high label risk: low AP@2 energy capture`;
- `high morphology risk: far range tail too long`;
- `high support risk: far near-zero-Doppler undercoverage`.

---

## M5: Real detector-error coverage

Code name: `M5_REC`

### Purpose

Ask whether a candidate contains the examples needed to fix the real errors of a fixed detector.

### Fixed detector

Use the existing dataset-specific detector trained on the best tuned simulator. Do not retrain it per candidate.

On each real training reference subset, collect:

- false negatives: labels with no detection within radius 2;
- false positives: detections with score >= 0.3 and no label within radius 4;
- low-score true positives as a reported sensitivity analysis only.

### Candidate example banks

- Positive bank: all labelled synthetic targets.
- Negative bank: fixed-detector false positives plus strong unlabelled synthetic peaks.

### Interpretable example features

Use:

- physical range;
- radial velocity;
- local CFAR margin;
- AP@2 energy capture;
- label-to-peak offset for positives;
- range and Doppler widths;
- range tails;
- peak prominence;
- ridge flag;
- existing target-response features.

Standardise using real-reference errors only and clip to `[-5, 5]`.

The primary M5 must use interpretable radar features. A frozen detector embedding may be an explicitly labelled sensitivity analysis, not a silent replacement.

### Coverage score

For every real error example:

1. find the nearest matching candidate example;
2. find the nearest same-type real example from another sequence;
3. compute:

   `ratio = candidate_distance / real_to_real_distance`.

Summarise separately for false negatives and false positives:

- median log ratio;
- 90th-percentile log ratio;
- uncovered share beyond the real-real 95th percentile;
- worst range/velocity stratum uncovered share.

Overall M5 risk is the maximum of:

- false-negative coverage risk;
- false-positive coverage risk;
- worst support-stratum risk.

If fewer than 20 real errors of a type are available, mark that side low-confidence and report M5 both with and without it.

---

## M6: Low-capacity monotone radar ranker

Code name: `M6_MRR`

### Purpose

Test whether M4/M5 components are useful but require learned relative importance.

### Inputs

Use at most these eight values:

1. M4 detectability risk;
2. M4 label-task risk;
3. M4 morphology risk;
4. M4 support risk;
5. M4 hard-negative risk;
6. M4 processing risk;
7. M5 false-negative coverage risk;
8. M5 false-positive coverage risk.

Do not feed dozens of raw probes to M6.

### Model

Fit a non-negative pairwise logistic ranker using development validation AP:

- form pairs only when validation AP differs by more than validation seed noise;
- input is the difference between component-risk vectors;
- all weights must be non-negative;
- no hidden layers;
- z-scale components on development data;
- fixed L2 regularisation = 1.0;
- weight pairs so each defect family contributes equal total weight.

Lower weighted risk is better.

Save:

- development candidates;
- pairs;
- family weights;
- fitted component weights;
- calibration commit/hash.

Do not train a separate budget-specific M6 unless at least 20 independent mixed-training candidate mechanisms become available.

---

## 5. Real-reference sample budgets

Compute M3-M6 at:

- 5 real training sequences, three fixed sequence-level subsets;
- 10 sequences, three subsets;
- 20 sequences, three subsets;
- all 50 sequences.

Primary reference for the final prospective selection:

- **10 sequences, subset 0**.

For every score, save:

- sequence indices;
- number of independent sequences;
- number of labels;
- labels in each stratum;
- number of real false negatives and false positives;
- sequence-bootstrap interval;
- low-confidence flags.

Reporting thresholds:

- fewer than 100 labels overall: low confidence;
- fewer than 20 labels in a stratum: low-confidence stratum;
- fewer than 20 false negatives or false positives: low-confidence M5 side.

These thresholds do not delete results. They control interpretation.

---

## 6. New code and output locations

Create:

- `src/utility_metric_v2.py`
- `src/utility_score_v2.py`
- `scripts/build_metric_v2_features.py`
- `scripts/evaluate_metric_v2.py`
- `scripts/prospective_v2.py`
- `scripts/plot_metric_v2.py`
- `tests/test_utility_metric_v2.py`
- `docs/notes/2026-11-XX-metric-v2-preregistration.md`

Write only under:

- `results/metric_v2/features/`
- `results/metric_v2/scores/`
- `results/metric_v2/prospective/`
- `results/metric_v2/figures/`

Every result JSON must contain:

- repository commit;
- feature-config version and hash;
- radar metadata;
- candidate source/cache;
- real-reference sequence indices;
- metric version;
- sample counts;
- runtime and hardware;
- low-confidence flags.

---

## 7. Required tests before metric evaluation

Add unit tests for:

- [ ] circular Doppler distance and windows;
- [ ] local CFAR background excludes target guard cells and nearby labels;
- [ ] a stronger target raises local CFAR margin;
- [ ] increased noise lowers local CFAR margin;
- [ ] stronger clutter can lower local CFAR margin even when target-over-floor is unchanged;
- [ ] a one-cell range shift changes geometry and AP@2 energy capture;
- [ ] a Doppler shift wraps correctly;
- [ ] an unrendered target lowers visibility;
- [ ] an added unlabelled target raises off-ridge unlabelled peaks;
- [ ] range blur increases physical range width;
- [ ] Doppler blur increases physical Doppler width;
- [ ] a range tail increases far-side extent/asymmetry;
- [ ] identical candidate/reference gives baseline risk;
- [ ] real-to-real error coverage has median ratio near one;
- [ ] the builder reads training splits only;
- [ ] fixed inputs give deterministic outputs.

### Causal feature checks

Using shared-seed controlled variants, verify that the intended feature moves in the expected direction with a paired sequence-bootstrap interval excluding zero:

- contrast/noise -> detectability;
- range/Doppler shift -> label-task geometry;
- blur/tail -> morphology;
- extra/unrender -> label consistency;
- support intervention -> support;
- an available hard-negative mechanism -> negative coverage.

If a feature fails its causal check:

1. fix the implementation before retrospective scoring, or
2. mark the feature unmeasured.

Do not declare the physical factor irrelevant merely because a probe failed.

---

## 8. Retrospective development protocol

This phase selects the Metric-v2 finalist. It is not the final proof.

Use existing candidates and the first prospective bank as development evidence. Use **validation AP**, not test AP, to choose between M3-M6.

### 8.1 Leave-one-family-out

For every radar:

- hold out the entire mechanism family and its base together;
- calibrate on all remaining families;
- rank the held-out base and levels;
- summarise each family once, regardless of number of severity levels.

### 8.2 Leave-one-radar-out

- calibrate on two radars;
- evaluate on the held-out radar;
- primary set: natural candidates only;
- all-candidate set: stress test only.

### 8.3 Finalist rule

Choose exactly one Metric-v2 finalist using validation outcomes only:

1. lowest mean leave-one-radar-out regret on natural candidates;
2. if within 0.01 AP, higher pairwise accuracy for pairs beyond seed noise;
3. if still tied, lower leave-one-family-out regret;
4. if still tied, prefer simpler methods in this order: M4, M3, M5, M6.

Write:

`results/metric_v2/finalist.json`

Commit this file before prospective bank-v2 scoring.

Do not delete losing metrics from reports.

---

## 9. Prospective candidate bank v2

The first bank's outcomes are known. It cannot be the final unseen test for newly designed metrics.

Create a second bank on the best tuned simulator. Use the same scene seeds across candidates where technically possible.

Required candidates:

1. `tuned_control`
   - unchanged best tuned simulator.

2. `scnr_narrow`
   - narrow joint randomisation of target and static/background strength;
   - preserve plausible target-to-clutter-plus-noise margins.

3. `scnr_wide`
   - broader joint target/background randomisation.

4. `shape_range`
   - randomise range blur and one-sided range tail;
   - keep Doppler shape near tuned.

5. `shape_doppler`
   - randomise Doppler blur and micro-motion;
   - keep range shape near tuned.

6. `label_stochastic`
   - draw label-to-return offsets from a distribution estimated only from permitted real training labels;
   - do not use validation/test labels.

7. `support_stratified`
   - sample or reweight scenes to better cover the four real range/velocity strata;
   - save the actual generated distribution.

8. `combined_moderate`
   - combine narrow SCNR, moderate range shape, and support stratification;
   - all ranges fixed in the preregistration.

Do not include pasted real target templates. The first bank showed local discontinuity artifacts.

Bank requirements:

- 200 synthetic sequences per candidate;
- CARRADA and scene-disjoint RADDet required;
- RADIal optional only if compute permits after the required radars;
- save every per-sequence random draw;
- do not replace a candidate because its metric score looks poor.

---

## 10. Prospective freeze and training procedure

For each required radar:

1. Build all bank-v2 caches.
2. Compute global, M1, M2, M3, M4, M5, M6, and proxy scores.
3. Use the same 10-sequence primary reference for every method.
4. Run three proxy seeds for every bank candidate.
5. Save complete rankings and one selected candidate per method.
6. Commit:
   - metric code and hashes;
   - bank specification;
   - reference indices;
   - rankings;
   - selections;
   - proxy outputs.
7. Confirm no full-detector bank-v2 result exists in the commit.
8. Only after the freeze, train the full detector on every bank-v2 candidate with three seeds using the existing Study-2 training/early-stopping protocol.
9. Evaluate AP@2, AP@4, and recall@2 on real test.
10. Do not change metrics after results arrive.

---

## 11. Primary evaluation

For every selector report:

- top-1 test regret: `oracle AP@2 - selected AP@2`;
- whether selected AP is within median seed noise of the oracle;
- pairwise ordering accuracy beyond seed noise;
- Kendall tau and Spearman rho as secondary statistics;
- top-3 recall;
- selection stability across reference subsets/budgets;
- runtime and hardware;
- dominant diagnostic risk;
- whether the diagnosis matches the candidate's known modified mechanism.

### Prospective success for the Metric-v2 finalist

Success on one radar requires:

- selected candidate within one median seed-noise width of the test oracle;
- lower regret than random expected regret;
- lower regret than the global score;
- actual modified mechanism appears among the two largest reported risk blocks.

Comparison with the proxy must be reported honestly. Do not redefine success if the proxy wins.

A strong result is achieved if the finalist succeeds on both required radars and its average regret is no worse than the proxy by more than half a seed-noise width.

---

## 12. Real-sample requirement analysis

Recompute every metric for 5, 10, 20, and 50 real sequences.

Report:

- selected candidate at each budget/subset;
- agreement with the 50-sequence selection;
- regret of each budget's selection;
- confidence interval width;
- label/error counts;
- smallest budget where the selection is stable across all three subsets.

Do not declare one universal sample requirement. Derive it separately by radar and metric.

Primary intended operating point remains 10 real sequences.

---

## 13. Augmentation-utility check

Synthetic-only and synthetic-plus-real utility are different targets.

After bank-v2 synthetic-only evaluation:

### RADDet

- train every bank-v2 candidate mixed with the fixed 5-real and 10-real sequence sets;
- use three seeds;
- compute:

  `Delta_AP_b = AP(candidate + b real) - AP(b real only)`.

Evaluate whether each metric's ranking predicts `Delta_AP_5` and `Delta_AP_10`.

Do not fit a new learned budget-specific ranker to this small set.

### CARRADA

Run mixed bank-v2 training only if the candidate spread at 5 real sequences exceeds twice seed noise. Otherwise report that candidate choice is practically unimportant at that budget.

---

## 14. Required outputs

### Tables

- metric comparison: regret, within-noise success, pairwise accuracy, runtime;
- prospective bank-v2 AP and each method's selected candidate;
- sample-budget stability;
- augmentation-benefit ranking for RADDet;
- dominant-risk diagnosis per candidate.

### Figures

- bank-v2 candidate AP with selections marked;
- risk profile by candidate: detectability, labels, morphology, support, negatives, processing;
- selection stability versus number of real sequences;
- false-negative and false-positive coverage plots;
- retrospective grouped validation, labelled clearly as development;
- prospective result, visually separated and treated as primary.

### Notes

Update the preregistration note with:

- completed commands;
- deviations;
- bugs/fixes;
- final metrics;
- prospective results;
- honest statement of whether the proxy or a metric won.

---

# 15. Agent work order

Follow this order exactly.

## Phase A - audit and preregister

- [ ] Run current tests.
- [ ] Record HEAD and environment.
- [ ] Create the Metric-v2 preregistration note.
- [ ] Fix bank-v2 parameter ranges and reference indices.
- [ ] Commit before evaluating new metric performance.

## Phase B - implement M3

- [ ] Add robust scales.
- [ ] Add weakest-link aggregation.
- [ ] Add unit tests.
- [ ] Verify that existing M1/M2 outputs remain unchanged.

## Phase C - implement M4

- [ ] Add local CFAR-margin features.
- [ ] Add AP@2 energy-capture features.
- [ ] Add morphology in physical units.
- [ ] Add range/velocity support strata.
- [ ] Add hard-negative adequacy.
- [ ] Add processing diagnostics.
- [ ] Run causal feature checks.

## Phase D - implement M5

- [ ] Extract real training false negatives and false positives.
- [ ] Build candidate positive/negative banks.
- [ ] Implement real-real normalised coverage ratios.
- [ ] Add low-sample confidence handling.

## Phase E - implement M6

- [ ] Build the eight fixed component inputs.
- [ ] Implement non-negative pairwise ranking.
- [ ] Verify family-balanced pairs and saved weights.

## Phase F - retrospective development

- [ ] Run grouped leave-one-family-out.
- [ ] Run leave-one-radar-out.
- [ ] Choose the finalist by the frozen rule.
- [ ] Write and commit `finalist.json`.

## Phase G - bank-v2 freeze

- [ ] Implement/render bank-v2 candidates.
- [ ] Compute every metric.
- [ ] Run all proxy seeds.
- [ ] Save rankings and selections.
- [ ] Commit before full detector training.

## Phase H - prospective full training

- [ ] Train every bank-v2 candidate, three seeds.
- [ ] Evaluate test AP@2/AP@4/recall@2.
- [ ] Compute regret and within-noise success.
- [ ] Do not alter metrics.

## Phase I - sample budgets and augmentation

- [ ] Run 5/10/20/50-sequence metric stability.
- [ ] Run RADDet bank-v2 mixed training at 5 and 10 sequences.
- [ ] Run CARRADA mixed training only if its spread gate passes.

## Phase J - final report

- [ ] Produce all tables and figures.
- [ ] Update the note with results and deviations.
- [ ] State which metric won.
- [ ] State whether it beat, tied, or lost to the proxy.
- [ ] State how many real sequences were actually needed.

---

## 16. Stopping rule

Stop metric development when bank-v2 selections are committed.

Do not add features, create M7, or introduce a deep regressor after seeing bank-v2 outcomes.

Interpret the possible outcomes as follows:

- **M3 wins:** existing probes were adequate; M1 scaling/aggregation was the main problem.
- **M4 wins:** radar-physics and AP-task features improved selection.
- **M5 wins:** coverage of actual real detector errors is more predictive than generic adequacy.
- **M6 wins:** scientific components are useful but require learned relative importance.
- **Proxy wins clearly:** per-candidate task probing is the best selector; retain M4/M5 for diagnosis.
- **None beats global/random:** the few-label reference or current features cannot identify candidate utility reliably.

Every outcome is reportable if this protocol remains frozen.
- Phase D-E: M5 (src/utility_metric_v2.m5) and M6 (PairwiseRanker, family-balanced pair weights, saved
  weights) implemented and unit-tested (tests/test_utility_metric_v2.py, 22 tests).
- Phase H-J done 2026-11-07 09:05: bank v2 trained (48 runs), RADDet augmentation (48 runs), reports, figures,
  verdict below.
- Phase G (bank-v2 freeze) committed 2026-11-07 ~07:50: renders, features, proxy (3 seeds), scores, rankings, selections; freeze-check: no bank-v2 full-detector result.
- Phase F done 2026-11-07 07:20 (results/metric_v2/scores/development.json, results/metric_v2/finalist.json).
