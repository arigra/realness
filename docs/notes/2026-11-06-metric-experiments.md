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

## Metric variants (Phase 2) — *TBD (before Phase 3 evaluation)*
Planned: the existing global realism score (results/score_frozen.json), global and conditional
target-patch energy distance, support only, label agreement only, the five components unweighted,
a domain classifier (diagnostic); tolerance (dead-zone) penalties calibrated on development
datasets/families only and frozen in `results/metric/frozen_metric.json`; scalar readouts: equal
sum, non-negative linear / rank regression. No high-capacity model.

## Validation (Phase 3) — *TBD details (before Phase 3)*
Leave-one-family-out within a radar; leave-one-radar-out (frozen before the held-out radar's outcomes
are read). Primary: top-1 regret (oracle AP@2 minus selected AP@2), P(selected within the dataset's
seed noise of the oracle), pairwise ordering accuracy. Secondary: Kendall tau, Spearman rho, top-3
recall, runtime, stability across reference subsets. Bootstrap by family for rank summaries, by real
sequence for reference stability; paired seeds for AP comparisons.

Decision gate: if held-out-family ranking is not better than both the existing global score and
random choice, stop before any learned score; report which probe failed and the vector as diagnosis.

## Proxy detector (Phase 4) — configuration fixed here
FrameDetector(width=8), 50 synthetic sequences per candidate, 1,000 steps, no early stopping, 1 seed
for the sweep then 3 for the top candidates, scored on 5 labelled real training sequences over several
fixed 5-sequence subsets. Record proxy AP@2, wall-clock and GPU time, ranking metrics, top-1 regret.

## Augmentation benefit (Phase 5) — *TBD (results/metric/augmentation_candidates.json, before runs)*

## Prospective selection (Phase 6) — *TBD (freeze point, before generating final candidates)*

## Stopping rules
- Phase 1 stops if a controlled intervention moves none of its intended probes: that probe is fixed
  or the factor is marked unmeasured; it is never declared irrelevant.
- Phase 3 decision gate above.
- No classification-metric work until Phase 8's gate passes.
