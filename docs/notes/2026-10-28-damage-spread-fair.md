# Damage spread, fair defects (pre-registered 2026-10-28)

Rerun of docs/notes/2026-10-27-damage-spread.md, which failed. Ari's two
objections: the map defects were not fair, and CARRADA may be too easy. Checks
afterwards agreed with the first: -10 dB still left targets 12.7 dB over the
background, CARRADA's static clutter sits only on the zero-Doppler column, and
blinking kept map and label consistent. New defects and scoring proposed by
Claude, accepted by Ari ("do it"), 2026-10-28; this note covers CARRADA, RADIal
follows. Written before any result.

## Versions (src/damage.py, src/versions.py, `scripts/damage_study.py --suite fair`)
Same 200 simulator sequences, same seeds; re-rendered defects reproduce the
clean scene except for the change (tests/test_damage.py).

| family | levels | what changes | measured (clean -> worst) |
|---|---|---|---|
| contrast | -6, -12, -20 dB | moving objects' RCS | target over map median 20.9 -> 7.7 dB; hit 0.99 -> 0.85 |
| unrendered | 10, 25, 50% | (object, frame) removed from the map, **label kept** | prominence 20.9 -> 10.9 dB |
| extra | 10, 25, 50% of sequences | one extra moving object, **no label** | off-ridge 6 dB cells/frame 17.9 -> 27.3 |
| smoothing | 25, 50, 75% | every cell pulled to its 4-neighbour mean, except +/-3 x +/-2 cells around labels | speckle 1.86 -> 1.28 / 0.79 / 0.66 dB |
| noise | +6, +12 dB | receiver noise figure | prominence 20.9 -> 9.5 dB |
| clutter (reported) | +/-6, +/-12 dB | ground and pole RCS | only the zero-Doppler ridge changes (static radar) |
| range (control) | 1, 2 bins | labels moved | -- |

Hit = peak 3 dB over its ring within +/-2 cells; pure noise passes this ~half
the time, so it understates removal (unrendered 50%: 0.79).
For scale: the RADIal DiT's speckle was 36% under real (smoothing 25% ~ that).

## Training and test
As the 2026-10-27 note: frame detector, settings **only** and **mix** (50 real),
seeds 1-3, CARRADA official test. Detections kept (results/fair_carrada_dets).

## Scores
- **AP within 2 bins**: primary, as before.
- AP within 4 bins and recall within 2 bins: reported with the same rule, so
  that placement precision does not hide missing or invented targets.

## Rule (unchanged)
noise = median seed range over versions; pass = best-minus-worst mean >= 3 x
noise and >= 3 ordered families among the five counted (contrast, unrendered,
extra, smoothing, noise). Clutter (two-sided) and range (control) are reported
only. Applied to "only"; "mix" reported.

## What each outcome means
- Pass: AP responds to fair, graded map defects; Step 2 (realism metrics on
  these versions) has something to predict.
- Fail again with these defects: on CARRADA the detector is insensitive to map
  realism short of misplaced labels; RADIal decides whether that is CARRADA.
- Only AP@4 or recall pass: the strict 2-bin AP hides target-level defects; the
  primary score should change (Ari's decision).

## Results
2026-10-28 ~02:40; results/fair_carrada_runs.json, results/fair_carrada_verdict.json.
**Primary (AP@2, only): FAIL** -- gap 0.371 = 6.9 x noise 0.054, but 2/5 counted
families ordered (extra, smoothing). AP@4 only passes (gap 7.5 x noise 0.077;
contrast, extra, smoothing), contrast only just (clean - worst 0.079 vs noise
0.077). Recall@2 only fails (1/5). Mix fails on every score (gap <= 1.6 x noise).

Mean test scores, 3 seeds, only / mix:

| version | AP@2 | AP@4 | recall@2 |
|---|---|---|---|
| clean | 0.295 / 0.441 | 0.463 / 0.607 | 0.677 / 0.774 |
| contrast -6/-12/-20 | 0.301 0.234 0.241 / 0.401 0.389 0.405 | 0.473 0.381 0.384 / 0.582 0.550 0.583 | 0.686 0.665 0.640 / 0.738 0.749 0.718 |
| unrendered 10/25/50% | 0.316 0.315 0.341 / 0.403 0.423 0.429 | 0.493 0.494 0.523 / 0.590 0.634 0.607 | 0.678 0.690 0.698 / 0.749 0.729 0.752 |
| extra 10/25/50% | 0.314 0.280 0.209 / 0.428 0.404 0.397 | 0.489 0.440 0.344 / 0.600 0.582 0.585 | 0.671 0.693 0.654 / 0.774 0.750 0.749 |
| smoothing 25/50/75% | 0.120 0.008 0.004 / 0.408 0.425 0.436 | 0.214 0.014 0.008 / 0.586 0.615 0.626 | 0.465 0.145 0.103 / 0.746 0.763 0.759 |
| noise +6/+12 | 0.306 0.293 / 0.430 0.426 | 0.498 0.516 / 0.618 0.604 | 0.663 0.669 / 0.751 0.732 |
| clutter +6/-6/+12/-12 | 0.337 0.375 0.313 0.352 / 0.414 0.425 0.450 0.416 | 0.497 0.582 0.471 0.550 / 0.610 0.613 0.639 0.611 | 0.705 0.685 0.696 0.663 / 0.760 0.738 0.762 0.727 |
| range 1/2 | 0.276 0.055 / 0.423 0.412 | 0.552 0.421 / 0.626 0.626 | 0.605 0.322 / 0.751 0.745 |

- **only**: smoothing is by far the strongest defect -- 25% (about the RADIal
  DiT's speckle deficit) cuts AP@2 0.295 -> 0.120, 50% leaves a useless
  detector. Unlabelled extra objects lower AP gradually (-0.09 at 50%). Weaker
  targets cost ~0.06 from -12 dB. Unrendered targets (label kept) and +12 dB
  noise do nothing; unrendered even sits above clean (within noise).
- **mix**: with 50 real sequences no defect is measurable, including smoothing
  75% (0.436 vs 0.441), which alone gives AP 0.004.
- By the outcomes above: not a clean pass on the primary; AP responds strongly
  to texture (smoothing) and unlabelled objects, weakly to contrast, not at all
  to missing signal under a kept label or to noise. AP@4 passing is the third
  case (Ari's decision on the primary score).

## RADIal (pre-registered 2026-10-28, before any RADIal training)
Ari chose the full suite (both settings, 3 seeds, 126 runs), after the CARRADA
results above; written before any RADIal result. Same 21 versions, rule, scores
and primary (AP@2) as CARRADA, applied to the first 200 RADIal engineer
simulator sequences (data/scene_engineer, re-rendered on the GPU: clean
re-render identical), 50 real sequences of fold 0, its val (32 spread) and its
143 test sequences (data/radial_fold0_n50). Moving vehicles carry the defects;
parked cars are scene. Clutter is every static scatterer; unlike CARRADA it
spreads over the targets' Doppler.

| family | measured (clean -> worst) |
|---|---|
| contrast | target over ring 23.7 -> 18.0 / 12.8 / 6.3 dB |
| unrendered | 23.7 -> 22.7 / 21.0 / 15.0 dB |
| extra | bright (15 dB) cells/frame 1785 -> 1790 / 1794 / 1807 -- weak: frames are busy |
| smoothing | speckle 1.84 -> 1.38 / 0.98 / 0.77 dB |
| noise | target over ring 23.7 -> 18.7 / 13.1 dB |
| clutter | bright cells 1785 -> 2650 / 1169 / 3658 / 846 (+6/-6/+12/-12) |

Run: `bash scripts/run_study.sh fair radial` (resumable; `sbatch
scripts/run_study.sbatch fair radial` continues it). Results:
results/fair_radial_runs.json, results/fair_radial_verdict.json, detections
results/fair_radial_dets.

### RADIal results
2026-10-30 ~20:20; results/fair_radial_runs.json, results/fair_radial_verdict.json.
**Primary (AP@2, only): FAIL** -- gap 0.358 = 4.6 x noise 0.077, but 2/5 counted
families ordered (contrast, smoothing). AP@4 only and recall@2 only fail the
same way (2/5: contrast, smoothing). Mix fails on every score (gap <= 1.9 x
noise; clean is the lowest mix version).

Mean test scores, 3 seeds, only / mix:

| version | AP@2 | AP@4 | recall@2 |
|---|---|---|---|
| clean | 0.312 / 0.804 | 0.438 / 0.882 | 0.677 / 0.895 |
| contrast -6/-12/-20 | 0.204 0.152 0.083 / 0.823 0.841 0.852 | 0.269 0.198 0.102 / 0.890 0.889 0.901 | 0.610 0.502 0.449 / 0.909 0.916 0.910 |
| unrendered 10/25/50% | 0.255 0.280 0.290 / 0.841 0.831 0.838 | 0.360 0.409 0.409 / 0.895 0.889 0.899 | 0.588 0.631 0.605 / 0.905 0.893 0.904 |
| extra 10/25/50% | 0.313 0.214 0.274 / 0.829 0.821 0.845 | 0.428 0.306 0.398 / 0.882 0.883 0.895 | 0.673 0.621 0.679 / 0.895 0.890 0.911 |
| smoothing 25/50/75% | 0.124 0.007 0.015 / 0.822 0.835 0.821 | 0.192 0.012 0.022 / 0.874 0.888 0.875 | 0.514 0.146 0.213 / 0.892 0.898 0.891 |
| noise +6/+12 | 0.234 0.314 / 0.824 0.849 | 0.348 0.449 / 0.875 0.893 | 0.581 0.639 / 0.902 0.904 |
| clutter +6/-6/+12/-12 | 0.269 0.353 0.329 0.252 / 0.808 0.824 0.827 0.823 | 0.349 0.493 0.400 0.346 / 0.866 0.887 0.886 0.881 | 0.644 0.634 0.711 0.569 / 0.900 0.888 0.891 0.904 |
| range 1/2 | 0.365 0.341 / 0.822 0.838 | 0.389 0.450 / 0.885 0.886 | 0.706 0.628 / 0.882 0.905 |

- **only**: smoothing again the strongest defect (25%: 0.312 -> 0.124; 50%
  leaves a useless detector). Unlike CARRADA, weaker targets cost a lot and in
  order (-20 dB: 0.083). Extra and unrendered objects stay within noise (extra
  was a weak defect here: +22 bright cells on 1785). Range shifts of 1-2 cells
  do nothing (CARRADA range 2: 0.055).
- contrast -12 and noise +12 leave the same target-over-ring (12.8 vs 13.1 dB)
  but AP 0.152 vs 0.314: what costs is targets sinking towards the scene's other
  scatterers, not towards the noise floor (a reading, untested).
- **mix**: nothing measurable, as on CARRADA; smoothing 50% gives 0.835 in the
  mix and 0.007 alone.
- Across both datasets only smoothing orders on every score; the second ordered
  family differs (CARRADA extra, RADIal contrast).
