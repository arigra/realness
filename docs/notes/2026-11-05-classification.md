# Classification study (pre-registered 2026-11-05, before any run)

## Question
Detection tolerated most differences between synthetic and real data except labels and target
strength (docs/notes/2026-11-02-study2b.md). Class depends on the target's signature (extent,
micro-Doppler, shape), where the simulators are crudest. Does classification depend more on how
targets look than detection did?

## Setting (Ari's choices, 2026-11-05)
- **Crops, not maps.** The classifier sees the range-Doppler patch around each labelled object,
  centred on the label: classes differ in typical speed and range, and a classifier on whole maps
  could learn that instead of the signature. The static radars' zero-Doppler ridge is blanked inside
  crops (it would reveal a slow object's speed) and every crop is standardised (its level follows
  range). Crops: 25 range x 17 Doppler cells (src/crops.py).
- **Classes.** CARRADA: pedestrian / cyclist / car. RADDet's six merged into the simulator's three
  models: person; bicycle + motorcycle; car + bus + truck. RADIal (vehicles only) is out.
- **Classifier.** Small CNN, class-balanced batches, Doppler-flip augmentation, early-stopped on crops
  of 32 real val sequences; 3 seeds (src/classify.py).
- **Metrics.** Balanced accuracy (mean recall over classes; the real test sets are imbalanced),
  primary; macro F1 and the confusion matrix reported. Whole real val decides, test is reported.

## Versions (same 200 scenes, seeds SEED0 + i, as the detection study)
| version | what |
|---|---|
| spec | the spec simulator |
| tuned | the best tuned simulator of the detection study |
| micro 0.5, 0 | limbs and wheels move 0.5x / not at all relative to the body |
| blur 1.5, 2, 3 | target response wider |
| tail 2, 4, 8 | target power smeared away from the radar (cells) |
| contrast -6, -12, -20 | targets weaker (dB) |
| real | the 50 real training sequences (reference) |

## Controls
- **Shortcut baseline:** a class-balanced logistic regression on (|radial speed|, range) of each label,
  trained on the real training labels, and on the spec and tuned simulators' labels, scored on real test:
  what the hidden information would have been worth.

## Readout
Balanced accuracy per version (mean of 3 seeds, seed range). The tuned-vs-spec gap and the tolerance
of each family, read as in Study 2 (onset = mildest level more than the seed noise below tuned that
stays so). Comparison with detection: families that cost classification earlier than detection.

## Results (2026-11-06)

First run (crops with the zero-Doppler columns set to a constant) leaked speed: the flat stripe's
position inside a slow target's crop is its speed. Archived in results/round1/*_constant_ridge.json;
the ridge is now filled with background samples (src/crops.py MODE). Numbers below are that rerun.

Balanced accuracy on real test, mean of 3 seeds [range] (chance 0.33):

| version | CARRADA | RADDet |
|---|---|---|
| real (50 seqs) | 0.60 [0.58, 0.62] | 0.64 [0.63, 0.65] |
| spec | 0.46 [0.45, 0.46] | 0.47 [0.42, 0.51] |
| tuned | 0.37 [0.37, 0.38] | 0.55 [0.53, 0.58] |
| micro 0.5 / 0 | 0.38 / 0.35 | 0.53 / 0.53 |
| blur 1.5 / 2 / 3 | 0.46 / 0.49 / 0.50 | 0.60 / 0.53 / 0.43 |
| tail 2 / 4 / 8 | 0.48 / 0.49 / 0.47 | 0.58 / 0.51 / 0.46 |
| contrast -6 / -12 / -20 | 0.36 / 0.40 / 0.37 | 0.51 / 0.54 / 0.38 |
| shortcut (speed, range), real labels | 0.62 | 0.60 |
| shortcut, spec / tuned labels | 0.55 / 0.61 | 0.65 / 0.58 |

Onsets (rule above; noise = median seed range: CARRADA 0.035, RADDet 0.056):
- **CARRADA: no readout.** The tuned simulator (best for detection) is at chance, below spec, so no
  defect can lower it; blur and tail *raise* it (to 0.50). The detection-tuned simulator's targets are
  wrong for classification here.
- **RADDet: blur 3, tail 8, contrast -20 dB; micro none** (val agrees, micro 0 on val only). Detection
  onsets were blur 2, tail 2, contrast -12: **classification tolerated more than detection**, the
  opposite of the question's expectation.
- Limb and wheel motion (micro) is not used by these classifiers on either dataset.

Controls (results/crop_controls.json, scripts/crop_controls.py): with the target erased (crop taken
40 rows away, ridge filled with noise) classifiers trained on real data still score 0.45 (CARRADA) and
0.50 (RADDet), well above chance: much of the class information is context, not signature. The
real-trained CNN does not beat the speed + range shortcut on CARRADA (0.60 vs 0.62) and is
only 0.04 above it on RADDet.

Reading: with crops this size and this classifier, real-data classification is mostly context plus
kinematics, so a test of "does target signature matter" is weak. A stronger test would need crops
with no context left (smaller, or masked to the labelled box) and an erased-crop score near chance first.
