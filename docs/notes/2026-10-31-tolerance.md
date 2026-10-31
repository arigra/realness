# Tolerance curves (pre-registered 2026-10-31, before any run)

## Question
Study 1 (2026-10-28) asked whether AP responds to damage at all, at three
strengths per defect. Some defects did nothing at those strengths. Here every
defect is pushed from barely visible to extreme, to find where it starts to
cost: its **tolerance**. A real-vs-synthetic gap inside a property's tolerance
is harmless; one outside it is worth fixing. Ari's choices: all three datasets
(CARRADA, RADIal, RADDet), RADDet with its own simulator built from its spec.

## Settings
Synthetic only (the 50-real mix showed nothing on CARRADA or RADIal), 3 seeds,
same detector, training, early stopping and scores as Study 1 (AP@2 primary,
AP@4, recall@2), same 200 clean simulator sequences per dataset.

## Families (mild -> extreme; a version changes one thing)
| family | levels | how |
|---|---|---|
| blur | target response 1.25 1.5 2 3 5 8 x wider | re-render, range and Doppler |
| sidelobes | -10 +5 +10 +15 +20 dB | re-render, window sidelobe level |
| smoothing | 5 10 25 50 75 90% | on the maps, objects included |
| contrast | -3 -6 -12 -20 -30 dB | re-render, targets weaker |
| noise | +3 +6 +12 +20 +30 dB | re-render |
| clutter | -20 -12 -6 +6 +12 +20 dB | re-render |
| unrender | 10 25 50 75 90% | re-render, labels kept |
| extra | 10 25 50 100 200% | re-render, unlabelled objects |
| range shift | 1 2 3 5 8 cells | on the labels |
| Doppler shift | 1 2 3 5 8 cells | on the labels |

Smoothing replaces Study 1's version, which left a box around each label
unsmoothed (a cue the detector could use).

## Readout
Per family, mean AP (3 seeds) against strength. noise = median seed range over
all versions of the dataset. **Onset** = the mildest level whose mean is more
than noise below clean and stays so at every stronger level; none if no level
qualifies. Reported per family and dataset; no pass/fail.

## RADDet
TI AWR1843 like CARRADA; spec from the RADDet README: 256 range bins to 50 m
(0.195 m/bin), 64 Doppler bins (0.42 m/s/bin), stationary radar, 6 classes
(person, bicycle, car, motorcycle, bus, truck). The simulator is the CARRADA
simulator with this spec and a RADDet road-user scenario taken from the paper's
statistics, written before any comparison with RADDet maps. Real data: RD maps
from the complex RAD cubes (power summed over azimuth), label = box centre in
range and Doppler. The test split is RADDet's own; val comes from its train
split. Only synthetic data trains, so the known train/test overlap of RADDet
frames does not reach the detector.
