# realness

Can an inexpensive realism measure predict the real-test detector utility of
synthetic radar range-Doppler sequences, when there is too little labelled real
data to check utility directly?

Split out of radSeq (at radSeq 5d1729e, 2026-10-30). The simulators, detector and
data loaders are copies, so changes in radSeq cannot change results here.

## Layout
- `src/damage.py`: controlled defects (label defects; map defects re-render the same simulator scene).
- `src/damage_viz.py`: the plots of `notebooks/damage_study.ipynb`.
- `scripts/damage_study.py`: train the frame detector on each version and score it on real test data.
- `src/carrada_sim.py`, `src/scene_sim.py`: CARRADA and RADIal simulators (from specs, never fitted to data).
- `src/detector*.py`, `scripts/radial_detector_*.py`: detector and training loop.
- `docs/notes/`: pre-registrations and results, one note per study.
- `data/`: symlinks to the radSeq caches (simulator, 50 real sequences with val/test, damaged versions).

## Studies
1. Does detector AP respond to controlled defects? `docs/notes/2026-10-27-damage-spread.md` (v1, failed),
   `docs/notes/2026-10-28-damage-spread-fair.md` (fair defects; CARRADA done, RADIal running).

## Run
    PYTHONPATH=. python3 -m pytest tests -q
    bash scripts/run_damage_radial.sh          # resumable; or from ece-hpc: sbatch scripts/damage_radial.sbatch
