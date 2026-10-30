# realness

**Question.** Can a cheap measure of how realistic synthetic radar data looks
predict how much it helps a detector on real data, in the setting where there
is too little labelled real data to check that directly?

Before any realism measure can be judged, the thing it must predict has to
move: a detector trained on worse synthetic data must score worse on real
test data. Study 1 checks that.

## Study 1: does detector AP respond to damaged synthetic data?

200 simulator sequences, and versions of the same scenes with one controlled
defect each (weaker targets, missing objects, unlabelled objects, smoothed
texture, more noise, more/less clutter, misplaced labels). A detector is
trained on each version, alone ("only") and mixed with 50 real sequences
("mix"), 3 seeds, and scored on real test data. CARRADA and RADIal.

- Pre-registrations and results: `docs/notes/2026-10-27-damage-spread.md` (first
  attempt, failed), `docs/notes/2026-10-28-damage-spread-fair.md` (fair defects).
- Everything visual (maps, labels, each defect, results, detections):
  `notebooks/damage_study.ipynb`.

## Code

| file | what it is |
|---|---|
| `src/radar.py` | the RADIal radar's published specification |
| `src/carrada_sim.py`, `src/radial_sim.py` | the two simulators, from published specs; nothing fitted to data |
| `src/damage.py` | the defects: label edits, smoothing, and simulators that re-render a scene with one change |
| `src/versions.py` | the datasets, the versions in each suite, and how each version is made |
| `src/data.py` | cache format: loading, writing, label packing |
| `src/detector.py` | the frame detector, peak picking, AP and recall |
| `src/train.py` | training batches, early-stopped training, test detections |
| `src/plots.py` | the notebook's figures |
| `scripts/damage_study.py` | trains and scores every version, applies the pre-registered rule |
| `scripts/run_study.sh`, `scripts/run_study.sbatch` | run or resume a study, here or on the cluster |
| `scripts/write_manifest.py` | records the exact code and data behind a run |

Results: `results/<suite>_<dataset>_runs.json` (every run's scores) and
`_verdict.json` (the rule applied); test detections in `_dets/` (local only).

## Data

`data/` holds links to caches built in radSeq (git 5d1729e): the clean
simulator sequences, the 50 real training sequences with their real val and
test splits, and the damaged versions (re-rendered on first use).
The simulators, detector and training loop were copied from radSeq and give
identical results (checked on data, weights, detections and scores).

## Run

    PYTHONPATH=. python3 -m pytest tests -q
    bash scripts/run_study.sh fair radial         # from ece-hpc: sbatch scripts/run_study.sbatch fair radial
