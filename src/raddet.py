"""RADDet real data: range-Doppler maps and labels from the released RAD cubes.

Conventions (measured 2026-11-01 on the files and RADDet's own code):
- RAD cube: 256 range x 256 angle x 64 Doppler, complex64. RD map = power
  summed over angle, in dB.
- Range: in the cube row 0 is far (RADDet util/helper.py: range = (255 - row) x
  0.195 m). Maps and labels here are flipped so row 0 is near, as in CARRADA.
- Doppler: zero at bin 32, approaching objects above it (label motion: +0.32
  range rows per frame per Doppler bin towards the radar, i.e. ~0.15 s frames).
- Labels: per object a box [range, angle, Doppler centre, sizes]; the label is
  the box centre. No instance ids, so objects are numbered within each frame.
The official split interleaves train and test frames; only synthetic data
trains in the studies, so that overlap does not reach the detector.
"""
import pickle
from pathlib import Path

import numpy as np

ROOT = Path("/truenas/home/arigra/permuter/ariGranevich/radar data/RADDet/raw")
N_RANGE, N_DOPPLER = 256, 64


def frames(split):
    """The official split's labelled frames in time order: [(part, frame id)].
    On disk test/gt/part_2 holds 1 of its 1,000 label files, so the test split
    here is 1,032 of RADDet's 2,032 frames."""
    paths = (ROOT / split / "gt").glob("part_*/*.pickle")
    return sorted(((p.parent.name, p.stem) for p in paths), key=lambda f: int(f[1]))


def rd_db(split, part, frame):
    rad = np.load(ROOT / split / "RAD" / part / f"{frame}.npy")
    power = (np.abs(rad) ** 2).sum(axis=1)
    return 10 * np.log10(power[::-1] + 1e-12)


def labels(split, part, frame):
    gt = pickle.load(open(ROOT / split / "gt" / part / f"{frame}.pickle", "rb"))
    return [{"id": k, "cls": c, "range_bin": N_RANGE - 1 - float(b[0]), "doppler_bin": float(b[2])}
            for k, (c, b) in enumerate(zip(gt["classes"], gt["boxes"]))]
