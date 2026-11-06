"""Tolerance readout of Study 2 and 2b (pre-registered: docs/notes/2026-10-31-tolerance.md,
docs/notes/2026-11-02-study2b.md).

noise = median seed range (max - min of the 3 seeds) over every version of the dataset
(Study 2 and 2b together). Onset of a family = the mildest level whose mean is more than
noise below clean and stays so at every stronger level; none if no level qualifies.
Families that go two ways are read per direction: clutter -/+, sidelobes (higher only;
-10 dB lower sidelobes is reported apart), confusers fewer/more.
"""
import json
from pathlib import Path

import numpy as np

FAMILIES = {          # family -> versions in increasing damage
    "blur": ["blur1.25", "blur1.5", "blur2", "blur3", "blur5", "blur8"],
    "sidelobes+": ["sidelobes+5", "sidelobes+10", "sidelobes+15", "sidelobes+20"],
    "smoothing": [f"smoothing{p}" for p in (5, 10, 25, 50, 75, 90)],
    "contrast": [f"contrast-{d}" for d in (3, 6, 12, 20, 30)],
    "noise": [f"noise+{d}" for d in (3, 6, 12, 20, 30)],
    "clutter-": ["clutter-6", "clutter-12", "clutter-20"],
    "clutter+": ["clutter+6", "clutter+12", "clutter+20"],
    "unrender": [f"unrender{p}" for p in (10, 25, 50, 75, 90)],
    "extra": [f"extra{p}" for p in (10, 25, 50, 100, 200)],
    "rangeshift": [f"rangeshift{b}" for b in (1, 2, 3, 5, 8)],
    "dopplershift": [f"dopplershift{b}" for b in (1, 2, 3, 5, 8)],
    "support": [f"support{p}" for p in (25, 50, 75, 100)],
    "annotation": [f"annotation+{d}" for d in (6, 12, 18)],
    "tail": [f"tail{c}" for c in (2, 4, 8, 16)],
    "confusers-": ["confusers0.5", "confusers0"],
    "confusers+": ["confusers2", "confusers4"],
}
EXTRA = ["sidelobes-10"]


def means(dataset, split, metric):
    """{version: (mean, seed range)} over study2 and study2b; split 'test' or 'val' (whole val)."""
    out = {}
    for suite in ("study2", "study2b"):
        p = Path(f"results/{suite}_{dataset}_runs.json")
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        runs = (r["runs"] if split == "test" else r["val"])["only"]
        for v, seeds in runs.items():
            if len(seeds) == 3 and (v not in out or suite == "study2"):
                x = [s[metric] for s in seeds.values()]
                out[v] = (float(np.mean(x)), float(np.ptp(x)))
    return out


def onset(clean, levels, noise):
    below = [m is not None and clean - m > noise for m in levels]
    for i in range(len(levels)):
        if all(below[i:]):
            return i
    return None


def readout(dataset, split, metric):
    m = means(dataset, split, metric)
    noise = float(np.median([r for _, r in m.values()]))
    clean = m["clean"][0]
    fams = {}
    for f, names in FAMILIES.items():
        lv = [m[n][0] if n in m else None for n in names]
        if any(x is None for x in lv):
            fams[f] = {"means": dict(zip(names, lv)), "onset": "incomplete"}
            continue
        i = onset(clean, lv, noise)
        fams[f] = {"means": dict(zip(names, lv)), "onset": None if i is None else names[i]}
    return {"clean": clean, "noise": noise, "families": fams,
            "apart": {n: m[n][0] for n in EXTRA if n in m}}
