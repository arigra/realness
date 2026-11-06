"""A scene-disjoint RADDet split (the official one interleaves train and test
frames of the same scenes; docs/notes/2026-11-02-simulators.md).

RADDet's frame ids run on one timeline. A scene change shows as a jump in the
static zero-Doppler ridge (the radar is static, so the ridge is the scene):
every 4th labelled frame's ridge profile (dB minus its median, rows 10-229) is
correlated with the next; correlation < 0.5 marks a cut. Whole scenes then go,
in time order, to test / val / train in a repeating pattern sized to about
28% / 12% / 60% of frames. Writes results/raddet_split.json:
{"cuts": [first frame id of each new scene], "scenes": [[split, first id, last id], ...]}.

    PYTHONPATH=. python3 scripts/split_raddet.py
"""
import json
from pathlib import Path

import numpy as np

from src import raddet

OUT = Path("results/raddet_split.json")


def labelled():
    """All labelled frames on disk, in time order: [(id, official split, part, frame)]."""
    return sorted([(int(f), "train", p, f) for p, f in raddet.frames("train")]
                  + [(int(f), "test", p, f) for p, f in raddet.frames("test")])


def main():
    frames = labelled()
    sel = frames[::4]
    z = np.stack([raddet.rd_db(s, p, f)[:, 32] for _, s, p, f in sel])
    z = z - np.median(z, 1, keepdims=True)
    corr = [np.corrcoef(z[i, 10:230], z[i + 1, 10:230])[0, 1] for i in range(len(z) - 1)]
    cuts = [sel[i + 1][0] for i, c in enumerate(corr) if c < 0.5]
    bounds = [frames[0][0]] + cuts + [frames[-1][0] + 1]
    ids = np.array([f[0] for f in frames])
    scenes = [(a, b - 1, int(((ids >= a) & (ids < b)).sum())) for a, b in zip(bounds, bounds[1:])]
    # deal scenes in time order to the split least filled relative to its share
    share, have, out = {"test": 0.28, "val": 0.12, "train": 0.60}, {"test": 0, "val": 0, "train": 0}, []
    for a, b, n in scenes:
        split = min(share, key=lambda s: have[s] / share[s])
        have[split] += n
        out.append([split, int(a), int(b)])
    OUT.write_text(json.dumps({"cuts": [int(c) for c in cuts], "scenes": out,
                               "frames": {k: int(v) for k, v in have.items()}}, indent=1))
    print(have, len(scenes), "scenes")


if __name__ == "__main__":
    main()
