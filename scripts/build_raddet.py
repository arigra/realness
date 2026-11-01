"""The real RADDet cache, in the format of the other datasets' (src/data.py).

Frames are grouped by 8 in time order. train: 50 groups and val: 38 groups,
spread over RADDet's train split and disjoint; test: every labelled frame of
its test split (129 groups; see src/raddet.frames). Objects past MAX_OBJECTS in a frame are dropped (frames hold
up to 14; few have more than 8).

    PYTHONPATH=. python3 scripts/build_raddet.py data/raddet_n50
"""
import sys

import numpy as np

from src import raddet
from src.data import write_cache


def groups(fs):
    return [fs[i:i + 8] for i in range(0, len(fs) - 7, 8)]


def main(out):
    train = groups(raddet.frames("train"))
    pick = np.linspace(0, len(train) - 1, 88).round().astype(int)
    to_val = set(np.linspace(0, 87, 38).round().astype(int))
    splits = {"train": ("train", [train[i] for k, i in enumerate(pick) if k not in to_val]),
              "val": ("train", [train[i] for k, i in enumerate(pick) if k in to_val]),
              "test": ("test", groups(raddet.frames("test")))}
    for name, (src, seqs) in splits.items():
        def make(i, src=src, seqs=seqs):
            fs = seqs[i]
            x = np.stack([raddet.rd_db(src, p, f) for p, f in fs])
            return x, [raddet.labels(src, p, f) for p, f in fs], {"recording": src, "samples": [f for _, f in fs]}
        write_cache(out, len(seqs), make, (raddet.N_RANGE, raddet.N_DOPPLER), 1,
                    slot_by_rank=True, split=name)
        print(name, len(seqs), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
