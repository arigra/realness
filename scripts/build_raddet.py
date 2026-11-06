"""The real RADDet cache, in the format of the other datasets' (src/data.py),
on the scene-disjoint split (src/raddet.scene_groups, scripts/split_raddet.py).

Frames are grouped by 8 in time order within a scene. train: 50 groups spread
evenly over the training scenes; val and test: every group of their scenes.
Objects past MAX_OBJECTS in a frame are dropped (frames hold up to 14; few have
more than 8).

    PYTHONPATH=. python3 scripts/build_raddet.py data/raddet_scenesplit
"""
import sys

import numpy as np

from src import raddet
from src.data import write_cache


def main(out):
    train = raddet.scene_groups("train")
    splits = {"train": [train[int(i)] for i in np.linspace(0, len(train) - 1, 50).round()],
              "val": raddet.scene_groups("val"), "test": raddet.scene_groups("test")}
    for name, seqs in splits.items():
        def make(i, seqs=seqs):
            fs = seqs[i]
            x = np.stack([raddet.rd_db(o, p, f) for o, p, f in fs])
            return x, [raddet.labels(o, p, f) for o, p, f in fs], {"recording": "scene", "samples": [(o, f) for o, _, f in fs]}
        write_cache(out, len(seqs), make, (raddet.N_RANGE, raddet.N_DOPPLER), 1,
                    slot_by_rank=True, split=name)
        print(name, len(seqs), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
