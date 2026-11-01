"""The clean RADDet simulator cache: N_SIM sequences, seeds SEED0 + i, as the
other datasets' simulator caches (src/versions.py).

    PYTHONPATH=. python3 scripts/build_raddet_sim.py data/raddet_sim
"""
import sys

import torch

from src.data import write_cache
from src.raddet_sim import RaddetSimulator
from src.versions import N_SIM, SEED0


def make(i):
    o = RaddetSimulator(generator=torch.Generator().manual_seed(SEED0 + i)).gen_sequence()
    return o["x"].numpy(), o["labels"], {"classes": o["classes"]}


if __name__ == "__main__":
    write_cache(sys.argv[1], N_SIM, make, (256, 64), 1, slot_by_rank=True)
