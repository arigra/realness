"""Sequence caches: how maps and labels are stored, written and loaded.

A cache directory holds, per split (train / val / test):
  <split>_x.npy        (n, 8, R, D) float16 dB maps, memory-mapped
  <split>_labels.pt    traj (n, M, 8, 2) range/Doppler bins of each object,
                       present (n, M, 8) bool, meta (one dict per sequence)
  stats.pt             mean/std of the training maps
Grids: RADIal 512 x 256, CARRADA 256 x 64. Doppler is circular on both.
Every source is normalised by its own statistics.
"""
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

MAX_OBJECTS = 8


def load(cache, split, limit=None):
    """The first `limit` sequences of a split, as a list of items in dB."""
    cache = Path(cache)
    x = np.load(cache / f"{split}_x.npy", mmap_mode="r")
    labels = torch.load(cache / f"{split}_labels.pt")
    stats = torch.load(cache / "stats.pt")
    n = len(x) if limit is None else min(limit, len(x))
    return [{"x": torch.from_numpy(np.array(x[i], dtype=np.float16)),
             "traj": labels["traj"][i], "present": labels["present"][i], "stats": stats,
             "recording": labels["meta"][i].get("recording")} for i in range(n)]


def normalised(item, t):
    return (item["x"][t].float() - item["stats"]["mean"]) / item["stats"]["std"]


def pack_labels(frame_labels, seq_len, slot_by_rank=False):
    """Per-frame label lists -> traj (M, L, 2), present (M, L), one slot per object id.

    Slots follow the object's id, not its position in each frame's list: when
    one object leaves the view the others would otherwise slide into its slot.
    RADIal ids are small integers and are the slot (moving vehicles first);
    with slot_by_rank (CARRADA) the sorted ids fill the slots in order.
    """
    ids = sorted({lab["id"] for fl in frame_labels for lab in fl})
    slot = ({i: m for m, i in enumerate(ids[:MAX_OBJECTS])} if slot_by_rank
            else {i: i for i in ids if i < MAX_OBJECTS})
    traj = torch.zeros(MAX_OBJECTS, seq_len, 2)
    present = torch.zeros(MAX_OBJECTS, seq_len, dtype=torch.bool)
    for t, frame in enumerate(frame_labels):
        for lab in frame:
            if lab["id"] in slot:
                m = slot[lab["id"]]
                traj[m, t, 0] = lab["range_bin"]
                traj[m, t, 1] = lab["doppler_bin"]
                present[m, t] = True
    return traj, present


def write_cache(out, n, make, grid, stats_every, slot_by_rank=False):
    """Write a training split of n sequences: make(i) -> (x (8, *grid) dB, frame labels, meta).

    stats_every: mean/std come from every stats_every-th sequence (CARRADA
    caches used all of them, RADIal every 4th, for memory)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    x = open_memmap(out / "train_x.npy", mode="w+", dtype=np.float16, shape=(n, 8, *grid))
    trajs, presents, metas = [], [], []
    for i in range(n):
        xi, labels, meta = make(i)
        x[i] = xi.astype(np.float16)
        t, p = pack_labels(labels, 8, slot_by_rank)
        trajs.append(t); presents.append(p); metas.append(meta)
    x.flush()
    torch.save({"traj": torch.stack(trajs), "present": torch.stack(presents), "meta": metas},
               out / "train_labels.pt")
    sample = np.asarray(x[::stats_every], dtype=np.float32)
    torch.save({"mean": float(sample.mean()), "std": float(sample.std())}, out / "stats.pt")
