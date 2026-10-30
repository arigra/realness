"""The versions of the synthetic training data: clean and each defect.

A version is the first N_SIM sequences of a simulator cache, either as cached
(clean), with a defect applied to the cached maps or labels, or re-rendered
from the same seeds with one physical change (written once to its own cache).
"""
import dataclasses
import json
from pathlib import Path

import torch

from src.damage import (DamagedCarradaSimulator, DamagedSceneSimulator, clutter_scenario,
                        drop_labels, noisy_spec, shift_labels, smooth_background)
from src.data import load, write_cache
from src.radial_sim import Scenario

N_SIM, SEED0 = 200, 10_000_000      # sequences per version; the simulator cache's seeds


@dataclasses.dataclass(frozen=True)
class Dataset:
    sim: str            # clean simulator cache
    real: str           # 50 real training sequences, plus the real val and test splits
    damage: str         # where re-rendered versions are written
    grid: tuple         # range x Doppler bins
    stats_every: int    # re-rendered stats from every n-th sequence, as the clean cache's


DATASETS = {
    "carrada": Dataset("data/carrada_sim_engineer", "data/carrada_seq_n50", "data/carrada_damage", (256, 64), 1),
    "radial": Dataset("data/scene_engineer", "data/radial_fold0_n50", "data/radial_damage", (512, 256), 4),
}

# name -> (defect, strength)
V1 = {"clean": ("clean", None)}
V1 |= {f"drop{p}": ("drop", p / 100) for p in (10, 25, 50, 75)}
V1 |= {f"range{b}": ("range", b) for b in (1, 2, 3)}
V1 |= {f"doppler{b}": ("doppler", b) for b in (1, 2, 3)}
V1 |= {f"blink{p}": ("blink", p / 100) for p in (10, 25, 50)}
V1 |= {f"contrast-{d}": ("contrast", -d) for d in (3, 6, 10)}
V1 |= {"clutter+6": ("clutter", 6.0), "clutter-6": ("clutter", -6.0)}
FAIR = {"clean": ("clean", None)}
FAIR |= {f"contrast-{d}": ("contrast", -d) for d in (6, 12, 20)}
FAIR |= {f"unrender{p}": ("unrender", p / 100) for p in (10, 25, 50)}
FAIR |= {f"extra{p}": ("extra", p / 100) for p in (10, 25, 50)}
FAIR |= {f"smooth{p}": ("smooth", p / 100) for p in (25, 50, 75)}
FAIR |= {f"noise+{d}": ("noise", d) for d in (6, 12)}
FAIR |= {f"clutter{d:+d}": ("clutter", float(d)) for d in (6, -6, 12, -12)}
FAIR |= {f"range{b}": ("range", b) for b in (1, 2)}
# suite -> (versions, families counted by the rule, families reported only);
# a family is a defect, its versions in increasing damage after clean
SUITES = {"v1": (V1, ("drop", "range", "doppler", "blink", "contrast"), ("clutter",)),
          "fair": (FAIR, ("contrast", "unrender", "extra", "smooth", "noise"), ("clutter", "range"))}
ON_CACHE = ("clean", "drop", "range", "doppler", "smooth")      # the rest are re-rendered


def family(versions, fam):
    return ["clean"] + [n for n in versions if n.startswith(fam)]


def _seeds(i):
    return {"generator": torch.Generator().manual_seed(SEED0 + i),
            "blink_generator": torch.Generator().manual_seed(i)}


def _carrada_sequence(kind, value, i):
    kw = _seeds(i)
    if kind == "contrast":
        kw["target_offset_db"] = value
    elif kind == "noise":
        kw["spec"] = noisy_spec(value)
    elif kind == "clutter":
        kw["scenario"] = clutter_scenario(value)
    else:                                       # blink, unrender, extra
        kw[f"{kind}_frac"] = value
    o = DamagedCarradaSimulator(**kw).gen_sequence()
    return o["x"].numpy(), o["labels"], {"classes": o["classes"]}


def _radial_sequence(kind, value, i):
    # on the GPU: the device the clean cache was built on, so its scene is reproduced exactly
    kw = _seeds(i) | {"device": torch.device("cuda")}
    if kind == "contrast":
        kw["target_offset_db"] = value
    elif kind == "noise":
        kw["scenario"] = dataclasses.replace(Scenario(), snr_offset_db=-value)
    elif kind == "clutter":
        kw["clutter_offset_db"] = value
    else:                                       # unrender, extra
        kw[f"{kind}_frac"] = value
    o = DamagedSceneSimulator(**kw).gen_sequence()
    return o["x"].numpy(), o["labels"], {"road_type": o["road_type"], "ego_speed": o["ego_speed"]}


def render(dataset, name, kind, value):
    """The cache of a re-rendered version, written on first use."""
    ds = DATASETS[dataset]
    out = Path(ds.damage) / name
    if not (out / "DONE").exists():
        seq = _carrada_sequence if dataset == "carrada" else _radial_sequence
        write_cache(out, N_SIM, lambda i: seq(kind, value, i), ds.grid, ds.stats_every,
                    slot_by_rank=dataset == "carrada")
        (out / "DONE").write_text(json.dumps({"kind": kind, "value": value}))
    return out


def version(dataset, name, kind, value, clean, n=N_SIM):
    """The first n training sequences of one version; clean = the clean cache's."""
    if kind not in ON_CACHE:
        return load(render(dataset, name, kind, value), "train", limit=n)
    g = torch.Generator().manual_seed(0)
    out = []
    for it in clean[:n]:
        it = dict(it)
        if kind == "drop":
            it["present"] = drop_labels(it["present"], value, g)
        elif kind == "range":
            it["traj"] = shift_labels(it["traj"], 0, value, it["x"].shape[-2], wrap=False)
        elif kind == "doppler":
            it["traj"] = shift_labels(it["traj"], 1, value, it["x"].shape[-1], wrap=True)
        elif kind == "smooth":
            it["x"] = smooth_background(it["x"].float(), it["traj"], it["present"], value).half()
        out.append(it)
    return out
