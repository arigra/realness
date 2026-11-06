"""The versions of the synthetic training data: clean and each defect.

A version is the first N_SIM sequences of a simulator cache, either as cached
(clean), with a defect applied to the cached maps or labels, or re-rendered
from the same seeds with one physical change (written once to its own cache).
"""
import dataclasses
import hashlib
import json
from pathlib import Path

import torch

from src.damage import (DamagedCarradaSimulator, DamagedSceneSimulator, RadialDamage, TrackDamage,
                        clutter_scenario, drop_labels, noisy_spec, shift_labels, smooth_background)
from src.data import load, write_cache
from src.raddet_tuned import TunedRaddetSimulator, range_gain
from src.radial_sim import Scenario
from src.tuned import TUNING, simulator

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
    "raddet": Dataset("data/raddet_sim", "data/raddet_scenesplit", "data/raddet_damage", (256, 64), 1),
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
# Study 1's smoothing left the cells around labels unsmoothed; this one smooths
# whole maps (docs/notes/2026-11-01-realness-score.md). Its clean runs are the
# fair suite's (same clean pool, seeds and training).
SMOOTHALL = {"clean": ("clean", None)} | {f"smoothall{p}": ("smoothall", p / 100) for p in (25, 50)}
# RADDet simulator tuned to the real maps, one fix per version and all together
# (src/raddet_tuned.py; docs/notes/2026-11-01-realness-score.md, step 6). The
# contrast offsets (target dB, static dB) are fitted per version and read from
# results/tuning/raddet_contrast_fit.json ("alone": no other fix, "all": with them).
TUNE = {"clean": ("clean", None),
        "processing": ("tuned", {"processing": True}),
        "rangegain": ("tuned", {"range_gain": True}),
        "contrast": ("tuned", {"contrast": "alone"}),
        "allfixes": ("tuned", {"processing": True, "range_gain": True, "contrast": "all"})}
# The tuned simulators (src/tuned.py): every fix on, and every fix left out in
# turn (RADIal has no processing or label fix, CARRADA and RADDet no body fix:
# leave those versions out with --versions).
FIXES = ("processing", "range_gain", "scenes", "label", "contrast", "body", "clutter_profile")
SIM2 = {"clean": ("clean", None), "all": ("tuned2", FIXES)}
SIM2 |= {f"all-{f}": ("tuned2", tuple(x for x in FIXES if x != f)) for f in FIXES}
# Study 2 and 2b on each dataset's best tuned simulator (docs/notes/2026-11-02-study2b.md).
# "t2": the best simulator with a defect (src/damage.py mixins; target_db / static_db move the
# tuned offsets further). Label shifts and smoothing act on its cache.
BEST = Path("results/tuning/best.json")       # {dataset: [switches]}, chosen on full real val
STUDY2 = {"clean": ("clean", None)}
STUDY2 |= {f"blur{k}": ("t2", {"blur": k}) for k in (1.25, 1.5, 2, 3, 5, 8)}
STUDY2 |= {f"sidelobes{d:+d}": ("t2", {"sidelobes_db": float(d)}) for d in (-10, 5, 10, 15, 20)}
STUDY2 |= {f"smoothing{p}": ("smoothall", p / 100) for p in (5, 10, 25, 50, 75, 90)}
STUDY2 |= {f"contrast-{d}": ("t2", {"target_db": -float(d)}) for d in (3, 6, 12, 20, 30)}
STUDY2 |= {f"noise+{d}": ("t2", {"target_db": -float(d), "static_db": -float(d)}) for d in (3, 6, 12, 20, 30)}
STUDY2 |= {f"clutter{d:+d}": ("t2", {"static_db": float(d)}) for d in (-20, -12, -6, 6, 12, 20)}
STUDY2 |= {f"unrender{p}": ("t2", {"unrender_frac": p / 100}) for p in (10, 25, 50, 75, 90)}
STUDY2 |= {f"extra{p}": ("t2", {"extra_frac": p / 100}) for p in (10, 25, 50, 100, 200)}
STUDY2 |= {f"rangeshift{b}": ("range", b) for b in (1, 2, 3, 5, 8)}
STUDY2 |= {f"dopplershift{b}": ("doppler", b) for b in (1, 2, 3, 5, 8)}
STUDY2B = {"clean": ("clean", None)}
STUDY2B |= {f"support{p}": ("t2", {"own_scenario_frac": p / 100}) for p in (25, 50, 75, 100)}
STUDY2B |= {f"annotation+{d}": ("t2", {"annotation_db": float(d)}) for d in (6, 12, 18)}
STUDY2B |= {f"tail{c}": ("t2", {"tail": float(c)}) for c in (2, 4, 8, 16)}
STUDY2B |= {f"confusers{k}": ("t2", {"confusers": k}) for k in (0, 0.5, 2, 4)}
BUDGET = {"clean": ("clean", None)}
# suite -> (versions, families counted by the rule, families reported only);
# a family is a defect, its versions in increasing damage after clean
SUITES = {"v1": (V1, ("drop", "range", "doppler", "blink", "contrast"), ("clutter",)),
          "fair": (FAIR, ("contrast", "unrender", "extra", "smooth", "noise"), ("clutter", "range")),
          "smoothall": (SMOOTHALL, ("smoothall",), ()),
          "tune": (TUNE, (), ()),
          "sim2": (SIM2, (), ()),
          "study2": (STUDY2, (), ("blur", "sidelobes", "smoothing", "contrast", "noise", "clutter",
                                  "unrender", "extra", "rangeshift", "dopplershift")),
          "study2b": (STUDY2B, (), ("support", "annotation", "tail", "confusers")),
          "budget": (BUDGET, (), ())}
ON_BEST = ("study2", "study2b", "budget")     # suites whose clean is the best tuned simulator
ON_CACHE = ("clean", "drop", "range", "doppler", "smooth", "smoothall")      # the rest are re-rendered


def best(dataset):
    return tuple(json.loads(BEST.read_text())[dataset])


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


def _raddet_sequence(kind, value, i):
    kw = {"generator": torch.Generator().manual_seed(SEED0 + i),
          "processing": value.get("processing", False)}
    if value.get("range_gain"):
        kw["range_gain_db"] = range_gain()
    if "contrast" in value:
        fit = json.loads(Path("results/tuning/raddet_contrast_fit.json").read_text())[value["contrast"]]
        kw |= {"target_offset_db": fit["target_db"], "static_offset_db": fit["static_db"]}
    o = TunedRaddetSimulator(**kw).gen_sequence()
    return o["x"].numpy(), o["labels"], {"classes": o["classes"]}


def _tuned_sequence(dataset, switches, tables, i, damage=None):
    dev = torch.device("cuda") if dataset == "radial" else None
    kw = {}
    if damage is not None:
        kw = {"mixin": RadialDamage if dataset == "radial" else TrackDamage,
              "damage_generator": torch.Generator().manual_seed(i), **damage}
    o = simulator(dataset, switches, torch.Generator().manual_seed(SEED0 + i), dev, tables, **kw).gen_sequence()
    return o["x"].numpy(), o["labels"], {}


def render(dataset, name, kind, value):
    """The cache of a re-rendered version, written on first use. Tuned versions
    live in data/tuned/<dataset>/ and are rendered again when their tuning
    table changes."""
    ds = DATASETS[dataset]
    if kind in ("tuned2", "t2"):
        text = (TUNING / f"{dataset}.json").read_text()
        switches, damage = (value, None) if kind == "tuned2" else (best(dataset), value)
        sub = "" if kind == "tuned2" else "damage_"
        out = Path("data/tuned") / dataset / (sub + name)
        done = {"kind": kind, "switches": list(switches), "damage": damage,
                "table_md5": hashlib.md5(text.encode()).hexdigest()}
        tables = json.loads(text)
        seq = lambda i: _tuned_sequence(dataset, switches, tables, i, damage)
    else:
        out, done = Path(ds.damage) / name, {"kind": kind, "value": value}
        f = {"carrada": _carrada_sequence, "radial": _radial_sequence, "raddet": _raddet_sequence}[dataset]
        seq = lambda i: f(kind, value, i)
    if not (out / "DONE").exists() or json.loads((out / "DONE").read_text()) != done:
        write_cache(out, N_SIM, seq, ds.grid, ds.stats_every, slot_by_rank=dataset != "radial")
        (out / "DONE").write_text(json.dumps(done))
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
        elif kind == "smoothall":
            it["x"] = smooth_background(it["x"].float(), it["traj"], it["present"], value,
                                        protect=False).half()
        out.append(it)
    return out
