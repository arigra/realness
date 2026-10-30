"""Does detector AP on real CARRADA respond to controlled defects in simulator data?
Pre-registration: docs/notes/2026-10-27-damage-spread.md (--suite v1) and
docs/notes/2026-10-28-damage-spread-fair.md (--suite fair).

Versions of the same 200 simulator sequences (clean + defects), each trained
in two settings with the training loop of radial_detector_study.py: "only" (the
version alone) and "mix" (half 50 real sequences, half the version). Test: real
CARRADA test. Resumable per (setting, dataset, seed); --verdict applies the rule.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import dataclasses

from numpy.lib.format import open_memmap

from scripts.build_carrada_cache import stats, write_split
from scripts.radial_detector_arms import dataset_pool
from scripts.radial_detector_study import LR, N_VAL, detections, draw_from, run_stage
from src import carrada
from src.damage import (DamagedCarradaSimulator, DamagedSceneSimulator, clutter_scenario,
                        drop_labels, noisy_spec, shift_labels, smooth_background)
from src.scene_data import pack_labels
from src.scene_sim import Scenario
from src.detector_scene import RadialVehicleDetector, average_precision_wrapped

N_SIM, SEED0 = 200, 10_000_000
# simulator cache (its first N_SIM training sequences), 50-real cache (with its val/test), outputs
CONFIGS = {"carrada": ("data/carrada_sim_engineer", "data/carrada_seq_n50", "data/carrada_damage"),
           "radial": ("data/scene_engineer", "data/radial_fold0_n50", "data/radial_damage")}
SIM, REAL, DAMAGE_DIR = CONFIGS["carrada"][0], CONFIGS["carrada"][1], Path(CONFIGS["carrada"][2])
DATASET = "carrada"
RENDERED = ("blink", "contrast", "clutter", "unrender", "extra", "noise")
# name -> (kind, value); label defects and smoothing act on the clean cache, the rest re-render it
V1 = {"clean": ("label", None)}
V1 |= {f"drop{p}": ("drop", p / 100) for p in (10, 25, 50, 75)}
V1 |= {f"range{b}": ("range", b) for b in (1, 2, 3)}
V1 |= {f"doppler{b}": ("doppler", b) for b in (1, 2, 3)}
V1 |= {f"blink{p}": ("blink", p / 100) for p in (10, 25, 50)}
V1 |= {f"contrast-{d}": ("contrast", -d) for d in (3, 6, 10)}
V1 |= {"clutter+6": ("clutter", 6.0), "clutter-6": ("clutter", -6.0)}
FAIR = {"clean": ("label", None)}
FAIR |= {f"contrast-{d}": ("contrast", -d) for d in (6, 12, 20)}
FAIR |= {f"unrender{p}": ("unrender", p / 100) for p in (10, 25, 50)}
FAIR |= {f"extra{p}": ("extra", p / 100) for p in (10, 25, 50)}
FAIR |= {f"smooth{p}": ("smooth", p / 100) for p in (25, 50, 75)}
FAIR |= {f"noise+{d}": ("noise", d) for d in (6, 12)}
FAIR |= {f"clutter{d:+d}": ("clutter", float(d)) for d in (6, -6, 12, -12)}
FAIR |= {f"range{b}": ("range", b) for b in (1, 2)}
# ordered families (clean first, then increasing damage) that count towards the rule;
# clutter is two-sided and range is the known control: both reported only
SUITES = {"v1": (V1, ("drop", "range", "doppler", "blink", "contrast"), ("clutter",)),
          "fair": (FAIR, ("contrast", "unrender", "extra", "smooth", "noise"), ("clutter", "range"))}
DATASETS, COUNTED, REPORTED = SUITES["v1"]


def render_radial(out, kind, value):
    """The first N_SIM RADIal training scenes re-rendered with one defect, on the GPU
    (the device the cache was built on, so the clean scene is reproduced exactly)."""
    dev = torch.device("cuda")
    x = open_memmap(out / "train_x.npy", mode="w+", dtype=np.float16, shape=(N_SIM, 8, 512, 256))
    trajs, presents, metas = [], [], []
    for i in range(N_SIM):
        kw = {"generator": torch.Generator().manual_seed(SEED0 + i), "device": dev,
              "blink_generator": torch.Generator().manual_seed(i)}
        if kind in ("unrender", "extra"):
            kw[f"{kind}_frac"] = value
        elif kind == "contrast":
            kw["target_offset_db"] = value
        elif kind == "noise":
            kw["scenario"] = dataclasses.replace(Scenario(), snr_offset_db=-value)
        else:
            kw["clutter_offset_db"] = value
        o = DamagedSceneSimulator(**kw).gen_sequence()
        x[i] = o["x"].numpy().astype(np.float16)
        t, p = pack_labels(o["labels"], 8)
        trajs.append(t); presents.append(p)
        metas.append({"road_type": o["road_type"], "ego_speed": o["ego_speed"]})
    x.flush()
    torch.save({"traj": torch.stack(trajs), "present": torch.stack(presents), "meta": metas},
               out / "train_labels.pt")
    sample = np.asarray(x[::4], dtype=np.float32)
    torch.save({"mean": float(sample.mean()), "std": float(sample.std())}, out / "stats.pt")


def render(name, kind, value):
    out = DAMAGE_DIR / name
    if (out / "DONE").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    if DATASET == "radial":
        render_radial(out, kind, value)
        (out / "DONE").write_text(json.dumps({"kind": kind, "value": value}))
        return out

    def make(i):
        kw = {"generator": torch.Generator().manual_seed(SEED0 + i),
              "blink_generator": torch.Generator().manual_seed(i)}
        if kind in ("blink", "unrender", "extra"):
            kw[f"{kind}_frac"] = value
        elif kind == "contrast":
            kw["target_offset_db"] = value
        elif kind == "noise":
            kw["spec"] = noisy_spec(value)
        else:
            kw["scenario"] = clutter_scenario(value)
        o = DamagedCarradaSimulator(**kw).gen_sequence()
        fl = [[{"iid": l["id"], "range_bin": l["range_bin"], "doppler_bin": l["doppler_bin"],
                "cls": l["cls"]} for l in f] for f in o["labels"]]
        t, p, c = carrada.pack(fl, 8)
        return o["x"].numpy(), t, p, c, {"classes": o["classes"]}

    stats(out, write_split(out, "train", N_SIM, make))
    (out / "DONE").write_text(json.dumps({"kind": kind, "value": value}))
    return out


def pool(name, clean):
    kind, value = DATASETS[name]
    if kind in RENDERED:
        return dataset_pool(render(name, kind, value), "train")
    g = torch.Generator().manual_seed(0)
    out = []
    for it in clean:
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


def recall_wrapped(detections, targets, n_doppler, radius=2.0):
    """Share of labelled objects with a detection within radius (greedy, as the AP)."""
    hit = n = 0
    for frame_dets, tgt in zip(detections, targets):
        n += len(tgt)
        used = torch.zeros(len(tgt), dtype=torch.bool)
        for _, r, d in sorted(frame_dets, key=lambda t: -t[0]):
            if not len(tgt):
                break
            dd = torch.remainder(tgt[:, 1] - d + n_doppler / 2, n_doppler) - n_doppler / 2
            dist = torch.sqrt((tgt[:, 0] - r) ** 2 + dd ** 2)
            dist[used] = float("inf")
            best = int(torch.argmin(dist))
            if float(dist[best]) <= radius:
                used[best] = True
        hit += int(used.sum())
    return hit / max(n, 1)


def scores(dets, tgts, n_doppler):
    return {"ap2": average_precision_wrapped(dets, tgts, n_doppler, radius=2.0),
            "ap4": average_precision_wrapped(dets, tgts, n_doppler, radius=4.0),
            "recall2": recall_wrapped(dets, tgts, n_doppler, radius=2.0)}


def verdict(report, metric="ap2", counted=COUNTED, reported=REPORTED):
    """v1 runs hold the AP at 2 bins as a number; later runs a dict of scores."""
    out = {}
    for setting, ds in report["runs"].items():
        val = lambda r: [v if isinstance(v, float) else v[metric] for v in r.values()]
        mean = {n: float(np.mean(val(r))) for n, r in ds.items()}
        rng = {n: float(np.ptp(val(r))) for n, r in ds.items()}
        noise = float(np.median(list(rng.values())))
        gap = max(mean.values()) - min(mean.values())
        fam = {}
        for f in counted + reported:
            names = ["clean"] + [n for n in ds if n.startswith(f)]
            m = [mean[n] for n in names]
            fam[f] = {"means": dict(zip(names, m)),
                      "ordered": bool(all(b - a <= noise for a, b in zip(m, m[1:]))
                                      and m[0] - m[-1] >= noise)}
        n_ord = sum(fam[f]["ordered"] for f in counted)
        out[setting] = {"metric": metric, "test_mean": mean, "seed_range": rng, "noise": noise,
                        "gap": gap, "gap_over_noise": gap / noise if noise else None,
                        "families": fam, "counted": list(counted), "n_ordered": n_ord,
                        "pass": bool(gap >= 3 * noise and n_ord >= 3)}
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--suite", default="v1", choices=tuple(SUITES))
    ap_.add_argument("--dataset", default="carrada", choices=tuple(CONFIGS))
    ap_.add_argument("--settings", default="only,mix")
    ap_.add_argument("--datasets", help="default: the whole suite")
    ap_.add_argument("--dets-dir", help="keep each run's test detections here")
    ap_.add_argument("--seeds", default="1,2,3")
    ap_.add_argument("--out", default="samples/damage_study.json")
    ap_.add_argument("--verdict", default="samples/damage_verdict.json")
    a = ap_.parse_args()
    global DATASETS, DATASET, SIM, REAL, DAMAGE_DIR
    DATASETS, counted, reported = SUITES[a.suite]
    DATASET = a.dataset
    SIM, REAL, DAMAGE_DIR = CONFIGS[a.dataset][0], CONFIGS[a.dataset][1], Path(CONFIGS[a.dataset][2])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real = dataset_pool(REAL, "train")
    full_val = dataset_pool(REAL, "val")
    val = [full_val[int(i)] for i in np.linspace(0, len(full_val) - 1, N_VAL).round()]
    test = dataset_pool(REAL, "test")
    clean = dataset_pool(SIM, "train", limit=N_SIM)
    names = a.datasets.split(",") if a.datasets else list(DATASETS)
    settings, seeds = a.settings.split(","), [int(s) for s in a.seeds.split(",")]

    out = Path(a.out)
    report = json.loads(out.read_text()) if out.exists() else {"sim": SIM, "real": REAL, "n_sim": N_SIM,
                                                              "runs": {}, "stops": {}}
    # one version in memory at a time (a RADIal version is ~420 MB)
    for name in names:
        todo = [(s, sd) for s in settings for sd in seeds
                if str(sd) not in report["runs"].get(s, {}).get(name, {})]
        if not todo:
            continue
        p = pool(name, clean)
        print(f"{name}: {len(p)} sequences", flush=True)
        for setting, seed in todo:
            runs = report["runs"].setdefault(setting, {}).setdefault(name, {})
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed)
            model = RadialVehicleDetector(width=32).to(dev)
            if setting == "only":
                draw = lambda g, p=p: draw_from(p, 8, dev, g)
            else:
                def draw(g, p=p):
                    x1, h1 = draw_from(real, 4, dev, g)
                    x2, h2 = draw_from(p, 4, dev, g)
                    return torch.cat([x1, x2]), torch.cat([h1, h2])
            st = run_stage(model, LR, draw, val, dev, gen, lambda m: None)
            dets, tgts, _ = detections(model, test, dev)
            n_d = test[0]["x"].shape[-1]
            if a.suite == "v1":
                runs[str(seed)] = average_precision_wrapped(dets, tgts, n_d, radius=2.0)
            else:
                runs[str(seed)] = scores(dets, tgts, n_d)
            if a.dets_dir:
                Path(a.dets_dir).mkdir(parents=True, exist_ok=True)
                torch.save({"dets": dets, "tgts": tgts},
                           Path(a.dets_dir) / f"{setting}_{name}_s{seed}.pt")
            report["stops"].setdefault(setting, {}).setdefault(name, {})[str(seed)] = st
            print(f"{setting} {name} seed {seed}: {runs[str(seed)]} {st}", flush=True)
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(report, indent=2))
            tmp.replace(out)
            del model
            torch.cuda.empty_cache()
        del p
    metrics = ("ap2",) if a.suite == "v1" else ("ap2", "ap4", "recall2")
    vs = {m: verdict(report, m, counted, reported) for m in metrics}
    Path(a.verdict).write_text(json.dumps(vs["ap2"] if a.suite == "v1" else vs, indent=2))
    for m, v in vs.items():
        for s, x in v.items():
            print(f"{m} {s}: gap {x['gap']:.3f} noise {x['noise']:.3f} "
                  f"ordered {x['n_ordered']}/{len(counted)} pass {x['pass']}")


if __name__ == "__main__":
    main()
