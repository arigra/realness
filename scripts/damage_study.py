"""Does real-test detector AP respond to controlled defects in synthetic data?

Pre-registrations: docs/notes/2026-10-27-damage-spread.md (--suite v1) and
docs/notes/2026-10-28-damage-spread-fair.md (--suite fair).

Every version (src/versions.py) trains the frame detector in two settings,
"only" (the version alone) and "mix" (each batch half from 50 real
sequences, half from the version), with 3 seeds ("real": the 50 real
sequences alone, a reference; run it with --versions clean); every detector is scored on
the real test split. Resumable per (setting, version, seed). Writes
results/<suite>_<dataset>_runs.json, _verdict.json and _dets/.

    PYTHONPATH=. python3 scripts/damage_study.py --suite fair --dataset radial
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data import load
from src.detector import FrameDetector, scores
from src.train import N_VAL, detections, frame_batch, train
from src.versions import DATASETS, N_SIM, SUITES, family, version


def verdict(report, metric, counted, reported):
    """The pre-registered rule. noise = median seed range over versions; a family
    is ordered if its means never rise by more than noise along increasing
    damage and clean - worst >= noise; pass = gap >= 3 x noise and >= 3 ordered
    counted families. v1 runs stored the AP at 2 cells as a bare number."""
    out = {}
    for setting, ds in report["runs"].items():
        val = lambda r: [v if isinstance(v, float) else v[metric] for v in r.values()]
        mean = {n: float(np.mean(val(r))) for n, r in ds.items()}
        rng = {n: float(np.ptp(val(r))) for n, r in ds.items()}
        noise = float(np.median(list(rng.values())))
        gap = max(mean.values()) - min(mean.values())
        fam = {}
        for f in counted + reported:
            names = family(ds, f)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="fair", choices=tuple(SUITES))
    ap.add_argument("--dataset", default="carrada", choices=tuple(DATASETS))
    ap.add_argument("--settings", default="only,mix")
    ap.add_argument("--versions", help="default: the whole suite")
    ap.add_argument("--seeds", default="1,2,3")
    a = ap.parse_args()
    ds = DATASETS[a.dataset]
    versions, counted, reported = SUITES[a.suite]
    stem = Path("results") / f"{a.suite}_{a.dataset}"
    runs_path, dets_dir = Path(f"{stem}_runs.json"), Path(f"{stem}_dets")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real = load(ds.real, "train")
    full_val = load(ds.real, "val")
    val = [full_val[int(i)] for i in np.linspace(0, len(full_val) - 1, N_VAL).round()]
    test = load(ds.real, "test")
    clean = load(ds.sim, "train", limit=N_SIM)
    names = a.versions.split(",") if a.versions else list(versions)
    settings, seeds = a.settings.split(","), [int(s) for s in a.seeds.split(",")]
    report = (json.loads(runs_path.read_text()) if runs_path.exists() else
              {"sim": ds.sim, "real": ds.real, "n_sim": N_SIM, "runs": {}, "stops": {}})

    for name in names:                  # one version in memory at a time (RADIal: ~420 MB)
        todo = [(s, sd) for s in settings for sd in seeds
                if str(sd) not in report["runs"].get(s, {}).get(name, {})]
        if not todo:
            continue
        pool = version(a.dataset, name, *versions[name], clean)
        print(f"{name}: {len(pool)} sequences", flush=True)
        for setting, seed in todo:
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed)
            model = FrameDetector(width=32).to(dev)
            if setting == "only":
                draw = lambda g, p=pool: frame_batch(p, 8, dev, g)
            elif setting == "real":                     # reference: the 50 real sequences alone
                draw = lambda g: frame_batch(real, 8, dev, g)
            else:
                def draw(g, p=pool):
                    x1, h1 = frame_batch(real, 4, dev, g)
                    x2, h2 = frame_batch(p, 4, dev, g)
                    return torch.cat([x1, x2]), torch.cat([h1, h2])
            stop = train(model, draw, val, dev, gen)
            dets, tgts = detections(model, test, dev)
            result = scores(dets, tgts, test[0]["x"].shape[-1])
            dets_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"dets": dets, "tgts": tgts}, dets_dir / f"{setting}_{name}_s{seed}.pt")
            report["runs"].setdefault(setting, {}).setdefault(name, {})[str(seed)] = result
            report["stops"].setdefault(setting, {}).setdefault(name, {})[str(seed)] = stop
            print(f"{setting} {name} seed {seed}: {result} {stop}", flush=True)
            tmp = runs_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(report, indent=2))
            tmp.replace(runs_path)
            del model
            torch.cuda.empty_cache()
        del pool

    v = {m: verdict(report, m, counted, reported) for m in ("ap2", "ap4", "recall2")
         if a.suite == "fair" or m == "ap2"}
    Path(f"{stem}_verdict.json").write_text(json.dumps(v, indent=2))
    for m, by_setting in v.items():
        for s, x in by_setting.items():
            print(f"{m} {s}: gap {x['gap']:.3f} noise {x['noise']:.3f} "
                  f"ordered {x['n_ordered']}/{len(counted)} pass {x['pass']}")


if __name__ == "__main__":
    main()
