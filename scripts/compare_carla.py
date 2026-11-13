"""Is the RadSimReal-style CARLA simulator better or worse than ours, on RADDet?

Same protocol as the damage studies (scripts/damage_study.py, setting "only"): the frame
detector is trained on one simulator's 200 sequences alone, stopped on 32 real val sequences,
and scored on the real scene-disjoint test split (data/raddet_scenesplit) and the whole real val
split. 3 seeds per source. Reference: the 50 real training sequences alone.

Sources:
  real            the 50 real training sequences (reference)
  ours_clean      data/raddet_sim (src/raddet_sim.py)
  ours_tuned      the best tuned simulator (results/tuning/best.json)
  carla_v1        RadSimReal/CARLA, before the road/noise fixes   (RadSimReal/scripts/export_rd_realness.py)
  carla_v2        RadSimReal/CARLA, measured road + measured noise texture

    PYTHONPATH=. python3 scripts/compare_carla.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data import load
from src.detector import FrameDetector, scores
from src.train import N_VAL, detections, frame_batch, train
from src.versions import DATASETS, N_SIM, best, render

OUT = Path("results/compare_carla_raddet.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="real,ours_clean,ours_tuned,carla_v1,carla_v2")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    ds = DATASETS["raddet"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real = load(ds.real, "train")
    full_val = load(ds.real, "val")
    val = [full_val[int(i)] for i in np.linspace(0, len(full_val) - 1, N_VAL).round()]
    test = load(ds.real, "test")
    pools = {"real": lambda: real,
             "ours_clean": lambda: load(ds.sim, "train", limit=N_SIM),
             "ours_tuned": lambda: load(render("raddet", "best", "tuned2", best("raddet")), "train", limit=N_SIM),
             "carla_v1": lambda: load("data/raddet_carla_v1", "train"),
             "carla_v2": lambda: load("data/raddet_carla_v2", "train")}
    report = json.loads(out.read_text()) if out.exists() else {"test": {}, "val": {}, "stops": {}}
    for name in a.sources.split(","):
        seeds = [s for s in a.seeds.split(",") if s not in report["test"].get(name, {})]
        if not seeds:
            continue
        pool = pools[name]()
        print(f"{name}: {len(pool)} sequences", flush=True)
        for seed in seeds:
            torch.manual_seed(int(seed))
            gen = torch.Generator().manual_seed(int(seed))
            model = FrameDetector(width=32).to(dev)
            stop = train(model, lambda g, p=pool: frame_batch(p, 8, dev, g), val, dev, gen)
            dets, tgts = detections(model, test, dev)
            report["test"].setdefault(name, {})[seed] = scores(dets, tgts, test[0]["x"].shape[-1])
            vd, vt = detections(model, full_val, dev)
            report["val"].setdefault(name, {})[seed] = scores(vd, vt, full_val[0]["x"].shape[-1])
            report["stops"].setdefault(name, {})[seed] = stop
            print(name, seed, report["test"][name][seed], flush=True)
            out.write_text(json.dumps(report, indent=1))
            del model
            torch.cuda.empty_cache()
    print(f"\n{'source':12}{'test AP@2':>12}{'test AP@4':>12}{'recall@2':>10}{'val AP@2':>10}")
    for name, runs in report["test"].items():
        m = {k: (np.mean([r[k] for r in runs.values()]), np.ptp([r[k] for r in runs.values()])) for k in ("ap2", "ap4", "recall2")}
        v = np.mean([r["ap2"] for r in report["val"][name].values()])
        print(f"{name:12}{m['ap2'][0]:8.3f}±{m['ap2'][1]/2:.3f}{m['ap4'][0]:8.3f}±{m['ap4'][1]/2:.3f}{m['recall2'][0]:10.3f}{v:10.3f}")


if __name__ == "__main__":
    main()
