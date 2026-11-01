"""Measure every version of a dataset against its 50 real training sequences
(docs/notes/2026-11-01-realness-score.md, step 2). Writes
results/measures_<dataset>.pt ({"real": measures, "versions": {name: measures}})
and results/distances_<dataset>.json (per measure and per group).

    PYTHONPATH=. python3 scripts/measure_versions.py --dataset carrada
"""
import argparse
import json
from pathlib import Path

import torch

from src.data import load
from src.measures import distances, group_distances, measure
from src.versions import DATASETS, N_SIM, SUITES, version


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="carrada", choices=tuple(DATASETS))
    ap.add_argument("--suites", default="fair,smoothall")
    a = ap.parse_args()
    ds = DATASETS[a.dataset]
    out = Path("results") / f"measures_{a.dataset}.pt"
    res = torch.load(out, weights_only=False) if out.exists() else {"real": None, "versions": {}}
    if res["real"] is None:
        res["real"] = measure(load(ds.real, "train"))
    clean = load(ds.sim, "train", limit=N_SIM)
    for suite in a.suites.split(","):
        for name, (kind, value) in SUITES[suite][0].items():
            if name in res["versions"]:
                continue
            res["versions"][name] = measure(version(a.dataset, name, kind, value, clean))
            torch.save(res, out)
            print(name, group_distances(distances(res["versions"][name], res["real"])), flush=True)
    d = {n: distances(m, res["real"]) for n, m in res["versions"].items()}
    Path("results", f"distances_{a.dataset}.json").write_text(json.dumps(
        {n: {"measures": x, "groups": group_distances(x)} for n, x in d.items()}, indent=2))


if __name__ == "__main__":
    main()
