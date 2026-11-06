"""Augmentation benefit at a small real-label budget (docs/notes/2026-11-06-metric-experiments.md,
Phase 5; candidates: results/metric/augmentation_candidates.json).

Each candidate's 200 synthetic sequences + the budget's evenly spaced real training sequences, each
batch 4 real + 4 synthetic frames, early stopping on the 32 real val sequences, 3 seeds: the budget
study's training (scripts/damage_study.py), so the real-only and tuned + real runs of
results/budget_<dataset>_real<b>_runs.json are its reference. Scored on real test (reported) and the
whole real val split. Resumable. Writes results/metric/augment/<dataset>_real<b>_runs.json and
detections to results/metric/dets/.

    PYTHONPATH=. python3 scripts/augment_study.py --dataset carrada --stage A
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.build_metric_features import load_candidate, registry
from src.data import load
from src.detector import FrameDetector, scores
from src.provenance import provenance
from src.train import N_VAL, detections, frame_batch, train
from src.versions import DATASETS

PLAN = Path("results/metric/augmentation_candidates.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="carrada", choices=tuple(DATASETS))
    ap.add_argument("--stage", default="A")
    ap.add_argument("--budget", type=int)
    a = ap.parse_args()
    stage = json.loads(PLAN.read_text())[f"stage_{a.stage}"]
    budget = a.budget or stage["budget"]
    ds = DATASETS[a.dataset]
    dev = torch.device("cuda")
    out = Path(f"results/metric/augment/{a.dataset}_real{budget}_runs.json")
    dets_dir = Path("results/metric/dets") / f"augment_{a.dataset}_real{budget}"
    report = json.loads(out.read_text()) if out.exists() else {"runs": {}, "val": {}, "stops": {}}
    real_all = load(ds.real, "train")
    idx = np.linspace(0, len(real_all) - 1, budget).round().astype(int)
    real = [real_all[int(i)] for i in idx]
    report["real_sequences"] = idx.tolist()
    full_val = load(ds.real, "val")
    val = [full_val[int(i)] for i in np.linspace(0, len(full_val) - 1, N_VAL).round()]
    test = load(ds.real, "test")
    reg = registry(a.dataset)
    for cid in stage["candidates"]:
        if cid == "tuned":
            continue                                    # the budget study's mix run
        todo = [s for s in stage["seeds"] if str(s) not in report["runs"].get(cid, {})]
        if not todo:
            continue
        pool, _ = load_candidate(reg[cid])
        for seed in todo:
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed)
            model = FrameDetector(width=32).to(dev)

            def draw(g, p=pool):
                x1, h1 = frame_batch(real, 4, dev, g)
                x2, h2 = frame_batch(p, 4, dev, g)
                return torch.cat([x1, x2]), torch.cat([h1, h2])
            stop = train(model, draw, val, dev, gen)
            d, t = detections(model, test, dev)
            res = scores(d, t, test[0]["x"].shape[-1])
            vd, vt = detections(model, full_val, dev)
            report["val"].setdefault(cid, {})[str(seed)] = scores(vd, vt, full_val[0]["x"].shape[-1])
            report["runs"].setdefault(cid, {})[str(seed)] = res
            report["stops"].setdefault(cid, {})[str(seed)] = stop
            dets_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"dets": d, "tgts": t}, dets_dir / f"mix_{cid.replace(':', '__')}_s{seed}.pt")
            report["provenance"] = provenance()
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(report, indent=1))
            tmp.replace(out)
            print(f"{cid} seed {seed}: {res} {stop}", flush=True)
        del pool


if __name__ == "__main__":
    main()
