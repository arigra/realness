"""Does the candidate ranking transfer to a different detector? (docs/notes/2026-11-06-metric-experiments.md,
Phase 7.1). The compact transformer detector (src/transformer_detector.py) trained synthetic-only on a
focused set of candidates, 3 seeds, Study 2's training and early stopping; real-test AP@2 / AP@4 /
recall@2, compared with the U-Net's AP of the same candidates. Resumable.
Writes results/metric/architecture/<dataset>.json.

    PYTHONPATH=. python3 scripts/architecture_check.py --dataset carrada \
        --candidates spec,tuned,study2:rangeshift1,study2:blur2,study2b:tail2,<selected>,<oracle>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.build_metric_features import REGISTRY, load_candidate
from src.data import load
from src.detector import scores
from src.provenance import provenance
from src.train import N_VAL, detections, frame_batch, train
from src.transformer_detector import build
from src.versions import DATASETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="carrada", choices=tuple(DATASETS))
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--seeds", default="1,2,3")
    a = ap.parse_args()
    ds = DATASETS[a.dataset]
    dev = torch.device("cuda")
    out = Path(f"results/metric/architecture/{a.dataset}.json")
    report = json.loads(out.read_text()) if out.exists() else {"runs": {}, "stops": {}}
    full_val = load(ds.real, "val")
    val = [full_val[int(i)] for i in np.linspace(0, len(full_val) - 1, N_VAL).round()]
    test = load(ds.real, "test")
    reg = {r["candidate"]: r for r in json.loads(REGISTRY.read_text())["candidates"] if r["dataset"] == a.dataset}
    for cid in a.candidates.split(","):
        todo = [s for s in map(int, a.seeds.split(",")) if str(s) not in report["runs"].get(cid, {})]
        if not todo:
            continue
        pool, _ = load_candidate(reg[cid])
        for seed in todo:
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed)
            model = build(*ds.grid, dev)
            t0 = time.time()
            stop = train(model, lambda g, p=pool: frame_batch(p, 8, dev, g), val, dev, gen)
            d, t = detections(model, test, dev)
            res = scores(d, t, test[0]["x"].shape[-1]) | {"seconds": time.time() - t0}
            report["runs"].setdefault(cid, {})[str(seed)] = res
            report["stops"].setdefault(cid, {})[str(seed)] = stop
            report["provenance"] = provenance()
            report["n_params"] = sum(p.numel() for p in model.parameters())
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(report, indent=1))
            tmp.replace(out)
            print(f"{cid} seed {seed}: {res} {stop}", flush=True)
        del pool


if __name__ == "__main__":
    main()
