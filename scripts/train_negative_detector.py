"""The fixed detector of the hard-negative probe (docs/notes/2026-11-06-metric-experiments.md,
Phase 1.4): FrameDetector(width=32) on the best tuned simulator alone, seed 1, Study 2's training
and early stopping (32 real val sequences). Writes results/metric/negative_detector_<dataset>.pt.

    PYTHONPATH=. python3 scripts/train_negative_detector.py --dataset carrada
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data import load
from src.detector import FrameDetector
from src.provenance import provenance
from src.train import N_VAL, frame_batch, train
from src.versions import DATASETS, N_SIM, best, render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="carrada", choices=tuple(DATASETS))
    a = ap.parse_args()
    out = Path(f"results/metric/negative_detector_{a.dataset}.pt")
    dev = torch.device("cuda")
    ds = DATASETS[a.dataset]
    pool = load(render(a.dataset, "best", "tuned2", best(a.dataset)), "train", limit=N_SIM)
    full_val = load(ds.real, "val")
    val = [full_val[int(i)] for i in np.linspace(0, len(full_val) - 1, N_VAL).round()]
    torch.manual_seed(1)
    gen = torch.Generator().manual_seed(1)
    model = FrameDetector(width=32).to(dev)
    t0 = time.time()
    stop = train(model, lambda g: frame_batch(pool, 8, dev, g), val, dev, gen)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "width": 32, "stop": stop, "seed": 1,
                "provenance": provenance(), "seconds": time.time() - t0, "switches": best(a.dataset)}, out)
    print(a.dataset, json.dumps(stop), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
