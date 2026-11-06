"""Cheap proxy detector baseline (docs/notes/2026-11-06-metric-experiments.md, Phase 4).

Per candidate: FrameDetector(width=8) on its first 50 synthetic sequences, 1,000 steps, batch 8, no
early stopping; scored (AP@2) on the budget-5 reference subsets 0, 1, 2 of the 50 real training
sequences; the proxy score is their mean. Seed 1 for every usable candidate, then seeds 2 and 3 for
the 5 best per dataset. Resumable. Writes results/metric/proxy/<dataset>.json.

    PYTHONPATH=. python3 scripts/proxy_detector.py --dataset carrada
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.build_metric_features import load_candidate, registry, subsets
from src.data import load
from src.detector import FrameDetector, average_precision, focal_loss
from src.provenance import provenance
from src.train import LR, detections, frame_batch
from src.versions import DATASETS

WIDTH, N_SEQ, STEPS, BATCH = 8, 50, 1000, 8
OUT = Path("results/metric/proxy")


def proxy(pool, real_subsets, dev, seed):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = FrameDetector(width=WIDTH).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    model.train()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(STEPS):
        x, heat = frame_batch(pool, BATCH, dev, gen)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(x)
        loss = focal_loss(logits, heat)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    train_s = time.time() - t0
    aps = []
    for sub in real_subsets:
        dets, tgts = detections(model, sub, dev)
        aps.append(average_precision(dets, tgts, sub[0]["x"].shape[-1], radius=2.0))
    return {"ap2_subsets": aps, "proxy": float(np.mean(aps)), "train_seconds": train_s,
            "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="carrada", choices=tuple(DATASETS))
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--candidates", default="all", help="'all' usable, 'bank' (Phase 6, seed 1 only, own file)")
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = OUT / (f"{a.dataset}.json" if a.candidates == "all" else f"{a.dataset}_{a.candidates}.json")
    report = json.loads(out.read_text()) if out.exists() else {
        "config": {"width": WIDTH, "n_seq": N_SEQ, "steps": STEPS, "batch": BATCH, "lr": LR,
                   "reference": "budget-5 subsets 0,1,2 of the real training sequences"}, "runs": {}}
    real = load(DATASETS[a.dataset].real, "train")
    subs = subsets(5, 3)
    real_subsets = [[real[int(i)] for i in subs[s]] for s in range(3)]
    report["reference_sequences"] = {str(s): subs[s].tolist() for s in range(3)}
    reg = registry(a.dataset)
    rows = [r for r in reg.values() if (r["usable"] if a.candidates == "all" else r["suite"] == "bank")]

    def run(row, seed):
        cid = row["candidate"]
        if str(seed) in report["runs"].get(cid, {}):
            return
        pool, _ = load_candidate(row)
        res = proxy(pool[:N_SEQ], real_subsets, dev, seed)
        report["runs"].setdefault(cid, {})[str(seed)] = res
        report["provenance"] = provenance()
        tmp = out.with_suffix(".tmp")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(report, indent=1))
        tmp.replace(out)
        print(f"{cid:24s} seed {seed}: proxy {res['proxy']:.3f} {np.round(res['ap2_subsets'], 3).tolist()} "
              f"{res['seconds']:.0f}s", flush=True)

    for row in rows:
        run(row, 1)
    if a.candidates != "all":
        return
    best = sorted(rows, key=lambda r: -report["runs"][r["candidate"]]["1"]["proxy"])[:a.top]
    for row in best:
        for seed in (2, 3):
            run(row, seed)
    report["top"] = [r["candidate"] for r in best]
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
