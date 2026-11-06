"""Learned-feature distance baselines (docs/notes/2026-11-06-metric-experiments.md, Phase 7.2).

Encoders: the U-Net (src/detector.FrameDetector) and the compact transformer
(src/transformer_detector.py), each trained as a detector on one half of the 50 real training
sequences (fold 0: even indices, fold 1: odd), early-stopped on the 32 real val sequences, then frozen.
An object's embedding = the encoder's features at its label cell (U-Net: decoder output and
bottleneck; transformer: its token and the decoder output). Cross-fitting: a real object is embedded
only by the encoder trained on the other fold; candidates by both, and each encoder is compared with
the real objects it did not see. Also raw multiscale patches (25 x 17 z patch and its 2x average-pooled
version, randomly projected to 64 dims). Distances: energy distance to the reference's objects
(features standardised by the reference), global and the mean over range-tercile x near-zero strata.
Writes results/metric/encoders/<dataset>_<arch>_fold<k>.pt and results/metric/learned/<dataset>.json.

    PYTHONPATH=. python3 scripts/learned_features.py --dataset carrada
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.build_metric_features import REGISTRY, load_candidate, subsets
from scripts.evaluate_utility_metric import REFERENCES
from src.data import load, normalised
from src.detector import FrameDetector
from src.provenance import provenance
from src.train import N_VAL, frame_batch, train
from src.transformer_detector import build
from src.utility_metric import RADARS, energy_distance, wrap

ENC = Path("results/metric/encoders")
OUT = Path("results/metric/learned")
K_R, K_D = 12, 8


def encoder(dataset, arch, fold, real, val, dev):
    path = ENC / f"{dataset}_{arch}_fold{fold}.pt"
    model = FrameDetector(32).to(dev) if arch == "cnn" else build(*real[0]["x"].shape[-2:], dev)
    if path.exists():
        model.load_state_dict(torch.load(path, weights_only=False)["state_dict"])
    else:
        pool = [it for i, it in enumerate(real) if i % 2 == fold]
        torch.manual_seed(1)
        stop = train(model, lambda g: frame_batch(pool, 8, dev, g), val, dev, torch.Generator().manual_seed(1))
        ENC.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "stop": stop, "fold": fold, "provenance": provenance()}, path)
    model.eval()
    return model


def _hooks(model, arch):
    feats = {}
    if arch == "cnn":
        model.dec1.register_forward_hook(lambda m, i, o: feats.__setitem__("fine", o))
        model.mid.register_forward_hook(lambda m, i, o: feats.__setitem__("coarse", o))
    else:
        model.dec.register_forward_hook(lambda m, i, o: feats.__setitem__("fine", o))
        model.up.register_forward_hook(lambda m, i, o: feats.__setitem__("coarse", i[0]))
    return feats


@torch.no_grad()
def embed(models, hooks, pool, radar, dev, proj):
    """Per labelled object: seq, range_m, near_zero, embeddings per model, raw patch projection."""
    rows = {"seq": [], "range_m": [], "near_zero": []}
    embs = {k: [] for k in models}
    raw = []
    for s, item in enumerate(pool):
        x = torch.stack([normalised(item, t) for t in range(item["x"].shape[0])])
        labs = [(t, r, d) for t in range(x.shape[0]) for r, d in item["traj"][item["present"][:, t], t].tolist()]
        if not labs:
            continue
        for k, m in models.items():
            m(x[:, None].to(dev))
            fine, coarse = hooks[k]["fine"].float().cpu(), hooks[k]["coarse"].float().cpu()
            R, D = x.shape[-2:]
            for t, r, d in labs:
                ri, di = min(max(int(round(r)), 0), R - 1), int(round(d)) % D
                cr = min(ri * coarse.shape[-2] // R, coarse.shape[-2] - 1)
                cd = min(di * coarse.shape[-1] // D, coarse.shape[-1] - 1)
                embs[k].append(torch.cat([fine[t, :, ri, di], coarse[t, :, cr, cd]]).numpy())
        for t, r, d in labs:
            R, D = x.shape[-2:]
            ri, di = min(max(int(round(r)), K_R), R - K_R - 1), int(round(d)) % D
            cols = (di + np.arange(-K_D, K_D + 1)) % D
            p = x[t, ri - K_R: ri + K_R + 1][:, cols]
            p2 = F.avg_pool2d(p[None, None], 2, ceil_mode=True)[0, 0]
            raw.append(proj @ torch.cat([p.flatten(), p2.flatten()]).numpy())
            rows["seq"].append(s)
            rows["range_m"].append(r * radar.range_res)
            rows["near_zero"].append(float(abs(wrap(d - radar.zero, D)) <= 1))
    out = {k: np.asarray(v) for k, v in rows.items()}
    out |= {f"emb_{k}": np.asarray(v) for k, v in embs.items()}
    out["raw_patch"] = np.asarray(raw)
    return out


def distance(C, R, c_strata, r_strata, min_n=5):
    """Energy distance (reference-standardised) and its mean over strata with >= min_n on both sides."""
    if len(C) < min_n or len(R) < min_n:
        return np.nan, np.nan
    mu, sd = R.mean(0), R.std(0) + 1e-6
    g = energy_distance((C - mu) / sd, (R - mu) / sd)
    per = []
    for b in np.unique(r_strata):
        cm, rm = c_strata == b, r_strata == b
        if cm.sum() >= min_n and rm.sum() >= min_n:
            per.append(energy_distance((C[cm] - mu) / sd, (R[rm] - mu) / sd))
    return g, (float(np.mean(per)) if per else np.nan)


def strata_of(t, edges):
    return np.digitize(t["range_m"], edges) * 2 + t["near_zero"].astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="carrada")
    a = ap.parse_args()
    ds = a.dataset
    from src.versions import DATASETS
    dev = torch.device("cuda")
    real = load(DATASETS[ds].real, "train")
    full_val = load(DATASETS[ds].real, "val")
    val = [full_val[int(i)] for i in np.linspace(0, len(full_val) - 1, N_VAL).round()]
    models = {f"{arch}_{f}": encoder(ds, arch, f, real, val, dev) for arch in ("cnn", "transformer") for f in (0, 1)}
    del full_val, val
    proj_dim = (2 * K_R + 1) * (2 * K_D + 1) + (K_R + 1) * (K_D + 1)
    proj = np.random.default_rng(0).normal(size=(64, proj_dim)).astype(np.float32) / np.sqrt(64)
    proj = torch.from_numpy(proj)
    radar = RADARS[ds]
    hooks = {k: _hooks(m, k.split("_")[0]) for k, m in models.items()}       # registered once
    R = embed(models, hooks, real, radar, dev, proj)
    fold = R["seq"] % 2
    out_path = OUT / f"{ds}.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    res["provenance"] = provenance()
    refs = {f"{b}_{s}": (subsets(b, 3)[s] if b < 50 else np.arange(50)) for b, s in REFERENCES}
    rows = [r for r in json.loads(REGISTRY.read_text())["candidates"] if r["dataset"] == ds and r["usable"]]
    for row in rows:
        cid = row["candidate"]
        if cid in res:
            continue
        t0 = time.time()
        pool, _ = load_candidate(row)
        C = embed(models, hooks, pool, radar, dev, proj)
        del pool
        out = {}
        for key, seqs in refs.items():
            in_ref = np.isin(R["seq"], seqs)
            edges = np.percentile(R["range_m"][in_ref], (100 / 3, 200 / 3)) if in_ref.sum() else [0, 0]
            rs, cs = strata_of(R, edges), strata_of(C, edges)
            d = {}
            for arch in ("cnn", "transformer"):
                g_, c_ = [], []
                for f in (0, 1):          # encoder trained on fold f compares with real objects of fold 1 - f
                    m = in_ref & (fold != f)
                    g, c = distance(C[f"emb_{arch}_{f}"], R[f"emb_{arch}_{f}"][m], cs, rs[m])
                    g_.append(g); c_.append(c)
                d[arch] = float(np.nanmean(g_)) if np.isfinite(g_).any() else None
                d[f"{arch}_conditional"] = float(np.nanmean(c_)) if np.isfinite(c_).any() else None
            g, c = distance(C["raw_patch"], R["raw_patch"][in_ref], cs, rs[in_ref])
            d["raw_patch"] = None if not np.isfinite(g) else float(g)
            d["raw_patch_conditional"] = None if not np.isfinite(c) else float(c)
            out[key] = d
        res[cid] = out | {"seconds": time.time() - t0}
        OUT.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res))
        print(f"{cid:24s} 5_0 {out['5_0']} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
