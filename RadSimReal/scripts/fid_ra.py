"""FID between sets of range-azimuth maps (paper Sec. 3.1 reports RADDet train vs test 6.76 and
RADDet train vs RadSimReal 6.54; it does not state how maps were turned into images).

Image rule (ours, identical for every set): linear power -> dB -> fixed window [lo, hi] dB ->
8-bit grey -> 3 channels; pytorch-fid Inception-v3 pool features (resized to 299 inside the model).
A second rule, per-image min-max normalisation, is reported too, to show sensitivity to that choice.
Equal sample sizes per comparison (FID is biased by sample size). Only image statistics are used;
no labels, no model selection.

  python scripts/fid_ra.py --n 1962 --out runs/figs/fid.json
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance


def real_paths(root, split):
    man = json.loads((Path(root) / "manifest.json").read_text())
    return [Path(root) / "ra_cache" / (r["id"].replace("/", "__") + ".npy") for r in man["splits"][split]]


def load(path):
    p = str(path)
    return np.load(p) if p.endswith(".npy") else np.load(p)["image"]


def to_grey(img, rule, lo, hi):
    d = 10 * np.log10(np.maximum(img, 1e-12))
    if rule == "fixed":
        d = (d - lo) / (hi - lo)
    else:
        d = (d - d.min()) / max(d.max() - d.min(), 1e-6)
    return np.round(np.clip(d, 0, 1) * 255) / 255.0


@torch.no_grad()
def features(paths, model, device, rule, lo, hi, batch=50):
    out = []
    for i in range(0, len(paths), batch):
        g = np.stack([to_grey(load(p), rule, lo, hi) for p in paths[i:i + batch]]).astype(np.float32)
        x = torch.from_numpy(g)[:, None].repeat(1, 3, 1, 1).to(device)
        out.append(model(x)[0].squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(out)


def stats(f):
    return f.mean(0), np.cov(f, rowvar=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="runs/raddet_prepared_v")
    ap.add_argument("--synthetic", nargs="+", default=["runs/raddet_carla_synthetic_vehicle:car-mounted",
                                                       "runs/raddet_carla_synthetic:sidewalk"])
    ap.add_argument("--n", type=int, default=1962)
    ap.add_argument("--lo", type=float, default=30.0); ap.add_argument("--hi", type=float, default=90.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    pick = lambda xs: [xs[i] for i in np.sort(rng.choice(len(xs), min(args.n, len(xs)), replace=False))]
    sets = {"real train": pick(real_paths(args.real, "train")), "real test": pick(real_paths(args.real, "test"))}
    for spec in args.synthetic:
        path, label = spec.split(":")
        sets[label] = pick(sorted(glob.glob(str(Path(path) / "train" / "train_*.npz"))))
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(args.device).eval()
    result = {"n_per_set": {k: len(v) for k, v in sets.items()}, "window_db": [args.lo, args.hi],
              "paper": {"RADDet train vs test": 6.76, "RADDet train vs RadSimReal": 6.54}}
    for rule in ("fixed", "per_image"):
        S = {k: stats(features(v, model, args.device, rule, args.lo, args.hi)) for k, v in sets.items()}
        ref = S["real train"]
        result[rule] = {f"real train vs {k}": float(calculate_frechet_distance(*ref, *S[k])) for k in S if k != "real train"}
        print(rule, json.dumps(result[rule], indent=1), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
