"""Range-azimuth maps drawn like the paper's Fig. 1: range in metres (cropped), azimuth in degrees,
jet colormap, each image scaled to its own dB range (median .. max), as the paper's panels appear to be.
Top row real RADDet, bottom row synthetic.

  python scripts/plot_ra_paper_style.py --synthetic runs/raddet_carla_synthetic_vehicle --out runs/figs/ra_paper_style.png
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from radsimreal.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="runs/raddet_prepared_v")
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--rmin", type=float, default=5.0); ap.add_argument("--rmax", type=float, default=35.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    radar = load_config(Path(args.synthetic) / "config.json").radar
    az = np.degrees(np.arcsin(np.clip(np.asarray(radar.angular_grid), -1, 1)))
    rng_m = radar.range_min + np.arange(radar.range_bins) * radar.range_step
    keep = (rng_m >= args.rmin) & (rng_m <= args.rmax)
    rng = np.random.default_rng(args.seed)
    man = json.loads((Path(args.real) / "manifest.json").read_text())
    train = man["splits"]["train"]
    real = [(f"real RADDet {r['sequence']}", np.load(Path(args.real) / "ra_cache" / (r["id"].replace("/", "__") + ".npy")))
            for r in (train[i] for i in rng.choice(len(train), args.n, replace=False))]
    files = sorted(glob.glob(str(Path(args.synthetic) / "train" / "train_*.json")))
    syn = []
    for i in rng.choice(len(files), args.n, replace=False):
        meta = json.loads(Path(files[i]).read_text())
        syn.append((f"synthetic {meta['source_capture'].split('/')[0]}", np.load(files[i].replace(".json", ".npz"))["image"]))
    fig, axes = plt.subplots(2, args.n, figsize=(4.2 * args.n, 8), constrained_layout=True)
    for ax, (title, img) in zip(axes.ravel(), real + syn):
        d = 10 * np.log10(np.maximum(img[keep], 1e-12))
        ax.pcolormesh(az, rng_m[keep], d, cmap="jet", vmin=np.median(d), vmax=d.max(), shading="auto")
        ax.set_title(title, fontsize=10); ax.set_xlabel("Azimuth [deg]"); ax.set_ylabel("Range [m]")
        ax.set_xlim(-90, 90)
    fig.suptitle("Paper-style display: jet, each panel scaled median..max dB")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
