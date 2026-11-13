"""Real vs synthetic range-azimuth map statistics, one figure.

Panels: (1) mean dB vs range, (2) mean dB vs azimuth, (3) pooled dB histogram,
(4) car-box distance histogram, (5) cars per frame. Also prints a summary table.

  python scripts/compare_ra_statistics.py --synthetic runs/raddet_carla_synthetic_vehicle:car-mounted \
      runs/raddet_carla_synthetic:sidewalk --out runs/figs/ra_statistics.png
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

RANGE_STEP = 0.1953125


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-12))


def load_real(root, n, rng):
    man = json.loads((Path(root) / "manifest.json").read_text())
    rows = man["splits"]["train"]
    imgs, boxes = [], []
    for i in np.sort(rng.choice(len(rows), n, replace=False)):
        r = rows[i]
        imgs.append(np.load(Path(root) / "ra_cache" / (r["id"].replace("/", "__") + ".npy")))
        lab = json.loads((Path(root) / r["prepared_annotation"]).read_text())
        boxes.append([b for b, l in zip(lab["boxes_bev"], lab["labels"]) if l in ("car", "truck", "bus")])
    return db(np.stack(imgs)), boxes


def load_syn(root, n, rng):
    files = sorted(glob.glob(str(Path(root) / "train" / "train_*.json")))
    imgs, boxes = [], []
    for i in np.sort(rng.choice(len(files), n, replace=False)):
        imgs.append(np.load(files[i].replace(".json", ".npz"))["image"])
        boxes.append([o["box_bev"] for o in json.loads(Path(files[i]).read_text())["objects"]])
    return db(np.stack(imgs)), boxes


def box_distances(boxes):
    out = []
    for frame in boxes:
        for b in frame:
            out.append(np.hypot((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="runs/raddet_prepared_v")
    ap.add_argument("--synthetic", nargs="+", required=True, help="path:label ...")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    sets = [("real RADDet",) + load_real(args.real, args.n, rng)]
    for spec in args.synthetic:
        path, label = spec.split(":")
        sets.append((label,) + load_syn(path, args.n, rng))
    colors = ["black", "tab:orange", "tab:blue", "tab:green"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    real_db = sets[0][1]
    print(f"{'set':<14}{'p1':>6}{'p50':>6}{'p90':>6}{'p99':>6}{'p99.9':>7}{'W1 vs real':>11}{'cars/frame':>11}{'empty%':>8}{'car dist med':>13}")
    for (label, d, boxes), c in zip(sets, colors):
        rr = np.arange(d.shape[1]) * RANGE_STEP
        ax[0, 0].plot(rr, d.mean(axis=(0, 2)), color=c, label=label)
        ax[0, 1].plot(np.arange(d.shape[2]), d.mean(axis=(0, 1)), color=c, label=label)
        ax[0, 2].hist(d[:, ::2, ::2].ravel(), bins=np.arange(20, 100, 1), density=True, histtype="step", color=c, label=label)
        dist = box_distances(boxes)
        ax[1, 0].hist(dist, bins=np.arange(0, 52, 2), density=True, histtype="step", color=c, label=label)
        n = np.array([len(b) for b in boxes])
        ax[1, 1].hist(n, bins=np.arange(-0.5, 12.5, 1), density=True, histtype="step", color=c, label=label)
        # near/far brightness: mean dB in 0-10 m vs 20-45 m
        near, far = d[:, : int(10 / RANGE_STEP)].mean(), d[:, int(20 / RANGE_STEP): int(45 / RANGE_STEP)].mean()
        ax[1, 2].bar(label, far - near, color=c)
        q = np.percentile(d, [1, 50, 90, 99, 99.9])
        w = wasserstein_distance(real_db[:, ::4, ::4].ravel(), d[:, ::4, ::4].ravel())
        print(f"{label:<14}" + "".join(f"{v:6.1f}" for v in q[:4]) + f"{q[4]:7.1f}{w:11.2f}{n.mean():11.2f}{100*np.mean(n==0):8.0f}{np.median(dist):13.1f}")
    ax[0, 0].set(xlabel="range (m)", ylabel="mean dB", title="Mean brightness vs range")
    ax[0, 1].set(xlabel="azimuth bin", ylabel="mean dB", title="Mean brightness vs azimuth")
    ax[0, 2].set(xlabel="dB", ylabel="density", title="Pixel brightness distribution")
    ax[1, 0].set(xlabel="distance (m)", ylabel="density", title="Labelled car distance")
    ax[1, 1].set(xlabel="cars per frame", ylabel="density", title="Cars per frame")
    ax[1, 2].set(ylabel="dB", title="Far (20-45 m) minus near (0-10 m) brightness")
    ax[1, 2].axhline(0, color="gray", lw=0.8)
    for a in ax.ravel()[:5]:
        a.legend(fontsize=9)
    fig.suptitle(f"Real RADDet vs synthetic, {args.n} train frames each")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
