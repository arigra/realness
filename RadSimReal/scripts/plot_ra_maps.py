"""Side-by-side range-azimuth maps: real RADDet train frames vs synthetic frames, same dB scale.

Car boxes (metres, sensor frame) are mapped to RA bins through the radar config's own
coordinate transform, so both rows are drawn identically.

  python scripts/plot_ra_maps.py --synthetic runs/raddet_carla_synthetic_vehicle --out figs/ra_maps.png
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
from radsimreal.radar.coordinates import points_to_indices


def box_to_ra(box, radar):
    x0, y0, x1, y1 = box
    xs, ys = np.meshgrid(np.linspace(x0, x1, 9), np.linspace(y0, y1, 9))
    pts = np.c_[xs.ravel(), ys.ravel(), np.zeros(xs.size)]
    idx, _ = points_to_indices(pts, np.zeros_like(pts), radar)
    return idx[:, 2].min(), idx[:, 0].min(), idx[:, 2].max(), idx[:, 0].max()  # az0, r0, az1, r1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="runs/raddet_prepared_v")
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    radar = load_config(Path(args.synthetic) / "config.json").radar
    rng = np.random.default_rng(args.seed)
    man = json.loads((Path(args.real) / "manifest.json").read_text())
    real_rows = [man["splits"]["train"][i] for i in rng.choice(len(man["splits"]["train"]), args.n, replace=False)]
    syn_files = sorted(glob.glob(str(Path(args.synthetic) / "train" / "train_*.json")))
    syn_files = [syn_files[i] for i in rng.choice(len(syn_files), args.n, replace=False)]
    panels = []
    for r in real_rows:
        img = np.load(Path(args.real) / "ra_cache" / (r["id"].replace("/", "__") + ".npy"))
        lab = json.loads((Path(args.real) / r["prepared_annotation"]).read_text())
        boxes = [b for b, l in zip(lab["boxes_bev"], lab["labels"]) if l in ("car", "truck", "bus")]
        panels.append(("real " + r["sequence"], img, boxes))
    for f in syn_files:
        img = np.load(f.replace(".json", ".npz"))["image"]
        lab = json.loads(Path(f).read_text())
        src = lab["source_capture"]; meta = lab["carla_meta"]
        panels.append((f"synthetic {src.split('/')[0]} ego {meta.get('ego_speed_m_s', 0):.1f} m/s", img,
                       [o["box_bev"] for o in lab["objects"]]))
    vmin, vmax = 35, 85
    fig, axes = plt.subplots(2, args.n, figsize=(4 * args.n, 8.4), constrained_layout=True)
    for ax, (title, img, boxes) in zip(axes.ravel(), panels):
        im = ax.imshow(10 * np.log10(np.maximum(img, 1e-12)), origin="lower", aspect="auto", vmin=vmin, vmax=vmax,
                       cmap="viridis")
        for b in boxes:
            a0, r0, a1, r1 = box_to_ra(b, radar)
            ax.add_patch(plt.Rectangle((a0, r0), a1 - a0, r1 - r0, fill=False, ec="red", lw=1.2))
        ax.set_title(title, fontsize=10); ax.set_xlabel("azimuth bin"); ax.set_ylabel("range bin (0.195 m)")
    fig.colorbar(im, ax=axes, shrink=0.6, label="dB")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
