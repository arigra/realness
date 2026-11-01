"""Plots for the damage study (docs/notes/2026-10-28-damage-spread-fair.md):
the synthetic maps and their labels, each defect next to the clean scene, the
detectors' test scores, and what trained detectors find on real test frames.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import load
from src.versions import DATASETS, FAIR, SUITES, version as _version

FAMILIES = SUITES["fair"][1] + SUITES["fair"][2]
RANGE_RES_M = 0.2               # both datasets


def _results(dataset, kind):
    """results/fair_<dataset>_runs.json, _verdict.json or _dets."""
    return Path("results") / f"fair_{dataset}_{kind}"


def family(name):
    """Version names of a family, clean first, in increasing damage."""
    return ["clean"] + [n for n in FAIR if n.startswith(name)]


def version(dataset, name, n=16):
    """The first n sequences of one version, as the study trains on them."""
    clean = load(DATASETS[dataset].sim, "train", limit=n)
    return _version(dataset, name, *FAIR[name], clean, n=n)


def real(dataset, split="train", n=16):
    return load(DATASETS[dataset].real, split, limit=n)


def labels(item, t):
    """(range_bin, doppler_bin) of the objects labelled in frame t."""
    return item["traj"][item["present"][:, t], t].float().numpy()


def _show(ax, m, lo, hi, marks=(), res=0.2, cmap="viridis", window=None):
    m = np.asarray(m, dtype=np.float32)
    ax.imshow(m, aspect="auto", origin="lower", vmin=lo, vmax=hi, cmap=cmap,
              extent=[0, m.shape[1], 0, m.shape[0] * res])
    for rb, db in marks:
        ax.plot(db, rb * res, "o", mfc="none", mec="red", ms=11, mew=1.5)
    if window is not None:
        (r0, r1), (d0, d1) = window
        ax.set_xlim(d0, d1)
        ax.set_ylim(r0 * res, r1 * res)


def _window(item, t, zoom, shape):
    if zoom is None or not len(labels(item, t)):
        return None
    r, d = labels(item, t)[0]
    return ((max(r - zoom, 0), min(r + zoom, shape[0])),
            (max(d - zoom // 2, 0), min(d + zoom // 2, shape[1])))


def show_real_and_sim(dataset, seqs=(0, 1, 2, 3), frame=0):
    """Real training maps (top) and clean simulator maps (bottom), red = labels.
    Each panel on its own scale: the two sources differ in absolute level."""
    rows = {"real": real(dataset, n=max(seqs) + 1), "simulator": version(dataset, "clean", max(seqs) + 1)}
    fig, axes = plt.subplots(2, len(seqs), figsize=(3.6 * len(seqs), 7), squeeze=False)
    for r, (label, items) in enumerate(rows.items()):
        for c, s in enumerate(seqs):
            m = items[s]["x"][frame].float().numpy()
            _show(axes[r][c], m, *np.percentile(m, (5, 99.5)), labels(items[s], frame), RANGE_RES_M)
            axes[r][c].set_title(f"{label}, sequence {s}, {len(labels(items[s], frame))} labelled")
        axes[r][0].set_ylabel("range (m)")
    for ax in axes[-1]:
        ax.set_xlabel("Doppler bin")
    fig.suptitle(f"{dataset}: frame {frame}; red circles = labels")
    fig.tight_layout()
    plt.show()


def show_sequence(dataset, name, seq, zoom=None):
    """All 8 frames of one sequence of one version, red = labels."""
    item = version(dataset, name, seq + 1)[seq]
    x = item["x"].float().numpy()
    lo, hi = np.percentile(x, (5, 99.7))
    fig, axes = plt.subplots(1, x.shape[0], figsize=(2.6 * x.shape[0], 3.4))
    for t, ax in enumerate(axes):
        _show(ax, x[t], lo, hi, labels(item, t), RANGE_RES_M, window=_window(item, 0, zoom, x.shape[1:]))
        ax.set_title(f"frame {t}")
    axes[0].set_ylabel("range (m)")
    fig.suptitle(f"{dataset}, {name}, sequence {seq}")
    fig.tight_layout()
    plt.show()


def pick_sequence(dataset, name, n=16, frame=0):
    """Among the first n sequences, the one with labels in `frame` that the defect
    `name` changes most -- so the change is visible."""
    clean, bad = version(dataset, "clean", n), version(dataset, name, n)
    best, score = 0, -1.0
    for s, (a, b) in enumerate(zip(clean, bad)):
        if not len(labels(a, frame)):
            continue
        d = float((b["x"][frame].float() - a["x"][frame].float()).abs().max())
        d += float((b["traj"] - a["traj"]).abs().sum())
        if d > score:
            best, score = s, d
    return best


def show_family(dataset, fam, seq=None, frame=0, zoom=None):
    """Top: the same scene, clean and at each damage level, on ONE colour scale
    (the clean map's), red = the labels the detector is trained with.
    Bottom: damaged minus clean, in dB (red = more power, blue = less)."""
    names = family(fam)
    if seq is None:
        seq = pick_sequence(dataset, names[-1], frame=frame)
    items = [version(dataset, n, seq + 1)[seq] for n in names]
    maps = [it["x"][frame].float().numpy() for it in items]
    lo, hi = np.percentile(maps[0], (5, 99.7))
    win = _window(items[0], frame, zoom, maps[0].shape)
    fig, axes = plt.subplots(2, len(names), figsize=(3.4 * len(names), 7), squeeze=False)
    for c, (n, it, m) in enumerate(zip(names, items, maps)):
        _show(axes[0][c], m, lo, hi, labels(it, frame), RANGE_RES_M, window=win)
        axes[0][c].set_title(n)
        diff = m - maps[0]
        _show(axes[1][c], diff, -12, 12, (), RANGE_RES_M, cmap="RdBu_r", window=win)
        axes[1][c].set_title(f"minus clean (max |change| {np.abs(diff).max():.0f} dB)", fontsize=9)
        axes[1][c].set_xlabel("Doppler bin")
    axes[0][0].set_ylabel("range (m)")
    axes[1][0].set_ylabel("range (m)")
    fig.suptitle(f"{dataset}: {fam}, sequence {seq}, frame {frame}")
    fig.tight_layout()
    plt.show()


def load_results(dataset):
    path = _results(dataset, "runs.json")
    return json.loads(path.read_text()) if path.exists() else None


def show_results(dataset, metric="ap2"):
    """One panel per family: real-test score of detectors trained on each version.
    Dots = the 3 seeds, line = their mean; grey band = clean mean +/- seed noise."""
    rep = load_results(dataset)
    if rep is None:
        print(f"{dataset}: no results yet")
        return
    fig, axes = plt.subplots(2, len(FAMILIES), figsize=(2.9 * len(FAMILIES), 6.4), sharey="row")
    for r, setting in enumerate(("only", "mix")):
        runs = rep["runs"].get(setting, {})
        vals = {n: [v[metric] for v in s.values()] for n, s in runs.items()}
        if not vals:
            continue
        noise = float(np.median([np.ptp(v) for v in vals.values()]))
        for c, fam in enumerate(FAMILIES):
            ax = axes[r][c]
            names = [n for n in family(fam) if n in vals]
            xs = np.arange(len(names))
            clean = np.mean(vals["clean"]) if "clean" in vals else np.nan
            ax.axhspan(clean - noise, clean + noise, color="0.88")
            for i, n in enumerate(names):
                ax.plot([i] * len(vals[n]), vals[n], "o", color="C0", alpha=0.5, ms=4)
            ax.plot(xs, [np.mean(vals[n]) for n in names], "-", color="C0", lw=2)
            ax.set_xticks(xs, [n.replace(fam, "") or n for n in names], rotation=45, fontsize=8)
            ax.set_title(fam if r == 0 else "")
        axes[r][0].set_ylabel(f"{setting}\n{metric} on real test")
    fig.suptitle(f"{dataset}: {metric}; top: synthetic only, bottom: 50 real + synthetic")
    fig.tight_layout()
    plt.show()


def print_verdict(dataset):
    path = _results(dataset, "verdict.json")
    if not path.exists():
        print(f"{dataset}: no verdict yet")
        return
    v = json.loads(path.read_text())
    print(f"{'score':8} {'setting':7} {'gap':>6} {'noise':>6} {'gap/noise':>9}  ordered families      pass")
    for metric, by_setting in v.items():
        for setting, x in by_setting.items():
            fams = [f for f in x["counted"] if x["families"][f]["ordered"]]
            print(f"{metric:8} {setting:7} {x['gap']:6.3f} {x['noise']:6.3f} {x['gap_over_noise']:9.1f}"
                  f"  {len(fams)}/5 {','.join(fams):16} {x['pass']}")


def show_detections(dataset, names, seq=None, frame=0, seed=1, setting="only", min_score=0.2, zoom=None):
    """One real test frame, as each detector sees it. Red circles = real labels,
    white crosses = the detector's detections (score >= min_score, written next
    to them). The detector was trained on the version named above the panel."""
    test = real(dataset, "test", n=None)
    if seq is None:
        seq = next(s for s, it in enumerate(test) if len(labels(it, frame)) >= 2)
    item = test[seq]
    m = item["x"][frame].float().numpy()
    lo, hi = np.percentile(m, (5, 99.5))
    idx = sum(it["x"].shape[0] for it in test[:seq]) + frame
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 4), squeeze=False)
    for ax, n in zip(axes[0], names):
        f = _results(dataset, "dets") / f"{setting}_{n}_s{seed}.pt"
        _show(ax, m, lo, hi, labels(item, frame), RANGE_RES_M,
              window=_window(item, frame, zoom, m.shape))
        if f.exists():
            for score, r, d in torch.load(f, weights_only=False)["dets"][idx]:
                if score >= min_score:
                    ax.plot(d, r * RANGE_RES_M, "x", color="white", ms=9, mew=2)
                    ax.text(d + 1, r * RANGE_RES_M, f"{score:.2f}", color="white", fontsize=7, clip_on=True)
        ax.set_title(f"trained on {n}" if f.exists() else f"{n}: not run yet")
        ax.set_xlabel("Doppler bin")
    axes[0][0].set_ylabel("range (m)")
    fig.suptitle(f"{dataset} real test sequence {seq}, frame {frame} ({setting}, seed {seed})")
    fig.tight_layout()
    plt.show()


def show_score_vs_ap(fits, metric="ap2"):
    """Realness score against synthetic-only AP, one panel per (dataset, score).
    fits: {dataset: {kind: Score}} (src/score.py); the CARRADA-fitted scores are
    applied unchanged to RADIal. Labels mark every version; rho = Spearman of -score with AP."""
    from src.score import points, rho
    rows = [(ds, kind) for ds, by in fits.items() for kind in by]
    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 4), squeeze=False)
    for ax, (ds, kind) in zip(axes[0], rows):
        p = points(ds, metric)
        names = list(p)
        D, ap = np.stack([p[n][0] for n in names]), np.array([p[n][1] for n in names])
        s = fits[ds][kind](D)
        ax.plot(s, ap, "o", color="C0")
        for x, y, n in zip(s, ap, names):
            ax.annotate(n, (x, y), fontsize=6, xytext=(2, 2), textcoords="offset points")
        r, ci = rho(s, ap)
        ax.set_title(f"{ds}, {kind}: rho {r:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]", fontsize=9)
        ax.set_xlabel("score (higher = less real)")
        ax.set_ylabel(f"{metric}, synthetic only")
    fig.tight_layout()
    plt.show()
