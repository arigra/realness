"""Figures and tables for the simulator and tolerance studies (notebooks/simulators_and_tolerances.ipynb;
docs/notes/2026-11-02-simulators.md, docs/notes/2026-11-02-study2b.md). Every figure reads the
results files the runs wrote; nothing here trains.

Colours: one fixed hue per dataset (CARRADA blue, RADDet orange, RADIal aqua), the first three
slots of the dataviz reference palette; real data is drawn in neutral ink.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import load, normalised
from src.detector import average_precision
from src.measures import range_cut
from src.tolerance import FAMILIES, readout
from src.versions import DATASETS, SIM2, best, render

DATASETS_ORDER = ("carrada", "raddet", "radial")
NAMES = {"carrada": "CARRADA", "raddet": "RADDet", "radial": "RADIal"}
COLOR = {"carrada": "#2a78d6", "raddet": "#eb6834", "radial": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#8a8984", "#e4e3df"
FIGDIR = Path("results/figures")


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.6, "lines.markersize": 5, "legend.frameon": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK})


def save(fig, name):
    """Write a figure for the paper: results/figures/<name>.pdf (and .png)."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{name}.png", bbox_inches="tight", dpi=200)


# ------------------------------------------------------------------ results files

def runs(suite, dataset, split="test", setting="only"):
    """{version: [per-seed scores]} of a suite; split 'test' or 'val' (the whole val split)."""
    p = Path(f"results/{suite}_{dataset}_runs.json")
    if not p.exists():
        return {}
    r = json.loads(p.read_text())
    block = r["runs"] if split == "test" else r.get("val", {})
    return {v: list(s.values()) for v, s in block.get(setting, {}).items()}


def best_version(dataset):
    """The sim2 version the pre-fixed rule chose (highest mean whole-val AP@2)."""
    val = runs("sim2", dataset, "val")
    return max((v for v in val if len(val[v]) == 3), key=lambda v: np.mean([s["ap2"] for s in val[v]]))


class Table:
    """A small table that shows as Markdown in a notebook and as text elsewhere."""

    def __init__(self, header, rows):
        self.header, self.rows = header, rows

    def _cell(self, v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    def _repr_markdown_(self):
        lines = ["| " + " | ".join(self.header) + " |", "|" + "---|" * len(self.header)]
        return "\n".join(lines + ["| " + " | ".join(self._cell(v) for v in r) + " |" for r in self.rows])

    def __repr__(self):
        w = [max(len(self._cell(r[i])) for r in [self.header] + self.rows) for i in range(len(self.header))]
        return "\n".join("  ".join(self._cell(v).ljust(n) for v, n in zip(r, w)) for r in [self.header] + self.rows)


def _dets(path):
    d = torch.load(path)
    return d["dets"], d["tgts"]


# ------------------------------------------------------------------ figures

def _with_labels(pool, start, t):
    """The first sequence from `start` on whose frame t has a labelled object."""
    for i in range(start, len(pool)):
        if bool(pool[i]["present"][:, t].any()):
            return pool[i]
    return pool[start]


def maps(datasets=DATASETS_ORDER, start=5, t=3):
    """One frame per source and dataset: real test, the spec simulator, the best tuned simulator
    (the first sequence from `start` with a labelled object in frame t). Maps are normalised
    (as the detector sees them); circles mark the labels."""
    style()
    fig, axs = plt.subplots(len(datasets), 3, figsize=(10, 3.4 * len(datasets)))
    axs = np.atleast_2d(axs)
    for row, ds in enumerate(datasets):
        D, n = DATASETS[ds], start + 20
        pools = {"real (test)": _with_labels(load(D.real, "test", limit=n), start, t),
                 "spec simulator": _with_labels(load(D.sim, "train", limit=n), start, t),
                 "tuned simulator": _with_labels(load(render(ds, "best", "tuned2", best(ds)), "train", limit=n), start, t)}
        for col, (title, it) in enumerate(pools.items()):
            ax = axs[row, col]
            ax.imshow(normalised(it, t).numpy(), origin="lower", aspect="auto", cmap="viridis", vmin=-2, vmax=5)
            lab = it["traj"][it["present"][:, t], t].numpy()
            ax.scatter(lab[:, 1], lab[:, 0], s=90, facecolors="none", edgecolors="white", linewidths=1.2)
            ax.set_title(f"{NAMES[ds]}: {title}")
            ax.grid(False)
            ax.set_xlabel("Doppler bin")
            ax.set_ylabel("range bin" if col == 0 else "")
    fig.tight_layout()
    return fig


def label_profile(datasets=DATASETS_ORDER, kr=12):
    """Mean normalised level along range through every label, real vs spec vs tuned simulator
    (training sequences). Where the label sits on the object's energy."""
    style()
    fig, axs = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.0), sharey=True)
    x = np.arange(-kr, kr + 1)
    for ax, ds in zip(np.atleast_1d(axs), datasets):
        D = DATASETS[ds]
        ax.plot(x, range_cut(load(D.real, "train"), kr).numpy(), color=INK, label="real")
        ax.plot(x, range_cut(load(D.sim, "train", limit=50), kr).numpy(), color=COLOR[ds], ls="--", label="spec sim")
        ax.plot(x, range_cut(load(render(ds, "best", "tuned2", best(ds)), "train", limit=50), kr).numpy(),
                color=COLOR[ds], label="tuned sim")
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(NAMES[ds])
        ax.set_xlabel("range cells from the label (+ = farther)")
    np.atleast_1d(axs)[0].set_ylabel("mean normalised level")
    np.atleast_1d(axs)[0].legend()
    fig.tight_layout()
    return fig


def radius(datasets=DATASETS_ORDER, radii=(2, 4, 8, 16)):
    """AP on real test against the match radius: real-only (50 sequences), the spec simulator
    and the best tuned simulator. A gap that closes with radius is localisation (where labels
    sit); one that stays is detection (what the data contains)."""
    style()
    fig, axs = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.0), sharey=True)
    for ax, ds in zip(np.atleast_1d(axs), datasets):
        n_d = DATASETS[ds].grid[1]
        b = best_version(ds)
        srcs = [("real only", INK, "-", f"results/budget_{ds}_real50_dets/real_clean_s{{}}.pt"),
                ("spec sim", COLOR[ds], "--", f"results/sim2_{ds}_dets/only_clean_s{{}}.pt"),
                ("tuned sim", COLOR[ds], "-", f"results/sim2_{ds}_dets/only_{b}_s{{}}.pt")]
        for name, c, ls, pat in srcs:
            if not Path(pat.format(1)).exists():
                continue
            ap = np.array([[average_precision(*_dets(pat.format(s)), n_d, radius=r) for r in radii] for s in (1, 2, 3)])
            ax.errorbar(radii, ap.mean(0), yerr=[ap.mean(0) - ap.min(0), ap.max(0) - ap.mean(0)],
                        color=c, ls=ls, marker="o", capsize=2, label=name)
        ax.set_xscale("log", base=2)
        ax.set_xticks(radii, [str(r) for r in radii])
        ax.set_title(NAMES[ds])
        ax.set_xlabel("match radius (cells)")
        ax.set_ylim(0, 1)
    np.atleast_1d(axs)[0].set_ylabel("AP on real test")
    np.atleast_1d(axs)[0].legend()
    fig.tight_layout()
    return fig


def ablation_table(dataset):
    """The sim2 ablation: whole-val and test AP@2 (mean over 3 seeds, seed range), AP@4, recall@2,
    sorted by whole-val AP@2 (what decided)."""
    val, test = runs("sim2", dataset, "val"), runs("sim2", dataset, "test")
    chosen, rows = best_version(dataset), []
    for v in test:
        if len(test[v]) < 3 or len(val.get(v, [])) < 3:
            continue
        a = lambda s, k: np.array([x[k] for x in s])
        rows.append([v + (" (chosen)" if v == chosen else ""), a(val[v], "ap2").mean(), np.ptp(a(val[v], "ap2")),
                     a(test[v], "ap2").mean(), np.ptp(a(test[v], "ap2")), a(test[v], "ap4").mean(),
                     a(test[v], "recall2").mean()])
    rows.sort(key=lambda r: -r[1])
    return Table(["version", "val AP@2", "val range", "test AP@2", "test range", "test AP@4", "test recall@2"], rows)


def ablation(datasets=DATASETS_ORDER):
    """Every fix left out in turn: whole-val AP@2 (filled, seed range as a bar; what decides)
    and test AP@2 (hollow; reported). Versions sorted by val."""
    style()
    fig, axs = plt.subplots(1, len(datasets), figsize=(4.0 * len(datasets), 3.4))
    for ax, ds in zip(np.atleast_1d(axs), datasets):
        rows = ablation_table(ds).rows[::-1]
        y = np.arange(len(rows))
        col = lambda k: np.array([r[k] for r in rows])
        ax.errorbar(col(1), y, xerr=col(2) / 2, fmt="o", color=COLOR[ds], capsize=2, label="whole val")
        ax.plot(col(3), y, "o", mfc="white", mec=COLOR[ds], label="test")
        ax.set_yticks(y, [r[0] for r in rows])
        ax.set_title(NAMES[ds])
        ax.set_xlabel("AP@2")
    h, l = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return fig


def budget(datasets=DATASETS_ORDER, ns=(1, 5, 10, 25, 50)):
    """Label budget: real only vs tuned simulator + real (half of each batch), against the number
    of real training sequences; the dotted line is the tuned simulator alone."""
    style()
    fig, axs = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.0), sharey=True)
    for ax, ds in zip(np.atleast_1d(axs), datasets):
        for setting, c, name in (("real", INK, "real only"), ("mix", COLOR[ds], "tuned sim + real")):
            pts = [(n, [s["ap2"] for s in runs(f"budget", f"{ds}_real{n}", "test", setting).get("clean", [])]) for n in ns]
            pts = [(n, np.array(x)) for n, x in pts if len(x) == 3]
            if pts:
                n_, x = zip(*pts)
                m = np.array([v.mean() for v in x])
                ax.errorbar(n_, m, yerr=[m - [v.min() for v in x], [v.max() for v in x] - m],
                            color=c, marker="o", capsize=2, label=name)
        sim = runs("study2", ds, "test").get("clean", [])
        if len(sim) == 3:
            ax.axhline(np.mean([s["ap2"] for s in sim]), color=COLOR[ds], ls=":", lw=1.2, label="tuned sim alone")
        ax.set_xscale("log")
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.set_title(NAMES[ds])
        ax.set_xlabel("real training sequences")
    np.atleast_1d(axs)[0].set_ylabel("AP@2 on real test")
    np.atleast_1d(axs)[0].legend()
    fig.tight_layout()
    return fig


def tolerance(datasets=DATASETS_ORDER, split="test", metric="ap2", families=tuple(FAMILIES), ncols=4):
    """Tolerance curves: the change in AP from the dataset's clean (best tuned) simulator at each
    level of a family, one line per dataset. Filled markers: more than that dataset's seed noise
    below clean (the pre-registered onset rule reads the first filled level after which all stay
    filled). Dotted lines: minus each dataset's noise."""
    style()
    res = {d: readout(d, split, metric) for d in datasets}
    nrows = int(np.ceil(len(families) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), sharey=True)
    for ax, f in zip(axs.flat, families):
        names = FAMILIES[f]
        x = np.arange(len(names) + 1)
        for d in datasets:
            r = res[d]
            m = [r["clean"]] + [r["families"][f]["means"].get(n) for n in names]
            ok = [i for i, v in enumerate(m) if v is not None]
            dm = np.array([m[i] - r["clean"] for i in ok])
            ax.plot(x[ok], dm, color=COLOR[d], label=NAMES[d])
            below = dm < -r["noise"]
            ax.plot(x[ok][below], dm[below], "o", color=COLOR[d])
            ax.plot(x[ok][~below], dm[~below], "o", mfc="white", mec=COLOR[d])
            ax.axhline(-r["noise"], color=COLOR[d], ls=":", lw=0.8)
        ax.axhline(0, color=MUTED, lw=0.8)
        prefix = f.rstrip("+-")
        ax.set_xticks(x, ["clean"] + [n.replace(prefix, "") for n in names], fontsize=7)
        ax.set_title(f)
    for ax in axs.flat[len(families):]:
        ax.set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel(f"{metric.upper()} - clean")
    axs.flat[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def onsets(datasets=DATASETS_ORDER, split="test", metric="ap2"):
    """Onset per family and dataset (None: no onset; incomplete: runs still missing)."""
    res = {d: readout(d, split, metric) for d in datasets}
    rows = [[f] + [str(res[d]["families"][f]["onset"]) for d in datasets] for f in FAMILIES]
    rows.append(["(clean AP, noise)"] + [f"{res[d]['clean']:.2f}, {res[d]['noise']:.2f}" for d in datasets])
    return Table(["family"] + [NAMES[d] for d in datasets], rows)
