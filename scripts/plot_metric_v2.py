"""Metric v2 figures (docs/notes/2026-11-06-metric-v2-preregistration.md, "Required outputs").

prospective.{pdf,png}   PRIMARY: bank-v2 test AP@2 per candidate (seed range), each method's selection marked
risk_profile.{pdf,png}  M4 block risks (detectability ... processing) and M5 FN / FP coverage risks per candidate
stability.{pdf,png}     test regret of each budget's selection vs number of real sequences (finalist, proxy, global)
development.{pdf,png}   DEVELOPMENT (val AP): leave-one-radar-out natural and leave-one-family-out regret per method
Palette: the validated reference categorical slots 1-3 (blue, orange, aqua) + neutral grays; risks on one blue ramp.

    PYTHONPATH=. python3 scripts/plot_metric_v2.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results/metric_v2/figures")
P = Path("results/metric_v2/prospective")
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DATASETS = ("carrada", "raddet")
METHODS = ("global", "M1", "M2", "M3", "M4", "M5", "M6", "proxy")

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def short(c):
    return c.split(":")[1].replace("_", " ")


def prospective(reps):
    fig, axes = plt.subplots(1, len(reps), figsize=(5.6 * len(reps), 3.9), squeeze=False,
                             gridspec_kw={"wspace": 0.45})
    for ax, (ds, r) in zip(axes[0], reps.items()):
        names = sorted(r["test_ap2"], key=lambda c: -r["test_ap2"][c])
        sel = json.loads((P / f"{ds}_runs.json").read_text())["runs"]
        y = np.arange(len(names))
        ap = np.array([r["test_ap2"][c] for c in names])
        lo = np.array([min(s["ap2"] for s in sel[c].values()) for c in names])
        hi = np.array([max(s["ap2"] for s in sel[c].values()) for c in names])
        ax.barh(y, ap, height=0.55, color=S1, edgecolor=SURF, linewidth=2)
        ax.errorbar(ap, y, xerr=[ap - lo, hi - ap], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
        ax.axvspan(ap.max() - r["noise"], ap.max(), color=GRID, zorder=0)
        for k, c in enumerate(names):
            who = [m for m in METHODS if r["methods"][m]["selected"] == c]
            if who:
                ax.text(hi[k] + 0.006, k, " ".join(who), va="center", fontsize=7.5, color=INK)
        ax.set_yticks(y, [short(c) for c in names])
        ax.invert_yaxis()
        ax.set_xlabel("real-test AP@2, synthetic-only (mean of 3 seeds; whiskers: seed range;\n"
                      "gray: within seed noise of the best); labels: methods that selected it")
        ax.set_title(f"{ds.upper()} bank v2: PROSPECTIVE result", fontsize=9, loc="left", color=INK)
        ax.set_xlim(0, max(hi) * 1.35)
    save(fig, "prospective")


def risk_profile(sels):
    blocks = ["A", "B", "C", "D", "E", "F"]
    labels = ["detect.", "label", "morph.", "support", "hard neg.", "process.", "M5 FN", "M5 FP"]
    fig, axes = plt.subplots(1, len(sels), figsize=(5.8 * len(sels), 3.6), squeeze=False,
                             gridspec_kw={"wspace": 0.55})
    for ax, (ds, s) in zip(axes[0], sels.items()):
        prim = s["references"]["10_0"]["candidates"]
        names = list(prim)
        M = np.array([[prim[c]["M4"]["blocks"][b]["risk"] if prim[c]["M4"]["blocks"][b]["risk"] is not None else np.nan
                       for b in blocks] + [np.nan if prim[c]["M5"][k] is None else prim[c]["M5"][k] for k in ("fn_risk", "fp_risk")]
                      for c in names], dtype=float)
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, "-" if not np.isfinite(v) else f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color=SURF if np.isfinite(v) and v > 0.6 else INK)
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(names)), [short(c) for c in names])
        ax.set_title(f"{ds.upper()}: risk profile (reference 10 sequences)", fontsize=9, loc="left", color=INK)
        ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="risk (0 = inside the safe range)")
    save(fig, "risk_profile")


def stability(reps):
    fig, axes = plt.subplots(1, len(reps), figsize=(5.0 * len(reps), 3.3), squeeze=False,
                             gridspec_kw={"wspace": 0.4})
    for ax, (ds, r) in zip(axes[0], reps.items()):
        fin = r["finalist"]
        for m, col, mk in ((fin, S1, "o"), ("proxy", S2, "s"), ("global", S3, "^"), ("M4", S4, "D")):
            b = r["budgets"][m]["mean_regret_by_budget"]
            xs = [5, 10, 20, 50]
            ax.plot(xs, [b[str(x)] if str(x) in b else b[x] for x in xs], color=col, marker=mk, lw=2, ms=6,
                    label=f"{m} (finalist)" if m == fin else m)
        ax.axhline(r["random"]["regret"], color=INK2, lw=1, ls="--")
        ax.text(50, r["random"]["regret"], " random", va="bottom", ha="right", fontsize=7.5, color=INK2)
        ax.axhspan(0, r["noise"], color=GRID, zorder=0)
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 50], ["5", "10", "20", "50"])
        ax.set_xlabel("real training sequences in the reference")
        ax.set_ylabel("test regret (mean over 3 subsets;\ngray: within seed noise)")
        ax.set_title(f"{ds.upper()}: selection vs reference size", fontsize=9, loc="left")
        ax.legend(frameon=False, fontsize=7.5)
    save(fig, "stability")


def development(dev):
    prim = dev["references"]["10_0"]
    methods = ["global", "M1", "M2", "M3", "M4", "M5", "M6", "proxy_v1", "random"]
    radars = ["carrada", "raddet", "radial"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, (title, get) in zip(axes, (
            ("leave one radar out, natural candidates", lambda d, m: prim["loro"][d]["natural"][m]["regret"]),
            ("leave one family out (mean over families)", lambda d, m: prim["lofo"][d]["summary"][m]["regret"]))):
        x = np.arange(len(methods))
        for k, (d, col) in enumerate(zip(radars, (S1, S2, S3))):
            v = [get(d, m) for m in methods]
            ax.bar(x + (k - 1) * 0.27, v, width=0.25, color=col, edgecolor=SURF, linewidth=1, label=d.upper())
            for i, y in enumerate(v):
                if y == 0:
                    ax.text(x[i] + (k - 1) * 0.27, 0.001, "0", ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xticks(x, [m.replace("_v1", "") for m in methods], rotation=30)
        ax.set_ylabel("val-AP@2 regret")
        ax.set_title(f"DEVELOPMENT (val AP): {title}", fontsize=9, loc="left")
        ax.legend(frameon=False, fontsize=7.5)
    save(fig, "development")


def main():
    dev = Path("results/metric_v2/scores/development.json")
    if dev.exists():
        development(json.loads(dev.read_text()))
    sels = {d: json.loads((P / f"{d}_selection.json").read_text()) for d in DATASETS if (P / f"{d}_selection.json").exists()}
    if sels:
        risk_profile(sels)
    reps = {d: json.loads((P / f"{d}_report.json").read_text()) for d in DATASETS if (P / f"{d}_report.json").exists()}
    if reps:
        prospective(reps)
        stability(reps)


if __name__ == "__main__":
    main()
