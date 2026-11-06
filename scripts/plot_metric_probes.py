"""Diagnostic figure of the Phase 1 probes (docs/notes/2026-11-06-metric-experiments.md).

Rows: datasets; columns: one probe per adequacy component that the controlled interventions are
meant to move. Each candidate (dot, 90% bootstrap over its sequences) against the real reference:
the line is all 50 real training sequences, the bands are the 5-95% range over random
5-sequence (light) and 10-sequence (dark) reference subsets: can a few labelled sequences resolve
the difference? Reads the raw tables of scripts/build_metric_features.py.
Writes results/metric/figures/probes.{pdf,png} and results/metric/probes_summary.json.

    PYTHONPATH=. python3 scripts/plot_metric_probes.py
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.build_metric_features import OUT, candidate_dir, load_raw, resample
from src.figures import COLOR, INK, MUTED, NAMES, style
from src.provenance import provenance
from src.utility_metric import PRIMARY_Q, rates, thresholds

CANDIDATES = ["spec", "tuned", "study2:rangeshift1", "study2:dopplershift1", "study2:extra50",
              "study2:unrender90", "study2:contrast-12", "study2:blur2", "study2b:tail2"]
SHORT = {"spec": "spec", "tuned": "tuned", "study2:rangeshift1": "range +1", "study2:dopplershift1": "Doppler +1",
         "study2:extra50": "extra 50%", "study2:unrender90": "unrender 90%", "study2:contrast-12": "contrast -12",
         "study2:blur2": "blur 2", "study2b:tail2": "tail 2"}
# (probe, label, statistic): statistic of the objects column, or a threshold rate
PROBES = [("peak_dr", "label offset in range\n(strongest cell - label, cells)", "median"),
          ("peak_dd", "label offset in Doppler\n(cells)", "median"),
          ("labels_with_peak_r2", "labels with a target-like\npeak within 2 cells", "rate"),
          ("unlabelled_off_ridge_per_frame", "unlabelled target-like peaks\noff the ridge, per frame", "rate"),
          ("label_prom", "target prominence\n(z over range-row median)", "median"),
          ("width_r2", "target range width\n(cells within 2 z of peak)", "mean"),
          ("far_ext2", "target extent behind peak\n(cells within 2 z)", "mean")]


def value(tables, probe, stat, thr):
    if stat == "rate":
        return rates(tables, thr)[probe]
    v = tables["objects"][probe]
    v = v[np.isfinite(v)]
    return float(np.median(v) if stat == "median" else np.mean(v)) if len(v) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="carrada,raddet")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-subsets", type=int, default=200)
    a = ap.parse_args()
    datasets = a.datasets.split(",")
    style()
    fig, axs = plt.subplots(len(datasets), len(PROBES), figsize=(2.3 * len(PROBES), 2.9 * len(datasets)),
                            squeeze=False)
    g = np.random.default_rng(0)
    summary = {"provenance": provenance(), "threshold_q": PRIMARY_Q, "datasets": {}}
    for i, ds in enumerate(datasets):
        real = load_raw(OUT / ds / "real" / "raw.npz")
        thr = thresholds(real)[PRIMARY_Q]
        n_real = int(real["frames"]["seq"].max()) + 1
        cands = {c: load_raw(candidate_dir(ds, c) / "raw.npz") for c in CANDIDATES}
        summary["datasets"][ds] = {"threshold": thr, "probes": {}}
        for j, (probe, label, stat) in enumerate(PROBES):
            ax = axs[i, j]
            full = value(real, probe, stat, thr)
            bands = {}
            for b, alpha in ((5, 0.12), (10, 0.25)):
                vals = []
                for _ in range(a.n_subsets):
                    seqs = np.sort(g.choice(n_real, b, replace=False))
                    vals.append(value(resample(real, seqs), probe, stat, thr))
                lo, hi = np.nanpercentile(vals, (5, 95))
                bands[b] = [float(lo), float(hi)]
                ax.axhspan(lo, hi, color=INK, alpha=alpha, lw=0)
            ax.axhline(full, color=INK, lw=1.0)
            rows = {}
            for k, c in enumerate(CANDIDATES):
                t = cands[c]
                n = int(t["frames"]["seq"].max()) + 1
                v = value(t, probe, stat, thr)
                boot = [value(resample(t, g.integers(n, size=n)), probe, stat, thr) for _ in range(a.n_boot)]
                lo, hi = np.nanpercentile(boot, (5, 95))
                ax.errorbar([k], [v], yerr=[[v - lo], [hi - v]], fmt="o", ms=4.5, capsize=0, lw=1.2,
                            color=COLOR[ds] if c != "spec" else MUTED)
                rows[c] = {"value": v, "ci90": [float(lo), float(hi)]}
            summary["datasets"][ds]["probes"][probe] = {"stat": stat, "real_50": full,
                                                        "real_subset_5_95": bands, "candidates": rows}
            ax.set_xticks(range(len(CANDIDATES)), [SHORT[c] for c in CANDIDATES], rotation=60, ha="right",
                          fontsize=7)
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(label, fontsize=8)
            if j == 0:
                ax.set_ylabel(NAMES[ds])
    fig.text(0.01, -0.02, "Line: all 50 real training sequences; bands: 5-95% over random 5-sequence (light) and "
             "10-sequence (dark) reference subsets. Dots: candidates (200 synthetic sequences), 90% bootstrap over "
             f"sequences; grey = spec simulator. Target-like = prominence >= the {PRIMARY_Q}th percentile of the "
             "real labelled objects'.", fontsize=7, color=MUTED, wrap=True)
    fig.tight_layout()
    (OUT.parent / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.parent / "figures/probes.pdf", bbox_inches="tight")
    fig.savefig(OUT.parent / "figures/probes.png", bbox_inches="tight", dpi=200)
    (OUT.parent / "probes_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
