"""Label-budget figure from the pushed results JSON (metric plan, Phase 0.2).

Real-test AP@2 against the number of labelled real training sequences (0, 1, 5, 10, 25, 50; one
sequence = 8 frames): real only, and the best tuned simulator + real (each batch half real, half
synthetic). Budget 0 is the tuned simulator alone (Study 2's clean). Mean of 3 seeds, bars = seed
min-max; points are joined by straight lines, nothing is smoothed or interpolated. Only the real
training pool changes with the budget; validation (early stopping) and test stay fixed.
Writes results/metric/figures/label_budget_ap2.{pdf,png} and results/metric/label_budget_summary.json.

    PYTHONPATH=. python3 scripts/plot_label_budget.py
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures import COLOR, INK, NAMES, style
from src.provenance import provenance

BUDGETS = (1, 5, 10, 25, 50)
OUT = Path("results/metric")


def points(dataset, metric="ap2"):
    """{setting: [(budget, [per-seed AP], source path)]}."""
    out = {"real": [], "mix": []}
    sim = Path(f"results/study2_{dataset}_runs.json")
    clean = json.loads(sim.read_text())["runs"]["only"]["clean"]
    out["mix"].append((0, [s[metric] for s in clean.values()], str(sim)))
    for n in BUDGETS:
        p = Path(f"results/budget_{dataset}_real{n}_runs.json")
        r = json.loads(p.read_text())["runs"]
        for setting in ("real", "mix"):
            out[setting].append((n, [s[metric] for s in r[setting]["clean"].values()], str(p)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="carrada,raddet")
    a = ap.parse_args()
    datasets = a.datasets.split(",")
    style()
    fig, axs = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.0), sharey=True)
    summary = {"provenance": provenance(), "metric": "ap2", "split": "test",
               "best_tuned": json.loads(Path("results/tuning/best.json").read_text()), "datasets": {}}
    for ax, ds in zip(np.atleast_1d(axs), datasets):
        pts = points(ds)
        summary["datasets"][ds] = {}
        for setting, c, name in (("real", INK, "real only"), ("mix", COLOR[ds], "tuned sim + real")):
            n = np.array([p[0] for p in pts[setting]])
            x = [np.array(p[1]) for p in pts[setting]]
            assert all(len(v) == 3 for v in x), (ds, setting)
            m = np.array([v.mean() for v in x])
            ax.errorbar(n, m, yerr=[m - [v.min() for v in x], [v.max() for v in x] - m],
                        color=c, marker="o", capsize=2, label=name)
            summary["datasets"][ds][setting] = [
                {"budget": int(b), "seeds": v.tolist(), "mean": float(v.mean()), "min": float(v.min()),
                 "max": float(v.max()), "source": src} for b, v, src in zip(n, x, [p[2] for p in pts[setting]])]
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks((0,) + BUDGETS, [str(b) for b in (0,) + BUDGETS])
        ax.minorticks_off()
        ax.set_title(NAMES[ds] + (" (scene-disjoint split)" if ds == "raddet" else ""))
        ax.set_xlabel("labelled real training sequences (8 frames each)")
    np.atleast_1d(axs)[0].set_ylabel("AP@2 on real test")
    np.atleast_1d(axs)[0].legend(loc="lower right")
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures/label_budget_ap2.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figures/label_budget_ap2.png", bbox_inches="tight", dpi=200)
    summary["caption"] = ("Real-test AP@2 against the number of labelled real training sequences (8 frames "
                          "each). Real only vs the best tuned simulator + real (half of each batch); 0 = the "
                          "tuned simulator alone. Mean of 3 seeds, bars = seed min-max. Only the real training "
                          "pool changes; validation (early stopping) and test are fixed.")
    (OUT / "label_budget_summary.json").write_text(json.dumps(summary, indent=2))
    for ds, s in summary["datasets"].items():
        for setting, rows in s.items():
            print(ds, setting, " ".join(f"{r['budget']}:{r['mean']:.3f}" for r in rows))


if __name__ == "__main__":
    main()
