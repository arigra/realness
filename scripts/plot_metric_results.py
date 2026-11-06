"""Figures of the retrospective validation (docs/notes/2026-11-06-metric-experiments.md, Phase 3-4).

Reads results/metric/scores/evaluation.json. Per dataset (rows): (a) leave-one-family-out mean top-1
regret per method, 90% bootstrap over families; (b) leave-one-radar-out regret, all candidates and
natural candidates; (c) leave-one-family-out regret of M1, the global score and the proxy across
reference subsets. Lower regret = better. Writes results/metric/figures/validation.{pdf,png}.

    PYTHONPATH=. python3 scripts/plot_metric_results.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures import COLOR, INK, MUTED, NAMES, style

EVAL = Path("results/metric/scores/evaluation.json")
ORDER = ["M1", "M2", "global", "target_energy", "target_conditional", "coverage_only", "label_only",
         "five_unweighted", "domain_classifier", "proxy", "random"]
LABEL = {"M1": "tolerance (M1)", "M2": "tolerance, NNLS (M2)", "global": "global score (2026-11-01)",
         "target_energy": "target energy dist.", "target_conditional": "target, conditional",
         "coverage_only": "coverage only", "label_only": "label only", "five_unweighted": "5 components, mean",
         "domain_classifier": "domain classifier", "proxy": "proxy detector", "random": "random choice"}


def boot(vals, n=2000, seed=0):
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)])
    if not len(vals):
        return np.nan, np.nan, np.nan
    g = np.random.default_rng(seed)
    b = [vals[g.integers(len(vals), size=len(vals))].mean() for _ in range(n)]
    return vals.mean(), np.percentile(b, 5), np.percentile(b, 95)


def main():
    ev = json.loads(EVAL.read_text())
    refs = ev["references"]
    prim = refs["5_0"]
    datasets = list(prim["lofo"])
    style()
    fig, axs = plt.subplots(len(datasets), 3, figsize=(13, 3.3 * len(datasets)), squeeze=False,
                            gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    for i, ds in enumerate(datasets):
        fam = prim["lofo"][ds]["families"]
        ax = axs[i, 0]
        for k, m in enumerate(ORDER):
            mu, lo, hi = boot([fam[f][m]["regret"] for f in fam if m in fam[f]])
            ax.errorbar([mu], [k], xerr=[[mu - lo], [hi - mu]], fmt="o", ms=4.5, lw=1.2,
                        color=COLOR[ds] if m in ("M1", "M2") else (MUTED if m == "random" else INK))
        ax.set_yticks(range(len(ORDER)), [LABEL[m] for m in ORDER], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("mean top-1 regret (AP@2), held-out families")
        ax.set_title(f"{NAMES[ds]}: leave one family out ({len(fam)} families)", fontsize=8)
        ax = axs[i, 1]
        if "loro" in prim:
            for k, m in enumerate(ORDER):
                for j, (subset, mk) in enumerate((("all", "o"), ("natural", "s"))):
                    v = prim["loro"][ds][subset].get(m, {}).get("regret")
                    if v is not None:
                        ax.plot([v], [k + (j - 0.5) * 0.3], mk, ms=4.5, color=COLOR[ds] if j == 0 else MUTED,
                                label=("all candidates" if j == 0 else "natural candidates") if k == 0 else None)
            ax.set_yticks(range(len(ORDER)), [LABEL[m] for m in ORDER], fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel("top-1 regret (AP@2), held-out radar")
            ax.set_title(f"{NAMES[ds]}: leave one radar out", fontsize=8)
            if i == 0:
                ax.legend(fontsize=7, loc="lower right")
        ax = axs[i, 2]
        keys = list(refs)
        for m, c in (("M1", COLOR[ds]), ("global", INK), ("proxy", MUTED)):
            ys = [refs[k]["lofo"][ds]["summary"][m]["regret"] for k in keys]
            ax.plot(range(len(keys)), ys, "o-", color=c, ms=4, label=LABEL[m])
        ax.set_xticks(range(len(keys)), [k.replace("_", "/") for k in keys], fontsize=7)
        ax.set_xlabel("reference (sequences / subset)")
        ax.set_ylabel("LOFO mean regret")
        if i == 0:
            ax.legend(fontsize=7)
    fig.tight_layout()
    out = Path("results/metric/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "validation.pdf", bbox_inches="tight")
    fig.savefig(out / "validation.png", bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    main()
