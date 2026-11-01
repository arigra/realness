"""Realness scores from group distances, and how well they rank AP
(docs/notes/2026-11-01-realness-score.md, steps 4-5). Higher score = less real.
"""
import itertools
import json

import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr

from src.measures import GROUPS

G = list(GROUPS)


def points(dataset, metric="ap2", suites=("fair", "smoothall")):
    """{version: (group distances (3,), mean synthetic-only AP)} for the measured versions."""
    d = json.load(open(f"results/distances_{dataset}.json"))
    ap = {}
    for s in suites:
        for n, r in json.load(open(f"results/{s}_{dataset}_runs.json"))["runs"]["only"].items():
            ap[n] = float(np.mean([v[metric] for v in r.values()]))
    return {n: (np.array([d[n]["groups"][g] for g in G]), ap[n]) for n in d if n in ap}


def _fit(D, ap):
    """AP ~ a - sum w_g D_g with a, w >= 0. Returns w."""
    sol, _ = nnls(np.column_stack([np.ones(len(D)), -D]), ap)
    return sol[1:]


class Score:
    """kind: 'sum' (nothing fitted), 'weighted', or 'threshold' (w_g max(0, d_g - tau_g))."""

    def __init__(self, kind):
        self.kind, self.w, self.tau = kind, np.ones(len(G)), np.zeros(len(G))

    def fit(self, D, ap):
        if self.kind == "weighted":
            self.w = _fit(D, ap)
        elif self.kind == "threshold":
            grid = [np.quantile(D[:, g], (0, .25, .5, .75)) for g in range(len(G))]
            best = None
            for tau in itertools.product(*grid):
                H = np.maximum(D - np.array(tau), 0)
                w = _fit(H, ap)
                a = np.mean(ap + H @ w)
                sse = float(((a - H @ w - ap) ** 2).sum())
                if best is None or sse < best[0]:
                    best = (sse, w, np.array(tau))
            _, self.w, self.tau = best
        return self

    def __call__(self, D):
        return np.maximum(D - self.tau, 0) @ self.w

    def describe(self):
        return {"kind": self.kind, "w": dict(zip(G, self.w.round(4).tolist())),
                "tau": dict(zip(G, self.tau.round(4).tolist()))}


def rho(score, ap, n_boot=2000, seed=0):
    """Spearman correlation of -score with AP, and a bootstrap 90% interval over versions."""
    r = spearmanr(-score, ap).statistic
    g = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        i = g.integers(len(ap), size=len(ap))
        if len(set(ap[i])) > 2 and len(set(score[i])) > 2:
            boot.append(spearmanr(-score[i], ap[i]).statistic)
    return float(r), [float(np.nanpercentile(boot, 5)), float(np.nanpercentile(boot, 95))]


def loo(kind, D, ap):
    """Leave-one-version-out scores (each version scored by a fit without it)."""
    out = np.zeros(len(ap))
    for i in range(len(ap)):
        keep = np.arange(len(ap)) != i
        out[i] = Score(kind).fit(D[keep], ap[keep])(D[i:i + 1])[0]
    return out


def disagreements(names, score, ap, k=5):
    """Versions whose AP rank and (-score) rank differ most: + means AP better than the score says."""
    ra = np.argsort(np.argsort(ap))
    rs = np.argsort(np.argsort(-score))
    res = ra - rs
    order = np.argsort(-np.abs(res))[:k]
    return [(names[i], int(res[i]), round(float(ap[i]), 3), round(float(score[i]), 3)) for i in order]
