"""Metric v2 selectors (docs/notes/2026-11-06-metric-v2-preregistration.md): M3_RTW, M4_RDA, M6_MRR.
M5_REC is computed directly in src/utility_metric_v2.m5 (no calibration).

RobustTolerance: safe interval = min..max over undamaged development candidates; scale = max(MAD over
development candidates, reference instability); a probe with neither is non-informative (excluded);
penalty = distance outside / scale, clipped to [0, 5]; an unmeasured (NaN) probe gets no penalty and is
flagged.
"""
import numpy as np
from scipy.optimize import minimize

CLIP = 5.0


class RobustTolerance:
    def fit(self, X, undamaged, instability):
        """X (n, p) development probe values; undamaged (n,) bool; instability (p,) reference instability."""
        p = X.shape[1]
        self.lower, self.upper, self.scale = np.full(p, np.nan), np.full(p, np.nan), np.full(p, np.nan)
        for j in range(p):
            ok = np.isfinite(X[:, j])
            u = ok & undamaged
            if u.any():
                self.lower[j], self.upper[j] = X[u, j].min(), X[u, j].max()
            mad = float(np.median(np.abs(X[ok, j] - np.median(X[ok, j])))) if ok.sum() >= 2 else np.nan
            inst = instability[j] if np.isfinite(instability[j]) else np.nan
            s = np.nanmax([mad, inst]) if np.isfinite([mad, inst]).any() else np.nan
            self.scale[j] = s if np.isfinite(s) and s > 0 else np.nan
        self.informative = np.isfinite(self.scale) & np.isfinite(self.lower)
        return self

    def penalties(self, X):
        """(n, p) penalties in [0, 5]; NaN where the probe is unmeasured or non-informative."""
        below = np.clip(self.lower - X, 0, None)
        above = np.clip(X - self.upper, 0, None)
        p = np.clip((below + above) / self.scale, 0, CLIP)
        p[:, ~self.informative] = np.nan
        p[~np.isfinite(X)] = np.nan
        return p

    def describe(self, names):
        return {n: {"lower": _f(lo), "upper": _f(up), "scale": _f(s), "informative": bool(i)}
                for n, lo, up, s, i in zip(names, self.lower, self.upper, self.scale, self.informative)}


def _f(x):
    return None if x is None or not np.isfinite(x) else float(x)


def groups_of(names):
    """Component / block of each probe name ('label.peak_dr' -> 'label', 'A.cfar_db_median' -> 'A')."""
    return [n.split(".")[0] for n in names]


# ------------------------------------------------------------------ M3

def m3(pen_row, names):
    """Weakest-link aggregation of one candidate's penalties: component risk = 80th percentile of its
    measured penalties; secondary = mean of its three largest. Returns (sort key, details)."""
    comps = {}
    for g in dict.fromkeys(groups_of(names)):
        v = pen_row[[i for i, n in enumerate(names) if n.split(".")[0] == g]]
        v = v[np.isfinite(v)]
        comps[g] = {"risk": float(np.percentile(v, 80)) if len(v) else np.nan,
                    "top3_mean": float(np.sort(v)[-3:].mean()) if len(v) else np.nan, "n_measured": int(len(v))}
    risks = np.array([c["risk"] for c in comps.values() if np.isfinite(c["risk"])])
    mx = float(risks.max()) if len(risks) else np.nan
    mean = float(risks.mean()) if len(risks) else np.nan
    dom = max((g for g in comps if np.isfinite(comps[g]["risk"])), key=lambda g: comps[g]["risk"], default=None)
    unmeasured = [g for g, c in comps.items() if c["n_measured"] == 0]
    return {"max": mx, "mean": mean, "key": (_inf(mx), _inf(mean)), "dominant": dom, "components": comps,
            "unmeasured_components": unmeasured}


def _inf(x):
    """Sort value: unmeasured (NaN) ranks last."""
    return float(x) if x is not None and np.isfinite(x) else np.inf


def lex_rank(keys):
    """Dense rank (0 = best) of sort-key tuples: equal tuples share a rank, so evaluate_set's tie rule
    (uniform choice among the tied best) applies to exact ties only."""
    order = sorted(set(keys))
    pos = {k: i for i, k in enumerate(order)}
    return np.asarray([pos[k] for k in keys], dtype=np.float64)


# ------------------------------------------------------------------ M4

BLOCK_NAMES = {"A": "detectability", "B": "label-task", "C": "morphology", "D": "support", "E": "hard negatives",
               "F": "processing"}


def m4(pen_row, names):
    """Risk per probe = min(p / 5, 1); block risk = max; M4 = 1 - prod(1 - block risk); diagnosis."""
    r = np.minimum(pen_row / CLIP, 1.0)
    blocks = {}
    for b in BLOCK_NAMES:
        idx = [i for i, n in enumerate(names) if n.split(".")[0] == b]
        v = r[idx]
        m = np.isfinite(v)
        if not m.any():
            blocks[b] = {"risk": np.nan, "top3_mean": np.nan, "driver": None, "n_measured": 0}
            continue
        k = idx[int(np.nanargmax(np.where(m, v, -1)))]
        blocks[b] = {"risk": float(np.nanmax(v)), "top3_mean": float(np.sort(v[m])[-3:].mean()),
                     "driver": names[k], "n_measured": int(m.sum())}
    br = np.array([blocks[b]["risk"] for b in BLOCK_NAMES])
    br0 = np.where(np.isfinite(br), br, 0.0)
    total = float(1 - np.prod(1 - br0))
    order = sorted(BLOCK_NAMES, key=lambda b: -(blocks[b]["risk"] if np.isfinite(blocks[b]["risk"]) else -1))
    diag = [f"{'high' if blocks[b]['risk'] >= 0.5 else 'some'} {BLOCK_NAMES[b]} risk: {blocks[b]['driver']}"
            for b in order[:2] if np.isfinite(blocks[b]["risk"]) and blocks[b]["risk"] > 0]
    # ties: lower maximum block risk, then label-task (B), then detectability (A); the name last (selection)
    key = (total, float(np.nanmax(br)) if np.isfinite(br).any() else 0.0, float(br0[1]), float(br0[0]))
    return {"M4": total, "key": key, "blocks": blocks, "top_blocks": order[:2], "diagnosis": diag}


# ------------------------------------------------------------------ M6

M6_INPUTS = ("A", "B", "C", "D", "E", "F", "fn_risk", "fp_risk")


class PairwiseRanker:
    """Non-negative linear pairwise logistic ranker on component risks (no hidden layer)."""

    def fit(self, R, val, ds, fam, noise, l2=1.0):
        """R (n, 8) risks; val (n,) val AP; ds / fam (n,) radar and family; noise {radar: val noise}."""
        Rf = np.where(np.isfinite(R), R, np.nanmean(R, 0))
        Rf = np.where(np.isfinite(Rf), Rf, 0.0)
        self.mu, self.sd = Rf.mean(0), Rf.std(0)
        self.sd[self.sd == 0] = 1.0
        Z = (Rf - self.mu) / self.sd
        pairs, fams = [], []
        for i in range(len(val)):
            for j in range(len(val)):
                if ds[i] == ds[j] and val[i] - val[j] > noise[ds[i]]:
                    pairs.append((i, j))                 # i better than j
                    fams.append(fam[j])
        self.pairs = pairs
        if not pairs:
            self.w = np.zeros(R.shape[1])
            return self
        fams = np.asarray(fams)
        counts = {f: (fams == f).sum() for f in set(fams)}
        pw = np.array([1.0 / counts[f] for f in fams])          # every family's pairs sum to 1
        self.pair_weights = pw
        self.pair_families = fams
        self.family_weights = {str(f): float(1.0 / counts[f]) for f in counts}
        D = np.array([Z[j] - Z[i] for i, j in pairs])    # worse minus better: a good w makes w.D > 0

        def loss(w):
            s = D @ w
            l = np.logaddexp(0, -s)
            g = -(pw * (1 / (1 + np.exp(s))))[:, None] * D
            return float((pw * l).sum() + l2 * (w ** 2).sum()), g.sum(0) + 2 * l2 * w
        res = minimize(loss, np.full(R.shape[1], 0.1), jac=True, method="L-BFGS-B",
                       bounds=[(0, None)] * R.shape[1])
        self.w = res.x
        return self

    def score(self, R):
        Rf = np.where(np.isfinite(R), R, self.mu)
        return ((Rf - self.mu) / self.sd) @ self.w
