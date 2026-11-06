"""Scores from the adequacy vector (docs/notes/2026-11-06-metric-experiments.md, Phase 2-3).

probes(vector) -> {probe: x_j}: the signed probe values the tolerance score reads.
Tolerance: interval of x_j over undamaged development candidates, scale = MAD over all development
candidates; penalty = distance outside the interval / scale. M1 = sum of penalties; M2 = NNLS weights
on the penalties fitted to development val AP. Baselines read the vector (or the raw objects table).
evaluate_set(): the ranking metrics of one candidate set.
"""
import numpy as np
from scipy.optimize import nnls
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from src.utility_metric import LABEL, TARGET, energy_distance

# (component, feature, how): how = "shift" | "w1" | "diff" (cand - ref) | "ratio-1"
PROBES = {
    "coverage": [("coverage", "joint_energy", "w1"), ("coverage", "n_in_frame", "shift"),
                 ("coverage", "near_zero_share", "diff")],
    "label": [("label", k, "shift") for k in ("peak_dr", "peak_dd", "cen_dr", "cen_dd", "label_prom", "over_clutter")]
    + [("label", "labels_with_peak_r2", "diff"), ("label", "unlabelled_per_frame", "shift"),
       ("label", "unlabelled_off_ridge_per_frame", "shift")],
    "target": [("target", k, "shift") for k in ("width_r2", "width_d2", "area2", "near_ext2", "far_ext2",
                                                 "tail_level", "n_peaks2", "entropy_r", "entropy_d")],
    "negatives": [("negatives", "fp_coverage", "ratio-1"), ("negatives", "fp_per_frame", "shift"),
                  ("negatives", "peak_prom", "shift")],
    "processing": [("processing", k, "shift") for k in ("spread1", "spread4", "corr_range", "corr_doppler",
                                                         "frame_q99", "ridge_width")]
    + [("processing", "rowmed_profile", "w1"), ("processing", "rowq99_profile", "w1"),
       ("processing", "ridge_profile", "shift")],
}
PROBE_NAMES = [f"{c}.{f}" for comp in PROBES.values() for c, f, _ in comp]
COMPONENTS = {name: name.split(".")[0] for name in PROBE_NAMES}


def probes(vector):
    out = {}
    for comp in PROBES.values():
        for c, f, how in comp:
            d = vector.get(c, {}).get(f)
            v = np.nan
            if d:
                if how == "diff":
                    v = d["cand"] - d["ref"] if d.get("cand") is not None and d.get("ref") is not None else np.nan
                elif how == "ratio-1":
                    v = d["ratio"] - 1 if d.get("ratio") is not None else np.nan
                else:
                    v = d.get(how)
            out[f"{c}.{f}"] = np.nan if v is None else float(v)
    return out


class Tolerance:
    """Dead-zone penalties: interval of each probe over undamaged development candidates."""

    def fit(self, X, undamaged):
        """X: (n, p) probe values of development candidates; undamaged: (n,) bool."""
        self.lower = np.full(X.shape[1], np.nan)
        self.upper = np.full(X.shape[1], np.nan)
        self.scale = np.full(X.shape[1], np.nan)
        for j in range(X.shape[1]):
            ok = np.isfinite(X[:, j])
            u = ok & undamaged
            if u.any():
                self.lower[j], self.upper[j] = X[u, j].min(), X[u, j].max()
            if ok.sum() >= 2:
                self.scale[j] = max(float(np.median(np.abs(X[ok, j] - np.median(X[ok, j])))), 1e-3)
        return self

    def penalties(self, X):
        below = np.clip(self.lower - X, 0, None)
        above = np.clip(X - self.upper, 0, None)
        p = (below + above) / self.scale
        return np.where(np.isfinite(p), p, 0.0)         # unmeasured / uncalibrated: no penalty

    def describe(self):
        return {n: {"lower": lo, "upper": up, "scale": s} for n, lo, up, s in
                zip(PROBE_NAMES, self.lower.tolist(), self.upper.tolist(), self.scale.tolist())}


def nnls_weights(P, ap):
    """AP ~ a - P w, a and w >= 0 (no intercept sign issue: AP is positive)."""
    sol, _ = nnls(np.column_stack([np.ones(len(P)), -P]), ap)
    return sol[1:]


def component_penalties(p):
    """Sum of the probe penalties per component."""
    out = {}
    for name, v in zip(PROBE_NAMES, p):
        out[COMPONENTS[name]] = out.get(COMPONENTS[name], 0.0) + float(v)
    return out


# ------------------------------------------------------------------ baselines from the raw objects table

def target_energy(cand_obj, ref_obj):
    """Energy distance between the target-response feature vectors (standardised by the reference)."""
    keys = [k for k in TARGET if k in cand_obj]
    C = np.column_stack([cand_obj[k] for k in keys])
    R = np.column_stack([ref_obj[k] for k in keys])
    C, R = C[np.isfinite(C).all(1)], R[np.isfinite(R).all(1)]
    if len(C) < 5 or len(R) < 5:
        return np.nan
    sd = R.std(0)
    sd[sd == 0] = 1
    mu = R.mean(0)
    return energy_distance((C - mu) / sd, (R - mu) / sd)


def domain_classifier(cand_obj, ref_obj, folds=5, seed=0):
    """Balanced accuracy of a logistic regression telling real objects from the candidate's, on label +
    target features, folds grouped by sequence (each source's sequences kept whole)."""
    keys = [k for k in LABEL + TARGET if k in cand_obj]
    C = np.column_stack([cand_obj[k] for k in keys])
    R = np.column_stack([ref_obj[k] for k in keys])
    cm, rm = np.isfinite(C).all(1), np.isfinite(R).all(1)
    C, R = C[cm], R[rm]
    if len(R) < folds or len(C) < folds:
        return np.nan
    g = np.random.default_rng(seed)
    if len(C) > 20 * len(R):                       # keep the problem balanced-ish and cheap
        keep = g.choice(len(C), 20 * len(R), replace=False)
        C, cs = C[keep], cand_obj["seq"][cm][keep]
    else:
        cs = cand_obj["seq"][cm]
    X = np.vstack([C, R])
    y = np.r_[np.zeros(len(C)), np.ones(len(R))]
    groups = np.r_[cs, 100_000 + ref_obj["seq"][rm]]
    n_groups = len(np.unique(groups[y == 1]))
    k = min(folds, n_groups)
    if k < 2:
        return np.nan
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            pred[te] = 0
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(class_weight="balanced", max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
        pred[te] = m.predict((X[te] - mu) / sd)
    return float(np.mean([(pred[y == c] == c).mean() for c in (0, 1)]))


# ------------------------------------------------------------------ ranking metrics

def evaluate_set(risk, ap, noise):
    """Ranking metrics of one candidate set: lower risk = predicted better; NaN risk ranks last;
    ties at the top are a uniform choice (expected regret)."""
    risk = np.where(np.isfinite(risk), risk, np.inf)
    oracle = ap.max()
    top = np.isclose(risk, risk.min()) | (risk == risk.min())
    sel = ap[top]
    out = {"n": int(len(ap)), "regret": float(oracle - sel.mean()),
           "within_noise": float((sel >= oracle - noise).mean()), "n_tied_top": int(top.sum())}
    pairs, correct = 0, 0.0
    for i in range(len(ap)):
        for j in range(i + 1, len(ap)):
            if abs(ap[i] - ap[j]) <= noise:
                continue
            pairs += 1
            better = i if ap[i] > ap[j] else j
            worse = j if better == i else i
            if risk[better] < risk[worse]:
                correct += 1
            elif risk[better] == risk[worse]:
                correct += 0.5
    out["pairwise"] = correct / pairs if pairs else np.nan
    out["n_pairs"] = pairs
    finite = np.isfinite(risk)
    if finite.sum() >= 3 and len(set(risk[finite])) > 1:
        out["kendall"] = float(kendalltau(-risk[finite], ap[finite]).statistic)
        out["spearman"] = float(spearmanr(-risk[finite], ap[finite]).statistic)
    else:
        out["kendall"] = out["spearman"] = np.nan
    if len(ap) > 3:
        true3 = set(np.argsort(-ap, kind="stable")[:3])
        pred3 = set(np.argsort(risk, kind="stable")[:3])
        out["top3_recall"] = len(true3 & pred3) / 3
    return out
