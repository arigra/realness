"""Retrospective validation of the utility metric (docs/notes/2026-11-06-metric-experiments.md,
Phase 2-3): leave-one-family-out within each radar and leave-one-radar-out, for the tolerance
score (M1), its NNLS-weighted readout (M2) and the baselines, on every reference subset.

Reads the vectors of scripts/build_metric_features.py (results/metric/features/), the registry and
the result files. Calibration uses development candidates' whole-val AP@2; evaluation uses test
AP@2. Baselines that need the raw objects table (target energy, domain classifier) are computed once
per (candidate, reference) and cached in results/metric/scores/baselines_<dataset>.json.
Writes results/metric/scores/evaluation.json and results/metric/frozen_metric.json.

    PYTHONPATH=. python3 scripts/evaluate_utility_metric.py
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts.build_metric_features import OUT, candidate_dir, feature_key, subsets
from src.provenance import provenance
from src.utility_score import (PROBE_NAMES, Tolerance, component_penalties, domain_classifier,
                               evaluate_set, nnls_weights, probes, target_energy)

REGISTRY = Path("results/metric/candidates.json")
SCORES = Path("results/metric/scores")
DATASETS = ("carrada", "raddet", "radial")
REFERENCES = [(5, 0), (5, 1), (5, 2), (10, 0), (10, 1), (10, 2), (50, 0)]
PRIMARY = (5, 0)
BASELINES = ("global", "target_energy", "target_conditional", "coverage_only", "label_only",
             "five_unweighted", "domain_classifier")
LEARNED = ("cnn", "cnn_conditional", "transformer", "transformer_conditional", "raw_patch", "raw_patch_conditional")
METHODS = ("M1", "M2") + BASELINES + ("proxy",) + tuple(f"learned_{k}" for k in LEARNED)
PROXY = Path("results/metric/proxy")
LEARNED_DIR = Path("results/metric/learned")
N_BOOT = 2000


def seeds_of(row, split):
    """AP@2 of every usable synthetic-only seed of a candidate (test or whole val)."""
    out = []
    for e in row["entries"]:
        if e["setting"] != "only" or not e["usable"]:
            continue
        r = json.loads(Path(e["result_path"]).read_text())
        block = r["runs"] if split == "test" else r.get("val", {})
        for v in block.get("only", {}).get(e["version"], {}).values():
            out.append(v if isinstance(v, float) else v["ap2"])
    return out


def load_rows():
    rows = []
    for r in json.loads(REGISTRY.read_text())["candidates"]:
        if not r["usable"]:
            continue
        test, val = seeds_of(r, "test"), seeds_of(r, "val")
        r = dict(r, test_ap=float(np.mean(test)), test_range=float(np.ptp(test)) if len(test) > 1 else np.nan,
                 val_ap=float(np.mean(val)) if val else np.nan,
                 val_range=float(np.ptp(val)) if len(val) > 1 else np.nan, n_test_seeds=len(test))
        rows.append(r)
    return rows


_KEYS = {}


def vector(row, ref):
    """The candidate's vector against a reference, if built with the current features and detector."""
    p = candidate_dir(row["dataset"], row["candidate"]) / f"reference_{ref[0]}_{ref[1]}.json"
    if not p.exists():
        return None
    v = json.loads(p.read_text())
    key = _KEYS.setdefault(row["dataset"], feature_key(row["dataset"]))
    return v if v.get("feature_key") == key else None


def load_objects(path):
    z = np.load(path)
    return {k.split("/")[1]: z[k].astype(np.float64) for k in z.files if k.startswith("objects/")}


def baselines(dataset, rows):
    """{candidate: {"b_s": {baseline: value}}}, cached."""
    path = SCORES / f"baselines_{dataset}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    real = load_objects(OUT / dataset / "real" / "raw.npz")
    refs = {f"{b}_{s}": (subsets(b, 3)[s] if b < 50 else np.arange(50)) for b, s in REFERENCES}
    changed = False
    for row in rows:
        cid = row["candidate"]
        v50 = vector(row, (50, 0))
        if v50 is None:
            continue
        if cache.get(cid, {}).get("_key") != v50["feature_key"]:
            cache[cid] = {"_key": v50["feature_key"]}       # features rebuilt: baselines are stale
        todo = [k for k in refs if k not in cache[cid]]
        if not todo:
            continue
        cand = load_objects(candidate_dir(dataset, cid) / "raw.npz")
        for k in todo:
            b, s = map(int, k.split("_"))
            v = vector(row, (b, s))
            if v is None:
                continue
            vec = v["vector"]
            m = np.isin(real["seq"], refs[k])
            ref = {c: x[m] for c, x in real.items()}
            sc = vec["scores"]
            strata = [x["score"] for x in vec["target_strata"].values() if x.get("score") is not None]
            cache.setdefault(cid, {})[k] = {
                "global": vec["global"]["sum"], "target_energy": target_energy(cand, ref),
                "target_conditional": float(np.mean(strata)) if strata else None,
                "coverage_only": nan(sc["coverage"]), "label_only": nan(sc["label"]),
                "five_unweighted": float(np.nanmean([nan(sc[c]) for c in ("coverage", "label", "target",
                                                                           "processing", "negatives")])),
                "domain_classifier": domain_classifier(cand, ref)}
            changed = True
    if changed:
        SCORES.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, default=lambda x: None if not np.isfinite(x) else float(x)))
    return cache


def nan(x):
    return np.nan if x is None else x


class Data:
    """Probe matrix, outcomes and baselines of every usable candidate for one reference."""

    def __init__(self, rows, base_cache, ref):
        self.rows, self.ref = [], ref
        X = []
        for r in rows:
            v = vector(r, ref)
            if v is None:
                continue
            self.rows.append(r)
            x = probes(v["vector"])
            X.append([x[n] for n in PROBE_NAMES])
        self.X = np.asarray(X, dtype=np.float64)
        key = f"{ref[0]}_{ref[1]}"
        self.B = {b: np.asarray([nan(base_cache[r["dataset"]].get(r["candidate"], {}).get(key, {}).get(b))
                                 for r in self.rows], dtype=np.float64) for b in BASELINES}
        proxy = {d: json.loads((PROXY / f"{d}.json").read_text())["runs"] if (PROXY / f"{d}.json").exists() else {}
                 for d in DATASETS}
        # proxy detector (Phase 4): the seed-1 sweep, mean AP@2 over the three 5-sequence subsets; risk = -AP
        self.B["proxy"] = np.asarray([-proxy[r["dataset"]].get(r["candidate"], {}).get("1", {}).get("proxy", np.nan)
                                      for r in self.rows], dtype=np.float64)
        learned = {d: json.loads((LEARNED_DIR / f"{d}.json").read_text()) if (LEARNED_DIR / f"{d}.json").exists() else {}
                   for d in DATASETS}
        for k in LEARNED:       # Phase 7.2: learned / raw-patch energy distances (missing -> NaN, ranks last)
            self.B[f"learned_{k}"] = np.asarray([nan(learned[r["dataset"]].get(r["candidate"], {}).get(key, {}).get(k))
                                                 for r in self.rows], dtype=np.float64)
        self.cost = {"metric_seconds": {}, "proxy_seconds": {}}
        for d in DATASETS:
            ms = [json.loads((candidate_dir(d, r["candidate"]) / "raw.json").read_text()).get("seconds", np.nan)
                  for r in self.rows if r["dataset"] == d and (candidate_dir(d, r["candidate"]) / "raw.json").exists()]
            ps = [v["1"]["seconds"] for v in proxy[d].values() if "1" in v]
            self.cost["metric_seconds"][d] = float(np.nanmedian(ms)) if ms else None
            self.cost["proxy_seconds"][d] = float(np.median(ps)) if ps else None
        self.ds = np.asarray([r["dataset"] for r in self.rows])
        self.fam = np.asarray([r["family"] for r in self.rows])
        self.base = np.asarray([r["base"] for r in self.rows])
        self.cid = np.asarray([r["candidate"] for r in self.rows])
        self.test = np.asarray([r["test_ap"] for r in self.rows])
        self.val = np.asarray([r["val_ap"] for r in self.rows])
        self.noise_test = {d: float(np.nanmedian([r["test_range"] for r in self.rows if r["dataset"] == d]))
                           for d in DATASETS}
        self.noise_val = {d: float(np.nanmedian([r["val_range"] for r in self.rows if r["dataset"] == d]))
                          for d in DATASETS}
        # undamaged: val AP within the val noise of the dataset's tuned simulator (natural candidates
        # and every tuned-based family are judged against it)
        tuned_val = {d: self.val[(self.ds == d) & (self.cid == "tuned")][0] for d in set(self.ds)}
        self.undamaged = np.asarray([np.isfinite(v) and v >= tuned_val[d] - self.noise_val[d]
                                     for v, d in zip(self.val, self.ds)])

    def calibrate(self, dev):
        """Tolerance and NNLS fitted on the development rows that have val AP."""
        m = dev & np.isfinite(self.val)
        tol = Tolerance().fit(self.X[m], self.undamaged[m])
        P = tol.penalties(self.X[m])
        w = nnls_weights(P, self.val[m])
        return tol, w, int(m.sum())

    def risks(self, tol, w, idx):
        P = tol.penalties(self.X[idx])
        out = {"M1": P.sum(1), "M2": P @ w}
        for b in METHODS[2:]:
            out[b] = self.B[b][idx]
        return out, P


def summarise_sets(results):
    """Mean of each metric over sets (families)."""
    keys = ("regret", "within_noise", "pairwise", "kendall", "spearman")
    return {k: float(np.nanmean([r[k] for r in results])) for k in keys}


def bootstrap_diff(per_family, a, b, n=N_BOOT, seed=0):
    """Mean regret of method a minus method b, bootstrap over families: mean and 95% interval."""
    fams = sorted(per_family)
    da = np.array([per_family[f][a]["regret"] for f in fams])
    db = np.array([per_family[f][b]["regret"] for f in fams])
    g = np.random.default_rng(seed)
    boot = [np.mean((da - db)[g.integers(len(fams), size=len(fams))]) for _ in range(n)]
    return {"mean": float(np.mean(da - db)), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def random_metrics(ap, noise):
    """Expected metrics of a uniformly random choice."""
    return {"regret": float(ap.max() - ap.mean()), "within_noise": float((ap >= ap.max() - noise).mean()),
            "pairwise": 0.5, "kendall": 0.0, "spearman": 0.0}


def lofo(D, dataset):
    """Leave-one-family-out within one radar."""
    in_ds = D.ds == dataset
    fams = sorted({f for f in D.fam[in_ds] if f != "natural" and (D.fam[in_ds] == f).sum() >= 2})
    per_family = {}
    for f in fams:
        dev = (D.fam != f) & in_ds
        tol, w, n_dev = D.calibrate(dev)
        sets = []
        for base in sorted(set(D.base[in_ds & (D.fam == f)])):
            members = in_ds & (D.fam == f) & (D.base == base)
            anchor = in_ds & (D.cid == base) if base in ("tuned", "spec") else np.zeros_like(members)
            if f == "fixes":
                anchor = in_ds & np.isin(D.cid, ["tuned", "spec"])
            idx = np.flatnonzero(members | anchor)
            sets.append(idx)
        res = {m: [] for m in METHODS + ("random",)}
        for idx in sets:
            ap = D.test[idx]
            noise = D.noise_test[dataset]
            risks, _ = D.risks(tol, w, idx)
            for m in METHODS:
                res[m].append(evaluate_set(risks[m], ap, noise))
            res["random"].append(random_metrics(ap, noise))
        per_family[f] = {m: summarise_sets(r) for m, r in res.items()}
        per_family[f]["n_dev"] = n_dev
        per_family[f]["sets"] = [[D.cid[i] for i in idx] for idx in sets]
    methods = METHODS + ("random",)
    summary = {m: {k: float(np.nanmean([per_family[f][m][k] for f in fams])) for k in
                   ("regret", "within_noise", "pairwise", "kendall", "spearman")} for m in methods}
    diffs = {b: bootstrap_diff(per_family, "M1", b) for b in ("random",) + METHODS[1:]}
    return {"families": per_family, "summary": summary, "M1_minus": diffs}


def loro(D, held):
    """Leave-one-radar-out: calibrate on the two other radars."""
    dev = D.ds != held
    tol, w, n_dev = D.calibrate(dev)
    out = {"n_dev": n_dev, "weights": dict(zip(PROBE_NAMES, w.tolist())), "tolerance": tol.describe()}
    noise = D.noise_test[held]
    for name, mask in (("all", D.ds == held),
                       ("natural", (D.ds == held) & np.isin(D.fam, ["natural", "fixes"]))):
        idx = np.flatnonzero(mask)
        ap = D.test[idx]
        risks, P = D.risks(tol, w, idx)
        out[name] = {m: evaluate_set(risks[m], ap, noise) for m in METHODS}
        out[name]["random"] = random_metrics(ap, noise)
        sel = idx[np.flatnonzero(np.isclose(risks["M1"], risks["M1"].min()))]
        out[name]["M1_selected"] = [D.cid[i] for i in sel]
        out[name]["oracle"] = D.cid[idx[np.argmax(ap)]]
        out[name]["diagnosis"] = {D.cid[i]: component_penalties(P[k]) for k, i in enumerate(idx)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    a = ap.parse_args()
    datasets = a.datasets.split(",")
    t0 = time.time()
    rows = [r for r in load_rows() if r["dataset"] in datasets]
    cache = {d: baselines(d, [r for r in rows if r["dataset"] == d]) for d in datasets}
    report = {"provenance": provenance(), "references": {}, "baseline_seconds": time.time() - t0}
    for ref in REFERENCES:
        D = Data(rows, cache, ref)
        key = f"{ref[0]}_{ref[1]}"
        r = {"n_candidates": {d: int((D.ds == d).sum()) for d in datasets}, "cost": D.cost,
             "noise_test": D.noise_test, "noise_val": D.noise_val,
             "n_undamaged": {d: int((D.undamaged & (D.ds == d)).sum()) for d in datasets},
             "lofo": {d: lofo(D, d) for d in datasets}}
        if len(datasets) == 3:
            r["loro"] = {d: loro(D, d) for d in datasets}
        report["references"][key] = r
        s = {d: r["lofo"][d]["summary"] for d in datasets}
        print(key, " | ".join(f"{d}: M1 {s[d]['M1']['regret']:.3f} glob {s[d]['global']['regret']:.3f} "
                              f"rand {s[d]['random']['regret']:.3f}" for d in datasets), flush=True)
        if ref == PRIMARY and len(datasets) == 3:
            D_all = D
    if len(datasets) == 3:
        tol, w, n_dev = D_all.calibrate(np.ones(len(D_all.rows), bool))
        Path("results/metric/frozen_metric.json").write_text(json.dumps({
            "provenance": provenance(), "reference": list(PRIMARY), "n_dev": n_dev,
            "tolerance": tol.describe(), "weights": dict(zip(PROBE_NAMES, w.tolist()))}, indent=1,
            default=lambda x: None))
    SCORES.mkdir(parents=True, exist_ok=True)
    (SCORES / "evaluation.json").write_text(json.dumps(report, indent=1,
                                                       default=lambda x: None if not np.isfinite(x) else float(x)))
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
