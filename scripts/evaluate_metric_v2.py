"""Metric v2 development evaluation (docs/notes/2026-11-06-metric-v2-preregistration.md, Phases B, E, F).

Development candidates: the usable rows of results/metric/candidates.json and the first bank
(results/metric/bank.json). Outcome: whole-val AP@2 (mean of seeds), read from the "val" blocks of the result
files only: this script never reads test AP. Candidates without val AP are scored but not used.

Selectors: M1 / M2 (src/utility_score, unchanged), M3_RTW, M4_RDA, M5_REC, M6_MRR (src/utility_score_v2), the
global realism score, the v1 proxy detector (development reference only: seed 1, budget-5 subsets) and random.
Protocols: leave-one-family-out within each radar (family and its base held out together), leave-one-radar-out
(natural candidates primary; all candidates with val AP as a stress test), at every reference; the finalist by
the frozen rule at the primary reference (10 sequences, subset 0) -> results/metric_v2/finalist.json.

    PYTHONPATH=. python3 scripts/evaluate_metric_v2.py            # development + finalist
    PYTHONPATH=. python3 scripts/evaluate_metric_v2.py --verify-v1   # M1/M2 reproduce v1's Phase 6 rankings
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import scripts.build_metric_features as v1b
from scripts.build_metric_v2_features import PRIMARY, REFERENCES, cdir, feature_key
from src.provenance import provenance
from src.utility_score import PROBE_NAMES, Tolerance, evaluate_set, nnls_weights, probes
from src.utility_score_v2 import BLOCK_NAMES, M6_INPUTS, PairwiseRanker, RobustTolerance, lex_rank, m3, m4

DATASETS = ("carrada", "raddet", "radial")
REGISTRY = Path("results/metric/candidates.json")
BANK = Path("results/metric/bank.json")
SCORES = Path("results/metric_v2/scores")
NATURAL = ("natural", "fixes", "bank")
NEW = ("M3", "M4", "M5", "M6")
METHODS = ("M1", "M2", "global") + NEW + ("proxy_v1",)
FINALIST_ORDER = ("M4", "M3", "M5", "M6")
# probes marked unmeasured on a radar by a failed causal check (note, deviation log)
UNMEASURED = {"carrada": ("E.fp_per_frame", "B.capture_r2"), "raddet": ("E.uncovered",), "radial": ("B.capture_r2",)}


# ------------------------------------------------------------------ outcomes (val only)

def val_seeds(row):
    """Whole-val AP@2 of every usable synthetic-only seed of a candidate. Only "val" blocks are read."""
    out = []
    if row["suite"] == "bank":
        r = json.loads(Path(f"results/metric/prospective/{row['dataset']}_runs.json").read_text())
        return [v["ap2"] for v in r.get("val", {}).get(row["candidate"], {}).values()]
    for e in row["entries"]:
        if e["setting"] != "only" or not e["usable"]:
            continue
        r = json.loads(Path(e["result_path"]).read_text())
        for v in r.get("val", {}).get("only", {}).get(e["version"], {}).values():
            out.append(v if isinstance(v, float) else v["ap2"])
    return out


def dev_rows(include_bank=True):
    rows = [r for r in json.loads(REGISTRY.read_text())["candidates"] if r["usable"]]
    if include_bank and BANK.exists():
        rows += json.loads(BANK.read_text())["candidates"]
    out = []
    for r in rows:
        v = val_seeds(r)
        out.append(dict(r, val_ap=float(np.mean(v)) if v else np.nan,
                        val_range=float(np.ptp(v)) if len(v) > 1 else np.nan, n_val_seeds=len(v)))
    return out


# ------------------------------------------------------------------ features

_V1_KEYS = {}


def v1_vector(dataset, cid, b, s):
    key = _V1_KEYS.setdefault(dataset, v1b.feature_key(dataset))
    for p in (v1b.candidate_dir(dataset, cid) / f"reference_{b}_{s}.json",
              cdir(dataset, cid) / f"v1_reference_{b}_{s}.json"):
        if p.exists():
            v = json.loads(p.read_text())
            if v.get("feature_key") == key:
                return v["vector"]
    return None


_V2_KEYS = {}


def v2_reference(dataset, cid, b, s):
    key = _V2_KEYS.setdefault(dataset, feature_key(dataset))
    p = cdir(dataset, cid) / f"reference_{b}_{s}.json"
    if p.exists():
        v = json.loads(p.read_text())
        if v.get("feature_key") == key:
            return v
    return None


def nanf(x):
    return np.nan if x is None else float(x)


class Features:
    """Probe matrices of a set of rows at every reference: v1 probes (M1-M3), M4 probes, M5, global."""

    def __init__(self, rows, refs=REFERENCES, proxy=True):
        self.rows, self.refs = rows, refs
        self.ds = np.asarray([r["dataset"] for r in rows])
        self.cid = np.asarray([r["candidate"] for r in rows])
        self.fam = np.asarray([r["family"] for r in rows])
        self.base = np.asarray([r["base"] for r in rows])
        self.val = np.asarray([r.get("val_ap", np.nan) for r in rows], dtype=np.float64)
        self.X1, self.X4, self.M5, self.glob, self.meta = {}, {}, {}, {}, {}
        self.m4_names = None
        missing = []
        for ref in refs:
            X1, X4, M5, G, meta = [], [], [], [], []
            for r in rows:
                v1 = v1_vector(r["dataset"], r["candidate"], *ref)
                v2 = v2_reference(r["dataset"], r["candidate"], *ref)
                if v1 is None or v2 is None:
                    missing.append((r["dataset"], r["candidate"], ref))
                x = probes(v1) if v1 else {}
                X1.append([nanf(x.get(n)) for n in PROBE_NAMES])
                p4 = v2["m4_probes"] if v2 else {}
                if self.m4_names is None and v2:
                    self.m4_names = list(p4)
                X4.append(p4)
                m = v2["m5"] if v2 else {}
                M5.append([nanf(m.get(k)) for k in ("fn_risk", "fp_risk", "stratum_risk", "M5", "M5_no_lowconf")])
                G.append(nanf(v1.get("global", {}).get("sum")) if v1 else np.nan)
                meta.append(v2["reference"] if v2 else None)
            self.X1[ref] = np.asarray(X1, dtype=np.float64)
            names = self.m4_names or []
            self.X4[ref] = np.asarray([[nanf(p.get(n)) for n in names] for p in X4], dtype=np.float64).reshape(len(rows), len(names))
            self.M5[ref] = np.asarray(M5, dtype=np.float64)
            self.glob[ref] = np.asarray(G, dtype=np.float64)
            self.meta[ref] = meta
        self.missing = missing
        self.m4_names = self.m4_names or []
        for d, probes_ in UNMEASURED.items():
            for n in probes_:
                if n in self.m4_names:
                    for ref in refs:
                        self.X4[ref][self.ds == d, self.m4_names.index(n)] = np.nan
        self.proxy = np.full(len(rows), np.nan)
        if proxy:
            for d in DATASETS:
                runs = {}
                for f in (Path(f"results/metric/proxy/{d}.json"), Path(f"results/metric/proxy/{d}_bank.json")):
                    if f.exists():
                        runs |= json.loads(f.read_text())["runs"]
                for i in np.flatnonzero(self.ds == d):
                    self.proxy[i] = runs.get(self.cid[i], {}).get("1", {}).get("proxy", np.nan)
        self.noise = {d: float(np.nanmedian([r.get("val_range", np.nan) for r in rows if r["dataset"] == d]))
                      for d in DATASETS if (self.ds == d).any()}
        tuned = {d: self.val[(self.ds == d) & (self.cid == "tuned")] for d in self.noise}
        self.undamaged = np.asarray([np.isfinite(v) and len(tuned[d]) and v >= tuned[d][0] - self.noise[d]
                                     for v, d in zip(self.val, self.ds)])


def instability(F, X, idx, budget):
    """Reference instability per probe: median over the development rows idx of |x_j(subset s) - median_s x_j|
    over the three subsets of the budget (the 50-sequence reference: the 20-sequence value)."""
    b = 20 if budget == 50 else budget
    stack = np.stack([X[(b, s)][idx] for s in range(3)])             # (3, n, p)
    if stack.shape[-1] == 0 or stack.shape[1] == 0:
        return np.full(stack.shape[-1], np.nan)
    with np.errstate(all="ignore"):
        dev = np.abs(stack - np.nanmedian(stack, axis=0, keepdims=True))
        return np.nanmedian(dev.reshape(-1, stack.shape[-1]), axis=0)


def m2_weights(P, ap):
    """M2's NNLS (src/utility_score.nnls_weights, unchanged); if SciPy's default iteration cap (3 x columns) is
    hit, the same problem is solved again with a larger cap."""
    try:
        return nnls_weights(P, ap)
    except RuntimeError:          # the same problem (min ||A x - ap||, x >= 0) by bounded-variable least squares
        from scipy.optimize import lsq_linear
        A = np.column_stack([np.ones(len(P)), -P])
        return lsq_linear(A, ap, bounds=(0, np.inf), method="bvls").x[1:]


class Calibration:
    """Every calibrated selector fitted on development rows `dev` (bool mask) at one reference."""

    def __init__(self, F, dev, ref):
        self.F, self.ref = F, ref
        m = dev & np.isfinite(F.val)
        self.idx = np.flatnonzero(m)
        X1, X4 = F.X1[ref], F.X4[ref]
        # M1 / M2: v1 code, unchanged
        self.tol1 = Tolerance().fit(X1[m], F.undamaged[m])
        self.w2 = m2_weights(self.tol1.penalties(X1[m]), F.val[m])
        # M3: v1 probes, robust scale, weakest link
        self.inst3 = instability(F, F.X1, self.idx, ref[0])
        self.tol3 = RobustTolerance().fit(X1[m], F.undamaged[m], self.inst3)
        # M4: v2 probes
        self.inst4 = instability(F, F.X4, self.idx, ref[0])
        self.tol4 = RobustTolerance().fit(X4[m], F.undamaged[m], self.inst4)
        # M6 on the eight component risks of the development rows
        R = self.components(self.idx)
        self.m6 = PairwiseRanker().fit(R, F.val[m], F.ds[m], F.fam[m], F.noise)
        self.n_dev = int(m.sum())

    def components(self, idx):
        """(n, 8): M4 block risks A-F, M5 FN and FP risks."""
        pen = self.tol4.penalties(self.F.X4[self.ref][idx])
        out = []
        for k, i in enumerate(idx):
            r = m4(pen[k], self.F.m4_names)
            out.append([r["blocks"][b]["risk"] for b in BLOCK_NAMES] + list(self.F.M5[self.ref][i, :2]))
        return np.asarray(out, dtype=np.float64)

    def risks(self, idx, details=False):
        """Risk per method for rows idx (lower = predicted better). M3 / M4 are exact lexicographic ranks."""
        F, ref = self.F, self.ref
        P1 = self.tol1.penalties(F.X1[ref][idx])
        out = {"M1": P1.sum(1), "M2": P1 @ self.w2, "global": F.glob[ref][idx], "proxy_v1": -F.proxy[idx]}
        P3 = self.tol3.penalties(F.X1[ref][idx])
        r3 = [m3(P3[k], PROBE_NAMES) for k in range(len(idx))]
        out["M3"] = lex_rank([r["key"] for r in r3])
        P4 = self.tol4.penalties(F.X4[ref][idx])
        r4 = [m4(P4[k], F.m4_names) for k in range(len(idx))]
        out["M4"] = lex_rank([r["key"] for r in r4])
        out["M5"] = F.M5[ref][idx, 3]
        R = np.asarray([[r["blocks"][b]["risk"] for b in BLOCK_NAMES] + list(F.M5[ref][i, :2])
                        for r, i in zip(r4, idx)], dtype=np.float64)
        out["M6"] = self.m6.score(R)
        if details:
            return out, {"M3": r3, "M4": r4, "P3": P3, "P4": P4, "R": R}
        return out

    def describe(self):
        return {"n_dev": self.n_dev, "reference": list(self.ref),
                "M3_tolerance": self.tol3.describe(PROBE_NAMES),
                "M3_instability": dict(zip(PROBE_NAMES, map(_j, self.inst3))),
                "M4_tolerance": self.tol4.describe(self.F.m4_names),
                "M4_instability": dict(zip(self.F.m4_names, map(_j, self.inst4))),
                "M6": {"inputs": list(M6_INPUTS), "weights": self.m6.w.tolist(), "mu": self.m6.mu.tolist(),
                       "sd": self.m6.sd.tolist(), "n_pairs": len(self.m6.pairs),
                       "family_weights": getattr(self.m6, "family_weights", {})},
                "M2_weights": dict(zip(PROBE_NAMES, self.w2.tolist()))}


def _j(x):
    return None if x is None or not np.isfinite(x) else float(x)


def random_metrics(ap, noise):
    return {"regret": float(ap.max() - ap.mean()), "within_noise": float((ap >= ap.max() - noise).mean()),
            "pairwise": 0.5, "kendall": 0.0, "spearman": 0.0}


def evaluate(cal, idx, noise):
    ap = cal.F.val[idx]
    risks = cal.risks(idx)
    out = {m: evaluate_set(risks[m], ap, noise) for m in METHODS}
    out["random"] = random_metrics(ap, noise)
    out["oracle"] = cal.F.cid[idx[np.argmax(ap)]]
    out["picks"] = {m: [cal.F.cid[i] for i in idx[np.flatnonzero(risks[m] == np.nanmin(np.where(np.isfinite(risks[m]), risks[m], np.inf)))]]
                    for m in METHODS}
    return out


KEYS = ("regret", "within_noise", "pairwise", "kendall", "spearman")


def mean_over(results, methods):
    return {m: {k: float(np.nanmean([r[m][k] for r in results if r[m].get(k) is not None]))
                if any(r[m].get(k) is not None for r in results) else None for k in KEYS} for m in methods}


def lofo(F, dataset, ref):
    """Leave-one-family-out within a radar: the family and its base(s) held out together."""
    has_val = np.isfinite(F.val)
    in_ds = (F.ds == dataset) & has_val
    fams = sorted({f for f in F.fam[in_ds] if f != "natural" and (F.fam[in_ds] == f).sum() >= 2})
    noise = F.noise[dataset]
    per = {}
    for f in fams:
        members = in_ds & (F.fam == f)
        bases = set(F.base[members])
        anchors = {"tuned", "spec"} if f == "fixes" else {b for b in bases if b in ("tuned", "spec")}
        anchor = in_ds & np.isin(F.cid, sorted(anchors))
        dev = in_ds & ~members & ~anchor
        cal = Calibration(F, dev, ref)
        res = []
        for base in sorted(bases):
            m = members & (F.base == base)
            a = in_ds & np.isin(F.cid, sorted(anchors)) if f == "fixes" else in_ds & (F.cid == base)
            idx = np.flatnonzero(m | a)
            if len(idx) >= 2:
                res.append(evaluate(cal, idx, noise))
        if not res:
            continue
        per[f] = mean_over(res, METHODS + ("random",)) | {
            "n_dev": cal.n_dev, "sets": [[str(c) for c in r["picks"]["M1"]] for r in res],
            "held_out": sorted(anchors) + [str(c) for c in F.cid[members]]}
    summary = {m: {k: float(np.nanmean([per[f][m][k] for f in per if per[f][m][k] is not None])) for k in KEYS}
               for m in METHODS + ("random",)}
    return {"families": per, "summary": summary, "n_families": len(per)}


def loro(F, held, ref):
    has_val = np.isfinite(F.val)
    cal = Calibration(F, (F.ds != held) & has_val, ref)
    out = {"n_dev": cal.n_dev, "calibration": cal.describe()}
    for name, mask in (("natural", np.isin(F.fam, NATURAL)), ("all", np.ones(len(F.rows), bool))):
        idx = np.flatnonzero((F.ds == held) & has_val & mask)
        out[name] = evaluate(cal, idx, F.noise[held]) | {"candidates": F.cid[idx].tolist(),
                                                         "val_ap": F.val[idx].tolist()}
        risks, det = cal.risks(idx, details=True)
        out[name]["diagnosis"] = {F.cid[i]: {"M4_top_blocks": det["M4"][k]["top_blocks"],
                                             "M4_diagnosis": det["M4"][k]["diagnosis"],
                                             "M3_dominant": det["M3"][k]["dominant"]} for k, i in enumerate(idx)}
    return out


def finalist(dev_primary):
    """Frozen rule: (1) lowest mean LORO natural regret; (2) within 0.01: higher mean LORO natural pairwise;
    (3) lower mean LOFO regret; (4) order M4, M3, M5, M6."""
    lo = dev_primary["loro"]
    reg = {m: float(np.mean([lo[d]["natural"][m]["regret"] for d in DATASETS])) for m in NEW}
    pw = {m: float(np.nanmean([np.nan if lo[d]["natural"][m]["pairwise"] is None else lo[d]["natural"][m]["pairwise"]
                               for d in DATASETS])) for m in NEW}
    lf = {m: float(np.mean([dev_primary["lofo"][d]["summary"][m]["regret"] for d in DATASETS])) for m in NEW}
    best = min(reg.values())
    step1 = [m for m in NEW if reg[m] <= best + 0.01 + 1e-12]
    trail = ["(1) LORO natural regret: " + ", ".join(f"{m} {reg[m]:.4f}" for m in NEW)
             + f"; within 0.01 of the best: {step1}"]
    cand = step1
    if len(cand) > 1:
        top = max(pw[m] for m in cand)
        cand = [m for m in cand if abs(pw[m] - top) < 1e-9]
        trail.append("(2) LORO natural pairwise: " + ", ".join(f"{m} {pw[m]:.4f}" for m in step1) + f" -> {cand}")
    if len(cand) > 1:
        low = min(lf[m] for m in cand)
        cand = [m for m in cand if abs(lf[m] - low) < 1e-9]
        trail.append("(3) LOFO regret: " + ", ".join(f"{m} {lf[m]:.4f}" for m in cand) + f" -> {cand}")
    if len(cand) > 1:
        cand = [min(cand, key=FINALIST_ORDER.index)]
        trail.append(f"(4) order M4, M3, M5, M6 -> {cand}")
    return {"finalist": cand[0], "loro_natural_regret": reg, "loro_natural_pairwise": pw, "lofo_regret": lf,
            "rule_trail": trail}


def verify_v1():
    """M1 / M2 through this code path reproduce v1's Phase 6 rankings (same rows, reference 5/0, LORO)."""
    rows = dev_rows(include_bank=False)
    F = Features(rows, refs=[(5, 0), (5, 1), (5, 2)], proxy=False)
    # v1's noise and undamaged used the same rows (val only)
    report = {}
    for held in ("carrada", "raddet"):
        sel = json.loads(Path(f"results/metric/prospective/{held}_selection.json").read_text())
        cal = Calibration(F, (F.ds != held) & np.isfinite(F.val), (5, 0))
        bank = [r for r in json.loads(BANK.read_text())["candidates"] if r["dataset"] == held]
        Fb = Features(bank, refs=[(5, 0), (5, 1), (5, 2)], proxy=False)
        P = cal.tol1.penalties(Fb.X1[(5, 0)])
        got = {c: (float(p.sum()), float(p @ cal.w2)) for c, p in zip(Fb.cid, P)}
        want = {x["candidate"]: (x["M1"], x["M2"]) for x in sel["ranking"] if x["bank"]}
        diff = max(max(abs(got[c][0] - want[c][0]), abs(got[c][1] - want[c][1])) for c in want)
        report[held] = {"max_abs_diff": diff, "ok": bool(diff < 1e-6 * max(1.0, max(abs(w[0]) for w in want.values())))}
        print(held, report[held])
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-v1", action="store_true")
    a = ap.parse_args()
    if a.verify_v1:
        rep = verify_v1()
        SCORES.mkdir(parents=True, exist_ok=True)
        (SCORES / "verify_v1.json").write_text(json.dumps({"provenance": provenance(), "checks": rep}, indent=1))
        return
    t0 = time.time()
    rows = dev_rows()
    F = Features(rows)
    if F.missing:
        raise SystemExit(f"{len(F.missing)} (candidate, reference) features missing, e.g. {F.missing[:3]}")
    report = {"provenance": provenance(), "outcome": "whole-val AP@2 (mean of seeds); test AP not read",
              "n_candidates": {d: int(((F.ds == d) & np.isfinite(F.val)).sum()) for d in DATASETS},
              "n_natural": {d: int(((F.ds == d) & np.isfinite(F.val) & np.isin(F.fam, NATURAL)).sum()) for d in DATASETS},
              "noise_val": F.noise, "n_undamaged": {d: int((F.undamaged & (F.ds == d)).sum()) for d in DATASETS},
              "references": {}}
    for ref in REFERENCES:
        r = {"lofo": {d: lofo(F, d, ref) for d in DATASETS}, "loro": {d: loro(F, d, ref) for d in DATASETS}}
        key = f"{ref[0]}_{ref[1]}"
        report["references"][key] = r
        print(key, " | ".join(f"{d}: " + " ".join(f"{m} {r['loro'][d]['natural'][m]['regret']:.3f}"
                                                  for m in ("M1", "M3", "M4", "M5", "M6", "global"))
                              for d in DATASETS), flush=True)
    prim = report["references"][f"{PRIMARY[0]}_{PRIMARY[1]}"]
    fin = finalist(prim)
    report["finalist"] = fin
    report["seconds"] = time.time() - t0
    SCORES.mkdir(parents=True, exist_ok=True)
    (SCORES / "development.json").write_text(json.dumps(report, indent=1, default=_j))
    Path("results/metric_v2/finalist.json").write_text(json.dumps(
        {"provenance": provenance(), "reference": list(PRIMARY), "outcome": "whole-val AP@2",
         "feature_keys": {d: feature_key(d) for d in DATASETS}} | fin, indent=1, default=_j))
    print("finalist:", fin["finalist"], *fin["rule_trail"], sep="\n")


if __name__ == "__main__":
    main()
