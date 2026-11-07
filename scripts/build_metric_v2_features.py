"""Metric v2 features (docs/notes/2026-11-06-metric-v2-preregistration.md, Phases C-D).

Per candidate (results/metric/candidates.json usable rows, the first bank results/metric/bank.json and,
once rendered, bank v2 results/metric_v2/bank_v2.json):
- raw v2 tables (src/utility_metric_v2.raw_v2, with the fixed negative detector's detections)
  -> results/metric_v2/features/<dataset>/<candidate>/raw_v2.npz;
- the v1 adequacy vector at every reference: read from results/metric/features when it exists and is
  current (never rewritten), else computed from the v1 raw tables and written under results/metric_v2/
  (budget 20 always; bank v2 everything, its v1 raw tables too);
- per reference (budgets 5, 10, 20 x subsets 0-2, and all 50): M4 probes and the M5 coverage summary
  -> .../reference_<budget>_<subset>.json.
The real references are the 50 real *training* sequences only; nothing here can read val or test.

--check: the preregistered causal checks (paired sequence bootstrap vs the tuned simulator).

    PYTHONPATH=. python3 scripts/build_metric_v2_features.py --dataset carrada --candidates all --part 0/4
"""
import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np
import torch

import scripts.build_metric_features as v1b
from src import data, versions
from src.provenance import config_hash, provenance
from src.utility_metric import RADARS, config as v1_config, raw, select, summarise
from src.utility_metric_v2 import config, m4_probes, m5, raw_v2

OUT = Path("results/metric_v2/features")
V1 = Path("results/metric/features")
BANK2 = Path("results/metric_v2/bank_v2.json")
N_REAL = 50
REFERENCES = [(b, s) for b in (5, 10, 20) for s in range(3)] + [(50, 0)]
PRIMARY = (10, 0)
_LOAD = data.load


def train_only(cache, split, limit=None):
    if split != "train":
        raise RuntimeError(f"the v2 feature builder reads training splits only, not {split!r} of {cache}")
    return _LOAD(cache, split, limit)


def ref_seqs(b, s):
    return v1b.subsets(b, 3)[s] if b < N_REAL else np.arange(N_REAL)


def registry(dataset):
    rows = v1b.registry(dataset)
    if BANK2.exists():
        rows |= {r["candidate"]: r for r in json.loads(BANK2.read_text())["candidates"] if r["dataset"] == dataset}
    return rows


def load_pool(row):
    if row["suite"] in ("bank", "bank2"):
        return train_only(row["cache"], "train", versions.N_SIM)
    return v1b.load_candidate(row)[0]


def cdir(dataset, cid):
    return OUT / dataset / cid.replace(":", "__")


def feature_key(dataset):
    """v2 feature definition + the fixed detector's weights (+ the v1 key, as v1 vectors are read)."""
    return config_hash(config() | {"v1": v1b.feature_key(dataset)})


def hardware():
    return {"host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def save_tables(path, tables, meta):
    v1b.save_raw(path, tables, meta)


def fresh(path, key):
    meta = path.with_suffix(".json")
    return path.exists() and meta.exists() and json.loads(meta.read_text()).get("feature_key") == key


def real_v2(dataset):
    path = OUT / dataset / "real" / "raw_v2.npz"
    key = feature_key(dataset)
    if fresh(path, key):
        return v1b.load_raw(path)
    t0 = time.time()
    pool = train_only(v1b.DATASETS[dataset].real, "train")
    tables = raw_v2(pool, RADARS[dataset], v1b.detector_dets(dataset, pool))
    save_tables(path, tables, {"provenance": provenance(), "hardware": hardware(), "cache": v1b.DATASETS[dataset].real,
                               "split": "train", "n_sequences": len(pool), "feature_config": config(),
                               "feature_key": key, "seconds": time.time() - t0})
    return tables


def real_v1(dataset):
    """v1 real tables: results/metric's (current), never rebuilt here."""
    path = V1 / dataset / "real" / "raw.npz"
    if not v1b.fresh(path, dataset):
        raise SystemExit(f"{path} is missing or stale; v1 tables are not rebuilt by the v2 builder")
    return v1b.load_raw(path)


class Candidate:
    """Raw v1 / v2 tables of one candidate, built once and cached."""

    def __init__(self, dataset, row):
        self.dataset, self.row, self.cid = dataset, row, row["candidate"]
        self.dir = cdir(dataset, self.cid)
        self._pool = self._dets = self._v1 = None
        self.seconds = {}

    def pool(self):
        if self._pool is None:
            self._pool = load_pool(self.row)
            t0 = time.time()
            self._dets = v1b.detector_dets(self.dataset, self._pool)
            self.seconds["detector"] = time.time() - t0
        return self._pool

    def v2(self):
        path, key = self.dir / "raw_v2.npz", feature_key(self.dataset)
        if fresh(path, key):
            return v1b.load_raw(path)
        pool = self.pool()
        t0 = time.time()
        tables = raw_v2(pool, RADARS[self.dataset], self._dets)
        self.seconds["raw_v2"] = time.time() - t0
        save_tables(path, tables, {"provenance": provenance(), "hardware": hardware(), "candidate": self.row,
                                   "n_sequences": len(pool), "feature_config": config(), "feature_key": key,
                                   "seconds": self.seconds["raw_v2"] + self.seconds["detector"]})
        return tables

    def v1_tables(self):
        if self._v1 is None:
            p = v1b.candidate_dir(self.dataset, self.cid) / "raw.npz"
            if v1b.fresh(p, self.dataset):
                self._v1 = v1b.load_raw(p)
            else:
                q = self.dir / "raw_v1.npz"
                if v1b.fresh(q, self.dataset):
                    self._v1 = v1b.load_raw(q)
                else:
                    pool = self.pool()
                    t0 = time.time()
                    self._v1 = raw(pool, RADARS[self.dataset], None, self._dets)
                    save_tables(q, self._v1, {"provenance": provenance(), "candidate": self.row,
                                              "n_sequences": len(pool), "feature_config": v1_config(),
                                              "feature_config_hash": config_hash(v1_config()),
                                              "feature_key": v1b.feature_key(self.dataset),
                                              "seconds": time.time() - t0})
        return self._v1

    def v1_vector(self, b, s, real1):
        """The v1 adequacy vector at a reference (results/metric's when current, else computed here)."""
        p = v1b.candidate_dir(self.dataset, self.cid) / f"reference_{b}_{s}.json"
        key = v1b.feature_key(self.dataset)
        if p.exists():
            v = json.loads(p.read_text())
            if v.get("feature_key") == key:
                return v["vector"]
        q = self.dir / f"v1_reference_{b}_{s}.json"
        if q.exists():
            v = json.loads(q.read_text())
            if v.get("feature_key") == key:
                return v["vector"]
        seqs = ref_seqs(b, s)
        vec = v1b.to_json(summarise(self.v1_tables(), select(real1, seqs)))
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(json.dumps({"provenance": provenance(), "dataset": self.dataset, "split": "train",
                                 "candidate": self.cid, "reference": {"budget": b, "subset": s,
                                                                      "sequence_indices": seqs.tolist()},
                                 "feature_key": key, "vector": vec}))
        return vec


def ref_meta(ref, b, s):
    o, n = ref["objects"], ref["negatives"]
    fn = int((o["det_score_r2"] == 0).sum())
    fp = int((n["kind"] == 1).sum())
    return {"budget": b, "subset": s, "sequence_indices": ref_seqs(b, s).tolist(),
            "n_sequences": int(len(ref_seqs(b, s))), "n_labels": int(len(o["seq"])), "n_fn": fn, "n_fp": fp,
            "low_confidence": {"labels": bool(len(o["seq"]) < 100), "fn": bool(fn < 20), "fp": bool(fp < 20)}}


def score_reference(cand2, real2, vec1, b, s):
    seqs = ref_seqs(b, s)
    ref = select(real2, seqs)
    med = float(np.median(ref["objects"]["range_m"]))
    P, meta = m4_probes(cand2, ref, vec1, med)
    M5 = m5(cand2, ref, med)
    return P, meta, M5, ref


def build(dataset, row, real1, real2, refs=REFERENCES):
    c = Candidate(dataset, row)
    key = feature_key(dataset)
    todo = []
    for b, s in refs:
        p = c.dir / f"reference_{b}_{s}.json"
        if not (p.exists() and json.loads(p.read_text()).get("feature_key") == key):
            todo.append((b, s))
    if not todo:
        return c.cid, 0.0
    t0 = time.time()
    cand2 = c.v2()
    raw_s = time.time() - t0
    for b, s in todo:
        t1 = time.time()
        vec1 = c.v1_vector(b, s, real1)
        P, meta, M5, ref = score_reference(cand2, real2, vec1, b, s)
        out = {"provenance": provenance(), "hardware": hardware(), "metric_version": "v2",
               "dataset": dataset, "candidate": c.cid, "family": row.get("family"), "level": row.get("level"),
               "suite": row.get("suite"), "version": row.get("version"), "cache": row.get("cache"),
               "radar": {k: float(v) if not isinstance(v, bool) else v for k, v in vars(RADARS[dataset]).items()},
               "feature_config": config(), "feature_config_hash": config_hash(config()), "feature_key": key,
               "reference": ref_meta(ref, b, s) | {"m4": meta},
               "n_candidate": {"objects": int(len(cand2["objects"]["seq"])),
                               "negatives": int(len(cand2["negatives"]["seq"])),
                               "frames": int(len(cand2["frames"]["seq"]))},
               "m4_probes": P, "m5": M5, "v1_global": vec1.get("global", {}).get("sum"),
               "raw_seconds": raw_s, "summary_seconds": time.time() - t1}
        (c.dir / f"reference_{b}_{s}.json").write_text(json.dumps(v1b.to_json(out)))
    return c.cid, time.time() - t0


# ------------------------------------------------------------------ causal checks (Phase C)

def _med(a):
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else np.nan


def _vis(objs, thr):
    return float(np.mean(objs["peak_prom"] >= thr)) if len(objs["peak_prom"]) else np.nan


# candidate -> [(feature, direction)], relative to the tuned simulator (same scene seeds); fixed in the note
# before the check is run. Feature names: raw v2 object columns (median), or the listed special probes.
CHECKS = {
    "study2:contrast-12": [("cfar_db", -1), ("cfar_z", -1), ("visibility", -1)],
    "study2:noise+12": [("cfar_db", -1), ("cfar_z", -1)],
    "study2:rangeshift1": [("off_r", -1), ("capture_r2", -1)],
    "study2:dopplershift1": [("off_d", -1), ("capture_r2", -1)],
    "study2:blur2": [("width_r6_m", +1), ("width_d6_mps", +1)],
    "study2b:tail2": [("far6_m", +1), ("asym6_m", +1)],
    "study2:unrender90": [("visibility", -1), ("capture_r2", -1)],
    "study2:extra50": [("v1_unlabelled_off_ridge_per_frame", +1)],
    "study2b:support100": [("support_under_worst", +1)],
    "study2b:confusers4": [("fp_per_frame", +1)],
    "study2b:confusers0": [("neg_uncovered", +1)],
}


def check_value(name, t2, t1, ref2, ref1, thr, med):
    from src.utility_metric import PRIMARY_Q, rates, thresholds
    o = t2["objects"]
    if name == "visibility":
        return _vis(o, thr)
    if name == "fp_per_frame":
        return float(np.mean(t2["frames"]["n_fp"]))
    if name == "support_under_worst":
        from src.utility_metric_v2 import strata_of
        cs = strata_of(o["range_m"], o["near_zero"], med)
        rs = strata_of(ref2["objects"]["range_m"], ref2["objects"]["near_zero"], med)
        return float(max(max(0.0, (rs == b).mean() - (cs == b).mean()) for b in range(4)))
    if name == "neg_uncovered":
        from src.utility_metric_v2 import NEG_FEATURES, _feat, _std, coverage, coverage_summary
        rn = ref2["negatives"]
        base = _feat(rn, NEG_FEATURES)
        Q = _std(base, base)
        tc, tr = coverage(Q, rn["seq"], Q, rn["seq"], _std(_feat(t2["negatives"], NEG_FEATURES), base))
        return coverage_summary(tc, tr)["uncovered"]
    if name.startswith("v1_"):
        return rates(t1, thresholds(ref1)[PRIMARY_Q])[name[3:]]
    return _med(o[name])


def run_checks(dataset, real1, real2, n_boot=300, seed=0, only=None):
    reg = registry(dataset)
    checks = {c: v for c, v in CHECKS.items() if only is None or c in only}
    ids = ["tuned"] + [c for c in checks if c in reg]
    t2 = {c: Candidate(dataset, reg[c]).v2() for c in ids}
    t1 = {c: Candidate(dataset, reg[c]).v1_tables() for c in ids}
    thr = float(np.quantile(real2["objects"]["peak_prom"][np.isfinite(real2["objects"]["peak_prom"])], 0.25))
    med = float(np.median(real2["objects"]["range_m"]))
    n = 200
    g = np.random.default_rng(seed)
    boots = [g.integers(n, size=n) for _ in range(n_boot)]
    rs = lambda tables, idx: v1b.resample(tables, idx)
    report = {"provenance": provenance(), "reference": "all 50 real training sequences", "n_boot": n_boot,
              "visibility_threshold": thr, "checks": {}}
    for cid in checks:
        if cid not in reg:
            continue
        res = {}
        for feat, sign in checks[cid]:
            val = lambda a2, a1: check_value(feat, a2, a1, real2, real1, thr, med)
            diff = val(t2[cid], t1[cid]) - val(t2["tuned"], t1["tuned"])
            bd = []
            for idx in boots:
                bd.append(val(rs(t2[cid], idx), rs(t1[cid], idx)) - val(rs(t2["tuned"], idx), rs(t1["tuned"], idx)))
            lo, hi = np.nanpercentile(bd, (2.5, 97.5))
            ok = bool(lo > 0) if sign > 0 else bool(hi < 0)
            res[feat] = {"expected": "up" if sign > 0 else "down", "diff": diff, "ci95": [lo, hi], "moved": ok}
            print(f"check {dataset} {cid:22s} {feat:34s} {'up  ' if sign > 0 else 'down'} {diff:+.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}] {'OK' if ok else 'NOT MOVED'}", flush=True)
        report["checks"][cid] = res
    path = Path("results/metric_v2/features") / dataset / ("causal_check.json" if only is None else "causal_check_recheck.json")
    path.write_text(json.dumps(v1b.to_json(report), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="carrada", choices=tuple(RADARS))
    ap.add_argument("--candidates", default="all", help="'all' (usable + bank), 'bank', 'bank2', or ids")
    ap.add_argument("--part", default="0/1", help="k/n: every n-th candidate from k (parallel workers)")
    ap.add_argument("--check", action="store_true", help="only the causal checks")
    ap.add_argument("--only", default=None, help="comma-separated check candidates (a re-check)")
    a = ap.parse_args()
    data.load = versions.load = v1b.data.load = v1b.versions.load = train_only    # never val or test
    v1b._LOAD = _LOAD
    real1, real2 = real_v1(a.dataset), real_v2(a.dataset)
    if a.check:
        run_checks(a.dataset, real1, real2, only=a.only.split(",") if a.only else None)
        return
    reg = registry(a.dataset)
    if a.candidates == "all":
        ids = [c for c, r in reg.items() if r.get("usable") or r["suite"] == "bank"]
    elif a.candidates in ("bank", "bank2"):
        ids = [c for c, r in reg.items() if r["suite"] == a.candidates]
    else:
        ids = a.candidates.split(",")
    k, n = map(int, a.part.split("/"))
    for cid in ids[k::n]:
        cid, sec = build(a.dataset, reg[cid], real1, real2)
        print(f"{a.dataset} {cid:28s} {sec:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
