"""Raw probes and adequacy vectors of candidates (docs/notes/2026-11-06-metric-experiments.md, Phase 1).

For each candidate of results/metric/candidates.json: the raw tables of its 200 synthetic sequences
(src/utility_metric.raw) -> results/metric/features/<dataset>/<candidate>/raw.npz, and its vector
against each real reference subset -> .../reference_<budget>_<subset>.json. The real reference is
the 50 real training sequences (raw once, .../real/raw.npz); subset 0 of a budget is evenly spaced
(the budget study's choice), subsets 1.. are random sequence draws (seeded). Only training splits
are read: never val or test.

--check: does each controlled intervention move its intended probes, relative to the tuned
simulator it was made from? Paired bootstrap over the 200 sequences (the versions share seeds and
scenes), 95% interval of the difference in medians / rates. Writes .../probe_check.json.

    PYTHONPATH=. python3 scripts/build_metric_features.py --dataset carrada \
        --candidates spec,tuned,study2:rangeshift1,study2:extra50,study2:unrender90 --check
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src import data, versions
from src.provenance import config_hash, provenance
from src.utility_metric import PRIMARY_Q, RADARS, config, raw, select, summarise, thresholds, rates
from src.versions import DATASETS, N_SIM, ON_BEST, SUITES, best, render, version

OUT = Path("results/metric/features")
REGISTRY = Path("results/metric/candidates.json")
N_REAL = 50
_LOAD = data.load


def train_only(cache, split, limit=None):
    if split != "train":
        raise RuntimeError(f"the feature builder reads training splits only, not {split!r} of {cache}")
    return _LOAD(cache, split, limit)


def classes_of(cache):
    p = Path(cache) / "train_labels.pt"
    if p.exists():
        lab = torch.load(p, weights_only=False)
        if "cls" in lab:
            return lab["cls"].numpy()
    return None


def load_candidate(row):
    ds = DATASETS[row["dataset"]]
    kind, value = SUITES[row["suite"]][0][row["version"]]
    clean = (train_only(render(row["dataset"], "best", "tuned2", best(row["dataset"])), "train", N_SIM)
             if row["suite"] in ON_BEST else train_only(ds.sim, "train", N_SIM))
    pool = version(row["dataset"], row["version"], kind, value, clean)
    cache = row["cache"]
    cls = classes_of(ds.sim) if cache.startswith("(on the spec") else (
        classes_of(cache) if Path(cache).is_dir() else None)
    return pool, (cls[:len(pool)] if cls is not None else None)


def save_raw(path, tables, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {f"{t}/{k}": v.astype(np.float32) for t, cols in tables.items() for k, v in cols.items()}
    np.savez_compressed(path, **flat)
    (path.with_suffix(".json")).write_text(json.dumps(meta, indent=1))


def load_raw(path):
    z = np.load(path)
    out = {}
    for key in z.files:
        t, k = key.split("/")
        out.setdefault(t, {})[k] = z[key].astype(np.float64)
    return out


def fresh(path):
    """A saved raw table built with the current feature definition."""
    meta = path.with_suffix(".json")
    return path.exists() and meta.exists() and \
        json.loads(meta.read_text())["feature_config_hash"] == config_hash(config())


def subsets(budget, n_subsets):
    out = {0: np.linspace(0, N_REAL - 1, budget).round().astype(int)}
    for s in range(1, n_subsets):
        g = np.random.default_rng(1000 * s + budget)
        out[s] = np.sort(g.choice(N_REAL, budget, replace=False))
    return out


def candidate_dir(dataset, cid):
    return OUT / dataset / cid.replace(":", "__")


def real_tables(dataset):
    path = OUT / dataset / "real" / "raw.npz"
    if fresh(path):
        return load_raw(path)
    ds = DATASETS[dataset]
    t0 = time.time()
    pool = train_only(ds.real, "train")
    tables = raw(pool, RADARS[dataset], classes_of(ds.real))
    save_raw(path, tables, {"provenance": provenance(), "cache": ds.real, "split": "train",
                            "n_sequences": len(pool), "feature_config": config(),
                            "feature_config_hash": config_hash(config()), "seconds": time.time() - t0})
    return tables


def to_json(x):
    if isinstance(x, dict):
        return {str(k): to_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


# ------------------------------------------------------------------ probe check

# candidate -> [(probe, direction)]: what the intervention must move, relative to the tuned simulator
EXPECT = {
    "study2:rangeshift1": [("peak_dr", -1), ("cen_dr", -1)],
    "study2:dopplershift1": [("peak_dd", -1), ("cen_dd", -1)],
    # the total unlabelled count falls: extra objects raise the map std, so every other peak shrinks
    # in z (each source is normalised by its own statistics); off the ridge the count tracks them
    "study2:extra50": [("unlabelled_off_ridge_per_frame", +1), ("unlabelled_per_frame", +1)],
    "study2:unrender90": [("labels_with_peak_r2", -1), ("label_prom", -1)],
    "study2:contrast-12": [("label_prom", -1), ("over_floor", -1)],
    "study2:blur2": [("width_r2", +1), ("width_d2", +1), ("area2", +1)],
    "study2b:tail2": [("far_ext2", +1), ("tail_asym2", +1), ("tail_level", +1)],
}


def probe_value(tables, probe, thr):
    if probe in ("unlabelled_per_frame", "unlabelled_off_ridge_per_frame", "labels_with_peak_r2"):
        return rates(tables, thr)[probe]
    v = tables["objects"][probe]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="carrada", choices=tuple(RADARS))
    ap.add_argument("--candidates", required=True, help="comma-separated candidate ids (registry)")
    ap.add_argument("--budgets", default="5,10")
    ap.add_argument("--subsets", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    data.load = versions.load = train_only          # nothing below may read val or test
    reg = {r["candidate"]: r for r in json.loads(REGISTRY.read_text())["candidates"] if r["dataset"] == a.dataset}
    real = real_tables(a.dataset)
    budgets = [int(b) for b in a.budgets.split(",")] + [N_REAL]
    cands = {}
    for cid in a.candidates.split(","):
        row = reg[cid]
        d = candidate_dir(a.dataset, cid)
        t0 = time.time()
        if fresh(d / "raw.npz"):
            tables = load_raw(d / "raw.npz")
        else:
            pool, cls = load_candidate(row)
            tables = raw(pool, RADARS[a.dataset], cls)
            save_raw(d / "raw.npz", tables, {"provenance": provenance(), "candidate": row,
                                             "n_sequences": len(pool), "feature_config": config(),
                                             "feature_config_hash": config_hash(config()),
                                             "seconds": time.time() - t0})
            del pool
        raw_s = time.time() - t0
        cands[cid] = tables
        for b in budgets:
            for s, seqs in (subsets(b, a.subsets) if b < N_REAL else {0: np.arange(N_REAL)}).items():
                t1 = time.time()
                vec = summarise(tables, select(real, seqs))
                (d / f"reference_{b}_{s}.json").write_text(json.dumps(to_json({
                    "provenance": provenance(), "dataset": a.dataset, "split": "train",
                    "candidate": cid, "family": row["family"], "level": row["level"], "suite": row["suite"],
                    "version": row["version"], "reference": {"budget": b, "subset": s,
                                                             "sequence_indices": seqs.tolist()},
                    "feature_config_hash": config_hash(config()), "raw_seconds": raw_s,
                    "summary_seconds": time.time() - t1, "vector": vec}), indent=1))
        v = json.loads((d / f"reference_{N_REAL}_0.json").read_text())["vector"]
        print(f"{cid:22s} n_obj {len(tables['objects']['seq']):5d}  scores vs 50 real: "
              + " ".join(f"{k} {x:.2f}" for k, x in v["scores"].items()), flush=True)
    if a.check:
        run_check(a.dataset, cands, real)


_BY_SEQ = {}


def resample(tables, idx):
    """The rows of sequences idx (with repeats), sequence by sequence."""
    if id(tables) not in _BY_SEQ:
        n = int(tables["frames"]["seq"].max()) + 1
        _BY_SEQ[id(tables)] = {name: [np.flatnonzero(cols["seq"] == i) for i in range(n)]
                               for name, cols in tables.items()}
    by_seq = _BY_SEQ[id(tables)]
    out = {}
    for name, rows in by_seq.items():
        take = np.concatenate([rows[i] for i in idx])
        out[name] = {k: v[take] for k, v in tables[name].items()}
    return out


def run_check(dataset, cands, real, n_boot=500, seed=0):
    thr = thresholds(real)[PRIMARY_Q]
    base = cands["tuned"]
    g = np.random.default_rng(seed)
    n = int(base["frames"]["seq"].max()) + 1
    report = {"provenance": provenance(), "threshold_q": PRIMARY_Q, "threshold": thr, "n_boot": n_boot, "checks": {}}
    for cid, probes in EXPECT.items():
        if cid not in cands:
            continue
        cand = cands[cid]
        boots = [g.integers(n, size=n) for _ in range(n_boot)]
        res = {}
        for probe, sign in probes:
            diff = probe_value(cand, probe, thr) - probe_value(base, probe, thr)
            bd = [probe_value(resample(cand, idx), probe, thr) - probe_value(resample(base, idx), probe, thr)
                  for idx in boots]
            lo, hi = np.nanpercentile(bd, (2.5, 97.5))
            ok = bool(lo > 0) if sign > 0 else bool(hi < 0)
            res[probe] = {"expected": "up" if sign > 0 else "down", "diff": diff, "ci95": [lo, hi], "moved": ok}
            print(f"check {cid:22s} {probe:22s} expected {'up  ' if sign > 0 else 'down'} "
                  f"diff {diff:+.3f} [{lo:+.3f}, {hi:+.3f}] {'OK' if ok else 'NOT MOVED'}", flush=True)
        report["checks"][cid] = {"probes": res, "any_moved": any(r["moved"] for r in res.values())}
    (OUT / dataset / "probe_check.json").write_text(json.dumps(to_json(report), indent=1))


if __name__ == "__main__":
    main()
