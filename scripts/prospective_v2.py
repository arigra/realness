"""Metric v2 prospective test on candidate bank v2 (docs/notes/2026-11-06-metric-v2-preregistration.md, G-J).

  render   write the bank-v2 caches (src/bank_v2.BANK2) and results/metric_v2/bank_v2.json
  proxy    the v1 proxy detector (width 8, 50 sequences, 1,000 steps), seeds 1-3, for every bank-v2 candidate;
           AP@2 on every reference subset of the real training sequences (primary: 10 sequences, subset 0)
  score    every selector (global, M1, M2, M3, M4, M5, M6, proxy) on the bank at every reference, calibrated on
           all development candidates of the three radars (leave-one-radar-out calibration: secondary);
           rankings, one selection per method, sequence-bootstrap intervals
           -> results/metric_v2/prospective/<dataset>_selection.json  (commit before training)
  freeze-check  confirm no bank-v2 full-detector result exists
  train    synthetic-only U-Net, Study 2's training (early stopping on 32 val sequences), seeds 1-3
  augment  RADDet: each bank-v2 candidate + the fixed 5 / 10 real sequences of the budget study, seeds 1-3
  report   regret, within-noise success, pairwise, Kendall / Spearman, top-3, stability, diagnosis

    PYTHONPATH=. python3 scripts/prospective_v2.py render --dataset carrada --names scnr_narrow
"""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import scripts.build_metric_features as v1b
from scripts.build_metric_v2_features import BANK2, PRIMARY, REFERENCES, cdir, feature_key, ref_seqs
from scripts.evaluate_metric_v2 import (DATASETS as ALL, Calibration, Features, _j, dev_rows)
from src.bank_v2 import BANK2 as SPEC, render as render_one
from src.provenance import provenance
from src.utility_score import PROBE_NAMES, evaluate_set
from src.utility_score_v2 import BLOCK_NAMES

OUT = Path("results/metric_v2/prospective")
DATASETS = ("carrada", "raddet")
METHODS = ("global", "M1", "M2", "M3", "M4", "M5", "M6", "proxy")
EXPECTED_BLOCKS = {"scnr_narrow": {"A"}, "scnr_wide": {"A"}, "shape_range": {"C"}, "shape_doppler": {"C"},
                   "label_stochastic": {"B"}, "support_stratified": {"D"}, "combined_moderate": {"A", "C", "D"},
                   "tuned_control": None}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_j))
    tmp.replace(path)


# ------------------------------------------------------------------ render

def render(a):
    ref = ref_seqs(*PRIMARY)
    names = a.names.split(",") if a.names else list(SPEC)
    for ds in a.dataset.split(","):
        for name in names:
            t0 = time.time()
            out, info = render_one(ds, name, ref)
            print(ds, name, out, f"{time.time() - t0:.0f}s", flush=True)
            row = {"dataset": ds, "candidate": f"bank2:{name}", "suite": "bank2", "version": name,
                   "family": "bank2", "level": None, "base": "tuned", "setting": "only", "n_sim_sequences": 200,
                   "cache": str(out), "usable": False, "entries": [],
                   "spec": {k: list(v) if isinstance(v, tuple) else v for k, v in SPEC[name].items()},
                   "primary_reference_sequences": ref.tolist(), "render_info": info}
            # merge with what is on disk now (several renderers run in parallel)
            BANK2.parent.mkdir(parents=True, exist_ok=True)
            with open(BANK2.with_suffix(".lock"), "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                rows = json.loads(BANK2.read_text())["candidates"] if BANK2.exists() else []
                rows = [r for r in rows if (r["dataset"], r["candidate"]) != (ds, row["candidate"])] + [row]
                write_json(BANK2, {"provenance": provenance(), "candidates": rows})


def bank_rows(dataset):
    rows = [r for r in json.loads(BANK2.read_text())["candidates"] if r["dataset"] == dataset]
    order = list(SPEC)
    return sorted(rows, key=lambda r: order.index(r["version"]))


# ------------------------------------------------------------------ proxy (3 seeds, every reference subset)

def proxy(a):
    from scripts.proxy_detector import BATCH, N_SEQ, STEPS, WIDTH, proxy as run_proxy
    from src.data import load
    from src.versions import DATASETS as DS
    dev = torch.device("cuda")
    out = OUT / f"{a.dataset}_proxy.json"
    report = json.loads(out.read_text()) if out.exists() else {
        "config": {"width": WIDTH, "n_seq": N_SEQ, "steps": STEPS, "batch": BATCH,
                   "reference": "AP@2 on each reference subset of the real training sequences; primary 10/0"},
        "runs": {}}
    real = load(DS[a.dataset].real, "train")
    subs = [[real[int(i)] for i in ref_seqs(b, s)] for b, s in REFERENCES]
    report["references"] = {f"{b}_{s}": ref_seqs(b, s).tolist() for b, s in REFERENCES}
    for row in bank_rows(a.dataset):
        cid = row["candidate"]
        todo = [s for s in (1, 2, 3) if str(s) not in report["runs"].get(cid, {})]
        if not todo:
            continue
        pool = v1b.train_only(row["cache"], "train", 200)
        for seed in todo:
            res = run_proxy(pool[:N_SEQ], subs, dev, seed)
            res["ap2_by_reference"] = {f"{b}_{s}": ap for (b, s), ap in zip(REFERENCES, res.pop("ap2_subsets"))}
            res["proxy"] = res["ap2_by_reference"][f"{PRIMARY[0]}_{PRIMARY[1]}"]
            res["gpu"] = torch.cuda.get_device_name(0)
            report["runs"].setdefault(cid, {})[str(seed)] = res
            report["provenance"] = provenance()
            write_json(out, report)
            print(f"{cid:28s} seed {seed}: proxy {res['proxy']:.3f} {res['seconds']:.0f}s", flush=True)


def proxy_scores(dataset, ref):
    p = OUT / f"{dataset}_proxy.json"
    runs = json.loads(p.read_text())["runs"] if p.exists() else {}
    key = f"{ref[0]}_{ref[1]}"
    return {c: float(np.mean([v["ap2_by_reference"][key] for v in r.values()])) for c, r in runs.items()
            if len(r) == 3}, {c: [v["ap2_by_reference"][key] for v in r.values()] for c, r in runs.items()}


# ------------------------------------------------------------------ score and select

def select_order(method, risk, details, names):
    """Bank order per method: lower risk first; ties by the method's own rule, then the name."""
    k = np.arange(len(names))
    if method == "M1":
        key = lambda i: (risk["M1"][i], risk["M2"][i], names[i])
    elif method == "M3":
        key = lambda i: details["M3"][i]["key"] + (names[i],)
    elif method == "M4":
        key = lambda i: details["M4"][i]["key"] + (names[i],)
    else:
        key = lambda i: (risk[method][i] if np.isfinite(risk[method][i]) else np.inf, names[i])
    return [names[i] for i in sorted(k, key=key)]


def score_bank(F_dev, dataset, ref, calibrate_on, n_boot=0):
    """Rankings of the bank at one reference; calibration on development rows of `calibrate_on` radars."""
    bank = bank_rows(dataset)
    Fb = Features(bank, refs=sorted({ref} | {(20 if ref[0] == 50 else ref[0], s) for s in range(3)}), proxy=False)
    if Fb.missing:
        raise SystemExit(f"bank-v2 features missing: {Fb.missing[:3]}")
    # the bank rows are evaluated with the development rows' calibration
    dev = np.isin(F_dev.ds, calibrate_on) & np.isfinite(F_dev.val)
    cal = Calibration(F_dev, dev, ref)
    joint = _Joined(F_dev, Fb)
    cal.F = joint
    idx = np.arange(len(F_dev.rows), len(F_dev.rows) + len(bank))
    risk, det = cal.risks(idx, details=True)
    cal.F = F_dev
    prox, prox_seeds = proxy_scores(dataset, ref)
    names = [r["candidate"] for r in bank]
    risk["proxy"] = np.asarray([-prox.get(c, np.nan) for c in names])
    out = {"reference": ref_meta(Fb, ref), "calibrated_on": list(calibrate_on), "n_dev": cal.n_dev,
           "orders": {m: select_order(m, risk, det, names) for m in METHODS},
           "candidates": {}}
    out["selected"] = {m: out["orders"][m][0] for m in METHODS}
    for k, c in enumerate(names):
        m3, m4 = det["M3"][k], det["M4"][k]
        out["candidates"][c] = {
            "global": risk["global"][k], "M1": risk["M1"][k], "M2": risk["M2"][k],
            "M3": {"max": m3["max"], "mean": m3["mean"], "dominant": m3["dominant"],
                   "components": m3["components"], "unmeasured_components": m3["unmeasured_components"],
                   "probes": {n: {"value": _j(Fb.X1[ref][k, j]), "penalty": _j(det["P3"][k, j])}
                              for j, n in enumerate(PROBE_NAMES)}},
            "M4": {"M4": m4["M4"], "blocks": m4["blocks"], "top_blocks": m4["top_blocks"],
                   "diagnosis": m4["diagnosis"],
                   "probes": {n: {"value": _j(Fb.X4[ref][k, j]), "penalty": _j(det["P4"][k, j])}
                              for j, n in enumerate(F_dev.m4_names)}},
            "M5": dict(zip(("fn_risk", "fp_risk", "stratum_risk", "M5", "M5_no_lowconf"),
                           map(_j, Fb.M5[ref][k]))),
            "M6": risk["M6"][k], "M6_inputs": dict(zip(list(BLOCK_NAMES) + ["fn_risk", "fp_risk"], map(_j, det["R"][k]))),
            "proxy": prox.get(c), "proxy_seeds": prox_seeds.get(c)}
    out["calibration"] = cal.describe()
    if n_boot:
        out["bootstrap"] = bootstrap(F_dev, cal, dataset, bank, ref, n_boot)
    return out


class _Joined:
    """Development rows followed by bank rows, with the development rows' noise / undamaged / m4 names."""

    def __init__(self, F, B):
        self.rows = F.rows + B.rows
        cat = lambda a, b: np.concatenate([a, b])
        self.ds, self.cid = cat(F.ds, B.ds), cat(F.cid, B.cid)
        self.fam, self.base, self.val = cat(F.fam, B.fam), cat(F.base, B.base), cat(F.val, B.val)
        self.m4_names = F.m4_names
        assert B.m4_names == F.m4_names
        self.X1 = {r: cat(F.X1[r], B.X1[r]) for r in B.X1 if r in F.X1}
        self.X4 = {r: cat(F.X4[r], B.X4[r]) for r in B.X4 if r in F.X4}
        self.M5 = {r: cat(F.M5[r], B.M5[r]) for r in B.M5 if r in F.M5}
        self.glob = {r: cat(F.glob[r], B.glob[r]) for r in B.glob if r in F.glob}
        self.proxy = cat(F.proxy, np.full(len(B.rows), np.nan))
        self.noise, self.undamaged = F.noise, cat(F.undamaged, np.zeros(len(B.rows), bool))


def ref_meta(Fb, ref):
    m = next((x for x in Fb.meta[ref] if x), None)
    return m


_BOOT = {}


def _boot_draw(draw):
    """One bootstrap draw of the reference sequences: every selector's value per bank candidate."""
    from src.utility_metric import summarise
    from src.utility_metric_v2 import m4_probes, m5
    from src.utility_score import probes
    from src.utility_score_v2 import lex_rank, m3, m4
    S = _BOOT
    r1, r2 = v1b.resample(S["real1"], draw), v1b.resample(S["real2"], draw)
    med = float(np.median(r2["objects"]["range_m"]))
    X1, X4, M5, G = [], [], [], []
    for c1, c2 in zip(S["t1"], S["t2"]):
        vec = v1b.to_json(summarise(c1, r1))
        x = probes(vec)
        X1.append([x[n] if x[n] is not None else np.nan for n in PROBE_NAMES])
        P, _ = m4_probes(c2, r2, vec, med)
        X4.append([np.nan if n in S["unmeasured"] else P.get(n, np.nan) for n in S["m4_names"]])
        mm = m5(c2, r2, med)
        M5.append([mm.get(k, np.nan) for k in ("fn_risk", "fp_risk", "stratum_risk", "M5", "M5_no_lowconf")])
        G.append(vec["global"]["sum"])
    cal = S["cal"]
    X1, X4, M5 = np.asarray(X1, float), np.asarray(X4, float), np.asarray(M5, float)
    P1, P3, P4 = cal.tol1.penalties(X1), cal.tol3.penalties(X1), cal.tol4.penalties(X4)
    r3 = [m3(p, PROBE_NAMES) for p in P3]
    r4 = [m4(p, S["m4_names"]) for p in P4]
    R = np.asarray([[r["blocks"][b]["risk"] for b in BLOCK_NAMES] + list(M5[k, :2]) for k, r in enumerate(r4)])
    v = {"global": np.asarray(G, float), "M1": P1.sum(1), "M2": P1 @ cal.w2,
         "M3": np.asarray([r["max"] for r in r3]), "M4": np.asarray([r["M4"] for r in r4]),
         "M5": M5[:, 3], "M6": cal.m6.score(R)}
    risk = dict(v, M3=lex_rank([r["key"] for r in r3]), M4=lex_rank([r["key"] for r in r4]))
    det = {"M3": r3, "M4": r4}
    return v, {m: select_order(m, risk, det, S["names"])[0] for m in v}


def bootstrap(F_dev, cal, dataset, bank, ref, n_boot, seed=0, workers=8):
    """Sequence bootstrap of the reference (resampled with replacement; a repeated sequence keeps its index),
    calibration fixed: interval of every selector's value per candidate and how often each is selected."""
    import multiprocessing as mp
    from scripts.build_metric_v2_features import Candidate, real_v1, real_v2
    from scripts.evaluate_metric_v2 import UNMEASURED
    seqs = ref_seqs(*ref)
    cands = [Candidate(dataset, r) for r in bank]
    names = [r["candidate"] for r in bank]
    _BOOT.clear()
    _BOOT.update(real1=real_v1(dataset), real2=real_v2(dataset), t2=[c.v2() for c in cands],
                 t1=[c.v1_tables() for c in cands], cal=cal, m4_names=F_dev.m4_names, names=names,
                 unmeasured=set(UNMEASURED.get(dataset, ())))
    g = np.random.default_rng(seed)
    draws = [g.choice(seqs, size=len(seqs), replace=True) for _ in range(n_boot)]
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_boot_draw, draws)
    out = {"n_boot": n_boot, "reference_sequences": seqs.tolist(), "intervals": {}, "selection_frequency": {}}
    for m in res[0][0]:
        V = np.asarray([r[0][m] for r in res], float)
        out["intervals"][m] = {c: [_j(np.nanpercentile(V[:, k], 2.5)), _j(np.nanpercentile(V[:, k], 97.5))]
                               for k, c in enumerate(names)}
        picks = [r[1][m] for r in res]
        out["selection_frequency"][m] = {c: picks.count(c) / n_boot for c in names}
    return out


def score(a):
    t0 = time.time()
    F = Features(dev_rows())
    if F.missing:
        raise SystemExit(f"development features missing: {F.missing[:3]}")
    ds = a.dataset
    result = {"provenance": provenance(), "dataset": ds, "metric_version": "v2",
              "feature_keys": {d: feature_key(d) for d in ALL},
              "finalist": json.loads(Path("results/metric_v2/finalist.json").read_text())["finalist"],
              "primary_reference": list(PRIMARY), "outcome_used_for_calibration": "development whole-val AP@2",
              "bank_spec": {r["candidate"]: r["spec"] for r in bank_rows(ds)},
              "caches": {r["candidate"]: r["cache"] for r in bank_rows(ds)},
              "hardware": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
              "references": {}, "loro_calibration": {}}
    for ref in REFERENCES:
        key = f"{ref[0]}_{ref[1]}"
        n_boot = a.boot if ref == PRIMARY else a.boot_other
        result["references"][key] = score_bank(F, ds, ref, ALL, n_boot=n_boot)
        result["loro_calibration"][key] = {
            k: v for k, v in score_bank(F, ds, ref, tuple(d for d in ALL if d != ds)).items()
            if k in ("orders", "selected", "n_dev", "calibrated_on")}
        print(ds, key, result["references"][key]["selected"], f"{time.time() - t0:.0f}s", flush=True)
    prim = result["references"][f"{PRIMARY[0]}_{PRIMARY[1]}"]
    result["selection"] = prim["selected"]
    result["selection"]["finalist"] = prim["selected"][result["finalist"]]
    result["seconds"] = time.time() - t0
    write_json(OUT / f"{ds}_selection.json", result)
    print(json.dumps(result["selection"], indent=1))


def freeze_check(a):
    bad = []
    for ds in DATASETS:
        for p in (OUT / f"{ds}_runs.json", OUT / f"{ds}_augment_real5.json", OUT / f"{ds}_augment_real10.json"):
            if p.exists():
                bad.append(str(p))
    tracked = __import__("subprocess").run(["git", "ls-files", str(OUT)], capture_output=True, text=True).stdout.split()
    bad += [t for t in tracked if "_runs" in t or "augment" in t]
    print("no bank-v2 full-detector result exists" if not bad else f"FOUND: {bad}")
    if bad:
        raise SystemExit(1)


# ------------------------------------------------------------------ full training (after the freeze)

def _train_runs(dataset, out, draw_for, seeds=(1, 2, 3)):
    from src.data import load
    from src.detector import FrameDetector, scores
    from src.train import N_VAL, detections, train
    from src.versions import DATASETS as DS
    ds = DS[dataset]
    dev = torch.device("cuda")
    report = json.loads(out.read_text()) if out.exists() else {"runs": {}, "val": {}, "stops": {}}
    full_val = load(ds.real, "val")
    val = [full_val[int(i)] for i in np.linspace(0, len(full_val) - 1, N_VAL).round()]
    test = load(ds.real, "test")
    for row in bank_rows(dataset):
        cid = row["candidate"]
        todo = [s for s in seeds if str(s) not in report["runs"].get(cid, {})]
        if not todo:
            continue
        pool = v1b._LOAD(row["cache"], "train", 200)
        for seed in todo:
            torch.manual_seed(seed)
            model = FrameDetector(width=32).to(dev)
            t0 = time.time()
            stop = train(model, draw_for(pool, dev), val, dev, torch.Generator().manual_seed(seed))
            d, t = detections(model, test, dev)
            report["runs"].setdefault(cid, {})[str(seed)] = scores(d, t, test[0]["x"].shape[-1])
            vd, vt = detections(model, full_val, dev)
            report["val"].setdefault(cid, {})[str(seed)] = scores(vd, vt, full_val[0]["x"].shape[-1])
            report["stops"].setdefault(cid, {})[str(seed)] = stop | {"seconds": time.time() - t0}
            report["provenance"] = provenance()
            write_json(out, report)
            print(cid, seed, report["runs"][cid][str(seed)], flush=True)
    return report


def train_bank(a):
    from src.train import frame_batch
    _train_runs(a.dataset, OUT / f"{a.dataset}_runs.json",
                lambda pool, dev: (lambda g, p=pool: frame_batch(p, 8, dev, g)))


def augment(a):
    from src.data import load
    from src.train import frame_batch
    from src.versions import DATASETS as DS
    real_all = load(DS[a.dataset].real, "train")
    for b in map(int, a.budgets.split(",")):
        idx = np.linspace(0, len(real_all) - 1, b).round().astype(int)      # the budget study's sequences
        real = [real_all[int(i)] for i in idx]

        def draw_for(pool, dev, real=real):
            def draw(g, p=pool):
                x1, h1 = frame_batch(real, 4, dev, g)
                x2, h2 = frame_batch(p, 4, dev, g)
                return torch.cat([x1, x2]), torch.cat([h1, h2])
            return draw
        out = OUT / f"{a.dataset}_augment_real{b}.json"
        rep = _train_runs(a.dataset, out, draw_for)
        rep["real_sequences"] = idx.tolist()
        write_json(out, rep)


# ------------------------------------------------------------------ report (after training)

def _mean_runs(path, key="runs", metric="ap2"):
    r = json.loads(Path(path).read_text())
    return ({c: float(np.mean([s[metric] for s in v.values()])) for c, v in r[key].items() if len(v) == 3},
            {c: float(np.ptp([s[metric] for s in v.values()])) for c, v in r[key].items() if len(v) == 3})


def method_eval(order, ap, noise, names):
    """A method's bank order against outcomes ap (dict): selection, regret, within noise, pairwise, tau, rho, top 3."""
    risk = np.asarray([order.index(c) for c in names], dtype=np.float64)
    a = np.asarray([ap[c] for c in names])
    ev = evaluate_set(risk, a, noise)
    sel = order[0]
    ev |= {"selected": sel, "selected_ap": ap[sel], "regret": float(a.max() - ap[sel]),
           "within_noise": bool(ap[sel] >= a.max() - noise)}
    return ev


M3_EXPECTED = {"scnr_narrow": {"label"}, "scnr_wide": {"label"}, "shape_range": {"target"},
               "shape_doppler": {"target"}, "label_stochastic": {"label"}, "support_stratified": {"coverage"},
               "combined_moderate": {"label", "target", "coverage"}}


def mechanism_match(finalist, cid, cand):
    """The modified mechanism among the finalist's two largest risks (note, deviation 5); None for the control."""
    name = cid.split(":")[1]
    if EXPECTED_BLOCKS[name] is None:
        return None
    if finalist in ("M4", "M6"):
        return bool(set(cand["M4"]["top_blocks"]) & EXPECTED_BLOCKS[name])
    if finalist == "M3":
        comps = {k: v["risk"] for k, v in cand["M3"]["components"].items() if v["risk"] is not None}
        top = sorted(comps, key=lambda k: -comps[k])[:2]
        return bool(set(top) & M3_EXPECTED[name])
    sides = {k: cand["M5"][k] or 0.0 for k in ("fn_risk", "fp_risk", "stratum_risk")}
    top = set(sorted(sides, key=lambda k: -sides[k])[:2])
    return bool(top & ({"stratum_risk"} if name == "support_stratified" else {"fn_risk", "fp_risk"}))


MECH = {"A": "detectability", "B": "label-task", "C": "morphology", "D": "support", "E": "hard negatives",
        "F": "processing"}


def report(a):
    ds = a.dataset
    sel = json.loads((OUT / f"{ds}_selection.json").read_text())
    names = [r["candidate"] for r in bank_rows(ds)]
    test, rng = _mean_runs(OUT / f"{ds}_runs.json")
    ap4, _ = _mean_runs(OUT / f"{ds}_runs.json", metric="ap4")
    rec2, _ = _mean_runs(OUT / f"{ds}_runs.json", metric="recall2")
    val, _ = _mean_runs(OUT / f"{ds}_runs.json", key="val")
    missing = [c for c in names if c not in test]
    if missing:
        raise SystemExit(f"bank-v2 runs missing: {missing}")
    noise = float(np.median([rng[c] for c in names]))
    oracle = max(names, key=lambda c: test[c])
    a_ = np.asarray([test[c] for c in names])
    prim = sel["references"][f"{PRIMARY[0]}_{PRIMARY[1]}"]
    out = {"provenance": provenance(), "dataset": ds, "noise": noise, "test_ap2": test, "test_ap4": ap4,
           "test_recall2": rec2, "val_ap2": val, "seed_range": rng, "test_oracle": oracle,
           "val_oracle": max(names, key=lambda c: val[c]), "finalist": sel["finalist"],
           "random": {"regret": float(a_.max() - a_.mean()), "within_noise": float((a_ >= a_.max() - noise).mean())},
           "methods": {}}
    for m in METHODS:
        out["methods"][m] = method_eval(prim["orders"][m], test, noise, names)
    vo = out["val_oracle"]
    out["methods"]["val_oracle"] = {"selected": vo, "selected_ap": test[vo], "regret": test[oracle] - test[vo],
                                    "within_noise": bool(test[vo] >= test[oracle] - noise)}
    # diagnosis: the modified mechanism among the two largest M4 risk blocks
    diag = {}
    for c in names:
        top = prim["candidates"][c]["M4"]["top_blocks"]
        exp = EXPECTED_BLOCKS[c.split(":")[1]]
        diag[c] = {"top_blocks": [MECH[b] for b in top], "diagnosis": prim["candidates"][c]["M4"]["diagnosis"],
                   "expected": None if exp is None else sorted(MECH[b] for b in exp),
                   "match": None if exp is None else bool(set(top) & exp),
                   "M3_dominant": prim["candidates"][c]["M3"]["dominant"]}
    out["diagnosis"] = diag
    fin = sel["finalist"]
    f = out["methods"][fin]
    fsel = f["selected"]
    mech_ok = mechanism_match(fin, fsel, prim["candidates"][fsel])
    for c in names:
        diag[c]["finalist_match"] = mechanism_match(fin, c, prim["candidates"][c])
    out["success"] = {"within_noise": f["within_noise"], "below_random": f["regret"] < out["random"]["regret"],
                      "below_global": f["regret"] < out["methods"]["global"]["regret"],
                      "mechanism_in_top2": mech_ok}
    out["success"]["success"] = all(out["success"].values())
    out["proxy_comparison"] = {"finalist_regret": f["regret"], "proxy_regret": out["methods"]["proxy"]["regret"],
                               "difference": f["regret"] - out["methods"]["proxy"]["regret"]}
    # sample budgets: selection per reference, regret, agreement with 50, stability across the 3 subsets
    bud = {}
    for m in METHODS:
        per = {}
        for key, r in sel["references"].items():
            c = r["selected"][m]
            per[key] = {"selected": c, "regret": test[oracle] - test[c],
                        "within_noise": bool(test[c] >= test[oracle] - noise),
                        "agrees_with_50": c == sel["references"]["50_0"]["selected"][m],
                        "n_labels": (r["reference"] or {}).get("n_labels"), "n_fn": (r["reference"] or {}).get("n_fn"),
                        "n_fp": (r["reference"] or {}).get("n_fp")}
            if "bootstrap" in r:
                iv = r["bootstrap"]["intervals"].get(m, {}).get(c)
                per[key]["ci95_selected"] = iv
                per[key]["ci95_width"] = None if not iv or None in iv else iv[1] - iv[0]
                per[key]["bootstrap_selection_frequency"] = r["bootstrap"]["selection_frequency"].get(m, {}).get(c)
        stable = [b for b in (5, 10, 20) if len({per[f"{b}_{s}"]["selected"] for s in range(3)}) == 1]
        bud[m] = {"per_reference": per, "budgets_stable_across_subsets": stable,
                  "smallest_stable_budget": min(stable) if stable else None,
                  "mean_regret_by_budget": {b: float(np.mean([per[f"{b}_{s}"]["regret"] for s in range(3)]))
                                            for b in (5, 10, 20)} | {50: per["50_0"]["regret"]}}
    out["budgets"] = bud
    out["loro_calibration"] = {m: {"selected": sel["loro_calibration"][f"{PRIMARY[0]}_{PRIMARY[1]}"]["selected"][m],
                                   "regret": test[oracle] - test[sel["loro_calibration"][f"{PRIMARY[0]}_{PRIMARY[1]}"]["selected"][m]]}
                               for m in METHODS}
    # augmentation (RADDet): Delta_AP_b = AP(candidate + b real) - AP(b real only)
    aug = {}
    for b in (5, 10):
        p = OUT / f"{ds}_augment_real{b}.json"
        if not p.exists():
            continue
        mix, mrng = _mean_runs(p)
        base = json.loads(Path(f"results/budget_{ds}_real{b}_runs.json").read_text())["runs"]
        real_only = float(np.mean([v["ap2"] for v in base["real"]["clean"].values()]))
        old_mix = float(np.mean([v["ap2"] for v in base["mix"]["clean"].values()]))
        delta = {c: mix[c] - real_only for c in names if c in mix}
        if len(delta) < len(names):
            continue
        n_aug = float(np.median([mrng[c] for c in names]))
        aug[b] = {"real_only_ap2": real_only, "mix_ap2": mix, "delta": delta, "noise": n_aug,
                  "budget_study_tuned_mix_ap2": old_mix,
                  "spread": float(max(delta.values()) - min(delta.values())),
                  "methods": {m: method_eval(prim["orders"][m], delta, n_aug, names) for m in METHODS}}
        da = np.asarray([delta[c] for c in names])
        aug[b]["random"] = {"regret": float(da.max() - da.mean())}
    out["augmentation"] = aug
    write_json(OUT / f"{ds}_report.json", out)
    print(json.dumps({m: {k: out["methods"][m][k] for k in ("selected", "regret", "within_noise")} for m in out["methods"]},
                     indent=1), "random", round(out["random"]["regret"], 3), "noise", round(noise, 3),
          "success", out["success"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("render", "proxy", "score", "freeze-check", "train", "augment", "report"))
    ap.add_argument("--dataset", default="carrada")
    ap.add_argument("--names", default="")
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--boot-other", type=int, default=20)
    ap.add_argument("--budgets", default="5,10")
    a = ap.parse_args()
    {"render": render, "proxy": proxy, "score": score, "freeze-check": freeze_check, "train": train_bank,
     "augment": augment, "report": report}[a.step](a)


if __name__ == "__main__":
    main()
