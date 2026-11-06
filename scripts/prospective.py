"""Prospective candidate selection (docs/notes/2026-11-06-metric-experiments.md, Phase 6).

  render   write the bank caches (src/natural.BANK) and results/metric/bank.json
  select   score the bank and the natural candidates of a held-out radar with the tolerance metric
           frozen on the two other radars (leave-one-radar-out calibration, reference = budget 5
           subset 0), write the ranking and the selection to results/metric/prospective/<dataset>_selection.json
           (commit it before training)
  train    synthetic-only U-Net, Study 2's training, 3 seeds, for every bank candidate
           -> results/metric/prospective/<dataset>_runs.json
  report   the selection against random, global score, proxy, val oracle and test oracle

    PYTHONPATH=. python3 scripts/prospective.py render
    PYTHONPATH=. python3 scripts/build_metric_features.py --dataset carrada --candidates bank
    PYTHONPATH=. python3 scripts/prospective.py select --dataset carrada
    PYTHONPATH=. python3 scripts/prospective.py train --dataset carrada
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.build_metric_features import BANK, candidate_dir, registry, subsets
from scripts.evaluate_utility_metric import PRIMARY, Data, baselines, load_rows, vector
from src.natural import BANK as SPEC, cache_dir, render_bank
from src.provenance import provenance
from src.utility_score import PROBE_NAMES, component_penalties, probes

OUT = Path("results/metric/prospective")
DATASETS = ("carrada", "raddet")


def render(_):
    rows = []
    ref = subsets(PRIMARY[0], 3)[PRIMARY[1]]          # the metric's reference: templates never come from it
    for ds in DATASETS:
        for name in SPEC:
            out = render_bank(ds, name, ref)
            print(ds, name, out, flush=True)
            rows.append({"dataset": ds, "candidate": f"bank:{name}", "suite": "bank", "version": name,
                         "family": "bank", "level": None, "base": "tuned", "setting": "only",
                         "n_sim_sequences": 200, "cache": str(out), "usable": False, "entries": [],
                         "spec": SPEC[name], "template_excluded_sequences": ref.tolist()})
    BANK.write_text(json.dumps({"provenance": provenance(), "candidates": rows}, indent=1))


def select(a):
    held = a.dataset
    rows = load_rows()
    cache = {d: baselines(d, [r for r in rows if r["dataset"] == d]) for d in ("carrada", "raddet", "radial")}
    D = Data(rows, cache, PRIMARY)
    tol, w, n_dev = D.calibrate(D.ds != held)
    bank = [r for r in registry(held).values() if r["suite"] == "bank"]
    natural = [r for r in rows if r["dataset"] == held and r["family"] in ("natural", "fixes")]
    ranking = []
    for r in bank + natural:
        v = vector(r, PRIMARY)
        if v is None:
            raise SystemExit(f"no current vector for {r['candidate']}: build its features first")
        x = np.asarray([[probes(v["vector"])[n] for n in PROBE_NAMES]])
        p = tol.penalties(x)[0]
        ranking.append({"candidate": r["candidate"], "bank": r["suite"] == "bank", "M1": float(p.sum()),
                        "M2": float(p @ w), "global": v["vector"]["global"]["sum"],
                        "components": component_penalties(p)})
    # selection among the bank (new candidates); tie rule: lower M2, then name
    order = sorted([x for x in ranking if x["bank"]], key=lambda x: (x["M1"], x["M2"], x["candidate"]))
    out = {"provenance": provenance(), "held_out": held, "reference": list(PRIMARY), "n_dev": n_dev,
           "calibrated_on": [d for d in ("carrada", "raddet", "radial") if d != held],
           "tolerance": tol.describe(), "weights": dict(zip(PROBE_NAMES, w.tolist())),
           "selected": order[0]["candidate"], "bank_order": [x["candidate"] for x in order],
           "global_selected": min([x for x in ranking if x["bank"]], key=lambda x: x["global"])["candidate"],
           "ranking": sorted(ranking, key=lambda x: x["M1"])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{held}_selection.json").write_text(json.dumps(out, indent=1, default=lambda x: None))
    print(held, "selected", out["selected"], "| bank order", out["bank_order"], "| global picks", out["global_selected"])


def train_bank(a):
    from src.data import load
    from src.detector import FrameDetector, scores
    from src.train import N_VAL, detections, frame_batch, train
    from src.versions import DATASETS as DS
    from scripts.build_metric_features import load_candidate
    ds = DS[a.dataset]
    dev = torch.device("cuda")
    out = OUT / f"{a.dataset}_runs.json"
    report = json.loads(out.read_text()) if out.exists() else {"runs": {}, "val": {}, "stops": {}}
    full_val = load(ds.real, "val")
    val = [full_val[int(i)] for i in np.linspace(0, len(full_val) - 1, N_VAL).round()]
    test = load(ds.real, "test")
    for r in [r for r in registry(a.dataset).values() if r["suite"] == "bank"]:
        cid = r["candidate"]
        todo = [s for s in (1, 2, 3) if str(s) not in report["runs"].get(cid, {})]
        if not todo:
            continue
        pool, _ = load_candidate(r)
        for seed in todo:
            torch.manual_seed(seed)
            model = FrameDetector(width=32).to(dev)
            stop = train(model, lambda g, p=pool: frame_batch(p, 8, dev, g), val, dev, torch.Generator().manual_seed(seed))
            d, t = detections(model, test, dev)
            report["runs"].setdefault(cid, {})[str(seed)] = scores(d, t, test[0]["x"].shape[-1])
            vd, vt = detections(model, full_val, dev)
            report["val"].setdefault(cid, {})[str(seed)] = scores(vd, vt, full_val[0]["x"].shape[-1])
            report["stops"].setdefault(cid, {})[str(seed)] = stop
            report["provenance"] = provenance()
            out.write_text(json.dumps(report, indent=1))
            print(cid, seed, report["runs"][cid][str(seed)], flush=True)


def report(a):
    held = a.dataset
    sel = json.loads((OUT / f"{held}_selection.json").read_text())
    runs = json.loads((OUT / f"{held}_runs.json").read_text())
    proxy_path = Path(f"results/metric/proxy/{held}_bank.json")
    proxy = json.loads(proxy_path.read_text())["runs"] if proxy_path.exists() else {}
    test = {c: float(np.mean([s["ap2"] for s in v.values()])) for c, v in runs["runs"].items()}
    rng = {c: float(np.ptp([s["ap2"] for s in v.values()])) for c, v in runs["runs"].items()}
    val = {c: float(np.mean([s["ap2"] for s in v.values()])) for c, v in runs["val"].items()}
    noise = float(np.median(list(rng.values())))
    oracle = max(test.values())
    picks = {"metric (M1)": sel["selected"], "global score": sel["global_selected"],
             "val oracle": max(val, key=val.get), "test oracle": max(test, key=test.get)}
    if proxy:
        picks["proxy"] = max(proxy, key=lambda c: proxy[c]["1"]["proxy"])
    out = {"held_out": held, "noise": noise, "test_ap": test, "seed_range": rng, "val_ap": val,
           "random_expected_regret": oracle - float(np.mean(list(test.values()))),
           "picks": {k: {"candidate": c, "test_ap": test[c], "regret": oracle - test[c],
                         "within_noise": test[c] >= oracle - noise} for k, c in picks.items()}}
    (OUT / f"{held}_report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["picks"], indent=1), "random regret", round(out["random_expected_regret"], 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("render", "select", "train", "report"))
    ap.add_argument("--dataset", default="carrada", choices=DATASETS)
    a = ap.parse_args()
    {"render": render, "select": select, "train": train_bank, "report": report}[a.step](a)


if __name__ == "__main__":
    main()
