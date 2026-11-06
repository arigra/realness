"""Classification redesign before any classification metric (docs/notes/2026-11-06-metric-experiments.md,
Phase 8).

Crop variants (src/crops.py modes): the current crops; smaller crops; crops masked to the target's
connected component with the background randomised; peak-centred; range- and speed-matched class
sampling (in each of 5 x 5 quantile bins of range and |speed|, every class keeps the same number of
crops). Controls: target erased (crop 40 rows away), ridge only. Each trained on the 50 real training
sequences (3 seeds), early-stopped on 32 real val sequences, scored on real test with a 95% interval
over test sequences; plus the speed/range logistic baseline on the same sampling.
Gate (per crop design): its erased and ridge-only controls consistent with chance (interval includes
1/3) and the real-trained classifier above the speed/range baseline. A design that passes is then
trained on the tuned simulator's crops (3 seeds).
Writes results/metric/classification/controls_v2.json.

    PYTHONPATH=. python3 scripts/classify_controls_v2.py
"""
import json
from pathlib import Path

import numpy as np
import torch

from scripts.classify_study import RES
from src.classify import predict, scores, shortcut_baseline, train
from src.crops import from_frames, real_frames, sim_frames
from src.provenance import provenance
from src.train import N_VAL
from src.tuned import simulator, table
from src.versions import DATASETS, N_SIM, SEED0, best

OUT = Path("results/metric/classification/controls_v2.json")
SMALL = {"kr": 6, "kd": 4}
MASK = SMALL | {"mask": "component", "bg": "noise"}
DESIGNS = {             # name -> (mode, matched)
    "base": ({}, False),
    "small": (SMALL, False),
    "small_mask": (MASK, False),
    "small_mask_peak": (MASK | {"align": "peak"}, False),
    "small_mask_matched": (MASK, True),
}
CONTROLS = {"erase": {"erase": True}, "ridge_only": {"ridge_only": True, "ridge": "keep"}}
CHANCE = 1 / 3


def crops(frames, dataset, mode):
    """x, y, shortcut features, sequence of each crop (frames: iterable of (seq, frame, objects))."""
    xs, ys, fs, seq = [], [], [], []
    for s, frame in frames:
        x, y, f = from_frames([frame], *RES[dataset], mode=mode)
        if not len(y):
            continue                    # its empty tensor has the default crop size
        xs.append(x); ys.append(y); fs.append(f); seq += [s] * len(y)
    return torch.cat(xs), torch.cat(ys), torch.cat(fs), torch.tensor(seq)


def real_split(dataset, split, mode):
    frames = ((i // 8, fr) for i, fr in enumerate(real_frames(dataset, DATASETS[dataset].real, split)))
    return crops(frames, dataset, mode)


def matched(data, seed=0, bins=5):
    """Subsample so that every class has the same count in each (range, |speed|) quantile bin."""
    x, y, f, s = data
    g = np.random.default_rng(seed)
    qs = [np.quantile(f[:, k].numpy(), np.linspace(0, 1, bins + 1)[1:-1]) for k in (0, 1)]
    cell = np.digitize(f[:, 0].numpy(), qs[0]) * bins + np.digitize(f[:, 1].numpy(), qs[1])
    keep = []
    for c in np.unique(cell):
        idx = [np.flatnonzero((cell == c) & (y.numpy() == k)) for k in range(3)]
        n = min(len(i) for i in idx)
        for i in idx:
            keep += g.choice(i, n, replace=False).tolist() if n else []
    keep = torch.tensor(sorted(keep), dtype=torch.long)
    return x[keep], y[keep], f[keep], s[keep]


def boot_ci(pred_by_seed, y, seq, n=1000, seed=0):
    """Mean balanced accuracy over seeds, 95% interval over test sequences."""
    g = np.random.default_rng(seed)
    seqs = np.unique(seq.numpy())
    rows = {q: np.flatnonzero(seq.numpy() == q) for q in seqs}
    est = float(np.mean([scores(p, y)["bal_acc"] for p in pred_by_seed]))
    boot = []
    for _ in range(n):
        take = np.concatenate([rows[q] for q in g.choice(seqs, len(seqs))])
        t = torch.from_numpy(take)
        boot.append(np.mean([scores(p[t], y[t])["bal_acc"] for p in pred_by_seed]))
    return est, [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def run(train_data, val_data, test_data, dev):
    pick = set(np.linspace(0, int(val_data[3].max()), N_VAL).round().astype(int).tolist())
    stop = torch.tensor([int(q) in pick for q in val_data[3]])
    preds = []
    for seed in (1, 2, 3):
        model, _ = train(train_data[0], train_data[1], (val_data[0][stop], val_data[1][stop]), dev, seed)
        preds.append(predict(model, test_data[0], dev))
    est, ci = boot_ci(preds, test_data[1], test_data[3])
    return {"bal_acc": est, "ci95": ci, "n_test": int(len(test_data[1])), "n_train": int(len(train_data[1]))}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res["provenance"] = provenance()
    for ds in ("carrada", "raddet"):
        out = res.setdefault(ds, {})
        sims = None
        for name, (mode, match) in DESIGNS.items():
            d = out.setdefault(name, {})
            variants = {"target": mode} | {c: mode | m for c, m in CONTROLS.items()}
            for v, md in variants.items():
                if v in d:
                    continue
                splits = [real_split(ds, s, md) for s in ("train", "val", "test")]
                if match:
                    splits = [matched(sp) for sp in splits]
                d[v] = run(splits[0], splits[1], splits[2], dev)
                if v == "target":
                    d["shortcut"] = shortcut_baseline(splits[0][2], splits[0][1], splits[2][2], splits[2][1])["bal_acc"]
                print(f"{ds} {name:20s} {v:10s} bal_acc {d[v]['bal_acc']:.3f} {np.round(d[v]['ci95'], 3).tolist()} "
                      f"n_test {d[v]['n_test']}", flush=True)
                OUT.parent.mkdir(parents=True, exist_ok=True)
                OUT.write_text(json.dumps(res, indent=1))
            d["gate"] = {"erase_at_chance": d["erase"]["ci95"][0] <= CHANCE <= d["erase"]["ci95"][1],
                         "ridge_at_chance": d["ridge_only"]["ci95"][0] <= CHANCE <= d["ridge_only"]["ci95"][1],
                         "beats_shortcut": d["target"]["ci95"][0] > d["shortcut"]}
            d["gate"]["pass"] = all(d["gate"].values())
            print(f"{ds} {name}: gate {d['gate']}", flush=True)
            if d["gate"]["pass"] and "tuned" not in d:
                if sims is None:
                    tb = table(ds)
                    sims = [simulator(ds, best(ds), torch.Generator().manual_seed(SEED0 + i), None, tb).gen_sequence()
                            for i in range(N_SIM)]
                frames = ((i // 8, fr) for i, fr in enumerate(sim_frames(sims)))
                sim = crops(frames, ds, mode)
                splits = [real_split(ds, s, mode) for s in ("val", "test")]
                if match:
                    sim, splits = matched(sim), [matched(sp) for sp in splits]
                d["tuned"] = run(sim, splits[0], splits[1], dev)
                print(f"{ds} {name:20s} tuned-trained bal_acc {d['tuned']['bal_acc']:.3f}", flush=True)
            OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
