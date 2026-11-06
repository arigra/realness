"""Classification study (pre-registered: docs/notes/2026-11-05-classification.md): classify
target crops (src/crops.py) with a CNN trained on each synthetic version (or the 50 real
training sequences), 3 seeds, early-stopped on crops of 32 real val sequences, scored on the
whole real val split (what choices are made on) and the real test split (reported).
Versions are the spec simulator, the best tuned one, and target-signature defects on it.
Writes results/classify_<dataset>_runs.json; crops are cached in data/crops/<dataset>/.

    PYTHONPATH=. python3 scripts/classify_study.py --dataset carrada
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.classify import predict, scores, shortcut_baseline, train
from src.crops import from_frames, real_frames, sim_frames
from src.damage import TrackDamage
from src.train import N_VAL
from src.tuned import simulator, table
from src.versions import DATASETS, N_SIM, SEED0, best

RES = {"carrada": (0.2, 0.42), "raddet": (0.1953125, 0.4197)}     # range (m), velocity (m/s) per bin
VERSIONS = {"spec": None, "tuned": {}}
VERSIONS |= {f"micro{m}": {"micro": m} for m in (0.5, 0.0)}
VERSIONS |= {f"blur{k}": {"blur": k} for k in (1.5, 2.0, 3.0)}
VERSIONS |= {f"tail{c}": {"tail": float(c)} for c in (2, 4, 8)}
VERSIONS |= {f"contrast-{d}": {"target_db": -float(d)} for d in (6, 12, 20)}


def sim_crops(dataset, name, damage):
    out = Path("data/crops") / dataset / f"{name}.pt"
    if out.exists():
        return torch.load(out)
    tb = table(dataset)

    def seqs():
        for i in range(N_SIM):
            g = torch.Generator().manual_seed(SEED0 + i)
            if damage is None:                                   # the spec simulator: no switch on
                yield simulator(dataset, (), g, None, tb).gen_sequence()
            else:
                yield simulator(dataset, best(dataset), g, None, tb, mixin=TrackDamage,
                                damage_generator=torch.Generator().manual_seed(i), **damage).gen_sequence()
    c = from_frames(sim_frames(seqs()), *RES[dataset])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(c, out)
    return c


def real_crops(dataset, split):
    out = Path("data/crops") / dataset / f"real_{split}.pt"
    if out.exists():
        return torch.load(out)
    cache = DATASETS[dataset].real
    n_seq = len(torch.load(f"{cache}/{split}_labels.pt")["meta"])
    parts, seq_of = [], []
    for i, frame in enumerate(real_frames(dataset, cache, split)):
        c = from_frames([frame], *RES[dataset])
        parts.append(c)
        seq_of += [i // 8] * len(c[1])
    c = tuple(torch.cat([p[k] for p in parts]) for k in range(3)) + (torch.tensor(seq_of),)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(c, out)
    print(f"real {split}: {len(c[1])} crops from {n_seq} sequences", flush=True)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=tuple(RES))
    ap.add_argument("--seeds", default="1,2,3")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(f"results/classify_{a.dataset}_runs.json")
    report = json.loads(path.read_text()) if path.exists() else {"runs": {}, "shortcut": {}, "n_crops": {}}

    tr, va, te = (real_crops(a.dataset, s) for s in ("train", "val", "test"))
    pick = set(np.linspace(0, int(va[3].max()), N_VAL).round().astype(int).tolist())
    stop = torch.tensor([int(s) in pick for s in va[3]])
    val_stop = (va[0][stop], va[1][stop])
    report["n_crops"] |= {"real_train": len(tr[1]), "real_val": len(va[1]), "real_test": len(te[1]),
                          "class_counts_test": torch.bincount(te[1], minlength=3).tolist()}

    report["shortcut"]["real_train"] = shortcut_baseline(tr[2], tr[1], te[2], te[1])
    sources = {"real": tr[:3]} | {n: None for n in VERSIONS}
    for name in sources:
        data = sources[name] if name == "real" else sim_crops(a.dataset, name, VERSIONS[name])
        report["n_crops"][name] = len(data[1])
        if name in ("spec", "tuned"):
            report["shortcut"][name] = shortcut_baseline(data[2], data[1], te[2], te[1])
        for seed in [int(s) for s in a.seeds.split(",")]:
            if str(seed) in report["runs"].get(name, {}):
                continue
            model, stopped = train(data[0], data[1], val_stop, dev, seed)
            res = {"test": scores(predict(model, te[0], dev), te[1]),
                   "val": scores(predict(model, va[0], dev), va[1]), "stop": stopped}
            report["runs"].setdefault(name, {})[str(seed)] = res
            print(f"{name} seed {seed}: test bal_acc {res['test']['bal_acc']:.3f} "
                  f"val {res['val']['bal_acc']:.3f}", flush=True)
            path.write_text(json.dumps(report, indent=1))
    path.write_text(json.dumps(report, indent=1))
    print("shortcut baselines (test bal_acc):", {k: round(v["bal_acc"], 3) for k, v in report["shortcut"].items()})


if __name__ == "__main__":
    main()
