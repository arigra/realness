"""Shortcut controls for the crop classifier (docs/notes/2026-11-05-classification.md, controls).

The first crops set the zero-Doppler columns to a constant: for a slow target the flat stripe's
position inside the crop is its speed. Here the same classifier is trained (on the 50 real
training sequences, and on the best tuned simulator) and scored on real test with each crop mode
(src/crops.MODE): ridge constant / noise / kept, label- or peak-centred, and with the target
erased (the crop taken 40 rows away, same Doppler columns). Erased crops must score at chance
(0.33): anything above is information that is not the target. 3 seeds, balanced accuracy.
Writes results/crop_controls.json.

    PYTHONPATH=. python3 scripts/crop_controls.py
"""
import json
from pathlib import Path

import numpy as np
import torch

from scripts.classify_study import RES
from src.classify import predict, scores, train
from src.crops import from_frames, real_frames, sim_frames
from src.train import N_VAL
from src.tuned import simulator, table
from src.versions import DATASETS, N_SIM, SEED0, best

MODES = {"label_constant": {"ridge": "constant"}, "label_noise": {"ridge": "noise"},
         "label_keep": {"ridge": "keep"}, "peak_noise": {"ridge": "noise", "align": "peak"},
         "erase_constant": {"ridge": "constant", "erase": True}, "erase_noise": {"ridge": "noise", "erase": True}}
OUT = Path("results/crop_controls.json")


def real(dataset, split, mode):
    """Crops of a real split, with the sequence index of each crop."""
    xs, ys, seq = [], [], []
    for i, frame in enumerate(real_frames(dataset, DATASETS[dataset].real, split)):
        x, y, _ = from_frames([frame], *RES[dataset], mode=mode)
        xs.append(x); ys.append(y); seq += [i // 8] * len(y)
    return torch.cat(xs), torch.cat(ys), torch.tensor(seq)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for ds in ("carrada", "raddet"):
        tb = table(ds)
        sims = [simulator(ds, best(ds), torch.Generator().manual_seed(SEED0 + i), None, tb).gen_sequence()
                for i in range(N_SIM)]
        for name, mode in MODES.items():
            done = res.get(ds, {}).get(name, {})
            if all(k in done for k in ("real", "tuned")):
                continue
            tr, va, te = (real(ds, s, mode) for s in ("train", "val", "test"))
            pick = set(np.linspace(0, int(va[2].max()), N_VAL).round().astype(int).tolist())
            stop = torch.tensor([int(s) in pick for s in va[2]])
            sim = from_frames(sim_frames(sims), *RES[ds], mode=mode)
            for src, (x, y) in (("real", tr[:2]), ("tuned", sim[:2])):
                accs = []
                for seed in (1, 2, 3):
                    model, _ = train(x, y, (va[0][stop], va[1][stop]), dev, seed)
                    accs.append(scores(predict(model, te[0], dev), te[1])["bal_acc"])
                res.setdefault(ds, {}).setdefault(name, {})[src] = accs
                print(f"{ds} {name:15s} trained on {src:5s}: test bal_acc {np.mean(accs):.3f} "
                      f"[{min(accs):.3f}, {max(accs):.3f}]", flush=True)
                OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
