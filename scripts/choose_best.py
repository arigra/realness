"""Choose a dataset's best tuned simulator: the sim2 version with the highest mean
AP@2 on the whole real val split (docs/notes/2026-11-02-simulators.md). Writes its
switches to results/tuning/best.json.

    PYTHONPATH=. python3 scripts/choose_best.py --dataset carrada
"""
import argparse
import json

import numpy as np

from src.versions import BEST, SIM2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    runs = json.load(open(f"results/sim2_{a.dataset}_runs.json"))["val"]["only"]
    val = {n: float(np.mean([r["ap2"] for r in s.values()])) for n, s in runs.items() if len(s) == 3}
    for n, v in sorted(val.items(), key=lambda kv: -kv[1]):
        print(f"{n:22s} whole-val AP@2 {v:.3f}")
    name = max(val, key=val.get)
    kind, switches = SIM2[name]
    chosen = json.loads(BEST.read_text()) if BEST.exists() else {}
    chosen[a.dataset] = [] if kind == "clean" else list(switches)
    BEST.write_text(json.dumps(chosen, indent=1))
    print("best:", name, chosen[a.dataset])


if __name__ == "__main__":
    main()
