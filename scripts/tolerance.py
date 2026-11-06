"""Print and save the tolerance readout (src/tolerance.py) of Study 2 and 2b.

    PYTHONPATH=. python3 scripts/tolerance.py --datasets carrada,raddet --split test
"""
import argparse
import json
from pathlib import Path

from src.tolerance import FAMILIES, readout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="carrada,raddet,radial")
    ap.add_argument("--split", default="test", choices=("test", "val"))
    ap.add_argument("--metric", default="ap2")
    a = ap.parse_args()
    res = {d: readout(d, a.split, a.metric) for d in a.datasets.split(",")}
    Path(f"results/tolerance_{a.split}_{a.metric}.json").write_text(json.dumps(res, indent=1))
    ds = list(res)
    print(f"{a.metric} on {a.split}: " + "  ".join(f"{d} clean {res[d]['clean']:.3f} noise {res[d]['noise']:.3f}" for d in ds))
    print(f"{'family':14s}" + "".join(f"{d:>16s}" for d in ds))
    for f in FAMILIES:
        print(f"{f:14s}" + "".join(f"{str(res[d]['families'][f]['onset']):>16s}" for d in ds))
    for f in FAMILIES:
        print(f"  {f}: " + " | ".join(d + " " + " ".join("-" if x is None else f"{x:.2f}" for x in res[d]["families"][f]["means"].values()) for d in ds))


if __name__ == "__main__":
    main()
