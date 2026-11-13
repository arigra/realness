"""Run beside your datasets, including from a VS Code remote terminal.

Example: python scripts/hpc_run.py --dataset raddet --root /data/RADDet
  --trusted-labels --reference-cartesian --out /scratch/radsimreal
  --selection /scratch/verified_selection.json --models unet raddet probabilistic
"""
from radsimreal.cli import main

if __name__ == "__main__":
    main(["native", "run", *__import__("sys").argv[1:]])
