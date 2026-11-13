"""Run from an installed project: python scripts/train.py --help."""
from radsimreal.cli import main

if __name__ == "__main__":
    main(["train", *__import__("sys").argv[1:]])
