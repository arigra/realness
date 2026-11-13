"""Run from an installed project: python scripts/infer.py --help."""
from radsimreal.cli import main

if __name__ == "__main__":
    main(["infer", *__import__("sys").argv[1:]])
