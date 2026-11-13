"""Run from an installed project: python scripts/evaluate.py --help."""
from radsimreal.cli import main

if __name__ == "__main__":
    main(["evaluate", *__import__("sys").argv[1:]])
