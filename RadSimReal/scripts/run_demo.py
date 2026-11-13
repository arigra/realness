"""Run from the project root after pip install -e ."""
from radsimreal.cli import main

if __name__ == "__main__":
    main(["demo", *__import__("sys").argv[1:]])

