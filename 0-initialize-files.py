"""This module builds the directory structure for the directories not under VCS."""
from os import makedirs

PATHS = [
    "data/raw",
    "data/processed",
    "out/models",
    "out/plots",
    "out/analysis",
    "out/eval",
]

if __name__ == "__main__":
    for p in PATHS:
        makedirs(p, exist_ok=True)
    print("Done.")
