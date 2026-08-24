"""Stitch cells/*.py + cells/*.md (in filename order) into the .ipynb.

Usage:
    .venv/Scripts/python.exe megaminx/kaggle_notebooks/az4_train_shareable/build_ipynb.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS_DIR = HERE / "cells"
OUT = HERE / "cayleypy-az4-trainer-megaminx.ipynb"


def main() -> int:
    cells = []
    for path in sorted(CELLS_DIR.iterdir()):
        if path.suffix not in (".py", ".md"):
            continue
        src = path.read_text(encoding="utf-8").rstrip("\n")
        # Notebook cell sources are lists of lines WITH trailing newlines (except last).
        lines = src.split("\n")
        source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        cell_id = path.stem  # stable id per cell file (nbformat >= 4.5 requires ids)
        if path.suffix == ".md":
            cells.append({"cell_type": "markdown", "id": cell_id, "metadata": {},
                          "source": source})
        else:
            cells.append({"cell_type": "code", "id": cell_id, "metadata": {},
                          "source": source, "outputs": [], "execution_count": None})
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
