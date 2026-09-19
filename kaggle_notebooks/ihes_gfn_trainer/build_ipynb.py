"""Stitch cells/*.py + cells/*.md (filename order) into the .ipynb."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "cayleypy-ihes-gfn-trainer.ipynb"


def main() -> int:
    cells = []
    for path in sorted((HERE / "cells").iterdir()):
        if path.suffix not in (".py", ".md"):
            continue
        src = path.read_text(encoding="utf-8").rstrip("\n")
        lines = src.split("\n")
        source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        cid = path.stem
        if path.suffix == ".md":
            cells.append({"cell_type": "markdown", "id": cid, "metadata": {},
                          "source": source})
        else:
            cells.append({"cell_type": "code", "id": cid, "metadata": {},
                          "source": source, "outputs": [], "execution_count": None})
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.10"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
