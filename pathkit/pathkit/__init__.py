"""pathkit -- path post-processing for Cayley-graph puzzles.

Read README.md for the methods, DISCIPLINE.md for the process rules, VERDICTS.md for what
has already been measured. Start with:

    python -m pathkit.cli selftest
    python -m pathkit.cli plan --preset <puzzle> --in <submission.csv>
"""
from __future__ import annotations

__version__ = "1.0.0"

from .puzzle import Puzzle

__all__ = [
    "Puzzle",
    "backend", "balls", "bridge", "cheap", "io", "ladder", "merge",
    "neural", "suffix", "table", "window",
]
