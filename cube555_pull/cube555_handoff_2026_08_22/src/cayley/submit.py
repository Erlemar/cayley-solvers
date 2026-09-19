"""Build a submission CSV.

We operate over a set of candidate per-puzzle solutions (from one or more solver runs).
For each puzzle id we:
  1. re-verify every candidate against the initial state,
  2. discard invalid ones,
  3. keep the shortest valid one,
  4. compare to a fallback CSV (e.g. v13/Kociemba or sample) and keep the shorter.
  5. write the result to `submission.csv` in the expected format.

The output format is `initial_state_id,path` with the path as a dot-separated string of
generator names (`d2.-f2.-d1...`). This matches `sample_submission.csv`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from cayley.puzzle import PictureCube
from cayley.verify import load_submission, load_test_states, verify_path


def _collect_candidates(candidates_csvs: Iterable[Path]) -> dict[int, list[list[str]]]:
    out: dict[int, list[list[str]]] = {}
    for csv_path in candidates_csvs:
        for pid, path in load_submission(csv_path).items():
            out.setdefault(pid, []).append(path)
    return out


def build_submission(
    puzzle: PictureCube,
    test_csv: str | Path,
    candidate_csvs: list[Path],
    fallback_csv: str | Path,
    out_path: str | Path,
) -> dict[str, int]:
    states = load_test_states(test_csv)
    fallback = load_submission(fallback_csv)
    candidates = _collect_candidates(candidate_csvs)

    rows: list[tuple[int, str]] = []
    stats = {"from_solver": 0, "from_fallback": 0, "invalid_candidates": 0}
    total_moves = 0

    for pid in sorted(states):
        state = states[pid]
        best: list[str] | None = None

        for path in candidates.get(pid, []):
            result = verify_path(puzzle, state, path)
            if not result.ok:
                stats["invalid_candidates"] += 1
                continue
            if best is None or len(path) < len(best):
                best = path

        fallback_path = fallback.get(pid, [])
        # The fallback must itself be valid — this is asserted by Phase 0 verification.
        if best is None or (fallback_path and len(fallback_path) < len(best)):
            best = fallback_path
            stats["from_fallback"] += 1
        else:
            stats["from_solver"] += 1

        if best is None:
            raise RuntimeError(f"no valid path for puzzle {pid}")
        total_moves += len(best)
        rows.append((pid, ".".join(best)))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["initial_state_id", "path"])
        for pid, path_str in rows:
            writer.writerow([pid, path_str])

    stats["total_moves"] = total_moves
    stats["n_puzzles"] = len(rows)
    return stats
