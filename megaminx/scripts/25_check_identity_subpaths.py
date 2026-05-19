"""T2.1 sanity: do our submitted paths contain residual identity sub-paths?

Premise: BFS-d6 window-replacement should have already eliminated any contiguous
sub-path that evaluates to identity (window size up to 12). If we find any,
post-processing wasn't run to fixpoint and there's free signal.

If we find 0, then identity-word INSERTION (the IHES-prototype mechanism) is
redundant with what we already do — needs a different IF mechanism (e.g.,
non-identity commutator insertion that re-routes the path).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "src"))

from megaminx.puzzle import Megaminx


def main() -> int:
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    move_names = list(puzzle.move_names)
    name_to_idx = {n: i for i, n in enumerate(move_names)}
    gens = [np.array(puzzle.generators[n], dtype=np.int64) for n in move_names]
    identity = tuple(range(len(puzzle.solved_state)))

    sub_path = PROJECT / "submissions" / "merge_plus_sym8_top20.csv"
    rows = list(csv.DictReader(open(sub_path)))
    print(f"loaded {len(rows)} pids from {sub_path.name}")

    # For each path, slide a window of size W; compute net perm; flag identity.
    n_total_windows = 0
    by_W = {}
    flagged_pids = []
    for W in (4, 6, 8, 10, 12):
        n_id = 0
        for r in rows:
            path = r['path'].split('.')
            if len(path) < W:
                continue
            path_idx = [name_to_idx[m] for m in path]
            for i in range(0, len(path_idx) - W + 1):
                state = identity
                for mi in path_idx[i:i + W]:
                    state = tuple(state[g] for g in gens[mi])
                if state == identity:
                    n_id += 1
                    if len(flagged_pids) < 5:
                        flagged_pids.append((int(r['initial_state_id']), i, W,
                                             '.'.join(path[i:i + W])))
                n_total_windows += 1
        by_W[W] = n_id
        print(f"  W={W:2d}: {n_id:>4d} identity windows")

    print(f"\nfirst 5 flagged windows (pid, pos, W, word):")
    for f in flagged_pids:
        print(f"  pid={f[0]:4d} pos={f[1]:3d} W={f[2]:2d}: {f[3]}")

    if all(v == 0 for v in by_W.values()):
        print("\n=> 0 identity sub-paths at any W in 4..12.")
        print("   Existing BFS-d6 window-replacement is already at fixpoint w.r.t.")
        print("   identity-collapse. Identity-word INSERTION mechanism is redundant.")
        print("   Need a different IF mechanism (non-identity commutator insertion +")
        print("   tail re-solve) or pivot to T1.6 SA/LAHC.")
    else:
        total = sum(by_W.values())
        print(f"\n=> Found {total} residual identity sub-paths! Free signal.")
        print(f"   Re-running full_post_process should remove them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
