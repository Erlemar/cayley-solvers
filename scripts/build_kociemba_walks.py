"""Build a (state, distance) training supplement from Kociemba's 1003 solutions.

For each test puzzle, Kociemba's path solves the scramble in ~38 avg moves. Starting
from `solved` and applying the path's inverse (reverse + invert each move), we visit
each intermediate state along the way. At step i from solved (i = 0..k), the state is
the `i`-th on the path; its distance-to-solved along this path is i (an upper bound).

These states come from the real test distribution (graded difficulty), complementing
the uniform distribution produced by random walks from solved.

    python scripts/build_kociemba_walks.py \\
        --kociemba data/kociemba_fallback.csv \\
        --out data/kociemba_walks.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.puzzle import PictureCube
from cayley.verify import load_submission, load_test_states, verify_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kociemba", type=Path, default=PROJECT / "data" / "kociemba_fallback.csv")
    ap.add_argument("--test", type=Path, default=PROJECT / "data" / "test.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--include-test-state", action="store_true",
                    help="also emit the scrambled test state with label = len(path)")
    args = ap.parse_args()

    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    states_map = load_test_states(args.test)
    subs = load_submission(args.kociemba)

    all_states: list[tuple[int, ...]] = []
    all_depths: list[int] = []

    for pid in sorted(states_map):
        path = subs.get(pid, [])
        test_state = states_map[pid]
        if not path:
            continue
        if not verify_path(puzzle, test_state, path).ok:
            print(f"WARN: Kociemba path for pid {pid} does not verify — skipping")
            continue
        # We want states along the solved → scrambled trajectory paired with
        # "distance-to-solved" = steps-remaining-to-solved along this path.
        #
        # From the puzzle's convention: `apply_path(scrambled, path) == solved`.
        # So `apply_path(solved, invert_path(path))` walks back to scrambled through
        # the same states (in reverse order). At step i from solved, state is s_i and
        # its distance-along-path to solved is i.
        k = len(path)
        inv_path = puzzle.invert_path(path)
        s = tuple(puzzle.solved_state)
        # Don't emit identity (depth 0) here — BFS-d5 already covers it.
        for i, m in enumerate(inv_path, start=1):  # i = 1..k
            s = puzzle.apply_move(s, m)
            if i < k or args.include_test_state:
                all_states.append(s)
                all_depths.append(i)

    state_size = len(puzzle.solved_state)
    states_np = np.array(all_states, dtype=np.int64)
    depths_np = np.array(all_depths, dtype=np.int64)
    assert states_np.shape == (len(all_states), state_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"states": states_np, "depths": depths_np}, f, protocol=5)

    print(f"emitted {len(all_states):,} (state, depth) pairs")
    print(f"depth histogram: min={depths_np.min()} max={depths_np.max()} "
          f"mean={depths_np.mean():.1f} p50={int(np.median(depths_np))} "
          f"p90={int(np.percentile(depths_np, 90))}")
    print(f"saved: {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
