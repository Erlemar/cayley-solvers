"""Test whether the commutator library catches real beam-search windows.

For each path in the current best submission:
  - Slide a window of size W (7 or 8) along the path.
  - Compute the net permutation of the window.
  - If that perm is in the commutator library AND the library word is shorter
    than W, count it as a potential saving.

This counts SAVINGS — does NOT actually rewrite the submission. We want to
know if there's signal before investing in the full rewrite logic (which
needs careful state-tracking when the replacement changes path geometry).
"""
from __future__ import annotations

import csv
import pickle
import sys
import time
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
    gens_np = [np.array(puzzle.generators[n], dtype=np.int64) for n in move_names]

    print("loading commutator library...")
    with open(PROJECT / "data" / "commutator_table.pkl", "rb") as f:
        d = pickle.load(f)
    table = d["table"]  # tuple_perm -> tuple_word_idxs
    print(f"  {len(table):,} entries, max_depth={d['max_depth']}")

    # Find shortest word per perm in case of duplicates
    shortest_word = {}
    for perm, word in table.items():
        if perm not in shortest_word or len(word) < len(shortest_word[perm]):
            shortest_word[perm] = word

    print("\nloading current best submission...")
    sub_path = PROJECT / "submissions" / "merge_plus_sym8_top20.csv"
    rows = list(csv.DictReader(open(sub_path)))
    print(f"  {len(rows)} pids, total moves {sum(len(r['path'].split('.')) for r in rows):,}")

    # For each window size W in {7, 8}, count savings.
    identity = tuple(range(len(puzzle.solved_state)))
    total_potential_savings = 0
    n_paths_with_match = 0
    matched_pids = []

    for W in (7, 8):
        n_matches_w = 0
        savings_w = 0
        for row in rows:
            pid = int(row["initial_state_id"])
            path = row["path"].split(".")
            if len(path) < W:
                continue
            path_idx = [name_to_idx[m] for m in path]
            for start in range(len(path) - W + 1):
                window = path_idx[start : start + W]
                # Compute net permutation of this window
                cur = np.arange(len(puzzle.solved_state), dtype=np.int64)
                for gi in window:
                    cur = cur[gens_np[gi]]
                perm = tuple(cur.tolist())
                if perm in shortest_word:
                    word = shortest_word[perm]
                    if len(word) < W:
                        n_matches_w += 1
                        savings_w += W - len(word)
                        if pid not in matched_pids:
                            matched_pids.append(pid)
        print(f"\nwindow size W={W}:")
        print(f"  matches found: {n_matches_w}")
        print(f"  potential savings (sum W - len(word)): {savings_w}")
        total_potential_savings += savings_w

    print(f"\ntotal potential savings (W=7+8 combined): {total_potential_savings}")
    print(f"unique pids with at least one match: {len(matched_pids)}")
    if matched_pids:
        print(f"  sample pids: {matched_pids[:10]}")

    return 0 if total_potential_savings > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
