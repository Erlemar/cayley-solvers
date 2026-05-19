"""Build (state, action, remaining_distance) dataset from a submission CSV.

Each pid contributes path_len tuples: at step i along the realized solution,
the state is the partial-application of the first i moves to the initial scramble,
the action is the move taken at step i, and the remaining distance is path_len - i.

These tuples are AlphaZero-style training data: policy_target = action_taken (one-hot),
value_target = remaining_distance.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/67_build_az_dataset.py \\
        --submission megaminx/submissions/merge_v7_curr_v3_pi_v2_rescue.csv \\
        --out megaminx/data/az_dataset_78029.pt
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from megaminx.puzzle import Megaminx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    state_size = len(puzzle.solved_state)
    move_to_idx = {name: i for i, name in enumerate(puzzle.move_names)}

    test_rows = list(csv.DictReader(open(PROJECT / "data" / "test.csv")))
    sub_rows = list(csv.DictReader(open(args.submission)))

    print(f"submission: {len(sub_rows)} rows; test: {len(test_rows)} pids", flush=True)

    all_states = []
    all_actions = []
    all_values = []  # remaining distances
    n_skipped = 0

    for sub_row in sub_rows:
        pid = int(sub_row["initial_state_id"])
        path_str = sub_row["path"].strip()
        if not path_str:
            n_skipped += 1
            continue
        path_moves = path_str.split(".")
        path_len = len(path_moves)

        s0 = tuple(int(x) for x in test_rows[pid]["initial_state"].split(","))
        cur = list(s0)

        for i, move_name in enumerate(path_moves):
            if move_name not in move_to_idx:
                # Unknown move — skip this pid
                print(f"  pid={pid}: unknown move {move_name!r} at step {i}, skipping",
                      flush=True)
                break
            action_id = move_to_idx[move_name]
            remaining = path_len - i  # moves left including this one
            all_states.append(np.array(cur, dtype=np.int8))
            all_actions.append(action_id)
            all_values.append(remaining)
            # Apply the move
            gen = puzzle.generators[move_name]
            cur = [cur[g] for g in gen]
        # Verify final state is solved (sanity check)
        if tuple(cur) != puzzle.solved_state:
            print(f"  pid={pid}: path doesn't reach solved (off-by-something?), kept tuples anyway",
                  flush=True)

    print(f"\nDataset: {len(all_states):,} (state, action, value) tuples from "
          f"{len(sub_rows) - n_skipped} solved pids ({n_skipped} skipped)", flush=True)

    states_t = torch.tensor(np.stack(all_states), dtype=torch.int8)
    actions_t = torch.tensor(all_actions, dtype=torch.int8)
    values_t = torch.tensor(all_values, dtype=torch.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "states": states_t,
        "actions": actions_t,
        "values": values_t,
        "source": str(args.submission),
    }, args.out)
    print(f"saved {args.out}: states {tuple(states_t.shape)}, "
          f"value range [{values_t.min():.0f}, {values_t.max():.0f}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
