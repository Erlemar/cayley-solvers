"""Build a solved-path policy dataset for Idea 4 (Policy-guided V + π search).

For each pid in a verified submission CSV, walk the path forward to enumerate
`(state_i, action_i, suffix_len_i)` tuples, where:
  - state_i = state after applying the first i moves to the initial test state
  - action_i = generator index of the (i+1)-th move
  - suffix_len_i = L - i  (remaining path length to solved)

The weight for each tuple is `1 / (1 + suffix_len)`: moves close to solved get
higher weight, where the model has tighter supervision (the realized path is
near-optimal in the last few moves).

Output: `data/policy_train.pt` with keys:
  states:      (N, 120) int8
  actions:     (N,)     int8
  suffix_lens: (N,)     int16
  weights:     (N,)     float32
  metadata:    dict

Usage:
    .venv/Scripts/python.exe megaminx/scripts/43_build_policy_dataset.py \\
        --submission megaminx/submissions/merge_v19b_plus_community.csv \\
        --out megaminx/data/policy_train.pt
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path,
                    default=PROJECT / "submissions" / "merge_v19b_plus_community.csv",
                    help="verified submission CSV; one row per pid with .-joined path")
    ap.add_argument("--test-csv", type=Path,
                    default=PROJECT / "data" / "test.csv")
    ap.add_argument("--out", type=Path,
                    default=PROJECT / "data" / "policy_train.pt")
    ap.add_argument("--include-final-solved-state", action="store_true",
                    help="also emit (state_L, *, 0) tuples — these have suffix_len=0 "
                         "and weight=1, but no clear next action. Default off.")
    args = ap.parse_args()

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    name_to_idx = {n: i for i, n in enumerate(puzzle.move_names)}
    gens = np.stack([puzzle.generators[n] for n in puzzle.move_names], axis=0).astype(np.int64)

    # Load test.csv: pid → initial state.
    test_states: dict[int, np.ndarray] = {}
    with open(args.test_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = int(row["initial_state_id"])
            test_states[pid] = np.array(
                [int(x) for x in row["initial_state"].split(",")], dtype=np.int8
            )

    # Load submission.
    sub: dict[int, list[str]] = {}
    with open(args.submission, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = int(row["initial_state_id"])
            path_str = row["path"]
            sub[pid] = path_str.split(".") if path_str else []

    print(f"submission: {len(sub)} pids; test: {len(test_states)} pids")

    # Walk each path forward; emit tuples.
    states_out: list[np.ndarray] = []
    actions_out: list[int] = []
    suffix_out: list[int] = []
    n_invalid = 0
    skipped_short = 0
    for pid in sorted(sub):
        path = sub[pid]
        L = len(path)
        if L == 0:
            skipped_short += 1
            continue
        if pid not in test_states:
            print(f"  WARNING: pid {pid} not in test.csv", file=sys.stderr)
            continue
        state = test_states[pid].copy()
        ok = True
        for i, move_name in enumerate(path):
            if move_name not in name_to_idx:
                print(f"  WARNING: pid {pid} has unknown move {move_name!r}", file=sys.stderr)
                ok = False
                break
            a = name_to_idx[move_name]
            states_out.append(state.copy())
            actions_out.append(a)
            suffix_out.append(L - i)  # i=0..L-1, so suffix=L..1; never 0 here
            # Apply the move to advance the state for next iteration.
            state = state[gens[a]]
        if not ok:
            n_invalid += 1
            continue
        # Optional: emit the final solved state with suffix_len=0 (no action).
        # We skip the action label (no clear "correct next action" at solved);
        # if --include-final-solved-state is set, we emit it with action=-1 and
        # downstream training filters those out before CE.
        if args.include_final_solved_state:
            states_out.append(state.copy())
            actions_out.append(-1)  # sentinel; trainer should mask out
            suffix_out.append(0)

    print(f"emitted: {len(states_out):,} tuples ({n_invalid} pids skipped with bad moves, "
          f"{skipped_short} with empty paths)")
    if not states_out:
        print("ERROR: no tuples emitted", file=sys.stderr)
        return 2

    states = torch.from_numpy(np.stack(states_out, axis=0)).to(torch.int8)
    actions = torch.tensor(actions_out, dtype=torch.int8)
    suffix = torch.tensor(suffix_out, dtype=torch.int16)
    # weight = 1 / (1 + suffix_len). suffix_len ranges 0..L; weight in (0, 1].
    weights = (1.0 / (1.0 + suffix.to(torch.float32))).to(torch.float32)

    # Diagnostics.
    suffix_np = suffix.numpy()
    print(f"  suffix_len: min={int(suffix_np.min())} median={int(np.median(suffix_np))} "
          f"p90={int(np.percentile(suffix_np, 90))} max={int(suffix_np.max())}")
    print(f"  weight:     min={float(weights.min()):.4f} median={float(weights.median()):.4f} "
          f"max={float(weights.max()):.4f}  sum={float(weights.sum()):.1f}")
    actions_np = actions.numpy()
    valid_actions = actions_np[actions_np >= 0]
    if len(valid_actions) > 0:
        bincount = np.bincount(valid_actions, minlength=len(puzzle.move_names))
        print(f"  action histogram: min_count={int(bincount.min())} max={int(bincount.max())} "
              f"mean={float(bincount.mean()):.0f}")
        if int(bincount.min()) == 0:
            zero_actions = [puzzle.move_names[i] for i in range(len(bincount)) if bincount[i] == 0]
            print(f"  WARNING: {len(zero_actions)} actions never appear: {zero_actions[:10]}")

    metadata = {
        "submission": str(args.submission),
        "n_pids": len(sub),
        "n_tuples": len(states_out),
        "include_final_solved_state": args.include_final_solved_state,
        "n_actions": len(puzzle.move_names),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "states": states,
        "actions": actions,
        "suffix_lens": suffix,
        "weights": weights,
        "metadata": metadata,
    }, args.out)
    print(f"wrote {args.out}: {len(states_out):,} tuples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
