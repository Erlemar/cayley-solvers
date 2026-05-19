"""Beam-stack rescue: re-attempt specific puzzles with runner-up backtracking.

Standard beam keeps top-B per layer; beam-stack also stores (B+1)..(2B) "runners"
at checkpoint layers. On failure, restart from a deepest stored runner and try
again. Targeted at puzzles where the simple beam misses by 1-2 wrong choices.

Usage:
    python megaminx/scripts/12_beam_stack_rescue.py \
        --checkpoint megaminx/models/m05_bellman_warm/epoch_0499.pt \
        --pids 490,920 \
        --beam 131072 --max-steps 120 \
        --max-retries 3 --checkpoint-every 10 \
        --out megaminx/submissions/rescue.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
LAB = PROJECT / "beam_lab"
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from beam_search import KhoruzhiiSearchConfig, setup_model_for_inference  # noqa: E402
from beam_search_stack import BeamStackSolver  # noqa: E402
from model import load_checkpoint  # noqa: E402
from puzzle import Megaminx  # noqa: E402


def load_puzzle_states(test_csv: Path, pids: list[int]) -> dict[int, tuple[int, ...]]:
    out: dict[int, tuple[int, ...]] = {}
    pid_set = set(pids)
    with open(test_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            pid = int(row["initial_state_id"])
            if pid in pid_set:
                out[pid] = tuple(int(x) for x in row["initial_state"].split(","))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--puzzle-info", type=Path, default=LAB / "data" / "puzzle_info.json")
    ap.add_argument("--test-csv", type=Path, default=PROJECT / "data" / "test.csv")
    ap.add_argument("--pids", type=str, required=True, help="comma-separated list of pids to rescue")
    ap.add_argument("--beam", type=int, default=131072)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bf16", action="store_true", default=True)
    args = ap.parse_args()

    pids = [int(x) for x in args.pids.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    puzzle = Megaminx.load(args.puzzle_info)
    states = load_puzzle_states(args.test_csv, pids)
    print(f"loaded {len(states)} puzzle states for pids: {sorted(states)}")

    print(f"loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(str(args.checkpoint), device=device)
    if args.bf16 and device == "cuda":
        model = model.to(torch.bfloat16)
    setup_model_for_inference(model)

    solver = BeamStackSolver(
        puzzle, model, device=device,
        internal_batch_size=16384,
        random_seed=0,
        state_dtype=torch.int8,
    )
    cfg = KhoruzhiiSearchConfig(
        beam_width=args.beam,
        num_steps=args.max_steps,
        num_attempts=1,
        internal_batch_size=16384,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print()
    print(f"{'pid':>6s} {'attempt':>8s} {'found':>5s} {'path':>5s} {'wall':>6s} {'valid':>5s}")
    print("-" * 50)
    for pid in pids:
        if pid not in states:
            print(f"{pid:>6d}  (state not in test.csv, skip)")
            continue
        state = states[pid]
        ok, plen, names, prof = solver.solve_with_backtrack(
            state, cfg,
            checkpoint_every=args.checkpoint_every,
            max_retries=args.max_retries,
        )
        # Verify path
        valid = False
        if ok:
            cur = tuple(state)
            for m in names:
                cur = puzzle.apply_move(cur, m)
            valid = puzzle.is_solved(cur)
        valid_mark = "OK" if valid else ("BAD" if ok else "-")
        print(
            f"{pid:>6d}  {prof.success_attempt:>+8d}  {int(prof.found):>5d}  "
            f"{plen if ok else 0:>5d}  {prof.total_s:>6.1f}  "
            f"{valid_mark:>5s}",
            flush=True,
        )
        if ok and valid:
            rows.append({"initial_state_id": pid, "path": ".".join(names)})

    # Write CSV
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["initial_state_id", "path"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {args.out}: {len(rows)} rescued pids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
