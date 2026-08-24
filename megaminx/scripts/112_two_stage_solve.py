"""Two-phase (Kociemba-style) megaminx solve + verify.

Stage 1: beam with the Phase-1 MaskedV on the FULL 24-generator puzzle, goal =
         every frozen (LL) sticker home (membership in subgroup H).
Stage 2: beam with the Phase-2 V on the RESTRICTED 12-generator puzzle, goal =
         fully solved.
Concatenate the two move sequences, verify against the original scramble, write
the solved rows to a CSV. Optionally also run a single-stage baseline V on the
full puzzle for a head-to-head length comparison (reproduces Vlad's graphs).

    .venv/Scripts/python.exe megaminx/scripts/112_two_stage_solve.py \
        --phase1-checkpoint megaminx/models/m_phase1_v0/epoch_1499.pt \
        --phase2-checkpoint megaminx/models/m_phase2_v0/epoch_1999.pt \
        --spread 11 --beam 131072 --bf16 \
        --out megaminx/submissions/two_stage_spread.csv

Only solved rows are written (min-merge against your current best for full coverage).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cayley.verify import load_test_states, verify_path
from megaminx.puzzle import Megaminx
from megaminx.two_phase import (
    build_phase2_puzzle,
    load_two_phase_model,
    make_frozen_goal_check,
    movable_frozen_positions,
    two_stage_solve_one,
)


def make_smoke_states(puzzle: Megaminx, depth: int, n: int, seed: int) -> dict[int, tuple]:
    """N shallow random scrambles (full-gen non-backtracking walk) from solved.

    For plumbing validation with briefly-trained models, since real test.csv
    scrambles are far too deep to solve at low training budget.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    names = puzzle.move_names
    out: dict[int, tuple] = {}
    for i in range(n):
        s = list(puzzle.solved_state)
        prev_inv = None
        for _ in range(depth):
            nm = names[int(rng.integers(0, len(names)))]
            while nm == prev_inv:
                nm = names[int(rng.integers(0, len(names)))]
            s = list(puzzle.apply_move(s, nm))
            prev_inv = puzzle.inverse_name(nm)
        out[i] = tuple(s)
    return out


def pick_pids(all_pids: list[int], args) -> list[int]:
    if args.pids:
        return [int(x) for x in args.pids.split(",") if x.strip()]
    if args.spread:
        n = len(all_pids)
        k = min(args.spread, n)
        # evenly spaced indices across the sorted pid list, inclusive of both ends
        idxs = [round(i * (n - 1) / max(1, k - 1)) for i in range(k)]
        return sorted({all_pids[i] for i in idxs})
    return all_pids[:10]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-checkpoint", required=True, type=Path)
    ap.add_argument("--phase2-checkpoint", required=True, type=Path)
    ap.add_argument("--baseline-checkpoint", type=Path, default=None,
                    help="optional single-stage V (full puzzle) for length comparison")
    ap.add_argument("--pids", type=str, default="", help="comma-separated pids")
    ap.add_argument("--spread", type=int, default=0,
                    help="if no --pids, pick this many evenly-spaced pids across the test set")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--max-steps-1", type=int, default=60)
    ap.add_argument("--max-steps-2", type=int, default=90)
    ap.add_argument("--num-attempts", type=int, default=1)
    ap.add_argument("--internal-batch-size", type=int, default=2**14)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--smoke-depth", type=int, default=0,
                    help="if >0, solve N shallow random scrambles of this depth "
                         "instead of test.csv (plumbing check)")
    ap.add_argument("--smoke-n", type=int, default=5)
    ap.add_argument("--smoke-seed", type=int, default=0)
    args = ap.parse_args()

    full = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    puzzle2 = build_phase2_puzzle(full)
    movable, frozen = movable_frozen_positions(full)

    if args.smoke_depth > 0:
        states = make_smoke_states(full, args.smoke_depth, args.smoke_n, args.smoke_seed)
        pids = sorted(states.keys())
        print(f"SMOKE mode: {args.smoke_n} scrambles of depth {args.smoke_depth}", flush=True)
    else:
        states = load_test_states(PROJECT / "data" / "test.csv")
        all_pids = sorted(states.keys())
        pids = pick_pids(all_pids, args)

    print(f"two-phase solve: {len(pids)} pids, beam={args.beam:,}, "
          f"steps={args.max_steps_1}/{args.max_steps_2}, attempts={args.num_attempts}",
          flush=True)
    print(f"frozen(goal) stickers={len(frozen)}, movable(masked)={len(movable)}, "
          f"phase2 gens={len(puzzle2.move_names)}", flush=True)

    v1 = load_two_phase_model(args.phase1_checkpoint, args.device)   # MaskedV
    v2 = load_two_phase_model(args.phase2_checkpoint, args.device)   # plain
    base = (load_two_phase_model(args.baseline_checkpoint, args.device)
            if args.baseline_checkpoint else None)
    if args.bf16 and args.device == "cuda":
        v1 = v1.to(torch.bfloat16)
        v2 = v2.to(torch.bfloat16)
        if base is not None:
            base = base.to(torch.bfloat16)

    solver1 = KhoruzhiiSolver(full, v1, device=args.device,
                              internal_batch_size=args.internal_batch_size)
    solver2 = KhoruzhiiSolver(puzzle2, v2, device=args.device,
                              internal_batch_size=args.internal_batch_size)
    solver_base = (KhoruzhiiSolver(full, base, device=args.device,
                                   internal_batch_size=args.internal_batch_size)
                   if base is not None else None)

    cfg1 = KhoruzhiiSearchConfig(beam_width=args.beam, num_steps=args.max_steps_1,
                                 num_attempts=args.num_attempts,
                                 internal_batch_size=args.internal_batch_size)
    cfg2 = KhoruzhiiSearchConfig(beam_width=args.beam, num_steps=args.max_steps_2,
                                 num_attempts=args.num_attempts,
                                 internal_batch_size=args.internal_batch_size)

    frozen_goal = make_frozen_goal_check(frozen, args.device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "w", newline="", encoding="utf-8")
    writer = csv.writer(out_f)
    writer.writerow(["initial_state_id", "path"])

    n_solved = 0
    sum_total = 0
    sum_s1 = 0
    sum_s2 = 0
    sum_base = 0
    n_base = 0
    fail_phase1 = []
    fail_phase2 = []

    for pid in pids:
        s0 = states[pid]
        t0 = time.time()
        r = two_stage_solve_one(solver1, solver2, frozen_goal, full, s0, cfg1, cfg2)
        wall = time.time() - t0
        if r["status"] == "phase1_fail":
            fail_phase1.append(pid)
            print(f"pid {pid:4d}: PHASE-1 FAIL ({wall:.1f}s)", flush=True)
            continue
        if r["status"] == "phase2_fail":
            fail_phase2.append(pid)
            print(f"pid {pid:4d}: PHASE-2 FAIL (s1 len {r['len1']}, {wall:.1f}s)", flush=True)
            continue
        if r["status"] == "verify_fail":
            print(f"pid {pid:4d}: VERIFY FAIL: {r['reason']} "
                  f"(s1={r['len1']}, s2={r['len2']})", flush=True)
            continue

        len1, len2, total, full_path = r["len1"], r["len2"], r["total"], r["path"]
        writer.writerow([pid, ".".join(full_path)])
        out_f.flush()
        n_solved += 1
        sum_total += total
        sum_s1 += len1
        sum_s2 += len2

        base_str = ""
        if solver_base is not None:
            fb, lb, pb = solver_base.solve(s0, cfg2)
            if fb and verify_path(full, s0, pb).ok:
                sum_base += lb
                n_base += 1
                base_str = f" | 1-stage {lb}"
            else:
                base_str = " | 1-stage FAIL"
        print(f"pid {pid:4d}: total {total:3d} (s1 {len1:2d} + s2 {len2:2d}) "
              f"{wall:.1f}s{base_str}", flush=True)

    out_f.close()

    print("", flush=True)
    print(f"solved {n_solved}/{len(pids)} two-stage", flush=True)
    if n_solved:
        print(f"  mean total {sum_total / n_solved:.2f}  "
              f"(stage1 {sum_s1 / n_solved:.2f} + stage2 {sum_s2 / n_solved:.2f})", flush=True)
    if n_base:
        print(f"  mean 1-stage {sum_base / n_base:.2f} over {n_base} pids "
              f"(compare same pids only)", flush=True)
    if fail_phase1:
        print(f"  phase-1 failures: {fail_phase1}", flush=True)
    if fail_phase2:
        print(f"  phase-2 failures (couldn't close in H): {fail_phase2}", flush=True)
    print(f"wrote {n_solved} solved rows -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
