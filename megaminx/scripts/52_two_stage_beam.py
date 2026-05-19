"""Two-stage beam search test: F2L target → V0 target, both using m_curr_v3.

Stage 1: beam from initial state, terminate when 35 F2L stickers correct.
Stage 2: beam from stage-1 end state, terminate when fully solved (V0).
Output: concatenated path; compare to single-stage beam-to-V0 directly.

If 2-stage path < 1-stage path, decomposition gain is real and worth investing
in V_F2L training. If not, decomposition is a dead end.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/52_two_stage_beam.py \\
        --v-checkpoint megaminx/models/m_curr_v3/epoch_0499.pt \\
        --qshort-student megaminx/models/m23_v2_sym_aware/epoch_0499.pt \\
        --pids 0,1,2,3,4 \\
        --beam 65536 --max-steps 120 \\
        --out megaminx/submissions/two_stage_smoke.csv
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
from cayley.model import ResMLPDistance
from megaminx.decomposition import F2L_STICKERS_FROZEN
from megaminx.puzzle import Megaminx


def load_model(path: Path, device: str, output_dim: int = 1):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    mc = ckpt.get("model_config", {})
    model = ResMLPDistance(
        state_size=mc.get("state_size", 120),
        num_classes=mc.get("num_classes", 120),
        hidden_dims=tuple(mc.get("hidden_dims", [2048, 512])),
        num_res_blocks=mc.get("num_res_blocks", 2),
        encoding=mc.get("encoding", "embedding"),
        embed_dim=mc.get("embed_dim", 16),
        output_dim=mc.get("output_dim", output_dim),
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  load: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model = model.to(device).eval()
    return model


def two_stage_solve(solver, qshort_solver, initial_state, cfg, V0, f2l_positions, stage2_solver=None):
    """Run two-stage beam: F2L goal then V0 goal.

    Returns (found, total_len, full_path_names, stage1_len, stage2_len, stage1_states_seen, stage2_states_seen).
    """
    s2_solver = stage2_solver or solver

    # Stage 1: F2L goal
    def f2l_goal(states):
        # states: (B, state_size) int8
        # V0: (state_size,) int8
        return (states[:, f2l_positions] == V0[f2l_positions]).all(dim=1)

    t0 = time.time()
    found1, len1, path1 = solver.solve(initial_state, cfg, goal_check_fn=f2l_goal)
    stage1_wall = time.time() - t0

    if not found1:
        return False, 0, [], 0, 0, stage1_wall, 0.0

    # Apply path1 to get stage-1 end state
    state = list(initial_state)
    for move_name in path1:
        gen = solver.puzzle.generators[move_name]
        state = [state[g] for g in gen]
    state_tuple = tuple(state)

    # Stage 2: from end-of-stage-1 state, solve to V0
    t0 = time.time()
    found2, len2, path2 = s2_solver.solve(state_tuple, cfg)
    stage2_wall = time.time() - t0

    if not found2:
        return False, 0, [], len1, 0, stage1_wall, stage2_wall

    return True, len1 + len2, path1 + path2, len1, len2, stage1_wall, stage2_wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v-checkpoint", required=True, type=Path,
                    help="V model for stage 2 (full puzzle V, e.g. m_curr_v3)")
    ap.add_argument("--v-f2l-checkpoint", type=Path, default=None,
                    help="V_F2L model for stage 1. If unset, uses --v-checkpoint for both.")
    ap.add_argument("--qshort-student", required=True, type=Path)
    ap.add_argument("--qshort-alpha", type=float, default=2.0)
    ap.add_argument("--pids", type=str, default="0,1,2,3,4")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--single-stage-only", action="store_true",
                    help="Skip two-stage, only run single-stage baseline (for comparison)")
    args = ap.parse_args()

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    test_rows = list(csv.DictReader(open(PROJECT / "data" / "test.csv")))
    pids = [int(x) for x in args.pids.split(",") if x.strip()]
    print(f"Two-stage beam test: {len(pids)} pids, beam={args.beam:,}", flush=True)

    # Load V model + Q-shortlister
    v_model = load_model(args.v_checkpoint, args.device, output_dim=1)
    q_model = load_model(args.qshort_student, args.device, output_dim=24)
    if args.bf16 and args.device == "cuda":
        v_model = v_model.to(torch.bfloat16)
        q_model = q_model.to(torch.bfloat16)

    # Stage-1 V (V_F2L) — defaults to v_model if not specified
    if args.v_f2l_checkpoint:
        v_f2l_model = load_model(args.v_f2l_checkpoint, args.device, output_dim=1)
        if args.bf16 and args.device == "cuda":
            v_f2l_model = v_f2l_model.to(torch.bfloat16)
        print(f"  V_F2L for stage 1: {args.v_f2l_checkpoint}", flush=True)
        print(f"  V_full for stage 2: {args.v_checkpoint}", flush=True)
    else:
        v_f2l_model = v_model
        print(f"  Single V model for both stages: {args.v_checkpoint}", flush=True)

    # Build solvers — separate for stage 1 (uses V_F2L) and stage 2 (uses V_full)
    solver_s1 = KhoruzhiiSolver(
        puzzle=puzzle, model=v_f2l_model, device=args.device,
        internal_batch_size=2**14,
    )
    solver_s2 = KhoruzhiiSolver(
        puzzle=puzzle, model=v_model, device=args.device,
        internal_batch_size=2**14,
    )
    solver = solver_s1  # for backward compat (single-stage uses V_full)

    cfg = KhoruzhiiSearchConfig(
        beam_width=args.beam,
        num_steps=args.max_steps,
    )

    f2l_positions = torch.tensor(list(F2L_STICKERS_FROZEN), dtype=torch.int64, device=args.device)
    V0 = torch.tensor(puzzle.solved_state, dtype=torch.int8, device=args.device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    f_csv = open(args.out, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["initial_state_id", "path", "single_stage_len", "two_stage_len",
                     "stage1_len", "stage2_len", "two_stage_wall_s"])

    print(f"\n{'pid':>4} | {'1-stage':>8} {'2-stage':>8} {'s1':>5} {'s2':>5} | "
          f"{'2-stage wall':>14} | verdict", flush=True)
    print("-" * 80, flush=True)

    n_two_stage_better = 0
    n_one_stage_better = 0
    n_tied = 0
    total_one_stage = 0
    total_two_stage = 0
    n_attempted = 0

    for pid in pids:
        s0 = tuple(int(x) for x in test_rows[pid]["initial_state"].split(","))
        n_attempted += 1

        # Single-stage baseline (uses V_full)
        t0 = time.time()
        found_one, len_one, path_one = solver_s2.solve(s0, cfg)
        wall_one = time.time() - t0

        if not found_one:
            print(f"{pid:>4} | {'N/F':>8} {'?':>8}    {0:>5} {0:>5} | {0.0:>14.1f}s | 1-stage failed",
                  flush=True)
            writer.writerow([pid, "", "", "", "", "", ""])
            continue

        if args.single_stage_only:
            print(f"{pid:>4} | {len_one:>8} | {wall_one:>14.1f}s | 1-stage only",
                  flush=True)
            writer.writerow([pid, ".".join(path_one), len_one, "", "", "", ""])
            continue

        # Two-stage: solver_s1 (V_F2L) for stage 1, solver_s2 (V_full) for stage 2
        found_two, len_two, path_two, s1_len, s2_len, s1_wall, s2_wall = two_stage_solve(
            solver_s1, None, s0, cfg, V0, f2l_positions, stage2_solver=solver_s2,
        )
        wall_two = s1_wall + s2_wall

        if not found_two:
            verdict = "2-stage failed"
            len_two_str = "F"
        else:
            total_one_stage += len_one
            total_two_stage += len_two
            if len_two < len_one:
                n_two_stage_better += 1
                verdict = f"2S WINS by {len_one - len_two}"
            elif len_two > len_one:
                n_one_stage_better += 1
                verdict = f"1S wins by {len_two - len_one}"
            else:
                n_tied += 1
                verdict = "TIED"
            len_two_str = str(len_two)

        print(f"{pid:>4} | {len_one:>8} {len_two_str:>8} {s1_len:>5} {s2_len:>5} | "
              f"{wall_two:>14.1f}s | {verdict}", flush=True)
        writer.writerow([pid, ".".join(path_two) if found_two else "",
                         len_one, len_two if found_two else "",
                         s1_len, s2_len if found_two else "", round(wall_two, 1)])
        f_csv.flush()

    f_csv.close()

    if not args.single_stage_only:
        n_compared = n_two_stage_better + n_one_stage_better + n_tied
        print(f"\n{'-' * 80}", flush=True)
        print(f"Compared {n_compared}/{n_attempted} pids:", flush=True)
        print(f"  2-stage WINS: {n_two_stage_better}", flush=True)
        print(f"  1-stage wins: {n_one_stage_better}", flush=True)
        print(f"  tied:         {n_tied}", flush=True)
        if n_compared > 0:
            print(f"  total 1-stage: {total_one_stage}", flush=True)
            print(f"  total 2-stage: {total_two_stage}", flush=True)
            delta = total_two_stage - total_one_stage
            print(f"  net delta:    {delta:+d} (negative = 2-stage saves moves)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
