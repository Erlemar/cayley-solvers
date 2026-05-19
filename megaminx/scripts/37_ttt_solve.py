"""T1.3 — Test-Time Training (TTT) per puzzle.

For each test puzzle:
  1. Snapshot V model weights.
  2. Generate ~32 random walks of length ~8 STARTING FROM the puzzle's initial
     state (not from solved). This collects states in the local neighborhood of
     the test puzzle.
  3. Compute Bellman targets on these local states using a frozen target_model
     (= the original V before any fine-tuning).
     target(s) = clip(1 + min_a target_model(apply(s, a)), 0, +inf).
  4. Fine-tune V for ~50 SGD steps (MSE on Bellman targets). The model is now
     locally Bellman-consistent near the test state.
  5. Beam-solve the test puzzle with locally-tuned V.
  6. Restore weights from snapshot. Repeat for next puzzle.

Hypothesis: m05's *global* loss averages over the whole graph; its *local*
error near a specific scramble is much higher. TTT closes the local gap by
making the model self-consistent in the test puzzle's neighborhood.

Distinct from solver-trace primary training (m37, OOD failure) — TTT keeps
the m05 backbone, doesn't replace its training distribution. It just nudges
predictions to be locally Bellman-consistent.

Acceptance gate: any net improvement to the submission via min-merge with the
current best counts.

Usage:
  PYTHONUTF8=1 .venv/Scripts/python.exe -u megaminx/scripts/37_ttt_solve.py \\
      --checkpoint megaminx/models/m05_bellman_warm/epoch_0499.pt \\
      --out megaminx/submissions/m05_ttt_strat5.csv \\
      --beams 65536 --max-steps 150 \\
      --stratified 5 --strat-seed 0 \\
      --bf16
"""
from __future__ import annotations

import argparse
import copy
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.data import GeneratorTable
from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cayley.search import load_model_checkpoint
from cayley.verify import load_test_states, verify_path
from megaminx.post_process import full_post_process
from megaminx.puzzle import Megaminx


@torch.no_grad()
def _walks_from_state(
    start_state: torch.Tensor,         # (state_size,) int64
    n_walks: int,
    walk_len: int,
    generators: torch.Tensor,          # (n_gen, state_size) int64
    seed: int,
    device: str,
) -> torch.Tensor:
    """Generate `n_walks` random walks of length `walk_len` starting from `start_state`.
    Returns (n_walks * walk_len, state_size) int64."""
    n_gen, state_size = generators.shape
    g = torch.Generator(device=device); g.manual_seed(seed)
    states = start_state.unsqueeze(0).expand(n_walks, state_size).clone()
    out = torch.empty((n_walks * walk_len, state_size), dtype=torch.int64, device=device)
    for k in range(walk_len):
        action = torch.randint(0, n_gen, (n_walks,), generator=g, device=device)
        gen_rows = generators[action]
        states = torch.gather(states, 1, gen_rows)
        out[k * n_walks:(k + 1) * n_walks] = states
    return out


@torch.no_grad()
def _bellman_target(
    target_model: torch.nn.Module,
    states: torch.Tensor,                # (B, state_size) int64
    generators: torch.Tensor,            # (n_gen, state_size) int64
    solved_state: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """y = 1 + min_a target_model(apply(s, a)) for each parent state s.
    Children that are solved get value 0 (boundary condition)."""
    B, S = states.shape
    n_gen = generators.shape[0]
    children = torch.gather(
        states.unsqueeze(1).expand(B, n_gen, S), 2,
        generators.unsqueeze(0).expand(B, n_gen, S),
    )
    children_flat = children.reshape(B * n_gen, S)
    is_solved = (children_flat == solved_state).all(dim=1)
    target_model.eval()
    vals = torch.empty(B * n_gen, dtype=torch.float32, device=states.device)
    for i in range(0, B * n_gen, chunk_size):
        chunk = children_flat[i:i + chunk_size]
        v = target_model(chunk).flatten().to(torch.float32)
        vals[i:i + chunk_size] = v
    vals = torch.where(is_solved, torch.zeros_like(vals), vals)
    vals = vals.view(B, n_gen)
    target = 1.0 + vals.min(dim=1).values
    return target.clamp(min=0.0)


def ttt_finetune(
    model: torch.nn.Module,
    target_model: torch.nn.Module,
    test_state: torch.Tensor,            # (state_size,) int64
    generators: torch.Tensor,
    solved_state: torch.Tensor,
    n_walks: int,
    walk_len: int,
    k_steps: int,
    batch_size: int,
    lr: float,
    device: str,
) -> None:
    """Fine-tune `model` in place on Bellman targets sampled near `test_state`."""
    seed = (hash(tuple(test_state.tolist())) ^ 0xa5f1) & 0x7FFFFFFF
    states = _walks_from_state(test_state, n_walks, walk_len, generators, seed, device)
    n_total = states.size(0)
    # Compute targets ONCE before fine-tuning (frozen target_model = original m05).
    target = _bellman_target(target_model, states, generators, solved_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for step in range(k_steps):
        idx = torch.randperm(n_total, device=device)[:batch_size]
        bs = states[idx]
        bt = target[idx]
        pred = model(bs).flatten().float()
        loss = F.mse_loss(pred, bt.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--beams", type=int, default=65536)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--stratified", type=int, default=5)
    ap.add_argument("--strat-seed", type=int, default=0)
    ap.add_argument("--strat-buckets", type=str, default=None,
                    help="Comma-separated bucket indices (e.g. '7,8,9,10' for hard tail)")
    ap.add_argument("--pids", type=str, default=None,
                    help="Optional explicit pid list (overrides --stratified).")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # TTT hyperparameters.
    ap.add_argument("--ttt-k-steps", type=int, default=50,
                    help="SGD steps per puzzle for TTT fine-tune.")
    ap.add_argument("--ttt-n-walks", type=int, default=32,
                    help="Number of random walks from test_state.")
    ap.add_argument("--ttt-walk-len", type=int, default=8,
                    help="Length of each walk from test_state.")
    ap.add_argument("--ttt-lr", type=float, default=1e-3)
    ap.add_argument("--ttt-batch-size", type=int, default=512)
    ap.add_argument("--internal-batch-size", type=int, default=2**14)
    ap.add_argument("--state-dtype", default="int8",
                    choices=["int8", "int16", "int32", "int64"])
    ap.add_argument("--bfs-table", type=Path, default=None,
                    help="optional BFS-d6 table for window post-proc")
    ap.add_argument("--bfs-max-window", type=int, default=12)
    args = ap.parse_args()

    state_dtype = {"int8": torch.int8, "int16": torch.int16,
                   "int32": torch.int32, "int64": torch.int64}[args.state_dtype]
    # TTT requires gradients through the model — load fp32 always. The --bf16
    # flag is accepted for compatibility but ignored during fine-tuning.
    model_dtype = torch.float32

    print(f"loading puzzle + states + model ...")
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    states_dict = load_test_states(PROJECT / "data" / "test.csv")
    all_ids = sorted(states_dict)

    # pid selection
    if args.pids:
        solve_ids = sorted(int(p) for p in args.pids.split(",") if p.strip())
    else:
        rng = random.Random(args.strat_seed)
        buckets: dict[int, list[int]] = {}
        for pid in all_ids:
            buckets.setdefault(pid // 100, []).append(pid)
        wanted = (set(int(b) for b in args.strat_buckets.split(",") if b.strip())
                  if args.strat_buckets else set(buckets))
        picked = []
        for b in sorted(buckets):
            if b not in wanted:
                continue
            picked.extend(sorted(rng.sample(buckets[b], min(args.stratified, len(buckets[b])))))
        solve_ids = picked

    print(f"solve_ids: {len(solve_ids)} pids "
          f"(buckets={sorted(set(p // 100 for p in solve_ids))})")

    bfs_table = None
    if args.bfs_table is not None and args.bfs_table.exists():
        from megaminx.bfs_bytes import BfsBytesTable
        bfs_table = BfsBytesTable.load(args.bfs_table)
        print(f"bfs-d6 table: {len(bfs_table.table):,} entries")

    # Load V model
    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=model_dtype)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} params")

    # Frozen target model = original m05 weights. Used for ALL puzzles' Bellman targets
    # (no contamination from prior TTT fine-tunes, since we restore between puzzles).
    target_model = copy.deepcopy(model).eval()
    for p in target_model.parameters():
        p.requires_grad = False

    # Solver setup
    solver = KhoruzhiiSolver(
        puzzle, model, device=args.device,
        internal_batch_size=args.internal_batch_size,
        state_dtype=state_dtype,
    )

    gens_table = GeneratorTable.from_puzzle(puzzle)
    generators = torch.from_numpy(gens_table.perms).to(args.device)
    solved_state_t = torch.tensor(puzzle.solved_state, dtype=torch.int64,
                                   device=args.device)

    cfg = KhoruzhiiSearchConfig(beam_width=args.beams, num_steps=args.max_steps,
                                 num_attempts=1)

    # Snapshot to restore between puzzles.
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}

    out_paths: dict[int, list[str]] = {}
    n_solved = 0
    n_total = 0
    t0 = time.time()
    for i, pid in enumerate(solve_ids):
        state = states_dict[pid]
        state_t = torch.tensor(state, dtype=torch.int64, device=args.device)
        t_pid = time.time()

        # Fine-tune locally
        t_ttt_start = time.time()
        ttt_finetune(
            model, target_model, state_t, generators, solved_state_t,
            n_walks=args.ttt_n_walks, walk_len=args.ttt_walk_len,
            k_steps=args.ttt_k_steps, batch_size=args.ttt_batch_size,
            lr=args.ttt_lr, device=args.device,
        )
        t_ttt = time.time() - t_ttt_start

        # Beam-solve
        t_beam_start = time.time()
        found, _, raw = solver.solve(state, cfg)
        t_beam = time.time() - t_beam_start
        path = None
        if found:
            cand = full_post_process(raw)
            if verify_path(puzzle, state, cand).ok:
                path = cand

        # Restore weights for next puzzle
        model.load_state_dict(snapshot)

        bucket = pid // 100
        n_total += 1
        if path is not None:
            out_paths[pid] = path
            n_solved += 1
            print(f"  pid={pid:4d} bucket={bucket} mlen={len(path):3d} "
                  f"ttt={t_ttt:.1f}s beam={t_beam:.1f}s total={time.time()-t_pid:.1f}s",
                  flush=True)
        else:
            print(f"  pid={pid:4d} bucket={bucket} UNSOLVED "
                  f"ttt={t_ttt:.1f}s beam={t_beam:.1f}s total={time.time()-t_pid:.1f}s",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\n=== TTT summary ===")
    print(f"  solved: {n_solved}/{n_total}")
    print(f"  total wall: {elapsed/60:.1f} min")
    if n_solved > 0:
        avg_path = sum(len(p) for p in out_paths.values()) / n_solved
        print(f"  mean path length (solved): {avg_path:.2f}")

    # Per-bucket breakdown
    print(f"\n  per-bucket:")
    print(f"  {'bucket':>8} {'n':>3} {'solved':>7} {'mean':>7}")
    by_bucket: dict[int, list[int]] = {}
    for pid in solve_ids:
        by_bucket.setdefault(pid // 100, []).append(pid)
    for b in sorted(by_bucket):
        pids_b = by_bucket[b]
        solved_b = [out_paths[p] for p in pids_b if p in out_paths]
        avg = sum(len(p) for p in solved_b) / max(len(solved_b), 1) if solved_b else -1
        print(f"  {b*100:4d}-{(b+1)*100-1:<4d} {len(pids_b):3d} "
              f"{len(solved_b):>2d}/{len(pids_b):<2d} {avg:7.2f}")

    # Write CSV (only solved pids; full submission needs fallback merge separately)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid in sorted(out_paths):
            w.writerow([pid, ".".join(out_paths[pid])])
    print(f"\nwrote {args.out} ({len(out_paths)} solved pids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
