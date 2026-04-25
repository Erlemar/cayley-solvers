"""Validation tool: evaluate one or more checkpoints on BFS-true-distance metrics.

For each checkpoint, computes:
  - mse_true:   MSE between predicted distance and TRUE BFS distance
                (random-walk-noise-free, unlike training-time MSE)
  - top1_acc:   % of states where the model's argmin-predicted neighbor is one of the
                optimal-direction neighbors (i.e., a neighbor at true distance d-1).
                This directly predicts beam-search quality.
  - top3_acc:   same but checking top-3 ranked neighbors.
  - mse_walk:   MSE on a fixed sample of random-walk states (label = walk depth as
                upper bound). Comparable to training-time MSE.

Validation set construction:
  - Sample `n_per_depth` states per BFS depth d in [1, max_d] from bfs_bytes_d6.
  - For each state s at depth d, identify optimal-direction neighbors:
    apply each generator, look up neighbor in BFS table; the neighbor is "optimal"
    iff its BFS depth is d-1 (so taking this move strictly reduces distance).

    python megaminx/scripts/validate_model.py \
        --checkpoints megaminx/models/m07_big_k80/epoch_3999.pt \
                      megaminx/models/m12_muon_k80/epoch_*.pt \
        --bfs-table megaminx/data/bfs_bytes_d6.pkl \
        --n-per-depth 500
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.data import generate_walks_torch
from cayley.search import load_model_checkpoint
from megaminx.bfs_bytes import BfsBytesTable
from megaminx.puzzle import Megaminx


def build_validation_set(
    puzzle: Megaminx,
    bfs: BfsBytesTable,
    n_per_depth: int,
    max_d: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Returns (states, true_distances, optimal_neighbor_lists).

    `states`: (N, 120) int8.
    `true_distances`: (N,) int — depth d in [1, max_d].
    `optimal_neighbor_lists[i]`: list of generator indices gi such that applying gi
                                 to state[i] lands in a state at depth d-1.
    """
    rng = random.Random(seed)
    n_gen = len(puzzle.move_names)
    gens = np.stack([np.array(puzzle.generators[n], dtype=np.int8) for n in puzzle.move_names])

    by_depth: dict[int, list[bytes]] = {d: [] for d in range(max_d + 1)}
    for k, w in bfs.table.items():
        d = len(w)
        if d <= max_d:
            by_depth[d].append(k)

    val_states: list[np.ndarray] = []
    val_dist: list[int] = []
    val_opt: list[list[int]] = []
    print(f"[val] states-per-depth available: " +
          ", ".join(f"d={d}: {len(by_depth[d]):,}" for d in range(max_d + 1)), flush=True)

    for d in range(1, max_d + 1):
        pool = by_depth[d]
        if not pool:
            continue
        sample = rng.sample(pool, min(n_per_depth, len(pool)))
        for sk in sample:
            s = np.frombuffer(sk, dtype=np.int8)
            opt: list[int] = []
            for gi in range(n_gen):
                nb = s[gens[gi]]
                nbk = nb.tobytes()
                w = bfs.table.get(nbk)
                if w is not None and len(w) == d - 1:
                    opt.append(gi)
            if opt:  # only keep if at least one neighbor is verifiably optimal
                val_states.append(s)
                val_dist.append(d)
                val_opt.append(opt)

    states_arr = np.stack(val_states, axis=0)
    dist_arr = np.array(val_dist, dtype=np.int32)
    print(f"[val] built {len(val_states)} states with at least one verifiably-optimal "
          f"neighbor (sampled {n_per_depth}/depth across d=1..{max_d})", flush=True)
    return states_arr, dist_arr, val_opt


@torch.no_grad()
def evaluate_one(
    model_path: Path,
    val_states_t: torch.Tensor,
    val_dist_t: torch.Tensor,
    val_opt: list[list[int]],
    walk_states_t: torch.Tensor,
    walk_depths_t: torch.Tensor,
    puzzle: Megaminx,
    device: str,
    bf16: bool,
) -> dict:
    dtype = torch.bfloat16 if bf16 else torch.float32
    model = load_model_checkpoint(model_path, device=device, dtype=dtype)
    base = getattr(model, "_orig_mod", model)
    base.inference_chunk_size = 4096

    # MSE on true-distance states.
    pred = model(val_states_t).flatten().to(torch.float32)
    mse_true = float(((pred - val_dist_t.float()) ** 2).mean().item())

    # MSE on walk-depth states (matches training metric).
    pred_w = model(walk_states_t).flatten().to(torch.float32)
    mse_walk = float(((pred_w - walk_depths_t.float()) ** 2).mean().item())

    # Top-K ordering accuracy: for each val state, predict its 24 neighbors,
    # rank, check if any optimal neighbor is in top-1/3.
    n_gen = len(puzzle.move_names)
    state_size = val_states_t.shape[1]
    gens_t = torch.tensor(
        np.stack([np.array(puzzle.generators[n], dtype=np.int64) for n in puzzle.move_names]),
        dtype=torch.int64, device=device,
    )

    n = val_states_t.shape[0]
    correct1 = 0
    correct3 = 0
    chunk = 256
    for i in range(0, n, chunk):
        s = val_states_t[i : i + chunk]              # (b, 120) int8
        b = s.shape[0]
        # neighbors: (b, n_gen, 120)
        neighbors = torch.gather(
            s.long().unsqueeze(1).expand(b, n_gen, state_size),
            2,
            gens_t.unsqueeze(0).expand(b, n_gen, state_size),
        )
        flat = neighbors.reshape(b * n_gen, state_size).to(val_states_t.dtype)
        scores = model(flat).flatten().to(torch.float32).reshape(b, n_gen)
        # Argmin per state
        ranked = scores.argsort(dim=1)  # ascending => lowest first
        for j in range(b):
            opt = set(val_opt[i + j])
            top1 = int(ranked[j, 0].item())
            top3 = {int(ranked[j, k].item()) for k in range(min(3, n_gen))}
            if top1 in opt:
                correct1 += 1
            if top3 & opt:
                correct3 += 1

    return {
        "mse_true": mse_true,
        "mse_walk": mse_walk,
        "top1_acc": correct1 / n,
        "top3_acc": correct3 / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    ap.add_argument("--bfs-table", type=Path,
                    default=PROJECT / "data" / "bfs_bytes_d6.pkl")
    ap.add_argument("--n-per-depth", type=int, default=500)
    ap.add_argument("--max-d", type=int, default=6)
    ap.add_argument("--n-walks", type=int, default=2000)
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    print(f"[val] loading BFS table {args.bfs_table.name}...", flush=True)
    t0 = time.time()
    bfs = BfsBytesTable.load(args.bfs_table)
    print(f"[val]   {len(bfs.table):,} states ({time.time() - t0:.0f}s)", flush=True)

    val_s, val_d, val_opt = build_validation_set(
        puzzle, bfs, args.n_per_depth, args.max_d, seed=args.seed,
    )
    val_s_t = torch.from_numpy(val_s).to(args.device, dtype=torch.int8)
    val_d_t = torch.from_numpy(val_d).to(args.device, dtype=torch.int32)

    # Walk-depth samples (fixed seed for reproducible per-checkpoint comparison).
    walk_s, walk_d = generate_walks_torch(
        puzzle, n_walks=args.n_walks, k_max=args.k_max,
        seed=args.seed + 1000, device=args.device, n_back=1,
    )
    print(f"[val] walk samples: {walk_s.shape[0]:,} states (k_max={args.k_max})", flush=True)

    print()
    print(f"  {'checkpoint':<60s} {'mse_true':>10s} {'mse_walk':>10s} "
          f"{'top1':>7s} {'top3':>7s} {'wall':>6s}")

    for ckpt in args.checkpoints:
        if not ckpt.exists():
            print(f"  {ckpt.name:<60s}  MISSING")
            continue
        t0 = time.time()
        try:
            r = evaluate_one(ckpt, val_s_t, val_d_t, val_opt, walk_s, walk_d,
                             puzzle, args.device, args.bf16)
        except Exception as e:
            print(f"  {ckpt.name:<60s}  ERROR: {e}")
            continue
        print(f"  {str(ckpt.relative_to(PROJECT.parent) if ckpt.is_relative_to(PROJECT.parent) else ckpt):<60s} "
              f"{r['mse_true']:>10.3f} {r['mse_walk']:>10.3f} "
              f"{r['top1_acc']:>7.3f} {r['top3_acc']:>7.3f} {time.time() - t0:>5.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
