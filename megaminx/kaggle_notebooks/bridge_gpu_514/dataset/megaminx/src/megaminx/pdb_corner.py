"""K-corner pattern database builder.

Tracks K labeled target corners through coord-space BFS from the solved configuration.
Each state in coord space is (positions, ori) where:
  - positions[k] = slot occupied by target corner k (in [0, 20), all distinct)
  - ori[k]       = orientation index of target corner k (in [0, 3))

Coord = lehmer_rank(positions) * 3^K + ori_rank

Total states: P(20, K) × 3^K
  K=4 →   116,280 ×   81 =       9,418,680  (~9 MB depth table)
  K=5 → 1,860,480 ×  243 =     452,096,640  (~452 MB)
  K=6 →27,907,200 ×  729 =  20,344,348,800  (way too big)
"""
from __future__ import annotations

import math
import pickle
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def precompute_move_tables(corner_perm: np.ndarray, corner_ori: np.ndarray, n_corners: int = 20):
    """Per-move slot transition + per-slot orientation delta.

    slot_target[m, slot_before] = slot_after that the corner at slot_before goes to under move m.
    ori_delta[m, slot_before] = orientation delta added to that corner's ori under move m.
    """
    n_moves = corner_perm.shape[0]
    slot_target = np.zeros((n_moves, n_corners), dtype=np.int64)
    ori_delta = np.zeros((n_moves, n_corners), dtype=np.int64)
    for m in range(n_moves):
        # σ_m = inverse permutation of corner_perm[m]
        sigma = np.zeros(n_corners, dtype=np.int64)
        for t in range(n_corners):
            j = int(corner_perm[m, t])
            sigma[j] = t
        slot_target[m] = sigma
        ori_delta[m] = corner_ori[m, sigma]
    return slot_target, ori_delta


def lehmer_rank_torch(positions: torch.Tensor, n: int) -> torch.Tensor:
    """Encode (B, K) distinct positions in [0, n) to (B,) lex rank.

    rank = sum_{k=0..K-1} r_k * (n-1-k)! / (n-K)!
    where r_k = positions[k] - count(j < k : positions[j] < positions[k])
    """
    B, K = positions.shape
    rank = torch.zeros(B, dtype=torch.int64, device=positions.device)
    coeff = 1
    for k in range(K - 1, -1, -1):
        rk = positions[:, k].clone().to(torch.int64)
        for j in range(k):
            rk -= (positions[:, j] < positions[:, k]).to(torch.int64)
        rank = rank + rk * coeff
        coeff *= (n - k)
    return rank


def encode_pdb_state(positions: torch.Tensor, ori: torch.Tensor, n: int) -> torch.Tensor:
    K = positions.shape[1]
    pos_rank = lehmer_rank_torch(positions, n)
    pow3 = 3 ** torch.arange(K, dtype=torch.int64, device=ori.device)
    ori_rank = (ori.to(torch.int64) * pow3).sum(dim=1)
    return pos_rank * (3 ** K) + ori_rank


def coord_max(K: int, n: int = 20) -> int:
    pos_max = 1
    for k in range(K):
        pos_max *= (n - k)
    return pos_max * (3 ** K)


def bfs_corner_pdb(
    target_corners: Sequence[int],
    corner_perm: np.ndarray,
    corner_ori: np.ndarray,
    device: str = "cuda",
    n_corners: int = 20,
    max_depth: int = 60,
):
    K = len(target_corners)
    target_corners = sorted(target_corners)
    slot_target, ori_delta = precompute_move_tables(corner_perm, corner_ori, n_corners)
    n_moves = slot_target.shape[0]
    cmax = coord_max(K, n_corners)

    print(f"K={K}  target_corners={target_corners}  coord_max={cmax:,}  device={device}", flush=True)

    slot_target_t = torch.from_numpy(slot_target).to(device).to(torch.int8)  # int8: values < 20
    ori_delta_t = torch.from_numpy(ori_delta).to(device).to(torch.int8)

    depth_table = torch.full((cmax,), 255, dtype=torch.uint8, device=device)

    init_pos = torch.tensor([target_corners], dtype=torch.int8, device=device)  # (1, K)
    init_ori = torch.zeros((1, K), dtype=torch.int8, device=device)
    init_coord = encode_pdb_state(init_pos, init_ori, n_corners)
    depth_table[init_coord] = 0

    frontier_pos = init_pos
    frontier_ori = init_ori
    depth = 0
    total_visited = 1
    t_start = time.time()

    # Per-chunk peak: chunk_size × n_moves × K × 1 byte (int8 pos/ori) + similar for coords (int64)
    # For chunk=50K, K=5, n_moves=24: 50K × 24 × 5 × 1 = 6 MB pos/ori; 50K × 24 × 8 = 9.6 MB coords
    # Plus dedupe sort buffers ~3x. So ~50 MB peak per chunk.
    chunk_size = 50_000

    while frontier_pos.shape[0] > 0 and depth < max_depth:
        Nf = frontier_pos.shape[0]

        # Each chunk marks visited inline; later chunks see depth_table updated, so
        # cross-chunk duplicates are naturally filtered. Accumulate new (pos, ori).
        new_pos_chunks = []
        new_ori_chunks = []
        n_new_total = 0

        for chunk_start in range(0, Nf, chunk_size):
            cs = chunk_start
            ce = min(cs + chunk_size, Nf)
            chunk_pos = frontier_pos[cs:ce]  # (Cf, K) int8
            chunk_ori = frontier_ori[cs:ce]  # (Cf, K) int8
            Cf = chunk_pos.shape[0]

            # Per-move expansion: (n_moves, Cf, K) int8
            chunk_pos_long = chunk_pos.to(torch.int64)
            sub_pos = slot_target_t[:, chunk_pos_long]  # (n_moves, Cf, K) int8
            sub_ori_delta = ori_delta_t[:, chunk_pos_long]  # (n_moves, Cf, K) int8
            sub_ori = (chunk_ori.unsqueeze(0) + sub_ori_delta) % 3  # (n_moves, Cf, K) int8

            sub_pos_flat = sub_pos.reshape(-1, K)  # (n_moves*Cf, K)
            sub_ori_flat = sub_ori.reshape(-1, K)

            sub_coords = encode_pdb_state(sub_pos_flat, sub_ori_flat, n_corners)  # (n_moves*Cf,)

            # Per-chunk dedupe via torch.unique
            unique_coords, inverse_idx = torch.unique(sub_coords, return_inverse=True)
            # First occurrence per unique: scatter min index
            B = sub_coords.shape[0]
            first_idx_per_unique = torch.full((unique_coords.shape[0],), B,
                                               dtype=torch.int64, device=device)
            first_idx_per_unique.scatter_reduce_(
                0, inverse_idx, torch.arange(B, dtype=torch.int64, device=device), reduce="amin"
            )

            # Unvisited filter
            unvis_mask = depth_table[unique_coords] == 255
            new_uniq_coords = unique_coords[unvis_mask]
            if new_uniq_coords.numel() == 0:
                continue
            new_first_idx = first_idx_per_unique[unvis_mask]

            new_pos = sub_pos_flat[new_first_idx]  # (n_new, K) int8
            new_ori = sub_ori_flat[new_first_idx]  # (n_new, K) int8

            depth_table[new_uniq_coords] = depth + 1

            new_pos_chunks.append(new_pos)
            new_ori_chunks.append(new_ori)
            n_new_total += new_uniq_coords.shape[0]

        if n_new_total == 0:
            depth += 1
            wall = time.time() - t_start
            print(f"  depth {depth:3d}: frontier 0  cumulative {total_visited:,}/{cmax:,}  "
                  f"{wall:7.1f}s", flush=True)
            break

        frontier_pos = torch.cat(new_pos_chunks)
        frontier_ori = torch.cat(new_ori_chunks)
        n_new = frontier_pos.shape[0]
        depth += 1
        total_visited += n_new
        wall = time.time() - t_start
        print(
            f"  depth {depth:3d}: frontier {n_new:>11,}  cumulative {total_visited:>11,}/{cmax:,} "
            f"({100*total_visited/cmax:.1f}%)  {wall:7.1f}s",
            flush=True,
        )
        if n_new == 0:
            break

    return depth_table.cpu().numpy(), total_visited


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--target-corners", type=str, default=None,
                    help="comma-separated corner IDs. Default: evenly-spaced selection.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.target_corners:
        target_corners = [int(x) for x in args.target_corners.split(",")]
    else:
        # Evenly spaced across [0, 20)
        target_corners = [int(round(i * 20 / args.K)) for i in range(args.K)]
    K = len(target_corners)
    assert K == args.K

    PROJECT = Path(__file__).resolve().parents[2]
    with open(PROJECT / "data" / "corner_tables.pkl", "rb") as f:
        tables = pickle.load(f)
    corner_perm = tables["corner_perm"]
    corner_ori = tables["corner_ori"]

    print(f"building corner PDB: K={K}, target={target_corners}", flush=True)
    depth_table, total = bfs_corner_pdb(
        target_corners, corner_perm, corner_ori, device=args.device,
    )

    cmax = coord_max(K)
    coverage = total / cmax
    print(f"\nDone. Visited {total:,}/{cmax:,} ({100*coverage:.2f}%)", flush=True)
    print(f"Depth distribution:")
    finite = depth_table[depth_table != 255]
    if len(finite) > 0:
        print(f"  min={finite.min()}  max={finite.max()}  mean={finite.mean():.2f}")
        for d in range(int(finite.max()) + 1):
            n_at_d = int((finite == d).sum())
            print(f"    depth {d:2d}: {n_at_d:>11,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({
            "K": K,
            "target_corners": sorted(target_corners),
            "depth_table": depth_table,  # uint8 array, 255 = unvisited
        }, f)
    print(f"\nwrote {args.out} ({depth_table.nbytes / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
