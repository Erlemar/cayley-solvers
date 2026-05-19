"""K-edge pattern database builder.

Edges have 2 stickers each; orientation is binary (flipped or not).

Coord: pos_rank(positions, n=30) * 2^K + ori_rank(ori, K).

Total states (K=5): P(30, 5) × 2^5 = 17,100,720 × 32 = 547,223,040 (~547 MB depth table)
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from megaminx.pdb_corner import lehmer_rank_torch, precompute_move_tables


def encode_edge_pdb_state(positions: torch.Tensor, ori: torch.Tensor, n: int = 30) -> torch.Tensor:
    K = positions.shape[1]
    pos_rank = lehmer_rank_torch(positions, n)
    pow2 = 2 ** torch.arange(K, dtype=torch.int64, device=ori.device)
    ori_rank = (ori.to(torch.int64) * pow2).sum(dim=1)
    return pos_rank * (2 ** K) + ori_rank


def coord_max_edge(K: int, n: int = 30) -> int:
    pos_max = 1
    for k in range(K):
        pos_max *= (n - k)
    return pos_max * (2 ** K)


def bfs_edge_pdb(
    target_edges: Sequence[int],
    edge_perm: np.ndarray,
    edge_ori: np.ndarray,
    device: str = "cuda",
    n_edges: int = 30,
    max_depth: int = 60,
):
    K = len(target_edges)
    target_edges = sorted(target_edges)
    slot_target, ori_delta = precompute_move_tables(edge_perm, edge_ori, n_corners=n_edges)
    n_moves = slot_target.shape[0]
    cmax = coord_max_edge(K, n_edges)

    print(f"K={K}  target_edges={target_edges}  coord_max={cmax:,}  device={device}", flush=True)

    slot_target_t = torch.from_numpy(slot_target).to(device).to(torch.int8)
    ori_delta_t = torch.from_numpy(ori_delta).to(device).to(torch.int8)

    depth_table = torch.full((cmax,), 255, dtype=torch.uint8, device=device)

    init_pos = torch.tensor([target_edges], dtype=torch.int8, device=device)
    init_ori = torch.zeros((1, K), dtype=torch.int8, device=device)
    init_coord = encode_edge_pdb_state(init_pos, init_ori, n_edges)
    depth_table[init_coord] = 0

    frontier_pos = init_pos
    frontier_ori = init_ori
    depth = 0
    total_visited = 1
    t_start = time.time()

    chunk_size = 50_000

    while frontier_pos.shape[0] > 0 and depth < max_depth:
        Nf = frontier_pos.shape[0]
        new_pos_chunks = []
        new_ori_chunks = []
        n_new_total = 0

        for chunk_start in range(0, Nf, chunk_size):
            cs = chunk_start
            ce = min(cs + chunk_size, Nf)
            chunk_pos = frontier_pos[cs:ce]
            chunk_ori = frontier_ori[cs:ce]
            Cf = chunk_pos.shape[0]

            chunk_pos_long = chunk_pos.to(torch.int64)
            sub_pos = slot_target_t[:, chunk_pos_long]
            sub_ori_delta = ori_delta_t[:, chunk_pos_long]
            sub_ori = (chunk_ori.unsqueeze(0) + sub_ori_delta) % 2  # mod 2 for edges

            sub_pos_flat = sub_pos.reshape(-1, K)
            sub_ori_flat = sub_ori.reshape(-1, K)

            sub_coords = encode_edge_pdb_state(sub_pos_flat, sub_ori_flat, n_edges)

            unique_coords, inverse_idx = torch.unique(sub_coords, return_inverse=True)
            B = sub_coords.shape[0]
            first_idx_per_unique = torch.full((unique_coords.shape[0],), B,
                                               dtype=torch.int64, device=device)
            first_idx_per_unique.scatter_reduce_(
                0, inverse_idx, torch.arange(B, dtype=torch.int64, device=device), reduce="amin"
            )

            unvis_mask = depth_table[unique_coords] == 255
            new_uniq_coords = unique_coords[unvis_mask]
            if new_uniq_coords.numel() == 0:
                continue
            new_first_idx = first_idx_per_unique[unvis_mask]

            new_pos = sub_pos_flat[new_first_idx]
            new_ori = sub_ori_flat[new_first_idx]

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
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--target-edges", type=str, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.target_edges:
        target_edges = [int(x) for x in args.target_edges.split(",")]
    else:
        target_edges = [int(round(i * 30 / args.K)) for i in range(args.K)]
    K = len(target_edges)
    assert K == args.K

    PROJECT = Path(__file__).resolve().parents[2]
    with open(PROJECT / "data" / "edge_tables.pkl", "rb") as f:
        tables = pickle.load(f)
    edge_perm = tables["edge_perm"]
    edge_ori = tables["edge_ori"]

    print(f"building edge PDB: K={K}, target={target_edges}", flush=True)
    depth_table, total = bfs_edge_pdb(
        target_edges, edge_perm, edge_ori, device=args.device,
    )

    cmax = coord_max_edge(K)
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
            "target_edges": sorted(target_edges),
            "depth_table": depth_table,
        }, f)
    print(f"\nwrote {args.out} ({depth_table.nbytes / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
