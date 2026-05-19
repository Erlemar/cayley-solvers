"""Verify K=5 corner PDB against known BFS distances.

For each state s in BFS-d6 (full puzzle states with known distance d ∈ [0, 6]),
compute pdb_lookup(s). Check admissibility: pdb_value ≤ d.

Also samples the PDB distribution across states.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from megaminx.bfs_bytes import BfsBytesTable
from megaminx.pdb_corner import (
    encode_pdb_state,
    precompute_move_tables,
)
from megaminx.puzzle import Megaminx


def state_to_corner_pdb_state(states_t: torch.Tensor, corner_slots, target_corners,
                               n_corners=20):
    """Convert (B, 120) full states to (B, K) positions and (B, K) ori for target corners.

    states_t: int8 (B, 120) on device.
    corner_slots: list of n_corners tuples, each tuple of 3 sticker positions (sorted).
    target_corners: K-list of corner IDs to track.

    Returns (positions, ori): both (B, K) int8 on device.
    """
    K = len(target_corners)
    B = states_t.shape[0]
    device = states_t.device

    # For each state, for each target corner T:
    #   1. Find which slot has corner T's home stickers (in any cyclic order).
    #   2. Compute orientation of that occurrence.

    # Precompute home sticker sets for each corner (sorted tuples → corner_id)
    home_sticker_sets = [tuple(slot) for slot in corner_slots]  # 20 corners, each a sorted 3-tuple

    # For each state and each slot, read the 3 sticker values and identify which
    # corner is there.
    # slot s in state has values (state[slot[0]], state[slot[1]], state[slot[2]]) (sorted positions).
    # The (sorted) values reveal the home corner ID via lookup against home_sticker_sets.

    # Build lookup table: tuple(sorted 3 sticker values) → corner_id
    home_lookup = {tuple(slot): cid for cid, slot in enumerate(corner_slots)}

    # For each state, for each slot:
    slots_arr = torch.tensor([list(slot) for slot in corner_slots],
                              dtype=torch.int64, device=device)  # (n_corners, 3)
    state_at_slots = states_t[:, slots_arr]  # (B, n_corners, 3) int8 — values at slot positions

    # For target corners only, find slot containing it
    # Approach: for each state, for each slot, identify which corner is there (sort+lookup).
    # Then build "slot_of_corner" array (B, n_corners), and select target_corners.

    # Sort the 3 values per slot to get canonical key
    sorted_vals, _ = torch.sort(state_at_slots, dim=-1)  # (B, n_corners, 3)

    # Convert to corner_id via lookup. For GPU, use a hash-based or direct numpy fallback.
    sorted_vals_cpu = sorted_vals.cpu().numpy().astype(np.int64)  # (B, n_corners, 3)
    corner_at_slot = np.zeros((B, n_corners), dtype=np.int64)  # (B, n_corners)
    # Use vectorized lookup via dict
    for b in range(B):
        for s in range(n_corners):
            key = tuple(sorted_vals_cpu[b, s])
            corner_at_slot[b, s] = home_lookup[key]
    corner_at_slot_t = torch.from_numpy(corner_at_slot).to(device)  # (B, n_corners)

    # Inverse: slot_of_corner[b, c] = slot s such that corner_at_slot[b, s] = c
    slot_of_corner = torch.zeros((B, n_corners), dtype=torch.int64, device=device)
    rows = torch.arange(B, device=device).unsqueeze(1).expand(B, n_corners)
    slot_of_corner[rows, corner_at_slot_t] = torch.arange(n_corners, device=device).unsqueeze(0).expand(B, n_corners)

    # positions[b, k] = slot_of_corner[b, target_corners[k]]
    target_t = torch.tensor(target_corners, dtype=torch.int64, device=device)
    positions = slot_of_corner[:, target_t]  # (B, K) int64

    # For ori: for each target corner T at slot S, the slot S has values (state[slot[0]], state[slot[1]], state[slot[2]])
    # in CANONICAL position order. The home stickers for T sorted are home_sticker_sets[T].
    # Cyclic offset r: (state[slot[0]], state[slot[1]], state[slot[2]]) is rotation of home_sticker_sets[T] by r.
    target_homes = torch.tensor(
        [list(corner_slots[t]) for t in target_corners], dtype=torch.int64, device=device
    )  # (K, 3)

    # state_at_target_slot[b, k, j] = states_t[b, slot_positions_for_targets_slot[b,k][j]]
    # First: positions[b, k] → slot index. Get slot positions: slots_arr[positions[b,k]]
    target_slot_positions = slots_arr[positions]  # (B, K, 3)
    state_at_target = states_t[
        torch.arange(B, device=device).unsqueeze(1).unsqueeze(2).expand(B, K, 3),
        target_slot_positions,
    ]  # (B, K, 3) int8

    # For each (b, k), find r ∈ {0, 1, 2} such that state_at_target[b, k] = rotate(home, r)
    # home = target_homes[k], rotate(home, r) = (home[(r+0)%3], home[(r+1)%3], home[(r+2)%3])
    ori = torch.full((B, K), -1, dtype=torch.int8, device=device)
    for r in range(3):
        rotated = torch.stack([
            target_homes[:, (r + 0) % 3],
            target_homes[:, (r + 1) % 3],
            target_homes[:, (r + 2) % 3],
        ], dim=-1)  # (K, 3)
        match = (state_at_target == rotated.unsqueeze(0)).all(dim=-1)  # (B, K)
        ori = torch.where(match, torch.tensor(r, dtype=torch.int8, device=device), ori)

    if (ori == -1).any():
        print(f"WARNING: some ori values not matched, state malformed?", flush=True)
    return positions.to(torch.int8), ori


def main():
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")

    # Load corner tables
    with open(PROJECT / "data" / "corner_tables.pkl", "rb") as f:
        ctables = pickle.load(f)
    corner_slots = ctables["corner_slots"]

    # Load PDB
    with open(PROJECT / "data" / "pdb_corner_K5.pkl", "rb") as f:
        pdb = pickle.load(f)
    target_corners = pdb["target_corners"]
    K = pdb["K"]
    depth_table_np = pdb["depth_table"]
    n_corners = 20

    print(f"PDB: K={K}, target={target_corners}, table size {depth_table_np.size:,}", flush=True)

    # Load BFS table (small one — d=4 or d=6)
    bfs_path = PROJECT / "data" / "bfs_bytes_d4.pkl"
    if not bfs_path.exists():
        bfs_path = PROJECT / "data" / "bfs_bytes_d6.pkl"
    print(f"loading {bfs_path}...", flush=True)
    bfs_table = BfsBytesTable.load(bfs_path)
    print(f"loaded {len(bfs_table.table):,} states", flush=True)

    # Build (state, depth) arrays from bfs table
    state_size = 120
    N = min(len(bfs_table.table), 100_000)
    items = list(bfs_table.table.items())[:N]
    states = np.empty((N, state_size), dtype=np.int8)
    depths = np.empty(N, dtype=np.int32)
    for i, (key_bytes, path_bytes) in enumerate(items):
        states[i] = np.frombuffer(key_bytes, dtype=np.int8)
        depths[i] = len(path_bytes)
    print(f"sampled {N:,} states", flush=True)

    device = "cuda"
    states_t = torch.from_numpy(states).to(device)
    depth_table_t = torch.from_numpy(depth_table_np).to(device)

    # Convert to PDB state
    print("converting states to PDB state...", flush=True)
    t0 = time.time()
    positions, ori = state_to_corner_pdb_state(states_t, corner_slots, target_corners,
                                                n_corners=n_corners)
    print(f"converted in {time.time()-t0:.1f}s", flush=True)

    # PDB lookup
    coords = encode_pdb_state(positions, ori, n_corners)
    pdb_vals = depth_table_t[coords].cpu().numpy().astype(np.int32)
    print(f"PDB lookups: min={pdb_vals.min()}, max={pdb_vals.max()}, mean={pdb_vals.mean():.2f}",
          flush=True)

    # Admissibility check: pdb_val ≤ depth
    violations = pdb_vals > depths
    n_violations = int(violations.sum())
    print(f"admissibility violations: {n_violations}/{N}", flush=True)
    if n_violations > 0:
        # Show a few examples
        idx = np.where(violations)[0][:5]
        for i in idx:
            print(f"  state {i}: depth={depths[i]} but pdb={pdb_vals[i]} (violation)")

    # Distribution by depth
    print(f"\npdb_val by full-puzzle depth:")
    for d in range(int(depths.max()) + 1):
        mask = depths == d
        if mask.sum() > 0:
            sub = pdb_vals[mask]
            print(f"  depth {d}: n={int(mask.sum()):>9,}  pdb min={sub.min()} max={sub.max()} "
                  f"mean={sub.mean():.2f}")

    return 0 if n_violations == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
