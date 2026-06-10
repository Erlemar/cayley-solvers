"""Multi-PDB heuristic: max over disjoint K=5 corner PDBs.

For state s: h(s) = max over PDBs p of pdb_p.lookup(s)
Each pdb_p tracks K=5 disjoint corners → admissible lower bound on full distance.
Max over PDBs preserves admissibility.

For 4 disjoint PDBs covering all 20 corners, max-of-4 is typically much tighter than
a single PDB because random scrambles often have at least one corner subset that's
maximally hard.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from megaminx.pdb_corner import encode_pdb_state


class CornerPDBHeuristic:
    """Bundle multiple corner PDBs and compute max heuristic for batches of states.

    Loads each PDB's depth_table to GPU. Pre-computes per-PDB target_corner indices
    and home stickers. lookup(states_t) returns (B,) int8 max-distance estimates.
    """

    def __init__(self, pdb_paths: list[Path], corner_tables_path: Path, device: str = "cuda",
                 n_corners: int = 20):
        self.device = device
        self.n_corners = n_corners

        with open(corner_tables_path, "rb") as f:
            ctables = pickle.load(f)
        self.corner_slots = ctables["corner_slots"]  # list of n_corners tuples

        # Slots tensor (n_corners, 3) int64 — sticker positions per slot
        self.slots_t = torch.tensor(
            [list(s) for s in self.corner_slots], dtype=torch.int64, device=device,
        )  # (n_corners, 3)

        # Home corner lookup (sorted 3-tuple → corner_id)
        self.home_lookup = {tuple(s): cid for cid, s in enumerate(self.corner_slots)}

        # Convert home_lookup to a hash table on device for fast vectorized lookup.
        # We use sorted-tuple → corner_id. To vectorize, build a 1D table indexed by
        # encoded sorted-tuple. With sticker_size=120 and 3 stickers, encoded =
        # s0*120^2 + s1*120 + s2 (s0 < s1 < s2).
        # Total: 120*120*120 = 1.7M entries, 1 byte each = 1.7 MB. Fine on GPU.
        sticker_size = 120
        homedict_t = torch.full((sticker_size**3,), -1, dtype=torch.int8, device=device)
        for slot, cid in self.home_lookup.items():
            s0, s1, s2 = slot  # already sorted
            idx = s0 * sticker_size**2 + s1 * sticker_size + s2
            homedict_t[idx] = cid
        self.homedict_t = homedict_t
        self.sticker_size = sticker_size

        # Load each PDB
        self.pdbs = []
        for path in pdb_paths:
            with open(path, "rb") as f:
                pdb = pickle.load(f)
            target_corners = pdb["target_corners"]
            K = pdb["K"]
            depth_table_np = pdb["depth_table"]
            depth_table_t = torch.from_numpy(depth_table_np).to(device)

            target_t = torch.tensor(target_corners, dtype=torch.int64, device=device)
            # For ori computation: home stickers per target corner (K, 3)
            target_homes = torch.tensor(
                [list(self.corner_slots[t]) for t in target_corners],
                dtype=torch.int64, device=device,
            )

            self.pdbs.append({
                "K": K,
                "target_corners": target_corners,
                "target_t": target_t,
                "target_homes": target_homes,
                "depth_table": depth_table_t,
            })

        print(f"loaded {len(self.pdbs)} PDBs covering corners "
              f"{sorted(set(c for p in self.pdbs for c in p['target_corners']))}", flush=True)

    def states_to_corner_state(self, states_t: torch.Tensor):
        """Convert (B, 120) int8 full states → (B, n_corners) corner_at_slot, (B, n_corners, 3) values_at_slot.

        For each state b, slot s: corner_at_slot[b, s] = corner ID at slot s in state b.
        values_at_slot[b, s] = (state[slots[s][0]], state[slots[s][1]], state[slots[s][2]]) sorted.
        """
        B = states_t.shape[0]
        # state_at_slots[b, s, j] = states_t[b, slots[s, j]]
        state_at_slots = states_t[:, self.slots_t]  # (B, n_corners, 3) int8
        sorted_vals, _ = torch.sort(state_at_slots, dim=-1)  # (B, n_corners, 3)

        # Vectorized corner-id lookup via flat hash table
        sv = sorted_vals.to(torch.int64)
        encoded = sv[..., 0] * self.sticker_size**2 + sv[..., 1] * self.sticker_size + sv[..., 2]
        # encoded: (B, n_corners) int64
        corner_at_slot = self.homedict_t[encoded]  # (B, n_corners) int8
        return corner_at_slot, sorted_vals

    def states_to_pdb_state(self, states_t: torch.Tensor, pdb):
        """Convert (B, 120) states → (B, K) positions, (B, K) ori for given PDB."""
        B = states_t.shape[0]
        K = pdb["K"]
        n_corners = self.n_corners

        corner_at_slot, _ = self.states_to_corner_state(states_t)  # (B, n_corners) int8

        # Inverse: slot_of_corner[b, c] = s where corner_at_slot[b, s] = c
        slot_of_corner = torch.zeros((B, n_corners), dtype=torch.int64, device=self.device)
        rows = torch.arange(B, device=self.device).unsqueeze(1).expand(B, n_corners)
        slot_of_corner[rows, corner_at_slot.to(torch.int64)] = (
            torch.arange(n_corners, device=self.device).unsqueeze(0).expand(B, n_corners)
        )

        # positions[b, k] = slot_of_corner[b, target_t[k]]
        positions = slot_of_corner[:, pdb["target_t"]]  # (B, K) int64

        # For ori: at slot positions[b, k], the values at the canonical 3 sticker positions
        # are some cyclic rotation of target_homes[k]. Find r ∈ {0, 1, 2}.
        # state_at_target[b, k, j] = states_t[b, slots[positions[b, k], j]]
        slot_positions = self.slots_t[positions]  # (B, K, 3) int64
        rows_b = torch.arange(B, device=self.device).unsqueeze(1).unsqueeze(2).expand(B, K, 3)
        state_at_target = states_t[rows_b, slot_positions]  # (B, K, 3) int8

        target_homes = pdb["target_homes"]  # (K, 3) int64

        ori = torch.full((B, K), -1, dtype=torch.int8, device=self.device)
        for r in range(3):
            rotated = torch.stack([
                target_homes[:, (r + 0) % 3],
                target_homes[:, (r + 1) % 3],
                target_homes[:, (r + 2) % 3],
            ], dim=-1)  # (K, 3)
            match = (state_at_target.to(torch.int64) == rotated.unsqueeze(0)).all(dim=-1)
            ori = torch.where(match, torch.tensor(r, dtype=torch.int8, device=self.device), ori)

        return positions.to(torch.int8), ori

    def lookup(self, states_t: torch.Tensor) -> torch.Tensor:
        """Return (B,) int max distance estimate."""
        B = states_t.shape[0]
        max_h = torch.zeros(B, dtype=torch.int8, device=self.device)
        for pdb in self.pdbs:
            positions, ori = self.states_to_pdb_state(states_t, pdb)
            coords = encode_pdb_state(positions, ori, self.n_corners)
            vals = pdb["depth_table"][coords].to(torch.int8)
            max_h = torch.maximum(max_h, vals)
        return max_h


def main():
    """Smoke-test heuristic on BFS states + random walks."""
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-paths", type=str, default=None,
                    help="comma-separated paths. Default: all 4 K5 PDBs in megaminx/data/")
    ap.add_argument("--n-bfs-samples", type=int, default=10_000)
    ap.add_argument("--n-walk-samples", type=int, default=10_000)
    ap.add_argument("--walk-len", type=int, default=50)
    args = ap.parse_args()

    if args.pdb_paths:
        pdb_paths = [Path(p) for p in args.pdb_paths.split(",")]
    else:
        pdb_paths = [
            PROJECT / "data" / f"pdb_corner_K5{suffix}.pkl"
            for suffix in ["", "_p1", "_p2", "_p3"]
        ]
    for p in pdb_paths:
        assert p.exists(), p

    from megaminx.bfs_bytes import BfsBytesTable
    from megaminx.puzzle import Megaminx

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    state_size = len(puzzle.solved_state)

    h = CornerPDBHeuristic(pdb_paths, PROJECT / "data" / "corner_tables.pkl", device="cuda")

    # Test 1: BFS-d6 sampled states
    print(f"\n=== Test 1: BFS-d6 admissibility ({args.n_bfs_samples:,} samples) ===")
    bfs = BfsBytesTable.load(PROJECT / "data" / "bfs_bytes_d6.pkl")
    items = list(bfs.table.items())[:args.n_bfs_samples]
    states = np.empty((len(items), state_size), dtype=np.int8)
    depths = np.empty(len(items), dtype=np.int32)
    for i, (k, p) in enumerate(items):
        states[i] = np.frombuffer(k, dtype=np.int8)
        depths[i] = len(p)
    states_t = torch.from_numpy(states).to("cuda")

    t0 = time.time()
    pdb_vals = h.lookup(states_t).cpu().numpy().astype(np.int32)
    wall = time.time() - t0
    print(f"  lookup wall: {wall:.2f}s ({len(items)/wall:,.0f} states/s)")
    n_violations = int((pdb_vals > depths).sum())
    print(f"  admissibility violations (max_pdb > full_depth): {n_violations}/{len(items)}")
    print(f"  pdb min/max/mean: {pdb_vals.min()}/{pdb_vals.max()}/{pdb_vals.mean():.2f}")
    print(f"  full depth min/max/mean: {depths.min()}/{depths.max()}/{depths.mean():.2f}")

    # Per-depth breakdown
    print(f"\n  pdb distribution by full depth:")
    for d in range(int(depths.max()) + 1):
        mask = depths == d
        if mask.sum() > 0:
            sub = pdb_vals[mask]
            print(f"    depth {d}: n={int(mask.sum()):>9,}  pdb min/max/mean={sub.min()}/{sub.max()}/{sub.mean():.2f}")

    # Test 2: deeper random walks (no known optimal distance, but heuristic distribution)
    print(f"\n=== Test 2: random walk states (depth ~{args.walk_len}) ===")
    rng = np.random.default_rng(0)
    n_gen = len(puzzle.move_names)
    gens = np.stack([np.array(puzzle.generators[n], dtype=np.int8) for n in puzzle.move_names])
    walk_states = np.zeros((args.n_walk_samples, state_size), dtype=np.int8)
    base = np.array(puzzle.solved_state, dtype=np.int8)
    for i in range(args.n_walk_samples):
        cur = base.copy()
        for _ in range(args.walk_len):
            gi = rng.integers(0, n_gen)
            cur = cur[gens[gi]]
        walk_states[i] = cur
    walk_states_t = torch.from_numpy(walk_states).to("cuda")

    t0 = time.time()
    walk_pdb_vals = h.lookup(walk_states_t).cpu().numpy().astype(np.int32)
    wall = time.time() - t0
    print(f"  lookup wall: {wall:.2f}s")
    print(f"  pdb min/max/mean: {walk_pdb_vals.min()}/{walk_pdb_vals.max()}/{walk_pdb_vals.mean():.2f}")
    # Histogram
    print(f"  pdb distribution:")
    for d in range(15):
        n = int((walk_pdb_vals == d).sum())
        if n > 0:
            print(f"    pdb={d}: n={n:>6}  ({100*n/args.n_walk_samples:.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
