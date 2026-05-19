"""T3.3 — Build BFS-d6 oracle Q-target dataset.

For every state s in the d<=5 shell of BFS-d6, compute the EXACT Q-targets:
    Q*(s, a) = 1 + true_d(apply(s, a))   for all 24 actions a

Children at d<=6 are in the BFS table (we have exact distances). For parents
at d=5, all children are at d in {4, 5, 6} — all in the table. For parents
at d=6, some children at d=7 are outside the table; we skip d=6 parents to
keep labels clean.

Output: `megaminx/data/oracle_q_d5.pt` with:
    {"states":   (N, 120) int8,
     "q_targets": (N, 24) int8,
     "depths":    (N,) int8     # d(s) for each state — for stratified eval}

Run: .venv/Scripts/python.exe megaminx/scripts/29_build_oracle_q_dataset.py
ETA: ~5-10 min on a fast machine.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "src"))

from megaminx.bfs_bytes import BfsBytesTable
from megaminx.puzzle import Megaminx


def main() -> int:
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    move_names = list(puzzle.move_names)
    gens = np.stack([np.array(puzzle.generators[n], dtype=np.int64) for n in move_names])
    n_gen = gens.shape[0]
    print(f"loaded {n_gen} generators")

    print("loading BFS-d6 table ...")
    t0 = time.time()
    bfs = BfsBytesTable.load(PROJECT / "data" / "bfs_bytes_d6.pkl")
    table = bfs.table
    print(f"  {len(table):,} entries (max_depth={bfs.max_depth}) in {time.time()-t0:.1f}s")

    # Bucket states by depth (= len(path_bytes))
    print("\nbucketing states by depth ...")
    by_depth: dict[int, list[bytes]] = {}
    for s_bytes, path_bytes in table.items():
        d = len(path_bytes)
        by_depth.setdefault(d, []).append(s_bytes)
    for d in sorted(by_depth):
        print(f"  d={d}: {len(by_depth[d]):,} states")

    # Filter to d<=5: those have all children in the d<=6 shell (so q-targets are exact)
    target_depths = [d for d in by_depth if d <= 5]
    parent_states_bytes: list[bytes] = []
    parent_depths: list[int] = []
    for d in target_depths:
        parent_states_bytes.extend(by_depth[d])
        parent_depths.extend([d] * len(by_depth[d]))
    n = len(parent_states_bytes)
    print(f"\ntotal d<=5 parent states: {n:,}")

    print("\ncomputing oracle Q-targets ...")
    t0 = time.time()
    states_arr = np.zeros((n, 120), dtype=np.int8)
    q_targets = np.full((n, n_gen), -1, dtype=np.int8)  # -1 = invalid sentinel

    # Vectorize the inner loop: for each parent state, apply all 24 generators.
    # The lookup itself is per-(parent, action) which is 33M dict lookups on bytes keys.
    n_invalid = 0
    n_solved_children = 0
    for i, (s_bytes, d) in enumerate(zip(parent_states_bytes, parent_depths)):
        if i % 200_000 == 0 and i > 0:
            print(f"  {i:>8,} / {n:,} ({100*i/n:.1f}%) - elapsed {time.time()-t0:.1f}s", flush=True)
        s = np.frombuffer(s_bytes, dtype=np.uint8)
        states_arr[i] = s.view(np.int8)
        # Compute all 24 children: shape (24, 120)
        children = s[gens]  # (24, 120) — uint8 fancy indexing
        for a in range(n_gen):
            child_bytes = children[a].tobytes()
            child_path = table.get(child_bytes)
            if child_path is None:
                n_invalid += 1
                continue
            q_targets[i, a] = 1 + len(child_path)
            if len(child_path) == 0:
                n_solved_children += 1

    print(f"\ndataset built: {n:,} states × {n_gen} actions in {time.time()-t0:.1f}s")
    print(f"  invalid Q-targets (child not in table): {n_invalid:,}")
    print(f"  solved-child Q-targets (=1): {n_solved_children:,}")

    # Sanity: distribution of Q-targets
    valid_mask = q_targets >= 0
    valid_q = q_targets[valid_mask]
    print(f"\nQ-target distribution (valid only):")
    from collections import Counter
    counter = Counter(int(q) for q in valid_q.flat[: 100_000])  # sample
    for q in sorted(counter):
        print(f"  Q={q}: {counter[q]:,}")
    print(f"  Q-target stats (valid): min={valid_q.min()}, max={valid_q.max()}, mean={valid_q.mean():.2f}")

    # Save as torch tensor file
    out = {
        "states": torch.from_numpy(states_arr),
        "q_targets": torch.from_numpy(q_targets),
        "depths": torch.from_numpy(np.array(parent_depths, dtype=np.int8)),
    }
    out_path = PROJECT / "data" / "oracle_q_d5.pt"
    torch.save(out, out_path)
    print(f"\nwrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
