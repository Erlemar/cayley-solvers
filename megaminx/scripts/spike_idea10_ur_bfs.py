"""Day-1 spike for Idea 10: BFS over the <U, R> subgroup of megaminx.

Determines |<U, R>| by counting states reachable from solved using only U/U'/R/R'.
Caps the search to keep this a quick (~10-30 min) feasibility check.

Decision matrix:
  - states at d<=12 stays under 10^6  -> bytes-keyed BFS table at depth ~24 is feasible
  - states explode past 10^8 in a few layers  -> needs coordinate-encoded PDB
  - states > 10^10 anywhere  -> abandon Idea 10, defer to T2.4
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from megaminx.puzzle import Megaminx


def main() -> int:
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    move_names = list(puzzle.move_names)
    # Pick the U,U',R,R' indices.
    target_names = ["U", "-U", "R", "-R"]
    gen_idxs = [move_names.index(n) for n in target_names]
    gens = np.stack([np.array(puzzle.generators[move_names[i]], dtype=np.int8)
                     for i in gen_idxs], axis=0)
    print(f"using {len(gen_idxs)} generators: {target_names}")
    print(f"state size: {len(puzzle.solved_state)}")

    state_size = len(puzzle.solved_state)
    identity = np.arange(state_size, dtype=np.int8)
    identity_key = identity.tobytes()

    # BFS layer by layer
    seen: set[bytes] = {identity_key}
    frontier: list[np.ndarray] = [identity]

    MAX_TOTAL = 100_000_000  # 100M cap (would be ~12GB at 120B per key; abort earlier)
    MAX_DEPTH = 20
    MAX_WALL_S = 1800  # 30 min cap

    t_start = time.time()
    print(f"\n{'depth':<6} {'new':<14} {'total':<14} {'elapsed_s':<10}")
    print(f"{'0':<6} {'1':<14} {'1':<14} {'0.0':<10}", flush=True)

    for d in range(1, MAX_DEPTH + 1):
        t0 = time.time()
        next_frontier_keys: set[bytes] = set()
        for state in frontier:
            for g in gens:
                child = state[g]
                k = child.tobytes()
                if k not in seen:
                    seen.add(k)
                    next_frontier_keys.add(k)
        new_count = len(next_frontier_keys)
        if new_count == 0:
            print(f"\nNo new states at depth {d}; subgroup fully enumerated.")
            print(f"|<U, R>| = {len(seen):,}")
            break
        # Materialize next frontier as np arrays for the next pass.
        next_frontier = [np.frombuffer(k, dtype=np.int8).copy() for k in next_frontier_keys]
        frontier = next_frontier
        elapsed = time.time() - t_start
        print(f"{d:<6} {new_count:<14,} {len(seen):<14,} {elapsed:<10.1f}", flush=True)
        if len(seen) > MAX_TOTAL:
            print(f"\nABORT: exceeded {MAX_TOTAL:,} states cap.")
            break
        if elapsed > MAX_WALL_S:
            print(f"\nABORT: exceeded {MAX_WALL_S}s wall cap.")
            break

    print(f"\nfinal: depth={d}, total_states={len(seen):,}, elapsed={time.time()-t_start:.1f}s")
    print()
    if len(seen) <= 1_000_000:
        print(f"DECISION: |<U,R>| ~= {len(seen):,}, FEASIBLE for bytes-keyed BFS table.")
    elif len(seen) <= 10_000_000:
        print(f"DECISION: |<U,R>| ~= {len(seen):,}, BORDERLINE for bytes-keyed (1.2 GB).")
    elif len(seen) <= 100_000_000:
        print(f"DECISION: |<U,R>| ~= {len(seen):,}, requires coordinate-encoded PDB (12 GB bytes-keyed).")
    else:
        print(f"DECISION: |<U,R>| > {len(seen):,}, infeasible at our scale; defer to T2.4 or abandon.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
