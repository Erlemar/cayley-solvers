"""Build a fast-load tensor dataset of (state, exact_distance) pairs from
the BFS-d6 table. For Option B training (BFS-d6 exact target mixin).

Output: (states, distances) tensors saved as `data/bfs_d6_train.pt`.

  states     : (N, 120) int8 — N = 19,352,405 unique shell states (d=0..6)
  distances  : (N,)     int8 — values 0..6

Memory: ~2.3 GB on disk + 1 brief peak in RAM during build (~3 GB).
Loading at train time: ~10s via torch.load on local 4090.

Why pre-build: random-sample shell states each epoch is fast on a tensor
(`states[idx]`); doing it on a 19M-entry dict-of-bytes is slow.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
# pickle.load needs the `megaminx` package importable to resolve BfsBytesTable
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

BFS_PATH = PROJECT / "data" / "bfs_bytes_d6.pkl"
OUT_PATH = PROJECT / "data" / "bfs_d6_train.pt"


def main() -> int:
    if not BFS_PATH.exists():
        print(f"ERROR: {BFS_PATH} not found", file=sys.stderr)
        return 2

    print(f"[bfs6] loading {BFS_PATH.name} ({BFS_PATH.stat().st_size / 1024**3:.2f} GB)...",
          flush=True)
    t0 = time.time()
    with open(BFS_PATH, "rb") as f:
        bfs = pickle.load(f)
    print(f"[bfs6]   {len(bfs.table):,} entries loaded ({time.time() - t0:.1f}s)", flush=True)

    n = len(bfs.table)
    print(f"[bfs6] preallocating tensors ({n:,} states × 120 bytes = {n * 120 / 1024**3:.2f} GB)...",
          flush=True)
    states = np.zeros((n, 120), dtype=np.int8)
    dists = np.zeros(n, dtype=np.int8)

    print(f"[bfs6] iterating dict (every 2M printed)...", flush=True)
    t0 = time.time()
    for i, (k, v) in enumerate(bfs.table.items()):
        # bytes (uint8) -> int8 view; values 0-119 fit either way
        states[i] = np.frombuffer(k, dtype=np.uint8).view(np.int8)
        dists[i] = len(v)
        if (i + 1) % 2_000_000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"[bfs6]   {i + 1:,} / {n:,} | {elapsed:.0f}s elapsed | "
                  f"{rate / 1e6:.1f}M/s | ETA {eta:.0f}s", flush=True)
    print(f"[bfs6]   done ({time.time() - t0:.1f}s)", flush=True)

    # Per-depth count check
    counts = np.bincount(dists, minlength=8)
    print(f"[bfs6] per-depth counts: " +
          ", ".join(f"d={d}:{counts[d]:,}" for d in range(min(8, len(counts)))),
          flush=True)

    print(f"[bfs6] saving to {OUT_PATH}...", flush=True)
    t0 = time.time()
    torch.save({
        "states": torch.from_numpy(states),
        "distances": torch.from_numpy(dists),
        "n_states": n,
        "max_depth": int(bfs.max_depth),
    }, OUT_PATH)
    sz = OUT_PATH.stat().st_size / 1024**3
    print(f"[bfs6]   saved {sz:.2f} GB ({time.time() - t0:.1f}s)", flush=True)

    print(f"[bfs6] DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
