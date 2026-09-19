"""Pack a legacy IHES ``BfsTable`` pickle for GPU MITM depth probes.

The legacy table stores 72-int Python tuples and shortest words.  GPU joining
only needs a deterministic Zobrist hash and the exact depth of each element.
The resulting NPZ is roughly 100 MiB for d6 instead of a 1.77 GiB pickle that
expands to about 10 GiB of Python objects.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.bfs_table import BfsTable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    t0 = time.time()
    print(f"loading {args.table}...", flush=True)
    table_obj = BfsTable.load(args.table)
    table = table_obj.table
    n = len(table)
    max_depth = int(table_obj.max_depth)
    first = next(iter(table))
    state_size = len(first)
    print(
        f"loaded {n:,} states, size={state_size}, max_depth={max_depth} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    ztab = rng.integers(0, 1 << 63, size=(state_size, state_size), dtype=np.int64)
    hashes = np.empty(n, dtype=np.int64)
    depths = np.empty(n, dtype=np.uint8)

    items = iter(table.items())
    offset = 0
    while offset < n:
        take = min(args.chunk_size, n - offset)
        keys: list[tuple[int, ...]] = []
        words: list[tuple[int, ...]] = []
        for _ in range(take):
            key, word = next(items)
            keys.append(key)
            words.append(word)
        states = np.asarray(keys, dtype=np.uint8)
        h = np.zeros(take, dtype=np.int64)
        for pos in range(state_size):
            h ^= ztab[pos, states[:, pos]]
        hashes[offset : offset + take] = h
        depths[offset : offset + take] = np.fromiter(
            (len(word) for word in words), dtype=np.uint8, count=take
        )
        offset += take
        if offset % 1_000_000 < take or offset == n:
            print(f"  packed {offset:,}/{n:,}", flush=True)

    order = np.argsort(hashes)
    hashes = hashes[order]
    depths = depths[order]
    collisions = int(np.count_nonzero(hashes[1:] == hashes[:-1]))
    if collisions:
        raise SystemExit(
            f"{collisions} Zobrist collisions; rerun with a different --seed"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        hashes=hashes,
        depths=depths,
        ztab=ztab,
        max_depth=np.asarray(max_depth, dtype=np.uint8),
        state_size=np.asarray(state_size, dtype=np.uint16),
        source_states=np.asarray(n, dtype=np.int64),
        seed=np.asarray(args.seed, dtype=np.int64),
    )
    print(
        f"wrote {args.out}: {args.out.stat().st_size / 2**20:.1f} MiB, "
        f"{n:,} unique hashes ({time.time() - t0:.1f}s total)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
