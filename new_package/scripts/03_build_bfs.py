"""BFS from the solved Tetraminx state: exact-distance anchors + endgame hash table.

Two products, both driven by the same sweep:

  1. ANCHORS (--states-out): (state, exact_depth) pairs for the Bellman trainer's
     exact-distance mixin.  Full states for every depth <= --full-depth, plus a
     random sample of --sample-per-depth states for deeper levels.
  2. ENDGAME TABLE (--hash-out): a sorted int64 Zobrist-hash array plus a
     parallel int8 depth array covering every state within --max-depth.  The beam
     looks up each frontier node; on a hit it finishes the puzzle exactly by
     descending depth k -> k-1, so the tail of every solve is provably optimal.

Level sizes (measured): 24 / 408 / 6592 / 105136 / 1659416, branching ~15.8,
so d6 ~ 26M states (208 MB of hashes) and d7 ~ 415M (out of scope).

    python3 tetraminx/scripts/03_build_bfs.py --max-depth 6 --full-depth 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]


def zobrist_table(state_size: int, n_values: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**63 - 1, size=(state_size, n_values), dtype=np.int64)


def hash_states(states: np.ndarray, ztab: np.ndarray) -> np.ndarray:
    """XOR-fold Zobrist hash; states is (N, S) of small ints."""
    out = np.zeros(states.shape[0], dtype=np.int64)
    for i in range(states.shape[1]):
        out ^= ztab[i][states[:, i]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzle-info", type=Path,
                    default=PROJECT / "tetraminx" / "data" / "puzzle_info.json")
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--full-depth", type=int, default=5,
                    help="keep every state up to this depth in the anchor file")
    ap.add_argument("--sample-per-depth", type=int, default=2_000_000,
                    help="random states kept per depth beyond --full-depth")
    ap.add_argument("--chunk", type=int, default=250_000,
                    help="frontier states expanded per batch (memory knob)")
    ap.add_argument("--states-out", type=Path,
                    default=PROJECT / "tetraminx" / "data" / "bfs_anchors.npz")
    ap.add_argument("--hash-out", type=Path,
                    default=PROJECT / "tetraminx" / "data" / "bfs_endgame.npz")
    ap.add_argument("--torch-out", type=Path,
                    default=PROJECT / "tetraminx" / "data" / "bfs_d6_train.pt",
                    help="anchors in the format cayley.bellman expects "
                         "(keys: states int8, distances int8)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    info = json.loads(args.puzzle_info.read_text(encoding="utf-8"))
    move_names = list(info["generators"].keys())
    gens = np.array([info["generators"][nm] for nm in move_names], dtype=np.int64)
    solved = np.array(info["central_state"], dtype=np.int64)
    n = len(solved)
    ztab = zobrist_table(n, n, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    frontier = solved.astype(np.uint8)[None, :]
    seen_hashes = hash_states(frontier, ztab)
    seen_sorted = np.sort(seen_hashes)

    all_hashes = [seen_hashes]
    all_depths = [np.zeros(1, dtype=np.int8)]
    anchor_states = [frontier.copy()]
    anchor_depths = [np.zeros(1, dtype=np.int8)]
    level_sizes = [1]

    t0 = time.time()
    for depth in range(1, args.max_depth + 1):
        new_states_parts, new_hash_parts = [], []
        for lo in range(0, frontier.shape[0], args.chunk):
            # keep children in uint8 -- an int64 gather here is 8x the memory
            block = frontier[lo:lo + args.chunk]
            children = block[:, gens.reshape(-1)].reshape(-1, n)
            h = hash_states(children, ztab)
            # drop duplicates inside this batch, then anything already seen
            uniq_h, first_idx = np.unique(h, return_index=True)
            pos = np.searchsorted(seen_sorted, uniq_h)
            pos = np.clip(pos, 0, len(seen_sorted) - 1)
            fresh = seen_sorted[pos] != uniq_h
            keep_h = uniq_h[fresh]
            keep_idx = first_idx[fresh]
            if keep_h.size:
                new_hash_parts.append(keep_h)
                new_states_parts.append(children[keep_idx].astype(np.uint8))
            del children, h
        new_h = np.concatenate(new_hash_parts) if new_hash_parts else np.zeros(0, np.int64)
        new_s = (np.concatenate(new_states_parts) if new_states_parts
                 else np.zeros((0, n), np.uint8))
        # dedup across chunks
        new_h, idx = np.unique(new_h, return_index=True)
        new_s = new_s[idx]

        frontier = new_s
        seen_sorted = np.sort(np.concatenate([seen_sorted, new_h]))
        all_hashes.append(new_h)
        all_depths.append(np.full(new_h.size, depth, dtype=np.int8))
        level_sizes.append(int(new_h.size))

        if depth <= args.full_depth:
            anchor_states.append(new_s.copy())
            anchor_depths.append(np.full(new_s.shape[0], depth, dtype=np.int8))
        elif args.sample_per_depth > 0:
            k = min(args.sample_per_depth, new_s.shape[0])
            sel = rng.choice(new_s.shape[0], size=k, replace=False)
            anchor_states.append(new_s[sel].copy())
            anchor_depths.append(np.full(k, depth, dtype=np.int8))
        print(f"depth {depth}: {new_h.size:,} new  (cum {seen_sorted.size:,})  "
              f"{time.time() - t0:.1f}s", flush=True)

    hashes = np.concatenate(all_hashes)
    depths = np.concatenate(all_depths)
    order = np.argsort(hashes)
    hashes, depths = hashes[order], depths[order]
    assert np.unique(hashes).size == hashes.size, "hash collision in endgame table"

    args.hash_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.hash_out, hashes=hashes, depths=depths, ztab=ztab,
             move_names=np.array(move_names), max_depth=args.max_depth)
    a_states = np.concatenate(anchor_states)
    a_depths = np.concatenate(anchor_depths)
    np.savez(args.states_out, states=a_states, depths=a_depths,
             move_names=np.array(move_names))

    if args.torch_out is not None:
        import torch
        torch.save({"states": torch.from_numpy(a_states.astype(np.int8)),
                    "distances": torch.from_numpy(a_depths.astype(np.int8)),
                    "move_names": move_names,
                    "level_sizes": level_sizes},
                   args.torch_out)
        print(f"bellman anchors -> {args.torch_out}", flush=True)

    print(f"level sizes: {level_sizes}", flush=True)
    print(f"endgame table: {hashes.size:,} states <= d{args.max_depth} -> {args.hash_out}",
          flush=True)
    print(f"anchors: {a_states.shape[0]:,} states -> {args.states_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
