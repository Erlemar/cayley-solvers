"""Cross-validate the numpy tail-BFS builder against the torch one.

Two independent implementations of the same table is the point: [[dual_codepath_drift]]
says a second code path rots silently unless something forces them to agree. Here the
agreement is exact and checkable -- same hash, same code packing -- so any divergence is
a hard failure, not a judgement call.

Compares against the .pt written by the patched pilgrim/searcher.py::_build_tail_bfs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
BUNDLE = PROJECT / "cube444" / "kaggle_inference" / "cube444_inference" / "solver"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax_endgame as eg  # noqa: E402


def load_puzzle():
    import torch

    sys.path.insert(0, str(BUNDLE))
    from pilgrim.utils import generate_inverse_moves, parse_generator_spec

    spec = json.loads((BUNDLE / "generators" / "p002.json").read_text(encoding="utf-8"))
    moves, move_names = parse_generator_spec(spec)
    all_moves = np.asarray(moves, dtype=np.int64)
    inverse_moves = np.asarray(generate_inverse_moves(move_names), dtype=np.int64)
    v0 = torch.load(BUNDLE / "targets" / "p002-t000.pt",
                    map_location="cpu", weights_only=False).numpy().astype(np.uint8)
    return all_moves, inverse_moves, v0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    all_moves, inverse_moves, v0 = load_puzzle()
    # "torch" here on purpose: this test compares against a table built by
    # pilgrim/searcher.py, which seeds hash_vec from torch's generator. The TPU beam
    # uses backend="numpy" -- see make_hash_vec.
    hash_vec = eg.make_hash_vec(v0.shape[0], args.seed, backend="torch")
    print(f"gens {all_moves.shape}  v0 {v0.shape}  depth {args.depth}")

    hashes, codes = eg.build_tail_table(
        all_moves, inverse_moves, v0, hash_vec, args.depth)
    total = int(np.sum(eg.LEVEL_SIZES[1:args.depth + 1]))
    print(f"numpy table: {len(hashes):,} states (expected {total:,})")

    cache = sorted((BUNDLE / "tail_cache").glob(f"tail_bfs_d{args.depth}_*.pt"))
    if not cache:
        print(f"no torch cache for depth {args.depth}; built-only check")
        return 0 if len(hashes) == total else 1

    blob = torch.load(cache[0], map_location="cpu")
    t_h = blob["hashes"].numpy()
    t_c = blob["codes"].numpy()
    print(f"torch table: {len(t_h):,} states  ({cache[0].name})")

    ok = len(t_h) == len(hashes)
    print(f"  count match      : {ok}")
    same_h = same_c = False
    if ok:
        same_h = bool(np.array_equal(t_h, hashes))
        same_c = bool(np.array_equal(t_c, codes))
        print(f"  hashes identical : {same_h}")
        print(f"  codes  identical : {same_c}")
        ok = same_h

    # Codes may legitimately differ: a state at depth d is reachable from several depth
    # d-1 parents, and whichever parent is visited first supplies the code. The two
    # builders chunk the frontier differently (262,144 vs torch's 16,384), so they
    # tie-break differently. What must hold is that EVERY code is a valid optimal tail:
    # replaying it from the state reaches solved, and its length equals the BFS depth.
    base = all_moves.shape[0] + 1
    rng = np.random.default_rng(0)
    sample = rng.choice(len(hashes), size=min(400, len(hashes)), replace=False)

    def state_from_code(code: int) -> np.ndarray:
        # s --m0--> ... --mk--> solved, so s = inv(m0) o ... o inv(mk) applied to solved.
        state = v0.copy()
        for move in reversed(eg.decode_code(code, base)):
            state = state[all_moves[inverse_moves[move]]]
        return state

    def solves(code: int, state: np.ndarray) -> bool:
        cur = state
        for move in eg.decode_code(code, base):
            cur = cur[all_moves[move]]
        return bool(np.array_equal(cur, v0))

    for label, table_codes in (("numpy", codes), ("torch", t_c)):
        good = same_len = 0
        for i in sample:
            code = int(table_codes[i])
            state = state_from_code(code)
            if int(eg._hash(state.reshape(1, -1), hash_vec)[0]) == int(hashes[i]) \
                    and solves(code, state):
                good += 1
            if len(eg.decode_code(code, base)) == len(
                    eg.decode_code(int(codes[i]), base)):
                same_len += 1
        print(f"  {label} codes replay to the filed state and solve it: "
              f"{good}/{len(sample)}   same length as numpy: {same_len}/{len(sample)}")
        ok = ok and good == len(sample) and same_len == len(sample)

    if same_h and not same_c:
        print("\n  NOTE: codes differ but both are valid optimal tails (different "
              "tie-break among equal-depth parents). The table's contract is 'a' "
              "shortest tail, not 'the' shortest tail.")

    print("\nENDGAME PARITY " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
