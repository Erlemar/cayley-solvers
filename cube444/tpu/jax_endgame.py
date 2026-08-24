"""Exact endgame (tail-BFS) table for the cube444 TPU beam.

The beam terminates as soon as any live state is within `depth` moves of solved and finishes
with the exact BFS path. Structurally this can only shorten a path, never lengthen one.

Two properties make this cheap enough to run inside a TPU kernel:

  * The hash is the SAME linear hash the torch solver uses -- `sum(state * hash_vec)` in
    int64 -- so a table built here and one built by `pilgrim/searcher.py` are byte-identical
    and either can validate the other.
  * The deepest level is never expanded again, so its 63.5M states never need to be kept.
    Storing only (hash, code) puts the depth-6 table at 67,041,676 x 16 B = 1.07 GB, which
    fits in one v5e chip's 16 GB HBM replicated -- `jnp.searchsorted` then runs inside the
    step with no host sync.

Level sizes are known exactly and are asserted: 1, 24, 468, 9000, 172914, 3316744, 63542526.

Functions:
    build_tail_table(all_moves, inverse_moves, v0, hash_vec, depth) -> (hashes, codes)
    decode_code(code, base)          -> list of move indices that solve the state
    lookup(table_hashes, table_codes, state_hashes) -> (found, codes)
"""
from __future__ import annotations

import numpy as np

# BFS level sizes from solved, measured exactly (level 6 first measured 2026-08-09;
# see cube444_stage3_heldout). Index = depth. Used as a build-time assertion.
LEVEL_SIZES = [1, 24, 468, 9000, 172914, 3316744, 63542526]

# Parents per expansion chunk. 262,144 x 23 children x 96 B = 579 MB of states in flight.
CHUNK_PARENTS = 262_144


def make_hash_vec(state_size: int = 96, seed: int = 0, backend: str = "numpy") -> np.ndarray:
    """Build the linear hash vector. `backend` is REQUIRED to be chosen deliberately.

    The two code paths in this repo seed it differently and the vectors are NOT equal:

        backend="numpy"  np.random.default_rng(seed)   <- gcp_beam_cube444.py, the TPU beam
        backend="torch"  torch.Generator(seed)         <- pilgrim/searcher.py, the GPU solver

    The endgame table is addressed BY HASH, so a table built with one vector and queried
    with the other never hits. That failure is silent: the beam runs, returns verified
    paths, and the endgame simply contributes nothing -- indistinguishable from "the
    endgame didn't help". Always build the table with the same vector the beam uses.
    """
    if backend == "numpy":
        return np.random.default_rng(seed).integers(
            0, int(1e15), size=state_size, dtype=np.int64)
    if backend == "torch":
        import torch

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        return torch.randint(
            0, int(1e15), (state_size,), generator=gen, dtype=torch.int64
        ).numpy()
    raise ValueError(f"backend must be 'numpy' or 'torch', got {backend!r}")


def _hash(states: np.ndarray, hash_vec: np.ndarray) -> np.ndarray:
    # int64 with deliberate wraparound, exactly like torch's sum over int64.
    with np.errstate(over="ignore"):
        return (states.astype(np.int64) * hash_vec).sum(axis=1)


def make_zobrist(state_size: int = 96, num_classes: int = 6,
                 seed: int = 0) -> np.ndarray:
    """(state_size, num_classes) table of full-width 64-bit randoms.

    WHY THIS EXISTS. The shipped hash is a LINEAR FORM, `sum(s_i * hash_vec_i)`, written
    for permutation puzzles where s_i spans 0..119. Ported to a colour cube its symbols
    are 0..5, the per-position multiplier collapses from ~60 to ~2.5, and the sum packs
    into a narrow band that never even wraps int64. Measured effective width:

        megaminx-style (120 pos, values 0..119)   2^59.1
        cube444        ( 96 pos, values 0..5 )    2^55.5     <- 12x fewer slots
        true 64-bit Zobrist                       2^64

    Against a 67M-entry table that is P(false hit) 1.3e-9 per probe vs 3.6e-12, and the
    beam probes ~5.7e8 times per 18-pid run -- i.e. ~0.7 phantom hits per run, ~40 over a
    full 1043-pid pass. Zobrist takes that to ~0.1 per full pass.

    Used ONLY for the tail table. Dedup keeps the linear hash: it compares ~524k
    candidates per step rather than probing a 67M set, so its exposure is ~1000x
    smaller and not worth perturbing a validated code path for.
    """
    rng = np.random.default_rng(seed ^ 0x5A17B0)   # distinct stream from hash_vec
    return rng.integers(-(2 ** 63), 2 ** 63 - 1, size=(state_size, num_classes),
                        dtype=np.int64)


def zobrist_hash(states: np.ndarray, table: np.ndarray) -> np.ndarray:
    """XOR-fold of per-(position, symbol) randoms. states: (B, state_size)."""
    cols = table[np.arange(states.shape[1]), states.astype(np.int64)]
    out = cols[:, 0]
    for j in range(1, cols.shape[1]):
        out = np.bitwise_xor(out, cols[:, j])
    return out


def build_tail_table(
    all_moves: np.ndarray,
    inverse_moves: np.ndarray,
    v0: np.ndarray,
    hash_vec: np.ndarray,
    depth: int,
    verbose: bool = True,
    zobrist: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """BFS out to `depth`, returning hash-sorted (hashes, codes).

    `code` packs the solving move sequence base-(n_gens+1), least significant digit first,
    with 0 reserved as the terminator -- identical to Searcher._build_tail_bfs so the two
    implementations cross-validate.
    """
    n_gens, state_size = all_moves.shape
    base = n_gens + 1

    # `zobrist` overrides the linear hash for the table's keys (see make_zobrist).
    def H(x):
        return _hash(x, hash_vec) if zobrist is None else zobrist_hash(x, zobrist)

    visited = np.sort(H(v0.reshape(1, -1)))
    frontier = v0.reshape(1, -1).astype(np.uint8)
    frontier_codes = np.zeros(1, dtype=np.int64)
    frontier_last = np.full(1, -1, dtype=np.int64)

    all_hashes: list[np.ndarray] = []
    all_codes: list[np.ndarray] = []

    for level in range(depth):
        is_last = level == depth - 1
        # Chunk the expansion: level 6 has 3.3M parents x 23 non-backtracking moves =
        # 76M children, which is 7.3 GB of states if materialised at once. Only the
        # (hash, code) pairs survive, so never hold more than one chunk of states.
        chunk = max(1, CHUNK_PARENTS)
        h_rows, code_rows, state_rows, last_rows = [], [], [], []
        for start in range(0, frontier.shape[0], chunk):
            stop = min(start + chunk, frontier.shape[0])
            n_par = stop - start
            parent_idx = np.repeat(np.arange(n_par, dtype=np.int64), n_gens)
            moves = np.tile(np.arange(n_gens, dtype=np.int64), n_par)
            # Non-backtracking: undoing the entering move lands at depth-1, seen already.
            last = frontier_last[start:stop][parent_idx]
            keep = (last < 0) | (moves != inverse_moves[np.maximum(last, 0)])
            parent_idx, moves = parent_idx[keep], moves[keep]
            if moves.size == 0:
                continue

            children = frontier[start:stop][parent_idx][
                np.arange(moves.size)[:, None], all_moves[moves]]
            child_h = H(children)
            child_codes = (inverse_moves[moves] + 1
                           + base * frontier_codes[start:stop][parent_idx])

            fresh = ~_isin_sorted(child_h, visited)
            child_h, child_codes = child_h[fresh], child_codes[fresh]
            h_rows.append(child_h)
            code_rows.append(child_codes)
            if not is_last:
                state_rows.append(children[fresh])
                last_rows.append(moves[fresh])
            del children

        child_h = np.concatenate(h_rows)
        child_codes = np.concatenate(code_rows)
        h_rows = code_rows = None

        order = np.argsort(child_h, kind="stable")
        child_h, child_codes = child_h[order], child_codes[order]
        uniq = np.empty(child_h.shape[0], dtype=bool)
        uniq[0] = True
        np.not_equal(child_h[1:], child_h[:-1], out=uniq[1:])
        child_h, child_codes = child_h[uniq], child_codes[uniq]

        all_hashes.append(child_h)
        all_codes.append(child_codes)
        if verbose:
            expected = LEVEL_SIZES[level + 1] if level + 1 < len(LEVEL_SIZES) else None
            flag = "" if expected is None else (" OK" if len(child_h) == expected
                                                else f" MISMATCH expected {expected:,}")
            print(f"tail bfs: level {level + 1} = {len(child_h):,} states{flag}",
                  flush=True)
        if is_last:
            break
        visited = np.union1d(visited, child_h)
        frontier = np.concatenate(state_rows)[order][uniq]
        frontier_codes = child_codes
        frontier_last = np.concatenate(last_rows)[order][uniq]
        state_rows = last_rows = None

    hashes = np.concatenate(all_hashes)
    codes = np.concatenate(all_codes)
    order = np.argsort(hashes, kind="stable")
    hashes, codes = hashes[order], codes[order]
    uniq = np.empty(hashes.shape[0], dtype=bool)
    uniq[0] = True
    np.not_equal(hashes[1:], hashes[:-1], out=uniq[1:])
    return hashes[uniq], codes[uniq]


def _isin_sorted(values: np.ndarray, sorted_ref: np.ndarray) -> np.ndarray:
    if sorted_ref.size == 0:
        return np.zeros(values.shape[0], dtype=bool)
    pos = np.searchsorted(sorted_ref, values)
    pos_clamped = np.minimum(pos, sorted_ref.size - 1)
    return (pos < sorted_ref.size) & (sorted_ref[pos_clamped] == values)


def decode_code(code: int, base: int = 25) -> list[int]:
    moves: list[int] = []
    code = int(code)
    while code:
        digit = code % base
        if digit:
            moves.append(digit - 1)
        code //= base
    return moves


def lookup(table_hashes, table_codes, state_hashes):
    """Device-side membership test. Returns (found_mask, codes).

    jnp.searchsorted over a 67M-entry sorted int64 array is a binary search per query --
    ~26 steps, fully vectorised, no host sync.
    """
    import jax.numpy as jnp

    pos = jnp.searchsorted(table_hashes, state_hashes)
    pos_clamped = jnp.minimum(pos, table_hashes.shape[0] - 1)
    found = (pos < table_hashes.shape[0]) & (
        jnp.take(table_hashes, pos_clamped) == state_hashes)
    return found, jnp.take(table_codes, pos_clamped)
