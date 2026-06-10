"""Bit-level test for the packed-backpointer layout (idea-4 / 96M debug).

The packed uint32 backpointer encodes (parent_local, parent_rank, move). The
ORIGINAL layout gave parent_local 23 bits, which silently overflowed at
B_global=96M (B_local=12.58M > 2^23) and corrupted the walkback -> found=False
despite real V0 hash hits. The fix is the 24/3/5 layout in jax_beam_spmd_v_qshort.

This test needs NO TPU / JAX device -- it validates pure integer arithmetic:
  1. NEW layout round-trips (pl, rank, move) losslessly for pl in [0, 2^24).
  2. OLD layout corrupts for pl >= 2^23 (reproduces the bug, as a guard against
     regressing to 23 bits).
  3. The exact 96M operating point (B_local=12,582,912) is covered.

It imports the REAL BPTR_* constants from the solver so the test tracks the
shipped layout, and replicates the pack/unpack with the identical shift/mask
formulas used at every JAX site.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pull the shipped constants. Importing the module pulls jax; if that is not
# available we fall back to the literal layout (kept in sync by the asserts).
try:
    from jax_beam_spmd_v_qshort import (
        BPTR_PL_BITS, BPTR_RANK_BITS, BPTR_MOVE_BITS,
        BPTR_RANK_SHIFT, BPTR_MOVE_SHIFT,
        BPTR_PL_MASK, BPTR_RANK_MASK, BPTR_MOVE_MASK,
        BPTR_MAX_B_LOCAL,
    )
    SRC = "imported from jax_beam_spmd_v_qshort"
except Exception as e:  # pragma: no cover - only if jax missing
    print(f"[warn] could not import solver ({e!r}); using literal layout")
    BPTR_PL_BITS, BPTR_RANK_BITS, BPTR_MOVE_BITS = 24, 3, 5
    BPTR_RANK_SHIFT = BPTR_PL_BITS
    BPTR_MOVE_SHIFT = BPTR_PL_BITS + BPTR_RANK_BITS
    BPTR_PL_MASK = (1 << BPTR_PL_BITS) - 1
    BPTR_RANK_MASK = (1 << BPTR_RANK_BITS) - 1
    BPTR_MOVE_MASK = (1 << BPTR_MOVE_BITS) - 1
    BPTR_MAX_B_LOCAL = 1 << BPTR_PL_BITS
    SRC = "literal fallback"


def pack(pl, rank, move, pl_bits):
    """Replicates the JAX packing: pl | (rank << pl_bits) | (move << pl_bits+3).

    Uses uint32 wraparound exactly like XLA's uint32 ops (& overflow behaviour),
    so an out-of-range pl corrupts the higher fields just as it would on device.
    """
    rank_shift = pl_bits
    move_shift = pl_bits + BPTR_RANK_BITS
    pl = np.uint32(pl)
    rank = np.uint32(rank)
    move = np.uint32(move)
    return np.uint32(pl | (rank << np.uint32(rank_shift)) | (move << np.uint32(move_shift)))


def unpack(rec, pl_bits):
    rec = int(rec)
    rank_shift = pl_bits
    move_shift = pl_bits + BPTR_RANK_BITS
    pl_mask = (1 << pl_bits) - 1
    pl = rec & pl_mask
    rank = (rec >> rank_shift) & BPTR_RANK_MASK
    move = (rec >> move_shift) & BPTR_MOVE_MASK
    return pl, rank, move


def main():
    print(f"layout source: {SRC}")
    print(f"BPTR_PL_BITS={BPTR_PL_BITS} RANK_BITS={BPTR_RANK_BITS} "
          f"MOVE_BITS={BPTR_MOVE_BITS}")
    print(f"RANK_SHIFT={BPTR_RANK_SHIFT} MOVE_SHIFT={BPTR_MOVE_SHIFT} "
          f"PL_MASK=0x{BPTR_PL_MASK:06X} MAX_B_LOCAL={BPTR_MAX_B_LOCAL:,}")
    assert BPTR_PL_BITS + BPTR_RANK_BITS + BPTR_MOVE_BITS == 32, "must fill uint32"
    assert BPTR_RANK_SHIFT == BPTR_PL_BITS
    assert BPTR_MOVE_SHIFT == BPTR_PL_BITS + BPTR_RANK_BITS

    rng = np.random.default_rng(0)
    B_LOCAL_96M = 100_663_296 // 8          # 12,582,912
    B_LOCAL_48M = 50_331_648 // 8           # 6,291,456
    N_GEN = 24
    WORLD = 8

    # --- Test 1: NEW layout is lossless across the full pl range incl. 96M. ---
    # Cover boundaries explicitly + a random sweep.
    boundary = [0, 1,
                2**23 - 1, 2**23, 2**23 + 1,         # the OLD overflow edge
                B_LOCAL_48M - 1,
                B_LOCAL_96M - 1,                      # 96M operating point
                BPTR_MAX_B_LOCAL - 1]                 # field max
    rand_pl = rng.integers(0, BPTR_MAX_B_LOCAL, size=200_000, dtype=np.int64)
    pls = np.concatenate([np.array(boundary, dtype=np.int64), rand_pl])
    ranks = rng.integers(0, WORLD, size=pls.shape, dtype=np.int64)
    moves = rng.integers(0, N_GEN, size=pls.shape, dtype=np.int64)

    bad = 0
    for pl, rk, mv in zip(pls.tolist(), ranks.tolist(), moves.tolist()):
        rec = pack(pl, rk, mv, BPTR_PL_BITS)
        u_pl, u_rk, u_mv = unpack(rec, BPTR_PL_BITS)
        if (u_pl, u_rk, u_mv) != (pl, rk, mv):
            bad += 1
            if bad <= 5:
                print(f"  NEW FAIL pl={pl} rank={rk} move={mv} -> "
                      f"({u_pl},{u_rk},{u_mv})")
    assert bad == 0, f"NEW layout lossy in {bad} cases"
    print(f"[PASS] NEW 24/3/5 layout lossless over {pls.shape[0]:,} cases "
          f"(incl. pl up to {BPTR_MAX_B_LOCAL-1:,})")

    # --- Test 2: OLD 23-bit layout corrupts the 96M frontier (the bug). ---
    # Every pl in [2^23, B_local_96M) must misdecode under 23 bits.
    old_overflow = [2**23, 2**23 + 7, B_LOCAL_96M - 1,
                    (2**23 + B_LOCAL_96M) // 2]
    n_corrupt = 0
    for pl in old_overflow:
        rec = pack(pl, rank=3, move=10, pl_bits=23)
        u_pl, u_rk, u_mv = unpack(rec, 23)
        corrupt = (u_pl, u_rk, u_mv) != (pl, 3, 10)
        n_corrupt += int(corrupt)
        print(f"  OLD(23b) pl={pl:,} rank=3 move=10 -> "
              f"pl={u_pl:,} rank={u_rk} move={u_mv}  "
              f"{'CORRUPT' if corrupt else 'ok'}")
    assert n_corrupt == len(old_overflow), (
        "expected OLD layout to corrupt ALL >=2^23 cases (the documented bug)")
    print(f"[PASS] OLD 23-bit layout corrupts all {n_corrupt} pl>=2^23 cases "
          f"(reproduces the 96M failure)")

    # --- Test 3: fraction of the 96M frontier the OLD layout would corrupt. ---
    frac = (B_LOCAL_96M - 2**23) / B_LOCAL_96M
    print(f"[info] OLD layout corrupts {frac*100:.1f}% of the 96M frontier "
          f"({B_LOCAL_96M - 2**23:,} of {B_LOCAL_96M:,} slots) -> "
          f"per-step corruption compounds over ~75 steps -> walkback ~always fails")
    assert B_LOCAL_96M > 2**23, "sanity: 96M must exceed the old field"
    assert B_LOCAL_96M <= BPTR_MAX_B_LOCAL, "96M must fit the new field"
    assert B_LOCAL_48M <= 2**23, "48M must fit the old field (why it worked)"

    print("\nALL BACKPOINTER PACKING TESTS PASSED")


if __name__ == "__main__":
    main()
