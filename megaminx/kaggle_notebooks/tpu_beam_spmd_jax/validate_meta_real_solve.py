"""Real multi-step solve validation for the idea-4 meta-materialize body.

WHY THIS EXISTS
---------------
sanity_check_neighborhood.py only ever ran DEEP test.csv scrambles at B=128,
which never solve in 22 steps -- so its "meta vs baseline length-safety" check
only ever compared found=False==found=False. The meta body's multi-step
backpointer chain + 2-hop materialize round-trip were NEVER exercised through a
real verified solve, at either alpha. That blind spot let the 96M failure ship.

This test closes it. It generates SHALLOW non-backtracking scrambles (so a real
solve is found in a handful of beam steps at modest B) and runs each through:
  - the WORKING streaming body (meta=False)  -- ground truth
  - the meta-materialize body (meta=True)     -- under test
at BOTH alpha=2 (qshort prefilter active) and alpha=24 (V-only-equivalent, the
exact mode the 96M run used and the one never tested before), with the
neighborhood early-stop ON (exercises the near-hit prefix+suffix walkback that
produced best_near in the failing run).

Assertions per (scramble, alpha):
  1. streaming solves AND its path verifies (re-apply moves -> SOLVED).
  2. meta solves     AND its path verifies.
  3. meta path_len == streaming path_len (identical-beam length-safety).

A pass means the meta body's backptr chain + materialize round-trip are correct
at small B. Combined with test_backptr_packing.py (which proves the 24/3/5
packing is lossless to 2^24, the part small-B can't reach), this covers both the
plumbing and the field-width fix.

NOTE: small B can't exceed 2^23, so this does NOT exercise the packing overflow
itself -- that's what test_backptr_packing.py is for. The two tests are
complementary.

Run: .venv/Scripts/python.exe megaminx/kaggle_notebooks/tpu_beam_spmd_jax/validate_meta_real_solve.py
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import json  # noqa: E402
from jax_beam import make_hash_vec  # noqa: E402
from jax_model import load_params_from_pt  # noqa: E402
from jax_beam_spmd_v_qshort import (  # noqa: E402
    beam_solve_v_qshort_spmd_packed,
    build_solved_neighborhood,
    make_mesh,
)

ROOT = HERE.parents[1]
PUZZLE_INFO = json.loads((ROOT / "data" / "puzzle_info.json").read_text())
GENERATORS = PUZZLE_INFO["generators"]
MOVE_NAMES = list(GENERATORS.keys())
SOLVED = tuple(PUZZLE_INFO["central_state"])
N_GEN = len(MOVE_NAMES)
STATE_SIZE = len(SOLVED)
ALL_MOVES_NP = np.array([GENERATORS[n] for n in MOVE_NAMES], dtype=np.int32)

_name_to_idx = {n: i for i, n in enumerate(MOVE_NAMES)}
INV_MOVE_IDX = np.full(N_GEN, -1, dtype=np.int32)
for _i, _nm in enumerate(MOVE_NAMES):
    _inv = _nm[1:] if _nm.startswith("-") else "-" + _nm
    INV_MOVE_IDX[_i] = _name_to_idx[_inv]
assert (INV_MOVE_IDX >= 0).all()


def verify_path(initial_state, path_idx):
    cur = list(initial_state)
    for m in path_idx:
        cur = [cur[g] for g in GENERATORS[MOVE_NAMES[m]]]
    return tuple(cur) == SOLVED


def make_scramble(depth, seed):
    """Non-backtracking random walk from SOLVED; returns the scrambled state."""
    rng = np.random.default_rng(seed)
    cur = np.array(SOLVED, dtype=np.int8)
    last = -1
    for _ in range(depth):
        choices = [g for g in range(N_GEN)
                   if last < 0 or g != int(INV_MOVE_IDX[last])]
        gi = int(rng.choice(choices))
        cur = cur[ALL_MOVES_NP[gi]]
        last = gi
    return cur.tolist()


def solve(s0, v_params, q_params, alpha, meta, nbhd, hash_vec, mesh, num_steps=20):
    nbhd_hash, nbhd_len, nbhd_mv = nbhd
    cfg = dict(
        B_local=256, K_per_peer=64, n_gen=N_GEN, state_size=STATE_SIZE,
        num_steps=num_steps, dtype=jnp.float32, internal_bs=64, parent_chunk=128,
        pack_v_score=True, progress_every=0,
        nbhd_hash_sorted=nbhd_hash, nbhd_suffix_len=nbhd_len,
        nbhd_suffix_moves=nbhd_mv,
        use_meta_materialize=meta, alpha_req=8.0,  # REQ_CAP=B_local -> zero overflow even if all survivors share a source
    )
    V0 = jnp.asarray(SOLVED, dtype=jnp.int8)
    all_moves = jnp.asarray(ALL_MOVES_NP)
    return beam_solve_v_qshort_spmd_packed(
        s0, v_params, q_params, alpha, all_moves, V0, hash_vec, mesh, **cfg)


def main():
    devices = jax.devices()
    print(f"JAX {jax.__version__}  devices={len(devices)}")
    if len(devices) != 8:
        print(f"FAIL: expected 8 logical CPU devices, got {len(devices)}")
        return 1
    mesh = make_mesh(devices)

    hash_vec_np = make_hash_vec(STATE_SIZE, seed=0)
    hash_vec = jnp.asarray(hash_vec_np)

    v_path = ROOT / "models" / "m_az_v4_v_only.pt"
    q_path = ROOT / "models" / "m23_v3_az_v4_sym" / "epoch_0199.pt"
    if not v_path.exists() or not q_path.exists():
        print(f"FAIL: production checkpoints missing ({v_path}, {q_path})")
        return 1
    print(f"loading V={v_path.name} (AZ v4, 2048/512) ...")
    v_params = load_params_from_pt(v_path, hidden_dims=(2048, 512))
    print(f"loading Q={q_path.parent.name}/{q_path.name} (m23_v3, 2048/1024 x3) ...")
    q_params = load_params_from_pt(q_path, hidden_dims=(2048, 1024), num_res_blocks=3)

    print("building solved neighborhood radius=4 ...")
    nbhd = build_solved_neighborhood(
        ALL_MOVES_NP, INV_MOVE_IDX, hash_vec_np, np.array(SOLVED), 4)
    nbhd = (jnp.asarray(np.asarray(nbhd[0], dtype=np.int64)),
            jnp.asarray(np.asarray(nbhd[1], dtype=np.int8)),
            np.asarray(nbhd[2], dtype=np.int8))
    print(f"  neighborhood M={int(nbhd[0].shape[0]):,}")

    # Shallow scrambles: deep enough to need multi-step beam (so the backptr
    # chain + materialize round-trip are exercised), shallow enough to solve
    # reliably at B_global=2048.
    scrambles = []
    for depth in (5, 7, 9):
        for seed in (1, 2):
            scrambles.append((depth, seed, make_scramble(depth, seed)))
    # Drop any that are accidentally already solved (shouldn't happen for d>=5).
    scrambles = [(d, s, st) for (d, s, st) in scrambles if tuple(st) != SOLVED]

    print(f"\n== meta vs streaming on {len(scrambles)} shallow scrambles, "
          f"alpha in (2, 24), neighborhood ON ==")
    n_fail = 0
    n_len_mismatch = 0
    n_unsolved = 0
    n_fp = 0
    rows = []
    for (depth, seed, s0) in scrambles:
        for alpha in (2, 24):
            t0 = time.time()
            rs = solve(s0, v_params, q_params, alpha, False, nbhd, hash_vec, mesh)
            rm = solve(s0, v_params, q_params, alpha, True, nbhd, hash_vec, mesh)
            wall = time.time() - t0
            oks = verify_path(s0, rs["path_idx"]) if rs["found"] else False
            okm = verify_path(s0, rm["path_idx"]) if rm["found"] else False
            # false positive = claims found but path doesn't verify
            if rs["found"] and not oks:
                n_fp += 1
            if rm["found"] and not okm:
                n_fp += 1
            len_ok = (rs["found"] == rm["found"]) and (
                (not rs["found"]) or rs["path_len"] == rm["path_len"])
            if not (rs["found"] and oks):
                n_unsolved += 1
            if not (rm["found"] and okm):
                n_unsolved += 1
            if not len_ok:
                n_len_mismatch += 1
            status = "OK" if (rs["found"] and oks and rm["found"] and okm and len_ok) else "FAIL"
            if status == "FAIL":
                n_fail += 1
            rows.append((depth, seed, alpha, rs, rm, oks, okm, len_ok, status, wall))
            print(f"  d={depth} seed={seed} a={alpha:>2} | "
                  f"stream found={rs['found']} len={rs['path_len']:>2} via={rs.get('via','-'):>12} verify={oks} | "
                  f"meta found={rm['found']} len={rm['path_len']:>2} via={rm.get('via','-'):>12} verify={okm} | "
                  f"len_match={len_ok} [{status}] {wall:4.1f}s")

    # --- Group 2: force the NEIGHBORHOOD near-hit walkback to be the winner by
    # capping num_steps so the beam reaches the radius-ball but not exact V0.
    # Exercises meta's near_* outputs + the wrapper's prefix+suffix assembly. ---
    print("\n== force-neighborhood group (num_steps capped, alpha=24) ==")
    n_nbhd_meta = 0
    for depth, cap in ((8, 6), (10, 7), (10, 8)):
        for seed in (1, 2):
            s0 = make_scramble(depth, seed)
            if tuple(s0) == SOLVED:
                continue
            t0 = time.time()
            rs = solve(s0, v_params, q_params, 24, False, nbhd, hash_vec, mesh, num_steps=cap)
            rm = solve(s0, v_params, q_params, 24, True, nbhd, hash_vec, mesh, num_steps=cap)
            wall = time.time() - t0
            oks = verify_path(s0, rs["path_idx"]) if rs["found"] else False
            okm = verify_path(s0, rm["path_idx"]) if rm["found"] else False
            if rs["found"] and not oks:
                n_fp += 1
            if rm["found"] and not okm:
                n_fp += 1
            len_ok = (rs["found"] == rm["found"]) and (
                (not rs["found"]) or rs["path_len"] == rm["path_len"])
            if not len_ok:
                n_len_mismatch += 1
            if rm["found"] and okm and rm.get("via") == "neighborhood":
                n_nbhd_meta += 1
            # Both must agree; if neither finds within the cap that's fine (not a
            # failure) -- but if one finds and the other doesn't, or lengths
            # differ, that's a meta-vs-stream divergence.
            print(f"  d={depth} cap={cap} seed={seed} | "
                  f"stream found={rs['found']} len={rs['path_len']:>2} via={rs.get('via','-'):>12} verify={oks} | "
                  f"meta found={rm['found']} len={rm['path_len']:>2} via={rm.get('via','-'):>12} verify={okm} | "
                  f"len_match={len_ok} {wall:4.1f}s")
    print(f"  meta solves that WON via neighborhood walkback (verified): {n_nbhd_meta}")
    if n_nbhd_meta == 0:
        print("  [warn] neighborhood walkback never won through meta -- path "
              "exercised in-flight each step but not as the chosen candidate")

    print()
    print(f"summary: {len(rows)} (scramble,alpha) primary cases | "
          f"unsolved-or-unverified={n_unsolved} | "
          f"false-positives={n_fp} | meta-vs-stream len mismatches={n_len_mismatch}")
    if n_fail or n_fp or n_unsolved or n_len_mismatch:
        print("\nFAIL: the meta body did NOT match streaming on every real solve.")
        return 1
    print("\nPASS: meta-materialize solves every shallow scramble through a real "
          "multi-step\n      backptr chain, every path verifies, and meta length "
          "== streaming length\n      at alpha=2 AND alpha=24. Meta plumbing is correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
