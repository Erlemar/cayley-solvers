"""Local correctness test for the qshort JAX prefilter logic.

Recreates the qshort step in isolation (no SPMD, no streaming) and checks:
  1. alpha_v values match V on the gathered alpha-children one-by-one.
  2. child_v_full[i, top_alpha_idx[i, k]] == alpha_v[i, k].
  3. child_v_full[i, m] == BIG_F32 for m not in top_alpha_idx[i].
  4. flatten order: child_v[i*n_gen + m] aligns with children[i*n_gen + m].
  5. Compare beam quality on a single step: top-B of qshort's child_v vs top-B
     of v_only's full child_v on the same parents. If recall=100% on the kept
     children, qshort is correct at the step level.

Usage:
    PYTHONUTF8=1 .venv/Scripts/python.exe megaminx/scripts/test_qshort_jax.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "kaggle_notebooks" / "tpu_beam_spmd_jax"))

from jax_model import load_params_from_pt, apply as model_apply


def main() -> int:
    import json
    puzzle_info = json.load(open(PROJECT / "data" / "puzzle_info.json"))
    gen_names = list(puzzle_info["generators"].keys())
    n_gen = len(gen_names)
    state_size = len(puzzle_info["central_state"])
    all_moves_np = np.array([puzzle_info["generators"][n] for n in gen_names], dtype=np.int8)
    all_moves = jnp.asarray(all_moves_np)
    print(f"n_gen={n_gen}, state_size={state_size}, all_moves shape={all_moves.shape}")

    # Load V + Q.
    v_path = PROJECT / "models" / "m_az_v4_v_only.pt"
    q_path = PROJECT / "models" / "m23_v3_az_v4_sym" / "epoch_0199.pt"
    print(f"loading V from {v_path}")
    v_params = load_params_from_pt(v_path, hidden_dims=(2048, 512), num_res_blocks=2)
    print(f"loading Q from {q_path}")
    q_params = load_params_from_pt(q_path, hidden_dims=(2048, 1024), num_res_blocks=3)

    # Build a small parent_chunk of realistic states: start from solved + 20 random walks.
    rng = np.random.default_rng(42)
    solved = np.array(puzzle_info["central_state"], dtype=np.int8)
    parent_chunk = 256  # small for fast testing
    parents = np.tile(solved, (parent_chunk, 1))
    for _ in range(20):
        moves = rng.integers(0, n_gen, size=parent_chunk)
        for i in range(parent_chunk):
            parents[i] = parents[i][all_moves_np[moves[i]]]
    states_chunk = jnp.asarray(parents)
    print(f"parent batch: {states_chunk.shape}")

    dtype = jnp.bfloat16
    BIG_F32 = jnp.float32(1e9)
    alpha = 2

    # === Replicate the GLOBAL top-αB qshort logic (corrected) ===
    def _chunked_apply(params, x, chunk_size):
        n, S = x.shape
        n_chunks_inner = max(1, n // chunk_size)
        chunks = x.reshape(n_chunks_inner, n // n_chunks_inner, S)
        def _scan_fn(_, c):
            return _, model_apply(params, c, dtype=dtype)
        _, outs = jax.lax.scan(_scan_fn, None, chunks)
        if outs.ndim == 2:
            return outs.reshape(n)
        return outs.reshape(n, -1)

    chunk_n = parent_chunk * n_gen
    children = states_chunk[:, all_moves].reshape(-1, state_size)
    parent_q = model_apply(q_params, states_chunk, dtype=dtype).astype(jnp.float32)
    q_flat = parent_q.reshape(-1)
    n_keep = alpha * parent_chunk  # global top-αB
    _, keep_idx = jax.lax.top_k(-q_flat, n_keep)  # smallest n_keep
    keep_idx = keep_idx.astype(jnp.int32)
    selected_children = children[keep_idx]
    selected_v = _chunked_apply(v_params, selected_children, 512).astype(jnp.float32)
    child_v = jnp.full((chunk_n,), BIG_F32, dtype=jnp.float32)
    child_v = child_v.at[keep_idx].set(selected_v)
    child_v_full = child_v.reshape(parent_chunk, n_gen)  # for downstream tests

    # === Reference: V on ALL children (v_only path) ===
    children_full = states_chunk[:, all_moves].reshape(-1, state_size)
    print(f"\nchildren_full shape: {children_full.shape}")
    child_v_ref = _chunked_apply(v_params, children_full, 512).astype(jnp.float32)
    print(f"child_v_ref shape: {child_v_ref.shape}")
    child_v_ref_2d = child_v_ref.reshape(parent_chunk, n_gen)

    # === Tests (corrected for GLOBAL top-αB algorithm) ===
    print("\n=== Tests ===")

    # Test 1: selected_v matches V on the same children from full reference
    selected_v_ref = child_v_ref[keep_idx]
    diff1 = float(jnp.max(jnp.abs(selected_v - selected_v_ref)))
    print(f"Test 1: selected_v vs ref via keep_idx, max abs diff = {diff1:.6f}")
    test1_pass = diff1 < 0.1

    # Test 2: child_v at kept positions equals selected_v
    test2_check = child_v[keep_idx]
    diff2 = float(jnp.max(jnp.abs(test2_check - selected_v)))
    print(f"Test 2: child_v[keep_idx] vs selected_v, max abs diff = {diff2:.6f}")
    test2_pass = diff2 < 1e-6

    # Test 3: child_v at non-kept positions = BIG_F32
    keep_mask = jnp.zeros((chunk_n,), dtype=jnp.bool_).at[keep_idx].set(True)
    n_sentinel_expected = int(jnp.sum(~keep_mask))
    n_sentinel_actual = int(jnp.sum((child_v == BIG_F32) & ~keep_mask))
    print(f"Test 3: sentinel count, expected {n_sentinel_expected}, actual {n_sentinel_actual}")
    test3_pass = n_sentinel_expected == n_sentinel_actual

    # Test 4: GLOBAL Q recall — do Q's top-αB include V's top-B?
    B_test = parent_chunk  # take "B" = parent_chunk for this test
    # V top-B globally
    _, v_top_idx = jax.lax.top_k(-child_v_ref, B_test)
    v_top_set = set(int(x) for x in v_top_idx)
    # Q top-αB globally
    q_top_set = set(int(x) for x in keep_idx)
    recall_global = len(v_top_set & q_top_set) / len(v_top_set)
    print(f"Test 4: GLOBAL Q top-{alpha}B recall vs V top-B = {recall_global:.4f}")
    test4_pass = recall_global >= 0.99

    # Test 5: per-step top-B selection. Does top-B by child_v (qshort) match top-B by child_v_ref (v_only)?
    # This is the actual "does the beam keep the same states as v_only?" test.
    _, qshort_topB_idx = jax.lax.top_k(-child_v, B_test)
    _, vonly_topB_idx = jax.lax.top_k(-child_v_ref, B_test)
    qshort_topB_set = set(int(x) for x in qshort_topB_idx)
    vonly_topB_set = set(int(x) for x in vonly_topB_idx)
    beam_overlap = len(qshort_topB_set & vonly_topB_set) / B_test
    print(f"Test 5: top-B beam overlap (qshort vs v_only) = {beam_overlap:.4f}")
    test5_pass = beam_overlap >= 0.99

    # Test 6: flat layout — child_v[i*n_gen + m] matches reshape
    for idx in [0, 100, 1000, 5000]:
        flat = float(child_v[idx])
        twod = float(child_v_full[idx // n_gen, idx % n_gen])
        assert flat == twod, f"flat layout mismatch at idx={idx}: {flat} vs {twod}"
    print(f"Test 6: flat reshape layout consistent. PASS")
    test6_pass = True

    print("\n=== Summary ===")
    print(f"  Test 1 (selected_v matches V):     {'PASS' if test1_pass else 'FAIL'}")
    print(f"  Test 2 (scatter correctness):      {'PASS' if test2_pass else 'FAIL'}")
    print(f"  Test 3 (sentinel coverage):        {'PASS' if test3_pass else 'FAIL'}")
    print(f"  Test 4 (global recall vs V top-B): {'PASS' if test4_pass else 'FAIL'}")
    print(f"  Test 5 (top-B beam overlap):       {'PASS' if test5_pass else 'FAIL'}")
    print(f"  Test 6 (flat reshape):             {'PASS' if test6_pass else 'FAIL'}")

    all_pass = test1_pass and test2_pass and test3_pass and test4_pass and test5_pass and test6_pass
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
