"""Single-device JAX beam search for megaminx.

Direct port of `beam_solve_xla_qshort` (megaminx/kaggle_notebooks/tpu_beam_smoke/
build_notebook.py CELL_5_BEAM) to pure JAX. Static-shape throughout so it JITs
cleanly. Used standalone (one TPU core or CPU) and also as the foundation for the
SPMD shared-beam version in jax_beam_spmd.py.

Algorithm (qshort + policy + V0-on-hash-equality):
  Step 0: V-only on the 24 first-move children; topk -> B initial states.
          Pad to B by repeating the last chosen state.
  Steps 1..NUM_STEPS-1:
    1. Student forward (Q model, B states -> B x N_GEN logits).
    2. Policy forward (pi model, B states -> B x N_GEN logits).
    3. Generate B*N_GEN children. Hash each child. Sort+restore dedup.
    4. student_score = student_q_flat + lambda_policy * neg_log_pi_flat.
       Mask dups with +inf. topk(alpha*B, smallest) -> shortlist indices.
    5. Shortlist states. Teacher forward on shortlist -> V values.
    6. topk(B, smallest) by teacher V -> chosen B states for next step.
    7. V0 detection on chosen states via tensor equality. Record found_step
       at first hit.

Path reconstruction: walk the tree backwards from (found_step, found_pos).
"""
from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jax_model import apply as model_apply


def make_hash_vec(state_size: int, seed: int = 0) -> np.ndarray:
    """Random int64 vector for state hashing. Returns numpy (caller moves to device)."""
    rng = np.random.default_rng(seed)
    # int64 positive values up to 1e15 — matches the torch_xla version's range.
    return rng.integers(0, int(1e15), size=state_size, dtype=np.int64)


def _model_predict_chunked(params: dict[str, Any], states: jnp.ndarray,
                           chunk_size: int, dtype: jnp.dtype) -> jnp.ndarray:
    """Chunked forward, used when (B, state_size) is too large for one pass.

    Requires states.shape[0] % chunk_size == 0 (caller ensures).
    Outputs (n,) for V or (n, num_out) for Q/policy.
    """
    n, S = states.shape
    n_chunks = n // chunk_size
    chunks = states.reshape(n_chunks, chunk_size, S)

    def scan_fn(_, chunk):
        return _, model_apply(params, chunk, dtype=dtype)

    _, outs = jax.lax.scan(scan_fn, None, chunks)
    if outs.ndim == 2:
        return outs.reshape(n)
    return outs.reshape(n, -1)


def _topk_smallest(values: jnp.ndarray, k: int):
    """Return (smallest_values, smallest_indices). jax.lax.top_k gives largest."""
    neg_v, idx = jax.lax.top_k(-values, k)
    return -neg_v, idx


@partial(jax.jit, static_argnames=(
    "B", "n_gen", "state_size", "alpha", "num_steps", "internal_bs",
    "dtype", "use_chunked_v", "use_chunked_q",
))
def beam_solve_qshort_jit(
    init_state: jnp.ndarray,         # (state_size,) int8
    teacher_params: dict[str, Any],
    student_params: dict[str, Any],
    policy_params: dict[str, Any],
    all_moves: jnp.ndarray,           # (n_gen, state_size) int32
    V0: jnp.ndarray,                  # (state_size,) int8
    hash_vec: jnp.ndarray,            # (state_size,) int64
    lambda_policy: float,
    B: int,
    alpha: int,
    n_gen: int,
    state_size: int,
    num_steps: int,
    internal_bs: int,
    dtype: Any = jnp.bfloat16,
    use_chunked_v: bool = False,
    use_chunked_q: bool = False,
):
    """Run beam search. Returns (states_final, tree_idx, tree_move, min_v_log,
    found_step, found_pos) where path is reconstructed by the caller from
    tree_idx / tree_move."""
    aB = alpha * B

    def forward_v(params, x):
        if use_chunked_v:
            return _model_predict_chunked(params, x, internal_bs, dtype)
        return model_apply(params, x, dtype=dtype)

    def forward_q(params, x):
        if use_chunked_q:
            return _model_predict_chunked(params, x, internal_bs, dtype)
        return model_apply(params, x, dtype=dtype)

    BIG_F32 = jnp.float32(1e9)

    # --- Step 0: V-only on 24 first-move children ---
    states_seed = jnp.expand_dims(init_state, 0)                       # (1, S)
    neighbors0 = states_seed[:, all_moves].reshape(-1, state_size)     # (24, S)
    # NEVER chunked at step 0: n=24 < any reasonable internal_bs, so a chunked
    # forward would reshape to (0, internal_bs, ...). Use model_apply directly.
    values0 = model_apply(teacher_params, neighbors0, dtype=dtype).astype(jnp.float32)  # (24,)
    k0 = min(B, n_gen)
    top_v0, top_idx0 = _topk_smallest(values0, k0)
    chosen0 = neighbors0[top_idx0]                                     # (k0, S)
    if k0 < B:
        pad = jnp.broadcast_to(chosen0[-1:], (B - k0, state_size))
        states = jnp.concatenate([chosen0, pad], axis=0)
    else:
        states = chosen0[:B]

    tree_idx = jnp.zeros((num_steps, B), dtype=jnp.int32)
    tree_move = jnp.full((num_steps, B), -1, dtype=jnp.int8)
    min_v_log = jnp.full((num_steps,), 1e6, dtype=jnp.float32)

    move0_full = jnp.full((B,), -1, dtype=jnp.int8)
    move0_full = move0_full.at[:k0].set(top_idx0.astype(jnp.int8))
    tree_move = tree_move.at[0].set(move0_full)

    # V0 detection at step 0 (init_state already excluded by caller).
    eq0 = jnp.all(states == V0, axis=1)
    any0 = jnp.any(eq0)
    pos0 = jnp.argmax(eq0.astype(jnp.int32))
    found_step = jnp.where(any0, jnp.int32(0), jnp.int32(-1))
    found_pos = jnp.where(any0, pos0.astype(jnp.int32), jnp.int32(-1))

    # --- Steps 1..num_steps-1 via lax.scan ---
    def beam_step(carry, j):
        states, tree_idx, tree_move, min_v_log, found_step, found_pos = carry

        # Forward student + policy on current B states.
        student_q = forward_q(student_params, states).astype(jnp.float32)   # (B, n_gen)
        policy_q = forward_q(policy_params, states).astype(jnp.float32)     # (B, n_gen)
        log_pi = policy_q - jax.scipy.special.logsumexp(policy_q, axis=-1, keepdims=True)
        neg_log_pi_flat = -log_pi.reshape(-1)                                # (B*n_gen,)

        # Generate children + hash.
        neighbors_2d = states[:, all_moves]                                  # (B, n_gen, S)
        neighbors = neighbors_2d.reshape(-1, state_size)                     # (B*n_gen, S)
        h = jnp.sum(neighbors.astype(jnp.int64) * hash_vec, axis=1)          # (B*n_gen,) int64

        # Static-shape dedup (sort+adjacent-equal+restore).
        sort_h = jnp.sort(h)
        sort_idx = jnp.argsort(h)
        is_dup_sorted = jnp.concatenate([
            jnp.zeros(1, dtype=jnp.bool_),
            sort_h[1:] == sort_h[:-1],
        ])
        restore = jnp.argsort(sort_idx)
        dup_mask = is_dup_sorted[restore]                                    # (B*n_gen,) bool

        # Student score = student_q + lambda * neg_log_pi. Mask dups.
        student_q_flat = student_q.reshape(-1)
        student_score = student_q_flat + lambda_policy * neg_log_pi_flat
        student_score_masked = jnp.where(dup_mask, BIG_F32, student_score)

        # Shortlist (alpha*B smallest by student score).
        _, shortlist_indices = _topk_smallest(student_score_masked, aB)      # (aB,) int32

        # Materialize shortlist states, run teacher rerank.
        shortlist_states = neighbors[shortlist_indices]                      # (aB, S)
        teacher_v = forward_v(teacher_params, shortlist_states).astype(jnp.float32)  # (aB,)

        # Top B by teacher V.
        top_v, chosen_local = _topk_smallest(teacher_v, B)                   # (B,)
        chosen_states = shortlist_states[chosen_local]                       # (B, S)
        chosen_global = shortlist_indices[chosen_local]                      # (B,) int32

        parent = (chosen_global // n_gen).astype(jnp.int32)
        move = (chosen_global % n_gen).astype(jnp.int8)

        # V0 detection via TENSOR equality (not hash; avoids collision risk).
        eq_v0 = jnp.all(chosen_states == V0, axis=1)
        any_hit = jnp.any(eq_v0)
        pos_hit = jnp.argmax(eq_v0.astype(jnp.int32)).astype(jnp.int32)
        is_first_hit = (found_step == -1) & any_hit
        new_found_step = jnp.where(is_first_hit, j, found_step)
        new_found_pos = jnp.where(is_first_hit, pos_hit, found_pos)

        new_tree_idx = tree_idx.at[j].set(parent)
        new_tree_move = tree_move.at[j].set(move)
        new_min_v_log = min_v_log.at[j].set(top_v[0])

        return (chosen_states, new_tree_idx, new_tree_move, new_min_v_log,
                new_found_step, new_found_pos), ()

    init_carry = (states, tree_idx, tree_move, min_v_log, found_step, found_pos)
    final_carry, _ = jax.lax.scan(beam_step, init_carry,
                                  jnp.arange(1, num_steps, dtype=jnp.int32))
    states_final, tree_idx, tree_move, min_v_log, found_step, found_pos = final_carry
    return states_final, tree_idx, tree_move, min_v_log, found_step, found_pos


def reconstruct_path(tree_idx_h: np.ndarray, tree_move_h: np.ndarray,
                     found_step: int, found_pos: int) -> list[int]:
    """Walk the tree backwards from (found_step, found_pos) to assemble the move
    sequence. Returns a list of generator indices. Strips leading -1 sentinels."""
    if found_step < 0:
        return []
    path = []
    pos = int(found_pos)
    for j in range(found_step, -1, -1):
        m = int(tree_move_h[j, pos])
        path.append(m)
        pos = int(tree_idx_h[j, pos])
    path.reverse()
    while path and path[0] < 0:
        path.pop(0)
    return path


def beam_solve_qshort(
    init_state_list: list[int],
    teacher_params: dict[str, Any],
    student_params: dict[str, Any],
    policy_params: dict[str, Any],
    all_moves: jnp.ndarray,
    V0: jnp.ndarray,
    hash_vec: jnp.ndarray,
    lambda_policy: float = 0.05,
    B: int = 131072,
    alpha: int = 2,
    n_gen: int = 24,
    state_size: int = 120,
    num_steps: int = 120,
    internal_bs: int = 32768,
    dtype: Any = jnp.bfloat16,
    use_chunked_v: bool = False,
    use_chunked_q: bool = False,
) -> dict[str, Any]:
    """High-level wrapper: prepare inputs, run jit, reconstruct path."""
    init_state = jnp.asarray(init_state_list, dtype=jnp.int8)
    if jnp.array_equal(init_state, V0):
        return {"found": True, "path_len": 0, "path_idx": [], "found_step": -1}

    (states_final, tree_idx, tree_move, min_v_log, found_step, found_pos
     ) = beam_solve_qshort_jit(
        init_state, teacher_params, student_params, policy_params,
        all_moves, V0, hash_vec, jnp.float32(lambda_policy),
        B=int(B), alpha=int(alpha), n_gen=int(n_gen), state_size=int(state_size),
        num_steps=int(num_steps), internal_bs=int(internal_bs),
        dtype=dtype,
        use_chunked_v=bool(use_chunked_v), use_chunked_q=bool(use_chunked_q),
    )
    fs = int(found_step)
    fp = int(found_pos)
    tree_idx_h = np.asarray(tree_idx)
    tree_move_h = np.asarray(tree_move)
    min_v_h = np.asarray(min_v_log).tolist()
    path = reconstruct_path(tree_idx_h, tree_move_h, fs, fp)
    return {
        "found": fs >= 0,
        "found_step": fs,
        "found_pos": fp,
        "path_idx": path,
        "path_len": len(path),
        "min_v_trajectory": min_v_h,
    }
