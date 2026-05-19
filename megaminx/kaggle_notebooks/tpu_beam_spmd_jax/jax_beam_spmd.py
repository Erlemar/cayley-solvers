"""SPMD shared-beam JAX implementation: B_global = world_size * B_local.

Design (mirrors the failed torch_xla v5 design, but in JAX which avoids both
the multi-tensor all_gather alignment bug and the stale-buffer reuse class):

  * Each TPU core ("rank") owns B_local states. Global beam = world_size * B_local.
  * Each state's "owner rank" is hash(state) % world_size — a fixed partition.
  * Per step:
      1. Each rank generates B_local * N_GEN children of ITS owned states.
      2. Per (sender, owner) pair, take top-K_PER_PEER children by student
         score where owner_rank == owner. Pack (state, parent_local, move)
         into a single uint8 PACK_SIZE-byte record.
      3. One jax.lax.all_to_all routes bucket S from each sender to rank S.
         Single packed tensor — no multi-call alignment risk.
      4. Receiver: in-rank dedup, teacher rerank, topk(B_local). Update its
         owned states + tree.
      5. V0 detection: tensor equality on owned states; cross-rank reduce
         picks the lowest-step finder (tie-break on lowest rank).

The Python beam loop is OUTSIDE the shard_map'd step (jax-ml/jax #26148: a
fori_loop inside shard_map with manual collectives crashes). Each step is a
jitted shard_map call with the carry threaded explicitly via donate_argnums.

Path reconstruction (cross-rank walkback):
  At end of search, walk back from (found_step, found_pos_local, found_pos_rank)
  one step at a time. Each rank contributes 0 unless it owns the current walk
  position; a lax.psum sums to the value owned by the actual owner.
"""
from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from jax_model import apply as model_apply

PACK_SIZE = 128


def make_mesh(devices=None):
    """Build the 1-D mesh over TPU cores."""
    if devices is None:
        devices = jax.devices()
    return Mesh(np.asarray(devices), axis_names=("cores",))


def _layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps) * gamma + beta


def _topk_smallest(values, k):
    neg_v, idx = jax.lax.top_k(-values, k)
    return -neg_v, idx


def _build_step_body(
    teacher_params, student_params, policy_params,
    all_moves, V0, hash_vec, V0_hash,
    B_local, world_size, K_per_peer, n_gen, state_size,
    lambda_policy, dtype, internal_bs,
):
    """Returns a closure beam_step_local(states, tree_..., found_step, ..., j).

    All static configuration is captured. The closure runs inside shard_map.
    Forward passes are chunked along the batch axis to avoid materializing
    the full (n, state_size, embed_dim) embedding intermediate in HBM (which
    OOM'd at B_local=512K on TPU v5e-8 with 16 GB/chip).
    """
    BIG_F32 = jnp.float32(1e9)
    BIG_INT = jnp.int32(2_000_000_000)
    aB_local = K_per_peer * world_size

    def _chunked_apply(params, x, chunk_size):
        """Run model_apply on x in chunks via lax.scan. x.shape[0] MUST be
        divisible by chunk_size. Returns (n,) for V (output_dim=1) or
        (n, num_out) for Q/policy."""
        n, S = x.shape
        n_chunks = n // chunk_size
        chunks = x.reshape(n_chunks, chunk_size, S)
        def _scan_fn(_, chunk):
            return _, model_apply(params, chunk, dtype=dtype)
        _, outs = jax.lax.scan(_scan_fn, None, chunks)
        if outs.ndim == 2:  # V model: each chunk -> (chunk_size,)
            return outs.reshape(n)
        return outs.reshape(n, -1)  # Q/policy: each chunk -> (chunk_size, num_out)

    def forward_v(params, x):
        return _chunked_apply(params, x, internal_bs).astype(jnp.float32)

    def forward_q(params, x):
        return _chunked_apply(params, x, internal_bs).astype(jnp.float32)

    # Precomputed per-child indexing (static).
    n_total = B_local * n_gen
    flat_idx = jnp.arange(n_total, dtype=jnp.int32)
    parent_local_per_child = (flat_idx // n_gen).astype(jnp.int32)
    move_per_child = (flat_idx % n_gen).astype(jnp.int8)
    arange_aB = jnp.arange(aB_local, dtype=jnp.int32)
    sender_rank_per_recv = (arange_aB // K_per_peer).astype(jnp.int8)

    def beam_step_local(states, tree_parent_local, tree_parent_rank, tree_move,
                       min_v_log, found_step, found_pos_local, found_pos_rank,
                       verify_state, j):
        """One step. axis_name='cores' is in scope via shard_map.

        shard_map keeps sharded axes (with local size). Squeeze axis-0 on entry
        so the body sees per-shard shapes natively, then re-add on return.
        verify_state captures chosen_states[pos_hit] at first V0 hit (per-rank).
        """
        states = states[0]                                # (B_local, state_size)
        tree_parent_local = tree_parent_local[0]          # (num_steps, B_local)
        tree_parent_rank = tree_parent_rank[0]
        tree_move = tree_move[0]
        min_v_log = min_v_log[0]                          # (num_steps,)
        found_step = found_step[0]                        # ()
        found_pos_local = found_pos_local[0]
        found_pos_rank = found_pos_rank[0]
        verify_state = verify_state[0]                    # (state_size,)
        # j is replicated scalar (no sharded axis); keep as-is.

        rank_int = jax.lax.axis_index("cores").astype(jnp.int32)

        # 1. Student + policy forward on owned states.
        student_q = forward_q(student_params, states)        # (B_local, n_gen)
        policy_q = forward_q(policy_params, states)          # (B_local, n_gen)
        log_pi = policy_q - jax.scipy.special.logsumexp(policy_q, axis=-1, keepdims=True)
        neg_log_pi_flat = -log_pi.reshape(-1)
        student_score = student_q.reshape(-1) + lambda_policy * neg_log_pi_flat  # (n_total,)

        # 2. Generate children + hash.
        neighbors = states[:, all_moves].reshape(-1, state_size)  # (n_total, S)
        h = jnp.sum(neighbors.astype(jnp.int64) * hash_vec, axis=1)
        owner = (h % jnp.int64(world_size)).astype(jnp.int32)

        # 3. Per-owner top-K_per_peer (loop over owner is unrolled in JIT — small).
        send_buckets = jnp.zeros((world_size, K_per_peer, PACK_SIZE), dtype=jnp.uint8)
        for S in range(world_size):
            mask_S = (owner == S)
            score_for_S = jnp.where(mask_S, student_score, BIG_F32)
            top_v_S, top_idx_S = _topk_smallest(score_for_S, K_per_peer)
            is_pad_S = top_v_S >= (BIG_F32 * 0.5)

            sel_states = neighbors[top_idx_S]              # (K_per_peer, S) int8
            sel_parent_local = parent_local_per_child[top_idx_S]  # (K_per_peer,) int32
            sel_move = move_per_child[top_idx_S]           # (K_per_peer,) int8

            zero_state = jnp.zeros_like(sel_states, dtype=jnp.uint8)
            zero_int32 = jnp.zeros_like(sel_parent_local)
            zero_int8 = jnp.zeros_like(sel_move)

            sel_states_u8 = jnp.where(is_pad_S[:, None], zero_state, sel_states.astype(jnp.uint8))
            sel_parent_local_z = jnp.where(is_pad_S, zero_int32, sel_parent_local)
            sel_move_z = jnp.where(is_pad_S, zero_int8, sel_move)

            # Pack into PACK_SIZE bytes per record:
            #   [0..119]  state (120 uint8)
            #   [120..123] parent_local int32 LE
            #   [124]     move uint8
            #   [125..127] pad zeros
            bucket = jnp.zeros((K_per_peer, PACK_SIZE), dtype=jnp.uint8)
            bucket = bucket.at[:, 0:120].set(sel_states_u8)
            bucket = bucket.at[:, 120].set((sel_parent_local_z & 0xFF).astype(jnp.uint8))
            bucket = bucket.at[:, 121].set(((sel_parent_local_z >> 8) & 0xFF).astype(jnp.uint8))
            bucket = bucket.at[:, 122].set(((sel_parent_local_z >> 16) & 0xFF).astype(jnp.uint8))
            bucket = bucket.at[:, 123].set(((sel_parent_local_z >> 24) & 0xFF).astype(jnp.uint8))
            bucket = bucket.at[:, 124].set(sel_move_z.astype(jnp.uint8))
            send_buckets = send_buckets.at[S].set(bucket)

        # 4. all_to_all: bucket S goes to rank S. split=0, concat=0 (the SPMD shape
        # invariant: in (W, K, P) -> out (W, K, P) where out[i] came from sender i).
        recv_buckets = jax.lax.all_to_all(
            send_buckets, axis_name="cores",
            split_axis=0, concat_axis=0,
            tiled=True,
        )

        # 5. Unpack received candidates.
        recv_flat = recv_buckets.reshape(-1, PACK_SIZE)  # (aB_local, PACK_SIZE)
        recv_states_u8 = recv_flat[:, 0:120]
        recv_states = recv_states_u8.astype(jnp.int8)
        recv_parent_local = (
            recv_flat[:, 120].astype(jnp.int32)
            | (recv_flat[:, 121].astype(jnp.int32) << 8)
            | (recv_flat[:, 122].astype(jnp.int32) << 16)
            | (recv_flat[:, 123].astype(jnp.int32) << 24)
        )
        recv_move = recv_flat[:, 124].astype(jnp.int8)
        recv_sender_rank = sender_rank_per_recv

        # Padding detection: real states sum to 0+1+...+119 = 7140, padding sums to 0.
        recv_state_sum = jnp.sum(recv_states.astype(jnp.int32), axis=1)
        is_padding = (recv_state_sum == 0)

        # In-rank dedup (sort+adjacent-equal+restore).
        recv_h = jnp.sum(recv_states.astype(jnp.int64) * hash_vec, axis=1)
        sort_h = jnp.sort(recv_h)
        sort_idx = jnp.argsort(recv_h)
        is_dup_sorted = jnp.concatenate([
            jnp.zeros(1, dtype=jnp.bool_),
            sort_h[1:] == sort_h[:-1],
        ])
        restore = jnp.argsort(sort_idx)
        dup_mask = is_dup_sorted[restore]

        # Teacher rerank + topk(B_local).
        teacher_v = forward_v(teacher_params, recv_states)
        teacher_v_masked = jnp.where(dup_mask | is_padding, BIG_F32, teacher_v)
        top_v_keep, keep_idx = _topk_smallest(teacher_v_masked, B_local)

        chosen_states = recv_states[keep_idx]
        chosen_parent_local = recv_parent_local[keep_idx]
        chosen_parent_rank = recv_sender_rank[keep_idx]
        chosen_move = recv_move[keep_idx]
        chosen_h = recv_h[keep_idx]  # int64 hashes of chosen states
        chosen_state_sum = jnp.sum(chosen_states.astype(jnp.int32), axis=1)
        chosen_is_real = (chosen_state_sum != 0)  # padding => sum=0; V0 => sum=7140

        new_tree_parent_local = tree_parent_local.at[j].set(chosen_parent_local)
        new_tree_parent_rank = tree_parent_rank.at[j].set(chosen_parent_rank)
        new_tree_move = tree_move.at[j].set(chosen_move)
        new_min_v_log = min_v_log.at[j].set(top_v_keep[0])

        # V0 detection via SCALAR HASH equality, gated on chosen_is_real to
        # exclude any padded entries that survived topk. v3 (tensor eq) and v4
        # (hash eq) both fired spuriously on padded [0]*120 — the non-real
        # filter is belt-and-suspenders against either of those failure modes.
        eq_v0 = (chosen_h == V0_hash) & chosen_is_real
        any_hit = jnp.any(eq_v0)
        pos_hit = jnp.argmax(eq_v0.astype(jnp.int32)).astype(jnp.int32)
        is_first_hit = (found_step == -1) & any_hit
        my_found_step = jnp.where(is_first_hit, j, found_step)
        my_found_pos_local = jnp.where(is_first_hit, pos_hit, found_pos_local)
        my_found_pos_rank = jnp.where(is_first_hit, rank_int, found_pos_rank)

        # Diagnostic: snapshot the state that triggered V0 detection (per-rank).
        # On host we compare verify_state[fpr_h] to V0 — if it matches, V0
        # detection was correct (any bug is in tree storage / walkback). If
        # not, the V0 check is firing spuriously and we have a deeper issue.
        candidate_state = chosen_states[pos_hit]  # (state_size,) int8
        new_verify_state = jnp.where(
            is_first_hit,
            candidate_state,
            verify_state,
        )

        # v6 (DIAGNOSTIC): cross-rank reduce REMOVED. Each rank's found_step
        # now reflects ONLY that rank's local first hit. Wrapper computes the
        # global winner on host. This lets us see exactly which rank locally
        # detected V0 (and at which iter) vs. what `pmin/psum` was previously
        # reporting. Per the expert's feedback after v5: V0_hash % 8 == 1, so
        # V0 should land on rank 1 if hash-partition + axis_index are consistent;
        # if found_pos_rank comes back as 0, host shard-ordering disagrees with
        # axis_index inside the body.
        new_found_step = my_found_step
        new_found_pos_local = my_found_pos_local
        new_found_pos_rank = my_found_pos_rank

        # Re-add the squeezed leading shard axis on return.
        return (chosen_states[None, :, :],
                new_tree_parent_local[None, :, :],
                new_tree_parent_rank[None, :, :],
                new_tree_move[None, :, :],
                new_min_v_log[None, :],
                new_found_step[None],
                new_found_pos_local[None],
                new_found_pos_rank[None],
                new_verify_state[None, :])

    return beam_step_local


def beam_solve_qshort_spmd(
    init_state_list: list[int],
    teacher_params, student_params, policy_params,
    all_moves: jnp.ndarray,
    V0: jnp.ndarray,
    hash_vec: jnp.ndarray,
    mesh: Mesh,
    B_local: int,
    K_per_peer: int,
    n_gen: int = 24,
    state_size: int = 120,
    num_steps: int = 120,
    lambda_policy: float = 0.05,
    dtype=jnp.bfloat16,
    internal_bs: int = 32768,
) -> dict[str, Any]:
    """High-level SPMD beam solver. Returns dict with path + metadata."""
    devices = mesh.devices.flatten()
    world_size = len(devices)
    assert K_per_peer * world_size > 0
    SOLVED = np.asarray(V0)
    init_state = np.asarray(init_state_list, dtype=np.int8)
    if np.array_equal(init_state, SOLVED):
        return {"found": True, "path_len": 0, "path_idx": [], "found_step": -1, "wall_s": 0.0}

    # Step 0: V-only on the 24 first-move children. Replicated (every rank does it
    # independently, deterministic — same seeds appear on every rank, then hash-
    # partition kicks in at step 1).
    init_dev = jnp.asarray(init_state)
    states_seed = jnp.expand_dims(init_dev, 0)
    neighbors0 = states_seed[:, all_moves].reshape(-1, state_size)
    values0 = model_apply(teacher_params, neighbors0, dtype=dtype).astype(jnp.float32)
    k0 = min(B_local, n_gen)
    _, top_idx0 = _topk_smallest(values0, k0)
    chosen0 = neighbors0[top_idx0]
    if k0 < B_local:
        pad = jnp.broadcast_to(chosen0[-1:], (B_local - k0, state_size))
        states0 = jnp.concatenate([chosen0, pad], axis=0)
    else:
        states0 = chosen0[:B_local]
    # The seeds-replicate-on-every-rank means cross-rank dedup will see massive dups
    # at step 1, but the first all_to_all routes them to owners which filter dups.

    # Replicate states0 to (world_size, B_local, state_size) via NamedSharding.
    states_global = jnp.broadcast_to(states0[None, :, :], (world_size, B_local, state_size))
    # Tree storage per-rank.
    tree_parent_local = jnp.zeros((world_size, num_steps, B_local), dtype=jnp.int32)
    tree_parent_rank = jnp.zeros((world_size, num_steps, B_local), dtype=jnp.int8)
    tree_move = jnp.full((world_size, num_steps, B_local), -1, dtype=jnp.int8)

    # tree_move[0] = move that produced state k. For k<24 it's top_idx0[k] (the
    # seed-move whose V-rank was k). For padded slots [24..B_local-1] all copies
    # of chosen0[-1] => move = top_idx0[23].
    top_idx0_i8 = top_idx0.astype(jnp.int8)  # (k0,)
    if k0 < B_local:
        last_move = top_idx0_i8[k0 - 1]
        pad_moves = jnp.broadcast_to(last_move, (B_local - k0,))
        move0_full = jnp.concatenate([top_idx0_i8, pad_moves])
    else:
        move0_full = top_idx0_i8[:B_local]
    move0_full = jnp.broadcast_to(move0_full[None, :], (world_size, B_local))
    tree_move = tree_move.at[:, 0, :].set(move0_full)

    min_v_log = jnp.full((world_size, num_steps), 1e6, dtype=jnp.float32)
    found_step = jnp.full((world_size,), -1, dtype=jnp.int32)
    found_pos_local = jnp.full((world_size,), -1, dtype=jnp.int32)
    found_pos_rank = jnp.full((world_size,), -1, dtype=jnp.int32)
    verify_state = jnp.zeros((world_size, state_size), dtype=jnp.int8)

    # Initial V0 check on step-0 states (replicated across ranks).
    eq0 = jnp.all(states0 == V0, axis=1)
    any0 = jnp.any(eq0)
    pos0 = jnp.argmax(eq0.astype(jnp.int32)).astype(jnp.int32)
    if bool(any0):
        # Found at step 0. The seed-move that produced states0[pos0] is
        # move0_full[pos0] = top_idx0[pos0] (for pos0 < k0; padded slots map to
        # top_idx0[k0-1]).
        pos0_h = int(pos0)
        if pos0_h < k0:
            seed_move = int(top_idx0[pos0_h])
        else:
            seed_move = int(top_idx0[k0 - 1])
        return {"found": True, "path_len": 1, "path_idx": [seed_move],
                "found_step": 0, "wall_s": 0.0}

    # Compute V0_hash on host as a Python int (passed as static constant into
    # the body, bypasses any closure-array capture issue under shard_map).
    V0_hash_host = int(np.sum(np.asarray(V0).astype(np.int64) * np.asarray(hash_vec)))

    # Build the per-shard step body (closure over static configuration).
    step_body = _build_step_body(
        teacher_params, student_params, policy_params,
        all_moves, V0, hash_vec, jnp.int64(V0_hash_host),
        B_local, world_size, K_per_peer, n_gen, state_size,
        float(lambda_policy), dtype, int(internal_bs),
    )

    # Jitted shard_map wrapper. We use named axis 'cores' from the mesh.
    # NOTE: donate_argnums removed in v3 — v2's off-by-one found_step pattern
    # (found_step = V0_actual_arrival - 1) suggested buffer aliasing across
    # consecutive step_fn calls in the Python loop. donate=() means JAX won't
    # reuse input buffers as outputs, costing ~2x memory but eliminating any
    # aliasing class. The verify_state extra slot lets us distinguish a tree-
    # storage bug from a false-positive V0 detection.
    @jax.jit
    def step_fn(states, tp_local, tp_rank, tmove, mv_log, fs, fpl, fpr, vstate, j_arr):
        try:
            from jax.experimental.shard_map import shard_map
        except ImportError:
            from jax import shard_map  # JAX 0.5+
        return shard_map(
            step_body,
            mesh=mesh,
            in_specs=(P("cores"), P("cores"), P("cores"), P("cores"),
                      P("cores"), P("cores"), P("cores"), P("cores"),
                      P("cores"), P()),
            out_specs=(P("cores"), P("cores"), P("cores"), P("cores"),
                       P("cores"), P("cores"), P("cores"), P("cores"),
                       P("cores")),
        )(states, tp_local, tp_rank, tmove, mv_log, fs, fpl, fpr, vstate, j_arr)

    # Shard the per-rank tensors using NamedSharding so step_fn sees (world_size, ...)
    # globally but each per-shard view is (B_local, ...) etc.
    from jax.sharding import NamedSharding
    sharding_states = NamedSharding(mesh, P("cores"))
    sharding_scalar = NamedSharding(mesh, P("cores"))

    states_d = jax.device_put(states_global, sharding_states)
    tp_local_d = jax.device_put(tree_parent_local, sharding_states)
    tp_rank_d = jax.device_put(tree_parent_rank, sharding_states)
    tmove_d = jax.device_put(tree_move, sharding_states)
    mv_log_d = jax.device_put(min_v_log, sharding_scalar)
    fs_d = jax.device_put(found_step, sharding_scalar)
    fpl_d = jax.device_put(found_pos_local, sharding_scalar)
    fpr_d = jax.device_put(found_pos_rank, sharding_scalar)
    vstate_d = jax.device_put(verify_state, sharding_states)

    import time
    t_start = time.time()
    first_iter_t = None

    for j in range(1, num_steps):
        t_iter = time.time()
        j_arr = jnp.int32(j)  # replicated scalar
        (states_d, tp_local_d, tp_rank_d, tmove_d,
         mv_log_d, fs_d, fpl_d, fpr_d, vstate_d) = step_fn(
            states_d, tp_local_d, tp_rank_d, tmove_d,
            mv_log_d, fs_d, fpl_d, fpr_d, vstate_d, j_arr,
        )
        if first_iter_t is None:
            jax.block_until_ready(fs_d)
            first_iter_t = time.time() - t_iter

    jax.block_until_ready(fs_d)

    # Extract scalars from rank 0 (all ranks have the same convergence after psum).
    # v6 (DIAGNOSTIC): extract ALL per-rank arrays, not just rank 0 / fpr_h.
    fs_per_rank   = np.asarray(fs_d)        # (world_size,) — each rank's first local hit, -1 if never
    fpl_per_rank  = np.asarray(fpl_d)       # (world_size,) — each rank's first-hit local position
    fpr_per_rank  = np.asarray(fpr_d)       # (world_size,) — each rank's recorded "owner rank" (= rank_int when hit fired)
    vstate_per_rank = np.asarray(vstate_d)  # (world_size, state_size) — each rank's captured candidate at first hit
    mv_per_rank   = np.asarray(mv_log_d)    # (world_size, num_steps) — each rank's min_v trajectory

    V0_host_np = np.asarray(V0)
    vstate_is_V0_per_rank = [
        bool(np.array_equal(vstate_per_rank[r], V0_host_np))
        for r in range(world_size)
    ]

    # Cross-rank winner selection ON HOST.
    INT_MAX = 2 ** 30
    fs_signed = np.where(fs_per_rank >= 0, fs_per_rank, INT_MAX)
    global_min_step = int(fs_signed.min())

    per_rank_summary = []
    for r in range(world_size):
        fs_r = int(fs_per_rank[r])
        min_v_at_fs = float(mv_per_rank[r, fs_r]) if fs_r >= 0 else None
        per_rank_summary.append({
            "rank": r,
            "fstep": fs_r,
            "fpos": int(fpl_per_rank[r]),
            "fprank_recorded": int(fpr_per_rank[r]),
            "vstate_head": vstate_per_rank[r, :12].tolist(),
            "vstate_tail": vstate_per_rank[r, -12:].tolist(),
            "vstate_is_V0": vstate_is_V0_per_rank[r],
            "min_v_at_fstep": min_v_at_fs,
        })

    # Host-side V0_hash + expected owner rank (for cross-checking shard order).
    V0_hash_check = int(np.sum(V0_host_np.astype(np.int64) * np.asarray(hash_vec)))
    expected_owner_rank = V0_hash_check % world_size

    if global_min_step >= INT_MAX:
        return {"found": False, "path_len": 0, "path_idx": [],
                "found_step": -1, "wall_s": time.time() - t_start,
                "first_iter_s": first_iter_t,
                "min_v_trajectory_rank0": mv_per_rank[0].tolist(),
                "per_rank_summary": per_rank_summary,
                "V0_hash": V0_hash_check,
                "expected_owner_rank": expected_owner_rank,
                "detected_is_V0": False}

    # Pick global winner: smallest fstep, tie-break smallest rank.
    winner_ranks = np.where(fs_signed == global_min_step)[0]
    winner_rank  = int(winner_ranks[0])
    fs_h  = int(fs_per_rank[winner_rank])
    fpl_h = int(fpl_per_rank[winner_rank])
    fpr_h = winner_rank

    # Cross-rank walkback (host-side, downloads tree slices on demand).
    tp_local_h = np.asarray(tp_local_d)  # (world_size, num_steps, B_local)
    tp_rank_h  = np.asarray(tp_rank_d)
    tmove_h    = np.asarray(tmove_d)

    path_idx = []
    cur_rank = fpr_h
    cur_pos  = fpl_h
    for j in range(fs_h, -1, -1):
        m = int(tmove_h[cur_rank, j, cur_pos])
        if m >= 0:
            path_idx.append(m)
        if j > 0:
            new_rank = int(tp_rank_h[cur_rank, j, cur_pos])
            new_pos  = int(tp_local_h[cur_rank, j, cur_pos])
            cur_rank = new_rank
            cur_pos  = new_pos
    path_idx.reverse()
    while path_idx and path_idx[0] < 0:
        path_idx.pop(0)

    # Detected state of the winning rank.
    detected_state = vstate_per_rank[fpr_h].tolist()
    detected_is_V0 = detected_state == V0_host_np.tolist()

    return {
        "found": True,
        "path_len": len(path_idx),
        "path_idx": path_idx,
        "found_step": fs_h,
        "found_pos_local": fpl_h,
        "found_pos_rank": fpr_h,
        "wall_s": time.time() - t_start,
        "first_iter_s": first_iter_t,
        "min_v_trajectory_rank0": mv_per_rank[0].tolist(),
        "detected_state_first12": detected_state[:12],
        "detected_state_last12": detected_state[-12:],
        "detected_is_V0": bool(detected_is_V0),
        "per_rank_summary": per_rank_summary,
        "V0_hash": V0_hash_check,
        "expected_owner_rank": expected_owner_rank,
    }
