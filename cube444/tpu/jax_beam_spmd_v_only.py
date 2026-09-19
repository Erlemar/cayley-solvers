"""SPMD shared-beam JAX implementation: V-ONLY variant (no qshort, no policy).

CUBE444 PORT (2026-07-23) of megaminx/kaggle_notebooks/tpu_beam_spmd_jax/
jax_beam_spmd_v_only.py. Only change from the megaminx original: the 6 packed
state-slice sites were `[:, 0:120]` (megaminx state width) and are now
`[:, 0:state_size]`, so the 96-byte color-cube state packs correctly. Everything
else -- parent-local at bytes state..+3 via the fixed 120..123 offsets, move at
124, optional bf16 V score at 125..126, PACK_SIZE=128 -- is unchanged and has
ample headroom for state_size<=120. The all-zero padding sentinel stays valid:
a real color-cube state sums to >=240 (16 each of colors 0..5), never 0.

Sibling of jax_beam_spmd.py for AZ-style V models (e.g. m_az_v4_v_only) where
m23_v2 qshort REGRESSES paths because m23_v2 was distilled from m05's V
landscape, not AZ's. Drop qshort entirely; rank all B_LOCAL*N_GEN children
by the V model directly.

Algorithm (per step inside the shard_map body):
  1. Each rank generates B_LOCAL * N_GEN children of its owned states.
  2. Run V on ALL children (chunked via lax.scan).
  3. Hash each child, owner = hash % world_size.
  4. Per-owner top-K_PER_PEER by V score; pack (state, parent_local, move)
     uint8 records.
  5. Single packed `lax.all_to_all` routes bucket S from each sender to rank S.
  6. Receiver: in-rank dedup (sort+adjacent-equal+restore), RE-RUN V on
     received states (cheap vs. step 2 — aB_LOCAL ≈ 2*B_LOCAL vs. 24*B_LOCAL),
     topk(B_LOCAL) by V.
  7. V0 detection: hash equality + non-padding filter (per-rank).

Cross-rank winner selection is HOST-SIDE (cross-rank reduce inside the body
mis-propagates on TPU v5e-8; see spmd_jax_recovery.md). Each rank carries
its own local first-hit state; wrapper picks the global winner on host
after the loop.

Re-running V on receive instead of packing V scores into the bucket keeps
PACK_SIZE=128 and the packing identical to jax_beam_spmd.py — simpler code,
lower bug surface. The ~8% extra compute (aB_LOCAL vs. 24*B_LOCAL) is
negligible.
"""
from __future__ import annotations

from functools import partial
from typing import Any

import gc
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from jax_model import apply as model_apply

PACK_SIZE = 128
BPTR_PL_BITS = 25
BPTR_RANK_BITS = 3
BPTR_MOVE_BITS = 5
BPTR_RANK_SHIFT = BPTR_PL_BITS
BPTR_MOVE_SHIFT = BPTR_PL_BITS + BPTR_RANK_BITS
BPTR_PL_MASK = (1 << BPTR_PL_BITS) - 1
BPTR_RANK_MASK = (1 << BPTR_RANK_BITS) - 1
BPTR_MOVE_MASK = (1 << BPTR_MOVE_BITS) - 1


def make_mesh(devices=None):
    if devices is None:
        devices = jax.devices()
    return Mesh(np.asarray(devices), axis_names=("cores",))


def _topk_smallest(values, k):
    neg_v, idx = jax.lax.top_k(-values, k)
    return -neg_v, idx


def _tail_hit(chosen_states, chosen_h, chosen_is_real, tail_hashes, tail_zobrist,
              state_size):
    """Membership test against the exact endgame table. None when it is disabled.

    ONE implementation shared by every step body -- a second copy of this would be the
    textbook [[dual_codepath_drift]] setup, since a wrong tail lookup still returns
    verified paths and just quietly stops helping.

    Keyed by a full-width Zobrist when `tail_zobrist` is given. The beam's own hash is
    the LINEAR form `sum(s_i * hv_i)`, written for permutation puzzles where s_i spans
    0..119; on a 6-colour cube it occupies only ~2^55.5 and produces ~0.3 phantom hits
    per pid at this width. Zobrist is 2^64.5 measured.
    """
    if tail_hashes is None:
        return None
    if tail_zobrist is not None:
        # TRANSPOSE FIRST, then walk rows. On TPU the last two dims tile as (8, 128), so
        # a tensor with trailing dimension 1 -- which is what `chosen_states[:, i]` is
        # before its reshape -- pads 1 -> 128 lanes: a 128x blowup that turned 8 MB into
        # 1.00 GB per temp and cost ~4 GB across the fold (measured, v11: HLO temp 17.07G
        # at 5.1% utilisation, unpadded 898 MB). Rows of the transpose have the huge axis
        # last, so they tile exactly. The transpose itself is uint8: 96 * B bytes.
        st_t = chosen_states.T                                   # (state_size, B)
        key = jnp.zeros(chosen_states.shape[0], dtype=jnp.int64)
        for i in range(state_size):
            key = jnp.bitwise_xor(
                key, jnp.take(tail_zobrist[i], st_t[i].astype(jnp.int32)))
    else:
        key = chosen_h
    pos = jnp.searchsorted(tail_hashes, key)
    pos_c = jnp.minimum(pos, tail_hashes.shape[0] - 1)
    return ((pos < tail_hashes.shape[0])
            & (jnp.take(tail_hashes, pos_c) == key)
            & chosen_is_real)


def _build_step_body_v_only(
    v_params,
    all_moves, V0, hash_vec, V0_hash,
    B_local, world_size, K_per_peer, n_gen, state_size,
    dtype, internal_bs,
):
    """Per-shard step body for V-only beam search."""
    BIG_F32 = jnp.float32(1e9)
    aB_local = K_per_peer * world_size


    def _chunked_apply(params, x, chunk_size):
        n, S = x.shape
        n_chunks = n // chunk_size
        chunks = x.reshape(n_chunks, chunk_size, S)
        def _scan_fn(_, chunk):
            return _, model_apply(params, chunk, dtype=dtype)
        _, outs = jax.lax.scan(_scan_fn, None, chunks)
        if outs.ndim == 2:
            return outs.reshape(n)
        return outs.reshape(n, -1)

    def forward_v(params, x):
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
        # shard_map keeps sharded axes with local size; squeeze leading shard axis.
        states = states[0]
        tree_parent_local = tree_parent_local[0]
        tree_parent_rank = tree_parent_rank[0]
        tree_move = tree_move[0]
        min_v_log = min_v_log[0]
        found_step = found_step[0]
        found_pos_local = found_pos_local[0]
        found_pos_rank = found_pos_rank[0]
        verify_state = verify_state[0]
        rank_int = jax.lax.axis_index("cores").astype(jnp.int32)

        # 1. Generate all children of owned states.
        neighbors = states[:, all_moves].reshape(-1, state_size)  # (n_total, S)

        # 2. V on ALL children (the V-only step — no qshort prefilter).
        child_v = forward_v(v_params, neighbors)  # (n_total,) float32

        # 3. Hash + owner per child.
        h = jnp.sum(neighbors.astype(jnp.int64) * hash_vec, axis=1)
        owner = (h % jnp.int64(world_size)).astype(jnp.int32)

        # 4. Per-owner top-K_per_peer by V score. Loop over owners is unrolled.
        send_buckets = jnp.zeros((world_size, K_per_peer, PACK_SIZE), dtype=jnp.uint8)
        for S in range(world_size):
            mask_S = (owner == S)
            score_for_S = jnp.where(mask_S, child_v, BIG_F32)
            top_v_S, top_idx_S = _topk_smallest(score_for_S, K_per_peer)
            is_pad_S = top_v_S >= (BIG_F32 * 0.5)

            sel_states = neighbors[top_idx_S]
            sel_parent_local = parent_local_per_child[top_idx_S]
            sel_move = move_per_child[top_idx_S]

            zero_state = jnp.zeros((K_per_peer, state_size), dtype=jnp.uint8)
            zero_int32 = jnp.zeros(K_per_peer, dtype=jnp.int32)
            zero_int8 = jnp.zeros(K_per_peer, dtype=jnp.int8)

            sel_states_u8 = jnp.where(is_pad_S[:, None], zero_state, sel_states.astype(jnp.uint8))
            sel_parent_local_z = jnp.where(is_pad_S, zero_int32, sel_parent_local)
            sel_move_z = jnp.where(is_pad_S, zero_int8, sel_move)

            bucket = jnp.zeros((K_per_peer, PACK_SIZE), dtype=jnp.uint8)
            bucket = bucket.at[:, 0:state_size].set(sel_states_u8)
            bucket = bucket.at[:, 120].set((sel_parent_local_z & 0xFF).astype(jnp.uint8))
            bucket = bucket.at[:, 121].set(((sel_parent_local_z >> 8) & 0xFF).astype(jnp.uint8))
            bucket = bucket.at[:, 122].set(((sel_parent_local_z >> 16) & 0xFF).astype(jnp.uint8))
            bucket = bucket.at[:, 123].set(((sel_parent_local_z >> 24) & 0xFF).astype(jnp.uint8))
            bucket = bucket.at[:, 124].set(sel_move_z.astype(jnp.uint8))
            send_buckets = send_buckets.at[S].set(bucket)

        # 5. all_to_all: bucket S from each sender goes to rank S.
        recv_buckets = jax.lax.all_to_all(
            send_buckets, axis_name="cores",
            split_axis=0, concat_axis=0, tiled=True,
        )

        # 6. Unpack received candidates.
        recv_flat = recv_buckets.reshape(-1, PACK_SIZE)
        recv_states_u8 = recv_flat[:, 0:state_size]
        recv_states = recv_states_u8.astype(jnp.int8)
        recv_parent_local = (
            recv_flat[:, 120].astype(jnp.int32)
            | (recv_flat[:, 121].astype(jnp.int32) << 8)
            | (recv_flat[:, 122].astype(jnp.int32) << 16)
            | (recv_flat[:, 123].astype(jnp.int32) << 24)
        )
        recv_move = recv_flat[:, 124].astype(jnp.int8)
        recv_sender_rank = sender_rank_per_recv

        # Padding detection: a real state has a nonzero sticker sum; padding is all-zero.
        # (megaminx: sum==7140; cube444: sum>=240, 16 each of colors 0..5). == 0 is generic.
        recv_state_sum = jnp.sum(recv_states.astype(jnp.int32), axis=1)
        is_padding = (recv_state_sum == 0)

        # In-rank dedup.
        recv_h = jnp.sum(recv_states.astype(jnp.int64) * hash_vec, axis=1)
        sort_h = jnp.sort(recv_h)
        sort_idx = jnp.argsort(recv_h)
        is_dup_sorted = jnp.concatenate([
            jnp.zeros(1, dtype=jnp.bool_),
            sort_h[1:] == sort_h[:-1],
        ])
        restore = jnp.argsort(sort_idx)
        dup_mask = is_dup_sorted[restore]

        # 7. Re-run V on received states (necessary because we didn't pack V
        # scores into the bucket — keeps PACK_SIZE=128, same as qshort body).
        # aB_local ≈ 2 * B_local is small vs. n_total = 24 * B_local so the
        # extra compute is ~8%.
        recv_v = forward_v(v_params, recv_states)
        recv_v_masked = jnp.where(dup_mask | is_padding, BIG_F32, recv_v)

        top_v_keep, keep_idx = _topk_smallest(recv_v_masked, B_local)

        chosen_states = recv_states[keep_idx]
        chosen_parent_local = recv_parent_local[keep_idx]
        chosen_parent_rank = recv_sender_rank[keep_idx]
        chosen_move = recv_move[keep_idx]
        chosen_h = recv_h[keep_idx]
        chosen_state_sum = jnp.sum(chosen_states.astype(jnp.int32), axis=1)
        chosen_is_real = (chosen_state_sum != 0)

        new_tree_parent_local = tree_parent_local.at[j].set(chosen_parent_local)
        new_tree_parent_rank = tree_parent_rank.at[j].set(chosen_parent_rank)
        new_tree_move = tree_move.at[j].set(chosen_move)
        new_min_v_log = min_v_log.at[j].set(top_v_keep[0])

        # V0 detection (per-rank; cross-rank reduce REMOVED -- host-side instead).
        eq_v0 = (chosen_h == V0_hash) & chosen_is_real
        any_hit = jnp.any(eq_v0)
        pos_hit = jnp.argmax(eq_v0.astype(jnp.int32)).astype(jnp.int32)
        is_first_hit = (found_step == -1) & any_hit
        new_found_step = jnp.where(is_first_hit, j, found_step)
        new_found_pos_local = jnp.where(is_first_hit, pos_hit, found_pos_local)
        new_found_pos_rank = jnp.where(is_first_hit, rank_int, found_pos_rank)
        candidate_state = chosen_states[pos_hit]
        new_verify_state = jnp.where(is_first_hit, candidate_state, verify_state)

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


def beam_solve_v_only_spmd(
    init_state_list: list[int],
    v_params,
    all_moves: jnp.ndarray,
    V0: jnp.ndarray,
    hash_vec: jnp.ndarray,
    mesh: Mesh,
    B_local: int,
    K_per_peer: int,
    n_gen: int = 24,
    state_size: int = 120,
    num_steps: int = 120,
    dtype=jnp.bfloat16,
    internal_bs: int = 32768,
) -> dict[str, Any]:
    """High-level V-only SPMD beam solver."""
    devices = mesh.devices.flatten()
    world_size = len(devices)
    init_state = np.asarray(init_state_list, dtype=np.int8)
    if np.array_equal(init_state, np.asarray(V0)):
        return {"found": True, "path_len": 0, "path_idx": [], "found_step": -1, "wall_s": 0.0}

    # Step 0: V on 24 first-move children (always unchunked).
    init_dev = jnp.asarray(init_state)
    states_seed = jnp.expand_dims(init_dev, 0)
    neighbors0 = states_seed[:, all_moves].reshape(-1, state_size)
    values0 = model_apply(v_params, neighbors0, dtype=dtype).astype(jnp.float32)
    k0 = min(B_local, n_gen)
    _, top_idx0 = _topk_smallest(values0, k0)
    chosen0 = neighbors0[top_idx0]
    if k0 < B_local:
        pad = jnp.broadcast_to(chosen0[-1:], (B_local - k0, state_size))
        states0 = jnp.concatenate([chosen0, pad], axis=0)
    else:
        states0 = chosen0[:B_local]

    states_global = jnp.broadcast_to(states0[None, :, :], (world_size, B_local, state_size))
    tree_parent_local = jnp.zeros((world_size, num_steps, B_local), dtype=jnp.int32)
    tree_parent_rank = jnp.zeros((world_size, num_steps, B_local), dtype=jnp.int8)
    tree_move = jnp.full((world_size, num_steps, B_local), -1, dtype=jnp.int8)

    top_idx0_i8 = top_idx0.astype(jnp.int8)
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

    eq0 = jnp.all(states0 == V0, axis=1)
    any0 = jnp.any(eq0)
    pos0 = jnp.argmax(eq0.astype(jnp.int32)).astype(jnp.int32)
    if bool(any0):
        pos0_h = int(pos0)
        if pos0_h < k0:
            seed_move = int(top_idx0[pos0_h])
        else:
            seed_move = int(top_idx0[k0 - 1])
        return {"found": True, "path_len": 1, "path_idx": [seed_move],
                "found_step": 0, "wall_s": 0.0}

    V0_hash_host = int(np.sum(np.asarray(V0).astype(np.int64) * np.asarray(hash_vec)))

    step_body = _build_step_body_v_only(
        v_params, all_moves, V0, hash_vec, jnp.int64(V0_hash_host),
        B_local, world_size, K_per_peer, n_gen, state_size,
        dtype, int(internal_bs),
    )

    @jax.jit
    def step_fn(states, tp_local, tp_rank, tmove, mv_log, fs, fpl, fpr, vstate, j_arr):
        try:
            from jax.experimental.shard_map import shard_map
        except ImportError:
            from jax import shard_map
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
        j_arr = jnp.int32(j)
        (states_d, tp_local_d, tp_rank_d, tmove_d,
         mv_log_d, fs_d, fpl_d, fpr_d, vstate_d) = step_fn(
            states_d, tp_local_d, tp_rank_d, tmove_d,
            mv_log_d, fs_d, fpl_d, fpr_d, vstate_d, j_arr,
        )
        if first_iter_t is None:
            jax.block_until_ready(fs_d)
            first_iter_t = time.time() - t_iter

    jax.block_until_ready(fs_d)

    # Host-side cross-rank winner selection.
    fs_per_rank = np.asarray(fs_d)
    fpl_per_rank = np.asarray(fpl_d)
    fpr_per_rank = np.asarray(fpr_d)

    INT_MAX = 2 ** 30
    fs_signed = np.where(fs_per_rank >= 0, fs_per_rank, INT_MAX)
    global_min_step = int(fs_signed.min())
    if global_min_step >= INT_MAX:
        return {"found": False, "path_len": 0, "path_idx": [],
                "found_step": -1, "wall_s": time.time() - t_start,
                "first_iter_s": first_iter_t}

    winner_ranks = np.where(fs_signed == global_min_step)[0]
    winner_rank = int(winner_ranks[0])
    fs_h = int(fs_per_rank[winner_rank])
    fpl_h = int(fpl_per_rank[winner_rank])
    fpr_h = winner_rank

    tp_local_h = np.asarray(tp_local_d)
    tp_rank_h = np.asarray(tp_rank_d)
    tmove_h = np.asarray(tmove_d)

    path_idx = []
    cur_rank = fpr_h
    cur_pos = fpl_h
    for j in range(fs_h, -1, -1):
        m = int(tmove_h[cur_rank, j, cur_pos])
        if m >= 0:
            path_idx.append(m)
        if j > 0:
            new_rank = int(tp_rank_h[cur_rank, j, cur_pos])
            new_pos = int(tp_local_h[cur_rank, j, cur_pos])
            cur_rank = new_rank
            cur_pos = new_pos
    path_idx.reverse()
    while path_idx and path_idx[0] < 0:
        path_idx.pop(0)

    return {
        "found": True,
        "path_len": len(path_idx),
        "path_idx": path_idx,
        "found_step": fs_h,
        "found_pos_local": fpl_h,
        "found_pos_rank": fpr_h,
        "wall_s": time.time() - t_start,
        "first_iter_s": first_iter_t,
    }


# =============================================================================
# Phase A+B variant: packed uint64 backpointers, host memmap tree, early stop.
# =============================================================================

def _build_step_body_v_only_packed(
    v_params,
    all_moves, V0, hash_vec, V0_hash,
    B_local, world_size, K_per_peer, n_gen, state_size,
    dtype, internal_bs,
    pack_v_score=False,
    owner_hash_vec=None,
    trace_prefix_hashes=None,
    q_score_fn=None,
    tail_hashes=None,
    tail_zobrist=None,
):
    """Per-shard V-only step body, Phase A+B (+ optional D) variant.

    `q_score_fn` (Q-native): if given, scores come from a Q head applied to the
    B_local PARENTS instead of a V head applied to all 24*B_local children. The
    child at flat index i is (parent = i // n_gen, move = i % n_gen), which is
    exactly the row-major layout of a (B_local, n_gen) Q output -- so
    `q_score_fn(states).reshape(-1)` aligns with `neighbors` elementwise with no
    permutation, and every downstream stage (owner routing, per-owner top-K,
    all_to_all, dedup, backptr) is untouched. 24x fewer model forwards, which is
    what makes a 57-token transformer affordable at width. REQUIRES
    pack_v_score=True: a received child's Q value lives on the SENDING rank, so
    the receive-side re-forward has to be skipped and the score carried in the
    bucket.

    `pack_v_score` (Phase D): if True, pack the send-side bf16 V score into
    bucket bytes 125-126 and skip the receive-side forward_v re-run. Saves
    ~8% V compute (the receive-side forward on aB_local states). Quality:
    bf16 ordering may tie-break differently than fp32 -- expect path-length
    drift of 0-2 moves per pid.

    Differences from `_build_step_body_v_only`:
      * Phase A: precomputed parent_local_per_child / move_per_child arrays
        removed. Per-owner parent/move derived from `top_idx_S` via integer
        div/mod after the topk. Saves ~B_local*n_gen*(4+1) bytes per rank
        (180 MB at B_local=4M).
      * Phase B: tree_* tensors removed from the carry. The body now emits a
        single packed uint64 backpointer per beam slot (shape (B_local,)):
          bits  0..24: parent_local (25 bits; supports b_local up to 2^25 = 33.55M)
          bits 25..27: parent_rank  (0..7)
          bits 28..32: move         (0..23, 5 bits)
        This supports 256M global on 8 chips. 128M would fit in 24/3/5 uint32,
        but 256M needs the extra parent bit.

    Body outputs (in shard-local shape, with leading rank axis re-added):
      (chosen_states, packed_backptr, min_v_log, found_step,
       found_pos_local, found_pos_rank, verify_state)

    The wrapper writes packed_backptr into a host np.memmap each step.
    Walkback unpacks records and follows (parent_rank, parent_local) chains.
    """
    BIG_F32 = jnp.float32(1e9)
    aB_local = K_per_peer * world_size
    if q_score_fn is not None and not pack_v_score:
        # Without packing, the receive side re-runs the scorer on states it did not
        # generate -- but a Q value is indexed by (parent, action) and the parent
        # lives on the sender. Silently scoring the wrong thing is exactly the
        # failure mode this guard exists to prevent.
        raise ValueError("q_score_fn requires pack_v_score=True")
    trace_enabled = trace_prefix_hashes is not None
    if trace_enabled:
        trace_prefix_hashes = jnp.asarray(trace_prefix_hashes, dtype=jnp.int64)

    def _chunked_apply(params, x, chunk_size):
        n, S = x.shape
        n_chunks = n // chunk_size
        chunks = x.reshape(n_chunks, chunk_size, S)
        def _scan_fn(_, chunk):
            return _, model_apply(params, chunk, dtype=dtype)
        _, outs = jax.lax.scan(_scan_fn, None, chunks)
        if outs.ndim == 2:
            return outs.reshape(n)
        return outs.reshape(n, -1)

    def forward_v(params, x):
        return _chunked_apply(params, x, internal_bs).astype(jnp.float32)

    def forward_q(x):
        """Chunked Q forward -> (n, n_gen) float32.

        Same lax.scan chunking as forward_v. Without it the transformer's
        (B, 56, 3, d_model) per-slot embedding blows HBM at shard width.
        """
        n, S = x.shape
        cs = min(int(internal_bs), n)
        if n % cs:
            raise ValueError(
                f"internal_bs ({cs}) must divide the shard width ({n})")
        _, outs = jax.lax.scan(
            lambda _c, chunk: (_c, q_score_fn(chunk)), None,
            x.reshape(n // cs, cs, S))
        return outs.reshape(n, -1).astype(jnp.float32)

    # Phase A: full per-child index arrays dropped. Only receive-side
    # sender-rank index remains (computed once, ~aB_local bytes int8).
    arange_aB = jnp.arange(aB_local, dtype=jnp.int32)
    sender_rank_per_recv = (arange_aB // K_per_peer).astype(jnp.int8)

    def beam_step_local(states, min_v_log, found_step, found_pos_local,
                        found_pos_rank, verify_state, j):
        # Squeeze leading shard axis (shard_map gives each rank shape (1, ...)).
        states = states[0]
        min_v_log = min_v_log[0]
        found_step = found_step[0]
        found_pos_local = found_pos_local[0]
        found_pos_rank = found_pos_rank[0]
        verify_state = verify_state[0]
        rank_int = jax.lax.axis_index("cores").astype(jnp.int32)

        # 1. Generate all children of owned states.
        neighbors = states[:, all_moves].reshape(-1, state_size)

        # 2. Child scores.
        if q_score_fn is None:
            # V-only: one forward per child (24 * B_local), chunked via lax.scan.
            child_v = forward_v(v_params, neighbors)
        else:
            # Q-native: one forward per PARENT (B_local), 24 scores each. Row-major
            # (B_local, n_gen) matches `neighbors`' (parent*n_gen + move) order exactly.
            # MUST be chunked like forward_v: the PieceTransformer's per-slot embedding
            # is (B, 56, 3, d_model), i.e. 168 rows per state, so an unchunked shard at
            # B_local=524,288 asks for 45 GB against 16 GB of HBM.
            child_v = forward_q(states).reshape(-1)

        # 3. Owner hash. Owner routing only needs uniform partitioning, not
        # collision resistance -- use cheaper uint32 hash if provided.
        # world_size is a power of 2 (8 on v5e-8) so AND-mask is cheaper than mod.
        if owner_hash_vec is not None:
            h_owner = jnp.sum(neighbors.astype(jnp.uint32) * owner_hash_vec, axis=1)
            owner = (h_owner & jnp.uint32(world_size - 1)).astype(jnp.int32)
        else:
            h = jnp.sum(neighbors.astype(jnp.int64) * hash_vec, axis=1)
            owner = (h % jnp.int64(world_size)).astype(jnp.int32)

        # 4. Per-owner top-K_per_peer by V score (loop unrolled in JIT).
        send_buckets = jnp.zeros((world_size, K_per_peer, PACK_SIZE), dtype=jnp.uint8)
        for S in range(world_size):
            mask_S = (owner == S)
            score_for_S = jnp.where(mask_S, child_v, BIG_F32)
            top_v_S, top_idx_S = _topk_smallest(score_for_S, K_per_peer)
            is_pad_S = top_v_S >= (BIG_F32 * 0.5)

            sel_states = neighbors[top_idx_S]
            # Phase A: derive parent_local / move from top_idx_S on demand.
            sel_parent_local = (top_idx_S // n_gen).astype(jnp.int32)
            sel_move = (top_idx_S % n_gen).astype(jnp.int8)

            zero_state = jnp.zeros((K_per_peer, state_size), dtype=jnp.uint8)
            zero_int32 = jnp.zeros(K_per_peer, dtype=jnp.int32)
            zero_int8 = jnp.zeros(K_per_peer, dtype=jnp.int8)

            sel_states_u8 = jnp.where(is_pad_S[:, None], zero_state, sel_states.astype(jnp.uint8))
            sel_parent_local_z = jnp.where(is_pad_S, zero_int32, sel_parent_local)
            sel_move_z = jnp.where(is_pad_S, zero_int8, sel_move)

            bucket = jnp.zeros((K_per_peer, PACK_SIZE), dtype=jnp.uint8)
            bucket = bucket.at[:, 0:state_size].set(sel_states_u8)
            bucket = bucket.at[:, 120].set((sel_parent_local_z & 0xFF).astype(jnp.uint8))
            bucket = bucket.at[:, 121].set(((sel_parent_local_z >> 8) & 0xFF).astype(jnp.uint8))
            bucket = bucket.at[:, 122].set(((sel_parent_local_z >> 16) & 0xFF).astype(jnp.uint8))
            bucket = bucket.at[:, 123].set(((sel_parent_local_z >> 24) & 0xFF).astype(jnp.uint8))
            bucket = bucket.at[:, 124].set(sel_move_z.astype(jnp.uint8))
            if pack_v_score:
                # Phase D: pack bf16 V score into bytes 125-126.
                top_v_S_bf16 = top_v_S.astype(jnp.bfloat16)
                top_v_S_u16 = jax.lax.bitcast_convert_type(top_v_S_bf16, jnp.uint16)
                bucket = bucket.at[:, 125].set((top_v_S_u16 & 0xFF).astype(jnp.uint8))
                bucket = bucket.at[:, 126].set(((top_v_S_u16 >> 8) & 0xFF).astype(jnp.uint8))
            send_buckets = send_buckets.at[S].set(bucket)

        # 5. all_to_all: bucket S from each sender goes to rank S.
        recv_buckets = jax.lax.all_to_all(
            send_buckets, axis_name="cores",
            split_axis=0, concat_axis=0, tiled=True,
        )

        # 6. Unpack received candidates.
        recv_flat = recv_buckets.reshape(-1, PACK_SIZE)
        recv_states_u8 = recv_flat[:, 0:state_size]
        recv_states = recv_states_u8.astype(jnp.int8)
        recv_parent_local = (
            recv_flat[:, 120].astype(jnp.int32)
            | (recv_flat[:, 121].astype(jnp.int32) << 8)
            | (recv_flat[:, 122].astype(jnp.int32) << 16)
            | (recv_flat[:, 123].astype(jnp.int32) << 24)
        )
        recv_move = recv_flat[:, 124].astype(jnp.int8)
        recv_sender_rank = sender_rank_per_recv

        # Padding detection: a real state has a nonzero sticker sum; padding is all-zero.
        # (megaminx: sum==7140; cube444: sum>=240, 16 each of colors 0..5). == 0 is generic.
        recv_state_sum = jnp.sum(recv_states.astype(jnp.int32), axis=1)
        is_padding = (recv_state_sum == 0)

        # In-rank dedup via sorted-order traversal (one argsort + gathers).
        # Top-k doesn't care whether candidates are in original receive order,
        # so we process everything in sorted layout and drop the second argsort
        # (the inverse-permutation "restore" that the prior version needed).
        recv_h = jnp.sum(recv_states.astype(jnp.int64) * hash_vec, axis=1)
        sort_idx = jnp.argsort(recv_h)
        sorted_h = recv_h[sort_idx]
        sorted_states = recv_states[sort_idx]
        sorted_parent_local = recv_parent_local[sort_idx]
        sorted_move = recv_move[sort_idx]
        sorted_sender_rank = recv_sender_rank[sort_idx]
        sorted_is_padding = is_padding[sort_idx]
        is_dup_sorted = jnp.concatenate([
            jnp.zeros(1, dtype=jnp.bool_),
            sorted_h[1:] == sorted_h[:-1],
        ])

        # 7. Re-run V on received states -- OR unpack the packed score (Phase D).
        if pack_v_score:
            recv_v_u16 = (
                recv_flat[:, 125].astype(jnp.uint16)
                | (recv_flat[:, 126].astype(jnp.uint16) << jnp.uint16(8))
            )
            recv_v_bf16 = jax.lax.bitcast_convert_type(recv_v_u16, jnp.bfloat16)
            recv_v = recv_v_bf16.astype(jnp.float32)
        else:
            recv_v = forward_v(v_params, recv_states)
        sorted_v = recv_v[sort_idx]
        sorted_v_masked = jnp.where(is_dup_sorted | sorted_is_padding, BIG_F32, sorted_v)

        top_v_keep, keep_sorted_idx = _topk_smallest(sorted_v_masked, B_local)

        chosen_states = sorted_states[keep_sorted_idx]
        chosen_parent_local = sorted_parent_local[keep_sorted_idx]
        chosen_parent_rank = sorted_sender_rank[keep_sorted_idx]
        chosen_move = sorted_move[keep_sorted_idx]
        chosen_h = sorted_h[keep_sorted_idx]
        chosen_state_sum = jnp.sum(chosen_states.astype(jnp.int32), axis=1)
        chosen_is_real = (chosen_state_sum != 0)

        # Phase B: pack backpointer into a single uint64 per beam slot.
        packed_backptr = (
            (chosen_parent_local.astype(jnp.uint64) & jnp.uint64(BPTR_PL_MASK))
            | (chosen_parent_rank.astype(jnp.uint64) << jnp.uint64(BPTR_RANK_SHIFT))
            | (chosen_move.astype(jnp.uint64) << jnp.uint64(BPTR_MOVE_SHIFT))
        )

        new_min_v_log = min_v_log.at[j].set(top_v_keep[0])

        if trace_enabled:
            trace_hash = trace_prefix_hashes[j]
            eq_trace = (chosen_h == trace_hash) & chosen_is_real
            trace_any = jnp.any(eq_trace)
            trace_pos = jnp.argmax(eq_trace.astype(jnp.int32)).astype(jnp.int32)
            trace_hit = trace_any.astype(jnp.int32)
            trace_v = jnp.where(trace_any, top_v_keep[trace_pos], BIG_F32)
            trace_cutoff = top_v_keep[-1]
        else:
            trace_hit = jnp.int32(0)
            trace_pos = jnp.int32(-1)
            trace_v = BIG_F32
            trace_cutoff = BIG_F32

        # V0 detection (per-rank; cross-rank reduce REMOVED -- host-side instead).
        eq_v0 = (chosen_h == V0_hash) & chosen_is_real
        _hit = _tail_hit(chosen_states, chosen_h, chosen_is_real, tail_hashes,
                         tail_zobrist, state_size)
        if _hit is not None:
            # An exact tail can only SHORTEN a path, and terminating here is where most
            # of the wall-clock saving comes from. The host decodes the stored tail from
            # verify_state and REPLAYS it before accepting.
            eq_v0 = eq_v0 | _hit
        any_hit = jnp.any(eq_v0)
        pos_hit = jnp.argmax(eq_v0.astype(jnp.int32)).astype(jnp.int32)
        is_first_hit = (found_step == -1) & any_hit
        new_found_step = jnp.where(is_first_hit, j, found_step)
        new_found_pos_local = jnp.where(is_first_hit, pos_hit, found_pos_local)
        new_found_pos_rank = jnp.where(is_first_hit, rank_int, found_pos_rank)
        candidate_state = chosen_states[pos_hit]
        new_verify_state = jnp.where(is_first_hit, candidate_state, verify_state)

        return (chosen_states[None, :, :],
                packed_backptr[None, :],
                new_min_v_log[None, :],
                new_found_step[None],
                new_found_pos_local[None],
                new_found_pos_rank[None],
                new_verify_state[None, :],
                trace_hit[None],
                trace_pos[None],
                trace_v[None],
                trace_cutoff[None])

    return beam_step_local


# -----------------------------------------------------------------------------
# Phase C: streaming child generation via lax.scan over parent chunks.
# Removes the 11.5 GiB neighbors materialization at B_local=4M (32M global).
# Same packed-backpointer output as the non-streaming body, so the wrapper
# can dispatch to either body via the `parent_chunk` parameter.
# -----------------------------------------------------------------------------

def _build_step_body_v_only_packed_streaming(
    v_params,
    all_moves, V0, hash_vec, V0_hash,
    B_local, world_size, K_per_peer, n_gen, state_size,
    dtype, internal_bs,
    parent_chunk,
    pack_v_score=False,
    owner_hash_vec=None,
    trace_prefix_hashes=None,
    q_score_fn=None,
    tail_hashes=None,
    tail_zobrist=None,
):
    """Per-shard V-only step body, Phase A+B+C (+ optional D): streamed.

    `pack_v_score` (Phase D): same as in the non-streaming body -- pack send-side
    score into bytes 125-126, skip receive-side forward_v. Quality budget:
    0-2 moves per pid from bf16 tie-break drift.

    Replaces `_build_step_body_v_only_packed`'s full-neighbors materialization
    (B_local * n_gen * state_size bytes; 11.5 GiB at B_local=4M) with a
    `lax.scan` over parent chunks. Per chunk:
      * generate (parent_chunk * n_gen, state_size) children
      * run V on the chunk's children (chunked internally by internal_bs)
      * route to owner buckets, merge with per-owner running top-K_per_peer
    Final per-rank send_buckets = (world_size, K_per_peer, PACK_SIZE) uint8.

    Everything downstream of the all_to_all is identical to the non-streaming
    body. Backpointer is the same packed uint64 format.

    `parent_chunk` MUST divide B_local. Typical value at 32M: 65536.
    """
    BIG_F32 = jnp.float32(1e9)
    aB_local = K_per_peer * world_size
    trace_enabled = trace_prefix_hashes is not None
    if trace_enabled:
        trace_prefix_hashes = jnp.asarray(trace_prefix_hashes, dtype=jnp.int64)
    assert B_local % parent_chunk == 0, (
        f"parent_chunk {parent_chunk} must divide B_local {B_local}"
    )
    n_chunks = B_local // parent_chunk
    chunk_n = parent_chunk * n_gen

    def _chunked_apply(params, x, chunk_size):
        n, S = x.shape
        n_chunks_inner = n // chunk_size
        chunks = x.reshape(n_chunks_inner, chunk_size, S)
        def _scan_fn(_, c):
            return _, model_apply(params, c, dtype=dtype)
        _, outs = jax.lax.scan(_scan_fn, None, chunks)
        if outs.ndim == 2:
            return outs.reshape(n)
        return outs.reshape(n, -1)

    def forward_v(params, x):
        return _chunked_apply(params, x, internal_bs).astype(jnp.float32)

    def forward_q(x):
        # Same lax.scan chunking as forward_v: the model's activation budget is set by
        # internal_bs, NOT by parent_chunk.
        n, S = x.shape
        cs = min(int(internal_bs), n)
        if n % cs:
            raise ValueError(f"internal_bs ({cs}) must divide parent_chunk ({n})")
        _, outs = jax.lax.scan(
            lambda _c, chunk: (_c, q_score_fn(chunk)), None, x.reshape(n // cs, cs, S))
        return outs.reshape(n, -1).astype(jnp.float32)

    # Receive-side sender_rank index (static, small).
    arange_aB = jnp.arange(aB_local, dtype=jnp.int32)
    sender_rank_per_recv = (arange_aB // K_per_peer).astype(jnp.int8)

    def beam_step_local(states, min_v_log, found_step, found_pos_local,
                        found_pos_rank, verify_state, j):
        # Squeeze leading shard axis.
        states = states[0]
        min_v_log = min_v_log[0]
        found_step = found_step[0]
        found_pos_local = found_pos_local[0]
        found_pos_rank = found_pos_rank[0]
        verify_state = verify_state[0]
        rank_int = jax.lax.axis_index("cores").astype(jnp.int32)

        # Streaming scan carry: per-owner running top-K_per_peer.
        # Initial: all-BIG_F32 scores, all-zero packs. Any real child wins.
        # Inside shard_map's manual mode, lax.scan requires the carry to be
        # 'varying' along the sharded axis. Constants from jnp.full/jnp.zeros
        # are replicated by default; pcast them to 'varying' so the
        # carry-in/carry-out types match.
        init_top_scores = jax.lax.pcast(
            jnp.full((world_size, K_per_peer), BIG_F32, dtype=jnp.float32),
            ("cores",), to="varying",
        )
        init_top_pack = jax.lax.pcast(
            jnp.zeros((world_size, K_per_peer, PACK_SIZE), dtype=jnp.uint8),
            ("cores",), to="varying",
        )

        def chunk_body(carry, chunk_i):
            top_scores, top_pack = carry
            parent_start = chunk_i * jnp.int32(parent_chunk)
            states_chunk = jax.lax.dynamic_slice(
                states, (parent_start, jnp.int32(0)),
                (parent_chunk, state_size),
            )
            children = states_chunk[:, all_moves].reshape(-1, state_size)  # (chunk_n, S)

            # Chunk scores. Q-native reads all 24 action values off ONE forward per
            # parent; `children` is (parent*n_gen + move) row-major, exactly the layout
            # of a (parent_chunk, n_gen) Q output, so reshape(-1) aligns elementwise.
            #
            # MUST be chunked by internal_bs. parent_chunk is sized for the CHILDREN
            # tensor (parent_chunk * n_gen * state_size), which is a different budget
            # from the model's activations: at parent_chunk=65536 in fp32 the 57-token
            # transformer's feed-forward intermediate alone is 65536*57*1024*4 = 15.3 GB
            # per layer, and the whole forward asks for ~47 GB against 15.75 GB of HBM.
            if q_score_fn is None:
                child_v = forward_v(v_params, children)
            else:
                child_v = forward_q(states_chunk).reshape(-1).astype(jnp.float32)

            # Owner partition. Owner routing only needs a uniform 0..world_size-1
            # bucket; use cheaper uint32 hash if provided. world_size is a power
            # of 2 (8 on v5e-8), so AND-mask is cheaper than mod.
            if owner_hash_vec is not None:
                h_owner = jnp.sum(children.astype(jnp.uint32) * owner_hash_vec, axis=1)
                owner = (h_owner & jnp.uint32(world_size - 1)).astype(jnp.int32)
            else:
                h = jnp.sum(children.astype(jnp.int64) * hash_vec, axis=1)
                owner = (h % jnp.int64(world_size)).astype(jnp.int32)

            new_top_scores = top_scores
            new_top_pack = top_pack
            for dest in range(world_size):
                # Mask children not owned by `dest`.
                masked_scores = jnp.where(owner == dest, child_v, BIG_F32)
                # Merge running top with this chunk's masked candidates.
                merged_scores = jnp.concatenate(
                    [new_top_scores[dest], masked_scores], axis=0,
                )  # (K_per_peer + chunk_n,) float32
                top_v_new, keep = _topk_smallest(merged_scores, K_per_peer)

                # Each kept slot is either from old (keep < K_per_peer) or new.
                from_old = keep < K_per_peer
                old_idx = jnp.clip(keep, 0, K_per_peer - 1)
                new_idx = jnp.clip(keep - K_per_peer, 0, chunk_n - 1)

                # Old pack: previous top_pack[dest] indexed by old_idx.
                old_pack = new_top_pack[dest][old_idx]  # (K_per_peer, PACK_SIZE)

                # New pack: pack chunk children at new_idx.
                sel_states = children[new_idx]  # (K_per_peer, state_size) int8
                sel_parent_local = (parent_start + (new_idx // n_gen)).astype(jnp.int32)
                sel_move = (new_idx % n_gen).astype(jnp.int8)

                # Padding marker (a kept slot is pad iff its score is BIG_F32).
                is_pad = top_v_new >= (BIG_F32 * 0.5)
                zero_state = jnp.zeros((K_per_peer, state_size), dtype=jnp.uint8)
                zero_int32 = jnp.zeros(K_per_peer, dtype=jnp.int32)
                zero_int8 = jnp.zeros(K_per_peer, dtype=jnp.int8)
                sel_states_u8 = jnp.where(is_pad[:, None], zero_state, sel_states.astype(jnp.uint8))
                sel_parent_local_z = jnp.where(is_pad, zero_int32, sel_parent_local)
                sel_move_z = jnp.where(is_pad, zero_int8, sel_move)

                new_pack = jnp.zeros((K_per_peer, PACK_SIZE), dtype=jnp.uint8)
                new_pack = new_pack.at[:, 0:state_size].set(sel_states_u8)
                new_pack = new_pack.at[:, 120].set((sel_parent_local_z & 0xFF).astype(jnp.uint8))
                new_pack = new_pack.at[:, 121].set(((sel_parent_local_z >> 8) & 0xFF).astype(jnp.uint8))
                new_pack = new_pack.at[:, 122].set(((sel_parent_local_z >> 16) & 0xFF).astype(jnp.uint8))
                new_pack = new_pack.at[:, 123].set(((sel_parent_local_z >> 24) & 0xFF).astype(jnp.uint8))
                new_pack = new_pack.at[:, 124].set(sel_move_z.astype(jnp.uint8))

                # Select between old_pack and new_pack per kept slot.
                merged_pack = jnp.where(from_old[:, None], old_pack, new_pack)

                if pack_v_score:
                    # Phase D: pack bf16 score (post-topk) into bytes 125-126.
                    # top_v_new[i] is the right score regardless of from_old/new.
                    top_v_bf16 = top_v_new.astype(jnp.bfloat16)
                    top_v_u16 = jax.lax.bitcast_convert_type(top_v_bf16, jnp.uint16)
                    merged_pack = merged_pack.at[:, 125].set((top_v_u16 & 0xFF).astype(jnp.uint8))
                    merged_pack = merged_pack.at[:, 126].set(((top_v_u16 >> 8) & 0xFF).astype(jnp.uint8))

                new_top_scores = new_top_scores.at[dest].set(top_v_new)
                new_top_pack = new_top_pack.at[dest].set(merged_pack)

            return (new_top_scores, new_top_pack), None

        (_, final_top_pack), _ = jax.lax.scan(
            chunk_body, (init_top_scores, init_top_pack),
            jnp.arange(n_chunks, dtype=jnp.int32),
        )

        send_buckets = final_top_pack  # (world_size, K_per_peer, PACK_SIZE)

        # all_to_all + receive side (identical to non-streaming body).
        recv_buckets = jax.lax.all_to_all(
            send_buckets, axis_name="cores",
            split_axis=0, concat_axis=0, tiled=True,
        )

        recv_flat = recv_buckets.reshape(-1, PACK_SIZE)
        recv_states_u8 = recv_flat[:, 0:state_size]
        recv_states = recv_states_u8.astype(jnp.int8)
        recv_parent_local = (
            recv_flat[:, 120].astype(jnp.int32)
            | (recv_flat[:, 121].astype(jnp.int32) << 8)
            | (recv_flat[:, 122].astype(jnp.int32) << 16)
            | (recv_flat[:, 123].astype(jnp.int32) << 24)
        )
        recv_move = recv_flat[:, 124].astype(jnp.int8)
        recv_sender_rank = sender_rank_per_recv

        recv_state_sum = jnp.sum(recv_states.astype(jnp.int32), axis=1)
        is_padding = (recv_state_sum == 0)

        # In-rank dedup via sorted-order traversal (one argsort + gathers).
        recv_h = jnp.sum(recv_states.astype(jnp.int64) * hash_vec, axis=1)
        sort_idx = jnp.argsort(recv_h)
        sorted_h = recv_h[sort_idx]
        sorted_states = recv_states[sort_idx]
        sorted_parent_local = recv_parent_local[sort_idx]
        sorted_move = recv_move[sort_idx]
        sorted_sender_rank = recv_sender_rank[sort_idx]
        sorted_is_padding = is_padding[sort_idx]
        is_dup_sorted = jnp.concatenate([
            jnp.zeros(1, dtype=jnp.bool_),
            sorted_h[1:] == sorted_h[:-1],
        ])

        # Re-run V on received states -- OR unpack the packed score (Phase D).
        if pack_v_score:
            recv_v_u16 = (
                recv_flat[:, 125].astype(jnp.uint16)
                | (recv_flat[:, 126].astype(jnp.uint16) << jnp.uint16(8))
            )
            recv_v_bf16 = jax.lax.bitcast_convert_type(recv_v_u16, jnp.bfloat16)
            recv_v = recv_v_bf16.astype(jnp.float32)
        else:
            recv_v = forward_v(v_params, recv_states)
        sorted_v = recv_v[sort_idx]
        sorted_v_masked = jnp.where(is_dup_sorted | sorted_is_padding, BIG_F32, sorted_v)

        top_v_keep, keep_sorted_idx = _topk_smallest(sorted_v_masked, B_local)

        chosen_states = sorted_states[keep_sorted_idx]
        chosen_parent_local = sorted_parent_local[keep_sorted_idx]
        chosen_parent_rank = sorted_sender_rank[keep_sorted_idx]
        chosen_move = sorted_move[keep_sorted_idx]
        chosen_h = sorted_h[keep_sorted_idx]
        chosen_state_sum = jnp.sum(chosen_states.astype(jnp.int32), axis=1)
        chosen_is_real = (chosen_state_sum != 0)

        packed_backptr = (
            (chosen_parent_local.astype(jnp.uint64) & jnp.uint64(BPTR_PL_MASK))
            | (chosen_parent_rank.astype(jnp.uint64) << jnp.uint64(BPTR_RANK_SHIFT))
            | (chosen_move.astype(jnp.uint64) << jnp.uint64(BPTR_MOVE_SHIFT))
        )

        new_min_v_log = min_v_log.at[j].set(top_v_keep[0])

        if trace_enabled:
            trace_hash = trace_prefix_hashes[j]
            eq_trace = (chosen_h == trace_hash) & chosen_is_real
            trace_any = jnp.any(eq_trace)
            trace_pos = jnp.argmax(eq_trace.astype(jnp.int32)).astype(jnp.int32)
            trace_hit = trace_any.astype(jnp.int32)
            trace_v = jnp.where(trace_any, top_v_keep[trace_pos], BIG_F32)
            trace_cutoff = top_v_keep[-1]
        else:
            trace_hit = jnp.int32(0)
            trace_pos = jnp.int32(-1)
            trace_v = BIG_F32
            trace_cutoff = BIG_F32

        eq_v0 = (chosen_h == V0_hash) & chosen_is_real
        _hit = _tail_hit(chosen_states, chosen_h, chosen_is_real, tail_hashes,
                         tail_zobrist, state_size)
        if _hit is not None:
            eq_v0 = eq_v0 | _hit
        any_hit = jnp.any(eq_v0)
        pos_hit = jnp.argmax(eq_v0.astype(jnp.int32)).astype(jnp.int32)
        is_first_hit = (found_step == -1) & any_hit
        new_found_step = jnp.where(is_first_hit, j, found_step)
        new_found_pos_local = jnp.where(is_first_hit, pos_hit, found_pos_local)
        new_found_pos_rank = jnp.where(is_first_hit, rank_int, found_pos_rank)
        candidate_state = chosen_states[pos_hit]
        new_verify_state = jnp.where(is_first_hit, candidate_state, verify_state)

        return (chosen_states[None, :, :],
                packed_backptr[None, :],
                new_min_v_log[None, :],
                new_found_step[None],
                new_found_pos_local[None],
                new_found_pos_rank[None],
                new_verify_state[None, :],
                trace_hit[None],
                trace_pos[None],
                trace_v[None],
                trace_cutoff[None])

    return beam_step_local


def beam_solve_v_only_spmd_packed(
    init_state_list: list[int],
    v_params,
    all_moves: jnp.ndarray,
    V0: jnp.ndarray,
    hash_vec: jnp.ndarray,
    mesh: Mesh,
    B_local: int,
    K_per_peer: int,
    n_gen: int = 24,
    state_size: int = 120,
    num_steps: int = 120,
    dtype=jnp.bfloat16,
    internal_bs: int = 32768,
    tree_path: str | None = None,
    parent_chunk: int | None = None,
    pack_v_score: bool = False,
    progress_every: int = 0,
    owner_hash_vec: jnp.ndarray | None = None,
    trace_prefix_hashes: np.ndarray | None = None,
    stop_on_trace_drop: bool = False,
    q_score_fn=None,
    tail_hashes=None,
    tail_zobrist=None,
) -> dict[str, Any]:
    """V-only SPMD solver, Phase A+B (+ optional C streaming, optional D V-packing):
    packed host-memmap tree + early stop.

    `parent_chunk`:
      * None (default): use the non-streaming body (full neighbors materialized).
        Safe up to B_GLOBAL=16M on v5e-8 with Phase A+B.
      * int (e.g. 65536): use the Phase C streaming body. `parent_chunk` must
        divide B_local. Required for B_GLOBAL >= 32M on v5e-8.

    `pack_v_score` (Phase D):
      * False (default): receive side re-runs V on packed candidates.
      * True: send side packs the bf16 V score into bucket bytes 125-126;
        receive side unpacks and skips the V forward. Saves ~8% V compute.
        bf16 tie-breaks may shift path lengths by 0-2 moves per pid.

    Differences from `beam_solve_v_only_spmd`:
      * Per-step backpointer is a packed uint64 emitted by the body and
        written to a host np.memmap (shape (num_steps, world_size, B_local)).
        Saves ~world_size*num_steps*B_local*6 bytes of device HBM
        (1.44 GB at B_local=2M, 2.88 GB at B_local=4M).
      * Wrapper exits the iter loop as soon as any rank locally detects V0
        (saves both host writes and TPU compute on remaining steps).
      * step_fn re-enables `donate_argnums` for the 6-tensor carry. JAX
        aliases each donated input buffer to its same-shaped output, saving
        another ~1.5 GB of HBM at B_local=2M.
      * Tree file is auto-deleted in a `finally` block.

    Memmap default location: /kaggle/working (Kaggle TPU scratch, ~20 GB),
    else `tempfile.gettempdir()`. Caller may override via `tree_path`.
    File size = num_steps * world_size * B_local * 8 bytes because the packed
    tree is uint64. At 256M global / 8 ranks / 100 steps this is about 200 GiB,
    so use v6e host RAM or a large scratch disk, not Kaggle's small working dir.
    """
    devices = mesh.devices.flatten()
    world_size = len(devices)
    init_state = np.asarray(init_state_list, dtype=np.int8)
    if trace_prefix_hashes is not None:
        trace_prefix_hashes = np.asarray(trace_prefix_hashes, dtype=np.int64)
        if trace_prefix_hashes.ndim != 1 or len(trace_prefix_hashes) < num_steps:
            raise ValueError("trace_prefix_hashes must be a 1D array with at least num_steps entries")
    if np.array_equal(init_state, np.asarray(V0)):
        return {"found": True, "path_len": 0, "path_idx": [], "found_step": -1, "wall_s": 0.0}

    if progress_every:
        print(
            f"[solve] B_local={B_local:,} K_per_peer={K_per_peer:,} "
            f"parent_chunk={parent_chunk} internal_bs={internal_bs} "
            f"pack_v_score={pack_v_score} num_steps={num_steps}",
            flush=True,
        )

    # Step 0: 24 first-move scores (same as non-packed variant; Q-native reads them
    # from a single forward on the seed instead of 24 child forwards).
    init_dev = jnp.asarray(init_state)
    states_seed = jnp.expand_dims(init_dev, 0)
    neighbors0 = states_seed[:, all_moves].reshape(-1, state_size)
    if q_score_fn is None:
        values0 = model_apply(v_params, neighbors0, dtype=dtype).astype(jnp.float32)
    else:
        values0 = q_score_fn(states_seed).reshape(-1).astype(jnp.float32)
    k0 = min(B_local, n_gen)
    _, top_idx0 = _topk_smallest(values0, k0)
    top_idx0_np = np.asarray(top_idx0, dtype=np.int32)
    chosen0_np = np.asarray(neighbors0[top_idx0]).astype(np.int8)

    # Seed shard on host (one core's worth). Do NOT build the padded
    # (world_size, B_local, S) seed as a JAX array: at large 8-chip widths that
    # leaves a multi-GB device buffer alive before the first step.
    states0_np = np.empty((B_local, state_size), dtype=np.int8)
    states0_np[:k0, :] = chosen0_np
    if k0 < B_local:
        states0_np[k0:, :] = 0

    # Seed-move array (populates memmap[0] -- parent_local=0, parent_rank=0).
    move0_full_np = np.empty((B_local,), dtype=np.int8)
    move0_full_np[:k0] = top_idx0_np.astype(np.int8)
    if k0 < B_local:
        move0_full_np[k0:] = 0

    min_v_log = jnp.full((world_size, num_steps), 1e6, dtype=jnp.float32)
    found_step = jnp.full((world_size,), -1, dtype=jnp.int32)
    found_pos_local = jnp.full((world_size,), -1, dtype=jnp.int32)
    found_pos_rank = jnp.full((world_size,), -1, dtype=jnp.int32)
    verify_state = jnp.zeros((world_size, state_size), dtype=jnp.int8)

    # Early V0 check on the seed beam.
    v0_np = np.asarray(V0).astype(np.int8)
    eq0 = np.all(chosen0_np == v0_np[None, :], axis=1)
    if bool(eq0.any()):
        pos0_h = int(np.argmax(eq0))
        if pos0_h < k0:
            seed_move = int(top_idx0_np[pos0_h])
        else:
            seed_move = int(top_idx0_np[k0 - 1])
        return {"found": True, "path_len": 1, "path_idx": [seed_move],
                "found_step": 0, "wall_s": 0.0}
    del init_dev, states_seed, neighbors0, values0, top_idx0, chosen0_np
    gc.collect()

    V0_hash_host = int(np.sum(np.asarray(V0).astype(np.int64) * np.asarray(hash_vec)))

    if parent_chunk is None:
        step_body = _build_step_body_v_only_packed(
            v_params, all_moves, V0, hash_vec, jnp.int64(V0_hash_host),
            B_local, world_size, K_per_peer, n_gen, state_size,
            dtype, int(internal_bs),
            pack_v_score=bool(pack_v_score),
            owner_hash_vec=owner_hash_vec,
            trace_prefix_hashes=trace_prefix_hashes,
            q_score_fn=q_score_fn,
            tail_hashes=tail_hashes,
            tail_zobrist=tail_zobrist,
        )
    else:
        step_body = _build_step_body_v_only_packed_streaming(
            v_params, all_moves, V0, hash_vec, jnp.int64(V0_hash_host),
            B_local, world_size, K_per_peer, n_gen, state_size,
            dtype, int(internal_bs),
            parent_chunk=int(parent_chunk),
            pack_v_score=bool(pack_v_score),
            owner_hash_vec=owner_hash_vec,
            trace_prefix_hashes=trace_prefix_hashes,
            q_score_fn=q_score_fn,
            tail_hashes=tail_hashes,
            tail_zobrist=tail_zobrist,
        )

    @partial(jax.jit, donate_argnums=(0, 1, 2, 3, 4, 5))  # NOT j_arr
    def step_fn(states, mv_log, fs, fpl, fpr, vstate, j_arr):
        try:
            from jax.experimental.shard_map import shard_map
        except ImportError:
            from jax import shard_map
        return shard_map(
            step_body,
            mesh=mesh,
            in_specs=(P("cores"), P("cores"), P("cores"), P("cores"),
                      P("cores"), P("cores"), P()),
            out_specs=(P("cores"),) * 11,
        )(states, mv_log, fs, fpl, fpr, vstate, j_arr)

    from jax.sharding import NamedSharding
    sharding_states = NamedSharding(mesh, P("cores"))

    # Host memmap for the packed tree.
    if tree_path is None:
        work_dir = "/kaggle/working"
        if not os.path.isdir(work_dir):
            work_dir = tempfile.gettempdir()
        fd, tree_path = tempfile.mkstemp(prefix="tree_", suffix=".u32", dir=work_dir)
        os.close(fd)
    tree_mm = np.memmap(
        tree_path, mode="w+", dtype=np.uint64,
        shape=(num_steps, world_size, B_local),
    )
    # Seed move at j=0: parent_local=0, parent_rank=0, move=top_idx0[k] (replicated).
    seed_packed = (move0_full_np.astype(np.uint64) << BPTR_MOVE_SHIFT)
    tree_mm[0, :, :] = np.broadcast_to(seed_packed[None, :], (world_size, B_local))

    import time

    # Pre-compile step_fn explicitly so compile time is separately measurable.
    if progress_every:
        print("[lower] tracing step_fn...", flush=True)
    t_lower_start = time.time()
    states_aval = jax.ShapeDtypeStruct(
        (world_size, B_local, state_size), jnp.int8, sharding=sharding_states)
    mv_log_aval = jax.ShapeDtypeStruct(
        (world_size, num_steps), jnp.float32, sharding=sharding_states)
    rank_i32_aval = jax.ShapeDtypeStruct(
        (world_size,), jnp.int32, sharding=sharding_states)
    vstate_aval = jax.ShapeDtypeStruct(
        (world_size, state_size), jnp.int8, sharding=sharding_states)
    lowered = step_fn.lower(
        states_aval, mv_log_aval, rank_i32_aval, rank_i32_aval,
        rank_i32_aval, vstate_aval, jnp.int32(1),
    )
    t_lower = time.time() - t_lower_start
    if progress_every:
        print(f"[lower] done in {t_lower:.1f}s", flush=True)
        print("[compile] compiling step_fn...", flush=True)
    t_compile_start = time.time()
    compiled_step = lowered.compile()
    t_compile = time.time() - t_compile_start
    if progress_every:
        print(f"[compile] done in {t_compile:.1f}s", flush=True)
        print("[compile] loading runtime executable...", flush=True)
    t_load_start = time.time()
    compiled_step.runtime_executable()
    t_load = time.time() - t_load_start
    if progress_every:
        print(f"[compile] runtime executable loaded in {t_load:.1f}s", flush=True)

    # Build the sharded seed beam directly: each core gets one (B_local, S)
    # shard, and the full beam is never materialized on one device.
    states_d = jax.make_array_from_callback(
        (world_size, B_local, state_size), sharding_states,
        lambda _idx: states0_np[None],
    )
    mv_log_d = jax.device_put(min_v_log, sharding_states)
    fs_d = jax.device_put(found_step, sharding_states)
    fpl_d = jax.device_put(found_pos_local, sharding_states)
    fpr_d = jax.device_put(found_pos_rank, sharding_states)
    vstate_d = jax.device_put(verify_state, sharding_states)

    t_start = time.time()
    first_iter_t = None
    last_completed_step = 0
    fs_per_rank = np.asarray(fs_d)  # initial state (all -1), re-read in loop.
    trace_rows = [] if trace_prefix_hashes is not None else None

    try:
        for j in range(1, num_steps):
            t_iter = time.time()
            j_arr = jnp.int32(j)
            (states_d, packed_d, mv_log_d,
             fs_d, fpl_d, fpr_d, vstate_d,
             trace_hit_d, trace_pos_d, trace_v_d, trace_cutoff_d) = compiled_step(
                states_d, mv_log_d, fs_d, fpl_d, fpr_d, vstate_d, j_arr,
            )
            # Block on device computation (any output works; fs is smallest).
            jax.block_until_ready(fs_d)
            t_device = time.time() - t_iter

            # Sync + transfer packed backptr to host memmap.
            t_copy_start = time.time()
            packed_h = np.asarray(packed_d)
            t_copy = time.time() - t_copy_start

            t_write_start = time.time()
            tree_mm[j, :, :] = packed_h
            t_write = time.time() - t_write_start
            # Keep very-wide runs under HBM: after the backpointer block is
            # copied into the host memmap, the device copy is no longer needed.
            try:
                packed_d.delete()
            except Exception:
                pass
            del packed_d, packed_h

            fs_per_rank = np.asarray(fs_d)
            last_completed_step = j

            trace_msg = ""
            if trace_rows is not None:
                trace_hit_h = np.asarray(trace_hit_d)
                trace_pos_h = np.asarray(trace_pos_d)
                trace_v_h = np.asarray(trace_v_d)
                trace_cutoff_h = np.asarray(trace_cutoff_d)
                hit_ranks = np.where(trace_hit_h > 0)[0]
                row = {
                    "step": int(j),
                    "hit": bool(len(hit_ranks)),
                    "hit_ranks": [int(x) for x in hit_ranks.tolist()],
                    "best_cutoff_v": float(np.min(trace_cutoff_h)),
                }
                if len(hit_ranks):
                    rank0 = int(hit_ranks[0])
                    row.update({
                        "rank": rank0,
                        "pos": int(trace_pos_h[rank0]),
                        "target_v": float(trace_v_h[rank0]),
                        "rank_cutoff_v": float(trace_cutoff_h[rank0]),
                    })
                    trace_msg = (
                        f" trace=hit r{rank0} pos={int(trace_pos_h[rank0])} "
                        f"v={float(trace_v_h[rank0]):.3f} "
                        f"cut={float(trace_cutoff_h[rank0]):.3f}"
                    )
                else:
                    trace_msg = f" trace=drop cut_min={float(np.min(trace_cutoff_h)):.3f}"
                trace_rows.append(row)

            if first_iter_t is None:
                first_iter_t = time.time() - t_iter

            if progress_every and (j == 1 or j % progress_every == 0 or np.any(fs_per_rank >= 0)):
                print(
                    f"[step {j:03d}/{num_steps-1}] device={t_device:.1f}s "
                    f"copy={t_copy:.2f}s write={t_write:.2f}s "
                    f"total={time.time()-t_iter:.1f}s "
                    f"fs={fs_per_rank.tolist()}{trace_msg}",
                    flush=True,
                )

            if trace_rows is not None and stop_on_trace_drop and not trace_rows[-1]["hit"]:
                break

            if np.any(fs_per_rank >= 0):
                break

        tree_mm.flush()

        fpl_per_rank = np.asarray(fpl_d)
        mv_per_rank = np.asarray(mv_log_d)

        INT_MAX = 2 ** 30
        fs_signed = np.where(fs_per_rank >= 0, fs_per_rank, INT_MAX)
        global_min_step = int(fs_signed.min())
        if global_min_step >= INT_MAX:
            out = {"found": False, "path_len": 0, "path_idx": [],
                   "found_step": -1, "wall_s": time.time() - t_start,
                   "first_iter_s": first_iter_t,
                   "last_completed_step": last_completed_step,
                   "lower_s": t_lower, "compile_s": t_compile,
                   "min_v_trajectory_rank0": mv_per_rank[0].tolist()}
            if trace_rows is not None:
                out["trace_prefix"] = trace_rows
            return out

        winner_ranks = np.where(fs_signed == global_min_step)[0]
        winner_rank = int(winner_ranks[0])
        fs_h = int(fs_per_rank[winner_rank])
        fpl_h = int(fpl_per_rank[winner_rank])

        # Walkback through the packed memmap.
        path_idx = []
        cur_rank = winner_rank
        cur_pos = fpl_h
        for jj in range(fs_h, -1, -1):
            rec = int(tree_mm[jj, cur_rank, cur_pos])
            parent_local = rec & BPTR_PL_MASK
            parent_rank = (rec >> BPTR_RANK_SHIFT) & BPTR_RANK_MASK
            move = (rec >> BPTR_MOVE_SHIFT) & BPTR_MOVE_MASK
            path_idx.append(int(move))
            if jj > 0:
                cur_rank = int(parent_rank)
                cur_pos = int(parent_local)
        path_idx.reverse()

        out = {
            "found": True,
            "path_len": len(path_idx),
            "path_idx": path_idx,
            "found_step": fs_h,
            "found_pos_local": fpl_h,
            "found_pos_rank": winner_rank,
            "wall_s": time.time() - t_start,
            "first_iter_s": first_iter_t,
            "last_completed_step": last_completed_step,
            "lower_s": t_lower, "compile_s": t_compile,
            "min_v_trajectory_rank0": mv_per_rank[0].tolist(),
        }
        if trace_rows is not None:
            out["trace_prefix"] = trace_rows
        return out
    finally:
        # Always release memmap + delete the tree file.
        try:
            del tree_mm
        except Exception:
            pass
        gc.collect()
        try:
            if os.path.exists(tree_path):
                os.unlink(tree_path)
        except OSError:
            pass
