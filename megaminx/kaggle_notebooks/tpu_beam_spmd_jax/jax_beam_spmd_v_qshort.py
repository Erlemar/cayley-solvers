"""SPMD shared-beam JAX implementation: V + QSHORT variant (AZ v4 + m23_v3 stack).

Built on jax_beam_spmd_v_only.py with one change: at the V-call site inside
the streaming chunk body, we add a Q-shortlister (m23_v3_az_v4_sym) that
prefilters children before V scoring. Matches the production PyTorch
QShortlisterSolver algorithm (megaminx/beam_lab/beam_search_qshort.py).

Algorithm: GLOBAL top-αB per chunk.
  - student Q produces 24 values per parent → flatten to (chunk_n,) score
  - keep the αB SMALLEST Q-values GLOBALLY across all chunk_n candidates
    (NOT per-parent top-α; that has only 62% recall vs V's choices)
  - V re-scores only those αB candidates → top-K_per_peer downstream

At α=2: V evaluates n_keep = 2·parent_chunk = 262K children per chunk
vs 24·parent_chunk = 3.15M without qshort. **12× fewer V forwards.**

Recall validation (local 2026-05-19, n_parents=4080, B=8192):
  m23_v3_az_v4_sym/epoch_0199.pt teacher=m_az_v4_v_only.pt
  α=2 GLOBAL recall = 100%
  α=1 GLOBAL recall = 98.97% (do NOT use α=1 in production).

Algorithm (per step inside the shard_map body):
  1. Each rank generates B_LOCAL parents.
  2. Per chunk of parent_chunk parents: Q over all 24·parent_chunk children
     in one forward; GLOBAL top-αB by Q-score.
  3. V over only those αB selected children.
  4. Build child_v (chunk_n,) with BIG_F32 sentinels for non-selected.
     Downstream owner routing + top-K_per_peer never picks sentinels.
  5-7. Identical to v_only: hash → owner buckets → all_to_all → dedup →
     top-B_local → V0 detection.

Cross-rank winner selection remains HOST-SIDE (see spmd_jax_recovery.md).

Only the STREAMING builder is exported here (Phase C); the non-streaming
variant is dropped to keep this file focused. The 48M kernel always uses
parent_chunk > 0 (streaming), so this is sufficient.

Recipe parameters:
  q_params: dict[str, Any] — JAX params for the Q-shortlister student.
            Load via load_params_from_pt(qshort_path, (2048, 1024), 3).
  alpha:    int — top-α children per parent. Production: 2 (recall 100%).
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

# ---------------------------------------------------------------------------
# Packed-backpointer bit layout: one uint32 per beam slot in the host memmap
# encodes (parent_local, parent_rank, move). parent_local needs
# ceil(log2(B_local)) bits, parent_rank needs ceil(log2(world_size)), move
# needs ceil(log2(n_gen)).
#
# The ORIGINAL layout gave parent_local only 23 bits (max 8,388,607). That is
# fine up to B_global=48M (B_local=6.29M) but SILENTLY OVERFLOWS at B_global=96M
# (B_local=12.58M > 2^23): the 2^23 bit of parent_local leaks into the rank
# field and the low 23 bits truncate, so the walkback reconstructs wrong
# parents -> _solves rejects the path -> found=False even though the V0 hash
# hit was real. (This is why 48M-streaming solves but 96M failed; the CPU smoke
# never hit it because small-B never exceeds 2^23.)
#
# New layout 24/3/5 fits a uint32 exactly (24+3+5=32) and supports B_local up
# to 2^24=16,777,216 (B_global up to 134M). For wider beams switch the memmap
# to uint64 and widen BPTR_PL_BITS.
BPTR_PL_BITS = 24                                   # parent_local : bits 0..23 (8 ranks: 2^24=16.7M/rank -> 134M global)
BPTR_RANK_BITS = 3                                  # parent_rank  : bits 24..26 (8 ranks need 3 bits)
BPTR_MOVE_BITS = 5                                  # move         : bits 27..31
BPTR_RANK_SHIFT = BPTR_PL_BITS                      # 24
BPTR_MOVE_SHIFT = BPTR_PL_BITS + BPTR_RANK_BITS     # 27
BPTR_PL_MASK = (1 << BPTR_PL_BITS) - 1              # 0x00FFFFFF (24-bit)
BPTR_RANK_MASK = (1 << BPTR_RANK_BITS) - 1          # 0x7 (8 ranks)
BPTR_MOVE_MASK = (1 << BPTR_MOVE_BITS) - 1          # 0x1F
BPTR_MAX_B_LOCAL = 1 << BPTR_PL_BITS                # 16,777,216 (8 ranks -> 134M global)


def make_mesh(devices=None):
    if devices is None:
        devices = jax.devices()
    return Mesh(np.asarray(devices), axis_names=("cores",))


def _topk_smallest(values, k):
    neg_v, idx = jax.lax.top_k(-values, k)
    return -neg_v, idx


def _approx_topk_smallest(values, k, recall_target=0.95):
    """Approximate k-smallest via jax.lax.approx_min_k (TPU-optimized).

    Returns (values, indices); the result is NOT guaranteed sorted and ~k*recall
    of the entries are the true smallest (the rest are slightly larger). Drop-in
    for `_topk_smallest` at the large one-shot selections (qshort prefilter,
    final beam cut). Keep `_topk_smallest` (exact) for the per-owner running
    merge so approximation errors don't compound across the chunk scan.
    """
    vals, idx = jax.lax.approx_min_k(values, k, recall_target=recall_target)
    return vals, idx.astype(jnp.int32)


def build_solved_neighborhood(generators, inv_move_idx, hash_vec, v0_state,
                              max_radius, canonical=True):
    """BFS from the solved state out to `max_radius` moves; return the arrays
    the kernel needs for in-beam neighborhood (K1-ball) early-stop.

    Mirrors `megaminx.bfs_bytes.build_bfs_bytes` (BFS from identity, canonical
    anti-inverse pruning) but also computes, per reachable state, the int64
    Zobrist hash (using the SAME `hash_vec` the kernel hashes states with) and
    the *shortest suffix* of moves that returns that state to solved.

    generators:   (n_gen, state_size) int; child[i] = parent[gen[i]].
    inv_move_idx: (n_gen,) int; inverse generator index of each move.
    hash_vec:     (state_size,) int64; identical to the kernel's hash vector.
    v0_state:     (state_size,) int; the solved/central state.

    Returns (hash_sorted int64[M], suffix_len int8[M], suffix_moves int8[M, R]),
    sorted ascending by hash so the device can binary-search. For a near-hit at
    sorted index i, `suffix_moves[i][:suffix_len[i]]` applied (in generator
    index space) to a state with that hash reaches solved. The solved state
    itself is included (suffix_len 0), so this also subsumes exact V0 detection.
    """
    gens = np.asarray(generators)
    n_gen = gens.shape[0]
    inv = np.asarray(inv_move_idx)
    hv = np.asarray(hash_vec).astype(np.int64)
    v0 = np.asarray(v0_state).astype(np.int8)
    state_size = v0.shape[0]

    # word = generator indices applied to V0 (left-to-right) to reach the state.
    table = {v0.tobytes(): b""}
    frontier = [(v0, b"")]
    for _d in range(max_radius):
        nxt = []
        for state, word in frontier:
            last = word[-1] if word else -1
            for gi in range(n_gen):
                if canonical and last >= 0 and gi == int(inv[last]):
                    continue
                child = state[gens[gi]]
                key = child.tobytes()
                if key not in table:
                    new_word = word + bytes((gi,))
                    table[key] = new_word
                    nxt.append((child, new_word))
        frontier = nxt

    M = len(table)
    hashes = np.empty(M, dtype=np.int64)
    suf_len = np.empty(M, dtype=np.int8)
    suf_mv = np.zeros((M, max_radius), dtype=np.int8)
    with np.errstate(over="ignore"):  # int64 hash wraps mod 2**64, same as XLA.
        for i, (key, word) in enumerate(table.items()):
            st = np.frombuffer(key, dtype=np.int8).astype(np.int64)
            hashes[i] = (st * hv).sum()
            # suffix (state -> V0) = inverse moves of `word` in reverse order.
            suffix = [int(inv[m]) for m in reversed(word)]
            suf_len[i] = len(suffix)
            for k, mv in enumerate(suffix):
                suf_mv[i, k] = mv
    order = np.argsort(hashes)
    return hashes[order], suf_len[order], suf_mv[order]


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
            bucket = bucket.at[:, 0:120].set(sel_states_u8)
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

        # Padding detection: real states sum to 7140, padding sums to 0.
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
    # Packed-backpointer field-width guards: parent_local/rank/move must fit the
    # uint32 memmap layout (see BPTR_* constants). Overflow corrupts the walkback
    # SILENTLY (found=False with valid V0 hash hits) -- fail loudly instead.
    if B_local > BPTR_MAX_B_LOCAL:
        raise ValueError(
            f"B_local={B_local:,} exceeds packed-backpointer capacity "
            f"{BPTR_MAX_B_LOCAL:,} ({BPTR_PL_BITS}-bit parent_local). "
            f"B_global <= {BPTR_MAX_B_LOCAL * world_size:,}. For wider beams "
            f"switch the tree memmap to uint64 and widen BPTR_PL_BITS."
        )
    if world_size > (1 << BPTR_RANK_BITS):
        raise ValueError(f"world_size={world_size} exceeds {1 << BPTR_RANK_BITS} ranks")
    if n_gen > (1 << BPTR_MOVE_BITS):
        raise ValueError(f"n_gen={n_gen} exceeds {1 << BPTR_MOVE_BITS} moves")
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
            check_rep=False,  # jax 0.6.2 strict VMA match (replicated vs varying).
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
# Phase A+B variant: packed uint32 backpointers, host memmap tree, early stop.
# =============================================================================

def _build_step_body_v_only_packed(
    v_params,
    all_moves, V0, hash_vec, V0_hash,
    B_local, world_size, K_per_peer, n_gen, state_size,
    dtype, internal_bs,
    pack_v_score=False,
    owner_hash_vec=None,
):
    """Per-shard V-only step body, Phase A+B (+ optional D) variant.

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
        single packed uint32 backpointer per beam slot (shape (B_local,)):
          bits  0..22: parent_local (supports up to 8,388,608 = 64M global)
          bits 23..25: parent_rank  (0..7)
          bits 26..30: move         (0..23, 5 bits)
          bit      31: unused (could later be a "valid" flag)

    Body outputs (in shard-local shape, with leading rank axis re-added):
      (chosen_states, packed_backptr, min_v_log, found_step,
       found_pos_local, found_pos_rank, verify_state)

    The wrapper writes packed_backptr into a host np.memmap each step.
    Walkback unpacks records and follows (parent_rank, parent_local) chains.
    """
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

        # 2. V on ALL children (chunked via lax.scan).
        child_v = forward_v(v_params, neighbors)

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
            bucket = bucket.at[:, 0:120].set(sel_states_u8)
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

        # Padding detection: real states sum to 7140, padding sums to 0.
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

        # Phase B: pack backpointer into a single uint32 per beam slot.
        packed_backptr = (
            chosen_parent_local.astype(jnp.uint32)
            | (chosen_parent_rank.astype(jnp.uint32) << jnp.uint32(BPTR_RANK_SHIFT))
            | (chosen_move.astype(jnp.uint32) << jnp.uint32(BPTR_MOVE_SHIFT))
        )

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
                packed_backptr[None, :],
                new_min_v_log[None, :],
                new_found_step[None],
                new_found_pos_local[None],
                new_found_pos_rank[None],
                new_verify_state[None, :])

    return beam_step_local


# -----------------------------------------------------------------------------
# Phase C: streaming child generation via lax.scan over parent chunks.
# Removes the 11.5 GiB neighbors materialization at B_local=4M (32M global).
# Same packed-backpointer output as the non-streaming body, so the wrapper
# can dispatch to either body via the `parent_chunk` parameter.
# -----------------------------------------------------------------------------

def _build_step_body_v_qshort_packed_streaming(
    v_params,
    q_params,
    alpha,
    all_moves, V0, hash_vec, V0_hash,
    B_local, world_size, K_per_peer, n_gen, state_size,
    dtype, internal_bs,
    parent_chunk,
    pack_v_score=False,
    owner_hash_vec=None,
    use_neighborhood=False,
    nbhd_hash_sorted=None,
    nbhd_suffix_len=None,
    use_approx_topk=False,
    approx_recall=0.95,
    profile_skip="",
    lean_merge=False,
):
    """Per-shard V+QSHORT step body, Phase A+B+C (+ optional D): streamed.

    Qshort prefilter: for each chunk of `parent_chunk` parents, runs Q on
    parents (one forward producing (parent_chunk, n_gen) Q-values), picks the
    α-smallest Q indices per parent, materializes α×parent_chunk children,
    runs V only on those. Builds a sentinel-padded (parent_chunk, n_gen)
    child_v tensor so downstream owner routing is unchanged.

    Net effect at α=2: V evaluates α×parent_chunk = 2×parent_chunk children
    per chunk instead of n_gen×parent_chunk = 24×parent_chunk. 12× fewer V
    forwards plus a small Q forward overhead.

    `pack_v_score` (Phase D): same as v_only -- pack send-side bf16 V score
    into bytes 125-126, skip receive-side forward_v. Quality budget:
    0-2 moves per pid from bf16 tie-break drift.

    `parent_chunk` MUST divide B_local. Typical value at 48M: 131072.
    """
    BIG_F32 = jnp.float32(1e9)
    aB_local = K_per_peer * world_size
    assert B_local % parent_chunk == 0, (
        f"parent_chunk {parent_chunk} must divide B_local {B_local}"
    )
    n_chunks = B_local // parent_chunk
    chunk_n = parent_chunk * n_gen
    M_nbhd = int(nbhd_hash_sorted.shape[0]) if use_neighborhood else 0

    def sel_topk(values, k):
        # Approx (approx_min_k) for the big one-shot selections when enabled;
        # exact top_k otherwise. The per-owner running merge always stays exact.
        if use_approx_topk:
            return _approx_topk_smallest(values, k, approx_recall)
        return _topk_smallest(values, k)

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

    def forward_q(params, x):
        # Q has output_dim=24; returns (n, n_gen) float32 (no chunked scan;
        # parent_chunk is already small enough for one Q forward).
        return model_apply(params, x, dtype=dtype).astype(jnp.float32)

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
        # lean_merge carries compact provenance (parent_local + move) instead of
        # the full (ws, K, 128) = 1.5 GB pack -- the per-chunk RMW of that carry
        # was ~85% of the step (ablation v14). Packs are built ONCE post-scan.
        if lean_merge:
            init_top_pl = jax.lax.pcast(
                jnp.zeros((world_size, K_per_peer), dtype=jnp.int32),
                ("cores",), to="varying",
            )
            init_top_mv = jax.lax.pcast(
                jnp.zeros((world_size, K_per_peer), dtype=jnp.int32),
                ("cores",), to="varying",
            )
        else:
            init_top_pack = jax.lax.pcast(
                jnp.zeros((world_size, K_per_peer, PACK_SIZE), dtype=jnp.uint8),
                ("cores",), to="varying",
            )

        def chunk_body(carry, chunk_i):
            if lean_merge:
                top_scores, top_pl, top_mv = carry
            else:
                top_scores, top_pack = carry
            parent_start = chunk_i * jnp.int32(parent_chunk)
            states_chunk = jax.lax.dynamic_slice(
                states, (parent_start, jnp.int32(0)),
                (parent_chunk, state_size),
            )
            # profile_skip="gen" replaces the all-24-children gather (18 GB/step)
            # with a cheap zeros memset (WRONG beam, timing-only) to measure the
            # generation gather's share of the step.
            if profile_skip == "gen":
                children = jnp.zeros((parent_chunk * n_gen, state_size), dtype=jnp.int8)
            else:
                children = states_chunk[:, all_moves].reshape(-1, state_size)  # (chunk_n, S)

            # QSHORT PREFILTER: GLOBAL top-αB by Q-score per chunk (NOT per-parent).
            # Matches the production PyTorch QShortlisterSolver
            # (megaminx/beam_lab/beam_search_qshort.py): target_k = alpha * B, then
            # torch.topk(student_score, target_k) GLOBALLY across all 24·B candidates.
            # Per-parent top-α was tested and FAILS (recall 62%); global has recall 100%
            # at alpha=2 per the 09_eval_q_recall.py bench (validated 2026-05-19).
            if profile_skip == "qfwd":
                parent_q = jnp.zeros((parent_chunk, n_gen), dtype=jnp.float32)
            else:
                parent_q = forward_q(q_params, states_chunk)  # (parent_chunk, n_gen) float32
            q_flat = parent_q.reshape(-1)  # (chunk_n,) — same flat layout as children
            n_keep = alpha * parent_chunk  # target_k from qshort spec
            # Global top-αB smallest Q values (smallest = predicted closest to solved)
            _, keep_idx = sel_topk(q_flat, n_keep)  # (n_keep,) int32 indices
            keep_idx = keep_idx.astype(jnp.int32)

            # Gather selected children + V on them. profile_skip="vfwd" skips the
            # V GEMM (zeros -> WRONG beam, timing-only) to measure the V forward's
            # share of the step; "qfwd" skips Q the same way (above).
            selected_children = children[keep_idx]  # (n_keep, S)
            if profile_skip == "vfwd":
                selected_v_flat = jnp.zeros((alpha * parent_chunk,), dtype=jnp.float32)
            else:
                selected_v_flat = forward_v(v_params, selected_children)  # (n_keep,)

            # Build (chunk_n,) child_v with BIG_F32 sentinels at non-selected positions.
            # Downstream owner-routing + top-K_per_peer never picks BIG_F32 entries.
            child_v = jnp.full((parent_chunk * n_gen,), BIG_F32, dtype=jnp.float32)
            child_v = child_v.at[keep_idx].set(selected_v_flat)

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
            if lean_merge:
                new_top_pl = top_pl
                new_top_mv = top_mv
            else:
                new_top_pack = top_pack
            for dest in range(world_size):
                # Mask children not owned by `dest`.
                masked_scores = jnp.where(owner == dest, child_v, BIG_F32)
                # Merge running top with this chunk's masked candidates.
                merged_scores = jnp.concatenate(
                    [new_top_scores[dest], masked_scores], axis=0,
                )  # (K_per_peer + chunk_n,) float32
                top_v_new, keep = sel_topk(merged_scores, K_per_peer)

                # profile_skip="packbuild" skips the 128 B bucket pack-building
                # (keeps top_k so upstream isn't DCE'd) -- isolates its share.
                if profile_skip == "packbuild":
                    new_top_scores = new_top_scores.at[dest].set(top_v_new)
                    continue

                # Each kept slot is either from old (keep < K_per_peer) or new.
                from_old = keep < K_per_peer
                old_idx = jnp.clip(keep, 0, K_per_peer - 1)
                new_idx = jnp.clip(keep - K_per_peer, 0, chunk_n - 1)
                is_pad = top_v_new >= (BIG_F32 * 0.5)
                sel_parent_local = (parent_start + (new_idx // n_gen)).astype(jnp.int32)
                sel_move = (new_idx % n_gen).astype(jnp.int32)

                if lean_merge:
                    # Carry compact provenance only; packs built once post-scan.
                    old_pl = new_top_pl[dest][old_idx]
                    old_mv = new_top_mv[dest][old_idx]
                    merged_pl = jnp.where(from_old, old_pl, jnp.where(is_pad, 0, sel_parent_local))
                    merged_mv = jnp.where(from_old, old_mv, jnp.where(is_pad, 0, sel_move))
                    new_top_scores = new_top_scores.at[dest].set(top_v_new)
                    new_top_pl = new_top_pl.at[dest].set(merged_pl)
                    new_top_mv = new_top_mv.at[dest].set(merged_mv)
                    continue

                # --- legacy: full 128 B pack carried in the scan (slow path) ---
                old_pack = new_top_pack[dest][old_idx]  # (K_per_peer, PACK_SIZE)
                sel_states = children[new_idx]  # (K_per_peer, state_size) int8
                zero_state = jnp.zeros((K_per_peer, state_size), dtype=jnp.uint8)
                zero_int32 = jnp.zeros(K_per_peer, dtype=jnp.int32)
                zero_int8 = jnp.zeros(K_per_peer, dtype=jnp.int8)
                sel_states_u8 = jnp.where(is_pad[:, None], zero_state, sel_states.astype(jnp.uint8))
                sel_parent_local_z = jnp.where(is_pad, zero_int32, sel_parent_local)
                sel_move_z = jnp.where(is_pad, zero_int8, sel_move.astype(jnp.int8))

                new_pack = jnp.zeros((K_per_peer, PACK_SIZE), dtype=jnp.uint8)
                new_pack = new_pack.at[:, 0:120].set(sel_states_u8)
                new_pack = new_pack.at[:, 120].set((sel_parent_local_z & 0xFF).astype(jnp.uint8))
                new_pack = new_pack.at[:, 121].set(((sel_parent_local_z >> 8) & 0xFF).astype(jnp.uint8))
                new_pack = new_pack.at[:, 122].set(((sel_parent_local_z >> 16) & 0xFF).astype(jnp.uint8))
                new_pack = new_pack.at[:, 123].set(((sel_parent_local_z >> 24) & 0xFF).astype(jnp.uint8))
                new_pack = new_pack.at[:, 124].set(sel_move_z.astype(jnp.uint8))
                merged_pack = jnp.where(from_old[:, None], old_pack, new_pack)
                if pack_v_score:
                    top_v_bf16 = top_v_new.astype(jnp.bfloat16)
                    top_v_u16 = jax.lax.bitcast_convert_type(top_v_bf16, jnp.uint16)
                    merged_pack = merged_pack.at[:, 125].set((top_v_u16 & 0xFF).astype(jnp.uint8))
                    merged_pack = merged_pack.at[:, 126].set(((top_v_u16 >> 8) & 0xFF).astype(jnp.uint8))
                new_top_scores = new_top_scores.at[dest].set(top_v_new)
                new_top_pack = new_top_pack.at[dest].set(merged_pack)

            if lean_merge:
                return (new_top_scores, new_top_pl, new_top_mv), None
            return (new_top_scores, new_top_pack), None

        if lean_merge:
            (final_scores, final_pl, final_mv), _ = jax.lax.scan(
                chunk_body, (init_top_scores, init_top_pl, init_top_mv),
                jnp.arange(n_chunks, dtype=jnp.int32),
            )
            # Build the 128 B send buckets ONCE: regenerate each survivor's state
            # via apply-move on the local frontier (generation is ~free, v13), then
            # byte-pack. 8 owner pack-builds total vs the old 8*n_chunks=384.
            # Build packs CHUNKED over all ws*K survivors (owner-major flatten),
            # so the per-row apply-move never materializes a full (ws*K, S) int32
            # generator-index (that un-chunked gather made v15 SLOWER than the
            # per-chunk path). Same chunking trick as idea-4's materialize.
            flat_sc = final_scores.reshape(-1)
            flat_pl = final_pl.reshape(-1)
            flat_mv = final_mv.reshape(-1)
            N_surv = world_size * K_per_peer
            MAT = parent_chunk
            assert N_surv % MAT == 0, f"ws*K={N_surv} not divisible by MAT={MAT}"

            def _pack_body(_c, ci):
                base = ci * jnp.int32(MAT)
                sc = jax.lax.dynamic_slice(flat_sc, (base,), (MAT,))
                pl = jax.lax.dynamic_slice(flat_pl, (base,), (MAT,))
                mv = jax.lax.dynamic_slice(flat_mv, (base,), (MAT,))
                is_pad = sc >= (BIG_F32 * 0.5)
                parent = states[jnp.clip(pl, 0, B_local - 1)]
                gen = all_moves[jnp.clip(mv, 0, n_gen - 1)]
                child = jnp.take_along_axis(parent, gen, axis=1)
                pl_z = jnp.where(is_pad, 0, pl)
                mv_z = jnp.where(is_pad, 0, mv)
                child_u8 = jnp.where(is_pad[:, None], jnp.uint8(0), child.astype(jnp.uint8))
                # Build the 128 B pack with ONE concatenate (not 6 scatter-set
                # byte ops) -- the .at[].set byte-packing is the suspected slow
                # primitive (lean still spent ~31s here). Little-endian bitcast
                # matches the receive-side unpack (byte 120 = LSB, etc.).
                pl_b = jax.lax.bitcast_convert_type(pl_z.astype(jnp.uint32), jnp.uint8)  # (MAT,4)
                mv_b = mv_z.astype(jnp.uint8)[:, None]                                   # (MAT,1)
                if pack_v_score:
                    sc_b = jax.lax.bitcast_convert_type(sc.astype(jnp.bfloat16), jnp.uint8)  # (MAT,2)
                    pk = jnp.concatenate(
                        [child_u8, pl_b, mv_b, sc_b, jnp.zeros((MAT, 1), jnp.uint8)], axis=1)
                else:
                    pk = jnp.concatenate(
                        [child_u8, pl_b, mv_b, jnp.zeros((MAT, 3), jnp.uint8)], axis=1)
                return _c, pk

            _, _packs = jax.lax.scan(_pack_body, None, jnp.arange(N_surv // MAT, dtype=jnp.int32))
            send_buckets = _packs.reshape(world_size, K_per_peer, PACK_SIZE)
        else:
            (_, final_top_pack), _ = jax.lax.scan(
                chunk_body, (init_top_scores, init_top_pack),
                jnp.arange(n_chunks, dtype=jnp.int32),
            )
            send_buckets = final_top_pack  # (world_size, K_per_peer, PACK_SIZE)

        # all_to_all + receive side. profile_skip="a2a" replaces the exchange
        # with identity (WRONG beam -- timing-only ablation) to measure a2a cost.
        if profile_skip == "a2a":
            recv_buckets = send_buckets
        else:
            recv_buckets = jax.lax.all_to_all(
                send_buckets, axis_name="cores",
                split_axis=0, concat_axis=0, tiled=True,
            )

        recv_flat = recv_buckets.reshape(-1, PACK_SIZE)
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

        recv_state_sum = jnp.sum(recv_states.astype(jnp.int32), axis=1)
        is_padding = (recv_state_sum == 0)

        # In-rank dedup + top-B using ONLY hash+score (small per-row), then defer
        # the state gather to the B survivors. Removes the 12M*120 sorted_states
        # gather (~1.4 GB at 48M) and the other full-width sorted_* gathers; the
        # recv side now does one B_local-sized gather via final_idx. Identical
        # selection -> length-safe. (approx-topk on the merge was rejected: ~0
        # speedup, +5 moves on pid 0 -- the merge top-k was NOT the bottleneck;
        # the memory-bound gathers are the next suspect for the 46.4s step.)
        recv_h = jnp.sum(recv_states.astype(jnp.int64) * hash_vec, axis=1)
        # profile_skip="argsort" replaces the dedup sort with identity (WRONG
        # beam -- timing-only) to measure the 12M argsort cost.
        if profile_skip == "argsort":
            sort_idx = jnp.arange(aB_local, dtype=jnp.int32)
        else:
            sort_idx = jnp.argsort(recv_h)
        sorted_h = recv_h[sort_idx]
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

        top_v_keep, keep_sorted_idx = sel_topk(sorted_v_masked, B_local)

        # Map kept sorted positions back to receive positions; gather states once.
        final_idx = sort_idx[keep_sorted_idx]
        chosen_states = recv_states[final_idx]
        chosen_parent_local = recv_parent_local[final_idx]
        chosen_parent_rank = recv_sender_rank[final_idx]
        chosen_move = recv_move[final_idx]
        chosen_h = recv_h[final_idx]
        chosen_state_sum = jnp.sum(chosen_states.astype(jnp.int32), axis=1)
        chosen_is_real = (chosen_state_sum != 0)

        packed_backptr = (
            chosen_parent_local.astype(jnp.uint32)
            | (chosen_parent_rank.astype(jnp.uint32) << jnp.uint32(BPTR_RANK_SHIFT))
            | (chosen_move.astype(jnp.uint32) << jnp.uint32(BPTR_MOVE_SHIFT))
        )

        # jnp.min not top_v_keep[0]: approx_min_k results aren't sorted.
        new_min_v_log = min_v_log.at[j].set(jnp.min(top_v_keep))

        eq_v0 = (chosen_h == V0_hash) & chosen_is_real
        any_hit = jnp.any(eq_v0)
        pos_hit = jnp.argmax(eq_v0.astype(jnp.int32)).astype(jnp.int32)
        is_first_hit = (found_step == -1) & any_hit
        new_found_step = jnp.where(is_first_hit, j, found_step)
        new_found_pos_local = jnp.where(is_first_hit, pos_hit, found_pos_local)
        new_found_pos_rank = jnp.where(is_first_hit, rank_int, found_pos_rank)
        candidate_state = chosen_states[pos_hit]
        new_verify_state = jnp.where(is_first_hit, candidate_state, verify_state)

        # Idea (2): neighborhood (K1-ball) early-stop. The host already-computed
        # chosen_h is binary-searched against the sorted solved-neighborhood
        # hashes; we emit this rank's smallest-suffix near-hit this step. The
        # wrapper tracks the global-best (step + suffix_len) across steps and
        # appends the BFS-optimal suffix. Subsumes exact V0 (suffix_len 0).
        if use_neighborhood:
            nb_pos = jnp.searchsorted(nbhd_hash_sorted, chosen_h)
            nb_pos = jnp.clip(nb_pos, 0, M_nbhd - 1)
            nb_match = (nbhd_hash_sorted[nb_pos] == chosen_h) & chosen_is_real
            nb_suf = jnp.where(
                nb_match, nbhd_suffix_len[nb_pos].astype(jnp.int32), jnp.int32(127)
            )
            near_argmin = jnp.argmin(nb_suf).astype(jnp.int32)
            near_any = jnp.any(nb_match).astype(jnp.int32)
            near_minlen = nb_suf[near_argmin].astype(jnp.int32)
            near_pos = near_argmin
            near_sufidx = nb_pos[near_argmin].astype(jnp.int32)
        else:
            near_any = jnp.int32(0)
            near_minlen = jnp.int32(127)
            near_pos = jnp.int32(0)
            near_sufidx = jnp.int32(0)

        return (chosen_states[None, :, :],
                packed_backptr[None, :],
                new_min_v_log[None, :],
                new_found_step[None],
                new_found_pos_local[None],
                new_found_pos_rank[None],
                new_verify_state[None, :],
                near_any[None],
                near_minlen[None],
                near_pos[None],
                near_sufidx[None])

    return beam_step_local


# -----------------------------------------------------------------------------
# Idea (4): compact-meta + deferred materialization (the GPU friend's design).
#
# The forward all-to-all ships a 16-byte CandidateMeta (hash + parent_local +
# move + score + valid), NOT the 128-byte full-state pack. Dedup + top-B run on
# hash/score only -- no candidate state is ever gathered (kills the 12M*120-byte
# sorted_states gather). The B survivors' states are then re-materialized via a
# 2-hop round-trip: owner routes (parent_local, move) back to the source rank,
# the source applies the move to its LOCAL frontier parent and returns the child
# state, the owner scatters it into next_frontier.
#
# Selects the IDENTICAL beam as _build_step_body_v_qshort_packed_streaming
# (same routing/dedup/top-B) so paths are unchanged; only state movement differs
# (12M candidate states -> 6M survivor states + 8x-smaller buckets). Wins memory
# (~2x width) and the memory-bound sort/gather/all-to-all cost. ALPHA_REQ
# over-sizes the per-source request buckets for load balance; the only quality
# risk is request-bucket overflow (degrades a survivor slot to padding), which is
# ~0 because survivor source-rank is uniform. Same 11 outputs as the streaming
# body, so the wrapper is unchanged.
# -----------------------------------------------------------------------------

def _build_step_body_v_qshort_meta_streaming(
    v_params,
    q_params,
    alpha,
    all_moves, V0, hash_vec, V0_hash,
    B_local, world_size, K_per_peer, n_gen, state_size,
    dtype, internal_bs,
    parent_chunk,
    alpha_req=1.5,
    use_neighborhood=False,
    nbhd_hash_sorted=None,
    nbhd_suffix_len=None,
    use_approx_topk=False,
    approx_recall=0.95,
):
    BIG_F32 = jnp.float32(1e9)
    assert B_local % parent_chunk == 0
    n_chunks = B_local // parent_chunk
    chunk_n = parent_chunk * n_gen
    M_nbhd = int(nbhd_hash_sorted.shape[0]) if use_neighborhood else 0
    META_SIZE = 16          # hash8 + parent_local4 + move1 + score2 + valid1
    REQ_REC = 12            # parent_local4 + move1 + valid1 + target4 + pad2
    RESP_REC = 128          # state120 + target4 + valid1 + pad3
    # Per-source request bucket capacity (static). Survivor source-rank is
    # uniform, so alpha_req ~1.25-1.5 leaves headroom; overflow -> padding slot.
    # Rounded up to a MAT_CHUNK multiple so the response materialization can scan
    # over fixed-size chunks (the un-chunked apply-move OOMs: a (ws*REQ_CAP, S)
    # generator-index gather is ~13 GB of HLO temp).
    MAT_CHUNK = parent_chunk
    _req_raw = -(-(int(alpha_req * B_local)) // world_size)  # ceil(alpha_req*B_local/ws)
    REQ_CAP = ((_req_raw + MAT_CHUNK - 1) // MAT_CHUNK) * MAT_CHUNK

    def sel_topk(values, k):
        if use_approx_topk:
            return _approx_topk_smallest(values, k, approx_recall)
        return _topk_smallest(values, k)

    def _chunked_apply(params, x, chunk_size):
        n, S = x.shape
        chunks = x.reshape(n // chunk_size, chunk_size, S)
        def _scan_fn(_, c):
            return _, model_apply(params, c, dtype=dtype)
        _, outs = jax.lax.scan(_scan_fn, None, chunks)
        return outs.reshape(n) if outs.ndim == 2 else outs.reshape(n, -1)

    def forward_v(params, x):
        return _chunked_apply(params, x, internal_bs).astype(jnp.float32)

    def forward_q(params, x):
        return model_apply(params, x, dtype=dtype).astype(jnp.float32)

    def _u8(a):
        return a.astype(jnp.uint8)

    def _bytes(a, dt):
        return jax.lax.bitcast_convert_type(a.astype(dt), jnp.uint8)

    arange_aB = jnp.arange(K_per_peer * world_size, dtype=jnp.int32)
    sender_rank_per_recv = (arange_aB // K_per_peer).astype(jnp.int32)
    owner_arange = jnp.arange(world_size, dtype=jnp.int32)

    def beam_step_local(states, min_v_log, found_step, found_pos_local,
                        found_pos_rank, verify_state, j):
        states = states[0]
        min_v_log = min_v_log[0]
        found_step = found_step[0]
        found_pos_local = found_pos_local[0]
        found_pos_rank = found_pos_rank[0]
        verify_state = verify_state[0]
        rank_int = jax.lax.axis_index("cores").astype(jnp.int32)

        init_top_scores = jax.lax.pcast(
            jnp.full((world_size, K_per_peer), BIG_F32, dtype=jnp.float32),
            ("cores",), to="varying")
        init_top_meta = jax.lax.pcast(
            jnp.zeros((world_size, K_per_peer, META_SIZE), dtype=jnp.uint8),
            ("cores",), to="varying")

        def chunk_body(carry, chunk_i):
            top_scores, top_meta = carry
            parent_start = chunk_i * jnp.int32(parent_chunk)
            states_chunk = jax.lax.dynamic_slice(
                states, (parent_start, jnp.int32(0)), (parent_chunk, state_size))
            children = states_chunk[:, all_moves].reshape(-1, state_size)

            # qshort prefilter (GLOBAL top-alphaB by Q), identical to streaming.
            parent_q = forward_q(q_params, states_chunk)
            q_flat = parent_q.reshape(-1)
            _, keep_idx = sel_topk(q_flat, alpha * parent_chunk)
            keep_idx = keep_idx.astype(jnp.int32)
            selected_children = children[keep_idx]
            selected_v = forward_v(v_params, selected_children)
            child_v = jnp.full((chunk_n,), BIG_F32, dtype=jnp.float32).at[keep_idx].set(selected_v)

            # int64 Zobrist hash -> owner (low bits) AND dedup key (shipped).
            h = jnp.sum(children.astype(jnp.int64) * hash_vec, axis=1)
            owner = (h & jnp.int64(world_size - 1)).astype(jnp.int32)

            new_top_scores = top_scores
            new_top_meta = top_meta
            for dest in range(world_size):
                masked = jnp.where(owner == dest, child_v, BIG_F32)
                merged = jnp.concatenate([new_top_scores[dest], masked], axis=0)
                top_v_new, keep = _topk_smallest(merged, K_per_peer)
                from_old = keep < K_per_peer
                old_idx = jnp.clip(keep, 0, K_per_peer - 1)
                new_idx = jnp.clip(keep - K_per_peer, 0, chunk_n - 1)
                old_meta = new_top_meta[dest][old_idx]

                sel_h = h[new_idx]
                sel_pl = (parent_start + (new_idx // n_gen)).astype(jnp.uint32)
                sel_mv = (new_idx % n_gen).astype(jnp.uint8)
                is_pad = top_v_new >= (BIG_F32 * 0.5)
                valid_b = jnp.where(is_pad, jnp.uint8(0), jnp.uint8(1))[:, None]
                sc_b = _bytes(top_v_new, jnp.bfloat16)              # (K, 2)
                new_meta = jnp.concatenate([
                    _bytes(sel_h, jnp.int64),                       # (K, 8)
                    _bytes(sel_pl, jnp.uint32),                     # (K, 4)
                    sel_mv[:, None],                               # (K, 1)
                    sc_b,                                          # (K, 2)
                    valid_b,                                       # (K, 1)
                ], axis=1)
                merged_meta = jnp.where(from_old[:, None], old_meta, new_meta)
                new_top_scores = new_top_scores.at[dest].set(top_v_new)
                new_top_meta = new_top_meta.at[dest].set(merged_meta)
            return (new_top_scores, new_top_meta), None

        (_, final_meta), _ = jax.lax.scan(
            chunk_body, (init_top_scores, init_top_meta),
            jnp.arange(n_chunks, dtype=jnp.int32))

        # Forward all-to-all of 16-byte metas (vs 128-byte state packs).
        recv_meta = jax.lax.all_to_all(
            final_meta, axis_name="cores", split_axis=0, concat_axis=0, tiled=True)
        recv_flat = recv_meta.reshape(-1, META_SIZE)
        recv_hash = jax.lax.bitcast_convert_type(recv_flat[:, 0:8], jnp.int64)
        recv_pl = jax.lax.bitcast_convert_type(recv_flat[:, 8:12], jnp.uint32).astype(jnp.int32)
        recv_move = recv_flat[:, 12].astype(jnp.int32)
        recv_sc_u16 = jax.lax.bitcast_convert_type(recv_flat[:, 13:15], jnp.uint16)
        recv_score = jax.lax.bitcast_convert_type(recv_sc_u16, jnp.bfloat16).astype(jnp.float32)
        recv_valid = recv_flat[:, 15] > 0

        # Dedup by hash + top-B by score -- NO state gather (small fields only).
        sort_idx = jnp.argsort(recv_hash)
        sorted_hash = recv_hash[sort_idx]
        sorted_score = recv_score[sort_idx]
        sorted_pl = recv_pl[sort_idx]
        sorted_move = recv_move[sort_idx]
        sorted_src = sender_rank_per_recv[sort_idx]
        sorted_valid = recv_valid[sort_idx]
        is_dup = jnp.concatenate([jnp.zeros(1, dtype=jnp.bool_),
                                  sorted_hash[1:] == sorted_hash[:-1]])
        sorted_score_m = jnp.where(is_dup | (~sorted_valid), BIG_F32, sorted_score)
        top_v_keep, keep = sel_topk(sorted_score_m, B_local)

        sel_hash = sorted_hash[keep]
        sel_pl = sorted_pl[keep]
        sel_src = sorted_src[keep].astype(jnp.int32)
        sel_move = sorted_move[keep]
        sel_is_real = top_v_keep < (BIG_F32 * 0.5)

        packed_backptr = (
            sel_pl.astype(jnp.uint32)
            | (sel_src.astype(jnp.uint32) << jnp.uint32(BPTR_RANK_SHIFT))
            | (sel_move.astype(jnp.uint32) << jnp.uint32(BPTR_MOVE_SHIFT)))

        # --- Materialize the B survivors' states via a 2-hop round-trip. ---
        target_idx = jnp.arange(B_local, dtype=jnp.int32)
        onehot = (sel_src[:, None] == owner_arange[None, :]).astype(jnp.int32)
        cumpos = jnp.cumsum(onehot, axis=0) - 1
        my_pos = jnp.take_along_axis(cumpos, sel_src[:, None], axis=1)[:, 0]
        req_valid = (my_pos < REQ_CAP) & sel_is_real
        safe_pos = jnp.where(req_valid, my_pos, REQ_CAP)  # overflow -> dump slot

        req_rec = jnp.concatenate([
            _bytes(sel_pl, jnp.uint32),                            # (B,4)
            sel_move.astype(jnp.uint8)[:, None],                   # (B,1)
            jnp.where(req_valid, jnp.uint8(1), jnp.uint8(0))[:, None],  # (B,1)
            _bytes(target_idx.astype(jnp.uint32), jnp.uint32),     # (B,4)
            jnp.zeros((B_local, 2), dtype=jnp.uint8),              # pad -> 12
        ], axis=1)
        req_send = jnp.zeros((world_size, REQ_CAP + 1, REQ_REC), dtype=jnp.uint8)
        req_send = req_send.at[sel_src, safe_pos].set(req_rec)
        req_send = req_send[:, :REQ_CAP, :]

        req_recv = jax.lax.all_to_all(
            req_send, axis_name="cores", split_axis=0, concat_axis=0, tiled=True)
        req_rflat = req_recv.reshape(-1, REQ_REC)          # owner = idx // REQ_CAP

        # Source materializes responses CHUNKED so the per-row apply-move never
        # allocates a full (ws*REQ_CAP, S) generator-index array. Each chunk does
        # a (MAT_CHUNK, S) gather + take_along_axis (tiny); lax.scan stacks the
        # (MAT_CHUNK, RESP_REC) outputs into the resp buffer needed for the a2a.
        n_mat = (world_size * REQ_CAP) // MAT_CHUNK

        def _mat_body(_carry, ci):
            base = ci * jnp.int32(MAT_CHUNK)
            sl = jax.lax.dynamic_slice(req_rflat, (base, jnp.int32(0)), (MAT_CHUNK, REQ_REC))
            c_pl = jax.lax.bitcast_convert_type(sl[:, 0:4], jnp.uint32).astype(jnp.int32)
            c_move = sl[:, 4].astype(jnp.int32)
            c_valid = sl[:, 5] > 0
            parent = states[jnp.clip(c_pl, 0, B_local - 1)]
            gen = all_moves[jnp.clip(c_move, 0, n_gen - 1)]
            child = jnp.take_along_axis(parent, gen, axis=1)
            child = jnp.where(c_valid[:, None], child, jnp.zeros_like(child))
            resp = jnp.concatenate([
                _u8(child),                                            # (.,120)
                sl[:, 6:10],                                           # target bytes (.,4)
                jnp.where(c_valid, jnp.uint8(1), jnp.uint8(0))[:, None],  # (.,1)
                jnp.zeros((MAT_CHUNK, 3), dtype=jnp.uint8),            # pad -> 128
            ], axis=1)
            return _carry, resp

        _, resp_chunks = jax.lax.scan(_mat_body, None, jnp.arange(n_mat, dtype=jnp.int32))
        resp_send = resp_chunks.reshape(world_size, REQ_CAP, RESP_REC)

        resp_recv = jax.lax.all_to_all(
            resp_send, axis_name="cores", split_axis=0, concat_axis=0, tiled=True)
        resp_rflat = resp_recv.reshape(-1, RESP_REC)
        rc_child = resp_rflat[:, 0:120].astype(jnp.int8)
        rc_target = jax.lax.bitcast_convert_type(resp_rflat[:, 120:124], jnp.uint32).astype(jnp.int32)
        rc_valid = resp_rflat[:, 124] > 0
        safe_t = jnp.where(rc_valid, rc_target, B_local)   # invalid -> dump row
        next_frontier = jnp.zeros((B_local + 1, state_size), dtype=jnp.int8)
        next_frontier = next_frontier.at[safe_t].set(rc_child)
        chosen_states = next_frontier[:B_local]

        chosen_h = sel_hash
        chosen_state_sum = jnp.sum(chosen_states.astype(jnp.int32), axis=1)
        chosen_is_real = (chosen_state_sum != 0)

        new_min_v_log = min_v_log.at[j].set(jnp.min(top_v_keep))

        eq_v0 = (chosen_h == V0_hash) & chosen_is_real
        any_hit = jnp.any(eq_v0)
        pos_hit = jnp.argmax(eq_v0.astype(jnp.int32)).astype(jnp.int32)
        is_first_hit = (found_step == -1) & any_hit
        new_found_step = jnp.where(is_first_hit, j, found_step)
        new_found_pos_local = jnp.where(is_first_hit, pos_hit, found_pos_local)
        new_found_pos_rank = jnp.where(is_first_hit, rank_int, found_pos_rank)
        new_verify_state = verify_state  # unused by the packed wrapper

        if use_neighborhood:
            nb_pos = jnp.clip(jnp.searchsorted(nbhd_hash_sorted, chosen_h), 0, M_nbhd - 1)
            nb_match = (nbhd_hash_sorted[nb_pos] == chosen_h) & chosen_is_real
            nb_suf = jnp.where(nb_match, nbhd_suffix_len[nb_pos].astype(jnp.int32), jnp.int32(127))
            near_argmin = jnp.argmin(nb_suf).astype(jnp.int32)
            near_any = jnp.any(nb_match).astype(jnp.int32)
            near_minlen = nb_suf[near_argmin].astype(jnp.int32)
            near_pos = near_argmin
            near_sufidx = nb_pos[near_argmin].astype(jnp.int32)
        else:
            near_any = jnp.int32(0)
            near_minlen = jnp.int32(127)
            near_pos = jnp.int32(0)
            near_sufidx = jnp.int32(0)

        return (chosen_states[None, :, :],
                packed_backptr[None, :],
                new_min_v_log[None, :],
                new_found_step[None],
                new_found_pos_local[None],
                new_found_pos_rank[None],
                new_verify_state[None, :],
                near_any[None],
                near_minlen[None],
                near_pos[None],
                near_sufidx[None])

    return beam_step_local


def beam_solve_v_qshort_spmd_packed(
    init_state_list: list[int],
    v_params,
    q_params,
    alpha: int,
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
    nbhd_hash_sorted=None,
    nbhd_suffix_len=None,
    nbhd_suffix_moves=None,
    use_approx_topk: bool = False,
    approx_recall: float = 0.95,
    use_meta_materialize: bool = False,
    alpha_req: float = 1.5,
    profile_skip: str = "",
    lean_merge: bool = False,
) -> dict[str, Any]:
    """V+QSHORT SPMD solver: streaming Phase C + packed host-memmap tree + early stop.

    Adds qshort prefilter to beam_solve_v_only_spmd_packed: per chunk, the
    student Q-head scores parents and selects top-α children per parent;
    V is then run only on those α×parent_chunk children (12× fewer V evals
    at α=2 vs. evaluating all 24×parent_chunk children).

    `parent_chunk` is REQUIRED (no non-streaming variant in this module).
    At B=48M (B_local=6M), use parent_chunk=131072 (47 chunks per step).

    `pack_v_score` (Phase D):
      * False (default): receive side re-runs V on packed candidates.
      * True: send side packs the bf16 V score into bucket bytes 125-126;
        receive side unpacks and skips the V forward. Saves ~8% V compute.
        bf16 tie-breaks may shift path lengths by 0-2 moves per pid.

    Differences from `beam_solve_v_only_spmd`:
      * Per-step backpointer is a packed uint32 emitted by the body and
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
    File size = num_steps * world_size * B_local * 4 bytes
    (e.g. 7.68 GB at B_local=2M, 15.36 GB at B_local=4M; both fit Kaggle's
    /kaggle/working).
    """
    devices = mesh.devices.flatten()
    world_size = len(devices)
    # Packed-backpointer field-width guards: parent_local/rank/move must fit the
    # uint32 memmap layout (see BPTR_* constants). Overflow corrupts the walkback
    # SILENTLY (found=False with valid V0 hash hits) -- fail loudly instead.
    if B_local > BPTR_MAX_B_LOCAL:
        raise ValueError(
            f"B_local={B_local:,} exceeds packed-backpointer capacity "
            f"{BPTR_MAX_B_LOCAL:,} ({BPTR_PL_BITS}-bit parent_local). "
            f"B_global <= {BPTR_MAX_B_LOCAL * world_size:,}. For wider beams "
            f"switch the tree memmap to uint64 and widen BPTR_PL_BITS."
        )
    if world_size > (1 << BPTR_RANK_BITS):
        raise ValueError(f"world_size={world_size} exceeds {1 << BPTR_RANK_BITS} ranks")
    if n_gen > (1 << BPTR_MOVE_BITS):
        raise ValueError(f"n_gen={n_gen} exceeds {1 << BPTR_MOVE_BITS} moves")
    init_state = np.asarray(init_state_list, dtype=np.int8)
    if np.array_equal(init_state, np.asarray(V0)):
        return {"found": True, "path_len": 0, "path_idx": [], "found_step": -1, "wall_s": 0.0}

    if progress_every:
        print(
            f"[solve] B_local={B_local:,} K_per_peer={K_per_peer:,} "
            f"parent_chunk={parent_chunk} internal_bs={internal_bs} "
            f"pack_v_score={pack_v_score} num_steps={num_steps}",
            flush=True,
        )

    # Step 0: V on 24 first-move children (same as non-packed variant).
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

    # Seed shard on host (one core's worth). Building the full
    # (world_size, B_local, S) beam on one device then resharding materializes
    # ~world_size*B_local*S bytes on a single core and OOMs past ~48M; instead
    # we hand each core its own copy of states0 directly (see make_array_from_callback below).
    states0_np = np.asarray(states0)  # (B_local, S) host

    # Seed-move array (populates memmap[0] -- parent_local=0, parent_rank=0).
    top_idx0_i8 = top_idx0.astype(jnp.int8)
    if k0 < B_local:
        last_move = top_idx0_i8[k0 - 1]
        pad_moves = jnp.broadcast_to(last_move, (B_local - k0,))
        move0_full = jnp.concatenate([top_idx0_i8, pad_moves])
    else:
        move0_full = top_idx0_i8[:B_local]
    move0_full_np = np.asarray(move0_full)  # (B_local,) int8

    min_v_log = jnp.full((world_size, num_steps), 1e6, dtype=jnp.float32)
    found_step = jnp.full((world_size,), -1, dtype=jnp.int32)
    found_pos_local = jnp.full((world_size,), -1, dtype=jnp.int32)
    found_pos_rank = jnp.full((world_size,), -1, dtype=jnp.int32)
    verify_state = jnp.zeros((world_size, state_size), dtype=jnp.int8)

    # Early V0 check on the seed beam.
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

    if parent_chunk is None:
        raise ValueError(
            "jax_beam_spmd_v_qshort: parent_chunk is REQUIRED (streaming-only). "
            "For B_GLOBAL >= 16M use parent_chunk=65536 or 131072."
        )
    # Idea (2): solved-neighborhood early-stop arrays. Hashes + suffix lengths
    # go to device (small); the actual suffix moves stay host-side for final
    # path assembly. None -> feature off (behaviour identical to before).
    use_neighborhood = nbhd_hash_sorted is not None
    if use_neighborhood:
        nbhd_hash_dev = jnp.asarray(np.asarray(nbhd_hash_sorted, dtype=np.int64))
        nbhd_suflen_dev = jnp.asarray(np.asarray(nbhd_suffix_len, dtype=np.int8))
        nbhd_suffix_moves_host = np.asarray(nbhd_suffix_moves, dtype=np.int8)
        if progress_every:
            print(f"[neighborhood] M={nbhd_hash_dev.shape[0]:,} "
                  f"max_suffix={nbhd_suffix_moves_host.shape[1]}", flush=True)
    else:
        nbhd_hash_dev = None
        nbhd_suflen_dev = None
        nbhd_suffix_moves_host = None

    if use_meta_materialize:
        # Idea (4): compact-meta forward + deferred state materialization.
        if progress_every:
            print(f"[meta-materialize] ON  alpha_req={alpha_req}", flush=True)
        step_body = _build_step_body_v_qshort_meta_streaming(
            v_params, q_params, int(alpha),
            all_moves, V0, hash_vec, jnp.int64(V0_hash_host),
            B_local, world_size, K_per_peer, n_gen, state_size,
            dtype, int(internal_bs),
            parent_chunk=int(parent_chunk),
            alpha_req=float(alpha_req),
            use_neighborhood=use_neighborhood,
            nbhd_hash_sorted=nbhd_hash_dev,
            nbhd_suffix_len=nbhd_suflen_dev,
            use_approx_topk=bool(use_approx_topk),
            approx_recall=float(approx_recall),
        )
    else:
        step_body = _build_step_body_v_qshort_packed_streaming(
            v_params, q_params, int(alpha),
            all_moves, V0, hash_vec, jnp.int64(V0_hash_host),
            B_local, world_size, K_per_peer, n_gen, state_size,
            dtype, int(internal_bs),
            parent_chunk=int(parent_chunk),
            pack_v_score=bool(pack_v_score),
            owner_hash_vec=owner_hash_vec,
            use_neighborhood=use_neighborhood,
            nbhd_hash_sorted=nbhd_hash_dev,
            nbhd_suffix_len=nbhd_suflen_dev,
            use_approx_topk=bool(use_approx_topk),
            approx_recall=float(approx_recall),
            profile_skip=profile_skip,
            lean_merge=lean_merge,
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
            # 7 carried/emitted + 4 neighborhood (near_any, near_minlen,
            # near_pos, near_sufidx).
            out_specs=(P("cores"),) * 11,
            check_rep=False,  # jax 0.6.2: replicated table vs varying query in
                              # searchsorted is fine; skip the strict VMA match.
        )(states, mv_log, fs, fpl, fpr, vstate, j_arr)

    from jax.sharding import NamedSharding
    sharding_states = NamedSharding(mesh, P("cores"))

    # Build the sharded seed beam directly -- each core gets its own copy of
    # states0 as a (1, B_local, S) shard; the full beam is never on one device.
    states_d = jax.make_array_from_callback(
        (world_size, B_local, state_size), sharding_states,
        lambda _idx: states0_np[None],
    )
    mv_log_d = jax.device_put(min_v_log, sharding_states)
    fs_d = jax.device_put(found_step, sharding_states)
    fpl_d = jax.device_put(found_pos_local, sharding_states)
    fpr_d = jax.device_put(found_pos_rank, sharding_states)
    vstate_d = jax.device_put(verify_state, sharding_states)

    # Host memmap for the packed tree.
    if tree_path is None:
        work_dir = "/kaggle/working"
        if not os.path.isdir(work_dir):
            work_dir = tempfile.gettempdir()
        fd, tree_path = tempfile.mkstemp(prefix="tree_", suffix=".u32", dir=work_dir)
        os.close(fd)
    tree_mm = np.memmap(
        tree_path, mode="w+", dtype=np.uint32,
        shape=(num_steps, world_size, B_local),
    )
    # Seed move at j=0: parent_local=0, parent_rank=0, move=top_idx0[k] (replicated).
    seed_packed = (move0_full_np.astype(np.uint32) << BPTR_MOVE_SHIFT)  # (B_local,) uint32
    tree_mm[0, :, :] = np.broadcast_to(seed_packed[None, :], (world_size, B_local))

    import time

    # Pre-compile step_fn explicitly so compile time is separately measurable.
    if progress_every:
        print("[lower] tracing step_fn...", flush=True)
    t_lower_start = time.time()
    lowered = step_fn.lower(
        states_d, mv_log_d, fs_d, fpl_d, fpr_d, vstate_d, jnp.int32(1),
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

    t_start = time.time()
    first_iter_t = None
    last_completed_step = 0
    fs_per_rank = np.asarray(fs_d)  # initial state (all -1), re-read in loop.

    # Idea (2): running best neighborhood hit (minimises step + suffix_len).
    best_near_total = 1 << 30
    best_near = None  # (step, rank, pos_local, suffix_idx)

    try:
        for j in range(1, num_steps):
            t_iter = time.time()
            j_arr = jnp.int32(j)
            (states_d, packed_d, mv_log_d,
             fs_d, fpl_d, fpr_d, vstate_d,
             near_any_d, near_minlen_d, near_pos_d, near_sufidx_d) = compiled_step(
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

            fs_per_rank = np.asarray(fs_d)
            last_completed_step = j

            # Idea (2): fold this step's per-rank near-hits into the running best.
            if use_neighborhood:
                na = np.asarray(near_any_d)
                nl = np.asarray(near_minlen_d)
                npos = np.asarray(near_pos_d)
                nsuf = np.asarray(near_sufidx_d)
                for r in range(world_size):
                    if na[r]:
                        # A chosen state at step j is reached by j+1 moves
                        # (seed = step 0 = 1 move); path = (j+1) + suffix_len.
                        total = (j + 1) + int(nl[r])
                        if total < best_near_total:
                            best_near_total = total
                            best_near = (j, r, int(npos[r]), int(nsuf[r]))

            if first_iter_t is None:
                first_iter_t = time.time() - t_iter

            v0_hit = bool(np.any(fs_per_rank >= 0))
            # No future step can beat the best neighborhood total once j+1 reaches
            # it (a step j' > j has min path (j'+1)+0 >= (j+2) > best_near_total).
            near_done = best_near is not None and (j + 1) >= best_near_total

            if progress_every and (j == 1 or j % progress_every == 0 or v0_hit or near_done):
                _bn = best_near_total if best_near is not None else -1
                _minv = float(np.asarray(mv_log_d)[:, j].min())  # global best V at step j
                print(
                    f"[step {j:03d}/{num_steps-1}] device={t_device:.1f}s "
                    f"copy={t_copy:.2f}s write={t_write:.2f}s "
                    f"total={time.time()-t_iter:.1f}s "
                    f"fs={fs_per_rank.tolist()} min_v={_minv:.3f} best_near={_bn}",
                    flush=True,
                )

            if v0_hit or near_done:
                break

        tree_mm.flush()

        fpl_per_rank = np.asarray(fpl_d)
        mv_per_rank = np.asarray(mv_log_d)

        INT_MAX = 2 ** 30
        PL_MASK = BPTR_PL_MASK

        def _walk(step, rank, pos):
            # Walk the packed memmap from (step, rank, pos) back to the seed,
            # returning the move sequence (rotated frame) in forward order.
            out = []
            cur_rank, cur_pos = int(rank), int(pos)
            for jj in range(int(step), -1, -1):
                rec = int(tree_mm[jj, cur_rank, cur_pos])
                out.append((rec >> BPTR_MOVE_SHIFT) & BPTR_MOVE_MASK)
                if jj > 0:
                    cur_rank = (rec >> BPTR_RANK_SHIFT) & BPTR_RANK_MASK
                    cur_pos = rec & PL_MASK
            out.reverse()
            return out

        all_moves_np = np.asarray(all_moves)
        v0_np = np.asarray(V0).astype(np.int8)

        def _solves(path):
            cur = np.asarray(init_state, dtype=np.int8)
            for m in path:
                cur = cur[all_moves_np[m]]
            return np.array_equal(cur, v0_np)

        # Candidate 1: exact V0 hit (earliest across ranks). Verified here too,
        # which incidentally guards the single-hash V0 detection.
        candidates = []  # (path_len, path, found_step, via)
        fs_signed = np.where(fs_per_rank >= 0, fs_per_rank, INT_MAX)
        global_min_step = int(fs_signed.min())
        if global_min_step < INT_MAX:
            wr = int(np.where(fs_signed == global_min_step)[0][0])
            fs_h = int(fs_per_rank[wr])
            fpl_h = int(fpl_per_rank[wr])
            p_v0 = _walk(fs_h, wr, fpl_h)
            if _solves(p_v0):
                candidates.append((len(p_v0), p_v0, fs_h, "v0"))

        # Candidate 2 (idea 2): neighborhood hit -> walkback prefix + suffix.
        if best_near is not None:
            nstep, nrank, npos, nsuf = best_near
            suffix_len = best_near_total - nstep - 1  # prefix is nstep+1 moves
            prefix = _walk(nstep, nrank, npos)
            suffix = nbhd_suffix_moves_host[nsuf][:suffix_len].astype(int).tolist()
            p_near = prefix + suffix
            if _solves(p_near):
                candidates.append((len(p_near), p_near, nstep, "neighborhood"))

        if not candidates:
            return {"found": False, "path_len": 0, "path_idx": [],
                    "found_step": -1, "wall_s": time.time() - t_start,
                    "first_iter_s": first_iter_t,
                    "last_completed_step": last_completed_step,
                    "lower_s": t_lower, "compile_s": t_compile,
                    "min_v_trajectory_rank0": mv_per_rank[0].tolist()}

        candidates.sort(key=lambda c: c[0])
        best_len, best_path, best_step, best_via = candidates[0]

        return {
            "found": True,
            "path_len": best_len,
            "path_idx": best_path,
            "found_step": best_step,
            "via": best_via,
            "wall_s": time.time() - t_start,
            "first_iter_s": first_iter_t,
            "last_completed_step": last_completed_step,
            "lower_s": t_lower, "compile_s": t_compile,
            "min_v_trajectory_rank0": mv_per_rank[0].tolist(),
        }
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
