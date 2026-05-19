# TPU JAX 48M/64M Beam Speed Plan

This document is a detailed implementation guide for speeding up the AZ v4
V-only JAX SPMD beam after the 32M memory work. It focuses on the current
`cayleypy-megaminx-beam-az-v4-32m-dev` notebook, which can run up to 48M in
version 7 but spends a long time silently after:

```text
mesh: Mesh('cores': 8, axis_types=(Auto,))
```

That log line is printed before the first call to
`beam_solve_v_only_spmd_packed(...)`. The next top-level notebook print only
happens after the whole `(pid, rotation)` pair finishes, so the immediate
priority is to instrument the solver and separate compile time, first-step
runtime, host tree transfer time, and steady per-step time.

## Current Dev Notebook Shape

Main files:

- `megaminx/kaggle_notebooks/tpu_beam_az_v4_32m_dev/build_notebook.py`
- `megaminx/kaggle_notebooks/tpu_beam_spmd_jax/jax_beam_spmd_v_only.py`
- `megaminx/BEAM_32M_OPTIMIZATION_PLAN.md`

Current version-7 config:

```python
B_GLOBAL = 48 * 1024 * 1024
ALPHA_QSHORT = 1
INTERNAL_BS = 4096
NUM_STEPS = 100
PARENT_CHUNK = 32768
PACK_V_SCORE = True
```

Derived 48M sizes:

```text
B_LOCAL = 6,291,456
K_PER_PEER = 786,432
parent chunks per step = 192
children per parent chunk = 32,768 * 24 = 786,432
model_apply chunks per parent chunk = 786,432 / 4,096 = 192
model_apply chunks per beam step = 192 * 192 = 36,864
owner top-k merges per beam step = 192 * 8 = 1,536
packed tree transfer per step = 48M * 4 bytes = 192 MiB
```

This is the likely cause of the long runtime: the notebook fits 48M by using
very small chunks, but small chunks create huge scan/top-k/model-call overhead.

Derived 64M sizes:

```text
B_LOCAL = 8,388,608
K_PER_PEER = 1,048,576  # with alpha=1
parent chunks per step at PARENT_CHUNK=32768 = 256
model_apply chunks per step at INTERNAL_BS=4096 = 49,152
owner top-k merges per step = 2,048
packed tree transfer per step = 64M * 4 bytes = 256 MiB
80-step packed tree = 20 GiB
```

64M is possible only if we reduce overhead and avoid default per-step tree
logging for failed runs.

## Implementation Order

Do these in order. Each step should have a small smoke run before moving on.

1. Add timing/progress instrumentation.
2. Add search-only mode and replay-with-tree mode.
3. Tune `INTERNAL_BS` and `PARENT_CHUNK`.
4. Batch the 8 owner top-k computations.
5. Use a cheaper owner hash.
6. Rewrite receive-side dedup to avoid the second argsort.
7. Add bounded per-chunk shortlist / periodic merge.
8. Try async host copy only if tree logging remains necessary.
9. Attempt 64M with search-only first pass.

## Step 1: Add Timing And Progress Instrumentation

Patch target:

- `megaminx/kaggle_notebooks/tpu_beam_spmd_jax/jax_beam_spmd_v_only.py`
- Function: `beam_solve_v_only_spmd_packed`

Add an optional argument:

```python
progress_every: int = 1,
```

Add a banner before compiling/running:

```python
print(
    f"[solve] B_local={B_local:,} K_per_peer={K_per_peer:,} "
    f"parent_chunk={parent_chunk} internal_bs={internal_bs} "
    f"pack_v_score={pack_v_score}",
    flush=True,
)
```

Split compile from first execution. Replace direct use of `step_fn(...)` with a
compiled callable:

```python
t_compile = time.time()
compiled_step = step_fn.lower(
    states_d, mv_log_d, fs_d, fpl_d, fpr_d, vstate_d, jnp.int32(1)
).compile()
print(f"[compile] step_fn compiled in {time.time() - t_compile:.1f}s", flush=True)
```

Then call `compiled_step(...)` inside the loop:

```python
(states_d, packed_d, mv_log_d,
 fs_d, fpl_d, fpr_d, vstate_d) = compiled_step(
    states_d, mv_log_d, fs_d, fpl_d, fpr_d, vstate_d, j_arr,
)
```

Add per-step timing:

```python
t_step = time.time()
(states_d, packed_d, mv_log_d,
 fs_d, fpl_d, fpr_d, vstate_d) = compiled_step(...)
t_device_done = time.time()

packed_h = np.asarray(packed_d)
t_copy_done = time.time()

tree_mm[j, :, :] = packed_h
t_write_done = time.time()

fs_per_rank = np.asarray(fs_d)
t_fs_done = time.time()

if progress_every and (j == 1 or j % progress_every == 0 or np.any(fs_per_rank >= 0)):
    print(
        f"[step {j:03d}/{num_steps - 1}] "
        f"device+sync={t_device_done - t_step:.1f}s "
        f"copy={t_copy_done - t_device_done:.1f}s "
        f"write={t_write_done - t_copy_done:.1f}s "
        f"fs={t_fs_done - t_write_done:.1f}s "
        f"total={t_fs_done - t_step:.1f}s "
        f"found={fs_per_rank.tolist()}",
        flush=True,
    )
```

Expected outcome:

- We know whether the silence after `mesh:` is compile, first execution, tree
  transfer, or normal steady-state runtime.
- The Kaggle log gives progress every step.

Acceptance test:

```python
B_GLOBAL = 8 * 1024 * 1024
START_PID = 0
END_PID = 1
NUM_STEPS = 5
PARENT_CHUNK = 32768
INTERNAL_BS = 4096
progress_every = 1
```

Verify that all timing lines appear and the path still verifies if found.

## Step 2: Add Search-Only Mode

Tree logging is expensive:

```text
48M: 192 MiB/step
64M: 256 MiB/step
```

For rescue, most very-wide runs may fail to find a shorter path. Writing a full
tree for failed runs is pure overhead. Add a first-pass mode that does not
return or store `packed_backptr`.

Patch target:

- `jax_beam_spmd_v_only.py`

Add a solver argument:

```python
record_tree: bool = True
```

Create a second body builder or a flag in the current builder:

```python
def _build_step_body_v_only_packed_streaming(..., record_tree=True):
    ...
    if record_tree:
        return (chosen_states[None, :, :],
                packed_backptr[None, :],
                new_min_v_log[None, :],
                new_found_step[None],
                new_found_pos_local[None],
                new_found_pos_rank[None],
                new_verify_state[None, :])
    else:
        return (chosen_states[None, :, :],
                new_min_v_log[None, :],
                new_found_step[None],
                new_found_pos_local[None],
                new_found_pos_rank[None],
                new_verify_state[None, :])
```

Use two separate `step_fn` definitions because output arity is static:

```python
if record_tree:
    @partial(jax.jit, donate_argnums=(0, 1, 2, 3, 4, 5))
    def step_fn(...):
        return shard_map(..., out_specs=(P("cores"),) * 7)(...)
else:
    @partial(jax.jit, donate_argnums=(0, 1, 2, 3, 4, 5))
    def step_fn(...):
        return shard_map(..., out_specs=(P("cores"),) * 6)(...)
```

When `record_tree=False`:

- Do not create `tree_mm`.
- Do not transfer `packed_d`.
- Do not return `path_idx`.
- Return `found`, `found_step`, `found_pos_local`, `found_pos_rank`,
  `last_completed_step`, timings, and `verify_state`.

Return shape:

```python
return {
    "found": global_min_step < INT_MAX,
    "path_len": 0,
    "path_idx": [],
    "found_step": fs_h if found else -1,
    "found_pos_local": fpl_h if found else -1,
    "found_pos_rank": winner_rank if found else -1,
    "needs_replay": bool(found),
    "wall_s": time.time() - t_start,
}
```

Then add a wrapper helper:

```python
def beam_solve_v_only_spmd_search_then_replay(...):
    r = beam_solve_v_only_spmd_packed(..., record_tree=False)
    if not r["found"]:
        return r
    replay_steps = int(r["found_step"]) + 1
    return beam_solve_v_only_spmd_packed(
        ..., num_steps=replay_steps, record_tree=True
    )
```

Important:

- This doubles compute only for successful pids.
- For failed pids, it removes all packed tree transfers and memmap writes.
- For successful rescue pids, it records only up to `found_step`, not default
  `NUM_STEPS`.

Notebook config knobs:

```python
SEARCH_THEN_REPLAY = True
RECORD_TREE = not SEARCH_THEN_REPLAY
```

Acceptance test:

1. Run `record_tree=True` and `record_tree=False` at small beam.
2. Confirm both report the same `found_step`.
3. Confirm `search_then_replay` reconstructs and verifies the same path.

## Step 3: Tune `INTERNAL_BS`

Current `INTERNAL_BS=4096` creates too many model chunks. Increase it until HBM
or compile fails.

Patch target:

- `build_notebook.py`, config cell.

Experiment ladder at 48M:

```python
INTERNAL_BS = 8192
INTERNAL_BS = 16384
INTERNAL_BS = 32768
```

Keep `PARENT_CHUNK=32768` for the first internal-batch sweep so only one
variable changes.

Expected model-call reduction at 48M:

```text
4096:  36,864 model chunks/step
8192:  18,432 model chunks/step
16384:  9,216 model chunks/step
32768:  4,608 model chunks/step
```

Risk:

- Larger `INTERNAL_BS` increases activation memory inside `model_apply`.
- It can force OOM at 48M/64M.

Acceptance:

- First step completes.
- Per-step timing improves.
- HBM remains stable.

Recommendation:

- Use the largest value that completes at least 5 steps.
- For 64M, start one rung below the largest safe 48M value.

## Step 4: Tune `PARENT_CHUNK`

Current `PARENT_CHUNK=32768` creates many outer scan iterations and owner top-k
merges. Increase it once `INTERNAL_BS` has been tuned.

Patch target:

- `build_notebook.py`, config cell.

Experiment ladder at 48M:

```python
PARENT_CHUNK = 65536
PARENT_CHUNK = 131072
PARENT_CHUNK = 262144
PARENT_CHUNK = 524288  # aggressive
```

Derived outer chunks at 48M:

```text
32768:  192 chunks/step
65536:   96 chunks/step
131072:  48 chunks/step
262144:  24 chunks/step
524288:  12 chunks/step
```

Derived outer chunks at 64M:

```text
32768:  256 chunks/step
65536:  128 chunks/step
131072:  64 chunks/step
262144:  32 chunks/step
524288:  16 chunks/step
```

Risk:

- Larger parent chunks materialize more child states and V activations:

```text
child states = PARENT_CHUNK * 24 * 120 bytes
65536:  180 MiB
131072: 360 MiB
262144: 720 MiB
524288: 1.44 GiB
```

Recommendation:

- Try `131072` first.
- If 48M still has HBM headroom, try `262144`.
- For 64M, target `131072` or `262144`.

## Step 5: Batch The Owner Top-K

Current streaming body loops:

```python
for dest in range(world_size):
    masked_scores = jnp.where(owner == dest, child_v, BIG_F32)
    merged_scores = jnp.concatenate([new_top_scores[dest], masked_scores], axis=0)
    top_v_new, keep = _topk_smallest(merged_scores, K_per_peer)
    ...
```

Patch target:

- `_build_step_body_v_only_packed_streaming`
- Inner `chunk_body`

Replace with a batched owner axis. The basic idea:

```python
dest_ids = jnp.arange(world_size, dtype=owner.dtype)[:, None]
masked_scores = jnp.where(
    owner[None, :] == dest_ids,
    child_v[None, :],
    BIG_F32,
)  # (world_size, chunk_n)

merged_scores = jnp.concatenate([top_scores, masked_scores], axis=1)
top_v_new, keep = _topk_smallest(merged_scores, K_per_peer)
```

Then gather old/new packs in batched form. Pseudocode:

```python
from_old = keep < K_per_peer
old_idx = jnp.clip(keep, 0, K_per_peer - 1)
new_idx = jnp.clip(keep - K_per_peer, 0, chunk_n - 1)

owner_rows = jnp.arange(world_size, dtype=jnp.int32)[:, None]
old_pack = top_pack[owner_rows, old_idx]  # (world_size, K_per_peer, PACK_SIZE)

sel_states = children[new_idx]  # (world_size, K_per_peer, state_size)
sel_parent_local = (parent_start + (new_idx // n_gen)).astype(jnp.int32)
sel_move = (new_idx % n_gen).astype(jnp.int8)
new_pack = pack_records_batched(sel_states, sel_parent_local, sel_move, top_v_new)

top_pack_new = jnp.where(from_old[:, :, None], old_pack, new_pack)
```

Add helper:

```python
def _pack_records_batched(sel_states, sel_parent_local, sel_move, scores, is_pad):
    # sel_states: (world_size, K, state_size)
    # returns: (world_size, K, PACK_SIZE)
```

Rationale:

- Reduces Python-unrolled HLO size.
- Gives XLA a single batched top-k over the owner axis.
- Removes 8 repeated chunks of similar code.

Risk:

- The batched `sel_states = children[new_idx]` may create a large
  `(world_size, K_per_peer, 120)` temporary. Watch HBM.
- If it OOMs, use this only after bounded per-chunk shortlist is implemented.

Acceptance:

- At 8M or 16M, compare against old streaming body on a few pids.
- Verify paths and path lengths.
- Benchmark first-step and steady-step time.

## Step 6: Cheap Owner Hash

The current child-side hash uses int64:

```python
h = jnp.sum(children.astype(jnp.int64) * hash_vec, axis=1)
owner = (h % jnp.int64(world_size)).astype(jnp.int32)
```

Owner routing only needs a uniform bucket from `0..7`. It does not need
collision resistance. Dedup and V0 detection still need the existing int64 hash
after receive.

Patch target:

- Add `owner_hash_vec` alongside `hash_vec`.
- Pass it into `_build_step_body_v_only_packed_streaming`.

Create once in `CELL_9_LOAD_PARAMS`:

```python
rng = np.random.default_rng(12345)
owner_hash_vec_np = rng.integers(
    0, np.iinfo(np.uint32).max,
    size=STATE_SIZE,
    dtype=np.uint32,
)
owner_hash_vec = jnp.asarray(owner_hash_vec_np)
```

Inside child streaming:

```python
h_owner = jnp.sum(children.astype(jnp.uint32) * owner_hash_vec, axis=1)
owner = (h_owner & jnp.uint32(world_size - 1)).astype(jnp.uint8)
```

Because `world_size=8`, bitmask is cheaper than modulo.

Keep receive-side dedup unchanged:

```python
recv_h = jnp.sum(recv_states.astype(jnp.int64) * hash_vec, axis=1)
```

Risk:

- Owner distribution must be uniform enough. Measure counts:

```python
owner_counts = jnp.sum(owner[None, :] == jnp.arange(world_size)[:, None], axis=1)
```

Do not print counts every step in production, but test once at small beam.

Acceptance:

- Owner counts are within a few percent for large chunks.
- Paths verify.

## Step 7: One-Sort Receive Dedup

Current receive dedup does:

```python
sort_h = jnp.sort(recv_h)
sort_idx = jnp.argsort(recv_h)
is_dup_sorted = ...
restore = jnp.argsort(sort_idx)
dup_mask = is_dup_sorted[restore]
recv_v_masked = jnp.where(dup_mask | is_padding, BIG_F32, recv_v)
top_v_keep, keep_idx = _topk_smallest(recv_v_masked, B_local)
```

This uses two argsorts plus a sort. Replace with one argsort and top-k in sorted
order.

Patch target:

- Receive side in both non-streaming and streaming packed bodies.

New flow:

```python
sort_idx = jnp.argsort(recv_h)
sorted_h = recv_h[sort_idx]
sorted_states = recv_states[sort_idx]
sorted_parent_local = recv_parent_local[sort_idx]
sorted_move = recv_move[sort_idx]
sorted_sender_rank = recv_sender_rank[sort_idx]
sorted_is_padding = is_padding[sort_idx]
sorted_v = recv_v[sort_idx]

is_dup_sorted = jnp.concatenate([
    jnp.zeros(1, dtype=jnp.bool_),
    sorted_h[1:] == sorted_h[:-1],
])
sorted_v_masked = jnp.where(is_dup_sorted | sorted_is_padding, BIG_F32, sorted_v)

top_v_keep, keep_sorted_idx = _topk_smallest(sorted_v_masked, B_local)
chosen_states = sorted_states[keep_sorted_idx]
chosen_parent_local = sorted_parent_local[keep_sorted_idx]
chosen_parent_rank = sorted_sender_rank[keep_sorted_idx]
chosen_move = sorted_move[keep_sorted_idx]
chosen_h = sorted_h[keep_sorted_idx]
```

Rationale:

- Removes `jnp.sort(recv_h)` because `recv_h[argsort]` gives sorted hashes.
- Removes `restore = jnp.argsort(sort_idx)`.
- Top-k does not require original order; sorted order is fine.

Tie behavior:

- May differ when equal scores exist.
- Verification remains the real acceptance criterion.

Acceptance:

- Small-beam comparison: path verifies and path length matches or is within
  expected tie drift.
- Measure receive-side timing if instrumentation allows.

## Step 8: Bounded Per-Chunk Shortlist

Current streaming is exact but wasteful:

```python
topk(concat(running_K, all chunk candidates), K_per_peer)
```

At 48M:

```text
K_PER_PEER = 786,432
chunk_n = 786,432
expected candidates per owner per chunk = 98,304
```

For each owner, the code still asks top-k for `786,432` results per chunk, most
of which cannot be real for that chunk. Use a bounded per-chunk shortlist first.

Patch target:

- Streaming `chunk_body`

Add static argument:

```python
chunk_owner_k: int | None = None
merge_every: int = 1
```

Suggested values:

```python
chunk_owner_k = min(K_per_peer, max(131072, parent_chunk * n_gen // world_size * 2))
```

For `PARENT_CHUNK=32768`:

```text
expected per owner = 98,304
chunk_owner_k = 196,608
```

For `PARENT_CHUNK=131072`:

```text
expected per owner = 393,216
chunk_owner_k = 786,432
```

Two-stage flow:

```python
# 1. Per-chunk, per-owner shortlist.
chunk_v, chunk_idx = _topk_smallest(masked_scores, chunk_owner_k)
chunk_pack = pack_records(...)

# 2. Merge running top K with chunk shortlist.
merged_scores = jnp.concatenate([top_scores[dest], chunk_v], axis=0)
top_v_new, keep = _topk_smallest(merged_scores, K_per_peer)
```

This helps most when `parent_chunk` is small. If `parent_chunk` is large enough
that `expected_owner_count >= K_per_peer`, the benefit shrinks.

Approximate mode:

- If `chunk_owner_k < actual good candidates for an owner`, this can drop a
  candidate that would later survive globally.
- In practice owner hash is random and the margin can be generous.

Exact fallback:

- Set `chunk_owner_k = K_per_peer`.

Instrumentation:

Add optional underfill count:

```python
owner_count = jnp.sum(owner == dest)
underfilled = owner_count > chunk_owner_k
```

Do not transfer this every step in production. Use small diagnostic runs.

Acceptance:

- Compare exact and bounded modes at 8M/16M on 5 pids.
- If path lengths are stable, try 48M.

## Step 9: Async Host Copy For Tree Mode

If tree logging is still needed in the main pass, overlap host copy with next
step where possible.

JAX arrays support `copy_to_host_async()`, which requests host-cache population
without waiting for immediate host access. This is useful when a later
`np.asarray(...)` would otherwise initiate a blocking copy.

Patch target:

- Host loop in `beam_solve_v_only_spmd_packed`

Use a one-step delayed queue:

```python
pending = None
for j in range(1, num_steps):
    outputs = compiled_step(...)
    states_d, packed_d, mv_log_d, fs_d, fpl_d, fpr_d, vstate_d = outputs

    packed_d.copy_to_host_async()
    fs_d.copy_to_host_async()

    if pending is not None:
        prev_j, prev_packed_d = pending
        tree_mm[prev_j, :, :] = np.asarray(prev_packed_d)

    fs_per_rank = np.asarray(fs_d)
    pending = (j, packed_d)

    if np.any(fs_per_rank >= 0):
        break

if pending is not None:
    prev_j, prev_packed_d = pending
    tree_mm[prev_j, :, :] = np.asarray(prev_packed_d)
```

Risk:

- Donation: do not donate `packed_d`; it is an output, not an input.
- Holding pending output for one step increases host/device bookkeeping but not
  major HBM carry.
- The next step depends on `states_d`, not `packed_d`, so overlap should be
  possible.

Acceptance:

- Per-step `copy` timing drops or is hidden.
- Path reconstruction still works.

## Step 10: Incumbent-Limited Steps

For rescue, we do not need any solution. We need a path shorter than the current
best path.

Add config:

```python
INCUMBENT_CSV = "/kaggle/input/.../current_best.csv"  # optional
MAX_IMPROVE_STEPS = None  # if set, overrides NUM_STEPS per pid
```

For each pid:

```python
incumbent_len = len(current_best_path[pid])
num_steps_pid = min(NUM_STEPS, incumbent_len - 1)
```

If no incumbent is supplied, use `NUM_STEPS`.

Rationale:

- Saves steps on pids where current best is already short.
- Reduces tree size and wall time.
- Exact for improvement search.

Risk:

- If the run is meant to produce any valid path from scratch, do not cap this
  way.

Acceptance:

- For a pid with incumbent length `L`, no returned path should have length
  `>= L`.

## Step 11: Approximate Top-K Experiment

JAX has `jax.lax.approx_min_k`. This can be tried where exact `top_k` dominates.
Keep it as an experiment only.

Patch:

```python
def _topk_smallest(values, k, approximate=False, recall_target=0.99):
    if approximate:
        return jax.lax.approx_min_k(
            values, k,
            recall_target=recall_target,
            aggregate_to_topk=True,
        )
    neg_v, idx = jax.lax.top_k(-values, k)
    return -neg_v, idx
```

Config:

```python
APPROX_TOPK = False
APPROX_RECALL = 0.99
```

Try only after exact 48M timing is known.

Acceptance:

- Compare exact vs approximate at 8M/16M on 10 pids.
- Track solved count and path length drift.

## Step 12: 64M Run Recipe

Do not attempt 64M with current version-7 settings. Use the speed stack.

First 64M smoke:

```python
B_GLOBAL = 64 * 1024 * 1024
ALPHA_QSHORT = 1
PACK_V_SCORE = True
SEARCH_THEN_REPLAY = True
RECORD_TREE = False  # first pass
INTERNAL_BS = 8192   # raise later if HBM allows
PARENT_CHUNK = 131072
NUM_STEPS = min(80, incumbent_len - 1)
progress_every = 1
```

If first step completes:

```python
INTERNAL_BS = 16384
PARENT_CHUNK = 262144
```

If OOM:

```python
INTERNAL_BS = 4096
PARENT_CHUNK = 65536
```

Do not enable full tree logging for all 64M steps unless necessary. A full
80-step 64M tree is approximately:

```text
80 * 8 * 8,388,608 * 4 bytes = 20 GiB
```

That is right at Kaggle scratch limits and slow to write.

## Patch Checklist

### Required Before More 48M Runs

- [ ] Add compile and per-step timing logs.
- [ ] Add `progress_every`.
- [ ] Add `SEARCH_THEN_REPLAY` / `record_tree=False`.
- [ ] Try `INTERNAL_BS=8192`.
- [ ] Try `PARENT_CHUNK=65536` and `131072`.

### Required Before 64M

- [ ] Search-only first pass works.
- [ ] Replay-with-tree reconstructs verified path after search-only hit.
- [ ] Larger `PARENT_CHUNK` works at 48M.
- [ ] Larger `INTERNAL_BS` works at 48M.
- [ ] Host tree is not written for failed search-only runs.

### High-Value After 64M First Step

- [ ] Batched owner top-k.
- [ ] Cheap `uint32` owner hash.
- [ ] One-sort receive dedup.
- [ ] Bounded per-chunk shortlist.
- [ ] Async host copy if tree logging remains hot.

## Measurement Table To Fill In

Record this for every run:

```text
tag:
B_GLOBAL:
B_LOCAL:
ALPHA_QSHORT:
PARENT_CHUNK:
INTERNAL_BS:
PACK_V_SCORE:
SEARCH_THEN_REPLAY:
record_tree:
NUM_STEPS:
compile_s:
first_step_device_s:
first_step_copy_s:
first_step_write_s:
steady_step_total_s:
found_step:
path_len:
verify_ok:
OOM/error:
notes:
```

Suggested run tags:

```text
48m_v7_baseline
48m_ibs8192
48m_ibs16384
48m_pc65536
48m_pc131072
48m_search_only
48m_search_replay
64m_search_pc131k_ibs8192
64m_search_pc262k_ibs8192
```

## Expected Biggest Wins

Expected impact order:

1. Search-only first pass: removes 192-256 MiB host transfer per step for failed
   runs.
2. Larger `INTERNAL_BS`: directly reduces tens of thousands of model chunks per
   step.
3. Larger `PARENT_CHUNK`: reduces outer scan and owner top-k merge count.
4. Bounded per-chunk shortlist: reduces top-k work when chunks are small.
5. One-sort dedup: reduces receive-side sorting overhead.
6. Cheap owner hash: reduces int64 child-side work.
7. Batched owner top-k: may reduce HLO/loop overhead, but must be checked for
   HBM pressure.
8. Async host copy: useful only if tree logging remains in the hot path.

## Notes For Future AZ-Compatible QShort

The deeper speed solution is to stop evaluating AZ V on all 24 children. Train a
Q-shortlister distilled from AZ v4 V:

```text
Q_AZ(s, a) ~= V_AZ(apply(s, a))
```

Then use the existing qshort SPMD design:

1. Run cheap Q on parent states.
2. Route only a shortlist.
3. Rerank with AZ V.

The old `m23_v2` qshort regressed AZ because it was distilled from a different
teacher landscape, not because qshort is inherently bad. This is likely the
largest long-term wall-time win, but it is a training project, not a notebook
patch.

## Summary

The current 48M notebook fits by making `PARENT_CHUNK` and `INTERNAL_BS` very
small. That creates the runtime overhead. The fastest path forward is:

1. Instrument first, so the post-`mesh` silence becomes measurable.
2. Remove tree I/O from the main run with search-only plus replay.
3. Raise `INTERNAL_BS` and `PARENT_CHUNK` until HBM says no.
4. Reduce top-k/hash/dedup overhead.
5. Attempt 64M only in search-only mode first.

Do not spend more TPU quota on blind 48M/64M runs without per-step timing logs.
