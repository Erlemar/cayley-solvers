# TPU JAX 32M Beam Optimization Plan

This note is a concrete implementation plan for getting the AZ v4 V-only JAX
SPMD beam from the current verified `B_GLOBAL=8M` ceiling to `B_GLOBAL=32M` on
Kaggle TPU v5e-8.

The target is an exact 32M shared beam first. 64M is intentionally out of scope
except where a 32M change should be designed so it does not block a later 64M
attempt.

## Current Baseline

Main files:

- `megaminx/kaggle_notebooks/tpu_beam_az_v4_v_only_jax_shareable/build_notebook.py`
- `megaminx/kaggle_notebooks/tpu_beam_spmd_jax/jax_beam_spmd_v_only.py`
- `megaminx/kaggle_notebooks/tpu_beam_spmd_jax/jax_model.py`

Current production settings:

```python
B_GLOBAL = 8 * 1024 * 1024
B_LOCAL = B_GLOBAL // 8
ALPHA_QSHORT = 2
K_PER_PEER = (ALPHA_QSHORT * B_LOCAL) // 8
INTERNAL_BS = 32768
NUM_STEPS = 120
```

The current V-only step materializes all children at once:

```python
neighbors = states[:, all_moves].reshape(-1, state_size)
child_v = forward_v(v_params, neighbors)
h = jnp.sum(neighbors.astype(jnp.int64) * hash_vec, axis=1)
```

At `B_GLOBAL=32M`, `B_LOCAL=4M`, this `neighbors` tensor alone is:

```text
4,194,304 parents * 24 moves * 120 bytes = 12,079,595,520 bytes
```

That is about 11.25 GiB per TPU core before model activations, scores, hashes,
owner arrays, all-to-all buckets, receive buffers, dedup buffers, and tree
storage. This is the main blocker.

## 32M Design Goal

For 32M, the per-rank memory shape should become:

- Keep only the current beam states on device.
- Stream child generation in parent chunks.
- Maintain per-owner top `K_PER_PEER` candidates as the streaming carry.
- Route one packed bucket per owner with `lax.all_to_all`.
- Move full backpointer tree storage to host or disk.
- Keep only the current step backpointers on device.
- Preserve host-side cross-rank winner selection.

This keeps the exact shared-beam semantics, except for possible tie-order
differences from chunked top-k. Ties are not expected to matter for verified
path length, but the validation plan below compares against the old body at
small beam.

## Improvement List

### 1. Raise JAX Memory Fraction

Set before `import jax` in the notebook setup cell:

```python
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
```

Rationale:

- Cheap to try.
- May make more of device memory available to JAX/XLA.
- Not sufficient for 32M by itself.

Risk:

- On TPU this may be less relevant than on GPU.
- It should not be counted as a real 32M fix.

Implementation target:

- `build_notebook.py`, `CELL_2_SETUP`, before `import jax`.

### 2. Re-enable JAX Buffer Donation

Current `step_fn` is jitted without donation. Re-enable donation for the step
carry:

```python
from functools import partial

@partial(jax.jit, donate_argnums=tuple(range(9)))
def step_fn(states, tp_local, tp_rank, tmove, mv_log, fs, fpl, fpr, vstate, j_arr):
    ...
```

For the host-tree version, the donated argument list will shrink because the
tree tensors are no longer part of the device carry. Donate all same-shape
carry-in/carry-out arrays, but do not donate `j_arr`.

Rationale:

- The old donation concern came from the earlier misdiagnosis of the
  cross-rank winner bug.
- Host-side winner selection removed the unsafe in-body `pmin`/`psum` pattern.
- Donation is especially valuable for the full tree carry before it is removed,
  and still valuable for `states`.

Risk:

- JAX will error if a donated buffer is reused after the call.
- The Python wrapper must overwrite each donated variable immediately:

```python
states_d, mv_log_d, fs_d, fpl_d, fpr_d, vstate_d, backptr_d = step_fn(...)
```

Validation:

- Run 1M or 8M with donation and verify all paths.
- Watch for donated-buffer reuse errors.

### 3. Reduce `INTERNAL_BS`

Change the default:

```python
INTERNAL_BS = 16384
```

Fallback if 32M still OOMs:

```python
INTERNAL_BS = 8192
```

Rationale:

- Reduces model forward activation peak inside `_chunked_apply`.
- Particularly useful while streaming children, because each parent chunk still
  runs V over `C * 24` children.

Risk:

- First run at a new value recompiles.
- Smaller chunks can reduce TPU utilization.

Validation:

- Benchmark first iteration time and steady per-step time at 8M.
- Keep `16384` unless `8192` is needed for HBM.

### 4. Remove Full Per-Child Index Arrays

Current code precomputes:

```python
n_total = B_local * n_gen
flat_idx = jnp.arange(n_total, dtype=jnp.int32)
parent_local_per_child = (flat_idx // n_gen).astype(jnp.int32)
move_per_child = (flat_idx % n_gen).astype(jnp.int8)
```

For 32M this is hundreds of MiB per rank. Derive parent and move only after
top-k selects candidate indices:

```python
sel_parent_local = (top_idx_S // n_gen).astype(jnp.int32)
sel_move = (top_idx_S % n_gen).astype(jnp.int8)
```

For streaming chunks, include the parent offset:

```python
sel_parent_local = parent_start + (child_idx // n_gen).astype(jnp.int32)
sel_move = (child_idx % n_gen).astype(jnp.int8)
```

Rationale:

- Simple memory reduction.
- Also simplifies the streaming implementation.

Risk:

- Need to ensure `parent_start` is `int32` and does not overflow. It is safe at
  32M and 64M local sizes.

### 5. Move Tree Storage Out Of The Device Carry

Current device carry includes full tensors:

```python
tree_parent_local = jnp.zeros((world_size, num_steps, B_local), dtype=jnp.int32)
tree_parent_rank = jnp.zeros((world_size, num_steps, B_local), dtype=jnp.int8)
tree_move = jnp.full((world_size, num_steps, B_local), -1, dtype=jnp.int8)
```

At 32M, this is about 2.8 GiB per rank for the local shard, and the carry can
also cause old/new buffer overlap without donation.

Replace this with a per-step packed backpointer output:

```python
packed_backptr = (
    chosen_parent_local.astype(jnp.uint32)
    | (chosen_parent_rank.astype(jnp.uint32) << jnp.uint32(23))
    | (chosen_move.astype(jnp.uint32) << jnp.uint32(26))
)
```

Bit layout:

```text
bits  0..22: parent_local, supports up to 8,388,607
bits 23..25: parent_rank, 0..7
bits 26..30: move, 0..23
bit      31: unused
```

This supports:

- 32M global: `B_LOCAL=4,194,304`, needs 22 bits.
- 64M global later: `B_LOCAL=8,388,608`, needs 23 bits.

Wrapper structure:

```python
tree_path = f"/kaggle/working/tree_pid{pid}_rot{rot_i}.u32"
tree_mm = np.memmap(
    tree_path,
    mode="w+",
    dtype=np.uint32,
    shape=(num_steps, world_size, B_local),
)

for j in range(1, num_steps):
    states_d, packed_d, mv_log_d, fs_d, fpl_d, fpr_d, vstate_d = step_fn(...)
    packed_h = np.asarray(packed_d)
    tree_mm[j, :, :] = packed_h
    tree_mm.flush()

    fs_per_rank = np.asarray(fs_d)
    if np.any(fs_per_rank >= 0):
        break
```

Path walkback becomes:

```python
rec = int(tree_mm[j, cur_rank, cur_pos])
move = (rec >> 26) & 31
parent_rank = (rec >> 23) & 7
parent_local = rec & ((1 << 23) - 1)
```

Rationale:

- Removes multi-GiB tree carry from HBM.
- Makes donation much easier.
- Keeps path reconstruction host-side, matching the current fixed design.

Cost:

- 32M packed transfer is:

```text
8 ranks * 4,194,304 states * 4 bytes = 128 MiB per step
```

- Worst-case 120 steps is 15 GiB of host/disk writes per pid.
- This is acceptable for rescue-style 32M, but use memmap and delete the file
  after reconstructing the path.

Risk:

- Host transfer introduces synchronization each step.
- If per-step compute is very large at 32M, the transfer cost should still be
  secondary. Measure it.

### 6. Stream Child Generation With Per-Owner Top-K

This is the mandatory 32M change.

Do not materialize:

```python
neighbors = states[:, all_moves].reshape(-1, state_size)
```

Instead, scan over parent chunks:

```python
PARENT_CHUNK = 65536
assert B_local % PARENT_CHUNK == 0
n_parent_chunks = B_local // PARENT_CHUNK
```

Carry one running top-k set per destination owner:

```python
top_scores: (world_size, K_per_peer) float32 or bfloat16
top_pack:   (world_size, K_per_peer, PACK_SIZE) uint8
```

Chunk body sketch:

```python
def chunk_body(carry, chunk_i):
    top_scores, top_pack = carry

    parent_start = chunk_i * PARENT_CHUNK
    states_chunk = lax.dynamic_slice(
        states,
        (parent_start, 0),
        (PARENT_CHUNK, state_size),
    )

    children = states_chunk[:, all_moves].reshape(-1, state_size)
    child_v = forward_v(v_params, children)
    h = jnp.sum(children.astype(jnp.int64) * hash_vec, axis=1)
    owner = (h % jnp.int64(world_size)).astype(jnp.uint8)

    child_idx_all = jnp.arange(PARENT_CHUNK * n_gen, dtype=jnp.int32)
    child_parent = parent_start + (child_idx_all // n_gen)
    child_move = (child_idx_all % n_gen).astype(jnp.int8)

    for dest in range(world_size):
        masked_scores = jnp.where(owner == dest, child_v, BIG_F32)
        merged_scores = jnp.concatenate([top_scores[dest], masked_scores], axis=0)
        _, keep = _topk_smallest(merged_scores, K_per_peer)

        from_old = keep < K_per_peer
        old_idx = jnp.clip(keep, 0, K_per_peer - 1)
        new_idx = jnp.clip(keep - K_per_peer, 0, PARENT_CHUNK * n_gen - 1)

        old_pack = top_pack[dest][old_idx]
        new_pack = pack_records(
            children[new_idx],
            child_parent[new_idx],
            child_move[new_idx],
        )
        merged_pack = jnp.where(from_old[:, None], old_pack, new_pack)

        top_scores = top_scores.at[dest].set(merged_scores[keep])
        top_pack = top_pack.at[dest].set(merged_pack)

    return (top_scores, top_pack), None
```

Important:

- The `lax.scan` output must be `None`. Do not return per-chunk arrays.
- The only large scan carry should be the running top-k.
- Use clamped indices before `jnp.where`, because both branches may be evaluated.
- Keep `PACK_SIZE=128` initially to avoid changing all-to-all layout.

After the scan:

```python
send_buckets = top_pack
recv_buckets = jax.lax.all_to_all(
    send_buckets,
    axis_name="cores",
    split_axis=0,
    concat_axis=0,
    tiled=True,
)
```

Rationale:

- Reduces child state peak from 11.25 GiB at 32M to:

```text
PARENT_CHUNK * 24 * 120 bytes
```

- With `PARENT_CHUNK=65,536`, child states are about 180 MiB.
- This is the difference between impossible and plausible.

Risk:

- Top-k over `K_per_peer + PARENT_CHUNK * 24` is expensive and runs for each
  owner and each chunk.
- Exact streaming can be slower than the current full-materialization body.
- 32M V-only is rescue-only compute either way.

Tuning:

- Start with `PARENT_CHUNK=65536`.
- Try `131072` if HBM allows and top-k overhead dominates.
- Keep `PARENT_CHUNK` a power of two that divides `B_LOCAL`.

### 7. Keep Receive-Side Dedup Initially

The receive side currently does:

```python
recv_flat = recv_buckets.reshape(-1, PACK_SIZE)
recv_states = recv_flat[:, 0:120].astype(jnp.int8)
recv_h = jnp.sum(recv_states.astype(jnp.int64) * hash_vec, axis=1)
sort_h = jnp.sort(recv_h)
sort_idx = jnp.argsort(recv_h)
...
recv_v = forward_v(v_params, recv_states)
top_v_keep, keep_idx = _topk_smallest(recv_v_masked, B_local)
```

At 32M with `ALPHA_QSHORT=2`, receive candidate count is:

```text
aB_local = K_PER_PEER * 8 = 8,388,608
```

This is large but likely manageable after the child tensor and tree carry are
removed.

Do not rewrite this first. Keep it stable for the first 32M attempt.

Fallbacks if receive side still OOMs:

1. Set `ALPHA_QSHORT = 1` for a memory smoke.
2. Pack V scores in the bucket and skip the receive-side V rerun.
3. Stream receive-side dedup/top-k in hash buckets.

### 8. Pack V Scores Into Buckets Later

Current V-only body reruns V on received states because the packed bucket does
not carry the score:

```python
recv_v = forward_v(v_params, recv_states)
```

There are spare bytes in the 128-byte record:

```text
0..119: state
120..123: parent_local
124: move
125..126: optional bfloat16 V score
127: pad
```

This can remove the receive-side V rerun:

```python
score_u16 = lax.bitcast_convert_type(score_bf16, jnp.uint16)
bucket = bucket.at[:, 125].set((score_u16 & 0xFF).astype(jnp.uint8))
bucket = bucket.at[:, 126].set((score_u16 >> 8).astype(jnp.uint8))
```

Keep this as phase 2, not phase 1.

Rationale:

- Saves compute.
- May reduce activation peak on receive side.

Risk:

- bf16 score ordering can differ from f32 rerank.
- For the first exact 32M implementation, preserve current rerank behavior.

### 9. Add Early Stop With Host-Side Winner Check

Current wrapper runs all `num_steps` then checks `fs_d` on host. With host-tree
storage, the wrapper is already syncing each step, so stop as soon as any rank
finds V0:

```python
fs_per_rank = np.asarray(fs_d)
if np.any(fs_per_rank >= 0):
    break
```

Then choose the global winner exactly as today:

```python
fs_signed = np.where(fs_per_rank >= 0, fs_per_rank, INT_MAX)
winner_rank = int(np.argmin(fs_signed))
winner_step = int(fs_per_rank[winner_rank])
winner_pos = int(fpl_per_rank[winner_rank])
```

Rationale:

- Saves runtime and host tree writes.
- Especially important at 32M where each step is expensive.

Risk:

- None if the host winner logic remains unchanged.

### 10. Keep `ALPHA_QSHORT=2` For First Quality Run

Although this is a V-only notebook, `ALPHA_QSHORT` controls receive oversampling:

```python
K_PER_PEER = (ALPHA_QSHORT * B_LOCAL) // N_DEVICES
aB_local = K_PER_PEER * world_size
```

For 32M:

```text
B_LOCAL = 4,194,304
K_PER_PEER alpha=2 = 1,048,576
aB_local = 8,388,608
```

Recommendation:

- Use `ALPHA_QSHORT=2` for the first real 32M quality comparison.
- Add `ALPHA_QSHORT=1` as an OOM smoke fallback only.

Rationale:

- Keeps behavior closest to 8M.
- Gives receive-side top-k enough slack after dedup.

Risk:

- all-to-all payload is about 1 GiB per rank per step at 32M.
- If this dominates wall or memory, alpha is the next knob.

## Proposed Implementation Order

### Phase A: Low-Risk Memory Knobs

Files:

- `build_notebook.py`
- `jax_beam_spmd_v_only.py`

Changes:

1. Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` before `import jax`.
2. Set `INTERNAL_BS=16384`.
3. Add `donate_argnums`.
4. Remove full per-child index arrays in the current non-streaming body.

Goal:

- Confirm 16M is reachable.
- De-risk donation before larger refactor.

### Phase B: Host Tree And Packed Backpointers

Files:

- `jax_beam_spmd_v_only.py`

Changes:

1. Remove full tree tensors from device carry.
2. Return one `(world_size, B_local)` packed `uint32` backpointer array per step.
3. Store backpointers in a host `np.memmap`.
4. Walk back through packed host records.
5. Add early stop after each step.

Goal:

- Remove the largest persistent HBM structure.
- Prepare for 32M streaming.

### Phase C: Streaming Child Generation

Files:

- Prefer a new function in `jax_beam_spmd_v_only.py`, for example:
  `beam_solve_v_only_spmd_streaming`.
- Keep the old function temporarily for comparison.

Changes:

1. Add static `parent_chunk` argument, default `65536`.
2. Replace full `neighbors` materialization with `lax.scan` over parent chunks.
3. Carry `(top_scores, top_pack)` through the scan.
4. Feed `top_pack` into the existing `all_to_all`.
5. Keep receive-side dedup and V rerank unchanged.

Goal:

- Fit 32M.
- Preserve old body for regression comparison.

### Phase D: Optional Receive-Side Optimizations

Only do these if 32M still OOMs or is too slow:

1. Pack bf16 V score into bucket and skip receive-side V rerun.
2. Try `ALPHA_QSHORT=1`.
3. Stream receive-side dedup/top-k.

## Validation Ladder

### 1. Static Notebook Validation

Run the existing notebook validator:

```powershell
.venv/Scripts/python.exe megaminx/kaggle_notebooks/tpu_beam_az_v4_v_only_jax_shareable/_validate_nb.py
```

Also rebuild the notebook after edits:

```powershell
.venv/Scripts/python.exe megaminx/kaggle_notebooks/tpu_beam_az_v4_v_only_jax_shareable/build_notebook.py
```

### 2. Small-Beam Equivalence

Run old and streaming bodies at a tiny shape:

```python
B_GLOBAL = 1 * 1024 * 1024
START_PID = 0
END_PID = 2
K_SYM = 1
```

Acceptance:

- `verify=True` for all found paths.
- Path lengths match old body where possible.
- If path differs due to top-k tie order, compare total path length and final
  verification.

### 3. 8M Regression

Run:

```python
B_GLOBAL = 8 * 1024 * 1024
START_PID = 0
END_PID = 1
K_SYM = 1
```

Acceptance:

- No OOM.
- Path verifies.
- First-iteration time is within a reasonable factor.
- Host tree file is deleted after path reconstruction.

### 4. 16M Capacity Smoke

Run:

```python
B_GLOBAL = 16 * 1024 * 1024
START_PID = 800
END_PID = 801
K_SYM = 1
```

Acceptance:

- No OOM.
- Verify true if found.
- Record first-iteration time, steady step time, and host-tree write time.

### 5. 32M Capacity Smoke

Run:

```python
B_GLOBAL = 32 * 1024 * 1024
START_PID = 900
END_PID = 901
K_SYM = 1
PARENT_CHUNK = 65536
INTERNAL_BS = 16384
ALPHA_QSHORT = 2
```

Acceptance:

- Compile succeeds.
- First step completes.
- No runtime OOM through at least 5 steps.
- If found, path verifies.

### 6. 32M Quality Test

Pick 5 hard pids where 8M has known paths. Run 32M K=1 and compare:

- solved count
- path length
- wall time
- first iteration time
- average step time
- host write time
- peak failure mode if any

Only after this should we spend quota on a larger rescue run.

## Expected Memory Shape At 32M

Approximate per-rank persistent or large transient arrays after Phase C:

```text
states:              4M * 120 * 1       ~= 480 MB
stream child states: 64K * 24 * 120     ~= 180 MB
top_pack/send:       8 * 1M * 128       ~= 1.0 GB
top_scores:          8 * 1M * 4         ~= 32 MB
recv_buckets:        8 * 1M * 128       ~= 1.0 GB
recv_states:         8M * 120 * 1       ~= 960 MB
recv_hash/sort idx:  8M * 8 + 8M * 4    ~= 96 MB plus XLA workspace
model params:        small relative to beam buffers
tree carry:          removed from HBM
```

The current impossible array is eliminated:

```text
full neighbors:      4M * 24 * 120      ~= 11.25 GiB
```

This is why streaming is the core 32M unlock.

## Risks And Fallbacks

Risk: streaming exact top-k is too slow.

Fallback:

- Increase `PARENT_CHUNK` to `131072`.
- Profile owner loop and top-k time.
- Consider approximate per-chunk shortlist only after exact 32M is proven too
  slow.

Risk: receive-side sort/top-k OOMs.

Fallback:

- Try `ALPHA_QSHORT=1`.
- Pack V scores and skip receive rerank.
- Stream receive-side by hash bucket.

Risk: host tree writes are too slow or disk-heavy.

Fallback:

- Keep packed tree in host RAM if available.
- Stop immediately on first found step.
- Store only up to current step and delete memmap after walkback.

Risk: donation causes invalid-buffer errors.

Fallback:

- Remove donation from one argument at a time.
- Keep donation for `states_d` at minimum.

Risk: path reconstruction breaks.

Fallback:

- For small beam, store both old tree tensors and new packed tree, then compare
  decoded `(parent_local, parent_rank, move)` records step by step.

## Recommendation

Implement in this order:

1. Low-risk knobs and donation.
2. Packed host tree.
3. Streaming child generation.
4. 32M smoke at one pid.
5. Optional receive-side score packing only if needed.

The 32M goal is reasonable with these changes. The non-negotiable change is
streaming child generation; everything else either creates the HBM headroom
needed for streaming to survive or keeps the wrapper practical enough to run
real rescue pids.
