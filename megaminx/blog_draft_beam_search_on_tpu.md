# Running beam search on TPU: from PyTorch to JAX, and the bug that almost made us give up

We've been running beam search for a Kaggle puzzle competition called CayleyPy
Megaminx — the goal is to solve 1001 scrambled megaminx (dodecahedron-shaped Rubik's
cube) instances with as few total moves as possible. Our solver is a learned-V-function
beam search: a neural network estimates "distance to solved" for each candidate
state, and a beam keeps the best `B` states per step over ~120 steps.

This post is about the part of that project where we tried to run a *shared* B=8M
beam across all 8 cores of a Kaggle TPU. It took five failed architectures, two
frameworks, and a critical diagnostic before we figured out the bug was in our
own code — not, as we'd been convinced, deep inside XLA.

The lesson, if there's one: **when you see the same failure across two completely
different frameworks built on the same backend, the temptation is to blame the
backend. Sometimes the right move is to look closer at what your code is actually
doing on each rank, before declaring the whole architecture a dead end.**

## The setup

The beam search is the standard "qshort" pattern:

```
for j in 1..NUM_STEPS:
    # 1. expand: each of the B parent states has N_GEN=24 children
    children = states[:, all_moves].reshape(B * N_GEN, state_size)

    # 2. cheap student model ranks them
    student_q = student(states)                  # (B, N_GEN) bf16
    student_score = student_q.reshape(-1)

    # 3. take top alpha*B by student
    _, shortlist = topk_smallest(student_score, alpha * B)

    # 4. expensive teacher reranks the shortlist
    teacher_v = teacher(children[shortlist])      # (alpha*B,)

    # 5. take top B by teacher
    _, chosen = topk_smallest(teacher_v, B)
    states = children[shortlist][chosen]
    # plus tree storage for path reconstruction at the end
```

Two small models — a "V" teacher (distance estimate) and a "Q" student (top-k
action shortlister) — work together: student is cheap and picks a 2B shortlist,
teacher is expensive and refines to B. With `B = 1,048,576` and 120 steps, this
beats the leaderboard's average path length by enough to matter.

## What worked: data-parallel on TPU v3-8 (PyTorch/XLA)

Our first working version ran on Kaggle TPU v3-8 (4 chips × 2 cores) via
`torch_xla.distributed.xla_multiprocessing.spawn`. 8 worker ranks, each running an
independent B=1M beam on a different `(pid, rotation)` pair:

```python
def _mp_fn(index, pid_rot_pairs):
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    rank = xm.get_ordinal()

    teacher = ResMLPDistance(hidden_dims=(2048, 512), output_dim=1)
    teacher.load_state_dict(torch.load("m_curr_v3.pt"))
    teacher = teacher.to(torch.bfloat16).to(device).eval()
    # ... same for student ...

    my_pairs = [pr for i, pr in enumerate(pid_rot_pairs)
                if i % world_size == rank]
    for pid, rot_i in my_pairs:
        result = beam_solve_xla_qshort(
            init_state[pid], teacher, student,
            device=device, B=131072, num_steps=120, alpha=2,
        )
        # write per-pair result to /kaggle/working/rank_<r>_partial.json
```

This works well. We pushed a "shareable" Kaggle kernel where collaborators fork
the notebook, pick a pid chunk, and run on their own quota — the project
covered the full 1001 puzzles across a few volunteers and merged the per-pid
minimum across all submissions.

Per-pair wall on v3-8 was about 150 seconds steady-state after a ~21-minute
first-pair compile. Memory: about 4 GB per rank, comfortable on the shared
16 GB HBM (two ranks per chip on v3-8).

## What we wanted next: a single B=8M beam across all 8 cores

For the hardest 200 puzzles, a B=1M beam wasn't finding optimal paths. We
hypothesized that the bottleneck wasn't the V model — it was beam diversity:
at B=1M, the beam was getting stuck on locally-attractive states and missing
the globally optimal trajectory. A B=8M beam (one beam, all 8 cores cooperating,
hash-partitioned across ranks) might find shorter paths on the long tail.

This is a textbook SPMD use case: 8 ranks share a single logical work pool,
coordinate via collective ops. We'd done data-parallel just fine; SPMD shared-beam
seemed natural.

The first design: each rank holds `B_LOCAL = B / 8 = 1M` states. At each step:

1. Each rank generates its `B_LOCAL × N_GEN = 24M` children.
2. Hash each child: `owner_rank = hash(child) % world_size`.
3. Per `(sender, owner)` pair, take the top `K_PER_PEER` children by V.
4. Cross-rank routing: `lax.all_to_all` (or `xm.all_to_all` in torch_xla)
   sends bucket S from every sender to rank S.
5. Each rank receives `aB_LOCAL ≈ 2 × B_LOCAL` candidates, picks top `B_LOCAL`
   by teacher V — those become the new beam.
6. Detect V0 (the solved state) on each rank; cross-rank-reduce to a single
   global winner.

Per-step memory is bounded by `B_LOCAL`, not `B_GLOBAL`. The cross-rank
communication is one packed all-to-all per step.

## The first time we tried it: four failed architectures in torch_xla

We built four versions, each with a different cross-rank communication pattern:

| ver | architecture | result |
|---|---|---|
| v1 | `xm.all_gather` of 5 separate tensors | OOM during XLA program load |
| v2 | multi-call `xm.all_gather` per tensor field | Ran, but only pid 1 (a real 1-move solve) verified |
| v3 | v2 + tensor-equality V0 detection | Same false positives |
| v4 | manual padded `xm.all_reduce(SUM)` instead of all_gather | Same false positives |
| v5 | hash-partition + single packed `xm.all_to_all` | Same false positives, slightly different paths |

The signature of the bug, identical across all four:

```
pid | iteration | found_step reported | path verifies
----+-----------+---------------------+--------------
 0  |     0     | 55-57               | False
 1  |     1     |  0                  | True (real 1-move solve)
 2  |     2     |  1                  | False
 3  |     3     |  2                  | False
 4  |     4     |  3                  | False
```

For pids 1+: `found_step = call_index - 1`. The algorithm reports finding the
solved state, but applying the recovered move sequence to the initial state
doesn't reproduce V0.

We searched torch_xla's GitHub issues, found two relevant bugs
(pytorch/xla#3824 about `all_gather` rank-order scrambling, pytorch/xla#3510
about all_gather memory regressions), and convinced ourselves the bug was deep
in torch_xla's collective dispatcher. After ~6 hours of debugging and two
sessions, we wrote up the failure as "SPMD shared-beam is a dead end on
Kaggle TPU v3-8 / torch_xla 2.8" and moved on.

## "But JAX will be different"

A few weeks later, we got curious. JAX is the framework that Google's own TPU
team writes against; if any framework's SPMD on TPU is well-tested, it's JAX.
The bug class we'd hit in torch_xla — cross-rank ordering scrambling — was
specifically called out as a torch_xla bug, not an XLA backend bug.

So we ported the V/Q/policy models to pure JAX (it's a ~200-line port from the
PyTorch `nn.Module` to a function over a dict-of-arrays; we verified the
output matched PyTorch within 1e-5 in float32), and rewrote the SPMD beam
body using JAX's canonical pattern:

```python
@partial(jax.jit, donate_argnums=...)
def step_fn(states, tree_..., found_..., j_arr):
    return shard_map(
        beam_step_local,
        mesh=mesh,
        in_specs=(P("cores"), ...),
        out_specs=(P("cores"), ...),
    )(states, tree_..., found_..., j_arr)
```

And inside `beam_step_local`, the per-step routing:

```python
def beam_step_local(states, tree_..., found_step, found_pos, j):
    # ... generate B_LOCAL * 24 children, hash, owner-partition ...
    send_buckets = pack_per_owner_topk(children, scores, K_per_peer)

    # Single packed all_to_all — no multi-tensor ordering risk
    recv_buckets = jax.lax.all_to_all(
        send_buckets, axis_name="cores",
        split_axis=0, concat_axis=0, tiled=True,
    )

    # Dedup, teacher rerank, topk(B_LOCAL)
    chosen_states, chosen_h, chosen_parent_local, chosen_move = ...

    # V0 detection
    eq_v0 = (chosen_h == V0_hash) & chosen_is_real
    is_first_hit = (found_step == -1) & jnp.any(eq_v0)
    new_found_step = jnp.where(is_first_hit, j, found_step)
    # ...

    # Cross-rank reduce: lowest-step finder wins, tie-break smallest rank
    my_step_signed = jnp.where(new_found_step >= 0, new_found_step, BIG_INT)
    global_min_step = jax.lax.pmin(my_step_signed, axis_name="cores")
    # ... pmin/psum to broadcast the global winner's (rank, pos) ...
    return chosen_states, tree_..., global_found_step, global_pos, global_rank
```

We were genuinely excited. JAX traces purely; closure-captured arrays are
constants; there's no autograd state to leak across pid calls; `jax.lax.all_to_all`
is the same primitive Google uses to train trillion-parameter models.
Whatever weird buffer-reuse pattern torch_xla had hit, JAX would side-step.

We pushed v1 to Kaggle TPU v5e-8 (the newer TPU, 8 chips × 1 core each, 16 GB
HBM per chip).

It OOM'd at compile time. Fixed via `lax.scan`-based chunked forward — the
student + policy embedding lookups were fusing into a 4 GB intermediate per rank.
v2 ran end to end.

Same bug. Identical fingerprint. `found_step = pid_index - 1` for sequential
calls; pid 0 anomaly at iter 55-57.

We tried four more fixes:
- v3: removed `donate_argnums` (eliminate any buffer aliasing across calls).
- v4: switched from tensor-equality V0 detection to scalar `chosen_h == V0_hash`.
- v5: added a non-padding filter so the V0 check can mathematically never fire
  on a padded `[0]*120` row.

After v5, the output was byte-identical to v3 and v4. Every defensive filter
we added did nothing. The same `found_step = call_index - 1`. The same
`detected_state = [0]*120` (the padded all-zero state).

We wrote a memory file declaring it an XLA-level dead end. Two frameworks,
seven architectures, same fingerprint — what else could it be? We started
drafting a minimal repro to file upstream with the JAX team.

## The diagnostic that changed everything

Before filing the repro, we got a careful read from a colleague who pointed
out two things we'd missed:

1. The `min_v_log` we'd been printing was rank-0-only. The reasoning "min_v at
   `found_step` is 1.84, not V(V0) ≈ 0.957, therefore V0 is not in chosen
   anywhere" assumed rank 0 was the rank with V0. But V0's hash determines
   its owner deterministically: `V0_hash % 8 = 1`. **V0 should land on rank 1,
   not rank 0.** Rank 0's `min_v` at iter 57 being 1.84 doesn't prove V0 is
   absent globally — it just proves rank 0 doesn't have V0.

2. The `detected_state = [0]*120` we'd been seeing was a consequence of
   reading `vstate_d[fpr_h]` on host, where `fpr_h` is whatever
   `found_pos_rank` got broadcast to after the in-body `pmin/psum` cross-rank
   reduce. If the reduce was producing wrong values, we'd be reading the wrong
   rank's `verify_state` — which on the non-finder ranks was the initial
   zeros.

In other words: we hadn't been debugging the algorithm. We'd been debugging
our incomplete view of it.

The proposed test: skip the in-body cross-rank reduce entirely. Let each rank
keep its own local `(found_step, found_pos, verify_state)`. After the loop,
extract all 8 ranks' values to host, do the cross-rank winner selection in
plain numpy, and walk back the tree from the winner's perspective.

We made the change and pushed once. Per-pid output (for the 8-pid smoke):

```
pid 0: V0_hash=3,952,374,295,417,501,729  expected_owner_rank=1
  rank 0: fstep=-1  (never locally detected V0)
  rank 1: fstep=57  fpos=5   is_V0=True  head=[0,1,2,3,4,5,6,7,8,9,10,11]
  rank 2: fstep=-1
  ... (ranks 3..7 all -1)
  winner = rank 1 → walkback → 58-move path → verify True

pid 2: same structure — V0 on rank 1 at fstep=1 → walkback → 2-move path → verify True
```

**The algorithm had been correct all along.** V0 was being detected by exactly
the rank that owned its hash. The path-walkback from that rank's tree was
correct. The bug was that the in-body `pmin/psum` was not correctly
propagating the global winner's rank to all ranks' carry slots, so the host
was extracting the wrong rank's view.

By moving the cross-rank reduce out of the JIT'd body and onto the host —
a few lines of `np.argmin` — every pid suddenly verified.

## The fix

Inside the per-step body, the change was a deletion:

```python
# OLD: cross-rank reduce inside the body
my_step_signed = jnp.where(found_step >= 0, found_step, BIG_INT)
global_min_step = jax.lax.pmin(my_step_signed, axis_name="cores")
# ... 15 more lines of pmin/psum to broadcast the winner ...
return chosen_states, tree_..., global_found_step, global_pos, global_rank

# NEW: each rank keeps its own local first-hit; no in-body reduce
return chosen_states, tree_..., found_step, found_pos, found_pos_rank
```

And on host, after the loop:

```python
fs_per_rank = np.asarray(fs_d)        # shape (8,)
fpl_per_rank = np.asarray(fpl_d)
fpr_per_rank = np.asarray(fpr_d)

INT_MAX = 2**30
fs_signed = np.where(fs_per_rank >= 0, fs_per_rank, INT_MAX)
global_min_step = fs_signed.min()
if global_min_step >= INT_MAX:
    return {"found": False}

winner_rank = int(np.argmin(fs_signed))
winner_step = int(fs_per_rank[winner_rank])
winner_pos = int(fpl_per_rank[winner_rank])

# walk back through the winner rank's tree slice
path = walkback(tree_data[winner_rank], winner_step, winner_pos)
```

Five lines on host replaced ~20 lines of in-body `pmin/psum` and worked
correctly on the first push.

## Results

The fix unlocked the actual feature we'd been chasing for weeks. We now have
B=8M shared-beam working on Kaggle TPU v5e-8:

| variant | per-pid wall (B=8M, v5e-8) | scope per kernel |
|---|---|---|
| qshort (V + Q + policy) | ~400s | ~60 pids |
| V-only (single V model) | ~2400s | ~10 pids |

That's ~7 hours for a 60-pid hard-tail rescue at the original target B=8M.
Compare to the data-parallel B=1M setup: 8 different pids in parallel at
B=1M each, ~80 seconds per pid amortized.

We built four production notebooks on top of the shared `jax_beam_spmd.py`
body — a research kernel with the v1-v9 history visible, a clean private
production tool, and two collaborator-facing shareables (one for the m_curr_v3
qshort stack, one for AZ V-only). The single-source-of-truth pattern means
fixes propagate to all four at build time.

## Closing the loop: porting the fix back to torch_xla

If the diagnosis was right — "the bug is in the in-body cross-rank reduce
pattern, common to both frontends" — then the same fix should work in
torch_xla too. So we went back and tried.

The torch_xla SPMD body has a `_convergence_reduce` helper structurally
identical to the JAX `pmin/psum` block:

```python
def _convergence_reduce(found_step, found_pos_local, found_pos_rank, ...):
    import torch_xla.core.xla_model as xm
    BIG_INT = torch.tensor(2_000_000_000, dtype=torch.int32, device=device)
    found_step_signed = torch.where(found_step >= 0, found_step, BIG_INT)
    global_min_step = xm.all_reduce(xm.REDUCE_MIN, found_step_signed)
    i_am_finder = (found_step_signed == global_min_step) & (found_step >= 0)
    tag = torch.where(i_am_finder, rank_tensor, BIG_INT)
    global_min_finder_rank = xm.all_reduce(xm.REDUCE_MIN, tag)
    # ... more all_reduces to broadcast winner_pos ...
    return found_step, found_pos_local, found_pos_rank
```

Same pattern, same primitives (`xm.all_reduce(REDUCE_MIN/SUM)`), same role
inside the per-step body. The fix translates directly: delete the
`_convergence_reduce` call from the body, keep per-rank state local, and
aggregate on host.

The trickier part is that torch_xla's `xmp.spawn` launches 8 separate Python
processes (4 chips × 2 threads on TPU v3-8). They can't share Python objects
in memory, so "aggregate on host" has to go through disk. The natural pattern,
which the project already uses for data-parallel aggregation, is per-rank
`.npz` dumps + `xm.rendezvous` to synchronize + a designated rank reading and
aggregating:

```python
# Inside each xmp.spawn worker, at the end of beam_solve_spmd (no in-body reduce).
np.savez(f"/tmp/rank_{rank}_spmd_state.npz",
         fs=my_found_step.cpu().numpy(),
         fpl=my_found_pos_local.cpu().numpy(),
         vstate=verify_state.cpu().numpy(),
         tree_move=tree_move.cpu().numpy(),
         tree_parent_local=tree_parent_local.cpu().numpy(),
         tree_parent_rank=tree_parent_rank.cpu().numpy())
xm.rendezvous("spmd_state_dumped")   # all ranks must finish writing first

if rank == 0:
    states = [np.load(f"/tmp/rank_{r}_spmd_state.npz") for r in range(world_size)]
    fs_per_rank = np.array([int(s["fs"]) for s in states])
    fs_signed = np.where(fs_per_rank >= 0, fs_per_rank, 2**30)
    winner = int(np.argmin(fs_signed))
    # walk back through states[winner]["tree_move"] / tree_parent_*
```

We pushed v6 of the original torch_xla SPMD notebook with this change on
Kaggle TPU v3-8 and got `verify_ok=True` for every pid. The notebook we'd
written off as a "dead end" four weeks earlier was four lines from working
the whole time.

Two practical asymmetries vs the JAX version:

- **Synchronization**. JAX runs single-process; host-side aggregation is
  just `np.asarray(device_array)` after the loop. torch_xla runs
  multi-process under `xmp.spawn`; per-rank state has to cross the process
  boundary, which means disk-via-`/tmp` (or a multiprocessing queue) plus
  an explicit `xm.rendezvous`.
- **Tree extraction cost**. The full per-rank tree is `(NUM_STEPS, B_LOCAL)`
  of int32 + int8 + int8 = ~360 MB per rank at B=1M. Dumping 8 of those to
  `/tmp` and reading them back on rank 0 adds ~2-3 seconds per pid. In the
  JAX version the same data is already host-visible without serialization.

For our use case neither asymmetry matters in practice — the per-pid wall
is dominated by the 120-step beam search itself, not by the post-loop
aggregation. The torch_xla port is now a viable fallback for collaborators
who'd rather stick with the older v3-8 TPU and the PyTorch/XLA stack they
already know.

## Going wider: from B=8M to B=48M

With the shared-beam architecture working, we wanted more. For the hardest
~100 puzzles, the B=8M beam still wasn't beating the public leaderboard.
We'd been thinking about beam quality as a function of B alone, but the
math hinted at something else — at B=8M with α=2 receive-side oversampling,
each rank only sees 2M candidates per step. Quadrupling to B=32M, or even
quadrupling again to B=64M, might unlock paths the smaller beam couldn't see.

The catch: B=8M already used ~5 GB of HBM per rank. Doubling crossed the
v5e-8 HBM ceiling on the very first attempt at B=16M. Going to B=32M and
beyond needed structural changes.

We landed on a 4-phase optimization stack plus two extra patches. Each phase
addresses a different persistent or transient buffer. Below is what each
phase actually freed, the one that turned out to do nothing, and the wall
of memory we eventually hit.

### Phase A: donation and dropping precomputed arrays

The lowest-hanging memory was a precomputed `parent_local_per_child` array
of shape `(B_LOCAL * 24,) int32` — used inside the per-step body to look up
which parent each child came from. At B_LOCAL=4M that's 384 MB per rank,
sitting in HBM all run. Replace it with two integer ops at the topk output:

```python
# OLD: precomputed once at body-build time, gather inside topk
sel_parent_local = parent_local_per_child[top_idx_S]
sel_move = move_per_child[top_idx_S]

# NEW: derive on demand from the topk index
sel_parent_local = (top_idx_S // n_gen).astype(jnp.int32)
sel_move = (top_idx_S % n_gen).astype(jnp.int8)
```

Phase A also re-enabled `donate_argnums` on the step_fn jit. We'd disabled
donation during the SPMD bug debug because we suspected buffer aliasing was
masking the bug. With the bug correctly localized to the in-body cross-rank
reduce, aliasing was safe again. Donating the 6 carry tensors saves another
~1.5 GB at B_LOCAL=2M.

### Phase B: pack the per-step backpointer, spill the tree to host

The biggest single device buffer was the path-reconstruction tree, held
in 3 separate arrays for parent_local (int32), parent_rank (int8), and
move (int8). At B_LOCAL=4M and 120 steps that's 2.88 GB per rank, sitting
in HBM the entire run just to support the final walkback.

We packed all three fields into one uint32 per beam slot:

```
bits  0..22  parent_local   (supports B_LOCAL up to 8,388,608 = 64M global)
bits 23..25  parent_rank    (0..7)
bits 26..30  move           (0..23, 5 bits)
bit     31   unused
```

Then we wrote the packed array to a host `np.memmap` each step. The
memmap lives on `/kaggle/working` (20 GB scratch on Kaggle TPU VMs), so
the tree is gone from HBM entirely — about 1.5 GB freed at B_LOCAL=2M,
2.9 GB at B_LOCAL=4M. Walkback unpacks the uint32 records and follows
the `(parent_rank, parent_local)` chain from the found step back to step 0.

We also added early stop: once any rank locally finds V0, the host loop
breaks. Cuts ~half the steps on found pids that solve before the 120-step
budget.

After Phase A+B, B=16M fit comfortably on v5e-8. A clean 2× scaling unlock.

### Phase C: stream child generation

B=32M didn't fit. At B_LOCAL=4M, the materialized children tensor —
`states[:, all_moves].reshape(B_LOCAL * 24, 120)` — is **11.5 GB** on its
own. All the persistent-buffer cuts from Phase A+B were dwarfed by this
single transient.

The fix is conceptually simple: process the parents in chunks. For each
chunk of (say) 65,536 parents, generate that chunk's 1.5M children
(180 MB), run V on them, hash + owner-route to per-owner buckets. The
per-owner top-K is the carry of a `lax.scan` over chunks — at the end
we have `(world_size, K_per_peer, PACK_SIZE)` send_buckets, exactly the
shape the all_to_all expects.

```python
def chunk_body(carry, chunk_i):
    top_scores, top_pack = carry
    parent_start = chunk_i * PARENT_CHUNK
    states_chunk = lax.dynamic_slice(states, (parent_start, 0),
                                     (PARENT_CHUNK, state_size))
    children = states_chunk[:, all_moves].reshape(-1, state_size)
    child_v = forward_v(v_params, children)
    h_owner = jnp.sum(children.astype(jnp.uint32) * owner_hash_vec, axis=1)
    owner = (h_owner & jnp.uint32(world_size - 1)).astype(jnp.int32)

    new_top_scores, new_top_pack = top_scores, top_pack
    for dest in range(world_size):
        masked_scores = jnp.where(owner == dest, child_v, BIG_F32)
        merged_scores = jnp.concatenate(
            [new_top_scores[dest], masked_scores], axis=0)
        top_v_new, keep = topk_smallest(merged_scores, K_per_peer)
        # ... assemble merged_pack from old top_pack or new chunk children ...
        new_top_scores = new_top_scores.at[dest].set(top_v_new)
        new_top_pack = new_top_pack.at[dest].set(merged_pack)
    return (new_top_scores, new_top_pack), None

(_, final_top_pack), _ = lax.scan(
    chunk_body, (init_top_scores, init_top_pack),
    jnp.arange(n_chunks, dtype=jnp.int32),
)
send_buckets = final_top_pack
```

Per chunk, per owner, we do `topk(K_per_peer + chunk_n, K_per_peer)` on
the concatenation of "running top-K so far" with "this chunk's children
for this owner". After all chunks complete, the running top-K is the
global top-K — same final candidates as the non-streaming body, just
computed incrementally.

The first 32M push with this body errored at trace time:

```
TypeError: scan body function carry input and carry output must have
equal types, but they differ:
  * carry[0] has type float32[8,1M] but corresponding output has type
    float32[8,1M]{V:cores}, so the varying manual axes do not match
```

JAX 0.10's `shard_map` manual mode tracks per-axis "varying" / "replicated"
type information. `jnp.full(...)` is replicated by default; the body returns
"varying" along the sharded `cores` axis. Carry types must match. The fix
is right in the error message — wrap each initial carry with
`jax.lax.pcast(..., to='varying')`:

```python
init_top_scores = jax.lax.pcast(
    jnp.full((world_size, K_per_peer), BIG_F32, dtype=jnp.float32),
    ("cores",), to="varying",
)
init_top_pack = jax.lax.pcast(
    jnp.zeros((world_size, K_per_peer, 128), dtype=jnp.uint8),
    ("cores",), to="varying",
)
```

After this one-line fix per init, the 32M streaming body verified pid 0
at 55 moves — identical path length to the (smaller) 16M run. The bigger
beam paid its compute cost — 57s per step at 32M vs 23s at 16M — but the
search was correct.

### Phase D: pack the V score, skip the receive-side rerun

The receive side currently re-runs V on the candidates it gets, because
the all_to_all only carries states (and parent metadata), not scores.
There were 3 spare bytes in our 128-byte packed record. We packed the
bf16 V score into bytes 125-126 on the send side, unpacked it on receive,
and dropped the second V forward.

Theory: receive-side V is the second-most-expensive op on the rank after
the chunked send-side V. Skipping it should save ~8% wall.

Reality: 0.3% wall savings at 32M. The before-and-after measurement was
3453s vs 3443s — a rounding error.

What happened? Per-step at 32M is dominated by *device compute*: the
per-owner top-K merges (8 owners × 64 chunks × `topk(2.5M, 1M)`) plus
the chunked send-side V on 1.5M children per chunk. Receive-side V on
8M total candidates is ~8% of *V flops*, but V flops are only ~half of
total step flops, and most of that 4% gets hidden behind concurrent
top-K work. Phase D shipped because it's quality-neutral and didn't
hurt anything, but the 8% projection was wrong — we'd never measured
what fraction of wall was actually receive-V before planning it.

### The 1h45m silent cancel

Before Phase D landed, we tried jumping straight to B=48M with α=1
(half the receive oversampling, lighter buffers) to see if the bigger
beam helped on hard pids. The push ran for 1h45m without producing a
single line of per-step output. We cancelled it and dug into the
partial log: nothing between

```
mesh: Mesh('cores': 8, axis_types=(Auto,))
```

— the last notebook cell's print — and the timeout. We had no idea
where the time went. Compile? First step? Stuck in some host transfer?
The TPU quota had been consumed by an unmeasured run.

The fix was four lines. Split `jax.jit`'s lazy compile into explicit
`lower().compile()` so we could time each separately, and add per-step
timing:

```python
# Pre-compile explicitly
t = time.time()
lowered = step_fn.lower(states_d, mv_log_d, fs_d, fpl_d, fpr_d, vstate_d, jnp.int32(1))
print(f"[lower] {time.time()-t:.1f}s", flush=True)
t = time.time()
compiled_step = lowered.compile()
print(f"[compile] {time.time()-t:.1f}s", flush=True)

# Per-step timing
for j in range(1, num_steps):
    t = time.time()
    outputs = compiled_step(*args)
    jax.block_until_ready(outputs[0])
    print(f"[step {j}] device={time.time()-t:.1f}s "
          f"copy={t_copy:.2f}s write={t_write:.2f}s", flush=True)
```

The next 48M push immediately self-explained: lower=0.5s, compile=70s,
per-step device=64.4s steady, host copy + write=0.23s. The 1h45m had
been `compile + 78 steps × 78s = ~1h45m`. Nothing was wrong; we just
hadn't been measuring.

The instrumentation also revealed something about Phase D: per-step
is **99.6% device-bound** at 32M+. Host I/O (per-step memmap write +
packed backpointer transfer) is ~0.4% of wall. A "search-only mode"
we'd planned — skip the per-step tree write entirely for failed pids
— would have saved <1% of wall. We dropped that work. Measure before
optimizing would have flipped Phase C/D priorities entirely.

### Alpha matters more than B

The 48M α=1 push that taught us about silent compile also taught us
about search quality. It did fit, and it did produce a verified path
— at **75 moves**. Pid 0 at 32M α=2 had been 55 moves. The bigger
beam was finding worse paths.

The mechanism: at 32M α=2, each rank receives 2× as many candidates as
beam slots (`aB_local = 2 × B_local`), so the receive-side top-K picks
the best of a meaningfully larger pool. At 48M α=1 the receive side
just barely fills the beam with no margin — search quality collapses.

After two patches (one-sort dedup, uint32 owner hash) freed ~300 MB
transient, 48M α=2 fit. Pid 0 came back to 55 moves at 78 min wall —
clean linear scaling vs 32M α=2's 55 moves at 57 min. Quality restored,
modest wall increase, bigger beam available for the hard pids where
32M wasn't optimal.

| version  | B   | α | per-step | path_len pid 0 |
|---|---|---|---|---|
| 32M α=2  | 32M | 2 | 57s      | 55             |
| 48M α=1  | 48M | 1 | 64s      | **75** (worse) |
| 48M α=2  | 48M | 2 | 85s      | 55             |

Rule of thumb: always scale α=2 with B. Dropping α to fit a bigger B in
HBM is a bad trade — quality regresses faster than B's improvement gains.

### The 64M wall

64M α=2 doesn't fit on v5e-8. The math: B_LOCAL=8M, K_per_peer=2M.
Persistent buffers total ~7 GB per rank — `top_pack` carry (2 GB),
`recv_buckets` (2 GB), `states` (960 MB), `recv_states` (1.9 GB). Plus
XLA workspace (~10 GB) at this scale. Total ~17 GB on a 15.2 GB allowed
budget (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` of 16 GB). The cleanest
path to 64M would be to stream the receive side too — process recv_buckets
in chunks instead of materializing all `aB_local` int8 states at once —
but that's an algorithmic refactor we haven't done yet.

**48M α=2 is the practical ceiling** for shared-beam SPMD V-only on
Kaggle TPU v5e-8, with the current algorithm.

### Where we landed

| variant  | per-pid wall (verified pid 0) | path_len | use case                         |
|---|---|---|---|
| 8M α=2   | ~40 min  | 55 | broad coverage, ~30 pids/kernel  |
| 32M α=2  | ~57 min  | 55 | mid-tier rescue, ~7-8 pids/kernel |
| 48M α=2  | ~78 min  | 55 | hard-tail rescue, ~4-6 pids/kernel |

Three shareable Kaggle notebooks now ship — one per tier. Collaborators
pick the tier appropriate to their TPU quota and the pid range they're
covering. The 6× beam scaling (8M → 48M) came from layered optimizations
totaling ~600 lines of net code change in the V-only solver. The biggest
single win was Phase C (streaming children) — without it 32M+ just
doesn't fit. The second-biggest was instrumentation; we couldn't tune
what we couldn't see. The smallest was Phase D, which the back-of-envelope
flop count had suggested would be biggest. We shipped it anyway —
quality-neutral cheap optimizations are always worth keeping when they
don't hurt anything — but the wall savings line on the changelog reads
"0.3%".

## What we learned

**Identical fingerprint across two frameworks ≠ "framework / backend bug".**
Both torch_xla and JAX compile through XLA, so seeing the same `found_step =
pid_index - 1` signature in both was a strong signal something below the
frontend was wrong. It wasn't. The bug was in an orchestration pattern we
happened to write the same way in both: a cross-rank `min/sum` reduce
running inside the per-step JIT'd body, with its output fed back into the
per-rank carry. Once we moved the reduce outside the body — five lines of
host-side numpy in JAX, four lines of `np.savez` + `xm.rendezvous` +
rank-0 aggregation in torch_xla — every pid verified, in both frameworks,
on both TPU generations (v5e-8 with JAX, v3-8 with torch_xla). The
"orchestration pattern" theory was the right model; it just needed a
second framework's confirmation to be sure.

**Diagnostic discipline beats patch-and-retry.** We spent five debug pushes
trying defensive filters on the V0 detection. One push that returned per-rank
state instead of rank-0 state located the bug in under an hour. The marginal
cost of "dump everything per-rank" on the first debug push is tiny; the cost
of patching the wrong layer for five push cycles is significant — especially
on Kaggle, where each TPU kernel push is ~30 minutes and the weekly quota is
20 hours.

**JAX is genuinely fast on Kaggle TPU v5e-8.** The first-pid compile for our
SPMD body is ~55 seconds, down from torch_xla's ~21 minutes on v3-8.
Steady-state per-pid wall at B=1M is ~80 seconds vs torch_xla's ~150 seconds —
about 2× throughput improvement, before any algorithmic changes. The framework
itself is in great shape on TPU; what bit us was the orchestration pattern,
not the framework.

**Memory math beats trial and error for capacity planning.** B=8M fits
comfortably (~5 GB / rank); B=16M OOMs at runtime by ~240 MB. We could have
predicted this with arithmetic before the push — `children_buffer =
B_LOCAL * 24 * 120` bytes alone is 5.76 GB at B_LOCAL=2M — but didn't bother
to compute it until after the OOM. One avoidable push lost.

Wrapping up: if you're running SPMD on TPU, dump per-rank state on your first
debug push, do your cross-rank reduces on host first and move them inline only
once the host version works, and don't trust "it works in pmap" or
"it works on CPU 8-device emulation" as proof that the SPMD body will run
correctly on real TPU hardware. The frontend frameworks have gotten very good;
the orchestration patterns we wrap around them are still where most bugs live.

In our case, "SPMD shared-beam doesn't work on Kaggle TPU" turned out to
mean "an in-body cross-rank reduce doesn't propagate correctly when its
result feeds the next iteration's per-rank carry". Four lines of patience
between "dead end" and "production tool". Worth checking yours.
