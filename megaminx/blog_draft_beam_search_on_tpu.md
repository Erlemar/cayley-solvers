# Running beam search on TPU: the bug that almost made us give up, and the climb from 8M to 512M

We've been running beam search for a Kaggle puzzle competition called CayleyPy
Megaminx — the goal is to solve 1001 scrambled megaminx (dodecahedron-shaped Rubik's
cube) instances with as few total moves as possible. Our solver is a learned-V-function
beam search: a neural network estimates "distance to solved" for each candidate
state, and a beam keeps the best `B` states per step over ~120 steps.

This post is about the part of that project where we tried to run a *shared* B=8M
beam across all 8 cores of a Kaggle TPU, then kept pushing the same architecture
until it reached 48M on Kaggle and 256M/512M on a larger v6e TPU. It took five
failed architectures, two frameworks, and a critical diagnostic before we figured
out the first bug was in our own code — not, as we'd been convinced, deep inside
XLA. The later scaling work had a different flavor: not "is the collective
wrong?", but "which one of these innocent-looking arrays just cost us 8 GB?".
a
The lesson, if there's one: **when you see the same failure across two completely
different frameworks built on the same backend, the temptation is to blame the
backend. Sometimes the right move is to look closer at what your code is actually
doing on each rank, before declaring the whole architecture a dead end.**

## What's a Cayley graph, and why would you solve one?

Before the TPU war stories, a few paragraphs on what we're actually computing —
"beam search over a Cayley graph" is doing a lot of quiet work in that first
sentence.

Start with the puzzle. Every legal move on a megaminx — turn one face a click
clockwise or counter-clockwise — is a permutation of the puzzle's pieces.
Compose two moves and you get another permutation; undo one and you get its
inverse; do nothing and you have the identity. That's a *group*: all the
configurations reachable from solved, with move-composition as the operation,
and the 24 face turns as its *generators* — the handful of operations every
other element is built from.

A **Cayley graph** is the picture of that group. Drop one vertex for every
element (every reachable configuration of the puzzle) and draw an edge between
two vertices whenever a single generator takes one to the other. The solved
state is just one distinguished vertex, the identity; a scramble is some other
vertex. *Solving* the puzzle is finding a path through the graph from the
scramble back to the identity, and solving it *well* — the thing the
competition scores — is finding a short one. The shortest possible path is the
scramble's true distance-to-solved; the largest such distance over all scrambles
is the graph's diameter, known to cubers as **"God's number."**

This reframing — puzzle as group, group as graph — is the whole game, not a cute
analogy. And it isn't special to twisty puzzles: by **Cayley's theorem** every
finite group is a group of permutations, so permutation puzzles are a concrete
handle on finite group theory in general.

### Why anyone cares

Cayley graphs are a main bridge between abstract algebra and geometry. Once a
group is a graph you can ask geometric questions of it — how fast the vertex
count grows as you walk outward, whether it's "negatively curved," what it looks
like from far away — and the answers encode deep algebraic facts. That's the
field of *geometric group theory*, and its central figure, Mikhail Gromov, has
spent decades at the very institute this competition is named after, studying
groups by treating their Cayley graphs as metric spaces.

They earn their keep in computer science too. The best-connected Cayley graphs —
*expanders*, first built explicitly by Margulis out of group generators — are the
machinery behind error-correcting codes, pseudorandom generators, and
fault-tolerant networks. And the basic questions are genuinely hard: nobody
proved the 3×3×3 cube's God's number is 20 until 2010, and for the megaminx the
diameter is simply unknown.

### Why it's a brutal search problem

Here's what makes this an engineering problem and not just a math one: these
graphs are far too big to write down. The 3×3×3 cube has about 4.3 × 10^19
states; the megaminx has roughly 10^68, astronomically more. You never store the
graph — you store one state and generate its 24 neighbors on demand. The graph is
defined *implicitly*, by its generators.

That rules out exact search. Breadth-first from solved is fine for a few moves —
on megaminx the shells are 1, 24, 408, 6,208, 90,144, 1.28M, then ~18M states at
depth six — but depth seven is ~250M and won't fit in memory, while a real
scramble is dozens of moves deep. You can't compute the true distance-to-solved,
so you *estimate* it. That estimate is the learned **V function** from the top of
this post: a network trained to predict distance-to-solved, used to steer a beam
that keeps the best `B` states at each step. The entire problem collapses to "how
good is your estimate, and how wide a beam can you afford" — which is precisely
why everything below is about cramming the largest possible beam onto a TPU.

It also makes a clean benchmark: one unambiguous number — total moves over a
fixed set of scrambles — for a question with no tractable exact answer, on a
graph nobody can fully see.

### CayleyPy, IHES, and megaminx

**CayleyPy** is the umbrella for this line of work — a research project and a pair
of 2025 papers (*A Machine Learning Approach That Beats Large Rubik's Cubes*,
arXiv:2502.13266, and *CayleyPy RL*, arXiv:2502.18663) on finding short paths in
Cayley graphs with learned heuristics and beam search instead of the hand-built
solvers cubers traditionally use. It spun off a series of Kaggle competitions,
each a different graph.

The flagship is the **IHES** cube — a "picture cube," a 3×3×3 whose facelets are
all distinct so orientation matters, with 1,003 scrambles to solve as short as
possible. It's named after the **Institut des Hautes Études Scientifiques**, the
French research institute outside Paris (mathematics and theoretical physics, the
rough European counterpart to Princeton's IAS — and, fittingly, Gromov's home).

**Megaminx**, the subject of this post, is the sibling competition: not a cube but
a dodecahedron — twelve pentagonal faces, each turning in fifths of a full
rotation. Our solver encodes a state as a length-120 vector with 24 generators
(twelve faces × two directions), and the competition ships **1,001 scrambles**,
scoring the total move count across all of them.

Two things make it a good target. First, unlike the 3×3×3 there's no
Kociemba-style two-phase algorithm to fall back on — no convenient subgroup
decomposition for a classical solver to exploit — so learned-heuristic search
isn't merely competitive, it's nearly the only game in town. (Tomas Rokicki, one
of the people who pinned down the cube's God's number in 2010, sits mid-leaderboard
here with a classical method; the top spots are all ML beam searches.) Second,
because the score sums over a thousand scrambles, every move shaved off every path
matters — exactly the relentless marginal pressure that pushed us from a 1M-state
beam to the 512M-state monster the rest of this post is about.

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

## Past the Kaggle ceiling: 256M on v6e-8

The 48M ceiling above was a Kaggle v5e-8 ceiling, not a law of beam search.
When we got access to an 8-chip v6e TPU VM, each chip had about twice the
usable HBM: XLA reported a 31.25 GB per-chip budget instead of the ~15 GB
budget we'd been fighting on Kaggle. That made the next question obvious:
could we keep the same SPMD design and simply push the beam another order
of magnitude?

The first serious target was `B_GLOBAL = 268,435,456` — a 256M global beam,
split across 8 chips:

```
B_GLOBAL = 268,435,456
world_size = 8
B_LOCAL  = 33,554,432
```

That broke a quiet assumption in our path tree. The old packed backpointer
layouts were designed around smaller local beams:

```
23/3/5  uint32  -> parent_local up to  8,388,607
24/3/5  uint32  -> parent_local up to 16,777,215
```

At 256M global, each rank owns 33,554,432 slots, so the parent index needs
25 bits. There is no way to fit `25 + 3 + 5 = 33` bits into uint32. The tree
had to become uint64:

```
bits  0..24  parent_local   (25 bits, up to 33,554,431)
bits 25..27  parent_rank    (3 bits, 8 ranks)
bits 28..32  move           (5 bits, 24 moves)
bits 33..63  unused
```

That sounds like a small type change, but it changes the operational shape
of the run. A 90-step, 8-rank, 256M tree is already about 193 GB on disk:

```
90 steps * 8 ranks * 33,554,432 slots/rank * 8 bytes = 193,273,528,320 bytes
```

The upside is that this cost is on host scratch, not HBM. Once the tree is a
per-step `.u64` memmap and the device only carries the live frontier, the TPU
doesn't care that the final path tree is hundreds of gigabytes.

We also added a practical "V-all" path to the qshort kernel. With
`student_alpha >= N_GEN` — in our case `student_alpha = 24` for 24 moves —
the student is no longer shortlisting anything. The kernel should degrade to
"score every child with V" instead of allocating a useless student shortlist
that is the same size as the full expansion. That made it possible to use the
same production runner for both qshort and V-only-equivalent experiments.

The 256M run itself was deliberately conservative:

```bash
gcp_beam_v6e.py \
    --b-global 268435456 \
    --start-pid 991 --end-pid 992 \
    --k-sym 1 \
    --num-steps 90 \
    --student-alpha 24 \
    --receive-alpha 1.25 \
    --alpha-req 1.25 \
    --parent-chunk 524288 \
    --internal-bs 131072 \
    --nbhd-radius 4
```

It was a full 256M shared beam across all 8 chips, but it was not a symmetry
ensemble. `k_sym=1`, `rot_idxs=[0]`, `rot_idx=0`: identity frame only. The
model filename happened to contain `_sym`, because it was trained with
symmetry-aware distillation, but the search itself did not rotate the puzzle.

The result:

```
[step 073/89] device=717.9s copy=1.18s write=2.18s total=721.3s ... min_v=0.009 best_near=74
[solve] pid=991 rot=0(idx0) found=True len=74 verify=True wall=53764s
```

Engineering-wise, this was a big success: the full 256M beam ran to a verified
solution, with a 33M-state local frontier per chip, per-step tree writes, and
host walkback. Competitively, it was not a win. The best public community path
we had for pid 991 was length 69, so length 74 is a valid data point, not a
merge candidate.

That distinction matters. A giant beam that verifies is a systems milestone.
A giant beam that beats the current best path is a search milestone. 256M gave
us the first one.

## Making 512M fit

Once 256M worked, 512M looked temptingly close. The arithmetic is brutal but
simple:

```
B_GLOBAL = 536,870,912
world_size = 8
B_LOCAL  = 67,108,864
```

`B_LOCAL` is exactly `2^26`, so the backpointer needed one more parent bit:

```
bits  0..25  parent_local   (26 bits, up to 67,108,863)
bits 26..28  parent_rank    (3 bits)
bits 29..33  move           (5 bits)
bits 34..63  unused
```

That part was easy. The hard part was all the accidental "only a few gigabytes"
temporaries that were harmless at 48M or 256M and fatal at 512M.

The first 512M compile failed with:

```
Used 33.73G of 31.25G hbm
```

After one round of cuts it still failed:

```
Used 32.59G of 31.25G hbm
```

At this scale, a single casual `(B_LOCAL, 8)` int32 array is about 2 GB. A
single `(B_LOCAL, 120)` uint8 frontier is 8 GB. The difference between "fits"
and "doesn't load" is not an abstract optimization; it is one buffer.

The final 512M fit came from four changes.

### 1. Stop materializing request positions as `(B, 8)`

The qshort receive path used to compute per-source request positions with a
one-hot matrix:

```python
onehot = (sel_src[:, None] == owner_arange[None, :]).astype(jnp.int32)
pos = jnp.cumsum(onehot, axis=0) - 1
```

At normal sizes this is just convenient. At 512M, `sel_src` is B-sized and
`owner_arange` has length 8, so the temporary is `(67M, 8) int32`: about
2 GB, plus more cumsum workspace.

The fix was to loop over sources and keep only one B-length mask live at a
time:

```python
for src in range(world_size):
    mask = sel_src == src
    pos_for_src = jnp.cumsum(mask.astype(jnp.int32)) - 1
    # scatter this source's requests, then move on
```

It is less elegant, but it removes the giant rank-axis matrix entirely.

### 2. Tile response materialization

The request/response phase routes parent states back to the rank that needs
to materialize the selected child. The old code stacked the full response
payload before the return all-to-all:

```
(world_size * REQ_CAP, state_size)
```

At 512M, with 120-byte states, that is another multi-gigabyte transient, and
XLA wants workspace around it. We changed response building to operate in
tiles (`REQ_TILE`) and concatenate only the bounded tile outputs the collective
expects.

This was a fit patch, not a speed patch. We later tried `REQ_TILE = 4 *
MAT_CHUNK` to reduce the number of response tiles. It compiled a little faster
but the first real step was essentially unchanged. The bottleneck wasn't the
number of response all-to-alls; it was the broader selection/materialization
structure and HBM traffic.

### 3. Scatter into the donated frontier, not a fresh zero frontier

The old survivor-materialization code started from a clean zero buffer:

```python
new_states = jnp.zeros((B_LOCAL + 1, state_size), dtype=jnp.uint8)
new_states = new_states.at[dst_pos].set(materialized_children)
```

At 512M, that zero buffer is another ~8 GB allocation. But the previous
frontier is already the right shape and is donated into the step. We don't
care about its old contents after the step. So the new version uses the
donated frontier as the scatter base and drops invalid writes out-of-bounds:

```python
new_states = states.at[dst_pos].set(
    materialized_children,
    mode="drop",
)
```

Same result for valid rows, no extra 8 GB base allocation.

### 4. Build the seed frontier on host, not device

This was the least glamorous bug and the one that finally made the executable
load. After the compile OOMs were gone, we hit a different failure:

```
Attempting to reserve 17.73G ... only 7.73G free
```

The step had compiled, but loading the runtime executable needed a big device
reserve. Why was only 7.73 GB free? Because the seed frontier had been built as
a large JAX array first and then converted into a sharded array. That left an
extra ~8 GB device buffer alive at exactly the wrong time.

The fix was to build the padded seed in NumPy on the host, and then use
`jax.make_array_from_callback` so each device receives only its shard. This
is the same lesson we learned earlier with `jnp.broadcast_to((8, B_LOCAL, 120))`
accidentally materializing the full beam on one device: for giant frontiers,
host-side sharding is not optional plumbing. It is part of the memory model.

## 512M ran. That doesn't mean it was worth running to completion.

After those patches, 512M finally ran:

```bash
gcp_beam_v6e.py \
    --b-global 536870912 \
    --start-pid 991 --end-pid 992 \
    --k-sym 1 \
    --num-steps 3 \
    --student-alpha 4 \
    --receive-alpha 1.03125 \
    --alpha-req 1.03125 \
    --parent-chunk 262144 \
    --internal-bs 131072 \
    --nbhd-radius 4
```

The smoke numbers:

```
compile: 143.9s
step 1: device=2515.6s copy=2.74s write=4.11s total=2522.5s  min_v=28.625
step 2: device=2509.3s copy=2.43s write=4.12s total=2515.8s  min_v=27.875
```

Call it 42 minutes per step. A full 90-step pid would be roughly 62 hours,
identity-only, before any symmetry or inverse-axis work. The tree would also
be enormous:

```
90 steps * 8 ranks * 67,108,864 slots/rank * 8 bytes = 386,547,056,640 bytes
```

That is ~386 GB decimal before logs, partial outputs, and filesystem slack.
Our 400 GB scratch disk was technically close, but operationally too tight;
any serious full 512M run wants 600-800 GB.

More importantly, the early search trace looked bad. The 512M smoke used
`student_alpha=4`, meaning the Q student shortlisted 4 children per parent.
That was the only way to keep the candidate flow small enough for the first
fit attempt. But quality was poor: after two steps, `min_v` was still 27.875.
In the 256M alpha24 run, `min_v` was already around 12.75 by step 2.

So the conclusion was not "launch 512M for three days". It was:

1. The 512M SPMD architecture now fits on 8 v6e chips.
2. The tested alpha4 qshort setting is too miscalibrated for this hard-pid
   endgame.
3. The next useful experiments are short 512M smokes at `student_alpha=8` or
   `12`, and possibly an alpha24 V-all step benchmark, before committing to a
   full run.

That's the uncomfortable but useful distinction: **capacity work can succeed
while the search experiment it enables is still a bad bet.** We made 512M
possible; we did not yet make 512M the right thing to spend two TPU-days on.

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

**Fitting a beam is not the same as making it a good beam.** The 512M smoke was
the cleanest example: after the host-seed and transient-buffer fixes, the
architecture ran. But alpha4 qshort had poor early search quality, and the full
run would have cost roughly 62 hours plus a larger scratch disk. That's still a
success — it tells us what experiment is now possible — but the next scientific
question is alpha8/12 or V-all quality, not "press go on the biggest number".

**For giant frontiers, host-side sharding is part of the algorithm.** Building a
`(8, B_LOCAL, 120)` seed or padding buffer as a JAX array can silently place an
extra multi-gigabyte object on device before sharding. At 8M that is annoying.
At 512M it prevents the compiled executable from loading. `make_array_from_callback`
and host NumPy construction stopped being implementation details and became
the difference between "fits" and "does not start".

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
