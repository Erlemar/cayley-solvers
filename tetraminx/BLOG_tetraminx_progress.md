# Solving Professor Tetraminx: from 29,622 to 28,467

*A working log of what we tried on the CayleyPy Professor Tetraminx competition, in
roughly the order we tried it — including the things that didn't work, which was most
of them.*

---

## The puzzle

Professor Tetraminx is a tetrahedral twisty puzzle. For our purposes it's a permutation
puzzle on **88 facelets**, all distinct — no colour degeneracy, so a state is a
permutation and not an equivalence class. That matters: it puts it in the same family as
the IHES picture cube, and let us reuse a solver stack essentially unchanged.

| Fact | Value |
|---|---|
| State | 88 facelets, all distinct |
| Generators | 24 = axes {D, F, BL, BR} × layers {2, 3, 4} × 2 directions; **every one of order 3** |
| Group order | 1.538 × 10³² (Schreier–Sims) |
| Non-backtracking branching | ~15.8 |
| BFS levels | 1 / 24 / 408 / 6,592 / 105,136 / 1,659,416 / 26,008,172 (d ≤ 6 cumulative: **27,779,749**) |
| Counting bound | average optimal ≥ ~26.7; realistic true optimum ~28 |
| Test set | 1,000 scrambles; 41 shallow (≤ 26 moves), 959 near-diameter |

The counting bound is the thing to internalise. A uniformly random state sits about **27
moves** from solved, and the whole leaderboard lives within a few percent of that. When
the theoretical floor is ~26.7 moves/pid and the leader is at 28.48, **a single move per
puzzle is an enormous margin.** Everything below is a fight over fractions of a move.

The symmetry group is worth stating too, because it becomes a lever later. The
facelet-automorphism group has order **15,552 = 24 × 648**, where the 648 is a
centralizer that acts trivially on reachable states. The useful quotient is the full
tetrahedral group T_d: 12 rotations plus 12 mirrors. Add **inverse antisymmetry** — solve
`s⁻¹`, then reverse and invert the resulting path — and you get **48 independent search
frames per puzzle**.

Where we started:

```
Rokicki       28,481   near-optimal / coset solver territory
Khoruzhii     28,863   CayleyPy beam + NN
webmaking     29,555   ┐ public "RW-Models2" notebook family
Arbidos       29,591   ┘ ~200K-param MLP, RW regression only, beam 2^21
public kernel 29,622   our starting floor, replay-verified 1000/1000
```

**Final result: 28,467** — 14 moves under Rokicki, 1,155 moves below the public floor
we started from.

---

## Phase 1: the standard stack, and what it bought

The initial approach was the one that works on the cube and megaminx: train a neural
network to estimate distance-to-solved (a **value function**, V), then run beam search
scored by it.

- **Stage A**: random-walk regression. Walk *k* steps from solved, train `V(s) ≈ k`.
- **Stage B**: Bellman refinement — `V(s) ← 1 + min_a V(s·a)`, bootstrapping toward true
  distance.
- **Exact BFS anchors** for states within d ≤ 6, where the truth is known.

This got us to **29,477** by 2026-07-26 — 145 moves below the public floor, from a
combination of tiered beam runs and a small post-processing pass.

It also produced the first result that shaped everything after it.

### Width is the lever, not the model

We ran a clean ablation on 40 pids (the 30 longest floor paths plus 990–999):

| beam | history | solved | total | vs floor |
|---|---|---|---|---|
| 128k | 1 | 31/40 | 1076 | +58 |
| 1M | 1 | 40/40 | 1323 | −1 |
| 1M | 4 | 40/40 | 1316 | −8 |
| **4M** | **1** | **40/40** | **1270** | **−54** |

128k → 1M was −65 on commonly-solved pids; 1M → 4M another −53. Meanwhile *three*
different V models — spanning a 5-point range of deep-scale calibration and two Bellman
recipes — benched **within noise of each other**, and a 25-epoch model matched a
125-epoch one.

The conclusion was uncomfortable but clear: **V refinement had plateaued and search width
had not.** Every subsequent decision followed from that.

### The exact endgame table

The single highest-value structural change was replacing the goal test. Instead of
searching until `state == solved`, the beam stops as soon as it lands **anywhere inside
the BFS d ≤ 6 table** (27.8M states), and the tail is replaced by the table's optimal
descent.

Why it matters: the min-V trace showed the beam descending to ~0.7 and then flatlining
for 30+ steps. It reaches the goal's neighbourhood and *cannot close*, because V is
confidently wrong there — precisely where the beam is narrowest and least reliable. The
table makes the last ~6 moves of every solve **provably optimal** and stops the search
early.

Every emitted path is replayed against the original scramble before being recorded, which
also catches any hash false positive.

---

## Phase 2: the objective change that unlocked width

Width was the lever, but width is compute. At 4M the beam was running ~416 s/pid. To buy
more width we needed a cheaper step, and that came from changing *what the network
predicts*.

### The problem with a value function

To rank a state's children, a V model must **expand every child and score it**: `B × 24`
forward passes per beam step.

Worse, the MSE-on-walk-depth objective has a subtle failure. At large *k* the conditional
variance of true distance is large, so the MSE-optimal prediction shrinks toward the mean
and **local discrimination flattens**. Measured on our own Bellman model, the mean gap
between the good child and the bad child by pivot depth:

| pivot band | 1–4 | 10–14 | 20–24 | **30–40** |
|---|---|---|---|---|
| `V(next) − V(undo)` | 1.85 | 1.78 | 1.22 | **0.44** |

Deep in the search — exactly where the beam needs to discriminate — V can barely tell a
good move from a bad one.

### Sparse-Q: an all-neighbours head

The fix came from Vlad Kuznetsov's `cayleypy-training-core`: an **all-neighbours Q head**
with 24 outputs, one per action, where `Q(s,a)` estimates the distance of the child
`s·a`. One forward on the *parent* scores all 24 children.

Measured end-to-end in our beam step:

| B | V per-child | Q exhaustive | Q + progressive top-k |
|---|---|---|---|
| 16,384 | 155.3 ms | 8.5 ms (18.3×) | **6.4 ms (24.4×)** |
| 65,536 | 578.7 ms | 33.7 ms (17.2×) | **26.4 ms (21.9×)** |

**~22× cheaper per step**, which converts directly into more width.

One caveat that's easy to lose: those numbers are from the **PyTorch** searcher, where
"progressive top-k" also avoids hashing every candidate — score all `(parent, action)`
pairs first, hash only the top ~αB, double until B uniques. **The TPU kernel does not do
that.** It materializes all 24 children, hashes all of them for owner routing, and only
then takes a per-owner top-K. So on TPU the Q head buys the **17.2×** "Q exhaustive"
column, not the 21.9× one — the saving is on the model forward alone.

That looks like it should matter for how wide we can go: children are fully materialized,
so the `B_local × 24 × 88` array is 4.4 GB per rank at 16M. Porting progressive top-k into
the JAX kernel to shrink that seemed obviously worthwhile — **and it turned out to be a
dead end.** See "Two things that closed after the submission" below: the port is correct,
strictly slower, and the memory ceiling it was built to relieve doesn't exist.

The training objective is the clever part. Walk *k* steps from solved, pick a pivot *p*,
and label **exactly two** of the 24 actions:

```
Q(s, undo) = p − 1        Q(s, next) = p + 1
```

Everything else is masked out of the loss. The two labels **always differ by exactly 2
with zero conditional variance**, so the loss *cannot* be reduced by flattening the gap.
The collapse above is forbidden by construction rather than merely discouraged.

We added four things over upstream:

1. **Exact 24-way anchors from BFS.** Every child of a d ≤ 5 state is at d ≤ 6, so all 24
   are in the table — the head gets 24 *true* distances for those states. They're 6.7% of
   rows but **46% of all supervision**, because they're dense where walk labels are 2-of-24.
2. **Symmetry-expanded label coverage** (14 rows/sample). Note it's 14, not 24: the group
   permutes the 4 axes but not the 3 layers, so an action's orbit has 8 members.
3. **Depth-tilted pivot sampling** — upstream's scheme is heavily shallow-weighted and our
   paths are 30–36 moves long.
4. `top1_margin_weight` as a live knob.

**A trap worth recording:** upstream uses `k_max = 70` against a ~29 diameter. That only
survives because its pivot scheme keeps the mass shallow. Past the diameter a walk is near
stationary, so "undo the last move" stops being reliably distance-reducing — the *ranking
label itself* degrades, not just its scale. We run `k_max 40`, tilt 0.5.

---

## Phase 3: the architecture bake-off

With the objective settled we ran a head-to-head: **ResMLP / GFlowNet / PieceTransformer
× no-AZ head / AZ head**, all on the same sparse-Q objective at a matched 1500-epoch
budget, then beam-evaluated on a fixed 15-pid stratified set.

| cell | arch | AZ head | params | 15-pid result |
|---|---|---|---|---|
| 1 | ResMLP | no | 5,008,280 | 468 |
| 2 | ResMLP | yes | 5,008,793 | 461 |
| 3 | PieceTransformer | no | 3,444,504 | 438 (best), then flat |
| 4 | **PieceTransformer** | **yes** | 3,444,761 | **424** (ep1500) |
| 5 | GFlowNet (trajectory balance) | — | ~3.4M | falsified |

The **PieceTransformer** tokenises the state by *physical cubie* rather than by facelet —
50 pieces for tetraminx, plus a CLS token. It was 30+ moves better than the ResMLP at the
same width despite having **fewer** parameters.

### What we killed

**A 6th cell was dropped as provably redundant.** `ResMLPGFlowNet` trained with sparse-Q
is bit-identical to the plain ResMLP: same trunk, and `policy_head` is the same
`Linear(512,24)` as `head`. A smoke test produced identical loss to four decimals. The
GFlowNet *architecture* contributes nothing — what makes a GFlowNet a GFlowNet is the
objective. So cell 5 trained true trajectory balance instead.

**Cell 5 (trajectory-balance GFlowNet): falsified.** Not a crash — a converged degenerate
solution. The backward policy collapsed below uniform (argmax P_B = 0.0260 against 1/24 =
0.0417), meaning the model had learned to distribute probability *away* from the
structure it was supposed to find. Killed at epoch 410.

**Path labels (`tq1`): rejected.** Feeding replay-verified solution paths as extra Q
labels — well-motivated, since walk labels degrade past the diameter — didn't beat sparse-Q
alone.

**Deeper BFS anchors (`tq3_d7anchor`): axis closed.** We built the d ≤ 7 table (433M
states) and A/B'd anchor depth. It did nothing. The same run also priced d8 at zero, so
that was never built.

**The AZ value head costs ~1 point of top-1 on the Q head** — and we used it anyway, for
reasons that took another week to become clear.

### Two architectures that recovered but didn't win

A **latent-relational** model compressed the state too aggressively (505 vs 462 at one
frame). Its replacement, **Generator-ISAB**, kept all 50 physical piece tokens, exact
generator geometry and a zero-initialised residual over a frozen ResMLP — and recovered
ResMLP quality (462 vs 468) without closing the gap to the transformer (438). Rejected as
a deployment candidate, retained as a diversity option.

Its most useful output was a *negative* finding, which we'd independently rediscover
later: **offline probe metrics do not select good beam checkpoints.** Across its 15
checkpoints, pair accuracy and top-1 were flat to three decimals while path totals swung
by 10 moves. Epoch 1400 had the best top-1 and scored 469; epoch 1300 had *lower* top-1
and scored 462.

> The Q probe measures average action prediction. Beam search is controlled by a small
> number of ranking mistakes near the beam cutoff over a long trajectory. Those objectives
> are not equivalent.

---

## Phase 4: the inference-side grind

By this point the model was strong and width was expensive. We spent a long day testing
everything that might improve search *without* retraining. Almost all of it failed, and
the failures are more instructive than the wins.

### Width vs diversity, at matched budget

The temptation with a beam is to run it K times with different seeds/checkpoints and take
the per-pid minimum. We tested that against simply spending the same compute on a wider
beam.

| config | 15-pid total | compute |
|---|---|---|
| 1M ×1 | 432 | 1 unit |
| 1M ×1, min over **4** checkpoints | 426 | 4 units |
| **4M ×1** | **419** | 4 units |
| 4M ×1, min over 2 checkpoints | 417 | 8 units |
| 8M ×1 | 417 | 8 units |

**Width beat checkpoint-diversity by 7 moves at identical cost.** And diversity
*saturates*: the min over {ep900, ep1000} was already 426 at 2 units, and adding two more
checkpoints to reach 4 units bought exactly **zero**.

The width curve has its own ceiling, though: 1M → 4M was −13, but 4M → 8M was only −2.

### Model soup: better on every metric, zero moves

We averaged the *weights* of three checkpoints (SWA / "model soup"). The souped model beat
every input on pair accuracy, top-1 and gap, in **every depth band** including the deep
ones where the beam struggles.

In beam search it was worth **exactly 0 moves** (432 vs 432 at 1M; 420 vs 419 at 4M).

That was the third independent confirmation of the probe-metrics lesson, and it's now a
standing rule: on this problem, only a replay-verified beam total counts.

### What actually worked

**Cross-architecture blending.** Averaging the transformer's Q scores with a **ResMLP's**
at 0.8/0.2 gave −2 moves. The economics are what make it viable: the ResMLP is ~1/17 of
the transformer's forward cost, so the blend runs at **~1.07×**, not 2×. A weight sweep
found a broad plateau from 0.70–0.95, and the two heads turned out to share a distance
scale almost exactly (std ratio 1.000), which is what makes raw averaging legitimate.

**V-consistency.** The AZ dual-head model emits both `Q(s,a)` and `V(s)` from **one trunk
pass**, so reading the value head is free. If the parent is at distance `V(s)`, the best
child should be at `V(s) − 1`, so we charge each candidate by how far its Q sits from that
expectation:

```
score(s,a) = Q(s,a) + λ · |Q(s,a) − (V(s) − 1)|
```

Worth −4 moves at zero cost. Decomposing it is instructive: for the common case
`Q ≥ V−1` this reduces to ranking by `Q(s,a) − 0.23·V(s)`, a *between-parent* bonus
favouring parents that have a genuinely progress-making move — a steepest-descent bias.
For `Q < V−1` it damps children whose Q is implausibly optimistic given the parent's own
estimate. One term, two jobs.

**This is why the AZ head earned its keep.** It costs ~1 point of top-1 on the Q head, and
the beam never reads its value output by default. But cell 4 (with AZ) kept improving for
800 epochs after cell 3 (without) went flat — 424 vs 438. **The head's main value is
trainability, not the value signal.** The −4 from consistency was a bonus on top.

### The frame discovery

Our symmetry ensemble had always used frame `(0, forward)`. The handoff recorded — from
much earlier ResMLP work — that the **inverse** frame does the work, not the spatial
rotation. We had never tested that on the transformer.

At 4M: forward frame 416, **inverse frame 414**, identical cost. The inverse frames also
ran ~9% *faster*.

Then we replayed a completed four-frame run per frame, which cost nothing:

| frames used | total |
|---|---|
| k1i alone | 405 |
| k0i alone | 408 |
| k0f alone | 411 |
| k1f alone | 413 |
| **best pair (k0f + k1i)** | **400** |
| best triple | 400 |
| all four | 400 |

**Frames saturate at two.** The third and fourth contributed exactly zero — half of a
12-hour run had been wasted. And the winning pair mixes *both* axes (one forward, one
inverse, different rotations); picking the two best individual frames (k1i + k0i) would
have been the wrong heuristic, because two inverse frames are more correlated with each
other.

### Things that came back neutral

- **`history_depth` beyond 1.** Excluding candidates seen in the last *n* layers: h0 → h1
  is worth 1 move, and **h4 was byte-identical to h1** across all 15 pids. Saturates
  immediately.
- **Q-shortlist → V-rerank** at α=2: −7 moves, but at 3.0× cost. A cheaper band-rerank
  variant got −5 at 1.4×. Both were superseded by consistency, which is free.
- **Window reduction.** Replacing any path window whose net group element sits in the d ≤ 6
  table with the table's optimal word. It earned −16 on the loose original floor and
  **exactly 0 rewrites** on our tight merge — the paths are locally optimal at the 6-move
  scale. The remaining slack is *structural*: it needs a different route through the
  group, not a local rewrite.

---

## The final inference stack

```
PieceTransformer Q head (ep1500, 3.44M params, AZ dual-head)
  + ResMLP+AZ blended at 0.8 / 0.2          (~1.07× cost, −2)
  + V-consistency λ = 0.3                    (free, −4)
  ↓
beam width 16M, sharded across 8 TPU cores  (one shared beam, not 8 independent)
  history_depth = 1                          (h4 measured identical)
  ↓
2 frames: k0-forward + k1-inverse            (frames saturate at 2, −5)
  ↓
exact d ≤ 6 endgame table (27.8M states)     (tail provably optimal)
  ↓
replay-verify every path against the original scramble
  ↓
n-way per-pid min across every result source
```

Running on a v6e-8 that's ~740 s/pid/frame at 16M, so ~25 min/pid at two frames.

**Some architecture of the kernel matters too.** It's a single beam of 16M *sharded across
all 8 cores* — children are hash-routed to their owner core with `all_to_all`, and dedup
plus top-k happen per owner. That's one beam of 16M, not eight independent beams of 2M.
Backpointers are packed into a uint32 (24 parent-local bits + 3 rank + 5 move) and
streamed to a host memmap, so beam width is bounded by HBM for the *states*, not by tree
storage.

---

## What the final push looked like

The last day was a merge grind more than a search grind. The n-way per-pid minimum across
every source is the actual submission, and the sources are scattered:

```
28,559  session start
28,517  our own beam JSONs from sibling sessions (tier runs never folded in)
28,500  top-20 long-path run at 16M × 2 frames
28,493  submission_publ.csv
28,484  submission_publ1 + publ2
28,481  a community GitHub result set  ← exactly level with the leader
28,475  submission_dad.csv             ← first submission, beat the bar
28,471  the completed top-20 run
28,468  a sibling session's file that surfaced on the 6th merge
28,467  full re-pull of all TPU JSONs  ← final submission
```

Two lessons from that column of numbers.

**The merge's source set is not stable.** Successive runs scanned 121, 112, 122, 128, and
finally 131 CSVs — because session scratchpads are temp directories that appear and vanish,
and because an early `scp` of TPU results went stale. The final full re-pull took the JSON
count from 78 to 121 and found moves. A stale copy caps the merge with no warning.

**Result files hide in the wrong folders.** Three were in an unrelated
`rogii-wellbore-geology-prediction/` directory; one was in the `megaminx/` folder but
contained *tetraminx* data. Together they were worth −22 moves. The only reliable test is
by **content**: does the header read `initial_state_id,path`, *and* is the move alphabet a
subset of this puzzle's generators? A megaminx file passes the first test and fails the
second.

Worth noting for attribution: several of those "external" files were **other people running
the notebook we published**. That isn't a third-party source — it's our own solver
returning results from hardware we didn't pay for. Of the day's −92, about −62 came from
runs we executed and −22 from our published solver running elsewhere.

---

## The meta-lessons

**Offline metrics don't predict beam quality.** Four separate times — a graph-transformer
V, an all-neighbours Q head, the Generator-ISAB sweep, and the model soup — something won
on pair accuracy, top-1, recall or calibration and moved **zero** moves in search. The
mechanism is consistent: probes measure *average* prediction quality; beam search is
decided by a handful of ranking mistakes near the cutoff over a 30-move trajectory.

**Match your controls or you'll measure nothing.** We spent most of a day comparing local
runs at `history_depth=0` against TPU baselines at `history_depth=1` and produced two
conclusions that had to be retracted. A related trap: if a config change produces
*byte-identical* results, the flag isn't wired — that's how we found `--history-depth` had
been parsed and never forwarded.

**Don't generalise an interaction from one measured pair.** We measured that souping
doesn't stack with history_depth and asserted the same of V-consistency, which had never
been tested. It turned out **90% additive** — the two win on disjoint pids.

**Yield estimates on a length-sorted list are biased optimistic.** Our running estimate on
the top-20 longest pids fell monotonically from −2.00 to −1.45 moves/pid as the sample
grew, because the list was sorted by the very quantity that predicts yield.

**Saturation is everywhere, and it's cheap to check.** Frames saturate at 2. History
saturates at 1. Checkpoint diversity saturates at 2. Width saturates around 4M. Anchor
depth saturates at d6. Almost every knob on this problem has a ceiling two or three steps
in, and the ones we found by *replaying existing logs* cost nothing to find.

---

## Two things that closed after the submission

**The training axis.** Cell 4 kept improving long enough that we extended its budget from
1500 to 2000 epochs. Every 100th checkpoint was beam-evaluated:

| ep | 1200 | 1300 | 1400 | **1500** | 1600 | 1700 | 1800 | 1900 | 2000 |
|---|---|---|---|---|---|---|---|---|---|
| total | 426 | 428 | 431 | **424** | 426 | 427 | 425 | 428 | 428 |

Band 424–431, no trend across 800 epochs, while training loss fell the whole way. **ep1500's
424 is the low tail of a noise distribution, not a peak** — and the extension was triggered
by a 2-move gap that this spread can't resolve. The honest reading is that more epochs of
the same objective buy nothing; the next training gain has to change *what* is taught.

**Progressive top-k on TPU.** The JAX kernel materializes and hashes all 24 children before
its per-owner top-K; only the PyTorch searcher selects first and hashes the survivors. Porting
that to the kernel looked obviously right — the children array is `B_local × 24 × 88`, 4.4 GB
per rank at 16M, and the handoff estimated 32M at ~8.8 GB as "likely too big".

The port works and is verified correct (identical paths). It is also **strictly slower**:
+33% at 1M, **+110% at 32M**, for identical output.

The premise was false. **32M runs unchunked without it**, no OOM — the estimate marked
*UNTESTED* in our own doc had been treated as fact. One control run would have killed the
idea before any code was written.

Why it loses on TPU but wins on GPU is the reusable part: `states[:, all_moves]` is one
*regular strided* gather that XLA fuses, and the port swaps it for two *irregular* gathers
plus a serial chain. And the "progressive" adaptivity — `while uniques < B: k *= 2`, which
almost always exits at the first `k` — is impossible under static shapes, so TPU always pays
the worst case. It's now default-off and documented, and the regime where it would help
(width > 32M) is one the saturation curve says is worthless anyway.

## What's still open

- **Sym-pooled beam** — one `K×B` beam over all rotated+inverse copies pooled, instead of
  K sequential B-beams, so width is allocated *competitively* between frames rather than
  uniformly. Validated on megaminx (a stubborn pid went 101 → 82 at equal budget), never
  ported here.
- **NISS** — building one solution from both ends. Structurally related to the inverse
  frame; megaminx measured it as redundant with frame diversity at 2× wall, so the prior
  is poor, but it's never been measured on this puzzle.
- **Reconciling walk labels with the exact BFS table.** We measured that 37% of walk pivots
  land in the d ≤ 6 table, and **26% of those** carry a walk index that isn't the true
  distance. The ordering survives 92.75% of the time — which is *why* sparse-Q tolerates it
  — but ~7% of labelled pairs assert a gap that doesn't exist. The table is already resident
  on-device for the anchors, so replacing those labels with exact distances is nearly free.
- **The second half of the top-40 longest pids**, which is where the remaining slack in the
  length-31 tier lives.
