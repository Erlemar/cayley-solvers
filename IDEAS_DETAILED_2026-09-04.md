# Seven ideas, worked out: structured scorers, better labels, robust selection, and a cube666 programme

Date: 2026-09-04. Companion to `RESEARCH_SYNTHESIS_2026-09-04.md`, which ranks these
against everything else. This file explains each idea well enough to implement and to
gate, with the code it attaches to and the measurement that would falsify it.

Notation used throughout:

- `s` a state, `a` a generator, `s.a` the child, `inv(a)` the inverse generator.
- `d(s)` true distance to the goal set; `Q(s,a)` the model's estimate of `d(s.a)`;
  `V(s)` the model's estimate of `d(s)`.
- `M` mixing depth = `log_b(index of the target set)` (the counting bound); `V_vis` the
  deepest depth at which the scorer's child ranking is still >= 2x chance;
  `Z = start distance - V_vis` (blind zone). A beam solves iff `Z <~ 10`
  (`CUBE666_MODEL_RESEARCH_2026-08-24.md:81-101`).
- Sparse-Q rw-middle label: walk `k` steps non-backtracking from solved, pick a pivot
  `p`, label `Q(s_p, undo) = p-1` and `Q(s_p, next) = p+1`, mask the other columns
  (`tetraminx/scripts/51_train_sparse_q.py:215-270`, `cube555/.../qtrain.py:125-190`).

Numbers that anchor everything below:

| puzzle | M | scorer | V_vis | V_vis / M | top-1 by depth (chance) |
|---|---|---|---|---|---|
| cube444 (colour) | 36.7 | V ResMLP 3.3M | ~M | ~1.0 | - |
| cube555 (super) | 66-72 | sparse-Q ResMLP 24.8M | >= 65 | ~0.9 | 0.246 @40, 0.182 @60 (0.033) |
| cube666 (super) | 102 | sparse-Q ResMLP 26.4M, k_max 122 ("arm A") | 75-80 | ~0.78 | 0.090 / 0.043 / 0.035 / 0.029 @40 / 60 / 72 / 85 (0.028) |
| tetraminx | ~27-29 | PieceTransformer-Q 3.4M | ~M | ~1.0 | 0.537 shallow, 0.105 deep band |

Every idea in sections 1-6 is a way to move `V_vis / M` up or to make the beam survive a
given `V_vis`; section 7 assembles them into a cube666 plan.

---

## 1. Cluster-factorised PieceTransformer-Q for n x n x n cubes

### 1.1 Why a structured scorer, and why not the flat transformer

The evidence that tokenisation + objective matter on permutation puzzles is the strongest
architecture result in the repo: PieceTransformer-Q 424 vs ResMLP-Q 461-468 on the
tetraminx 15-pid bench (-8%), and on cube444 Vlad Kuznetsov's transformer beat our whole
V-beam pipeline on 916/1043 pids with 0 losses, uniformly across depth bands
(`cube444/EXPERIMENTS.md:626-645`). On 666 the flat ResMLP-Q is at chance by depth 85 and
already weak at depth 40, which is not a label-physics limit (555 gets 0.246 at the same
relative depth) but a representation limit.

A flat piece transformer on 666 has 152 tokens (8 corners + 48 wings + 96 centres). Attention
cost scales with `T^2 d`: `152^2 / 51^2 = 8.9x` the tetraminx model per layer, and the MLP
part `T d^2` is 3x, so ~4x per state overall. The tetraminx run consumed ~14.7B token-rows in
74 h on a v6e (`TORCHTPU_FINDINGS.md`), and attention at T=51 already sits at ~2.5% MXU
occupancy. A 4x-more-expensive model at 5-10B rows is a multi-week TPU job. That cost is
avoidable because the cube has structure the flat model has to rediscover.

### 1.2 The structure

`cube666/scripts/00_probe_structure.py` finds nine invariant 24-sticker orbits; physically
that is one corner cluster (8 pieces x 3 stickers), four centre clusters (24 x 1) and two
wing clusters (24 x 2), 152 pieces. Every generator acts as a permutation *within* each
cluster; the only coupling between clusters is that one move touches several of them at
once (an outer turn touches all seven, an inner slice touches four centre clusters and two
wing clusters but no corners). The joint distance is emphatically **not** a function of
per-cluster distances (the sum-of-orbits scorer sits in the stabiliser trap, `kept 24/24/0`,
`CUBE666_SOLVER_FROM_SCRATCH.md:130-138`), but the *input* factorises exactly, and the
classical solver exploits precisely this factorisation.

### 1.3 The model

```
Stage A  per-cluster encoder, weights shared across clusters of the same type
         (types: corner, centre, wing; 4 centre clusters share one encoder, 2 wing
         clusters share one, corners have their own)

  for cluster c with n_c slots (24, or 8 for corners):
      token_j = sum_s  E_val[type(c)][slot-local sticker s on slot j]   (folded one-hot
                                                                          matmul, as in
                                                                          _folded_piece_tokens)
              + E_slot[type(c)][j]                                       (slot position
                                                                          within cluster)
      h_c = EncoderBlock x L_A over the n_c tokens          (L_A = 2, d = 256, 8 heads)
      u_c = LN(attention-pool(h_c)) + E_cluster[c]          (one summary per cluster;
                                                             E_cluster distinguishes the
                                                             four centre clusters etc.)

Stage B  cross-cluster mixer
      z = EncoderBlock x L_B over [CLS, u_1 .. u_7]         (L_B = 2-4, 8 tokens)

Heads   from CLS:   Q(s, .) in R^36   (the beam scorer)
                    V(s)    in R      (AZ value head, feeds qv-consistency)
        from u_c:   auxiliary per-orbit heads (1.4), training only
```

Cost per state: Stage A attention is `7 x 24^2` instead of `152^2` (a 17x reduction on the
quadratic part), Stage B is negligible; the MLP part is the same as the flat model. Net
cost is close to the tetraminx model (T=51) rather than 4x it, so the 5-10B-row budget is
the same order as the 555 ResMLP run (6.14B rows, 23 h on one A100).

Everything else is inherited: the folded one-hot input stage and the one-hot matmul
instead of `index_select` (`tetraminx/src/tetraminx/models.py:359-375` explains why each
is load-bearing), pre-norm blocks with the `attn_impl="einsum"` path on TPU, `get_model_config`
for checkpoint interchange, and a JAX mirror in the style of `cube555/tpu/jax_model.py` for
the beam kernel.

A cheaper sibling for primitive-move beams where the forward dominates the step: replace
Stage A's attention with a per-cluster MLP over the concatenated 24 slot embeddings and
keep Stage B; this is "orbit-factored one-hot" (`CUBE555_PROGRESS.md`, untried) plus a
mixer, and it costs about what the current ResMLP costs.

### 1.4 Auxiliary per-orbit heads

The point of the auxiliary heads is to force each cluster summary `u_c` to carry the
permutation structure of its cluster explicitly, so that Stage B mixes *structure* rather
than raw stickers. Two families of targets, both free:

1. **Exact structural features of the state** (self-supervised, deterministic): per
   cluster, pieces at home, number of cycles, `(n - cycles) / 2` (the 3-cycle unit count
   the classical residual uses), parity. These are computed on the fly from the state.
2. **Per-cluster diffusion depth from the same walk.** In a walk of `k` moves, cluster `c`
   is touched by only the moves that act on it (corners by the 12 outer turns only), so the
   induced walk on cluster `c`'s Schreier graph has length `k_c <= k`. `k_c` is a
   diffusion-distance label for the *coset* distance of cluster `c` in isolation, and it
   comes out of the sampler for free by counting per-cluster touches during the walk loop.
   Its use is as a target for `u_c`, not as a scorer: a sum of these heads is exactly the
   scorer that was measured stuck.

Weight the auxiliary terms at ~0.1 of the sparse-Q term by gradient-norm ratio, and anneal
them to zero over the last 20% of training so they cannot compete with the joint ranking
objective at the end.

The 555 progress doc lists 5-piece PDB values (`24P5 = 5.1M` entries, ~5 MB each) as
candidate input features and auxiliary targets; those fit the same slot (family 1) and are
cheap to add later.

### 1.5 What the tokeniser needs

`tetraminx/scripts/50_derive_piece_layout.py` already derives pieces from the generators
alone (facelets grouped by stabiliser, verified as a block system) and reproduced the
hand-written IHES 26-piece layout exactly. It needs one extension: the **orbit of each
piece under the group** (cluster id and cluster type), which `00_probe_structure.py`
computes for stickers. Output `piece_layout.json` gains `cluster_id`, `cluster_type`,
`slot_in_cluster`. The same file drives 555 (`8 + 24 + 12 + 24 + 24` pieces) and 777.

### 1.6 Gate and cost

Gate is visibility, nothing else, exactly as P1 states: top-1-by-depth on a fixed held-out
walk bank at depths 40/60/72/85/95/105 (>= 1,024 states per depth) against arm A's
0.090/0.043/0.035/0.029; report `V_vis`; the sibling gap `Q(next) - Q(undo)` by depth; then
the descent probe with `fix_mean >= 9.0` before any 2^21 beam. Promotion to a full beam
only at `V_vis >= 90`.

Falsifier: the cluster model's recall-by-depth is no better than the flat ResMLP at matched
wall (then the 666 problem is labels, not representation, and sections 2-4 carry it).

Cost: 1 GPU-day to build and smoke, ~1-2 TPU-days to train to 5B rows.

---

## 2. `k_max = 0.9 M` as a rule, with pivot mass re-tilted into the band that decides survival

### 2.1 Mechanism

The sparse-Q label asserts a gap of exactly 2 between `undo` and `next` at the pivot. That is
true only when the walk is locally geodesic there. In the linear regime the probability that
`undo` is descending is `(1 + drift) / 2 = 0.972` on 666 (drift 0.944,
`CUBE666_WINDOW_PLAN.md`), but as the walk approaches the mixing depth `M` the walk state
becomes a random group element at the typical distance and `undo` is descending only about
half the time. Past `M` the ranking label is a coin flip with an asserted gap of 2, and the
absolute label `p` overstates the true distance by a growing margin.

Tetraminx measured this directly: `k_max = 40` against a diameter of ~29, and "past the
diameter 'undo the last move' stops being distance-reducing -- the ranking label itself
degrades" (`tetraminx/TRANSFORMER_AZ_RECIPE.md:240-244`). Arm A on 666 used `k_max = 122`
against `M = 102` with `pivot_tilt = 0.5`, so roughly 20% of its pivots were past mixing and
the tilt oversampled exactly that band. The CayleyPy paper uses `K_max` of 0.9-1.0 M
everywhere (26 on 3x3x3, 45 on 444, 65 on 555).

### 2.2 The rule, and the one thing it must not break

`k_max = 0.9 M`, with `M` taken from the counting bound of the target set: 444 -> 33,
555 -> 60-65, 666 -> 92, tetraminx -> ~26. For coset targets (section 7's rungs) `M` is the
log-index of that rung's target set, not the full group's.

The window plan argues the opposite (`CUBE666_WINDOW_PLAN.md:234-248`: "keep `k_max` at 120
and move the pivot distribution"), because the teacher arm's `k_max = 40` compressed the Q
ceiling from 72 to 35 and cost solves: a beam wanders above its start depth and needs
*level* headroom there. Both are right about different things. The ranking assertion past
`0.9 M` is noise; the absolute level past `0.9 M` is still needed so that a global top-B can
compare deep parents. So split the two:

- **Ranking stream** (`undo`/`next` pair with the 2-gap): pivots only at `p <= 0.9 M`.
- **Level stream** for pivots with `p > 0.9 M`: supervise the value head `V(s_p) = p` (or,
  without a value head, set *both* labelled columns to `p`, asserting no gap), so the Q
  scale keeps its headroom but no fictional sibling ordering is taught.

This is a two-line change in the sampler (the row's mask and target depend on whether
`p > 0.9 M`), and it keeps `k_max` at whatever the beam's depth excursion needs (1.2 M is
fine for the level stream).

### 2.3 Re-tilting the pivot mass

The sampler draws `k ~ U[k_min, k_max]` and `p = floor(u^gamma (k-1)) + 1` with
`gamma = 1 / (1 + tilt)`; `tilt = 0.5` moves `E[p / (k-1)]` from 0.50 to 0.60
(`qtrain.py:134-135`). The resulting pivot-depth histogram is a triangle-ish mixture with
most mass in the middle third of the walk, and the trainer already logs it at startup
(`51_train_sparse_q.py:788`, `hist = bincount(vp)`), so any change is verifiable in one line.

"The band that decides survival" is where the beam currently loses the path: on 666 that is
depth 40-72, where top-1 is between 3.2x and 1.25x chance (signal exists but is thin); the
target stated in the window plan is **0.090 at depth 40 -> 0.15-0.20 at depth 45**. On a
new puzzle, read the band off the recall-by-depth curve as the interval where top-1 is
between ~1.5x and ~4x chance.

Implement it as a target histogram rather than a tilt exponent: draw `p` from a trapezoid
over `[0.25 M, 0.55 M]` with 60% of the mass, uniform elsewhere on `[k_min, 0.9 M]` with the
remaining 40% (so the shallow bands that feed the endgame and the anchors are not starved),
then draw `k ~ U[p + 1, k_max]`. This decouples the pivot depth from the walk length, which
the current scheme cannot do.

### 2.4 Gate and cost

Retrain with the matched control (same seed family, same rows). Gate on the recall-by-depth
curve: top-1 at depth 45 must rise from 0.09 toward 0.15 with no loss at depth 20-30, and
the level-stream diagnostic (`min_a Q` on random states at depth 100-120) must not compress
below the control's. Only then a beam. Cost is a config change plus one retrain; the
change cannot be applied mid-run because it is an objective change.

---

## 3. Reconcile walk labels with the exact table where they overlap

### 3.1 The measured defect

Two training streams label overlapping states by different rules: anchors use exact BFS
distances for a state and all of its children; walks use the walk index. Non-backtracking
forbids only the immediate inverse, so a walk is not geodesic (with an order-3 generator,
`g, g` has walk index 2 at true distance 1). `tetraminx/scripts/57_label_consistency.py`
measured it on 200k pivots (`k in [2,40]`, tilt 0.5; `tetraminx/HANDOFF.md:515-546`):

| quantity | value |
|---|---|
| pivots whose state is inside the d<=6 table | 37.1% |
| of those, walk index `p != d(s)` (always `p >= d`) | **25.9%** |
| labelled `undo` child: label != true distance | 21.9% |
| labelled `next` child: label != true distance | 31.0% |
| ordering `d(undo) < d(next)` correct | 92.75% |
| ordering tie (the asserted gap of 2 is fictional) | 6.14% |
| ordering inverted (teaches the wrong ranking) | 1.11% |
| mean true gap `d(next) - d(undo)` | 1.717 (label asserts 2) |

Scaled by the in-table rate, ~2.7% of all walk rows carry a false gap and ~0.4% teach an
inverted ranking, and the absolute scale of the walk stream conflicts with the anchor
stream on the same states. The 6.14% ties are the same benign-alternative-optima problem
that neutralised the megaminx rank losses, sitting inside the primary objective.

### 3.2 Three fixes, in order of strength

- **Replace (recommended).** Hash the pivot and its two labelled children (3 lookups per
  row; the anchor stream already does 12,288 per step) and, when a state is in the table,
  substitute the exact distance. Stronger form: if the *pivot* is in the table at depth
  `<= D-1`, every child is at depth `<= D`, so the whole row can become a full exact anchor
  row (all 24/36 columns). That upgrades 37% of tetraminx walk rows from approximate to
  ground truth and removes every tie and inversion inside the table.
- **Clamp.** Set `p := d(s_p)` when the pivot is in-table. Fixes the level conflict, keeps
  the 2-gap assertion (so ties stay wrong). Cheapest, weakest.
- **Drop.** Exclude in-table pivots from the walk stream and let the anchors own that
  region. Clean, but it removes rows from the band where the endgame handoff happens.

Machinery already exists: baked anchors (`45_bake_anchors.py`, `baked_anchors_d5.npz`) and
the on-device Zobrist hash + `searchsorted` that the beam's endgame test uses
(`eg_hashes`, `eg_ztab` in the kernels). On TPU the anchor pool is HBM-resident (0.19 GB on
tetraminx), so the three extra lookups are a gather, not a host round trip.

### 3.3 What it is worth on each puzzle

The in-table fraction of pivots is set by the table depth relative to `k_max`: 37% on
tetraminx (d<=6 against `k_max` 40); on cube444 the anchors reach d<=6 (67,041,677 states)
against `k_max` 60; on 555 d<=4 (ball(<=5) = 10.7M) against 80; on 666 ball(<=4) = 928,804
or ball(<=5) ~ 26.9M against 92-122. On the big cubes the direct effect is a few percent
of rows, but those rows are exactly the last 5-6 moves of every solve, where the beam is
narrowest and the endgame splice takes over, and the 25.9% in-table mislabel rate is the
only *measured* estimate of how non-geodesic the walks are; the out-of-table rate is
unknown and at least as large, which is what sections 2 and 4 address by other means.

### 3.4 Gate and cost

An objective change, so a new run with its matched control. Measure (a) calibration inside
the table on held-out d<=6 states (MAE of Q against exact); (b) recall-by-depth unchanged or
better at depth 20-30; (c) the beam total on the standard bench. On tetraminx the noise band
is 424-431, so a claim needs <= 416. Cost: half a day to wire, one retrain.

---

## 4. The slack-2 hinge

### 4.1 Derivation

For any parent `s` and any two generators `a`, `u`: `s.a -> s -> s.u` is a two-move path, so
`|d(s.a) - d(s.u)| <= 2` unconditionally. In particular every column of a consistent Q
vector satisfies `Q(s,a) >= Q(s,undo) - 2`. This is a constraint on the model's own output
vector, not a label, which is why it survives non-geodesic walk prefixes and depths where
no label is trustworthy.

```python
q      = model(states)                                   # (B, n_actions)
q_undo = q.gather(1, undo[:, None]).detach()             # supervised column, stop-grad
loss_low = torch.relu(q_undo - margin - q).mean()        # margin = 2.0
```

Three points that are not obvious from the formula (`CUBE666_WINDOW_PLAN.md:164-228`):

- **Stop-gradient on the reference.** Violations must push suspicious columns up, not let
  the loss cheat by pulling the `undo` value down.
- **The reference must be a supervised column.** The fully pairwise version,
  `relu(q.max(1) - q.min(1) - 2)`, is more general but has no anchor, and the cheapest way
  to satisfy it is to move an unsupervised value. `Q(s,undo)` is pinned by the sparse MSE.
- **Do not use `next` as a second reference.** Its label is `p+1`, an over-estimate under
  walk slack, so the floor `Q(next) - 2` sits above the truth and suppresses good children.
- **Skip the upper hinge.** The beam selects on low Q; the upper tail is never sampled, and
  compressing it eats the headroom the sparse-Q gap term is trying to build.
- **Never the zero-margin "undo is argmin" version**, not even as a control: it assumes the
  undo child is the minimum, which fails exactly on the deepest rows that `pivot_tilt`
  oversamples, and it validates clean on shallow probes.

### 4.2 Why it targets the measured pathology

The 666 descent probe showed a 2^21-2^22 beam converging on states where Q is spuriously
low (frontier `fix_mean` 1.3 against a random-state 9.0). A false-low column is what wins a
global top-B slot. The hinge lifts any column more than `margin` below its parent's `undo`
column, which is the tail the beam selects from. It is orthogonal to qv-consistency (which
compares `Q` to `V - 1` across the two heads) and to section 5 (which attacks the same tail
at inference).

### 4.3 The free pre-flight, and an inference-time twin

Before spending a GPU-day: score ~100k walk states across depth bands with the current model
and histogram `Q(s,undo) - min_a Q(s,a)` at thresholds 0 / 2 / 4 / 8 by depth; separately,
of the columns the beam actually selects into its top-B, measure the fraction that sit more
than 2 below their parent's `undo` column. If the tail is not there, or the false-low
columns are not the ones winning slots, the training arm is inert by construction.

The same constraint is available at inference for free, because in a beam the child's
`undo` column is the inverse of the move that produced it: `Q(child, inv(m))` points back
at the parent. Clamping `Q(child, a) := max(Q(child, a), Q(child, inv(m)) - 2)` before the
global top-B costs one gather and tests the hypothesis without training. Run it as part of
the pre-flight on the descent probe.

### 4.4 Margin, weight, arms, gate

`margin = 2` is exact in true-distance units. The model's Q is compressed at depth (a true
2-move gap maps to a predicted gap well under 1 at depth 60+ on 666), so a flat 2 is a
loose rail there; sweep `{1, 2, 4}` and pick by the deep-band metrics. Set the weight by
gradient-norm ratio at 5-10% of the sparse-Q gradient norm and re-measure at ~10% of
training (the hinge is sparse, so its norm is near zero at init and can spike; `.mean()`
over 36 columns with 1-2 violating dilutes the gradient ~20x and the ratio absorbs that).

Arms: H0 existing objective; H2 slack-2 hinge; H3 margin sweep. Gate on recall-by-depth and
the sibling-gap curve, **not** on undo top-1, which improves mechanically under any hinge.

---

## 5. Variance reduction inside the scorer where width is anti-productive

### 5.1 The mechanism

A beam's global top-B over `B x n_actions` candidates is an extreme order statistic. If the
scorer's error on a candidate is noise with standard deviation `sigma`, the selected set is
biased toward negative error by roughly `sigma * sqrt(2 ln(B n_actions))`, and more width
selects a deeper tail. Where the per-step signal (one move) is smaller than `sigma`, width
selects error rather than progress. That is the measured picture on 666 (`fix_mean` 7.0-14.1
at 2^21 vs 1.3 at 2^22, both against random 9.0) and the likely reason width is
non-monotonic on 555 (2^20 2/6, 2^21 6/6, 2^22 3/6, 2^23 0/1). The general rule the probe
earned: a metric the search optimises cannot be its progress indicator; `min_a Q` fell
74.6 -> 11.2 while the frontier ended below a random state.

None of the remedies below trains anything. Each is gated on the descent probe (three pids,
2^21, 500 steps): `fix_mean >= 9.0` sustained, frontier max fixed stickers against the
calibration max, and exact d<=4 ball hits.

### 5.2 Remedies

1. **Frame-averaged and frame-max Q** (`CUBE666_WINDOW_PLAN.md:129-160`):

       Q_avg(s,a) = mean_k Q(sym(s,k), relabel[k,a])     # variance reduction
       Q_max(s,a) = max_k  Q(sym(s,k), relabel[k,a])     # pessimistic, suppresses underestimates

   Only the spatial frames are legal for Q rows (inversion transports V, not Q:
   `74_train_avi.py` docstring). `sym` and `relabel` exist in every kernel
   (`to_frame` / `RELABEL` in `cube555/tpu/build_notebook.py:406-420`). Cost is K forwards
   per parent; on the selection-bound TPU step (V-forward ~8% of a 48M megaminx step) K=8
   roughly doubles the step, and the point is to spend the width budget on better-scored
   candidates: 2^19 x 8 frames instead of 2^22 x 1. Pre-check in minutes, no beam: recompute
   top-1-by-depth for `Q_avg` on the existing bank. Stays ~0.03 at depth 85 -> the error is
   systematic bias and frames are dead in every form; rises toward 0.10 -> the signal is
   noise-masked and this is the fix. Check first whether the model was trained with
   symmetry expansion (555/666 were: 4-of-48 random frames), in which case it may already
   be near-equivariant and `Q_avg ~ Q`.
   This is distinct from frames-as-retries (12 separate beams, measured 0/20 on 666), which
   lets each beam select its own left tail.

2. **Conservative ensemble scoring**: `score = mu + beta * sigma` over K independently
   trained checkpoints (different seeds, or different recipes: the flat ResMLP, the
   cluster transformer, a dense-V model), `beta ~ 0.5-1`. Independent error realisations
   average out; the disagreement `sigma` is also a usable confidence signal for the
   diversity rule below. Memory lists "diverse small-net ensemble within the beam" as the
   only genuinely untried residue of the megaminx compass check.

3. **Row statistics of the child instead of the parent's single column.** In the Q-native
   beam a child is scored by `Q(parent, a)`; its own 36-column row is computed at the next
   step and discarded. `1 + min_a Q(child, a)` is the implied V and is itself biased low
   (min over 36 noisy columns). Re-scoring survivors by `mean_a Q(child, a) - c` (a
   lower-variance level estimate) before the final top-B is the qv-rerank pattern
   (`qv_alpha`, band rerank in `src/cayley/khoruzhii_search.py`) with a different statistic.

4. **Lineage-smoothed score**: `score_t = lambda * score_{t-1}(parent) + (1 - lambda) * Q`,
   averaging noise along the path. The PHS cumulative score is the same family (validated,
   marginal on megaminx at `w = 0.03`); on 666 the noise is far larger, so the trade-off is
   different and worth one probe.

5. **Diversity stratification** (IDEAS A7, untried anywhere): bucket candidates by a cheap
   exact signature (per-cluster cycle-count vector or pieces-at-home on cubes) and cap each
   bucket's share of the beam, so a single false basin cannot take the whole width. The
   666 failure is a collapse of diversity as much as a scorer bias; the beam keeps 3.26M
   globally unique states and still ends below a random state on fixed stickers.

### 5.3 Order of operations

`Q_avg` top-1 check (minutes) -> inference-time hinge clamp (section 4.3) -> descent probe
with `Q_max` -> ensemble `mu + beta sigma` -> stratified beam. Stop at the first one that
keeps `fix_mean >= 9.0` and produces ball hits; that one is the deployed selection rule, and
the training-side ideas (sections 1-4) are then measured under it.

---

## 6. AVI as a visibility extender, gated per round (cube666)

### 6.1 What Beam-AVI is, and the three verdicts it has

Approximate value iteration on the Q head with a perfect dynamics model:

    Q(s,a) <- 0                              if s.a is in the exact table (table value)
              1 + min_a' Q_target(s.a, a')   otherwise

with states harvested from the beam itself (the backup is on the GPU anyway: the child's
forward at step j+1 gives `min_a' Q(child)`; +0.2% wall, targets verified within one bf16
ULP, `BEAM_AVI_METHOD.md:80-160`), a `--full-expand` dense stream of uniformly sampled
parents to correct the survivor bias, exact anchors in every batch (do not Huber-cap the
level term: 18/18 -> 5/18 solved when that was tried), and a value-head term so
qv-consistency keeps reading the same scale.

| puzzle | scorer state | precondition (on-path percentile, null 0.5) | result |
|---|---|---|---|
| cube444 | signal at depth, not at ceiling | 0.034 (z = 30.2) | **-4.2% below the floor**, 37W/4L/13T; the same budget on random-walk states +528, 2W/46L |
| tetraminx | at the training-axis ceiling | 0.055-0.102 (passed) | 1 round ties, 5 rounds lose 1W/7L; Q-space flattens (deep gap 0.438 -> 0.270, cross-parent sd 6.83 -> 6.23) |
| cube666 flat | blind past depth 85 | fails at 85+ | monotone destructive, beam wins 13 -> 7 -> 6 -> 1 |

The pattern: AVI transports information from where the scorer has it to where the search
goes; it cannot create information (666 flat) and it has nothing to add at a ceiling
(tetraminx). The 666 scorer is far from any ceiling (top-1 0.09 at depth 40 against
tetraminx's 0.54 shallow) and has real signal to depth ~75. The information it lacks (true
distances at depth 80-100) is not in any walk label, but it is reachable by backing up
from depth-75 states whose ranking is real.

### 6.2 The loop as a curriculum

Round `r` starts with a measured frontier `F_r = V_vis` (from the recall-by-depth bank).

1. **Generate roots in the boundary band, not at the test scrambles.** Random-walk states
   at walk depth in `[F_r - 10, F_r + 10]`, in a random spatial frame. The test scrambles
   are at ~105, entirely past the frontier; a beam started there wanders blind and its
   harvested rows are junk. This is the difference from the "different machine" DAgger
   loop, which harvested from full-depth roots.
2. **Run the deployed beam** (2^18 is enough here, ~200 roots) and harvest the three
   streams of `73_gen_harvest.py`: sparse (survivors, biased), dense (`--full-expand 256`,
   unbiased), value; exact-table override where a survivor is inside ball(<=5).
3. **Train one round** with the four terms of `74_train_avi.py` (sparse masked MSE, dense
   MSE, exact anchor MSE, value MSE), warm from the current weights, **keeping the original
   sparse-Q walk stream in the mixture at ~50% weight**. The tetraminx loop replaced the
   walk stream with the shard; on 666 the walk stream is what holds the shallow band, and
   "distribution beats budget" cuts both ways.
4. **Re-measure**: `V_{r+1}` from the bank; the matched level check
   (`77_level_check.py`: implied `1 + min_a Q` on one fixed state population under every
   checkpoint, which predicted both tetraminx gate outcomes in two minutes); the deep-band
   top-1 gap and the cross-parent sd.

Continue while `V` rises by >= 2 per round. Stop and revert on any of: `V` unchanged;
shallow-band (20-30) top-1 down by more than 10% relative; cross-parent sd down by more
than 5% (the flattening signature); implied level *flat* at depth (the bootstrap carried
no information, the cube444 pathological arm).

### 6.3 Precondition, measured the right way for 666

Run `72_path_rank_probe.py`'s test at the frontier band, not on KMC paths. KMC paths are not
geodesic (2.66 bits/move against the 4.86 ceiling), so their survival table measures
imitation, not distance. Known near-geodesic references at depth `<= 0.9 M` are free: the
inverse of a random walk of that depth (drift 0.944). The cross-parent percentile of the
on-path candidate at depths 60-75 must beat 0.5 clearly (444 called 0.034 ample); at
85-100 it will be ~0.5, which is the point: the loop is supposed to move that boundary.

### 6.4 Budget and expectation

Each round is one generation pass plus 4,000-9,000 training steps, ~1-2 h on a TPU. Value
iteration moves the boundary by at most a few moves per round, so 75 -> 95 is 5-10 rounds
if it works, about two TPU-days. Falsifier after round one: `V` unchanged with the level
flat. Expected if it works: `V_vis >= 95` on the flat 666, which by the blind-zone law is
the threshold for direct solves.

---

## 7. Suggestions for the cube666, rewritten

### 7.1 What is fixed

- Puzzle: 216 distinct stickers, 36 quarter-turn slice generators, branching 29.02,
  `|G| = 3.1e149`, counting bound 102.2, not bipartite (12 odd / 24 even generators). Nine
  invariant 24-sticker orbits; 152 physical pieces in 7 clusters.
- Test set: pids 12-1011 are a scramble-length ladder `L = 1..1000` plus 12 duplicates;
  ~92% have `L > 80` and are random states. On the ladder below `L ~ 100` our file is
  ~0.92 L (post-processed sample words); on the 800 deep pids it costs 186.8/pid.
- Standing 2026-09-04: Diener 166,421 (~182 on deep pids), ours 169,769 (#2), deadline
  2026-11-20.
- Flat scorer (arm A, ResMLP-Q 26.4M, 3M updates, `k_max 122`): `V_vis` 75-80, `Z` 25-30,
  zero direct solves at any width, width anti-productive above 2^21.
- Imitation of the classical solver (the other machine's scalar-V line, 1.06M states from
  8,874 KMC paths, DAgger, symmetry, pair losses, q96 relabelling, C2500 distillation):
  every variant improved an offline metric, none solved PID 502, known path evicted at
  depth 3-4. Its labels are upper bounds of a non-Markov, phase-structured cost; the line
  is closed. Its assets (C2500, q96 pairs) are usable as an evaluation set, not as data.
- A complete model-only solver exists: the factorised 3-cycle finisher, 1,012/1,012
  replay-verified at ~705 moves/pid (`cube666/MACRO_POLICY_TRAINING.md:73-77`). "Solves
  fully" is done; "competitively" means `< 165/pid` to lead and `< 140` to matter.
- Classical reference (KMC): exact corners 10.35 + parity ~1 + SA rough phase + 3-cycle
  insertion; state 500 = 11 + 67 + 126 = 193. Its slack is global (the decomposition into
  cycle units); primitive-move re-solves of its windows at spans 34/58 gave 0.

### 7.2 Definition of done

Model-only: corners by a learned scorer (a 2x2x2-supercube-sized problem), every rung by a
learned scorer, the endgame by the exact d<=5 ball (a table, not a solver), every path
replay-verified. That is the same standard as tetraminx. The 3-cycle library behind the
finisher is classical knowledge and is the fallback only.

### 7.3 The programme

**S0. Calibrate on 555 first** (1-2 GPU-days). Our 555 stack is ~1.9x worse than the
CayleyPy paper on the same states (`CUBE666_MODEL_RESEARCH_2026-08-24.md:37-45`), and 555
has `M = 72`, the regime every 666 rung has to work in. Arms at matched budget and 2^21:
A0 sparse-Q ResMLP (control); A1 paper recipe verbatim (dense V, `K_max` 65, BN, 4M
ResMLP); A2 sparse-Q cluster-factorised transformer (section 1); A3 A1 with four seeds
min-merged. Gate: `V_vis / M` and inflation on 20 fixed pids. This tells us whether the
gap is labels (A1 wins), representation (A2 wins) or the paper's agent portfolio (only A3
wins), and each answer has a different 666 consequence.

**S1. A structured scorer trained properly** (2-3 TPU-days). Section 1's model with
section 2's split streams (`k_max` ranking cut at 92, level stream to 120, pivot mass
re-tilted into depth 25-55), section 3's exact-table reconciliation against ball(<=5),
section 4's hinge at the pre-flight-chosen margin, 4-of-48 random frames, >= 5B rows.
Gate: recall-by-depth bank at 40/60/72/85/95/105. `V_vis >= 95` opens the direct route
(predicted 1.2-1.3x the bound, 125-135/pid, ~130k); 85-95 goes to S1b then S2;
`<= 85` closes the flat family for this recipe.

**S1b. AVI as a visibility extender** (section 6), at most ten rounds, each gated.

**S2. A two-rung ladder whose rungs sit inside the measured `V_vis / M`.**
- *Colour-relaxation ladder (new).* Rung 1 solves the colour-cube projection of the state
  (relabel 216 stickers to 6 colours). The target set is the colour-stabiliser subgroup
  `S`; the coset graph has ~390 bits (500.6 minus ~110 bits of within-face centre identity;
  wings and corners stay distinguishable by colour), so `M ~ 80`, the 555 regime. Train it
  as a colour cube (walk-depth labels on colour patterns; 48 frames; no inverse frame
  because `S` is not normal; colour symmetry needs the recolouring conjugation the 444
  code has). Rung 2 solves the residual, an element of `S` (~110 bits, counting ~23 moves,
  realistically 40-60 because it is commutator-bound), with the flat supercube scorer,
  whose visibility of ~78 covers it (`Z < 0`). Predicted total at paper-quality rung 1:
  ~112 + ~50 = ~160/pid; at our current 555 quality (2.4x) rung 1 fails, which is why S0
  comes first. Gate (i), no training: build residual states by composing 5-40 random
  centre 3-cycles from `artifacts/three_cycle_library.json` onto the solved state, solve
  20 of each depth with arm A at 2^18-2^20 in the forward frame; need 20/20 at < 70 moves.
  Gate (ii): rung-1 `V_vis / M` on the colour graph. Note the colour cube is used as a
  relaxation *target*, not as the puzzle model; the deep-research report was rightly
  rejected for the latter.
- *Corners-first six-rung orbit ladder (docs' P2).* Rung scorers `D_k` = distance to
  "clusters 1..k solved", trained on the ladder-start distribution and refined by AVI
  on the rung (where the precondition holds by construction), never a sum of per-cluster
  values. Counting says `6 x ~21 + 11 ~ 137` at rung-1 quality, ~180 at 28/rung. The
  two-cluster gate (P2-b: 200 ladder-start states, 2^16-2^18, walk-only vs AVI-tuned) has
  never been run and decides it in a GPU-hour.

**S3. Search-side robustness for whichever scorer survives** (section 5): `Q_avg` /
`Q_max`, conservative ensemble, child-row rerank, stratified diversity, `history_depth`,
the d<=5 ball as goal set, sym-pooled allocation over 96 trajectories. Gate on the descent
probe (pids 927 / 852 / 665 / 502): `fix_mean >= 9.0`, ball hits.

**S4. Fallback and coverage.** The factorised finisher covers any pid the ladder misses.
If the 3-cycle library is disqualifying, the fallback is the flat beam plus ladder with no
macro stage and coverage has to be measured.

**S5. Score insurance for 2026-11-20** (not model-only): the docs' P3' anytime prefix
optimiser with the exact finisher as terminal oracle (`A(p,a) = C(p) - C(pa)`, every beam
state terminal, never worse than the fallback, ceiling ~160k); the KMC parameter gate;
96-trajectory KMC merges; one ledger reconciling the three incumbents on disk (169,769
submitted, 171,019 and 170,051 in docs).

### 7.4 Expected numbers, so the gates mean something

| route | condition | moves/pid | total |
|---|---|---|---|
| today (KMC + merges) | - | 168-170 | 169,769 |
| Diener | - | ~164 | 166,421 |
| colour ladder | rung-1 scorer at paper quality (1.4x) | ~160 | ~162k |
| direct flat beam | `V_vis >= 95` | 125-135 | ~130k |
| counting floor x 1.13 | the 110k target in the docs | 116 | ~117k |

### 7.5 First week, concrete

1. Restore `models/q666_a_final_ARMA_deployed.pt`; build the depth bank (1,024 states x
   depths 40/60/72/85/95/105) and the recall-by-depth harness with chance lines.
2. Section 4.3 pre-flight on arm A: violation histogram by depth; fraction of beam-selected
   columns more than 2 below their parent's `undo`; the inference-time clamp on the descent
   probe.
3. Section 5.2 item 1: `Q_avg` top-1 by depth on the bank (minutes); `Q_max` descent probe
   on pid 927 at 2^21.
4. S2 gate (i): residual states from the 3-cycle library, arm A forward-frame solves.
5. Launch S0 on 555 (four arms, 20 fixed pids, 2^21).
6. Extend `50_derive_piece_layout.py` with cluster ids and build the cluster-factorised
   model (section 1) on the 555 layout first, so it enters S0 as A2.

### 7.6 Do not

Another flat full-puzzle arm at any capacity with the arm A recipe; width above 2^21 on the
current scorer; frames-as-retries; Bellman on the full puzzle from a blind scorer; teacher
paths or KMC-suffix labels as a distance target; sum-of-orbit scorers; a strict nested
coset ladder with more than two or three rungs (rung cost is cumulative bits / 4.86, and a
3-rung reduction's counting bound is already 266 moves).
