# Megaminx Path-Shortening Strategy: Novel Ideas, Architectures, Training Regimes, Losses, and Feature Preparation

**Scope:** CayleyPy Megaminx / Cayley graph pathfinding.  
**Goal:** significantly decrease the length of found paths, not merely reduce runtime or training loss.  
**Core assumption:** the problem is a finite-group/permutation search problem, not a natural time-series, image, or tabular problem.

---

## 0. Executive summary

The strongest next directions are not “train another slightly different value MLP.” The current evidence suggests the old value-model recipe is saturated. The most promising improvements are likely to come from:

1. **Post-solution path compression with learned bridges**: replace bad middle segments, not only tails.
2. **Cross-solution relinking**: recombine prefixes and suffixes from different valid paths for the same pid.
3. **Solution-graph optimization**: run Dijkstra over original edges, tail-resolve edges, bridge edges, and macro edges.
4. **Action-centric architectures**: train models that preserve or rank good actions, rather than only predict scalar distance.
5. **Graph or permutation-matrix state encoders**: exploit that the state is a permutation under 24 generators.
6. **Bridge / relative-state models**: train `D(s_from, s_to)` and `Q(s_from, s_to, a)` from path windows.
7. **Teacher-compatible or multi-teacher qshort**: avoid pruning useful actions from AZ v4, m05, and other diverse teachers.
8. **Macro-Q with mined macros**: expand the action space only through a trained action shortlister.
9. **Suffix specialist models**: train narrowly on near-solved/tail distributions for tail repair.
10. **Search-generated training**: train on beam mistakes, hard negatives, frontier states, bridge successes, and SA/tail-resolve improvements.

The most important conceptual shift:

> Optimize for **candidate ordering and path shortening**, not scalar regression loss.

---

## 1. Problem properties that should drive architecture choice

The Megaminx competition state has several unusual properties:

- The state is a **120-element permutation**.
- There are **24 deterministic primitive generators**.
- Every move is a known permutation of the 120 positions.
- The objective is **shortest/shorter path length**, measured after a verified sequence of moves.
- The main solver is beam-style search guided by learned values / qshort / policy / symmetry.
- Path quality depends on **relative child ordering**, not only absolute value calibration.
- There are strong symmetries and action conjugations.
- Verified solution paths create a large training corpus of state-action-suffix examples.
- Multiple solvers find different valid paths, so path diversity is real and exploitable.

Therefore, architectures should expose:

- forward permutation: `state[i] = sticker in slot i`;
- inverse permutation: `inv_state[j] = slot containing sticker j`;
- slot geometry;
- sticker identity and home metadata;
- generator/action structure;
- symmetry/conjugation structure;
- prefix/suffix states along paths;
- residual permutations between two arbitrary states;
- action costs for macros.

---

## 2. Highest-EV novel path-shortening ideas

These are designed to reduce path lengths directly. Most are non-regressing if every replacement is verified before acceptance.

---

### 2.1 Neural bridge compression

#### One-liner

Replace arbitrary subpaths inside an already-valid solution by solving the relative residual between two intermediate states.

#### Motivation

Existing tail-resolve improves only suffixes that end at the solved state. BFS-d6 window replacement improves only very short local windows. Neural bridge compression generalizes both:

> Given two prefix states `S_i` and `S_j` from a valid path, try to find a shorter path from `S_i` to `S_j`.

If a shorter bridge is found, replace `path[i:j]`.

#### Core construction

For a valid path:

```text
S_0 --m_0--> S_1 --m_1--> ... --m_{n-1}--> S_n = solved
```

Pick a window `[i, j]`. The original bridge length is `j - i`.

We want a sequence `B` such that:

```text
apply_path(S_i, B) = S_j
```

Construct a residual state `R_ij` so that solving `R_ij -> identity` gives a bridge from `S_i` to `S_j`.

The exact composition order should be unit-tested against known windows because the repository convention is `new_state[k] = state[gen[k]]`. The expected residual form is one of:

```text
R_ij = inverse(S_j) ∘ S_i
```

or

```text
R_ij = S_i ∘ inverse(S_j)
```

Test both on simple windows and assert that the returned path transforms `S_i` into `S_j`.

#### Algorithm

```python
for pid, path in candidate_paths:
    prefix_states = compute_prefix_states(initial_state, path)
    best_path = path

    for window_len in [12, 16, 20, 30, 40, 60, 80]:
        for i in selected_positions(path, window_len):
            j = i + window_len
            residual = make_residual(prefix_states[i], prefix_states[j])
            found, bridge = solve_residual(residual, max_steps=window_len - 1)

            if found and len(bridge) < window_len:
                candidate = path[:i] + bridge + path[j:]
                if verify(candidate) and len(candidate) < len(best_path):
                    best_path = candidate

    return best_path
```

#### Candidate window selection

Start cheap:

- top 100 longest pids;
- windows ending near the tail;
- windows where V trajectory is flat or oscillatory;
- windows where many same-face reductions happen nearby;
- windows where different solutions diverge/rejoin.

Then expand to all pids if positive.

#### Verification

Every accepted replacement must pass:

```text
apply_path(original_initial_state, candidate_path) == solved_state
len(candidate_path) < len(current_best_path)
```

#### Expected impact

Potentially high. This is one of the best ideas because it can shorten already-valid paths without training a new global solver. If current paths contain mid-path detours, the savings can be much larger than BFS-d6 window replacement.

#### First experiment

```text
Dataset: top 50 or top 100 longest current-best pids
Solver: current best V/qshort stack at small beam first
Windows: 20, 30, 40
Acceptance: any verified net decrease
```

If positive, convert accepted windows into training data for a bridge model.

---

### 2.2 Cross-solution neural relinking

#### One-liner

Recombine the prefix of one valid solution with the suffix of another valid solution using a learned or searched bridge between their intermediate states.

#### Motivation

For a single pid you often have multiple valid paths from different solvers/configurations:

```text
A = A_prefix + A_suffix
B = B_prefix + B_suffix
C = C_prefix + C_suffix
```

Whole-path min-merge only picks the shortest complete path. It discards useful parts of the others. Cross-solution relinking tries:

```text
A[:i] + bridge(A_i -> B_j) + B[j:]
```

#### Algorithm

```python
for pid in pids:
    paths = all_valid_paths_for_pid(pid)
    prefix_cache = {path_id: compute_prefix_states(path) for path in paths}
    best = shortest(paths)

    for A, B in path_pairs(paths):
        for i in sample_prefix_positions(A):
            for j in sample_suffix_positions(B):
                if i + len(B[j:]) >= len(best):
                    continue
                residual = make_residual(prefix_cache[A][i], prefix_cache[B][j])
                max_bridge_len = len(best) - i - len(B[j:]) - 1
                bridge = solve_residual(residual, max_steps=max_bridge_len)
                candidate = A[:i] + bridge + B[j:]
                if verify(candidate) and len(candidate) < len(best):
                    best = candidate
```

#### Candidate pair filtering

Avoid all-pairs explosion by filtering:

- only pids with at least 2 distinct valid solutions;
- only paths with different length/trajectory;
- only prefix/suffix pairs where `i + suffix_len < best_len`;
- optional Hamming-distance or residual-V prefilter;
- prioritize pids where solution diversity has historically helped.

#### Why it is promising

The project already has evidence that different search configurations and models contribute unique pids. Cross-solution relinking extracts more value from that diversity than whole-path min-merge.

#### Expected impact

Medium to very high. If multiple paths for the same pid diverge and reconverge in useful ways, this can save many moves.

---

### 2.3 Solution-graph Dijkstra

#### One-liner

For each pid, build a graph whose nodes are intermediate states from known solutions and whose edges are original subpaths, BFS shortcuts, tail-resolve shortcuts, neural bridges, and macro repairs. Then run shortest path.

#### Motivation

Greedy replacement can miss combinations. Dijkstra over a solution graph finds the best global combination of available edges.

#### Graph definition

For each pid:

```text
Nodes:
  prefix states from every known valid solution
  prefix states produced by accepted bridge/tail/macro attempts

Edges:
  original path segment edges
  BFS-d6 replacement edges
  neural bridge edges
  tail-resolve suffix edges
  macro-insert repair edges
  same-face reduction edges
```

Run:

```text
shortest_path(initial_state_node, solved_state_node)
```

#### Practical version

Start as a DAG over positions in one path:

```text
nodes = positions 0..n
edge i->j = replacement path from S_i to S_j
```

Then extend to multiple paths with merged state nodes.

#### Expected impact

High if bridge compression works. This is the natural optimizer over all discovered local improvements.

---

### 2.4 AZ-v4-specific qshort and multi-teacher qshort

#### One-liner

Train a q-shortlister that is compatible with the actual teacher(s) being used, especially AZ v4, and then generalize it to preserve the union of useful actions across teachers.

#### Motivation

A qshort model trained on one teacher can prune the good children of another teacher. The qshort should be trained either for the exact teacher in production or as a high-recall union shortlister.

#### AZ-v4-specific qshort

Train:

```text
student Q_azv4(s, a) ≈ V_azv4(apply(s, a))
```

Validate recall under:

- identity states;
- NISS states;
- symmetry-rotated states;
- bucket 4-7 states, where AZ v4 has been useful;
- hard-tail states.

Then test:

```text
AZ v4 V only
AZ v4 + qshort_azv4
AZ v4 + qshort_azv4 + sym4
AZ v4 + qshort_azv4 + sym4 + NISS
```

#### Multi-teacher union qshort

Train positives as actions that any teacher or known solution wants:

```text
positive action if:
  a ∈ top-k_m05(s)
  OR a ∈ top-k_azv4(s)
  OR a ∈ top-k_mdd(s)
  OR a is first action on a known short path
  OR a appears in a successful bridge/tail repair
```

Loss:

```text
L = BCE(action_good)
  + λ_false_negative * missed_good_action_penalty
  + KL_to_each_teacher_distribution
```

Metric:

```text
recall@αB by teacher, bucket, symmetry, and NISS setting
```

#### Expected impact

High. It can reduce wall time while preserving diverse useful trajectories, enabling more symmetry, larger beam, and more hard-tail retries.

---

### 2.5 Search-time 2-ply / 3-ply lookahead reranker

#### One-liner

Instead of ranking beam candidates only by immediate `V(child)`, rerank a shortlist by shallow minimization:

```text
score_1(s) = min_a [1 + V(apply(s, a))]
score_2(s) = min_{a,b} [2 + V(apply(apply(s,a), b))]
```

#### Motivation

The beam fails when immediate V misranks a child whose value improves after one or two moves. A shallow lookahead can catch these cases without changing the model.

#### Algorithm

```text
1. qshort produces αB candidates.
2. teacher V picks top γB candidates, e.g. γ=2.
3. For each of γB candidates, expand one extra ply.
4. Rank by min child value + 1.
5. Keep final B.
```

#### Where to use

- hard-tail pids;
- states with high qshort entropy;
- steps where V gap between candidates is small;
- windows in bridge compression.

#### Expected impact

Medium. It is compute-expensive but directly attacks local ranking errors.

---

### 2.6 Frontier regret training

#### One-liner

Train on the solver’s actual mistakes: children kept by beam that led to long suffixes versus children later shown to lead to shorter paths.

#### Data sources

- beam frontier logs;
- paths improved by TailResolve;
- paths improved by MacroInsert;
- bridge compression wins;
- cross-solution relinking wins;
- pids where AZ v4 beats m05;
- pids where community/colleague path beats the project path, if allowed for training analysis.

#### Pairwise target

For a parent state `s`:

```text
child_good = child that led to shorter verified suffix
child_bad  = child selected by current V/qshort but led to longer suffix
```

Loss:

```text
L_pair = log(1 + exp(Q(s, good) - Q(s, bad)))
```

Assuming lower Q/value is better.

#### Why it matters

This trains exactly what beam needs: local ordering.

#### Expected impact

High if the same mistakes recur across pids or buckets.

---

### 2.7 Rotation selector for symmetry ensemble

#### One-liner

Instead of choosing K random symmetries, predict which rotations are likely to help a specific pid.

#### Motivation

Symmetry ensembling works, but random rotations waste compute. A selector can concentrate K solves on rotations that produce diverse/short paths.

#### Features

For each pid and candidate rotation:

```text
bucket
fallback length
current best length
V(initial rotated state)
qshort entropy
low-beam probe best V
low-beam probe V slope
stagnation signal
number of near-solved candidates
historical rotation win rate
```

#### Training target

```text
rotation_good = 1 if rotation produced a shorter path than identity or baseline
```

or regression target:

```text
predicted_saving = baseline_len - rotated_len
```

#### Expected impact

Medium-high in compute efficiency. It may get K=8-quality diversity for K=4 cost.

---

### 2.8 Macro mining from the project’s own path corpus

#### One-liner

Mine useful macros from successful path differences before scraping human algorithm databases.

#### Motivation

Human Megaminx algorithms require notation conversion and may not match the competition distribution. Your own solution corpus is already in the correct move notation and distribution.

#### Mining process

```text
For each pid with multiple paths A, B:
  find subpath pairs where B[p:q] replaces A[i:j] with fewer moves
  extract B[p:q] as macro candidate
  compute net permutation
  dedupe by net permutation
  score by support, average saving, and verified usefulness
```

#### Macro score

```text
score(macro) =
  support_count
  * average_saving
  * distinct_pid_count
  * verification_rate
  / macro_length_penalty
```

#### Deployment

Do not add all macros directly. Train Macro-Q to shortlist them.

---

### 2.9 Suffix specialist model

#### One-liner

Train a model only on near-tail states and use it for tail-resolve and suffix repair.

#### Motivation

The global V model must cover a huge distribution. Tail repair is narrower and has better labels from verified suffixes.

#### Training data

From each best path:

```text
last 20-80 states
state -> exact remaining suffix length along best path
state -> next move
siblings -> ranking negatives
```

Include:

- tail-resolve wins;
- SA wins;
- bridge compression windows;
- near-solved BFS labels.

#### Deployment

Use `V_suffix` only in:

- TailResolve;
- bridge compression;
- final beam steps;
- MacroInsert suffix repair.

#### Expected impact

Medium, with low risk.

---

### 2.10 Learned subgoal routing

#### One-liner

Learn intermediate landmarks from existing paths and route through them instead of solving directly to identity.

#### Data

Collect states along best paths. Cluster by:

- depth band;
- face/piece displacement features;
- V value;
- action suffix patterns;
- symmetry-canonical features.

Train:

```text
subgoal_id(s)
V_to_subgoal(s, z)
π_to_subgoal(a | s, z)
```

#### Deployment

For hard pids, select likely subgoal(s):

```text
solve initial -> z
solve z -> solved
```

#### Expected impact

Speculative but potentially large if direct-to-solved value is too noisy.

---

### 2.11 PHS-style path scoring

#### One-liner

Use cumulative policy cost plus heuristic value instead of only local V:

```text
score(node) = g(node) + w_h * V(node) + w_p * cumulative_neg_log_policy(path)
```

#### Motivation

A policy prior can preserve paths that are locally value-ambiguous but globally likely under successful solves.

#### Difference from simple policy penalty

A local child penalty only changes one step. Cumulative policy cost changes which partial paths survive over depth.

#### First sweep

```text
w_p in {0.01, 0.03, 0.05, 0.1, 0.2}
w_h fixed to current V scale
```

Use only after qshort compatibility is established.

---

### 2.12 Dynamic beam allocation by uncertainty

#### One-liner

Use a wide beam only when the model is uncertain, rather than shrinking monotonically or escalating only after failure.

#### Signals

```text
qshort entropy
V gap between best and median candidate
number of duplicates
stagnation recurrence
V slope over recent steps
policy disagreement
teacher disagreement
rotation disagreement
```

#### Rule sketch

```text
if uncertainty high:
    keep or expand beam
elif uncertainty low and V gap is large:
    shrink beam
if stagnation detected:
    split beam into diversity buckets
```

#### Expected impact

Medium. Past adaptive beam failed because it was quality-destructive; uncertainty-based allocation is different.

---

### 2.13 Structural diversity buckets

#### One-liner

Force the beam to preserve candidates from different structural classes.

#### Buckets

```text
last move face
last two face classes
Hamming band
V band
symmetry orbit feature
center/corner/edge displacement counts
predicted subgoal id
qshort top-action class
```

Then select:

```text
top B_k per bucket, merge, rerank
```

#### Expected impact

Medium. This is a low-code way to reduce beam collapse.

---

### 2.14 Per-pid test-time calibration

#### One-liner

For a hard pid, fine-tune only a tiny correction layer using that pid’s current best path and siblings.

#### What to tune

Do not tune the full network. Tune:

- scalar affine head;
- small residual head on frozen embeddings;
- layernorm scales;
- action-bias vector.

#### Training data

For that pid:

```text
states along known path -> suffix length
siblings -> hard negatives
tail-resolve/bridge successes -> positives
```

#### Expected impact

Speculative but suitable for top 20-100 hard pids where minutes per pid are acceptable.

---

### 2.15 Bridge library / case-based repair

#### One-liner

Build a library of previously successful residual repairs and query it for similar new residuals.

#### Library entries

```text
residual_features
bridge_path
saving
source_pid
window_len
success_count
```

#### Query features

```text
Hamming pattern
per-face displacement counts
V residual value
qshort top actions
generator-orbit counts
```

#### Use

Before expensive neural bridge solving, try nearest known bridge repairs.

---

## 3. Neural network architectures

### 3.0 Architecture survey verdict (2026-05) — every encoder hits the same saturation ceiling

**The §3 architecture survey is CLOSED.** Across the model families tried as the V/Q scorer,
none beats the ~6M ResMLP baseline, and the binding reason is the same every time: the
deep-depth **saturation** property (Rule 23 — a working V flattens near the puzzle diameter
~29 on deep random walks) is **problem-intrinsic** (it comes from the Bellman bootstrap's
optimistic-min bias + megaminx's combinatorics), **not** a representation/encoder limit.
Better fit (lower MSE) does NOT mean better saturation and does NOT mean better beam.

| § | architecture | outcome | one-line why |
|---|---|---|---|
| 3.1 | Representation-upgraded ResMLP (gated feature bundle) | REJECTED | bundle tripled mid-depth V variance -> beam collapse (`m_repr_v0`, [[repr-upgrade-bundle-rejected]]) |
| 3.2 | Bipartite Slot-Sticker Graph Transformer | REJECTED | recall == flat GT-Q (action nodes add ~0); far below ResMLP-Q; 30-50x slower |
| 3.3 | Permutation-matrix axial Transformer | not built | its lightweight form *is* the rejected 3.2 |
| 3.4 | Symmetry-equivariant V/Q | settled | consistency-loss beam-neutral (§13.4); symmetry pays at INFERENCE, not in the model |
| 3.5 | Relative-state / Bridge Transformer | REJECTED | residual-distance D learns window-len (v0) or collapses to saturation (Bellman); scorer was never the bridge bottleneck |
| 3.9 | Perceiver IO | REJECTED | drifts (V@d80-d40 gap frozen +14.5); the GT-V signature |
| 3.8 | Dodecahedral CNN / geometric GNN | REJECTED | gap-canary "saturates" was a FALSE POSITIVE -- V collapsed (d80 slid 60->21, under-predicting); can't beam-solve pid 0; ~100x inference cost |

**Three unifying findings:** (1) saturation near the diameter is the load-bearing property for
beam, and it is set by the Bellman bootstrap + the problem, not the encoder (state_inv, GT-V,
Perceiver, Bridge-Bellman all drift identically); (2) V's child-ordering is already near-optimal
at 6M (frontier-regret §13.2), so a better Q/policy reranker has ~0 solve headroom; (3) attention
encoders are 30-135x more expensive to train/infer and would deploy only via distillation back
into the ResMLP anyway.

**Conclusion: stop building global scorer architectures. The path to <70K is PURE-INFERENCE** —
sym-ensemble at full-1001, multi-seed beam, rescue/merge, curated macros (see §12 and
`to_do_shortlist.md`). Reusable probe harness retained: the two-stage pretrain->Bellman +
`V@d80-V@d40` saturation gate (`scripts/88`, `scripts/89`) is the cheap go/no-go for any
future encoder idea -- but add an ABSOLUTE-calibration check (d80 must sit near the diameter
~29, not just `gap <= 10`), since the Dodeca collapsed uniformly and the gap-only canary
false-positived "SATURATES".

---

### 3.1 Representation-upgraded ResMLP

#### Purpose

Low-risk ablation over the existing MLP family.

#### Problem with raw state

Raw input:

```text
state[i] = sticker in slot i
```

This does not explicitly expose:

- where each sticker currently is;
- slot/sticker geometry;
- face and local position;
- piece/orbit relationships;
- generator action context.

#### Recommended input

For each slot `i`:

```text
x_i = concat_or_sum(
  Emb_slot_id[i],
  Emb_sticker_id[state[i]],
  Emb_inverse_position[inv_state[i]],
  Emb_slot_face[i],
  Emb_slot_local_pos[i],
  Emb_sticker_home_face[state[i]],
  Emb_piece_or_orbit_id[state[i]],
  scalar_flags
)
```

Scalar flags can include:

```text
same_home_face = slot_face[i] == sticker_home_face[state[i]]
slot_is_solved = state[i] == i
```

#### Heads

```text
V scalar
Q24 action values
π24 action logits
uncertainty scalar
```

#### Losses

```text
MSE/Huber value loss
Q-distillation
child-ranking CE
symmetry consistency
policy CE from known paths
```

#### Expected impact

Medium. Cheap and worthwhile, but unlikely to be the biggest breakthrough alone.

---


### 3.2 Bipartite Slot-Sticker Graph Transformer / GraphGPS

#### Purpose

Best architecture-class fit for the permutation nature of the problem.

#### Graph

Node types:

```text
slot nodes:    120
sticker nodes: 120
action nodes:  24 optional
```

Edges:

```text
assignment: slot_i -- sticker_state[i]
slot geometry: slot_i -- slot_j if adjacent on Megaminx geometry
generator: slot_i -- slot_gen_a[i], edge_type=a
solved-piece: sticker_j -- sticker_k if same solved piece/orbit
action-affects-slot: action_a -- slot_i if action a moves slot i
```

#### Node features

Slot node:

```text
slot_id
face_id
local_pos
slot_orbit_id
solved-neighborhood id
```

Sticker node:

```text
sticker_id
home_slot
home_face
piece/orbit id
```

Action node:

```text
action_id
face_id
inverse_action_id
affected_slots summary
move_cost = 1
```

#### Architecture

Each layer can combine:

1. local graph message passing over assignment/geometry/generator edges;
2. global attention over slots/stickers/actions;
3. action-node updates using affected slots;
4. residual MLP blocks.

#### Heads

```text
V = pooled graph token -> scalar
Q(a) = action_node_a -> scalar
π(a) = action_node_a -> logit
```

#### Why this is promising

The model can score actions based on exactly the slots they affect, rather than inferring action effects from a flat vector.

#### Training priority

Train first as a **qshort / Q model**, not as full scalar V replacement.

#### Result (2026-05-25) — TESTED, REJECTED (no architecture win)

Built the full stack (bipartite graph features, model, distillation trainer, depth-stratified
recall eval, runbook) and trained on GCP. The bipartite Q-shortlister's child-ranking recall
**tracks the flat graph-transformer Q** (sticker-tokens + CLS readout, `scripts/75_train_gt_q.py`)
essentially identically at matched size (~3.5M) and budget — the action nodes add ~nothing. Both
stay far below the production ResMLP-Q (`m23_v3_az_v4_sym`, 12.4M) on the beam-relevant deep buckets
(e29 alpha=2: d60 0.47 vs 0.81, d80 0.54 vs 0.79) and improve too slowly to close it. Better MSE fit
did NOT translate to better recall (same lesson as the GraphTransformer-V failure). Combined with
the frontier-regret finding (V child-ordering already near-optimal -> ~0 solve-quality headroom for
any Q reranker) and the GT's ~30-50x inference cost (deploy-only-via-distill = back to the ResMLP-Q),
the architecture bet does not pay. Stopped at epoch 30/120. Full write-up: `EXPERIMENTS.md`
(2026-05-25), `bipartite_gt_q_runbook.md`, memory `bipartite-gt-q-shortlister`.

---

### 3.3 Permutation-matrix axial Transformer

#### Purpose

Expose the state as a true bijection.

#### Input

Build:

```text
M[i, j] = 1 if slot i contains sticker j
```

This explicitly exposes both:

```text
row i: what is in slot i
column j: where sticker j is
```

#### Lightweight alternative

Avoid materializing 120×120 by using two streams:

```text
slot_stream[i]    = features for sticker in slot i
sticker_stream[j] = features for current slot of sticker j
```

Use cross-attention between the two.

#### Architecture

```text
row attention
column attention
active assignment attention
global latent pooling
V/Q/π heads
```

#### Pretraining tasks

```text
masked assignment reconstruction
inverse permutation prediction
transition move prediction from (state_t, state_t+1)
first inverse move prediction
```

#### Expected impact

Medium-high. Easier than full graph model, but still exposes bijection better than flat MLP.

---

### 3.4 Symmetry-equivariant V/Q model

#### Purpose

Turn the working inference-time symmetry into a model constraint.

#### Constraints

For symmetry `R`:

```text
V(R s R^-1) = V(s)
Q(R s R^-1, conjugate_R(a)) = Q(s, a)
```

#### Implementations

##### A. Consistency loss

```text
L_sym_V = |V(s) - V(RsR^-1)|²
L_sym_Q = ||rotate_back(Q(RsR^-1)) - Q(s)||²
```

##### B. Group-pooled head

Evaluate a small number of rotations inside the model:

```text
V = mean_k V_backbone(R_k s R_k^-1)
Q = mean_k rotate_back(Q_backbone(R_k s R_k^-1))
```

##### C. Rotation selector

Use a small model to choose which rotations to use in expensive search.

#### Warning

Naive training-time rotation augmentation has already regressed in the current 6M regime. Treat symmetry as a **constraint or selector**, not just more samples.

---

### 3.5 Relative-state / Bridge Transformer

#### Purpose

Model path shortening between arbitrary states, not only distance-to-solved.

#### Input

For pair `(s_from, s_to)`:

```text
residual = compose(inverse(s_to), s_from)
```

Use residual as the main input, with forward and inverse residual representations.

#### Outputs

```text
D(s_from, s_to)
Q_bridge(s_from, s_to, a)
π_bridge(a | s_from, s_to)
```

#### Training data

From every path:

```text
for random i < j:
  input = (S_i, S_j) or residual(S_i, S_j)
  target distance = j - i
  target first action = path[i]
```

#### Losses

```text
Huber(D, j-i)
CE(first_action)
pairwise child ranking
Bridge Bellman consistency:
  D(s,t) ≈ 1 + min_a D(apply(s,a), t)
```

#### Deployment

Use in:

- neural bridge compression;
- cross-solution relinking;
- solution-graph edge generation;
- tail repair.

#### Expected impact

Very high if bridge replacement succeeds.

#### Result (2026-05-25) — TESTED, REJECTED (saturation ceiling + scorer is not the constraint)

Built the relative-distance model end-to-end and tested in three stages. Key identity: under
the repo convention `apply(s,a)` maps the residual `X=make_residual(s,t)` to `apply(X,a)`, so
`D(s,t)` is exactly distance-to-solved of the residual `X`, and Bridge-Bellman is standard
Bellman on `X`. So D is just a `ResMLPDistance` on residual states; no new architecture needed.
Files: `scripts/85_build_bridge_distance_data.py` (path-window residual dataset, balanced by
window length, pid-split), `scripts/86_train_bridge_distance.py` (regress-to-wlen + per-bucket
saturation table vs the production V), `scripts/87_train_bridge_bellman.py` (Bridge-Bellman via
`cayley.bellman._bellman_targets` with `wlen` as the per-state `clip_upper`), and a
`--scorer-checkpoint` flag added to `scripts/81_bridge_compression.py` for the deploy A/B.

- **v0 (regress-to-wlen).** D extends the reliable calibration horizon from V's ~20 to ~40
  (val d30->d70 slope: D +19 vs V +9; D ~= true to wlen 30) — the first model-side idea this
  cycle with a measured positive. BUT the target is wrong: `j-i` is window length (an upper
  bound), so D learns *window-length*, not *true distance*, and so carries no compressibility
  signal beyond what `wlen` already says. Generalization ceiling at val MAE ~10 (train MAE ->0;
  weight decay does not move it).
- **v1 (Bridge-Bellman true-distance).** Warm-started from v0 and refined with
  `clip(1 + min_a D(child), 0, wlen)`. It COLLAPSES the v0 curve toward V's saturated plateau
  (d30->d70 slope 19 -> 15 -> 13, heading for V's 9) via the optimistic-min bias `bellman.py`
  itself documents (the m05/m17/m26/m27 ceiling). `clip_upper(wlen)` only caps from above and
  cannot resist the downward collapse. Worse than v0. The compressibility signal (pull D below
  wlen on compressible windows) and the collapse bias (pull D below wlen everywhere) are the
  same `min` op — inseparable without ground-truth deep distances we do not have.
- **Deployment A/B (the deploy-level gate).** Swapped v0-D in as the bridge *scorer* (solver
  held = V, same pids/windows/budget) on merge_v12 (77,214) top-15 long pids, windows 20-40.
  Arm V (scorer=V): 5 wins / **7 moves**. Arm D (scorer=v0-D): 4 wins / **6 moves** — noise-level
  WORSE, and D MISSED pid 342 that V caught. Mechanism: score = `wlen - scorer`; V's saturation
  (under-prediction) inflates predicted-save, making it aggressively propose marginal windows,
  and every real win here is a 1-2-move sliver, so aggression helps. D's accuracy makes it
  conservative and it skips the marginal wins. Every winning window was 1-2 moves of slack,
  identical across scorers -> **the binding constraint is path near-optimality, not scorer
  calibration** (confirms [[bridge-compression-findings]]).

Net: the Rule-23 saturation ceiling re-asserts itself in residual/bridge space, and the bridge
scorer was never the bottleneck. §3.5 closed. Code retained (inert/default-off for `81`).
Full write-up: EXPERIMENTS.md (2026-05-25), memory [[bridge-residual-distance-rejected]].

---

### 3.6 Multi-teacher qshort architecture

#### Purpose

Preserve useful actions from multiple teacher landscapes.

#### Input

Same state representation as upgraded MLP/graph/transformer.

#### Output

```text
Q24 or logits24
```

#### Target

Multi-label action set:

```text
good_actions(s) = union(
  top-k_teacher1,
  top-k_teacher2,
  known_path_action,
  bridge_success_action,
  tail_success_action
)
```

#### Loss

```text
BCE(good_actions)
+ false_negative_penalty
+ KL to teacher distributions
+ ranking loss
```

#### Metric

Not MSE. Use:

```text
recall@αB per teacher
recall@αB per bucket
recall@αB under sym rotations
full stratified solve quality
```

---

### 3.7 Macro-Q model

#### Purpose

Learn when to expand primitive actions and macro actions.

#### Action set

```text
A = 24 primitive moves + M macro moves
```

#### Action metadata

Each action needs:

```text
action_id
action_type: primitive/macro
real_move_cost
affected_slots_mask
net_permutation
macro_length
macro_family
symmetry_orbit_id
historical_success_rate
```

#### Two architecture options

##### Fixed output head

```text
Q(s) -> R^(24+M)
```

Good for small `M`.

##### Factorized state-action model

```text
h_s = Encoder(state)
h_a = ActionEncoder(action_metadata)
Q(s,a) = MLP([h_s, h_a, h_s * h_a])
```

Better for large macro libraries.

#### Loss

```text
Q_target(s,a) = V_teacher(apply_action(s,a)) + cost(a)
```

Use class-balanced sampling so primitives do not dominate.

---

### 3.8 Dodecahedral CNN / geometric convolution

#### Purpose

Cheap geometric inductive bias.

#### Warning

A normal 2D CNN over `12 faces × 10 stickers` is not enough. It creates fake adjacencies and misses true dodecahedron boundaries.

#### Representation

For each slot:

```text
same-face cyclic neighbors
adjacent-face boundary neighbors
same-piece neighbors
generator-image neighbors
```

Layer:

```text
h_i' = MLP(
  h_i,
  mean_same_face(h_j),
  mean_adjacent_face(h_j),
  mean_generator_neighbors(h_j),
  mean_piece_neighbors(h_j)
)
```

This is essentially a local GNN.

#### Best use

- suffix specialist;
- macro applicability classifier;
- local repair scorer.

#### Result (2026-05-27) — TESTED, REJECTED (collapse masquerading as saturation; can't beam)

Built `src/megaminx/dodeca_cnn.py` (the doc's spec: per-slot message passing, each slot
mean-aggregates over same-face / generator-edge / same-piece / stride-2 neighbours from the
relation matrix, MLP-mixes, residual; NO attention -> cheap, no OOM) + `scripts/89_dodeca_v_probe.py`
(same two-stage pretrain->Bellman + saturation-gate harness as the Perceiver). Trained on GCP L4
(3.29M, d=256/4-layer).

**The instructive part:** Dodeca is the ONLY encoder whose gap-canary printed "SATURATES"
(gap +4.2 at bel-e19, vs the Perceiver's +14.7 drift) -- and it was a **FALSE POSITIVE**. The
whole V curve slid down monotonically and never converged: d80 = 60.2 -> 32.1 -> 28.9 -> 23.4
-> 20.7 across the canaries, ending a heavily **compressed, under-predicting** V (range 0->21
where a healthy V spans 0->~29; d60/d80 only ~3 apart -> no depth resolution). That is the
optimistic-min Bellman **collapse** (the §3.5 family), and the gap stayed small only because
d40 and d80 collapsed *together*. **A beam bench confirmed it**: best.pt (e19, the least-collapsed
checkpoint) at beam 65k did not solve even pid 0 after 91 min (a healthy V: seconds), GPU pinned
at 100% / 21 GB -- the GNN message passing is ~100x the ResMLP's inference cost AND the collapsed
V can't navigate beam. Killed.

**Two lessons:** (1) the saturation gate needs an **absolute-calibration** check (d80 must sit
*near* the diameter, ~29, not just `gap <= 10`) -- Dodeca is the case that proves the gap-mean
alone is insufficient (Rule 21 / repr-bundle, now with a concrete false-positive). (2) Local
geometric message passing did NOT escape the wall; it found a *different* failure (collapse) than
the attention encoders' drift. Loader wired into `cayley.search` (`DodecaCNNV` branch), inert.
Full write-up: EXPERIMENTS.md (2026-05-27), memory [[dodeca-cnn-rejected]].

---

### 3.9 Perceiver IO

#### Purpose

Flexible architecture for multi-output V/Q/π with latent bottleneck.

#### Input tokens

```text
120 slot tokens
120 inverse-position sticker tokens
24 optional action tokens
optional macro action tokens
```

#### Latents

```text
64-256 learned latent tokens
```

#### Output queries

```text
V query
Q action queries
policy query
uncertainty query
macro action queries
```

#### Expected use

Good compromise if graph transformer is too heavy.

#### Result (2026-05-26) — TESTED, REJECTED (drifts; same saturation wall as GT-V)

Built a Perceiver V (`src/megaminx/perceiver_v.py`: 64-128 learned latents cross-attend to
the 240 slot+sticker tokens reusing `bipartite_features.pt`, self-attend, mean-pool -> scalar
V; NO attention mask anywhere, so it uses flash SDPA and dodges the rule-22/27 SDPA-mask
blow-up that hit the bipartite GT) and a two-stage saturation probe
(`scripts/88_perceiver_v_probe.py`: walk-depth MSE pretrain -> Bellman refine via
`cayley.bellman._bellman_targets` with V0/d1 anchors). Gate = the deep-depth saturation
canary `V@d80 - V@d40` on random walks (rule 23): a working V saturates near the diameter
(~29); the rejected GraphTransformer-V drifted (V@d80=42).

**Result (2.76M Perceiver, d=256/64-latent/3-layer, on GCP L4):** the gap is FROZEN across
40 Bellman epochs -- pretrain +13.3, bel-e19 +14.7, bel-e39 +14.5 -- with V@d80 stuck at
45-53 (diameter ~29). Bellman translates the whole curve DOWN uniformly toward the low-end
anchors (V0~=0.3, V(d1)~=0.9, d30~=24 all calibrate fine) but CANNOT bend the deep end into
saturation -- the exact GT-V signature (calibrates low, drifts high, gap won't close). Killed
at e40 (verdict certain; 30 more epochs only confirm a frozen gap). So the Perceiver encoder
hits the same problem-intrinsic saturation ceiling as ResMLP-state_inv, GraphTransformer-V,
and the §3.5 Bridge-Bellman -- representation power is not the bottleneck.

**Ops notes** (reusable): d=256/128-latent/4-layer at batch 4096 OOMs an L4 24GB (training
activations accumulate across self-blocks) -> use batch <=2048 + fewer latents/layers +
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The Bellman target's 24x child forwards
through an attention model are expensive (~135 s/epoch on L4, like the GT) and the target-net
forward thrashes at large `target_net_chunk` (557s/epoch at 16384, caught locally) -> keep
`target_net_chunk` small (4096) and run target forwards under bf16 autocast. Code retained,
inert-by-default. Full write-up: EXPERIMENTS.md (2026-05-26), memory [[perceiver-v-rejected]].

---

### 3.10 Mamba / Hyena / RWKV-style sequence models

#### Best use

Not as static state V replacement. Use for:

- move-sequence modeling;
- policy priors;
- autoregressive repair generation;
- macro proposal generation;
- state + path prefix -> next move.

#### Required representation

Mamba and similar sequence models require a meaningful token order. Options:

```text
face-major order with consistent winding
generator-orbit order
BFS order on slot adjacency graph
piece/orbit grouped order
multiple orders with pooling
```

#### Training

```text
CE(next_move | state, path_prefix)
DPO/preference: prefer shorter path sequence over longer one
masked move-span reconstruction
```

#### Expected impact

Medium. Useful for policy/macro generation, less likely as main V scorer.

---

### 3.11 Time-series foundation model ideas: TimesFM, Chronos, MOIRAI, Lag-Llama

#### Verdict

Use architecture ideas, not pretrained weights.

The pretrained weights are trained for continuous temporal forecasting. Megaminx is a discrete finite-group permutation problem. Direct transfer is unlikely to help.

#### Useful adaptations

##### Chronos-like tokenization

```text
tokens = sticker IDs or move IDs
objective = masked token reconstruction / next inverse move prediction
```

##### TimesFM / PatchTST-like patching

Patch by puzzle structure, not arbitrary time windows:

```text
patch = face
patch = piece/orbit group
patch = generator-affected slot set
patch = path segment
```

##### MOIRAI-like masked modeling

```text
mask slots, predict stickers
mask inverse positions, predict slots
mask moves in path, predict move tokens
```

##### Lag-Llama-like autoregression

Apply only to trajectories:

```text
state_t, move_t, state_t+1, move_t+1, ...
```

#### Recommendation

Do not start here unless using them as generic sequence-model baselines. Puzzle-specific graph/bridge/action models are more promising.

---

### 3.12 Tabular meta-models

#### Purpose

Not for raw state scoring. Use for compute allocation and solver selection.

#### Features

```text
pid bucket
current best length
fallback length
V_m05(initial)
V_azv4(initial)
qshort entropy
first-pass solved/fail
symmetry win count
NISS win/loss
tail profile
previous rescue attempts
```

#### Outputs

```text
which solver to run
which rotations to choose
which pids deserve K=8
whether TailResolve/SA is likely to help
expected move saving per GPU-hour
```

#### Expected impact

Medium via better compute allocation.

---

## 4. Losses and training regimes

---

### 4.1 Bellman value loss

Current value-style objective:

```text
target(s) = 1 + min_a V_target(apply(s,a))
```

Use Huber rather than MSE if outliers dominate:

```text
L_value = Huber(V(s), target(s))
```

But do not rely on this alone. Prior experiments indicate scalar Bellman refinements saturate.

---

### 4.2 Child-ranking loss

For each state `s`, evaluate all children:

```text
c_a = apply(s, a)
```

Create a target distribution:

```text
P_target(a) ∝ exp(-target_child_distance(a) / τ)
```

Train:

```text
L_rank = CE(P_target, softmax(-Q(s,a)))
```

or pairwise:

```text
L_pair = log(1 + exp(Q_good - Q_bad))
```

This should be a core loss for any Q model.

---

### 4.3 Q-distillation loss

For teacher V:

```text
Q_target(s,a) = V_teacher(apply(s,a))
```

Loss:

```text
L_Q = Huber(Q_student(s,a), Q_target(s,a))
```

For qshort, combine with recall-focused losses, because preserving top actions matters more than fitting all actions.

---

### 4.4 Union-qshort loss

Multi-label target:

```text
y_a = 1 if action a should be preserved
```

Loss:

```text
L_union = weighted_BCE(logits, y)
```

Use higher weight on false negatives:

```text
weight_positive >> weight_negative
```

Metric:

```text
recall@αB, not average BCE
```

---

### 4.5 Policy loss from verified paths

For state `s_i` on a known path:

```text
π_target = move_i
```

Loss:

```text
L_policy = CE(π(s_i), move_i)
```

Weight shorter/better paths more:

```text
w_i = 1 / (1 + remaining_suffix_len)
```

or:

```text
w_path = exp(-path_len / T)
```

---

### 4.6 Symmetry consistency loss

For sampled symmetry `R`:

```text
s_R = R s R^-1
```

Value:

```text
L_sym_V = |V(s) - V(s_R)|²
```

Q:

```text
Q_R_back = rotate_actions_back(Q(s_R))
L_sym_Q = ||Q(s) - Q_R_back||²
```

This is better than naive augmentation because it enforces consistency without simply multiplying noisy samples.

---

### 4.7 Bridge Bellman loss

For pair `(s,t)`:

```text
D(s,t) ≈ 1 + min_a D(apply(s,a), t)
```

Loss:

```text
L_bridge = Huber(D(s,t), known_window_len)
         + λ * Huber(D(s,t), 1 + min_a D(apply(s,a), t))
```

Policy:

```text
L_bridge_policy = CE(π_bridge(s,t), first_bridge_move)
```

---

### 4.8 Macro-Q loss

For action `a`, including macros:

```text
target(s,a) = V_teacher(apply_action(s,a)) + cost(a)
```

Loss:

```text
L_macro_Q = Huber(Q(s,a), target(s,a))
```

Add ranking:

```text
macro should beat primitive only if target_macro < target_primitive
```

Use class-balanced action sampling.

---

### 4.9 Expert iteration / AlphaZero-style beam training

Use the current solver as policy improvement:

```text
1. Solve with current model + search.
2. Improve paths with tail/bridge/SA/relinking.
3. Train V/Q/π on improved paths and frontier mistakes.
4. Repeat.
```

This adapts AlphaZero-style policy improvement to deterministic shortest-path search without needing full MCTS.

---

### 4.10 Frontier replay / DAgger

Log real frontier states from search:

```text
kept states
pruned states
states leading to long suffixes
states later shown useful
states from failed pids
```

Train on this distribution with:

- value suffix labels;
- ranking labels;
- qshort positives;
- policy labels.

This attacks the random-walk-vs-beam distribution mismatch.

---

### 4.11 PHS-style path-level scoring loss

Train policy and value so that search score works:

```text
score(node) = g(node) + w_h V(node) + w_p cumulative_neg_log_policy(path)
```

Tune `w_h`, `w_p` by stratified solve quality.

---

### 4.12 GFlowNet / trajectory-balance for diversity

Use this as a path/macro generator, not necessarily the main solver.

Reward:

```text
R(path) = exp(-len(path) / T) if path solves else 0
```

Train to sample diverse short paths. Mine macros and alternative trajectories from generated paths.

---

### 4.13 DPO / preference loss over paths

For same pid, if path A is shorter than path B:

```text
A preferred to B
```

Train sequence/policy model with preference objective:

```text
log P(A | state) > log P(B | state)
```

Useful for move-sequence models and repair proposal models.

---

## 5. Feature preparation for models

---

### 5.1 Forward and inverse permutation

Always prepare both:

```python
state[i] = sticker in slot i
inv_state[j] = slot containing sticker j
```

This is likely the cheapest important feature upgrade.

---

### 5.2 Slot metadata

For each slot `i`:

```text
slot_id
face_id
local_position_on_face
face_boundary_group
adjacent_slot_ids
same-face neighbor ids
generator_images: gen_a[i] for each action a
generator_preimages: gen_a^-1[i]
slot_orbit_id under symmetry group
```

---

### 5.3 Sticker metadata

For each sticker `j`:

```text
sticker_id
home_slot = j in solved state
home_face
home_local_position
piece_id or orbit_id, if derivable
sticker_orbit_id under symmetry group
```

If true Megaminx piece decomposition is difficult, start with symmetry/generator orbits as proxy piece/orbit metadata.

---

### 5.4 Action metadata

For each primitive move `a`:

```text
action_id
inverse_action_id
face_id
affected_slots_mask
permutation gen_a
cycle structure
order = 5
cost = 1
conjugation map under symmetries
```

For macros:

```text
macro_id
word_names
net_permutation
length/cost
affected_slots_mask
macro_family
source: mined / human / SA / bridge
historical_success_rate
symmetry_orbit_id
```

---

### 5.5 Graph construction features

Build and cache:

```text
slot adjacency graph
slot generator graph for each action
assignment edge list from current state
sticker solved-neighborhood graph
action-slot incidence matrix
```

For each state, only assignment edges change. Static graphs can be cached.

---

### 5.6 Permutation-matrix features

For axial/2D models:

```text
M[i,j] = 1 if state[i] = j
```

But for efficiency, store sparse coordinates:

```text
rows = arange(120)
cols = state
```

Also prepare inverse coordinates:

```text
rows_inv = arange(120)
cols_inv = inv_state
```

---

### 5.7 Residual bridge features

For bridge models:

```text
residual = compose(inverse(target_state), source_state)
residual_inv = inverse(residual)
```

Add:

```text
window_len_original
source_depth_along_path
target_depth_along_path
same_path_or_cross_path flag
```

Use residual as the main state input.

---

### 5.8 Symmetry features

Cache:

```text
rotations array
inverse rotations
action conjugation maps
canonical representative under small subset of rotations
rotation id embeddings
historical win rates by rotation
```

Use for:

- equivariance loss;
- qshort validation under rotations;
- rotation selector;
- macro symmetry expansion.

---

### 5.9 Search-derived features

For meta-models and compute allocation:

```text
pid bucket
current best path length
fallback length
V_m05(initial)
V_azv4(initial)
qshort entropy
policy entropy
teacher disagreement
first-pass solved/fail
NISS win/loss
symmetry win/loss
beam stagnation count
V slope over steps
tail-resolve improvement history
macro-insert improvement history
```

---

## 6. Pretraining tasks

---

### 6.1 Masked sticker reconstruction

Mask random slots and predict missing stickers.

```text
input: partial permutation with masked stickers
target: original stickers
```

Add a constraint or loss to avoid duplicate sticker predictions.

---

### 6.2 Inverse permutation reconstruction

Given forward state, predict inverse positions.

This forces the model to learn bijection structure.

---

### 6.3 Move transition prediction

Given `(s_t, s_{t+1})`, predict move `a`.

Good for action-aware encoders.

---

### 6.4 First inverse move prediction

For random walks from solved, train model to predict the first move of a shortest or known inverse path.

This is policy pretraining.

---

### 6.5 Path-window bridge pretraining

From known paths:

```text
input: residual(S_i, S_j)
target: j-i and first move path[i]
```

This directly supports bridge compression.

---

### 6.6 Teacher-ensemble qdistill pretraining

Use several teachers and known paths to define good action sets.

```text
output: Q/action logits
objective: high recall of useful actions
```

---

### 6.7 Denoising random walks

Apply k random moves, predict:

- inverse first move;
- k;
- all 24 child values;
- path back to solved for small k.

---

## 7. Pretrained model families: what to use and what not to use

---

### 7.1 Puzzle-specific pretrained models

Most useful if available. Use them for:

- training recipe transfer;
- qshort architecture transfer;
- beam implementation ideas;
- teacher initialization if state/action dimensions can be adapted.

Direct transfer from cube to Megaminx is limited by different state size and generator structure.

---

### 7.2 Time-series foundation models

Models like TimesFM, Chronos, MOIRAI, and Lag-Llama should not be used directly as pretrained state scorers.

Use ideas only:

```text
Chronos -> discrete tokenization / cross-entropy over tokens
TimesFM/PatchTST -> structural patching
MOIRAI -> masked multivariate modeling
Lag-Llama -> trajectory autoregression
```

---

### 7.3 Image CNN/ViT pretrained models

Low priority. Sticker IDs are categorical permutation entries, not visual pixels.

If using ViT/CNN, train from scratch with puzzle-specific representation.

---

### 7.4 Language models

Potentially useful for move-sequence modeling or macro generation, not for raw state V.

Train/finetune on:

```text
state tokens + solution move tokens
path preference pairs
macro strings
```

---

### 7.5 Tabular pretrained models

Use only for meta-selection and compute allocation.

---

## 8. Experiment queue

---

### Experiment 1: Neural bridge compression v0

**Goal:** verify that arbitrary middle segments can be shortened.

```text
Input: top 50 longest current-best pids
Windows: 20, 30, 40
Solver: current best model at modest beam
Output: verified shorter paths
Acceptance: any net positive; >100 moves saved is strong
```

Implementation tasks:

1. Implement residual composition and unit tests.
2. Compute prefix states for paths.
3. Try selected windows.
4. Verify replacements.
5. Log successful residuals for bridge training.

---

### Experiment 2: Cross-solution relinking v0

**Goal:** exploit multiple paths per pid.

```text
Input: pids with >=2 valid paths
Candidate: A[:i] + bridge(A_i, B_j) + B[j:]
Acceptance: verified shorter than current best
```

Start with top 50 long pids.

---

### Experiment 3: AZ-v4 qshort

**Goal:** make AZ v4 full-stack cheaper and non-regressing.

```text
Train Q_azv4(s,a) ≈ V_azv4(apply(s,a))
Validate recall under identity, NISS, sym rotations
Run strat-5 with AZ v4 + qshort_azv4 + sym4
```

---

### Experiment 4: Multi-teacher qshort

**Goal:** preserve useful actions across m05, AZ v4, and path data.

```text
Train union action target
Validate recall per teacher
Run hard-tail pids
```

---

### Experiment 5: Bridge model pretraining

**Goal:** train `D(s,t)` and `Q_bridge(s,t,a)` from path windows.

```text
Data: all verified paths
Samples: random windows and cross-path windows
Loss: distance + first-action CE + bridge Bellman
```

Deploy in bridge compression.

---

### Experiment 6: Suffix specialist

**Goal:** improve TailResolve and final-step search.

```text
Train on last 20-80 states of best paths
Deploy only in tail-resolve
Compare acceptance rate and saved moves
```

---

### Experiment 7: Graph-Q shortlister

**Goal:** test graph/bipartite architecture as action shortlister.

```text
Build slot-sticker-action graph
Train Q24 on multi-teacher targets
Compare recall and solve quality to m23/m23_v2
```

---

### Experiment 8: Macro mining + Macro-Q pilot

**Goal:** validate learned macro action expansion.

```text
Mine 20-50 macros from own path corpus
Train Q over 24+M actions
Run hard-tail pids
```

Acceptance requires actual macro actions appearing in verified shorter final paths.

---

### Experiment 9: Rotation selector

**Goal:** make sym-ensemble smarter.

```text
Run cheap probes for many rotations
Train selector to predict useful rotations
Compare random K=4 vs selected K=4
```

---

### Experiment 10: Solution-graph Dijkstra

**Goal:** globally combine all discovered path improvements.

```text
Nodes: prefix states
Edges: original, BFS, neural bridge, tail-resolve, macro repair
Run shortest path
```

Start with single-path DAG, then multi-path graph.

---

## 9. Recommended priority order

| Priority | Item | Why |
|---:|---|---|
| 1 | Neural bridge compression | Direct non-regressing path shortening; uses existing solver |
| 2 | Cross-solution neural relinking | Extracts more from existing path diversity |
| 3 | AZ-v4-specific qshort | Fixes current teacher-student mismatch and reduces full-stack cost |
| 4 | Multi-teacher qshort | Preserves ensemble diversity inside one search |
| 5 | Bridge model | Turns successful path repair into a trained capability |
| 6 | Suffix specialist | Narrow distribution, good labels, low risk |
| 7 | Graph-Q shortlister | Best architectural fit for permutation + action graph |
| 8 | Macro mining + Macro-Q | High ceiling if action-space expansion is gated correctly |
| 9 | Rotation selector | Converts symmetry from random diversity to targeted diversity |
| 10 | Solution-graph Dijkstra | Global optimizer over all discovered improvements |

---

## 10. Implementation notes and pitfalls

### 10.1 Do not judge by training MSE

The repo history already shows that lower MSE can produce worse search. Evaluate by:

```text
stratified solve quality
unique wins in merge
recall@αB for qshort
saved moves after verification
bucket-specific behavior
```

### 10.2 Separate replacement models from diversity contributors

A model can be valuable even if it does not dominate current best everywhere. Track:

```text
standalone score
stratified mean
unique pids improved
bucket where it wins
compatibility with qshort/sym/NISS
```

### 10.3 Verify every path replacement

Any bridge, macro, relink, or tail replacement must be verified from original initial state to solved.

### 10.4 Avoid raw pretrained time-series weights as a main path

Use their tokenization/masking ideas only. The data distribution is wrong for direct transfer.

### 10.5 Macro actions must include cost

A macro that costs 8 real moves should not be treated like one primitive move. Always score:

```text
V(result_state) + cost(macro)
```

or a calibrated equivalent.

### 10.6 Qshort must match the teacher

Do not pair a qshort student with a teacher whose value landscape it did not learn, unless recall is validated.

### 10.7 Symmetry is better as inference, constraint, or selector than naive augmentation

Naive rotation augmentation has already regressed. Prefer:

```text
symmetry consistency loss
rotation-aware qshort validation
rotation selector
inference-time sym ensemble
```

---

## 11. Minimal feature bundle to standardize immediately

For every future state model, prepare a common feature object:

```python
features = {
    "state": state,                       # forward permutation, shape (120,)
    "inv_state": inv_state,               # inverse permutation, shape (120,)
    "slot_id": slot_id,                   # 0..119
    "slot_face": slot_face,
    "slot_local_pos": slot_local_pos,
    "sticker_home_face": sticker_home_face[state],
    "sticker_home_slot": state,
    "is_solved_slot": (state == slot_id),
    "generator_images": generator_images, # cached static table [24,120]
}
```

For action models:

```python
action_features = {
    "action_id": action_id,
    "inverse_action_id": inverse_action_id,
    "face_id": action_face_id,
    "affected_slots_mask": affected_slots_mask,
    "cost": action_cost,
}
```

For bridge models:

```python
bridge_features = {
    "source_state": s_from,
    "target_state": s_to,
    "residual": residual(s_from, s_to),
    "residual_inv": inverse(residual),
    "original_window_len": j - i,
}
```

This standardization will make architecture comparisons cleaner.

---

## 12. Final recommendation

The best next push should combine two tracks:

### Track A: path-shortening without new global model

1. Implement neural bridge compression.
2. Implement cross-solution neural relinking.
3. Add solution-graph Dijkstra over accepted edges.
4. Mine successful bridge residuals and macros.

This has the highest chance of reducing existing found paths immediately.

### Track B: action-centric neural modeling

1. Train AZ-v4-specific qshort.
2. Train multi-teacher union qshort.
3. Train bridge model from path windows.
4. Train graph-Q or permutation-matrix Q model.
5. Train Macro-Q with mined macros.

This attacks the underlying search weakness: useful actions are being pruned or never considered.

The most promising architecture overall is:

> **Bipartite slot-sticker-action graph encoder with Q/policy/value heads, trained with multi-teacher qshort, child-ranking, symmetry consistency, and frontier regret losses.**

The most promising path-shortening mechanism overall is:

> **Neural bridge compression, upgraded to cross-solution relinking and solution-graph Dijkstra.**

Together, these are more likely to create a step-change than another scalar value-model ablation.

---

## 13. Selected next batch (2026-05-24): concrete, evidence-grounded plans

This section turns six ideas chosen for the next push into implementation-ready specs:
**PHS path scoring**, **PHS-style cumulative scoring**, **frontier regret training**,
**pairwise on real frontier mistakes**, **macro mining from own corpus**, and
**symmetry consistency loss**. They collapse into **four themes** (PHS path scoring +
cumulative scoring are one mechanism at two granularities; frontier regret + pairwise on
real mistakes are one training signal).

**Why this batch, given the track record.** The last several sessions were training-side
V experiments that all regressed (state_inv encoding, repr-upgrade bundle, GT-V, 11.8M
trunk), while every actual score win since 80K came from **inference-side** mechanisms
(sym-ensemble, bridge compression, hard-tail rescue, merges). Two lessons bind every spec
below:

1. **6M cluster ceiling** — re-encoded/bigger V trunks regress; the *only* honest gate is
   strat-51 with the **production** recipe (`--sym-ensemble 4 --beams 16384,65536
   --max-steps 60,150 --niss --bf16`), not a 10-pid bench, not training loss (Rule 21).
2. **V variance at d≈20 is the beam-quality proxy**, not saturation-mean. m_dd_v0 sits at
   d=20 std ≈ 1.73; the rejected repr bundle looked fine on saturation-mean but tripled
   that variance to ≈5.3 and collapsed beam (2/20). `v_canary.py` prints it — weight it.

Inference-side ideas (13.1, 13.3) are gated by **min-merge** (any verified net win counts,
cannot regress the submission). Training-side ideas (13.2, 13.4) are gated by the binding
strat-51 gate above **plus** d=20 std ≤ ~2.5.

---

### 13.1 PHS cumulative path scoring  [LEAD — inference-side, testable now]

Covers *PHS path scoring* + *PHS-style cumulative scoring* (doc §2.11, §4.11).

#### One-liner

Score a beam candidate by its value plus the **accumulated** policy cost of the entire
path that produced it, threaded through each surviving beam state:

```text
score(child) = V(child) + w_p * sum_{t<=d} ( -log pi(a_t | s_t) )
```

not by a single immediate step.

#### Why this is different from the local penalty that already regressed

The repo already has `--policy-model` / `--lambda-policy` (`03_solve.py:205`,
`beam_search.py:519`, `beam_search_qshort.py:155`). That path adds
`+lambda * (-log pi(a | parent))` to one candidate's V — **memoryless**. AZ v4 V + that
local penalty at λ=0.05 regressed **−156 moves** (HANDOFF, strat-5). The doc flagged the
distinction explicitly (§2.11): *"a local child penalty only changes one step; cumulative
policy cost changes which partial paths survive over depth."* A path the policy
consistently liked accrues low cumulative cost and survives even when a single `V(child)`
is locally ambiguous — the untested axis.

#### Why now (no training needed)

AZ v4 already ships a usable π head: `models/m_az_v4_pi_only.pt` (output_dim=24, top-1
≈28.5%, well above 4.2% random). `m_pi_v2` is an alternative prior. Both load with the
existing `--policy-model` plumbing.

#### Implementation (bounded change to the beam loop)

- Thread a per-beam accumulator `cum_neg_log_pi` (one float per live beam state) through
  `_do_greedy_step` and the solve loop in `beam_search.py`, mirrored in
  `beam_search_qshort.py`. Initialize to zeros for the root beam.
- Per candidate, reuse the **exact** `log_pi_per_cand` already computed at
  `beam_search.py:524`: `cum_cand = cum_neg_log_pi[parent_of_idx1] + (-log_pi_per_cand)`.
  Then `score = value + w_p * cum_cand` (keep the existing macro `cost_penalty`).
- After top-B selection, carry `cum_cand[chosen_local]` forward as the next step's
  `cum_neg_log_pi`. (One policy forward per step on the parents — same cost the local
  penalty already pays; negligible vs the nB child-V forwards.)
- New mode flag `--phs-cumulative` (reuse `--lambda-policy` as `w_p`, or add `--w-policy`);
  keep the local-penalty mode intact so the two don't collide.
- Unit test: on a tiny beam, assert `cum_neg_log_pi[i]` equals Σ(−log π) summed along the
  parent-backpointer path reconstructed for state `i`.

#### Acceptance gate

Inference-side. Sweep `w_p ∈ {0.01, 0.03, 0.05, 0.1, 0.2}` on strat-51 (production recipe)
vs the AZ-v4-V-only baseline on the same pids. If best `w_p` ties or beats baseline, run a
top-N long-pid pass and **min-merge** — cannot regress the submission. Expected: small but
free; the value is in long-tail pids where V's local ranking is noisiest.

#### Risk / cost

Low. Worst case it behaves like the local penalty and never beats baseline at any `w_p`,
costing only the strat-51 sweep. No checkpoints at risk.

---

### 13.2 Frontier regret / pairwise on real beam mistakes

Covers *frontier regret training* + *pairwise on real frontier mistakes* (doc §2.6, §4.2).

#### One-liner

Train an auxiliary pairwise loss on triples `(parent s, good_child, bad_child)` harvested
from **real** beam runs, where `good_child` lies on the eventually-verified shorter path
and `bad_child` is the one the current V/beam ranked higher but that led to a longer (or
no) solution.

#### Why this is different from the m_rank_v0 tie and from existing frontier replay

- **vs m_rank_v0 (neutral):** m_rank_v0 applied child-rank CE to *random-walk-sampled*
  children ordered by Bellman target — and a working V already orders RW children fine, so
  it tied (mean 89.9 vs 89.4). The new lever is the **data**: pairs where V's ordering
  *actually failed during beam*. That targeted negative set is exactly what m_rank_v0
  lacked.
- **vs existing frontier replay:** `frontier_states.pt` (used at `frontier_fraction: 0.25`
  in the m_dd_v0 recipe) mixes frontier *states* with on-the-fly Bellman *value* targets —
  a better state distribution, but **no ranking/regret signal**. `42_log_frontier_states.py`
  logs states only.

#### Why it attacks the real problem

The RW-vs-beam distribution mismatch is repeatedly flagged (HANDOFF key-learning #3;
to-do B9). Beam expands 23 off-path children per candidate; its failures are local
ordering errors on beam-frontier states — precisely the regret signal, which scalar
Bellman refinement (saturated) cannot capture.

#### Implementation

- **Data pipeline (the real work):** extend `42_log_frontier_states.py` to also record,
  per surviving beam state, whether it lies on the eventually-returned verified path or on
  a pruned branch, plus realized remaining-suffix length. Harvest triples from a strat-51 /
  top-N run where the verified best path per pid is known. Label good vs bad by realized
  suffix length to solved.
- **Loss:** reuse `cayley/listwise_loss.py::pairwise_hinge_loss`
  (`L_pair = log(1 + exp(Q_good − Q_bad))`, lower = better) as a small auxiliary term in
  `60_train_admissible.py` (the m_dd_v0 trainer), warmstart from m_dd_v0 or AZ-v4-V, small λ.
- **Variance guard:** run `v_canary.py` every few epochs — the regret term must not inject
  d≈20 variance the way the repr bundle did.

#### Acceptance gate

Training-side: strat-51 production recipe, **≥ +3 solves AND mean ≤ 0.95 × 89.4 (≤ 84.9)**,
**AND d=20 std ≤ ~2.5**. Most training variants fail this; it is deliberately hard.

#### Risk / cost

Medium–high. The data pipeline is the bulk of the effort and the 6M ceiling + variance
risk are real — but this is the one training-side idea with a clean "genuinely different
from what already failed" story, so it is the highest-conviction *training* bet.

---

### 13.3 Macro mining from own corpus

Covers *macro mining from own corpus* (doc §2.8). Deploys only through Macro-Q.

#### One-liner

Mine candidate macros from **path differences inside our own verified corpus** (where one
path's subpath reaches the same residual as another's in fewer moves), dedupe by net
120-permutation, and deploy them **only** through the trained Macro-Q shortlister
(`41_macro_qhead.py`) — never as raw beam actions.

#### Why this is different from the macro failures on record

Brute-force d=4 commutators as macros **hurt** (+9 to +41 moves on 5 hard pids), and the
37K-entry commutator window-replacement gave **0 matches** — because those macros sit
*outside* the beam's structured-permutation distribution. Macros mined from our own corpus
are by construction in the competition move-distribution. This is also distinct from
`31_curate_macros.py` (speedcubing/commutator sourced).

#### Honest caveat (from the cross-relink failure)

Cross-path mid-states are typically Hamming 35–66 apart, so exact-residual matches for
extraction will be sparse. Expect most mined macros to come from **short local windows**
(the same place bridge compression wins — front-of-path detours, windows 40–60), i.e. a
long tail of small, high-support macros rather than a few big ones. Set expectations
accordingly.

#### Implementation

- New mining script (e.g. `84_mine_corpus_macros.py`): for each pid with ≥2 verified paths,
  align prefix states; find positions where `S_i == S'_p` and `S_j == S'_q` with
  `(q−p) < (j−i)`; extract the shorter subword as a macro candidate; key by net perm;
  score by `support × avg_saving × distinct_pids`.
- Emit `curated_macros_corpus.pkl` in the format `41_macro_qhead.py` already consumes
  (`name, word_idxs, word_names, perm, ...`).
- Train m24-style Macro-Q (built), then run macro-augmented beam
  (`KhoruzhiiSolver(macros=...)`, shipped) on hard-tail pids.

#### Acceptance gate

Inference-side, but the specific historical gate applies: **a mined macro action must
appear in a verified *shorter* final path**, not merely exist in the action set. Min-merge
any wins.

#### Risk / cost

Highest effort of the four and the poorest macro track record — but the corpus angle is the
one untried macro variant and has the highest ceiling if it pays. Queue last, or when the
others stall.

---

### 13.4 Symmetry consistency loss

Covers *symmetry consistency loss* (doc §4.6, §3.4).

#### One-liner

Penalize `V(s) ≠ V(R s R⁻¹)` (and Q equivariance) as a **constraint** that ties existing
predictions together — adding **no** new target samples.

#### Why this is different from m31 (rotation augmentation, regressed)

m31 added rotated states as *extra Bellman samples*, diluting the 6M signal across 360
orbit-equivalents (REJECTED, 95.76). A consistency loss introduces no new targets; it only
enforces agreement between `V(s)` and `V(RsR⁻¹)`. The `lambda_sym` field already exists in
`bellman.py` (added during the repr-bundle work).

#### Honest caveat

`lambda_sym` was part of the **toxic** m_repr_v0 bundle. Forensics named solver-trace as
prime suspect, but sym was **not** independently exonerated (only child-rank was, via
m_rank_v0). So this experiment doubles as the forensic isolation of `lambda_sym`.

#### Implementation

Isolate `lambda_sym` on m_dd_v0 (encoding=embedding, no other bundle terms), exactly the
`m_rank_v0` / `m_rank_ctrl` template (those configs already exist as a pattern): treatment
sweep `λ_sym ∈ {0.05, 0.1, 0.3}`, control arm `λ_sym=0` to measure re-fine-tune drift.
Watch d=20 variance throughout.

#### Acceptance gate

Same training-side gate as 13.2 (strat-51 production + d=20 std ≤ ~2.5).

#### Risk / cost

Low cost (the field exists; runs in minutes like `m_rank_ctrl`), low–medium EV (6M ceiling
+ partial negative signal). Worth running as a cheap forensic + cheap shot in parallel with
13.1's benching.

---

### 13.5 Sequencing for this batch

1. **13.1 PHS cumulative scoring** — implement now. Inference-side, no training, fast
   strat-51 signal, cannot regress under min-merge.
2. **13.4 symmetry consistency loss** — cheap to launch as a background training arm (and a
   useful forensic on the repr bundle) while 13.1 is benched.
3. **13.2 frontier regret / pairwise on real mistakes** — highest-conviction *training*
   idea; queue after 13.1 lands, since it needs the real-mistake data pipeline first.
4. **13.3 macro mining from corpus** — highest effort / highest ceiling; queue last or when
   the others stall.

Binding reminder for all four: inference-side → min-merge (any net win counts); training-
side → strat-51 production recipe + d=20 std ≤ ~2.5. Never trust a 10-pid bench (Rule 21)
or saturation-mean alone (the repr-bundle lesson).
