# Graph Transformer Plan for Megaminx

Status: proposal / design note.

Goal: test a puzzle-native architecture that is more structured than the
current ResMLP, without jumping straight into a multi-week exact-equivariance
project.

The current production neural family is a residual MLP over a flattened
120-sticker embedding. It is fast and has been hard to beat, but it is also
structurally blind: it must learn from data which stickers are adjacent, which
stickers move together under generators, which positions form pieces, and which
states are equivalent under rotations. A graph transformer gives the model
those relational hints while keeping the same external interface:

```text
input:  states, int tensor of shape (B, 120)
output: scalar V, or (V, policy logits), or Q logits over actions
```

This note assumes the megaminx project state as of 2026-05-18:

- Best submission: 75,200.
- AZ v4 is a real signal improvement: 51/51 strat-5, mean 87.5 V-only.
- Pure training-side recipe sweeps around the ResMLP cluster mostly saturate.
- Naive rotation augmentation regressed, but sym-aware Q-shortlisting works.
- Bigger MLP trunks did not reliably improve beam metrics.

The graph transformer should therefore be treated as a targeted inductive-bias
test, not as "same recipe, bigger model".

## Recommendation

Run this only after the lower-risk AZ-v5 work:

1. Rebuild the AZ dataset from the latest best merged paths.
2. Add target-side symmetry averaging or a small symmetry-consistency loss.
3. Distill an AZ-compatible sym-aware Q-shortlister.
4. Then test graph transformer as a new V/AZ trunk.

If we do run it, start small and judge by beam score early. Do not train for
days just because the training loss is improving.

## Why Graph Transformer Instead of Vanilla Transformer

A vanilla transformer treats the state as a 120-token sequence with learned
position embeddings. That is already more expressive than a flattened MLP, but
the token order is arbitrary. The model must rediscover the puzzle graph.

A graph transformer keeps global attention, but adds relational bias:

- same-face neighborhood
- generator-induced edges
- inverse-generator edges
- piece-mate edges, if available
- symmetry/orbit metadata
- optional shortest graph distance bucket between sticker positions

This is the minimum architecture change that says, "these tokens are not words
in a sentence; they are stickers in a group action."

## Core Architecture

Use one token per sticker position.

```text
state token i:
    sticker_id = state[i]
    position_id = i
    face_id / local_slot_id
    orbit_id under generator action
    optional piece_slot_id / piece_type
```

Initial embedding:

```text
h_i =
    sticker_embedding[state[i]]
  + position_embedding[i]
  + face_embedding[face(i)]
  + slot_embedding[slot(i)]
  + orbit_embedding[orbit(i)]
```

Suggested v0 dimensions:

| knob | v0 value | rationale |
|---|---:|---|
| d_model | 192 or 256 | small enough to train quickly |
| layers | 4 | architecture smoke, not a 4-day bet |
| heads | 4 or 8 | head_dim 32-64 |
| ffn_dim | 4 * d_model | standard transformer ratio |
| dropout | 0.0 | match existing no-dropout regime |
| activation | GELU or SwiGLU | GELU for simplicity, SwiGLU if stable |
| norm | pre-norm RMSNorm or LayerNorm | pre-norm for stability |
| readout | attention pool or solved-token pool | better than plain mean-pool |

Start with a 3M-8M parameter model. Only scale to 15M-30M if the small version
matches AZ v4 or beats ResMLP on strat-5.

## Graph Bias Options

There are three increasingly complex ways to inject the graph.

### Option A: Additive Attention Bias

Precompute a relation matrix:

```text
rel[i, j] in {same_position, same_face, generator_edge, inverse_generator_edge,
              same_piece, adjacent_face, other}
```

Each attention head has a learned scalar bias per relation:

```text
attention_score[h, i, j] =
    q[h, i] dot k[h, j] / sqrt(d_head)
  + relation_bias[h, rel[i, j]]
```

This is the best v0. It keeps full attention, costs almost nothing, and is easy
to ablate by setting the bias to zero.

### Option B: Shortest-Distance Buckets

Build an undirected graph over sticker positions using:

- generator edges: i -> g(i), for all primitive generators g
- same-face adjacent slots
- optional same-piece links

Precompute shortest path distance between all sticker positions and bucket it:

```text
dist_bucket[i, j] = min(shortest_path_distance(i, j), 8)
```

Then add a learned distance bias per head:

```text
score += dist_bias[h, dist_bucket[i, j]]
```

This is similar to graphormer-style structural bias. It is probably the most
useful single graph prior once the relation table is working.

### Option C: Sparse Local Message Passing + Global Attention

Before each transformer layer, run one small message-passing block on the
fixed puzzle graph:

```text
m_i = aggregate_j in N(i) MLP([h_i, h_j, edge_type(i,j)])
h_i = h_i + MLP([h_i, m_i])
```

Then run global self-attention. This is stronger but slower and more code.
Do not start here.

## Minimal Model Variants

### GT-v0: Graph-Biased V Model

Purpose: answer "does graph bias help value ranking?"

Output:

```text
V(s) -> scalar distance-like score
```

Training:

- warm-start is probably not possible from ResMLP, so train from scratch
- use the m_dd_v0 / AZ-v4 value recipe, not plain random-walk MSE only
- include V0/d=1 anchors
- include BFS-d6 mixin
- use target net Bellman with target_update_every_epochs=10
- evaluate early

Do not judge this only by train loss. Beam ranking is the metric.

### GT-AZ-v1: Dual-Head Graph Transformer

Purpose: combine graph bias with the AZ-v4 signal.

Outputs:

```text
policy_logits(s): (B, 24)
V(s):             (B,)
```

Training:

```text
loss = beta * value_loss + alpha * policy_ce
```

Use the AZ-v4 dataset recipe, but rebuild from the current best merged
submission rather than the older 76,304 paths.

Important: the policy head is equivariant under symmetries. If training on
rotated states or adding consistency, action labels must be relabeled through
the conjugation table.

### GT-Q-v1: Graph Transformer Q-Shortlister

Purpose: graph model as a smarter action shortlister, not a full V replacement.

Output:

```text
Q(s, a) for all 24 primitive actions
```

Target:

```text
Q_target(s, a) = V_teacher(apply(s, a))
```

Loss:

```text
0.5 * MSE(Q_pred, Q_target)
+ 0.5 * KL(softmax(-Q_target / T) || softmax(-Q_pred / T))
```

This is lower risk than replacing V. A graph-aware shortlister could improve
child ordering while the proven AZ/ResMLP V still does final scoring.

## Token Features

Start with features that are static and cheap.

Required:

- sticker id: `state[i]`
- position id: `i`
- face id: `i // 10` if face layout is contiguous, otherwise derive from puzzle
- local slot id within face
- orbit id under the generator group

Optional:

- moved-by-generator bitset, or compressed embedding of which primitive moves
  affect position i
- piece id and piece orientation features, if a reliable megaminx piece
  decomposition is available
- distance-to-center or face-ring class, if useful for the 120-sticker layout

Avoid features that are constant over all reachable states, such as global
parity invariants. They add no useful signal.

## Graph Construction

Create a helper script:

```text
megaminx/scripts/72_build_graph_features.py
```

Outputs:

```text
megaminx/data/graph_features.pt
```

Suggested schema:

```python
{
    "state_size": 120,
    "face_id": LongTensor[120],
    "slot_id": LongTensor[120],
    "orbit_id": LongTensor[120],
    "relation_id": LongTensor[120, 120],
    "dist_bucket": LongTensor[120, 120],
    "edge_index": LongTensor[2, E],
    "edge_type": LongTensor[E],
}
```

Relation IDs should be deterministic and documented:

```text
0 other
1 same_position
2 same_face
3 generator_edge
4 inverse_generator_edge
5 same_piece
6 same_orbit
7 adjacent_face
```

For v0, `same_piece` can be omitted or set to 0 until piece decomposition is
trusted.

Unit tests:

1. `relation_id` is symmetric for undirected relation types.
2. Every generator edge appears with the expected type.
3. `dist_bucket[i, i] == 0`.
4. The graph is connected.
5. Static features have expected cardinalities.

## Action Relabel Table

Any policy/Q symmetry training needs this first.

Build:

```text
action_relabel[rot_idx, action_idx] = rotated_action_idx
```

Definition:

```text
apply(R*s*R_inv, rotated_action) == R*apply(s, action)*R_inv
```

Use the same conjugation logic as `megaminx/scripts/03_solve.py`.

Unit test on random states:

```python
s1 = rotate(apply(s, a), R)
s2 = apply(rotate(s, R), action_relabel[R, a])
assert s1 == s2
```

Do this before any graph-transformer policy or Q work. It is the highest-risk
bug surface.

## Readout Choices

Mean pooling is acceptable for a smoke test, but not ideal. Better options:

### Attention Pool

Use a learned query vector:

```text
alpha_i = softmax(q_pool dot W h_i)
pooled = sum_i alpha_i h_i
V = linear(pooled)
```

This lets the model focus on currently informative stickers.

### Solved-Reference Pool

Append one learned `[SOLVED]` or `[CLS]` token that attends to all sticker
tokens. Read from that token. This is simple and probably good enough.

### Multi-Pool

Use several learned pool queries and concatenate them. Useful if one query
collapses to a coarse Hamming-distance-like signal.

Start with `[CLS]` or attention pool. Avoid mean-pool if the first run is
intended to be decisive.

## Training Recipe

For a first serious run, do not use plain random-walk pretrain alone. That has
already been a weak proxy for beam quality.

### GT-v0 Smoke

Purpose: check feasibility and speed.

```text
d_model=192
layers=3
heads=4
ffn_dim=768
batch_size=2048
samples_per_epoch=300000
epochs=50
loss=random-walk MSE only
```

Gate:

- no OOM
- loss decreases sanely
- one forward benchmark is not catastrophically slower than transformer m46

This run is not expected to beat anything.

### GT-v1 Value Run

Purpose: compare graph bias against ResMLP value training.

```text
d_model=256
layers=4
heads=8
ffn_dim=1024
batch_size=2048 or 4096
samples_per_epoch=500000
value recipe=m_dd_v0 style
target_update_every_epochs=10
bfs_d6_fraction=0.10
n_anchor_v0=32
n_anchor_d1=4
epochs=100-200 with eval every 25
```

Gate:

- V(V0) near 0
- d=1 anchors near 1
- strat-5 mean competitive by epoch 50-100

Stop if V calibration drifts while training loss improves.

### GT-AZ-v1

Purpose: test graph transformer with the strongest signal source.

Use:

- rebuilt AZ dataset from latest best merged submission
- Bellman value branch
- policy CE branch
- early stopping

Suggested initial weights:

```text
beta_value = 1.0
alpha_policy = 0.5 or 1.0
lr = 2e-4 to 5e-4
policy_batch_size = 512 or 1024
rw_batch_size = 4096 if memory permits
```

The early-stop lesson from AZ v4 is binding. Save checkpoints every 5 epochs
and run a small beam bench frequently.

### GT-Q-v1

Purpose: use graph transformer where action relations matter most.

Teacher:

- AZ v4 V-only or AZ-v5 V, not m05 unless the student will only pair with m05.

Dataset:

- random-walk states
- optional beam-frontier states if available
- optional rotated states, with targets recomputed on rotated state

Gate:

- recall@alpha=1 and recall@alpha=2 against the teacher top-B candidates
- strat-5 with teacher rerank does not regress

This may be the highest-EV graph-transformer use because it can improve search
ordering without replacing the proven V scorer.

## Inference Considerations

The graph transformer will probably be slower than ResMLP. That matters because
beam search calls the model on millions of candidates.

Mitigations:

1. Use it as a Q-shortlister, not final V.
2. Distill graph transformer predictions back into a ResMLP Q-head.
3. Use it only on hard-tail rescues.
4. Use it as a teacher for AZ-v5 data generation.
5. Compile only if fixed batch padding is implemented for inference.

Do not deploy a slow V model full-1001 unless strat-5 gains are very large.

## Symmetry Interaction

The graph transformer can combine with symmetry in three ways.

### Safe First Step: Target-Side Orbit Averaging

During Bellman target generation:

```text
V_target_sym(c) = mean_g V_target(g*c*g_inv)
target(s) = 1 + min_a V_target_sym(apply(s, a))
```

This smooths targets without forcing the trainable model to consume rotated
states as main examples.

### Auxiliary Consistency

For scalar V:

```text
L_inv = MSE(V(s).detach(), V(g*s*g_inv))
      + MSE(V(s), V(g*s*g_inv).detach())
```

For policy/Q:

```text
logits_back = unrotate_action_logits(logits(g*s*g_inv), g)
L_eq = KL(softmax(logits(s)), softmax(logits_back.detach()))
```

Start with tiny weights:

```text
lambda_v = 0.02 to 0.05
lambda_pi = 0.02
```

Do not use strong consistency on the first graph-transformer run.

### Exact Equivariance

This is a later project. If target averaging and small consistency help, then
an exact group-equivariant graph model becomes more justified.

## Evaluation Gates

Use beam metrics, not only loss.

Minimum gates:

1. `61_eval_v_at_solved.py` calibration:
   - V(V0) close to 0
   - mean V@d=1 close to 1
   - no large undershoot/overshoot surprise on BFS-d6 states
2. Strat-5:
   - compare to m05, m29, AZ v4, and latest AZ-v5 if available
   - pass only if mean improves materially or adds unique min-merge wins
3. Recall for Q-head:
   - recall@alpha=2 should be near 100 percent against its exact teacher
4. Wall:
   - report seconds per pid at beam 65k
   - a slow model needs enough path improvement to justify hard-tail-only use

Suggested acceptance thresholds:

```text
GT-v1 V:       strat-5 mean <= 87.5 or clear unique-win contribution
GT-AZ-v1:     strat-5 mean <= 86.5 to justify replacing AZ v4
GT-Q-v1:      no path regression at alpha=2, plus 2x+ wall reduction or better
hard-tail use: improves top-50 long pids by >=100 moves
```

## Failure Modes

### Training Loss Improves, Beam Gets Worse

This has happened repeatedly. Beam cares about sibling ranking at the frontier,
not global MSE.

Response:

- add rank/KL losses over children
- train Q-head instead of scalar V
- mix beam-frontier states

### Model Is Too Slow

Response:

- use graph model as teacher
- distill into ResMLP
- deploy only for hard-tail rescue
- shrink d_model/layers before increasing beam cost

### Policy Memorizes and Damages V

AZ v4 already showed this.

Response:

- checkpoint every 5 epochs
- early-stop by beam bench
- lower policy loss weight
- freeze lower graph layers for first few policy epochs

### Graph Bias Is Wrong

Bad relation tables can silently hurt.

Response:

- keep a no-bias transformer ablation
- unit-test graph edges
- log learned relation bias magnitudes

### Symmetry Consistency Removes Useful Diversity

Inference-time sym-ensemble benefits partly because model errors differ by
frame.

Response:

- prefer target-side averaging first
- keep consistency weights tiny
- judge by min-merge contribution, not orbit error alone

## Implementation Plan

### Phase 0: Static Tables

Files:

```text
megaminx/scripts/72_build_graph_features.py
megaminx/data/graph_features.pt
tests/test_megaminx_graph_features.py
```

Deliverables:

- relation table
- distance buckets
- action relabel table if symmetry/Q/policy work is included
- tests

### Phase 1: Model Class

Add either:

```text
src/cayley/graph_transformer.py
```

or put the class in a megaminx-specific module:

```text
megaminx/src/megaminx/graph_transformer.py
```

The model must expose:

```python
forward(states) -> Tensor[B] or tuple(policy_logits, value)
output_dim = 1 or 24
num_parameters()
inference_chunk_size
```

Keep the API compatible with `03_solve.py` and Bellman target code.

### Phase 2: Tiny Smoke

Train GT-v0 for 50 epochs. Verify:

- no OOM
- no NaNs
- checkpoint loads
- forward works under bf16
- solver can call it on 2-3 pids

### Phase 3: Value or Q Experiment

Pick one path:

- GT-Q-v1 if the goal is production usefulness soon
- GT-AZ-v1 if the goal is architecture breakthrough

Do not run both at once. We want a clean read.

### Phase 4: Distillation

If graph transformer improves but is slow, distill:

```text
teacher = graph transformer V or Q
student = ResMLPDistance output_dim=1 or 24
loss = MSE + ranking KL over children
```

This may be the practical endpoint: graph transformer finds a better target
landscape, ResMLP carries it cheaply inside beam search.

## Concrete First Experiment

Recommended first serious run:

```text
name: gt_q_az_v4_v1
type: graph-transformer Q-shortlister
teacher: AZ v4 V-only
student:
  d_model: 256
  layers: 4
  heads: 8
  ffn_dim: 1024
  relation_bias: true
  dist_bias: true
training:
  samples_per_epoch: 300000
  batch_size: 2048
  epochs: 200
  k_max: 80
  lr: 3e-4
  loss: 0.5*MSE + 0.5*KL
  rotation_aug_prob: 0.25
eval:
  recall alpha=1,2 vs AZ v4 V
  strat-5 AZ v4 V + gt_q alpha=2
```

Why Q first:

- action ranking is where graph structure is most likely to matter
- recall is a cheap diagnostic
- final V rerank can remain the proven AZ v4 V
- if slow, Q can be distilled back into a ResMLP shortlister

Second run, only if Q looks good:

```text
name: gt_az_v1
type: dual-head graph transformer
dataset: latest AZ dataset rebuilt from best merged submission
value: Bellman + BFS-d6 + anchors
policy: CE, early-stopped
symmetry: target-side averaging K=2 only
```

## Notes on CNN and Mamba

Standard 2D CNN is not recommended as a first bet. The megaminx state is not a
rectangular image, and arbitrary face-layout choices create fake locality.
A "CNN" that respects face adjacency is essentially a graph or equivariant
model, which is what this plan targets.

Mamba/SSM is also not a first bet. It is attractive for long ordered sequences,
but this state has only 120 tokens and no natural 1D order. A multi-scan SSM
over face cycles might be interesting later, but graph attention is a cleaner
match to the puzzle.

## Final Recommendation

The graph transformer should enter the roadmap as a focused architecture bet:

1. Build static graph/relabel tables.
2. Try graph-transformer Q-shortlisting against AZ v4 V.
3. If Q recall/path quality is good, try dual-head GT-AZ.
4. Distill any useful graph model back into the fast ResMLP family.

This gives us a meaningful test of "the ResMLP is structurally blind" without
committing to a slow full replacement model before we know the graph prior pays.
