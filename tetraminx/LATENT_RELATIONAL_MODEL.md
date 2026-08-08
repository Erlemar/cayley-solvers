# Standalone latent relational Q model for Tetraminx

## Objective

Build a standalone all-neighbours Q scorer that preserves the PieceTransformer's
relational inductive bias without carrying 51 width-256 tokens through four Transformer
layers.

The model must have the same deployment interface as the existing scorers:

```text
input:  states, integer tensor (batch, 88)
output: Q(s, a), floating tensor (batch, 24), lower is better
```

It must run by itself during beam search. The first experiment does **not** use knowledge
distillation, an online teacher, a hybrid beam, an auxiliary AZ target, or path-label data.
It is trained from scratch on the same sparse-Q objective, sampler, symmetry coverage,
and exact BFS anchors as the ResMLP and PieceTransformer comparison cells.

The architecture is intended to answer one question cleanly:

> Can an early learned latent bottleneck retain the useful piece-to-piece reasoning of
> the PieceTransformer at approximately ResMLP inference cost?

## Evidence motivating the design

The current PieceTransformer is better than the ResMLP on deep random-walk pivots and
on the 15-pid Q@1M beam gate, but costs about 47 times as much per beam in measured wall
time. Its cost is not dominated by the 51-by-51 attention matrix. It is dominated by
carrying a width-256 representation for every token through the FF and projection layers
of four blocks.

The latent model keeps a learned relational read over all 50 physical pieces, then
compresses the state to eight latent tokens. All repeated reasoning operates on those
eight tokens. The output is decoded by 24 learned action queries, so each generator can
extract a different view of the compressed state.

## Architecture

### Default configuration

| Component | Setting |
|---|---:|
| Physical piece tokens | 50: 4 size-3, 30 size-2, 16 size-1 |
| Model width | 96 |
| Attention heads | 4 |
| Latent tokens | 8 |
| Latent self-attention blocks | 2 |
| Latent FF width | 192 (2x model width) |
| Action queries | 24 |
| Action decoder FF width | 192 |
| Activation | SiLU |
| Dropout | 0 |
| Output | 24 calibrated Q-values |

The model width is divisible by the head count, giving a head width of 24. The narrow
2x feed-forward expansion is deliberate: the existing 4x expansion is the largest
component of Transformer inference cost.

### Stage 1: physical-piece encoding

Use the existing self-verified `tetraminx/data/piece_layout.json` block system. Each
state becomes 50 physical-piece tokens. A token contains the sticker values currently
occupying the piece's one, two, or three local positions.

The encoder retains the PieceTransformer's three learned signals:

1. local sticker-value embeddings;
2. physical piece-position embeddings;
3. piece-type embeddings for sizes 3, 2, and 1.

The local concat-plus-projection is evaluated in folded form. If the original projection
is

```text
piece_projection(concat(E_0[value_0], E_1[value_1], E_2[value_2]))
```

then the implementation folds each local slice of `piece_projection.weight` into its
corresponding value table and sums the resulting lookups. This is mathematically
equivalent while avoiding a `(batch, pieces, local_slots, width)` activation.

During training, one-hot matrix multiplication is retained because the existing profile
showed that dense GEMMs avoid the severe scatter-add contention in embedding backward.
An inference-only cached-gather path is a later optimization, not part of the first
scientific gate.

All 50 pieces are retained. The 16 singleton pieces are cheap once compression happens
after this stage, and removing them would confound the architectural test with an
information ablation.

### Stage 2: learned compression into eight latents

Create eight learned latent query vectors. For each state, the queries cross-attend once
to the 50 normalized piece tokens:

```text
L_0 = learned_latents
L_1 = L_0 + CrossAttention(LayerNorm(L_0), LayerNorm(piece_tokens))
L_2 = L_1 + FF(LayerNorm(L_1))
```

This is the only point where the relational trunk reads all 50 pieces. The eight queries
are not assigned hand-written semantics. They are free to specialize into useful global
substructures, conflicts, piece families, or distance factors.

The compression stage has its own residual FF sublayer. Without it, each latent would be
only a weighted linear combination of local piece descriptors before entering the latent
reasoner.

### Stage 3: latent reasoning

Process the eight latents with two ordinary pre-norm self-attention blocks:

```text
L = L + SelfAttention(LayerNorm(L))
L = L + FF_96_to_192_to_96(LayerNorm(L))
```

Self-attention uses fused QKV projection and
`torch.nn.functional.scaled_dot_product_attention`, matching the optimized
PieceTransformer implementation. With only eight tokens, both attention and per-token FF
costs are small.

### Stage 4: action-query decoder

Create one learned query for each of the 24 generators. The queries cross-attend to the
final eight latents:

```text
A_0 = learned_action_queries
A_1 = A_0 + CrossAttention(LayerNorm(A_0), LayerNorm(latents))
A_2 = A_1 + FF(LayerNorm(A_1))
advantage_a = shared_linear(A_2[a])
```

This decoder is preferable to a single pooled state vector followed by a 24-wide linear
head. It permits each move to retrieve different relational evidence while retaining one
forward pass per parent state.

### Stage 5: calibrated dueling-Q output

The Tetraminx beam performs a global top-B across actions from different parent states.
It therefore needs both within-parent action ordering and absolute cross-parent
calibration.

Compute a scalar state baseline from the mean of the normalized latent tokens:

```text
value = value_head(mean(LayerNorm(latents), token_axis))
Q(s, a) = value + advantage_a - mean_a(advantage_a)
```

The centering makes the decomposition identifiable: `value` is exactly the mean predicted
Q for a parent. This scalar is not an auxiliary AZ head and receives no separate value
loss. It is an internal, jointly trained part of the 24 Q outputs.

The public model interface still reports `has_value_head = false`, because inference and
training consume only the combined Q tensor.

## Estimated compute

Ignoring the small input-table construction, normalization, bias, and activation costs:

| Scorer | Approximate MACs/state |
|---|---:|
| Existing ResMLP | 5.0M |
| Latent relational, d=96, 8 latents | about 4M |
| Existing PieceTransformer | 165.8M |

The estimate is a design target, not a throughput claim. The acceptance gate uses
measured wall time at realistic beam batch sizes because small attention operations and
kernel launch overhead may prevent MAC ratios from translating directly to speed.

## Standalone training plan

### Controlled comparison

Use the same data and objective as `mx_tf.yaml` and the ResMLP comparison cell. The only
changed variable is the model architecture.

| Training setting | Value |
|---|---:|
| Epochs | 1500 |
| Base batch before symmetry expansion | 2048 |
| Steps per epoch | 256 |
| Samples per epoch | identical to the existing matched cells |
| Random-walk depth | `k_min=2`, `k_max=40` |
| Pivot tilt | 0.5 |
| Symmetry coverage | enabled, full greedy coverage width |
| Exact anchor batch | 512 |
| Maximum anchor depth | 5 using the d6 table |
| Sparse-Q loss | MSE on the two labelled actions |
| Anchor loss | MSE on all 24 exact actions |
| Top-1 margin loss | disabled (`weight=0`) |
| Path labels | disabled |
| Auxiliary AZ value loss | disabled |
| Optimizer | fused AdamW |
| Learning rate | 3e-4 |
| Weight decay | 3e-3 |
| Gradient clip | 1.0 |
| Precision | bf16 autocast |
| Training compilation | enabled, fixed shapes |
| Seed | 0 |

The base batch is restored to the ResMLP-sized 2048 because the latent architecture does
not retain four full-width 51-token activation stacks. If this pages or OOMs, reduce the
base batch and increase steps per epoch by the same factor so the sample budget remains
identical.

### Checkpoints and stopping policy

Save every 50 epochs and run the existing validation probe at every checkpoint. Do not
select the deployment checkpoint from validation loss alone. Previous Tetraminx runs
showed that probe metrics can flatten while beam quality continues improving.

Required checkpoint gates:

1. epoch 250: early representational signal and throughput;
2. epoch 500: matched against the currently reported Transformer checkpoint;
3. epoch 1000;
4. epoch 1500: matched full-budget comparison.

Continue to epoch 1500 unless there is a clear failure such as non-learning, NaNs, or a
large beam regression. A flat probe alone is not a stopping rule.

## Evaluation plan

### Gate A: correctness

Before training:

- forward output is finite and has shape `(batch, 24)`;
- gradients reach the piece encoder, latent queries, latent blocks, action queries,
  advantage head, and value head;
- configuration round-trips through `get_model_config()` and `model_from_config()`;
- saved and reloaded state dicts reproduce outputs exactly in evaluation mode;
- the solver identifies the model as an all-neighbours Q scorer.

### Gate B: scorer throughput

Benchmark bf16 inference using the same device and realistic fixed chunks used by beam
search. Report states/second, peak memory, and wall time for at least two chunk sizes.

Targets:

- at least 10x faster than the existing PieceTransformer;
- preferably no more than 2x slower than the ResMLP;
- no silent Windows WDDM paging at the selected chunk size.

Do not enable compiled beam inference until the search path pads all forward calls to a
small set of fixed batch shapes and pre-warms them. Variable-shape compilation is a known
project failure mode.

### Gate C: decision-quality probe

Run `52_eval_q.py` on the common fixed probe, not the model's own training-log
distribution. Report pair accuracy, top-1 accuracy, gap, and undo-score calibration by
pivot-depth band.

The critical bands are 10-14, 20-24, 25-29, and 30-40. The first quality target is to
land between the ResMLP and PieceTransformer at depth while retaining calibration.

Useful reference values from the existing comparison:

| Band | ResMLP top-1 / gap | PieceTransformer top-1 / gap |
|---|---:|---:|
| 10-14 | 0.537 / 1.555 | 0.576 / 1.599 |
| 20-24 | 0.249 / 0.857 | 0.307 / 0.978 |
| 25-29 | 0.151 / 0.481 | 0.203 / 0.651 |
| 30-40 | 0.105 / 0.302 | 0.128 / 0.354 |

### Gate D: beam quality

The binding gate is the same replay-verified beam comparison used in
`RESMLP_VS_TRANSFORMER.md`:

- 15 stratified pids;
- Q@1M;
- one symmetry frame initially;
- exact endgame;
- `history_depth=1`;
- `--no-merge` for raw architectural comparison;
- every path replayed and asserted to solve.

References on that gate are ResMLP 468 moves and PieceTransformer 441 moves. A result
near 450 at substantially lower wall time is a clear success. A probe win without a beam
win is not accepted.

After node-matched evaluation, run a measured equal-wall-clock comparison. Choose widths
and frame counts from observed seconds rather than assuming a nominal width ratio.

## Ablation order if the default misses

Change one axis at a time in this order:

1. **Latent count:** 8 to 12 or 16. This increases bottleneck capacity cheaply.
2. **Model width:** 96 to 128. This is more expensive but still far below the current
   Transformer.
3. **Latent depth:** 2 to 3 blocks if the first two changes fail.
4. **Action decoder:** compare learned action cross-attention against a flat linear head
   only as a diagnostic.
5. **Singleton handling:** compress the 16 singleton pieces through a side pool only
   after proving they dominate cost or add no quality.

Do not begin by removing attention, replacing it with linear attention, increasing the FF
ratio to 4x, adding an auxiliary value objective, mixing path labels, or changing the
sampler. Those changes either optimize a non-bottleneck or confound the standalone
architecture test with axes already shown to be risky.

## Optional later phase: teacher-assisted training

Knowledge distillation is explicitly outside the first experiment. If the standalone
model is fast and relationally stronger than the ResMLP but remains below the
PieceTransformer, a later run may use the PieceTransformer's dense 24-way predictions as
additional offline training targets. The trained latent model would still be completely
standalone at inference.

That later phase must be recorded as a separate experiment and checkpoint family; it
must not replace or obscure the clean from-scratch comparison specified here.

## Deliverables

The first implementation consists of:

1. `LatentRelationalQ` in `tetraminx/src/tetraminx/models.py`;
2. `arch: latent_relational` support in `build_model()` and checkpoint round-tripping;
3. trainer construction support in `tetraminx/scripts/51_train_sparse_q.py`;
4. a matched standalone config in `tetraminx/configs/mx_latent_relational.yaml`;
5. focused unit tests for shape, gradients, dueling calibration, state/config round-trip,
   and invalid configuration handling.

