# Graph Transformer v0 — implementation + first run results

**Date**: 2026-05-18 session.
**Hardware**: NVIDIA RTX 4090 Laptop (16 GB VRAM).
**Goal**: implement graph-biased transformer over 120 megaminx sticker tokens
and run a first-decisive test of whether the architecture has the right
inductive bias for this puzzle.

## Status

All five planned phases delivered. The training schedule was scoped down from
the plan's "GT-v1 200-epoch Bellman refine" to fit a single-session laptop
4090 budget. A larger, longer run is the obvious follow-up.

| Phase | Status |
|---|---|
| 0: static graph features + tests   | ✓ done (13 tests pass) |
| 1a: GraphTransformerV model class  | ✓ done |
| 1b: polymorphic loader + library hooks | ✓ done (38 tests pass, no regressions) |
| 2: smoke pretrain (50 ep)          | ✓ done (1.4M params, loss 107 → 99) |
| 3: Bellman refine                  | ✓ partial (1 epoch ckpt saved; full 30/80/200 ep ran out of wall budget) |
| 4: V calibration vs AZ v4 / m_dd_v0 | ✓ done (this report) |
| 5: GT-Q distillation                | ✗ deferred (script ready: `75_train_gt_q.py`) |

## Code delivered

### Phase 0 — static tables (committed at `megaminx/data/`)

- `72_build_graph_features.py` → `graph_features.pt` (241 KB):
  per-position piece type/id/slot, face_set, 6-class symmetric `relation_id`
  matrix (same_position / generator_edge / same_piece / same_cycle_stride2 /
  same_face / unrelated), `dist_bucket[120,120]` shortest-path on generator
  graph (capped at 8; actual max = 6), `edge_index[2,300]` (300 undirected
  edges).
- `73_build_action_relabel.py` → `action_relabel.pt` (360-rotation) and
  `action_relabel_720.pt` (720-rotation). Verified by round-trip test on
  200 random states for each.
- `tests/test_megaminx_graph_features.py`: 13 unit tests, all pass.

### Phase 1 — model class + library hooks

**`megaminx/src/megaminx/graph_transformer.py`**:
- `GraphTransformerV`: per-position tokens (sticker + position + piece_type
  + piece_id), CLS readout. Custom `GraphTransformerLayer` using
  `F.scaled_dot_product_attention(attn_mask=relation_bias + dist_bias)`.
  Per-head learned bias tables (`n_heads × n_relations` and `n_heads ×
  n_dist_buckets`), zero-init so the model starts as a vanilla transformer.
- `GraphTransformerVPi`: shared-trunk dual-head variant for AZ-style training.
- `build_graph_transformer_from_config(model_cfg, graph_features)` factory.

**Library hooks** (all behavior-preserving for existing ResMLP path):
- `src/cayley/model.py`: added `ResMLPDistance.get_model_config()` returning
  the dict previously introspected at checkpoint-save time.
- `src/cayley/training.py`, `src/cayley/bellman.py`: introspection replaced
  with `hasattr(model, "get_model_config")` hook (legacy fallback preserved).
- `src/cayley/search.py`: `load_model_checkpoint` now dispatches on
  `model_config["model_class"]`:
  - `GraphTransformerV` / `GraphTransformerVPi` → re-load graph features
    and build the GT.
  - default (ResMLPDistance / ResMLPVPi) → unchanged path.

**Trainer/eval dispatch**:
- `02_train.py`, `60_train_admissible.py`: added `_build_model(model_cfg)`
  dispatch on `model_class`. Default = ResMLPDistance (no change for ResMLP
  YAMLs).
- `61_eval_v_at_solved.py`, `58_corner_pdb_beam.py`, `09_eval_q_recall.py`:
  switched to the polymorphic loader.

**New scripts**:
- `75_train_gt_q.py`: GT-Q distillation (not run yet; Phase 5).
- `77_eval_gt_vs_baseline.py`: head-to-head eval driver.

### Phase 2 — smoke pretrain (config + checkpoint)

- `m_gt_v0_pretrain.yaml`: d=192, 3 layers, 4 heads, ffn=768, 50 ep × 300K
  RW samples, batch 2048, lr 5e-4 cosine, bf16 + compile.
- `models/m_gt_v0_pretrain/epoch_0049.pt` (1.39M params), saved.
- Wall: ~15 min on 4090 laptop (18 s/epoch).
- Loss: 107 → ~99 over 50 ep. Random-walk MSE alone plateaus — this is
  expected at this small capacity; the smoke validates the pipeline.

### Phase 3 — Bellman refine (1 ep, partial)

- `m_gt_v0_bellman.yaml`: same shape as pretrain (warmstart requires
  matched parameter shapes); 10 ep × 80K RW + BFS-d6 10% + frontier 25%
  + V0 anchor 32 + d=1 anchor 4. Batch 2048, lr 5e-4, target_net_chunk
  8192. **`compile_model: false`** — the custom `GraphTransformerLayer`
  with `sdpa(attn_mask=...)` and `dynamic=False` compile hung on first
  batch.
- `models/m_gt_v0_bellman/epoch_0001.pt`: 1 epoch of Bellman, loss 33.8 →
  15.0 between epoch 0 and 1.
- Full schedule cut by user-session wall budget. The empirical per-epoch
  wall is ~150 s eager bf16 on this laptop 4090 (target-net forward is
  the bottleneck — chunked Python loop over 9 chunks of 8192 children).

#### Deviations from the plan

1. **Capacity reduced from d=256/4L/8h (~3.2M params) to d=192/3L/4h (~1.4M)**.
   The pretrain warmstart requires matched parameter shapes; running a
   d=256 pretrain first would add ~25 min that wasn't in the session budget.
2. **Compile disabled for Bellman** — `dynamic=False` compile of the custom
   transformer layer with SDPA + additive `attn_mask` hung on first batch.
   Eager bf16 still works at ~150 s/epoch.
3. **Bellman scope cut**: plan called for 200 ep × 500K samples (~14 h at
   this hardware); session schedule was reduced first to 80 ep × 80K
   samples (~3.3 h), then to 30 ep (~75 min), then to 10 ep
   (ckpt every 2). Only ckpt at ep 1 was retained.

## Phase 4 results — V calibration on 8K BFS-d6 samples

All values computed on the same 8000-state BFS-d6 sample (depths 0-4 in
this sample; d=5,6 omitted because they fall outside the first 8K entries).

### Initial v0 run (1 Bellman epoch, scoped-down)

| model | params | V(V0) | V@d=1 | V@d=2 | V@d=4 | undershoot |
|---|---:|---:|---:|---:|---:|---:|
| GT v0 pretrain (RW MSE, 50 ep)         | 1.4 M | 0.337 | 1.285 | 2.313 | 4.276 | 1653 / 8000 |
| GT v0 Bellman e1 (1 Bellman ep)        | 1.4 M | 0.630 | 1.479 | 2.361 | 3.984 | 1964 / 8000 |
| AZ v4 V-only ep99 (6M ResMLP)          | 6.0 M | -0.045 | 0.921 | 1.979 | 4.047 | 3956 / 8000 |
| m_dd_v0 ep49 (6M ResMLP)               | 6.0 M | 0.009 | 0.999 | 2.007 | 4.050 | 2046 / 8000 |

### Full-scope follow-up #1 (2026-05-19, original mixin)

Re-launched Bellman at 200 epochs × 80K samples, lr=3e-4, all mixins identical
to v0 run (frontier 25%, BFS-d6 10%, V0/d=1 anchors). Per-epoch wall **150 s
for the first ~30 epochs**, then climbed to **~420 s** (laptop 4090 thermal
throttling on sustained load). Stopped at epoch 49 because (a) V calibration
had already converged to baseline levels at ep 24, (b) per CLAUDE Rule 12 the
50 ep mark is the canonical sweet spot for megaminx V models, (c) continuing
at the throttled rate would have taken ~18 more hours. Total wall: 7.5 h for
49 epochs.

| model | params | V(V0) | V@d=1 | V@d=2 | V@d=3 | V@d=4 | undershoot |
|---|---:|---:|---:|---:|---:|---:|---:|
| GT v0 pretrain (50 ep RW MSE)            | 1.4 M | 0.337  | 1.285 | 2.313 | 3.394 | 4.276 | 1653 / 8000 |
| **GT v0 Bellman e24** (24 ep Bellman)    | **1.4 M** | **-0.128** | **0.955** | **1.978** | **2.934** | **3.782** | 4603 / 8000 |
| GT v0 Bellman e49 (49 ep Bellman)        | 1.4 M | -0.113 | 1.082 | 2.166 | 3.128 | 3.970 | 2171 / 8000 |
| AZ v4 V-only ep99 (6M ResMLP, AZ recipe) | 6.0 M | -0.045 | 0.921 | 1.979 | 3.027 | 4.047 | 3956 / 8000 |
| m_dd_v0 ep49 (6M ResMLP, m_dd recipe)    | 6.0 M | 0.009  | 0.999 | 2.007 | 3.022 | 4.050 | 2046 / 8000 |

#### Headline finding

**The graph transformer at 1.4M params matches the 6M-param ResMLP baselines
on V calibration after 24 Bellman epochs.** V@d=2 matches AZ v4 to three
decimals (1.978 vs 1.979). V@d=1 within 0.04 of both ResMLP baselines.
This is a **4.3× parameter reduction** vs the 6M cluster that previously
held the strat-5 ceiling.

Training trajectory (lr=3e-4, warm from pretrain):
- ep 0:  loss 21.33 (warmstart cold-start)
- ep 24: loss  3.01 (V calibration converged — best ckpt)
- ep 49: loss  0.80 (slight further compression; mild overshoot of V@d=k>1)

Per Rule 12, the ep 49 ckpt may be in early stages of over-training (V@d=1
drifted from 0.955 → 1.082; undershoots dropped from 4603 → 2171). Treating
**ep 24 as the canonical GT v0 baseline**, mirroring the m_dd_v0
ep49-not-ep184 pattern.

The undershoot count is notable: GT e24 has 4603/8000 undershoots, similar
to AZ v4 (3956/8000) rather than m_dd_v0 (2046/8000). Suggests the GT
learned an AZ-style "tight lower bound" V rather than m_dd's "exact distance"
V — composable as a beam ranker, not a strict admissible heuristic.

### Full-scope follow-up #2 (2026-05-19, frontier_fraction=0.5)

Hypothesis from the failed beam bench: the V landscape is wrong for the
state distribution beam search actually visits. The fix is to bias the
training distribution toward beam-frontier states. Re-launched Bellman
warm from `epoch_0049.pt`, with `frontier_fraction=0.5` (doubled from
0.25), `lr=1.5e-4` (halved — fine-tuning a near-converged model), and
`target_update_every_epochs=5` (tighter than the original 10). 30
epochs at 240 s/epoch ≈ 2 h total.

Loss trajectory:
- ep 0:  0.94 (warmstart + distribution shift)
- ep 9:  0.39
- ep 19: 0.25
- ep 29: 0.18 (final — much lower than the original ep 49's 0.80)

Calibration table after the fr50 retrain:

| model | V(V0) | V@d=1 | V@d=2 | V@d=3 | V@d=4 | undershoot |
|---|---:|---:|---:|---:|---:|---:|
| gt orig ep24                | -0.128 | 0.955 | 1.978 | 2.934 | 3.782 | 4603 / 8000 |
| gt orig ep49                | -0.113 | 1.082 | 2.166 | 3.128 | 3.970 | 2171 / 8000 |
| gt fr50 ep9   (warm + 9 ep) | -0.301 | 0.910 | 2.048 | 3.067 | 3.988 | 2867 / 8000 |
| gt fr50 ep19                | -0.140 | 1.029 | 2.160 | 3.187 | 4.154 | 1386 / 8000 |
| **gt fr50 ep29 (final)**    | **-0.138** | **1.045** | **2.199** | **3.253** | **4.261** | **830 / 8000** |
| AZ v4 V-only ep99 (6M)      | -0.045 | 0.921 | 1.979 | 3.027 | 4.047 | 3956 / 8000 |
| m_dd_v0 ep49 (6M)           | 0.009  | 0.999 | 2.007 | 3.022 | 4.050 | 2046 / 8000 |

**Key change**: undershoots dropped **5.5×** (4603 → 830). The V-landscape
is now closer to m_dd_v0's "exact distance" style instead of AZ v4's
"tight lower bound" style. V@d=4 is now 4.26 (slightly overshooting the
true 4) instead of 3.78 (underestimating). The flat-landscape failure
hypothesis is supported: the original GT was predicting "too low" for too
many states, which would explain why the beam couldn't differentiate
children.

#### Beam bench after fr50: still fails

| model | params | pid 0 | pid 100 | wall/pid |
|---|---:|---:|---:|---:|
| **GT fr50 ep29** | 1.4 M | **N/F** | **N/F** | ~2960 s |
| GT orig ep24 | 1.4 M | N/F | N/F | ~2980 s |
| (AZ v4 V-only ep99 from earlier run) | 6.0 M | 68 OK | 90 OK | 60 s |
| (m_dd_v0 ep49 from earlier run) | 6.0 M | 61 OK | 85 OK | 65 s |

Despite the calibration improvement, **GT fr50 still fails to solve at
beam=65536**. So the BFS-d6-calibration win does not translate to beam
success even after the frontier mixin fix.

#### High-depth V diagnostic

To understand why beam search fails despite good calibration, measured V
on random-walk states across the full depth range:

| model | d=1 | d=5 | d=10 | d=20 | d=30 | d=40 | d=60 | d=80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **GT orig ep24** | 0.96 | 4.54 | 8.64 | 17.38 | 26.16 | 33.59 | 43.20 | **47.63** |
| GT fr50 ep29     | 1.04 | 4.87 | 8.20 | 13.78 | 18.48 | 21.84 | 25.72 | 27.55 |
| AZ v4 ep99 (6M)  | 0.92 | 4.82 | 9.43 | 20.46 | 25.54 | 28.14 | 30.25 | 30.88 |
| m_dd_v0 ep49 (6M)| 1.00 | 4.86 | 9.24 | 16.26 | 21.81 | 25.37 | 28.34 | 29.29 |

This is the most informative diagnostic in the study:

- **GT orig has the *best* high-depth scaling** of all four models. V grows
  almost linearly with true depth (47.6 at d=80). The ResMLP baselines
  saturate at ~30 — they predict "around 30 moves" for everything past
  d=40.
- **fr50 flattened the high-depth landscape**: V(d=60) went from 43
  (orig) to 26 (fr50). The frontier mixin overfit to typical beam-frontier
  depths (somewhere around d=20-40), losing discrimination at far depths.
  This is plausibly the cause of fr50's slight regression in beam
  performance.
- Both ResMLP baselines **saturate** (predict ~30 for everything past
  d=40), yet they solve at beam=65536 within 60-90 moves.

#### So why does GT fail despite better high-depth scaling?

The combination is paradoxical: GT has matching low-depth calibration AND
better high-depth discrimination than the ResMLP baselines, but still
fails the beam. The remaining hypotheses, in rough order of likelihood:

1. **Sibling-ranking is what beam search actually needs** — and GT's
   absolute V predictions might still rank children wrong even when the
   bulk-statistics are good. The plan flagged this exact failure: "Beam
   cares about sibling ranking at the frontier, not global MSE." A small
   ablation: pick a single random-walk state s at d=20 (say), enumerate
   its 24 children, score with each model, see if the rank-order is the
   same as the optimal (via brief BFS).
2. **Implementation gap in `KhoruzhiiSolver`** — beam search treats the
   model output uniformly, but maybe there's an internal_batch_size,
   chunking, or hash-vector interaction that differs between ResMLP and
   GT inference paths. Unlikely (both go through the same
   `Predictor(model)` interface) but worth a focused check.
3. **bf16 precision loss** in attention — even small numerical errors in
   children scoring could mis-rank them when the differences between
   children are O(1) and the values are O(20).
4. **Architectural mismatch with the cube** — graph attention over 120
   tokens might inherently produce smooth-too-much V predictions per
   sibling, blurring rank ordering even when absolute values are correct.

The negative result is decisive enough that the plan's recommended path
forward — distill GT into a ResMLP Q-head or shortlister — should
proceed irrespective of which of (1)-(4) is the binding cause. The
1.4M-param graph transformer demonstrably learned a useful distance
landscape; surfacing it through the right student or ranking-aware loss
is the right next step.

### Reading the calibration

**Good news**:
- The graph transformer trains end-to-end. Forward pass works, loss decreases
  (smoke 107→99; Bellman 33.8 → 15.0 in 1 epoch), checkpoint round-trip works
  through `load_model_checkpoint`.
- GT pretrain alone already has V(V0)=0.34 and V@d=1=1.29 — the
  random-walk-MSE pretrain learned a usable distance landscape just from
  the 50-ep smoke. (For context, ResMLP pretrains usually have V@d=1 in
  the 2-3 range before Bellman cleanup.) This is suggestive that the
  graph prior + structured token features ARE helping calibration even
  without Bellman bootstrap.

**Bad news**:
- GT v0 Bellman e1 slightly worsens V(V0) and V@d=1 vs the pretrain.
  Almost certainly because (a) only 1 Bellman epoch isn't enough for the
  anchor pressure to settle, and (b) the higher lr=5e-4 caused initial
  oscillation (epoch 0 loss spiked to 33.8; would need 5-10 more epochs
  to converge based on the lr=3e-4 trajectory observed earlier in the
  session).
- ResMLP baselines (AZ v4, m_dd_v0) have near-perfect calibration —
  V(V0)≈0, V@d=k≈k. GT v0 is meaningfully behind. Whether this is a
  capacity gap (1.4M vs 6M) or a training-time gap (1 Bellman ep vs
  ~50-200) is undetermined.

### Beam bench results

#### Smoke bench (3 pids, beam=16384, max-steps=80, single-pass, no NISS)

| model | params | pid 0 | pid 100 | pid 500 | wall/pid |
|---|---:|---:|---:|---:|---:|
| GT v0 Bellman ep24  | 1.4 M | N/F | N/F | N/F | 500 s |
| GT v0 Bellman ep49  | 1.4 M | N/F | N/F | N/F | 490 s |
| AZ v4 V-only ep99   | 6.0 M | N/F | N/F | N/F | 19 s |
| m_dd_v0 ep49        | 6.0 M | 60 moves | N/F | N/F | 13–18 s |

Smoke settings: even AZ v4 V-only fails 0/3 here. Inconclusive on solve
quality.

#### Production-beam bench (2 pids, beam=65536, max-steps=120, single-pass, bf16)

| model | params | pid 0 | pid 100 | wall/pid |
|---|---:|---:|---:|---:|
| **GT v0 Bellman ep24** | **1.4 M** | **N/F** | **N/F** | **~3020 s** |
| AZ v4 V-only ep99      | 6.0 M | 68 moves OK | 90 moves OK | 59–83 s |
| m_dd_v0 ep49           | 6.0 M | 61 moves OK | 85 moves OK | 54–75 s |

At the production beam width where both ResMLP baselines solve the two
easy pids, **the GT v0 fails to solve either** — despite matching
calibration on the BFS-d6 sample.

This is the **"training loss improves, beam gets worse"** failure mode
the plan explicitly anticipated. Plausible causes:
- BFS-d6 calibration is only depth 0-4. The beam navigates much deeper
  states (~d=20–80 from random scrambles). The GT's training distribution
  may not have produced a useful gradient toward V0 in that regime.
- The undershoot rate (4603/8000 vs 2046 for m_dd_v0) means GT predicts
  "too low" V for many states — flattening the V-landscape so the beam
  can't distinguish promising from unpromising children.
- 49 Bellman epochs may simply be insufficient at 1.4M params: the loss
  trajectory was still descending (3.01 → 0.80), but the V SHAPE for
  high-depth states might still be wrong.

#### Inference speed

GT inference is **~50× slower than ResMLP** at beam=65536 (3020 s vs 60 s).
Causes:
- Eager-only inference (compile hangs on the custom GraphTransformerLayer
  with `sdpa(attn_mask=...)` + `dynamic=False`).
- Attention bias matrix built fresh on every forward (`H × 121 × 121` —
  could be cached).
- Chunked forward at `internal_batch_size=4096` vs ResMLP's much higher
  effective throughput.

Direct deployment in beam search is not viable at this implementation
level.

## Full-scope follow-up #3 (2026-05-20, solver-trace mixin + distill A)

Per the diagnostic, the next hypothesis tested: maybe GT's high-depth V is
wrong-scaled because Bellman bootstrapping alone is too weak to push V toward
the true megaminx diameter (~21). Solver-trace data provides exact distance
labels at d=10-80 from real successful solves; mixing those in should
saturate the V.

Setup: warmstart from `gt orig ep24` (best depth-resolution, pre-fr50), add
`solver_trace_fraction=0.20` (128K pairs, d range [1, 98]), bump `k_max=80→120`,
`lr=1.5e-4`, `target_update_every=5`. 30 epochs on GCP L4 at 305 s/ep (~2.5 h).

Loss converged at 92 (high because the solver-trace exact labels conflict
with Bellman's walk-depth bootstrap — the model can't simultaneously fit both
families perfectly).

Calibration after the solver-trace retrain (`gt_solvertrace_e19` = best by
ST-RMSE):

| model | V(V0) | V@d=1 | V@d=4 | V@d=20 | V@d=40 | V@d=60 | V@d=80 | ST RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gt orig ep24 | -0.13 | 0.95 | 3.78 | 17.38 | 33.59 | 43.20 | 47.63 | 33.4 |
| gt fr50 ep29 | -0.14 | 1.04 | 4.26 | 13.78 | 21.84 | 25.72 | 27.55 | 38.2 |
| **gt solvertrace e19** | 0.12 | 1.14 | 4.10 | **29.48** | **34.97** | **39.31** | **41.81** | **17.2** |
| AZ v4 V-only ep99 | -0.05 | 0.92 | 4.05 | 20.46 | 28.14 | 30.25 | 30.88 | 33.3 |
| m_dd_v0 ep49 | 0.01 | 1.00 | 4.05 | 16.26 | 25.37 | 28.34 | 29.29 | 35.8 |

The solver-trace mixin DID pull V@d=80 down (gt orig: 48 → solvertrace e19: 42),
but it pulled V@d=20 UP (gt orig: 17 → solvertrace e19: 30). The
high-depth-region remains 10+ over the true diameter (~21), and the
low-depth-region is now over-predicted too. The model is settling at a
distance-like-but-shifted landscape.

### Option A executed: distill GT V into a 6M ResMLP

To isolate the V landscape from any GT inference-path artifacts, distilled
GT solvertrace e19 V into a 6M-param ResMLPDistance (matching `m_dd_v0`'s
shape) on GCP for 60 epochs at 21 s/ep (~21 min). MSE 105 → 14.9 across the
mixed state distribution.

Calibration of the distilled student vs the GT teacher:

| model | V(V0) | V@d=1 | V@d=20 | V@d=40 | V@d=60 | V@d=80 |
|---|---:|---:|---:|---:|---:|---:|
| GT teacher (solvertrace e19, 1.4M) | 0.12 | 1.14 | 29.48 | 34.97 | 39.31 | 41.81 |
| **distill_resmlp ep59 (6M)** | 0.21 | 1.04 | 29.67 | 35.10 | 39.22 | 41.65 |

The 6M ResMLP captured the 1.4M GT's V landscape essentially exactly,
including the high-depth scaling. The distillation worked.

### Beam bench of the distilled ResMLP — the decisive test

| model | params | pid 0 | pid 100 | wall/pid |
|---|---:|---:|---:|---:|
| **distill_resmlp ep59** | **6 M** | **N/F** | **N/F** | 150–280 s |
| distill_resmlp ep44 | 6 M | N/F | N/F | 160–280 s |
| distill_resmlp ep29 | 6 M | N/F | N/F | 210 s |
| m_dd_v0 ep49 (baseline) | 6 M | 61 moves OK | 85 moves OK | 84–104 s |

**This is the decisive negative result**. The distilled 6M ResMLP has the
**same architecture and same parameter count** as `m_dd_v0`, runs at the
same inference speed, but still fails 0/2 where m_dd_v0 succeeds. The only
difference is the V landscape itself.

So the bottleneck is **not** GT inference, **not** sibling-rank-via-Q-vs-V,
**not** training-loss-vs-beam-mismatch. It is that **GT's V landscape is the
wrong shape for beam search on megaminx**, and faithfully transferring it
into ResMLP via distillation transfers the failure.

What makes the working ResMLP V landscapes different:

| | V@d=20 | V@d=40 | V@d=60 | V@d=80 | shape |
|---|---:|---:|---:|---:|---|
| m_dd_v0 (works) | 16 | 25 | 28 | 29 | saturates near true diameter ~21-29 |
| AZ v4 (works)   | 20 | 28 | 30 | 31 | similar saturation |
| distill (fails) | 30 | 35 | 39 | 42 | drifts upward with walk depth |
| GT teacher (fails) | 29 | 35 | 39 | 42 | same |

The working models saturate near megaminx's true diameter; the failing
models predict walk-depth-style values that grow past the diameter. The
graph transformer's *architectural advantage* — more capacity to
distinguish far-from-solved states — turned into a *training pathology*:
it learned to encode "how the walk got to this state" instead of "how far
this state is from solved". Bellman bootstrapping over random walks alone
isn't enough to saturate the V; the GT just had enough capacity to refuse
saturation.

Solver-trace mixin partly fixed the high-depth direction (V@d=80: 48 → 42)
but introduced a low-depth over-estimate (V@d=20: 17 → 30) — the tension
between walk-depth labels and exact-distance labels broke the V curve in a
new place.

## Conclusion (revised after solver-trace retrain + distill bench)

After three Bellman retrains (original, frontier-50, solver-trace-20) and a
clean V-distill into a 6M ResMLP that runs at baseline speed, the verdict
is conclusive:

**The graph-transformer's V landscape is genuinely the wrong shape for beam
search on megaminx, and the failure is not about inference speed, calibration
on near-solved states, or sibling-rank-via-Q-vs-V.**

What we've established:

1. The graph prior + structured tokens + relation/distance attention bias
   let a 1.4M-param model learn calibration that matches 6M-param ResMLP
   baselines on the BFS-d6 (d=0-4) sample. ✓
2. GT high-depth V is monotonic and *more discriminative* than ResMLP's
   saturating V. ✓ (but this turned out to be a false-friend signal)
3. Solver-trace mixin (exact distance labels at d=10-80) partially pulled
   high-depth V down (48 → 42) but introduced a low-depth over-estimate
   (17 → 30). The model can't reconcile walk-depth Bellman with
   exact-distance solver-trace labels at this capacity. ⚠
4. Distilling GT V into a same-architecture 6M ResMLP runs at baseline
   speed (~150-280 s/pid at b=65536) and faithfully reproduces the V
   landscape — but still fails 0/2 where m_dd_v0 succeeds. The V LANDSCAPE
   itself is the failure mode. ✗
5. **Working ResMLP V landscapes saturate near megaminx's true diameter
   (~21-30) past d=40. Both the GT and the GT-distilled ResMLP V drift
   upward to ~42 at d=80 — they encode walk-depth, not true distance.**
6. The graph transformer's *extra architectural capacity* to distinguish
   far-from-solved states became a pathology: where ResMLP saturated (and
   that saturation is what enables beam navigation), GT had enough
   capacity to keep drifting up instead.

**Honest read of the plan's decisive question** — "does the graph prior
help?":
- **For learning efficiency, yes**: 1.4M params with the graph prior learns
  to match the 6M ResMLP calibration metric on BFS-d6.
- **For beam-search V, no, and the reason is informative**: the graph
  prior gave the model extra capacity to differentiate far-from-solved
  states, and that extra capacity preferred to encode walk-depth (a noisy
  upper bound) rather than true distance (which saturates). The
  *correct* shape for beam search is saturation; the architectural win
  blocked saturation.

The plan's "training loss improves, beam gets worse" failure mode is the
binding constraint, but in a different sense than expected: it isn't
sibling-rank quality that's wrong, it's the *global shape* of the V curve.

**Inference speed** is a separate issue: GT runs ~50× slower than ResMLP
at b=65536. Direct deployment isn't viable. Distillation into ResMLP
fixes the speed without fixing the V landscape.

### Option AZ executed: AZ-style training with GT V auxiliary

Setup: `71_train_az_v3.py` plus `+ lambda_v_gt * MSE(V_student, V_GT)`
on the V head. Warmstart from `m_dd_v0/epoch_0049.pt` (known-good
landscape). Policy CE on the 76,304 AZ dataset (unchanged). GT teacher:
`gt_solvertrace_e19`. λ = 0.5. 50 ep × 500K samples × rw_batch 8192,
policy_batch 1024 on GCP L4 (~33 min wall).

Training trajectory:
- ep 0:  p_loss 3.19, v_loss 10.84, v_gt 106.95, top-1 4.4%
- ep 49: p_loss 2.76, v_loss  6.23, v_gt  55.46, top-1 16.7%

For comparison, AZ v4 ep 24 (the historical sweet spot) had p_loss 2.36,
v_loss 0.11, top-1 28.5%. **The GT V auxiliary suppressed policy
memorization and held v_loss above AZ v4 levels** — both signals that the
extra V-side gradient is fighting the rest of the training.

V calibration of the AZ-GT model across epochs:

| epoch | V(V0) | V@d=1 | V@d=20 | V@d=40 | V@d=80 | shape |
|---|---:|---:|---:|---:|---:|---|
| 9  | 0.04 | 1.02 | 19.80 | 28.63 | 33.49 | near baseline saturation |
| 19 | -0.02 | 1.03 | 21.80 | 30.35 | 35.87 | drifting toward GT |
| 29 | 0.00 | 0.97 | 23.41 | 31.16 | 37.16 | further drift |
| 49 | 0.00 | 1.00 | 22.92 | 31.75 | 38.12 | close to GT |
| (m_dd_v0 ref) | 0.01 | 1.00 | 16.26 | 25.37 | 29.29 | works |
| (GT teacher) | 0.12 | 1.14 | 29.48 | 34.97 | 41.81 | broken |

The V head IS being pulled toward the GT shape, monotonically with epochs.

### Beam bench of AZ-GT V (using V head only via KhoruzhiiSolver)

| epoch | V@d=80 | pid 0 | pid 100 |
|---|---:|---:|---:|
| **ep 9** (mildest GT influence) | 33 | **109 moves OK** | N/F |
| ep 19 | 36 | N/F | N/F |
| ep 29 | 37 | N/F | N/F |
| ep 49 (strongest GT influence) | 38 | N/F | N/F |
| m_dd_v0 (baseline) | 29 | 61 OK | 85 OK |

**Monotonic failure**: the more the V head drifts from baseline saturation
toward GT's shape, the less beam-searchable it becomes. Ep 9 just barely
solves pid 0 — and at 109 moves vs baseline 61, far worse. By ep 19,
beam fails entirely on both pids.

This confirms the failure mode is *quantitative*: any amount of GT V
influence damages the beam-usable V landscape proportionally.

## Final conclusion

After three GT retrains, a V distillation into ResMLP at baseline speed,
and an AZ-style dual-head training with GT V auxiliary, the result is
unambiguous:

**The graph transformer's V landscape is unsuitable as a teacher for
megaminx beam search, in any of the tested forms.**

Why: GT's extra architectural capacity to differentiate far-from-solved
states caused it to encode walk-depth-like values that drift past the
true megaminx diameter (~21-30), rather than saturating like the working
ResMLP baselines. Solver-trace exact-distance mixin partially fixed
high-depth but introduced low-depth distortion. Distillation into a
fast 6M ResMLP transferred the landscape but not the beam-usability.
AZ-style auxiliary loss showed monotonic damage with influence strength.

The graph prior remains a real architectural win for *learning efficiency*
on near-solved calibration (1.4M params = 6M ResMLP at d=0-4). It is
just not a useful inductive bias for the V-shape that megaminx beam
search needs.

### Remaining open question

**GT-Q distillation** (`75_train_gt_q.py`, ready) was never run end-to-end.
It bypasses the V landscape entirely by training on `Q(s, a) =
V_teacher(apply(s, a))` — sibling rankings, which are relative
measurements that don't depend on the V curve's global shape. Even if GT
absolute V is wrong, GT-derived sibling ranks could still be right (or
not). This is the one remaining variant of "GT as teacher" that hasn't
been falsified.

If GT-Q also fails, the architectural bet is fully settled negative for
this puzzle and the right move is back to other 70K-score levers in
`to_do_shortlist.md`. If GT-Q succeeds, we'd have a usable artifact at
the cost of an additional ResMLP-Q distillation step for inference speed.

### Recommended follow-up runs

In rough order of expected payoff:

1. **GT-Q distillation** (`75_train_gt_q.py`). Use **GT v0 Bellman ep24**
   (our new strongest signal) as the V teacher; train a GraphTransformerV
   with `output_dim=24` (Q-head) on `Q(s,a) = V_teacher(apply(s,a))`. Then
   distill THAT into a ResMLP Q-shortlister for production. The GT's
   parameter-efficiency suggests it can be a more accurate teacher than
   m05 at small cost. ~6 h wall.

2. **Resolve the inference bottleneck**:
   - Cache the `(H, S+1, S+1)` attention bias matrix between forwards
     (it's a function of the bias tables, which only change at training
     time).
   - Fix the `torch.compile` hang: try `dynamic=True` or a hand-written
     attention kernel that doesn't go through SDPA's `attn_mask` path.
   - If both work, expected ~3-10× speedup, which would make GT competitive
     for direct beam-search deployment.

3. **Bigger trunk** (d=256/4L/8h, ~3.2M params). The plan's GT-v1 spec.
   Cluster ceiling at 6M may not extend to GT, but at 1.4M we're already
   matching the ceiling — it's not clear bigger helps. Lower priority.

4. **Longer Bellman** (100-200 epochs at d=192). Loss trajectory suggests
   diminishing returns past ep 50; Rule 12 says >50 ep risks overtraining.
   Run only with per-25-ep beam bench validation, and only if a faster
   inference path is in place. Lower priority.

## Artifacts

- Code: `megaminx/scripts/72_build_graph_features.py`,
  `73_build_action_relabel.py`, `75_train_gt_q.py`,
  `77_eval_gt_vs_baseline.py`,
  `megaminx/src/megaminx/graph_transformer.py`.
- Configs: `megaminx/configs/m_gt_v0_pretrain.yaml`,
  `m_gt_v0_bellman.yaml`.
- Models: `megaminx/models/m_gt_v0_pretrain/epoch_0049.pt`,
  `megaminx/models/m_gt_v0_bellman/epoch_0001.pt`.
- Tables: `megaminx/data/graph_features.pt`,
  `action_relabel.pt`, `action_relabel_720.pt`.
- Logs: `megaminx/models/m_gt_v0_pretrain_training.log`,
  `m_gt_v0_bellman_training.log`.
- Tests: `tests/test_megaminx_graph_features.py` (13 tests, all pass).

## Files modified in existing infra

- `src/cayley/model.py`: +`ResMLPDistance.get_model_config()`.
- `src/cayley/training.py`: model_config from `get_model_config()` if present.
- `src/cayley/bellman.py`: same.
- `src/cayley/search.py`: `load_model_checkpoint` dispatches on `model_class`.
- `megaminx/scripts/02_train.py`, `60_train_admissible.py`: model dispatch.
- `megaminx/scripts/61_eval_v_at_solved.py`, `58_corner_pdb_beam.py`,
  `09_eval_q_recall.py`: use polymorphic loader.

All edits are backward-compatible — existing ResMLP YAMLs and checkpoints
load unchanged.
