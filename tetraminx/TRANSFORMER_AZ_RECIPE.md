# PieceTransformer + AZ dual head — the full training recipe

The deployed tetraminx scorer: **`models/mx_tf_az/epoch_1500.pt`**, 3,444,761 params,
trained from scratch for 2000 epochs on the sparse-Q objective. This file consolidates
what was previously spread across `configs/mx_tf_az.yaml`, `HANDOFF.md`,
`BLOG_tetraminx_progress.md`, `RESMLP_VS_TRANSFORMER.md` and `TORCHTPU_TRAINING.md`.
Written 2026-09-04 from the code, the configs, the checkpoint's own stored
`train_config`, and fresh measurements where a source doc disagreed with the code.

**It is ONE continuous stage.** No pretrain, no Bellman refine, no warm start —
`mx_tf_az.yaml` has no `warmstart_base`, and `51_train_sparse_q.py` only reads one if the
config sets it. The staged pipeline in this repo is the *ResMLP V* line (`tv0_pretrain`
Stage A -> `tv0_bellman` Stage B); the transformer does not use it.

> **`AZ_HEAD_COMPARISON.md` (repo root) is about a DIFFERENT model.** That file covers
> `21_train_az.py` / `taz_v1`: a ResMLP warm-started from a Bellman V, with a 24-logit
> *policy* head trained by cross-entropy on solution paths. Here, `az_head: true` adds a
> **scalar value head** to the transformer trunk inside the sparse-Q trainer. Different
> script, different head, different objective.

---

## 1. Data

### 1.1 What is generated, what is on disk

**There is no static training set for the main objective.** Random-walk pivot states are
generated on the training device every step by `SparseQSampler` (`51_train_sparse_q.py`),
so the model never sees the same state twice and there is no dataset to version. Only the
*anchor* supervision is precomputed, and it comes from exact BFS.

| artifact | built by | size | used for |
|---|---|---|---|
| `data/puzzle_info.json` | competition input | 6.8 KB | `central_state` + 24 generators; the root of everything below |
| `data/test.csv` | competition input | 1000 rows | evaluation only — **never** touches training |
| `data/tetra_symmetries.npy` `_inv.npy` `tetra_move_relabel.npy` | `01_build_symmetries.py` | 4.4 KB each | the 24 frames + action transport `sigma` |
| `data/piece_layout.json` | `50_derive_piece_layout.py` | 4.5 KB | the 50-piece tokenisation |
| `data/bfs_d6_train.pt` | `03_build_bfs.py` | 3,771,577 states int8 | anchor STATES (filtered to d <= 5 at load) |
| `data/bfs_endgame.npz` | `03_build_bfs.py` | 27,779,749 hashes + int8 depths | the d <= 6 lookup that labels anchor children |
| `data/baked_anchors_d5.npz` | `45_bake_anchors.py` | 1,771,577 x (88 state, 24 Q, depth) | TPU only — the same labels, precomputed |

Build order (each depends on the previous):

```bash
python tetraminx/scripts/01_build_symmetries.py
python tetraminx/scripts/50_derive_piece_layout.py \
    --puzzle tetraminx/data/puzzle_info.json --out tetraminx/data/piece_layout.json
python tetraminx/scripts/03_build_bfs.py --max-depth 6 --full-depth 5
# TPU only:
python tetraminx/scripts/45_bake_anchors.py --train tetraminx/data/bfs_d6_train.pt \
    --endgame tetraminx/data/bfs_endgame.npz --max-depth 5 \
    --out tetraminx/data/baked_anchors_d5.npz --verify
```

### 1.2 The 24 symmetries

`01_build_symmetries.py` solves `P^-1 . g_{sigma(m)} . P = g_m` for every generator, which
gives 24 relabelings P of the 88 facelets plus the move-relabel table `sigma`. The
consequence used in training is

```
apply(conj(s), m) == conj(apply(s, sigma(m)))
```

so a label on action `a` of state `s` transports to action `sigma^-1(a)` of `conj(s)`.
`verify_symmetry: true` asserts this at startup over all 24 frames x 24 actions before a
single gradient step — it has caught nothing, and it costs seconds, so leave it on.

The group permutes the 4 axes but **not** the 3 layers, so an action's orbit has 8
members, not 24. That is why one label pair covers at most 16 of 24 columns (section 1.5).

Inverse antisymmetry (solve `s^-1`, then reverse+invert the path) doubles this to 48
usable *search* frames, but it is an inference-side trick only — training uses the 24
spatial frames.

### 1.3 The piece layout — where the tokens come from

`50_derive_piece_layout.py` derives the physical-cubie partition from the generators
alone: facelets rigidly attached to the same piece have the **same stabilizer**, so group
slots by stabilizer and then verify the result is a genuine block system (every generator
maps each block onto some block). For tetraminx that yields:

| | count |
|---|---|
| pieces (tokens) | **50** |
| size-3 pieces | 4 |
| size-2 pieces | 30 |
| size-1 pieces (centres) | 16 |
| `max_piece_size` | 3 |
| `num_piece_types` | 3 |

Validated by re-running it on the IHES cube, where it reproduces cayleypy-training-core's
hand-written 26-piece layout exactly. Upstream hard-codes layouts for megaminx and IHES
only; this script is what made a PieceTransformer possible here at all.

### 1.4 The BFS anchor tables

Level sizes measured from solved: 24 / 408 / 6,592 / 105,136 / 1,659,416, branching ~15.8.

The cap is structural: `51_train_sparse_q.py` asserts `max_anchor_depth < table_depth`,
because **every child of a d <= k state is at d <= k+1** and must be findable. With the d6
table (27,779,749 states) that puts anchors at **d <= 5 = 1,771,577 states**, and every one
of their 24 children resolves to a true distance. `BFSAnchors.sample` asserts a total hit
rate — a miss is a corrupt table, not a soft failure.

`bfs_d6_train.pt` holds 3.77M states (full through d5 plus a d6 sample); the `d <= 5`
filter at load is what selects the 1.77M anchors.

**d7 was built and rejected by measurement.** `03b_build_bfs_deep.py` produced
`bfs_endgame_d7.npz` (433,385,579 states, 3.90 GB), which would lift anchors to d <= 6 =
27.8M states, 15.7x more. The `tq3_d7anchor` A/B found **no improvement**, and the same
run priced d8 at zero. The d7 table still earns its keep on the *search* side (MITM radius
12-13 rewriting), just not as anchor supervision. Do not re-run this ablation.

### 1.5 What one training step actually consumes

Per step, from `batch_size: 512`:

| rows | count | labels each | labels |
|---|---|---|---|
| random-walk, symmetry-expanded | 512 x 14 = **7,168** | 0-2 (sparse; mean 13.57 per *sample*) | ~6,945 |
| BFS anchors | **512** | 24 (dense, exact) | 12,288 |
| **total** | **7,680** | | **~19,233** |

**Anchors are 6.7% of the rows but ~64% of all Q supervision.** They are dense where walk
labels are 2-of-24, and they are exact where walk labels are merely an upper bound.

> **Correction to `BLOG_tetraminx_progress.md`**, which states 46%. That figure assumes 2
> labels on each of the 14 symmetry rows (14,336 rw labels). Measured from
> `Symmetries.coverage_table()`: the greedy expansion emits **13.57 labels per sample on
> average** (min 8, max 16), and only **9.15 of the 14 rows carry a label at all** — the
> rest are padding to a static shape. Those rows are still forward-passed and still train
> the *value* head (section 3.3); they just contribute nothing to the sparse Q loss.

Per epoch: 1024 steps x 512 samples = **524,288 pivot states**, ~7.86M rows.
Over 2000 epochs: **~1.05B pivot states**, none of them repeated.

### 1.6 On disk but NOT used by this cell

- `data/path_labels_28821.pt` — `PathLabels` supervision from our own solution paths.
  Wired into the trainer (`path_mse_w`, `path_rank_w`) but **both weights default 0.0** and
  `mx_tf_az.yaml` does not set them. See `72_path_rank_probe.py`.
- `data/az_dataset.pt`, `data/solver_trace.pt` — inputs to `20_build_az_dataset.py` /
  `21_train_az.py`, i.e. the *other* AZ model (`taz_v1`). Not this one.
- `data/b6_front.npz`, `bfs_endgame_d7.npz` — search-side MITM rewriting.
- `box_*` / `profile_*` knobs in the trainer — all default 0.0, see
  `TRIANGLE_BOX_REJECTED.md`.

---

## 2. The model

`tetraminx/src/tetraminx/models.py::PieceTransformerQ`. One token per physical piece,
carrying the stickers currently on it — *not* one token per facelet.

| | |
|---|---|
| tokens | 50 pieces + 1 CLS = **T = 51** |
| `d_model` | 256 |
| `nhead` | 8 (head_dim 32) |
| `num_layers` | 4 |
| `ff_dim` | 1024 |
| `dropout` | 0.0 |
| activation | silu |
| params | **3,444,761** (3,444,504 without the value head) |

Structure: folded input stage -> CLS concat -> LayerNorm -> 4 x pre-norm encoder block ->
output LayerNorm -> read CLS -> `head: Linear(256, 24)` and `value_head: Linear(256, 1)`.

**Both heads come from ONE trunk pass.** `model.return_value = True` during training, so
the dual-head cell is not silently charged a second forward.

Three implementation details that are load-bearing, not incidental:

1. **Folded input stage.** `piece_projection` is pushed into the value table, so a token is
   `sum_j table[j, v_j] * mask_j + bias`. This never materialises the `(B, P, K, D)`
   intermediate — 2.3 GB at training batch, ~10 GB at TPU batch. Required to run, not an
   optimisation.
2. **One-hot matmul, not `index_select`.** A gather's backward is a scatter-add, and 384k
   gradients per slot contended atomically for an 88-row table. Freezing the input stage
   alone took the A100 step 472 -> 158 ms, i.e. the embedding gradient was **67% of the
   entire step**. Recast as `(N, C) @ (C, D)` and both directions are dense GEMMs.
3. **`_SelfAttn`, deliberately not `nn.MultiheadAttention`.** MHA takes the math path in
   training mode and saves a `(B, H, T, T)` tensor per layer for backward. Parameter
   *names* are kept identical (`in_proj_weight`, `in_proj_bias`, `out_proj.*`) so
   checkpoints and the JAX loader in `jax_model.py` keep working unchanged.

Attention is only **3.2%** of this model's MACs at T=51 — the cost is the per-token FF
(64.5%) and the QKV/output projections (32.3%). Consequence: cheaper-attention research
cannot help this model.

`attn_impl` is `"sdpa"` (default, GPU — gets flash attention) or `"einsum"` (TPU — 1.69x on
fwd+bwd there). Same parameters, `state_dict` interchanges `strict=True`, CPU fp32
agreement 5.8e-07. Set it per platform; it is not a model change.

---

## 3. The objective

### 3.1 Sparse-Q — the core

Port of Vlad Kuznetsov's random-walk-middle objective
(`github.com/AnanasClassic/cayleypy-training-core`). Walk `k` steps from solved
(non-backtracking), pick a pivot `p` in the middle, label exactly **two** of the 24
actions:

```
Q(s, undo_last_move) = p - 1        Q(s, next_walk_move) = p + 1
```

Everything else is masked out of the loss.

**Why this beats regressing V on walk depth.** An MSE against `k` has large conditional
variance at depth, so the optimal prediction shrinks toward the mean and *local*
discrimination flattens. Here the two labels always differ by exactly 2 with **zero
conditional variance**, so the loss cannot be reduced by flattening the gap — only the
absolute level is free to saturate. Measured on `tv0_bellman` ep24, the V's mean gap
`V(next) - V(undo)` decays from 1.91 at depth 2 to 0.75-1.03 at depth 22-28; the sparse-Q
head holds ~1.42.

The masked mean is written as sum/count, not `masked_select` — the latter has a
data-dependent output size, which on TPU is a host round-trip plus a recompile per
distinct count.

### 3.2 Our four additions over upstream

1. **Exact 24-way BFS anchors** (section 1.4). `anchor_weight: 1.0`, `anchor_batch: 512`,
   `max_anchor_depth: 5`.
2. **Symmetry-expanded label coverage** (sections 1.2, 1.5). `sym_coverage: true`,
   `sym_rows: 0` = use the full greedy width (14). Row 0 is always the identity frame, so
   the original sample is always present.
3. **Depth-tilted pivot sampling.** `pivot_tilt: 0.5` -> `gamma = 1/(1+tilt)`, pivots drawn
   as `u^gamma * (k-1) + 1`. Upstream's plain scheme is heavily shallow-weighted; our paths
   are 28-36 moves long. Measured histogram at tilt 0.5 spans depth 1-39, ~123 samples at
   d1 falling to ~6 at d39.
4. **`top1_margin_weight` as a live knob.** Upstream ships it at 0.0 and so do we
   (`top1_margin: 1.0`, `top1_margin_weight: 0.0`) — it is available, not used. The margin
   term is still *reported* every epoch as a metric.

**The `k_max` trap.** Upstream uses `k_max = 70` against a ~29 diameter. That only survives
because its pivot scheme keeps the mass shallow. Past the diameter a walk is
near-stationary and "undo the last move" stops being reliably distance-reducing — the
**ranking label itself** degrades, not merely its scale. Since `pivot_tilt` pushes mass
deeper, `k_max` must come down to compensate. We run **`k_max: 40`, `k_min: 2`, tilt 0.5**.
`k_max`/`pivot_tilt` is the first ablation axis if a variant underperforms.

### 3.3 The AZ value head

`value_weight` defaults to 1.0 and the config does not override it. The target is the
classic V objective, on the same rows already in the batch:

| rows | value target |
|---|---|
| random-walk (all 7,168, including padded symmetry rows) | the sample's pivot depth `p` |
| BFS anchors (512) | the **exact** distance `d0` |

Distance is conjugation-invariant, so every symmetry row inherits its source sample's
pivot — `src_row = arange(n_rw) // used`.

### 3.4 Total loss

```
loss = sparse
     + anchor_weight   * anchor_mse      # 1.0
     + margin_weight   * margin_loss     # 0.0  -- off
     + path_mse_w      * path_mse        # 0.0  -- off
     + path_rank_w     * path_rank       # 0.0  -- off
     + value_weight    * value_loss      # 1.0
     + box                               # 0.0  -- off
```

Four of the seven terms are zero in this recipe. They are live knobs for other cells; do
not read their presence in the trainer as part of this recipe.

---

## 4. Resolved hyperparameters

From `epoch_1500.pt`'s own stored `train_config` — the config file says 1500 epochs and
`checkpoint_every_epochs: 50`; the actual run used `--set` overrides.

```yaml
n_epochs: 2000              # config file says 1500; extended, see section 5
steps_per_epoch: 1024
batch_size: 512             # samples BEFORE symmetry expansion (x14 rows after)
k_min: 2
k_max: 40
pivot_tilt: 0.5
lr: 3.0e-4                  # constant, no schedule
weight_decay: 3.0e-3
grad_clip: 1.0
sym_coverage: true
sym_rows: 0
verify_symmetry: true
anchor_batch: 512
max_anchor_depth: 5
anchor_weight: 1.0
top1_margin: 1.0
top1_margin_weight: 0.0
val_size: 2048
checkpoint_every_epochs: 10   # 50 in the yaml; the run used 10
amp: true                     # bf16 autocast
compile_model: true
fused_optimizer: true
seed: 0
```

Optimizer is AdamW (fused on CUDA). No LR schedule — constant 3e-4 for all 2000 epochs.

**`compile_model: true` is correct here.** An earlier note in this repo blamed a
`torch.compile` hang (CLAUDE.md Rule 22). That was a misdiagnosis: the real cause was
memory — batch 512 peaks at 17.3 GB and a 16.4 GB Windows card **pages via WDDM instead of
OOMing**, taking the step from 340 ms to 3,617 ms with no error and no traceback.
`torch.compile` costs 18.6 s of warmup on the A100 and is ~12% faster.

### Where it ran

Spot A100. Steady state **142.7 s/epoch** (mean of epochs 10-30), total logged wall
**79.9 h** over 2,011 log rows.

> The **159 s/epoch** in `RESMLP_VS_TRANSFORMER.md` is epoch 1, which carries compile
> warmup. 142.7 is the number to compare against.

`train_log.csv` has 2,011 rows for 2,000 epochs — **10 duplicate epoch numbers, which are
spot-preemption resume points, not training phases.** (Cell 3, `models/mx_tf/`, has 94.)
`--resume <ckpt>` restores model + optimizer + epoch. Spot VMs created with
`--instance-termination-action=STOP` stay stopped after preemption and GCP never restarts
them; the boot `startup-script` that resumes from the newest checkpoint is what makes an
80-hour run survivable.

---

## 5. Why epoch 1500 — and why 2000 epochs bought nothing

Every 100th checkpoint was beam-evaluated on the same 15 stratified pids at Q@1M, 1
forward frame, so the whole curve is comparable.

**Descent phase** (cell 4 = transformer + AZ, vs cell 3 = transformer, no AZ):

| ep | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| cell 4 total | 459 | 448 | 445 | 438 | 443 | 442 | 436 | 440 | **432** | **432** |
| cell 3 total | 444 | 440 | 446 | 447 | — | 449 | 438 | 443 | — | — |

**Plateau** (the 1500 -> 2000 extension):

| ep | 1200 | 1300 | 1400 | **1500** | 1600 | 1700 | 1800 | 1900 | 2000 |
|---|---|---|---|---|---|---|---|---|---|
| total | 426 | 428 | 431 | **424** | 426 | 427 | 425 | 428 | 428 |
| avg>1 | 30.36 | 30.50 | 30.71 | **30.21** | 30.36 | 30.43 | 30.29 | 30.50 | 30.50 |

Band 424-431, mean ~426.4, **no trend across 800 epochs**, while training loss kept falling
the whole way (13.88 at ep1500 -> 13.79 at ep2000; `top1_acc` 0.5119 -> 0.5158). Eight
checkpoints past ep1500 and not one beat it. **The training axis is closed** — ep1500 wins
by a 2-move gap this bench cannot resolve, so the honest reading is "ep1500 is as good as
anything past it", not "ep1500 is optimal".

**Two findings worth keeping:**

- **The AZ head's main value is trainability, not the value signal.** Cell 3 goes flat
  around ep200-400; cell 4 keeps descending for another ~800 epochs, ending 424 vs 438.
  The beam never reads the value output by default. Confirmed on the 30-pid discriminating
  set: cell 3 ep700 = 951, cell 4 ep700 = 945, cell 4 ep900 = **938**.
- **Per-pid noise between adjacent checkpoints is +-3**, so a 4-move total on 15 pids is
  inside noise. Use the 30-pid set to discriminate; the 15-pid `avg>1` and the 30-pid mean
  are **not** comparable (31.3 vs 29.9 — the 30-pid set is genuinely harder).

For contrast, cell 3's own selection is in `models/mx_tf_best/README.md`: `best.pt` =
epoch_0700, and that model is converged by ~epoch 200 — the transformer reaches full
quality in ~13% of the budget the ResMLPs needed.

---

## 6. Verifying a checkpoint

`scripts/46_verify_checkpoint.py` scores against labels known exactly — the d <= 5 anchor
pool plus the d=1 states:

| | A100 ep200 | **A100 ep1500** | TPU ep5 (reference point) |
|---|---|---|---|
| undo ranked strictly best (d=1) | 100% | **100%** | 100% |
| MAE vs exact 24-way Q on anchors | 0.110 | **0.069** | 0.347 |
| argmin picks a TRUE optimal move | 98.9% | **99.7%** | 65.6% |
| einsum vs sdpa on the same weights | 1.20e-06 | 3.95e-06 | 1.29e-06 |

A correct checkpoint is **41,411,849 bytes** on CUDA (41,408,457 from TorchTPU — 62
tensors, 3,444,761 params, plus 62 optimizer entries). Size is the cheap first check.

> **The obvious sanity check is broken.** "All 24 children of solved are at d=1, so Q
> should be ~1.0" *false-fails the deployed model*: converged checkpoints predict 3.8-3.9
> on the solved state while a 5-epoch model predicts 0.48. The solved state is
> out-of-distribution — the sampler pivots at depth >= 1 and it appears exactly once in
> 1.77M anchors. It is kept as an informational line only. A threshold that fails your
> known-good model is a broken threshold.

---

## 7. Porting: TPU

`configs/mx_tf_az_tpu.yaml` is a matched re-run — every training field is identical except
two, and neither changes the math (`45_bake_anchors.py --verify` asserts the targets are
row-identical):

- `anchor_baked: baked_anchors_d5.npz` — precomputed instead of a live hash lookup.
- `anchor_resident: true` — the anchor pool lives in HBM (0.19 GB).

Plus `attn_impl: einsum` (section 2). Result: **132.0 s/epoch on a v6e-4 vs the A100's
142.7**. Full detail, including the four port changes and the DDP checkpoint deadlock, is
in `TORCHTPU_TRAINING.md`. `models/mx_tf_az_tpu5/` holds only `epoch_0005.pt` — that run is
a **benchmark, not a production model**.

---

## 8. Deploying it

```
PieceTransformer Q head (ep1500, 3.44M params, AZ dual-head)
  + ResMLP+AZ blended 0.8 / 0.2        (~1.07x cost, -2)
  + qv-consistency lambda 0.3          (free, -4, additive with history)
beam 16M sharded across 8 TPU cores, history_depth 1
2 frames: k0-forward + k1-inverse      (frames SATURATE at 2)
exact d<=6 endgame splice -> replay-verify -> n-way per-pid min
```

`--qv-consistency` scores `Q(s,a) + lam*|Q(s,a) - (V(s)-1)|` with **both heads from one
trunk pass** — genuinely free, and it recovers the entire ep700->ep1000 training gain
without training. This is the only place the value head is read at inference, and it is
what makes the AZ head pay twice.

Three gotchas that cost real time:

1. **`--chunk-size 4096`, not the old 32768 default.** Peak transient is linear in CHUNK and
   independent of beam width. Measured on this checkpoint (bf16 autocast, forward only):
   **1.42 GiB at 4096 vs 11.21 GiB at 32768**. On a 16 GB Windows card the big one does not
   OOM — WDDM silently spills to host RAM and the only symptom is wall clock (1.0 vs 9.2
   s/step at B=65,536). Throughput is flat over chunk 2048-8192. Results are bit-identical
   across chunk sizes (`max_abs_diff` exactly 0) because nothing in the scoring path
   reduces across rows.
   > `HANDOFF.md` explains this as an explicit `attn_mask` materialising a `(chunk, H, 88,
   > 88)` score matrix. That mechanism does not match this model: `_EncoderBlock` runs
   > **unmasked** self-attention over **T = 51**, chosen precisely so SDPA takes the fused
   > path. The fix and its measured numbers stand; the stated cause does not. The peak is
   > linear in chunk because every activation is.
2. **Never run more than 1 frame on the transformer.** Frames saturate at 2 and the inverse
   frame carries the win, not the spatial rotation (at 4M: forward 416, inverse 414, and
   inverse ran ~9% faster).
3. **`30_solve.py --history-depth` defaults to 0; the TPU driver uses 1** — worth ~4 moves
   on the 15-pid set. Match it on any local run being compared to a TPU number (CLAUDE.md
   Rule 28).

---

## 9. Reproducing

```bash
# GPU
python tetraminx/scripts/51_train_sparse_q.py \
    --config tetraminx/configs/mx_tf_az.yaml \
    --output tetraminx/models/mx_tf_az \
    --set training.n_epochs=2000 --set training.checkpoint_every_epochs=10

# resume after preemption
python tetraminx/scripts/51_train_sparse_q.py \
    --config tetraminx/configs/mx_tf_az.yaml \
    --output tetraminx/models/mx_tf_az \
    --resume tetraminx/models/mx_tf_az/epoch_XXXX.pt \
    --set training.n_epochs=2000

# TPU (see TORCHTPU_TRAINING.md; launch_ddp.sh sets TPU_DEFER_AND_FUSE=1)
bash ~/launch_ddp.sh /home/and-l/runs/<name>
```

Startup prints the resolved config, the verified symmetry transport, the coverage width,
the anchor counts and the param count — **read that header before trusting a "matched"
re-run** (CLAUDE.md Rule 13).

---

## 10. Closed — do not retry without new justification

| axis | verdict |
|---|---|
| more epochs past 1500 | 8 checkpoints, no trend across 800 epochs |
| deeper BFS anchors (d7) | `tq3_d7anchor`: nothing. d8 priced at zero in the same run |
| bigger / other architectures | ResMLP 468; GeneratorISAB recovers ResMLP quality but does not close the gap; GFlowNet-TB falsified at ep410 (flat flow, no terminal reward) |
| checkpoint averaging / model soup | beam-neutral — averaging relocates scorer errors, it does not reduce them |
| hand-built cycle-structure features | falsified: analytic invariants fall to or below chance past depth 10, the mirror of where the transformer's advantage lives |
| frames > 2 | saturate at 2 |
| `path_mse` / `path_rank` / `box` / `profile` terms | shipped default-off; see `TRIANGLE_BOX_REJECTED.md`, `72_path_rank_probe.py` |
| Beam-AVI bootstrap | rejected — probe passes, bootstrap still flattens Q-space |

The open axis is **not** training. `RESMLP_VS_TRANSFORMER.md` section 6 frames it: the
transformer is a better scorer per node and a much worse one per second, so the question is
*where the width curve saturates*, not which model is better in the abstract.

---

## Sources

- `configs/mx_tf_az.yaml`, `configs/mx_tf_az_tpu.yaml` — the configs
- `models/mx_tf_az/epoch_1500.pt` — stored `train_config` / `model_config` (authoritative)
- `models/mx_tf_az/train_log.csv` — 2,011 rows, per-epoch metrics and seconds
- `scripts/51_train_sparse_q.py` — objective, sampler, anchors, symmetry expansion
- `src/tetraminx/models.py::PieceTransformerQ` — the model
- `HANDOFF.md` — deployed stack, epoch curves, "training axis closed"
- `RESMLP_VS_TRANSFORMER.md` — architecture rationale and cost (transformer at ep500; stale)
- `TORCHTPU_TRAINING.md` — the matched TPU port
- `BLOG_tetraminx_progress.md` — narrative (its 46% supervision figure is corrected in 1.5)
- `models/mx_tf_best/README.md` — cell 3 (no AZ) checkpoint selection
