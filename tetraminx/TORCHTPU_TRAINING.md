# Training the PieceTransformer on TPU with TorchTPU

Measured 2026-08-24 on a GCP `ct6e-standard-4t` (4x v6e chips, europe-west4-a),
TorchTPU `0.1.1.dev20260824100827` / torch 2.13.0+cpu / libtpu 0.0.46.

**Result: 132.0 s/epoch against the A100's 142.7 -- 1.08x faster, and 1.78x faster
than the naive port.** Same model, same recipe, same samples/epoch, and the
checkpoint verifies against exact BFS ground truth.

Steady state is the mean of epochs 2-5 (131.0 / 132.9 / 133.3 / 130.8); epoch 1 is
135.9 with a warm tier-2/3 compilation cache, 146.0 cold.

## The bar

The A100 reference is `models/mx_tf_az/train_log.csv`, steady-state **142.7 s/epoch**
(mean of epochs 10-30). Note this is NOT the 159 s/epoch quoted in
`RESMLP_VS_TRANSFORMER.md` -- that figure is epoch 1, which carries compile warmup.

The comparison is matched: `epoch_0200.pt`'s stored `train_config` is byte-identical
to `configs/mx_tf_az_tpu.yaml` on batch_size 512, steps_per_epoch 1024, k 2-40,
pivot_tilt 0.5, lr 3e-4, weight_decay 3e-3, grad_clip 1.0, sym_coverage, anchor_batch
512, amp, compile_model, val_size 2048 -- and the same `model_config`. Global batch
stays 512 under DDP (128/rank x 4), so samples/epoch match exactly.

## Where the time went

| config | s/epoch | vs A100 |
|---|---|---|
| naive port (default eager, `compile(model)`) | 234.8 | 1.65x slower |
| + `TPU_DEFER_AND_FUSE=1` | 176.5 | 1.24x slower |
| + `attn_impl: einsum` | **132.0** | **1.08x FASTER** |

### Lever 1 -- `TPU_DEFER_AND_FUSE=1` (one env var, 234.8 -> 176.5)

TorchTPU's default execution mode is `DEFER_NEVER`: every op outside
`torch.compile` dispatches one at a time. A trainer is full of such code -- our
40-iteration random-walk sampler loop, `expand_labels`, the loss, the metrics.
Fused Eager defers and fuses them.

| | eager (default) | DEFER_AND_FUSE | torch.compile |
|---|---|---|---|
| data pipeline | 61.3 ms | **13.1 ms** | 29.7 ms |

It BEAT `torch.compile` on the same code -- many tiny ops fuse better than they
compile. Metrics came out byte-identical to the unfused run (epoch 1: loss
60.5967, pair 0.6222, top1 0.1211 both ways), so this lever is free.

### Lever 2 -- `attn_impl: einsum` (176.5 -> 132.0)

Attention was ~90% of model time on ~3% of the FLOPs. Reformulated so the heads
never leave dim 2, which removes the 5-D permute:

```python
q, k, v = qkv.view(B, T, 3, H, Dh).unbind(2)        # no permute
s = torch.einsum("bthd,bshd->bhts", q, k) * scale
p = s.float().softmax(-1).to(v.dtype)               # fp32 softmax, see below
a = torch.einsum("bhts,bshd->bthd", p, v)
```

25.14 -> **14.85 ms** per layer fwd+bwd (1.69x), x4 layers.

**This was nearly missed.** Judged on FORWARD time it looks worse (5.51 vs 4.56 ms)
and was rejected on that basis in a first pass. The permute is only 0.06 ms
forward; unwinding it around SDPA's saved tensors in the BACKWARD is the cost.
Judge any training-path rewrite on fwd+bwd.

Opt-in (`attn_impl`, default `"sdpa"`): on GPU, SDPA gets flash attention and
wins. Parameters are unchanged and `state_dict` interchanges `strict=True`, so a
checkpoint trained either way loads into the other.

**Numerics.** einsum and SDPA disagree rel 3.3e-3 on TPU, so both were checked
against a CPU float64 reference rather than against each other:

| | rel error vs CPU float64 |
|---|---|
| CPU float32 | 6.6e-07 (noise floor) |
| TPU fp32 SDPA | 4.836e-03 |
| TPU fp32 einsum | 4.555e-03 |
| TPU bf16 SDPA | 8.192e-03 |
| TPU bf16 einsum, naive | 1.505e-02 |

Both TPU paths sit at ~4.6e-3 -- that is the MXU's default 1-pass bf16 multiply
(`torch.tpu.precision`), not a defect in either formulation. On CPU fp32 the two
agree to **5.8e-07**, i.e. mathematically equivalent. Under bf16 the naive einsum
was worse because TorchTPU's fused SDPA accumulates in fp32 internally; promoting
only the softmax to fp32 (elementwise on `(B,H,T,T)`, nearly free) closes it.

## Why attention is the bottleneck at all

A GEMM roofline at our own shapes says the chip is fine:

| | TFLOP/s |
|---|---|
| square 8192^3 | 361 |
| QKV / FF GEMMs at our shapes | 212-259 |
| FF sublayer in situ | 235 |
| **attention (SDPA)** | **3.0** |

Every attention matmul is 51x32x51 on a 128x128 MXU -- roughly 2.5% tile
occupancy, because T=51 tokens and head_dim=32. That is arithmetic, not a bug, and
no reshape fixes it. It is why this model is fine on a GPU (flash attention, 16x16
tensor cores) and awkward on TPU. The einsum rewrite does not fix the occupancy;
it removes the permute traffic layered on top of it.

## Falsified -- do not retry

| hypothesis | result |
|---|---|
| Folded input-stage / embedding gradient (the A100's culprit, 67% of its step) | **0.96 ms, 0.8%**. The 53 ms first seen was eager-dispatch overhead in the probe, not real work. |
| Eager AdamW + `clip_grad_norm_` over 62 param tensors | 7.4 ms and 3.3 ms |
| head_dim zero-padded 32->128 with explicit `scale` (bit-exact) | 0.83x -- SLOWER |
| Token dim padded 51->64 (masked) | 0.81x -- slower |
| Compiling the loss into the graph with the model | 134.83 -> 134.78 ms, nothing |
| `capturable=True` AdamW | 147 ms, worse |
| `foreach=True`, TorchTPU's AdamW fork | neutral (128.8-128.9) -- stock AdamW is fine |
| einsum attention judged on forward only | rejected it; wrong test, see Lever 2 |

Note the first row: the fix that mattered most on the A100 is a non-issue on TPU,
and the TPU's dominant cost (attention layout) is a non-issue on the A100. Neither
platform's profile transfers.

## Port changes the TPU required

All verified byte-identical on CUDA against the pre-change trainer (fp32
reduction-order noise at the 8th significant digit only), so there is one code path,
not a TPU fork:

1. `masked_select` in the sparse loss -> masked sum/count.
2. Boolean-mask indexing in `expand_labels` -> static-shape `scatter_` with a
   sentinel column. Both of these lower to `nonzero`, whose output size is
   data-dependent: a host round-trip plus a recompile per distinct count.
3. Twelve per-step `.item()` metric reads -> one device-side accumulator, read back
   once per epoch.
4. Anchors baked to a device-resident table (`scripts/45_bake_anchors.py`, verified
   row-identical to the live `BFSAnchors` path) to remove a per-step host round-trip.

## The trap: never diverge around a materialization

TorchTPU's distributed golden rule is "avoid `if rank == 0` blocks around
operations that trigger materialization." Checkpointing looks exempt -- writing a
file is obviously rank-0 work -- but it is not:

```python
if is_main:
    torch.save(payload, path)        # DEADLOCK under DEFER_AND_FUSE + DDP
```

Under Fused Eager the `state_dict` tensors are still DEFERRED. Rank 0 pulling them
executes a graph that can contain that step's DDP collectives, while ranks 1-3 are
already waiting at the `dist.barrier()` below. The mesh hangs.

It does not look like a hang. The symptom is a **part-written checkpoint that
stops growing** -- 1.1 MB, then 2.2 MB, then frozen, against an expected 41 MB --
with all four ranks alive and burning CPU. It reads as "checkpointing from TPU is
slow", which is the wrong diagnosis.

Fix: materialise on every rank, write on one.

```python
cpu_state = {k.removeprefix("_orig_mod."): v.detach().to("cpu")
             for k, v in model.state_dict().items()}     # ALL ranks
cpu_opt = _optimizer_state_to_cpu(opt)                   # ALL ranks
if is_main:
    torch.save({"state_dict": cpu_state, "optimizer": cpu_opt, ...}, path)
```

A correct checkpoint is 41,408,457 bytes (62 tensors, 3,444,761 params, plus 62
optimizer entries) against the A100's 41,411,849 -- so **size is the cheap check**.
Note this trap is coupled to the speedup: it only fires under `DEFER_AND_FUSE`,
which is exactly the mode Lever 1 turns on.

## Does the model actually work?

Speed is worthless if the weights are wrong, and file size only proves the tensors
are present. `scripts/46_verify_checkpoint.py` scores a checkpoint against labels
we know exactly (the d<=5 BFS anchor pool, plus the d=1 states):

| | TPU ep5 | A100 ep200 | A100 ep1500 |
|---|---|---|---|
| undo ranked strictly best (d=1) | 100% | 100% | 100% |
| MAE vs exact 24-way Q on anchors | 0.347 | 0.110 | 0.069 |
| argmin picks a TRUE optimal move | 65.6% | 98.9% | 99.7% |
| einsum vs sdpa on the same weights | 1.29e-06 | 1.20e-06 | 3.95e-06 |

A 5-epoch model is early, and it reads exactly like an early point on the A100's
own trajectory. The last row is the one that matters for the rewrite: the two
attention paths agree to ~1e-06 on identical weights, so a checkpoint is portable
between them.

**A warning about the obvious test.** The natural sanity check -- "all 24 children
of the solved state are at d=1, so Q should be ~1.0" -- is worthless here, and it
was in the first version of this script as a pass/fail gate. The converged A100
models predict **3.8-3.9** on the solved state while the 5-epoch TPU model predicts
0.48, so the check false-fails the deployed model and flatters the untrained one.
The solved state is out of distribution for this objective: the sampler pivots at
depth >= 1 and the anchor pool contains it exactly once in 1.77M. It is kept as an
informational line only. Running the reference checkpoints through the same script
is what exposed it -- a threshold that fails your known-good model is a broken
threshold, not a broken model.

## Reproducing

```bash
# on the TPU VM
bash ~/launch_ddp.sh /home/and-l/runs/<name> --set training.n_epochs=5
```

`launch_ddp.sh` sets `TPU_DEFER_AND_FUSE=1`, the tier-2/3 compilation caches, and
picks a FREE torchrun master port each run. That last one matters: with a fixed
port, a previous run's socket lingers, the store fails to bind with EADDRINUSE, one
rank dies and the survivors spin in a collective wait that looks exactly like slow
training.

## Untried

- `TORCH_TPU_INTERNAL_XLA_OPTIONS="xla_optimization_level=O3"` (launcher supports it
  via `TT_XLA_OPTS`).
- More chips. Rows/s RISES as the per-rank batch falls (88.5k @ 960 rows vs 64.2k @
  7680), so a v6e-8 at 960 rows/rank should scale better than linearly. Needs the
  Queued Resources API -- the 8-chip `instances create` path backend-faults.
- `nhead` 8->2 (head_dim 128) measured 2.7x on SDPA alone, but it is a DIFFERENT
  model, so it belongs in a quality ablation, not here.
