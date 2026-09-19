# TorchTPU: what it is, and what it took to beat an A100

Session of 2026-08-24/25. Everything below was either read from the `torch_tpu`
source or measured on a GCP `ct6e-standard-4t` (4x v6e). Where a claim is
measured, the number is given; where it is read from source, it says so.

Detailed training write-up: [`tetraminx/TORCHTPU_TRAINING.md`](tetraminx/TORCHTPU_TRAINING.md).

---

## 1. Headline

Two workloads were ported to TorchTPU: **training** (now faster than the A100 it
was originally trained on) and **beam search** (matches the production JAX
kernel's search quality, at 5.7x its wall clock).

### Training

| config | s/epoch | vs A100 (142.7) |
|---|---|---|
| naive port | 234.8 | 1.65x slower |
| `+ TPU_DEFER_AND_FUSE=1` | 176.5 | 1.24x slower |
| `+ attn_impl: einsum` | **132.0** | **1.08x FASTER** |

Same model, same recipe, same samples/epoch; checkpoint verified against exact
BFS ground truth.

### Beam search (beam 1M, 4x v6e, matched settings, all paths replay-verified)

| | total moves / 6 pids | mean | s/pid |
|---|---|---|---|
| JAX SPMD kernel (production) | 185 | 30.83 | 57.5 |
| **TorchTPU (this port)** | **186** | **31.00** | **329** |

Quality is a dead heat (0.17 moves/pid). The 5.7x wall-clock gap is the model
forward plus a 2.35x factor the profile does not explain -- see section 7.

---

## 2. What TorchTPU actually is

**It is not PyTorch/XLA.** A fresh backend: ATen kernels lower *directly to
StableHLO MLIR* (`torch_tpu/ops/<op>/<op>.cc` builds `stablehlo::` ops), then
PJRT/libtpu. No LazyTensor IR, no `xm.mark_step`, no `xla_device()`.

```python
import torch
device = torch.device("tpu")     # that is the whole init
```

The backend autoloads on `import torch` via PyTorch's `TORCH_DEVICE_BACKEND_AUTOLOAD`
plus a PrivateUse1 rename, so **user code should not `import torch_tpu`**.
`torch.tpu.*` mirrors `torch.cuda.*`.

### Execution modes -- the thing that matters most

| mode | how | note |
|---|---|---|
| `DEFER_NEVER` | **the default** | op-at-a-time, async. Lazy is NOT the default, unlike torch_xla |
| `DEFER_NEVER_AND_LAUNCH_BLOCKING` | `TPU_LAUNCH_BLOCKING=1` | synchronous; debug only |
| `DEFER_AND_FUSE` | `TPU_DEFER_AND_FUSE=1` | defers and fuses across op boundaries |
| compiled | `torch.compile(fn, backend="tpu", dynamic=False)` | AOT; `torch.compile` is monkeypatched to default to the tpu backend |

`sync.synchronize(tensors, wait=False)` (`torch_tpu._internal.sync`) is the
`mark_step` analogue: it cuts the graph *without* a host round-trip.

### Hard constraints (read from source)

- **`dynamic=False` is mandatory.** A SymInt in the graph raises. Every shape
  change is a full recompile.
- **`scan` is the only higher-order op.** `torch.ops.higher_order.scan` lowers to
  `stablehlo.while` and works in eager too. There is **no `cond`, no
  `while_loop`** -- zero hits repo-wide.
- **No silent CPU fallback.** Unimplemented ops throw; opt in with
  `execution_mode.enable_cpu_fallback = True`. Better than torch_xla's silent
  cliff.
- `@torch.inference_mode` + `torch.compile` **crashes**. Use `@torch.no_grad()`.
- Bounded dynamism exists (`dynamism.mark_dynamic(t, dim, lo, hi)`) but allows
  **exactly one dynamic dim per tensor** and needs `DEFER_AND_FUSE`.

### Ops that force a host round-trip

These call `.cpu().item()` internally to learn their own output size, then compile
a graph specialised to that count -- so they recompile per distinct value:

> `nonzero`, `masked_select`, `unique`, `unique_consecutive`, `bincount`,
> `repeat_interleave.Tensor`, `histc`, `ctc_loss`

Poison for any inner loop. Note `x[bool_mask]` is `nonzero` in disguise.
`take` also calls `.cpu().item()`, but only under
`TORCH_TPU_INTERNAL_ENABLE_DEBUG_CHECKS` (off by default), so it is safe.

Everything else we need is static and registered: `topk` (native `chlo::TopK`, the
same primitive `jax.lax.top_k` uses), `sort`, `gather`, `scatter*`,
`index_select`, `index_put`, `where`, `cumsum`, `searchsorted`, `bucketize`,
`embedding`. Coverage ground truth is `tests/native_functions_data.py` -- it lists
what is MISSING. `torch.topk` returns **int64** indices (I32 internally, widened
for ATen parity), which doubles index memory versus JAX.

### Distributed

Multi-process SPMD only -- **no GSPMD, no `mark_sharding`**. Each rank sees its
core as `tpu:0`; DDP takes no `device_ids`.

```bash
eval $(python -m torch_tpu._internal.distributed.launchers.singlehost_wrapper)
torchrun --nproc_per_node=$WORLD_SIZE script.py
```

with `dist.init_process_group(backend="tpu_dist")`. Collectives: allgather,
allreduce, alltoall, reduce_scatter. DTensor and a `topology_aware_mesh()` helper
exist.

### The JAX escape hatch (read from source, not exercised)

`torch_tpu._internal.pallas.jax_op(name, jax_fn, mesh=..., input_partition_specs=...)`
registers **any jittable JAX function** as a `torch.library.custom_op` by exporting
it to StableHLO (`jax.export.export`) and splicing it into the torch graph -- no
JAX runtime, no device copy. It works multi-chip with `jax.shard_map`
(`tests/pallas/pallas_communication_test.py`). This would let existing JAX SPMD
kernels be called from PyTorch, but it is explicitly NOT used in the beam-search
port, which is pure TorchTPU by request.

---

## 3. How the A100 gap was closed

The method that worked, twice: **roofline the chip at your own shapes first**, then
attribute, then verify the fix numerically. Guessing produced three wrong
diagnoses.

### Lever 1 -- `TPU_DEFER_AND_FUSE=1` (234.8 -> 176.5 s/epoch)

Everything outside `torch.compile` runs op-at-a-time by default, and a trainer is
full of such code: a 40-iteration random-walk sampler loop, the loss, the metrics.

| data pipeline | eager (default) | DEFER_AND_FUSE | torch.compile |
|---|---|---|---|
| | 61.3 ms | **13.1 ms** | 29.7 ms |

It **beat `torch.compile`** on the same code -- many tiny ops fuse better than they
compile. Metrics byte-identical, so the lever is free.

### Lever 2 -- attention without the 5-D permute (176.5 -> 132.0 s/epoch)

```python
q, k, v = qkv.view(B, T, 3, H, Dh).unbind(2)     # heads never leave dim 2
s = torch.einsum("bthd,bshd->bhts", q, k) * scale
p = s.float().softmax(-1).to(v.dtype)            # fp32 softmax, deliberate
a = torch.einsum("bhts,bshd->bthd", p, v)
```

25.14 -> **14.85 ms** per layer fwd+bwd. Shipped as `attn_impl` on `_SelfAttn`,
default `"sdpa"` so GPUs keep flash attention; parameters unchanged and
`state_dict` interchanges `strict=True`.

**It was almost discarded.** Judged on FORWARD time it looks worse (5.51 vs
4.56 ms) and was rejected on that basis. The permute is 0.06 ms forward; unwinding
it around SDPA's saved tensors in the BACKWARD is the cost. *Judge any
training-path rewrite on fwd+bwd.*

### Why attention is the bottleneck

| | TFLOP/s |
|---|---|
| square 8192^3 | 361 |
| our QKV / FF GEMM shapes | 212-259 |
| FF sublayer in situ | 235 |
| **attention (SDPA)** | **3.0** |

Every attention matmul is 51x32x51 on a 128x128 MXU -- ~2.5% tile occupancy,
because T=51 tokens and head_dim=32. Arithmetic, not a bug. The einsum rewrite
removes the permute traffic layered on top; it does not fix occupancy.

### Numerics: never let one TPU path judge another

einsum and SDPA disagreed rel 3.3e-3 on TPU. Against a CPU **float64** reference:

| | rel error |
|---|---|
| CPU float32 | 6.6e-07 (noise floor) |
| TPU fp32 SDPA | 4.836e-03 |
| TPU fp32 einsum | 4.555e-03 |
| TPU bf16 SDPA | 8.192e-03 |
| TPU bf16 einsum, naive | 1.505e-02 |

**Both** TPU paths are ~4.6e-3 off -- that is the MXU's default 1-pass bf16
multiply (`torch.tpu.precision`), not a defect in either. On CPU fp32 they agree
to 5.8e-07. The bf16 gap was SDPA's internal fp32 accumulation; promoting only the
softmax to fp32 closes it for free.

---

## 4. Falsified -- do not retry

| hypothesis | result |
|---|---|
| Folded input stage / embedding gradient (**67% of the A100's step**) | **0.96 ms, 0.8%** on TPU. The 53 ms first seen was eager-dispatch overhead in the probe. |
| Eager AdamW + `clip_grad_norm_` over 62 param tensors | 7.4 ms and 3.3 ms |
| head_dim zero-padded 32->128 with explicit `scale` (bit-exact) | **0.83x -- slower** |
| Token dim padded 51->64 (masked) | 0.81x -- slower |
| Compiling the loss into the graph with the model | 134.83 -> 134.78 ms, nothing |
| `capturable=True` AdamW | 147 ms, worse |
| `foreach=True` / TorchTPU's AdamW fork | neutral (128.8-128.9); stock AdamW is fine |
| einsum attention judged on forward only | rejected it -- wrong test |

The first row is the general lesson: **neither platform's profile transfers.** The
A100's dominant cost is negligible on TPU, and the TPU's dominant cost is
negligible on the A100.

Also measured: **rows/s FALLS as batch grows** (88.5k @ 960 rows -> 64.2k @ 7680),
so buy throughput with more chips at a smaller per-rank batch, not a bigger batch.

---

## 5. Traps that cost real time

**Never diverge around a materialization.** This is in TorchTPU's own docs and I
broke it anyway:

```python
if is_main:
    torch.save(payload, path)     # DEADLOCK under DEFER_AND_FUSE + DDP
```

Under fused eager the state_dict tensors are still deferred; rank 0 pulling them
runs a graph containing that step's DDP collectives while ranks 1-3 wait at the
barrier. The symptom is a **part-written checkpoint that stops growing** (1.1 MB,
2.2 MB, frozen, vs 41 MB expected) with all ranks alive -- it reads as "TPU
checkpointing is slow", which is the wrong diagnosis. Fix: materialise on every
rank, write on one. Note the trap only fires under the mode that gives the
speedup.

**A threshold that fails your known-good model is a broken threshold.** My
checkpoint verifier gated on "Q(solved) should be ~1.0". It failed the 5-epoch TPU
model -- and then failed the deployed A100 ep1500 model too, which predicts 3.8
there. The solved state is out of distribution for this objective (the sampler
pivots at depth >= 1; it is 1 of 1.77M anchors). Running the reference checkpoints
through the same script is what exposed it.

**A fixed torchrun `--master_port` strands the next run.** The previous run's
socket lingers, the store fails to bind with EADDRINUSE, one rank dies and the
survivors spin in a collective wait that looks exactly like slow training. Pick a
free port per launch.

**Eager measurements of compiled code are meaningless.** The input stage measured
53 ms eager and 1.08 ms compiled. Measure the mode you will ship.

---

## 6. Infrastructure playbook (GCP)

- The private Artifact Registry **is reachable with a normal GDE account** --
  the Google Drive wheels in the quickstart are not needed:
  ```bash
  pip install --pre --index-url \
    "https://oauth2accesstoken:$(gcloud auth print-access-token)@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" \
    torch_tpu
  ```
  This pulled torch_tpu `0.1.1.dev20260824100827`, **torch 2.13.0+cpu** (not the
  2.11 the docs claim) and libtpu 0.0.46. Needs Python 3.12 (deadsnakes on
  Ubuntu 22.04).
- `ct6e-standard-4t` create needs **`--maintenance-policy=TERMINATE`** or it fails
  with a confusing `onHostMaintenance` error.
- **`--max-run-duration` is not updatable in place** (no flag on `instances
  update`, GA or beta). Set 168h (the max) at create time.
- Boot disk defaults to 10 GB; pass `--boot-disk-size`.
- Compilation caches are worth setting -- they removed the epoch-1 warmup entirely
  (146.0 -> 135.9 s):
  `TORCH_TPU_TIER2_COMPILATION_CACHE` (a NAME, lands in `/dev/shm/torch_tpu_cache/`),
  `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` (a path, can be `gs://`).

---

## 7. Beam search, fully in TorchTPU

`tetraminx/src/tetraminx/beam_tpu.py` + `tetraminx/scripts/47_beam_tpu.py`.
**No JAX and no `jax_op`** -- every op is one torch_tpu registers. Runs unchanged
on CUDA, which is how the TPU numbers were checked.

### The existing solver could not be ported

`cayley.khoruzhii_search.KhoruzhiiSolver` is TPU-hostile three separate ways:
`torch.unique` for dedup (host round-trip + recompile per distinct count),
boolean-mask indexing (`nonzero` in disguise), and genuinely **dynamic shapes**
(`cand.index_select(0, unique_pos[:select_B])`, whose size depends on how many
uniques were found). `dynamic=False` is mandatory on TPU. So this is a rewrite.

### What replaces `unique`

Sort, compare neighbours, permute back -- all fixed-shape:

```python
sort_h, sort_idx = h.sort()
dup_sorted = cat([False, sort_h[1:] == sort_h[:-1]])
dup = dup_sorted[sort_idx.argsort()]
score = where(dup | ~parent_valid, +BIG, score)   # never REMOVE, just mask
```

Duplicates are scored +inf rather than dropped, because dropping is a dynamic
shape. `topk` then never picks them.

### Multi-chip

One code path serves any world size (W=1 degenerates naturally, so the
single-chip path cannot rot separately). Per step each rank expands and scores
its `24 * B_local` children, assigns each an OWNER from its hash so identical
states anywhere in the mesh land on the same rank, does a per-owner `topk(K)`,
exchanges with `dist.all_to_all_single`, then dedups and re-selects locally.
Scores travel WITH the candidates rather than being recomputed on arrival.

`all_reduce(MIN)` on int64 **fails to compile** on torch_tpu, so the cross-rank
first-hit latch uses `all_gather` + a host-side min -- which is what
[[spmd_jax_dead_end]] recommends anyway. `all_to_all_single` works for
int8/int32/int64/float32 and 2-D.

Three more things the port needed:

* **Two hashes.** Dedup runs on all `B*24` children, so it uses a cheap
  dot-product hash. The endgame probe needs the table's own Zobrist hash, so it
  runs on the `B` survivors only.
* **int8 states.** `children.to(int64) * hash_vec` materialises a `(B*24, 88)`
  int64 tensor -- gigabytes at beam scale, against N*88 bytes for int8 states.
* **Tree bookkeeping OUTSIDE the compiled region.** Writing `tree[j] = ...`
  inside `step()` makes the level index part of the graph and recompiles every
  level.

### HEAD TO HEAD vs the production JAX kernel

Beam **1,048,576 global on 4x v6e**, matched on every knob: same checkpoint
(`mx_tf_az/epoch_1500.pt`, sha256 `e7c0b761cd4c`), B_LOCAL 262,144,
K_PER_PEER 131,072, alpha 2, 1 frame, 36 max steps, same d<=6 endgame table,
Q scorer, bf16. Every path replay-verified independently on both sides.

| pid | JAX len | TorchTPU len | JAX s | TorchTPU s |
|---|---|---|---|---|
| 990 | 31 | 32 | 58 | 383* |
| 991 | 29 | **30** | 53 | 309 |
| 992 | **32** | 31 | 60 | 332 |
| 993 | **30** | 29 | 56 | 285 |
| 994 | 32 | 32 | 60 | 332 |
| 995 | 31 | 32 | 58 | 332 |
| **total** | **185** | **186** | **345** | **1,973** |
| **mean** | **30.83** | **31.00** | **57.5** | **329** |

*includes compile warmup

**Quality is a dead heat**: 185 vs 186 moves over 6 pids (0.17/pid). JAX wins
990/991/995, TorchTPU wins 992/993, tied on 994 -- inside run-to-run noise.
**Speed: JAX by 5.7x.**

### Where the time goes (profiled, B_local=262,144)

| component, per step | cost |
|---|---|
| **model forward (Q head)** | **~3.9 s (~83%)** |
| goal probe (Zobrist + searchsorted) | 641 ms |
| dedup sort+argsort over 6.29M children | 82 ms |
| per-owner topk x4 | 37 ms |
| hash, 88-slot loop | 35 ms |
| expand children | 10 ms |
| **the 4 `all_to_all` exchanges** | **0.6 ms** |
| hit-latch + tree bookkeeping | ~1 ms |

**Multi-chip communication is essentially free.** The cost is the model, for the
same reason as in training: attention runs at ~3 TFLOP/s because T=51 and
head_dim=32 fill ~2.5% of a 128x128 MXU tile.

Two clear wins found but not yet applied:

* **The fused multiply-reduce hash is 4.4x faster than the 88-slot loop**
  (7.9 vs 35.1 ms). The loop was written to avoid an (N,88) int64 intermediate,
  but XLA fuses the cast into the reduction -- so the "optimization" cost 4.4x
  for no memory saving. Keep the loop only where N is the full child array.
* The goal probe (641 ms) could drop to ~10 ms by rebuilding the endgame table's
  index with the cheap hash instead of Zobrist.

### UNRESOLVED: a 2.35x the profile does not explain

The step benchmarks at **4.96 s** in isolation but runs at **11.65 s** in situ --
same compiled function, same shapes, same world size. 4.96 s is 17.5 TFLOP/s,
matching the training measurements; 11.65 s is 7.5.

Three explanations were tested and **falsified**:

| hypothesis | result |
|---|---|
| deferred execution skips outputs nothing demands | +3.0 ms |
| holding per-step tree tensors in Python lists | +0.9 ms |
| random ints vs real permutation states | +38 ms |

Per-step time is also flat across 26 steps (4958-4962 ms), so it is not
graph growth. **If that 2.35x is recoverable the real gap to JAX is ~2.4x, not
5.7x.** Recorded as open rather than attributed to a guess.

Artifacts: `tetraminx/results/torchtpu_vs_jax/`.

## 8. Port changes the trainer needed

All verified byte-identical on CUDA against the pre-change trainer (fp32
reduction-order noise at the 8th significant digit only), so there is **one code
path, not a TPU fork**:

1. `masked_select` in the sparse loss -> masked sum/count.
2. Boolean-mask indexing in `expand_labels` -> static-shape `scatter_` with a
   sentinel column.
3. Twelve per-step `.item()` metric reads -> one device-side accumulator, read
   back once per epoch.
4. Anchors baked to a device-resident table (`tetraminx/scripts/45_bake_anchors.py`,
   verified row-identical to the live `BFSAnchors` path).

Artifacts: `tetraminx/configs/mx_tf_az_tpu.yaml`,
`tetraminx/scripts/46_verify_checkpoint.py`,
`tetraminx/models/mx_tf_az_tpu5/` (5-epoch checkpoint + log).
