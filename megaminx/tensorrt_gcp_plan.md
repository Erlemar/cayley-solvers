# TensorRT on GCP — execution plan

The only remaining speed lever with predicted >5% gain (model_s = 96% of total
wall, so direct model-forward acceleration is the only way past the async ceiling
and the rejected quality-trading optimizations). torch_tensorrt fails to build on
Windows; GCP L4 (Linux + CUDA 12.9 + PyTorch 2.9) is the target.

Prereq state (from `reference_gcp_cayley_vm.md`):
- VM: `cayley-gpu`, zone `us-east1-b`, image
  `pytorch-2-9-cu129-ubuntu-2204-nvidia-580`
- Python 3.10.12, PyTorch 2.9.1+cu129, NVIDIA driver 580
- L4 (24 GB, sm_89) — engine will be sm_89-specific (won't transfer to A100/4090
  without rebuild)

## Goal

Replace the `torch.compile`d model forward in `beam_lab` with a TensorRT engine.
Acceptance gate: **path-sum on the 24-puzzle stratified sample is identical to
current --compile baseline (2013)**, wall reduced ≥20% on model_s.

If gate passes, deploy in qshort solver for full 1001 — this is the lever that
could move us from 88,195 toward Rokicki's 21,840 zone (combined with already-
deployed qshort + beam-stack rescue).

## Step 1 — environment (15 min on the VM)

```bash
ssh and-l@<VM_IP>
python3 -m pip install --upgrade pip
# torch_tensorrt 2.9.x ships pre-built wheels for the matching torch 2.9 + cu129
python3 -m pip install torch-tensorrt --index-url https://download.pytorch.org/whl/cu129
python3 -c "import torch_tensorrt; print(torch_tensorrt.__version__)"
nvidia-smi --query-gpu=compute_cap --format=csv,noheader   # confirm sm_89
```

Fall-back if torch-tensorrt 2.9 is unavailable: pin torch to whichever torch_tensorrt
version is freshest on PyPI (`pip index versions torch-tensorrt`); the deeplearning-
platform image lets us reinstall a matched torch.

## Step 2 — export script (30 min)

New file: `beam_lab/export_tensorrt.py`. Single-purpose: load a checkpoint, build
the engine, save it.

```python
import torch
import torch_tensorrt
from model import load_checkpoint

ckpt = "models/m05_bellman_warm/epoch_0499.pt"
batch = 16384      # MUST match internal_batch_size in solver
device = "cuda"
precision = torch.float16   # primary target; also try torch.bfloat16

model = load_checkpoint(ckpt, device=device).to(precision).eval()
# disable internal chunking so the whole batch flows through in one call
if hasattr(model, "inference_chunk_size"):
    model.inference_chunk_size = None

# torch_tensorrt expects example inputs — we use int64 because that's what
# the embedding layer takes. TensorRT may downcast to int32 internally.
example = torch.zeros((batch, 120), dtype=torch.int64, device=device)

trt_engine = torch_tensorrt.compile(
    model,
    inputs=[
        torch_tensorrt.Input(
            min_shape=(batch, 120), opt_shape=(batch, 120), max_shape=(batch, 120),
            dtype=torch.int64,
        ),
    ],
    enabled_precisions={precision},
    workspace_size=4 << 30,           # 4 GB build workspace
    truncate_long_and_double=True,    # int64 → int32 on TRT side
    require_full_compilation=False,   # allow TRT to drop unsupported subgraphs back to torch
)

# Smoke: 5-iter forward, check output dtype + magnitude
with torch.inference_mode():
    for _ in range(5):
        out = trt_engine(example)
print("out dtype:", out.dtype, "mean:", out.mean().item(), "shape:", out.shape)

torch.save(trt_engine, "models/m05_trt_fp16_b16384_sm89.ts")
```

Notes:
- `truncate_long_and_double=True` is the standard knob for handling the int64
  input. If it fails, we cast inputs to int32 in `_model_predict` (small change).
- If compile errors on the embedding layer, fallback is to extract the embedding
  output (float bf16) and feed that to TRT; route the embedding through eager
  PyTorch.
- Build time is one-shot (~minutes); runtime is fast.

## Step 3 — solver integration (30 min)

In `beam_lab/beam_search.py`, add:

```python
def setup_model_for_tensorrt(engine_path, batch_size: int = 16384):
    """Load a pre-built TRT engine and return it as a drop-in for the model.

    The engine is shape-fixed to (batch_size, 120) — caller MUST ensure
    `internal_batch_size == batch_size` and `pad_to_batch_size=True`.
    """
    import torch_tensorrt  # noqa: F401  (registers TS_extension)
    eng = torch.load(engine_path, weights_only=False)
    eng.eval()
    return eng
```

In `run_benchmark.py`, add `--tensorrt-engine PATH` flag (mutually exclusive
with `--compile` and `--cuda-graphs`). When set:
- `model = setup_model_for_tensorrt(args.tensorrt_engine, args.internal_batch_size)`
- `pad_to_batch_size = True`
- skip the `setup_model_for_inference` call (engine already has internal chunking baked out)

## Step 4 — validation gates (45 min)

Run sequentially on GCP, halting on first failure:

1. **Smoke (1 puzzle, beam 16k)** — engine forward works, no NaN, path is valid:
   ```
   python3 run_benchmark.py --tensorrt-engine models/m05_trt_fp16_b16384_sm89.ts \
       --beam 16384 --max-steps 60 --pid 0 --verify
   ```
2. **3-puzzle (beam 131k)** — A/B vs --compile, path-sum must match (254):
   ```
   python3 run_benchmark.py --tensorrt-engine ... \
       --beam 131072 --max-steps 120 --samples data/sample_3.csv --verify \
       --out results/trt_3p.csv
   ```
3. **24-puzzle stratified (beam 131k)** — path-sum gate (must equal 2013):
   ```
   python3 run_benchmark.py --tensorrt-engine ... \
       --beam 131072 --max-steps 120 --samples data/sample_puzzles.csv --verify \
       --out results/trt_24p.csv
   ```
   Hard gate: `path-sum != 2013` → REJECT this engine, retry with bf16 or
   debug numerics.

If gate fails:
- Try `precision=torch.bfloat16` instead of fp16 (m05 was trained in bf16, so
  numerics may match more cleanly).
- Try `enabled_precisions={torch.float16, torch.float32}` (mixed) — lets TRT
  fall back to fp32 for layers that lose accuracy in fp16.
- If still failing, the TRT path is REJECTED on quality grounds (consistent with
  the user's "no metric regressions" rule) and we shelve it.

## Step 5 — qshort integration (30 min)

If the V (m05) engine validates, build a second engine for the Q-shortlister
student (same procedure, different checkpoint), add `--tensorrt-student PATH`
flag to qshort mode.

24-puzzle qshort A/B: current local (compile) baseline vs TRT engines on both
teacher and student. Path-sum should match.

## Step 6 — full 1001 deployment (one shot)

```bash
python3 src/megaminx/scripts/03_solve.py \
    --searcher khoruzhii \
    --teacher-engine models/m05_trt_fp16_b16384_sm89.ts \
    --student-engine models/m23_trt_fp16_b16384_sm89.ts \
    --beam 524288 --max-steps 180 \
    --out submissions/full_1001_trt.csv \
    --fallback data/kociemba_fallback.csv
```

(03_solve.py needs the same `--tensorrt-*` flag plumbing — copy from
run_benchmark.py.)

Submit only after `verify_submission` passes.

## Risks (in priority order)

| risk | likelihood | mitigation |
|---|---|---|
| fp16 numerics shift path-sum | medium | bf16 fallback, mixed precision fallback, hard gate at step 4 |
| TRT can't compile embedding layer | medium | route embedding through eager torch, TRT for the rest of the network |
| torch_tensorrt 2.9 ABI mismatch | low | reinstall matched torch + trt versions |
| Engine build OOMs on L4 (24 GB) | low | reduce workspace_size to 2 GB |
| Engine is L4-specific (sm_89) | known | document, rebuild on each GPU arch we ever use |

## Time budget

| step | wall time |
|---|---|
| 1. env | 15 min |
| 2. export | 30 min |
| 3. integration | 30 min |
| 4. validation | 45 min |
| 5. qshort | 30 min |
| **active work** | **2.5 hours** |
| 6. full 1001 solve (background) | ~12 h GPU |

## Decision tree

```
Step 4 24-puzzle gate
├─ path-sum = 2013 + wall reduced ≥20%
│     → proceed to step 5/6, deploy
├─ path-sum = 2013 but wall not improved (or worse)
│     → REJECT — TensorRT not a win on this workload, document
└─ path-sum ≠ 2013
     ├─ try bf16, mixed precision
     │     → if path-sum matches, proceed
     └─ else REJECT — quality regression
```

## Anti-patterns

- Don't deploy the engine without the 24-puzzle path-sum gate. Submitting a
  worse-quality engine to Kaggle wastes a daily submission slot.
- Don't bake int8 quantization in the first pass. INT8 needs a calibration
  dataset and adds another quality-risk knob — only consider after FP16 is
  validated.
- Don't try to compile the entire `_do_greedy_step` (incl. neighbor / dedup /
  topk) into one TRT engine. The savings are <4% (per the speed_optimizations
  empirical findings) and the engineering cost is enormous. Only the model
  forward needs TRT.
