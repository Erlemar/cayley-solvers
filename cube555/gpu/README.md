# cube555 on the 2xT4 MultiGPUBeamSearch engine — analysis, and where it stops

Analysis of `alexandervc/cayleypy-2xt4-tetraminx-transformer-v3-29len` and an attempt
to build the cube555 equivalent.

**Conclusion: that engine cannot run cube555.** Not because of the model — that part is
solved below — but because cube555's state is longer than the engine's packed state
representation allows. The cap is on the *puzzle*, not the scorer, so no checkpoint work
gets around it.

---

## 1. What the reference notebook actually is

It is not a beam search. It is a **thin adapter** around a third-party CUDA engine:

- engine: `TryDotAtwo/MultiGPUBeamSearch`, pinned at commit `f679504b…`, cloned at runtime
- it exports a PyTorch checkpoint into the engine's own weight bundle, then shells out to
  `python -m tools.run_cayleypy_public --config-json … --output-dir …`
- runs at `BEAM_WIDTH = 2**26` on 2x Tesla T4
- the notebook's own contribution is four small local patches so the engine's export layer
  and two C++ manifest validators accept tetraminx's `(88, 88, 50, 3)` shape, plus a
  `pilgrim.model` shim and a combined generator JSON

The discipline throughout is explicit: *"no beam-search or CUDA math is touched anywhere."*
Every adaptation is a dict entry or a validator branch. That is what makes the method
safe, and it is also what bounds it.

Around the engine the notebook adds the same frame machinery this project already uses:
`(sym, inverted)` frames, per-frame solve, translate back, replay-verify, per-pid min-merge
into `submission.csv`.

## 2. The model side: solved

The engine has three checkpoint families — `batchnorm-folded` MLP, `resmlp-layernorm` MLP,
and the piece Transformer. cube555 has no transformer, so the target is
`resmlp-layernorm`. Its contract (read off `tools/export_stream1_mlp.py` and
`tools/stream1_weight_io.hpp`) is encoded in `test_export_contract.py`.

Checked against the shipped checkpoints: **7 violations**, the important one structural —
the exporter reads `sd["input_stack.3.weight"]` unconditionally, i.e. a **two-level** input
stack, while `ResMLPQ` has one (it is built from a single `d_model`). The C++ backs this up:
*"hidden1 must be >= hidden2 because Stream1 reuses hidden1 scratch for residual output."*

**Distillation does not fix it.** A two-level front end trained onto the teacher's
activation drove MSE down 12x (0.405 -> 0.034) while argmin agreement stayed at chance
(1/30); a Q-space objective with a centered ranking term did no better. The reason is
scale — the within-parent action spread is ~0.6 against a Q level of ~44, so the front end
must be right to well under a percent before the ranking survives, and that is far past
where gradient descent got.

**The algebraic rewrite does fix it, exactly and with no training** (`make_engine_model.py`).
Write the teacher's front end as `v = W0 e + b0`, `h = ReLU(LN(v))`, and set

    W0' = [W0 ; -W0]   b0' = [b0 ; -b0]        ->  u = [v ; -v]
    LN1: gamma=1, beta=0                        ->  LN1(u) = [v ; -v] / s
    ReLU                                        ->  [v+ ; v-] / s
    W3  = k * [I , -I]   b3 = 0                 ->  k*v / s
    LN2: teacher's gamma, beta                  ->  LN2(v)   (LN is scale-invariant)
    ReLU                                        ->  h

The intermediate ReLU is exactly what the ± pair absorbs. Residual blocks and head are
copied verbatim, so the student *is* the teacher.

`k` matters only through LayerNorm's epsilon, which is not scale-invariant: the student's
effective eps is `eps*s^2/k^2` with `s = sqrt(mean(v^2)) ≈ 6–9`. Swept, not guessed:

| `--w3-scale` | max abs dQ | argmin | full 30-action ordering |
|---|---|---|---|
| 1 | 2.796e-03 | 512/512 | 503/512 |
| 8 | 1.297e-04 | 512/512 | 511/512 |
| **32** | **1.755e-04** | **512/512** | **512/512** |
| 128 | 1.755e-04 | 512/512 | 512/512 |

k=32 and k=128 are identical, so 1.755e-4 is the fp32 round-off floor of the extra
2048-wide matmul, not error. For scale: the JAX/TPU port of the same model measured
2.7e-4 with 255/256 ordering. All four checkpoints convert with perfect ranking agreement.

That takes the contract from 7 violations to **1**, and the survivor is a one-line Python
guard (`if embed_dim != 16: raise`) whose own fold, `(embedding @ block.t()).t()`, is
embed_dim-agnostic — `state_len = 3600 // 24 = 150` still derives correctly, and embed_dim
never reaches the CUDA because the runtime consumes the folded
`(state_len x num_classes x hidden1)` table. Patching it is the same kind of edit the
reference notebook already makes. embed_dim 24 is *kept* deliberately: the per-position
table has rank up to 24 and an embed_dim=16 student could only reach rank 16.

## 3. Where it stops

`tools/cayleypy_public/data.py`:

    if not 1 <= state_len <= 120:
        raise ValueError(
            "public runner requires 1 <= state_len <= 120 for the State128 logical payload")

**cube555's state_len is 150.** The engine's own README says the same thing in prose.

This is the packed state type, not a contract dict. Raising it means changing the CUDA
core — precisely the thing the reference notebook's method is built to avoid, and not
something that can be validated without the hardware. Nor can cube555 be squeezed under
the cap: the state is a permutation of 150 distinct labels, the fixed-centre orbit is only
6 of them, and 144 is still over.

The engine's other supported shapes are all under it — IHES 72, tetraminx 88, cube4 96,
p900 120 — which is consistent with 120 being a deliberate ceiling rather than an oversight.

## 4. What is here

    test_export_contract.py    the engine's contract, restated and checkable offline
    make_engine_model.py       exact ResMLPQ -> two-level ResMLPDistance rewrite
    engine_models/             all four checkpoints, converted and verified
    _ref/                      the pulled reference notebook

`make_engine_model.py` keeps its value regardless: it is a lossless re-parameterisation
that makes the cube555 scorer consumable by any two-level-ResMLP consumer, and the
±-duplication trick generalises to any single-level LayerNorm MLP that needs to become a
two-level one.

`test_export_contract.py` reports HARD (the CUDA's shape) separately from PATCH (a
Python-side guard), so a future checkpoint can be checked in one command.

## 5. Options

1. **A 2xT4 GPU beam for cube555 on our own solver.** `KhoruzhiiSolver` already has the
   30-wide Q head, the `d<=5` endgame ball, `history_depth`, frames and replay
   verification. Sharding it over two T4s is the honest "large beam on GPU" for this
   puzzle, and unlike the engine it can be tested here before it burns a session.
2. **Ask the engine's author for a wider payload.** A `State256` variant would take
   cube555 (150) and cube666 (216) at once. Nothing else blocks us — the model side is
   already done and verified.
3. **Point the engine at a puzzle it accepts.** tetraminx (88) is what the reference does;
   cube444 (96) also fits.

---

# The 2xT4 beam that was actually built

Option 1 from above. Runs **our** solver (`KhoruzhiiSolver`, the same one `30_solve.py`
drives) with one beam per GPU, both cards at once.

    gpu_beam.py             single-GPU worker over a (pid, frame) task list
    run_multi_gpu.py        splits tasks across devices, merges best-of-frames
    build_notebook.py       generates the Kaggle notebook + metadata
    test_notebook_local.py  executes the notebook's own cells against a staged mount
    kaggle_dataset/         362 KB: cayley + cube555 packages, workers, anchor builder

## Task-parallel, not a sharded beam

Width on cube555 is measured non-monotonic -- 2^20 2/6, 2^21 6/6, 2^22 3/6, 2^23 0/1,
with no OOM anywhere. The optimum is 2^21 and one 2^21 beam fits a single 16 GB T4, so a
second card is worth more as a second *worker* than as half of a bigger beam. It buys
frames (each ~33-50% independent, so `k` frames solve `1-(1-q)^k`) and pids. Two
processes, zero cross-device communication, no failure mode where one card stalls the
other.

## Measured decisions

**fp16, not bf16.** T4 is Turing with no native bf16. On 512 real test states vs fp32:

| dtype | max abs dQ | argmin agreement |
|---|---|---|
| **fp16** | **0.119** | **501/512** |
| bf16 | 0.876 | 427/512 |

**Best-of-frames.** `30_solve.py:269` breaks on the first frame that solves; this keeps
the per-pid minimum over every frame run. RESULTS.md s7 lists that as untested and
"strictly better" -- the second GPU is what pays for it.

**The endgame ball is held once.** `30_solve.py:203/215` keeps `eg_states` *and* its
sorted copy alive, so the d<=5 ball costs ~3.2 GB rather than 1.6 GB. Free on an 80 GB
A100; not free on a 16 GB T4. Fixed in `gpu_beam.py`.

## Two traps hit while building this

**The repo's `src/cayley` shadowed the handoff's.** They have diverged -- 641 lines
against 761 -- and `sys.path.insert(0, ...)` in a loop puts the LAST entry first, so
listing the repo last is what makes the handoff win. The symptom was a `TypeError` on
`no_backtrack`, a config field that exists in one copy and not the other. Silent
otherwise: it would just have run a different searcher.

**`KhoruzhiiSolver` returns move NAMES, not indices.** Everything downstream here works
in indices, so the conversion happens at that one boundary.

## Verified locally (2026-08-23)

- `run_multi_gpu.py` with `--devices cuda:0,cuda:0`: 8 tasks (4 pids x 2 frames) over two
  concurrent workers, 8/8 solved at optimal length, best-of-frames merged, 0 invalid rows.
- `test_notebook_local.py`: the generated notebook's own cells against a staged
  `/kaggle/input` -- mount discovery, working-tree staging, the d<=4 anchor build,
  subprocess wiring, report and submission. 3 pids x 2 frames, 3/3 replay-verified,
  3/3 improved on the baseline.

Two workers cannot share one 16 GB card at `ENDGAME_DEPTH=5` -- they stall at ~15.9 GB.
That is a single-GPU-box artifact: on Kaggle each T4 has its own 16 GB. The local tests
use d<=4 for that reason.

## The T4 OOM, and the wrong diagnosis I gave it first

The first 2xT4 run OOM'd asking for 12.00 GiB. I attributed it to `KhoruzhiiSolver`
materialising the whole `(B, 30, 150)` child block -- 17.6 GiB at 2^21 in int16 -- and
recommended dropping to 2^20 with a d<=4 ball. **That was wrong.** The full child block
lives in `_get_neighbors`, which is called from exactly one place (`_do_greedy_step`,
the V path) and even there on already-chunked input. `use_q_function=True` routes to
`_do_greedy_step_q_topk`, which scores from the parents and only materialises the
progressive top-k shortlist. The Q path never allocates it.

The real cause was in THIS file's endgame `_lookup`, inherited from `30_solve.py`:

    hit &= (eg_sorted[pos].to(torch.int64) == states_t.long()).all(dim=1)

The Q path hands that the whole shortlist -- `k = max(2B, q_topk_min)`, about 4.2M rows
at B=2^21 -- and every `(N, 150)` int64 temporary is **4.69 GiB**, which is the exact
figure in the earlier anchor-build OOM too. Three of them are live at once.

Fixed by chunking the lookup and comparing in the TABLE's dtype instead of casting both
sides to int64 (8x smaller). Measured peak on a real deep pid, `--endgame-depth 5`:

| beam | peak |
|---|---|
| 2^19 | 2.65 GiB |
| 2^20 | 2.89 GiB |
| **2^21** | **3.46 GiB** |

So 2^21 with the full d<=5 ball fits a 14.56 GiB T4 with room to spare, and the shipped
defaults did not need to change -- only the lookup did.

Two lessons worth keeping: measuring peak on SHALLOW pids understates it badly (the beam
never reaches full width), and `torch.cuda.max_memory_allocated` must be reset *before*
the task, not after, or the first reading folds in one-time setup. An early reading of
25.55 GiB on a 16 GB card -- impossible, and identical across two dtypes -- was the tell.

`uint8` state dtype is available via `--state-dtype` and gives byte-identical paths to
int16, but it is nearly free either way now that the child block is known not to be on
the hot path (2.65 vs 2.67 GiB at 2^19).

## Invariant: never widen a state-shaped tensor to int64 unchunked

Bit twice. `state_count x 150 x 8 bytes` is the whole problem:

| site | shape | unchunked | now |
|---|---|---|---|
| endgame table build | 10,739,017 x 150 | **12.00 GiB** | 1.17 GiB (1M chunk) |
| `_lookup` shortlist | up to 4.2M x 150 | **4.69 GiB** | 0.07 GiB (>=65536 chunk) |
| `eg_descend` | 1 x 150 | negligible | - |

Both of those numbers appeared verbatim in a 2xT4 OOM. The lookup was fixed first and
the construction was missed, because the fix was made against the instance that had
been *measured* rather than against the pattern. Any new `.long()` / `.to(torch.int64)`
on something with a leading state-count dimension needs a chunk loop; the current sweep
(`grep -n '\.long()\|to(torch.int64)' gpu_beam.py`) is clean.

Note this is invisible on a Windows dev box: WDDM spills CUDA allocations into host
memory, so a 12 GiB allocation *succeeds* on a 16 GB 4090 and shows up only as an
absurd `max_memory_allocated` reading. Linux/T4 has no such fallback.

## Not yet verified

No successful Kaggle GPU run yet. The first session after this fix should be 2 pids at
the shipped defaults before any long queue.

## Kaggle accelerator gotcha (cost one session, 2026-08-23)

`kernel-metadata.json` with `enable_gpu: true` and **no `machine_shape`** was scheduled
onto a single **Tesla P100 (sm_60)**. The image's torch ships
`sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`, so the first CUDA call died with

    torch.AcceleratorError: CUDA error: no kernel image is available for execution
    on the device        (cudaErrorNoKernelImageForDevice)

and the notebook reported `SystemExit: anchor build failed (rc=1)` -- 70 s into a BFS,
pointing nowhere near the cause.

Two fixes, both shipped:

1. `"machine_shape": "NvidiaTeslaT4"` in the kernel metadata. `enable_gpu` alone does
   not pin the accelerator.
2. An architecture preflight in the SETUP cell that compares each device's
   `torch.cuda.get_device_capability()` against `torch.cuda.get_arch_list()` and exits
   in ~2 s with the actionable message. Check the ARCHITECTURE, not the device name --
   the name tells you nothing about whether this image's torch has kernels for it.

---

# The engine path, reopened and made to work

`README` above concluded the MultiGPUBeamSearch engine could not take cube555. That was
right about the symptom and wrong about the depth of the cause: the `state_len <= 120`
guard is a stale Python policy, not a runtime limit, and the engine now runs cube555
end to end on 2xT4. Three bugs, all in our fork
`Erlemar/MultiGPUBeamSearch@cube555-engine`:

**1. Payload sizing (`cmake` + `config.hpp`).** `BEAM_STATE_LOGICAL_BYTES` is already
inferred from `puzzle_info.json`; pointed at cube555 an unmodified checkout reports
`150 / 160 / 30`. The `1 <= state_len <= 120` check in `tools/cayleypy_public/data.py`
described one particular build. Raising it alone would have been unsafe -- Zobrist rows
are indexed by state VALUE with `STATE_VALUE_PAD` pinned at 128, so a 150-class alphabet
would read into the next position's row silently. The pad is now inferred alongside the
other sizes with a `static_assert` keeping them in step.

**2. `stream1.cu` shared-state under-copy.** The guard reads
`threadIdx.x < STATE_STORAGE_LEN` (160) but the only launch site is `dim3
input_block(128)`, so bytes 128..159 were never written -- 22 uninitialised bytes for
cube555, 88 for cube666. Every shipped puzzle is <= 120, which is why it never fired.
Fixed with a strided loop, padding zeroed.

**3. `output_dim % 8` -- the one that actually crashed it.** `stream1_cutlass_linear_cuda`
passes `output_cols` as the leading dimension of B/C/D, and CUTLASS's default `half_t`
alignment is 8 elements (16 bytes). compute-sanitizer on 2xT4:

    Invalid __global__ read of size 16 bytes
      at cutlass::Kernel<...MmaPipelined<GemmShape<128,64,32>...Sm75...>>
      Address 0x7c8a4948801c is misaligned
      inside the nearest allocation at 0x7c8a49487e00 of size 61,440 bytes

61,440 B = 30,720 half = `hidden2(1024) x output_dim(30)` -- the head weight. Offsets
540/556/572/588 are row 9 of a 60-byte stride; 540 % 16 = 12. Every shipped puzzle has
24 moves (24 % 8 == 0); cube555 has 30 and cube666 would have 36. Fixed with an
alignment-1 GEMM taken only when `output_cols % 8 != 0`, leaving the aligned branch
untouched.

## Does it earn its place? Not yet.

Same pid, same checkpoint, scorer verified identical (512/512 argmin):

| solver | width | endgame | dedup | pid 1034 | wall |
|---|---|---|---|---|---|
| **our PyTorch beam** | 2^20 | d<=5 | history 4 | **122 moves** | ~28 min |
| our TPU SPMD kernel | 2^21 | d<=5 | history 4 | not found | ~16 min/frame |
| MultiGPUBeamSearch | 2^18 | touch 4 | - | not found | 3.8 min |
| MultiGPUBeamSearch | 2^21 | touch 4 | - | not found | 26.6 min |

Two uncontrolled variables before calling the engine weaker, and both favour ours by
construction rather than by search quality:

- **goal width** -- ours stops inside the exact `d<=5` ball (10,739,017 goal states);
  the engine's `touch_bfs_radius=4` is a far smaller target.
- **no `history_depth` equivalent configured** -- measured at **-14% path length** on
  this puzzle, which is enough to decide a marginal solve.

So: "not yet configured to win", not "architecturally worse", and n=1 pid on one frame.
Do not spend more sessions on the engine until it beats 122 on a pid we already solve;
raise `touch_bfs_radius` and find its dedup knob first.

## Sequence of wrong diagnoses, kept as a warning

Three in a row on this one crash, each plausible and each refuted by the next
measurement: (a) the full child block in `KhoruzhiiSolver` -- wrong, the Q path never
materialises it; (b) `FINAL_RESPONSE_TARGET_LOCAL_IDX_OFFSET` at a non-4-aligned byte
150 -- wrong, that accessor is byte-by-byte; (c) the `stream1` shared-memory tail --
real, but not the crash. The sanitizer settled it in one run. Reach for
`compute-sanitizer` earlier than feels necessary; reading CUDA for alignment bugs is
guessing.

## Dataset versions are not live when `datasets version` returns

`datasets version` returns as soon as the upload finishes; Kaggle then PROCESSES the
version asynchronously. A `kernels push` issued 20 s later mounted the PREVIOUS version,
so a run failed with the exact OOM that had already been fixed -- and the traceback
showed the old source line, which is the only reason it was caught rather than being
blamed on the fix not working.

Before pushing a kernel that depends on a fresh dataset version, VERIFY THE CONTENT,
not just that the push returned:

    kaggle datasets download -d artgor/cube555-gpu-code -f gpu_beam.py -p /tmp --unzip
    grep -q "_hc" /tmp/gpu_beam.py || echo "not live yet"

Sibling of the `kernels output` staleness rule: with Kaggle, "the command returned 0" is
not the same as "the thing is there".
