**TPU beam-search improvements, 5 September 2026**

Implemented and tested a more memory-efficient JAX search, fixed invalid frontier entries consuming useful beam capacity, reduced TorchTPU executable storage, and added optional faster attention with correct compilation-cache handling. A 268,435,456-state JAX frontier executed on eight GCP v6e chips. The integrated 4M search found 89 moves across three puzzles, versus 91 at the same control width. A billion-state search has not been demonstrated.

The preceding [implementation and Kaggle notebook review](C:/Users/and-l/cayley/tetraminx/TPU_BEAM_REVIEW_2026-09-05.md) covers all three requested notebooks and their retrieved outputs. This report covers the subsequent code changes and GCP experiments. Here 1M means 2^20 states; GiB means 2^30 bytes.

**JAX changes and measured capacity**

The maintained [JAX kernel](C:/Users/and-l/cayley/tetraminx/kaggle_notebooks/tpu_beam_tetraminx/jax_beam_spmd_v_only.py) now:

1. Carries a float32 score and uint32 child ID through streamed owner selection. It constructs the final selected child records once, after processing all parent chunks.
2. Combines the receive-sort and final-selection permutations before gathering states. State gathers and endgame Zobrist lookups run in bounded 65,536-row tiles.
3. Exchanges int8 states, a uint32 parent/move word and optional bf16 scores directly. The typed payload is 94 bytes per Tetraminx candidate, avoiding byte-packing and unpacking intermediates.
4. Masks empty parents before owner selection and keeps rejected survivors empty. The previous code could score padding as though it were a real parent, allowing it to displace legal children before being discarded later.
5. Uses four-byte ancestry records through 2^24 slots per rank, then eight-byte records with 27 parent bits. Parent indices through a 1B global beam on eight ranks are representable; that is an encoding limit, not a memory-fit result.

The important memory fix was bounding the gather and endgame intermediates. Merely carrying compact child IDs still ran out of memory at 128M. This agrees with the previously retrieved Kaggle allocation report, which showed severe padding of large gather-index buffers.

All capacity rows below use the same eight v6e chips, 0.8 PieceTransformer / 0.2 ResMLP Q blend, Q/V consistency 0.3, history depth 1, model batch 8,192 and parent chunk 65,536.

| Implementation | Global beam | Compiler temporary storage per chip | Evidence |
|---|---:|---:|---|
| Original control | 128M | 39.53 GiB required | Compilation OOM; 31.25 GiB available |
| Compact child IDs alone | 128M | 39.53 GiB required | Compilation OOM |
| Compact IDs + tiled gathers/endgame | 128M | 15.46 GiB | Compiled |
| Add typed exchange | 128M | 12.21 GiB | Compiled |
| Integrated implementation, validity enabled | 128M | 12.00 GiB | Compiled and executed one step, 345.25 s |
| Integrated implementation, wider ancestry | 256M | 23.71 GiB | Compiled and executed one step, 1,019.58 s |

The 128M temporary requirement fell by about 70% relative to the failing control. These are compiler temporary-storage figures, **not measured total peak HBM**. The result JSONs retain argument/output/alias/code sizes and device allocator statistics separately. The 256M execution's compiler gate reserved more than 10% planned headroom.

Large-width execution used a repeated-goal frontier to establish capacity. It did not reconstruct complete 128M/256M solutions or establish their path quality. Its timings are single synchronized capacity steps, not steady-state estimates for ordinary diverse frontiers. Doubling width from 128M to 256M took about 2.95 times as long with these fixed chunks, so memory capacity alone does not make enormous searches economical.

Increasing only the parent chunk from 65,536 to 262,144 reduced the 128M synthetic step from 345.25 to **295.42 seconds**, a 14.4% reduction. Compiler temporary storage stayed essentially unchanged at 12.00 GiB. The candidate passed a post-compilation execution gate: arguments + outputs - aliases + temporaries + code totaled 15.12 GB, below the 30 GB limit. This identifies a useful throughput setting to test on complete puzzles; it is not yet a verified path-quality setting or a steady-state speed estimate.

**Verified complete paths**

The quality workset fixes the transformer checkpoint, frame k0-forward, depth-six endgame, history 0, Q/V consistency 0, model batch 8,192, parent chunk 32,768 and receive oversampling 2. Every returned move sequence was independently replayed against the original puzzle, rather than trusting the solver's success flag.

| Arm | Beam | pid 990 | pid 991 | pid 992 | Total |
|---|---:|---:|---:|---:|---:|
| Original control | 1M | 31 | 29 | 32 | 92 |
| Compact child IDs | 1M | 31 | 29 | 32 | 92 |
| Validity fix alone | 1M | 31 | 29 | 31 | 91 |
| Integrated implementation | 1M | 31 | 29 | 31 | 91 |
| Original control | 4M | 30 | 29 | 32 | 91 |
| Compact + tiled | 4M | 30 | 29 | 32 | 91 |
| Integrated implementation | 4M | 30 | 29 | 30 | 89 |

The compact memory-only arms preserved the exact full control sequences, not just their lengths. At 1M, masking invalid entries shortened pid 992 by one move. At 4M, the integrated implementation shortened it by two moves relative to the same-width control. This is a three-puzzle smoke comparison and does not estimate a competition-wide gain. The deterministic best-of-tested merge totals 89 moves (30/29/30), also achieved by the integrated 4M arm alone, versus 92 for the 1M control.

At 4M, the original, compact+tiled and integrated implementations logged median device steps of 5.8, 5.7 and 5.8 seconds. Logging rounds to 0.1 seconds, and total walls were 511.1, 506.1 and 493.0 seconds including compilation. The integrated arm needed fewer search levels. Steady step speed was effectively unchanged; this does not establish a precise kernel speedup. At 1M the corresponding logged steps were approximately 1.3–1.4 seconds.

An adversarial padding test retained only 22 of 24 legal children in the control and all 24 after the fix. Empty frontiers stayed empty. CPU and TPU differential checks covered Q/V scoring, score exchange enabled/disabled, history, inverse-move filtering and Q/V consistency: 156 matching output comparisons per memory implementation. Validity was tested separately because it deliberately changes search policy. Ancestry boundary checks cover the uint32-to-uint64 transition and the highest supported parent/rank/move values.

Production 1M finished pid 990 before the first VM was interrupted by maintenance. Only missing pids 991/992 were resumed under a separate immutable manifest. The recovered union preserves both source records; it does not manufacture completion of the interrupted run.

**TorchTPU improvements and diagnosis**

The maintained [Torch search](C:/Users/and-l/cayley/tetraminx/src/tetraminx/beam_tpu.py) supports a scan over fixed-size model chunks. The [CLI](C:/Users/and-l/cayley/tetraminx/scripts/47_beam_tpu.py) enables it for compiled TPU inference; `--model-loop` restores the original unrolled loop. `--bf16 --half-sdpa` enables the faster attention option, which can change scores and paths. The scalar scan carry must be allocated and synchronized before compilation: creating it inside the exported scan triggered a native deferred-buffer assertion in the tested backend.

| Metric | Unrolled loop | Scan |
|---|---:|---:|
| Model-only steady median | 1.758 s | 1.768 s |
| Model compile + first call | 37.97 s | 27.69 s |
| Model cached executable HBM | 264.04 MiB | 55.21 MiB |
| Full-step steady median | 2.289 s | 2.295 s |
| Full-step compile + first call | 117.89 s | 58.38 s |
| Full-step cached executable HBM | 363.22 MiB | 162.27 MiB |
| Warm complete solve, pid 990 | 57.23 s | 55.90 s |
| Path length | 30 | 30, exact same sequence |

The actual maintained class was also tested, with a 2.295-second median step and the same replay-verified path. Cached executable HBM measures compiled code storage, not all device memory. First-call times include compilation and depend on previous cache contents: the final isolated scan configuration took 109.87 seconds for compile + first call, versus 118.95 seconds for the isolated loop control. Therefore the earlier 117.89-to-58.38 observation is not a robust twofold cold-compilation speedup. The reliable scan benefit is the much smaller compiled code footprint; steady step speed is unchanged.

Model-only JAX and Torch timings used exactly matched inputs: 1M states split across eight ranks, seed 990+rank, 20 legal moves from puzzle 990, chunk 8,192, same checkpoint. JAX took 0.830 seconds versus Torch's 1.758 seconds. Explicitly passing replicated runtime weights to JAX gave 0.827 seconds, essentially equal to closed-over weights at 0.827 seconds. Constant-weight specialization does not explain this gap in this experiment.

The historical extra 2.35x slowdown inside a full Torch solve did not reproduce: approximately 24 useful iterations at 2.29 seconds are consistent with the observed 57-second solve. The principal remaining measured gap is inside model inference, not unexplained host waiting. This does not establish the cause of the historical run's penalty.

Attention lowering contributes a measurable difference. The local TorchTPU [attention implementation](C:/Users/and-l/torch_tpu/torch_tpu/ops/scaled_dot_product_attention/scaled_dot_product_attention_shlo.cc:136) promotes the entire attention module to float32 for this model's sequence length 51 unless half-precision SDP reduction is enabled. Allowing that reduced model-only time to 1.622 seconds, but changed the preferred action on 4.8% of 1,024 sampled states. The cache-isolated full search confirmed a useful speed gain:

| Freshly compiled setting | Median full step | Warm complete solve | Verified path |
|---|---:|---:|---|
| Default precision, loop | 2.293 s | 56.91 s | 30 moves |
| Half-SDPA, loop | 2.139 s | 53.41 s | 30 moves, different sequence |
| Excess precision, loop | 2.289 s | 57.23 s | 30 moves, different sequence |

Half-SDPA was 6.7% faster per step and about 6.2% faster for this complete solve. It is opt-in because its numerical ranking changes are real and only one complete Torch puzzle was compared. The final maintained scan with corrected cache handling measured **2.293 s / 56.91 s** for default attention and **2.138 s / 53.29 s** with half-SDPA; both returned independently verified 30-move paths. Their compiled executable storage was 162.27 and 162.49 MiB, respectively.

A further source-level difference is that TorchTPU [explicitly defaults](C:/Users/and-l/torch_tpu/torch_tpu/common/excess_precision.cc:60) `xla_allow_excess_precision` to false, while XLA's default is true. A valid isolated comparison measured 1.759 seconds for the default model and 1.755 seconds with excess precision enabled: no meaningful speed improvement. It changed preferred actions on 6.25% of the sample. This setting was not promoted.

The precision experiments exposed a correctness issue in the benchmark environment: the PyTorch AOT graph cache reused a compiled artifact despite changed global precision settings. A separate TPU Tier-2 cache namespace was insufficient. The cached excess-precision model returned exactly the preceding half-SDPA model's predictions, while a fresh excess-precision compile returned different predictions. Cached full-search variants likewise reused the baseline sequence; fresh variants changed it. These earlier runs remain preserved but their causal precision comparisons are rejected in `results/precision_cache_correction.json` and the failure ledger.

The maintained CLI disables AOT graph-cache loading and constructs `TpuBackend(enable_serialization=False)` for compiled TPU inference. Native TPU compiled-kernel caching remains available. The constructor argument matters: passing `enable_serialization=False` through the generic `torch.compile(options=...)` dictionary is ignored by this backend version. The final tests use the actual constructor policy.

The remaining roughly twofold model-only gap between JAX and TorchTPU is not fully explained at the kernel level. The experiments locate the dominant difference in inference, measure the attention contribution, and rule out unrolled model chunks and closed-over JAX weights as its main explanation. Detailed device-kernel profiling remains necessary to explain the rest; this report does not claim that Torch now matches JAX throughput.

Rejected inference variants include gathered token inputs (1.783 seconds, no improvement), einsum attention (1.950 seconds, slower) and a compiled hit latch (57.56-second solve, no benefit). Scan predictions were bit-identical on all 24 actions of 1,024 sampled states. The JAX and Torch implementations are not bit-identical: baseline action agreement was 90.3%, so their paths cannot be treated as identical numerical controls across frameworks.

**Using the changes and next capacity work**

For JAX, use the maintained `gcp_beam_tetraminx.py` alongside the updated kernel and existing `jax_model.py`. Supply `--parent-chunk` to select the streaming implementation; leaving it unset selects the older non-streaming body. Invalid-slot masking defaults on in streaming mode, and `--legacy-padding` is its comparison flag. The memory changes need no additional feature flag. Keep JAX x64 enabled; the driver sets this explicitly. Preserve the intended checkpoint, blend, history, endgame and frame settings when comparing runs.

Host ancestry capacity is independent of TPU memory. At eight ranks and 60 stored levels: 64M needs 15 GiB, 128M needs 30 GiB, 256M needs 120 GiB, and 1B needs 480 GiB. The record size doubles above 128M because the local parent index outgrows 24 bits. Set `--tree-dir` to storage with sufficient capacity. The tests used GCP shared memory; this does not establish that the same configuration fits Kaggle's TPU or scratch storage.

The next substantial memory candidate is the included [lossless piece codec](C:/Users/and-l/cayley/tetraminx/tpu_experiments/compact_20260905/piece_codec.py). Tetraminx's 88 stickers can be represented by 50 piece codes, each requiring at most four bits: **25 bytes per state**, a 71.6% logical reduction. It passed round trips on all 1,000 test states, all 24 moves for each state and 51,200 chained move checks. It is not integrated into TPU selection/exchange or the model, so no 1B capacity claim follows from it. A useful next implementation would keep compressed frontiers and exchange records, decode only bounded inference/endgame tiles, and measure TPU layouts before a large launch.

The other remaining scaling issue is repeated owner top-k over the retained pool. With a fixed parent chunk, both the number of chunks and retained pool size grow with beam width. Parent-chunk tuning and a hierarchical merge/selection design deserve priority before simply multiplying beam size again. Different chunking or approximate selection can alter tie handling or retained states, so full-path quality must remain a separate gate.

The exact-endgame minimum-selection variant passed its targeted oracle but gave no improvement on the workset. A complete depth-six table probed every layer normally makes first new hits have the same six-move tail; selecting a minimum tail is therefore not a general improvement in this setup. It was not promoted.

**Reproducibility, scope and cost**

Experiments use Python 3.12.14, JAX/jaxlib 0.10.2, libtpu 0.0.42, Torch 2.13.0+cpu and TorchTPU 0.1.1.dev20260824100827. Eight single-host v6e chips were used. Local TorchTPU source observations come from checkout 988759f; that checkout may postdate the installed wheel by several hours, so runtime measurements take precedence.

The [experiment directory](C:/Users/and-l/cayley/tetraminx/tpu_experiments/compact_20260905/SESSION.md) contains immutable source arms, artifact hashes, passing preflight manifests, failure-inclusive ledgers, raw measurements and independent replay scripts. `results/final_verification.json` binds maintained executable syntax to the source tested on TPU. `results/verified_benchmark_best.json` records deterministic winner attribution and all source hashes; its CSV covers only these three benchmark puzzles.

The changes live in the maintained Python modules. Existing dirty notebook and staged-dataset revisions were preserved and need a targeted reconciliation before publishing a new Kaggle version. No competition submission was made.

All uploads and TPU rentals were within the approved $200 initial cap. The two allocations had maximum TPU charges of $43.20 and $21.60, respectively: **$64.80 combined TPU upper bound**, plus storage/network charges. This is a bound from allocation durations and the $1.35/chip-hour rate, not a retrieved billing statement. Results are durably stored in the private bucket under `beam-improvement/compact_20260905/`.

Cloud cleanup completed and was independently checked at **18:33:45 UTC**. Both allocated nodes, their queued requests and the early unallocated request are absent. [Provider cleanup evidence](C:/Users/and-l/cayley/tetraminx/tpu_experiments/compact_20260905/results/provider_cleanup.json) records each check. The final raw VM result archive was verified locally with SHA256 `ac143dfd015b3cfbfad793d1dd226ec86c6a0abf20c0d41ff10c6f1f5f09a65d` and saved to the private GCS prefix before deletion.
