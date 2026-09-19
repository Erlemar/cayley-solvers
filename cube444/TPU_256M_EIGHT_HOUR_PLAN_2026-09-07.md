# Making cube444 256M practical within eight hours

**Validated on Kaggle, September 9:** notebook version 3 completed a 256M PID 1001
solve in 44 moves in about 6h10m, with scratch held at 1.96 GiB and recorded host
RSS peaking at 109.86 GiB. See the [measured result and reproducible recipe](tpu_experiments/speed_ram_20260908/RESULTS.md).
The final recipe combines the faster decoder/selection/scoring work with anonymous
RAM ancestry; all source, replay, runtime and artifact checks passed.

The rest of this document is the original design and its then-current evidence.
Preserve the frozen `beam256_20260906` attempt; historical estimates below are
superseded by the completed GCP and Kaggle measurements linked above.

## Measured budget

The full-width run demonstrated 268,435,456 global states with compiled program memory of 11,967,378,944 bytes per chip. Mature steps cost 1,306.7 seconds including approximately 4 seconds of host copying/writing. Host work is 0.31% of step time; further ancestry compression alone cannot provide the required speedup.

The 1M control reached the endgame after 41 device steps, reconstructing a 42-move prefix and a six-move tail. Its 48-move result and the earlier 46-move 48M result motivate a **40–45 step planning range**, not a prediction or guarantee for a new 256M policy.

| Planning case | Current step time, search only | 550 s/step plus 40 min setup/reserve |
|---|---:|---:|
| 40 steps | 14.52 h | 6.78 h |
| 45 steps | 16.33 h | 7.54 h |

Target **550 seconds per mature step**, approximately 2.38 times faster than current execution, with a strict 28,800-second whole-session budget. An intermediate 600-second result may suffice for 40 steps but has little margin for 45. The next launcher must use the user's eight-hour constraint, replacing the previous nine-hour provider allowance.

The first six full-width device steps consumed 7,989.5 seconds (2 h 13 m) while much of the frontier was padding. This is an opportunity, not a measured removable cost: receive-side work and real candidates still require processing.

## Prioritized changes

### 1. Profile a full-width step before choosing the main rewrite

Capture JAX/XProf device activity, with a warm executable and synchronized output. Attribute time to parent scoring, owner routing and per-owner selection, winner packing/collectives, receive sorting/deduplication, final selection, and endgame lookup. Use bounded trace capture if the full 22-minute trace is too large. Separate stage microbenchmarks are supporting evidence because splitting a fused executable can change its cost.

Use a dense legal frontier for mature-step comparisons and a real sparse frontier for startup comparisons. Artificial all-identical or solved states are capacity tests, not representative speed/quality tests. Retain fp32/highest `s3`, alpha 2, exact d6, and the same frames while testing search mechanics.

JAX officially supports TPU traces and requires synchronization to capture device execution: [JAX profiling](https://docs.jax.dev/en/latest/profiling.html). Current documentation was checked on 2026-09-07; validate the capture API on the pinned Kaggle JAX 0.10.2 runtime.

### 2. Derive owner ranks without generating full children

The Q model already emits all 24 child scores from each parent. The current Q path nevertheless constructs `(parent_chunk, 24, 96)` children solely to calculate owner hashes. Final winner states are independently reconstructed from compact parent/move IDs.

For move permutation `p`, the existing owner rule has the exact identity:

```
owner(s[p]) = sum_i s[i] * hash_vector[inverse(p)[i]] mod 8
```

Precompute the 96-by-24 coefficient matrix modulo 8. A small integer-exact dot computes every owner directly from parents. A proposed TPU implementation uses exactly representable bf16 integer inputs and float32 accumulation: factors are 0..5 and 0..7, products at most 35, sums at most 3,360. This does not reduce the precision of neural inference. TPU code-generation/throughput still need measurement.

CPU proof: 600,960 exact owner matches across all test states and 24 recoloured frames, plus padding. The current child tensor is 576 MiB at the 262,144-parent chunk before any physical padding or cast intermediates. Removing it should create room to test larger chunks, but the new compiled peak must be checked rather than adding nominal buffer savings to the old memory estimate.

### 3. Reduce repeated large selections

At 256M there are 128 parent chunks per chip. Each performs eight top-k merges, retaining 8,388,608 candidates per destination from 8,388,608 old entries plus 6,291,456 new entries. Many new entries are masked because they belong to other destinations.

| Parent chunk | Chunks/chip | Owner top-k calls/chip/step | Total score entries across those merges |
|---:|---:|---:|---:|
| 262,144 | 128 | 1,024 | 15.03 billion |
| 524,288 | 64 | 512 | 10.74 billion |
| 1,048,576 | 32 | 256 | 8.59 billion |

The 1M chunk reduces this work-count proxy by 43%, not necessarily wall time by 43%. Keep model batch 4,096 fixed for the first comparison so activation memory stays controlled. Test 512K then 1M chunks after direct owner hashing, with separate memory gates and parity checks. Sorting algorithms can behave differently at the larger shapes.

If selection dominates after this change, investigate an exact threshold/radix selection design that avoids repeatedly reprocessing the entire retained set. It must preserve fp32 cutoff ordering, per-owner capacities and stable tie handling. A two-pass design may duplicate model computation unless scores are stored; storing all scores costs 3 GiB per chip at this width, so this is a memory/dataflow design decision. Fixed per-parent or per-chunk quotas are approximations and must not be represented as equivalent global selection.

### 4. Avoid doing full-width work on empty startup slots

The copied source sets `mask_invalid=False`. Consequently, the existing empty-chunk bypass is disabled and padding parents are scored/routed. Invalid-score states can also remain as duplicate records after selection.

Add explicit validity, skip completely empty chunks, and measure per-rank valid occupancy. These flags change the copied source's padding behavior and potentially tie ordering; treat them as a correctness/policy change and run matched path-quality checks.

A stronger startup design uses a small set of precompiled capacities while the real frontier grows, switching to 256M before any smaller beam or sender-owner quota would prune candidates. Keep the 256M owner mapping, transform ancestry addresses correctly on migration, and record overflow/cutoff evidence. A simple fixed-depth 1M-to-256M schedule is not proof of equivalence: it can lose candidates before the wide phase.

### 5. Compute only CLS in the final transformer block

After four transformer blocks, the model reads only `h[:, 0]`. In the last block, keep full-token keys and values, but compute the query, attention output, residual and feed-forward result only for the classification token. Earlier blocks retain all tokens.

This is mathematically equivalent because final-layer updates to the other tokens cannot affect the final CLS result. Floating-point operation shapes change, so numerical equivalence must be checked. The first CPU prototype tested 1,088 padded examples: maximum Q difference 7.6293945e-06 and zero best-action mismatches. This is not a TPU path-parity result.

The change removes much of the fourth block's work, roughly a fifth of transformer arithmetic by operation count. That does not imply a fifth of whole-step wall time, nor prove a TPU speedup. Also consider concatenated Q/K/V projections and model-batch 4,096 versus 8,192 only after measuring model time and memory separately.

### 6. Evaluate lower precision only under path-quality gates

Keep the original fp32/highest scorer as the first control. The source notebook's older precision probe showed substantial action-ranking changes with bf16; it used a blend and an imperfect default-precision reference, so it cannot settle the quality of a new `s3`-only mixed-precision implementation.

If model time remains dominant, compare selective lower-precision layers against fp32/highest on real search states and full solves. Keeping normalization and outputs in fp32 may help, but does not guarantee preserved paths. A bf16 shortlist followed by fp32 reranking is approximate unless candidate recall is proven; it can discard the right branch before reranking.

JAX documents the accuracy/throughput tradeoff explicitly: [matmul precision](https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html). Current docs are guidance; retain the pinned runtime and judge the actual checkpoint by replay-verified path quality.

## Short sequential benchmark ladder

1. Unchanged dense 256M control: compile/memory, one warm step, device trace.
2. Direct-owner variant at 262K chunks: exact output parity and mature-step comparison.
3. Direct-owner variant at 512K and then 1M chunks: one family varied, independent memory gates.
4. Best search kernel with final-CLS model: numeric, action-ranking, full-step and small-solve checks.
5. Startup validity/skipping as a separate policy arm, including occupancy and matched quality.
6. Only if needed, mixed precision or a different selection algorithm in separately identified arms.

Use short, fixed-step full-width benchmarks before spending another full solve session. Accept the combined configuration only with safe HBM margin, approximately 550-second mature steps, correct collectives/ancestry, and satisfactory replay-verified path quality. Record every failed/slow/rejected arm as well as winners.

The original campaign spent about 30.7 minutes on checks and small controls before the full solve. Once a source/configuration is validated, move the full qualification suite to its own benchmark run; production retains fast source/runtime checks and replay verification. This recovers session time but cannot close the performance gap alone. Precompute/hash the exact endgame when convenient; do not remove the d6 tail, which already avoids several beam levels.

## Prototype artifacts

`cube444/tpu_experiments/speed_design_20260907/prove_optimizations.py` independently checks the direct-owner identity and last-CLS prototype against the copied source.

`cpu_proof.json`, `cls_model_prototype.py`, and `budget.json` contain the results and calculations. CPU JAX was 0.10.0; real TPU validation and speed measurements remain pending. No production source, live notebook version, beam width, scorer precision or experiment manifest was changed by this analysis.
