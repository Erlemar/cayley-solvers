# CayleyPy 4x4x4 — PieceTransformer training plan

**Date:** 2026-08-08 · **Machine:** A100 80GB (`NVIDIA PG509-210`), idle
**Inputs:** `~/cube444_a100_handoff/` (docs + code), `~/Downloads/cube4_full_ensemble_inference_w040.tar.gz` (Vlad Kuznetsov's trained models, extracted to `~/ensemble_w040/`)
**Competition:** https://www.kaggle.com/competitions/cayley-py-444-cube — deadline 2026-09-22

**Bottom line:** the handoff's central premise — "the scorer is blind past depth 40" — is a
measurement artifact of using random-walk index as the depth label. On geodesic pivots the
shipped transformer holds pair accuracy 0.877 at depth 38-42, not 0.564. Train it standalone,
supervise the deep band with near-geodesic paths rather than deeper random walks, and stop
treating it as a reranker.

---

## 1. The bundle is drop-in compatible (verified)

`cube4_full_ensemble_inference_w040` ships two trained Q heads plus the model/problem source
(`unified_training/`), but **not** the trainer — sampler and objective have to be written.

| model | provider | params | `complete` | notes |
|---|---|---|---|---|
| `transformer` | `piece_transformer` | 3,383,064 | **false** | run never finished its scheduled 16,384 epochs |
| `mlp_x16` | `pair_qmlp` | 47,013,144 | true | best epoch 16,104, val loss 8.546 |

Both checkpoints are **raw `state_dict`s** — no optimizer state, no epoch counter. Warm-start
means restarting Adam.

Convention check against `code/data/puzzle_info.json`:

```
move names identical : True
all 24 perms identical: True
target == central_state : True
```

Move names in order: `f0 -f0 f1 -f1 f2 -f2 f3 -f3 r0 -r0 r1 -r1 r2 -r2 r3 -r3 d0 -d0 d1 -d1 d2 -d2 d3 -d3`.
Byte-identical, so the weights drop into our beam with no action remapping.

**Architecture** (`unified_training/models.py`, `_cube4_layout`): 8 corners + 24 wings + 24
centres = 56 piece tokens + CLS = 57. `d_model=256, 4 layers, 8 heads, ff=1024`, ReLU, CLS
pooling, dropout 0. Exactly the derivation doc 04 predicted — and it keeps the centres, which
doc 04 correctly insisted on for an unreduced-cube scorer.

**Author's training config** (from `models/transformer/model.json`): objective
`sparse_depth_pair` with `top1_margin_weight: 0.0`, sampler `rw_middle_sparse` with
`k_min=2, k_max=45, move_filter=inverse`, augmentation **`identity`** (i.e. none),
batch 512, 256 steps/epoch, lr 1e-4, wd 3e-3, bf16.

---

## 2. Measurement 1 — the transformer is our best scorer at every depth

Run with `code/scripts/35_eval_vlad_q.py`, which reuses `32_eval_q.py`'s exact protocol
(same walk generator, same seed 12345, same DEPTHS) so the numbers are directly comparable
to the table in `04_TRAIN_SPARSE_Q.md`.

```
transformer: 3,383,064 params  provider=piece_transformer  complete=False
mlp_x16: 47,013,144 params  provider=pair_qmlp  complete=True

shared pivots: 2,048 walks, seed 12345, k_max 60

                             transformer                           mlp_x16
 depth        gap       pair        top1        gap       pair        top1
--------------------------------------------------------------------------
     2      1.973      0.991       0.838      1.962      0.992       0.845
     5      1.901      0.983       0.799      1.896      0.984       0.819
     8      1.816      0.974       0.747      1.783      0.971       0.733
    12      1.572      0.949       0.607      1.509      0.936       0.558
    16      1.384      0.906       0.451      1.241      0.873       0.379
    21      1.074      0.817       0.301      0.915      0.789       0.235
    26      0.656      0.723       0.162      0.530      0.681       0.134
    32      0.358      0.631       0.111      0.270      0.605       0.086
    40      0.149      0.554       0.065      0.124      0.549       0.058
    50      0.060      0.534       0.053      0.057      0.522       0.050
    58      0.004      0.510       0.044      0.010      0.505       0.038

  transformer: gap 1.97 (d=2) -> 0.66 (d=26)  decay 67%   | d=40 pair 0.554 top1 0.065
  mlp_x16:     gap 1.96 (d=2) -> 0.53 (d=26)  decay 73%   | d=40 pair 0.549 top1 0.058

reference (04_TRAIN_SPARSE_Q.md, our own arms, same protocol):
  our V   : d2 gap 1.954 pair .809 | d26 gap 0.319 | d40 gap 0.075 pair 0.448 top1 .052
  our Q(B): d2 gap 1.953 pair .842 | d26 gap 0.354 | d40 gap 0.111 top1 .030
```

Two conclusions:

1. The transformer beats **our V, our cell-B Q, and the 47M MLP** at every depth past 8 —
   with 14× fewer params than the MLP. That settles cells C/D: **at matched objective the
   architecture does matter.** Doc 04's "neither — it's the Q head's inference economics"
   was concluded from a 3.3M ResMLP-Q and does not generalize to a transformer.
2. Our deployed V is below chance at depth 40 (0.448); the transformer is above it (0.554).

---

## 3. Measurement 2 — their sampler is far shallower than assumed

The trainer isn't shipped, so I recovered the pivot distribution by scoring candidate
distributions against the `last_validation` block the author records for `mlp_x16`
(`sparse_mse 8.5763, pair 0.88599, top1 0.58881`), using the shipped weights.

First, the naive reading of `rw_middle_sparse` (all pivots `p ∈ [2,44]`, which is what our
`src/cayley/sparse_q.py` does) is badly wrong:

```
-- all pivots p in [2,44], seed 12345, n=704,512
   mse mean-of-two 33.1652 | sum-of-two 66.3304 | pooled 33.1652
   pair_acc 0.75477 | top1(excl next) 0.32290
```

Against the reported 8.576 / 0.886 / 0.589 — off by ~4× on MSE. Scoring all candidates
against all three metrics at once:

```
  candidate pivot distribution      mse    pair    top1    E[p]   rel err
                      REPORTED     8.58   0.886   0.589       ?
-------------------------------------------------------------------------
         k~U[2,45], p~U[1,k-1]     8.38   0.898   0.602    11.8     0.059   <-- MATCH
           uniform p in [1,24]     8.32   0.899   0.546    12.5     0.117
           uniform p in [2,24]     8.68   0.895   0.527    13.0     0.128
           uniform p in [1,26]     8.97   0.883   0.516    13.5     0.173
           uniform p in [2,22]     7.68   0.911   0.560    12.0     0.181
             k~U[2,45], p=k//2     7.33   0.915   0.580    11.5     0.193
           uniform p in [1,28]     9.44   0.867   0.488    14.5     0.292
           uniform p in [1,20]     6.05   0.930   0.615    10.5     0.389
           uniform p in [1,16]     3.14   0.956   0.692     8.5     0.888
           uniform p in [1,44]    32.49   0.761   0.338    22.5     3.355
           uniform p in [2,44]    33.25   0.756   0.323    23.0     3.476
```

**The sampler is: draw walk length `k ~ U[2,45]`, then pivot `p ~ U[1,k-1]` inside that
walk.** The `1/(k-1)` weighting makes it heavily shallow:

```
author pivot distribution  k~U[2,45], p~U[1,k-1]
  E[p] = 11.75   max p = 44
  P(pivot depth >= 10) =  47.97%
  P(pivot depth >= 20) =  21.19%
  P(pivot depth >= 25) =  12.90%
  P(pivot depth >= 30) =   7.00%
  P(pivot depth >= 35) =   3.06%
  P(pivot depth >= 40) =   0.80%
  P(pivot depth >= 46) =   0.00%   <- hard cap: k_max=45
```

The label convention itself is confirmed identical to ours (undo column = `p-1`, next column
= `p+1`, masked MSE on those two only) — that's why the MSE matches once the distribution is
right.

Consequence — the model saturates, because it has never been asked for a value above ~44 and
almost never above ~20:

```
transformer output saturation (walks to depth 60):
   p   E[q_undo]   target   E[q] all 24    pair
   2        1.05        1          2.95   0.991
   8        7.50        7          9.21   0.973
  16       16.57       15         17.90   0.899
  24       24.73       23         25.51   0.751
  32       28.96       31         29.29   0.625
  40       30.67       39         30.82   0.570
  48       31.38       47         31.43   0.520
  56       31.65       55         31.70   0.522
```

Tracks the target well to p≈24, then flattens at ~31.5. At p=40 the undo action scores 30.67
against an all-24-action mean of 30.82 — a 0.15 spread, i.e. no usable discrimination.

---

## 4. Measurement 3 — the decisive one: the *label* goes blind at depth, not the model

`04_TRAIN_SPARSE_Q.md` and `CLAUDE.md` both rest on this claim:

> past depth ~40 the deployed V's pair accuracy is 0.448 — BELOW CHANCE — and that is where
> ~1000 of the 1043 test pids live

That was measured on random-walk pivots, where the walk index is *asserted* to be the depth.
It isn't. A non-backtracking walk of length 58 lands at true distance ~44 (the 46,662 file
solves random test states in 44.7 moves on average, against Rokicki's 44.4). So at deep `p`
the "undo" move is not reliably one step closer to solved than any other move — **the label
itself is at chance.**

Test: re-measure the same model, same metric, on pivots where the deep label *is*
trustworthy — states replayed from the verified `cube4_submission_46662.csv` (1043
near-geodesic paths, 45,619 usable pivots). For a solution `s_0 → … → s_L = solved` via moves
`m_0..m_{L-1}`, state `s_i` sits at depth `d = L - i`, toward-solved column = `m_i`,
away column = `inverse(m_{i-1})`.

```
path pivots from 46,662-move verified file: 45,619 states, depth 1..49

A. GEODESIC-PATH pivots (label is trustworthy at depth)
  depth band        n      gap    pair    top1
         2-6    5,194    1.741   0.964   0.849
        8-12    5,156    1.035   0.838   0.588
       14-18    5,121    0.714   0.760   0.406
       20-24    5,093    0.930   0.827   0.469
       26-30    5,049    0.892   0.809   0.429
       32-36    5,016    1.154   0.868   0.417
       38-42    4,956    1.371   0.877   0.305
       44-50    1,957    0.825   0.792   0.166

B. RANDOM-WALK pivots (label asserts depth = walk index)
  depth band        n      gap    pair    top1
         2-6   16,995    1.921   0.985   0.827
        8-12   16,908    1.693   0.962   0.692
       14-18   17,022    1.397   0.907   0.480
       20-24   16,965    0.974   0.796   0.276
       26-30   16,944    0.577   0.694   0.155
       32-36   16,951    0.282   0.601   0.097
       38-42   16,969    0.170   0.564   0.071
       44-50   23,793    0.067   0.526   0.055
```

**At depth 38-42: pair 0.877 on geodesic pivots vs 0.564 on random walks. Gap 1.371 vs 0.170.**

The scorer was never blind at depth. Six levers in the handoff were measured against a broken
diagnostic and declared dead.

Two secondary observations worth keeping:

- The **hardest band is 14-18** (pair 0.760), not depth 40 — late-solve states in the reduced
  regime, where many moves look alike. The by-depth story is U-shaped, not monotone.
- The **weakest top-1 is 44-50** (0.166), which is exactly where every solve starts.

**Caveat, not controlled:** those 1043 paths are themselves solver-found, so there is some
selection effect toward states scorers rank well. The magnitude (0.56 → 0.88) is far too
large to be only that, but a clean control would re-measure on paths from an independent
solver.

---

## 5. Measurement 4 — standalone is affordable

`bench_piece_transformer.py`, under exactly what `solve_ensemble_submission.py` does
(`model.half()`, `inference_mode`, `torch.compile(mode="reduce-overhead")`, fixed batch):

```
model=transformer  device=NVIDIA PG509-210
                      config     batch     ms/step    M child-scores/s
----------------------------------------------------------------------
                  fp32 eager      4096      112.52                0.87
                  fp32 eager     16384      450.01                0.87
                  fp32 eager     65536  FAILED: CUDA error: invalid configuration argument
                  fp16 eager      4096       26.66                3.69
                  fp16 eager     16384      105.06                3.74
                  fp16 eager     65536  FAILED: CUDA error: invalid configuration argument
fp16 compile reduce-overhead      4096       26.84                3.66
fp16 compile reduce-overhead     16384      125.26                3.14
```

**3.7M child-scores/s at fp16** — 4.2× the fp32 figure, and above doc 04's 2.2-2.5M 4090-laptop
quote. `torch.compile` adds nothing here; fp16 eager is the best config.

Standalone beam arithmetic (a Q head scores all 24 children with one forward on the parent):

| beam B | ms/step | 150 steps | all 1043 pids, 1×A100 |
|---|---|---|---|
| 2^16 | 425 ms | 64 s | 18 GPU-h |
| **2^18** | **1.70 s** | **4.3 min** | **~74 GPU-h** |
| 2^21 | 13.6 s | 34 min | ~590 GPU-h (shard it) |

Non-model beam cost (gather + hash + unique) is ~9 ms at B=65536 per doc 04, so ~36 ms at
B=2^18 — negligible against 1.7 s.

**Implementation note:** the forward hits a CUDA grid limit past ~32k rows (the 57-token
attention). Chunk beam inference at ≤16384, which is what the author's `inference_batch_buckets`
already does.

Training throughput, warm-started, bf16 autocast, including on-GPU walk generation:

```
   batch   ms/step    pivots/s
     512      58.4       8,766
    2048     116.3      17,609
    8192     303.8      26,967
```

The author's entire original run (16,384 epochs × 256 steps × batch 512 ≈ 2.1e9 pivots) is
~22 GPU-h at batch 8192. Continuing is cheap.

---

## 6. Where the existing plan is wrong

| doc 04 / CLAUDE.md says | status |
|---|---|
| "Build cell D as a RERANKER, not a replacement scorer" | **Superseded.** It rested on our V being below chance on the band. The transformer isn't. Train standalone. |
| "The transformer must beat ResMLP-V at ~1/34 the beam width" | **Wrong arithmetic, stale numbers.** Measured 3.7M child-scores/s fp16 → ~74 GPU-h for the full test set at B=2^18. |
| "The answer is neither architecture nor objective: it's inference economics" | **Partly wrong.** Concluded from a 3.3M ResMLP-Q. At matched objective the transformer beats a 47M MLP with 14× fewer params. |
| "Past depth ~40 the scorer is below chance" | **Artifact.** 0.877 pair accuracy on geodesic pivots at depth 38-42. |
| Open follow-up (c): depth-tilted pivot sampling | **Right direction, wrong mechanism.** Raising `k_max` adds label noise; the rw label is already unreliable past ~25. |
| Rule 8: V scale-collapse is benign | **Still holds** — but the transformer's saturation at 31.5 is coupled to the shallow pivot distribution, so it is worth fixing here. |

---

## 7. Recommended training recipe

### Stage 1 — faithful continuation (sanity, ~2 h)

Rebuild the author's exact recipe: `k ~ U[2,45]` then pivot uniform inside the walk, masked
MSE on the two labelled columns, bf16, wd 3e-3, grad clip 1.0. Warm-start from
`models/transformer/model.pth`. No optimizer state ships, so restart Adam at **~5e-5 with
warmup** — the same `resume_lr` the author used when resuming the MLP.

Gate: validation should land near `sparse_mse ≈ 8.4, pair ≈ 0.90, top1 ≈ 0.60`, not diverge.
This proves the trainer is faithful before anything changes.

### Stage 2 — fix the deep supervision (the main run)

**Not deeper random walks.** Near-geodesic path labels:

1. **Seed set:** 45,619 pivots replayed from `cube4_submission_46662.csv`, available now.
2. **×24 via correct symmetry augmentation.** `sym(s,R) = color_map_R[s[rotation_R]]` — the
   recolour form, not slot-permutation-only (gotcha #3, silent if wrong). Assets already
   exist and pass: `code/data/color_maps_24.npy`, `rotations_24.npy`, `move_relabel_24.npy`,
   `test_symmetry.py` prints `SYMMETRY TABLES OK`. The author trains with
   `augmentation: identity`, so this is a free 24× multiplier they never used. It also lands
   the two labelled columns on *different* action indices, labelling more of the 24 per sample.
3. **Expert iteration (the scalable part):** beam-solve fresh random scrambles with the
   current model, keep the verified solutions, feed them back as path labels, repeat. This
   generates unlimited correct deep supervision in the exact operating distribution. This is
   the natural standalone training loop.
4. **Mix with rw sparse-Q for the shallow band**, where it demonstrably works (pair 0.96+ to
   depth 12). A curriculum, not a replacement.

Cheap add-ons worth taking in the same run:

- **Turn on `top1_margin`** — the author ships it at weight 0.0. It is scale-free, so it does
  not fight saturation, and it targets top-1 accuracy, which is what the beam depends on.
  Already implemented in `src/cayley/sparse_q.py::sparse_q_loss`.
- **Exact 24-column anchors** from the d≤4 BFS ball. `QAnchors` already exists in
  `sparse_q.py` (182,407 states, all 24 children exact). Rule 9 says anchors were load-bearing
  for V; for Q they are strictly stronger. The author uses none.

### Stage 3 — Q-Bellman refinement

`Q(s,a) ← 1 + min_a' Q(apply(s,a), a')` — doc 04's open follow-up (a), and the V arm's single
largest jump came from its Bellman stage. Needs new code (the scalar recursion in `bellman.py`
does not apply). Complementary to Stage 2: fixes label self-consistency without needing a
solver.

### Gating

- **Replace the rw-pivot diagnostic with the path-pivot one.** The old protocol systematically
  mismeasures the band we care about. Use `~/path_vs_walk_pivots.py`'s method A.
- Then a real beam bench per **rule 7** (loss is not a proxy), judged as **per-pid min against
  the 46,662 floor** per **rule 4**, at matched wall clock.
- Keep the **0.60 × transformer + 0.40 × mlp_x16 ensemble** for the final run — the author
  measured mean 50.93 vs 53.00 for the transformer alone at B=2^18 on 30 scrambles
  (paired −2.07 moves, 95% CI [−3.23, −0.90]). Both models ship; it is free.

### Optional parallel arm

One capacity scale-up (`d_model` 256→384, layers 4→6, ~10M params). The handoff's "capacity
doesn't help" result was measured on ResMLP-V 3.3M→15M and does not transfer to a transformer
— the 3.38M transformer already beats a 47M MLP. Costs throughput roughly linearly, so gate
it on a wall-clock-matched beam, not equal epochs.

---

## 8. Artifacts produced this session

| path | purpose |
|---|---|
| `~/cube444_a100_handoff/code/scripts/35_eval_vlad_q.py` | gap-by-depth for the shipped models, `32_eval_q.py` protocol |
| `~/identify_sampler.py` | recovers the author's pivot distribution from reported val metrics |
| `~/path_vs_walk_pivots.py` | geodesic-path vs random-walk pivots — the decisive experiment |
| `~/bench_piece_transformer.py` | fp16/compile throughput sweep |
| `~/verify_objective_match.py` | checks our objective against the author's reported numbers |
| `~/depth_mass_and_train_speed.py` | pivot-mass table, saturation curve, training step time |
| `~/ensemble_w040/` | extracted bundle |

## 9. Open items

1. **Doc corrections not yet written.** `04_TRAIN_SPARSE_Q.md` and `CLAUDE.md` still tell the
   next session to build a reranker and not to re-litigate levers that were killed by the
   broken metric. This is the highest-value cleanup.
2. **Re-measure our own V on path pivots.** Its "below chance at depth 40" may be an artifact
   too. The `c_bells2/epoch_0399.pt` checkpoint is not on this box.
3. **Control for path selection bias** — re-run measurement 3 on solutions from an
   independent solver.
4. The bundle's own `baseline_submission.csv` totals 47,588 (mean 45.63), worse than our
   46,662 file (mean 44.738). Not a merge source worth chasing, but worth knowing.
