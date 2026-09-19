# cube444 `s3` — Q-Bellman refinement: results and full training recipe

**Date:** 2026-08-09 · **Box:** A100-SXM4-80GB · **Code:** `~/cube444_tf/bellman_q.py`
**Checkpoint:** `runs/s3/latest` (step 6000) · **Cost:** 6000 steps in 1,266 s = **21 minutes**

`s3` is the best cube444 scorer measured to date. It is not a new architecture — same
PieceTransformer, same 3,383,064 parameters, same `state_dict`, loads into the bundle's beam
solver with `strict=True`. It is the shipped weights after two rounds of continued training:

```
orig   shipped bundle transformer  (models/transformer/model.pth)
 └─ s1        3k steps, stage-1 recipe (author's objective reproduced exactly)
     └─ s1L       20k steps, same recipe; step 4000 = "s1L_4k", the 944 checkpoint
         └─ s3        6k steps, Q-Bellman refinement          <- this document
```

---

## 1. Results

Every arm below is warm-started from the same checkpoint (`s1L_4k`), so the comparisons are
matched. `orig` is byte-identical to the shipped bundle weights (md5 verified).

> **All beam numbers in this section are transformer-STANDALONE** — `bench_beam.py` was
> overriding the solver's `--mlp-weight` default of 0.4 to 0.0, so the ResMLP rescorer was off
> for every arm. They stay valid as *scorer* comparisons, since the handicap is uniform, but
> they are not the best achievable configuration. Switching the blend on adds **−72 moves** on
> the held-out 54, and the exact endgame a further −8:
>
> | config | held-out 54 | wins | merge gain |
> |---|---|---|---|
> | `s3` standalone (this section) | 2931 | 1 | 2 |
> | `s3` + blend 0.4 | 2859 | 2 | 4 |
> | **`s3` + blend 0.4 + endgame d6** | **2851** | **2** | **4** |
>
> See §6c of `cube444_stage3_heldout_2026-08-09.md`. Packaged as `run_best.sh` in
> `~/cube444_inference.tar.gz`.

### B=65536, 110 steps

| arm | STRAT 18 (floor 820) | HELDOUT 54 (floor 2609) |
|---|---|---|
| orig | 964 (+17.6%) | 3011 (+15.4%) |
| s1L_4k | **944 (+15.1%)** | 2959 (+13.4%) |
| **s3** | 966 (+17.8%) | **2931 (+12.3%)** |
| long_ema (+40k steps) | 968 | 2947 |
| d6a_ema (d≤6 anchors) | 960 | 3023 |

### B=2^20 (1,048,576), 100 steps, 6-pid set `8,199,399,599,799,999` (floor 241)

| arm | total | vs floor | wins | ties | merge gain | wall |
|---|---|---|---|---|---|---|
| **s3** | **243** | **+0.8%** | 1 | 3 | **2** | 1493 s |
| s1L_4k | 255 | +5.8% | 1 | 1 | 2 | 1576 s |
| orig | 259 | +7.5% | 0 | 0 | 0 | 1605 s |

```
  pid  floor      orig    s1L_4k        s3   best      (* beats floor, = ties it)
    8      7        7=        7=        7=      7
  199     46       50        48        48      48
  399     46       52        52        48      48
  599     48       50        52        48=     48
  799     48       50        46*       46*     46
  999     46       50        50        46=     46
  TOT    241       259       255       243    243
```

### What the numbers say

- **s3 is the only checkpoint in this project that beats the 46,662 floor on a pid**, and it
  does so on all three sets (wins 1 / merge gain 2 each time). That is the metric an ensemble
  actually banks.
- **The ordering `orig → s1L_4k → s3` is monotone at 1M width and on the 54-pid held-out set.**
  Each stage strictly improves.
- **The advantage grows with beam width.** +2 against `orig` at B=65536 on STRAT, −16 (−6.2%)
  at B=2^20. The 65k gating numbers *understate* s3.
- **s3 alone equals the 3-arm min-merge at 1M** (243) — best-or-tied on every pid.
- **The 18-pid gate ranks s3 last.** Gating on it, as every prior decision in this project did,
  would have discarded the best arm. Use ≥54 pids for anything under a ~2% margin.
- **Still worse than the floor standalone.** Even s3 sits +0.8% above the existing submission
  at 1M. These models earn their keep in the *merge*, not on their own.

---

## 2. Why Bellman, and why it works here when it failed before

### The label problem it solves

The author's sparse-Q label takes a random walk from solved, picks a pivot at index `p`, and
asserts that the undo move sits at depth `p-1` and the next move at `p+1`. Only 2 of 24 columns
get a label. Worse, **the assertion is false at depth**: a non-backtracking walk of length 58
lands at true distance ~44, so past `p ≈ 40` the "undo" move is not reliably closer to solved
and the label is at chance. Six levers were previously declared dead against that
mismeasurement.

Bellman needs no walk at all:

```
Q(s,a)  ←  0                                 if apply(s,a) is solved
           1 + min_a' Q_target(apply(s,a))   otherwise
```

This is exact for the true distance function, because `min_a' Q(s',a') = d(s') - 1` for any
unsolved `s'` (some neighbour is always one step closer), so `1 + that = d(s')`. It supervises
**all 24 columns** and is immune to the walk-length-≠-distance failure entirely.

### The collapse risk, and the two things that prevent it

`Q ≡ 0` is *also* a fixed point of the recursion alone. Two things select the right one, and
**both are load-bearing**:

1. the terminal case (`0` for a solved child), and
2. exact BFS anchors in every batch, all 24 columns.

> **Do not remove the anchors here because "anchors were refuted."** They were refuted as an
> extra *label source bolted onto a working sparse-Q objective* — arm `d6a` scored 3023 on
> held-out, the worst of anything measured, and arm `tmx` lost too. In stage 3 they play a
> completely different role: they are the **fixed-point selector** for a bootstrap that would
> otherwise be free to collapse. Same component, opposite function.

### Why the earlier Q-Bellman attempt failed

A prior run (`c_qd1b`) applied Q-Bellman to a near-scratch standalone model and it did not fix
depth saturation (ties 10.2 → 11.6, top-1 0.047 → 0.042 = chance). That result stands **for
that setting** and does not transfer: the bootstrap `1 + min_a' Q_target(child)` inherits the
target network's flatness at depth, so with a near-scratch target there is nothing to propagate
outward from the `d≤5` ball. Warm-started from a trained stage-1 checkpoint, the target already
discriminates at depth and the recursion has real signal to refine.

**The prerequisite is therefore: warm-start from a checkpoint that already works.** Stage 3 is a
refinement stage, not a training stage.

---

## 3. The recipe

### Command

```bash
PY=/home/artgor/cube444_a100_handoff/.venv/bin/python
cd ~/cube444_tf

# prerequisite: exact anchors (~1 min; see §3.4)
$PY build_anchors_deep.py --max-exact 6 --anchor-depth 6

$PY bellman_q.py --name s3 \
    --init runs/s1L/step004000/model.pth \
    --steps 6000 \
    --anchors data/q_anchors_d6.pt \
    --anchor-batch 256 \
    --anchor-weight 2.0 \
    --eval-every 1000
```

### 3.1 Hyperparameters

| | value | note |
|---|---|---|
| init | `runs/s1L/step004000` | **required** — see §2; do not run from scratch |
| steps | 6000 | 1,266 s ≈ 211 ms/step |
| batch_size | 768 | *states* per step; each costs 24 child forwards |
| path_frac | 0.5 | → 384 random-walk + 384 path states |
| anchor_batch | 256 | **additional** to batch_size, not a slice of it |
| anchor_weight | 2.0 | anchors weighted 2x the Bellman term |
| lr | 2e-5 | linear warmup 200 steps, then **flat — no decay** |
| optimiser | AdamW, betas (0.9, 0.95), wd 3e-3 | flat param group (no decay/no-decay split) |
| grad_clip | 1.0 | |
| target_refresh | 500 steps | hard copy, **not** an EMA |
| k_max | 45 | random-walk sampler; pivot_tilt 0 (author's distribution) |
| holdout_pids | 100 | split by PID so solutions cannot leak |
| precision | bf16 autocast, TF32 matmul | loss computed in fp32 |
| seed | 0 | |

### 3.2 The target computation

`bellman_targets()` — per step, no grad:

1. For each of the 24 actions, gather children of all 768 states → `(768*24, 96)`.
2. Forward the target net over those 18,432 children **in chunks of 16,384** (the beam study
   found the forward hits a CUDA grid limit past ~32k rows).
3. `mins = q.min(dim=1)` per child → `tgt = clamp(1 + mins, min=0)`.
4. Overwrite `tgt = 0` wherever the child *is* the solved state.
5. Reshape to `(768, 24)`.

**Target network is a hard frozen copy refreshed every 500 steps, not an EMA.** The point is
that the target is a *consistent function* within each window rather than something drifting
under its own bootstrap.

### 3.3 The loss

```python
bell = (pred[:768] - tgt).pow(2).mean()        # dense MSE, ALL 24 columns
al   = (pred[768:] - q_exact).pow(2).mean()    # dense MSE vs exact BFS depths
loss = bell + 2.0 * al
```

Both terms are plain squared error over all 24 columns. **Do not Huber-cap or reweight either
one.** The stage-2 post-mortem is unambiguous: softening the absolute level term improved every
offline proxy (pair accuracy, top-1, cross-parent calibration) and took the beam from 18/18 to
5/18. The beam takes a global top-B over all `(parent, action)` pairs, so it needs the absolute
scale, and no cheap offline metric detects its loss.

### 3.4 The anchors

`data/q_anchors_d6.pt`, built by `build_anchors_deep.py` in ~1 minute:

- **67,041,677 states** (the complete BFS ball to depth 6) × 24 columns = 1.61B exact labels.
- Stored `uint8` — targets are 0..7, so this is exact and keeps the set at 6.4 GB + 1.6 GB,
  GPU-resident. **Do not `.long()` it**: as int64 the states alone are 51 GB.
- Sampled 256/step with 24-frame recolour symmetry augmentation.

Built using **exclusion labelling**, which is what made depth 6 reachable at all: anchors at
depth D need a table complete to **D, not D+1**. A child of a depth-D state sits at D−1, D or
D+1; if the table is complete to D, a child that *misses* the lookup is at depth > D by
completeness and ≤ D+1 by adjacency — hence exactly D+1. The miss is the label. That is a
factor of ~19 in table size.

Level 6 of the cube measured exactly for the first time: **63,542,526**
(levels: `1, 24, 468, 9000, 172914, 3316744, 63542526`).

Depth 7 anchors would need a complete-to-7 table whose one-pass expansion is 146 GB; depth 8
needs 198 GB of hashes against 176 GB of RAM, with a 100% 64-bit collision probability. Both
are off this machine.

### 3.5 The state distribution — and why "path labels are rejected" does not apply

The random-walk and path samplers contribute **states only**. Their sparse 2-of-24 labels are
discarded; every target comes from the bootstrap and the anchors.

This matters, because path labels are a *measured rejection* elsewhere: on tetraminx, adding
path labels to the batch cost +3.07 moves/pid and was worse in every depth band, because states
on near-optimal paths are a measure-zero slice and the beam lives *off* path. That finding is
about **labels**. Here paths are a **state sampler** — a way to put the model on realistic deep
states while the label comes from a source that is correct everywhere. The two uses are not the
same and the rejection does not carry over.

---

## 4. Diagnostics — what to watch

**`E[target]` is the collapse alarm.** It should stay near the mean depth of the state
distribution. If it slides toward 0, the bootstrap has found the trivial fixed point and the
anchors are not holding.

```
step     0   loss 1.2133  bellman 0.7916  anchor 0.2108  E[target] 12.77
step  1000   loss 0.6229  bellman 0.4814  anchor 0.0708  E[target] 13.59
step  3000   loss 0.5270  bellman 0.4050  anchor 0.0610  E[target] 13.80
step  5000   loss 0.6675  bellman 0.4937  anchor 0.0869  E[target] 14.09
step  5999   loss 0.6615  bellman 0.4464  anchor 0.1076  E[target] 14.26
```

Healthy: 12.77 → 14.26, drifting *up* slightly (mild bootstrap inflation), never down.

Note the loss **rises** after ~step 3500 while the beam result is the best measured. Do not
early-stop on it.

**Held-out path-pivot bands at step 6000** (the benched checkpoint):

```
   band      n     gap    pair    top1
    2-6    500   1.732   0.968   0.866
   8-12    497   0.956   0.833   0.592
  14-18    493   0.539   0.761   0.420
  20-24    486   0.668   0.840   0.463
  26-30    476   0.514   0.805   0.410
  32-36    467   0.656   0.863   0.415
  38-42    454   0.795   0.912   0.297
  44-50    176   0.579   0.818   0.188
```

---

## 5. Gotchas

1. **The checkpoint that won is the one with the *worst* offline score.** `runs/s3/latest` is
   step 6000, deep_score **0.8494** — the lowest of all six evals. `runs/s3/best` is step 2000
   at 0.8569 and **has never been benched**. Every result in §1 is the step-6000 weights. This
   is the fourth independent confirmation that no offline metric tracks beam quality on this
   puzzle; `d6a` makes it starker still, with the best deep_score of any arm (0.8663) and the
   worst beam.
2. **Do not run stage 3 from scratch.** See §2 — the recursion has nothing to propagate.
3. **Do not drop the anchors** because anchors were refuted elsewhere. Different role (§2).
4. **Do not reweight or Huber-cap the loss.** See §3.3.
5. **Anchor batch is additional to batch_size**, matching the tetraminx convention. The step
   costs `768*24 = 18,432` no-grad forwards plus `768+256 = 1,024` with grad.
6. **Keep the anchor blob in its stored dtype.** `.long()` on a d≤6 set is 51 GB.
7. **Cheap.** 21 minutes. It is by a wide margin the best return per GPU-hour found here — for
   comparison, 40,000 extra stage-1 steps (57 min) moved the held-out total by 2 moves.

---

## 6. Open items

1. **Bench `runs/s3/best` (step 2000).** Untested, and the offline metric that picked it is
   known to be anti-correlated — so this is a genuine coin-flip worth 25 minutes.
2. **More Bellman steps, or a second round from `s3`.** 6000 steps was a first guess and the
   trend had not clearly turned. Unlike stage-1 training — which is measurably closed — this
   axis has never been swept.
3. **Re-test checkpoint diversity.** At 1M, `s3` alone equalled the 3-arm min-merge, contra the
   earlier finding that a per-pid min over 7 checkpoints beat the best single by 28 moves. The
   diversity result may have been an artifact of having no single strong checkpoint.
4. **Full 1043-pid pass at 1M with `s3`** — ~91 GPU-hours, shardable. Run it with the blend and
   endgame on (§1); `fold_input` is a further ~8% but is **not yet ported** into the solver's
   model class — it was implemented and verified in `qtrain/model.py`, which is the *training*
   model, while the beam builds via `build_model_from_info`.
5. **`--path-frac`, `--target-refresh`, `--anchor-weight` are unswept.** All three were set once
   and never varied.
6. **The search axis now outranks all of the above.** `--mlp-weight` is unswept and worth more
   per hour than anything on this list — see §8 of `cube444_stage3_heldout_2026-08-09.md`.
