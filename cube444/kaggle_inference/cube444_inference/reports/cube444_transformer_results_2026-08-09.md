# CayleyPy 4x4x4 — PieceTransformer: training + 1M beam results

**Date:** 2026-08-09 · **Machine:** A100 80GB (`NVIDIA PG509-210`)
**Companion to:** `cube444_transformer_training_plan_2026-08-08.md` (the plan; this is what happened)
**Code:** `~/cube444_tf/` · **Solver:** `~/cube444_tf/solver/` (patched copy of the bundle)

---

## Headline

**A verified submission at 46,658 — 4 moves under the best public file (46,662).**

```
rows 1043  PASS 1043  FAIL 0  total moves 46658
VERDICT: PASS
```

Produced by a standalone PieceTransformer Q head at beam width 2^20, min-merged with the
46,662 floor. File: `~/cube444_tf/solver/results/bench_s1_B1M.csv`.

The gain is one pid (797: **46 vs the floor's 50**), plus 7 pids tied exactly at the floor
out of the 18 deepest. Thin sample — 1.7% of the test set — but replay-verified.

---

## 1. What was built

`~/cube444_tf/` — a standalone trainer for the PieceTransformer Q head.

| file | purpose |
|---|---|
| `qtrain/model.py` | PieceTransformer, parameter names identical to the bundle so checkpoints load both ways |
| `qtrain/puzzle.py` | generators, inverses, 24-frame recolour symmetry; self-tests the transport identity |
| `qtrain/sampler.py` | random-walk pivots (author's distribution + `pivot_tilt`), geodesic-path pivots, exact anchors, symmetry **coverage** expansion |
| `qtrain/loss.py` | sparse-Q objective with an explicit level/gap split |
| `qtrain/evaluate.py` | by-depth gating on path pivots and random-walk pivots |
| `qtrain/checkpoint.py` | saves `model.json` + `model.pth` in the bundle's own schema, so the solver runs them unmodified |
| `train.py` | presets `stage1` / `stage2` / `tmx` |
| `bellman_q.py` | Stage-3 Q-Bellman refinement (written, not run — see §6) |
| `build_anchors.py` | BFS ball + exact 24-column anchors |
| `bench_beam.py` | matched multi-arm beam A/B on a fixed pid set |
| `run_sweep.sh` | chained train → beam-bench |

Verified before any training:

- generators, move-name order and solved state **byte-identical** to the bundle
- BFS level sizes reproduce exactly: `1, 24, 468, 9000, 172914, 3316744`
- symmetry transport identity `sym(apply(s,g),R) == apply(sym(s,R), relabel[k,g])` passes
- exact anchors: 182,407 states at d<=4, all 24 columns, `q[solved] = [1,1,1,...]`
- `stage1` preset reproduces the bundle objective **bit-for-bit** (`loss == sparse_mse == 6.7832`,
  and step 50 = 6.9153 identical to an independent earlier run)

---

## 2. Stage results

| stage | what changed | beam result |
|---|---|---|
| **Stage 1** (`s1`) | nothing — just 3k more steps of the shipped recipe | **18/18 vs shipped model's 16/18**, +2 merge |
| **Stage 2** (`s2`) | path labels + Huber level + 4x gap weight + margin + anchors + symmetry | **5/18 solved, +87% vs floor** — catastrophic |
| **tmx** | tetraminx recipe: `pivot_tilt 0.5` + coverage expansion + anchors, objective untouched | 988 vs s1's 970 — mildly worse |
| **Stage 3** | Q-Bellman | not run (see §6) |

### Stage 2 is the important negative result

Stage 2 improved **every** offline metric and destroyed the beam:

| arm | held-out deep pair acc | cross-parent acc | beam B=65536, 18 deepest pids |
|---|---|---|---|
| shipped | 0.845 | 0.912 | 16/18, +9.9% |
| s1 | 0.854 | 0.909 | **18/18, +9.7%, +2 merge** |
| s2 | **0.858** | **0.932** | **5/18, +87%** |

Cross-parent calibration was the obvious suspect and was explicitly checked — s2's `minQ`
tracked true depth far better (39.3 at d=45 vs the shipped model's 29.3). It still could not
solve. Doc 04's "KEEP THE ABSOLUTE MSE TERM ... the beam takes a global top-B over all
(parent, action) pairs" survives even a *Huber-softened* version of the term.

Methodological note: Stage 2 changed six things at once, violating rule 5. The attribution to
objective reweighting is consistent with the evidence but was not isolated. The `tmx` run
above is the clean test of the two data levers (anchors, coverage), and they came out mildly
negative on their own.

---

## 3. The beam-gated sweep — the main experiment

After three failures where offline metrics pointed the wrong way, selection moved entirely to
the beam. Training drops step-indexed checkpoints; no metric picks anything.

**18 stratified pids** (floor length >= 40, evenly spaced), B=65536, 110 steps, 1 attempt,
standalone transformer. Floor total on this set = **820**.

| arm            | cumulative steps | own total | vs floor   | matches per-pid best |
| -------------- | ---------------- | --------- | ---------- | -------------------- |
| orig (shipped) | 0                | 964       | +17.6%     | 6/18                 |
| s1             | 3k               | 952       | +16.1%     | 8/18                 |
| **s1L_4k**     | **7k**           | **944**   | **+15.1%** | **11/18**            |
| s1L_8k         | 11k              | 976       | +19.0%     | 5/18                 |
| s1L_12k        | 15k              | 980       | +19.5%     | 5/18                 |
| s1L_16k        | 19k              | 974       | +18.8%     | 4/18                 |
| s1L_20k        | 23k              | 960       | +17.1%     | 7/18                 |

**The curve is non-monotone and peaks early**: best at ~7k cumulative steps, then degrades
*past the shipped model*, then partially recovers. Training loss sat flat at ~6.5 across the
entire range while the beam swung 944 -> 980 -> 960.

`s1L_4k` wins on both axes — lowest total and most per-pid bests — which is a more robust
signal than the total alone.

### Per-pid detail (the variance is the story)

```
  pid floor     orig       s1   s1L_4k   s1L_8k  s1L_12k  s1L_16k  s1L_20k
   97    46       48       48       48       48       48       48       48
  118    47       51       55       55       55       55       57       53
  133    40       50       50       44       48       48       48       48
  193    48       50       54       54       54       54       54       54
  294    45       55       53       53       55       55       55       55
  299    46       56       52       58       56       54       54       54
  346    47       53       53       51       51       55       51       51
  467    44       56       52       52       56       56       58       58
  487    46       54       52       52       52       52       52       52
  582    45       59       57       55       55       55       55       55
  610    47       55       55       55       53       53       55       55
  685    46       52       52       52       54       58       54       52
  707    48       56       60       54       58       54       58       54
  846    45       55       49       55       57       59       55       55
  848    47       49       53       51       51       51       51       51
  881    46       52       54       52       60       64       60       60
  946    43       57       53       53       57       53       53       49
  985    44       56       50       50       56       56       56       56
TOTAL   820      964      952      944      976      980      974      960
```

### Checkpoint diversity beats checkpoint selection

```
per-pid MIN over all 7 arms = 916   vs best single arm 944   (-28 moves)
```

**28 moves better than the best checkpoint**, from models that are individually worse. Per-pid
swings are large and uncorrelated: pid 881 goes 52/54/52/60/64/60/60; pid 133 goes
50/50/44/48/48/48/48. They explore differently.

This is handoff rule 4 and README point 3 ("prefer several diverse runs to one wider one")
reproducing on **checkpoints** rather than seeds. It is probably the most actionable result in
this document.

### Caveat

Single seed. Per-pid swings of 6-12 moves between adjacent checkpoints mean total differences
of 8-20 moves over 18 pids are within plausible noise — `s1L_4k` over `s1` (8 moves) is not
defensible on this evidence alone. The 944-vs-980 spread and the 11/18-vs-4/18 split are the
parts worth trusting.

Also: **no arm beats the floor on a single pid at B=65536.** This width ranks checkpoints; it
does not produce submissions. Wins only appear at 2^20.

---

## 4. The 1M beam

`s1`, standalone, B=2^20 (1,048,576), 100 steps, 1 attempt, 18 deepest pids
(floor length 49-50, floor total 884). Wall 5,664 s = **315 s/pid**.

| | |
|---|---|
| solved | **18/18** |
| own total | 904 vs floor 884 (**+2.3%**) |
| wins / ties | **1 win, 7 exact ties** |
| merge | 880 (**-4 moves**) |

```
  pid   60      len  51 floor  49 (+2)
  pid   76 TIE  len  49 floor  49 (+0)
  pid  102      len  51 floor  49 (+2)
  pid  134 TIE  len  49 floor  49 (+0)
  pid  148      len  51 floor  49 (+2)
  pid  248      len  53 floor  49 (+4)
  pid  250      len  51 floor  49 (+2)
  pid  274 TIE  len  49 floor  49 (+0)
  pid  278 TIE  len  49 floor  49 (+0)
  pid  297 TIE  len  50 floor  50 (+0)
  pid  312      len  51 floor  49 (+2)
  pid  330 TIE  len  49 floor  49 (+0)
  pid  342      len  51 floor  49 (+2)
  pid  364      len  53 floor  49 (+4)
  pid  380      len  51 floor  49 (+2)
  pid  420 TIE  len  49 floor  49 (+0)
  pid  422      len  51 floor  49 (+2)
  pid  797 WIN  len  46 floor  50 (-4)
```

Width is doing real work: the same model is **+9.7% at B=65536 with zero wins** and **+2.3% at
B=2^20 with one win and seven ties**. That is the width-capped regime the tetraminx analysis
says the transformer belongs in.

Throughput measured on this A100, fp16 (`torch.compile` adds nothing; fp16 eager is best):

| batch | ms/step | M child-scores/s |
|---|---|---|
| 4096 | 26.7 | 3.69 |
| 16384 | 105.1 | 3.74 |

Forward hits a CUDA grid limit past ~32k rows — chunk beam inference at <=16384.

**Full-set cost:** 315 s/pid x 1043 = **~91 GPU-hours** for a complete 1M-width pass, shardable.

---

## 5. Findings that change the handoff docs

### (a) "The scorer is blind past depth 40" is a measurement artifact

The single most consequential finding. Same model, same metric, two pivot sources:

| depth band | random-walk pivots | geodesic-path pivots |
|---|---|---|
| 26-30 | pair 0.694 | **0.809** |
| 38-42 | pair 0.564, gap 0.170 | **0.877, gap 1.371** |
| 44-50 | pair 0.526, gap 0.067 | **0.792, gap 0.825** |

A non-backtracking walk of length 58 lands at true distance ~44, so past depth ~40 the "undo"
move is not reliably closer to solved and **the label goes to chance, not the model**.
`04_TRAIN_SPARSE_Q.md` and `CLAUDE.md` both build on the random-walk number; six levers were
declared dead against it.

Caveat not controlled: the 46,662 paths are solver-found, so some selection effect is possible.
Not 0.56->0.88 worth.

### (b) The bundle author's sampler was mis-read, and is very shallow

Recovered by scoring candidate pivot distributions against the `last_validation` block shipped
for `mlp_x16` (sparse_mse 8.576 / pair 0.886 / top1 0.589):

- **`k ~ U[2,45]`, then `p ~ U[1,k-1]`** -> 8.38 / 0.898 / 0.602 (match, 6% rel err)
- uniform `p in [2,44]` (the naive reading) -> 33.2 / 0.756 / 0.323 (off by 4x)

Consequences: `E[p] = 11.75`, `P(depth >= 40) = 0.80%`, hard zero past 45. The shipped
transformer has barely been trained where the test set lives, and saturates at ~31.5 (at pivot
40 it scores the undo action 30.67 against an all-24-action mean of 30.82).

### (c) The transformer beats the 47M MLP with 14x fewer params

At matched objective, on the shipped weights, `piece_transformer` (3.38M) beats `mlp_x16`
(47M) at every depth past 8. That settles cells C/D: architecture does matter, and doc 04's
"neither — it's inference economics" was concluded from a 3.3M ResMLP-Q that does not
generalise.

### (d) No offline metric tracks beam quality — confirmed three times

| experiment | offline says | beam says |
|---|---|---|
| s2 (reweighted objective) | better (pair 0.858, x-parent 0.932) | catastrophic (5/18) |
| tmx (tilt + coverage + anchors) | best of all (deep_score 0.8646) | worse than s1 |
| s1L sweep | loss flat at ~6.5 throughout | swings 944 -> 980 -> 960 |

Rule 7 ("loss is not a proxy for beam quality") is stronger than it reads: it applies to
discrimination metrics too, not just loss. Checkpoint selection by `deep_score` was actively
harmful — it is anti-correlated with the objective.

### (e) The tetraminx recipe does not transfer as-is

From `new_package/RESMLP_VS_TRANSFORMER.md`, the config that beat ResMLP (`mx_tf.yaml`) uses
`pivot_tilt 0.5`, `sym_coverage: true`, exact anchors, `top1_margin_weight: 0.0`, and **no path
labels**. Ported faithfully to cube444 (`tmx` preset), it came out mildly worse than plain
continuation. Two structural differences worth noting:

- cube444's 24 rotations give generator orbits of 6, so coverage expansion caps at 12 of 24
  columns — measured **width 10 rows/sample, mean 10.70 columns covered**. Tetraminx reaches
  width 14. The 48 mirror frames would lift this to ~18.26 but are not derived on this box.
- the tetraminx transformer was trained from scratch for 500-1500 epochs; here everything is a
  warm start off an already-trained checkpoint, so the regimes are not comparable.

---

## 6. Not done

1. **Stage 3 (Q-Bellman).** Written (`bellman_q.py`), not run. Deprioritised on evidence: a
   parallel session already ran Q-Bellman on cube444 and it did **not** fix depth saturation
   (ties 10.2 -> 11.6, top-1 0.047 -> 0.042 = chance).
2. **Full 1043-pid 1M pass.** Only 18 pids run. ~91 GPU-hours for the full set.
3. **48 mirror frames.** Would raise coverage 10.70 -> 18.26 columns/pivot and is also an
   untested ~2 moves/pid in the beam's `--sym-ensemble`. Derivation script is not on this box.
4. **Second seed on the sweep.** Needed before defending the fine ordering of checkpoints.
5. **Ensemble with `mlp_x16`.** The bundle author measured 0.60/0.40 at -2.07 moves vs the
   transformer alone. Both models ship; never tested here.
6. **Training-speed fixes.** `RESMLP_VS_TRANSFORMER.md` §5 found the embedding gradient was
   **67% of the entire training step**; recasting the input stage as `one_hot(vals) @ table`
   took 450 -> 162 ms. A ~2.9x training speedup, untouched here.

---

## 7. Recommended next step

The diversity result points away from hunting for a better single checkpoint — three attempts
have failed — and toward **min-merging several diverse checkpoints at full width**. The
per-pid min over 7 checkpoints was 28 moves better than the best one at B=65536; the same
mechanism at 2^20 is the cheapest untried lever.

Concretely: `s1`, `s1L_4k`, `s1L_20k` at B=2^20 on the 18 deepest pids, min-merged against the
floor. ~4.7 h. Then, if it holds, shard the full 1043 across the same three checkpoints.

---

## Appendix — reproduce

```bash
PY=/home/artgor/cube444_a100_handoff/.venv/bin/python
cd ~/cube444_tf

# exact anchors (self-checks BFS level sizes)
$PY build_anchors.py

# faithful continuation
$PY train.py --name s1 --preset stage1 --steps 3000 --batch-size 4096

# beam-gated sweep
./run_sweep.sh

# 1M beam, standalone, 18 deepest pids
$PY bench_beam.py --arms "s1=runs/s1/best" --beam 1048576 --steps 100 \
   --pids 60,76,102,134,148,248,250,274,278,297,312,330,342,364,380,420,422,797
```

Stratified 18-pid gating set:
`133,946,467,985,294,582,846,97,299,487,685,881,118,346,610,848,193,707` (floor 820)
