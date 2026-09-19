# CayleyPy 4x4x4: training the PieceTransformer as a standalone scorer

**Date:** 2026-08-08 | **Machine:** single A100 80GB | **Repo:** `~/cayley`
**Competition:** https://www.kaggle.com/competitions/cayley-py-444-cube (deadline 2026-09-22)

Executed the 8-step plan for cell D of the `cube444_a100_handoff` 2x2: build and train the
PieceTransformer Q head as a **standalone** scorer (not a reranker). All 8 steps ran to
completion. The headline result is not the one the plan expected.

---

## 1. Headline: the handoff's central premise is a measurement artifact

Everything in the handoff since 2026-08-08 rests on one number, from `32_eval_q.py`:

> "past depth ~40 the deployed V's pair accuracy is 0.448 -- BELOW CHANCE -- and that is
> where ~1000 of the 1043 test pids live"

That diagnostic builds pivots from **random walks** and asserts the true gap is 2 at every
depth. But `ANALYSIS_AND_PLAN.md` 1.3 measures that walks on this puzzle are **fully mixed
by length 26-30** (hamming 79.5 -> 80.0 = chance). Past mixing, `s_{p-1}` and `s_{p+1}` are
both just uniform-random states at true distance ~38: the real gap is ~0 and the real pair
accuracy is ~0.5. **A perfectly calibrated scorer scores 0.5 on that diagnostic at depth.**

The two statements contradict each other, and the reference trainer already said so --
`tetraminx_51_train_sparse_q.py::PathLabels` line 216: *"past the diameter ... 'undo the
last move' stops being reliably distance-reducing."*

New script `33_eval_path.py` runs the same shape of measurement on labels that are near-true
at depth: pivots replayed from the verified 46,662 submission (45,619 samples; that file
averages 44.74 against a 36.8 counting bound, so its paths are within a few moves of
geodesic). For a state at path position `i` with remaining `r`:

- on-path action -> child at remaining `r-1`
- back action (inverse of the entering action) -> child at remaining `r+1`

| remaining | V `c_bells2` pair | top-1 | Q `c_qb` pair | top-1 |
|---|---|---|---|---|
| 1-5 | 0.993 | 0.911 | 0.961 | 0.840 |
| 11-20 | 0.701 | 0.288 | 0.641 | 0.255 |
| 21-30 | 0.727 | 0.315 | 0.687 | 0.266 |
| 31-35 | 0.779 | 0.303 | 0.732 | 0.218 |
| **36-40** | **0.842** | **0.265** | 0.735 | 0.163 |
| **41-45** | **0.779** | **0.167** | 0.617 | 0.087 |
| **46-60** | **0.693** | **0.101** | 0.560 | 0.076 |

Chance: pair 0.500, top-1 0.042.

**The deployed V is not blind at depth.** At remaining 36-40 it is 6x chance on top-1, where
the walk diagnostic reported below chance. Six levers (walk-label fix, 4.5x BFS ball,
Bellman second pass, AZ dual head, 3.3M -> 15M capacity, beam width to 2^20) were declared
dead against this mismeasurement.

Second-order finding: **V beats Q at every band on every metric.** Cell B's win was purely
inference economics, not any parity of scorer quality -- which sharpens rather than
contradicts the handoff's conclusion.

---

## 2. Cell D standalone: it failed

`c_qd1`: PieceTransformerQ, `d_model=256, nhead=8, num_layers=4, ff_dim=1024` on the
57-token layout = **3,383,321 params**, capacity-matched against the deployed V
(3,291,137) and the cell-B Q head (3,302,936). Trained 1900 epochs / 5,510 s, matched to
cell B's 5,730 s budget.

Beam bench, B=8192, sym-4, 18 pids, two symmetry seeds, per-arm 2-seed min:

| arm | solved | total moves | vs floor | wall/seed |
|---|---|---|---|---|
| **D1 PieceTransformer** | **9/18** | 214 | +52.9% | 216 s |
| cell-B Q `c_qb` | 14/18 | 576 | +54.8% | 22 s |
| V `c_bells2` | 17/18 | 736 | +43.8% | 88 s |
| merge of all three | 18/18 | 802 | -- | -- |

**D1's merge contribution is 0**: zero pids it alone solved, zero pids where it is strictly
shorter, zero pids where it beats the floor. It solved only shallow pids (longest path 18
moves). D1's "+52.9%" is not comparable to the others -- it is computed over a much easier
9-pid subset.

The iso-wall-clock round was deliberately skipped: iso-wall gives the *competitors* more
width, so it can only be worse for D1 than the iso-width result above.

### Failure mode: saturation at depth

Visible only with a **tie-aware** metric. The first version of `33_eval_path.py` reported
pair accuracy *below* top-1, which is impossible without ties (rank 0 implies
`q_on < q_back`). A saturated head emits a near-constant 24-vector, and then "nothing
strictly beats the on-path action" is true of all 24 actions. Mean actions tied with the
on-path action:

| remaining | V `c_bells2` | Q `c_qb` | D1 stage 1 | D1 + Q-Bellman |
|---|---|---|---|---|
| 1-5 | 1.0 | 1.1 | 1.1 | 1.1 |
| 21-30 | 1.4 | 1.5 | 2.9 | 3.3 |
| 36-40 | 1.5 | 2.3 | 7.4 | 8.5 |
| 46-60 | 2.4 | 5.0 | 10.2 | 11.6 |

Ties track inversely with Bellman refinement -- but the Q-Bellman stage **did not fix it**
(top-1 at 46-60 went 0.047 -> 0.042 = chance). The bootstrap
`Q(s,a) <- 1 + min_a' Q_target(T(s,a), a')` inherits the target net's flatness at depth, and
50 target refreshes did not propagate exactness from the d<=5 exact ball out to depth 40.
Rule 9's canary was clean throughout (`Q(solved) = 0.95-1.08`, not the ~2 the rule warns of),
so this is not an anchor failure.

### This does not refute the architecture

At matched wall clock D1 saw **5.8x fewer training rows** (272M vs cell B's 1,584M). For a
model this much slower per state, "undertrained" and "too slow to train at this budget" are
the same statement. Two further self-imposed handicaps: `sym_rows=4` of the available 10,
and no mirror frames (see section 5).

---

## 3. Two corrections made mid-run

**Sampler (large).** Stage 1 was first built on `cayley.sparse_q.sparse_q_batch` -- fixed
`k_max=60`, emitting *every* pivot, giving E[pivot depth] 30 and P(depth>=40) = 33.9%. Pair
accuracy went *below* chance and stayed there. Killed at epoch 677 and restarted on the
shipped cube4 ensemble's actual sampler config
(`{kind: rw_middle_sparse, k_min: 2, k_max: 45, selection: skip_index}`): k ~ U[2,45] then
p ~ U[1,k-1], verified E=11.79 / P(>=40)=0.76% against the recovered 11.75 / 0.80%.
Same epoch after the change: **pair 0.649 vs 0.385, gap 0.36 vs 0.04.** This is a fourth
silent deviation in the v1 port, on top of the three its own docstring lists.

**Throughput (changes the deployment arithmetic).** The handoff's "0.24x / 529 ms at
B=65536" is a 4090-laptop number at ~20 TFLOP/s, about 6% of A100 bf16 peak. Re-measured
here (`bench_q_arch.py`):

| ratio | handoff (4090, fp16) | A100 model-only | A100 end-to-end beam |
|---|---|---|---|
| D1 vs deployed-V beam | 4.1x | 2.13x | **2.45x** |
| D1 vs cell-B Q | 34x | 50.6x | **9.8x** |

End-to-end ratios are much smaller than model-only against cell B, because the beam's
gather/hash/unique/top-k cost is not amortised for a fast scorer. `torch.compile` helps D1
1.6x but the ResMLP 2.8x, so it *widens* the gap -- and gotcha 11 blocks it in the beam
anyway.

**Hard constraint found:** SDPA's flash kernel caps the **batch** dimension at 65535 rows
independent of nhead (B=65535 runs, B=65536 dies with "invalid configuration argument"). So
unlike the ResMLP -- where `--inference-chunk-size 0` is the documented fastest setting --
the transformer must stay chunked, or a beam at width 2^17 crashes inside the first
attention layer. Encoded as a default of 32768 plus an explicit error.

---

## 4. What was built

All in `~/cayley`, following repo conventions (ASCII output, `encoding="utf-8"`, the
`parents[1]` PROJECT convention).

| file | purpose |
|---|---|
| `scripts/50_derive_layout_444.py` | 57-token piece layout, centres KEPT (`78_piece_model.py` drops them for the reduced 3x3x3 phase). Validated: 8/24/24 partition, block system in both directions, and the published 3x3x3 QTM level counts to depth 6 (1, 12, 114, 1068, 10011, 93840, 878880). |
| `src/cube444/piece_transformer.py` | `PieceTransformerQ`, ported from the tetraminx reference with the colour-cube deltas (`num_classes=6`, `n_actions=24`, no `invert_state`, 56 pieces). Folded input stage, SDPA not MHA. |
| `src/cube444/qtrain.py` | `Symmetries444` (recolour conjugation + coverage table), `expand_labels`, `PathLabels444`, `profile_loss`, `SparseQSampler444`. |
| `scripts/36_child_profile.py` | Measures the sorted child profile on the exact ball -> `profile_high_from`. |
| `scripts/37_train_qd1.py` + `configs/qd1_piece_transformer.yaml` | Stage 1 trainer. |
| `scripts/35_bellman_q.py` + `configs/qd1_bellman.yaml` | **New code**: the Q-form Bellman recursion. `bellman.py` implements only the scalar form. |
| `scripts/33_eval_path.py` | The path-pivot depth diagnostic, tie-aware. |
| `scripts/51_derive_mirrors.py` | Mirror symmetry derivation (section 5). |
| `scripts/bench_q_arch.py`, `run_d1_bench.sh` | Throughput microbench and the beam bench. |

**Everything the v1 port stripped is now wired in**: the author sampler, exact 24-way
anchors (d<=4, 182,407 states), recolour symmetry expansion, 1,119,888 path labels
(46,662 x 24 frames), the measured sorted-profile loss, and the auxiliary value head.
Deliberately still off, following the reference: `margin_weight=0`, `path_rank_weight=0`
(the reference measured 56.9% of such disagreements to be benign alternative optima), the
triangle box (the reference rejected it -- satisfied exactly by a constant), `pivot_tilt=0`.

**The sorted profile does not transfer from tetraminx.** Measured on the exact ball, cube444
gives `profile_high_from = 6` against tetraminx's 9; inheriting 9 would have left ranks 6-8
unsupervised for no reason. cube444's profile is much sharper (mean 1.19 distance-reducing
children of 24, and exactly zero same-distance children -- the QTM graph is bipartite here).

---

## 5. New capability: 48 symmetry frames, not 24

`ANALYSIS_AND_PLAN.md` 1.5(b) says *"Mirror reflections would extend this to 48"* and nobody
built them. They are now derived and verified.

Rotation-only label expansion is capped at **12 of 24 columns**, and this is
group-theoretic, not an implementation limit: the 24 generators form four orbits of 6 under
the rotations, split by chirality --

```
{f0, -f3, r0, -r3, d0, -d3}    {-f0, f3, -r0, r3, -d0, d3}
{f1, -f2, r1, -r2, d1, -d2}    {-f1, f2, -r1, r2, -d1, d2}
```

so a labelled (undo, next) pair can only reach `orbit(undo) U orbit(next)`.

Mirrors are valid on a colour cube even though `invert_state` is not: a reflection maps the
solved **colouring** to a recolouring of itself, whereas inversion is undefined on a coset
graph. `51_derive_mirrors.py` derives them with no geometry hardcoded -- constraint
propagation on `M[move_g[p]] = move_{sigma(g)}[M[p]]`, seeded once per closed slot orbit,
accepted only on the full transport identity that `test_symmetry.py` applies to the
rotations.

Result: **48/48 composed frames verify, all 48 slot permutations distinct**, orbits fuse
[6,6,6,6] -> [12,12], and label coverage goes from a mean 10.70 of 24 columns (24 frames,
greedy width 10) to **18.26** (48 frames, width 20). Representative mirror swaps axes f<->r
with a global direction flip (`f0 -> -r0`). Tables written to
`data/mirror_{rotations,color_maps,move_relabel}_48.npy`.

---

## 6. What I would do next, in order

1. **Wire the 48 frames into the beam's `--sym-ensemble`.** This is the largest measured-
   value item and it needs no training. `05_BEAM_SEARCH.md` measures ~-2 moves per doubling
   of K, and 24 -> 48 is one more doubling that has never been available. The solver
   currently loads `rotations_24.npy` directly; pointing it at the 48-frame tables is a
   small change. Untested -- treat the ~2 moves/pid as a hypothesis, and judge it per rule 4
   on the per-pid min against the floor.
2. **Re-audit the six "dead" levers against `33_eval_path.py`.** They were all killed on the
   walk diagnostic. At minimum the Bellman second pass and the 15M capacity arm deserve a
   re-read, since both showed gap-by-depth movement that the walk metric could not resolve.
3. **Fix D1's depth saturation before any retry.** Gate on tie-aware top-1 at remaining
   36-45, not on loss and not on the walk metric. The specific untried knobs: raise the
   sorted-profile weight (0.5 is a guess -- the reference's own final weights are not in the
   shipped bundle), raise `sym_rows` to 10 (or 20 with mirrors), and initialise Q-Bellman
   from a model that is not already flat at depth.
4. **A cheaper standalone transformer only if 1-3 do not settle it.** At 9.8x end-to-end
   against cell-B Q, D1 must win a width contest it currently loses at equal width.

## 7. Caveats

- The path diagnostic and the path labels both come from the 46,662 file, which is a
  community artifact -- `README.md` flags that this changes the provenance story for an
  otherwise self-generated project. Our own beams cannot substitute: 0 of 343 own-beam paths
  are shorter than it on shared pids, so they would only supply looser bounds.
- Those paths are solver-found, so some selection effect on the diagnostic is possible. The
  magnitude of the difference (0.842 vs 0.448) is far too large to be only that, but it is
  not controlled.
- The 18-pid bench has a real noise floor: the same checkpoint swings ~4 pids of symmetric
  difference on a symmetry-seed change alone. Two seeds were run per arm; the D1 vs V gap
  (9/18 vs 17/18) is far outside that floor, but smaller differences in these tables are not.
- The shipped reference transformer (`cube4_full_ensemble_inference_w040`) is not on this
  box, so D1 could not be compared against the author's own trained weights -- which would
  be the cleanest possible control and is drop-in compatible with our beam.

---

## CORRECTION (added same day, after review)

**Section 2's negative result is not a valid test of cell D. Disregard it.**

I sized the run to cell B's 5,730 s on the doc-04 rule "compare at matched budget". That is
the right rule for a deployment A/B between two *trained* models; it is the wrong rule for
asking whether an architecture can learn the task. The reference PieceTransformer on
tetraminx took **2-3 days** to train. My run was **1.53 h -- 2-3% of the compute the
architecture is known to need.**

The training log says the same thing plainly. Loss by epoch, and the drop over the previous
200 epochs:

| epoch | loss | ptop1 | lr | drop vs -200 ep |
|---|---|---|---|---|
| 800 | 52.79 | 0.142 | 1.87e-04 | +19.0% |
| 1000 | 43.89 | 0.154 | 1.37e-04 | +16.9% |
| 1200 | 37.28 | 0.157 | 8.95e-05 | +15.0% |
| 1400 | 32.45 | 0.164 | 4.82e-05 | +13.0% |
| 1600 | 30.38 | 0.166 | 1.80e-05 | +6.4% |
| 1899 | 28.55 | 0.167 | 0.00e+00 | +4.8% |

Loss was still falling 13-15% per 200 epochs at epoch 1400 and only flattened once the
cosine schedule annealed LR to zero. **With `T_max = n_epochs`, a run always looks converged
at its own horizon** -- the schedule manufactured the appearance of convergence. The
saturation-at-depth diagnosis in section 2 is therefore a property of an undertrained model,
not established as a property of the architecture.

### Relaunched properly

`c_qd1_long`, running detached, with the two handicaps removed that only existed to fit 92
minutes:

| | failed run | relaunch |
|---|---|---|
| horizon | 1,900 ep / 1.53 h | **24,000 ep / 60 h (2.5 days)** |
| training rows | 272M | **12.8B (47x)** |
| symmetry frames | 24 rotations | **48 (rotations x mirrors)** |
| sym rows/pivot | 4 of 10 | **20 of 20** |
| columns labelled | 6.35 of 24 | **18.26 of 24** |
| anchors / path rows per step | 1024 / 2048 | 2048 / 4096 |

Unchanged: author sampler, exact 24-way anchors, sparse-Q masked MSE, path labels, measured
sorted profile, value head. Checkpoints every 500 epochs, so intermediate ones can be gated
with `33_eval_path.py` while the run continues.

---

## LEVER RE-AUDIT (added same day)

Section 1 showed the six levers were killed on a diagnostic that cannot see depth. Five have
surviving checkpoints; re-measured on path pivots, tie-aware top-1, 45,619 samples:

| lever | 36-40 | 41-45 | 46-60 | deep avg | vs base | corrected verdict |
|---|---|---|---|---|---|---|
| base V 3.3M (old BFS) | 0.265 | 0.167 | 0.101 | 0.214 | -- | -- |
| L1 walk-label fix | 0.264 | 0.166 | 0.103 | 0.212 | -0.6% | confirmed dead |
| L2 BFS ball 5.7M -> 25.3M | 0.263 | 0.165 | 0.099 | 0.211 | -1.1% | confirmed dead |
| L3 Bellman 2nd pass (frontier) | 0.258 | 0.161 | 0.105 | 0.207 | -3.1% | **worse, not better** |
| L4 AZ dual head | 0.262 | 0.162 | 0.132 | 0.211 | -1.2% | confirmed dead |
| **L5 capacity 3.3M -> 15M** | **0.293** | **0.172** | **0.134** | **0.231** | **+8.3%** | **NOT dead** |

(L6, beam width to 2^20, is not a scorer property and is not testable this way.)

**L3 resolves an open puzzle.** Frontier replay improved gap-by-depth 10-18% on the walk
metric and then regressed the beam 16/18 -> 15/18, which was written off as inside the noise
floor. On true labels it is -3.1%. The walk-metric "improvement" was the model fitting a
label that is false at depth better; the beam regression was real signal.

**L5 reopens the capacity axis.** The 15M trunk is the best scorer of all six and uniformly
so -- highest pair accuracy at every deep band (0.858 vs 0.842 at remaining 36-40) and +33%
top-1 at remaining 46-60. It was killed on a **15-pid** beam delta of -1.36%, which the
handoff itself calls "inside the noise floor" (individual deep pids swing +-30 moves on a
symmetry-seed change). CLAUDE.md says "do not re-litigate" capacity; on 45,619 path pivots
that verdict does not hold.

Caveat: this is a scorer-quality result, not a deployment one. The 15M still costs 2.7x wall
clock, so "deploy the 3.3M at fixed wall" may survive. What is contradicted is the narrower
claim "no capacity gain" -- there is one, at exactly the depths that matter, and it was
measured away by a diagnostic that could not see depth.

**Revised priority order** (supersedes section 6): (1) wire the 48 frames into the beam's
`--sym-ensemble`; (2) re-test the 15M trunk on a proper pid set, judged per rule 4 on the
per-pid min against the floor; (3) let `c_qd1_long` finish and gate it on `33_eval_path.py`;
(4) treat L1/L2/L3/L4 as settled dead.

---

## CORRECTION 2: the tetraminx package invalidates two of my loss terms

`new_package.zip` (the full tetraminx work) arrived after the above. Three of my choices
are directly contradicted by measurements already in it, and one of them explains the
depth saturation I diagnosed as an architecture property.

### The deployed recipe is exactly three label sources

`configs/mx_tf_az.yaml` is the config behind the model actually deployed on tetraminx
(cell 4, PieceTransformer + AZ head, ep1500). It uses **sparse-Q rw-middle labels, exact
24-way BFS anchors, and symmetry label expansion** -- and nothing else. No path labels, no
profile, no box.

### What I added, and what it cost

**Path labels -- REJECTED as `tq1`, 2026-07-28.** Same 15 pids, Q@1M, matched epoch 500:
`tq0` (walk labels only) **459**; `tq1` (+ path labels) **505**. That is +3.07 moves/pid,
and the probe agreed -- worse in EVERY depth band, *including the 30-40 band the change was
designed to fix* (gap 0.248 -> 0.186, top1 0.091 -> 0.074). Their diagnosis: states on
near-optimal paths are a measure-zero slice, the beam spends nearly all its time OFF-path,
so correct labels there do not generalise and they divert capacity from the walk
distribution that matters. **They ran it at 3.4 pct of each batch; I ran it at 15.4 pct --
4.5x the already-rejected dose.**

This falsifies the central argument of section 1 of this report. "Deep supervision must
come from near-geodesic paths" is wrong as a *training* claim. The walk label really is
noisy at depth, but training on paths is worse, because it is off-distribution for the
search. Using verified paths for EVALUATION (`33_eval_path.py`) is unaffected -- tq1 is a
statement about training distribution, not about what makes a valid test set.

**Sorted profile -- REJECTED as `tq6`, and the whole family is closed.** From
`TRIANGLE_BOX_REJECTED.md`: *"sorting routes the gradient by the model's OWN current
ranking, so a permutation-invariant target entrenches the existing ordering instead of
correcting it -- harmless where the ordering is already right, damaging where it is
near-random (i.e. at depth)."*

**That is precisely the failure I measured and attributed to the architecture.** My ties
grew 1.1 -> 10.2 with depth while the two ResMLP baselines stayed at 2.4-5.0; a term that
entrenches a near-random ordering at depth is a sufficient explanation, and it was running
at weight 0.5/0.5.

Worse, I ran their own pre-flight check and it passed: best-constant MSE 0.190 > 0, which I
reported as "SEPARATING". **tq6 passed the identical check at 0.247 and still lost.** The
pre-flight is necessary, not sufficient -- my "separating" verdict was the same false
positive.

### Hyperparameters I had wrong

| | mine | theirs |
|---|---|---|
| `pivot_tilt` | 0.0 | **0.5** (biases pivots deeper) |
| `weight_decay` | 0.01 | **3.0e-3** |
| path rows / batch | 15.4 pct | **0** |
| profile weight | 0.5 / 0.5 | **0 / 0** |

### The budget question, answered from their numbers

Their transformer: **159 s/epoch on an A100**, 1500 epochs x 1024 steps x (512 x 14 sym +
512 anchor) = **11.8B rows in 2.76 days**. And the horizon is not conservative -- **the
training axis is CLOSED**: beam total by epoch was 426 / 428 / 431 / **424** / 426 / 427 /
425 / 428 / 428 at ep1200..2000, so 800 epochs past 1500 bought nothing.

Relaunched as `c_qd1_dep` on `configs/qd1_deployed_recipe.yaml`: 4300 epochs x 256 steps x
(512 x 20 sym + 512 anchor) = **11.8B rows in 2.54 days** -- matched. One deviation, flagged
for ablation: 48 mirror frames at width 20 instead of 24 rotations at width 14. That is the
same objective on more columns rather than a new constraint shape, which is what separates
it from the rejected family.

### The result this reframes: TWO REGIMES

`RESMLP_VS_TRANSFORMER.md` section 6 is the honest verdict on tetraminx, and it is not
"the transformer wins":

| regime | winner | evidence |
|---|---|---|
| width still buyable | **ResMLP** | ResMLP 4M x 4 frames = 179 vs transformer 1M = 189 (7 pids), at 1/3 the wall |
| width capped / saturated | **transformer** | equal width (1M, 1 frame): 441 vs 468 -- transformer by 27 |

So even on tetraminx the transformer loses at equal wall clock while width remains
purchasable. It wins per node, and on tight pids where width has saturated -- which is where
the remaining deficit to the leader lives.

Two consequences for cube444. (1) My step-8 iso-width loss is the meaningful one: theirs
*won* iso-width, mine lost, which is a clean statement that my model was undertrained and
mis-regularised rather than that the architecture fails. (2) Skipping the iso-wall round was
right, but for a better reason than I gave -- iso-wall is the regime the ResMLP is already
known to win, so it was never the discriminating test.

Their cost figure is ~17x beam ratio against a 33x FLOP ratio, with an explicit warning that
a beam ratio ABOVE the FLOP ratio indicates a thermal/measurement error. My A100 numbers
pass that check: 50.6x model-only against a ~56x FLOP ratio, and 9.8x end-to-end.
