# ResMLP vs PieceTransformer as a tetraminx beam scorer

Measured 2026-07-29..31. Both models are **all-neighbours Q heads** trained on the
identical sparse-Q objective, identical sampler (`k_max` 40, `pivot_tilt` 0.5, 24-frame
symmetry expansion, BFS-d≤5 exact anchors), identical data. The only variable is the
architecture.

> **Read the budgets before the numbers.** The ResMLPs are at **epoch 1500**; the
> transformer is at **epoch 500** and still training. Every transformer number below is
> therefore a lower bound on its converged quality, and every comparison is in its
> favour on that axis. The cost numbers are not affected.

| | ResMLP (cell 1) | ResMLP + AZ (cell 2) | PieceTransformer (cell 3) |
|---|---|---|---|
| params | 5,008,280 | 5,008,793 | **3,444,504** |
| shape | 1408 → 2048 → 512, 2 res blocks | same + scalar value head | 50 piece tokens + CLS, d=256, 4 layers, 8 heads, ff 1024 |
| epochs | 1500 | 1500 | 500 |

---

## 1. Decision-quality probe

`52_eval_q.py`, 16,384 random-walk-middle pivots, k ≤ 40, tilt 0.5. `top1` = does the
move that undoes the last scramble step score better than all other 22; `gap` = mean
score(next) − score(undo), which the label says should be 2 at every depth.

| metric | ResMLP | ResMLP+AZ | **Transformer** |
|---|---|---|---|
| pair | — | 0.8816 | **0.8977** |
| **top1** | 0.5095 | 0.4965 | **0.5350** |
| gap | 1.392 | 1.359 | **1.432** |

By pivot depth (`top1` / `gap`):

| band | ResMLP | ResMLP+AZ | **Transformer** |
|---|---|---|---|
| 1–4 | **0.780** / 1.877 | 0.780 / 1.876 | 0.776 / 1.853 |
| 10–14 | 0.537 / 1.555 | 0.510 / 1.501 | **0.576 / 1.599** |
| 20–24 | 0.249 / 0.857 | 0.236 / 0.798 | **0.307 / 0.978** |
| 25–29 | — | 0.151 / 0.481 | **0.203 / 0.651** |
| 30–40 | 0.105 / 0.302 | 0.097 / 0.240 | **0.128 / 0.354** |

**The advantage is entirely at depth.** At 1–4 the ResMLP is marginally ahead. From
depth 10 the transformer pulls away, and at 20–29 it is ~25% better on top1 and ~35%
better on gap. That is the band where 30-move solves actually spend their time.

## 2. Beam quality — the binding gate

15 stratified pids, Q@1M, 1 frame, exact endgame, `--no-merge`, all paths replayed
against `test.csv` and asserted to solve.

| | total | /pid | vs merged-best | ties best | wall (cool card) | wall (hot card) |
|---|---|---|---|---|---|---|
| merged-best (community ∪ ours) | 423 | 28.20 | — | — | — | — |
| ResMLP | 468 | 31.20 | +3.00 | 2 | 143 s | ~370 s |
| ResMLP + AZ | 461 | 30.73 | +2.53 | 3 | 140 s | ~370 s |
| **Transformer** | **441** | **29.40** | **+1.20** | **6** | (not measured cold) | **6,305 s** |

Two wall columns because this machine throttles to half clock under sustained load
(§3). The ResMLP rows were measured cold; the transformer row hot. **Compare only
within a column** -- the cross-column ratio is the 47x error described in §3.
The equal-wall test in §6 is unaffected: both of its runs were measured hot.

Transformer vs ResMLP head-to-head: **better on 9 pids, tied on 4, worse on 2**;
0 invalid paths. It more than halves our deficit against the community bar.

**The probe and the beam agree here.** Worth stating explicitly because they disagreed
three times earlier in this project (path-label model, tq0 plateau, AZ head) — the probe
is a proxy and is not trusted on its own.

## 3. Cost

### Inference

| | MACs / state | measured beam ratio |
|---|---|---|
| ResMLP | ~5.0 M | 1× |
| Transformer | ~166 M | **~17×** |

Per pid at Q@1M, both measured on a heat-soaked card: **~25 s vs ~420 s**.

> **Thermal-state warning — this bit me.** An earlier draft reported **47×**, from
> comparing a ResMLP run taken on a COOL card (9.5 s/pid) against a transformer run
> taken after 105 min of continuous load. This laptop 4090 throttles to **1515 MHz
> against a 3105 MHz max** (49%, 83 °C, throttle reason 0x04 = SW power cap), so the
> same job measures 9.5 s/pid cold and 24-27 s/pid hot — a 2.5x swing with nothing
> else changed. **Wall-clock comparisons on this machine are only valid within one
> thermal state.** The tell that something was wrong: 47x EXCEEDED the 33x FLOP ratio,
> which is backwards — the ResMLP's beam carries non-model overhead (child gather,
> hashing, top-k) that the transformer pays too, so the beam ratio must come in BELOW
> the FLOP ratio. 17x does; 47x did not.

**Attention is only 3.2% of the transformer's MACs** (26,112 of 812,544 per token per
layer). The cost is the per-token FF (64.5%) and the QKV/output projections (32.3%).
Consequence: cheaper-attention research (linear/sparse attention, and the Kimi K3
mechanisms specifically) cannot help this model — the token count and depth are the
cost, not the O(T²) term, at T=51.

It is also unambiguously the bottleneck: at B=65,536 the ResMLP beam step is ~26 ms of
which ~22 ms is the model; at ~17× model cost the transformer's step is still ~97% model.
Gather, hash and top-k are noise.

Memory differs too: the transformer needs `--chunk-size 8192` where the ResMLP is fine
at 32768 (at 32768 its per-chunk activations are ~855 MB per tensor).

### Training

| | s/epoch (A100) |
|---|---|
| ResMLP | 12.6 |
| Transformer, as first written | 470 |
| Transformer, after three fixes | **159** |

The 2.9× came from profiling, not guessing — see §5.

---

## 4. Why the quality differs — hypothesis, and a falsified one

**The shape of the evidence:** the transformer wins *only at depth*, and wins with
**31% fewer parameters**. So it is not capacity, and it is not a general modelling
advantage — it is specific to states far from solved.

**Working hypothesis.** Distance on a permutation puzzle is governed by the *relational*
structure between pieces, not by per-slot facts. The transformer's tokens are physical
pieces (50 of them, derived and self-verified as a block system by
`50_derive_piece_layout.py`), so attention computes piece↔piece interactions
structurally. The ResMLP sees a flat 88-slot embedding and must learn any relation from
scratch. Near the goal few pieces are displaced and per-slot information suffices — which
is exactly where the ResMLP holds its own.

**A cheap version of that hypothesis is FALSIFIED.** If the edge were coarse cycle
structure, zero-parameter invariants should rank children well. They do not:

| heuristic | top1 overall | 1–9 | 10–19 | 20–29 | 30–40 |
|---|---|---|---|---|---|
| uniform baseline | 0.0417 | | | | |
| misplaced pieces | 0.1785 | 0.324 | 0.062 | 0.052 | **0.036** |
| misplaced − cycle count | 0.1792 | 0.326 | 0.058 | 0.059 | **0.027** |
| *ResMLP learned Q* | *0.5095* | | | | |
| *Transformer learned Q* | *0.5350* | | | | |

Analytic cycle structure works only near the goal and falls **to or below chance past
depth 10** — the mirror image of where the transformer's advantage lives. By depth ~30
essentially every piece is displaced and the cycle count saturates, leaving these
invariants no resolution. So whatever the transformer has learned at depth is *not* a
low-order function of the permutation, and hand-built coarse features will not transfer
it. This closes off the cheapest proposed shortcut.

## 5. Why the inference time differs — and what was avoidable

Structural, not incidental: the transformer carries a 256-dim vector **per token across
51 tokens** through 4 layers; the ResMLP carries one 512-dim vector. That is the 33×.

Three implementation fixes recovered 2.9× of what was on top of it. All verified
numerically exact against a saved reference (max abs diff 6.6e-7 / 1.3e-6, argmin
agreement 1.000):

1. **Folded input stage** — push `piece_projection` into the value table so a token is
   `Σⱼ table[j, vⱼ]·maskⱼ + bias`. Removes the `(B, P, K, D)` intermediate (2.3 GB at
   training batch, ~10 GB at TPU batch). Not an optimisation — required to run at all.
2. **Fused-QKV SDPA** replacing `nn.MultiheadAttention` (`_SelfAttn`, identical parameter
   names so checkpoints and the JAX loader keep working): 484 → 450 ms. Small, and the
   3.2% attention share explains why.
3. **One-hot matmul input stage**: 450 → **162 ms**, the real win. A gather's backward is
   a scatter-add, and 384k gradients per slot contended atomically for an 88-row table.
   Freezing the input stage alone took the step 472 → 158 ms — the embedding gradient was
   **67% of the entire training step**. Recast as `one_hot(vals) @ table`, both directions
   become dense GEMMs.

**Diagnosis method worth reusing.** A GEMM roofline at our *own* matrix shapes gave
157–202 TFLOP/s on an A100 against ~17 achieved, proving the matmuls were never the
limit; then freezing parameter groups localised the cost in a single measurement.
Guessing had already produced two wrong diagnoses (a `torch.compile` hang that was
actually WDDM paging at the 16 GB wall, and an attention-materialisation theory that
was worth only 7%).

## 6. What this means: TWO REGIMES, not one verdict

The transformer is a **substantially better scorer per node and a much worse one per
second**. The obvious test is equal-wall-clock — ResMLP at 4M×4 frames against
transformer at 1M — and on the first 7 pids the ResMLP won decisively (179 vs 189,
and 179 beat even the merged-best 181), using only **a third** of the transformer's
wall clock (~140 s/pid vs ~420 s).

**But that test embeds an assumption: that width is purchasable.** The ResMLP only
converts its 47× speed into quality while there is width left to buy. Two things stop
that conversion:

* Note the premium is **~17×, not the 47× first reported** (see the thermal warning in
  §3). A smaller premium means the ResMLP can buy less extra width per second, so the
  crossover between the two regimes arrives SOONER than the original figure implied.
* **Memory ceiling.** Beam width is capped by memory, and the beam's state arrays are
  the same size for either model — the transformer costs TIME, not memory. So both
  models face the SAME width ceiling, and at that ceiling the ResMLP's speed advantage
  buys it nothing.
* **Width saturation**, which can arrive before the memory cap and arrives unevenly.
  Our own ablation found width "worthless on tight pids" while still paying −53 moves
  at 1M→4M on loose ones.

So there are two regimes, and the evidence for both is already in this document:

| regime | winner | evidence |
|---|---|---|
| width still buyable | ResMLP | 4M×4 = 179 vs transformer 1M = 189, at 1/3 the wall |
| **width capped / saturated** | **transformer** | **at EQUAL width (1M, 1 frame): 441 vs 468 — transformer by 27 moves** |

The equal-width row is the width-capped answer and it was measured before the
equal-wall test. Reporting only the equal-wall result would overstate the ResMLP's case.

**Practical consequence.** Tight pids — where width has already saturated — are exactly
the pids standing between us and the leader. That predicts something specific and
testable: *on tight pids the transformer should win even at equal wall clock.*

**The measurement that actually settles it** is a quality-vs-compute CURVE, not a single
point. Run the ResMLP at 1M / 4M / 16M on the same 15 pids and find where its curve
flattens, then compare the transformer's 441 against the ResMLP's SATURATED value rather
than its 4M value. If the ResMLP flattens above 441 the transformer is genuinely better
wherever we are compute-limited; if below, the transformer wins at the ceiling and
belongs there. The ceiling itself lives on the TPU (v6e-8 has demonstrated 32M), so that
sweep should end there.

## 7. What to do next

Ranked follow-ups:

0. **Width-saturation sweep (NEW, now the top priority).** ResMLP at 1M / 4M / 16M on the
   15-pid set, then the same on the TPU toward 32M. This locates the crossover between
   the two regimes above and tells us which model to deploy WHERE, rather than which
   model is "better" in the abstract. Everything below is contingent on it.
1. **Distillation** — train the ResMLP against the transformer's 24 Q-values. Value is
   contingent on regime: in the width-buyable regime we would be distilling a teacher
   that loses to the student's own wider-beam configuration, which is close to
   pointless. It matters in the width-capped regime, where per-node quality is the only
   lever. §4 already showed the gap is not a hand-buildable feature, so distillation is
   also the clean test of *representability*: if the loss floors well above zero at
   depth, attention is doing something a flat MLP structurally cannot.
2. **Transformer size ablations** — `d_model` 256→128 is ~4× cheaper (quadratic in FF and
   projections), 4→2 layers is 2×, and 16 of the 50 tokens are singleton centres that may
   carry little relational signal. Finds the quality/cost frontier.
3. **Hybrid re-rank** — `khoruzhii_search.py` already has `cutoff_model` /
   `cutoff_pool_mult`: score the whole beam with the ResMLP, re-rank only an
   overgenerated near-cutoff band with the transformer. At a 10% band that is ~1.7×
   instead of ~17×, spending the expensive model exactly where the decision is marginal.
   This is the fallback that works even if (1) shows the gap is irreducible.
