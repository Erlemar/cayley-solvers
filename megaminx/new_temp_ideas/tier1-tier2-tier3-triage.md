# Megaminx — Triaged Priorities (T1 / T2 / T3)

Triage of the ~70 ideas in `i-want-you-to-gentle-crayon.md` (data axis),
`capacity-axis-brainstorm.md` (capacity axis), and `paradigm-axis-brainstorm.md`
(paradigm axis). Anchored to:

- **Goal**: 88,195 → <70,000 (−20% moves). This is a heuristic-quality problem;
  speed wins only count to the extent they fund larger beams.
- **Acceptance gate**: ≥+3 strat-5 solves AND mean model_avg ≤ 0.95× m05's 89.4,
  no bucket regresses by more than 1.
- **Cluster summary**: m05 / m17 / m26 / m27 all sit at 50–51 strat-5 solves /
  mean 89–92. Capacity-scaling at the same Bellman recipe is exhausted; the
  question is whether the Bellman *recipe* is also exhausted, or only its naive
  form.

Ordering inside each tier is by **expected moves saved per engineering hour**.

---

## Already in `to_do_shortlist.md` (do not re-propose)

| Idea (in temp files) | Already queued as |
|---|---|
| T1.A "Option B" BFS-d6 mixin into Bellman | Item 5 (Exact BFS-d6 anchoring inside Bellman target) |
| T2.A m23 retrain against new teacher | Conditional on item 5 passing |
| T2.B m22 K-step lookahead | Item 8 |
| T2.C / C2 m24 V+π policy multi-task | Item 9 |
| T1.D NISS retest on m05+m23 stack | Item 11 |
| T0.1 TRT bf16 deploy | Items 13–15 |
| T1 cayleypy `nbt`/`bfs` walks A/B | Item 10 |

The Bellman bias-mitigation cluster (Double Bellman, frontier replay,
target-shrinkage tracking, n_back sweep) is items 3–7 of the shortlist and is
not duplicated here.

---

## TIER 1 — Must add. Highest leverage. Start these first.

### T1.1 — Speedcubing macro library + macro-augmented beam
**Source**: paradigm-axis P1.4 + P2.2.
**What**: Scrape ~100 named Megaminx algorithms from speedsolving.com /
cubingdb.com / jPerm. Each is 5–15 moves with a precomputed permutation. Extend
the beam's action space from 24 generators to 24 + ~100 meta-actions.
**Why this priority**: This is the **single highest-EV unclaimed item across
all three files**. Two pieces of hard evidence:
1. **Rokicki (the literal God-number cuber) is below us at 93,606.** Classical
   brute search has saturated on Megaminx's structure.
2. **WCA Megaminx world records solve in ~250 moves with macros**, vs our
   88,195 / 1001 ≈ 88 per puzzle. The path-length cliff is precisely what
   speedcubers exploit and our pipeline does not.
Risk is provably one-sided: macros are mathematical truths, beam can ignore
any unhelpful one, can never lose by adding them.
**Effort**: 3–5 days (scrape + validate + wire into `KhoruzhiiSolver`).
**Expected**: −10K to −20K total moves. Plausibly the lever that crosses 70K.

### T1.2 — Test-Time Training (TTT) per puzzle
**Source**: capacity-axis A3 (also overlaps D3 LoRA TTT).
**What**: Per puzzle, clone m05, run 50–200 self-supervised Bellman gradient
steps on random walks emanating from the test scrambled state, then beam-solve
with the adapted clone. Akyurek 2024 (TTT for ARC) is recent strong evidence.
**Why this priority**: Targets the exact failure mode our cluster summary
points at — m05's *global* Bellman loss 0.094 averages over the whole graph;
*near a specific scramble* the local error is much higher and that's where
beam search wastes wall. Strat-5 understates this; gate is full 1001
submission. Cheap, novel for this project, no model retraining.
**Effort**: ~3h code + ~50 min compute on full 1001 (3s/puzzle × 1001).
**Expected**: −2K to −5K (concentrated on the hard tail).

### T1.3 — SWA average of m05 checkpoints 400–499
**Source**: capacity-axis A5.
**What**: Average the state_dicts of 5 m05 checkpoints (400, 425, 450, 475,
499). Izmailov 2018 SWA evidence.
**Why this priority**: 30 minutes total wall, uses existing checkpoints, zero
training cost. If it doesn't help, throw away. Zero downside.
**Effort**: 30 min.
**Expected**: small (−500 moves if it works), but it's *free*. Always do free shots first.

### T1.4 — Aggressive beam-stack auto-rescue (top decile of long-path pids)
**Source**: data-axis T1.C.
**What**: Auto-identify the 50–100 longest-path pids in our current best
submission, run beam-stack rescue (`12_beam_stack_rescue.py`) on all of them
with a wider runner-up beam (2B). Splice winners back into the submission.
**Why this priority**: Current best (88,195) rescued only 2 pids and saved 700+
moves between them. The mechanism is proven; we just haven't applied it
broadly. Cannot regress (only swaps shorter paths in).
**Effort**: ~1 day code (~50-line wrapper around existing rescue) + ~1.5h compute.
**Expected**: −500 to −1500 moves. Near-guaranteed positive.

### T1.5 — Local search on completed paths (Simulated Annealing / LAHC)
**Source**: paradigm-axis P4.1.
**What**: SA on the move sequence with neighbor ops {delete inverse pair,
insert random move, swap adjacent}. Verify each candidate; only accept
shorter. Standard for TSP.
**Why this priority**: Different from BFS-d6 windows — explores non-monotonic
local moves. Can never make a path longer (verifier guards). Bounded
compute. Independent of any model retraining.
**Effort**: ~3 days code (~150 lines). 5 sec/puzzle × 1001 ≈ 90 min compute.
**Expected**: −1K to −3K total moves (TSP analog gets 3-7%; expect 1-3% here).

---

## TIER 2 — Strong. Run once T1 is in motion or has stalled.

### T2.1 — Insertion-Finder port (FMC-style)
**Source**: paradigm-axis P2.1 (pairs naturally with T1.1 macro library).
**What**: For each gap in a solved path, try inserting each commutator/macro,
re-canonicalize via existing `full_post_process`, accept only if total
length strictly decreases.
**Why this priority**: FMC competition literature reports −5 to −15 moves per
Rubik's solution; proportional Megaminx scaling → −3K to −10K. Pairs cleanly
with T1.1 — macros become insertion candidates. Free post-processing,
won't regress. Worth doing AFTER T1.1 (so the macro library is built).
**Effort**: 3–4 days port.
**Expected**: −3K to −10K total moves (heavy on hard tail).

### T2.2 — Tail re-solve post-processing
**Source**: data-axis T2.D.
**What**: For each completed path, re-solve the last K=20-30 moves with a
narrow beam from the intermediate state. Replace if shorter. Wire into
`full_post_process`.
**Why this priority**: Cheap, additive, won't regress (verifies final state).
Independent of model retraining. Sits on top of every other lever.
**Effort**: ~3h port + 1h validation.
**Expected**: −50 to −200 moves. Small but truly free.

### T2.3 — Listwise rank loss training (m30_listwise from m05 body)
**Source**: capacity-axis A1.
**What**: Replace MSE on V with ListNet-style soft ranking loss over the 24
children of each parent. Train m05's body for ~500 ep with the new loss.
**Why this priority**: Beam at every step uses ONLY relative ordering of
children; MSE optimizes absolute distance. The MSE-64 plateau is largely
*scale* noise — ranking loss is robust to scale and only needs *order*. Clean
theoretical fix targeting the exact objective↔inference mismatch.
**Effort**: ~6h training + 45 min strat-5. ~2× wall vs MSE Bellman per epoch.
**Expected**: plausibly +3 to +5 strat-5 solves; could be the lever that
breaks the cluster ceiling.

### T2.4 — Inversion-pair check on test set
**Source**: paradigm-axis P4.4 / P5.1.
**What**: Hash all 1001 test states + their state-inverses; report any matches.
**Why this priority**: 15 minutes. Almost certainly returns zero matches
(random walks rarely produce exact inverses), but the answer is free and
defensive. Just run it on day 1.
**Effort**: 15 min.
**Expected**: 0 pairs (90% prior). If non-zero, free pids.

---

## TIER 3 — Second-wave if T1/T2 stall, or research bets

### T3.1 — Transformer m29 retry on 4090 (resolves m18 rejection)
**Source**: capacity-axis B1.
**What**: 6-layer GPT-style transformer, d_model=384, ~6.5M params. ~12.5h
training on 4090 (90s/ep × 500 ep).
**Why this priority**: m18 was rejected on Kaggle wall-time only (HANDOFF: "Not
infeasible in principle, just slower-per-wall"). On 4090 it fits. Different
inductive bias = the kind of thing that breaks the cluster ceiling if anything
does. **Run only ONE arch lottery (this one), not multiple.**
**Effort**: 1d code + 12.5h training.
**Expected**: uncertain. ±5 strat-5 solves either way; high variance bet.

### T3.2 — Bellman-aux during RW pretrain (C1)
**Source**: capacity-axis C1.
**What**: Add Bellman consistency as secondary loss alongside walk-depth MSE
during the RW pretrain phase (epoch 1, not just last phase).
**Why this priority**: Could break the RW MSE-64 plateau by reshaping the
pretrain landscape. +40% per epoch. Reasonable theoretical motivation —
m05's two-phase recipe artificially separates signals that could co-train.
**Effort**: ~12h training.
**Expected**: small (RW pretrain is upstream; gains only flow through if the
Bellman fine-tune also breaks the plateau).

### T3.3 — DQN supervised on BFS-d6 oracle (P3.4)
**Source**: paradigm-axis P3.4.
**What**: Train Q-head directly on EXACT optimal Q*(s,a) = 1 + true_d(child)
for all 19.4M BFS-d6 states. No Bellman bootstrap, no noise.
**Why this priority**: Cleanest possible labels. Limited to d≤6 coverage but
that's exactly the boundary that matters for endgame. Pairs naturally with
shortlist item 5 (BFS-d6 anchoring).
**Effort**: 3–5 days.
**Expected**: 3–5%; could replace m23 student or augment qshort.

### T3.4 — Adversarial state mining (P4.2)
**Source**: paradigm-axis P4.2.
**What**: Compute |V_m05(s) − true_d(s)| for all 19.4M BFS-d6 states. Take
top 100K by error. Oversample 10× into next training run.
**Why this priority**: Standard hard-mining. Cheap (1 day). Targets the model's
actual blind spots (BFS-d6 is ground truth; m05's predictions there can be
*verified*).
**Effort**: 1 day.
**Expected**: 2–3% on subsequent retrains.

### T3.5 — V + per-piece distance auxiliary (C3)
**Source**: capacity-axis C3.
**What**: Auxiliary head outputs Hamming distance per piece. Loss:
`MSE_V + 0.05 · MSE_per_piece`. Body shared.
**Why this priority**: Cheap auxiliary signal (+5% per epoch). Low risk
regularizer. Could improve generalization.
**Effort**: ~12h training.
**Expected**: small (+1 to +2 solves if it helps; nothing if it doesn't).

### T3.6 — Multi-seed beam ensemble (T1.B)
**Source**: data-axis T1.B (also overlaps D4).
**What**: 3 different RNG seeds, take min path per pid.
**Why this priority**: Verify FIRST that `KhoruzhiiSolver` actually consumes
`random_seed` for tie-breaks (the data plan flags this risk). If RNG isn't
wired, this becomes a research item with no value. Deploy AFTER TRT funds the
3× wall (currently TRT bf16 is in flight per HANDOFF §13).
**Effort**: ~5 lines code (if RNG is wired); 3× wall on solve.
**Expected**: −2% to −4% paths from independent tie-break sampling.

---

## Explicitly NOT doing (skip with reasons)

These ideas appeared in the temp files but **don't run them**:

| Idea | Source | Reason |
|---|---|---|
| Per-bucket difficulty specialists | A4 | 10× training cost (~50h) at the same plateaued signal. Bucket boundaries are arbitrary k_max ranges. |
| GNN over piece adjacency | B2 | Speculative arch. Pick T3.1 transformer; running multiple arch lotteries burns wall. |
| Mamba / SSM | B3 | Same as B2 — pick one arch lottery. |
| Permutation-equivariant DeepSets | B4 | Position semantics matter; provably-permutation-invariant nets discard signal. (Author's own skip note.) |
| MCTS within beam nodes | D1 | Complex, c-tuning sensitive. At beam 131k+ the wall budget shifts compute around with uncertain quality outcome. |
| Looped V refinement w/ MC dropout | D2 | Requires dropout / noise injection; m05 has minimal dropout. Architectural change for unclear gain. |
| Adaptive beam width | D5 | Already partially explored. Prior benchmark showed −13% paths regression. |
| SAM / Lion / Lookahead | E2 | Muon hit the same plateau as AdamW. No mechanism to expect SAM/Lion to escape. |
| k_max curriculum | E3 | Already rejected (IHES E7). |
| V + cycle-decomposition aux | C4 | Speculative correlation; cheaper alternatives in T3 already cover the aux-signal angle. |
| Thistlethwaite-style decomposition | P1.1 | 1–2 week math project. Author honestly notes "useful chain may not exist" for icosahedral group. The 1-week probe is OK if we have spare bandwidth, but commitment is high-risk. |
| Stream-of-Search transformer | P1.2 | Research-grade; high uncertainty at our scale; ~2 weeks. |
| PPO RL fine-tune of m05 | P1.3 | Sparse reward, famously slow. Proxy↔objective gap may be small here. ~1 week. |
| Coset PDBs / B5 additive | P1.5 / B5 | Admissible heuristic only matters for A*/IDA*; at beam 131k+ A* unlikely to beat. Both gated on a multi-week IDA* implementation we haven't committed to. |
| twsearch port | P2.3 | Gated on whether the competition data ships a `.tws` file. Day-1 check; if no, porting is 1–2 weeks. |
| Schreier-Sims short words | P2.4 | Same use case as PDBs; same gating problem. |
| Commutator library standalone | P2.5 | Subsumed by T1.1 macros + T2.1 insertion finder. |
| HER goal-conditioned policy | P3.1 | 2 weeks. Lower-confidence than Tier-1/2 levers. |
| GFlowNet for diverse paths | P3.2 | 2–3 weeks. GFlowNet on permutation puzzles is unproven. |
| D3PM discrete diffusion | P3.3 | Author's own skip recommendation in favor of P1.2. |
| MuZero | P3.5 | Heavy machinery for uncertain gain when true dynamics are public. |
| Online learning during inference | P4.3 | Model-drift risk. Needs EMA/safeguard. T1.2 TTT is a cleaner version of this idea. |
| Inverse model I(s,s') | P4.5 | Use case (re-ranking) is poorly defined. |
| Distributional V (C51) | A2 | Subsumed by T2.3 ranking loss for the same target failure mode. |
| LLM as Megaminx strategy synthesizer | P-bold note | Speculative; LLMs unlikely to produce useful Megaminx zero-shot. |
| Foundation model across permutation puzzles | P-bold note | Cross-puzzle transfer is research; payoff unclear. |

---

## Recommended sequence (next 2–3 weeks)

**Day 1 (free wins, ~1h total):**
- T2.4 inversion-pair check (15 min).
- T1.3 SWA average of m05 (30 min).
- Check whether competition data ships a `.tws` file (informs whether P2.3
  twsearch port becomes viable later).

**Days 2–4 (near-guaranteed wins):**
- T1.4 aggressive beam-stack auto-rescue (~1 day code + 1.5h compute).
- T2.2 tail re-solve post-processing (~4h code).

**Days 4–9 (the big bet):**
- T1.1 macro library scrape + integration. The single highest-EV unclaimed
  item; 70K target plausibly turns on this.

**Parallel during macro work (compute-light, can interleave):**
- T1.2 TTT integration (~3h code + ~50 min compute on full 1001).
- T2.3 listwise rank loss training (~6h on 4090).

**Days 9–14:**
- T2.1 insertion-finder port (pairs naturally with T1.1 macros).
- T1.5 local search on completed paths.

**Week 3 (second-wave if needed):**
- Whichever Tier-1 levers didn't deliver → pick T3.1 transformer or T3.2
  Bellman-aux as the next training experiment.
- T3.4 adversarial state mining feeds the next retrain.

---

## Stacking note

These levers are **stackable**, not exclusive. The 70K target plausibly needs
2–3 of them landing in series:

- T1.1 macros change the search space.
- T1.2 TTT improves the heuristic per-puzzle.
- T1.4 + T1.5 + T2.1 + T2.2 are post-processing layers that compose
  associatively on top of any model.
- T2.3 / shortlist items 3–7 attack the heuristic-recipe ceiling.

Pick winners from each layer; stack at the end.

---

## Honest meta-observations

1. **The three temp files are siloed.** P1.4 macros (paradigm) is plausibly
   higher-EV than every data-axis item, but the data plan never references it.
   Combining the plans rather than committing to one axis is the right read.

2. **"Different inductive bias" is one bet, not three.** B1 transformer / B2
   GNN / B3 Mamba / P1.2 Stream-of-Search all share the same
   "might-break-plateau" justification. Run *one* (T3.1).

3. **The 70K target is gated by macros OR a real heuristic break.** Bellman
   recipe variations (shortlist items 3–7) gain you maybe −5K if any work.
   To cross 70K, T1.1 macros is the highest-leverage unclaimed bet, and T1.2
   TTT is the second.
