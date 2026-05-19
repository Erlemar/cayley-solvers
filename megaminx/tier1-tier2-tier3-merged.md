# Megaminx — Merged Priority Triage (T1 / T2 / T3)

Single source-of-truth for the priority triage of all brainstorm files. Built by
merging:

1. `new_temp_ideas/tier1-tier2-tier3-triage.md` — triage of the three
   `new_temp_ideas/*.md` brainstorms (data axis, capacity axis, paradigm axis).
2. `new_big_ideas/*.md` — six independent brainstorm files (gemini_ideas,
   codex_ideas, chatgpt_ideas, deep-research-report-codex,
   RESEARCH_alternatives_claude, claude_research) with literature citations.

Anchored to:

- **Goal**: 88,195 → <70,000 (−20% moves). Heuristic-quality problem; speed
  wins only count when they fund larger beams.
- **Acceptance gate**: ≥+3 strat-5 solves AND mean model_avg ≤ 0.95× m05's
  89.4, no bucket regresses by more than 1.
- **Cluster ceiling**: m05 / m17 / m26 / m27 all sit at 50–51 strat-5 solves /
  mean 89–92. The naive Bellman recipe is exhausted. Items in this triage
  contest the recipe ceiling, the search-space ceiling, or both.

This file SUPERSEDES `new_temp_ideas/tier1-tier2-tier3-triage.md`. The older
file is left in place for provenance.

---

## Already in `to_do_shortlist.md` (do not re-propose)

| Idea | Source files | Already queued as |
|---|---|---|
| BFS-d6 mixin into Bellman target | T1.A (temp), all big_ideas | Item 5 (Exact BFS-d6 anchoring) |
| Double Bellman (Hasselt 2010) | codex §3.1, RESEARCH §3.5 | Item 4 |
| Beam-frontier (failed-state) replay | codex §2.1, chatgpt §2.6, RESEARCH §2.5 | Item 6 |
| n_back sweep | T7 (temp), every big_ideas file | Item 7 |
| n-step / K-step Bellman | RESEARCH §3.1, claude_research §5.2 | Item 8 (m22) |
| V + π policy multi-task | T2.C (temp), DeepCubeA-style | Item 9 (m24) |
| cayleypy `nbt`/`bfs` walks | (already in shortlist) | Item 10 |
| NISS retest on m05+m23 | T1.D (temp) | Item 11 |
| TRT bf16 deploy | T0.1 (temp) | Items 13–15 |
| Symmetry derivation v3 | RESEARCH §1.7, §3.6, §5.8, claude_research §3.4 | Item 16 |
| Per-depth target shrinkage diagnostic | (my addition) | Item 3 |

---

## DAY 1 — free wins (<1 h each, all do)

These are tiny code changes or trivial verifications. Ship them all in one sitting.

1. **SWA average of m05 checkpoints 400–499** (was T1.3 in old triage). 30 min, uses existing checkpoints, Izmailov 2018.
2. **Inversion-pair check on test set** (was T2.4). 15 min. Almost certainly returns 0 pairs but free defensive check.
3. **Add soft-Bellman temperature param to `bellman.py`**. ~5-line change: replace `min_a V_target(s')` with `−T·logsumexp(−V_target(s')/T)`, expose `bellman.softmin_temperature` in config. Anneal T → 0 recovers hard min. Activates on next Bellman retrain.
4. **Add Polyak/EMA target update to `bellman.py`**. ~5-line change: replace "every 10 epochs" hard refresh with `target := τ·model + (1−τ)·target` per step (τ ≈ 0.005). Toggle via config.
5. **Web-verify HKHLR Megaminx 82-move solver claim** before scoping the Minkwitz port (T2.4 below). 15 min.

---

## TIER 1 — must do, highest leverage

Ordered by **expected moves saved per engineering hour**, anchored to evidence.

### T1.1 — Speedcubing macro library + macro-augmented beam
**Source**: paradigm-axis P1.4 + P2.2.
**Why this priority**: still the highest-EV unclaimed lever. Two empirical
anchors: (1) Rokicki — the literal God-number cuber — is *below us* at 93,606,
direct evidence classical brute search has saturated on Megaminx; (2) WCA
Megaminx world records solve in ~250 moves with macros, vs our ~88 average.
Path-length cliff is the mechanism humans exploit and our pipeline doesn't.
Risk is one-sided: macros are mathematical truths, beam ignores any unhelpful
one. Plausibly the lever that crosses 70K alone.
**Effort**: 3–5 days (scrape from speedsolving.com / cubingdb.com / jPerm,
validate, wire as 24+~100 meta-actions in `KhoruzhiiSolver`).
**Expected**: −10K to −20K total moves.

### T1.2 — Multi-agent ensemble (8–30 independently-trained agents) [NEW from big_ideas]
**Source**: `claude_research.md` Tier-1 #1, citing CayleyPy paper directly
(arXiv:2502.13266 + arXiv:2502.18663). Also gemini_ideas "Expert Iteration".
**Why this priority**: **most-cited and most-empirically-grounded lever in any
big_ideas file**. CayleyPy explicitly states this is what took them from
"single-agent good" to "beats every team in Santa 2023" on 4×4×4 and 5×5×5.
10 agents reduced 5×5×5 average length 92.9→92.16 and 4×4×4 49→46.51, scaling
near log-linearly with agent count (their final SOTA needed 69 agents on
5×5×5, 29 on 4×4×4). Each agent is individually mediocre; diversity of
learned representations does the work. **Distinct from the multi-seed beam
ensemble (now T3.6)** — that's one model with different RNG; this is many
independently-trained models.
**Effort**: 8 agents × ~80 min training/agent = ~10 h training. Beam-search
cost is the dominant term and is genuinely expensive — see below.
**Deployment reality**: naive full-1001 ensemble at 8 agents = **~84 h on
4090 (3.5 days continuous) or ~$60 GCP**. Don't ship that. Practical
defaults, in order of preference:
1. **Hard-tail-only ensembling**: run primary agent (m05) full at beam
   131k, then ensemble the other 7 agents only on the top-50 to top-100
   longest-path pids. Cost drops to ~17 h for ~80% of the diversity
   benefit — almost all per-puzzle-min savings come from the hard tail;
   easy pids are already near-optimal.
2. **Smaller per-agent beam** (CayleyPy's actual pattern — they ran many
   fast agents, not few big ones). 8 agents at beam 32k–65k ≈ 40 h total.
   Trades single-agent strength for diversity; the *min* over diverse-but-
   weaker agents can beat one strong agent.
3. **Heuristic-averaging fallback** (`V_ensemble = mean(V_i)` driving one
   beam): one beam wall + N× model-forward. Cheap sanity check that
   agents are actually different, but a different mechanism — gives one
   path per puzzle, payoff ~1% not 2–5%.
Per-puzzle min ensembling stacks naturally with T1.5 auto-rescue (run
ensemble + beam-stack rescue on the same hard-tail pids; take min over both).
**Expected**: −2K to −5K moves (2–5%). Confirmed empirical evidence on
4×4×4 / 5×5×5 cubes; transfer to Megaminx is suggestive but not certain.

### T1.3 — Test-Time Training (TTT) per puzzle
**Source**: capacity-axis A3, Akyurek 2024.
**Why this priority**: targets the cluster's exact failure mode — m05's
*global* Bellman loss 0.094 averages over the whole graph; *near a specific
scramble* local error is much higher and that's where beam wastes wall.
Strat-5 understates this (TTT helps most on the hard tail); gate is full 1001.
**Effort**: ~3 h code + ~50 min compute on full 1001 (3s/puzzle × 1001).
**Expected**: −2K to −5K moves on the hard tail.

### T1.4 — Solver-trace mining / iterative supervised learning [NEW from big_ideas]
**Source**: gemini_ideas "Expert Iteration" + codex_ideas "Solver trajectory
replay" + chatgpt_ideas §3.4 + RESEARCH §2.6 + claude_research Tier-1 #5 +
Arfaee-Zilles-Holte 2010 *Bootstrap Learning of Heuristic Functions* (ICAPS).
**Why this priority**: cited by 5 of 6 big_ideas files independently. For each
successfully-solved puzzle, every state along the path gets a strict
upper-bound label `remaining_path_length`. **Tighter than walk-depth labels by
construction.** Mix into next Bellman cycle at ~10–25%.
**Distinct from shortlist item 6 (beam-frontier replay)** — that captures
*failed-search* states (model overconfidence), this captures *successful
trajectories* (model correctness). Both target different failure modes.
**Effort**: ~2–3 days (instrument `03_solve.py` to dump `(state, dist)` pairs
on every solve; mix into Bellman trainer).
**Expected**: −2K to −6% combined with Bellman.

### T1.5 — Aggressive beam-stack auto-rescue (top decile)
**Source**: data-axis T1.C.
**Why this priority**: the existing rescue saved 700+ moves on just 2 pids
(490, 920). Auto-identify the 50–100 longest-path pids in our best
submission, rescue all with widened (2B) runner-up beam. Cannot regress (only
swaps shorter paths in).
**Effort**: ~1 day code + ~1.5 h compute.
**Expected**: −500 to −1500 moves. Near-guaranteed positive.

### T1.6 — Local search on completed paths (Simulated Annealing / LAHC)
**Source**: paradigm-axis P4.1.
**Why this priority**: TSP-standard; never tried on permutation-puzzle paths
in our pipeline. Different from BFS-d6 windows (explores non-monotonic local
moves). Cannot make a path longer (verifier guards).
**Effort**: ~3 days (~150 lines code) + 5s/puzzle × 1001 ≈ 90 min compute.
**Expected**: −1K to −3K total moves (TSP analog 3–7%; expect 1–3% here).

### T1.7 — Beam-width concentration on the hard tail [NEW from big_ideas]
**Source**: `claude_research.md` Tier-1 #3, citing CayleyPy + EfficientCube
empirics that path length scales linearly in log₂(beam) up to W ≈ 2²⁴.
**Why this priority**: post-TRT bf16 deploy (shortlist 13–15), push beam to
524k–1M on the top-100 longest-path pids only (rescue mode), keeping wall
bounded. Concrete next-step that's already half-implemented in our
`--beams 16384,65536` escalation pattern.
**Effort**: ~half a day (extend escalation logic in `03_solve.py`) + GCP
burst for the rescue solve.
**Expected**: −2% to −5% on the hard tail.

---

## TIER 2 — strong, run after Tier 1 has shipped or stalled

### T2.1 — Insertion-Finder port (FMC-style)
**Source**: paradigm-axis P2.1. Pairs naturally with T1.1 macros (macros
become insertion candidates).
**Why this priority**: FMC literature reports −5 to −15 moves per Rubik's
solution; proportional Megaminx scaling → −3K to −10K. Free post-processing.
**Effort**: 3–4 days port.

### T2.2 — Tail re-solve post-processing
**Source**: data-axis T2.D.
**Why this priority**: cheap (~3 h port + 1 h validation), additive, won't
regress. Sits on top of every other lever.
**Expected**: −50 to −200 moves.

### T2.3 — Listwise rank loss training (m30_listwise from m05 body)
**Source**: capacity-axis A1 + chatgpt_ideas §4.5 + RESEARCH §3.5 +
claude_research §5.3 + DeepCubeA's value+policy joint targets.
**Why this priority**: beam at every step uses ONLY relative ordering of
children; MSE optimizes absolute distance (whose plateau is mostly *scale*
noise). ListNet soft ranking is robust to scale, only needs *order*. Clean
theoretical fix targeting the objective↔inference mismatch.
**Effort**: ~6 h training + 45 min strat-5.

### T2.4 — Schreier-Sims-Minkwitz / HKHLR-style classical fallback [NEW from big_ideas]
**Source**: RESEARCH §5.1 + §2.4 + claude_research §4.6 + §7.1 (HKHLR Hessen
HPC project).
**Why this priority**: **claude_research claims HKHLR Megaminx solver
achieves avg 82 moves at 80 GB RAM — better than our 88.** If this is real,
classical SSM gives a strict floor below our current best by itself, and
`min(beam_path, minkwitz_path)` is an additive win on every puzzle.
There's a working Minkwitz implementation in the Kaggle Santa 2023 community
notebooks to port from. CPU-bound, doesn't compete with GPU.
**Gated on Day-1 web verification of the HKHLR claim.** If it doesn't
verify, this drops to T3.
**Effort**: 3–5 days port + ~1 h to run on all 1001 + post-shorten.
**Expected**: potentially huge floor improvement; low-bound −1K to −5K.

### T2.5 — Distributional V head (C51 / QR-DQN, quantile-based beam scoring) [NEW from big_ideas]
**Source**: RESEARCH §3.2 + claude_research §5.3 + Bellemare-Dabney-Munos
2017 + Dabney et al. 2018.
**Why this priority**: predict 32 quantiles instead of scalar mean; beam
selects on lower quantile (optimistic preference for confident-close states).
**Different mechanism from T2.3 listwise loss** — listwise targets ordering,
distributional captures uncertainty. Beam can use `μ − β·σ` or median for
risk-aware selection. CEA loss variant (arXiv:2509.22626) gives near-admissible
heuristics that beat compressed PDBs on 3×3×3.
**Effort**: ~1 day arch change + retrain.

### T2.6 — Prioritized Bellman backups (PER) [NEW from big_ideas]
**Source**: RESEARCH §3.4 + claude_research Tier-2 + Schaul et al. 2016.
**Why this priority**: sample states with prob ∝ |target − model|. Standard
RL trick we're not using; high-error states get more gradient signal.
Replace uniform shuffle in `_iterate_batches` with weighted sampler.
**Effort**: ~1 day port.

### T2.7 — Inversion-pair check on test set (free)
**Source**: paradigm-axis P4.4. Now in DAY-1 free-wins above; kept in this
list for completeness.

---

## TIER 3 — second-wave / research-grade / conditional

### T3.1 — Transformer m29 retry on 4090
**Source**: capacity-axis B1.
**Why**: m18 was rejected on Kaggle wall-time only. On 4090 (~2.5× P100):
12.5h training fits. Different inductive bias = the kind of thing that
breaks the cluster ceiling if anything does. **Run only ONE arch lottery
(this), not B2 GNN / B3 Mamba.**

### T3.2 — Bellman-aux during RW pretrain (C1)
**Source**: capacity-axis C1.
**Why**: integrate Bellman consistency from epoch 1 alongside walk-depth
MSE. Could break the RW MSE-64 plateau. +40% per epoch. Reasonable theory.

### T3.3 — DQN supervised on BFS-d6 oracle (Q-oracle)
**Source**: paradigm-axis P3.4.
**Why**: train Q-head directly on EXACT optimal Q*(s,a) = 1 + true_d(child)
for all 19.4M BFS-d6 states. No bootstrap, no noise. Pairs with shortlist
item 5 (BFS-d6 anchoring).

### T3.4 — Adversarial state mining
**Source**: paradigm-axis P4.2 + RESEARCH §1.8 + claude_research.
**Why**: |V_m05 − true_d| on all 19.4M BFS-d6 states; oversample top 100K
worst into next training run. Standard hard-mining. ~1 day.

### T3.5 — V + per-piece distance auxiliary (C3)
**Source**: capacity-axis C3.
**Why**: aux head outputs Hamming distance per piece. Loss `MSE_V +
0.05·MSE_per_piece`. Body shared. +5% per epoch. Low risk regularizer.

### T3.6 — Multi-seed beam ensemble (single model, different RNG) [downgraded]
**Source**: data-axis T1.B + claude_research §6.4.
**Why downgraded**: T1.2 multi-AGENT ensemble (different trained models) is
strictly stronger and better-evidenced. T3.6 is the cheaper variant —
useful only if multi-agent ensemble proves too expensive AND
`KhoruzhiiSolver` actually consumes `random_seed` for tie-breaks (verify
first).

### T3.7 — BFS-d7 partial shell [NEW from big_ideas]
**Source**: RESEARCH §4.1 + claude_research §4.1.
**Why**: extend our d=6 shell (~19.4M states) to d=7 (~250M–500M states).
~30 GB on disk with bit-packed encoding. Anchors training and extends MITM
zone. Tight on local 4090 (16 GB), feasible with GCP burst on a 32 GB+
machine.
**Effort**: 1–2 days BFS run + integration.

### T3.8 — EfficientCube-style policy head as ensemble member [NEW from big_ideas]
**Source**: claude_research §4.3 + Takano 2023.
**Why**: predict inverse of last move via cross-entropy (24-way). Different
inductive bias than V-regression → adds ensemble diversity to T1.2. Pair
with multi-agent ensemble: 8 V-agents + 4 π-agents.
**Effort**: ~6 h training + ensemble integration.

### T3.9 — Generalized non-backtracking (CayleyPy formulation) [NEW from big_ideas]
**Source**: claude_research Tier-1 #2 + arXiv:2502.18663.
**Why**: refinement of shortlist item 7. CayleyPy uses (1) n_back=32, (2)
ban states across multiple trajectories simultaneously, (3) ban not only
visited states but their 1-neighbors. They report this lets a single-epoch
1-min-trained MLP solve S₂₀ LRX optimally where ordinary RW fails entirely.
For us: extend item 7's n_back sweep with cross-trajectory + neighbor banning.
**Effort**: ~1 day code + retrain in next Bellman cycle.

### T3.10 — ReduceFactor DAG shortening [from IDEAS.md E6, skeptical]
**Source**: IDEAS.md E6 + IHES synthesis N4.
**Why**: generalization of our greedy BFS-d6 window post-processing. Build
solution DAG with edges weighted by BFS-d6 shortcut lengths; Dijkstra-over-
windows finds globally-optimal combined shortcuts rather than first-improvement
greedy. Distinct from T2.1 insertion-finder (inserts macros) and T1.6
SA/LAHC (perturbs locally).
**Skeptical priority**: HANDOFF.md backlog flagged this as "high code cost,
marginal gains beyond BFS-d6 windows on 25K-floor submissions." We're at 88K
not 25K so the gain *might* be larger now, but T2.1 + T1.6 cover similar
post-processing territory at lower implementation cost. Run only if T2.1 +
T1.6 land but leave clear move-savings on the table.
**Effort**: 1–2 days code.

### T3.11 — Multi-round pseudo-labeling (BirdCLEF self-distillation) [from IDEAS.md P1, skeptical]
**Source**: IDEAS.md P1 + BirdCLEF 2025 1st-place writeup.
**Why**: train round-1 model → predict on RW states → sharpen predictions
via γ-power transform → use as round-2 targets → repeat 3–4× with
StochasticDepth. BirdCLEF 2025 1st place got +0.058 LB this way.
**Skeptical priority**: BirdCLEF is audio classification (categorical,
softmax outputs); we're regression-to-distance. γ-power sharpening doesn't
have a clean analog for regression — you'd be power-scaling residuals, not
probabilities. And we already have Bellman, which is a more principled
"self-target with refinement" loop (with the `1 + min_a V` bootstrap step
that pseudo-labeling lacks). Pseudo-labeling without bootstrap is strictly
weaker than Bellman as a refinement signal.
**Run only if**: shortlist items 3–7 (Bellman bias-mitigation cluster) all
land their gains and we still need a heuristic improvement. ~3–5 h per
round on 4090.
**Expected**: 1–2% at best.

---

## Skip list — explicitly NOT doing

These appeared across multiple files but **don't run them**:

| Idea | Source(s) | Reason |
|---|---|---|
| Per-bucket difficulty specialists | A4 | 10× training cost at the same plateaued signal. Bucket boundaries arbitrary. |
| GNN over piece adjacency / Mamba SSM / DeepSets | B2/B3/B4 | Speculative arch lottery. Pick T3.1 transformer; running multiple burns wall. |
| MCTS / AlphaZero on Megaminx | RESEARCH §5.5 + claude_research §6.2 | DeepCube tried it, abandoned for weighted A\*; Konen 2023 only partial 3×3×3. Beam ≈ MCTS at depth-1 with infinite parallelism. |
| Looped V refinement w/ MC dropout | D2 | m05 has minimal dropout; needs arch change for unclear gain. |
| Adaptive beam width (uniform escalation) | D5 | Already explored; prior benchmark −13% paths regression. |
| SAM / Lion / Lookahead | E2 | Muon hit same plateau as AdamW; no mechanism to expect SAM/Lion to escape. |
| k_max curriculum | E3 | Already rejected (IHES E7). |
| V + cycle-decomposition aux | C4 | Speculative correlation; cheaper alternatives in T3 cover the aux-signal angle. |
| Thistlethwaite decomposition | P1.1 | 1–2 week math project; "useful chain may not exist" for icosahedral group. |
| Stream-of-Search transformer | P1.2 | Research-grade; high uncertainty at our scale. ~2 weeks. |
| PPO RL fine-tune of m05 | P1.3 + claude_research §6 | Sparse reward, famously slow. Lin-Liang 2024 only got 99.4% on 2×2×2. |
| Coset PDBs / additive heuristic | P1.5 + claude_research §4.5 | Korf-style PDBs likely infeasible at full scale on Megaminx (no clean small piece subset). |
| twsearch port | P2.3 | Gated on whether competition data ships a `.tws` file. |
| Schreier-Sims short words (standalone) | P2.4 | Same use case as PDBs; subsumed by T2.4 Minkwitz. |
| HER goal-conditioned policy | P3.1 + RESEARCH §4.5 | Single-goal task makes HER less applicable; ~2 weeks. |
| GFlowNet for diverse paths | P3.2 + claude_research §4.7 | 2–3 weeks. Unproven on permutation puzzles. |
| D3PM discrete diffusion | P3.3 | Author's own skip recommendation. |
| MuZero | P3.5 + claude_research §6 | True dynamics public — learning latent ones is wasteful. |
| Online learning during inference | P4.3 | Drift risk. T1.3 TTT is the cleaner version. |
| Inverse model I(s,s') | P4.5 | Use case (re-ranking) poorly defined. |
| LLM-as-solver | claude_research §6.3 | Useless on Megaminx; CoT solvers fail beyond 2–3 cube moves. |
| Replica exchange / parallel tempering walks | claude_research §3.3 | Skip on competition timeline; revisit only on multi-week run. |
| Coupling-from-the-past, quasi-random walks | various | Tiny gain, low ROI. |
| Energy-based models / score matching | RESEARCH §4.9 | Equivalent up to normalization to value learning. |
| Genetic algorithms on paths | RESEARCH §5.7 | T1.6 SA/LAHC is strictly simpler and likely does the same job. |
| SAT / ILP / RRT / quantum | claude_research §7 | All inapplicable at this scale. |
| Foundation models / cross-puzzle transfer | various | Speculative; payoff unclear. |

### Weak claims flagged

A few items in the big_ideas files don't hold up to scrutiny — call out before
investing:

- **gemini_ideas Bloom-filter MITM at d=8**: Bloom filters have *false
  positives*. A false positive in MITM would splice an *invalid* move sequence
  into a path. Without verification (which costs O(path_len)), this silently
  corrupts submissions. The proposal as stated would break solves.
- **gemini_ideas "Diffusion Models for Cayley Graphs (2024/2025 breakthroughs)"**:
  not verifiable as a specific paper. The Sun-Yang 2024 DiffuSCO work is real
  but for combinatorial optimization broadly, not cube-specific. Treat as
  aspirational.
- **deep-research-report-codex citation markers** (`citeturn22search11` etc.):
  unrendered browser-tool tokens. Underlying papers are probably real but each
  needs dereferencing before commitment.
- **claude_research's HKHLR 82-move solver claim**: most consequential claim in
  any file. **Day-1 web verification before scoping the Minkwitz port (T2.4)**.

---

## Recommended sequence

**Day 1 (free wins, ~1.5 h total)**
- Items 1–5 from the DAY-1 list above. The soft-Bellman + Polyak code changes
  ride into the next Bellman retrain at zero marginal cost.

**Days 2–4 (near-guaranteed wins, additive post-processing)**
- T1.5 auto-rescue (~1 day code + 1.5 h compute).
- T2.2 tail re-solve (~4 h code).

**Days 4–9 (the big unclaimed bets — run in parallel)**
- T1.1 macro library scrape + integration (3–5 days; engineering-heavy).
- T1.2 multi-agent ensemble: launch 8 ResMLP retrains on different seeds in
  series. ~10 h training + 8 full solves overnight.

**Parallel during macro work (compute-light, can interleave)**
- T1.3 TTT integration (~3 h code + ~50 min compute on full 1001).
- T1.4 solver-trace mining: instrument `03_solve.py`, capture from current
  production runs, mix into next Bellman cycle.

**Days 9–14**
- T2.1 insertion-finder port (pairs with T1.1 macros).
- T1.6 local search SA/LAHC on completed paths.
- T1.7 beam-tail concentration (post-TRT bf16).

**Week 3 (Tier 2 + verification-gated items)**
- T2.4 Minkwitz fallback (gated on Day-1 web verification of HKHLR claim).
- T2.3 listwise rank loss training.
- T2.5 distributional V head.
- T2.6 prioritized Bellman backups.

**Week 4+ (research / second wave if Tier 1 stalls)**
- T3.1 transformer m29 retry.
- T3.2 Bellman aux during RW.
- T3.7 BFS-d7 partial shell (GCP burst).
- T3.8 EfficientCube policy head as 9th ensemble member.
- T3.9 Generalized non-backtracking (refines shortlist item 7).

---

## Stacking note

These levers compose cleanly. The 70K target plausibly needs 3–4 landing in
series:

- **Search-space changes** (T1.1 macros, T2.1 insertion-finder, T2.4 Minkwitz):
  change the action space, can stack on any model.
- **Post-processing** (T1.5 auto-rescue, T1.6 SA/LAHC, T1.7 beam-tail, T2.2
  tail re-solve): associative on top of any path; multiple can apply.
- **Heuristic improvements** (T1.3 TTT, T1.4 solver-trace, T2.3 listwise, T2.5
  distributional, shortlist 3–7 bias-mitigation): replace m05 with a sharper
  teacher.
- **Inference compute** (T1.2 multi-agent, T3.6 multi-seed): orthogonal to
  heuristic changes; min-merge across many runs.

Pick winners in each layer; stack at the end.

---

## Honest meta-observations

1. **The big_ideas files are mostly recapitulations of common literature, not
   original suggestions.** The genuinely new high-EV items are: multi-agent
   ensemble (T1.2), solver-trace mining (T1.4), soft Bellman + Polyak (DAY-1),
   beam-tail concentration (T1.7), Minkwitz fallback (T2.4 if HKHLR verifies),
   distributional V (T2.5), prioritized PER (T2.6), BFS-d7 (T3.7). The other
   recommendations duplicate what's in `new_temp_ideas/` or shortlist.

2. **The CayleyPy paper (arXiv:2502.13266) is the most-directly-applicable
   evidence base** — same problem class, same architecture (ResMLP + beam),
   same regime. Their consistent message: **multi-agent ensemble + larger beam
   ≫ refined Bellman.** That should reorder our priorities away from
   Bellman-recipe variations toward (a) multi-agent and (b) beam scale.

3. **Bellman is load-bearing for a single agent at our scale — don't read #2
   as "rip out Bellman".** The CayleyPy ratio holds at THEIR scale (29–69
   agents, beam up to 2²⁴). At ours (8 agents max, beam ~2¹⁷): m07 (walk-depth
   only, no Bellman) → m05 (Bellman warmstart) was **+7 strat-5 solves AND
   −15% mean path length** — directly the difference between rank #4 territory
   and our current rank #3 88K submission. Mechanism: bigger beam + many
   agents absorb heuristic noise that Bellman would otherwise need to fix; we
   can't afford either luxury at scale, so per-agent heuristic sharpness still
   matters. Concretely: keep m05 as production teacher, run shortlist items
   3–7 (bias-mitigation cluster) as the *last* Bellman work, then stop —
   m17/m26/m26b already proved the recipe fixed point is reached past that.

4. **The Bellman cluster ceiling is real but partially-binding.** Items 3–7 in
   shortlist contest the recipe ceiling. Even if all of them work, gains are
   bounded — CayleyPy-RL Table 1 shows DQN refinement only raises Spearman
   0.92 → 0.98 at S₁₀, modest. The bigger lever is changing the search space
   (T1.1, T2.1, T2.4) or the inference compute (T1.2, T1.7).

5. **"Different inductive bias" remains one bet, not three.** B2 GNN / B3 Mamba
   / P1.2 Stream-of-Search all share the same "might break plateau"
   justification. T3.1 transformer is the most defensible single arch-lottery
   ticket because m18's prior rejection was wall-cost only.

6. **The 70K target almost certainly requires both** a search-space change
   (macros or Minkwitz) AND a post-processing layer (insertion-finder + SA +
   tail re-solve). Bellman-recipe variations (shortlist 3–7) probably gain
   −5K alone; the 70K crossing needs the macro/Minkwitz cliff plus
   post-processing additive gains.
