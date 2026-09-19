# How to build a model that solves 6x6x6 with beam search -- research memo

**Date:** 2026-08-24
**Inputs read:** every `CUBE666_*.md`, `HANDOFF_666.md`, `BIGCUBES_PLAN.md`,
`CUBE555_PROGRESS.md`, `555_ideas.md`, `AZ_HEAD_COMPARISON.md`, `rl_approach.md`
(cube444 Beam-AVI), `cube666/README.md`, `cube666/MACRO_POLICY_TRAINING.md`, all
`cube666/reports/*.md`, `cayley-py-666-cube/codex_666_solution.md`, `EXPERIMENTS.md`
2026-08-23/24, the vendored KMCoders README + `scores.csv`, and the memory notes on
cube444 / sparse-Q / two-phase. External: CayleyPy paper (arXiv 2502.13266) and repo,
CayleyPy RL (2502.18663), Q* search (Agostinelli), kSubS / AdaSubS / "What matters in
hierarchical search" (2406.03361), GFlowNet shortest paths (2603.01786), Douglas
"Diffusion models for Cayley graphs" (2503.05558), Santa 2023 KMCoders write-up.
**Revised the same day** after reading `codex_666_report.md` and `deep-research-report.md`
(section 7 lists what they changed; the bits/move ledger in 0b and the P3 correction are the
main edits).

---

## 0. Where we actually are (live leaderboard, 2026-08-24)

| team | 666 | mean/pid | 555 |
|---|---:|---:|---:|
| Stanislav Diener | **166,421** | 164.4 | 109,512 |
| us (submitted / on disk) | 176,889 / **171,019** | 174.8 / 169.0 | 187,780 |
| webmaking | 194,001 | 191.7 | 117,000 |

Deep pids (210-1011) cost us ~188 each and Diener ~182. Counting bound 102.2.
**The 110k target means 116 moves per deep pid = 1.13x the counting bound.** Nobody is
within a factor of 1.4 of that on any big cube: the CayleyPy paper's own best 555 result
(69 agents x 2^24 beam) is 92.16 = 1.39x its bound; Diener's 555 deep pids are ~1.7x;
Diener's 666 is 1.6x. So 110k is a research target, not a tuning target. A model-based
solver at **1.35-1.45x = 138-148 moves (140-150k)** would be a clear #1 and is the
realistic ambition of this memo.

Two facts from the outside world that the existing docs never use:

1. **Our 555 stack is ~1.9x worse than the published CayleyPy stack on identical states.**
   CUBE555_PROGRESS: "our ~177 loses to [the Santa pids'] 93.6 baseline". The paper gets
   92.16 on those Santa states. Same puzzle, same metric, same states. So "the 555 recipe
   transfers, the 666 one doesn't" is a statement about *our* recipe (sparse-Q, ResMLPQ,
   k_max past mixing, 2^21, 6 frames), not about the model family. Their recipe: dense
   diffusion-distance labels on *every* walk state, K_max = 0.9-1.0 x mixing (65 for 555,
   45 for 444), 4M-param ResMLP with BatchNorm, 8B states, beam 2^24, 29-69 independently
   trained agents, keep the shortest path.
2. **The CayleyPy repo ships a 6x6x6 config (group 004, K_max = 150) with no reported
   result**, the paper reports nothing past 5x5x5, and the organisers' own 666 leaderboard
   was empty for nine months. Read that as: they tried, it did not solve, and nobody
   published why. It is consistent with our findings, not a refutation of them.

Reference from Santa 2023 (KMCoders `scores.csv`): their colour 6x6x6 solutions are
154-165 moves in this metric; supercube 6x6x6 is harder (Diener 164, us 169). So the
classical pipeline family sits at ~1.5-1.65x bound everywhere it has been run.

### 0b. One yardstick for every phase, rung and model: bits per move

Every solver phase removes some information (bits of state) at some primitive-move cost;
the ceiling is log2(branching) = 4.86 bits/move on 666 (4.59 on 555, 4.26 on 444). This
normalises across puzzles, phases, rungs and models, and it calibrates the 110k target
against the whole field:

| solver | bits/move | fraction of ceiling |
|---|---:|---:|
| 3x3x3 optimal (20.5 QTM) | 3.18 | 0.99 |
| 444 Rokicki 44.39 / CayleyPy paper 46.51 | 3.54 / 3.37 | 0.83 / 0.79 |
| 555 CayleyPy paper 92.16 | 3.34 | 0.73 |
| 555 Diener deep pids ~113 | 2.73 | 0.59 |
| 555 ours 169.7 | 1.81 | 0.40 |
| 666 KMC pipeline (ours) deep ~188 / Diener ~182 | 2.66 / 2.75 | 0.55 / 0.57 |
| 666 single-orbit rung model (79 bits in 20.5 moves) | 3.86 | 0.79 |
| 666 orbit ladder at rung-1 quality (137) | 3.65 | 0.75 |
| **666 at 110k (116 moves)** | **4.32** | **0.89** |

The best learned solvers on any big cube run at 0.73-0.79 of ceiling; the classical
pipelines at 0.55-0.59; 0.89 has only ever been reached on the 3x3x3 with optimal solvers.
Use this column, not "inflation", to judge a phase: a KMC phase at 2.3 bits/move and a
neural rung at 3.9 bits/move are directly comparable.

---

## 1. One law that fits every measurement: the blind zone

Define for a (model, target set) pair:

- **mixing depth** M = log_b(index of the target set) -- the depth at which random walks
  from the target stop carrying distance information (b = effective branching);
- **visibility** V = deepest depth at which the model's child ranking is still clearly
  above chance (top-1 by depth >= ~2x chance);
- **blind zone** Z = (typical start distance) - V.

Every number in the 666/555 record fits **"a beam solves iff Z <~ 10, and inflation grows
with Z"**:

| problem | M | V (measured) | start dist | Z | outcome |
|---|---:|---:|---:|---:|---|
| 666 flat, arm A | ~102 | ~75-80 (top-1 3.2x at 40, 1.5x at 60, 1.25x at 72, chance at 85) | ~105-110 | 25-30 | 0 solves at any width; wider beam goes *below random* on fixed stickers |
| 555 flat, ours | ~72 | >=65 (7.5x chance at ~72) | ~72 | 0-7 | solves 84% w/ 6 frames, **2.35x** inflation |
| 555 flat, CayleyPy paper | ~72 | >=72 (inferred) | ~72 | ~0 | 92.16 = **1.28x** true distance |
| 1 centre orbit rung (`rung_o5/o6`) | ~16-20 (b_eff<29) | ~14-16 | ~19-22 | 2-6 | 24/24 in 18.8-22.3; width is a lever (2^11 42% -> 2^16 100%) |
| 2 orbits, joint 48-slot model | ~32 | ~24 | 17-32 | 8+ | 31-33% |
| 3 orbits | ~49 | ~37 | -- | 12+ | 0%, `kept 24/24/0` |

Corollaries that drive everything below:

- **The flat route needs V >= ~95 on the full puzzle.** Nothing measured here or published
  anywhere reaches that. Width, frames, capacity, Bellman-from-the-same-net all leave V
  where it is; the descent probe showed width *amplifies* the error past V.
- **A rung's V is ~0.75-0.9 of that rung's own M.** So a rung is safe only if its start
  distance is inside ~0.8 M. A single centre orbit (M ~ 16-20, start ~ 19) is marginal;
  that is why width mattered there and why it still solved.
- **The 2- and 3-orbit "composition cliff" was measured in the wrong regime for a ladder.**
  Distance from a ladder-start state (orbits 1..k-1 solved, orbit k random) to S_k is
  bounded below by log_29(24!) = 16.2 and is plausibly 20-25 -- *inside* a joint k-orbit
  model's visibility for every k >= 2 (V ~ 24 at k=2, ~37 at k=3, ~50 at k=4). The
  measured failures used either a SUM of per-orbit heuristics (provably stuck in the
  stabiliser: every move breaks a solved orbit by more than it fixes) or a joint model
  trained on walks from S_k that never contain ladder-start states (orbit 1 *exactly*
  solved + orbit 2 *uniformly* random is measure-zero on the walk manifold). The joint
  model's failure is therefore a **distribution-mismatch hypothesis, not a visibility
  fact**, and it has a cheap test (section 4, P2-a).

The rest of this memo is organised around the only three ways to keep Z small:
(A) raise V of a flat model as far as the family allows; (B) shrink M per rung with a
coset ladder; (C) hand the model only the last ~35-55 moves of a classical rough solve.

---

## 2. What the literature adds (and what it does not)

| source | what it establishes | how it bears on 666 |
|---|---|---|
| CayleyPy paper, arXiv 2502.13266 | 4M ResMLP+BN, dense walk-index labels, K_max ~ 0.9-1.0 M, 8B states, beam 2^24, 29-69 agents: 444 = 46.51, 555 = 92.16 (1.39x bound) | Best known flat recipe; **never reported on 666**, repo has a 666 config. Our 555 run is 1.9x worse than this on the same states, so our 666 "V ~ 80" is a lower bound on what the family can do, not an upper bound. |
| CayleyPy RL, 2502.18663 | solution length ~ linear in log(beam width); DQN/Bellman phase after warm-up helps on LRX | Confirms width is log-linear at best -- a 25-move blind zone is not a width problem. |
| cube444 Beam-AVI (`rl_approach.md`, ours) | bootstrapping on the **beam's own state distribution** beat the floor by 4.2%; the same budget on random-walk states *destroyed* the model; precondition = on-path candidate rank in the beam far above chance | The right training loop for any regime where the model already has signal (inside a rung, inside a residual). Precondition provably fails on the flat 666 at depth 85+. |
| Q* search / DeepCubeAQ (Agostinelli et al.) | a Q-head prices a 1,872-meta-action space at <4x the cost of 12 primitives | Macro-action beams are cheap with a Q-head; but our macro line already showed the limiting factor is *insertion-aware cost*, not action pricing. |
| kSubS / AdaSubS / "What matters in hierarchical search" (2406.03361) | subgoal search is robust to value noise (solves ~40% of cubes with sigma=100 value noise where best-first collapses); helps most when the value is hard to learn / trained on heterogeneous experts; with homogeneous data low-level search matches it | Directly relevant: our failure is a noisy value at depth. A *k-step-ahead state generator* (subgoal proposer) is an untested lane here; the "trajectory-memory" and "path-context" lines are not that. |
| GFlowNet shortest paths (2603.01786) | competitive lengths with smaller test-time search budgets on Rubik's/permutations | Our GFN port exists ([[gfn-pathfinding-ported-gated]]); no evidence it changes V. Long shot. |
| Douglas & Fraser-Taliente, diffusion models for Cayley graphs (2503.05558) | systematises "reverse process" pathfinding; proposes a reversed-score ansatz | Framing only; no big-cube results. |
| KMCoders Santa 2023 | corners exact -> parity via GF(2) -> SA over inner-slice words with (length + alpha x misplaced) -> greedy 4-move commutators -> insert 3-rots at arbitrary times (<=14 moves each, 24,288 of them) | Already vendored and producing 193-195/pid. Its cost split (section 4, P3) is where the hybrid win is. |

Nothing in the literature solves a 6x6x6 supercube with a learned heuristic, and the one
group that could have (CayleyPy) did not publish a result. Expect no shortcut.

---

## 3. What is closed (do not re-run) and what only *looks* closed

Closed by measurement (six independent levers, all null or worse): capacity 26M->120M;
ResMLPQ v1->v2; orbit-factored one-hot as a pure encoding change; flat vs tilted pivots;
width 2^16->2^22 (anti-productive past 2^21); frames-as-retries (0/20, and no per-frame
success to compound); one-step Bellman from the same net on the full puzzle (13->7->6->1);
fixed-CSV imitation (0.977 = memorisation); stabilizer transversal policy as a scorer
(solves projected stage-0 but 650-2,300 primitive moves); trajectory-chunk hierarchy
(52,904 options, zero new wins); fitted Bellman policy on macro depth-6 (0/8);
transition reranker (training-state recovery only).

Only *looks* closed:

1. **"666 models are at chance from depth 85" is a property of arm A's recipe.** No arm
   used dense diffusion labels, K_max <= mixing, BN, or the paper's data volume; the 555
   calibration above says our recipe is far from the family's frontier.
2. **"Composition kills rungs at 3 orbits."** Measured with a sum scorer and with a joint
   model trained off the ladder-start distribution (section 1).
3. **"Bellman cannot work here."** True for the full puzzle from arm A. Untested inside a
   rung or on the classical residual, where the model *does* have signal and the cube444
   Beam-AVI precondition holds.
4. **Frame aggregation inside the scorer** (`Q_avg`/`Q_max` over 48 frames, WINDOW_PLAN
   W2) and **row-mean scoring** were never run; both attack the exact pathology the probe
   measured (selection on the left tail of per-state error). Neither creates signal, but
   they decide whether a beam uses the signal a model has.

Logistics: `models/q666_a_final_ARMA_deployed.pt` (arm A, 317 MB) is **not on this
machine** -- restore it from `gdrive:artgor_meta/cube666_handoff_2026-08-20/models/`
before any inference-side experiment. The puzzle/symmetry substrate (`cube_nnn/`), the
KMC binary with the `KMC_ROUGH_PATH` / `KMC_CORNER_PATH` hooks, the 24,288-macro library,
the corner IDA*, and the r=4 window rewriter are all local and verified.

---

## 4. Proposals, ranked by (expected score gain) x P(works) / cost

### P1. Calibrate the family: reproduce the CayleyPy flat recipe on 666 (1-2 GPU-days)

Not because it will score -- even if it solves, 2x+ inflation gives 200+ moves and loses
to the classical 169 -- but because **every other proposal trains rung/residual models
with the same recipe**, and we do not know how much of V ~ 80 is the puzzle and how much
is the recipe.

Arms (one variable each, matched budget, same seed family):

- A0: arm A control (recall-by-depth from `53_probe_compare.py`: 0.090/0.043/0.035/0.029
  at 40/60/72/85).
- A1: paper recipe verbatim: V head, dense labels on all walk states, non-backtracking
  walks, **K_max = 95** (0.9 x mixing; arm A used 122 with `pivot_tilt`, i.e. ~20% of its
  labels were past mixing = noise, oversampled), ResMLP 1024/256 x 1 res block + BN,
  batch 10,000, ~8B states.
- A2: A1 with K_max = 105 and 115 (the K_max sweep the teacher post-mortem says was never
  run in isolation).
- A3: sparse-Q PieceTransformer (Vlad's `cayleypy-training-core`, piece tokens = cubies)
  -- on cube444 this beat our whole TPU V-beam pipeline by ~5 moves/pid; it is the only
  architecture change in this project that ever moved a colour/big cube.

**Gate = visibility, nothing else:** top-1-by-depth curve on a fixed held-out walk bank
at r = 40/60/72/85/95/105 (>= 1,024 states per depth), and the sibling gap Q(next)-Q(undo)
by depth. Report V = deepest depth with top-1 >= 2x chance. Promote to a 2^22 descent
probe only if V >= 90; run the probe with `fix_mean >= 9.0` as the acceptance metric.

Decision rule: if the best arm's V is still <= 85, the flat route is closed *for the
family*, and P2/P3 inherit the best recipe. If V >= 95, run the 2^24 / multi-agent beam on
the TPU kernel (PACK_SIZE >= 221) purely as a capability result and as a source of
verified deep paths for P2-c.

### P2. The corners-first orbit ladder with joint, ladder-distribution-trained rung models

This is the only route in which every learned decision sits inside a rung with Z ~ 0,
and it is the route MODEL_STRATEGY already recommends -- with two changes that address
why the earlier composition runs failed.

Structure:

1. **Corners exact first** (IDA*, 10.35 mean, already built), then the 6-bit parity
   repair (0-2 moves) as in KMC. From here on every rung target set includes "corners
   solved"; all 36 generators remain legal (outer turns may disturb corners mid-rung).
2. **Six rungs, one 24-piece cluster each** (4 centre orbits, 2 wing clusters). Rung k's
   scorer is a single joint model D_k(s) = estimated distance to S_k = {clusters 1..k and
   corners solved}, on the k-cluster projection plus corners. **Never a sum of per-cluster
   models** (measured stuck: `kept 24/24/0`).
3. **Soft boundaries:** score = D_k(s) while rung k is active, but allow rung k+1 to start
   from any beam state with D_k <= 1-2 and let D_{k+1} pay for the repair. Portfolio over
   cluster orders (the stabilizer audit's 0-4-5-2-3-1 is for the exact 3-cycle library;
   for the soft ladder any order is legal and a few should be tried) and over 48 frames.
4. **Exact tail:** the last rung ends in the exact d<=5 ball or the one-macro endgame.

Why the counting says it can beat 166k: each rung removes 79 bits; at rung-1's measured
quality (19-22 moves) the ladder totals 6 x ~21 + 11 = **~137** (~135k). At 28 moves/rung
it is ~180 (a loss). The whole bet is rung quality at k >= 2, which is exactly what the
gates below measure before anything expensive is built.

The two changes vs. the earlier attempts:

- **(P2-a) Train rung models on the ladder's own start distribution.** Walk-from-S_k data
  never contains "clusters 1..k-1 exactly solved, cluster k uniform". Add those states
  explicitly, and label them by **Beam-AVI on the rung** (cube444 recipe: run the rung
  beam from ladder-start states, harvest `1 + min_a' Q_target(child)` on survivors plus a
  256-parent full expansion, exact anchors every batch, warm-start from the walk model).
  This is the one training loop in this repo with a verified positive result on a
  beam-deployed scorer, and its precondition (on-path rank far above chance) holds inside
  a rung by construction (Z ~ 0).
- **(P2-b) Measure the 2-cluster rung *properly* before training anything:** joint
  48-slot walk model (already exists or is 1 GPU-hour), 200 ladder-start states (cluster 1
  solved, cluster 2 random), beam 2^16-2^18, all generators, goal = S_2 exact. Report
  solve rate and moves. Then the same with the P2-a fine-tune. Expected under the
  distribution-mismatch hypothesis: walk-only model 30-40%, AVI-fine-tuned model >= 90%
  at ~20-25 moves. If the AVI model stays < 60%, the ladder is dead and P3 is the plan.

Model inputs for rung and residual scorers (from the codex report, and consistent with the
from-scratch finding that a learned joint scorer beats any combination of exact tables):
per piece, orbit/position/target, **exact cycle id, cycle length and position-in-cycle**,
the per-cluster unrestricted 3-cycle count, and several **5-piece PDB values**
(24P5 = 5.1M entries, ~5 MB each) as input features and auxiliary targets -- never as the
scorer. On residual-type states these exact features carry the information a raw sticker
one-hot has to rediscover. A cross-cluster attention trunk is affordable in a macro beam
(10-30 decisions); the primitive-move rung beam needs the cheap ResMLP-Q variant with the
same features, and the two must be compared at matched accelerator-seconds, not matched
width.

Gates in order (from MODEL_STRATEGY, kept): B (2-cluster oracle/learned >= 90% on 100+
held-out), D (3-cluster composition with a credible path to high coverage), E (24-48 fully
mixed held-out states replayed to identity), F (min-merge vs 171,019). Report bits/move
per rung at every gate; abort if a rung falls below ~2.9 bits/move (~27 moves) at gate D.
Two diagnostics from the deep-research report belong in every rung gate: the
**beam-survival waterfall** (rank of the known-good child at every layer -- the layer where
it is evicted separates proposer failure from critic failure) and **counterfactual frontier
regret** (when the good route is lost, wide-probe the survivors to check whether they were
genuinely worse or an equivalent route).

Cost: ~1 GPU-day per rung model; the k=5,6 models see 120-144 slots + corners and need the
P1 recipe. The 96-frame x order portfolio at the end is embarrassingly parallel on Kaggle.

### P3. Hybrid: classical rough phase + neural finisher on the residual

The KMC trace on state 500 splits as corners+parity **11**, SA rough path **67**,
3-cycle insertion finisher **126** (net, after cancellation; 29 x 4.34) = 193 (204 raw).
In the bits/move ledger, with the SA residual at ~10-12 misplaced pieces per cluster
(78 misplaced stickers; residual set 248-293 bits, **counting distance 51-60**):

| KMC phase | moves | bits removed | bits/move |
|---|---:|---:|---:|
| corners + parity | 11 | 26 | 2.4 |
| SA rough path (inner slices) | 67 | 182-227 | 2.7-3.4 |
| 3-cycle insertion finisher | 126 | 248-293 | **2.0-2.3** |
| whole pipeline | 193 | 501 | 2.6 |

So the finisher is the least efficient phase and the largest in absolute moves, but only
by ~1.3-1.5x over the SA phase -- not the "2x-inflated outlier" a naive reading of 126
moves for a distance-55 residual suggests. The true distance of the residual state is
**unmeasured**: counting gives 51-60 as a lower bound, the finisher gives 126 as an upper
bound, and residual-type states (corners solved, ~10 scattered misplaced pieces per
cluster) are far off the random-walk manifold, so their distances may sit well above the
counting bound. That number is the first thing the gate below measures.

What a neural finisher would be worth, under the two scenarios the gate distinguishes:

- residual truly ~60-70 deep and a neural beam reaches ~3.3-3.7 bits/move on it (the
  rung-1 / CayleyPy-555 band): finisher ~75-85 moves -> ~140-150/pid -> **~140-150k**;
- neural competence only to ~35 on residual-type states: push SA further (at its
  2.7-3.4 bits/move) and let the beam finish the last ~35: ~11 + 95 + 40 = ~146-155
  -> **~150k**, but only if the deeper SA residual really is within competence.

Either scenario beats 166k; both use a *bounded-horizon* model exactly where the docs
say the model works, and the classical pipeline supplies coverage (no pid is ever
unsolved). If the gate shows the residual is 90+ deep, P3 is dead and the finisher's
cost is intrinsic to the residual, which strengthens P2.

Cheap gate first (no training, 1 day, needs arm A restored): take 30 KMC SA endpoints at
three residual sizes (tune `alpha` / anneal length to leave ~10, ~18, ~29 three-cycle
units), run arm A at 2^20 with the d<=5 ball as goal set and `history_depth=4`, and
compare `achieved` against KMC's own finisher cost on the same residual (`KMC_ROUGH_PATH`
relabelling gives that number exactly). Splice, commute-reduce, r=4-rewrite the seam,
replay. Any residual size where achieved < KMC finisher is deployable immediately.
Remember the ~28-move "prefix context" effect: the KMC finisher cancels inside the rough
prefix, an appended neural path does not, so the comparison must be end-to-end path
length, not finisher-vs-finisher.

If arm A is competent only to residual ~10-12 units (distance ~35), extend competence on
**this** distribution with Beam-AVI: start states = SA endpoints (or synthetic
solved + m random 3-cycles / commutators, m = 5..30), goal = exact tail, curriculum m
outward, warm-start from the best P1 recipe. The macro line's certified-return and
exact-endgame machinery (`121`, `135`, `136`) is reusable as the anchor/endgame here.

### P3'. The codex route: model as rough-prefix optimiser, exact finisher as terminal oracle

`codex_666_report.md` proposes the mirror image of P3: keep the KMC insertion finisher and
put the model *before* it. Define, for a corner/parity-normalised prefix p, C(p) = length
of the complete replay-verified solution after running the exact finisher on p. Then every
beam state is terminal, A(p,a) = C(p) - C(pa) is a verified counterfactual return, the
search is anytime and never worse than the fallback, and no value is ever bootstrapped
from the model itself. This is the formalisation of the prefix-context cascade that
already produced 171,343 -> 171,019 (EXPERIMENTS 2026-08-23/24), and its engineering
advice is sound and cheap:

- **instrument the Rust solver to export populations** (diverse rough prefixes at several
  depths, endpoints, residual features, completed lengths, config) instead of querying one
  candidate at a time -- dozens of comparable labels per run;
- **label comparability** (fixed completion beam/anneal/seeds/frames; the same min-of-k
  attempts at train and deploy) and **root-disjoint splits** -- both were learned the
  hard way here (the beam-4,000 vs 20,000 label bug, the cache-key bug);
- **oracle-ceiling gate before any training**: best wide-completed path among 256 diverse
  candidates per root over 100 fresh roots. If that pool averages ~160 moves, no ranker
  reaches 116 and the action grammar, not the model, is the constraint;
- active expert iteration on the beam's own retained / wrongly-discarded / uncertain
  candidates; ranking losses weighted by verified move differences; a beam-aware loss
  (penalise the layer where the oracle-good trajectory is evicted).

Its ceiling is the point: the report's own milestone for this route is ~160k, because the
finisher's 2.0-2.3 bits/move stays in every completed path. **P3 and P3' compound rather
than compete: the terminal oracle should become C(p) = min(KMC finisher, neural finisher
from P3), so every improvement in the finisher lifts the prefix ranker's labels.** Run the
oracle-ceiling measurement first (it costs a day and also yields the SA endpoints P3
needs), then decide how much of the two weeks goes to each side of the finisher.

### P4. Search-side fixes to the noise-exploitation pathology (hours each, use everywhere)

These do not create signal. They stop the beam from selecting the model's errors, which
the descent probe measured at 2^21-2^22 (`fix_mean` 1.3 vs random 9.0). Gate every one of
them on the descent probe with `fix_mean >= 9.0` over 500 steps, not on `min Q`.

1. **Frame-averaged / frame-max Q** (`Q_avg`, `Q_max` over k of 48 frames; W2). First
   measure the across-frame spread of Q on deep states -- if arm A is already
   near-equivariant the spread is small and this resolves in one line.
2. **Row-mean scoring.** Score a *child* by the mean of its own 36-column Q row (a
   36-sample estimate of V) instead of the single Q(parent, a) entry. Costs 36x per
   scored state, i.e. 2^21 -> 2^16 width -- which is the right trade to test given that
   width was measured anti-productive.
3. **Lineage-smoothed score** (PHS-cumulative analogue that helped on megaminx): rank by
   an EMA of Q along the last k ancestors, so a single spurious underestimate cannot win a
   slot.
4. **Independent-agent averaging** (2-4 models from P1 with different seeds) as a scorer
   ensemble, not a portfolio.
5. **Conservative (uncertainty-penalised) scoring**, which both external reports arrive at
   independently: rank by `mu + beta * sigma` for a cost (equivalently subtract
   `beta * sigma` from predicted savings), with `mu`/`sigma` over frames or ensemble
   members. Averaging reduces noise; this additionally *penalises disagreement*, which is
   the cost-minimisation analogue of conservative offline RL and targets the "confidently
   wrong" states the probe found directly. Sweep beta in {0, 0.5, 1, 2} on the descent
   probe.

### P5. Long shots worth one bounded experiment each

- **k-step subgoal generator (kSubS-style).** An autoregressive model that emits the
  *state* k=8-12 moves ahead on a near-geodesic walk, verified by a local exact/beam
  bridge. The literature's specific claim is robustness to value noise at high sigma --
  our exact failure mode. Cheap to gate on the 2-cluster rung.
- **GFlowNet shortest-path training** on a rung model (port exists) -- competitive lengths
  at smaller test budgets in the 2026 paper; unknown effect on V.
- **Wider rung vocabulary via inner-slice-only rungs.** The 24 inner generators fix the
  corners for free (KMC's step 2); a rung restricted to them never pays corner repair, at
  the cost of a smaller action set for wings. Measure bits/move before adopting.

---

## 5. What not to do (consistent with the closed list, plus three new items)

- No new full-puzzle arm on the arm-A recipe at any capacity, width, or frame count.
- No Bellman on the full puzzle from a net that is at chance at 85.
- No sum-of-rung heuristics, ever (`kept 24/24/0` is the signature).
- New: do not train rung models only on walks from S_k and then start them from
  ladder-start states -- that is the untested mismatch behind the composition cliff.
- New: do not judge any rung or residual model by loss, `min Q`, or pooled top-1; judge by
  top-1 *by depth* and by `fix_mean` under the beam, then by replayed moves vs 171,019.
- New: do not extend the macro-option hierarchy line; its own gates showed options repair
  horizon and retrieval but do not create cheaper trajectories.
- New: do not port the deep-research report's encoding advice. It treats 666 as a
  **colour** cube (216 x 6 one-hot, `num_classes=6`, colour-relabel symmetry, no
  `invert_state`). This competition's 666 is a **supercube**: 216 distinct stickers,
  solved = identity, 9 orbits of 24, `invert_state` legal, 48 frames x inverse = 96.

---

## 6. Two-week plan

| day | item | deliverable / gate |
|---|---|---|
| 1 | restore arm A; build the depth bank (1,024 states x r in {40,60,72,85,95,105}) | recall-by-depth harness with chance lines |
| 1-2 | patch the KMC solver to export rough-prefix populations + completed costs (P3'); dump SA endpoints at 3 residual sizes | one instrumented run -> comparable labels, residual states, oracle-ceiling pool |
| 2-3 | **P3 cheap gate**: 30 SA endpoints x 3 residual sizes through arm A; **P3' oracle ceiling**: 100 roots x 256 candidates, wide-completed | residual true-distance band; achieved vs KMC finisher (replayed, deploy any win); best-of-pool mean (~160 => prefix ranking alone cannot reach 116) |
| 2-3 | **P4-1/2**: frame spread + `Q_avg`/row-mean on the descent probe | `fix_mean >= 9.0` yes/no |
| 2-5 | **P1** A1/A2/A3 training (paper recipe; K_max sweep; PieceTransformer) | V per arm; best recipe chosen |
| 3-4 | **P2-b**: joint 2-cluster rung from ladder-start states, walk-only | solve rate / moves (the decisive number) |
| 5-7 | **P2-a**: Beam-AVI fine-tune of that rung on ladder-start states | >= 90% at <= 25 moves, or ladder closed |
| 8-10 | rung 3 with the winning recipe; gate D | inflation per rung <= 27 |
| 8-12 | **P3 full**: residual-distribution Beam-AVI if the cheap gate showed competence < needed | end-to-end 1,012-pid pass vs 171,019 |
| 12-14 | whichever of P2/P3 is ahead: 96-frame portfolio, r=4 seam rewriting, min-merge, replay, submit | first model-built submission under 166,421 |

Expected value, honestly: P3 is the most likely to put a model-built path under Diener's
file within two weeks (it needs the model only where it is already competent). P2 is the
only route whose ceiling (~130-140k) is meaningfully below the classical family's
~160k, and its decisive test costs a day. P1 is cheap insurance that we are not
concluding "impossible" from a recipe that is provably 1.9x off the family's frontier on
the sibling puzzle. 110k stays out of reach for every method here; say so in any plan
that quotes it.

---

## 7. What the two external reports add, and where they are wrong

**`codex_666_report.md`** (read after this memo was first written) is the strongest
statement of the prefix-optimisation route and is adopted above as P3'. Its lasting
contributions are methodological and apply to every proposal here: verified
counterfactual returns instead of any bootstrapped value; the oracle-ceiling gate before
training; instrumented population export from the classical solver; comparable labels and
root-disjoint splits; conservative scoring; the beam-aware loss; and the milestone ladder
160k (learned search reliably improves KMC) / 145k (cross-cluster planning) / 130k
(global macro optimisation) / 110k (near-optimal). Where it is incomplete: it keeps the
KMC insertion finisher as the terminal authority, so its own ceiling is bounded by that
phase's 2.0-2.3 bits/move; P3 attacks exactly that phase, and P2 removes the phase
structure altogether. Its model proposal (geometry trunk with exact cycle/PDB features,
cross-orbit attention, proposer + joint reranker, V_geometry - B_context decomposition) is
adopted for the macro-beam side and for residual-state scorers. Reading it also forced the
correction in P3: "the finisher is 2x inflated" became the ledger in 0b, where the finisher
is the least efficient phase by ~1.3-1.5x and the residual's true distance is an unmeasured
number in [51-60, 126].

**`deep-research-report.md`** is a generic neural-beam programme written without access
to the 666 docs (it says so) and it gets the puzzle wrong: it plans for a colour cube. Its
top recommendation -- flat sparse-Q ResMLP + very wide beam, then a PieceTransformer 2x2
-- is what arm A already is, and it is measured dead at 2^22 by the descent probe. It also
does not know that width was measured anti-productive. What survives and is folded in:
select scorers at **matched accelerator-seconds** (a 5x slower scorer must win at 1/5 the
width); the beam-survival waterfall and counterfactual-frontier-regret diagnostics;
conservative mu + beta*sigma scoring; label hygiene (dedup-min, exact override, splits by
scramble family); the memory table (2^24 with materialised children is 121 GiB -- the
streaming TPU kernel is not optional); and the pointer to monotonic / beam-stack variants
for hard-tail rescue (low priority: our width pathology is a wrong scorer, which
monotonicity in the scorer's own terms cannot fix).

Net effect on the ranking: P1-P4 stand; P3' is added and shares P3's data engine; the
first two days now produce the residual-distance band, the oracle ceiling and comparable
labels from one instrumented KMC run before any model is trained.
