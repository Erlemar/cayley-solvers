# Megaminx experiment backlog

Ideas ranked by **expected payoff per hour of work**. Check off as done and point to
the EXPERIMENTS.md row. Based on a synthesis of:
- Our IHES experiments (EXPERIMENTS.md in project root, NEW_IDEAS_SYNTHESIS*.md, PROGRESS.md)
- Literature: DeepCubeA (2019), CayleyPy paper 2502.13266 (Feb 2025), CayleyPy-RL 2502.18663,
  EfficientCube TMLR 2023, Q* search 2102.04518, DeepCubeAQ 2102.04518, PHS 2103.11505,
  Levin Tree Search (IJCAI 2023), Limited-horizon updates 2511.10264
- Public Kaggle Megaminx kernels (data points, not ground truth — top votes reflect clarity
  not leaderboard rank; Kuznetsov/DrozdovDan/Rokicki haven't shared code)

**Current best (submitted, 2026-04-27)**: **88,195** moves, rank **#3 of 16 teams**.
**Behind us**: Rokicki 93,606. **Next target (DrozdovDan, #2)**: 81,946. **Top
(Kuznetsov, #1)**: 79,971. **Next milestone**: break 80K barrier.

> **NOTE (2026-04-29)**: Active priority triage now lives in
> [`tier1-tier2-tier3-merged.md`](./tier1-tier2-tier3-merged.md). This file is
> retained as a historical backlog/status tracker. Items below tagged
> `[→ COVERED]` are subsumed by the merged triage — don't re-process them
> here. Items tagged `[REJECTED]` were moved into the Rejected section below.
>
> **NOTE (2026-05-04)**: Alternative-paradigm brainstorm with detailed
> per-idea analysis lives in [`novel_ideas.md`](./novel_ideas.md). Covers
> macro-Q shortlist, frontier replay, policy-guided search, coordinate
> solver hybrid (Botz phase tables), ReduceFactor DAG, path relinking,
> twsearch integration, learned bidirectional, per-phase specialist, and
> megaminx insertion-finder. Each entry has falsifiable tests + queue
> mapping. Items will be promoted to `to_do_shortlist.md` as they're
> picked up.

---

## Active sprint (2026-04-27+) — past the 96%-model_s ceiling

See `megaminx/speed_optimizations.md` for the full ranked list with status, and
`megaminx/tensorrt_gcp_plan.md` for the next concrete execution step.

**Where we landed**: 88,195 (rank #3) by stacking m05 fresh + Phase B + beam-stack
rescue. The next jump requires a faster model forward (TensorRT) OR a sharper
heuristic (m26/m26b capacity scaling, multi-task training).

**Recently confirmed dead ends (don't re-prototype)**:
- Manual CUDA Graphs alone — −34% slower than `--compile` on 3-puzzle. Inductor
  fusion is the win, not graph capture.
- beam_decay (geometric narrowing) — wall savings real but +3.5% paths (pid 492
  +7 moves). Quality-gated out.
- Stochastic beam — not a speed lever; same compute per step, just different which-
  states-survive.
- Async pipeline (skip-syncs + deferred solved-check) — within noise; the 96%
  model_s ceiling caps async wins at ~4%, in practice <1%.
- Adaptive beam escalation (16k→65k→131k early-exit) — −78% wall but +13% paths.
- cayleypy `iterated` mode w/ history_depth=10 — +124% wall AND +10 paths vs ours.
- cayleypy `simple` + MITM (BFS-d6 hashed_neighbourhood) — +17% wall vs ours;
  m05 already navigates d≤6 shell as a side-effect.

**Live work**:
- TensorRT on GCP (PLANNED, see `tensorrt_gcp_plan.md`) — only remaining pure-speed
  lever with predicted >5% gain. 6 steps, ~2.5h active + ~12h GPU full solve. Hard
  gate: 24-puzzle path-sum must equal 2013.
- m26 + m26b Kaggle training (RUNNING) — capacity-scaling A/B vs m05's
  [2048,512]×2. Acceptance gate: ≥+3 strat-5 solves AND mean model_avg ≤0.95×.

**Production speed levers already deployed**:
- `--compile` flag on `beam_lab/beam_search.py` (`mode=reduce-overhead`,
  pad-to-batch-size). −27% wall on 12 puzzles, paths IDENTICAL.
- `m23` Q-shortlister at α=2 (recall=100%) — 4.4× wall at beam 131k, +6%
  paths funded by scaling beam from 131k to 524k.
- Beam-stack backtracking (`beam_lab/beam_search_stack.py`) — rescued pid 490
  (407→126) and pid 920 (758→137) for our submitted score.
- BFS-d6 window post-processing — −0.2% on top of BFS-d5 (saturated).

**Acceptance gate (still binding for any new heuristic)**: ≥+3 strat-5 solves AND
mean `model_avg` ≤ 0.95× current best, no bucket regresses by more than 1.

**Anti-patterns confirmed (across sessions)**:
- Tier-1 CUDA micro-opts (topk, inference_mode, hash reuse, GPU tree, implicit
  indices, no double chunking) — within noise on CUDA. Algorithmic only.
- Bellman round 2+ on top of m05 (m17 was a no-op).
- Transformer at n=120 (m18 hit Kaggle 12h before reaching MLP-equivalent).
- Distillation to smaller V model (m06 lost ordering; rejected by user as a quality
  risk for current 88K submission).
- Internal_batch_size > 16k on 4090 — GPU compute-bound; no gain at 32k or 65k.
- Multi-puzzle batched beams on L4 — compute-bound, packing K puzzles slows down
  ~3.7× because the per-batch model call dominates differently.

---

## Strategic frame

**Root-cause diagnosis** (from IHES NEW_IDEAS_SYNTHESIS.md, re-derived for Megaminx):
- Heuristic ranking errors dominate (≥50% of gap). Tighten the heuristic first.
- Beam pruning of correct paths (~30%). Bigger beam, int8, Q-function all help.
- Non-backtracking constraints (~15%). NISS / six-axis ensemble add cheap diversity.
- Post-processing ceiling (~5%). Already at same-face + adjacent-inverse.

**Guiding principle**: start with our innovations (IHES-verified wins), then evaluate
canonical baselines; submit only when we can beat pp_fallback by meaningful margin
(not 8 moves).

---

## HIGHEST priority (tier 1) — STATUS

1. [x] **C1 — int8 state encoding** — DONE. In production beam_lab + 03_solve.py via `state_dtype=torch.int8`.

2. [x] **C2 — Adaptive beam per puzzle** — partially DONE. `--beams 32k,131k` syntax supported in 03_solve.py. Auto-escalation on first-pass-fail still pending (would shave ~30% on full 1001 wall).

3. [x] **C7 — NISS** — DONE. Submitted as `m07_phase1_partial440.csv` etc; doubles wall but adds path diversity.

4. [x] **A1 — Bigger arch m03 [2048,512]×2 k_max=80** — DONE. m03 (Kaggle) and m07 (local replica) both trained; m07 is current canonical V-baseline.

5. [x] **E3+C3 — BFS-d5/d6 tables** — DONE. `bfs_table_d5.pkl` (1.4M states), `bfs_bytes_d6.pkl` (19.4M states). Used for window post-processing (`pp_bfs6_fallback`) and MITM (in `beam_search_mitm.py`).

## HIGH priority (tier 2) — STATUS

6. [x] **A3 — Bellman auxiliary loss (m05)** — DONE, NEW BEST HEURISTIC. Used for Phase 2 retry + Phase B; produces 95,682 submission with m07. Strat-5: 50/51, 89.4 model_avg.

7. [x] **A2 — Bellman from scratch (m04)** — DONE (rejected). 15/21 strat-2, less than m07's 18/21.

8. [x] **A4 — Q-function head (m06)** — DONE (rejected). 10× speedup but lost ranking. **Redo planned as `m23` Q-shortlister with hybrid MSE+KL loss; script ready (`09_train_q_shortlister.py`).**

9. [ ] **D1 — Multi-seed ensemble** — REJECTED for m05 (deterministic beam => same paths across seeds). Only useful when paired with diversity from a different model (m17, m21, m22).

10. [ ] **B3 — Negated-beam data augmentation** — not started. Lower priority than Tier-S items in `speed_optimizations.md`.

## MEDIUM priority (tier 3)

11. [REJECTED 2026-04-27] **C4 — CayleyPy `beam_mode="iterated"`** — moved to "Rejected" section below.

12. [REJECTED 2026-04-27] **C5 — Distance-adaptive beam / beam_decay** — moved to "Rejected" section below.

13. [ ] **C10 — GCP L4 (24 GB) for beam 2^18–2^20**. Tail re-solve on unsolved
    hardest puzzles. $1–2 of compute per run.

14. [ ] **E6 — ReduceFactor DAG shortening** (N4 in IHES synthesis). Build solution
    DAG, Dijkstra over windows for globally-optimal combined shortcuts. 1 day code.

15. [✓ DONE 2026-04-29] **B1 — Symmetry derivation (v3)**. Found **360**
    sticker permutations forming the group **A_5 × C_6** (60 geometric
    icosahedral rotations × an extra C_6 of non-geometric piece-twist
    automorphisms that preserve the generator set under conjugation).
    `data/rotations.npy` (360, 120) int8. Code: `scripts/19_symmetry_v3.py`.
    Math verified: `scripts/20_test_sym_translation.py` (all 360 preserve
    solved state, conj maps valid, real path round-trips 16/16).
    - **History**: v1 (BFS-over-stickers) FAILED (0/60) — face-centers were
      thought to be in separate orbits, but in this representation NO sticker
      is fixed by all generators. v2 enumerated face permutations + tried
      to extend sticker maps but the multi-orbit seed-search was unprincipled.
      v3 uses backtracking + forward-checking on (sigma, P) jointly, plus
      the observation that the 120 stickers form 2 orbits of 60 each (so 2
      anchors fully determine P given sigma).
    - **Inference-time use (KEPT)**: `--sym-ensemble K` in `03_solve.py`
      provides path-translation via the conjugation map. Apply K random
      rotations to a puzzle, run beam on each, take min path. Untested at
      strat-5 scale (in flight).
    - **Training-time use (REJECTED)**: m31_rot_aug strat-5 50/51, mean
      95.76 — REGRESSES vs m05's 89.4. At 6M params, augmentation across
      360 orbit-equivalents dilutes signal rather than sharpening it.
      Don't retry rotation aug at this scale.

16. [→ COVERED] **E7 — Arbitrary-position insertion finder**. → see
    `tier1-tier2-tier3-merged.md` T2.1 (Insertion-Finder port). Try inserting correction
    subsequences at every position, not just the end. Cheap-ish IHES idea, untested.

17. [ ] **E8 — Move deletion + BFS repair**. Delete each move, use BFS-d5 to repair if
    cost <1. Needs BFS table.

18. [→ COVERED] **A5 — CEA loss** (cross-entropy admissibility) — → see
    `tier1-tier2-tier3-merged.md` T2.5 (Distributional V head, CEA variant).
    Penalize overestimating true distance. Pushes heuristic toward admissible.

19. [→ COVERED] **A6 — Pairwise / ranking loss**. → see
    `tier1-tier2-tier3-merged.md` T2.3 (Listwise rank loss training).
    Beam only cares about neighbor ordering, not absolute values.

## Training infrastructure (added 2026-04-24)

T0. [x] **Muon optimizer** (torch.optim.Muon, PyTorch 2.9+; Keller Jordan et al., Dec 2024).
   Newton-Schulz orthogonalizes momentum on 2D hidden matrices; AdamW on embeddings +
   biases + LN + head. Rationale: our MSE plateau at ~64 regardless of arch size may be
   optimizer-conditioning-limited. Implementation (`src/cayley/optimizers.py`,
   `TrainConfig.optimizer`): DONE 2026-04-24. Next: run **m09 sanity test** — same arch
   as m07, 1000 ep with Muon@lr=2e-2 + AdamW-aux@lr=2e-3; compare MSE vs m07's
   checkpoint at ep 999. If dips clearly below m07, run full 4000 ep (`m10_muon_4k`).
   If tracks m07, skip — bottleneck isn't the optimizer.



T1. [ ] **Generous early stopping.** Observed on m02 (plateau at MSE 66 from ~ep 500)
   and feared on Kaggle m03 (7h wall with no visible progress). Design:
   - Track rolling-mean loss over last N epochs (e.g. N=200)
   - If rolling_mean doesn't improve by `min_delta` (e.g. 0.1 MSE) for `patience` epochs
     (e.g. 500), stop. `patience` ≫ N so we're genuinely past convergence, not bouncing.
   - Save a separate `best_epoch_*.pt` when rolling_mean sets a new minimum — our
     "final" checkpoint may not be the actual best due to cosine LR decay.
   - Expose via `TrainConfig.early_stop_patience`, `early_stop_min_delta`,
     `early_stop_window` (0 = disabled, default).

T2. [ ] **Better scheduler for long runs.**
   - Current: `CosineAnnealingLR` with fixed `T_max=n_epochs`. Fine for known-length runs
     but drops LR to ~0 at the end, precluding warm-restart continuations.
   - Options:
     - `CosineAnnealingWarmRestarts` (T_0=500, T_mult=2) — cyclic cosine, resets LR at
       each cycle. Good for very long runs where you want occasional re-exploration.
     - `ReduceLROnPlateau(patience=200, factor=0.5)` — drops LR when loss stops improving.
       Pairs well with early stopping.
   - Add a `scheduler` field to `TrainConfig` (currently hardcoded cosine). Default `cosine`
     for backwards compat.

T3. [ ] **Optimizer variants for long runs.**
   - AdamW is already fine; Lion / Sophia might help on very-long runs but not worth
     experimenting with before infrastructure T1+T2 is in.
   - Gradient accumulation — effective larger batch without VRAM cost. Useful if we
     ever want batch 65k+ on a 16 GB card.

## LOW priority / defer

20. [→ COVERED] **A7 — Transformer (kodurd recipe)**. → see
    `tier1-tier2-tier3-merged.md` T3.1 (Transformer m29 retry on 4090).
    CayleyPy-RL paper: MLPs beat transformers on permutation Cayley graphs with n>15
    without hand features. m18 was rejected on Kaggle wall-time only; on 4090 it fits.

21. [ ] **A8 — Piece decomposition features** for dodecahedron. IHES E8 regressed on
    cube; untested with Bellman. Risky; high porting cost (~6h derivation).

22. [ ] **C6 — Q\* search algorithm** with the A4 Q-model. 129× faster expansions on
    small domains in the paper; unclear how it transfers to Megaminx. 2h after A4.

23. [ ] **C9 — Six-axis ensemble** (60 whole-puzzle rotations). Blocked on B1 symmetry
    derivation.

24. [→ COVERED] **B4 — Exact-label (BFS-sourced) training data mixin**. → see
    `to_do_shortlist.md` item 5 (Exact BFS-d6 anchoring inside Bellman target).
    Now using BFS-d6 (19.4M states) not d5; eliminates RW label noise in near-goal regime.

## Pending — non-priority

P1. [ ] **Multi-round pseudo-labeling / Noisy-Student-style refinement.** BirdCLEF 2025
   1st place got +0.058 LB from 4 rounds of self-distillation with PowerTransform
   sharpening. For us: train round-1 model → predict on random-walk states → sharpen
   predictions (γ=2 power) → use as targets for round-2 model → repeat 3-4× with
   StochasticDepth regularization. Direct distance-regression analog of their method.
   Requires committing real GPU time on speculative gain; defer until we've exhausted
   simpler levers (m05/m13/m14 first). Estimate ~3-5h per round on 4090.

## NEW ideas (2026-05-03 session)

- **L_upper one-sided overestimate penalty (m40)** — DESIGNED, QUEUED. Adds
  `lambda * mean(max(0, pred - walk_depth)^2)` to standard Bellman MSE. Penalizes
  pred > walk_depth (a provable upper bound on d(s)). Free constraint with one
  hyperparameter. Code committed (`src/cayley/bellman.py` `lambda_upper`),
  config (`m40_upper_penalty.yaml`), plan (`m40_plan.md`). Will launch when
  GPU slot frees. **If it fails at 50/51 / 89, abandon the broader 4-term
  composite-loss proposal too** (the rest is redundant/failed-equivalent).

- **3-stage LR ramp-down for Q-shortlister training** (from AnanasClassic/cayleypy-
  neighbour-model-training repo audit). Their qshort uses 1e-4 → 5e-5 → 2e-5 over
  4096 ep × 16 steps. Ours used single-stage. **Action only if** we ever retrain
  the Q-shortlister (e.g., against a new V like m37 if it passes acceptance).
  Expected ~0.5-1% recall improvement. Cheap to try (a config change).

- **Continue T1.6 SA on more pids** — top-200 SA gave -471 → -266 merged. Next
  could be top-300 or full-1001 SA. Cost scales linearly: top-200 = 36h L4, so
  full-1001 = ~180h = a week of GCP. Probably better wall-cost-ratio: parallelize
  across multiple GCP instances OR target only pids that didn't get rescued by
  TPU v19a/v19b (the new "long-tail" of the current 79,946 submission).

- **Multi-seed Bellman ensemble** — train 3 fresh V models with different seeds
  on Kaggle (using existing 30h/week quota). Each ~8h on P100. Per CayleyPy
  paper §multi-agent: ensemble breaks single-model ceiling at scale. Lower
  priority than m40 / m37 result, but **next-to-try if we run out of single-model
  recipe ideas**. ~24h Kaggle time total.

- **GCP T1.6 v3 with bigger pid coverage AND fresh base** — re-run the SA on
  top-300 longest pids of the 79,946 submission (not the 81,357 base T1.6 v2
  used). Different pids will be in the long-tail. Cost ~50h L4. Expected
  -200 to -400 in min-merge (on top of 79,946).

## NEW ideas (2026-05-04 / 2026-05-05 session)

- **T1.2 multi-agent ensemble (saved-for-last big bet, NOW PROMOTED)** — given
  the training-side track is empirically airtight closed across 44× param range,
  T1.2 is the highest-EV remaining lever. We have **9+ trained cluster-V models**
  (m05, m17, m26, m26b, m29, m38, m39a, m40, m42, possibly m44 + m45). Hard-tail-
  only ensemble (top-50 to top-100 longest pids) at ~17h on 4090 OR split across
  GCP. Min-merge per-pid: each member contributes 0-15 wins, cumulative ~-300 to
  -800 vs current best. CayleyPy paper validated mechanism. **Next session priority.**

- **T2.4 Schreier-Sims-Minkwitz proper port** — sympy `coset_factor` doesn't give
  generator-words; needs Minkwitz proper (Knuth-Bendix-style with Schreier tree).
  Effort: 3-5 days from scratch OR port from Kaggle Santa 2023 community kernels.
  Per-pid min-merge add (cannot regress). Expected -50 to -500. **Defer unless
  T1.2 multi-agent stalls.**

- **T1.3 v2 with full-beam first pass (instead of cheap beam)** — 8k×60 quick
  beam can't solve hard tail; using full beam for both passes loses the cheap-then-
  full mechanism. Could try a MIDDLE ground: 32k×80 first pass. Probably still
  marginal; T1.3 family essentially closed.

- **m45 transformer scaling test (m46)** — DECISION GATED on m45 result. If m45
  (49.6M) clusters, m46 transformer is the last orthogonal architectural test.
  ~3-4 day GCP commitment. See `m46_transformer_plan.md`.

- **Curated macros Phase 3 deepening** — Phase 2 yielded 45 macros at modest gain
  (-1.6 moves/pid on top-50). Phase 3 would scrape megaminx-specific algorithm
  databases (jPerm.net Megaminx tutorials, cubingdb.com, YDH method tables). Could
  yield 100-200 more macros. But total compute for full-1001 SA at curated rate
  is ~180h L4 ≈ 1 week of GCP. Per-cycle gain ~-200 to -500. Defer unless multi-
  agent T1.2 also disappoints.

- **Beam-frontier replay (DAgger-style training data augmentation)** — log
  states from real `03_solve.py` runs that beam DEEMED wrong (visited but
  pruned), mix into next Bellman epoch at ~25%. Different from solver-trace
  (which was REJECTED at 25% mixin in m43): frontier states are RW-style
  distribution (not path-concentrated), so should be compatible. ~80 lines code.
  Untried. Estimate: cluster-or-marginal-improvement.

## Rejected (with reasons — do not retry without new justification)

This list mirrors the JOURNAL in `EXPERIMENTS.md`. Quick reference:

- **m04 — Bellman-from-scratch** (Pearcatcher recipe, no RW pretraining). 15/21
  stratified-2 NISS-off vs m07's 18/21. Self-consistent Bellman target with no
  walk-depth grounding produces good d≤6 ordering (top-1 0.995) but doesn't
  generalize as a beam-search heuristic at our scale. **Walk-depth + AdamW remains
  the better foundation; Bellman as REFINEMENT (m05) wins, Bellman from scratch loses.**

- **m09–m12 — Muon LR sweep** at lr ∈ {0.005, 0.01, 0.02, 0.05}. All plateau at
  MSE ~64-67. m12 (lr=0.05) had the LOWEST training MSE in the sweep (59) but
  WORST stratified solve rate (9/21) — both at ep 99 (noise minimum) and ep 999
  (final). Muon converges 4× faster than AdamW but doesn't beat the AdamW plateau
  on this problem at this arch. **Don't tune Muon without changing arch first.**

- **m06 — Q-distillation from m07.** 10× faster inference confirmed (10.8 s/puzzle
  vs 111 s for V-head at beam 131k), but solve rate dropped to 20/51 — distillation
  lost ordering quality. **Q-distill from a noisy V-teacher inherits & amplifies
  noise.** Possibly retry with m05 as the (better) teacher → that's m14, currently
  training.

- **Validation-tool top-1 accuracy as ranker** (not just filter). m12 ep999 had
  top-1 0.998 but solved only 9/21. m04 had top-1 0.995 (higher than m07's 0.990)
  but solved fewer puzzles. **Top-1 separates "broken" from "OK" but doesn't rank
  good models. Stratified-5 solve count is the only trustworthy comparison.**

- **C4 — CayleyPy `beam_mode="iterated"` with `history_depth=10`** (REJECTED 2026-04-27,
  see EXPERIMENTS.md and HANDOFF.md §6). Lab benchmark on 3 puzzles: 410.1s wall vs
  ours 183.6s (+124%); paths +10 moves vs simple mode. Russian commenter docs explicitly
  warn "this slows down" — it's a quality lever, not a speed lever. m05's sharp
  Bellman heuristic doesn't need non-backtracking enforcement. Moved here from item 11
  in MEDIUM priority (which was stale).

- **C5 — Distance-adaptive beam width / beam_decay** (REJECTED 2026-04-27, see
  HANDOFF.md §6). Geometric narrowing per step (decay=0.99, floor=32k) on 4 puzzles:
  wall 167.2s (−18%) but paths 358 vs 346 baseline (+3.5%); pid 492 92→99 moves
  (+7). On full 1001 the +2-3.5% inflation would push 88,195 → ~91k Kaggle score.
  Speed-without-quality goal not met. Code stays under `--beam-decay` flag. Moved
  here from item 12 in MEDIUM priority (which was stale).

- **m34 — Soft-Bellman + Polyak EMA + solver-trace mixin** (REJECTED 2026-05-02,
  CATASTROPHIC). 0/51 strat-5 — V completely broken. Ablations isolated
  `softmin_temperature=0.5` as the killer (T not << V scale ~10). Polyak alone
  OK; mixin alone OK. **Lesson: T must be << V scale for soft-min approximation
  to recover hard min.** Don't try softmin without scaling T to V-magnitude.

- **m36 — Deeper-but-narrower [1024, 512]×6 ResBlocks** (REJECTED 2026-05-03).
  Strat-5 50/51 / mean ~91. Same cluster ceiling as every other Bellman variant.
  Depth doesn't help any more than width (m26b) did. Adds a data point: the
  cluster ceiling is robust across architecture variants of similar parameter count.

- **m38 — Bellman + listwise rank loss (ListNet) auxiliary** (REJECTED 2026-05-03).
  λ=0.5 with linear warmup. Strat-5 48/51 / mean ~99 — REGRESSES by -2 solves
  AND +10 mean vs m05. Listwise rank loss as a Bellman auxiliary doesn't help;
  if anything it disrupts the standard fixpoint. **Adds another data point to
  the cluster-ceiling-is-information-bound theory** (regularizers don't help
  when labels are noisy).

- **h18 hamming-filtered T1.6 SA macros** (REJECTED 2026-05-03). `commutator_table_h18.pkl`
  (2,520 macros, hamming distance = 18, "minimum disruption") replacing the unfiltered
  ~37K commutator pool in `27_path_sa.py`. Pre-kill signal -17 / 11 pids vs v1's
  -30 / 10 (tracking ~43% behind). Killed before completion due to GPU contention
  with active game; even at projected pace would have underperformed. **Conclusion:
  minimum-disruption macro filter doesn't help; diversity beats hamming-minimality
  for SA insertion.** Use the unfiltered `commutator_table.pkl` instead.

- **Parallel-SA-on-single-GPU with active game / heavy contender** (REJECTED 2026-05-03,
  infrastructure lesson). 4-way `27_path_sa.py` workers + DemonLord.exe on GPU =
  full CUDA queue domination by game. Even after game ended, 4-way contention made
  each beam call ~4× slower → no pid completion in reasonable time. **For SA-style
  work on shared GPU, prefer 1-2 workers max OR ensure no other CUDA process active.**
  Filed in operational lessons.

- **m37 — solver-trace PRIMARY training** (REJECTED 2026-05-04, CATASTROPHIC).
  Strat-5: 3/51, only easy bucket-0 solved. Pure solver-trace training is OOD on
  hard scrambles — solver paths concentrate on visited states, not the broader
  orbit. Random-walk Bellman is load-bearing for V quality. **Don't replace
  walk-depth with solver-trace; only mix in.**

- **m43 — Standard Bellman + 25% solver-trace MIXIN** (REJECTED 2026-05-04,
  CATASTROPHIC). Strat-5: 0/51. Even 25% mixin proportion is enough to break
  Bellman convergence. Solver-trace data is OOD vs random-walks at any non-trivial
  proportion. **BFS-d6 (d≤6, near-solved) mixin is compatible with RW; solver-trace
  (full distance distribution, path-concentrated) is NOT.**

- **m40 — Bellman + L_upper one-sided overestimate penalty (T2.5 v1)** (REJECTED
  2026-05-04). Strat-5: 48/51 / ~99. L_upper steady-state near zero (model
  converges to fixed point satisfying both MSE and clip-upper). Combined with
  m38's listwise null, **the 4-term composite-loss proposal (L_sup + L_lip +
  L_upper + L_anchor) is dead** — 2 of 4 terms tested individually, both null;
  others redundant.

- **m42 — Distributional V (QR-DQN, 32 quantiles)** (REJECTED 2026-05-04).
  Strat-5 median 48/51 / ~97; lower-q25 48/51 / ~95 (slightly better, distributional
  inference IS real but small). Both fail acceptance gate. Lower-q25 might be
  useful as ensemble member but not standalone breaker.

- **m44 — TINY 1.12M-param V** (REJECTED 2026-05-05). Strat-5: 47/51 / ~102.
  Cluster confirmation at small-scale boundary. Combined with m05 (6M), m26b
  (13M), m39a (11M), confirms cluster ceiling holds across **44× param range**.

- **T1.3 v1 — TTT (Akyurek-style)** (REJECTED 2026-05-04). 16/16 mean 93.44 vs
  m05 baseline 93.25 (+3 / +0.2%, within noise). Bellman-self-consistency on
  m05 itself = weak gradient signal because m05 is already locally consistent.

- **T1.3 v2 — TTT with self-distilled labels (DAGGER-style)** (REJECTED 2026-05-05).
  0/16 hard-tail solved — cheap first beam (8k×60) cannot solve buckets 7-10
  puzzles → no labels to fine-tune on → mechanism N/A. **T1.3 family closed:
  fundamental incompatibility with the regime where TTT would help most.**

- **T1.1 brute-force commutators (h18 filter)** (REJECTED 2026-05-03). 2,520
  hamming-18 macros worse than unfiltered ~37K pool in T1.6 SA. Diversity beats
  hamming-minimality for SA insertion.

- **T1.1 Phase 2 curated macros (vs brute-force A/B)** (KEPT MECHANISM, modest
  contribution only). 45 unique perms from 54-algorithm scrape. A vs B on top-50
  longest of 79,946: A wins (-79 / 24 wins) > B (-2 / 2 wins extrapolated).
  But absolute gain is small (-1.6 moves/pid) at 17h+ wall. Not the 70K-cracker
  the tier doc promised. Decision: don't deepen the multi-day scrape.

## Avoid (confirmed regressions or weak evidence)

- `n_back > 8` — IHES MSE 14.4 → 15.84 at n_back=40. Stick with n_back=1 (maybe
  sweep {2, 4, 8} cheaply in tier 3 if bored).
- `k_max >> 120` — IHES: k_max=45 vs 30 zero gain on 3×3 (diameter ~26). Megaminx
  diameter is larger but still ≲ 60; k_max=80 already covers the training-depth range.
- `1/k` curriculum weighting — IHES E7 regressed (-102 moves).
- Large sequential MLPs without residuals — IHES B2: `[5000,1000]` and `[2048,1024]×8`
  both worse than `[1024,256]×1`. Residual blocks matter more than raw width.
- 3+ step window shortcuts — IHES 2026-04-21: 0 gain over 2-step.
- Commutator library (depth-4/6 perm enumeration) — IHES 0 gain; real beam paths
  don't hit the commutator subset.
- Transformer as primary heuristic — CayleyPy-RL paper evidence.
- Diffusion / policy-gradient without tree search — no benchmarks above DeepCubeA.
- MuZero / deep RL — weeks of effort for uncertain gain.
- Public submissions min-merge — user policy: don't submit community-merged.
- Hybrid classical Megaminx solvers — no Kociemba-analog exists; best classical avg
  is ~89-95 moves (speedsolving forum) vs leader 80. ML already strictly better.

## DONE

- [x] **Port puzzle class, training loop, beam search** (m01, 2026-04-24). Duck-typed
  `Megaminx` class; shared `cayley.*` modules accept it unchanged.
- [x] **Same-face order-5 run reduction + adjacent inverse cancellation** (2026-04-24).
  Sample post-processed: 500,572 → 457,810 (-8.54%). See `megaminx/post_process.py`.
- [x] **Stratified sampling (k per 100-bucket) for smoke tests** (2026-04-24).
- [x] **Stratified eval default = 5/bucket (51 puzzles)** (2026-04-25). Earlier drift to 2/bucket (21) was noisy at the high-solve-rate end (m04 vs m07 differed by 3 solves out of 21 — within noise). 5/bucket gives ±3 noise floor on 51, statistically solid for ranking close models.
- [x] **m01** (fast k_max=40 200ep) — model works on easy puzzles, MSE 9.76.
- [x] **m02** (k_max=80 2000ep, same arch) — MSE 66; heuristic too noisy on medium/hard.
