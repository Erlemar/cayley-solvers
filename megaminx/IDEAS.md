# Megaminx experiment backlog

Ideas ranked by **expected payoff per hour of work**. Check off as done and point to
the EXPERIMENTS.md row. Based on a synthesis of:
- Our IHES experiments (EXPERIMENTS.md in project root, NEW_IDEAS_SYNTHESIS*.md, PROGRESS.md)
- Literature: DeepCubeA (2019), CayleyPy paper 2502.13266 (Feb 2025), CayleyPy-RL 2502.18663,
  EfficientCube TMLR 2023, Q* search 2102.04518
- Public Kaggle Megaminx kernels (data points, not ground truth — top votes reflect clarity
  not leaderboard rank; Kuznetsov/DrozdovDan/Rokicki haven't shared code)

**Current best (submitted)**: none yet. Local floor: **415,521** (pp + BFS-d5 on raw sample, 2026-04-24).
**Target (Rokicki, #3 LB)**: 93,606. **Target (Kuznetsov, #1)**: 79,971.

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

## HIGHEST priority (tier 1 — execute first)

1. [ ] **C1 — int8 state encoding** in khoruzhii beam search. 0 hours (already in code,
   just pass `state_dtype=torch.int8`). Unlocks beam 500K+ on 16 GB. IHES: -186 moves. **m02 + int8 re-solve first.**

2. [ ] **C2 — Adaptive beam per puzzle**. First pass beam 16k max_steps 60 (easy);
   retry unsolved at beam 65k max_steps 150. Code-only, 1 hour. Saves ~5× GPU time on
   easy puzzles, gives hard puzzles the budget they need. Unlocks full solves in ~1h
   instead of 8h.

3. [ ] **C7 — NISS (inverse-scramble search)**. Solve σ⁻¹, invert the path back.
   `Megaminx` already has `invert_state`/`invert_path`. 1 hour of code. IHES: -432
   moves in ensemble; directional anisotropy means the forward/inverse solves are
   genuinely different paths.

4. [ ] **A1 — Bigger arch m03 [2048,512]×2 k_max=80** (canonical recipe, not because it
   has a known LB but because DeepCubeA/CayleyPy papers use similar). **Running on Kaggle
   now as m03.** Expected -200 to -800 moves vs m02. First real data point for "how far
   does canonical ML take us."

5. [ ] **E3 + C3 — Build BFS-d5 Megaminx table** (~1.3M states, ~150MB). Use for both
   (a) MITM target set — every beam-search state that lands in d≤5 shell gets the
   optimal tail for free; (b) window post-processing (IHES: 30-80 moves/submission).
   3 hours to build + integrate. Extend to d6 (18M, ~2GB) if memory allows. **d7
   (250M, ~30GB) infeasible on 16GB laptop; punt unless we move to GCP.**

## HIGH priority (tier 2 — after tier 1 data lands)

6. [ ] **A3 — Bellman auxiliary loss** (m05) — warm-start from m03, add `y_bellman =
   1 + min_a V(apply(s,a))` as secondary target. IHES E6: -42 moves standalone. 2h code
   + Kaggle retrain.

7. [ ] **A2 — Bellman-from-scratch** (m04) — pearcatcher's recipe: no RW pretraining,
   `bfs_for_boundary=0`, discount 0.999, softmin, 1000 iterations. Untested claim in
   our stack; worth one run to test "is RW target necessary?". Kaggle parallel to m03.

8. [ ] **A4 — Q-function head** (m06). Distill m03 into a 24-output head predicting
   V(apply(s, a)) for each generator. One forward per beam step instead of 24. IHES:
   8× inference speedup → wider effective beam in same wall time. 2h code + Kaggle
   retrain.

9. [ ] **D1 — Multi-seed ensemble of m03**. 3 seeds in parallel on Kaggle, min-merge
   across solves. IHES: 3–5% per added model up to 3–5 seeds. Depends on m03 working.

10. [ ] **B3 — Negated-beam far-from-center data augmentation** (kieserel's trick).
    Run beam search with a NEGATED predictor to harvest high-distance states, feed
    them back as training examples. Addresses the hard-tail gap random walks don't
    cover. Estimated 3 hours code.

## MEDIUM priority (tier 3)

11. [ ] **C4 — CayleyPy `beam_mode="iterated"` with `history_depth=10`**. One
    alexandervc kernel uses this; cheap to try. 1h.

12. [ ] **C5 — Distance-adaptive beam width** (N5): wide beam early, narrow late.
    Saves compute without losing optimality. 2h.

13. [ ] **C10 — GCP L4 (24 GB) for beam 2^18–2^20**. Tail re-solve on unsolved
    hardest puzzles. $1–2 of compute per run.

14. [ ] **E6 — ReduceFactor DAG shortening** (N4 in IHES synthesis). Build solution
    DAG, Dijkstra over windows for globally-optimal combined shortcuts. 1 day code.

15. [ ] **B1 — Icosahedral symmetry augmentation (60×)**. Derive the 60 rotational
    symmetries of the dodecahedron acting on the 120-state. Augments training 60×.
    IHES's 24× version added +24 moves in ensemble; Megaminx's 60 might scale.
    4–8h derivation + training.

16. [ ] **E7 — Arbitrary-position insertion finder**. Try inserting correction
    subsequences at every position, not just the end. Cheap-ish IHES idea, untested.

17. [ ] **E8 — Move deletion + BFS repair**. Delete each move, use BFS-d5 to repair if
    cost <1. Needs BFS table.

18. [ ] **A5 — CEA loss** (cross-entropy admissibility) — penalize overestimating true
    distance. Pushes heuristic toward admissible. IHES-identified, untested. 4h.

19. [ ] **A6 — Pairwise / ranking loss**. Beam only cares about neighbor ordering, not
    absolute values. Triplet, listwise, BPR losses. IHES-identified, untested.

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

20. [ ] **A7 — Transformer (kodurd recipe)**. CayleyPy-RL paper: MLPs beat transformers
    on permutation Cayley graphs with n>15 without hand features. kodurd's recipe is
    plausibly good for diversification but not a free win. One experiment only if
    tier 2 plateaus.

21. [ ] **A8 — Piece decomposition features** for dodecahedron. IHES E8 regressed on
    cube; untested with Bellman. Risky; high porting cost (~6h derivation).

22. [ ] **C6 — Q\* search algorithm** with the A4 Q-model. 129× faster expansions on
    small domains in the paper; unclear how it transfers to Megaminx. 2h after A4.

23. [ ] **C9 — Six-axis ensemble** (60 whole-puzzle rotations). Blocked on B1 symmetry
    derivation.

24. [ ] **B4 — Exact-label (BFS-sourced) training data mixin**. 20% of each epoch from
    BFS-d5 with true labels. Eliminates RW label noise in near-goal regime.

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
