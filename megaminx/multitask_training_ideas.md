# Multi-task & multi-stage training experiments — backlog

Ideas to explore *after* beam search speed optimization is done. Goal: push past
m05's plateau (final Bellman MSE ~0.094, leaderboard score 88,195 = rank #3 as of
2026-04-27) with smarter training signals. Companion to `speed_optimizations.md`
and `tensorrt_gcp_plan.md`.

## Current state (2026-04-27)

- **Speed track**: every speed optimization that doesn't reduce model forward time
  has been tested and either deployed or rejected. The 96% model_s ceiling makes
  TensorRT (FP16) the only remaining pure-speed lever (see `tensorrt_gcp_plan.md`).
- **Capacity track**: m26 / m26b (bigger / wider arch with walk-depth pretraining,
  Kaggle) are RUNNING. If either beats m05 at the strat-5 gate, integrate as the
  new teacher and retrain m23 student. If both tie m05 → confirms 6.0M-param
  ResMLP is the right capacity for this problem; pivot to multi-task work below.
- **Multi-task track (this doc)**: NOT STARTED. Highest-EV next step IF m26/m26b
  don't dominate, since simple capacity scaling will have plateaued.

## Why m05 might already be at the limit

- m07 (random walk targets) plateaus at MSE ~64 on walk-depth.
- m05 (Bellman warmstart from m07, 500 ep) reaches the **Bellman fixed point**:
  predicting `V(s) = 1 + min_a V(apply(s, a))` already.
- m17 (Bellman r2 from m05) confirmed: training loss flat from epoch 0
  (0.097 → 0.095 → 0.098). **More of the same Bellman signal does not help.**

So new improvements have to come from **different signal types**, not more of the
same.

## Three multi-task ideas with growth potential

### Option A — two-head architecture (V-walk + V-Bellman)

```
shared body → walk_head (1 output, MSE on walk_depth)
           → bellman_head (1 output, MSE on Bellman target)
```

- Both heads trained jointly. Body benefits from cleaner gradient on walk-depth
  + the tightening signal from Bellman.
- Use bellman_head at inference.
- **Why it could help**: walk-depth provides smooth, dense supervision on
  diverse states; Bellman tightens at the boundary. Body sees both.
- **Risk**: walk-depth permanently biases body toward upper-bound predictions.
  May not be a real win over m05's 2-stage approach.
- Effort: ~1 hour to write, ~1 hour to train (warmstart from m07).

### Option B — walk-depth + BFS-d6 exact targets (m29) ⭐

Mixed labels per training sample:
```
target(s) = walk_depth(s)     if s ∉ BFS-d6 shell  (cheap upper bound)
            true_BFS_distance  if s ∈ BFS-d6 shell  (exact ground truth)
```

- For shell states (where we KNOW the exact distance), use the exact value as
  the regression target instead of the noisy walk-depth or self-consistent
  Bellman estimate.
- 19.4M shell states (d ≤ 6). Sampling them per epoch is fast (<1s, table loaded).
- This is fundamentally different from Bellman — **no bootstrap circularity**,
  exact ground truth at the boundary.
- **Why it should help**: m05's Bellman is self-consistent but not anchored to
  true distance. Anchoring at the d ≤ 6 region gives the model a sharp,
  veridical signal it can extrapolate from.
- Risk: low — exact labels can only help, never hurt.
- Effort: ~2 hours code, ~30 min training (warmstart from m05).

### Option C — V + π multi-task (m24 expanded)

Train one model with TWO heads:
```
shared body → V_head (1 output, MSE on Bellman/walk-depth target)
           → π_head (24 outputs, cross-entropy on solved-path triplets)
```

- Beam scoring at inference: `score(s, a) = V(apply(s, a)) + λ · (-log π(a|s_parent))`
- Adds a NEW signal type (move probability from successful paths), not just a
  variant of value learning.
- Used in PHS / Levin Tree Search — proven to help heuristic search.
- **Why it could help**: solves the "value plateau" by adding an orthogonal
  signal. Disambiguates among same-V candidates.
- Effort: ~3 hours code (training + beam integration), ~1 hour training.
  We already have `11_train_policy_head.py` for the policy head alone — can
  fork into multi-task version.

### Combinations
- **Option B + C**: walk-depth + BFS-d6 exact V + policy head. Probably the
  strongest combination. ~6 hours total.

## What NOT to do

- **Pure walk-depth + Bellman MTL from epoch 0**: equivalent to 2-stage with
  smoother boundary; the Bellman bootstrap still requires warmup. No expected
  gain over current m05.
- **More Bellman rounds** (m17 told us r3+ is a no-op).
- **Bellman from scratch** (m04 told us it doesn't converge without warmstart).

## Cheap model evaluations (no beam search) — pre-screen before strat-5

Critical insight: full beam-search benchmark on 12 puzzles takes ~10-20 min;
strat-5 takes ~30 min. We need cheap signals to decide whether a model is
"promising enough" to spend that time on.

### Already have: `validate_model.py`
- MSE on walk-depth labels (training-style)
- Top-1 / top-3 ordering accuracy on BFS-true distances
- MSE on BFS-true distances
- **Caveat**: m12 had top-1 = 0.998 (best) but solved only 9/51 puzzles. Top-1
  is necessary but not sufficient.

### Better cheap proxies (build later)

#### 1. Pairwise child-ranking accuracy ⭐
For 4096 random parent states from BFS-d6:
- Generate 24 children, get their true BFS distances
- Score with model
- Q: does model rank child with smallest true distance first?
- This is THE quantity beam uses for top-B picks. Directly relevant.
- Cost: ~30s per model on 4090.

#### 2. Top-K child recall (mirror of Q-shortlister test)
For each parent: `P(model_top_K ⊇ true_top_K)`.
- Already have a version in `09_eval_q_recall.py`.
- Adapt for V-models.

#### 3. Greedy-rollout from sampled BFS states
For state `s` of known true distance d:
- Simulate beam at width 1: `state ← argmin_a model(apply(state, a))`
- Run until solved or step cap
- Compute ratio `d_greedy / d_true`
- Aggregate across 1024 states sampled at d ∈ {5..30}: gives "greedy efficiency"
- Strong proxy for beam quality. Cost: ~30s per model.

#### 4. Expected-progress sanity check
For state s: `mean_a[ V(apply(s,a)) - V(s) ]` should be ≈ -1.
- If positive: model is broken. Catches regressions.
- Cost: ~5s per model.

## Proposed eval pipeline for new models

```
new model trained
   │
   ├─ cheap eval (~30s):   pairwise ranking + greedy rollout
   │     │
   │     ├─ promising → continue
   │     └─ poor      → reject, don't waste 30 min on strat-5
   │
   ├─ strat-5 (51 pids, ~30 min):  full beam search on representative sample
   │     │
   │     ├─ +3 solves vs m05 → continue
   │     └─ tied/worse        → reject
   │
   └─ full 1001 solve (~hours, GCP $$):  candidate for production submission
```

Saves time by filtering bad models early. Especially valuable when running
multiple architectures (m26, m26b, multi-task variants, etc.) in parallel.

## Execution order (when GPU is free)

1. **Build cheap eval script** (`megaminx/scripts/13_cheap_eval.py`) — 1 hour
   - Pairwise ranking + greedy rollout on a fixed BFS-d6 sample
2. **Run cheap eval on m05, m07, m17, m21, m23 (Q-shortlister)** as baseline
3. **Option B** (walk-depth + BFS-d6 exact) — 30 min train + 30s eval
4. **Option C** (V + π) — 1 hour train + 30s eval
5. **Option B + C combined** — 1 hour train + 30s eval
6. Best candidate → strat-5 → full 1001

## Decision tree (post m26/m26b)

```
m26 + m26b strat-5 results
├─ either beats m05 by ≥+3 solves AND ≤0.95× mean model_avg
│    → integrate winner as new teacher
│    → retrain m23 student against new teacher
│    → re-deploy qshort solver, full 1001 → submit
│    → multi-task work below DEFERRED until plateau hits again
├─ both tie m05 (within ±2 solves, mean within 5% of m05)
│    → 6M-param ResMLP is right capacity; capacity scaling exhausted
│    → execute Options B and B+C from this doc next
│    → priority: B (BFS-d6 exact targets) — exact ground truth, lowest risk
└─ both worse than m05
    → revert; investigate hyperparam diffs
    → re-run m05 baseline with same recipe to confirm reproducibility
