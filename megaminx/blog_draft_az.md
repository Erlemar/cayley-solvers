# The AZ track: an AlphaZero-style model that broke our value ceiling

*(Megaminx side of the project. The IHES picture cube kept pure distance models; AZ is
where the megaminx model line ended up.)*

## Why we tried it

Our solver is a wide beam search guided by a learned value model V(state) ~ distance to
solved. By May 2026 that pipeline had a well-documented ceiling: every 6M-param pure-V
model we trained -- regardless of recipe tweaks -- landed at mean ~89 moves on our
51-puzzle stratified benchmark, and bigger trunks actively regressed. So we spent a
session deliberately trying training paradigms from the literature: Trajectory Balance
(GFlowNets), dataset distillation, admissibility-aware losses, and AlphaZero.

AlphaZero was attractive for two reasons:

1. **A policy head is a move shortlister.** Megaminx has 24 moves per state; if a policy
   can rank them, the beam only needs to expand the top few children -- a large
   wall-clock saving.
2. **A second training signal.** Joint policy+value training on a shared trunk might
   teach the trunk something pure distance regression cannot.

What survived from AlphaZero was the *architecture* (shared trunk, policy head + value
head, joint loss `CE(policy) + MSE(value)`), not the self-play loop. Our final recipe
uses expert imitation for the policy and Bellman bootstrapping for the value -- no MCTS.

## Iterations: three failures, one lesson, two successes

All variants share the model: a 6M ResMLP trunk with a 24-way policy head and a scalar
value head. What changed was the training data.

| Version | Policy targets | Value targets | Bench | Verdict |
|---|---|---|---|---|
| AZ v0 | reversed random walks ("synthetic self-play") | walk depth | 0/10 | value only calibrated near solved |
| AZ v1 | our best submission's paths (78K moves) | remaining path length | 0/10 | beam's off-path children never seen |
| AZ v2 | same paths | v1 + 23 sibling states per path state, labeled by a teacher V (1.87M states) | 0/10 | conflicting targets destabilize the value head |
| AZ v3 | same paths | full Bellman recipe (below) | 10/10, +8% path length | first working dual head; policy top-1 = 50.7% |
| AZ v4 | 3-way min-merge of best submissions | same as v3 | **51/51, mean 87.5** | the breakthrough |

The failures taught us the single most important lesson of the track: **the value head
must be trained with broad off-path coverage.** Solver paths cover a narrow tube through
the state space; beam search expands 24 children per candidate, 23 of which are off that
tube. A value head that has never seen sibling states cannot rank them (v0, v1). And you
cannot patch this by labeling siblings with a teacher's predictions (v2): path states
carry realized path length (an upper bound on distance) while siblings carry predicted
true distance, and the head cannot fit both target semantics at once.

AZ v3 fixed it by refusing to compromise on the value side: keep the *entire* proven
pure-V training recipe (random walks for coverage, Bellman bootstrapped targets, exact
short-distance mixins), and let the policy cross-entropy ride along on the shared trunk.

## v3 -> v4: better policy data, and stopping early

Two changes turned a working model into the best one:

1. **Cleaner policy targets.** v4's policy dataset is a per-puzzle min-merge of three
   strong submissions (76,304 total moves). Best-of-three paths are more consistent, so
   the policy signal is less ambiguous.
2. **Early stopping at epoch 24** -- the counter-intuitive part. Trained longer, the
   shared trunk starts memorizing the finite policy dataset, and value calibration pays
   for it:

| Epoch | Policy loss | Value loss | Policy top-1 |
|---:|---:|---:|---:|
| 0 | 3.20 | 0.096 | 4.5% |
| **24** | **2.36** | **0.110** | **28.5%** <- picked |
| 49 | 1.71 | 0.140 | 48.1% |
| 74 | 0.91 | 0.250 | 74.2% |
| 99 | 0.45 | 0.481 | 91.1% |

Beam benchmarks confirm epoch 24 is the sweet spot; by epoch 99 the model fails puzzles
it solved at 24. We had previously blamed the v3 gap on a "dual-head tax" (two heads
sharing trunk capacity). That was a misdiagnosis -- v3 had simply trained past its own
optimum. Rule of thumb we now use: once policy top-1 passes ~35-40%, the value head is
degrading.

The result: AZ v4 epoch 24 solves 51/51 on the stratified benchmark with **mean path
87.5**, vs 89.4 for the best pure-V model of the same size -- the first 6M model to break
the 89-move cluster with no inference-side tricks.

## Final architecture and the full training pipeline

The shipped model is the last stage of a 5-stage chain; each stage fixes a failure mode
of the previous one.

```
stage 1  "pretrain"    random-walk MSE distance regression        4000 ep  (from scratch)
stage 2  "curriculum"  mixed-length curriculum + EMA + early stop ~6k ep   (warm from 1)
stage 3  "bellman"     Bellman bootstrap + frontier + BFS-d6 mix  500 ep   (warm from 2)
stage 4  "bellman_dd"  + exact anchors at solved / depth-1        50 ep    (warm from 3)
stage 5  "az"          dual-head: policy CE + Bellman value       ~25 ep   (warm from 4)
                       -> export the VALUE head = the beam-search model
```

- **pretrain** learns "random-walk depth" -- an upper bound on true distance, but a
  usable starting landscape.
- **curriculum** mixes walk lengths to calibrate mid-depth states; EMA smooths.
- **bellman** replaces the walk-depth label with the self-consistent target
  `1 + min_a V(child)` (DeepCubeA-style), and mixes in 25% beam-frontier states (the
  distribution beam search actually visits) and 10% exact-distance states (all 19.3M
  states within 6 moves of solved).
- **bellman_dd** pins `V(solved)=0` and `V(depth-1)=1` with exact anchors in every batch
  (the bootstrap alone leaves V(solved) near 2 -- a constant bias that misranks the
  endgame).
- **az** adds the policy head. Joint loss `1.0*CE + 1.0*MSE`, policy CE on the merged
  expert paths, value recipe unchanged from stage 4, early stop ~epoch 24.

One property worth calling out: the finished V *saturates* -- predictions level off near
the puzzle's diameter (~29) instead of growing with scramble depth. Our later
architecture survey (7 encoder families) found this is the load-bearing property for
beam search, and AZ v4 has it.

## The punchline: we throw the policy head away

At inference we extract the value head and run it alone. The policy head -- the whole
original motivation -- does not pay as a shortlister: composing AZ v4's V with our
distilled move-shortlister *regressed* paths by ~240 moves on the benchmark (the
shortlister was calibrated to a different V's landscape), and even a re-distilled one
hurt on the hardest puzzles. V-only is the production configuration on GPU and TPU.

So the policy earned its keep purely as a *training-time* signal: cross-entropy on
strong solutions, through a shared trunk, made the value head better than any value-only
recipe we found. AZ v4's extracted V has been the production scorer ever since.

The full pipeline is reproducible on Kaggle: the public
[AZ4 trainer notebook](https://www.kaggle.com/code/artgor/cayleypy-az4-trainer-megaminx)
runs all 5 stages (~25-30h on a T4, auto-resuming across sessions), or reproduces just
stage 5 from a shipped stage-4 checkpoint in ~30 minutes.
