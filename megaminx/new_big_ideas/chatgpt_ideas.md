Below is a research/brainstorm map. The main conclusion: your current process is reasonable, but it is only one point in a much larger design space:

```text
sample states → assign approximate / bootstrapped targets → train heuristic → solve with search
```

Your current version is:

```text
non-backtracking random walks from solved
→ train diffusion-distance model
→ Bellman refinement with frozen target net
→ wide neural beam search
```

This is close in spirit to DeepCubeA / Autodidactic Iteration: generate states by scrambling from solved, use shallow search + current network to produce value targets, then use search at inference. DeepCubeA explicitly generates training samples from the solved cube, backs up values through children, trains value/policy targets, and then uses MCTS for solving. ([ar5iv][1]) Your code does the same core thing but with scalar distance regression and beam search instead of value+policy and MCTS.

---

# 1. Why random walks are used at all

Random walks solve the **data generation problem**.

Exact distances are impossible at scale. BFS gives exact labels only near solved; beyond that, the state space explodes. Random walks let you cheaply generate unlimited states at nominal depths 1…`k_max`.

Your own code says the label is not true distance but a **diffusion-distance heuristic**: the state after `i` walk steps receives label `i`, which is an upper bound because the walk may revisit nearby states or contain cancellations.  A recent Rubik’s Cube paper uses similar language: random walks from solved create training data, and the average walk length can be interpreted as a “diffusion distance.” 

So random walks provide:

1. broad state coverage;
2. a rough depth curriculum;
3. cheap labels;
4. enough structure for the model to learn a representation;
5. a warm start for Bellman refinement.

They are not enough because the label is **path-generation depth**, not **shortest-path distance**. Bellman refinement tries to convert this rough diffusion heuristic into a locally self-consistent distance-like heuristic.

---

# 2. Variations of random walks

## 2.1 Increase `n_back`

Current setup: `n_back=1`, `k_max=80`.

Your generator already supports larger `n_back`: it bans the inverse of any of the last `n_back` actions, not just the immediately previous action. The comment says a previous Jan-2025 experiment reported a large improvement using `n_back=40`, although it can force repeat-heavy sequences when too many inverse pairs are banned. 

This is the first thing I would test.

Why it may help:

* `n_back=1` only prevents `U, -U`.
* It does not prevent `U, R, -U, -R`, commutator loops, or longer local cancellations.
* Larger `n_back` makes the generated path more “outward moving” in the Cayley graph.

Why it may hurt:

* Too-large `n_back` biases the action distribution.
* If many inverse pairs are effectively banned, sampling becomes unnatural.
* You may train on weird trajectories unlike beam-search states.

Experiment grid:

```text
n_back ∈ {1, 2, 4, 8, 16, 32, 40}
k_max  ∈ {60, 80, 100}
```

Do not evaluate by validation MSE only. Evaluate by downstream beam solve rate / path length.

---

## 2.2 Non-backtracking but with smarter local cycle avoidance

`n_back` bans inverse moves, but it does not detect that a short suffix returns to a previous state.

A stronger version:

```text
Maintain last M state hashes per walk.
Reject action if child hash appeared in the last M steps.
```

This is self-avoiding-walk-like sampling.

Pros:

* Fewer local loops.
* Better depth-label quality.
* More outward states at the same `k_max`.

Cons:

* More expensive.
* Biased away from natural random scrambles.
* Could overrepresent states that are hard to reach by realistic solve trajectories.

Use small `M`, e.g.:

```text
M ∈ {4, 8, 16}
```

This is more precise than `n_back=40`: it bans actual local revisits, not just inverse action patterns.

---

## 2.3 Depth-stratified random walks

Right now each walk emits every prefix from 1 to `k_max`. That creates many shallow examples and correlated examples from the same trajectory.

Alternative:

```text
Sample target depth k ~ distribution
Run exactly k steps
Emit only final state
```

Possible depth distributions:

```text
Uniform:             k ~ U[1, 80]
Hard-biased:          p(k) ∝ k
End-heavy:            p(k) ∝ exp(k / T)
Mixture:              30% shallow, 40% medium, 30% deep
Beam-relevant prior:  match observed solution-depth histogram
```

Why this may help:

* Less correlation between adjacent prefixes.
* More control over depth distribution.
* Easier to over-sample hard depths.

Why it may hurt:

* Prefixes provide useful curriculum.
* Shallow states are important anchors.

Best version: keep prefixes but reweight losses by depth bucket.

---

## 2.4 Random walks starting from BFS shell, not only solved

You already have exact BFS-d6 infrastructure and optional BFS-d6 mix-in in Bellman training. The Bellman code can mix random-walk states with exact BFS-d6 states, anchoring training near solved. 

A more aggressive version:

```text
sample state s from BFS-d6
run random walk of length k from s
label ≤ d_bfs(s) + k
```

This gives many states whose upper bound is slightly better than `k` from solved if you start from non-identity BFS states.

Even better:

```text
sample s from BFS-d6 at exact depth d
run outward-ish walk of length k
label = d + k upper bound
```

This broadens the near-goal manifold while preserving known ground-truth anchors.

---

## 2.5 Random walks backwards from real Kaggle states

Instead of only scrambling from solved, start from actual initial states and walk toward/around them.

For each Kaggle initial state `x`:

```text
take random walk from x for k steps
state = y
label target is unknown
but y is test-distribution-adjacent
```

How to use it:

* not for raw supervised walk-depth labels;
* use Bellman consistency only;
* use teacher model pseudo-labels;
* use contrastive/ranking losses: original `x` and neighboring states should have consistent local Bellman values.

This addresses a big distribution gap: random walks from solved may not match the states beam search sees around hard Kaggle instances.

---

## 2.6 Beam-policy-guided walks

Generate training states using the current solver/model, not uniform random moves.

Procedure:

```text
for each real or random scramble:
    run weak beam / greedy / stochastic beam for T steps
    collect states from beam frontier and near-misses
    train Bellman / ranking targets on those states
```

This is closer to DAgger / expert iteration: train on the states your policy actually visits.

Why likely useful:

* Beam search errors are distributional.
* The model needs to rank states on the beam frontier, not random states uniformly.
* Hard failures often come from a few bad local ordering decisions.

Risk:

* Feedback loop: model trains on its own blind spots and may reinforce them.
* Need diversity, entropy, or mixing with random walks.

I would mix:

```text
50% random-walk states
25% BFS / near-solved exact states
25% beam-frontier states
```

---

## 2.7 Stochastic beam as data generator

Run stochastic beam where candidate probability is:

```text
p(child) ∝ exp(-h(child) / T)
```

Collect trajectories that solve or nearly solve.

This is useful because deterministic top-B beam may collapse onto the same regions. Stochastic beam gives harder negatives and alternative solution routes. Your research notes already mention stochastic/sample-based beam as an inference alternative. 

For training, it may be more valuable than for inference.

---

## 2.8 Commutator / conjugation-biased walks

DeepCubeA observed that learned Rubik’s Cube solutions heavily used conjugation-like patterns and that these are structurally important for manipulating pieces while preserving others. ([ar5iv][1])

For Megaminx, random walks could be biased to include macro-patterns:

```text
A B A^-1 B^-1       # commutator
A B A^-1            # conjugation
A sequence A^-1     # local manipulation
```

Why this matters:

* Pure random walks may underrepresent useful algebraic motifs.
* Solving twisty puzzles often depends on structured local transformations.
* A model trained only on uniform moves may learn “distance” but not enough useful local algebra.

Use as a data augmentation mixture, not a replacement:

```text
80% ordinary non-backtracking walks
10% commutator-biased walks
10% conjugation / macro-biased walks
```

---

## 2.9 Generator-weighted random walks

Not all moves are equally useful under a given scramble distribution or search policy.

Instead of uniform action sampling:

```text
p(a) ≠ constant
```

Possible weights:

* learned from solution paths found so far;
* learned from beam frontier actions;
* favor moves that change more stickers;
* favor underrepresented faces;
* temperature-smoothed policy head if you add one.

Caution: if the final solver can use all moves uniformly, training should not collapse onto a narrow action distribution. Keep entropy high.

---

## 2.10 Metropolis / MCMC-style sampling by target difficulty

From physics/MCMC: instead of letting random walks drift naturally, sample states according to an energy function.

Define energy:

```text
E(s) = target desired difficulty mismatch
```

For example, accept/reject moves to keep model-predicted distance near a desired depth bucket:

```text
accept child with probability min(1, exp(-(E(child)-E(parent))/T))
```

Use this to create balanced datasets around difficult bands:

```text
predicted h(s) ≈ 20, 40, 60, 80
```

This is not for labels directly; it is for **state selection**. Then assign targets via Bellman, teacher, or search.

Risk: the model’s own prediction controls sampling, so early models can bias the dataset badly. Use after warm-start.

---

# 3. Alternatives to random walks

## 3.1 Exact BFS datasets near solved

You already have this in partial form. The code has `sample_from_bfs_table`, which samples exact `(state, distance)` pairs from a BFS table.  Bellman training also supports BFS-d6 exact-target mixing. 

This is the cleanest non-random-walk source.

Pros:

* exact labels;
* strong boundary condition;
* stabilizes Bellman;
* removes ambiguity near solved.

Cons:

* shallow only;
* can overfit to near-goal states;
* does not cover hard search-frontier states.

Recommendation: use BFS data not as a separate pretraining stage only, but continuously as an anchor during all refinement.

---

## 3.2 Pattern databases / abstract subproblem distances

Classic heuristic-search solvers for Rubik’s Cube and sliding-tile puzzles rely heavily on pattern databases: precomputed exact distances for abstractions/subgoals. Culberson & Schaeffer define a pattern database as a table of distances to subgoals; for 15-puzzle, pattern databases reduced searched nodes by a factor of 1038 compared with Manhattan distance. ([webdocs.cs.ualberta.ca][2]) Korf’s Rubik’s Cube work also emphasizes that more memory allows larger pattern databases, larger heuristic values, and faster IDA* search. ([cs.princeton.edu][3])

For Megaminx:

```text
Select subset of stickers/pieces.
Project full state → abstract state.
BFS abstract state space.
Use abstract distance as lower bound / feature / target.
```

Ways to use PDBs:

1. Direct heuristic:

   ```text
   h(s) = max(PDB_1(s), PDB_2(s), ...)
   ```

2. Neural input feature:

   ```text
   model input includes PDB distances
   ```

3. Target lower-bound regularizer:

   ```text
   loss += penalty if model(s) < PDB(s)
   ```

4. Search pruning:

   ```text
   if g + PDB(s) > threshold, prune
   ```

This is probably the strongest “classical AI” alternative to pure random-walk learning.

---

## 3.3 Perimeter search / meet-in-the-middle datasets

Your MITM idea is already related to perimeter search: precompute a shell around the goal, then search forward until hitting it. Korf’s paper explicitly discusses perimeter search as related to bidirectional search. ([cs.princeton.edu][3]) Your research notes already describe MITM with BFS-d6 as high ROI and exact-quality: if a beam state enters the BFS-d6 shell, splice the known optimal tail. 

Training alternative:

```text
Use the BFS shell not only for solving, but for data generation.
Train model to predict distance-to-shell + BFS tail distance.
```

This changes the target from:

```text
distance to solved
```

to:

```text
distance to any state in known perimeter shell
```

This may be easier because the goal set is millions of states instead of one identity state.

---

## 3.4 Kociemba-like / solver-derived trajectories

Your uploaded `data.py` has a `load_kociemba_walks` function for precomputed solution-derived states: replay solver solutions backward and use remaining path length as an upper-bound label.  For Megaminx you may not have Kociemba, but the general idea applies:

```text
collect any valid solution paths
replay backward from solved
state at t gets label remaining_length
```

Sources:

* your own beam solutions;
* MITM solutions;
* larger-beam offline solutions;
* human/domain algorithms if available;
* external solvers if allowed.

This gives labels from **solution-like trajectories**, not random scramble trajectories. They are still upper bounds, but much closer to useful policy behavior.

---

## 3.5 Bootstrap learning from easy instances

There is a directly relevant heuristic-learning line: Arfaee, Zilles, and Holte propose bootstrapping stronger heuristics from weaker ones. If the weak heuristic can solve easy instances, those solved instances become training data for a stronger heuristic; if it is too weak, they use random walks to create increasingly difficult instances. They tested this on sliding-tile puzzles, pancake puzzles, and blocks world, producing heuristics that solved random instances quickly with near-optimal solutions. ([AAAI Publications][4])

This maps almost perfectly to your setup:

```text
h0 = random-walk model
solve easy Megaminx instances
train h1 on solved trajectories
solve harder instances
train h2
...
```

This is broader than Bellman refinement. It uses actual solved instances, not only one-step bootstraps.

---

## 3.6 Go-Explore-style archive

Go-Explore’s core idea for hard exploration is: remember promising states, return to them, then explore further. ([arXiv][5]) For Megaminx, you have a deterministic simulator, so this is natural.

Build an archive of states:

```text
cell key = coarse abstraction of state
score = best known path length / lowest h / novelty / distance bucket
```

Then repeatedly:

```text
sample archived state
return to it by stored path
explore with random / beam / model-guided perturbations
add better states to archive
```

This can generate much better training data than fresh random walks from solved because it preserves promising frontiers.

---

## 3.7 Direct real-instance self-training

Since Kaggle test/validation initial states are known in the competition data, you can train specifically on states around those instances if rules allow.

Process:

```text
for each initial state:
    run current solver
    collect frontier states
    collect found paths / failed paths
    train heuristic or policy on those states
```

This is transductive optimization, not general puzzle learning. For Kaggle, that is often exactly what wins.

---

# 4. Variations of Bellman refinement

Your current Bellman target is:

```text
target(s) = clip(1 + min_a target_model(a(s)), 0, walk_depth)
```

with exact zero for solved children.  This is essentially value iteration with function approximation.

## 4.1 Multi-step Bellman target

DeepCubeA uses depth-1 BFS for target creation but explicitly says the process can be generalized to deeper searches. ([ar5iv][1])

Instead of:

```text
1 + min over 1-step children
```

use:

```text
d + min over depth-d descendants
```

For `d=2`:

```text
target(s) = 2 + min_{a,b} f_target(a_b(s))
```

Pros:

* Less myopic.
* Better handles cases where all immediate children look bad but a 2-step route is good.
* More like small local search.

Cons:

* Branching grows: 24² = 576 descendants per state.
* Need dedup and chunking.

Practical variant:

```text
sample top-k first moves using current model
expand only those to depth 2 or 3
```

---

## 4.2 Soft Bellman backup instead of hard min

Hard min is brittle: one underestimated child can drag the target down.

Use softmin:

```text
target(s) = 1 + softmin_a f(a(s))
```

where:

```text
softmin(x) = -T * logsumexp(-x / T)
```

As `T → 0`, this becomes min. At higher `T`, it averages over plausible good children.

Pros:

* More stable.
* Less sensitive to one bad child.
* Provides richer gradients.

Cons:

* Biases target upward/downward depending on temperature.
* Need tune `T`.

Good experiment:

```text
T ∈ {0.25, 0.5, 1.0, 2.0}
hard min baseline
```

---

## 4.3 Double-Bellman / clipped double target

Borrow from Double Q-learning / TD3:

Maintain two target models:

```text
target = 1 + min_a max(f1(a(s)), f2(a(s)))
```

or:

```text
target = 1 + min_a mean(f1(a(s)), f2(a(s)))
```

Why: hard min over noisy estimates causes systematic underestimation. If one child is accidentally predicted too low, it dominates. Double estimators reduce this.

This is particularly relevant because your Bellman target uses a `min` over 24 actions.

---

## 4.4 Conservative Bellman target

Current target can become too optimistic if the model finds fake low-value children.

Use:

```text
target = max(PDB_lower_bound(s), 1 + min child)
```

or:

```text
target = max(BFS_shell_lower_bound, Bellman_target)
```

Your research notes mention admissible clipping using BFS lower bounds. 

Even a weak lower bound helps prevent collapse.

---

## 4.5 Ranking Bellman, not only regression Bellman

Beam search needs ordering more than calibrated distance.

For each state `s`, compute children values:

```text
v_a = f_target(a(s))
```

Instead of only training:

```text
f(s) ≈ 1 + min_a v_a
```

also train a policy/ranking head:

```text
best_action = argmin_a v_a
```

Loss:

```text
MSE(value)
+ CE(policy, best_action)
+ pairwise ranking loss over children
```

DeepCubeA trains both value and policy targets from the child that maximizes value. ([ar5iv][1]) Your current scalar-only model throws away the action-ranking supervision produced during Bellman target computation.

This is one of the highest-value changes.

---

## 4.6 Q-value Bellman

Instead of `V(s)`, train:

```text
Q(s, a) ≈ 1 + min_b Q(apply(s,a), b)
```

Then beam expansion can score 24 children from one parent forward pass.

Your research notes already propose Q-distillation v2: distill child values from the Bellman-refined m05 teacher, use MSE + KL over child rankings, and potentially get ~10× model-forward reduction if quality holds. 

This is not only an inference optimization. It may be a better learning target because the model is directly trained on action ordering.

---

## 4.7 Bellman with solved-trajectory anchors

For any known solution path:

```text
s0 → s1 → ... → solved
```

you know an upper-bound target:

```text
target(s_t) ≤ remaining_length
```

Use this in Bellman:

```text
target = min(
    path_remaining_length,
    walk_depth,
    1 + min child target_model
)
```

This anchors Bellman to real paths and prevents target drift.

---

## 4.8 Prioritized sweeping / prioritized Bellman refinement

Classic RL emphasizes prioritized sweeping: spend backups on states where the value estimate changes most or where accuracy matters for likely future decisions. Sutton & Barto discuss focusing backups around states where the approximate value function most urgently needs to be accurate. ([Stanford University][6])

For your case:

```text
priority(s) = |f(s) - BellmanTarget(s)|
```

or:

```text
priority(s) = beam visitation count × Bellman error
```

Train more on:

* high Bellman residual states;
* states often seen in beam;
* states where child rankings are unstable;
* states near successful/failed frontier boundaries.

This is probably better than uniform random-walk state sampling during refinement.

---

## 4.9 Target-network update variations

Current target net refreshes every 10 epochs.  Alternatives:

```text
hard update every {1, 5, 10, 25} epochs
EMA target: θ_target ← τ θ + (1-τ) θ_target
ensemble target
delayed target update after loss plateau
```

EMA is often smoother than hard copies.

---

## 4.10 Huber / quantile / distributional targets

Current config uses MSE.  But targets are noisy upper bounds + bootstrapped estimates.

Alternatives:

* Huber loss: less sensitive to noisy inflated walk labels.
* Quantile regression: predict lower quantile of solution distance, not mean.
* Distributional value: predict histogram over distance buckets.

For beam search, the **lower tail** may matter more than the mean: “is there a route from here?” not “average random continuation quality.”

---

# 5. Alternatives to Bellman refinement

## 5.1 Autodidactic Iteration with value + policy

Full DeepCubeA-style ADI:

```text
generate scrambled states from solved
expand children / shallow BFS
target value = backed-up child value
target policy = best child action
train joint value+policy net
solve with MCTS or beam
```

DeepCubeA used this approach to solve 100% of randomly scrambled Rubik’s Cubes in its test setup, with median solution length 30. ([arXiv][7]) The paper also says the policy output reduces breadth and the value output reduces depth in MCTS. ([ar5iv][1])

Your current method is “ADI-lite”:

```text
value only
depth-1 Bellman
beam search
```

A natural upgrade is:

```text
value + policy + ranking loss
```

---

## 5.2 Expert iteration from your own solver

Run expensive solver offline:

```text
large beam / MITM / ensemble / stochastic restarts
```

Collect:

```text
(state, best next move, remaining path length)
```

Train a policy/value model.

Then solve with smaller beam.

This is likely better than pure Bellman once you have enough solved paths.

---

## 5.3 Imitation learning from best known solutions

If your goal is Kaggle path length, you do not necessarily need a globally accurate heuristic. You need to reproduce/shorten good solution paths for the given instances.

Dataset:

```text
all states along best known solution paths
label = next move
label = remaining length
```

Train:

```text
policy head
value head
```

Then use:

```text
policy-prior beam search
```

Score candidates by:

```text
score = value(child) - λ log policy(move | parent)
```

This can greatly reduce branching.

---

## 5.4 Direct policy-gradient RL

A recent paper argues for solving Rubik’s Cube without near-solved sampling, using policy gradients and cost-pattern predictions between states, tested on 2x2x2 with >99.4% solve rate. ([arXiv][8])

For Megaminx, I would not put this first. Sparse reward and long horizon make direct RL expensive. But a hybrid is interesting:

```text
pretrain from random walks / Bellman
fine-tune policy with successful solve rewards
```

Use ranked reward rather than absolute reward: Ranked Reward was proposed because AlphaZero-style self-play does not directly apply to single-player combinatorial optimization; it turns single-agent returns into relative performance against past attempts. ([InstaDeep][9])

For Kaggle:

```text
reward = improvement over current best path length for same puzzle
```

That is a natural ranked/self-play objective.

---

## 5.5 MCTS instead of Bellman/beam-only

DeepCubeA combines the trained network with MCTS. It uses network priors and values to guide tree expansion; after finding a solution, it runs BFS over the discovered search tree to remove cycles and shorten the path. ([ar5iv][1])

For your project:

* beam is simpler and GPU-friendly;
* MCTS may help hard cases by focusing non-uniform search budget;
* MCTS is less naturally batched but can exploit a policy head.

I would test MCTS only after adding a policy/Q head.

---

## 5.6 Classical PDB + IDA*

IDA* with pattern databases is the traditional optimal-search baseline. Korf’s Rubik’s Cube work shows how PDB memory trades directly into search speed. ([cs.princeton.edu][3]) Your README already lists A*/IDA* as an alternative but notes the memory issue and suggests BFS-d6 shell as a natural cutoff. 

For Kaggle, pure optimal IDA* is probably too slow. But hybrid IDA* is plausible:

```text
neural h + PDB lower bound + BFS shell termination
```

Use IDA* for a small subset of hard puzzles where beam gives long paths.

---

# 6. Alternatives to the whole process

## Alternative A: Classical search-first pipeline

```text
Build abstractions / PDBs
Use symmetry pruning
Use IDA* / perimeter search / MITM
Use neural model only as tie-breaker
```

Pros:

* More principled.
* Better path quality.
* Less target-noise dependence.

Cons:

* Engineering-heavy.
* Megaminx abstractions may be hard.
* Memory/time constraints.

Best use: hard cases only.

---

## Alternative B: Neural policy-first pipeline

```text
Generate solved paths from current solver
Train policy+value on solution trajectories
Run policy-guided beam / MCTS
Iterate
```

This shifts from “learn distance from random scrambles” to “learn how successful solvers move.”

Pros:

* Better aligned with Kaggle objective.
* Uses your own solver as teacher.
* Policy reduces branching.

Cons:

* Can imitate suboptimal paths.
* Needs solved trajectory corpus.
* Risk of overfitting to current solver style.

This is likely high ROI.

---

## Alternative C: Q-model pipeline

```text
Train Q(s, a) from Bellman-refined teacher
One forward pass scores all 24 moves
Use much wider beam or more restarts
```

Your notes already identify this: the first Q-distillation failed because the teacher was m07, the student was not strong enough, and MSE did not preserve ranking; v2 should use m05 teacher and ranking/KL loss. 

Pros:

* Potentially huge inference speedup.
* Directly matches beam’s child-ranking need.
* Allows larger beams within same wall time.

Cons:

* Quality risk.
* Ranking loss must be designed carefully.

This is the most attractive model-side alternative.

---

## Alternative D: Archive / Go-Explore pipeline

```text
Maintain archive of promising states
Explore from archived states
Collect successful routes and near-misses
Train model/policy
Repeat
```

Pros:

* Better exploration than random walks.
* Strong for sparse-reward deterministic environments.
* Naturally discovers hard/intermediate states.

Cons:

* Needs state abstraction for archive cells.
* More moving parts.

This is a good research direction if you want a nontrivial project, not just leaderboard improvement.

---

## Alternative E: Transductive Kaggle optimizer

```text
For each puzzle:
    run multiple solvers/settings
    collect best paths
    locally improve paths
    train/fine-tune on states from that puzzle bucket
```

This is less elegant but probably most Kaggle-effective.

Components:

* large beam;
* MITM BFS shell;
* stochastic beam restarts;
* path simplification;
* bidirectional search;
* model ensemble only for unresolved hard cases;
* per-puzzle solution replay distillation.

This optimizes the actual benchmark rather than a general Megaminx solver.

---

# 7. Most promising concrete experiments

## Tier 1: cheap, likely useful

### 1. Sweep `n_back`

Your own code comment strongly suggests this is worth revisiting. 

```text
n_back: 1, 4, 8, 16, 32, 40
k_max: 80
train short models
evaluate beam solve/path length
```

### 2. Add depth-weighted loss

Try:

```text
weight(k) = sqrt(k)
weight(k) = min(k / 20, 1)
weight(k) = bucket-balanced
```

Reason: current prefix emission overrepresents shallow/easy states.

### 3. BFS-d6 mix-in during Bellman

Your BellmanConfig already supports this.  Try:

```text
bfs_d6_fraction ∈ {0.05, 0.10, 0.20, 0.30}
```

This may stabilize Bellman and improve endgame ranking.

### 4. Soft Bellman backup

Replace hard min with softmin.

```text
T ∈ {0.25, 0.5, 1.0}
```

Evaluate path length, not MSE.

---

## Tier 2: medium effort, high upside

### 5. Add policy/ranking head

When computing Bellman targets, you already know:

```text
best_action = argmin child value
```

Train:

```text
value MSE/Huber
+ policy CE(best_action)
+ child-ranking KL/pairwise loss
```

This is directly inspired by DeepCubeA’s value+policy target construction. ([ar5iv][1])

### 6. Beam-frontier dataset

Run current m05 beam, collect:

```text
states expanded
top candidates
states on successful paths
states pruned but close
```

Fine-tune Bellman/ranking on this distribution.

This directly targets beam’s actual failure mode.

### 7. Solution-path replay dataset

Use best known solutions:

```text
state along path → remaining path length
state → next move
```

Train value+policy.

This is probably better than more random walks after you have many solved paths.

---

## Tier 3: larger but strategically important

### 8. Q-distillation v2

Use m05 as teacher, not m07. Use ranking/KL loss, not just MSE. Your research notes already diagnose the previous failure and propose this. 

### 9. PDB abstraction prototype

Pick a small subset of stickers/pieces.

Goal:

```text
Can we build a useful admissible lower bound cheaply?
```

Even weak PDB features may regularize Bellman and improve search pruning.

### 10. Bidirectional / perimeter search

Your notes already describe bidirectional beam and symmetry pruning.  Classical literature supports perimeter/bidirectional search as a major memory-search tradeoff in cube-like domains. ([cs.princeton.edu][3])

---

# 8. My strongest recommendation

I would not spend most effort on “better random walks” alone. Random walks are only the bootstrap distribution. The bigger gain is to make the later stages more aligned with search.

Priority order:

```text
1. n_back sweep + BFS-d6 Bellman mix-in
2. Add policy/ranking head from Bellman children
3. Fine-tune on beam-frontier states
4. Train from solution-path replay
5. Q-distillation v2 for fast child scoring
6. PDB / symmetry / bidirectional search as hard-case tools
```

The conceptual shift:

```text
Current:
learn approximate distance from random scrambles

Better:
learn to rank actions/states on the distribution produced by your solver
```

Random walks are good for representation. Bellman makes the representation more distance-like. But Kaggle performance will likely come from **search-aligned training data**: BFS anchors, beam frontiers, known solution paths, Q/ranking targets, and hard-case transductive refinement.

[1]: https://ar5iv.org/pdf/1805.07470 "[1805.07470] Solving the Rubik’s Cube Without Human Knowledge"
[2]: https://webdocs.cs.ualberta.ca/~jonathan/publications/ai_publications/compi.pdf "CompI.pdf"
[3]: https://www.cs.princeton.edu/courses/archive/fall06/cos402/papers/korfrubik.pdf "rubik.dvi"
[4]: https://ojs.aaai.org/index.php/SOCS/article/view/18159 "
		Bootstrap Learning of Heuristic Functions
							\| Proceedings of the International Symposium on Combinatorial Search
			"
[5]: https://arxiv.org/abs/1901.10995 "[1901.10995] Go-Explore: a New Approach for Hard-Exploration Problems"
[6]: https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf?utm_source=chatgpt.com "Reinforcement Learning: An Introduction"
[7]: https://arxiv.org/abs/1805.07470 "[1805.07470] Solving the Rubik's Cube Without Human Knowledge"
[8]: https://arxiv.org/html/2411.19583v1 "Solving Rubik’s Cube Without Tricky Sampling"
[9]: https://instadeep.com/research/paper/ranked-reward-enabling-self-play-reinforcement-learning-for-combinatorial-optimization/ "Ranked Reward: Enabling Self-Play Reinforcement Learning for Combinatorial Optimization | InstaDeep - Decision-Making AI For The Enterprise"
