# Brainstorm — alternatives to the random-walk + Bellman + beam pipeline

Companion to `RESEARCH.md`. That file ranks micro and meso optimizations of the
**existing** pipeline. This file zooms out and asks: what does the option space
look like if we allow ourselves to swap entire components, or change paradigms?

Scope: **four questions** from the original ask.
1. Variations of random walks (§1)
2. Alternatives to random walks (§2)
3. Variations of Bellman refinement (§3)
4. Alternatives to Bellman (§4)

Plus (§5) whole-process alternatives, (§6) a tier table, (§7) suggested 1-day
and 1-week experiments.

I've been honest about which ideas are speculative vs which have published
evidence. Some are obviously bad bets for our timeline; I included them anyway
because the brief was "brainstorm and analyze," and because thinking through
why something is wrong sharpens the case for what's right.

---

## §1. Variations on random walks

The current generator is `generate_walks_torch(n_back=1, k_max=80)`: a
non-backtracking simple random walk (NBRW) of length 80 starting from solved,
with the inverse of the previous action banned. Below: things to vary while
keeping the walk-from-solved structure.

### 1.1 Tune `n_back` — the cheapest experiment in this entire document

`n_back` is the length of the action history used to ban inverses. The codebase
comment explicitly flags that `n_back=40` was reported to produce a Kaggle
score of 10,281 in a Jan-2025 chat. NBRW theory backs this up: Alon, Benjamini,
Lubetzky and Sodin (2007, *Non-backtracking random walks mix faster*) prove
that on regular expanders an NBRW mixes up to **2× faster** than a simple
random walk, and the closer the graph is to Ramanujan the bigger the gap.
Megaminx's Cayley graph is regular (degree 24) and very expander-like at
moderate depths, so the larger the inverse ban, the more "ballistic" the walk
becomes — labels at step k correspond to genuinely-far states more often
instead of states the walk wandered back near.

The 24 generators form 12 inverse pairs, so `n_back=40` doesn't mean "ban 40
distinct moves" — most of the recent history bans collide with each other,
and the effective ban set is closer to 9–11 inverse pairs. The walk still has
~13–15 valid choices per step, plenty for diffusion.

**Smallest experiment**: re-train m05's Bellman recipe at `n_back ∈ {1, 5, 10,
20, 40}`, holding everything else fixed. Eval on strat-5. Picks the best.
Likely 1-day total on local 4090.

**Tier**: S. This should be the first knob touched.

### 1.2 Self-avoiding walks (full history ban)

Push n_back to the limit: ban *every* state visited so far in the current walk
(not just the last action's inverse). Self-avoiding walks (SAWs) on regular
graphs have been studied since the 1950s in the polymer context; key result is
that SAWs are *more* spread out per step than NBRW, with end-to-end distance
scaling closer to k^(3/4) on Z^2 vs k^(1/2) for SRW.

For Megaminx, the practical implementation: maintain a per-walk hash set of
visited states; resample if a candidate revisits. Given walk length 80 vs
state-space size ~10^68, collisions are vanishingly rare anyway (the walk is
already self-avoiding with high probability), so the marginal gain over high
`n_back` is small. The real value is for shorter walks where loops do happen.

**Honest take**: probably 1.0–1.05× on label tightness, not worth implementing
unless 1.1 doesn't deliver.

**Tier**: C.

### 1.3 Lévy walks / heavy-tailed step distribution

Instead of "one move per step," sample step **lengths** from a heavy-tailed
distribution and apply that many moves at once. With weight ∝ ℓ^(-α), the walk
has occasional macro-jumps that cover much more of the state space than a
diffusive walk of the same total move count.

Why it could matter: random walks oversample states the walk *recurs* near (a
recurrence bias inversely scaling with diameter). Heavy-tailed steps break
recurrence and produce a flatter empirical depth distribution → more
informative training pairs at large k.

Caveat: the *label* must reflect the actual word length applied (sum of step
lengths, not the number of step-events), or the model learns the wrong scale.

**Smallest experiment**: 1-day. Add a step-length sampler to
`generate_walks_torch`. Compare label-vs-true-distance histograms on the
BFS-d6 ground truth. If looser bound at large k, ship.

**Tier**: B.

### 1.4 Stratified / depth-flattened walks

Random walks place ≥ 70% of their mass in the second half of the walk (steps
40–80) because that's where most of the time accumulates. The walk *itself*
is uniform per step but the *resulting label distribution* is not — at k=80
you get N samples, at k=1 you get N samples, but k=80's samples cluster around
the same diffusion ball.

Fix: for each walk, sample a target depth k* uniformly from [1, 80] **before
running the walk**, and run only that long. Trains a uniform-over-depths
distribution, which is what the model needs for sharp ranking at all k.

Even better — Wang-Landau-style adaptive sampling (see §2.3): after each
batch, count how many examples we have at each depth, weight further sampling
inversely. Drives toward perfect flatness.

**Smallest experiment**: 0.5 days. Re-run m05 Bellman with depth-uniform walks.
Compare validation MAE per depth bucket.

**Tier**: A.

### 1.5 Pivot moves and cluster moves (statistical-physics MCMC)

In polymer Monte Carlo, the *pivot algorithm* (Madras-Sokal 1988) makes
non-local moves: pick a random midpoint, rotate the rest of the chain. For
Megaminx the analogue: take a current walk, pick a midpoint state s, replace
the suffix with a random walk from s. This recombines existing diversity into
new walks at low cost.

The Wolff cluster algorithm (1989) is the gold standard for ergodic sampling
on the Ising model — it flips correlated clusters of spins simultaneously,
breaking the critical slowing-down that local moves suffer from. The Megaminx
analogue would be flipping coordinated *blocks* of stickers (a face rotation
*is* such a block, so this is essentially what generators are already doing).
Useful as a theoretical lens, not directly importable.

**Tier**: D for direct port. Useful as inspiration for §2 ideas.

### 1.6 Bridge walks (random start → solved)

Reverse the framing: sample a *random* state (uniformly from the group, easy
via sequential random generators) as the start, then run a backward walk
ending at solved. The walk is biased toward solved by reweighting (Brownian
bridge construction) or by importance sampling.

Why interesting: random-walks-from-solved oversample near-solved states.
Bridge walks can be tuned to oversample states at any specified depth, e.g.
diameter-scale states that beam search struggles with.

This is essentially what Kociemba walks do (`load_kociemba_walks` in
`data.py`): use Kociemba to get a known path from a random scramble to
solved, then sample states along that path. Same family of idea.

The cleaner version when no oracle is available: take a random end-state from
the orbit of solved (= any group element via word generation) as the "fake
goal," run a walk back from it via the inverse generators, and re-label as
distance from real solved by reapplying the inverse path. This produces a
sample whose true distance to solved equals the walk length, regardless of
how that target was reached.

**Tier**: B. Already partially used via Kociemba walks.

### 1.7 Symmetry-aware walks (canonical orbit reps)

Megaminx has icosahedral rotational symmetry (order 60). Every state s has 60
"twins" R(s) for R in the symmetry group, and they have the same true
distance to solved. Random walks generate redundant samples within each
orbit.

If we canonicalize each walk-state to its orbit's lex-min representative
before adding to the dataset, we reduce dataset size 60× without losing
information, and the model sees 60× more *distinct* training examples per
epoch budget. This is RESEARCH.md §1.3 (symmetry pruning) applied to the data
side.

Same blocker: deriving the 60 rotation permutations rigorously. Two prior
attempts failed in the project. The construction-by-enumeration sketch in
RESEARCH.md §1.3 is the right path; once those 60 rotations exist they unlock
both data-side and search-side wins.

**Tier**: B (gated on §1.3 of RESEARCH.md being unblocked).

### 1.8 Adversarial / model-aware walks

Generate walks that **target the model's current weak spots**. After m05 is
trained, identify the depths and state regions where its prediction error is
worst (high MAE on a holdout BFS-d6 sample). Bias the next round of walks to
visit those regions more.

Implementation: after each training cycle, take 100k random states, score
them with the current model AND with the next-round Bellman target. The
states with the largest |target - model| disagreement are "informative." Run
walks that pass through them more frequently (e.g. start the walk near them,
or accept walks that visit them with higher probability).

This is an active-learning loop, and there's strong evidence it helps in
heuristic-learning settings (Chrestien et al. 2021, *Heuristic Search
Planning with Imitation, Attention and Curriculum*).

**Smallest experiment**: 2 days. Build the disagreement-mining infra, run
1 round.

**Tier**: A.

### 1.9 Quasi-random / Sobol walks

Replace `torch.randint` with a low-discrepancy sequence (Sobol, Halton).
Reduces clustering in the action sequence, marginal gain for diffusion-style
training, well-studied in MCMC. Probably 1.05× at most on a NBRW; not worth
the engineering.

**Tier**: D.

### 1.10 Two-sided walks

For each walk, record both the forward and backward views. Each pair (s,
walked-to-s', k) yields two training pairs: (s', k from s) AND (s', distance
from solved if s itself was at known distance from solved — which it is, =
0). Trivially: every state at walk step k is at distance ≤ k from solved.
This is what we already use. The "two-sided" twist is adding the *backward*
label: from s' back to s also takes k moves, so if we pretend s is the goal,
s' is at distance k from goal-of-s. This is goal-conditioned/HER (§4.5).

**Tier**: see §4.5.

---

## §2. Alternatives to random walks (data generation)

Different paradigms entirely, not parameterizations of the same paradigm.

### 2.1 Reverse BFS shells (the DeepCubeA framing)

DeepCubeA's data generation: scramble from solved with K random moves,
where K is sampled uniformly from [1, K_max]. Same as our random walks except
the depth is *labeled by scramble depth K, not by walk index*. Equivalent in
the limit, but the framing matters: every batch has labels at every depth
(uniform), not concentrated at large depth.

DeepCubeA + Approximate Value Iteration → ~60% optimality on 3×3×3, 100%
solve rate. EfficientCube extended this and showed the same reverse-scramble
data generation works for the 15-puzzle and Lights Out — supervised learning
on (state, scramble-depth) pairs is *all you need* (Takano 2021, *Self-supervision
is all you need for solving Rubik's Cube*) when paired with a wide enough
beam. For Megaminx we already do this (just framed differently).

**Already in pipeline**: yes (random walks ARE reverse-scramble walks). The
relevant variation is §1.4 (depth-uniformity).

### 2.2 BFS-leaves + extension walks (Bettker et al. 2024)

Bettker, Minini, Pereira, Ritt published *Understanding Sample Generation
Strategies for Learning Heuristic Functions in Classical Planning* (JAIR 2024,
arXiv:2211.13316) with an explicit comparison of sampling strategies. Their
main finding: **BFS expansion from goal up to some depth d, followed by
random walks starting from the BFS leaves, dominates pure random walks.**

The mechanism: BFS gives perfect labels in the near-goal region. Random walks
from BFS leaves extend coverage into the far region without re-traversing the
near-goal area. Compared to random-walks-only, this gives more uniform
coverage *and* tighter labels in the boundary region.

For us: we already have the BFS-d6 shell (19.4M states with exact distances).
The Bettker recipe suggests: instead of random walks from solved, do random
walks from each BFS-d6 *leaf*. Walk length k → label = 6 + k (for the leaf at
exactly d=6) or = exact_d(leaf) + k (for the partial-shell leaves at d<6).
The tightness of the boundary label is preserved.

We already use BFS-d6 as a target-mixin in `bellman.py`. The variation is to
also use it as the **starting distribution** for walks.

**Smallest experiment**: 1 day. Modify `generate_walks_torch` to start from
sampled BFS-d6 states with `start_depth = bfs_label`. Re-run m05 Bellman.

**Tier**: A.

### 2.3 Wang-Landau / flat-histogram sampling

Statistical physics method (Wang & Landau 2001) for sampling from energy
distributions when the energy spectrum spans many orders of magnitude. Idea:
maintain an estimate ĝ(E) of the density of states at energy E, sample states
with weight 1/ĝ(E), and update ĝ adaptively to push toward a flat histogram
in E.

For Megaminx: energy = walk depth (or equivalently, distance from solved
under our heuristic). Maintain ĝ(d) over depth bins [0, ..., diameter]. Each
round, sample walks (or modify walks via Metropolis steps) such that the
acceptance probability is ∝ 1/ĝ(d). After convergence, ĝ flattens and our
training data is uniformly distributed over depths.

This is §1.4 (stratified walks) on steroids — adaptive instead of fixed
target. Practical implementation needs Metropolis-style accept/reject on
walks, which is a small amount of new engineering.

There's a real literature on Wang-Landau for combinatorial state spaces; the
3D Ising spin-glass paper (Alder et al. 2004) is the canonical reference for
NP-hard regimes. Worth borrowing the algorithm even if we don't formally need
the density-of-states estimate.

**Smallest experiment**: 2-3 days. Implement WL-flavored sampler, train m20
with it, eval.

**Tier**: B. High variance: could be a 5% gain or a 30% gain.

### 2.4 Schreier-Sims-Minkwitz factorization

This is the **classical** way to solve permutation puzzles. Schreier-Sims
(1970) computes a base + strong generating set (BSGS) for a permutation
group; Minkwitz (1998) augments this to track *words* of bounded length
representing each coset rep. Combined: given any state s, output a word of
length O((diam) × something) that solves it. For Megaminx the words tend to
be 1500–3000 moves long — terrible compared to our beam search's 50–100 —
but each word is a **valid solution**, with no model needed.

How to use as data: run Schreier-Sims-Minkwitz on a few thousand random
states, get words. Each word is a (start, length-of-remainder) pair at every
prefix. These give *exact* upper-bound labels (length of remainder = exact
shortest path from this state under the SSM solver). Train on them.

The labels are loose (since SSM words are far from optimal), but they're
**exact for the SSM-induced path**, which means they're consistent with each
other — the model has a chance to learn the SSM solution structure and
shortcut it. There's a Kaggle notebook (Vicens Gaitan, Santa 2023) with a
working Minkwitz implementation; we'd port and run it once.

**Smallest experiment**: 3 days. Port Minkwitz from existing public code,
generate 100k labeled states, train a model on them only, see if it's
competitive with random-walk-trained.

**Tier**: B. Could surface as a strong "always-feasible" baseline even if
beam search is faster on average.

### 2.5 Adversarial / hard-state mining

Take the current production solver and run it on millions of synthetic
scrambles. The ones it FAILS on (or solves with abnormally long paths) are
exactly the states our heuristic is wrong about. Add them to training (with
labels = whatever the best beam search we have can find, or 1 + min_a f(a) at
those states).

This converges to a self-improving loop: model fails → identify failures →
augment data → retrain → model fails on harder states → ...

Same idea as DAgger in imitation learning, and it's how AlphaZero bootstraps
itself. Limit: requires the failure detection to be cheap (fast CPU solver,
or beam search with low budget that "almost" solves). For Megaminx, 1k beam
search in 5s/puzzle on the 4090 is fine.

**Smallest experiment**: 2 days. Run beam-2k on 100k random scrambles, take
the slowest-solved 1% as adversarial set. Add to training. Retrain.

**Tier**: A.

### 2.6 Self-play harvesting

Variant of 2.5: run the current solver on real scrambles, and for *every*
solve, harvest the (state, distance-remaining) pairs along the solution path.
These are real, exact-or-near-exact distance labels, *grounded in actual
search behavior*.

The trick: to be useful as training data, the path must be near-optimal,
otherwise the labels are loose. With m05 + beam 131k we get paths that are
optimal-ish on easy buckets and loose on hard buckets. Restrict harvesting to
puzzles where the path looks tight (e.g. compare to BFS-d6 lower bound where
applicable).

**Tier**: A. Cheap to run alongside production.

### 2.7 Targeted PDB-style coverage

Pattern databases (Culberson & Schaeffer 1998) precompute exact distances
within a *projection* of the state space — typically a coset space where you
ignore some pieces. For Megaminx, projections to consider:
- Just the corner stickers (60 stickers, much smaller subspace)
- Just the edge stickers (60 stickers)
- Just one "color set" of stickers

Each PDB gives admissible heuristic info for the full state. Combine via
max(corner_PDB, edge_PDB), the standard PDB recipe. This is the dominant
technique for the 15-puzzle (Korf 1985, Korf-Felner 2002).

For training data: each PDB lookup is an exact-distance label on its
projection, which trains the network to be admissible-on-projections. The
network can learn to combine projections.

**Tier**: B. Real investment (1–2 weeks to build PDBs + integrate). The
classical analog has been thoroughly engineered for cubes; less explored on
the dodecahedral side.

### 2.8 Symmetry-canonicalized sampling

Same as §1.7 but on a different axis: instead of canonicalizing within
walks, canonicalize at the orbit-rep level, only train on lex-min reps. 60×
distinct training examples per token of GPU compute.

**Tier**: B (same blocker as §1.7).

---

## §3. Variations on Bellman refinement

Current implementation: `y = clip(1 + min_a target(apply(s,a)), 0, walk_depth)`,
with target net refreshed every 10 epochs, MSE loss, optional BFS-d6 mixin.

### 3.1 N-step Bellman / TD(λ)

Instead of single-step bootstrap `y = 1 + min_a target(s')`, look n moves
ahead: `y = n + min over n-step paths target(s_n)`. With n=2,3 the bootstrap
sees more of the local geometry per training pair.

The full-blown version is TD(λ), which mixes 1-step, 2-step, ..., ∞-step
targets with exponentially decaying weights (1-λ)·λ^(k-1). Eligibility
traces. For value iteration on Cayley graphs the "k-step" math is nice
because there are exactly 24^k k-step children, all enumerable for small k.

n=2 means 24×24 = 576 lookups per state; chunk them. n=3 = 13824 per state,
gets memory-heavy but is once per training step.

**Why it might help**: 1-step Bellman propagates the solved-anchor by one
move per training round. Multi-step propagates faster, and crucially makes
the target function a contraction with a faster convergence rate (the
contraction coefficient is γ for 1-step, γ^n for n-step).

**Tier**: A. Real gain potential, real engineering.

### 3.2 Distributional Bellman (C51 / QR-DQN / IQN)

Bellemare-Dabney-Munos (2017) showed that predicting the **distribution** of
returns instead of the expected return gives significantly better learning,
especially when the return has multimodal structure. C51 parameterizes 51
fixed-support atoms; QR-DQN (Dabney et al. 2018) parameterizes quantiles;
IQN (Dabney et al. 2018b) parameterizes a continuous quantile function.

For our distance setting, the "distribution" is over possible distances given
state-uncertainty. A sharply-peaked distribution at d=N means the model is
confident; a wide one means it's not. Beam search can use this: instead of
top-B by mean distance, top-B by some quantile (e.g. 25th percentile) which
prefers states the model is *confident* are close.

This connects to the existing observation that "ranking is what matters,
not absolute distance" — distributional models expose ranking *uncertainty*
naturally.

QR-DQN with say 32 quantiles is a small architecture change: head goes from
1 output to 32. Bellman becomes per-quantile. Loss is quantile Huber.
Inference reads the median or any quantile.

**Tier**: B. Real research, but established literature and not too much code.

### 3.3 Soft / entropy-regularized Bellman

Replace the hard `min_a` with `softmin_a` = -T · log Σ_a exp(-target(a)/T).
Tunable temperature T. As T → 0, recovers the standard Bellman; as T → ∞,
becomes the average. Smoother targets, less brittle to occasional noisy
predictions in the min-arg.

Energy interpretation: the soft-min is the negative free energy of the local
neighborhood at temperature T. If our model is noisy, soft-Bellman doesn't
let one bad child poison the whole training signal.

**Smallest experiment**: 0.5 days. Add temperature parameter, sweep T ∈ {0.1,
0.5, 1.0, 2.0}, eval.

**Tier**: A. Trivially small change, possibly meaningful effect.

### 3.4 Prioritized / weighted Bellman backups

Schaul et al. (2016) — sample training examples in proportion to their TD
error, not uniformly. The high-error examples are the ones the model is
struggling with, which is where gradient signal is concentrated.

For us: maintain an importance score per state in our walk dataset. On each
round of training, sample with probability ∝ |y - model(s)|. As the model
catches up on easy regions, training automatically focuses on the hard ones.

Easy to implement: replace uniform shuffle in `_iterate_batches` with a
weighted sampler. The weights need maintenance (recompute after each epoch).

**Tier**: A. Standard RL trick, straightforward port.

### 3.5 Multi-target ensemble Bellman

Use K independent target nets (different random seeds, different past
checkpoints, etc.) and target = mean or max of their predictions. Reduces
target variance, which is a known source of Bellman instability.

**Tier**: B. K-fold compute cost on target net forward, only worth it if
target instability is actually hurting.

### 3.6 Symmetry-orbit Bellman

If we have the 60 icosahedral rotations: target(s) = average over the orbit:
`(1/|Orbit|) · Σ_R target(R(s))`. Equivariance enforced at the target level.
Reduces orbit-induced label noise in a single training run.

Or even simpler: target(s) := target(canonical(s)). Same effect, simpler
implementation.

**Tier**: B. Gated on §1.7.

### 3.7 Periodic vs Polyak target updates

Currently the target net is hard-copied every 10 epochs. The Polyak-averaging
alternative: `target := τ · model + (1-τ) · target` after every step, with
τ ≈ 0.005. Smoother target evolution, more stable training in DDPG and
descendants.

Trivial to implement. Worth A/B-testing.

**Tier**: A.

### 3.8 Munchausen Bellman

Vieillard et al. (2020) — augment the Bellman target with a log-policy
regularizer: `y = r + γ V(s') + α log π(a|s)`. Empirically robust improvement
over vanilla Q-learning across Atari. For value-only methods like ours, the
adaptation is to clip the target by an MSE-induced log-prior on past
predictions. Mostly a "stabilizer," ~5% gain.

**Tier**: C.

### 3.9 Bellman with pseudo-gradient consistency

Add an auxiliary loss: `L_consistency = Σ (target(s) - 1 - min_a target(s'))²`
on the model itself, not just the target net. Forces the model to be a
consistent value function during training, not just at refresh time.

Risk: this can collapse the model toward `target = 0 ∀ s`. Needs careful
weighting and probably the BFS-d6 anchor mixin to prevent collapse.

**Tier**: C.

### 3.10 Admissibility-constrained Bellman

Add a constraint that target(s) ≤ true_distance(s) wherever we know it
(BFS-d6 shell). Penalty: `λ · max(0, target(s) - bfs_d(s))²`. Keeps the
heuristic admissible-ish, which would unblock IDA* (RESEARCH.md §4.1).

Pamphlet (2009): admissibility is overrated for greedy search — beam doesn't
need it. But for IDA* and weighted A*, admissibility-or-near-admissibility
matters. If we ever switch to those algorithms, this Bellman variant is
required.

**Tier**: B (gated on switching to IDA*/A*).

---

## §4. Alternatives to Bellman

If we drop the value-iteration framing entirely:

### 4.1 Pure supervised regression on full BFS shells

Build BFS-d_k shells for as deep as feasible (we have d=6, ~19.4M states; d=7
would be ~250M; d=8 is ~5B which is intractable to materialize but tractable
to *sample*). Train regression on (state, exact_distance). Skip Bellman
entirely — it's only needed because we can't enumerate beyond d=6.

Cost: BFS-d6 to BFS-d7 is one more BFS pass over the d6 shell with hash
dedup. Memory: 250M × 120 bytes = ~30 GB, fits on disk.

Once we have d=7 exact labels and random walks for d>7, the walk-depth
overestimate problem is restricted to d>7 — a much smaller fraction of
training, with shorter overestimate window.

**Tier**: A. Big infra investment, big payoff. Building d7 is on the order
of 1-2 days of GCP if you can spare a 256 GB RAM machine.

### 4.2 Contrastive temporal distance learning

Recent work (Wang et al. 2024, *Learning Temporal Distances*) shows that
learning a distance via contrastive InfoNCE-style objectives produces a
*quasimetric* that beats Bellman-trained value functions on combinatorial
generalization. The training signal: for trajectory s_0, ..., s_T, the
positive pair is (s_t, s_{t+k}) labeled k, the negatives are random
non-trajectory states.

Quasimetric architectures (Wang & Isola 2022) hardcode the triangle
inequality: d(x,y) ≤ d(x,z) + d(z,y) by construction. This lets the model
*compose* distances correctly even at depths it didn't train on.

For us: replace `y = 1 + min_a target(s')` with a contrastive loss on
random-walk pairs at known offsets. Architecture change to enforce
quasimetric structure on the head.

**Tier**: B. Real research investment, but published evidence of advantage
on similar pathfinding problems.

### 4.3 Q-learning (predict per-action next-distance directly)

m06 was a Q-distillation attempt and lost ranking. RESEARCH.md §2.1 lays out
a fix path. The "alternative to Bellman" framing is: Q-learning trains
end-to-end on (s, a, s') triples instead of bootstrapping via min over
children. Each training example has 24 implicit labels (from the 24
generators) instead of 1.

The hybrid shortlist+rerank framing in RESEARCH.md §3.5 is exactly this —
use Q for the cheap shortlist, V for the careful rerank.

**Tier**: A (already in RESEARCH.md as the highest-ROI model-side bet).

### 4.4 Successor representation

Dayan (1993) — represent state s as the expected discounted occupancy vector
M(s, ·): for each state s', M(s, s') = expected discounted count of visits to
s' starting from s under some fixed policy. Plus the reward r(·): V(s) = M(s,
·) · r(·).

For shortest-path-to-solved: represent each state by its expected hitting
time of solved under the random-walk policy. The SR is an N×N matrix (N =
state space size, infeasible directly), but can be approximated by a neural
net f_SR(s) producing a feature vector φ(s), with V(s) = w·φ(s). The
features are *task-agnostic*; w is *task-specific*. Useful for transfer:
train SR once, use across many goals.

For us with one goal (solved), SR isn't a clean win — we already have V(s)
directly. But the SR formulation might enable goal-conditioning if we ever
want to solve "bring me to state X" not just "bring me to solved."

**Tier**: D for the current competition.

### 4.5 Hindsight Experience Replay (HER) / goal-conditioned

Andrychowicz et al. (2017): when training a goal-conditioned policy, every
trajectory generates valid samples for *any* goal it actually visited. For
us: every random walk from solved producing s_1, ..., s_k can be used as
training data for the goal "reach s_k from s_0=identity," in addition to
"reach s_0 from s_k."

For the value function, HER-style training: for each walk, sample target
goals from the walk's visited states, train V(s, g) = distance(s, g) on
those.

Why it might matter: trains the model on a richer set of (start, goal) pairs
than just the single goal=solved we need at test time. Tests on Megaminx
where every state has a "natural goal" structure (the symmetry group acts
transitively on states) might give us *implicit data augmentation by 60×* —
each state can stand in for any of its orbit reps as a goal.

**Tier**: C. Won't help directly for our single-goal task, but interesting
research direction.

### 4.6 Diffusion / score-based heuristic

Sun-Yang 2024 (DiffuSCO) and others train a diffusion model whose denoising
trajectory is biased by a combinatorial loss. The score function ∇ log p(x)
ends up being a heuristic-like signal pointing toward optimal solutions.

For shortest-path: train a diffusion model on optimal-path distributions
conditioned on (start, goal). At test time, sample from p(path | start, goal)
and pick the shortest sample. Computationally heavy but well-studied.

**Tier**: D. Far from our timeline.

### 4.7 GFlowNet for path generation

Bengio et al. 2021. Train a stochastic policy that samples paths with
probability ∝ exp(-length). Naturally produces diverse high-quality solutions
(Zhang et al. 2023, *Let the Flows Tell*).

Why it could be relevant: beam search is mode-seeking — it commits to one
high-confidence trajectory early. GFlowNet naturally produces *many* good
trajectories, and we can pick the shortest. On hard puzzles where beam gets
stuck, GFlowNet might find an alternate path beam couldn't see.

**Tier**: C. Real research investment; specialized for the problem class but
likely 2-3 weeks of work to even prototype.

### 4.8 Imitation learning from existing solver

Train the model to imitate beam-search-with-budget. For each state, the
solver provides a "best move" prediction; we train a policy to match it.
Equivalent to behavioral cloning of our own solver, useful as a fast policy
head that can prune candidates before the value head is queried.

Stacks well with §4.3 (Q-learning) — the Q-head IS this policy.

**Tier**: A.

### 4.9 Energy-based model with contrastive divergence

Train E_θ(s) such that solved is the unique minimum. Train via contrastive
divergence: pull energy down on observed states, push up on noise samples.
Standard Hopfield-like construction.

For our task, this is essentially what we're already doing (V(s) = energy,
solved = 0-energy). The CD framing might be more numerically stable than
Bellman bootstrapping, but no clear win.

**Tier**: D.

---

## §5. Whole-process alternatives

Replacing not just Bellman or random walks, but the entire model+search
sandwich.

### 5.1 Schreier-Sims-Minkwitz + path shortening

Pure classical: SSM gives a guaranteed solution in time polynomial in the
group size description. For Megaminx, words come out 1500–3000 moves. Then
post-shorten via local search (Lin-Kernighan-style: swap subwords, accept if
shorter and equivalent). State of the art classical methods get to ~80-150
moves for Megaminx.

This is a strong **always-feasible baseline**. Even if our beam search is
better on average, Minkwitz-shortened solutions are something we can fall
back on for the few puzzles where beam fails. Minkwitz is also fully CPU-bound
and doesn't compete with the GPU for resources, so it's almost free to run
in parallel.

**Tier**: A as a fallback. Run on every puzzle, take min(beam, minkwitz).

### 5.2 Megaminx-specific 2-phase Kociemba-style

Kociemba's algorithm for 3×3×3 works by decomposing the cube group as G ⊃ H
where H is a "nice" subgroup (R, L, U, D, F2, B2). Phase 1: get to H. Phase
2: solve within H. Each phase is its own short search.

Megaminx has analogous coset structures (Xu's MIT IMO paper covers some), but
nobody has published a clean 2-phase decomposition for it. The required work:
identify a useful subgroup (e.g. one where face orientations are fixed,
edge orientations are fixed, etc.), build the coset table, train two
heuristics — one for phase 1 (target = coset rep), one for phase 2 (target =
identity within H).

**Effort**: 2-4 weeks, math-heavy. **Payoff**: potentially huge — Kociemba
brought 3×3×3 average solve from 50+ moves to 18-22.

**Tier**: B. Project-defining bet. Probably won't fit competition timeline,
but worth seeding for follow-on work.

### 5.3 Bidirectional beam (already in RESEARCH.md §1.2)

Skip; covered there. Tier S there, agree.

### 5.4 IDA* with neural heuristic (DeepCubeA's actual algorithm)

DeepCubeA uses **weighted A***, not beam, with `f(s) = g(s) + λ · h(s)` where
λ ∈ [1, ∞) is a weight that biases toward h. With λ=1 you get classic A* (and
admissibility = optimality); with λ→∞ you get pure greedy best-first.
DeepCubeA finds 60% optimal at λ=1 on 3×3×3.

IDA* is the memory-efficient cousin: depth-bounded DFS with the depth
threshold raised iteratively. Much smaller memory footprint than beam.

Why we'd switch: IDA* at threshold k uses NO memory beyond the current path,
whereas beam holds B states per layer (131k × 120 bytes × 100 layers = 1.5
GB just for the beam tensor). On hard puzzles where beam runs out of memory,
IDA* keeps going.

**Tier**: B-A. Real switch, real win on the hardest puzzles. Might be the
right answer for the long tail of pid-492-type cases.

### 5.5 AlphaZero-style MCTS + policy + value

McAleer et al.'s original DeepCube (before DeepCubeA) used MCTS. Found that
on Rubik's MCTS produced longer solutions than weighted A*, abandoned for
A*. Konen 2023 retried MCTS+symmetry on Rubik 3×3×3 and got reasonable
results but still under DeepCubeA.

For Megaminx specifically there's no reason to expect MCTS to outperform
beam — beam is essentially MCTS with depth-1 lookahead and infinite parallel
threads.

**Tier**: D. Done research, demonstrably worse than the alternatives.

### 5.6 GFlowNet as solver (4.7 framed as the whole pipeline)

Train a GFlowNet to sample paths from any state to solved with prob ∝
exp(-len). Use it directly as the solver, no beam search.

**Tier**: C-D. Speculative; nothing in puzzle-solving literature points to
GFlowNet beating beam+heuristic in the regime we're in. Worth tracking, not
worth investing.

### 5.7 Genetic / evolutionary algorithm on paths

Borschbach 2010 implemented an evolutionary strategy for Thistlethwaite-style
phase 1 of Rubik 3×3×3. Population of candidate words, mutation = swap a
move, crossover = combine subwords, fitness = state distance from phase
goal.

For Megaminx: probably 2-3× worse than beam due to lower per-puzzle compute
budget. But interesting on pathological scrambles where beam fails — GA can
find solutions in regions beam never explores due to its early commits.

**Tier**: C.

### 5.8 Equivariant neural network over icosahedral symmetry

Cohen-Welling 2016 et seq. shows that hardcoding group equivariance in the
network architecture lets the model see 60× effective training examples per
gradient step. On Megaminx with the icosahedral group, this would mean: for
any rotation R, the network output satisfies f(R(s)) = f(s) (invariance) or
f(R(s)) = R(f(s)) (equivariance, depending on output type).

Implementation: replace the standard linear layers with G-CNN layers using
the discrete icosahedral group. Maurice Weiler's e2cnn library has the
infrastructure for cyclic and dihedral groups; icosahedral is a stretch but
the math is the same.

**Effort**: 1-2 weeks to retrain a model from scratch with the new arch.
**Payoff**: 60× more "effective examples" per gradient step → faster
convergence and possibly tighter heuristic. Stacks with everything.

**Tier**: B. Real engineering, real payoff. Same blocker as §1.7 (need the
60 rotations).

### 5.9 Pattern Database (PDB) heuristics — the classical king

For 15-puzzle and Rubik's 3×3×3, the state-of-the-art *for a long time* was
PDBs (Korf 1985, Korf-Felner 2002): exact-distance lookup tables on coset
projections, combined via max. PDBs are admissible by construction, fast to
look up (one hash table read), and parallelizable.

For Megaminx: the relevant projections might be (a) corners-only, (b)
edges-only, (c) one-color-only. Each has a much smaller state space (10^15
vs 10^68). Build BFS over each, store, look up at search time.

PDBs lose to neural heuristics when the state space is *very* large because
PDBs need exhaustive enumeration of the projection. For Megaminx the corner
subspace is still ~10^12, possibly tractable with disk-backed storage but
not trivial.

**Tier**: C. Mostly interesting as an admissible-heuristic source for IDA*.

### 5.10 Hybrid: neural beam + Minkwitz last-mile

When beam search makes it to depth ~6 or so without solving, switch to BFS-d6
table for the final stretch. We already do this with the MITM solver
(RESEARCH.md §1.1). The hybrid generalization: if MITM hash-misses but the
state is *close* to the shell, run a short bounded-depth BFS expansion on the
spot and rejoin the shell.

A more aggressive variant: if beam stagnates, fall back to Minkwitz on the
current best leaf, get a guaranteed solution from there, and report the
spliced path.

**Tier**: A. Strict improvement, no quality loss, modest engineering.

### 5.11 LLM as solver

GPT-4-class models can solve trivial scrambles via tokenized move prediction
but fail badly at depths > 10–15 unless given external tools. The training
data (random walks of moves) is what we already have, the architecture is
just a transformer over move tokens.

Two reasons to consider it: (a) transformer scaling laws are aggressive, and
(b) the m18 attempt at transformer architectures was reportedly infeasible at
the project's wall budget. BUT: a *small* transformer (10–50M params)
trained on (state, optimal_move_sequence) pairs for short scrambles, fine-
tuned with self-play on harder scrambles, might be a competitive one-shot
move predictor. Use it as an additional shortlist signal in beam.

**Tier**: D for in-competition. C as a research direction.

---

## §6. Tier table

Tiers reflect "expected ROI within this competition's timeline" — same scale
as RESEARCH.md.

| idea | section | tier | speedup/quality | quality risk | effort |
|---|---|---|---|---|---|
| `n_back` sweep | §1.1 | **S** | unknown, likely big | none | <1 day |
| Self-play harvesting | §2.6 | **A** | 1.1–1.3× | none | 1 day |
| BFS-leaves + extension walks | §2.2 | **A** | 1.05–1.2× | none | 1 day |
| Adversarial / hard-state mining | §1.8, §2.5 | **A** | 1.1–1.3× | none | 2 days |
| Stratified depth-uniform walks | §1.4 | **A** | 1.05–1.15× | none | 0.5 days |
| Soft Bellman (temperature) | §3.3 | **A** | 1.0–1.1× | low | 0.5 days |
| Polyak target updates | §3.7 | **A** | 1.0–1.1× | none | 0.5 days |
| Prioritized Bellman backups | §3.4 | **A** | 1.05–1.15× | low | 1 day |
| Q-learning hybrid (RESEARCH §2.1) | §4.3 | **A** | 5–10× model | medium | 1 week |
| Imitation cloning of solver | §4.8 | **A** | shortlist | low | 2 days |
| Hybrid neural+Minkwitz fallback | §5.10 | **A** | strictly ≥0 | none | 2 days |
| Build BFS-d7 partial shell | §4.1 | **A** | data quality | none | 2 days |
| Wang-Landau / flat-histogram | §2.3 | **B** | 1.05–1.3× | low | 2-3 days |
| Schreier-Sims-Minkwitz baseline | §2.4, §5.1 | **B** | quality fallback | none | 3 days |
| N-step Bellman | §3.1 | **B** | 1.1–1.4× | medium | 3 days |
| Distributional Bellman (QR-DQN) | §3.2 | **B** | 1.0–1.2× | medium | 1 week |
| Symmetry-canonical training | §1.7, §3.6 | **B** | up to 60× data efficiency | low | 1 week |
| Lévy walks | §1.3 | **B** | 1.05–1.15× | low | 1 day |
| Equivariant network | §5.8 | **B** | 60× effective data | medium | 2 weeks |
| Bridge walks via Kociemba | §1.6 | **B** | 1.05–1.15× | none | already partial |
| Megaminx 2-phase Kociemba | §5.2 | **B** | potentially 2-3× | medium | 4 weeks |
| IDA* with neural heuristic | §5.4 | **B** | 1.0–1.5× on hard | none | 2 weeks |
| Quasimetric + contrastive | §4.2 | **B** | unknown | high | 2 weeks |
| Genetic algorithm on paths | §5.7 | **C** | unclear | medium | 1 week |
| GFlowNet solver | §4.7, §5.6 | **C** | unclear | high | 3 weeks |
| Pattern databases | §5.9 | **C** | admissible heuristic | none | 2 weeks |
| Successor representation | §4.4 | **D** | n/a single goal | n/a | n/a |
| MCTS / AlphaZero | §5.5 | **D** | demonstrably ≤ beam | n/a | n/a |
| LLM solver | §5.11 | **D** | speculative | high | 3+ weeks |
| Diffusion heuristic | §4.6 | **D** | speculative | high | 3+ weeks |
| Self-avoiding walks | §1.2 | **C** | small over §1.1 | none | 1 day |
| Quasi-random walks | §1.9 | **D** | tiny | none | 1 day |

---

## §7. Suggested smallest experiments

### Day 1 (a single 4090 day, no GCP needed)

Three experiments, all stackable:

1. **`n_back` sweep**. Re-run m05 Bellman recipe at `n_back ∈ {1, 10, 40}`,
   500 epochs each. ~4-6 hours per run. Pick the winner. This is the
   highest-confidence single-experiment in this document.

2. **Stratified walks**. Modify the data generator to sample `target_depth`
   uniformly per walk (instead of always `k_max`). Re-run Bellman 500 epochs.
   Eval per-depth MAE on the BFS-d6 test split.

3. **Soft Bellman**. Add temperature parameter to `_bellman_targets`, sweep
   T ∈ {0.0 (current), 0.5, 1.0}.

Each is independently testable in <8 hours. If any clears the strat-5 quality
gate, ship to production.

### Week 1 (4090 + maybe a small GCP burst)

1. **BFS-d7 partial shell**. Extend the d6 shell to d7 (first-layer BFS). 250M
   states, ~30 GB on disk. Use as exact-label anchor for training. Same place
   in the pipeline as current d6 mixin, larger shell.

2. **Q-learning shortlist + V-rerank**. Train a small Q-distilled student
   from m05, plug into the beam loop as an α·B shortlist (RESEARCH.md §3.5).
   Validate recall vs the teacher. If recall ≥ 99% at α=4, integrate.

3. **Self-play harvesting + adversarial mining**. Run beam-2k on 100k random
   scrambles. Mine the top-1% slowest as adversarial set, the bottom-50% as
   self-play examples. Add both to next training cycle.

### Week 2-4 (longer-horizon bets, parallelizable)

4. **Schreier-Sims-Minkwitz fallback solver**. Port the Kaggle Santa 2023
   notebook implementation, run on all 1001 scrambles, take min(beam_path,
   minkwitz_path). Strict improvement, no quality risk.

5. **Megaminx symmetry derivation v3**. Use the construction-by-enumeration
   approach (RESEARCH.md §1.3) to derive the 60 icosahedral rotations.
   Validate by checking `R · gen · R⁻¹ ∈ generators` for all gens. Once this
   exists, unlock §1.7, §3.6, §5.8 simultaneously.

6. **n-step Bellman**. Implement n=2 lookahead in the target computation. ~3
   days work.

### Anti-recommendations (consistent with RESEARCH.md)

- **MCTS / AlphaZero**: published evidence is that beam beats MCTS on this
  problem class.
- **GFlowNet, diffusion, LLM**: 3+ week investments with speculative payoff.
  Nothing in the literature suggests these beat what we already have.
- **More Transformer experiments** (m18 was a no-op).
- **More Bellman rounds on m17** (already concluded: r2 was a no-op).

---

## §8. Honest reality-check

A lot of this brainstorm is speculative. The interventions with the highest
combination of (cheap, tested, likely-to-help) are concentrated in §1.1 (n_back),
§2.2 (BFS-leaves), §2.5–§2.6 (adversarial/self-play), §3.3 (soft Bellman),
§3.4 (prioritized backups), §3.7 (Polyak target), and §5.10 (Minkwitz fallback).
Ship those before the speculative ones.

The speculative ones with the highest *upside* are §3.1 (n-step Bellman),
§3.2 (distributional), §4.2 (contrastive quasimetric), and §5.8 (equivariant
network). Each is 1-2 weeks of work with 1.1×–1.5× expected payoff if it
lands.

The speculative ones with the highest *downside-protected* upside are §5.1
(Minkwitz baseline) and §4.1 (BFS-d7 shell) — both add bottom-line floor
without risk to the current best.

The right priority order, given Andrey's project context:
1. n_back sweep (today)
2. Soft Bellman + Polyak (today)
3. Prioritized backups + stratified walks (week 1)
4. Self-play harvesting + adversarial mining (week 1)
5. Q-learning shortlist (week 1-2)
6. BFS-d7 partial shell (week 2)
7. Minkwitz fallback (week 2)
8. n-step Bellman (week 3)
9. Symmetry derivation v3 + equivariant network (week 4+)

Total cost: <$50 GCP, ~2 weeks of focused 4090 time. Expected combined wall
+ quality improvement: 2-3× wall reduction and possibly 5-10% leaderboard
score reduction.
