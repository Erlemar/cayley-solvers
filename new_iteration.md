# New Iteration: Train the Search, Not Just the Value

Date: 2026-05-27

This note captures the next conceptual direction for the Megaminx work: stop treating
the neural net as only a scalar distance predictor, and instead train or use it in ways
that directly help the beam find shorter verified paths.

The key constraint from the current shortlist still binds: new global scorer
architectures are effectively closed unless the mechanism is fundamentally different.
The best evidence says the 6M V/Q scorer family has hit a cluster ceiling. Therefore,
the next iteration should prefer inference-side, verification-gated mechanisms and
search-aware training data over another value-model ablation.

Current decision baseline:

- current submitted best: 75,200;
- current standalone best: 77,086;
- do not use old `m05` strat numbers as binding decision gates for new work.

## Thesis

The current approach is not wrong because `V(state) ~= distance` is useless. It is
wrong if we keep optimizing scalar value loss and expect beam search to improve.

Beam search needs:

- correct relative ordering among children;
- preservation of useful alternative futures;
- diversity across symmetries, seeds, directions, and path shapes;
- candidate paths that can be verified and min-merged;
- repair mechanisms for already-valid solutions.

So the next question should be:

> Can we train or generate things that improve verified path length, even if scalar V
> loss does not improve?

## Recommendation

Do not start with fully differentiable end-to-end beam search. It has bad credit
assignment, discrete top-k selection, sparse rewards, and long horizons.

Instead, use an outer-loop search-aware training process:

1. Run the current solver stack.
2. Log what the beam considered, kept, pruned, and later regretted.
3. Find verified shorter paths or shorter path segments.
4. Label candidate states/actions by whether they helped produce short valid solutions.
5. Train a model/head to make those useful candidates survive.
6. Re-run beam with the new scorer.
7. Repeat.

This is closer to Expert Iteration / AlphaZero-style improvement than pure supervised
V learning. DeepCubeA's autodidactic iteration is relevant too: bootstrap the solver
from self-generated states and learned search guidance instead of relying on human
examples. Policy-guided heuristic search is also relevant because it explicitly
combines heuristic and policy signals.

The important difference from earlier failed ranking probes is label quality. Do not
train "this child was on one recorded path." Train "this child helped the search
distribution produce a shorter verified solution."

## Track 1: Solution-Graph Path Repair

This is the safest outer harness because every accepted edge is verified and Dijkstra
over verified positive-cost edges cannot regress the chosen path.

Bridge compression by itself has already shown only small effect. Keep bridge search,
but treat it as one edge generator among several, not as the whole plan. The stronger
idea is to collect all verified local alternatives into a per-pid solution graph and
let the graph optimizer combine them.

### 1. Solution graph

Build one graph per pid.

Node:

```text
state_hash
exact state bytes / permutation
pid
source path id
prefix length
```

Nodes must be keyed by exact state, not by path index. If two different paths reach
the same state, merge them into one node. This is where cross-solution relinking
becomes powerful: a prefix from one path can connect to a suffix from another path
as soon as they meet the same state or a verified bridge connects them.

Edge:

```text
from_node
to_node
move_word
cost = len(move_word)
source = original_path | bridge | relink | tail_resolve | macro | diffusion
verified = true / false
```

Only `verified = true` and `cost > 0` edges are allowed into the final Dijkstra run.

Then run Dijkstra:

```text
shortest verified path from initial_state_node to solved_state_node
```

This gives non-regression automatically: the original solution path is already in
the graph as verified edges, and every extra accepted edge can only lower or preserve
the shortest-path cost.

Start with:

- original path edges from all known verified paths for the pid;
- tail-resolve edges;
- macro-insert / SA accepted edges;
- bridge / relink edges if they verify;
- diffusion-generated bridge edges if they verify.

The graph optimizer should become part of the path-repair harness, not a later
optional add-on.

### 2. Bridge compression as an edge generator

Given a valid solution path:

```text
S_0 -> S_1 -> ... -> S_i -> ... -> S_j -> ... -> solved
```

try to solve the bridge `S_i -> S_j` in fewer than `j - i` moves. If a shorter bridge
is found and the endpoint verifies, add it as a graph edge:

```text
S_i -> S_j, move_word = bridge, cost = len(bridge), source = bridge
```

Why it matters:

- tail-resolve only fixes suffixes;
- BFS window replacement is saturated;
- bridge compression can repair middle detours;
- accepted bridges become high-quality training data.
- even small standalone bridge wins may combine with other graph edges.

Pilot:

- top 50 or top 100 longest current-best pids;
- windows of length 20, 30, 40, 60;
- current production V/qshort/sym stack at small beam first;
- accept only verified shorter full paths.

Gate:

- any net min-merge improvement counts;
- scale only if the solution graph gains real moves after combining bridge edges with
  original, tail, macro, relink, or diffusion edges.

### 3. Cross-solution relinking as graph connectivity

For a pid with multiple valid paths, try:

```text
A[:i] + bridge(A_i -> B_j) + B[j:]
```

This extracts value from path diversity that whole-path min-merge discards.

In graph terms, relinking adds a verified bridge edge from a node on path A to a node
on path B. Once that edge exists, Dijkstra can choose A's prefix, the relink edge, and
B's suffix if the combination is shorter.

Pilot:

- only pids with at least two distinct verified paths;
- only prefix/suffix pairs where a short bridge could beat the current best;
- reuse the bridge solver and verifier from Track 1.1.

Gate:

- verified shorter path for at least a handful of pids;
- if positive, convert accepted bridges into a training set.

## Track 2: Search-Aware Q / Ranking Training

This is the best version of "teach the neural net how to be useful for beam search."

The model should learn not only distance, but which child/action is worth preserving.

### Beam-utility head

The most promising concrete model is a beam-utility residual head, not another V.

Keep the existing V as the calibrated distance-ish signal, and add a head that predicts:

> Will this child be useful for the beam under the actual production solver
> configuration?

For a parent `s`, action `a`, child `s'`, depth `t`, and optional path context:

```text
score(s, a, s', t) =
    V(s')
  + lambda_pi * cumulative_policy_cost(path + a)
  + lambda_u  * U_theta(s, a, s', t)
  + lambda_d  * diversity_or_novelty_bonus
```

`U_theta` is not trained to approximate distance. It is trained to predict beam
usefulness.

This utility is config-specific. It depends on:

- beam width;
- qshort settings;
- symmetry/NISS configuration;
- depth;
- what other candidates are in the beam;
- whether the goal is standalone score or min-merge diversity.

That is acceptable. Train it for the production solver stack rather than pretending
it is a universal value function.

First implementation preference:

- freeze the production V trunk;
- add a small 24-action utility head on parent states, or a lightweight
  `(parent, action, child, depth)` scorer;
- train it as a residual reranker over existing V/qshort candidates;
- keep `lambda_u` small at first so it cannot destroy depth guidance.

Keep explicit diversity separate at first. Diversity is set-level: it depends on the
other candidates in the beam. A per-child utility head cannot fully learn that. Start
with:

```text
score = V + lambda_pi * policy + lambda_u * U
then apply explicit diversity buckets / novelty selection
```

Only fold diversity into the learned head after an explicit diversity rule wins.

### Label hierarchy

Do not use current-beam survival as truth by itself. That clones the current solver's
biases.

Use labels in this order:

1. Gold labels: child or bridge leads to a verified shorter continuation.
2. Strong labels: child appears in a stronger teacher search.
3. Weak labels: child survives the current beam.

Gold examples:

- tail-resolve wins;
- bridge-compression wins;
- cross-solution relinking wins;
- SA / macro-insert wins if the final path verifies shorter.

Strong examples:

- child selected by wider beam;
- child selected by sym-ensemble;
- child selected by multi-seed search;
- child selected by a longer rollout that beats current best;
- child lies on a known better min-merge path.

Weak examples are useful for diagnostics and weighting, but dangerous as direct
training truth.

The old frontier-regret result is the warning: many apparent "misranks" were benign
alternative optima. So labels must mean "helps produce a shorter verified result,"
not "matches this one path."

### Verified-regret Q reranker

Train on triples:

```text
(parent_state, good_child, bad_child)
```

where `good_child` is proven to lead to a shorter verified continuation than
`bad_child`.

Important lesson from the existing frontier-regret v0:

- apparent misranks are often benign alternative optima;
- do not train on "the path we recorded" versus "a sibling V liked better";
- train only on confirmed regret, where suffix_good is shorter than suffix_bad.

Loss:

```text
pairwise_loss = softplus(Q_good - Q_bad)
```

assuming lower Q is better.

Use pairwise/listwise loss within the same parent or local candidate set. Beam
selection is relative; the loss should also be relative.

Use different regularization depending on whether the production value model is frozen.

Frozen residual-head loss:

```text
loss =
    pairwise_or_listwise_utility_loss
  + lambda_reg    * ||U_theta||^2
  + lambda_center * mean(U_theta)^2
```

Shared-trunk loss:

```text
loss =
    pairwise_or_listwise_utility_loss
  + lambda_distill * mse(V_new, V_teacher)
  + lambda_var     * variance_guardrail
```

Scale guard:

```text
std(lambda_u * U_theta) should start much smaller than std(V)
```

This prevents the utility residual from overwhelming depth guidance. Increase
`lambda_u` only after stratified solve tests show the residual is helping rather
than merely perturbing the beam.

Deployment:

- as a reranker over qshort candidates;
- or as a small frozen-trunk action head;
- not as a full replacement for production V until it wins stratified tests.

Gate:

- training-side gate remains strict, but must be tied to the current production
  baseline, not old `m05` numbers: compare against the current solver stack and the
  current submitted / standalone baselines;
- if used only as an inference-side reranker with verification/min-merge, any net win
  is acceptable.

### Negative-sample rules

Negative examples need the same care as positives.

Good negatives:

- V-attractive children that fail to produce shorter continuations;
- bridge proposals that reach the wrong endpoint;
- bridge proposals that verify but are not shorter;
- children selected by the current beam but rejected by stronger search;
- diffusion samples that are invalid, longer, or duplicate known paths.

Avoid:

- arbitrary siblings not on the recorded path;
- current-beam-pruned states without stronger evidence;
- treating one solution path as unique truth.

This directly protects against the old frontier-regret failure mode, where many
apparent path-step misranks were benign alternative optima.

### Expected value and failure modes

Why this could work:

- distills expensive search into cheaper search;
- teaches a 65k beam to preserve branches a 524k/sym/multiseed teacher would preserve;
- preserves min-merge diversity that scalar V undervalues;
- trains directly on the survival decisions beam search actually makes.

Main failure modes:

- labels are self-confirming current-beam labels;
- positives are just one arbitrary solution path;
- `U_theta` learns path style instead of path quality;
- the utility head duplicates V and adds no new ranking signal;
- `lambda_u` is too large and destabilizes depth guidance;
- evaluation uses loss/AUC instead of solve/min-merge wins.

Therefore evaluate in this order:

1. teacher-good action recall on held-out hard pids;
2. strat-51 production recipe;
3. min-merge contribution against current best;
4. full deployment only if the min-merge signal survives.

## Track 3: Diffusion-Style Models

The AlphaFold 3 analogy is interesting, but it must be translated carefully.

AlphaFold 3 uses diffusion over continuous 3D atom coordinates. Megaminx is not a
continuous geometry problem; it is an exact discrete permutation and path problem.
A generated answer is useful only if the resulting move sequence verifies.

So a diffusion model should not be another global V scorer. It should be a candidate
generator or repair model whose outputs are verified.

## How Diffusion Fits This Iteration

Diffusion is not a separate strategy. It is a submodule inside the broader
search-aware iteration.

The combined loop should be:

```text
current solver stack
  -> verified paths
  -> bridge windows and cross-solution relink candidates
  -> diffusion / masked generator proposes alternate move sequences
  -> exact verifier rejects invalid or longer candidates
  -> shorter verified candidates improve the submission
  -> accepted and rejected candidates become training data
  -> Q/ranking model learns which choices are actually useful
```

This makes the division of labor clear:

- beam search and existing V/Q models produce strong baseline paths;
- bridge/relink code identifies local places where a shorter path might exist;
- diffusion samples diverse candidate repairs for those places;
- the verifier supplies truth;
- min-merge supplies non-regression;
- the verified-regret Q reranker learns from the accepted and rejected repairs.

In other words:

```text
new_iteration = the outer expert-iteration framework
diffusion = one candidate generator inside the verified repair loop
```

The diffusion model should feed Track 1 first, not replace Track 2. Once Track 1
produces verified wins and losses, Track 2 gets cleaner labels than the old
frontier-regret v0 labels.

### Why this fixes the frontier-regret label problem

The previous frontier-regret probe found many apparent misranks, but most were
benign alternative optima. A diffusion-backed bridge loop gives stricter labels:

```text
good = candidate bridge verifies and is shorter
bad  = candidate bridge is invalid, longer, or fails endpoint verification
```

That is much closer to the true objective than "this child was on the recorded path."
It lets the Q/ranking model learn from confirmed useful repairs rather than arbitrary
path-choice preferences.

### Correct sequencing

1. Build the bridge/relink harness first.
2. Run nonlearned bridge/relink pilots to establish whether windows are repairable.
3. Train a small masked bridge generator on verified path windows.
4. Use the generator to sample many candidate repairs on hard windows.
5. Accept only exact shorter repairs.
6. Train the Q/ranking model on verified good-vs-bad repair decisions.

This order keeps diffusion grounded. If the nonlearned bridge harness cannot find any
repairable windows, diffusion has little clean signal. If the harness finds even a
small number of wins, diffusion becomes a way to multiply that search.

### Bad version: global state denoising V replacement

Avoid:

```text
noisy permutation -> denoised lower-distance permutation
```

as a direct replacement for beam scoring.

Problems:

- generated states may not correspond to useful paths;
- mapping denoised states back to legal moves is hard;
- it risks becoming another saturated global scorer;
- no non-regression unless wrapped in expensive search.

### Better version A: bridge diffusion

Train a conditional discrete diffusion or masked-token model:

```text
input: source_state, target_state, noisy_bridge_tokens
output: cleaned bridge move sequence
```

Use it to propose candidate bridges for Track 1.

This is the most aligned diffusion idea because:

- source and target states are exact;
- the model generates a move sequence, not a vague heuristic;
- every proposal is verified by applying it to source_state;
- accepted bridges directly shorten existing solutions.

Training data:

- windows from verified paths;
- accepted bridge-compression wins;
- cross-solution relinking wins;
- tail-resolve and SA local-search improvements.

Corruption process:

- mask moves;
- insert random moves;
- delete moves;
- replace local chunks;
- optionally scramble order inside short blocks only if the verifier filters outputs.

Sampling:

- generate N candidate bridges per window;
- reject invalid bridges;
- keep only bridges shorter than the original segment;
- verify the full final path before accepting.

Pilot gate:

- on top 50 or top 100 long pids, produce at least one verified shorter bridge;
- scale only if it saves real moves beyond nonlearned bridge search.

### Better version B: path denoising / path repair

Train:

```text
noisy_or_suboptimal_path -> cleaner_path
```

conditioned on the initial state and maybe the solved target.

This can learn to remove detours, replace local chunks, and repair bad suffixes.

Risk:

- sequence lengths are long;
- exact validity is hard;
- output needs aggressive verification;
- it may duplicate what tail-resolve and bridge compression already do.

Use only after Track 1 creates a corpus of before/after path improvements.

### Better version C: move-policy diffusion

Train a denoising policy over future move sequences:

```text
state + noisy_future_moves -> plausible useful future_moves
```

This is closer to a generative policy than to AlphaFold-style structure prediction.
It may be useful as a diverse candidate generator for hard-tail pids.

Gate:

- generated candidates must beat the current solver on verified paths;
- otherwise it is just an expensive policy model.

## Why Diffusion Might Help

Diffusion-like generation is attractive because our search has a diversity problem.
Beam search collapses onto locally plausible futures, while diffusion can sample many
different globally plausible move sequences.

The useful role is therefore:

```text
generate diverse candidate repairs -> verify -> min-merge
```

not:

```text
replace V with a diffusion model
```

## Why Diffusion Might Fail

Main risks:

- legal move sequences are easy to generate, but useful shorter sequences are sparse;
- exact endpoint conditioning is hard;
- sampling may be too slow versus beam rescue;
- if trained only on current paths, it may imitate their detours;
- without bridge/relink wins, there may be too little positive "shortening" signal.

This is why bridge compression should come before diffusion. First find real repair
examples, then train a generative model to propose more of them.

## Concrete Next Batch

### Step 1: Build the solution-graph path-repair harness

Implement or harden:

- prefix-state cache for every verified path;
- exact state-keyed graph nodes;
- verified positive-cost graph edges;
- Dijkstra over merged state nodes;
- residual/bridge construction with unit tests;
- bridge candidate verifier;
- full-path verifier;
- per-pid min-merge reporting.

Deliverable:

- a script that builds a per-pid solution graph from known verified paths and accepted
  repair edges, runs Dijkstra, and emits a verified improved CSV if it finds wins.

### Step 2: Run nonlearned bridge and relinking pilots

Use the current production stack to search bridge and relink candidates before training
any new model. Treat these as graph-edge generators.

Deliverable:

- list of accepted bridge replacements;
- saved moves by pid/window length;
- failed-window statistics;
- training examples for later models.

### Step 3: Train verified-regret Q reranker

Only after confirmed shorter alternatives exist, harvest real regret triples.

Deliverable:

- small frozen-trunk beam-utility Q/rank head;
- strat-51 production evaluation;
- min-merge evaluation if deployed as inference-side reranker.

### Step 4: Diffusion bridge pilot

Train a small masked-token or discrete diffusion bridge generator from verified path
windows and accepted bridge wins.

Deliverable:

- N sampled bridge candidates per hard window;
- verified acceptance rate;
- move savings beyond nonlearned bridge search.

### Step 5: Feed verified repairs into Q/ranking

Use diffusion and bridge-search outputs to build confirmed preference pairs:

```text
(source_state, target_state, good_bridge, bad_bridge)
```

or local child triples extracted from the first divergent action:

```text
(parent_state, good_child, bad_child)
```

Deliverable:

- a cleaner verified-regret dataset than `frontier_regret_triples.pt`;
- a Q/ranking training run whose positives are proven shortening decisions, not
  merely recorded-path actions.

## Current Priority Ranking

1. Solution-graph path-repair harness.
2. Cross-solution relinking / bridge edges as graph connectivity.
3. Tail-resolve, macro, SA, and diffusion edges inside the graph.
4. Beam-utility residual head / verified-regret Q reranker.
5. Bridge diffusion / masked bridge generator.
6. Path denoising model.
7. Global diffusion-style solver.

The final item is intentionally last. It is intellectually fun, but the first five are
more likely to move the Kaggle score.

## Short Answer on the Diffusion Idea

Yes, a diffusion model is worth considering, but only as a verified candidate generator
for bridges or path repairs.

The most promising version is:

```text
condition on (source_state, target_state)
generate candidate bridge move sequences
verify exact endpoint
accept only if shorter
min-merge into the submission
```

That gives us AlphaFold-style generative diversity without giving up the hard verifier
that makes this project safe.
