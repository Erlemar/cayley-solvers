# CUBE666 model research: updated analysis and execution plan

**Date:** 2026-08-24  
**Status:** consolidated plan after rereading the revised
`CUBE666_MODEL_RESEARCH_2026-08-24.md`, `deep-research-report.md`, the latest
`EXPERIMENTS.md`, and the current cube666 implementation notes.

## Executive decision

The next CUBE666 model should not be trained as a larger version of the current
full-state primitive value model, and it should not be asked to guide a beam from a
fully mixed state to the identity in one uninterrupted search.

The highest-probability scoring route is:

1. solve corners and parity exactly;
2. use the classical KMC rough phase to enter a bounded residual distribution;
3. run a neural residual beam on that distribution;
4. make every neural frontier state finishable by an exact, replay-verified completion
   oracle;
5. select the shortest completed path, then commute-reduce, locally rewrite and replay
   it independently.

The highest-ceiling research route is a six-rung orbit ladder whose joint rung models
are trained on the actual ladder-start distribution rather than generic walks from the
rung goal. The flat full-puzzle route remains a bounded calibration experiment: it is
worth reproducing the strongest published CayleyPy recipe, but only a large measured
increase in deep visibility can justify another enormous flat beam.

In short:

- **P3/P3' hybrid residual search is the primary path to a score below 166,421.**
- **P2 orbit ladder is the primary path with a plausible 130-150k ceiling.**
- **P1 flat search is a calibration/control lane, not the default scoring plan.**
- **110k is a near-optimal research target, not the expected result of one better
  network.**

## 1. Correct problem definition

This competition's CUBE666 is a supercube with 216 distinct sticker labels. The exact
group decomposition is nine invariant orbits of 24 positions. After exact corner solving
and parity normalization, the hard residual is represented losslessly by six even
24-position permutations: four centre clusters and two wing clusters.

This invalidates the colour-cube encoding in `deep-research-report.md`:

- the input is not `216 x 6` colour one-hot;
- `num_classes == 6` is not a valid invariant;
- colour-relabel symmetry is not the relevant state symmetry;
- a physical-piece model must preserve exact sticker identity and orientation;
- inversion is legal for the uniquely labelled permutation state.

The report's Q-vector economics, streamed beam design, matched-wall-time evaluation and
search diagnostics remain useful. Its colour-coset data analysis and proposed six-colour
input do not transfer.

## 2. What the evidence now says

### 2.1 Capability is not the main blocker

The factorized finisher model already solved 1,012/1,012 normalized competition states
with neural beam search. Its paths were approximately 700 primitive moves per state, so
it established autonomous capability but failed the scoring geometry.

The Schreier-transversal model independently solved unseen arbitrary projected states at
short high-level horizon, yet compiled to hundreds or thousands of primitive moves.
Again, clean labels and short horizon produced generalization; the chosen action cost did
not correspond to the competition objective.

These results reject the claim that CUBE666 simply needs a more capable or much larger
network. The model can execute difficult algebra when supervision and horizon are clean.

### 2.2 The flat primitive model has a blind zone

The current flat primitive model has useful child ranking at moderate depth but falls to
approximately chance by depth 85, while fully mixed states require roughly 105-110
moves. Wider beam then selects increasingly optimistic value errors rather than genuine
progress. Beam widths above roughly 2^21 did not repair this and produced frontiers with
fewer fixed stickers than random controls.

Define the diagnostic quantities:

- `M`: a mixing or information depth for the target distribution;
- `V`: deepest depth where child ranking is clearly above chance;
- `Z = start_distance - V`: the blind-zone estimate.

`Z` is an empirical diagnostic, not a theorem. The observed outcomes are nevertheless
consistent with large beams working only when `Z` is small. Every future model should
report visibility by depth, not a pooled action accuracy.

### 2.3 The labels fail before model capacity does

The experiments have rejected:

- fixed-path imitation as a discovery mechanism;
- self-Bellman targets from the current scalar value;
- local transition ranking without independent continuation returns;
- a larger path-context transformer trained on only hundreds of completion labels;
- a 52,904-option trajectory hierarchy that reproduces known paths but creates no new
  cheap paths;
- wider beam around the current optimistic scorer.

The missing supervision is counterfactual, off-policy, multi-step completion quality on
the state distribution actually visited by the deployed search.

### 2.4 Production completion is path-contextual

KMC's insertion finisher can insert and cancel moves inside the existing rough prefix.
Two different prefixes reaching the same 216-sticker state can therefore have different
completed path lengths. A production cost model over the cube state alone is not Markov.

The correct production search state is at least `(exact cube state, reduced prefix
context)`. Empty-context completion remains valuable for learning geometry, but it must
not be mixed with prefix-context labels as though the targets were identical.

## 3. Two useful diagnostics, with limits

### 3.1 Bits per move

Bits per primitive move is a useful common yardstick for phases, rungs and solvers. It
helps expose that the target score is qualitatively different from the current classical
regime.

It is not a rigorous additive accounting system:

- the nominal action count is 36, while an effective no-backtracking branching estimate
  may be closer to 29;
- group relations make `log2(branching)` only a local upper reference;
- information assigned to phases can overlap;
- a counting lower bound is not the true distance of a particular residual manifold.

Use bits/move as a comparative metric and abort signal, never as proof that a projected
score is attainable.

### 3.2 Completion-oracle returns

Let `O_c(p)` be a completion algorithm under a frozen contract `c`, including fixed beam,
annealing budget, seed set, frames, insertion semantics and post-processing. Define:

```text
J_c(p) = length of the independently replay-verified path returned by O_c(p)
```

For a candidate action `a`:

```text
Delta_c(p, a) = J_c(p) - J_c(p + a)
```

`Delta_c` is an exact measured difference for that frozen completion contract. It is not
the optimal action advantage and `J_c` is not the true distance; it is a verified upper
bound. This distinction matters, but the target is still far stronger than a value
bootstrapped from the model itself.

When the oracle contract changes, the dataset version must change. Rows from beam 1,000,
beam 20,000 and different min-of-seed policies may be used as separate fidelity fields,
not silently merged into one scalar target.

## 4. Unified production architecture

```text
full scramble
  -> exact corner solver
  -> exact parity normalization
  -> KMC rough-state generator
  -> neural residual/prefix beam
  -> exact completion of selected frontier prefixes
  -> shortest verified full path
  -> commuting reduction + radius-4 rewrite
  -> independent replay and strict min-merge
```

The neural search is anytime and safe:

- the empty neural prefix is always a candidate;
- every selected frontier prefix can be completed;
- a neural error may waste search, but cannot force a submission regression;
- the production file changes only through strict replay-verified improvements.

The terminal completion set should eventually be:

```text
O(p) = min(
    KMC insertion finisher from p,
    neural residual solver from p + exact tail,
    current incumbent/fallback path
)
```

P3 and P3' therefore compound. Improving the residual finisher improves both final paths
and the training labels used by the rough-prefix ranker.

## 5. Primary route: residual neural beam

### 5.1 No-training capability gate

Before training a residual model:

1. restore the deployed arm-A checkpoint;
2. extract KMC endpoints at approximately 10, 18 and 29 unrestricted three-cycle units;
3. use at least 30 independent endpoints in each band;
4. run the existing primitive Q beam with an exact shallow endgame;
5. also allow early neural stopping followed by exact completion;
6. compare complete end-to-end paths after seam reduction and replay.

This gate measures:

- whether the current model has signal on the residual manifold;
- the largest residual band on which it can improve a complete path;
- whether requiring identity is unnecessarily harder than finishable-prefix search;
- how much prefix-context cancellation changes the result.

It does **not** measure the true residual distance. The counting calculation is a lower
bound; successful neural and KMC paths are upper bounds.

### 5.2 Residual training distribution

If the no-training gate shows any useful band, generate training roots from:

- real KMC SA endpoints under several alpha and annealing settings;
- deeper KMC rough paths that leave smaller residuals;
- solved states perturbed by 5-30 verified 3-cycles or short commutators;
- frontier states retained and rejected by the current residual beam;
- exact shallow states and reusable endgame-table states.

Curriculum should be competence-based. Concentrate on bands where the current beam solves
roughly 20-80% or where completion regret is nontrivial. Do not order the curriculum by
random-walk length after mixing.

### 5.3 Residual supervision

Use four target classes:

1. exact distances and optimal actions inside the exact shell;
2. verified suffix upper bounds from complete paths;
3. comparable counterfactual completion returns `J_c(p+a)`;
4. independently verified multi-step search returns from stronger residual beams.

Beam-AVI is permitted only after the deployed residual model has measurable on-path
signal. Each iteration must retain exact anchors and independently replay the improved
paths. Model-only fitted Bellman minima remain forbidden.

### 5.4 Residual model comparison

Run a matched objective/data comparison:

- fast ResMLP-Q over all 36 primitive actions;
- unique-sticker/orbit-aware PieceTransformer-Q;
- optional two-stage macro proposer plus joint state-action reranker if primitive horizon
  remains too long.

Evaluate at matched accelerator-seconds. A slower transformer must justify its lower
beam width with better verified path quality, not merely higher action accuracy.

## 6. High-ceiling route: orbit ladder

The previous composition experiments do not conclusively close the orbit ladder. They
used either a sum of independent orbit heuristics or a joint model trained on walks from
the rung goal. Neither matches the deployed distribution where earlier clusters are
exactly solved and the next cluster is broadly scrambled.

### 6.1 Correct rung formulation

For rung `k`, train one joint scorer:

```text
D_k(s) = estimated cost to S_k
S_k = corners solved and clusters 1..k solved
```

The model sees the corners plus the first `k` cluster permutations. All primitive moves
remain legal during the rung; exact target membership is checked only at the boundary.
Never sum independent per-cluster values.

Use soft boundaries: preserve several states close to `S_k` and allow the next rung to
repair a one- or two-step boundary error if the joint continuation is cheaper.

### 6.2 Decisive two-rung experiment

1. Generate 200 ladder-start states with cluster 1 solved and cluster 2 independently
   scrambled subject to reachability.
2. Evaluate the existing joint 48-slot model without fine-tuning.
3. Run Beam-AVI on those exact start states and their beam frontiers.
4. Evaluate on at least 100 root-disjoint ladder-start states.
5. Record solve rate, primitive moves, bits/move, good-path survival and counterfactual
   frontier regret.

Decision:

- `>=90%` solved at roughly `<=25` moves: promote to rung 3;
- `60-90%`: allow one further targeted data/AVI iteration;
- `<60%` after the targeted iteration: close the ladder and concentrate on the residual
  hybrid.

At rung 3 and later, abort if a rung falls below approximately 2.9 comparative bits/move
or rises above roughly 27 primitive moves without a clear compensating downstream gain.
The final acceptance criterion is end-to-end path length, not the sum of isolated rung
estimates.

## 7. Calibration route: strongest flat recipe

Our 555 implementation is substantially weaker than the published CayleyPy result on
the same family of states. Therefore one bounded calibration is justified before calling
the entire flat learned-heuristic family closed.

Use a shared held-out depth bank at 40, 60, 72, 85, 95 and 105. Compare:

- the current arm-A ResMLP-Q control;
- the paper-style dense V recipe with BatchNorm and pre-mixing `K_max`;
- the same data/trunk family with a Q head where feasible;
- an orbit-aware transformer Q model trained on the same data.

Report:

- top-1 and top-k reducing-action recall by depth;
- sibling margins by depth;
- cross-parent calibration;
- ensemble/frame disagreement;
- descent `fix_mean` rather than minimum predicted Q;
- fixed-wall-time beam performance.

Promotion rules:

- `V <= 85`: close the flat route for this recipe family;
- `V = 90-94`: retain as residual/rung initialization only;
- `V >= 95` plus descent `fix_mean >= 9`: permit one large capability beam;
- no score-scale campaign until the capability beam solves independently.

## 8. Data engine and KMC instrumentation

### 8.1 Population export

Modify the Rust KMC solver to export the internal rough-search population rather than
only its winner. Each candidate record should contain:

- root ID and immutable root hash;
- symmetry frame and inversion flag;
- complete reduced prefix and prefix hash;
- endpoint 216-state hash and six-cluster representation;
- primitive prefix length and residual statistics;
- generator configuration, seed and annealing budget;
- cheap, medium and wide completion fields kept separately;
- final completed path hash, length and replay result;
- parent candidate and generation depth where available.

This amortizes rough-search generation, provides within-root counterfactual candidate
sets, and supports search-aware training.

### 8.2 Oracle-ceiling pilot

A fully wide completion of `100 x 256` candidates may be unnecessarily expensive. Use a
two-part pilot:

- **unbiased small pool:** 20 roots x 64 diverse candidates, all completed under the same
  wide contract;
- **scale pool:** 100 roots x 256 candidates completed cheaply, followed by uniform
  wide completion of the model/cheap top set plus randomly sampled and diversity-strata
  controls.

The small pool estimates the true candidate-pool ceiling without shortlist bias. The
scale pool measures cheap-to-wide recall and supplies active-learning data.

If the unbiased wide pool barely improves the matched KMC control, the action/prefix
generator is the bottleneck. Do not train a larger ranker. Expand the rough generator,
residual solver or macro grammar first.

### 8.3 Split and label hygiene

- Split by independent root scramble before candidate generation.
- Keep all descendants, frames and prefixes of a root in one split.
- Hash-check for state and prefix overlap after splitting.
- Never compare candidates completed under different budgets as if their labels were
  uniform.
- Store failures as censored observations with the explored budget, not arbitrary large
  scalar costs.
- Replay every labelled completed path independently.

## 9. Model representation and objective

### 9.1 State features

For each exact permutation element, include:

- cluster/orbit ID;
- current and target position;
- cycle ID, cycle length and position within cycle;
- misplaced indicator;
- exact unrestricted three-cycle counts;
- selected exact pattern-database values.

A five-labelled-piece projection has `24P5 = 5,100,480` states and can fit in roughly
5 MB with one-byte distances before indexing overhead. Build only a small set initially.
Use the maximum of admissible pattern values as a lower-bound diagnostic and the complete
vector as model features/auxiliary targets. Do not assume their sum is admissible without
explicit cost partitioning.

### 9.2 Geometry/context decomposition

Train separate but interacting components:

```text
V_geometry(s)      empty-context or append-only completion geometry
B_context(p, s)    insertion/cancellation bonus from the actual prefix
Q_joint(p, s, a)   candidate action return after exact state transition
```

The context component should receive the reduced primitive prefix, recent moves and
deterministic insertion-frontier descriptors. It should predict a residual correction,
not relearn the full cube geometry from a small production corpus.

### 9.3 Losses

Use:

- listwise within-root ranking by comparable verified completion cost;
- pairwise margins weighted by actual move differences;
- regression to measured upper-bound returns with fidelity masks;
- exact-shell distance and optimal-action auxiliaries;
- PDB/cycle auxiliary prediction;
- a beam-aware loss at the first layer where an oracle-good route is evicted;
- explicit hard negatives selected by the previous deployed model.

For predicted savings `S`, conservative deployment uses:

```text
S_safe = mean(S) - beta * std(S)
```

or, equivalently for cost, `mean(cost) + beta * std(cost)`. Sweep beta rather than
assuming uncertainty penalization is always beneficial.

## 10. Search design

### 10.1 Primitive residual beam

- one model forward per parent emits all primitive action Q values;
- prune immediate inverses and exact duplicates;
- stream candidate `(parent, action)` scores;
- materialize full child states only for survivors or a bounded shortlist;
- use exact goal/endgame tests;
- preserve sufficiently wide parent/backpointer fields and test power-of-two boundaries;
- periodically complete selected prefixes and update the verified incumbent.

### 10.2 Macro prefix beam

- propose structured, replay-verified short macros;
- preserve primitives and analytically useful macros in the candidate grammar;
- use a cheap proposer followed by a joint state/action/context reranker;
- deduplicate by exact resulting state plus relevant context, not state alone;
- evaluate selected nodes with the frozen completion oracle;
- rank unevaluated nodes using conservative predicted completion/savings.

### 10.3 Search-noise ablations

Test one at a time:

1. frame mean and frame maximum;
2. independent-model mean;
3. uncertainty-penalized mean;
4. child-row mean Q;
5. lineage-smoothed Q;
6. monotonic or beam-stack rescue on a small hard set.

The first five must pass a 500-step descent control with structural progress at least as
good as random (`fix_mean >= 9` in the current diagnostic) before any full solve. These
methods reduce noise exploitation; they do not create missing deep signal.

## 11. Mandatory diagnostics

Every model/search report must include:

- child top-1/top-k and regret by depth or residual band;
- good-path beam-survival waterfall;
- counterfactual completion regret of survivors versus pruned oracle-good states;
- predicted cost mean/median/tails by layer;
- uncertainty/disagreement by layer;
- fixed stickers and per-cluster residual progression;
- unique-state fraction and duplicate rate;
- action distribution and immediate-inverse rejection rate;
- nodes and unique states per accelerator-second;
- found, verified, used, strict-win and fallback counts;
- path length at fixed accelerator-seconds;
- independent full 216-sticker replay.

Do not promote on training loss, pooled correlation, minimum Q, or recovery of one known
teacher path.

## 12. Ordered execution plan

### Phase 0: correctness and immutable baselines

1. Freeze the current 171,019 submission and verification report.
2. Restore and hash the arm-A checkpoint.
3. Freeze root-disjoint depth, residual and ladder evaluation suites.
4. Verify move, inverse, frame, inversion, packing and independent replay tests.
5. Measure a 100-PID phase ledger rather than extrapolating from state 500 alone.

Deliverable: immutable manifest containing artifact hashes, PID/root lists, oracle
contracts and baseline metrics.

### Phase 1: decisive low-cost gates

Run in this order:

1. instrumented KMC population export;
2. unbiased oracle-ceiling pilot;
3. residual no-training gate at three residual bands;
4. two-cluster ladder-start baseline;
5. frame/row-mean/ensemble descent probes;
6. flat recipe visibility controls.

These gates determine which training work has evidence behind it.

### Phase 2: targeted training

Priority order conditional on Phase 1:

1. residual-distribution Beam-AVI/verified-return training if any residual band shows
   signal;
2. two-rung ladder Beam-AVI if the targeted baseline is nontrivial;
3. prefix-return proposer/ranker if the oracle pool contains substantial wins that cheap
   selection misses;
4. flat recipe continuation only if visibility materially exceeds arm A.

### Phase 3: composition

- residual route: KMC rough prefix + neural residual beam + best exact tail;
- ladder route: rung 1 -> rung 2 -> rung 3, with end-to-end replay after every boundary;
- prefix route: model rough-prefix beam using the improved terminal oracle;
- merge all strict verified wins with the 171,019 incumbent.

### Phase 4: scale and score

Only after clean gates:

- larger beam or more frames;
- 2-4 model conservative ensemble;
- broader macro grammar;
- rung orders and symmetry portfolio;
- full 1,012-PID campaign;
- commuting reduction, radius-4 rewriting, verification and submission.

## 13. Hard promotion and stop rules

| Component | Promote | Stop or redirect |
|---|---|---|
| Flat model | visibility >=95 and descent `fix_mean >=9` | visibility <=85 after the calibrated recipe |
| Residual gate | verified end-to-end wins in at least one residual band | no wins and near-chance action rank at every band |
| Residual training | better paths at fixed wall time on root-disjoint endpoints | training-only rank gains or optimistic completion regressions |
| Two-rung ladder | >=90% solved near <=25 moves | <60% after one targeted AVI iteration |
| Rung 3+ | credible end-to-end path and >=2.9 comparative bits/move | >27 moves/rung without downstream compensation |
| Prefix ranker | held-out regret reduction and fresh online strict wins | oracle pool has no meaningful wins, or learned proposals match random controls |
| Search ensemble | structural descent improves, then verified path improves | only minimum predicted Q improves |
| Larger model | scaling curve improves verified path at fixed accelerator time | better loss but worse or equal search economics |

All numeric gates are operational thresholds and may be revised once the frozen pilot
establishes confidence intervals. Revisions must be recorded before looking at the final
holdout.

## 14. Hardware allocation

- **RTX Pro 6000 / marimo:** large fixed-shape training, transformer/rung models, large
  batched residual beams and matched-throughput measurements.
- **Local RTX 4090:** controls, data validation, smaller ResMLP-Q training, exact-feature
  generation, replay, active-learning iteration and independent reproductions.
- **CPU/Rust:** KMC rough population export, exact insertion completion, path reduction
  and dataset materialization.

Do not let the faster training device determine the experiment order. The acceptance
gates, not available GPU memory, decide what is scaled.

## 15. Expected score ladder

The current verified score is 171,019. A practical milestone ladder is:

- **<166,421:** first model-built route that beats the current public 666 leader quoted
  in the research memo;
- **~160k:** learned residual/prefix search reliably improves the KMC family;
- **145-150k:** strong bounded residual solver or early successful orbit ladder;
- **130-140k:** efficient multi-rung cross-cluster planning;
- **110k:** roughly 116 moves per deep state, near the counting floor and beyond the
  normalized efficiency demonstrated by current big-cube learned solvers.

The first four are suitable engineering/research milestones. The last requires a new
near-optimal search regime and should not be used as the pass/fail expectation for the
first model.

## 16. Immediate next actions

The next concrete sequence is:

1. restore and hash arm A;
2. freeze three evaluation suites: depth visibility, KMC residual bands and ladder-start
   states;
3. patch KMC to export comparable rough populations and completion metadata;
4. run the small unbiased oracle-ceiling pool;
5. run the no-training residual gate;
6. run the two-cluster ladder-start baseline;
7. choose residual, ladder or both for targeted Beam-AVI based on those results;
8. start no large model and no million-wide beam before these gates report.

This sequence produces the maximum information before committing substantial GPU/search
time, and it converts each surviving route into a falsifiable program rather than another
open-ended model experiment.

## References within this repository

- `CUBE666_MODEL_RESEARCH_2026-08-24.md`
- `deep-research-report.md`
- `CUBE555_VS_CUBE666_COMPARISON.md`
- `CUBE666_MODEL_STRATEGY.md`
- `CUBE666_SOLVER_FROM_SCRATCH.md`
- `cube666/MACRO_POLICY_TRAINING.md`
- `cube666/README.md`
- `EXPERIMENTS.md`

