# Strategy for a model capable of solving 6x6x6

**Date:** 2026-08-22  
**Status:** recommended next direction after the full-width descent probe  
**Related evidence:** `CUBE666_REBUTTAL_AND_PROBE_RESULTS.md`,
`CUBE666_TEACHER_POSTMORTEM.md`, `HANDOFF_666.md`, `CUBE555_PROGRESS.md`  
**Existing operational guide:** `CUBE666_SOLVING_GUIDANCE.md`

## Executive recommendation

Do not train another larger flat primitive-move Q model using the existing random-walk
labels. The strongest measurements now show that this model family is locally competent
but globally miscalibrated on 666. At fully mixed depth, wider search does not average out
the error: it deliberately selects the states on which the model is most optimistic and
most wrong.

The highest-probability route to a full solver is instead:

1. construct a **stabilizer-chain hierarchy** of verified macro actions;
2. train small **stage-specific policy/Q models** to solve one orbit or a necessary coupled
   set of orbits at a time;
3. keep every learned decision inside a bounded subproblem with a horizon comparable to the
   model's demonstrated 30--35-move reliable range;
4. use exact or independently computed projected heuristics as the teacher, rather than
   bootstrapping from the current deep Q values;
5. compose the stages, verify the complete primitive path, and accept a long first solution;
6. once this hybrid solver works, use it as the deep teacher for on-policy distillation into
   a more general network.

If a runnable exact-supercube solver that accepts arbitrary states becomes available, the
fastest alternative is to use it as that teacher directly. A fixed CSV of competition paths
is not enough.

The objective of the next programme should be **full held-out solve capability first**.
Path compression comes after coverage.

---

## 1. What the evidence now says

The current deployed model is `models/q666_a_final_ARMA_deployed.pt`, a 26.4M-parameter
Q model trained for 3M updates. It is not simply a slightly weak global heuristic.

The decisive observations are:

- Deep action top-1 falls from 0.090 at depth 40 to 0.029 at depth 85, essentially the
  1/36 chance rate.
- Whole-pid searches at widths `2^18` and `2^20` ran for roughly 665--682 steps and solved
  none of the three deep pids. This is not a short step-cap artefact.
- Twelve-frame portfolios have already returned zero deep solves.
- Orbit-factored one-hot models at 119.9M parameters tied the smaller model rather than
  fixing the failure.
- One orbit is solved 24/24, two orbits only about 31--33%, and three orbits 0%. The useful
  clue is therefore the scaling cliff, not raw model capacity.
- At width `2^21` and `2^22`, `min_a Q` fell from about 75 to 11--13 while no search entered
  the exact depth-4 ball.
- At width `2^22`, the frontier finished at about 1.3 correctly placed stickers versus the
  random-state expectation of 9. The larger beam selected extreme underestimates and moved
  objectively away from solved.

The resulting diagnosis is:

> The current Q model is a useful local optimiser, but its values are not trustworthy at
> the 60--110-move depths required for a global 666 solve. Primitive-action beam width,
> frames and self-Bellman backups cannot manufacture the missing information.

This explains why the same broad recipe can solve 555 but not 666. The issue is not just
the additional generators or state size. The 555 search crosses its approximately
72-move global horizon while its scorer retains enough action preference for frames and
width to amplify. On 666, action preference has collapsed before the approximately
102--110-move global horizon, and width amplifies structured error instead.

---

## 2. Target architecture: a hierarchical macro-action solver

### 2.1 Core idea

Factor the global solve into a sequence of subgroup or coset problems. At each stage:

- some sticker orbits are declared **locked**;
- the available actions are macros whose **net permutation fixes every locked orbit**;
- the active model scores progress on the next orbit or coupled orbit set;
- the stage ends only when an exact goal predicate is satisfied;
- subsequent stages operate inside progressively smaller stabilizer subgroups.

A macro is allowed to disturb locked pieces during its internal primitive sequence. It must
restore them at the macro boundary. Requiring every primitive move to preserve the locked
set would make the action space unnecessarily weak and would reproduce the stabilizer
valley already seen by the primitive beam.

The intended shape is:

```text
fully mixed state
    -> solve orbit/group A
    -> macros fixing A solve orbit/group B
    -> macros fixing A+B solve orbit/group C
    -> ...
    -> exact final residual
    -> identity
```

This changes the learning problem. The model no longer has to represent a single
100-move global potential over the entire 666 group. It represents several shorter,
better-labelled potentials over projected state spaces.

### 2.2 Why this direction fits the measurements

The one-orbit rung models are the only strong positive control in the 666 experiments:
24/24 solves in roughly 19--22 moves. Their failure under naive composition occurred
because a sum of independent orbit heuristics could not justify crossing a roughly
12-move temporary valley. Macro actions are designed to cross that valley atomically.

This is qualitatively different from the already-tested orbit-factored encoder. Changing
the input representation left the global primitive-action problem intact. The proposed
hierarchy changes the action graph, goal definition, label source and horizon.

### 2.3 Do not assume a strict one-orbit order

Some orbit configurations may be coupled by parity or other group invariants. Therefore
the stage chain must be derived from the actual generated subgroups, not from a visual
ordering of centres, wings and corners.

Possible stages may need to solve two orbits jointly. The correct sequence is the one that
passes reachability tests and produces short macro solutions, not necessarily the most
intuitive physical sequence.

---

## 3. Phase 0: algebra and macro preflight

Do this before any new neural training.

### 3.1 Build the stabilizer chain

Use permutation-group machinery such as a BSGS/Schreier--Sims construction to obtain, for
each candidate locked set:

- the subgroup that fixes the locked positions;
- generators or Schreier generators for that subgroup;
- primitive words for every derived generator;
- reachability and orbit information for the next projected stage.

Every stored macro must include:

- its exact 216-sticker permutation;
- its primitive generator word;
- its inverse macro;
- primitive length before and after safe local reduction;
- the locked sets it fixes;
- the active projected permutation it induces.

### 3.2 Verify actions, do not trust their construction

For every macro library:

1. replay the primitive word from identity;
2. compare the result with the stored macro permutation;
3. verify that all claimed locked positions are fixed;
4. verify inverse round trips;
5. deduplicate identical projected actions;
6. reject identity macros and unexpectedly long duplicates.

The parity statement in the older action plan must not be reused: 12 generators are odd
and 24 are even, so the full graph is not bipartite.

### 3.3 Measure whether a rung is economically useful

Before training, measure for each stage:

- number of distinct reachable projected states in controlled samples;
- effective branching after inverse removal and deduplication;
- average new information or projected-distance gain per macro;
- mean, median and tail primitive length per macro;
- primitive path inflation implied by sample stage solves;
- whether important configurations require joint-orbit stages.

The macro ladder is useful only if it both reaches the required states and produces
reasonable primitive paths. A hierarchy that solves everything in 450 moves may still be
valuable for coverage on the deepest pids, but it is a different outcome from a 200--300
move solver and should be evaluated honestly against the 362,371 floor.

### 3.4 First hard gate

The first decisive experiment is:

> Starting from random reachable two-orbit states, can an oracle or classical projected
> search solve orbit 2 while orbit 1 remains fixed at every macro boundary?

Run at least 100 held-out cases. If this fails, do not train a network. Fix the stage
definition, coupled-orbit choice or macro library first.

---

## 4. Obtain independent stage heuristics

The descent probe demonstrates that a model cannot safely serve as both student and deep
teacher. Each stage needs a source of information that is independent of the Q model being
trained.

### 4.1 Preferred label sources

In descending order of trust:

1. **Exact projected BFS distances**, wherever the projected space or a useful shallow
   ball fits.
2. **Pattern databases** over selected sticker subsets, with cost partitioning when several
   patterns are added.
3. **IDA*, A* or beam search guided by those independent pattern heuristics**, producing
   verified stage solutions.
4. **A runnable external 666 supercube solver**, queried on arbitrary generated states.
5. **Verified successful macro paths**, used as policy demonstrations and upper bounds,
   not asserted to be exact distances.

Pattern databases need not encode a complete 24-sticker orbit to be useful. Multiple
smaller patterns can provide truthful lower bounds and action discrimination at fully mixed
depth. If their costs are added, move costs must be partitioned so the resulting heuristic
remains valid; otherwise take a maximum or use the values as non-admissible features with
that limitation recorded.

### 4.2 Why current-Q Bellman targets are not enough

A target of the form

```text
1 + min(frontier Q_target)
```

repeats the operation that failed in the probe. The minimum is an extreme selector for
optimistic errors. Increasing the frontier size can make the target look better while the
states become objectively worse.

Limited-horizon Bellman learning becomes credible only when the frontier is terminated or
ranked by an independent exact/PDB/teacher signal. A second copy or delayed copy of the same
model does not remove a shared calibration failure.

---

## 5. Stage-model training design

### 5.1 Train separate models first

Use one small model per stage during the research phase. Separate models make it possible to
identify exactly which subgroup transition fails. A shared stage-conditioned model can be
considered after the full hierarchy works.

Suggested inputs:

- lossless orbit-local one-hot features for the active and coupled orbits;
- invariant or coupling features from unsolved orbits where reachability depends on them;
- stage ID and locked-orbit mask;
- optionally the target coset or goal description for a goal-conditioned model.

Suggested outputs:

- a policy over the current macro library;
- optionally a value in **macro steps**;
- optionally auxiliary projected/PDB predictions for each active pattern.

The first version should prioritise a well-calibrated policy over a precise scalar value.
Search primarily needs reliable action ranking.

### 5.2 Training data

Build data from successful classical stage solves:

1. sample legal states in the current stabilizer subgroup;
2. solve the projected stage with the independent teacher;
3. record all states on the verified solution path;
4. also collect nearby off-path states produced by the student search;
5. query the teacher again on those off-path states;
6. iterate until the student succeeds on its own state distribution.

This on-policy aggregation is essential. Training only on teacher corridors produced the
previous 0.977-accuracy memorisation result without general solving ability.

Retain exact shallow-ball states in every batch to anchor the solved boundary and absolute
scale.

### 5.3 Losses

Recommended initial loss mixture:

- policy cross-entropy on one or several teacher-approved macros;
- pairwise ranking of teacher actions against hard negative macros;
- value regression to verified teacher cost or exact/PDB targets, with label provenance
  tracked;
- exact boundary loss for solved and shallow-ball states;
- symmetry consistency only for symmetries that preserve the stage definition;
- optional auxiliary prediction of pattern-database values.

Do not treat the length of a random generating walk as exact graph distance. It is a valid
upper bound and supplies a known reverse action, but group relations may provide a shorter
route or a different best action.

### 5.4 The all-column argmin hinge

The all-column hinge is useful only when its preferred action is certified.

For a cost-valued Q head, a zero-margin form is:

```text
sum_a max(0, stop_gradient(Q(s, a_best)) - Q(s, a))
```

It prevents another action from being predicted below the certified best action while the
stop-gradient prevents the anchor from moving merely to satisfy the regulariser.

Apply it when `a_best` comes from:

- exact distance labels;
- an exact projected ball;
- a pattern-database comparison that certifies the inequality;
- a sufficiently strong independent stage teacher.

Do not apply it globally merely because `a_best` is the undo step of a random walk. The undo
child is one step earlier on that generated walk, but it is not guaranteed to be an argmin
of true graph distance. Incorrectly imposing the inequality on all other columns can turn a
noisy label into 35 false constraints.

### 5.5 Curriculum

Curriculum should advance over **verified stage difficulty**, not the number of visible
orbits in a global primitive-action task.

A suitable progression is:

1. exact shallow macro ball;
2. deeper states solved reliably by the independent teacher;
3. student-search states close to its current failure boundary;
4. fully mixed projected states for that stage;
5. only then the next stabilizer stage.

Do not promote based on training loss alone.

---

## 6. Search and composition

### 6.1 Stage search

For each stage, use moderate-width beam, weighted A* or best-first search over macros.
Maintain:

- full-state hashes for correctness and deduplication;
- exact locked-set checks at every macro boundary;
- an exact projected stage-goal predicate;
- sufficient path history to reconstruct primitive words;
- an exact/PDB endgame where available.

Model predictions should select work, but the goal test must remain exact.

Frames can be reintroduced only after a stage model has a demonstrated nonzero solve rate.
They are a portfolio multiplier, not a source of capability.

### 6.2 Compose stages

After each successful stage:

1. replay its primitive expansion from the current full state;
2. verify the new stage goal;
3. verify every prior locked set;
4. append the word to the full path;
5. begin the next stage from the resulting exact state.

Use exact search for the final residual wherever feasible. The corner state space
`8! * 3^7 = 88,179,840` is large but structured enough to treat as an exact-table or
specialised-search problem rather than another neural stage.

### 6.3 Initial success criterion

The first end-to-end solver should be judged on:

- replay to the exact 216-sticker identity;
- held-out coverage;
- primitive path length after safe cancellation;
- deterministic artifact provenance.

It should not be rejected merely because its first paths are longer than desired. A full
solver creates the deep teacher needed for subsequent improvement.

---

## 7. Bootstrap from the hierarchical solver to a stronger general model

Once the hierarchy solves arbitrary held-out states, generate a broad teacher dataset:

- fresh random supercube states, not only the 1012 competition states;
- complete teacher solutions and phase boundaries;
- teacher action distributions or several search-approved actions per state;
- states reached by the student's own failed searches;
- local perturbations around phase transitions;
- solution costs, projected heuristic values and macro labels.

Then train either:

1. a shared stage-conditioned macro policy/value network; or
2. a global network that predicts phase, subgoal and primitive/macro policy jointly.

Continue to alternate student rollouts with teacher relabelling. This directly addresses
the distribution-shift problem that made fixed-path imitation memorise.

The hierarchy is therefore not a permanent concession. It is a way to create a competent
deep expert when none currently exists.

---

## 8. Faster alternative: distil a runnable external solver

If the external solver that produced the strong 666 CSV can be obtained as a program and
can solve arbitrary exact-supercube states, use it immediately.

Required properties:

- accepts the full 216-distinct-sticker goal, not only colour-solved 6x6x6 states;
- can solve newly generated random states, not just competition pids;
- exposes complete paths or at least phase/action decisions;
- produces enough diverse trajectories to query on student-induced states.

Training would then follow the same on-policy aggregation process described above. Phase
labels from a classical solver may be more valuable than raw remaining-length regression,
because they expose the long-range structure the current Q model lacks.

A single solution CSV is useful for baseline paths and local bridge training, but it is a
measure-zero training corridor. Symmetry expansion produces equivalent views of the same
corridor; it does not replace state diversity.

---

## 9. Short-term score route while the full solver is being built

The existing Q model remains useful inside its reliable local range. Use it for:

- bridge replacement on 20--35-move segments;
- waypoint-to-waypoint goal-conditioned solving;
- compression of 200--300-move windows by decomposing them into local subgoals;
- shortening paths produced by a classical or hierarchical full solver.

A particularly practical model is `Q(s, goal, a)` trained on local perturbation tubes
around known paths. Waypoints should be at most roughly 20--30 moves apart initially. This
can make reproduction of a supplied full solution robust and may skip redundant waypoints,
but it is not a from-scratch global solver because it still requires a baseline path for
each target.

Keep this score work separate from the capability programme so a few saved moves do not get
mistaken for evidence of deep global guidance.

---

## 10. Experimental gates

Use the following sequence. A failed gate stops downstream work until its cause is fixed.

### Gate A: macro correctness

- 100% primitive replay agreement;
- 100% preservation of claimed locked sets;
- inverse round trips pass;
- projected action library is deduplicated and nontrivial.

### Gate B: two-orbit oracle

- at least 100 held-out reachable states;
- target orbit solved while the first stays locked;
- target success at least 90%;
- primitive path inflation recorded.

### Gate C: learned two-orbit stage

- matched oracle/state distribution;
- at least 80--90% held-out stage success;
- clear gain over random macro ranking;
- no widening-induced reversal in independent progress metrics.

### Gate D: three-orbit composition

- nonzero success is insufficient; require a credible path toward high coverage;
- verify every locked set after every macro;
- compare strict single-orbit and necessary joint-orbit variants.

### Gate E: complete hierarchy

- solve at least 24--48 fully mixed held-out states;
- replay every primitive path to exact identity;
- report coverage and path-length distribution, not only selected examples.

### Gate F: competition evaluation

- run against the current verified floor;
- min-merge only replay-valid improvements;
- report full source provenance and per-pid ownership.

Every harness should have positive controls. Timeouts, unfinished searches and failed
replays must remain separate outcomes.

---

## 11. What not to spend compute on now

Do not repeat these without genuinely new evidence:

- larger versions of the same primitive-action ResMLPQ;
- more training updates on the same random-walk label distribution;
- orbit-factored one-hot as a representation-only change;
- global active-orbit curricula using primitive moves;
- width beyond the already probed range with the same scorer;
- larger frame portfolios before a nonzero deep per-frame success rate exists;
- Bellman minima backed up solely from the present Q model;
- training on one fixed teacher CSV, even with symmetry augmentation;
- judging by loss, `min Q`, top-1 on shallow states or a few hand-picked pids.

These experiments leave the missing long-range signal unchanged.

---

## 12. Concrete next work package

The next bounded implementation project should be:

1. select the strongest existing one-orbit rung as stage 1;
2. derive a macro library in its stabilizer for the next orbit or coupled pair;
3. build replay, inverse and locked-set verification for every macro;
4. measure reachability, effective bits per macro and primitive inflation;
5. build the smallest useful exact/PDB heuristic for the active projection;
6. run the 100-state two-orbit oracle gate;
7. only after it passes, generate teacher solutions and train the stage-2 macro policy;
8. run a matched learned-vs-oracle search evaluation;
9. attempt the three-orbit gate;
10. document the result before expanding the hierarchy.

This work package directly tests the proposed escape from the observed three-orbit cliff.
It is small enough to fail informatively and strong enough that success would materially
change the outlook for a complete 666 solver.

## Final position

A flat global 666 model is currently blocked by the absence of trustworthy deep labels,
not by insufficient width or network size. There are two credible ways past that block:

1. obtain a real arbitrary-state expert and distil it on-policy; or
2. create that expert ourselves by composing learned, independently supervised macro stages
   inside stabilizer subgroups.

The second route is available with the assets already demonstrated in this project. It
turns the one-orbit success from an isolated curiosity into the foundation of a full solver,
and it provides a disciplined path toward a stronger unified model later.
