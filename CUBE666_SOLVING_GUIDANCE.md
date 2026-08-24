# Solving the 6x6x6 picture cube with neural guidance

**Status:** recommended next programme after the 2026-08-21 555 and 666 reviews  
**Primary evidence:** [`CUBE555_PROGRESS.md`](CUBE555_PROGRESS.md),
[`HANDOFF_666.md`](HANDOFF_666.md), and
[`CUBE666_TEACHER_POSTMORTEM.md`](CUBE666_TEACHER_POSTMORTEM.md)

This document answers one practical question: **what should be done next to make a
model solve full 666 picture-cube states?** It deliberately separates the cheap,
immediate experiment with the existing model from the slower programme for a new model.

The most important correction from the 555 work is that full-puzzle neural solving is
not structurally impossible on a large, fully mixed picture cube. The 555 sparse-Q model
solves fully mixed random-walk pids at useful rates. The 666 result is better understood
as a **searchability threshold**: the existing model retains useful guidance through the
shallow and middle region but loses action discrimination before the approximately
110-move random-state scale. The first job is to determine whether the complete 555
inference stack can push the existing model across that threshold. Only then should a new
training run be judged necessary.

---

# Part I — Run inference with the existing model

## 1. Objective

Use the deployed checkpoint:

```text
models/q666_a_final_ARMA_deployed.pt
```

Do **not** start with the teacher-path checkpoints or a Bellman-refined checkpoint.
Arm A is the cleanest full-puzzle random-walk model and the 666 Bellman experiment
monotonically reduced matched bridge wins. The immediate question is:

> Does arm A solve full 666 states when given the inference machinery that made the
> 555 model operational: a larger Q beam, multiple symmetry frames, inverse frames,
> history filtering, an exact endgame goal set, fixed-shape compiled inference, and a
> step cap appropriate for 666?

This is an inference experiment, not a new-model experiment. Freeze the checkpoint and
change one search family at a time.

## 2. Restore and freeze the baseline

On the original machine, restore `code_and_state.tgz` and the deployed checkpoint as
described in `HANDOFF_666.md`. Before running a science benchmark:

1. Hash the checkpoint and record the hash in the run manifest.
2. Record the exact generator list and inverse map. Expect 216 positions and approximately
   36 primitive generators; assert the actual values from `puzzle_info` rather than
   hard-coding them.
3. Print the resolved model configuration at startup, including encoding, trunk size,
   number of Q outputs, state size, dtype, compile mode, and checkpoint step.
4. Run the historical arm-A inference configuration byte-for-byte on its original
   benchmark. This is the matched control for every later search result.
5. Replay every returned path through a second verification path. A hash hit or a
   non-empty path is not proof of a solve.

Do not compare a new run against a remembered historical number. Produce the matched
control on the same machine, code revision, pid set, checkpoint, and output verifier.

## 3. Port the complete 555 inference stack

The following components should be available together before the final gate. Each must
first pass its own positive control.

### 3.1 Q-head parent scoring

Score all primitive children with one Q forward on each parent. Materialize and hash only
the selected candidates if the current search implementation supports progressive top-k.
The 30-wide Q head was a decisive throughput lever on 555; the same principle applies to
the 36-wide 666 head.

Measure states/second and whole beam-step time. Do not infer that a wider beam is active
from a parsed flag—log the realized width at every step and assert that the peak live beam
reaches it on a nonsolving control.

### 3.2 Fixed-shape compiled inference

Compilation is allowed only with fixed-size padded model calls:

1. Choose a fixed internal inference chunk size.
2. Pad every forward to that size.
3. Pre-warm the compiled graph at exactly that shape.
4. Verify byte-identical paths between compiled and eager inference on several controls.
5. Confirm from compiler logs or timing that there is one compiled shape, not a
   shape-dependent recompile loop.

The expected result is a throughput improvement, not a change in paths.

### 3.3 History filtering

Test `history_depth=4` against the matched historical control. It shortened 555 solutions
materially, but do not assume the 555 parity argument transfers unchanged. First compute
the parity of every 666 generator permutation and inspect actual revisit lags.

Acceptance requirements:

- identical or better solve coverage;
- shorter paths or lower wall time on the same successful states;
- no accidental pruning of the only valid short solution on shallow exact controls.

### 3.4 Exact endgame goal set

Build or restore a complete exact ball, initially d≤4 and preferably d≤5 if memory and
lookup cost are practical. The beam should stop when it reaches any verified ball state
and splice the exact descent to identity.

Required tests:

- exact level counts agree with an independently generated shallow BFS;
- every stored descent replays to identity in exactly its recorded depth;
- membership uses a hash to find candidates but compares the full state before accepting;
- a false hash match is non-fatal and continues the search;
- known d≤5 states solve exactly through the splice path.

The ball changes the goal from one identity state to millions of valid entry states. It
can improve coverage even when the learned scorer itself is unchanged.

### 3.5 Symmetry frames

Use the verified cube automorphism group already derived for the project. For every frame:

1. Transform the state.
2. Solve the transformed state.
3. Relabel the returned generators back to the base frame.
4. Replay against the original state.

The frame table must pass group closure, generator conjugation, inverse-generator, and
round-trip tests. A frame is not accepted merely because it produces a path.

Run frames in fixed sets of six and early-exit on the first verified solve when optimizing
coverage. For score optimization, retain every verified candidate and keep the shortest.
Record the winning frame for each pid so frame frailty and diminishing returns can be
measured rather than assumed.

### 3.6 Inverse/NISS frames

666 is a picture cube: the state identifies a group element, so inverse-state solving is
legal. For each base or symmetry-frame state, also solve its inverse, translate the result
back, and replay against the original state.

Before the science run, use constructed positive controls with known words. A translated
inverse path that does not solve is a direction bug, not evidence that inversion is
unhelpful.

Twenty-four rotations plus inversion provide 48 trajectories; a verified 48-element
automorphism set plus inversion provides 96. Start with small unused frame sets and scale
only after the translation tests pass.

### 3.7 Step cap

The 555 model finds solutions of length 128–250 with mean 169 for states whose random-state
distance scale is about 72. Scaling that stretch to a 666 distance scale near 110 suggests
that a successful neural path could easily be 250–300 moves.

Use a science cap of at least 350 and preferably 400 unless memory/backpointer constraints
require a smaller value. Log whether failures terminate because of the cap, stagnation,
OOM, or exhausted frames. A 60- or 120-step cap would test the wrong hypothesis.

## 4. Positive controls before real pids

The pipeline has many ways to return zero while being broken. Run these controls first:

1. **Shallow solve control:** at least 32 fresh random walks each at depths 10, 20, and 30.
2. **Near-envelope control:** at least 32 each at depths 40 and 60.
3. **Frame control:** a known base solution transformed through every enabled frame and
   translated back.
4. **Inverse control:** known words solved through inverse-state translation.
5. **Endgame control:** random members at every exact depth in the ball.
6. **Width control:** a hard nonsolving state whose log proves that the requested live beam
   was reached.
7. **Resume control:** one solved and one failed pid written durably, followed by a resume
   that repeats neither. Use fallback rows or a separate attempt journal so failures are
   recorded.

Do not proceed if any control is ambiguous.

## 5. Benchmark design

Tiny gates caused the teacher-tube misread. Use two separate sets.

### 5.1 Diagnostic random-walk bank

Generate a fixed, held-out bank with at least 64 states at each of:

```text
r = 30, 40, 60, 80, 100, 110, 122
```

Store the state, generating word, state hash, seed, and depth. The generating word is a
valid upper-bound solution and makes every row independently verifiable. This bank is for
action discrimination, shallow solve rate, and checkpoint comparison—not for claiming
competition score.

Report for every depth:

- undo-action top-1 and mean rank;
- Q gap between undo and median action;
- pair accuracy for undo versus next;
- `min Q`, Q standard deviation, and action-margin distribution;
- beam solve rate and verified path length on the affordable depth bands.

The key comparison is against chance, approximately 1/36, and against the 555 signature,
where deep Q accuracy remained many times above chance.

### 5.2 Real-pid science gate

Use at least 48 real fully mixed pids:

- 24 sampled without reference to current solution length;
- 24 from the longest current baseline paths, because those carry the largest score EV.

Reserve a separate holdout set for the final configuration. Do not repeatedly tune on the
same 48 and then quote their solve rate as an unbiased estimate.

For every pid and every attempted frame, journal:

- frame and inverse flag;
- requested and realized beam width;
- solve/failure reason;
- steps reached and best endgame depth;
- path length and replay verdict;
- wall time, model time, hashing time, peak device memory;
- improvement versus `FINAL_submission.csv`;
- whether this path wins the n-way per-pid merge.

## 6. Staged inference experiment

Use the same pid set and checkpoint for all arms.

| Arm | Configuration | Question |
|---|---|---|
| I0 | Historical arm-A inference | Matched control |
| I1 | I0 + fixed compile + verified endgame + history-4 | Do production mechanics help? |
| I2 | I1 + six symmetry frames at beam 2^20 | Do frames expose a nonzero success basin? |
| I3 | I2 + inverse versions of those frames | Does directional anisotropy help? |
| I4 | Best of I2/I3 at beam 2^21 | Does the 555 production width cross the threshold? |
| I5 | Residue only at beam 2^22 | Is width still a useful rescue lever? |

Frames should be tested before indiscriminate width escalation. On 555, fresh frames were
more valuable per GPU-second than doubling width. Do not assume independence: measure
success by frame round and expect the residue to become progressively harder.

## 7. Decision rules

Use these as operational rules, not mathematical impossibility proofs.

### If the existing model solves real pids

If the full gate produces at least several verified full solves:

1. Run fresh symmetry/inverse frame sets on the residue before increasing width.
2. Queue pids deepest-first by current baseline length or expected moves saved per GPU-second.
3. Early-exit after the first solve for coverage, then revisit successful long paths for
   score improvement with more frames.
4. Keep every result as a candidate source; the model need not beat the current file as a
   standalone submission to contribute through per-pid min-merge.
5. Re-run the existing neural bridge and r=4 window pipeline on any newly produced loose
   solution paths.

### If success is rare but nonzero

If only one to four of 48 pids solve, do not declare victory or failure. Test all remaining
unused frames and inverse frames on those successes and on a fresh 48-pid holdout. Determine
whether success is reproducible and whether it concentrates in a measurable state band.

This is still valuable training evidence: successful and near-successful frontiers become
the best source distribution for limited-horizon Bellman training.

### If the full stack returns 0/48

Zero of 48 after verified frames, inverse, a 350–400 step cap, endgame, and at least beam
2^21 puts the approximate 95% upper bound on configuration-level solve probability near 6%.
At that point:

- do not launch a full beam-2^22 campaign;
- retain a small beam-2^22 matched check only if deep action metrics still show a clear
  above-chance signal;
- proceed to new-model training;
- keep arm A for bridge/window selection, where it already has measured value.

## 8. Production and verification

Every attempt must be durable. A failure should be journalled just as explicitly as a
success. The final sequence is:

```text
solve candidates
→ replay each candidate independently
→ per-pid min-merge with all stable sources
→ bridge/window post-processing where indicated
→ write a complete 1012-row file
→ read the written file back
→ independently replay all 1012 rows
→ compare total and md5 of the staged copy
```

Never quote an in-memory or unverified merge total.

---

# Part II — Train a new model

## 9. Training objective

The new model should not attempt to learn a better absolute estimate of the submitted
solver's remaining path length. It should preserve and strengthen **action discrimination
at r≈80–110**, where arm A becomes ineffective.

The target signature is the one observed on 555:

- Q top-1 at random-state depth remains several times above chance;
- undo-versus-next and good-action margins remain positive;
- warm-started search backup improves, rather than erodes, beam performance;
- multiple symmetry frames have a nonzero per-frame success probability;
- the model contributes verified full solves at deployment width.

Train and select checkpoints on those properties. Training loss, policy accuracy on a fixed
teacher corpus, and absolute Q scale are diagnostic only.

## 10. Preserve the successful 555 recipe

The base control should faithfully reproduce arm A before introducing a new idea:

1. Sparse-Q random-walk middles with absolute MSE on the two labelled columns.
2. `k_max=122` or the exact arm-A value—not 40.
3. The proven pivot distribution from arm A; do not rebundle flat/tilted sampling with an
   architecture change.
4. Exact shallow anchors in every batch, with all Q columns labelled where exact.
5. Random verified symmetry frames per sample.
6. Q and V heads with the same conventions as the deployed searcher.
7. Fresh procedural states rather than a fixed path dataset.
8. The same update, batch, optimizer, and learning-rate budget for matched controls.

The control is not optional. Without it, a machine, code, or throughput change can be
mistaken for a scientific improvement.

## 11. Proposed architecture: orbit-factored sparse-Q

### 11.1 Derive the nine orbits

Derive position orbits from the generator permutations rather than hard-coding geometry:

1. Build the graph on the 216 sticker positions induced by every generator.
2. Compute connected components.
3. Assert exactly nine components of 24 positions.
4. Assert that every generator maps each component to itself, or explicitly records any
   automorphism-induced component permutation.
5. For every reachable state, assert that the sticker occupying a slot belongs to the
   slot's position orbit.
6. Assign each sticker a stable local ID 0–23 inside its orbit.

The representation for one orbit is then 24 slots × 24 possible sticker IDs. Across all
nine orbits the lossless input has:

```text
216 × 24 = 5,184 binary features
```

This has the same conceptual advantage as the proposed 555 orbit-factored one-hot: it is
lossless but does not waste 216 classes at every slot.

Symmetry tests must additionally verify how whole-cube frames permute orbit indices and
local sticker IDs.

### 11.2 Network structure

A strong first implementation is:

```text
per orbit: 24×24 one-hot → shared orbit encoder → 256-d token
9 orbit tokens + learned orbit/type embeddings
small 9-token mixer (MLP mixer or 1–2 attention blocks)
flatten/pool → 1024-wide residual trunk
global Q head: 36 outputs
global V head: 1 output
auxiliary orbit heads: 9 × 36 outputs
```

Attention over nine orbit tokens is cheap; sticker-level dense attention is not. Keep the
total parameter count near the arm-A control for the first comparison so architecture and
capacity are not confounded. A 1024-wide residual trunk can preserve the approximately
25M-parameter budget while changing the representation and supervision.

Include an interaction/residual route so the global score is not forced to be a simple
sum of independent orbit scores. Conceptually:

```text
Q_global(s,a) = Σ_i w_i(s) Q_i(s_i,a) + Q_interaction(s,a)
```

This explicitly preserves individually learnable orbit signals while letting the model
learn trades between orbits.

### 11.3 Auxiliary orbit labels

Do **not** naively apply the full-walk index to every orbit head. A generator can be a
no-op on a particular projected orbit, and an orbit projection can mix much sooner than
the full state.

Generate a separate projected sparse-Q stream for each orbit or active-orbit mask:

1. Use the generator permutations induced on that projection.
2. Drop generators that are identity on the projection.
3. Generate a non-backtracking projected random walk.
4. Count projected/effective steps, not full-cube primitive steps.
5. Label the projected undo and next actions with projected walk indices.
6. Use exact projected tables wherever affordable.

The existing one-orbit rung success is the positive control for this data stream. The
global head continues to receive the original full-state sparse-Q labels so its scores
remain comparable across parents in the production beam.

Start auxiliary loss at a modest weight and report its gradient norm relative to the
global Q loss. The purpose is to prevent orbit signals from disappearing, not to let nine
easy auxiliary tasks dominate the full problem.

## 12. Active-orbit curriculum

The 555 result suggests that a monolithic model can coordinate approximately six moving
24-sticker orbits, while 666 must coordinate nine. Locate the scaling cliff explicitly.

Construct projected tasks with active-orbit counts:

```text
1 → 2 → 4 → 6 → 7 → 8 → 9
```

For an active mask, use the induced generator action on only those orbit components and
drop projected no-op actions when generating training walks. Inactive components should
be represented as solved plus an explicit active/locked mask; do not silently zero them
without telling the model which task it is solving.

Two valid experimental designs are:

1. **Diagnostic models:** train small matched models independently at each orbit count to
   measure where solve rate collapses.
2. **One conditioned model:** sample an active mask and train a shared network, expanding
   the maximum active count only after the prior rung passes its held-out beam gate.

The conditioned version is the eventual target, but the diagnostic ladder is easier to
interpret. Do not advance based on training loss. Advance when the current rung solves a
predeclared held-out set with a fixed beam and preserves above-chance deep action metrics.

This curriculum is not intended to make the final model solve nine independent rungs
sequentially. It teaches cross-orbit composition at controlled scales before asking for
all nine simultaneously.

## 13. Training data

Use exactly three primary sources for the first new-model run.

### 13.1 Full-state sparse-Q walks

Retain the arm-A random-walk distribution and `k_max`. Label only undo and next columns,
preserving the absolute MSE term so Q values remain comparable across parents. A pure
pairwise ranking loss is insufficient for a global top-B over all parent/action pairs.

Log by pivot-depth band rather than only an aggregate loss. The r80–122 rows are the
reason for the new model.

### 13.2 Exact shallow anchors

Include exact anchors in every batch. At a minimum:

- identity and all d1 states every step or at high fixed frequency;
- complete exact rows through the affordable BFS depth;
- all 36 child columns labelled exactly;
- solved-child clamps verified independently.

Keep enough anchor weight to pin absolute scale, especially during any later Bellman
phase. Log anchor loss separately.

### 13.3 Symmetry expansion

Draw verified symmetry frames randomly per sample. Ensure action-pair column coverage
without reusing the same small fixed state set. The augmentation must transform state,
active-orbit mask, orbit IDs, and action labels consistently.

### 13.4 Shallow label correction

Where random walks enter the exact table:

- replace walk labels with exact child distances;
- drop sparse-Q rows whose asserted ordering is contradicted by exact distances;
- deduplicate identical shallow states using their best known label;
- log generated rows, exact-table hits, labels changed, and rows dropped.

This will not fix the deep problem by itself, but it prevents avoidable contradictory
supervision from contaminating the action margins.

### 13.5 What not to add

Do not add any of the following to the first orbit-factored run:

- remaining length along `FINAL_submission.csv` as a value or Q target;
- fixed on-path imitation pairs;
- `k_max=40`;
- one-step Bellman targets;
- sorted-profile or permutation-invariant label families;
- a new loss, a new architecture, and a new data mixture in the same arm.

The teacher tube can remain a separately trained local bridge/recovery model. Its resampled
signal is useful near existing solution corridors, but it should not define the global
666 value scale.

## 14. Base training schedule

Use the 555/arm-A scale as the reference budget: approximately 3M optimizer updates and
billions of fresh procedural states. The exact wall time will differ on 216-position
inputs, so budget by updates and states, not epochs or hours.

Recommended stages:

### Stage A — wiring pilot

Run only long enough to prove:

- every loss decreases on its own positive control;
- all 36 columns receive supervision;
- every orbit head beats chance on its own projected bank;
- symmetry round trips and action relabeling are exact;
- the global model solves shallow full-state walks;
- no label source silently contributes zero.

A wiring pilot may reject broken code. It must not be used to declare the architecture
scientifically worse.

### Stage B — matched full control

Reproduce arm A under the current code and hardware with its original representation.
Run the complete predeclared budget.

### Stage C — orbit-factorized architecture only

Change representation/architecture while holding data, objective, seed family, update
budget, optimizer, and evaluation fixed. This isolates whether factorization preserves
more deep action signal.

### Stage D — active-orbit curriculum and auxiliary heads

Warm-start from the successful Stage C trunk only if Stage C passes its gate. Add the
projected auxiliary streams and active-mask curriculum. Compare against a Stage C
continuation with the same extra updates and learning-rate schedule.

### Stage E — limited-horizon Bellman

Only run after a checkpoint demonstrates meaningful r100 action discrimination or a
nonzero full-state solve rate. Bellman cannot create a reliable target from a completely
nondiscriminating teacher.

## 15. Limited-Horizon Bellman Learning

The 555 report identifies this as the most plausible next training improvement. It is
especially relevant to 666 because a one-step neighbor difference may be too weak while a
short sequence can reveal progress.

### 15.1 Source states

Mix:

- held-out random-walk states at r60–122;
- actual beam frontiers from failed and near-successful full-pid searches;
- states just outside the exact endgame ball;
- a controlled fraction of ordinary random-walk pivots to avoid overfitting a few pids.

Store source provenance and never evaluate checkpoint quality on the exact same frontier
bank used for targets.

### 15.2 Target construction

For each root action `a`:

1. Apply `a` to get child `x0`.
2. Run a small bounded search from `x0`, initially horizon H=4 and a modest local beam.
3. For a frontier state `xd` reached after `d` more actions:
   - if `xd` is in the exact ball, candidate target is `d + exact_distance(xd)`;
   - otherwise candidate target is `d + 1 + min_b Q_target(xd,b)`.
4. Back up the best valid candidate as the target for `Q(s,a)`.
5. Preserve exact solved/anchor overrides.

Keep H and the local beam small initially. Taking a minimum over a huge noisy frontier
creates an optimistic-selection bias and is a plausible route to the ceiling collapse
seen in the 666 one-step Bellman run.

### 15.3 Reduce bootstrap bias

Prefer a double-target construction:

- target network A selects the frontier/action;
- independently initialized or EMA target network B evaluates it.

Alternatively, select with the online model and evaluate with an ensemble mean of frozen
targets. The essential point is not to use the same noisy estimator both to search a large
frontier and to assign the minimum numerical target.

Retain exact anchors in every batch at the strong Bellman weight used successfully on
555. Refresh targets slowly, use a small learning rate, and checkpoint frequently.

### 15.4 Stop conditions

Evaluate at short fixed intervals. Stop the Bellman leg if any two consecutive checkpoints
show:

- falling r100/r110 action accuracy;
- shrinking undo-versus-median Q margin;
- rapid decline in `E[target]` or the deep Q ceiling;
- worse matched beam coverage;
- improving loss while all search metrics worsen.

Loss improvement alone is never a reason to continue.

## 16. Checkpoint evaluation

### 16.1 Offline metrics

On fixed held-out banks at r30, 40, 60, 80, 100, 110, and 122, report:

- undo top-1, top-3, top-8, and mean rank;
- undo-versus-next pair accuracy;
- undo Q minus median-action Q;
- fraction of states with any action margin above a fixed numerical threshold;
- `min Q`, Q dispersion, V, and `V-minQ`;
- symmetry consistency by action relabeling;
- results separately by active-orbit count.

Always include chance lines. A numerically compressed Q is acceptable if action ordering
remains useful; a well-calibrated Q at chance ranking is not.

### 16.2 Search metrics

Use two gates:

1. A moderate-width checkpoint gate on at least 24–48 fixed states, used frequently.
2. A production-width holdout gate with the complete Part I inference stack, used only for
   checkpoints that pass offline canaries.

Report verified solve count first, then path length on the common solved set, then wall
time. A shorter mean over fewer successes is a regression when fallback paths are long.

### 16.3 Acceptance criteria

A new checkpoint advances only if it does at least one of:

- materially raises r100/r110 action discrimination versus the matched arm-A control;
- produces verified full solves where arm A produces none;
- adds per-pid min-merge wins at matched inference budget;
- lowers the compute required for the same verified solve set.

Probe accuracy, policy memorization, or lower loss without a search improvement does not
qualify.

## 17. Required ablation order

Run one family at a time:

| Run | Difference from prior run | Purpose |
|---|---|---|
| T0 | Exact arm-A reproduction | Current matched floor |
| T1 | Orbit-factored representation/trunk only | Test representation |
| T2 | T1 + projected orbit auxiliary heads | Test signal preservation |
| T3 | T2 + active-orbit curriculum | Test 6→9 composition |
| T4 | T3 continuation control | Matched extra-update control |
| T5 | T3 + limited-horizon Bellman | Test search-backed refinement |

Do not compare T5 against T0 and attribute the whole difference to Bellman. Every run must
have a matched continuation/control at the same cumulative update count.

Use sufficiently large science gates. A 0/4 or 0/6 result is a canary, not a capability
verdict. Where solve rates are low, compare identical states and identical attempted
frames, and include confidence intervals or an exact test.

## 18. If the direct model still fails

If an orbit-factored model with useful per-orbit heads still collapses when active count
reaches seven to nine, the next step is a hierarchical macro-action solver rather than a
larger flat model.

### 18.1 Build a valid subgroup chain

Choose an orbit order or coupled-orbit stages. For each stage, construct commutator and
conjugate macros whose **net** permutation preserves all locked orbits while acting on the
active orbit set. Temporary disturbance inside the macro is allowed.

Verify with group computation that each stage action set reaches every required projected
state. Parity or coupled invariants may require solving multiple late orbits together.

### 18.2 Train a macro-Q model

Train reverse random walks in macro space, keeping each stage inside its own mixing
envelope. Inputs are the active projection plus stage/locked mask; outputs score all valid
macros. Use exact shallow macro tables and search-backed targets.

The one-orbit rung model is the positive control. The critical new test is whether a
macro-space solver can solve orbit two while provably preserving orbit one.

### 18.3 Gate before scaling

Before training for all nine stages:

1. Prove two-stage reachability with an oracle heuristic.
2. Solve at least 100 held-out two-stage states with the learned scorer.
3. Repeat for three stages.
4. Measure primitive-move inflation of the macros.

If oracle macro search cannot compose two stages, fix the subgroup/action library rather
than changing the network.

## 19. Teacher paths and the new model

The teacher postmortem establishes two distinct facts:

1. The original 0/96 gate did not refute teacher-tube supervision; it was underpowered and
   confounded by `k_max=40`.
2. Fixed on-path policy supervision is memorization and does not produce score gains.

Therefore:

- do not use teacher remaining length as global distance;
- do not select a model by accuracy on the fixed 364k path states;
- do retain unbounded resampled tubes for a separate bridge/recovery head;
- if expert imitation is revisited, collect states induced by the current policy and query
  a real supercube solver there, rather than repeatedly replaying a fixed submission.

A full imitation route requires a solver that reaches the exact 216-sticker identity and
can label off-corridor states. A colour-only 666 solver is insufficient because it leaves
the supercube centre-permutation residual.

## 20. Artifact and reporting discipline

Every run should write an immutable manifest containing:

- code revision and dirty-worktree status;
- checkpoint input hash and output hash;
- puzzle/generator hash;
- symmetry-table and endgame-table hashes;
- resolved model and optimizer configuration;
- exact data-source proportions;
- seeds and procedural-state counts;
- cumulative updates, not only epochs;
- benchmark pid/state bank hashes;
- inference flags and realized beam width;
- verifier version.

Write append-only per-attempt journals during long solves. A run that stops or is preempted
should lose at most the current pid/frame, not its scientific record. Count which workers
actually emit records; a nominally running process that produces nothing is a failed
worker.

The final report for every arm must state:

1. matched-control solve count;
2. new-arm solve count;
3. common-set path delta;
4. per-pid min-merge contribution;
5. wall time and moves saved per GPU-second;
6. failures, timeouts, OOMs, and unwired/broken attempts;
7. replay status of every claimed path.

---

# Recommended execution order

1. Restore and reproduce historical arm-A inference.
2. Add positive-controlled endgame, frames, inverse translation, history filtering, and
   fixed-shape compile.
3. Run the 48-pid staged inference gate through beam 2^21; use 2^22 only as a targeted
   residue check.
4. If there are solves, scale frame/inverse coverage deepest-first and min-merge results.
5. In parallel, build the permanent diagnostic random-walk bank and orbit derivation tests.
6. Reproduce arm A as training control.
7. Train orbit-factored T1, then auxiliary-head T2, with one change per arm.
8. Locate the active-orbit scaling cliff and train the 1→9 curriculum.
9. Run limited-horizon Bellman only from a checkpoint with measurable r100 guidance.
10. If the direct nine-orbit model remains below threshold, switch to a verified
    subgroup-preserving macro solver.

The highest-value near-term action is **not** another large training launch. It is the
complete Part I inference gate. The 555 evidence shows that a modest per-frame success
probability can become high coverage under symmetry retries, and the existing 666 model
has not yet been judged under that full operating regime. The highest-value new-model
idea is then to preserve the already-proven one-orbit signal explicitly while learning
the six-to-nine-orbit composition that separates 555 from 666.
