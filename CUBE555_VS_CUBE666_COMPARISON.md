# Comparing the 5x5x5 and 6x6x6 supercubes

**Date:** 2026-08-24  
**Purpose:** separate the intrinsic path-length difference from the neural-model
capability difference, and clarify what a 110k cube666 score actually requires.

## Short conclusion

Three facts initially look contradictory but are all true:

1. A neural beam can solve many fully mixed cube555 states end to end.
2. Our cube666 neural beams cannot solve fully mixed unseen states end to end.
3. The current cube666 classical paths are not abnormally long relative to the strongest
   cube555 file once the intrinsic sizes of the two groups are taken into account.

The path-length gap and the model-capability gap are different phenomena. The former is
mostly expected from the geometry of the puzzles. The latter is a failure of the learned
heuristic beyond its reliable label horizon.

My current position is:

> Cube666 is not merely cube555 with 66 more stickers. Its typical optimal distance is
> roughly 50% larger. The current classical cube666 solution has approximately the same
> *relative* inefficiency as the external cube555 solution. What fails to transfer is the
> neural model's ability to maintain a useful direction signal all the way to the goal.

## 1. These are supercubes, not ordinary colour cubes

On an ordinary colour cube, centres of the same colour are interchangeable. Here every
sticker is distinct and the solved state is the exact identity permutation. This makes an
additional 24-position orbit a full permutation problem rather than a set of visually
equivalent pieces.

Measured structure from `BIGCUBES_PLAN.md`:

| quantity | cube555 | cube666 | ratio |
|---|---:|---:|---:|
| state entries | 150 | 216 | 1.44 |
| generators | 30 | 36 | 1.20 |
| measured effective branching | 24.092 | 29.023 | 1.20 |
| position orbits | 6 x 24 + 1 x 6 | 9 x 24 | -- |
| orbit-decomposition information | 307.93 bits | 500.62 bits | 1.63 |
| exact group size | 6.1983e91 | 3.1440e149 | about 5e57 larger |
| counting lower bound | 66.43 moves | 102.20 moves | 1.54 |

The extra branching on cube666 only partly compensates for the much larger amount of state
information. Before considering models or solver inefficiency, a typical cube666 solution
should already be about 36 moves longer than a cube555 solution near their respective
counting bounds.

The main structural increase is in the exact permutations of centres and wings:

- cube555: two 24-piece centre orbits and one 24-piece wing orbit, plus midges, corners and
  face-centre state;
- cube666: four 24-piece centre orbits and two 24-piece wing orbits, plus corners.

This is why intuition from ordinary 5x5x5 versus 6x6x6 solving understates the difference.

## 2. The current path-length gap is close to the expected gap

The two current replay-verified files are:

| artifact | rows | total | mean | median | maximum |
|---|---:|---:|---:|---:|---:|
| external cube555, post-processed | 1,035 | 109,288 | 105.59 | 113 | 124 |
| current cube666 merged solution | 1,012 | 171,019 | 168.99 | 187 | 203 |

The raw means differ by 63.40 moves, which looks enormous. Normalising by the counting lower
bounds gives a different picture:

```text
cube555: 105.59 / 66.43 = 1.59 x lower bound
cube666: 168.99 / 102.20 = 1.65 x lower bound
```

If the cube555 mean is scaled only by the intrinsic lower-bound ratio, it predicts:

```text
105.59 * (102.20 / 66.43) = 162.46 moves
```

The actual cube666 mean is 168.99, only 6.53 moves above that crude prediction. Relative to
the information-theoretic floor, the cube666 file is only about 4% less efficient than the
cube555 file.

This comparison is not a proof about optimal distances: the test sets contain shallow
scramble ladders, counting bounds are not exact distances, and the two solution generators
are different. It is nevertheless strong evidence against the claim that the current
cube666 paths are mysteriously inflated by 50--60 moves. Most of the increase is what the
group size predicts.

## 3. “The cube555 model solves” does not mean “the cube555 model is short”

The strongest cube555 file and our documented cube555 neural model should not be conflated.
The 109,288 file is an external community artifact. Its generation method is not established
by our controlled experiments.

Our own documented cube555 neural campaign found model paths with a mean around 169.7 moves.
The model was valuable because it reached the exact goal on many deep states, not because
those paths were close to optimal. A single frame had only roughly one-third success
probability; multiple symmetry frames raised aggregate coverage substantially.

This gives the most useful like-for-like observation:

```text
cube555 neural path when successful: approximately 170 moves
cube666 current classical path:        approximately 169 moves
cube666 neural path on a deep PID:     none
```

The surprising difference is therefore binary reachability, not the length of an already
found path. Cube555's model can wander inefficiently for about 170 moves and still enter the
exact endgame. Cube666's model loses its direction signal before it reaches the endgame, so
allowing 300, 500 or 680 steps does not help.

## 4. Why another 30--40 moves creates a sharp neural-search cliff

Beam search does not pay linearly for additional solution depth. It must preserve at least
one productive ancestry through every selection layer. A slightly unreliable policy can
survive a 70-move problem with a very wide beam and several independent frames, then fail
almost completely on a 102--110-move problem.

The measured deep action signal illustrates the cliff:

| random-walk depth | cube666 action top-1 | chance |
|---:|---:|---:|
| 40 | 0.090 | 0.028 |
| 60 | 0.043 | 0.028 |
| 72 | 0.035 | 0.028 |
| 85 | 0.029 | 0.028 |

By depth 85 the primitive Q model is at essentially chance action selection, while a typical
cube666 state is still materially outside the endgame. The remaining search is not a modest
linear extension: with an effective branching factor near 29, 30 additional uninformed
layers contain on the order of `29^30` possible continuations.

Worse, the heuristic is not merely flat. The full-width descent probe showed structured
optimistic error:

- at widths 2^21 and 2^22, predicted `min Q` fell from roughly 75 to 11--13;
- no run entered the exact depth-4 ball, even after 500 steps;
- the 2^22 frontier ended with about 1.3 correctly placed stickers, versus a random-state
  expectation around 9.

The wider beam searched more effectively for states on which the model was maximally wrong.
This explains why a million-wide or four-million-wide beam does not repair cube666.

## 5. What does not explain the failure

The evidence rejects several simpler explanations.

### The step cap was too short

Definitive whole-PID runs used approximately 665--682 steps on representative deep PIDs.
They still found no solution. The model is not merely trying to produce a 300-move path in a
250-step run.

### The beam was too narrow

Widths through 2^22 were tested. Wider search amplified value error instead of restoring
progress.

### We needed a larger flat model

Increasing capacity from approximately 26.4M to 119.9M parameters did not restore deep
action discrimination or full-PID solving. This is consistent with a supervision problem:
a larger network can fit the available target more accurately, but the target does not
contain the missing global direction.

### More symmetry frames would automatically rescue it

Frames multiply a non-zero per-frame success probability. They helped cube555 because its
single-frame probability was already substantial. The cube666 descent curves show no such
near-success mechanism to compound.

### Imitating the classical trajectory should be enough

The classical path is a verified upper bound, not a shortest-distance label. Memorising its
states teaches a narrow corridor in an enormous group. It can reproduce known paths without
learning how to rank unseen off-path states or how to discover a cheaper continuation.

## 6. The label-horizon interpretation

The primitive Q training source labels predecessor and successor actions on random walks
from solved. This is useful while walk index remains correlated with true distance. Near
mixing, two nearby walk states can have almost the same unknown true distance even though
their walk indices differ. The constructive direction then stops being a reliable global
distance direction.

Cube555's typical horizon is just short enough that residual action preference survives and
width plus frames can exploit it. Cube666 requires another roughly 35 true moves, but the
measured action preference has collapsed by then. Larger capacity, more training on the same
labels and self-Bellman backup from the same miscalibrated value function do not introduce
new information.

This is also consistent with the macro experiments:

- short-horizon macro states can be solved reliably through depth 4 and mostly through
  depth 5;
- models can reproduce represented teacher trajectories;
- models have found a few verified shorter local windows;
- generalisation on clean unseen depth-6 states remains weak;
- locally improved action ranks do not survive multi-parent beam competition;
- self-generated optimistic Bellman targets fail replay-based capability gates.

The model has local competence. It lacks independently trustworthy multi-step return
information for off-teacher states.

## 7. What a 110k cube666 score means

The score is not simply 1,012 times one path length because shallow pids keep their shorter
fallback paths. From the measured score-cap table:

| uniform deep-state solve length | cube666 total |
|---:|---:|
| 100 | 95,830 |
| 120 | 113,702 |

Interpolating puts a 110k total at roughly 116 moves per fully mixed state. That is only
about 14 moves over the 102.2 counting lower bound, approximately 1.13 times the floor.

By contrast, the external cube555 file's mean is approximately 1.59 times its counting
floor. Therefore “110k on cube666” is not the analogue of “109k on cube555.” It asks for a
far more efficient solver in normalized terms. Merely transferring the current cube555
neural behaviour would not reach it.

The 110k target may still be attainable because a counting bound is not an algorithm and
classical phase structure can be very efficient. But it should be treated as a near-optimal
search target, not as the expected result of making the existing model somewhat stronger.

## 8. Consequences for the model-based strategy

The comparison suggests the following division of labour.

1. **Use the flat random-walk model as a local optimiser, not a global cube666 compass.**
   Its demonstrated range is useful for windows, endgames and bounded projected stages.
2. **Obtain global reachability from structure.** A classical reduction, stabilizer chain,
   exact projected search or pattern-database-guided stage must carry the solve through the
   part where the learned flat value is untrustworthy.
3. **Use the classical submission as more than an imitation corpus.** It can define valid
   stage boundaries, alternative subproblems and starting upper bounds. To beat it, training
   labels must include independently verified off-path continuations, failures or lower
   bounds—not only the teacher's chosen next action.
4. **Train on multi-step verified returns.** The missing target is: among competing states
   reached by model search, which ones truly admit a short completion under a stronger,
   independent oracle?
5. **Gate on untouched full scrambles.** Local rank, imitation accuracy and training-split
   recovery are diagnostics. Promotion requires a unique unseen solve or a replay-verified
   path shorter than the known upper bound.

## 9. The clean mental model

I would summarise the comparison this way:

- **Intrinsic geometry:** cube666 really is about 1.5 times longer than cube555 in the
  competition metric.
- **Current path quality:** 169 versus 106 moves is broadly consistent with that scaling;
  the classical cube666 solver is not obviously defective from this comparison alone.
- **Neural solvability:** cube555 sits just inside the useful horizon of the learned scorer;
  cube666 sits outside it. This creates an abrupt solve-rate collapse rather than a gradual
  path-length regression.
- **Model size:** not the binding constraint under current supervision.
- **110k:** requires roughly 116-move deep solves and is much more ambitious than matching
  the normalized quality of the 109k cube555 file.
- **Required breakthrough:** reliable global or staged supervision for off-path states, not
  more imitation or a wider beam around the same erroneous value function.

## Source notes

- Structural counts and score-cap arithmetic: `BIGCUBES_PLAN.md`.
- Cube555 neural results and path statistics: `CUBE555_PROGRESS.md`.
- Cube666 full-width failure instrumentation: `CUBE666_REBUTTAL_AND_PROBE_RESULTS.md`.
- Cube666 model and hierarchy rationale: `CUBE666_MODEL_STRATEGY.md`.
- Latest short-macro and transition-ranker gates: `EXPERIMENTS.md`, dated 2026-08-24.
