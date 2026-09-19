## Short version

The 176,889 solution does not use a neural model. It is a classical exact-sticker solver adapted from KMCoders’ Santa 2023 cube solver, combined with our own optimal corner solver, correctness patches, path reduction, per-state ensembling, and full replay verification.

Its central trick is to avoid searching directly through the enormous 6×6×6 state space. Instead, it:

1. Solves the eight corners optimally.
2. Represents everything else as six permutations of 24 physical pieces.
3. Repairs their parity.
4. Uses simulated annealing to find a short partial solution.
5. Decomposes the remaining error exactly into 3-cycles.
6. Uses a 20,000-wide beam to insert exact 3-cycle algorithms where they cancel best with the existing path.
7. Keeps the shorter of this result and the existing fallback for every puzzle.
8. Replays every move against all 216 stickers.

No model checkpoint or learned heuristic is involved.

## 1. What the puzzle actually is

Every test state contains 216 uniquely labelled stickers. The target is the exact identity arrangement:

\[
(0,1,2,\ldots,215)
\]

This is stronger than solving an ordinary colour cube. Two stickers of the same apparent face colour are not interchangeable: every individual sticker must return to its precise target position.

There are 36 legal directed moves:

- 18 positive quarter turns: six layers on each of three axes.
- Their 18 inverses.

Every primitive slice quarter-turn costs one move. A half-turn therefore costs two moves.

The local replay convention is:

\[
\text{new\_state}[i] = \text{old\_state}[\text{generator}[i]]
\]

Using the opposite permutation convention would produce plausible-looking but invalid paths.

## 2. The structural decomposition

Rather than hard-coding a conventional cube layout, our local code discovers the invariant sticker orbits directly from the supplied move permutations.

The 216 stickers split into nine orbits of 24 stickers:

- One corner-sticker orbit:
  - 24 stickers
  - eight physical corners × three stickers.
- Four centre orbits:
  - four independent permutations of 24 centre pieces.
- Four wing-sticker orbits:
  - paired into two physical wing clusters.
  - each physical cluster contains 24 two-sticker wing pieces.

After pairing the wing stickers, the non-corner problem becomes exactly six permutations of 24 physical objects:

\[
4\text{ centre clusters} + 2\text{ wing clusters}
\]

This is the important compression. We no longer think about 192 independent non-corner stickers. We think about six structured \(S_{24}\) permutations.

The discovery and parity logic is in [classical.py](C:/Users/and-l/cayley/src/cube666/classical.py).

## 3. Exact optimal corner solving

Only outer face turns affect the corners. The corner group has:

\[
8! \times 3^7 = 88,179,840
\]

states, which is small enough for exact coordinate search.

Our corner solver uses IDA* with several exact pattern databases:

- Complete eight-corner permutation distance.
- Complete corner-orientation distance.
- A four-corner location-plus-orientation PDB.
- A complementary PDB for the other four corners.

The heuristic is the maximum of those exact lower bounds, so it remains admissible. Search pruning removes:

- Immediate inverse moves.
- Three identical quarter turns, which should be one inverse turn.
- Duplicate orderings of commuting opposite-face turns.
- Coordinates already on the current search path.

Consequently, the returned corner path is optimal in the competition’s quarter-turn metric.

On all 1,012 states, the measured corner solution was:

- Mean: 10.35 moves.
- Maximum: 13 moves.

The implementation is [corners.py](C:/Users/and-l/cayley/src/cube666/corners.py).

This optimal path is passed into the external solver through `KMC_CORNER_PATH`. That matters because KMCoders’ built-in 2×2 corner routine assumes a particular face convention. It could produce valid paths after our other fixes, but its paths were longer: on state 500, the native corner route eventually produced 211 moves versus 197 with our exact corner injection under the matched smaller configuration.

## 4. Parity repair

An isolated 3-cycle is an even permutation. Therefore, a cluster with odd permutation parity cannot be finished using only 3-cycle algorithms.

After solving the corners, the solver calculates a six-bit parity vector:

\[
p=(p_0,p_1,\ldots,p_5),\qquad p_i\in\{0,1\}
\]

Each inner-slice turn has a known six-bit parity effect. The solver constructs a \(6\times12\) linear system over \(\mathrm{GF}(2)\) and finds an inner-slice sequence that makes all six cluster permutations even.

This is a tiny exact algebra problem, not a cube search.

The earlier structural scan found that an optimal parity repair needs:

- Zero moves for 218 states.
- One move for 520 states.
- Two moves for 274 states.

After this step:

- The corners are solved.
- Every remaining cluster permutation is even.
- The remainder is guaranteed to be expressible using isolated 3-cycles.

## 5. Simulated annealing creates the rough path

The solver does not try to finish directly from the corner-plus-parity state. That would require too many expensive 3-cycle algorithms.

Instead, it optimizes a sequence of inner-slice turns using simulated annealing. Its effective objective is:

\[
E(P) = |P| + \alpha\,M(P)
\]

where:

- \(|P|\) is the path length.
- \(M(P)\) is the number of exact stickers still misplaced.
- \(\alpha=3\) in the production run.

The annealer makes four types of mutation:

- Replace one inner-slice turn.
- Swap two adjacent turns.
- Insert two parity-compatible turns.
- Delete two parity-compatible turns.

The use of paired insertion/deletion preserves the parity class required by the exact finisher.

The temperature falls linearly from 3 to 0 over 20 million mutations. The result is not a solution; it is a comparatively short path that leaves a small, even residual.

The solver then tries additional bulk operations—parallel slice packs and corner-fixing commutators—at different positions in the current path. It keeps changes that repair many stickers for little net path cost.

This is where the method differs fundamentally from our failed primitive-move PDB beam. The annealer searches over structured sequences that can cross the 8–12-move valleys where a primitive greedy beam gets stuck.

## 6. Exact residual decomposition

For each of the six clusters, the solver now computes its exact remaining permutation and decomposes it into cycles.

An even permutation can be expressed as 3-cycles. The solver calculates the minimum unrestricted number of 3-cycle units needed. For a cycle decomposition, this is based on terms of the form:

\[
\left\lfloor\frac{\text{cycle length}}{2}\right\rfloor
\]

with paired handling of even cycles so the whole permutation remains even.

This count is an exact algebraic residual, but not an exact quarter-turn distance. It says how many abstract 3-cycle corrections are required, not how many primitive moves their algorithms will cost.

As an independent validation, our local workbench generated every directed isolated 3-cycle for every cluster:

\[
6 \times 2 \times {24 \choose 3}=24,288
\]

All of them have algorithms no longer than 14 quarter turns:

| Macro length | Count |
|---:|---:|
| 8 | 4,032 |
| 10 | 12,256 |
| 12 | 7,232 |
| 14 | 768 |

The production solver uses KMCoders’ native `rotate.txt` and `rotate_all.txt` tables, but our independently generated library confirmed that the required exact macro family really exists.

## 7. The 3-cycle insertion beam

Simply appending 29 ten-move algorithms would add roughly 290 moves. That would not be competitive.

The clever part is inserting each 3-cycle at the best location in the path.

Suppose the path is divided into prefix \(A\) and suffix \(S\), and we insert macro \(M\):

\[
A\,M\,S
\]

The final contribution of \(M\) is conjugated by the suffix. Therefore, the solver chooses a lookup-table macro whose conjugated effect performs the desired 3-cycle in the final residual coordinates.

That gives it freedom to insert a correction beside turns with which it can commute or cancel.

For every beam state, it considers:

- Every cluster that is not yet solved.
- Every 3-cycle that reduces the exact residual cost by one.
- Every possible insertion point in the current path.
- Alternative macro words from the lookup table.
- Boundary cancellations with adjacent parallel slice turns.

The production beam retains up to 20,000 alternatives. Beam states preserve different:

- Residual permutations.
- Orders of 3-cycle corrections.
- Insertion locations.
- Cancellation contexts.

Each accepted layer must reduce the recomputed exact algebraic cost. Consequently, after \(k\) residual units, the beam performs exactly \(k\) reducing insertions and finishes at zero residual.

### Representative state 500

The production trace illustrates the entire process:

- Corner plus parity setup: 11 moves.
- Annealed rough path: 67 moves.
- Exact misplaced stickers after rough phase: 78.
- Exact residual: 29 unrestricted 3-cycles.
- Final path after the insertion beam: 193 moves.

So 29 exact macros added only:

\[
193-67=126
\]

net moves, or about 4.34 moves per correction. Raw macros cost 8–14 moves, but insertion and cancellation recover most of that cost.

The exact-finisher trace descended monotonically:

\[
29\rightarrow28\rightarrow\cdots\rightarrow1\rightarrow0
\]

while selecting among many different path histories.

## 8. The critical integration bug

The largest breakthrough was correcting the move ordering.

Cayley’s JSON lists the positive move families in:

```text
f, r, d
```

But KMCoders’ hard-coded 3-cycle tables encode operation numbers in:

```text
d, f, r
```

If the original JSON ordering is preserved, ordinary moves still look legal because each supplied permutation is valid. But every lookup-table macro becomes attached to the wrong axis family.

That is a particularly nasty failure mode: the initial stages appear to work, while the exact macro finisher is operating in the wrong coordinate system.

The runner now serializes the 18 positive generators in explicit `d,f,r` order while retaining each move’s real Cayley permutation.

This correction changed state 500, under a five-million-step/beam-1,000 test, to 197 moves.

The integration is in [06_run_kmcoders_beam.py](C:/Users/and-l/cayley/cube666/scripts/06_run_kmcoders_beam.py).

## 9. Safety patches to the external solver

I also patched two places where the external beam could incorrectly believe it had progressed.

### Stale mismatch deltas

An insertion candidate stores a cheap predicted change in mismatches. After path overlap and cancellation, that prediction can become stale.

The original code could therefore report zero error for a path that did not actually solve the cube.

The patched version materializes the complete 216-sticker state and recomputes the exact mismatch count before ranking or termination.

### False algebraic reduction

Sometimes an inserted macro cancels completely against the surrounding path. Its net effect is then identity.

The original code could still subtract one from the residual cost, eventually reaching a fictional cost of zero while leaving the cube unchanged.

The patched solver recomputes the residual permutations after every insertion and rejects any child whose exact residual cost did not decrease.

These changes do not merely improve score; they make the solver’s completion condition trustworthy.

The patched Rust solver is [solve_cube_beam.rs](C:/Users/and-l/cayley/external/third_party/kmcoders_santa2023/solution/src/bin/solve_cube_beam.rs).

## 10. How the production parameters were chosen

Small matched probes were run before the full pass.

### Annealing weight

On state 500, with five million annealing steps and beam 1,000:

| Alpha | Final moves |
|---:|---:|
| 2 | 207 |
| **3** | **197** |
| 4 | 225 |
| 5 | 213 |
| 6 | 219 |

So \(\alpha=3\) was selected.

### Exact-finisher beam width

On state 800, using five million annealing steps:

| Beam | Final moves |
|---:|---:|
| 1,000 | 219 |
| 5,000 | 211 |
| **20,000** | **203** |
| 100,000 | 207 |

The search is not strictly monotonic in beam width because candidate truncation, deduplication and heuristic ordering interact. Beam 100,000 was also much slower. Beam 20,000 gave the best quality/runtime compromise.

### Annealing duration

On state 800:

- Five million steps, beam 20,000: 203 moves.
- Twenty million steps, beam 20,000: 189 moves.

That justified the final 20-million-step configuration.

The production settings were:

```text
corner source: exact
move order: d,f,r
annealing seed: 50001
annealing steps: 20,000,000
annealing alpha: 3
exact insertion beam: 20,000
workers: 16
threads per worker: 1
```

Sixteen single-threaded processes avoided nested parallelism and CPU oversubscription.

## 11. Production coverage and merging

The expensive solver was run on IDs 210–1011: 802 states. These are the deeper states where the existing paths were long enough for a roughly 196-move solver to help.

The raw production result for those 802 states was:

- Total: 157,113 moves.
- Mean: 195.90.
- Median: 196.
- Minimum: 174.
- Maximum: 214.
- Wall time: approximately 5 hours 4 minutes.

Running the classical solver on every shallow state would be counterproductive because its setup and exact-finisher overhead tends toward roughly 190–200 moves even when an existing shallow path is much shorter.

The final merge therefore considered, independently for every state:

1. The existing valid path.
2. Any KMC-produced path.
3. A commuting reduction of every candidate.
4. The shortest replay-valid result.

Consecutive turns on parallel slices commute. Thus:

```text
f1.f3.-f1
```

can be reordered and reduced to:

```text
f3
```

The reducer treats each consecutive same-axis block as an exponent vector modulo four, allowing cancellation to cascade across empty blocks.

The original 1,012 fallback paths had:

- Raw total: 502,399.
- Commuting-reduced total: 463,203.

The final selection was:

- 797 states won by the new classical solver.
- 215 states retained an existing path.
- IDs 0–209 contributed 19,992 moves.
- IDs 210–1011 contributed 156,897 moves.

Therefore:

\[
19,992+156,897=\boxed{176,889}
\]

The merge implementation is [07_merge_verify.py](C:/Users/and-l/cayley/cube666/scripts/07_merge_verify.py).

## 12. Verification

The external solver is treated as an untrusted proposal generator.

Before accepting a path, the runner:

1. Checks every move name against the supplied generator dictionary.
2. Starts from the exact test state.
3. Applies the complete path to all 216 stickers.
4. Requires exact equality with the identity target.
5. Writes the result atomically only after verification.

The final merger repeats replay verification:

- Before comparing each candidate.
- After commuting reduction.
- For every one of the 1,012 selected winners.

Final verification results:

- Replay-valid states: 1,012/1,012.
- Missing states: 0.
- Invalid states: 0.
- Total: 176,889.
- Mean: 174.79.
- Median: 193.
- Maximum: 214.
- Focused regression tests: 12/12 passed.

The resulting files are:

- [Submission CSV](C:/Users/and-l/cayley/submissions/cube666_classical_merged.csv)
- [Verification report](C:/Users/and-l/cayley/cube666/reports/classical_merged.json)
- [Production report](C:/Users/and-l/cayley/cube666/results/dfr_20m_b20k_full_a3_b20000/report.json)

## 13. Why this succeeded where the models failed

The models had to estimate a useful global distance or action preference on an astronomical state space. Empirically, their signal disappeared around depth 85, while many paths were hundreds of moves long. A beam cannot preserve a correct trajectory for hundreds of decisions when its ranking model is effectively at chance.

Teacher-path training did not solve that:

- High apparent accuracy came from memorizing narrow teacher corridors.
- The teacher paths were very loose upper bounds, not true distances.
- Those corridors occupy a negligible fraction of the full group.

The classical solver never asks, “Which primitive move globally approaches the target?”

Instead, it asks a sequence of much more tractable questions:

- What is the exact optimal corner solution?
- What parity class is each cluster in?
- Can annealing find a short state with a modest residual?
- What is the exact residual permutation?
- Which 3-cycle reduces its exact algebraic cost?
- Where can that 3-cycle be inserted most cheaply?

The hard global problem is converted into:

- One exact 88-million-state corner problem.
- One six-bit linear system.
- A stochastic sequence optimizer.
- About 25–35 guaranteed exact macro corrections.

That is why the solution generalizes to every test state without learning anything from a training distribution.

## What is exact and what remains heuristic

Exact or guaranteed:

- Cube structure and generator actions.
- Optimal corner paths.
- Cluster parity calculation.
- Residual permutations.
- 3-cycle correctness.
- Monotonic residual reduction.
- Final replay validity.
- Per-state shortest-choice merge among available candidates.

Heuristic or non-optimal:

- The annealed rough path.
- The choice of annealing seed and duration.
- The 20,000-wide insertion beam.
- The resulting quarter-turn path length.

So 176,889 is guaranteed locally valid, but it is not claimed to be globally optimal. Multiple annealing seeds and targeted reruns of the 205–214-move outliers should reduce it further. It also remains a local replay-verified score until the CSV is submitted and confirmed by Kaggle.