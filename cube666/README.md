# Classical 6x6x6 solver workbench

This directory contains a clean, data-driven classical solver for the exact-sticker
`cayley-py-666-cube` instance. It discovers the cube structure from
`puzzle_info.json`; no sticker indices are hard-coded.

## Current status

The pipeline is replay-valid end to end:

1. Discover nine invariant 24-sticker orbits.
2. Recover one corner orbit, four centre clusters, and two paired physical wing
   clusters.
3. Solve the eight corners optimally in the quarter-turn metric with an exact IDA*
   solver and pattern databases.
4. Repair the six cluster parities optimally with a search over the 64 possible
   parity vectors.
5. Apply short parallel commutators for bulk residual reduction.
6. Finish every even cluster permutation with a complete isolated 3-cycle library.
7. Insert each finishing macro at the best point in the existing path and replay all
   216 stickers to the identity target.

The complete finisher has all `6 * 2 * C(24, 3) = 24,288` directed cluster
3-cycles. Path lengths are:

- 4,032 macros of length 8;
- 12,256 of length 10;
- 7,232 of length 12;
- 768 of length 14.

This independently recovers the reported “all 3-rotations in at most 14 turns”
property of the strongest classical architecture.

## Measured facts on the real 1,012-state test set

The structural scan in `reports/structure_probe.json` found:

- optimal corners: mean 10.3547 moves, maximum 13;
- parity repair after corners: 0 moves for 218 states, 1 for 520, and 2 for 274;
- corner plus parity setup: mean 11.4101 moves;
- exact post-setup unrestricted 3-cycle residual: total 65,311, mean 64.5366,
  maximum 70.

The basic four-turn macro family contains 768 unique effects:

- 384 inner-inner commutators, each executing two parallel centre 3-cycles;
- 384 inner-outer commutators, each executing two 5-cycles and two 3-cycles,
  equivalent to six unrestricted 3-cycle units in four turns.

One setup conjugation expands this bulk library to 19,680 effects. On the first
three real states, strict greedy reduction removed 70 of 203 residual units, but
stalled with 43-47 units left. This confirms that a non-myopic search is required.

Two exact replay examples are saved in `results/`:

- state 0: 370 moves with greedy bulk reduction, versus 440 without it;
- state 1011: 366 moves, versus 1,000 in the sample submission.

These are diagnostic rows, not complete submissions.

## Commands

Always use the repository virtual environment:

```powershell
.venv\Scripts\python.exe cube666\scripts\00_probe_structure.py `
  --json-out cube666\reports\structure_probe.json

.venv\Scripts\python.exe cube666\scripts\01_probe_commutators.py `
  --conjugator-depth 1

.venv\Scripts\python.exe cube666\scripts\03_probe_nested_commutators.py `
  --inner-conjugator-depth 1 --finisher-conjugator-depth 3

.venv\Scripts\python.exe cube666\scripts\04_solve_classical_baseline.py `
  --start-index 1011 --limit 1

.venv\Scripts\python.exe -m pytest tests\test_cube666_classical.py -q
```

The complete finisher is cached as
`artifacts/three_cycle_library.json`. The file includes a SHA-256 digest of the
ordered move definition and is rejected if it does not match `puzzle_info.json`.

## What to implement next

The correctness components are no longer the bottleneck. The score bottleneck is
the bulk macro optimizer.

### 1. Simulated annealing over macro sequences

Use the 19,680 depth-1 bulk effects as the first move set. A search state is the six
24-element cluster permutations plus a macro sequence. Mutations should insert,
delete, replace, or move a macro—not merely append forever. Score the complete
candidate as:

```text
reduced bulk path length
+ estimated insertion cost for the exact residual
```

The exact algebraic residual is useful as a lower bound, but optimizing it alone
reproduces the current greedy local minimum. Periodically run the real insertion
finisher on the best candidates to recalibrate the estimate.

### 2. Add depth-2 conjugated bulk effects selectively

Do not materialize every raw conjugate. Retain effects only if they introduce a new
cluster-cycle signature or improve the shortest path for an existing signature.
Store physical-cluster permutations as 144-byte keys. This should give much broader
placement coverage without the memory cost of full 216-integer Python tuples.

### 3. Optimize insertion jointly with cycle decomposition

The current finisher chooses the first shortest algebraic reducer and then optimizes
its insertion time. Improve it with a small beam over alternative reducing
3-cycles. Beam state should contain the six residual permutations and current path;
branch score is the actual reduced path length after arbitrary-time insertion.

### 4. Scale only after an ablation passes

Use a stratified set containing shallow, medium, and deepest IDs. Compare:

- exact finisher only;
- greedy bulk plus finisher;
- annealed bulk plus finisher;
- annealed bulk plus beam insertion.

The target is below roughly 190 moves per puzzle on average before spending hours on
all 1,012 states. Always merge per-state minima with the sample or best existing
classical paths, and replay every final row before submission.
