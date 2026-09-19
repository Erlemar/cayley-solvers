# 666 from scratch: what got built, and the wall the classical route hits

2026-08-22. Work log for the from-scratch solver attempt. Code in `cube_nnn/`.
Companions: `CUBE666_WINDOW_PLAN.md`, `CUBE666_REBUTTAL_AND_PROBE_RESULTS.md`,
`BIGCUBES_PLAN.md`.

**Bottom line: no working solver. The infrastructure is built and verified, and
the reason the classical search route stalls is now measured rather than
guessed. It is a structural wall, not a tuning failure.**

---

## 1. Target, restated exactly

Score model validated against the shipped data (sample total reproduces 502,399
exactly; our own 364,107 inverts to M=475, matching the file's median 447 /
max 562):

| | total | moves/pid |
|---|---|---|
| counting bound | -- | 103 |
| **180k goal** | 180,000 | **198** |
| webmaking #1 | 194,001 | 216 |
| John McFacker | 355,425 | 459 |
| ours | 364,107 | 475 |

webmaking reached 194,001 in **one submission**, and hit 555/666/777 the same
day. That is a general-purpose solver run once, untuned -- so 216 is its
out-of-the-box output, not a tuned result.

---

## 2. Built and verified (`cube_nnn/`)

Everything checked against shipped competition data, not asserted.

| component | check |
|---|---|
| `puzzle.py` generators from 3D geometry | **24/24 exact n=4, 36/36 exact n=6**; n=5/7 free from the same code |
| `symmetry.py` 48 frames | 48/48 bijective relabel, solved-fixing, conjugation round-trip; **96/96** valid same-length solutions on a real pid |
| `reduction.py` G1 membership | 300/300 no false negatives, 0/500 no false positives |
| `endgame.py` G1 coordinates | 0 malformed over 200 random G1 elements |
| `g1_action.py` coordinate action | **0/400** mismatches vs real moves |
| `tables/corner_pdb.npy` | **88,179,840 / 88,179,840 filled, max depth 14**, 88 MB, 185 s |
| `pdb.py` orbit PDBs | k=4 exact: 255,024 = 24*23*22*21, calibrated h ~ 1.75*d |
| data pipeline | 1012/1012 replay, sample total 502,399 exactly |

The corner table filling **100% with zero holes** is independent proof the
coordinate and action chain is correct -- a bug leaves unreachable states.

### Structure, measured

    branching        n=5 24.092, n=6 29.023   (matches BIGCUBES to 3 dp,
                                               independently derived)
    parity           invariant exists iff n is ODD. n=6 is 6 odd / 12 even
                     forward gens (12/24 with inverses). n=5, n=7 all odd.
    G1 = <12 outer turns>   0 centre stickers cross faces -> G1 IS the 3x3x3
                            supercube group, 76.23 bits
    endgame bound    21.3 moves (12 generators); realistic optimal ~26-32
    reduction        [G:G1] = 424.4 bits -> bound 87.3 moves
    two-phase total  108.6 vs global 103.0  ->  phasing costs only 5.4%

That last line is the good news and it still stands: the decomposition is nearly
free, so every move lost is search inefficiency, not phase structure.

---

## 3. Three bugs, two caught only by verifying

1. **Corner orientation chirality.** Ordering a corner's 3 stickers by face
   index flips handedness between corners, so orientation stops composing
   additively: **235/400** action mismatches. A right-handed cyclic order
   (det of the normals > 0) fixes it -> 0/400. Caught before spending BFS
   compute, because the action was verified first.
2. **Beam dedup in Python.** `[i for i in uniq if hh[i] not in seen]` loops over
   millions of candidates per step. Prune first, dedup the survivors.
3. **Integer overflow in the non-backtracking sentinel.** Banning a move with
   `h = int64max//4`, then `key = h*4096 + jitter`, wraps NEGATIVE -- banned
   children sort FIRST and fill the beam with backtracking moves. Fixing it
   moved the face-rung plateau from h=5 to h=2. Use a small sentinel (1<<40).

---

## 4. The wall

### 4.1 What was tried

Target: one face's 16 centres (4 pieces from each of the 4 centre orbits),
counting bound **14.8 moves**, heuristic = exact k=4 PDBs, one per orbit,
each measuring exactly its own 4 target pieces.

| variant | result |
|---|---|
| sum of components, strict top-B | descends 22 -> 2, then hard plateau, 0/3 |
| max of components | 0/3 |
| max*10 + sum tiebreak | 0/3 |
| stratified selection, level_cap 0.25 / 0.05 / 0.01 | 0/3, 0/3, 0/3 |
| widths 4096 / 8192 / 16384 | 0/3 throughout |

Also 0 on the single-orbit rung (24 pieces, bound 16.3) and on the G1 endgame
(0/4 on random G1 elements at width 16384).

### 4.2 Why -- and it is not tuning

**No exact table covers more than about 6 pieces.** A k-piece placement table on
a 24-slot orbit costs 24*23*...*(25-k) entries: k=6 is 96.9M (buildable), k=7 is
1.7e9, k=8 is 2.96e10. So any target involving more than ~6 pieces MUST be
scored by combining several component tables. And then:

* **`sum` over-punishes.** Near the goal most components read 0; the move that
  fixes the last one breaks the others, so every escape scores worse. The beam
  saturates with basin states. Measured: clean descent 22 -> 2, then flat
  forever. This is the same mechanism as `kept 24/24/0` in the rung-composition
  failure, now reproduced classically with *exact* tables.
* **`max` under-informs.** Admissible and valley-free, but it saturates at the
  component diameter (8 for k=4, 9 for the edge tables, 14 for corners) against
  targets of 15-32 moves. Almost no gradient.
* **Selection policy does not help.** Stratified selection across h-levels was
  the direct fix for the local minimum and is a clean negative once the beam is
  actually kept full (all caps ran ~215 s, so the beam was not starved).

Escaping the valley needs a ~8-12 move commutator; at branching 36 that is
36^10 = 3.6e15, far past any beam width.

### 4.3 The consequence worth carrying forward

The rung models (`rung_o5.pt`, `rung_o6.pt`) solve a random 24-slot centre orbit
**24/24 in 18.8-22.3 moves** -- inside 1.15-1.37x of the 16.3 bound. A learned
scorer approximates the JOINT distance directly, which no combination of exact
component tables can do at any feasible table size.

**So on centre reduction the learned scorer is not a convenience, it is doing
something exact tables provably cannot.** That is the opposite of the usual
"PDBs beat learning" intuition and it is the main technical finding here.

The endgame is different: it fails for a plain range reason (corner abstraction
diameter 14 vs a 26-32 move problem), and the standard fix is Kociemba-style
two-phase, which works because phase 2's subgroup is generated by cheap moves
and so never pays the preservation cost.

---

## 5. What to do next, in order

1. **Two-phase endgame** (~1 day). Phase 1 to `<U,D,L2,R2,F2,B2>`; pruning
   tables (corner-orientation x slice) = 1.08M and (edge-orientation x slice) =
   1.01M, both trivially buildable. Caveat: with only half turns on L/R/F/B
   those face rotations move in steps of 2 and cannot reach odd values, so the
   supercube coordinate interacts with the subgroup choice -- this is why
   supercube solvers are rarer than colour-cube ones. Needed by every route.
2. **Rung models for centres**, composed with the verified G1/reduction
   skeleton. Score the JOINT coset distance with a mask-conditioned head, never
   a sum of per-orbit heuristics -- s4.2 is now the classical confirmation of
   why the sum fails.
3. **GPU beam.** The numpy engine runs ~265k states/s; production needs 2^21.
4. **96-frame portfolio** on whatever solver exists. Free and cannot fail: a
   deterministic solver conjugated by 48 symmetries x inverse gives 96
   independent lengths, verified 96/96 valid and same-length.

Do NOT re-run: sum/max/stratified aggregation of component PDBs on a multi-piece
centre target (s4.1), or further width escalation on it.

---

## 6. Reproduce

    .venv/Scripts/python.exe cube_nnn/scripts/01_verify_structure.py
    .venv/Scripts/python.exe cube_nnn/scripts/02_verify_data.py
    .venv/Scripts/python.exe cube_nnn/scripts/03_verify_reduction.py
    .venv/Scripts/python.exe cube_nnn/scripts/05_endgame.py
    .venv/Scripts/python.exe cube_nnn/scripts/06_build_corner_pdb.py   # 185 s
    .venv/Scripts/python.exe cube_nnn/scripts/07_endgame_pdb.py
    .venv/Scripts/python.exe cube_nnn/scripts/08_face_rung.py

---

## 7. The endgame: Thistlethwaite, 4 phases (2026-08-22, later)

A working -- but not yet complete -- endgame solver for G1 (the 3x3x3 supercube
in the 12 outer turns). Scripts 09-14, tables in `cube_nnn/tables/`.

| phase | target | table | status | cost |
|---|---|---|---|---|
| 1 | H = <U,D,L,R,F2,B2> | 8,192 / 16,384 = 2^11 x 4 exactly | EXACT, verified | mean 6.2, max 9 |
| 2 | G2 = <U,D,L2,R2,F2,B2> | 4,330,260 / 4,330,260 (100%) | EXACT, verified | mean 10.4, max 13 |
| 3 | G3 = <U2,D2,L2,R2,F2,B2> | 2,508,800 / 5,017,600 | coordinate INCOMPLETE | mean 17.2 |
| 4 | identity | 5,308,416 reachable, max 16 half turns | EXACT, cross-validated | mean 22.8, max 32 |

**Phases 1+2 alone: 20/20 reach G2, all replay-verified, 16.1 QTM.**
**Full chain: 12/24 random G1 states fully SOLVED and replay-verified, mean 56.7
QTM** (counting bound 21.3; true optimal ~26-32, so Thistlethwaite's ~1.8x
penalty is as expected). The other 12 get stuck at the phase 3 -> 4 boundary.

### Everything here was DERIVED, and each derivation caught a real error

* **Corner orientation, U/D axis.** Ordering a corner's stickers by face index
  flips handedness between corners so orientation stops composing additively:
  235/400 action mismatches. Right-handed cyclic order (det of normals > 0)
  fixes it -> 0/400, then 480/480 and 960/960 on the subgroup check.
* **Edge orientation.** On a 6x6 the wings are ORIENTATION-FREE (2 orbits of 24,
  pure permutation), so the standard per-facelet F/B convention does not
  transfer -- all 4096 per-slot assignments fail. Derived instead by asking
  which flips are reachable inside H for each (home, slot) pair: 0 of 144 pairs
  admit both, so a separable invariant exists, and it is
  `o = flip XOR sigma(home) XOR sigma(slot)` with
  `sigma = [0,1,1,0,0,1,1,1,0,0,0,1]`. Verified H-invariant 1500/0.
* **G3 structure**, by random walk inside it: corner tetrads {0,3,5,6} /
  {1,2,4,7}; edge orbits {0,2,6,10}, {1,3,8,11}, {4,5,7,9} (the last IS the
  UD-slice); face rotations in {0,2} only.
* **Rotations must reach 0, not merely even.** Phase 4 reaches only 8 of 64
  rotation codes; leaving them at (2,2,2,2,2,2) lands outside G3 and phase 4
  reads 255.

### The open gap, stated precisely

Phase 3's coordinate (corner tetrad x edge tetrad x rotation code x parity x
UD-slice parity) does NOT determine the phase-4 coset. G3 sits at index 96
inside the within-tetrad group. Known constraints: only 8 of 32 parity
signatures occur (`p(cA)=p(cB)` and `p(e0)^p(e1)^p(e2)=0`, i.e.
parity(corners)=0 and parity(edges)=0), and only 32 of the 8 x 64
(signature, rotation) pairs are valid.

Mitigation in place: phase 3 is retried with randomised descent plus a short
random G2 prefix, since different phase-3 solutions land in different cosets.
That fixes half the states and no more -- **the split is exactly 12/24, i.e. one
binary condition**, and it is deterministic per state.

Falsified as the missing bit (measured, none separates success):
parity(corners), parity(edges), UD-slice permutation parity, and each of the six
face rotations individually. Tracking parity(corners) alone instead of the sum
produced **byte-identical** output -- rule 28's unwired-flag signature -- so that
is not it either.

**The proper fix** is the standard construction rather than more ad-hoc bits:
phase 3's coordinate should be the corner-permutation coset modulo G3's corner
subgroup and the edge-permutation coset modulo G3's edge subgroup, ranked
directly, instead of tetrad membership plus parity patches.
