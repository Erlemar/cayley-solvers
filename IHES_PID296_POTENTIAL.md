# Mathematical lower-bound functions for IHES puzzle 296

The requested goal is an explicit function F with F(solved)=0, F(296)=22,
and F(s)-F(move(s)) <= 1 for every legal move and state. Such a function would
prove the valid 22-move path optimal. **No new function reaching 22 was found.**
The existing full-search proof remains the basis for the optimality claim.

## An explicit coefficient function with a global certificate

For each physical piece p, let j_p(s) be its slot and o_p(s) its orientation.
There are three piece orbits: corners (8 pieces, 3 orientations), edges (12,2),
and centers (6,4). Define

    S(s) = sum_p C[p,j_p(s),o_p(s)] - sum_p C[p,j_p(solved),0]
    F(s) = max(0, S(s)/1000005).

The 624 integer coefficients are in `data/ihes_pdb/pid296_piece_potential.json`.
Their array convention is `[orbit][piece][slot][orientation]`, using the physical
piece conventions of `scripts/build_picture_cube_kpuzzle_v2.py`.

This function is independently evaluable from the configuration and those fixed
coefficients. It does not use the full puzzle distance or the 22-move proof.
For puzzle 296:

    S(296) = 7545297
    F(296) = 7545297/1000005 = approximately 7.54525927.

Consequently it proves at least 8 moves, not 22. The synthesis LP found an
unrounded objective of approximately 7.54529324; that floating optimization
result is not used to certify validity of the rounded function.

### Why no move can decrease F by more than one

For every generator m and orbit, the certificate supplies integer values
u[m,piece] and v[m,slot]. For every piece, slot and orientation the verifier checks

    C[p,j,o] - C[p,j_after_m,o_after_m] <= u[m,p] + v[m,j].

Every valid configuration uses each piece and each slot exactly once per orbit.
Adding the inequalities therefore gives

    S(s)-S(m(s)) <= sum_orbits(sum_p u[m,p] + sum_j v[m,j]) <= 1000005.

This holds for ALL legal configurations. The certificate even allows arbitrary
orientations and assignments beyond those actually reachable, so parity and
orientation restrictions cannot invalidate the conclusion. The inverse moves
are included too. Dividing by 1000005 gives a one-move bound; taking max with zero
preserves it. Subtracting the solved coefficient sum ensures F(solved)=0.

The independent checker verifies all **11,232** local inequalities using exact
integer arithmetic, plus each move budget and the start/goal values. It does not
invoke the linear-programming optimizer. Its callable `F(s)` returns an exact
Python Fraction from a 72-entry facelet state.

    .venv/Scripts/python.exe scripts/52_verify_ihes_potential.py

Synthesis is reproducible with `scripts/51_synthesize_ihes_potential.py`.
Correctness still assumes the puzzle-to-piece encoding is implemented correctly;
this is a finite checked certificate, not a formal proof-assistant development.

## A stronger existing projected-state function: 12

Let A(s) be the exact minimum number of outer-layer quarter turns needed to solve
the abstract state consisting of all corners and the total center orientation
modulo four. Let B(s) be the minimum number of middle-slice quarter turns needed
to solve just the center permutation. Define H(s)=A(s)+B(s).

- An outer turn leaves the center permutation unchanged, so it leaves B fixed;
  it changes A by at most one.
- A middle turn leaves corners and total center orientation unchanged, so it
  leaves A fixed; it changes B by at most one.
- Both values are zero at the solved state.

Thus H is also a globally valid one-move lower bound. For puzzle 296, the existing
abstract-distance tables give **A=10, B=2, H=12**. These are distances in smaller
projections; this is not the circular definition F=the original puzzle distance.
It nevertheless uses precomputed tables, rather than a short elementary formula.
The values are recorded in `data/ihes_pdb/pid296_existing_bound.json`.

Taking max(F,H) is valid too, but still gives only 12 at puzzle 296. Neither
construction supplies the desired independent analytic lower bound of 22.
