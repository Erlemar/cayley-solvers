# A universal diameter upper bound with a linear error term

Date: 2026-09-19. This strengthens the earlier O(n^(3/2)) upper bound.
Its coefficient C=4 has since been improved to C=3/4 in
[COMBINED_EXTRACTION_UPPER_BOUND.md](COMBINED_EXTRACTION_UPPER_BOUND.md). The balanced-lift
decomposition proved here is used in that stronger result.

Let G_n be the inverse-closed set of consecutive wrapped 4-cycles, and
let D(n) be the diameter of Cay(S_n,G_n). Positions are fixed; rotation
of the entire permutation is not a free operation.

## Theorem

For every n >= 5,

\[
\boxed{D(n)\le
\left\lfloor\frac{\lfloor n^2/4\rfloor+12n-16}{3}\right\rfloor
\le \frac{n^2}{12}+4n-\frac{16}{3}.}
\tag{1}
\]

Together with the previously proved lower bound, this gives

\[
\left\lceil\frac{n^2}{12}\right\rceil+\mathbf1_{3\mid n}
\le D(n)\le
\left\lfloor\frac{\lfloor n^2/4\rfloor+12n-16}{3}\right\rfloor.
\tag{2}
\]

In particular D(n)=n^2/12+O(n). This does not determine the linear
coefficient or the exact diameter. It proves that C=4 is admissible in
an upper estimate n^2/12+Cn+B.

The argument is constructive. It uses the known cyclic adjacent-swap
sorting theorem and a fully enumerated base of 120 permutations on five
positions. All those finite identities are supplied as certificates.

## 1. Sorting an interval efficiently

Write I(p) for the ordinary inversion number of a list p of m distinct
entries. In this section all generators are contained in the interval;
there is no wrapping across its endpoints.

**Lemma 1.** For every m >= 5, the list can be sorted with at most

\[
\frac{I(p)}3+2m-4
\tag{3}
\]

consecutive 4-cycles and inverses.

We use i+ for a left rotation of the four entries beginning at position
i, and i- for its inverse. Thus

\[
[a,b,c,d]\xrightarrow{i^+}[b,c,d,a].
\]

At each step place the smallest remaining entry into the first remaining
position. Suppose its current displacement from that position is
j=3q+r, with 0 <= r <= 2. Move it left by three q times, using q inverse
4-cycles. Each removes exactly three inversions. For the last r places,
apply the forward 4-cycle at the first remaining position r times.
All these moves are inside the remaining interval if it has at least
six entries, the range in which we use this step.

Let delta be the total decrease in ordinary inversion number, and c the
number of moves in this placement. The residual cases are:

- r=0: c=q and delta=3q, so 3c-delta=0.
- r=1: [a,x,b,c] becomes [x,b,c,a], with x the smallest. Its inversion
  decrease is at least -1. Thus 3c-delta <= 4.
- r=2: [a,b,x,c] becomes [x,c,a,b]. The two inversions involving x are
  removed, and the exchanges of c with a,b contribute at least -2.
  Thus the decrease is at least zero and 3c-delta <= 6.

After fixing one entry, repeat on the remaining interval. Stop at five
entries. If I_5 is their inversion number, the preceding placements cost

\[
3L_{\rm before}\le I(p)-I_5+6(m-5).
\tag{4}
\]

For the final five entries, explicit words satisfy 3L_5 <= I_5+18.
Here is the complete summary of the finite certificate by inversion
number; each entry is the largest word length among the supplied words
with that inversion number.

| I_5 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L_5, at most | 0 | 5 | 6 | 7 | 4 | 5 | 6 | 5 | 4 | 5 | 6 |

The file [five_position_certificates.json](five_position_certificates.json)
contains one explicit word for each of the 120 permutations, using only
the generators starting at positions 0 and 1. Inverting each word gives
a sorting word of the same length. The independent checker
[verify_linear_bound.py](verify_linear_bound.py) verifies that all 120
permutations are present exactly once, replays every identity, and checks
3L_5-I_5 <= 18. No optimality claim for these finite words is needed.

Adding this base bound to (4) gives

\[
3L\le I(p)+6(m-5)+18=I(p)+6m-12,
\]

which proves Lemma 1.

## 2. Balanced lifted target positions

We recall the form of the cyclic adjacent-swap theorem used here.
For an array p on n fixed cyclic positions, assign integers w_i congruent
to p_i modulo n, with sum w_i=n(n-1)/2. They are lifted target positions.
The displacements d_i=w_i-i can be chosen so that max d_i-min d_i <= n.
Starting with such a balanced choice, repeatedly swapping adjacent
out-of-order lifted targets gives a minimum adjacent-swap sorting and
uses at most floor(n^2/4) swaps. These facts follow from Lemma 2 and
Theorems 4 and 7 of van Zuylen, Bieron, Schalekamp and Yu,
[An Upper Bound on the Number of Circular Transpositions to Sort a
Permutation](https://arxiv.org/pdf/1402.4867).

For clarity about wrapping, extend w periodically by

\[
w_{i+n}=w_i+n.
\]

An adjacent step is allowed in this sorting when w_i>w_{i+1}. At the
boundary it compares w_{n-1} with w_0+n, and replaces the window entries
by w'_{n-1}=w_0+n, w'_0=w_{n-1}-n. This is an ordinary cyclic adjacent
swap of the underlying labels.

The balanced choice is constructive: start with w_i=p_i. If two
displacements differ by more than n, subtract n from the larger and add
n to the smaller. Their absolute-value sum strictly decreases, so this
terminates. Congruences and the total sum are preserved.

## 3. A reduced adjacent-swap sorting with three stages

The following structural observation lets us use Lemma 1 twice.

**Lemma 2.** A balanced lift admits an adjacent-swap sorting consisting of:

1. sorting one window of length n on a line;
2. moving one entry clockwise by q positions, with 0 <= q <= n-1;
3. sorting an interval of length n-1 on a line, with a fixed entry as its cut.

If the inversion numbers of the two linear stages are I_1 and I_2, then

\[
I_1+q+I_2\le\left\lfloor\frac{n^2}{4}\right\rfloor.
\tag{5}
\]

**Proof.** First sort w_0,...,w_{n-1} into increasing order
z_0<...<z_{n-1}, using ordinary inversion-removing swaps. Let t=z_{n-1}.
In a swap of two out-of-order lifted targets, the larger displacement
decreases by one and the smaller increases by one. The displacement
range cannot increase. Hence the sorted window still has displacement
range at most n, implying

\[
t-z_0\le2n-1.
\tag{6}
\]

Let q be the number of entries z_i<t-n. They form an initial segment.
There is no entry equal to t-n because residues are distinct. Replacing
those q entries by z_i+n makes the window's entries precisely

\[
t-n+1,t-n+2,\ldots,t.
\]

Indeed, (6) puts every replacement in this interval, and all n residues
are represented once. Comparing sums gives

\[
\frac{n(n-1)}2+nq=nt-\frac{n(n-1)}2,
\qquad t=n-1+q.
\tag{7}
\]

The entry t is at position n-1. Move it clockwise across those q entries
in the following period. Each is less than t in the lift, so these q
adjacent swaps all remove inversions. Its final lifted position is
n-1+q=t, exactly its target.

Moreover this entry, and every periodic copy of it, is now in the
correct order relative to every other entry: before the move no earlier
entry was larger, and the move passed every later entry smaller than it.
It therefore separates the remaining lifted array into intervals of
length n-1, each containing exactly its own consecutive target integers.
Sort any one such interval on a line. The periodic copies undergo the
same operations, so this sorts the original cyclic permutation exactly.

All three stages use only adjacent swaps of out-of-order lifted targets.
The balanced cyclic sorting theorem therefore bounds their total length
by floor(n^2/4), proving (5). Notice that no free rotation or change of
the target permutation has been used. QED.

## 4. Converting the three stages to allowed moves

For n >= 6, apply Lemma 1 to the first interval of length n and to the
last interval of length n-1. Their costs are at most

\[
L_1\le I_1/3+2n-4,\qquad
L_2\le I_2/3+2n-6.
\tag{8}
\]

For the middle stage, three consecutive swaps moving the same entry
clockwise are one allowed 4-cycle. For any leftover one or two swaps,
the following local five-position identities suffice:

| Operation on an identity array of length five | Word | Cost |
|---|---|---:|
| [1,0,2,3,4] | 0-, 1-, 0-, 1+, 1+ | 5 |
| [1,2,0,3,4] | 1+, 1+, 0-, 1- | 4 |

These identities also work when the five-position interval crosses the
cyclic boundary. They realize exactly the required insertion and fix
all other entries. Thus the middle cost satisfies

\[
L_0\le q/3+14/3.
\tag{9}
\]

The actual implementation also uses a shorter four-move identity for
a residual displacement of four; this improves particular words but is
not needed in (9).

Combining (5), (8), and (9),

\[
L_1+L_0+L_2
\le\frac{I_1+q+I_2}{3}+4n-\frac{16}{3}
\le\frac{\lfloor n^2/4\rfloor+12n-16}{3}.
\]

Word length is integral, giving (1) for n >= 6. For n=5 the finite base
gives an upper bound of seven moves, already below the right side of (1).

The 4-cycle sorting itself need not remove inversions at every move.
Equation (5) concerns the hypothetical reduced adjacent-swap sorting;
Lemma 1 accounts explicitly for all additional inversion changes in the
implemented 4-cycle words. This distinction is essential to the bound.

## 5. Verification and remaining problem

- All 120 finite five-position words were independently checked.
- The complete new sorting construction was independently replayed on
  every permutation for 5 <= n <= 8: 46,200 permutations in total.
- Another 1,788 instances were checked, including reversals, half rotations,
  and random permutations, covering every n=5..150 and n=250,500,1000.
- Every run checked the stage decomposition (5), both interval estimates,
  the middle-stage estimate, and the resulting total bound.

The algorithm is in [linear_error_sort.py](linear_error_sort.py); results
are recorded in `linear_error_sort_checks.json` and
`linear_bound_exhaustive_checks.json`. The earlier run-based construction
is in `affine_insertion.py` and is superseded for the universal upper bound.

The original exact-diameter objective remains open. In particular,
neither C=1/6 nor the candidate D(n)=ceil(n^2/12)+1_{3|n} has been proved
or disproved by this upper bound. The progress here is the improvement
of the proved error term from O(n^(3/2)) to O(n), with an explicit C=4.
