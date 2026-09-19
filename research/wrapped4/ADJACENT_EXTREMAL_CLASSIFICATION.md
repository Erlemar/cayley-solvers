# Which permutations maximize the auxiliary cyclic adjacent-swap distance?

Write ell(p) for word length in the cyclic adjacent transpositions, and
K_n=floor(n^2/4). This is the auxiliary metric, not the requested 4-cycle
metric. We prove the following structural statement for every n >= 2.

**Theorem.** The permutations with ell(p)=K_n are exactly the rotations

\[
\boxed{p(i)=i+a\pmod n,\qquad
a\in\{\lfloor n/2\rfloor,\lceil n/2\rceil\}.}
\tag{1}
\]

There is one such permutation for even n and two for odd n. Thus the
previous complete enumerations at n=10 and n=11 reflect an all-n fact.

A stronger stability result is now proved in
[ROTATION_ASCENT_LEMMA.md](ROTATION_ASCENT_LEMMA.md): an element of
length K_n-delta is at most delta arbitrary transpositions from a
rotation. That proof also classifies the next three adjacent-length
layers and supplies an alternative proof of (1).

## 1. Exact distance after moving one entry of a rotation

Let tau_a=[a,a+1,...,n-1,0,...,a-1], with 0 <= a <= n-1. For
0 <= l <= n-1, let q_(a,l) be the array obtained by rotating its first
l+1 entries one place to the left: the first entry moves l positions
clockwise. Then

\[
\boxed{\ell(q_{a,l})=a(n-a)-l+2\max(0,l-a).}
\tag{2}
\]

The same formula holds for an interval beginning at any cyclic position,
by conjugating by a rotation, which preserves the generating set.

**Proof.** Put b=n-a. A rotation tau_a has length ab. In a balanced
lift, its displacement at each position may be chosen as either a or
-b, with exactly b choices of a and a choices of -b. Such a lift has
sum of displacements zero, range n, and represents an optimal cyclic
adjacent-swap sorting.

If l <= a, choose displacement a for the entry at position 0 and -b
for the entries at positions 1,...,l. Moving the first entry right
across them is a sequence of l inversion-removing swaps in this lift.
At the step crossing position j, its lifted target a exceeds the other
target j-b, since 1 <= j <= l <= a < n. Each step decreases ell by one.
Hence ell(q_(a,l))=ab-l. The case a=l=0 is immediate.

If l > a, the same final array is obtained from tau_(a+1) by rotating
the suffix at positions l,...,n-1 one place to the right. This moves
one entry counterclockwise across r=n-1-l entries. The rotation has
parameters a+1 and b-1, and r <= b-1. Applying the preceding decreasing-
swap argument with directions reversed gives

\[
\ell(q_{a,l})=(a+1)(b-1)-r=ab+l-2a.
\]

This proves (2). The algebraic change of base rotation does not treat a
rotation as a free operation: its full distance is included in each
displayed calculation.

The standard balanced-lift criterion used here is Theorem 4 of
[van Zuylen, Bieron, Schalekamp and Yu](https://arxiv.org/html/1402.4867).

**Corollary.** If q is a nonrotation that becomes cyclically increasing
after deletion of one entry, then

\[
\ell(q)\le K_n-1.
\tag{3}
\]

Indeed, such an array is obtained by moving one entry of some rotation.
It can be represented as a left rotation of a cyclic interval of length
l+1 with 1 <= l <= n-2. A full interval merely changes the global rotation
and is excluded. In the case l <= a, (2) is at most K_n-1. Otherwise,

\[
\ell(q)=a(n-2-a)+l
\le\left\lfloor\frac{(n-2)^2}{4}\right\rfloor+n-2=K_n-1.
\]

## 2. Delete an extremal-displacement entry

We use the following consequence of Lemmas 5-6 and the proof of Theorem 7
of the same paper. From an optimal adjacent-swap sorting on n positions,
one can delete an extremal-displacement entry k and all swaps involving
it. The resulting sorting on the remaining cyclically ordered labels
is optimal, has balanced displacement range at most n-1, and removes
at most floor(n/2) swaps from the original sequence. Its initial and final
circular orders are the original orders with k deleted.

This is a statement about deleting crossings from an adjacent-swap
sorting. It does not assert that generators of a smaller 4-cycle graph
can be embedded across a gap at no cost.

We now prove (1) by induction. For n=2 it is immediate. Suppose
ell(p)=K_n, and perform the deletion above. Write L' for the length of
the remaining optimal sorting and t for the number of deleted swaps.
Since

\[
K_n=L'+t\le K_{n-1}+\lfloor n/2\rfloor=K_n,
\]

both inequalities are equalities. In particular L'=K_(n-1). By the
inductive hypothesis, the remaining initial array is a rotation of its
remaining final array. Therefore deleting k from p leaves the labels in
increasing circular order.

If p were not a rotation, (3) would give ell(p) <= K_n-1, a contradiction.
So p is a rotation. Finally, ell(tau_a)=a(n-a) equals K_n precisely for
the two central choices in (1), which coincide when n is even.

## 3. Consequence and limitation for the 4-cycle diameter

The maximal layer of the auxiliary metric is now completely classified.
Its 4-cycle distances are already given by the exact rotation theorem in
[DIAMETER_STATUS.md](DIAMETER_STATUS.md).

This does not classify the 4-cycle diameter elements. Such elements can
lie below the maximal adjacent-distance layer: the exhaustive n=12
calculation already exhibits this. To prove a sharp 4-cycle upper bound,
one must also control lower adjacent-distance layers, or find a different
global argument.

[check_adjacent_extremal_classification.py](check_adjacent_extremal_classification.py)
independently checks (1) and (2) against complete cyclic adjacent-swap
BFS at n=2,...,8. These finite checks validate the formulas; the induction
above is the proof for arbitrary n.
