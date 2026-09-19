# A finite-certificate method for central rotations with a fixed small core

This note describes a sufficient proof method. It does not assert that
the required finite words exist for every core, or that the diameter
conjecture has been proved.

Let alpha be a fixed permutation on t cyclically ordered marked positions,
with no fixed points. Insert g_i unmarked fixed positions after its i-th
position to obtain sigma on n=t+sum(g_i) positions. Consider

\[
p=\tau_a\sigma,\qquad a\in\{\lfloor n/2\rfloor,\lceil n/2\rceil\}.
\]

The existing strand-inflation lemma applies to any unmarked position,
independently of the core. A monotone positive strand has travel a and
grows the other block by three; a monotone negative strand has travel
b=n-a and grows a by three. The initial relabelling accounts exactly
for the cyclic-origin displacement. See TWO_DEFECT_EXACT.md, Section 2.

## 1. Finite roots

Fix a minimum permitted size N. In every gap retain

\[
r_i=g_i\quad(g_i<2),\qquad
r_i=2+((g_i-2)\bmod3)\quad(g_i\ge2).
\]

Let m=t+sum(r_i)<=5t. Choose the smallest n0>=max(N,m) congruent to
n modulo six. As m is already congruent to n modulo three, this entails
restoring triples only. Moreover

\[
n_0\le\max(N+5,5t+3),\qquad n_0\le n.
\]

The upper bound follows by separating m<N from m>=N; in the latter
case n0 is either m or m+3. A base is irreducible precisely when

\[
n_0-6<\max(N,m).
\]

Restore the required triples in any gaps from which they were removed.
The remaining number R of triples is even. The central parameters each
differ from their original values by 3R/2.

For a defect delta core, the support theorem gives t<=2delta. Choosing
N=max(12,4delta) is sufficient to preserve the elementary rank conditions
for every such core, and all root sizes are at most max(17,10delta+3).
For delta=3, N=12 and root sizes are therefore at most 33.

## 2. Sufficient anchor and refinement certificates

At a base provide a word sorting its stated p to the identity, together
with selected unmarked anchors in some or all gaps of length at least two.
Each covered gap needs one monotone positive strand of travel a and one
monotone negative strand of travel b. Check that same-direction chosen
anchors cross zero times and opposite-direction ones once, in the
adjacent-exchange expansion of the word.

If all gaps with remaining triples are covered, insert half the triples
using positive anchors and half using negative anchors. The choices
among gaps are unrestricted. The anchor properties survive every insertion,
so this restores the target gaps and central rotation exactly.

If a growing gap i is not covered, insert one of its triples in the base,
and insert a second triple in any gap j where one remains; j=i is allowed.
Such a second triple exists because R is positive and even. Move to a
separately certified word for this child. Its size is n0+6, with both
central blocks increased by three.

Thus a node with missing anchors must supply all children obtained by
choosing a pair of potentially growing gaps, with repetition, at least
one of them missing an anchor. Cases where no triples remain are covered
by the node's own word. Cases where only covered gaps grow use that word
and inflation without refinement.

A complete finite directed certificate, whose edges all increase size
by six and whose leaves have all anchors, proves coverage for arbitrarily
large gaps. Symmetry identifications are permitted under dihedral
conjugation and inversion, with the sorting words and anchors transported.
No negative search conclusion is needed.

## 3. Propagation of the candidate diameter bound

Put

\[
H(n)=\lceil n^2/12\rceil+\mathbf1_{3\mid n}.
\]

Balanced inflation from n to n+6 adds exactly n+3 letters. Also

\[
H(n+6)-H(n)=n+3.
\]

Consequently, if every base word in a complete certificate has length
at most H at its size, the same is true for every inflated target in
that core family. It is unnecessary to attain the adjacent-length lower
bound h(ell) at every base. In fact that stronger statement fails for
some deficit-three permutations at n=12.

Every allowed generator is odd, so searches may restrict the budget to
the largest integer <=H(n) congruent to ell(p) modulo two. This restriction
is preserved by balanced inflation.

## 4. What has and has not been certified

The defect-two instance is complete: 942 roots, 947 total nodes, and
five refinement edges. Its words even attain h(K-2), yielding the
stronger exact theorem in TWO_DEFECT_EXACT.md.

For defect three, `sparse_gap_roots.py` generates 177291 rooted models,
42661 geometric orbits, and maximum root size 33. The seven cyclic core
types consist of all 4-cycles, the permitted disjoint 3-cycle/transposition
cores, and the two cyclic types of noncrossing triple matchings.

The geometric symmetry routine was compared completely with the independent
defect-two root set, and checked against full-permutation symmetries on
300 sampled deficit-three orbits. These checks do not by themselves
constitute a complete deficit-three coverage audit.

The deficit-three search certificates are still incomplete. A complete
independent root enumeration, replay, anchor audit, and child-coverage
audit will be necessary before applying the implication in Section 3
to any newly completed family.
