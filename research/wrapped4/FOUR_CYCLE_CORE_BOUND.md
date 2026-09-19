# Central rotation followed by an arbitrary 4-cycle

Let tau_a(i)=i+a modulo n, let c be any 4-cycle on four distinct positions
(not necessarily consecutive), and put

\[
K=\lfloor n^2/4\rfloor,\qquad
h(k)=\lfloor k/3\rfloor+(k\bmod3),\qquad
H(n)=\lceil n^2/12\rceil+\mathbf1_{3\mid n}.
\]

**Theorem.** For n>=12 and a=floor(n/2) or ceil(n/2),

\[
\boxed{h(K-3)\le |\tau_a c|_4\le H(n).}
\tag{1}
\]

If n is congruent to 2,3,4 modulo six, parity sharpens this to

\[
\boxed{|\tau_a c|_4=h(K-3)=H(n)-1.}
\tag{2}
\]

For residues 0,1,5, the proved interval consists of the two possibilities
h(K-3) and h(K-3)+2=H(n). We do not claim that the longer supplied words
are optimal when the lower bound is two moves smaller.

This is a complete computer-assisted proof for this family, using
positive finite identities and induction in the unmarked gaps. It does
not prove an upper bound H(n) for arbitrary permutations.

## 1. Lower bound and reduction to finite bases

The fixed-deficit core theorem in ROTATION_ASCENT_LEMMA.md gives
ell(tau_a c)=K-3, where ell is cyclic adjacent-transposition length.
Each generator is odd and is a product of three adjacent transpositions,
so |tau_a c|_4>=h(K-3).

Describe c by its permutation alpha of four marked positions, in their
cyclic order, and let g_0,...,g_3 be the intervening unmarked gap lengths.
There are six choices for alpha, reduced to three types by cyclic relabelling.
Retain a gap of length zero or one unchanged; in any longer gap retain
2+((g_i-2) mod 3) positions. The reduced size m is at most 20.

Restore enough triples to obtain the smallest n0>=max(12,m) congruent
to n modulo six. Then 12<=n0<=23 and n0<=n. The remaining number R
of triples is even; each central rotation block must grow by 3R/2.

Thus every target reduces to a core and gaps at one of the finite bases
characterized by n0-6<max(12,m). Dihedral conjugation and inversion
preserve the generator set and transport both words and the anchor
property described below. There are exactly 1705 base orbits.

## 2. Monotone anchors and the complete certificate

The strand-inflation lemma of TWO_DEFECT_EXACT.md, Section 2, applies
without change to this four-position core. In an unmarked gap, one
monotone strand of positive travel a permits an insertion of three fixed
positions increasing b=n-a by three, at a cost of a letters. One monotone
negative strand of travel b permits increasing a by three, at a cost of b.
The lemma includes the winding correction and fixes the actual identity.

When both strands are available in every gap to be grown, allocate R/2
insertions to each direction, distributing them among the prescribed gaps.
The selected same-direction strands cross zero times, and opposite-direction
strands once. These properties persist under inflation, so the construction
can be repeated arbitrarily many times.

The certificate `four_cycle_core_certificate.json` has 1720 nodes:
1705 roots and 15 additional bases. Each contains a sorting word of length
at most H at its size. Of the nodes, 1717 have both anchors in every gap
of length at least two. Three roots instead have partial anchor coverage.
Up to the geometric symmetries their core/gap patterns are

| n0 | Core in one-line notation | Gaps |
|---:|---|---|
| 12 | (1,2,3,0) | (2,2,2,2) |
| 12 | (1,3,0,2) | (2,2,2,2) |
| 13 | (1,2,3,0) | (2,2,2,3) |

If only covered gaps must grow, use the node's word and its available
anchors. Otherwise choose a growing uncovered gap i and consume one
triple there. Since the total number of remaining triples is positive
and even, a second triple is available in some gap j, possibly the same
gap. Consume it as well and use the separately certified word at that
child of size n0+6.

The certificate includes every such child involving an uncovered gap,
up to symmetry: 15 edges and 15 added nodes in all. Every child has all
required anchors, so no second refinement is necessary. The point case
R=0 is covered by the parent's own word. The largest base size remains 23.

This proves complete coverage of arbitrary target gaps; it does not
depend on a negative search assertion. The precise sufficient refinement
rule is also stated in SPARSE_GAP_CERTIFICATION.md.

## 3. Propagate H(n) through the inflation

Increasing both central blocks by three changes n to n+6 and adds
exactly n+3 letters. On the other hand,

\[
H(n+6)=H(n)+n+3.
\]

More generally, for R/2 insertions of each direction, the total added
length is (K-K0)/3=H(n)-H(n0), independent of their order or gap allocation.
Every certified base word therefore extends to a word of length at most
H(n), proving the upper bound in (1).

Direct residue arithmetic gives H(n)=h(K-3)+1 in classes 2,3,4 modulo
six, and H(n)=h(K-3)+2 in the other three classes. Word length has the
parity of K-3, hence the intermediate value h(K-3)+1 is impossible.
This proves (2) and the two-value statement following it.

## 4. Independent audit and explicit constructor

`verify_four_cycle_core.py` uses only standard-library code and the
elementary replay/symmetry routines from the independent defect-two audit.
It imports neither the new search nor either gap enumerator or constructor.
It independently enumerates all six 4-cycles on marked position subsets,
obtaining 14982 models and exactly the same 1705 root orbits. It then:

* replays all 1720 words, comprising 37998 moves, and checks their lengths;
* measures anchor directions, travel, and every required pairwise crossing;
* regenerates all 15 required edges from uncovered gaps;
* checks that the directed certificate covers all roots, is closed, and
  terminates with anchored leaves.

The complete report is `four_cycle_core_audit.json`.

`four_cycle_core_construct.py` implements the induction for unrestricted
gap lengths. Independent full replay of 2254 constructed words through
n=151 exercises 150 refinements and 183658 moves. The results are in
`four_cycle_core_construct_checks.json`. These large examples check the
implementation; the finite audit and induction are the all-size proof.

## 5. Remaining diameter question

Together with THIRD_LAYER_NONCENTRAL.md, this removes all noncentral
families and the single-4-cycle central family from possible violations
of D(n)=H(n) in layer ell=K-3 (for n>=13). The central disjoint
3-cycle/transposition family and the central noncrossing triple-matching
family remain unsettled, as do deeper auxiliary layers. The universal
upper bound remains the one with linear coefficient 3/4.
