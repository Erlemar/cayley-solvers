# Exact lengths for arbitrary rotations with a defect-two core

Write `tau_a(i)=i+a mod n`, put `b=n-a`, and let sigma be either a
3-cycle or a product of two disjoint noncrossing transpositions. The
noncrossing condition refers to the cyclic order of their four positions.
Set

\[
h(k)=\lfloor k/3\rfloor+(k\bmod3).
\]

**Theorem.** If a,b>=12, then

\[
\boxed{|\tau_a\sigma|_4=h(ab-2).}
\]

The rotation need not be central. This follows from the complete
central-rotation certificate in [TWO_DEFECT_EXACT.md](TWO_DEFECT_EXACT.md),
with no further searched identities. It is an exact family result;
the full diameter remains undetermined.

## 1. An anchor property of the central construction

For every central target of this type, at every size n>=12, the explicit
construction in TWO_DEFECT_EXACT.md produces a shortest word with both
a positive and a negative monotone unmarked strand in **every gap of
length at least three**. Their travels are a and b respectively.

To see this, first inspect the finite certificate. Every leaf has both
anchors in every gap of length at least two. The only two nonterminal
words have missing anchors only in gaps of length exactly two. If such
a gap must grow, the construction passes to a certified anchored child.
If it does not grow, its final length stays two. All other anchors survive
inflation. The symmetry transports preserve the property as well.

This establishes the statement for every constructed central word, not
just the finite bases. The same-direction anchors never cross, and
opposite-direction anchors cross once, as required for repeated inflation.

## 2. Reduce the larger rotation block

Let c=min(a,b)>=12. There is a unique epsilon in {-1,0,1} congruent to
|a-b| modulo three. Replace the larger block by c+epsilon and keep the
smaller block equal to c. Call the resulting parameters a0,b0.

The new rotation is central: |a0-b0|<=1. Moreover

\[
n_0=a_0+b_0=2c+\epsilon\ge23,
\qquad n_0\le n,\qquad n-n_0\equiv0\pmod3.
\]

There are t=3 or 4 moved positions. Rotate their first position to zero,
and denote their cyclic unmarked gaps by g_i. Retain initially

\[
r_i=\begin{cases}
g_i,&g_i<3,\\
3+((g_i-3)\bmod3),&g_i\ge3.
\end{cases}
\]

The retained size m=t+sum(r_i) is at most 6t<=24 and is congruent to
n0 modulo three. It is therefore at most n0: the only potentially
problematic pair m=24,n0=23 has different residues modulo three.

Restore any (n0-m)/3 of the removed triples. Enough are available since
n0<=n. This gives base gaps with total size n0, and every gap still
requiring growth has retained at least three unmarked positions.

Apply the central theorem to these gaps and the central rotation a0.
By Section 1, every gap that still needs growth has both monotone anchors.

## 3. Inflate in one direction only

If a>=b, inflate negative anchors: each insertion of three unmarked
positions increases a by three, leaves b fixed, and adds exactly b moves.
If b>a, inflate positive anchors: each insertion increases b by three,
leaves a fixed, and adds exactly a moves. Allocate the insertions to the
prescribed gaps. Their anchors persist after each insertion.

The winding correction is the one proved in TWO_DEFECT_EXACT.md:
the initial inflated permutation is relabelled by the actual final
cyclic-origin displacement. This gives precisely the stated new rotation
parameter, with a word ending at the fixed identity. No rotation is
treated as a free move.

After all insertions we have exactly the original a,b and gaps. Finally,

\[
h(k+3s)=h(k)+s
\]

shows that each cost increment agrees with the required h(ab-2).
The finite-core formula gives adjacent length ab-2, so the parity and
three-adjacent-swaps lower bound matches this constructed upper bound.
This proves the theorem.

## 4. Verification and consequence for the next auxiliary layer

`check_two_defect_unbalanced.py` independently replays the output words
and measures strand motion using the standard-library checker of the
central certificate. The report `two_defect_unbalanced_checks.json`
records:

* all 947 finite nodes and their 1905 gaps of length at least three;
* 1790 complete words through n=151, covering both growth directions and
  every residue, with 257980 moves replayed;
* 4940 one-direction inflation steps and 59 central refinements;
* 72 additional anchor checks arising from the two exceptional bases.

These checks validate the constructor. Sections 1-3 are the argument
for unrestricted sizes.

For even n=2m>=26, the noncentral members of the auxiliary layer K-3
are tau_(m-1)sigma and tau_(m+1)sigma of the above type. Thus all of
these have exact 4-cycle length h(K-3). The central part of that layer
is not settled by this theorem.
