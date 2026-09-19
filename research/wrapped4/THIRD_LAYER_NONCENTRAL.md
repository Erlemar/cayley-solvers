# Exact lengths for the noncentral part of the auxiliary layer K-3

Put K=floor(n^2/4) and h(k)=floor(k/3)+(k mod 3). The metric ell uses
cyclic adjacent transpositions; |.|_4 uses the wrapped consecutive
4-cycles and their inverses. Write tau_a(i)=i+a modulo n.

**Theorem.** All the following permutations have exact 4-cycle length
h(K-3):

* n=2m>=14: tau_(m-1)sigma and tau_(m+1)sigma, where sigma is any
  3-cycle or any product of two disjoint noncrossing transpositions;
* n=2m+1>=13: tau_(m-1)t and tau_(m+2)t, where t is any transposition.

By [ROTATION_ASCENT_LEMMA.md](ROTATION_ASCENT_LEMMA.md), these are precisely
the noncentral families in the layer ell=K-3. The central families in
that layer remain unresolved in general. In particular this theorem
does not determine the diameter.

The lower bound h(K-3) follows from the finite-core adjacent-length
formula, the odd parity of each generator, and its adjacent length three.
It remains to construct words of that length.

## 1. Even n and a gap of length at least six

The core has t=3 or 4 marked positions. Let g_i be their cyclic unmarked
gap lengths. Suppose n>=16 and some g_i>=6. Delete three unmarked
positions in that gap. The new size n0=n-3 is odd and at least 13.

For the target a=m-1 choose base parameters

\[
(a_0,b_0)=(m-1,m-2).
\]

For the target a=m+1 choose instead

\[
(a_0,b_0)=(m-2,m-1).
\]

Both base rotations are central. Apply the exact central defect-two
construction from [TWO_DEFECT_EXACT.md](TWO_DEFECT_EXACT.md). The shortened
gap has length at least three. The anchor property proved in Section 1
of [TWO_DEFECT_UNBALANCED.md](TWO_DEFECT_UNBALANCED.md) supplies both a
positive and a negative monotone strand there.

In the first case inflate the positive strand, increasing b0 by three
and adding a0 moves. In the second case inflate the negative strand,
increasing a0 by three and adding b0 moves. This restores precisely
the original gap, core, and rotation. The winding correction in the
inflation lemma ensures the word ends at the actual identity.

Since h(k+3s)=h(k)+s, the resulting word has length

\[
h((m-1)(m+1)-2)=h(m^2-3)=h(K-3).
\]

## 2. The finite even boundary

If all gaps are at most five, then n=t+sum(g_i)<=6t<=24. Thus Section 1
leaves only these finite cases:

* all core and gap patterns at n=14;
* n=16,18,20,22,24 with every gap at most five.

For each even size it suffices to use a=m-1. Inversion sends a=m+1
to a=m-1, rotating and inverting the relative core. The listed core
family is invariant under those changes.

`third_layer_noncentral_words.json` gives 218 representatives under
dihedral conjugation and inversion, with a word for each:

| n | Representative orbits | Length of every supplied word |
|---:|---:|---:|
| 14 | 123 | 16 |
| 16 | 45 | 21 |
| 18 | 30 | 26 |
| 20 | 14 | 33 |
| 22 | 5 | 40 |
| 24 | 1 | 47 |

Every length equals h(K-3). This includes the cases where the diameter
candidate H(n) would allow two extra moves; no extra moves were needed.

## 3. Odd n

For odd n>=15, both block sizes of the noncentral rotations are at least
six. The one-transposition theorem in
[STRAND_INFLATION.md](STRAND_INFLATION.md) gives

\[
|\tau_{m-1}t|_4=|\tau_{m+2}t|_4
=h((m-1)(m+2)-1)=h(K-3).
\]

At n=13, cyclic conjugation and inversion reduce the question to
tau_5(0 d), d=1,...,6. The same certificate supplies six words of length
13=h(39), covering that boundary. This finishes the all-size proof.

## 4. Independent checks and the remaining family

`verify_third_layer_noncentral.py` uses the standard-library replay and
full-permutation symmetry routines of the independent central checker.
It imports neither the new root generator nor either search implementation.
It enumerates marked position subsets, including both noncrossing pairings,
and obtains exactly the same 224 orbits as the certificate. It replays all
4480 moves and checks every word attains the lower bound. The complete
report is `third_layer_noncentral_audit.json`.

`third_layer_noncentral_construct.py` constructs words for arbitrary
allowed n, gaps, and either noncentral rotation. Independent replay of
1346 complete words through n=151 exercises 275 finite lookups, 791
single-inflation constructions, and 280 uses of the one-transposition
theorem. All 245838 moves replay correctly. These are implementation
checks; the reduction and complete finite coverage prove the theorem
for unrestricted n.

For n>=13, an element of the auxiliary layer K-3 that could exceed H(n)
must therefore belong to its **central** family: a central rotation
followed by a 4-cycle, a disjoint 3-cycle and transposition with the stated
cyclic adjacency condition, or a noncrossing triple matching. These
families are listed precisely in ROTATION_ASCENT_LEMMA.md. Deeper layers
remain uncontrolled as well. The universal C=3/4 bound is unchanged.
