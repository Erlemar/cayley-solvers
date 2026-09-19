# Rotations followed by an arbitrary transposition

Let tau_a(i)=i+a modulo n, put b=n-a, and let t be any transposition,
not necessarily between adjacent positions. Let ell denote cyclic
adjacent-transposition length, and let |.|_4 denote the metric generated
by the wrapped consecutive 4-cycles and their inverses. Define

\[
h(k)=\min\{L\ge0:3L\ge k,\ L\equiv k\pmod2\}
=\lfloor k/3\rfloor+(k\bmod3).
\]

## Theorems

For every 1 <= a <= n-1 and every transposition t,

\[
\boxed{\ell(\tau_a t)=ab-1.}
\tag{1}
\]

The full wrapped-metric theorem is now proved for every a,b >= 6:

\[
\boxed{|\tau_a t|_4=h(ab-1)\quad\text{for every transposition }t.}
\tag{8}
\]

The interval construction below handles most cases. The remaining
short separations are settled by the strand-inflation induction in
[STRAND_INFLATION.md](STRAND_INFLATION.md), with three explicit base
words and careful control of the cyclic origin.

For reference, the interval construction alone gives, for a,b >= 6,

\[
\boxed{h(ab-1)\le |\tau_a t|_4\le h(ab-1)+2.}
\tag{2}
\]

Moreover, if a and b are not congruent to the same nonzero residue
modulo 3, the sharper equality holds:

\[
\boxed{|\tau_a t|_4=h(ab-1).}
\tag{3}
\]

In particular, for every half rotation a=floor(n/2) and every n >= 12,
the new theorem (8) gives equality with h(floor(n^2/4)-1).
Since H(n)=max(h(floor(n^2/4)),h(floor(n^2/4)-1)),

\[
|\tau_{\lfloor n/2\rfloor}t|_4=h(\lfloor n^2/4\rfloor-1)\le H(n).
\tag{4}
\]

Thus no family consisting of a half rotation followed by one arbitrary
transposition can witness a positive linear term in the diameter. This
does not rule out other families, including products of many
transpositions, and does not prove D(n)=H(n).

An intermediate refinement of the interval construction, established
below, handles all even half rotations of circular separation at least 3:

\[
\boxed{|\tau_{n/2}t|_4=h(n^2/4-1)\qquad(n\ge12\text{ even}).}
\tag{7}
\]

The short separations formerly left by these sufficient conditions
are all settled by (8): separation 2 when n=2 modulo 6, and separations
1 and 2 when n=4 modulo 6. The new proof also covers unequal a,b.

## 1. Reducing to an exchange with two marked entries

Conjugation by a cyclic rotation preserves both generating sets and
commutes with tau_a. Rotate the positions until the two positions of t
lie in different blocks of the array

\[
[a,\ldots,a+b-1,\ 0,\ldots,a-1].
\]

Such a cut exists: among all cyclic intervals of any fixed size strictly
between zero and n, there is one containing exactly one of two distinct
positions. Consequently it is enough to consider the array P(a,b,x,y)
obtained by exchanging the entries a+y and x, where 0 <= x < a and
0 <= y < b. Equivalently, start with two increasing labelled blocks
A B, exchange the blocks, and exchange one marked entry from each block.

Before this last exchange there are ab ordinary inversions. Between the
two marked positions, every B-entry has value larger than a+y, and every
A-entry has value smaller than x. Their total inversion contributions
are unchanged by the exchange. Only the inversion between the marked
entries disappears. Thus

\[
I(P)=ab-1.
\tag{5}
\]

For the raw lift w_i=P_i, unmarked displacements w_i-i are a or -b.
The two marked displacements are x-y and a+y-b-x, both lying between
-b and a. Their range is therefore at most a+b=n.

Ordinary bubble sorting of this lift swaps each pair at most once.
Jerrum's minimum-length criterion, stated as Theorem 4 of
[van Zuylen et al.](https://arxiv.org/html/1402.4867), consequently proves
that this sorting is also minimal among cyclic adjacent-swap words.
Together with (5), this proves (1). Since every allowed 4-cycle is odd
and is a product of three cyclic adjacent swaps, (1) also proves the
lower bound in (2).

## 2. Four extensions by a triple

Suppose a word exchanges A,B and exchanges their two marked entries.
Let T be a block of three unmarked entries. Each of the following
extensions uses exactly one extra move per entry of the opposite block:

* From T A B, first apply the old word on A B, then move T across the
  resulting first block of length b. Cost increase: b.
* From A T B, first exchange T with B in b moves, then apply the old
  word on A B. The final T is already at the end of the A-block.
* From A T B, with T now belonging to the beginning of the B-block,
  first exchange A with T in a moves, then apply the old word on A B.
* From A B T, first apply the old word, then exchange the resulting
  A-block with T in a moves.

Each exchange of a triple with one entry is one permitted 4-cycle.
The entries of T preserve their order and are never marked. These
identities hold for arbitrary contents of A and B; in particular they
remain valid after the marked entries have been exchanged.

The comparison bound changes by exactly the same amount:

\[
h((a+3)b-1)=h(ab-1)+b,\qquad
h(a(b+3)-1)=h(ab-1)+a.
\tag{6}
\]

Starting from any a,b >= 6, remove triples before or after the marked
entry of a block until its size is one of 6,7,8. This is always possible:
when a block has size at least 9, its two unmarked portions have total
size at least 8, so one of them has at least three entries. The same
argument applies to the other block. Reversing these removals and using
the four extensions reduces (2)-(3) to finitely many bases.

## 3. Complete finite base certificate

There are

\[
\sum_{a,b\in\{6,7,8\}}ab=(6+7+8)^2=441
\]

marked base configurations. The file
[twisted_exchange_certificates.json](twisted_exchange_certificates.json)
contains a sorting word for every one, using only nonwrapped 4-cycles
inside its interval. All words have length at most h(ab-1)+2. All 328
bases needed for (3) have length exactly h(ab-1).

| a | b | Number of bases | At lower bound | Two moves above lower bound |
|---|---|---:|---:|---:|
| 6 | 6 | 36 | 36 | 0 |
| 6 | 7 | 42 | 42 | 0 |
| 6 | 8 | 48 | 48 | 0 |
| 7 | 6 | 42 | 42 | 0 |
| 7 | 7 | 49 | 45 | 4 |
| 7 | 8 | 56 | 56 | 0 |
| 8 | 6 | 48 | 48 | 0 |
| 8 | 7 | 56 | 56 | 0 |
| 8 | 8 | 64 | 58 | 6 |

The two-move excess in this table concerns these interval words. It is
not a lower bound in the wrapped graph. In particular, separate wrapped
words attain the lower bound for the exceptional separations tested at
n=14 and n=16.

The checker in [twisted_exchange_check.py](twisted_exchange_check.py)
independently verifies complete base coverage, replays each interval
word, and checks its length. It also constructs and independently replays
7350 extended words with both block sizes between 6 and 40. The latter
checks supplement the four all-size identities; the finite size range
is not the justification for induction.

The certificate and the four extension identities prove (2) and (3).
The six residue classes of n give (4): when n=0,1,3,5 modulo 6 the
sharper equality (3) applies, and when n=2,4 modulo 6 we have
h(floor(n^2/4)-1)=H(n)-1.

The exact diameter remains unresolved.

## 4. Retaining triples proves the separation refinement

Write n=2m and let 3 <= d <= m be the circular separation of the
transposition. After rotation of the positions and, if necessary,
reflection of the circle, take the marked block indices x=0 and y=m-d.
Conjugation by reflection preserves the generators and commutes with
the half rotation.

Put c=6+(m mod 3). Reduce both blocks to size c, keeping x_0=0 and
choosing a retained B-index y_0 congruent to y modulo 3 with

\[
\max(0,c-d)\le y_0\le\min(y,c-1).
\]

These inequalities say exactly that the removed prefix and suffix have
nonnegative lengths divisible by three. Initially choose the largest
possible y_0. The finite certificate shows that all bases (c,c,0,y_0)
attain the lower bound except y_0=5 for c=7, and y_0=6,7 for c=8.

If one of these indices occurs, replace y_0 by y_0-3. This remains
feasible: for (c,y_0)=(7,5) or (8,6), the congruence forces d=2 modulo 3,
so d >= 5; for (c,y_0)=(8,7), it forces d=1 modulo 3, so d >= 4.
The new index is respectively 2,3,4 and is sharp. The triple-extension
identities now prove (7). For c=7 and d=1, the retained index is 6,
which is also sharp, giving the stated additional adjacent case.
