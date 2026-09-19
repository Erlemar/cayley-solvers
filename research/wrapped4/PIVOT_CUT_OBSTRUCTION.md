# Why one exact pivot insertion does not retain the required quadratic work bound

The three-stage balanced-lift construction has total hypothetical adjacent
work at most K=floor(n^2/4). A tempting simplification is to choose one
token x, move it directly to its correct fixed position by adjacent swaps
in one direction, then sort the other n-1 positions linearly after cutting
immediately after x.

This note proves that the adjacent-work count of that simplified scheme
can be asymptotically 5n^2/16, not n^2/4. It is an obstruction to this
work-accounting argument, **not a lower bound for the original Cayley
graph**, and does not rule out other constructions or uses of temporary
helpers across the cut.

## A four-block family

Let n=4b. Reverse each of the four consecutive b-position blocks, then
apply the half rotation. Explicitly, for i=qb+r, 0<=r<b,

\[
p(i)=((q+2)\bmod4)b+b-1-r.
\]

Define F(p) as the minimum, over all choices of token and both direct
directions, of

\[
\text{number of adjacent swaps in the pivot insertion}
\; +\;
\text{ordinary inversion number of the remaining cut interval}.
\]

**Proposition.**

\[
\boxed{F(p)=5b^2-2b+(b\bmod2).}
\tag{1}
\]

Thus, on even b,

\[
F(p)=\frac{5n^2}{16}-\frac n2,
\qquad F(p)-K=\frac{n^2}{16}-\frac n2.
\]

In particular no bound F(p)<=K+Cn+B with fixed constants C,B holds for
all permutations. One cannot replace the established three-stage
adjacent-work lemma by this simpler one-pivot assertion.

## Proof

Conjugation by a b-position rotation preserves p, so it suffices to
choose a token starting at position r in the first block. Its label and
target position are x=3b-1-r. The direct clockwise and counterclockwise
travel distances are

\[
d_+=3b-1-2r,\qquad d_-=b+1+2r.
\]

Use cyclic target ranks beginning at x+1 for the other n-1 labels.
After the counterclockwise insertion the remaining list consists of six
decreasing pieces A,B,C,D,E,F, with respective lengths

\[
(r+1,\ b,\ r,\ b-r-1,\ b,\ b-r-1).
\]

Their increasing value order is C,E,A,F,B,D. Writing a=r+1,
c=r, and d=b-r-1, the within-piece inversions total

\[
\binom a2+2\binom b2+\binom c2+2\binom d2,
\]

and their between-piece inversions total

\[
ac+ab+bc+b^2+2bd+d^2.
\]

Consequently the remaining inversion number is

\[
I_-(r)=4r^2+6r-4br+6b^2-7b+3.
\]

Adding d_- gives

\[
F_-(r)=4\left(r-\frac{b-2}{2}\right)^2+5b^2-2b.
\]

For the other insertion direction, the remaining list is obtained by
moving its first token to its end. That token has target rank b+2r;
therefore its inversion change is 4b-2-2(b+2r)=2b-2-4r. The difference
d_+-d_- has the same value. It follows that

\[
F_+(r)=F_-(r)+4b-4-8r
=4\left(r-\frac b2\right)^2+5b^2-2b.
\]

Minimizing these expressions over integer r in 0,...,b-1 gives a zero
square contribution when b is even and a contribution one when b is odd.
This proves (1), including b=1.

## Checks

`pivot_cut_probe.py` independently executes both insertions as ordinary
array swaps and counts the remaining inversions. Full enumeration through
n=8 already finds violations of F<=K, beginning at n=6. The structured
checks in `pivot_cut_block_obstructions.json` give, for example,
F=480 versus K=400 at n=40,b=10, and F=1960 versus K=1600 at n=80,b=20.
These agree with (1). The algebra above establishes the infinite family.
