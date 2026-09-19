# An alternating matching perturbation: a local obstruction

Let n=2m be divisible by four, with n >= 8, and define

\[
p(i)=i+m+(-1)^i\pmod n.
\]

This is a half rotation followed by the swaps (0,1),(2,3),...,(n-2,n-1).
The following bounds are proved:

\[
\ell(p)=m(m-1),\qquad
\boxed{|p|_4\ge 2\left\lceil\frac{m(m-1)+2}{6}\right\rceil.}
\tag{1}
\]

Here ell is cyclic adjacent-transposition length. The second bound uses
an obstruction beyond division of ell by three: **every first allowed
move changes ell by only one**, rather than being able to decrease it
by three.

## Proof

A balanced lift of the target positions is

\[
w_i=\begin{cases}i-m+1&i\text{ even},\\i+m-1&i\text{ odd}.
\end{cases}
\]

Its displacements are -(m-1) and m-1 and have range n-2. In the affine
inversion sum

\[
\Phi(w)=\sum_{i<j}\left|\left\lfloor\frac{w_j-w_i}{n}\right\rfloor\right|,
\]

same-parity pairs contribute zero. There are m(m-1)/2 contributions
from each order of opposite-parity pairs, giving Phi=m(m-1). The
balanced-lift minimum criterion proves the first equality of (1).

Conjugation by rotation through two positions fixes p. Reflection
i -> 1-i also fixes p and interchanges the two directions of a 4-cycle
while preserving the parity of its starting position. It therefore
suffices to examine left rotations of windows starting at 0 and at 1.

For a window starting at an odd position, the three adjacent swaps
change Phi by -1,+1,-1 respectively. The new displacement range is
n-1, so the lift remains balanced. The cyclic adjacent length is
therefore ell(p)-1.

For the window starting at 0, the initial lifted entries are

\[
[1-m,\ m,\ 3-m,\ m+2].
\]

The left rotation gives [m,3-m,m+2,1-m]. Rebalance by subtracting n
from the first entry and adding n to the fourth entry. The resulting
first four entries are

\[
[-m,\ 3-m,\ m+2,\ m+1].
\]

All later lifted entries are unchanged, and the displacement range is
now n, so this is a balanced lift. The six pairs inside the displayed
window contribute 3, compared with 2 before the move. Their total
contribution with entries outside the window is unchanged: aside from
reordering the four entries, one low value decreased by one and one
high value increased by one, and neither change crosses an external
multiple-of-n comparison threshold. Thus ell increases by one.

All 2n neighboring states consequently have cyclic adjacent length
ell(p)-1 or ell(p)+1. If a sorting word has L letters, its first letter
can remove at most one unit of ell, and each remaining letter removes
at most three. Hence

\[
m(m-1)\le1+3(L-1),\qquad 3L\ge m(m-1)+2.
\]

The permutation p is even, so L must be even. Rounding upward with
this parity gives (1).

## Exact result at n=24

At n=24 the lower bound is 46, and a word of exactly that length has
now been found and independently replayed. Therefore

\[
\boxed{|p_{24}|_4=46.}
\]

Here is the complete sorting word. An entry i+ left-rotates the four
positions i,i+1,i+2,i+3 modulo 24; i- is its inverse. Read from left
to right and apply to p_24:

```text
3- 5- 7- 0+ 23+ 9- 2- 6+ 13+ 3- 12+ 15-
9- 10- 5- 21+ 17- 0+ 11- 20+ 23+ 19+ 22+ 8+
18+ 12- 16+ 2- 13- 14- 21+ 11+ 5- 19+ 17+ 14-
8+ 7+ 4- 15- 11+ 10+ 7+ 5+ 3+ 1+
```

Only the first, second, and fourth moves decrease ell by one; every
other move decreases it by three. Thus its total decrease is
3*1+43*3=132, agreeing with ell(p_24).

The independent, standard-library-only checker
[verify_matching_certificates.py](verify_matching_certificates.py)
replays this word directly on the array, checks its length, and combines
it with the analytic lower bound (1). The canonical certificate is
[alternating_matching_certificates.json](alternating_matching_certificates.json).

The earlier failed searches at budget 48 were merely inconclusive;
the new word demonstrates why they could not be used as lower bounds.
In particular this permutation does not refute H(24)=49 and is not a
diameter element, since D(24) >= 49 > 46.

## Result at n=36

A replayed word of length 108 is available, while (1) gives a lower
bound of 104. Thus

\[
104\le |p_{36}|_4\le108 < H(36)=109\le D(36).
\]

Its exact distance is not yet proved by these bounds. It is already
proved not to be a diameter element. The proposed all-n equality with
the lower bound in (1) remains unproved.
