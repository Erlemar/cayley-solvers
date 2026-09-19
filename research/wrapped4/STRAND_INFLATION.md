# Exact wrapped 4-cycle length of every rotation followed by a transposition

Write tau_a(i)=i+a modulo n, put b=n-a, and let t be any transposition.
Let ell denote cyclic adjacent-swap length and let |.|_4 denote length
in wrapped consecutive 4-cycles and their inverses. Set

\[
h(k)=\min\{L\ge0:3L\ge k,\ L\equiv k\pmod2\}
=\lfloor k/3\rfloor+(k\bmod3).
\]

**Theorem.** For every a,b>=6 and every transposition t,

\[
\boxed{|\tau_a t|_4=h(ab-1).}
\tag{1}
\]

This closes the three families left unsettled in the earlier interval
construction. It is an all-size constructive theorem, not an extrapolation
from the searches. It does not determine the diameter for arbitrary
permutations.

The lower bound follows from the previously proved identity
ell(tau_a t)=ab-1: every 4-cycle has adjacent length at most three and
is odd. We prove the matching upper bound.

## 1. Inflating one strand while keeping the cyclic origin

Represent a permutation as a row of tokens, whose labels give their
final positions. In a sorting word, expand each 4-cycle into its three
ordinary cyclic adjacent exchanges. A positive move sends the first
token of its four-position window three places clockwise and the other
three tokens one place counterclockwise. A negative move does the reverse.

For a token x let D_x be its signed displacement in this expanded word,
let c_x be its total distance travelled, and let i_x be its initial
position. Its winding number is the integer

\[
k_x=\frac{i_x+D_x-x}{n}.
\tag{2}
\]

Replace x by four ordered clones in both the initial permutation and
the final label order, leaving every other token a singleton. This
creates a permutation of n+3 labels. Every original 4-cycle can be
simulated by allowed 4-cycles on these bundles, preserving the order
inside each bundle:

* If x is absent, keep the original move on the four singleton bundles.
* If x is the token moving three places, exchange its four clones with
  the other three tokens. This takes four moves, an increase of three.
* If x is one of the tokens moving one place, exchange the singleton
  moving three places with the now six-token opposite block. This takes
  two moves, an increase of one.

These block exchanges are explicit: pass successive triples across one
token at a time. More generally, exchanging blocks of lengths A,B takes
AB/3 moves whenever at least one length is divisible by three, with all
tokens in each block moving in the same direction. The construction is
valid on an interval crossing the cyclic cut as well.

The resulting word has exactly

\[
L'=L+c_x
\tag{3}
\]

letters. However, its endpoint must be checked in the **fixed-position**
graph: a global rotation is not free.

Let gamma track the physical position of the beginning of old position
zero in the enlarged circle. During an adjacent bundle exchange at the
old cut, let X be the bundle at old position n-1 and Y the one at old
position zero. Their exchange changes gamma by

\[
\gamma'-\gamma=\operatorname{size}(Y)-\operatorname{size}(X).
\]

An exchange away from that cut does not change gamma. Bundle sizes are
one, except that x has size four. Consequently a clockwise crossing
of the old cut by x subtracts three from gamma, and a counterclockwise
crossing adds three. Thus the final displacement of the cyclic origin is

\[
\boxed{\gamma=-3k_x\pmod{n+3}.}
\tag{4}
\]

This accounting can be done with the three adjacent bundle exchanges
underlying each old 4-cycle; the efficient block simulation has exactly
the same endpoint. At the end the enlarged bundles are in increasing
cyclic order, and their first label zero is at gamma.

**Inflation lemma.** If k_x=0, the constructed word sorts the inflated
permutation to the actual identity, with length L+c_x. There is no
unpaid global rotation.

Also, if another token y crosses x exactly once, inflating x increases
y's travel by three. If y moved in only one direction before, it still
does: each block exchange moves every constituent in the same direction
as its original strand. Any chosen clone of x has the same travel and
direction as x had. The crossing between a chosen clone and y still
occurs exactly once.

## 2. Two independent extensions of a short twisted rotation

Let

\[
P(a,b,d)=\tau_a(0\ d),\qquad d\in\{1,2\}.
\]

Suppose a word sorting P(a,b,d) has two distinguished, unmarked labels

\[
0\le u<a,\qquad a+d<v<a+b,
\]

with the following properties:

1. Token u moves only counterclockwise, travelling b positions in total.
2. Token v moves only clockwise, travelling a positions in total.
3. They cross each other exactly once.

Their initial positions are b+u and v-a respectively, so both have
winding number zero by (2).

Inflating u increases the first rotation block from a to a+3. The
initial inflated row is exactly P(a+3,b,d): the two marked labels shift
together, remain singletons, and retain their separation. By the lemma
the word still sorts to the fixed identity, and its length increases by b.
A chosen clone of u still travels b counterclockwise, while v now travels
a+3 clockwise. Their single crossing is preserved.

Similarly, inflating v gives P(a,b+3,d), with cost increase a. Token u
now travels b+3 counterclockwise, and a chosen clone of v travels a
clockwise. Thus the same three properties hold after either extension.
They can be repeated independently in any order.

In the residue classes used below, ab-1 is divisible by three. The
length increases agree exactly with the proposed optimal lengths:

\[
\frac{(a+3)b-1}{3}-\frac{ab-1}{3}=b,
\qquad
\frac{a(b+3)-1}{3}-\frac{ab-1}{3}=a.
\tag{5}
\]

## 3. Three explicit base identities

Notation i+ means the positive wrapped 4-cycle starting at position i,
and i- its inverse. Apply the listed moves from left to right to the
row P(a,b,d); all indices are modulo a+b.

| a,b,d | Distinguished u,v | Word length |
|---|---|---:|
| 7,7,2 | 0,10 | 16 |
| 5,5,1 | 0,7 | 8 |
| 5,5,2 | 3,9 | 8 |

The words are:

```text
(7,7,2): 4- 5- 6- 7- 8- 11- 12- 1- 2- 3- 9- 6+ 0+ 3- 4- 9-
(5,5,1): 1- 2- 9+ 8+ 5+ 4+ 1- 2-
(5,5,2): 7+ 6+ 8- 0- 3- 4- 7+ 6+
```

Direct application gives the identity. Tracking u and v in the same
calculation gives displacements -b and +a with total travels b and a,
and exactly one mutual crossing. Thus all hypotheses of Section 2 hold.
These are short, finite permutation identities; they are also retained
and independently checked in
[rotation_twist_inflation_bases.json](rotation_twist_inflation_bases.json).

It follows by induction that the lower bound is attained in all three
families:

\[
\begin{array}{ll}
a\equiv b\equiv1\pmod3,\quad a,b\ge7,& d=2,\\
a\equiv b\equiv2\pmod3,\quad a,b\ge5,& d=1,2.
\end{array}
\tag{6}
\]

No computation at larger sizes is used to justify this induction.

## 4. Reduction of every remaining transposition to known interval words

For clarity, the earlier separation argument works for unequal block
sizes too. Inversion replaces tau_a t by tau_b times a conjugate
transposition and preserves its length. We may therefore assume b>=a.
Let d be the smaller circular separation of t, so d<=n/2<=b. Conjugating
by a cyclic rotation lets us place the endpoints at b-d and b. This
is the marked interval exchange P_interval(a,b,x=0,y=b-d) from
[TWISTED_ROTATIONS.md](TWISTED_ROTATIONS.md).

When a,b do not share a nonzero residue modulo three, the existing
interval construction already has length h(ab-1), for every marked
position. Suppose they do share that residue and put c=6+(a mod 3),
so c is seven or eight. Remove triples to retain both blocks at size c,
keeping x_0=0 and an index y_0 with

\[
\max(0,c-d)\le y_0\le\min(b-d,c-1),
\qquad y_0\equiv b-d\pmod3.
\tag{7}
\]

The four interval-extension identities in the earlier proof recover
the original sizes without changing the excess over h(ab-1).
The complete interval-base table shows that with x_0=0 the only
non-sharp retained entries are y_0=5 when c=7, and y_0=6,7 when c=8.

Choose the largest index allowed in (7). If d>=3 and that index is
non-sharp, decrease it by three. This stays feasible: y_0=5 for c=7,
or y_0=6 for c=8, forces d=2 modulo three and hence d>=5; y_0=7 for
c=8 forces d=1 modulo three and hence d>=4. The new retained base is
sharp. Thus every separation d>=3 is already settled for all a,b>=6.
For c=7,d=1 the forced index y_0=6 is sharp as well.

The only cases left are exactly (6). Section 3 settles them using the
strand-inflation induction. Conjugation by a rotation moves the short
transposition to any desired positions; inversion handles a>b.
This proves (1).

## 5. Consequence for the diameter problem

Put K=floor(n^2/4) and H(n)=ceil(n^2/12)+1_(3 divides n).
The classification in
[ROTATION_ASCENT_LEMMA.md](ROTATION_ASCENT_LEMMA.md) says that the layer
ell=K-1 consists of central rotations followed by one transposition,
and, for even n, the two neighboring rotations themselves. Formula (1)
and the known exact rotation lengths therefore give

\[
\boxed{\ell(p)=K-1\quad\Longrightarrow\quad |p|_4=h(K-1)
\qquad(n\ge12).}
\tag{8}
\]

The layer ell=K consists only of central rotations and has exact
4-cycle length h(K). Hence the largest distance on the top two
adjacent-length layers is exactly H(n). Any permutation contradicting
D(n)=H(n) must have ell<=K-2.

This does not control the lower layers. The previously proved universal
upper estimate with C=3/4 remains unchanged, and the exact diameter
and its proposed linear coefficient are still open in this work.

The inflation method has since been extended to nonzero-winding fixed
strands: an explicit relabelling absorbs the computed origin shift into
the new rotation parameter. This proves the whole layer ell=K-2 as well;
see [TWO_DEFECT_EXACT.md](TWO_DEFECT_EXACT.md). The maximum on the top
three auxiliary layers is therefore H(n), for every n>=12.

## 6. Reproducible checks

[rotation_twist_inflation.py](rotation_twist_inflation.py) implements
both independent extensions and a complete optimal-word constructor
for tau_a(0 d), for all a,b>=6 and every circular separation d.
The three short base words above suffice for the new part; the
previous interval certificates supply the other cases.

[verify_twist_inflation.py](verify_twist_inflation.py) uses a separate
standard-library replay routine and no search code. It verifies the
base identities and distinguished strands, then checks 120 random
applications of the exact winding formula (4), including 57 nonzero
winding cases. It also constructs and independently replays all 13,669
rotation/transposition cases with 6<=a,b<=32 and every circular
separation, checking that every word attains h(ab-1). This replays
1,932,366 individual 4-cycles. Report:
[twist_inflation_independent_audit.json](twist_inflation_independent_audit.json).

The separate extension check covers 432 unequal-size targets with
block sizes through 40. A paired-strand version also verifies the
n-to-n+6 induction through n=254,256. Those finite checks support the
construction; the lemmas and the three explicit identities prove
the unrestricted statements.
