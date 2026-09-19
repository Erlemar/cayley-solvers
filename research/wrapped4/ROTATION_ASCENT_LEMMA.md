# Increasing paths to rotations in the cyclic adjacent-swap metric

Let ell(p) be word length in the cyclic adjacent transpositions of S_n,
let tau_a(i)=i+a modulo n, and put K=floor(n^2/4). These are statements
about the **auxiliary adjacent-swap metric**, not a determination of the
diameter for wrapped 4-cycles.

## 1. Stability theorem

**Theorem.** Every permutation p with ell(p)=K-delta can be changed into
a rotation tau_a by at most delta arbitrary transpositions. More precisely,
there is such a path along which ell strictly increases at every step,
and, if T is the number of steps, then

\[
T\le a(n-a)-\ell(p)\le\delta,
\qquad T\equiv a(n-a)-\ell(p)\pmod2.
\tag{1}
\]

Consequently p=tau_a sigma for a permutation sigma moving at most 2delta
positions, and

\[
\left(a-\frac n2\right)^2
\le\delta+\frac{n\bmod2}{4}.
\tag{2}
\]

The transpositions in (1) need not be adjacent. Their number does not
give the same bound in the adjacent-swap or 4-cycle metric.

### Balanced lifts

Choose a balanced affine lift w of p: w(i+n)=w(i)+n,
w(i) congruent to p(i) modulo n,

\[
\sum_{i=0}^{n-1}w(i)=\frac{n(n-1)}2,
\qquad \max_i(w(i)-i)-\min_i(w(i)-i)\le n.
\]

The standard minimum-length criterion for cyclic adjacent sorting gives

\[
\ell(p)=\Phi_n(w):=
\sum_{0\le i<j<n}\left|\left\lfloor
\frac{w(j)-w(i)}n\right\rfloor\right|.
\tag{3}
\]

This is the balanced-lift form of Jerrum's criterion; see Theorem 4 of
[van Zuylen, Bieron, Schalekamp and Yu](https://arxiv.org/html/1402.4867).
The sum counts crossings in affine adjacent sorting. The displacement
range condition makes that sorting optimal after projection to the circle.

Choose any integer a with 0<=a<n and

\[
\max_i(w(i)-i)\le a\le\min_i(w(i)-i)+n.
\]

Such an a exists since the displacement sum is zero. Set
f(i)=w(i)+n-a. Then f is a bijection of the integers, periodic of period n,
and

\[
i\le f(i)\le i+n,
\qquad \sum_{i=0}^{n-1}(f(i)-i)=n(n-a).
\tag{4}
\]

Adding a constant to the values leaves Phi unchanged. Moreover Phi is
independent of the choice of n consecutive domain positions: moving the
first position to the end replaces each affected summand |floor(x)| by
|floor(1-x)|=|floor(x)|, since x is nonintegral.

### An increasing swap always exists

Suppose i<A=f(i)<i+n. There is an integer j such that

\[
i<j\le A<B=f(j)\le i+n.
\tag{5}
\]

Indeed, if there were no such j among i+1,...,A, each f(j) would be
less than A or greater than i+n. In the first case its residue can be
represented in {i+1,...,A-1}. In the second case f(j)<=j+n<=A+n;
equality f(j)=A+n is impossible because f(i+n)=A+n. Thus f(j)-n again
belongs to {i+1,...,A-1}. The A-i inputs would have distinct residues in
a set of size A-i-1, a contradiction.

Swap f(i) with f(j), and also all their periodic translates. The new
function f' still satisfies (4): (5) ensures that both new values stay
inside their allowed intervals. The displacement sum is unchanged.
Accordingly w'=f'-(n-a) is again a balanced lift, now of the permutation
p' obtained from p by exchanging the positions i and j modulo n.

The change in length is exactly

\[
\Phi_n(f')-\Phi_n(f)
=1+2\#\{k:i<k<j,\ A<f(k)<B\}.
\tag{6}
\]

To verify (6), move the window to start at i and subtract i from domain
and range. Now i=0 and 0<j<=A<B<=n. The pair (0,j) contributes an
increase of one. For 0<k<j, all relevant differences lie strictly
between -n and n; its two summands increase by two exactly when
A<f(k)<B, and otherwise do not change. For k>j the two summands simply
interchange. This proves (6), which by (3) is also the change in ell.

### Termination and conclusion

There are finitely many functions satisfying (4) in a window, and Phi
strictly increases at every swap. The process therefore terminates.
By (5), it can terminate only when every f(i) equals i or i+n. Its
underlying residue permutation is then the identity, so the underlying
permutation of w is tau_a. There are n-a upper endpoints and a lower
endpoints. Each mixed pair contributes one to Phi and each unmixed
pair contributes zero, giving Phi=a(n-a).

Summing the positive odd increments (6) proves (1). A product of T
transpositions moves at most 2T positions. Finally a(n-a)>=K-delta
gives (2). This proves the theorem.

In particular, the only permutations of length K are the central
rotations tau_floor(n/2) and tau_ceil(n/2). This supplies another proof
of [ADJACENT_EXTREMAL_CLASSIFICATION.md](ADJACENT_EXTREMAL_CLASSIFICATION.md).

## 2. Deleting an endpoint preserves the defect

For any periodic bijection f satisfying i<=f(i)<=i+n, define its rank
k by sum_i(f(i)-i)=nk and its defect by

\[
\mathcal D_n(f)=k(n-k)-\Phi_n(f).
\tag{7}
\]

**Deletion lemma.** Suppose f(h) is h or h+n. Delete that residue from
both domain and range and compress their cyclic orders, obtaining a
bounded affine bijection g of period n-1. If f(h)=h, its rank is k and
Phi_n(f)-Phi_(n-1)(g)=k. If f(h)=h+n, its rank is k-1 and this difference
is n-k. In both cases

\[
\mathcal D_{n-1}(g)=\mathcal D_n(f).
\tag{8}
\]

**Proof.** Rotate the window so that h=0. The increasing bijection
compressing the integers other than multiples of n is

\[
C(x)=x-\lfloor x/n\rfloor-1,
\qquad C(x+n)=C(x)+n-1.
\]

For i=1,...,n-1 put g(i-1)=C(f(i)); boundedness follows by applying C
to i<=f(i)<=i+n. For two undeleted values x,y,

\[
\left\lfloor\frac{C(y)-C(x)}{n-1}\right\rfloor
=\left\lfloor\frac{y-x}n\right\rfloor.
\tag{9}
\]

To see this, write x=qn+r and y=q'n+r', with 1<=r,r'<=n-1. Both sides
are q'-q minus the indicator that r'<r. Thus all pair contributions
not involving zero are unchanged.

For i=1,...,n-1 one has 0<f(i)<2n and f(i) is not n. The rank equals
the number of values f(i) at least n in the full window. When f(0)=0,
the removed terms count the k values f(i)>n. The compressed displacement
sum is nk-k=k(n-1), so the rank stays k. When f(0)=n, the removed
terms count the n-k values f(i)<n. The compressed displacement sum is
nk-n-(k-1)=(k-1)(n-1), so the rank decreases to k-1. Substitution into
(7) proves (8).

## 3. A finite core computes a rotation perturbation exactly

Let sigma be a permutation moving exactly t positions. Order those
positions around the circle starting at the chosen cut, and relabel
them 0,...,t-1. Let alpha be the resulting fixed-point-free permutation
and let e be its number of excedances alpha(i)>i. Assume

\[
e\le a,\qquad t-e\le b=n-a.
\tag{10}
\]

Define the bounded affine core

\[
g_\alpha(i)=
\begin{cases}
\alpha(i),&\alpha(i)>i,\\
\alpha(i)+t,&\alpha(i)<i.
\end{cases}
\]

Then

\[
\boxed{\ell_n(\tau_a\sigma)
=ab-\bigl(e(t-e)-\Phi_t(g_\alpha)\bigr).}
\tag{11}
\]

For sigma equal to the identity the parenthesized defect is zero.

**Proof.** At each moved position put f(i)=sigma(i) if sigma(i)>i,
and f(i)=sigma(i)+n otherwise. At the fixed positions choose a-e lower
endpoints and b-(t-e) upper endpoints; (10) guarantees that these
nonnegative counts sum to n-t. The rank of f is b. Then w=f-b is
a balanced lift of tau_a sigma, so its length is Phi_n(f). Delete all
fixed positions using (8). The remaining function is g_alpha of rank
t-e, so its defect is e(t-e)-Phi_t(g_alpha). This proves (11).

The formula is independent of the gaps between moved positions. It
reduces the length calculation to their cyclic order and the finite core.

For the cores needed below it gives:

| Perturbation sigma | t | e | Phi(core) | Defect ab-ell |
|---|---:|---:|---:|---:|
| One transposition | 2 | 1 | 0 | 1 |
| Either orientation of a 3-cycle | 3 | 1 or 2 | 0 | 2 |
| Two disjoint, noncrossing transpositions | 4 | 2 | 2 | 2 |
| Two disjoint, crossing transpositions | 4 | 2 | 0 | 4 |

For four positions x1<x2<x3<x4, the noncrossing pairings are
(x1 x2)(x3 x4) and (x1 x4)(x2 x3); the crossing pairing is
(x1 x3)(x2 x4). Their respective cores g are [1,4,3,6], [3,2,5,4],
and [2,3,4,5]. Formula (11) applies to all these cases when a,b>=2.
For a single transposition it applies whenever a,b>=1.

Together with the stability theorem, (11) gives a finite-core method
for describing every layer ell=K-delta with delta fixed: only rotations
satisfying (2) and perturbations with support at most 2delta are needed.

More precisely, for each a set d=a(n-a)-(K-delta) and discard a if d<0.
Enumerate all fixed-point-free cores alpha on t<=2d positions satisfying
(10) and e(t-e)-Phi_t(g_alpha)=d, and embed their ordered positions in
every t-subset of the n positions. Include the empty core when d=0.
The resulting union is exactly the layer ell=K-delta. The converse is
(11). For completeness, apply the increasing path with its fixed a to
an arbitrary element of the layer: its initial bounded f has residue
permutation sigma=tau_a^{-1}p and at most 2d moved positions. At these
positions f is uniquely determined by sigma, and its endpoint counts
force (10). Deletion gives the required core equation.

This enumeration is implemented in
[adjacent_deficit_layers.py](adjacent_deficit_layers.py). For fixed
delta its core list is finite and independent of n.

## 4. Complete classification of the next three layers

Let T be the set of all transpositions. Let U be the set consisting of
all 3-cycles and all products of two disjoint noncrossing transpositions.
Products tau_a T and tau_a U denote sets of permutations.

### Layer K-1

For n=2m>=4,

\[
\{p:\ell(p)=K-1\}
=\tau_m T\ \cup\ \{\tau_{m-1},\tau_{m+1}\},
\qquad \#=\binom n2+2.
\tag{12}
\]

For n=2m+1>=5,

\[
\{p:\ell(p)=K-1\}
=\tau_m T\ \cup\ \tau_{m+1}T,
\qquad \#=n(n-1).
\tag{13}
\]

For even n, a rotation tau_(m+s) has deficit K-a(n-a)=s^2.
For odd n, tau_(m+s) has deficit s(s-1). By (1), the possible terminal
rotations have deficit at most one; the path has zero steps if that
deficit is one, and exactly one step if it is zero. This proves the
set containments in (12)-(13). The transposition row of the table
proves the converse.

The displayed sets are disjoint. For odd n>=5 the difference between
the two rotations is an n-cycle, whose minimum number n-1 of arbitrary
transpositions is greater than two. For even n the additional rotations
likewise cannot be a single transposition from the central rotation.
This proves the counts.

### Layer K-2

For n=2m>=6,

\[
\boxed{\{p:\ell(p)=K-2\}
=\tau_m U\ \cup\ \tau_{m-1}T\ \cup\ \tau_{m+1}T,}
\tag{14}
\]

with cardinality

\[
2\binom n3+2\binom n4+2\binom n2.
\tag{15}
\]

For n=2m+1>=7,

\[
\boxed{\{p:\ell(p)=K-2\}
=\tau_m U\ \cup\ \tau_{m+1}U
\ \cup\ \{\tau_{m-1},\tau_{m+2}\},}
\tag{16}
\]

with cardinality

\[
4\binom n3+4\binom n4+2.
\tag{17}
\]

**Proof.** For a terminal central rotation the total increase in (1)
is two. Since each increment is positive and odd, the path has exactly
two steps. Its perturbation is therefore a 3-cycle or two disjoint
transpositions; two equal transpositions would return to the same
permutation and are impossible. Formula (11) excludes the crossing
double transpositions, whose deficit is four rather than two. It also
proves that all the remaining perturbations have the required length.

For even n, the only other possible terminal rotations have s=+/-1,
deficit one. The path to them must be a single transposition. For odd
n, the only other possibilities have s=-1 or 2, deficit two. The path
then has no steps, so the original permutation is that rotation itself.
This proves (14) and (16).

For completeness, the all-transposition length of tau_c is
n-gcd(n,c). In (14), overlaps between the central family and a neighboring
family would express tau_1 using at most three transpositions, impossible
for n>=6. An overlap between the two neighboring families would express
tau_2 using at most two, also impossible. In (16), an overlap between
central families would express tau_1 using at most four, impossible for
n>=7. An outer rotation cannot belong to a central family, since their
difference is tau_1 or tau_2, each requiring n-1>2 transpositions for
odd n. Finally U has 2 binomial(n,3)+2 binomial(n,4) elements. This
proves (15) and (17).

### Layer K-3

Let V consist of the following perturbations:

* any 4-cycle;
* a disjoint 3-cycle and transposition, where the transposition's two
  endpoints are consecutive in the cyclic order of the five moved positions;
* three disjoint transpositions whose three chords are pairwise noncrossing.

There are respectively 6,10,5 possible cores on four, five, six positions,
so write

\[
V_n=6\binom n4+10\binom n5+5\binom n6,
\qquad U_n=2\binom n3+2\binom n4.
\]

For n=2m>=8 the layer is

\[
\tau_m V\ \cup\ \tau_{m-1}U\ \cup\ \tau_{m+1}U,
\qquad \#=V_n+2U_n.
\tag{18}
\]

For n=2m+1>=9 it is

\[
\tau_m V\ \cup\ \tau_{m+1}V\ \cup\
\tau_{m-1}T\ \cup\ \tau_{m+2}T,
\qquad \#=2V_n+2\binom n2.
\tag{19}
\]

Here is a direct check of the required core classification. An ascent
to a central rotation has total increment three, hence uses one or
three transpositions. One is excluded by its defect-one formula.
A permutation of minimum arbitrary-transposition length three is a
4-cycle, a disjoint 3-cycle and transposition, or three disjoint
transpositions. All six 4-cycles have core defect three: for e=1 or 3
the core Phi is zero, and for e=2 it is one. In the second case, the
two orientations of the 3-cycle give e=2 or 3. Their core Phi is three
when the other pair is consecutive in the cyclic order, and one
otherwise, giving defects three and five. Rotation and reflection of
the five positions reduce this calculation to [1,2,0,4,3] and
[2,3,4,1,0], and their inverses. There are five consecutive pairs and
two orientations, hence ten cores of defect three.

For r disjoint transpositions on 2r moved positions, the core has rank
r. Each individual paired pair contributes zero to Phi. Between two
transpositions, the four cross-pair terms contribute two if the chords
do not cross and zero if they cross. Thus, if c is the number of
crossing chord pairs,

\[
\Phi_{2r}(g)=r(r-1)-2c,
\qquad \mathcal D(g)=r+2c.
\tag{20}
\]

For r=3, precisely the five noncrossing perfect matchings have defect
three. This proves the classification of V. Other terminal rotations
have deficit one in the even case and two in the odd case, reducing
to the already classified U and T. This proves the unions (18)-(19).

The unions are disjoint in the stated ranges. In (18), an overlap
with the central family would express tau_1 in at most five arbitrary
transpositions; an overlap between the outer families would express
tau_2 in at most four. In (19), possible overlaps would express tau_1
in at most six, tau_1 or tau_2 in at most four, or tau_3 in at most two.
The formula n-gcd(n,c) for the transposition length of tau_c excludes
each possibility when n>=8 is even or n>=9 is odd, respectively.
The counts follow.

## 5. Implications and limits for wrapped 4-cycles

Let h(k)=floor(k/3)+(k mod 3), and H(n)=ceil(n^2/12)+1_(3 divides n).
Every permutation in layer K-j satisfies the 4-cycle lower bound h(K-j).
The classifications above alone give no matching 4-cycle upper bound.

Combining (12)-(13) with the now complete theorem in
[STRAND_INFLATION.md](STRAND_INFLATION.md) settles the 4-cycle length
of every element in layer K-1 for every n>=12: each length is exactly
h(K-1). The three short-separation families previously left by the
interval construction have all been closed by an all-size induction.
Consequently the maximum on the top two auxiliary layers is H(n),
and a counterexample to D(n)=H(n) must have ell<=K-2.

The entire next layer is now settled too:
[TWO_DEFECT_EXACT.md](TWO_DEFECT_EXACT.md) proves |p|_4=h(K-2) whenever
ell(p)=K-2, for every n>=12. It uses the core classification above,
monotone-strand inflation, and a complete finite word certificate.
Thus a counterexample to D(n)=H(n) must in fact have ell<=K-3.

At n=15, (16) has 7282 elements and (19) has 126700. The certificates
described below now establish their exact 4-cycle lengths, 18 and 19
respectively. Together with the central-rotation and twisted-rotation
theorems, this gives the complete top four layers:

| Adjacent length ell | Number of permutations | Exact 4-cycle length |
|---:|---:|---:|
| 56 | 2 | 20 |
| 55 | 210 | 19 |
| 54 | 7282 | 18 |
| 53 | 126700 | 19 |

Thus any permutation at n=15 with 4-cycle length greater than H(15)=20
must have ell<=52. This does not prove D(15)=20: all the lower layers
remain to be controlled. Neither this theorem nor the certificates
determine the diameter's linear term.

## 6. Independent checks

[rotation_ascent.py](rotation_ascent.py) checked all 46,232 permutations
at n=2,...,8 against an independent cyclic-adjacent BFS, including
278,350 individual ascent increments, and 50 further random paths at
n=9,12,20,40,100. It also checked the complete sets (12)-(13).

[check_rotation_defect.py](check_rotation_defect.py) independently checked:

* 92,457 instances of (11) against the same exact small graphs;
* all 125,670 bounded affine lifts at n=2,...,8 and their 251,326
  endpoint deletions;
* 500 further core formulas at n=12,15,20,40,100;
* the full sets (14)-(16) against BFS at n=6,7,8 and the saved complete
  enumerations at n=10,11.

The latter layer sizes are respectively 100,282,308,750,1982. Reports:
[rotation_ascent_checks.json](rotation_ascent_checks.json) and
[rotation_defect_checks.json](rotation_defect_checks.json). These are
checks of the proofs above; the all-n conclusions do not extrapolate
from the finite data.

The general finite-core enumerator was also checked against the full
layers of deficits zero through three at n=2,...,8 and n=10,11; see
[adjacent_deficit_layers_checks.json](adjacent_deficit_layers_checks.json).
The defect-three sizes at n=10,11 are 6150 and 17930, in agreement with
(18)-(19).

For the new exact n=15 4-cycle assertions, the files
[second_layer_four_n15.json](second_layer_four_n15.json) and
[adjacent_layer_d3_four_n15.json](adjacent_layer_d3_four_n15.json) contain
18-letter words for all 151 symmetry classes of ell=54 and 19-letter
words for all 2232 classes of ell=53. The lower bounds are h(54)=18 and
h(53)=19. Conjugation by a dihedral symmetry and taking inverses preserve
both lengths, so these words prove the matching upper bounds.

The standard-library checker
[verify_second_layer_certificate.py](verify_second_layer_certificate.py)
does not import the search implementation or the core enumerator. It
generates the candidate layers instead from the bounded-transposition
balls supplied by the ascent theorem, filters by the balanced length,
checks complete orbit coverage, and explicitly replays every transformed
word. It checked 9060 and 133920 words, including repeated images from
stabilizers. The reports are
[second_layer_four_n15_independent_audit.json](second_layer_four_n15_independent_audit.json)
and [adjacent_layer_d3_four_n15_independent_audit.json](adjacent_layer_d3_four_n15_independent_audit.json).
These are finite certificates of exact distances for the stated complete
layers, not samples or negative conclusions drawn from search limits.
