# Diameter under wrapped consecutive 4-cycles: proved results and open question

Date: 2026-09-19. Positions are fixed and indexed by Z/nZ. Global rotations
are permutations with a cost, not free identifications.

Let G_n consist of (i,i+1,i+2,i+3) and its inverse for every i modulo n,
and let D(n) be the diameter of Cay(S_n,G_n), for n >= 5.
An adjacent transposition has a five-letter expression given below, so
these generators do generate S_n.

## Status

The exact reflection formulas in [PROOF.md](PROOF.md) are proved.
The exact general diameter is **not proved** here. The established results are

\[
\boxed{D(n)\ge H(n):=\left\lceil\frac{n^2}{12}\right\rceil+
\mathbf1_{3\mid n}\qquad(n\ge5),}
\tag{1}
\]

\[
\boxed{D(n)\le\left\lfloor\frac{4\lfloor n^2/4\rfloor+9n+233}{12}\right\rfloor,
\quad D(n)=\frac{n^2}{12}+O(n).}
\tag{2}
\]

If d_1,d_2 denote the two reflection distances, then also

\[
\boxed{D(n)-\max\{d_1(n),d_2(n)\}\ge
\left\lfloor\frac n6\right\rfloor+(n\bmod2)>0
\qquad(n\ge13).}
\tag{3}
\]

For the alternating matching perturbation p(i)=i+n/2+(-1)^i modulo n,
the individual distance at n=24 is now proved to be exactly 46, and at
n=36 it is at most 108. Thus these two candidate permutations are not
diameter elements: the general lower bounds there are 49 and 109.
See [ALTERNATING_MATCHING.md](ALTERNATING_MATCHING.md).

Thus neither reflection is a diameter element for any n >= 13. Equality
with a reflection at small n does not extend to arbitrarily large n.

Equation (2) proves the leading coefficient 1/12 with a linear error bound;
C=3/4 is admissible in an upper estimate n^2/12+Cn+B. For n >= 32, the
constant 233 in (2) improves to 148. The proof is in
[COMBINED_EXTRACTION_UPPER_BOUND.md](COMBINED_EXTRACTION_UPPER_BOUND.md). It does not determine
the sharp linear coefficient or a periodic bounded remainder. In
particular, the proposed formula n^2/12+n/6+B(n mod 12) remains unproved.

An alternative extraction lemma now permits a permuted unselected tail,
charging its inversion number explicitly. Its complete hybrid certificate
has 1447 main local words and reproduces C=3/4; see
[PARTIAL_EXTRACTION_METHOD.md](PARTIAL_EXTRACTION_METHOD.md). Attempts at
smaller coefficients are independently audited but have uncovered cases,
so they do not sharpen (2). The one-pivot simplification is also ruled
out as a K+O(n) adjacent-work argument by the explicit infinite family in
[PIVOT_CUT_OBSTRUCTION.md](PIVOT_CUT_OBSTRUCTION.md); this is an obstruction
to that proof strategy, not to the proposed diameter formula.

The entire auxiliary layer ell=K-2 is now also settled for every n>=12:
every element has exact 4-cycle length h(K-2). The proof in
[TWO_DEFECT_EXACT.md](TWO_DEFECT_EXACT.md) combines a general strand
inflation lemma with a complete positive certificate of 947 words and
942 root orbits. Thus all three top auxiliary layers, ell=K,K-1,K-2,
have exact lengths h(K),h(K-1),h(K-2). Their maximum is H(n), so any
counterexample to D(n)=H(n) must have ell<=K-3. This does not control
all the lower layers or improve the universal upper bound (2).

The noncentral part of the next layer ell=K-3 is also exact for all
n>=13: its distance is h(K-3). See
[THIRD_LAYER_NONCENTRAL.md](THIRD_LAYER_NONCENTRAL.md). A single strand
inflation from the central defect-two theorem covers all large gaps;
an independently verified certificate of 224 words covers the finite
boundary. Only the central families in that layer remain unresolved.
More generally, [TWO_DEFECT_UNBALANCED.md](TWO_DEFECT_UNBALANCED.md)
proves |tau_a sigma|_4=h(a(n-a)-2) for every defect-two core sigma and
both rotation blocks at least 12, with no centrality assumption.

For a central rotation followed by an arbitrary 4-cycle, the upper bound
H(n) is now proved for every n>=12 by a separate complete certificate:
1705 roots, 1720 words, and 15 refinement edges. The distance is exactly
h(K-3) when n=2,3,4 modulo six. See
[FOUR_CYCLE_CORE_BOUND.md](FOUR_CYCLE_CORE_BOUND.md). In layer K-3 the
remaining central core types are a disjoint 3-cycle/transposition and
a noncrossing triple matching. Deeper layers remain uncontrolled.

There is now a structural stability theorem for the auxiliary metric:
if ell(p)=floor(n^2/4)-delta, then p differs from some rotation by at
most delta arbitrary transpositions. An increasing path proves this
for every n. A deletion identity reduces each fixed-deficit layer to
finitely many cores on at most 2delta positions. The layers of deficits
zero, one, two and three are now completely classified; see
[ROTATION_ASCENT_LEMMA.md](ROTATION_ASCENT_LEMMA.md).

At n=15 the exact 4-cycle lengths on these four entire layers are:

| Auxiliary length ell | Number of permutations | Exact 4-cycle length |
|---:|---:|---:|
| 56 | 2 | 20 |
| 55 | 210 | 19 |
| 54 | 7282 | 18 |
| 53 | 126700 | 19 |

The new last two rows have complete independently replayed certificates,
not just sampled searches. Thus a counterexample to D(15)=20 must have
ell<=52. The many lower layers remain uncontrolled, so this does not
settle D(15) or improve the universal upper bound (2).

## 1. A parity-sensitive comparison with cyclic adjacent swaps

Write ell(pi) for word length with respect to all cyclic adjacent
transpositions s_i=(i,i+1 mod n), and write |pi|_4 for the required metric.
Each generator in G_n is a product of three of the s_i and is odd. Therefore

\[
3|\pi|_4\ge\ell(\pi),\qquad
|\pi|_4\equiv\ell(\pi)\pmod2.
\tag{4}
\]

Define

\[
h(k)=\min\{L\in\mathbb Z_{\ge0}:3L\ge k,\ L\equiv k\pmod2\}
=\left\lfloor\frac k3\right\rfloor+(k\bmod3).
\tag{5}
\]

Note that h is not monotone. Whenever only a lower bound ell(pi) >= k
is used, the parity of pi must be checked separately before claiming
|pi|_4 >= h(k).

### Rotations

For a+b=n, let tau_a be the rotation i -> i+a modulo n. Then

\[
\ell(\tau_a)=ab.
\tag{6}
\]

Here is an elementary lower-bound proof. Track the signed displacement of
each token during any adjacent-swap word for tau_a. Its final displacement
is a+n k_i for an integer k_i. Since each swap moves one token clockwise
and another counterclockwise, their sum is zero, so sum_i k_i=-a.
For every integer k,

\[
|a+nk|\ge a+(2a-n)k.
\]

For k >= 0 the difference between the two sides is 2(n-a)k; for k <= -1
it is -2a(k+1). Summing gives sum_i |a+n k_i| >= 2ab. Each adjacent swap
contributes two to the total distance travelled, proving ell(tau_a) >= ab.
Exchanging two ordinary adjacent blocks of sizes a,b uses ab adjacent swaps
and realizes a rotation (possibly its inverse). This proves equality.

Set K=floor(n^2/4) and choose a=floor(n/2). Thus ell(tau_a)=K and
|tau_a|_4 >= h(K). If s is any cyclic adjacent transposition, the triangle
inequality gives

\[
\ell(\tau_a s)\ge K-1,
\qquad \operatorname{sgn}(\tau_a s)=(-1)^{K-1}.
\]

Consequently |tau_a s|_4 >= h(K-1), and

\[
D(n)\ge\max\{h(K),h(K-1)\}=H(n).
\]

The last equality follows by substituting n modulo 6. In particular,

\[
H(n)=\frac{n^2+b_{n\bmod6}}{12},\qquad
b=(12,11,8,15,8,11).
\tag{7}
\]

This proves (1). The witness is a half rotation or an adjacent
perturbation of it; the proof does not assume the diameter data.

There is now an explicit optimal word for a witness at distance exactly
H(n) for every n >= 5. Use the half rotation for n=2,3,4 modulo 6, and
the half rotation followed by an adjacent transposition for n=0,1,5
modulo 6. See [DIAMETER_WITNESSES.md](DIAMETER_WITNESSES.md) for the
three-base induction proving the perturbed-rotation distances.

The arbitrary-transposition result is now exact in every case:
for a,b >= 6 and any transposition t,

\[
|\tau_a t|_4=h(ab-1).
\]

The earlier interval proof is in [TWISTED_ROTATIONS.md](TWISTED_ROTATIONS.md).
The three remaining short-separation families are closed by the new
strand-inflation induction in [STRAND_INFLATION.md](STRAND_INFLATION.md).
It extends three explicit base words by adding three positions to either
rotation block independently, while preserving the cyclic origin.
Thus the entire auxiliary layer ell=K-1 has exact distance h(K-1) for
all n>=12. Together with the central rotations, the maximum on the top
two auxiliary layers is exactly H(n). The newer deficit-two theorem
above extends this conclusion to the top three layers; the many lower
layers are still not controlled.

## 2. Exact distances of nontrivial block rotations

The rotations by floor(n/2) and ceil(n/2) are now proved to be the only
permutations maximizing the auxiliary cyclic adjacent-swap length, for
every n >= 2. See
[ADJACENT_EXTREMAL_CLASSIFICATION.md](ADJACENT_EXTREMAL_CLASSIFICATION.md).
This does not imply they maximize the 4-cycle length: the two metrics
have different extremal sets.

An additional constructive statement is

\[
\boxed{|\tau_a|_4=h(a(n-a))\quad\text{if }2\le a\le n-2.}
\tag{8}
\]

The lower bound is (4)-(6). For the upper bound, it is enough to exchange
adjacent blocks A,B of sizes a,b, preserving the order inside both blocks.
If a is divisible by 3, move its triples across the b entries of B,
starting with the rightmost triple. Each triple crosses one entry with
one allowed move, so the cost is ab/3. The case 3|b follows by inversion.

For the other residue classes, use the following four base identities.
Here i+ left-rotates the four array entries starting at i, i- right-rotates
them, and the listed operations are applied from left to right to the
identity array. Positions in these words do not wrap.

| (a,b) | Word producing [a,...,a+b-1,0,...,a-1] | Length |
|---|---|---:|
| (2,2) | 0+, 0+ | 2 |
| (4,2) | 2+, 2+, 0+, 0+ | 4 |
| (2,4) | 0+, 0+, 2+, 2+ | 4 |
| (4,4) | 1-, 2-, 0+, 3-, 4-, 2+ | 6 |

An exchange for (a,b) extends to one for (a+3,b) at an additional cost b:
in [T A B], with |T|=3, first exchange A,B, then move T across B.
Likewise (a,b+3) costs an additional a. These increments agree with
h((a+3)b)-h(ab)=b and h(a(b+3))-h(ab)=a. The four bases cover all pairs
of nonzero residues modulo 3, proving (8) by induction.

## 3. Neither reflection is peripheral for n >= 13

The exact reflection theorem gives

\[
\max(d_1,d_2)=\frac{n^2-2n+c_{n\bmod12}}{12},
\quad c=(12,1,12,9,16,9,12,1,12,9,16,9).
\]

Subtract this from (7). Checking the twelve residue classes yields

\[
H(n)-\max(d_1,d_2)=\lfloor n/6\rfloor+(n\bmod2).
\]

Combining with (1) proves (3). The restriction n >= 13 is essential:
the shifted-reflection formula has small exceptions, including n=12.

## 4. Constructive upper bound for every permutation

The best proved linear coefficient now comes from

\[
D(n)\le\left\lfloor\frac{4\lfloor n^2/4\rfloor+9n+233}{12}\right\rfloor
\le n^2/12+(3/4)n+233/12\qquad(n\ge5).
\]

For n >= 32, the numerator's constant 233 improves to 148. See
[COMBINED_EXTRACTION_UPPER_BOUND.md](COMBINED_EXTRACTION_UPPER_BOUND.md) for the complete
computer-assisted proof. It combines the arbitrary-interval certificate
(10815 cases) with the two-list merge certificate (5794 cases) from
[REFINED_MERGE_UPPER_BOUND.md](REFINED_MERGE_UPPER_BOUND.md).
The earlier, simpler C=4 bound is in
[LINEAR_UPPER_BOUND.md](LINEAR_UPPER_BOUND.md).
It sorts a balanced lifted permutation in three stages: one linear
interval, one insertion across the wrap, and another linear interval.
Efficient 4-cycle sorting of the intervals gives the stated linear error.
The finite five-position base has explicit certificates for all 120 cases.

### Earlier, weaker construction

The following argument remains valid but its O(n^(3/2)) error is superseded
by the linear bound above.

The construction uses the following known theorem: every permutation of
M fixed cyclic positions can be sorted with at most floor(M^2/4) cyclic
adjacent swaps. An optimal sorting can be chosen with net token
displacements of sum zero and range at most M. See van Zuylen, Bieron,
Schalekamp and Yu, [An Upper Bound on the Number of Circular Transpositions
to Sort a Permutation](https://arxiv.org/pdf/1402.4867), Theorems 1 and 4
and Lemma 2. This is the fixed-position problem, not sorting modulo free
rotations.

We give the reduction to that theorem explicitly. Let

\[
r=n\bmod3,\quad N=n-r,\quad M=N/3+\mathbf1_{r>0}=\lceil n/3\rceil.
\]

Choose any positive multiple B of 3 with B <= N, and put q=ceil(N/B).
For every permutation, the following bound holds:

\[
\boxed{D(n)\le
3\left\lfloor\frac{M^2}{4}\right\rfloor+
5\left[r(n-1)+2N\left\lceil\frac NB\right\rceil+
\frac{N(B-1)}2\right].}
\tag{9}
\]

### Local swap identity

On five consecutive positions, the word

\[
0^-,1^-,0^-,1^+,1^+
\]

takes [0,1,2,3,4] to [1,0,2,3,4]. Shifting the indices cyclically
therefore realizes any adjacent transposition in five allowed moves.
The five positions are distinct for n >= 5.

### Preparing triples

First put the r labels N,...,n-1 into their final positions, in order,
using at most r(n-1) adjacent swaps. They will form a single small block.

Partition the other target labels into q consecutive classes, each of
size a multiple of 3 and at most B. In the current order of the first N
tokens, divide the occurrences of each class into successive triples.
Order all triples by the position of their last occurrence, and bring
each triple together, preserving its internal order.

This rearrangement uses at most 2Nq adjacent swaps. To prove it, a triple
whose positions are p1<p2<p3 can be charged at most
(p3-p1)+(p3-p2) <= 2(p3-p1) inversions: in a reversed pair from two triples,
charge the token in the triple with the later last occurrence. That token
must be one of the first two occurrences, and the other token lies between
it and its triple's last occurrence. For each class the intervals
[p1,p3] belonging to successive triples are disjoint, and their total
length is at most N. Summing over classes proves the claim.

### Sorting the blocks

There are M blocks: N/3 triples and, if r>0, the one small block of length
r. Give the triples a target block order according to their classes;
their relative order within one class can be arbitrary. Give the small
block the last position, which it already occupies.

Apply the cyclic adjacent-swap theorem to these M individually labelled
blocks. Exchanging two neighboring triples costs exactly three allowed
4-cycles. Exchanging a triple and the small block costs r <= 2 moves.
Thus this step costs at most 3 floor(M^2/4).

There is a positional issue when block lengths differ, which must be
checked. Keep a physical origin for coarse block position zero. Swapping
the last and first coarse blocks of lengths a,b shifts this origin by
b-a; other swaps leave it unchanged. Equal triples cause no shift.
Each boundary crossing by the small block changes the origin by plus or
minus (3-r), according to its direction.

Choose the balanced-displacement sorting from the cited theorem. Every
initial displacement has absolute value less than M: range at most M and
sum zero rule out a value >= M or <= -M. The small block starts and ends
at the same coarse position, so its displacement is a multiple of M;
hence it is zero. Its net boundary winding is therefore zero. The total
origin shift vanishes, and all class intervals end at their actual target
positions, not just at a common rotation of them.

### Sorting inside the classes

Each interval now contains exactly its target class of labels. Sort each
interval internally by adjacent swaps. If their sizes are b_j <= B, the
number required is at most

\[
\sum_j\binom{b_j}{2}\le\frac{N(B-1)}2.
\]

The small block is already internally sorted. Implement each adjacent
swap by the five-letter identity. Adding these three stages proves (9).

Taking B=min(N,3 ceil(2 sqrt(N)/3)) in (9), as n tends to infinity, gives

\[
D(n)\le\frac{n^2}{12}+10n^{3/2}+O(n).
\]

Together with (1), this independently proves the leading asymptotic
lim D(n)/n^2=1/12, with a weaker error than the current equation (2).
The large error term is a limitation of this construction, not evidence
for an actual n^(3/2) term in the diameter.

## 5. Exact computations and a candidate formula

The supplied diameter data and the lower bound compare as follows.

| n | D(n) | H(n) | D(n)-H(n) |
|---:|---:|---:|---:|
| 5 | 4 | 3 | 1 |
| 6 | 5 | 4 | 1 |
| 7 | 6 | 5 | 1 |
| 8 | 7 | 6 | 1 |
| 9 | 8 | 8 | 0 |
| 10 | 10 | 9 | 1 |
| 11 | 11 | 11 | 0 |
| 12 | 13 | 13 | 0 |
| 13 | 15 | 15 | 0 |
| 14 | 17 | 17 | 0 |

Complete breadth-first searches independently confirmed every diameter
for 5 <= n <= 12. The n=12 search reached all 479,001,600 permutations.
The entries n=13,14 are user-supplied data, not independently enumerated
in this task.

A possible sharper conjecture is

\[
D(n)\stackrel{?}=H(n)\qquad(n\ge11).
\tag{10}
\]

This has linear coefficient zero and a bounded term of period 6 (hence
also period 12). It matches the four known consecutive values n=11..14,
but those four values are not sufficient to establish it. The missing
statement is a universal upper bound |pi|_4 <= H(n) for every pi in S_n.
Neither (8), which only handles rotations, nor (9) proves that statement.
The data do not rule out the originally proposed positive linear term
for all sufficiently large n with a different periodic remainder.

Further evidence: 2,000 tested instances each at n=15 and n=16 all have
verified words of length at most H(n), after retrying initially
node-limited searches. These are samples, not full enumerations, and
do not establish D(15)=20 or D(16)=22. See `candidate_search_verification.json`
and [DIAMETER_WITNESSES.md](DIAMETER_WITNESSES.md).

For reference, the following are **proved lower bounds**, and only
conjectural exact diameters under (10):

| n | H(n) | n | H(n) | n | H(n) |
|---:|---:|---:|---:|---:|---:|
| 15 | 20 | 22 | 41 | 29 | 71 |
| 16 | 22 | 23 | 45 | 30 | 76 |
| 17 | 25 | 24 | 49 | 31 | 81 |
| 18 | 28 | 25 | 53 | 32 | 86 |
| 19 | 31 | 26 | 57 | 33 | 92 |
| 20 | 34 | 27 | 62 | | |
| 21 | 38 | 28 | 66 | | |

## 6. Reproducibility

- `rotation_bounds.py`: explicit rotation-word constructor; independently
  replayed all 1,521 words with 2 <= a,b <= 40; checked the residue and gap
  identities for every 13 <= n <= 10,000. Results: `diameter_bounds.json`.
- `general_sort.py`: implements the construction proving (9). Independently
  replayed words sorting 510 random permutations, covering every n=5..100
  and n=150,151,152,300,301,302. Results: `general_sort_checks.json`.
- `linear_error_sort.py`: implements the stronger linear-error bound.
  All 46,200 permutations for n=5..8 and 1,788 further instances through
  n=1000 were checked by full word replay. Its five-position base has
  all 120 explicit certificates in `five_position_certificates.json`.
  See `linear_bound_exhaustive_checks.json` and `linear_error_sort_checks.json`.
- `adaptive_sort.py`: independently checks the complete finite extraction
  certificate and implements the improved C=4/3 bound. Its proof is in
  `ADAPTIVE_UPPER_BOUND.md`; verification records are
  `adaptive_sort_checks.json` and `adaptive_boundary_checks.json`.
- `merge_sort.py`: independently checks all 396 cases of the merge
  extraction certificate and implements the C=1 bound. It verified
  8,188 short merges with helpers and 1,656 complete words through n=1000.
  Proof: `MERGE_UPPER_BOUND.md`; results: `merge_sort_checks.json`.
- `verify_merge_tree.py` and `refined_merge_sort.py`: check the complete
  5794-case certificate and implement the stronger C=5/6 bound. All 2909
  certificate words, 8188 short merges, and 1240 full sorting words were
  checked. Proof: `REFINED_MERGE_UPPER_BOUND.md`; results:
  `merge_refined_audit.json` and `refined_merge_sort_checks.json`.
- `verify_refined_extraction.py` and `combined_extraction_sort.py`: verify
  the complete 10815-case arbitrary-interval tree, the six-helper charge
  bound, and the stronger C=3/4 construction. Results:
  `combined_sort_checks_c7_4.json`, including 5912 boundary permutations
  and 1220 complete words through n=1000.
- `check_adjacent_extremal_classification.py`: exact adjacent-swap BFS
  for all 46232 permutations at n=2..8, validating the all-n classification
  proved in `ADJACENT_EXTREMAL_CLASSIFICATION.md`.
- `twisted_exchange_check.py`: verifies all 441 twisted-exchange bases
  and 7350 extended words. Proof: `TWISTED_ROTATIONS.md`; certificates:
  `twisted_exchange_certificates.json`; results: `twisted_exchange_checks.json`.
- `wide_search.py`: extends bounded search to n <= 40. All 6480 tested
  packed generator actions and both bounds for all 148 known n=8
  peripheral states were checked. Its node-limit results are inconclusive.
- `diameter_bfs.py` and `diameter_scan.py`: exhaustive finite calculations;
  result files are in `diameter_exact/`. The scanning implementation was
  compared with the queue implementation at n=8 before running n=12.

The finite checks validate the constructions and small cases. They do
not replace the all-n arguments in Sections 1-4 and do not prove (10).
