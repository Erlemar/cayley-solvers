# Exact 4-cycle lengths on the entire auxiliary layer K-2

Let ell(p) denote cyclic adjacent-transposition length, let |p|_4 denote
length in wrapped consecutive 4-cycles and their inverses, and put

\[
K=\lfloor n^2/4\rfloor,\qquad
h(k)=\lfloor k/3\rfloor+(k\bmod3).
\]

**Theorem.** For every n>=12,

\[
\boxed{\ell(p)=K-2\quad\Longrightarrow\quad |p|_4=h(K-2).}
\tag{1}
\]

This is a computer-assisted proof for all n. Its finite part consists
of explicit positive word identities, direction checks, and a complete
case-coverage certificate. Negative search results are not used.
It is not a proof of the full diameter.

## 1. Which permutations must be handled?

The classification in
[ROTATION_ASCENT_LEMMA.md](ROTATION_ASCENT_LEMMA.md) shows that the layer
K-2 consists of the following families. Write tau_a(i)=i+a modulo n.

* A central rotation tau_a, with a=floor(n/2) or ceil(n/2), followed by
  a 3-cycle or by two disjoint noncrossing transpositions.
* When n=2m is even: tau_(m-1) or tau_(m+1), followed by one transposition.
* When n=2m+1 is odd: the rotations tau_(m-1) and tau_(m+2) themselves.

For the central family write p=tau_a sigma, b=n-a. The fixed-point-free
core of sigma, in the cyclic order of its moved positions, is one of

\[
(1,2,0),\ (2,0,1),\ (1,0,3,2),\ (3,2,1,0).
\tag{2}
\]

Its defect is two, so ell(p)=ab-2. Each allowed generator is odd and
has adjacent length at most three; hence |p|_4>=h(ab-2). We construct
words attaining that bound for the whole central family.

The two other families are treated in Section 6.

## 2. Inflating a monotone unmarked strand, including nonzero winding

We need a slight extension of the lemma in
[STRAND_INFLATION.md](STRAND_INFLATION.md). Suppose sigma fixes position
j, and the token x=p(j) in a sorting word moves in only one direction.
Its signed travel is either +a or -b. Replace x by four ordered clones
and simulate every old 4-cycle by exchanging the corresponding bundles.

Recall the exact facts proved there. If x has total travel c and winding

\[
k_x=\frac{j+D_x-x}{n},
\]

the new word has length L+c, and its endpoint is the cyclic identity
with origin gamma=-3k_x modulo n+3. In array notation that endpoint
is tau_(-gamma), not necessarily the identity.

Let p^[x] be the initial inflated permutation. Relabel every entry by
adding gamma modulo n+3. The **same word** now sorts
tau_gamma p^[x] to the identity. This is an identification of a new
target permutation, not an assumption that rotations have zero cost.

Let sigma^[j] be obtained by replacing the fixed position j of sigma
by four consecutive fixed positions and preserving all other relative
orders. Directly from the two rotation blocks,

\[
p^{[x]}=\tau_{a+3\varepsilon}\sigma^{[j]},
\qquad \varepsilon=\mathbf1_{j\ge b}.
\]

For positive travel D_x=a one has k_x=epsilon. For negative travel
D_x=-b one has k_x=epsilon-1. Therefore

\[
\begin{array}{c|c|c|c}
\text{direction of }x&\text{new target}&(a',b')&\text{length increase}\\\hline
+a&\tau_a\sigma^{[j]}&(a,b+3)&a\\
-b&\tau_{a+3}\sigma^{[j]}&(a+3,b)&b.
\end{array}
\tag{3}
\]

Thus an unmarked gap can be lengthened by three using either a positive
or a negative monotone strand. No winding restriction remains, because
its exact effect on the new rotation parameter has been included.

### Anchors that survive repeated inflation

In each gap that we want to enlarge, select one unmarked positive strand
of travel a and one unmarked negative strand of travel b. Call them its
anchors. We also require that two selected anchors of the same direction
never cross and two of opposite directions cross exactly once, in the
three-adjacent-exchange expansion of the word.

These properties persist under (3). Chosen clones inherit the direction
and travel of the old strand. Other anchors of the same direction do not
cross it and keep their travel; each opposite anchor crosses its four
clones instead of one token, so its travel increases by three. These
are precisely the changes required by the new parameters a',b'. The
crossing pattern between chosen anchors remains unchanged.

The gap's two directions therefore remain available after any number
of inflations, including inflations in other gaps. The finite certificate
explicitly checks this crossing condition for all its chosen anchors.

Finally, the core in (2) is unchanged by inflating a fixed position. Its
adjacent length remains a'b'-2, and

\[
h((a+3)b-2)=h(ab-2)+b,\qquad
h(a(b+3)-2)=h(ab-2)+a.
\tag{4}
\]

Consequently an optimal base word stays optimal after every inflation.

## 3. Reduction to finitely many cyclic gap patterns

Let t=3 or 4 be the number of moved positions. Rotate their first position
to zero, and let g_0,...,g_(t-1) count the unmarked positions in the
successive cyclic gaps. Thus n=t+sum(g_i). A gap of length zero or one
will be kept fixed. For a longer gap retain initially

\[
r_i=2+((g_i-2)\bmod3)\in\{2,3,4\};
\qquad r_i=g_i\text{ if }g_i<2.
\tag{5}
\]

Put m=t+sum(r_i), so m<=5t<=20 and n-m is divisible by three. Choose
n_0 to be the smallest integer at least max(12,m) congruent to n modulo
six. Then

\[
12\le n_0\le23,\qquad n_0\le n.
\tag{6}
\]

To obtain the base gaps, restore any (n_0-m)/3 of the removed triples.
There are enough since n_0<=n. All restored triples go to gaps of length
at least two. If a=floor(n/2), choose a_0=floor(n_0/2); use the corresponding
ceilings for the other central rotation. Both rotation parameters have
decreased by (n-n_0)/2, a multiple of three.

The complete finite root list is therefore:

1. a core from (2), up to cyclic rotation of the marked positions;
2. gap lengths whose sum gives 12<=n_0<=23;
3. the irreducibility condition
   n_0-6<max(12,t+sum(r_i)), where r_i is obtained from those base gaps by (5).

The two noncrossing double-transposition cores are cyclic rotations of
each other, so one can use just (1,0,3,2) when enumerating every gap vector.
This gives 3533 rooted models. Dihedral conjugation and inversion reduce
them to 942 permutation orbits.

These symmetries preserve the relevant anchor property. Rotations
translate positions and labels. Reflection reverses strand directions
and exchanges a,b. For inversion, the reversed word reverses each
strand's motion; an unmarked position j becomes position j+a modulo n,
so the unmarked gaps are cyclically translated, while a,b are exchanged.
Having both monotone directions available in a gap is preserved.
Thus it suffices to certify one representative per orbit, even when
the representative uses the other central rotation.

## 4. The complete finite certificate

The file
[two_defect_inflation_complete.json](two_defect_inflation_complete.json)
contains 947 nodes: the 942 roots and five additional bases reached by refinement.
Every node contains an explicit word of length h(floor(n_0^2/4)-2)
sorting its stated permutation to the identity. Of these nodes, 945
have suitable anchors in every gap of length at least two.

Two roots need a more precise coverage rule. Their words still have the
optimal length and provide anchors in some gaps. If all gaps that must
grow have anchors, that word already suffices. Otherwise select a gap i
that must grow but lacks anchors. Consume one of its removed triples.
The total number of removed triples is even, because n-n_0 is divisible
by six. Since it is positive, at least one more triple remains. Consume
that triple in any gap j where it is available, allowing j=i.

The refined base has six extra positions, both central rotation blocks
increase by three, and its gaps are obtained by adding three to g_i and
three to g_j. The certificate contains every such child, up to the same
symmetries, whenever at least one of i,j lacks anchors. The remaining
number of triples is still even.

The two nonterminal nodes are:

| n_0 | Central shift | Core permutation on actual positions | Gaps | Gaps with anchors | Child orbits |
|---:|---:|---|---|---|---:|
| 12 | 6 | (0 9)(3 6) | (2,2,2,2) | 1,3 | 3 |
| 15 | 7 | (0 14)(10 11) | (9,0,2,0) | 0 | 2 |

Gap indices start at zero, beginning after the first marked position.
All five child nodes have anchors in every potentially growing gap.
Thus every branch closes after at most one such refinement; the largest
base size anywhere in the certificate is still 23.

The word at a nonterminal node also covers the case of no remaining
triples. This detail is why the result includes small sizes n>=12,
not just sufficiently large n.

### Completing the construction

After this finite reduction, let R_i be the remaining number of triples
to insert in gap i. Every gap with R_i>0 now has both anchors. The sum
R=sum(R_i) is even. Insert exactly R/2 triples using negative anchors
and R/2 using positive anchors, distributing these choices arbitrarily
among the prescribed gaps.

Formula (3) increases each rotation block by 3R/2, producing the original
central parameters and exactly the desired original gap lengths. The
core's cyclic order is unchanged. Formula (4) proves that the resulting
word has length h(K-2). This establishes the upper bound for the entire
central family, for every n>=12.

## 5. Independent verification of the finite part

[verify_two_defect_certificate.py](verify_two_defect_certificate.py)
uses only the Python standard library. It imports neither the search
implementation nor the root enumerator or inflated-word constructor.
It performs the following checks:

* independently generates the root cases by enumerating marked position
  subsets, including both noncrossing pairings, rather than gap compositions;
* obtains exactly the same 942 root orbits;
* replays all 947 words and checks their exact lengths;
* measures the selected strands' signed and total travel and all required
  pairwise crossings;
* regenerates every required refinement child from every uncovered gap,
  verifies all five edges, and checks full reachability and termination;
* verifies the six small boundary words used in Section 6.

The independent enumeration has 6030 models because it includes both
noncrossing pairings before taking symmetries. Its orbit set agrees
exactly with the certificate. The 947 main words contain 19,974 moves.
The audit is recorded in
[two_defect_certificate_audit.json](two_defect_certificate_audit.json).

[two_defect_construct.py](two_defect_construct.py) implements the
unrestricted constructor. A separate replay check covers all 3533 root
models, 300 larger cases through n=150, and 60 deliberately chosen
growth patterns from the two exceptional roots. All 3893 complete words
replay correctly; 47 refinement steps are exercised. Report:
[two_defect_construct_checks.json](two_defect_construct_checks.json).
These larger checks validate the implementation; the induction and
finite coverage above are the proof for unrestricted n.

## 6. The noncentral members of the layer

For even n=2m>=14, the remaining elements are tau_(m-1)t and tau_(m+1)t.
Both rotation blocks have size at least six. The theorem in
[STRAND_INFLATION.md](STRAND_INFLATION.md) gives their exact length
h((m-1)(m+1)-1)=h(K-2).

At n=12, the rotations have block sizes 5,7. By cyclic conjugation and
inversion it suffices to check tau_5(0 d) for d=1,...,6. The file
[two_defect_n12_boundary.json](two_defect_n12_boundary.json) contains
six independently replayed words of length 12=h(34), closing this case.

For odd n=2m+1>=13, the remaining permutations are the two rotations
by m-1 and m+2. Their adjacent length is (m-1)(m+2)=K-2, and the exact
rotation theorem gives their 4-cycle length h(K-2).
This completes the proof of (1).

## 7. What this proves about the diameter

The top three auxiliary layers are now all settled, for every n>=12:

\[
\ell(p)=K-j\quad\Longrightarrow\quad |p|_4=h(K-j),
\qquad j=0,1,2.
\]

Their largest 4-cycle distance is

\[
H(n)=\lceil n^2/12\rceil+\mathbf1_{3\mid n}.
\]

Consequently a counterexample to D(n)=H(n) must have ell(p)<=K-3.
This still leaves many permutations. The universal upper bound with
linear coefficient 3/4 is unchanged, and neither the exact diameter
nor its proposed linear term has been determined.
