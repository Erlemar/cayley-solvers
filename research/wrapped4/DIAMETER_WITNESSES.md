# Explicit elements attaining the diameter lower bound

Date: 2026-09-19. These are exact word-metric results for specific elements,
not a proof that no other element is farther away.

Let tau_a be the rotation with one-line form
[a,a+1,...,n-1,0,1,...,a-1], and let s_i swap positions i and i+1 modulo n.
All distances below use consecutive wrapped 4-cycles and their inverses.

## An exact distance for perturbed rotations

**Theorem.** If a,b >= 2, a+b=n, and 3 divides ab, then

\[
\boxed{|\tau_a s_i|_4=\frac{ab}{3}+1}
\tag{1}
\]

for every cyclic adjacent transposition s_i. Products are understood as
applying s_i to the positions of the one-line rotation array.

**Lower bound.** The cyclic adjacent-transposition distance of tau_a is
ab. Hence that of tau_a s_i is at least ab-1. A word of L allowed 4-cycles
satisfies 3L >= ab-1 and L congruent to ab-1 modulo 2. Write ab=3q.
The first condition gives L >= q, while the parity condition excludes q.
Thus L >= q+1.

**Upper bound.** First suppose b is divisible by 3. We construct a block
exchange

\[
[A,B]\longmapsto[B',A],
\]

where B' is B with its first two entries interchanged and all other
entries unchanged. For b=3 and a=2,3,4, use these words on the identity
array. The notation i+ means left rotation of the four entries starting
at i, i- its inverse, and operations are applied in the displayed order.

| a | b | Word | Length |
|---:|---:|---|---:|
| 2 | 3 | 0-, 1+, 1+ | 3 |
| 3 | 3 | 1-, 2+, 2+, 0+ | 4 |
| 4 | 3 | 2+, 0-, 3+, 0-, 2- | 5 |

All three are ordinary nonwrapped identities and have length ab/3+1.

To add three entries to A, write [T,A,B], with |T|=3. Apply the existing
word to A,B, giving [T,B',A], and move T across B' in b moves, one per
entry of B'. The result is [B',T,A], the desired larger exchange.
Its length increases by b, exactly the change in ab/3+1.

To add three entries to B, write [A,B,U], with |U|=3. First obtain
[B',A,U], then move U across A in a moves, giving [B',U,A]. This again
has the required perturbation and the required length increment.
Induction covers every a >= 2 and every positive b divisible by 3.

All choices of the adjacent edge i have equal distance: simultaneous
cyclic relabelling of positions and labels preserves the generator set,
commutes with tau_a, and sends s_i to any other s_j. This is conjugating
every letter of a word, not making a rotation for free.

If a is divisible by 3 but b is not, use the construction with a,b
interchanged and invert its word. Since tau_b^{-1}=tau_a, the result is
tau_a times an adjacent transposition on another edge. The preceding
conjugation observation finishes the proof of (1).

## A uniform family at distance H(n)

Put

\[
H(n)=\left\lceil\frac{n^2}{12}\right\rceil+\mathbf1_{3\mid n},
\qquad a=\lfloor n/2\rfloor.
\]

Define w_n by

\[
w_n=\begin{cases}
\tau_a s_0,&n\equiv0,1,5\pmod6,\\
\tau_a,&n\equiv2,3,4\pmod6.
\end{cases}
\]

Then for every n >= 5,

\[
\boxed{|w_n|_4=H(n).}
\tag{2}
\]

Indeed, in the first three residue classes, K=a(n-a)=floor(n^2/4) is
divisible by 3 and H(n)=K/3+1, so (1) applies. In the other classes, the
exact rotation theorem gives |tau_a|_4=h(K)=H(n), where
h(k)=floor(k/3)+(k mod 3). See Section 2 of
[DIAMETER_STATUS.md](DIAMETER_STATUS.md) for the rotation theorem.

For example, at n=13 the element

\[
[7,6,8,9,10,11,12,0,1,2,3,4,5]
\]

has distance 15. At n=14, the half rotation

\[
[7,8,9,10,11,12,13,0,1,2,3,4,5,6]
\]

has distance 17. These match the supplied diameters. At n=15 and n=16,
the half rotations have distances 20 and 22 respectively, giving
rigorous lower bounds for those still-undetermined diameters.

Thus, if the candidate D(n)=H(n) is eventually true, (2) supplies an
explicit diameter element in every residue class. Equation (2) itself
does not establish that candidate: for example D(5)=4 but H(5)=3.

## Verification and search evidence

The constructor [diameter_witnesses.py](diameter_witnesses.py) implements
the induction above. Direct replay checked:

- 1,960 perturbed block exchanges, every pair 2 <= a,b <= 60 with 3|ab;
- 248 complete witnesses, for every n=5..250 and for n=501,1000.

Results and example words are in `diameter_witnesses_checks.json`.

Separate counterexample searches tested 2,000 instances at n=15 and
2,000 at n=16. Each collection mixed perturbed half rotations, perturbed
reflections, inflations of exact n=12 peripheral states, and random
permutations. Every tested instance has a verified solution of length
at most H(n), after retrying the initial node-limited searches.
These are finite samples, with possible repeated permutations; they do
not exhaust S_15 or S_16 and do not establish either diameter.

The search uses the exact cyclic adjacent length as an admissible lower
bound and a complete radius-four endgame. Exhaustion at a node limit is
never treated as a nonexistence proof. As a check on the search logic,
it correctly rejected length-five solutions for all 148 known diameter
elements of S_8, and length-eight solutions for all 20 known diameter
elements of S_10. Successful larger words were independently replayed.
