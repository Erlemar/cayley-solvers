# Exact word lengths of the two cyclic reflections under consecutive 4-cycles

Date: 2026-09-19.

Let

\[
G_n=\{g_i^{\pm1}:i\in\mathbb Z/n\mathbb Z\},\qquad
g_i=(i,i+1,i+2,i+3).
\]

All positions in a generator are interpreted modulo \(n\). Write

\[
\rho_1(i)=-i\pmod n,\qquad \rho_2(i)=n-1-i\pmod n.
\]

Their one-line forms are, respectively, \([0,n-1,\ldots,1]\) and
\([n-1,n-2,\ldots,0]\). Let \(d_j(n)=|\rho_j|_{G_n}\).

**Theorem.** The two formulas in the supplied screenshot hold:

\[
d_1(n)=\frac{n^2-2n+s_{n\bmod12}}{12}\quad(n\ge13),
\qquad
s=(12,1,0,9,4,9,0,1,12,9,16,9),
\]

\[
d_2(n)=\frac{n^2-2n+r_{n\bmod12}}{12}\quad(n\ge12),
\qquad
r=(0,1,12,9,16,9,12,1,0,9,4,9).
\]

In particular, the linear coefficient is **\(-1/6\)**, as in the screenshot.
The proof below is constructive and uses no extrapolation from computed distances.

## 1. A lower bound from lifted token paths

First allow only cyclic adjacent transpositions \((j,j+1\bmod n)\).
Label each token by its starting position \(i\in\{0,\ldots,n-1\}\).
During a sequence of adjacent swaps, lift its position to an integer \(x_i\),
initially \(x_i=i\). When the tokens at positions \(j\) and \(j+1\pmod n\)
are swapped, increase the lift of the former by one and decrease the lift
of the latter by one. This also defines the lifts for the edge \((n-1,0)\).

Define

\[
\Phi(x)=\sum_{0\le i<j<n}
\left|\left\lfloor\frac{x_j-x_i}{n}\right\rfloor\right|.
\]

Initially \(\Phi=0\). Each adjacent swap changes \(\Phi\) by exactly one
in absolute value. Indeed, for a pair involving an unmoved third token,
the difference changes by at most one and cannot pass a multiple of \(n\):
the moving token crosses only its swap partner. Only the summand for the
two swapped tokens changes; its floor changes by one, so its absolute
value also changes by one. Equivalently, this is the crossing count of
the periodically repeated lifted token paths.

For any reflection \(i\mapsto c-i\pmod n\), the final lifts have the form

\[
x_i=c-i+n k_i,\qquad k_i\in\mathbb Z.
\]

Consequently,

\[
\Phi(x)=\sum_{i<j}|k_j-k_i-1|.
\]

If \(k_i\) and \(k_j\) have the same parity, their summand is at least one.
If there are \(E\) even and \(O=n-E\) odd integers among the \(k_i\), then

\[
\Phi(x)\ge\binom E2+\binom O2
=\binom n2-EO
\ge\binom n2-\left\lfloor\frac{n^2}{4}\right\rfloor
=\left\lfloor\frac{(n-1)^2}{4}\right\rfloor.
\]

Each allowed 4-cycle, in either direction, is a product of three cyclic
adjacent transpositions. Therefore any word of length \(L\) representing
either reflection satisfies

\[
3L\ge A_n,\qquad A_n:=\left\lfloor\frac{(n-1)^2}{4}\right\rfloor. \tag{1}
\]

Furthermore every 4-cycle is odd. Thus

\[
L\equiv p_1(n):=\left\lfloor\frac{n-1}{2}\right\rfloor\pmod2
\quad\text{for }\rho_1,
\qquad
L\equiv p_2(n):=\left\lfloor\frac n2\right\rfloor\pmod2
\quad\text{for }\rho_2. \tag{2}
\]

Let \(\operatorname{round}_p(t)\) denote the least integer at least \(t\)
congruent to \(p\pmod2\). Equations (1)-(2) prove the lower bounds

\[
d_j(n)\ge\operatorname{round}_{p_j(n)}(A_n/3). \tag{3}
\]

The winding numbers \(k_i\) need not be optimized or classified. The
parity argument applies to every possible sequence of moves.

## 2. Optimal reversal of a linear interval

**Lemma.** For every \(m\ge6\), the reversal of a linear interval of length
\(m\) can be performed, using only 4-cycles contained in that interval,
in exactly

\[
F(m)=\left\lceil\frac{m(m-1)}6\right\rceil
\]

moves.

The lower bound follows from the usual inversion number: reversal has
\(m(m-1)/2\) inversions, and one generator changes it by at most three.

For the upper bound, here are explicit base words. The notation \(i^+\)
means a left rotation of the four entries at positions \(i,i+1,i+2,i+3\),
and \(i^-\) means the inverse right rotation. Apply the listed operations
in order to the identity array. All indices are zero based.

| \(m\) | Length | Word reversing the interval |
|---|---:|---|
| 6 | 5 | \(0^+,2^-,0^+,2^-,0^+\) |
| 7 | 7 | \(0^+,2^-,0^+,2^-,0^+,3^-,0^-\) |
| 8 | 10 | \(0^+,0^+,1^-,3^+,2^-,0^+,2^-,4^-,1^-,0^+\) |
| 9 | 12 | \(0^+,1^-,0^+,2^-,3^-,1^+,4^-,5^-,2^-,0^+,2^-,0^+\) |
| 10 | 15 | \(0^-,1^-,2^-,3^-,4^-,6^+,4^-,6^+,4^-,0^-,2^+,0^-,1^-,3^+,2^-\) |
| 11 | 19 | \(0^-,1^-,2^-,3^-,4^-,5^-,6^-,7^+,0^-,3^+,0^-,1^-,4^+,6^+,5^-,1^-,4^+,1^-,0^+\) |

These are finite permutation identities, directly checkable by applying
the rotations. They establish the lemma for six consecutive values of \(m\).

To pass from \(m\) to \(m+6\), split the interval into \(A B\) with
\(|A|=6\), \(|B|=m\). Reverse both blocks, using \(5+F(m)\) moves.
Then exchange the two blocks, preserving their internal order, in \(2m\)
moves. To do so, move the rightmost triple of the six-element block across
\(B\), then its leftmost triple. A triple crosses one entry in one move,
using

\[
[a,b,c,x]\longmapsto[x,a,b,c].
\]

Thus the final array is \(B^{\mathrm{rev}}A^{\mathrm{rev}}\), the desired
reversal, and its word has length

\[
F(m)+2m+5
=\left\lceil\frac{m(m-1)}6\right\rceil+2m+5
=\left\lceil\frac{(m+6)(m+5)}6\right\rceil.
\]

This proves the lemma for every \(m\ge6\).

## 3. Cutting a cyclic reflection into two interval reversals

For \(a+b=n\), reverse the intervals \([0,a-1]\) and \([a,n-1]\).
Their product is the reflection

\[
\sigma_a(i)=a-1-i\pmod n.
\]

Conjugation by the rotation \(i\mapsto i+t\) changes its parameter from
\(a-1\) to \(a-1+2t\). This conjugation preserves the generator set:
it simply replaces every index \(i\) in the word by \(i+t\pmod n\).
**No rotation is being counted as a free move.** We are relabeling a word
via a graph automorphism, and every resulting letter is still an allowed
generator.

When \(n\) is odd, \(2\) is invertible modulo \(n\), so either reflection
is conjugate to \(\sigma_a\) for any \(a\). Choose

\[
a=\lfloor n/2\rfloor,\qquad b=\lceil n/2\rceil.
\]

When \(n=2m\) is even, \(\rho_1\) requires \(a\) odd, whereas \(\rho_2\)
requires \(a\) even. Choose the balanced split of the required parity:

\[
\begin{array}{c|cc}
&\rho_1&\rho_2\\ \hline
m\text{ even}&(a,b)=(m-1,m+1)&(a,b)=(m,m)\\
m\text{ odd}&(a,b)=(m,m)&(a,b)=(m-1,m+1).
\end{array}
\]

Both interval lengths are at least six for \(\rho_1\) when \(n\ge13\),
and for \(\rho_2\) when \(n\ge12\). The lemma therefore gives

\[
d_j(n)\le U_j(n):=
\left\lceil\frac{a(a-1)}6\right\rceil+
\left\lceil\frac{b(b-1)}6\right\rceil. \tag{4}
\]

## 4. The upper and lower bounds agree

Substitute \(n=12q+t\) in (3) and (4). For each \(t\in\{0,\ldots,11\}\),
both give \((n^2-2n+c_t)/12\), with:

| \(t\) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| \(s_t\), shifted flip | 12 | 1 | 0 | 9 | 4 | 9 | 0 | 1 | 12 | 9 | 16 | 9 |
| \(r_t\), full flip | 0 | 1 | 12 | 9 | 16 | 9 | 12 | 1 | 0 | 9 | 4 | 9 |

For transparency, this last arithmetic also has a short form. If
\(n=2m+1\), the common upper bound is

\[
\left\lceil\frac{m(m-1)}6\right\rceil+
\left\lceil\frac{m(m+1)}6\right\rceil
=\left\lceil\frac{m^2}{3}\right\rceil.
\]

It has parity \(m\), so equals the parity-rounded lower bound because
\(A_n=m^2\). The identity follows by checking \(m\bmod6\).
If \(n=2m\), then \(A_n=m(m-1)\); inserting either the split \((m,m)\)
or \((m-1,m+1)\) and checking \(m\bmod6\) gives the two even columns.

Thus (3) and (4) coincide, proving the theorem. In particular both
reflections obey the sharp uniform upper estimate

\[
d_j(n)\le\frac{n^2}{12}-\frac n6+\frac43
\]

in their stated ranges. No smaller constant coefficient of \(n\) can
work with a bounded additive term, because the exact formulas have
linear coefficient \(-1/6\).

## 5. Verification and provenance

- `verify_theorem.py` is a self-contained certificate checker and word
  constructor using only Python's standard library. It contains the six
  base words above; no search is needed to run the proof construction.
- All linear reversal words for \(6\le m\le250\) were replayed.
- 485 complete reflection words were replayed: every applicable size
  from 12 through 250, and sizes 503, 1000, 1001, and 1012.
- Both residue formulas were checked against the parity-rounded lower
  bound for every \(13\le n\le10000\).
- The lift-potential change was checked on 13,000 random adjacent swaps.
- Exhaustive breadth-first search for \(5\le n\le9\) confirmed that small
  exceptions really occur. These checks supplement the proof; they do
  not replace either the induction or the lower-bound argument.
- `verification.json` records the checks. `explore.py` and `monotone.py`
  are discovery scripts; their search results are not assumed in the proof.

Source context: the supplied screenshot states exactly the two formulas
proved here. Section 9 of [Chervov et al., arXiv:2603.22195](https://arxiv.org/pdf/2603.22195)
discusses wrapped consecutive cycles (section opening on PDF page 88;
full-flip discussion in section 9.8). The proof above is self-contained
and does not assume the paper's experimental tables or conjectures.
