# Critical analysis: "Cayleypy diameter bounding" note

**Subject:** `megaminx/Cayleypy_diameter_bounding.pdf` (4 pp., unsigned) — lower and upper
bounds for the diameter `D^n_C` of the Cayley graph of `<C>` where `C` is a full
conjugacy class of S_n with cycle type `2^{m2} 3^{m3} ... k^{mk}`.
**Analysis date:** 2026-07-11. All numerics computed from scratch for this review
(exact BFS over the full Cayley graph for n <= 8; conjugation-invariant
class-level BFS for n <= 20; methodology in section 3.1).

**Notation:** `S = sum j*m_j` (generator support), `S' = sum (j-1)*m_j`
(transposition cost of one generator), `c(sigma)` = number of cycles *including*
fixed points, `e(sigma)` = number of even-length cycles, `supp` = moved points.
The literature's `delta(C)` / `r(D)` is identical to the note's `S'`.

---

## 0. Verdict (TL;DR)

1. **The mathematics is essentially sound.** Theorem 2 (lower bound), Lemma 5,
   Theorem 7 (scaling) and Corollary 8 are correct. Theorem 4 (subadditivity) is
   correct as stated, but its proof has one real gap (a parity issue: the
   block-restricted permutation may be odd while the sub-class generates only
   A_n), one wrong numeric claim ("< 3" should be `< K + K' + 2`), and it
   silently uses an unproven (true) lemma that bounded-support elements cost
   O(1) generators. All three are fixable — fixes in section 2. Lemma 6 has no
   proof in the note; a complete route is given in section 2.5.
2. **Every numeric claim checks out** against BFS: `D_(2) = n-1` (n <= 12),
   `D_(3) = floor(n/2)` (n <= 20), and `D_{2^a 3^b} = n/(a+2b) + O(1)` with
   additive error in {0, 1} (n <= 18).
3. **Literature:** the lower bound and both lemmas are known (3-cycle distance:
   Suceava–Stong 2003; class `2^k` diameters known *exactly*,
   Phongpattanacharoen–Siemons 2013; single l-cycle classes known exactly,
   Bertram 1972 ... Kishnani–Kundu–Mishra 2022/24; the Theta(n/S') order is
   Liebeck–Shalev 2001). **The subadditivity theorem as a packaged general
   statement, and Corollary 8 for mixed `2^a 3^b` (and pure `3^b`, b >= 2),
   appear to be new.** Details in section 4.
4. **The natural extrapolation of the note — `D = n/S' + O(1)` for every class —
   is FALSE.** It fails for every class containing a cycle of length >= 4 that
   is not compensated by 2-cycles: `D_(4) = ceil(n/2)` (not ~n/3), `D_(5) ~ n/3`
   (not n/4), `D_(6) ~ n/4` (not n/5) — all BFS-verified to n = 16..20. The
   obstruction is a Riemann–Hurwitz/genus count on fixed-point-free involutions
   (section 5), and the corrected general rate conjecture (section 5.3) unifies
   *all* known exact results. The block calculus made a genuine advance
   prediction — `D_(5)(16) = 6` where the naive fit from n <= 14 said 5 — that
   BFS then confirmed.
5. **Exact quasipolynomial formulas** for ~12 class families (proven or
   BFS-verified conjectures) are collected in section 6: in every verified case
   the diameter is eventually `ceil((n - eps)/R(C))` plus an explicit
   parity-periodic correction.

---

## 1. What the note proves (recap)

* **Thm 2 (lower bound).** `D >= (n-1)/S'` when `<C> = S_n`; `D >= (n-2)/S'`
  when `<C> = A_n`. Proof: an n-cycle (resp. (n-1)-cycle) needs `n-1` (resp.
  `n-2`) transpositions; each generator supplies at most `S'`.
* **Thm 4 (subadditivity).** If `D_m <= n/K + O(1)` and `D_{m'} <= n/K' + O(1)`
  then `D_{m+m'} <= n/(K+K') + O(1)`. Device: split `[n]` into two blocks of
  sizes ~`alpha K`, ~`alpha K'`; factor the block restrictions of sigma by the
  two sub-classes; pad both factorizations to a common length; pair the i-th
  factors (disjoint supports => each pair is one type-(m+m') generator).
* **Lemma 5.** `D_(2) = n-1`. **Lemma 6 (proof omitted).** `D_(3) = floor(n/2)`.
* **Thm 7 (scaling).** `m -> d*m` multiplies the rate `K` by `d`.
* **Cor 8.** For classes of 2- and 3-cycles only: `D = n/(m2 + 2 m3) + O(1)`.

## 2. Logic verification, theorem by theorem

### 2.1 Theorem 2 — correct; can be sharpened for free

Airtight: distance is monotone in the metric `d_T(sigma) = n - c(sigma)` (a
transposition changes `c` by exactly +-1; one generator moves `d_T` by <= `S'`).

Two costless improvements:

* State it with a **ceiling**: `D >= ceil((n-1)/S')` / `ceil((n-2)/S')` — the
  LHS is an integer. With the ceiling the bound is *exactly attained* for class
  `2^2` at every n tested (5..20): `D_{(2,2)} = ceil((n-2)/2)`; likewise for
  class `(2,4)`: `D = ceil((n-2)/4)` (7..16).
* Record the **per-element** version `dist(sigma) >= ceil((n - c(sigma))/S')` —
  that is what applications (admissible search heuristics) actually use.

Nits: the class needs `n >= S` to be nonempty; Remark 3 (normality => `<C>` is
S_n or A_n for n >= 5) is correct — worth one citation to simplicity of A_n.

### 2.2 Theorem 4 — statement correct; one genuine gap, two slips

The core idea — pair up two padded factorizations of disjoint permutations to
get a factorization by the merged class — is elegant, correct, and per the
literature search (section 4) does not appear in this generality anywhere.

Issues, decreasing severity:

* **(Gap) Parity of the block restriction.** The proof factors `tau` (sigma
  restricted to the first block, modified at <= 1 point) as a product of
  class-m elements "by our assumption". But the assumption bounds the diameter
  of `<C_m>`, which may be the *alternating* group on the block, while `tau`
  may be **odd** — then tau has no factorization at all and the step is
  vacuous. Concrete instance: m = (3) (3-cycles), m' = (2), sigma an odd
  element of `<C_{(2,3)}> = S_n` whose first-block restriction is odd.
  **Fix (cheap):** when tau has the wrong sign, factor `tau * (x y)` instead
  (transposition inside the block); rho then disagrees with sigma at 2 more
  points, absorbed by the final O(1) cleanup. Same for tau'.
* **(Slip) The "< 3" count.** Points untouched by both blocks:
  `n - floor(alpha K) - floor(alpha K') = (n - alpha(K+K')) + {alpha K} + {alpha K'}`,
  and `n - alpha(K+K') = (K+K') * {n/(K+K')}` can be up to `K + K' - epsilon`.
  Correct bound: `< K + K' + 2`, not `< 3` (e.g. K = K' = 5, n = 19: alpha = 1,
  nine points untouched). Constants only — the O(1) conclusion stands — but the
  displayed equation is false as written.
* **(Unproven lemma) Final step.** "It suffices to bound the number of
  non-fixed points of `sigma rho^{-1}`" presumes: *any element of `<C>` with
  support <= B is a product of `f(B, type)` generators, uniformly in n.* True
  but needs proof. Sketch: for `g` in C and a 3-cycle `t`,
  `[g, t] = (g t g^{-1}) * t^{-1}` is a product of two class elements (classes
  are inverse-closed) and is a nontrivial even permutation of support <= 6 for
  suitable t; so `C^2` contains a whole nontrivial class `W` of bounded
  support; bounded powers of `W` contain all 3-cycles; every even
  bounded-support element is then in `C^{O(B)}`, odd ones cost one extra
  generator.
* **(Missing one-liner) The WLOG.** Writing sigma with consecutive-interval
  cycles is legitimate because *distance from id is a class function* (the
  generating set is conjugation-closed): relabeling = conjugating. Used
  silently; also load-bearing for the numerics below.
* The padding trick (append `g g^{-1}` pairs; when the count parity is wrong,
  replace the target tau by `tau * pi`) is correct as written.

### 2.3 Theorem 7 — correct (and deletable)

Both directions verify; the key inequality `D_m <= d * D_{dm} + 1` is proven
cleanly (split each type-dm generator into d disjoint type-m elements; one
extra generator repairs parity). In the reformulation of section 5.4, Theorem 7
is a corollary of Theorem 4 + Theorem 2 and could be a remark.

### 2.4 Lemma 5 — correct (classical)

`dist(sigma) = n - c(sigma)` exactly (BFS-verified for every type, n <= 12);
diameter n-1, witnesses = n-cycles only. Folklore/Cayley; also
Phongpattanacharoen–Siemons 2013, Prop. 2.1.

### 2.5 Lemma 6 — true; the missing proof

The exact per-element formula is known (**Suceava–Stong, Amer. Math. Monthly
110 (2003) 162**): for even sigma the minimal number of 3-cycles is

    dist(sigma) = (supp - s)/2 = (n - c + e)/2

(`s` = number of odd cycles of length >= 3; the two forms are identical).
**BFS-verified for every cycle type at every n in 5..12 — zero mismatches.**
Diameter: maximize `sum_cycles (l - 1 + [l even])` over even types; the max is
`2*floor(n/2)` with witnesses

* n = 0 mod 4: `2^{n/2}`;
* n = 2 mod 4: `2^{(n-4)/2} 4` (the 4-cycle repairs the sign of e) — exactly
  the BFS witness at n = 6 ((2,4) is the *unique* type at distance 3) and n=10,14;
* n odd: same plus a fixed point.

Note the witness is **not** the n-cycle. This is the first hint that Theorem
2's witness family stops being extremal for other classes — section 5.

### 2.6 Corollary 8 — correct given Lemma 6

With the ceiling version of Theorem 2:
`ceil((n-2)/(m2+2m3)) <= D <= n/(m2+2m3) + O(1)`. The BFS truth is within +1 of
the lower bound, the +1 exactly predictable by parity (section 6).

---

## 3. Numerical verification

### 3.1 Methodology

* **Exact BFS** over the full Cayley graph for n <= 8, all class types with
  S <= 6 (plus spot checks at n = 9). Ground truth.
* **Class-level BFS** for n <= 20: the generating set is conjugation-closed, so
  distance is a class function and the neighbour *types* of a type T can be
  computed from one representative x (all generators), vectorized in numpy.
  Nodes = partitions of n (p(20) = 627). **Validated against exact BFS on all
  20 overlapping (n, class) cases — perfect agreement.**

### 3.2 Diameter table (BFS values `D^n_C`)

S' = the note's rate; R = true asymptotic rate (section 5). Bold row groups:
{2,3}-classes (R = S'), then deficient classes (R < S').

| class | S' | R | n=5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (2) | 1 | 1 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | . | . | . | . | . | . | . | . |
| (3) | 2 | 2 | 2 | 3 | 3 | 4 | 4 | 5 | 5 | 6 | 6 | 7 | 7 | 8 | 8 | 9 | 9 | 10 |
| (2,2) | 2 | 2 | 2 | 2 | 3 | 3 | 4 | 4 | 5 | 5 | 6 | 6 | 7 | 7 | 8 | 8 | 9 | 9 |
| (2,3) | 3 | 3 | 3 | 3 | 3 | 3 | 4 | 4 | 4 | 5 | 5 | 5 | 6 | 6 | . | 7 | . | . |
| (2,2,2) | 3 | 3 | . | 5 | 4 | 3 | 4 | 4 | 4 | 5 | 5 | 5 | 6 | 6 | . | 7 | . | . |
| (3,3) | 4 | 4 | . | 3 | 2 | 2 | 2 | 3 | 3 | 3 | 3 | 4 | 4 | 4 | . | . | . | . |
| (2,4) | 4 | 4 | . | 2 | 2 | 2 | 2 | 2 | 3 | 3 | 3 | 3 | . | 4 | . | . | . | . |
| (2,2,2,2) | 4 | 4 | . | . | . | 4 | 3 | 3 | . | 3 | . | 3 | . | . | . | . | . | . |
| **(4)** | 3 | **2** | 3 | 3 | 4 | 4 | 5 | 5 | 6 | 6 | 7 | 7 | 8 | 8 | 9 | 9 | 10 | 10 |
| **(5)** | 4 | **3** | 2 | 2 | 2 | 3 | 3 | 3 | 4 | 4 | 4 | 5 | 5 | 6 | 6 | 6 | . | 7 |
| **(6)** | 5 | **4** | . | 3 | 3 | 3 | 3 | 4 | 4 | 4 | 5 | 5 | . | 5 | . | . | . | . |
| **(3,4)** | 5 | **4** | . | . | 3 | 3 | . | . | . | 4 | 4 | 5 | 5 | . | . | . | . | . |
| (2,5) | 5 | 5? | . | . | 3 | 3 | . | . | . | 4 | 4 | 4 | 4 | . | . | . | . | . |
| (2,2,4) | 5 | 5 | . | . | . | 3 | . | . | . | 3 | . | . | . | . | . | . | . | . |
| (7) | 6 | 4 | . | . | 2 | 2 | 2 | . | . | . | 3 | . | . | . | . | . | . | . |
| (4,4) | 6 | 4 | . | . | . | 2 | . | 2 | . | 3 | . | . | . | . | . | . | . | . |

(Compare the (2,4) row with the (4) row — the added 2-cycle repairs the
4-cycle's deficiency, dropping the diameter from ~n/2 to ~n/4. Class (2,2,3)
was not run; predicted R = 4.)

Headline observations:

* **Theorem 2 is never violated**, and for `(2,2)` and `(2,4)` its ceiling
  version is *exactly* attained at every n tested.
* **Corollary 8 confirmed to n = 18**: all-{2,3} classes track `n/S'` with
  additive error in {0, 1}, the +1 being a computable parity correction.
* **Classes with an uncompensated cycle of length >= 4 run strictly above
  `n/S'`**: `(4)` sits at `ceil(n/2)` — 50% above the Theorem-2 line — through
  n = 20; `(5)` at ~n/3 (S' = 4); `(6)`, `(7)` at ~n/4 (S' = 5, 6).
* **Diameter witnesses flip type between the regimes**: long cycles for the
  {2,3} classes (witness (13) for (2,2) at n=13, (12) for (2,3) at n=12) vs
  **fixed-point-free involutions and near-involutions** for the deficient
  classes (2^10 for (4) and (5) at n=20, 2^7 for (3,4) at n=14).
* **Advance prediction confirmed:** from n <= 14 data alone the (5)-class fit
  was `ceil((n-1)/3)` (predicting D(16) = 5), while the genus calculus of
  section 5 predicted `d(2^8) = 6`. BFS at n = 16: diameter 6, witness `2^8`.

### 3.3 Per-element distance formulas (verified against full distance tables)

| class | distance formula | verification |
|---|---|---|
| (2) | `n - c` | all types, exact |
| (3) | `(n - c + e)/2` = Suceava-Stong `(supp - s)/2` | all types, exact |
| (2,2) | `ceil((n-c)/2)` | exact except sigma = one 3-cycle (needs 2) |
| (4) | Ghaffari-Mostaghim `supp/3 + (c2-c1)/3 + [c0 odd]`, c_i = nontrivial cycles of length = i mod 3 | exact except sigma = one transposition (needs 3) |

The exceptions are honest small-support boundary cases. These formulas are the
engine behind the exact diameter quasipolynomials of section 6.

---

## 4. Placement in the literature

* **The asymptotic order is 25 years old.** Liebeck–Shalev (Ann. of Math. 2001):
  `diam Cay(G, C) = Theta(log|G|/log|C|)` for any class of any finite simple
  group — for fixed cycle type, `Theta(n/S')` with an *unspecified absolute
  constant*. The note's target — the exact constant — is a regime L–S do not
  address.
* **The lower bound is standard.** `S'` is the literature's
  `delta(C) = n - c` (Dvir 1985: `r(D)`); `D >= (n-1)/S'` is the trivial
  direction of every covering-number computation. Dvir Thm 10.2: `[D]^3 = A_n`
  once `r(D) >= (n-1)/2`. Stavi/Dvir: the covering number of A_n is
  `floor(n/2)` (+1), attained by the 3-cycles — i.e. Lemma 6 is the extremal
  case of a known theorem.
* **Known exact results for the note's specific classes:**
  * 3-cycles: Suceava–Stong 2003; Ghaffari–Mostaghim 2017 (diam = floor(n/2)).
  * `2^k` (pure m2): **Phongpattanacharoen–Siemons 2013** determine every
    metric sphere for n >= 4k: diam = `ceil((n-2)/k)` (k even) and a two-case
    max formula (k odd). The m3 = 0 slice of Corollary 8 is known *exactly*.
  * single l-cycles: Bertram 1972 (two conjugate cycles); Bertram–Herzog 2001
    (3 and 4 cycles); Herzog–Kaplan–Lev 2008; **Kishnani–Kundu–Mishra 2022/24**
    close the problem with exact thresholds `n(k,l) = (2/3)kl + O(1)` (mod-3
    and mod-4 case formulas). As diameters: rate = `floor(2l/3)`.
* **What appears new in the note:** (a) the subadditivity theorem as a general
  packaged statement (the device is used ad hoc inside Bertram/HKL/KKM/P–S
  proofs, but no general lemma is stated anywhere found); (b) Corollary 8 for
  mixed `2^a 3^b`, a,b >= 1, and pure `3^b`, b >= 2 — Ghaffari–Mostaghim
  explicitly flag the disjoint-class case as *open*. Best positioning: *"the
  exact diameter constant for all {2,3}-classes, via a general subadditivity
  lemma."*
* **CayleyPy papers** (arXiv 2502.13266, 2502.18663): no overlap — their
  diameter material concerns bounded generating sets (LRX), not classes.

---

## 5. Where the program breaks, and the corrected general picture

### 5.1 The involution obstruction (why long cycles are transposition-inefficient)

Theorem 2's metric `n - c` is blind to a second constraint. Build the
fixed-point-free involution `2^{n/2}` from l-cycles and split the factorization
into components (orbits of the group the factors generate). A component with d
factors and support s obeys the Riemann–Hurwitz / Frobenius genus identity

    d(l-1) = s + c(sigma_P) - 2 + 2g,   g >= 0 integer.

For an involution `c = s/2`, so `d(l-1) = (3/2)s - 2 + 2g`: at most 2/3 of each
factor letter converts into involution support — **plus integrality** (s even,
g integer >= 0, one-factor components are just the generator itself). Optimizing
`s/d` over feasible blocks:

| l | best block | transpositions per generator | involution cost |
|---|---|---|---|
| 3 | 2 gens -> 2^2 (g=0) | 1 | n/2 |
| 4 | 2 gens -> 2^2 (g=1); *no* g=0 block exists (3d = (3/2)s - 2 insoluble) | 1 | n/2 |
| 5 | 4 gens -> 2^6 (g=0) | 3/2 | n/3 |
| 6 | 2 gens -> 2^4 (g=0) | 2 | n/4 |
| 7 | 2 gens -> 2^4 (g=1) | 2 | n/4 |

Every row matches the BFS witnesses/costs *and* the KKM exact theorems. Closed
form: **rate `R_l = floor(2l/3)`** — one formula unifying KKM's three
congruence cases, Bertram's theorem (the d = 2 rows are Bertram's
`l >= ceil(3t/2)` refined by the parity constraint that two conjugate cycles
make an even permutation, so t must be even when l is odd), and Lemma 6 (l = 3
is the last cycle length with `floor(2l/3) = l - 1`).

Hence for l >= 4 the diameter witness is the involution family, not the
n-cycle, and `D_(l) = n/floor(2l/3) + O(1) > n/(l-1) + O(1)`. **Theorem 2 is
asymptotically tight exactly for classes built of 2- and 3-cycles** — the note
stops at precisely the right boundary.

### 5.2 Mixed classes: cycles rescue each other

A generator's different cycles may land in different components, so the block
calculus composes. Data-confirmed consequences:

* `(2,4)`: pair the 4-parts Bertram-style (2 transpositions per 2 gens), use
  the 2-parts directly (2 more) => rate 2/gen => `R = 4 = S'` — the 2-cycle
  *repairs* the 4-cycle's mod-3 deficiency. BFS: `D = ceil((n-2)/4)`, 7..16,
  witnesses `2^t` at cost exactly `t/2`. Contrast pure `(4)`: R = 2, `ceil(n/2)`.
* `(3,4)`: 3-parts pair at rate 1, 4-parts at rate 1 => R = 4 < S' = 5. BFS:
  D(12..14) = 4, 4, 5 with involution witness `2^7` at n = 14 (parity forces
  the odd count) — consistent with ~n/4, clearly below the S' line n/5.
* `(2,5)`: predicted R = 5 (= S'), but the optimal 5-part block needs 4
  generators and ~20 points; convergence is slow and BFS at n <= 15 (D = 4, 4,
  4, 4) cannot yet separate R = 5 from R = 4. **First genuinely open test
  case.** Directional hint: at n = 15 and equal S' = 5, D_{(2,5)} = 4 <
  D_{(3,4)} = 5 — the ordering the conjecture predicts (R = 5 vs 4).

### 5.3 Corrected general conjecture

**Conjecture.** For every fixed cycle type C,

    D^n_C = n/R(C) + O(1),
    R(C)  = min( S'(C),  2 m2 + sum_{j>=3} m_j floor(2j/3) ).

First term: Theorem 2 (long-cycle targets). Second: the involution-family cost
(each generator 2-cycle contributes a full transposition — worth 2 in this
bookkeeping vs 1 in S'; each j-cycle contributes floor(2j/3) <= j-1).
Equivalently: `R = S'` iff `m2 >= sum_{j>=4} ceil((j-3)/3) m_j` ("enough
2-cycles to amortize the long cycles"; 3-cycles are exactly neutral).

This reproduces **every proven exact result** — transpositions (R=1), 3-cycles
(R=2), `2^k` (R=k, P–S), l-cycles (R = floor(2l/3), KKM), all {2,3}-classes
(R = S', the note's Corollary 8) — and matches all our BFS data for (2,4),
(3,4), (4,4), (2,2,4), (6), (7) in their verified ranges.

Division of labor for a proof: the note's own Theorem 4 + the KKM single-cycle
results already give the upper bound `D <= n/(2 m2 + sum m_j floor(2j/3)) + O(1)`
coordinate-wise; the synergy cases (min jumping to S', e.g. (2,4)) need one
finite pairing construction per class. **The genuinely open half is the
matching lower bound for mixed types** — the genus argument above needs an
existence-free version when generator cycles split across components.

### 5.4 Simplification: the rate calculus

Define `rho(C) = lim n/D^n_C`. The note compresses to:

* Thm 2:   `rho(C) <= S'(C)` (refined by 5.1: `rho(C) <= R(C)`).
* Thm 4:   rho is **superadditive**: `rho(m + m') >= rho(m) + rho(m')`.
* Thm 7:   rho is **homogeneous** — now a corollary (superadditivity + squeeze),
  so Theorem 7 can be deleted.
* Lemmas 5, 6: `rho(2) = 1`, `rho(3) = 2`.
* Cor 8:   squeeze `S' >= rho >= m2 rho(2) + m3 rho(3) = S'`.

This makes the general program transparent — *rho is a superadditive,
homogeneous function on cycle types, bounded by min(S', involution rate);
computing it reduces to single-cycle classes plus synergy corrections* — and
exposes in one line why Corollary 8 cannot extend past {2,3}: superadditivity
gives `rho(m) >= sum m_j rho(j)`, but `rho(j) = floor(2j/3) < j-1` for j >= 4,
so the squeeze fails, and the data says the failure is real.

---

## 6. Exact quasipolynomial formulas

All verified diameters are eventually quasipolynomial in n. "Proven" = in the
literature; "conj" = BFS-backed conjecture with verified range.

| class | exact diameter | status |
|---|---|---|
| `2` | `n - 1` | proven (classical) |
| `3` | `floor(n/2)` | proven (Suceava–Stong; extremal case of Stavi/Dvir); BFS 4..20 |
| `2^k`, k even | `ceil((n-2)/k)` (n >= 4k) | proven (P–S 2013); k=2 BFS 5..20 |
| `2^k`, k odd | `max(2 ceil((n-2)/(2k)), 2 ceil((n-k-1)/(2k)) + 1)` (n even, n >= 4k; sibling for n odd) | proven (P–S 2013); k=3 BFS agrees for all n >= 8 |
| `4` | `ceil(n/2)` (n >= 5) | proven (G–M 2017); BFS 5..20 |
| `5` | `min{ k : n <= 3k + [k = 0 mod 4] }` (period-12 quasipolynomial) | KKM thresholds; BFS 6..20 exact |
| `6` | `ceil(n/4) + 1` (10 <= n <= 16 verified; parity wiggle at odd n) | conj |
| `l` general | `n/floor(2l/3) + O(1)`, exact thresholds with mod-3(l)/mod-4(k) corrections | proven (KKM 2022/24) |
| `2,3` | `ceil((n+1)/3)` (n >= 6) | conj [6..18]; = Thm-2 bound +1 exactly when the n-cycle witness needs a sign fix: class (2,3) is odd, so d must satisfy `(-1)^d = sgn(witness)` |
| `3,3` | `floor((n+2)/4)` (n >= 7) | conj [7..16] |
| `2,4` | `ceil((n-2)/4)` (n >= 7) | conj [7..16] — Thm 2 ceiling exactly tight, like (2,2) |
| `2,2,2,2` | `ceil((n-2)/4)` (n >= 12) | P–S prove n >= 16; BFS closes 12..15 |

**General conjectured shape** (consistent with every row): with R = R(C) from
5.3 and eps = 1 (S_n) or 2 (A_n),

    D^n_C = ceil((n - eps)/R) + chi(n)      for n >= n_0(C),

`chi(n) in {0,1}` equal to 1 exactly when every family attaining the ceiling
has a sign obstruction (`(-1)^{d}` vs `sgn C^d` — computable per residue class
of n). Proving this in general contains KKM as a special case, so it is not
cheap; as a *conjecture generator* it turns each new class into a finite
computation.

---

## 7. Recommendations for the note's author

1. Add the ceiling to Theorem 2; record the per-element version
   `dist(sigma) >= ceil((n - c(sigma))/S')`.
2. Patch Theorem 4: (i) parity of tau (factor `tau * (xy)`, +2 defect points);
   (ii) replace "< 3" by "< K + K' + 2"; (iii) state + prove the
   bounded-support lemma (commutator trick, 2.2); (iv) one sentence justifying
   the relabeling WLOG by conjugation-invariance.
3. Prove Lemma 6 via Suceava–Stong (with the identity
   `(supp - s)/2 = (n - c + e)/2` the diameter maximization is two lines) —
   and note the witness is an involution-type, not the n-cycle.
4. Demote Theorem 7 to a remark (follows from Thm 4 + Thm 2).
5. Cite: Liebeck–Shalev 2001; Phongpattanacharoen–Siemons 2013 (subsumes the
   m3 = 0 case exactly); Suceava–Stong 2003; KKM 2022/24 (why the program stops
   at {2,3}); Dvir 1985 (S' = delta tradition).
6. The strongest framing available: *Theorem 2 is asymptotically tight if and
   only if the class consists of 2- and 3-cycles* — given KKM, the note is one
   lemma short of this clean boundary theorem.
7. Natural next results, in increasing difficulty: (a) prove
   `D_{(2,4)} = ceil((n-2)/4)` (all ingredients on hand: Thm 4 + Bertram
   pairing + direct 2-parts); (b) settle `(2,5)` — the first case where the
   conjectured min jumps to S' via a *large* block (BFS inconclusive below
   n ~ 20); (c) the general involution lower bound for mixed types (genus
   argument of 5.1 with generator cycles split across components); (d) the
   chi(n) parity layer of section 6.

---

## Appendix A: key references

* M.W. Liebeck, A. Shalev, *Diameters of finite simple groups: sharp bounds and
  applications*, Ann. of Math. 154 (2001) 383–406.
* Y. Dvir, *Covering properties of permutation groups*, in: Products of
  Conjugacy Classes in Groups, LNM 1112, Springer 1985 (Thm 10.2; r(D) = S').
* J.L. Brenner, *Covering theorems for FINASIGs VIII*, J. Austral. Math. Soc. A
  25 (1978) — cn(2^{n/2}) = 4: an early S'-inefficient example.
* E. Bertram, *Even permutations as a product of two conjugate cycles*, JCTA 12
  (1972) 368–380.
* E. Bertram, M. Herzog, *Powers of cycle-classes in symmetric groups*, JCTA 94
  (2001) 87–99.
* G. Boccara, *Nombre de representations d'une permutation comme produit de
  deux cycles de longueurs donnees*, Discrete Math. 29 (1980) 105–134.
* M. Herzog, K.B. Reid, *Number of factors in k-cycle decompositions of
  permutations*, LNM 560 (1976).
* M. Herzog, G. Kaplan, A. Lev, *Covering the alternating groups by products of
  cycle classes*, JCTA 115 (2008) 1235–1245.
* H. Kishnani, R. Kundu, S.C. Mishra, *Alternating groups as products of cycle
  classes* I: Discrete Math. 346 (2023) 113470 (arXiv:2207.03165); II: J.
  Algebraic Combin. (2024) (arXiv:2210.15354).
* B. Suceava, R. Stong, *The fewest 3-cycles to generate an even permutation*,
  Amer. Math. Monthly 110 (2003) 162.
* T. Phongpattanacharoen, J. Siemons, *Metric intersection problems in Cayley
  graphs and the Stirling recursion*, Aequat. Math. 85 (2013) 387–408
  (arXiv:1202.4493).
* M.H. Ghaffari, Z. Mostaghim, *Distance in Cayley graphs on permutation groups
  generated by k m-cycles*, Trans. Comb. 6 (2017) 45–59 (state the disjoint
  class-`m^k` diameter as an open problem).
* N. Keller, N. Lifshitz, O. Sheinfeld, *Improved covering results for
  conjugacy classes of symmetric groups via hypercontractivity*, Forum Math.
  Sigma (2024).
* A.J. Malcolm, *The p-width of the alternating groups*, arXiv:1710.04972.

## Appendix B: reproduction

Scripts (session scratchpad, `bfs_exact.py`, `bfs_class.py`, `verify_formulas.py`):

* exact BFS: permutations as tuples, generators = full class, BFS from id,
  per-type max distances;
* class BFS: types = partitions; neighbours of a type = cycle types of
  (representative x every generator), numpy-vectorized cycle extraction,
  `np.unique` on count signatures; correctness rests on distance being a class
  function (conjugation-closed generators);
* runtimes: n = 20 single-cycle classes seconds-to-minutes; the largest jobs
  (~2 x 10^8 products) minutes.
