# A sharper universal bound: C = 4/3

This bound has since been improved to C=3/4 in
[COMBINED_EXTRACTION_UPPER_BOUND.md](COMBINED_EXTRACTION_UPPER_BOUND.md),
using stronger extraction certificates for both interval stages.

Date: 2026-09-19. This is a computer-assisted proof with an exhaustive
finite case certificate, followed by an all-n argument. It is not an
extrapolation from random permutations.

## Theorem

For the inverse-closed wrapped consecutive 4-cycle generators, and every
n >= 5,

\[
\boxed{D(n)\le
\left\lfloor\frac{\lfloor n^2/4\rfloor+4n+32}{3}\right\rfloor
\le\frac{n^2}{12}+\frac43n+\frac{32}{3}.}
\tag{1}
\]

This improves the previous admissible linear coefficient from 4 to 4/3.
The exact diameter, the proposed coefficient 1/6, and the candidate
D(n)=ceil(n^2/12)+1_{3|n} remain unresolved.

The finite certificate is [adaptive_extraction_c2.json](adaptive_extraction_c2.json).
The independent coverage and word checker is the `independent_tree_check`
function in [adaptive_sort.py](adaptive_sort.py). The verifier uses only
Python's standard library and elementary permutation operations. The
search that discovered the words is not needed to verify the proof.

## 1. Stable extraction and its inversion cost

In a linear list, mark its k smallest entries. A stable extraction places
them, in increasing order, at the front, leaving every unmarked entry in
its original relative order. Write Delta for the inversion decrease of
this operation. It counts the inversions involving at least one marked
entry; the inversions between unmarked entries do not change.

First process the marked entries in their current left-to-right order.
Move each one left across groups of three unmarked entries until fewer
than three unmarked entries precede it after the previous marked entry.
Each such move is an allowed inverse 4-cycle and removes exactly three
inversions. It preserves the relative orders within the marked and
unmarked classes.

After this preparation, every gap before a marked entry has size 0, 1,
or 2. The prefix ending at the last marked entry has length at most 3k.
Its residual pattern records:

- the marked labels 0,...,k-1, in their actual order;
- the unmarked entries, given artificial labels k,k+1,... in their
  left-to-right order.

These artificial labels describe a permutation identity; no assumption
is made that the actual unmarked values were sorted. Sorting the
artificial pattern performs exactly the required stable extraction.
Its ordinary inversion number J is precisely the remaining contribution
to Delta. If preparation used q moves, then Delta=3q+J.

Consequently a residual sorting word of length t satisfying

\[
3t-J\le2k
\tag{2}
\]

gives a complete extraction word of length L=q+t with

\[
3L-\Delta\le2k.
\tag{3}
\]

## 2. The finite extraction lemma

Start with k=3. There are 3! times 3^3 = 162 possible residual patterns.
For each pattern either use a certified word satisfying (2), or add the
next-smallest unmarked entry to the marked set and repeat.

Every possible refinement is explicitly enumerated. The new marked
entry can lie at any unmarked position before the current last marked
entry. Otherwise it lies after that entry; preparation removes whole
groups of three from this final gap, leaving exactly three possibilities.
Thus a pattern p of prefix length ell has the following complete list
of refinements: mark any unmarked position in p, or put the new marked
entry at position ell, ell+1, or ell+2. Relabel the remaining unmarked
entries in their positional order. These are all possibilities; there
is no bound on the original unmarked gaps hidden in this enumeration.

The certificate has the following complete coverage:

| Marked entries k | Patterns encountered | Certified words used | Refined further |
|---:|---:|---:|---:|
| 3 | 162 | 120 | 42 |
| 4 | 200 | 153 | 47 |
| 5 | 198 | 176 | 22 |
| 6 | 85 | 78 | 7 |
| 7 | 26 | 24 | 2 |
| 8 | 11 | 11 | 0 |

In total, there are 682 abstract cases and 562 successful words.
For every successful case the certificate gives an explicit nonwrapped
4-cycle word sorting the artificial permutation. Every word satisfies
(2). Its support is the initial max(9,ell) positions, where ell is the
pattern's prefix length; the largest support is 14 positions.

The independent checker verifies all of the following, rather than
trusting the search's success or failure flags:

1. The initial set is exactly the 162 possible three-entry patterns.
2. Every successful word uses only valid nonwrapped generators, replays
   to the required permutation, and satisfies (2).
3. Every refined pattern has all possible refinements in the next stage,
   with no omitted abstract case.
4. There are no unresolved cases after k=8.

It is immaterial whether a refined case actually had a shorter word.
The proof needs only the successful identities and complete refinement
coverage. Search nonexistence claims play no role in the certificate.

This proves that one can always stably extract between three and eight
of the smallest entries, with the deficit bound (3), provided the
necessary helper positions are available.

## 3. Finishing an interval with temporary helpers

Apply these extractions repeatedly to sort an interval of length m in
a circle of size n >= 9. Previously placed entries may be used as
temporary helpers, but every operation must restore them when its local
word finishes.

When fewer than 14 entries remain, append the following entries around
the circle as helpers, to obtain min(14,n) distinct positions. Assign
them artificial ranks above all the remaining entries, in their current
order. There are enough entries to follow the decision tree through
k=8. A residual prefix has length ell <= n, so each required word has
support max(9,ell) <= n. For n >= 14 this is immediate from the maximum
support of 14; for 9 <= n <= 13 it follows from the actual prefix length.

The resulting finite identity preserves the relative order of all
unmarked entries. If some helpers are marked, they follow all the real
remaining entries in the artificial target order. Therefore all helpers
return to their exact original positions at completion of the local
word. Only the unsorted interval changes. This remains true if helpers
are numerically smaller than the active entries: artificial labels are
used solely to specify the permutation identity.

The preparatory moves cross only active unmarked entries. If k exceeds
the number r of active entries, all r active entries are marked already,
and the additionally marked helpers occur consecutively afterward; no
preparatory move crosses a helper. Thus Delta in (3) is exactly the
inversion decrease inside the active interval. No inversion cost from
temporary motion of helpers has been omitted.

Stop immediately if the remaining interval is sorted. Otherwise, the
total number of entries charged across all extractions is at most m+5:
all but the final extraction charge only real entries, and in the final
one k <= 8. For r >= 3, this adds at most 8-r <= 5 helpers. For r=2,
the only unsorted pattern is an adjacent transposition; its certificate
path succeeds by k=7, again adding at most five helpers. For r=1 there
is nothing to sort.

Summing (3), with I the original interval inversion number, proves

\[
\boxed{3L\le I+2m+10.}
\tag{4}
\]

All generators act on at most n distinct positions. They may cross the
physical boundary of the circle, but each local word realizes exactly
its stated finite permutation and restores every helper. No rotation
of the target is allowed or used for free.

## 4. From the interval bound to the diameter bound

The balanced-lift argument in Sections 2-3 of
[LINEAR_UPPER_BOUND.md](LINEAR_UPPER_BOUND.md) supplies a reduced cyclic
adjacent-swap sorting with three stages:

- sort an interval of length n, with inversion number I_1;
- move one entry clockwise by q positions;
- sort the remaining interval of length n-1, with inversion number I_2.

It satisfies

\[
I_1+q+I_2\le K:=\lfloor n^2/4\rfloor.
\]

Apply (4) to the two intervals. The middle stage has the previously
proved conversion bound 3L_0 <= q+14. Therefore

\[
\begin{aligned}
3L
&\le (I_1+2n+10)+(q+14)
 +(I_2+2(n-1)+10)\\
&\le K+4n+32.
\end{aligned}
\]

This proves (1) for n >= 9. The helper argument is compatible with the
lifted windows: each local word has a finite support interval of at most
n positions, returns helper copies to the same lifted positions, and
performs exactly the desired permutation of the active entries.

For n=5,6,7,8, complete independent enumeration already gives diameters
4,5,6,7, which satisfy (1). Additionally, exhaustive replay of the older
construction on every permutation at these sizes gives maximum word
lengths 7,17,20,24, respectively, also below (1); thus this small extension
does not require trusting the exact-diameter computation.

## 5. Verification scope and open problem

The 682-case certificate is exhaustive over the states used by this
proof, so it is a finite component of the all-n theorem. The following
additional checks validate the implementation:

- the new construction on 1,788 instances, including reversals, half
  rotations and random permutations, through n=1000;
- every permutation for n=5..8;
- every permutation of an eight-entry final interval at n=9 and n=14,
  checking helper windows that cross the cyclic boundary.

Results are in `adaptive_sort_checks.json` and
`adaptive_boundary_checks.json`. The coefficient-3/2 extraction search
in `adaptive_extraction_c3_2.json` is incomplete and is not part of this
proof or of the claimed bound.

This improves a proved universal upper estimate. It does not establish
an exact diameter formula or show that the coefficient 4/3 is optimal.
