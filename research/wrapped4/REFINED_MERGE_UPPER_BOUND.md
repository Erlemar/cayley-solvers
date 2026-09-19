# A universal diameter bound with linear coefficient 5/6

The global coefficient is now improved to 3/4 in
[COMBINED_EXTRACTION_UPPER_BOUND.md](COMBINED_EXTRACTION_UPPER_BOUND.md).
The merge lemma proved here remains a component of that result.

Let D(n) be the diameter of S_n with the inverse-closed generators
(i,i+1,i+2,i+3), with positions read modulo n. Put K=floor(n^2/4).

**Theorem.**

\[
\boxed{D(n)\le\left\lfloor\frac{2K+5n+73}{6}\right\rfloor
\le\frac{n^2}{12}+\frac56n+\frac{73}{6}\qquad(n\ge30).}
\tag{1}
\]

A uniform version for every n >= 5 is

\[
\boxed{D(n)\le\left\lfloor\frac{2K+5n+109}{6}\right\rfloor
\le\frac{n^2}{12}+\frac56n+\frac{109}{6}.}
\tag{2}
\]

This is a computer-assisted proof of an upper bound. It does not prove
the exact diameter or the conjectured linear coefficient 1/6. The finite
part consists solely of explicit permutation identities and exhaustive
coverage checks. No inference from an unsuccessful search is needed.

## 1. The improved merge lemma

Consider an interval of m entries that is the concatenation of two
increasing lists A and B. Let I be its ordinary inversion number. If at
least 30 distinct cyclic positions are available, including surrounding
positions that may be used and restored as helpers, then

\[
\boxed{3L_{\rm merge}\le I+\frac m2+13.}
\tag{3}
\]

We prove this by stable extraction of k smallest entries at a time.
Choose a entries from the beginning of A and b from the beginning of B,
where a+b=k. Write |A|-a=3u+r, with r in {0,1,2}. Moving each chosen
B-entry left across u triples of unchosen A-entries uses bu generators
and removes exactly 3bu inversions. The remaining extraction pattern is

\[
p=[\text{A-labelled indices of }\sigma],\ [k,\ldots,k+r-1],\
  [\text{B-labelled indices of }\sigma],
\tag{4}
\]

where sigma records the source A or B of the chosen entries in increasing
value order. Unchosen entries are given artificial labels in positional
order. Sorting this pattern extracts all k chosen entries in increasing
order and preserves the relative order of every unchosen entry.

Let J be the inversion number of (4). The inversion decrease of the
complete extraction is Delta=3bu+J. If a word of length t sorts (4) and
satisfies 3t-J <= k/2, then the extraction cost L=bu+t satisfies

\[
3L-\Delta\le k/2.
\tag{5}
\]

Start at k=3 with the 24 pairs (sigma,r). If no word is supplied for the
current pair, select one further entry. The complete refinement rule is

\[
(\sigma,r)\longmapsto
 (\sigma A,r-1\bmod3)\quad\hbox{or}\quad(\sigma B,r).
\tag{6}
\]

Selecting an A-entry reduces |A|-a by one; selecting a B-entry leaves it
unchanged. Selection and refinement are decided before any moves, so
refining adds no cost. Including an impossible choice from an exhausted
list only enlarges the set of cases being checked.

The complete finite certificate is
[merge_extraction_c1_2_wide_trim.json](merge_extraction_c1_2_wide_trim.json).
It has 5794 cases, 2909 terminal words, no unresolved branch, maximum
k=28, and maximum support width 30. Every terminal word uses nonwrapped
consecutive 4-cycles on max(9,k+r) positions and satisfies (5).

The independent checker
[verify_merge_tree.py](verify_merge_tree.py) reconstructs all roots and
both children of every nonterminal case. It independently reconstructs
each pattern, counts its inversions, replays each supplied word using
ordinary array rotations, and checks the length inequality. Its report
is [merge_refined_audit.json](merge_refined_audit.json). The checker uses
only the Python standard library; the search implementation is not used.

After extraction, the remaining list is again a concatenation of two
increasing lists. Repeat the construction. When h < 30 actual entries
remain, append the following fixed cyclic positions as helpers, giving
them artificial ranks larger than every active entry and regarding them
as an extension of B. At most 30 distinct cyclic positions are used.

Each finite identity restores every helper exactly. If selected k > h,
all real entries of A have already been selected, so |A|-a=0. Thus no
preparatory triple move crosses a helper. The inversion decrease Delta
in (5) counts only real inversions even in the final extraction.

Only the last extraction can charge helpers. A nontrivial last interval
has h >= 2, and k <= 28, so at most 26 helpers are charged. The sum of
charged k is at most m+26. Summing (5) gives

\[
3L_{\rm merge}\le I+(m+26)/2=I+m/2+13,
\]

proving (3). Actual helper values need not be increasing: artificial
labels specify a permutation identity that returns them to their exact
positions. The argument concerns fixed positions, with no free rotation.

## 2. Apply the three-stage cyclic sorting decomposition

The balanced-lift decomposition in
[LINEAR_UPPER_BOUND.md](LINEAR_UPPER_BOUND.md) sorts any permutation in
three stages whose cyclic adjacent-swap costs satisfy

\[
I_1+q+I_2\le K.
\tag{7}
\]

The first stage is an arbitrary interval of length n. The second moves
one pivot clockwise by q positions. The last stage has length n-1 and
is a merge of two increasing lists, as proved in
[MERGE_UPPER_BOUND.md](MERGE_UPPER_BOUND.md).

The previously proved arbitrary-interval lemma and pivot conversion give

\[
3L_1\le I_1+2n+10,\qquad 3L_0\le q+14.
\]

For n >= 30, (3) applies to the final stage:

\[
3L_2\le I_2+(n-1)/2+13.
\]

Adding these inequalities and using (7) yields

\[
3(L_1+L_0+L_2)\le K+\frac52n+\frac{73}{2}.
\]

The total length is integral, giving (1).

For 5 <= n <= 15, the earlier bound 3D(n) <= K+4n+32 gives
3D(n) <= K+(5/2)n+109/2. For 16 <= n <= 29, the earlier bound
3D(n) <= K+3n+35 gives the same conclusion, since n/2+35 <= 99/2.
Together with (1), these prove (2).

## 3. Verification and scope

[refined_merge_sort.py](refined_merge_sort.py) implements the lemma and
the full cyclic construction. It requires a complete independently
verified tree before using a certificate. Supplementary checks replay
all 8188 binary merges of lengths 2 through 12 with restored helpers,
and 1240 full sorting words for reversals, half rotations, and random
permutations at every size 30 through 150, plus 250, 500, and 1000.
Results are stored in
[refined_merge_sort_checks.json](refined_merge_sort_checks.json).

The all-n theorem follows from the finite coverage certificate and the
argument above; the sample checks are implementation validation.
The exact diameter remains unresolved.
