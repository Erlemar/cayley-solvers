# A universal diameter bound with linear coefficient 3/4

Let D(n) be the diameter for the inverse-closed wrapped consecutive
4-cycles on S_n, and put K=floor(n^2/4).

**Theorem.** For n >= 32,

\[
\boxed{D(n)\le\left\lfloor\frac{4K+9n+148}{12}\right\rfloor
\le\frac{n^2}{12}+\frac34n+\frac{37}{3}.}
\tag{1}
\]

For all n >= 5, a uniform version is

\[
\boxed{D(n)\le\left\lfloor\frac{4K+9n+233}{12}\right\rfloor
\le\frac{n^2}{12}+\frac34n+\frac{233}{12}.}
\tag{2}
\]

This improves the previously proved coefficient 5/6. It is a complete
computer-assisted proof of an upper bound, not an exact diameter formula.
All finite identities and an independent checker are supplied. Search
failures are not used as nonexistence statements anywhere in the proof.

## 1. An improved arbitrary-interval lemma

For an interval of m entries with I ordinary inversions, surrounded by
at least 32 distinct cyclic positions in total, there is a sorting word
that restores every outside helper and satisfies

\[
\boxed{3L\le I+\frac74m+\frac{21}{2}.}
\tag{3}
\]

Use stable extraction as in
[ADAPTIVE_UPPER_BOUND.md](ADAPTIVE_UPPER_BOUND.md). Mark the k smallest
entries. Process them from left to right, moving each left across triples
of unmarked entries until every preceding unmarked gap has length at
most two. Each preparatory generator removes three true inversions.
Give the marked entries labels 0,...,k-1 in increasing-value order and
the remaining prefix entries labels k,k+1,... in positional order.

Let p be this residual pattern and J its ordinary inversion number.
Sorting p places the chosen entries first, in increasing order, and
preserves the order of all unchosen entries. If preparation took q
moves, the complete extraction removes Delta=3q+J inversions. Thus any
residual word of length t satisfying

\[
3t-J\le\frac74k
\tag{4}
\]

gives an extraction of cost q+t with

\[
3(q+t)-\Delta\le\frac74k.
\tag{5}
\]

### Complete finite coverage

Start at k=3. The 162 initial cases are the six possible marked orders
and the 27 choices of three gaps in {0,1,2}. At a nonterminal case,
mark the next-smallest entry. It lies either in an unmarked position
of the current prefix, or after that prefix with a residual gap of
0, 1, or 2. These alternatives are the complete refinement rule.
Relabel unmarked entries in positional order after each refinement.
Preparatory moves made along a branch all remove three inversions;
they are included in q in (5).

The certificate
[adaptive_extraction_c7_4_pruned.json](adaptive_extraction_c7_4_pruned.json)
has 10815 cases and 9101 terminal words. Every branch terminates by
k=24. Every word uses only nonwrapped consecutive 4-cycles on at most
32 positions, sorts its stated pattern with any padded helpers restored,
and satisfies (4).

[verify_refined_extraction.py](verify_refined_extraction.py) independently
reconstructs the entire tree from the 162 roots, checks that every
refinement is covered, replays every terminal word as an array permutation,
and directly verifies (4). It uses only the Python standard library.
The proof does not depend on the algorithm that discovered the words.

### The final batch charges at most six helpers

Repeat stable extraction until the interval is sorted. If fewer than 32
real entries remain, append following cyclic positions as helpers with
artificial ranks above all active entries, in their current order.
Each finite permutation identity restores these helpers exactly. Only
the final extraction can select any helpers.

For precision, let h >= 2 be the number of real entries in a nontrivial
final batch, and suppose k > h. All real entries are then marked.
The selected helpers are the first k-h following entries. No preparatory
move crosses a helper: before all real entries are marked, all selected
entries lie inside the real interval; afterward the helpers already
follow them consecutively. Consequently the terminal pattern has length
exactly k and fixes all positions h,...,k-1.

For a terminal pattern p of length k, put

\[
u(p)=\max\bigl(\{i+1:p_i\ne i\}\cup\{0\}\bigr).
\]

The actual h is at least max(2,u(p)), so the number of charged helpers
is at most k-max(2,u(p)). The checker evaluates this expression on all
323 terminal patterns of length k and obtains

\[
\max_p\{k-\max(2,u(p))\}=6.
\tag{6}
\]

For example, the adjacent swap [1,0,2,...,7] at k=8 attains this maximum.
Thus the total number of charged entries is at most m+6. Summing (5)
over the extractions gives

\[
3L\le I+\frac74(m+6),
\]

which is (3). Numerical helper values are irrelevant: artificial ranks
specify identities that restore the actual entries to their exact positions.

## 2. Combine the cyclic sorting stages

The balanced-lift decomposition from
[LINEAR_UPPER_BOUND.md](LINEAR_UPPER_BOUND.md) has three stages:
an arbitrary interval of length n, one pivot travelling q positions,
and a merge of two increasing lists on n-1 positions. Their adjacent-swap
costs satisfy

\[
I_1+q+I_2\le K.
\tag{7}
\]

Use (3) for the first stage. The existing pivot bound is 3L_0 <= q+14.
The merge lemma in
[REFINED_MERGE_UPPER_BOUND.md](REFINED_MERGE_UPPER_BOUND.md) gives
3L_2 <= I_2+(n-1)/2+13. Its support requirement is 30 positions and
is therefore satisfied for n >= 32. Adding yields

\[
\begin{aligned}
3(L_1+L_0+L_2)
&\le I_1+\frac74n+\frac{21}{2}
 +q+14+I_2+\frac{n-1}{2}+13\\
&\le K+\frac94n+37.
\end{aligned}
\]

Since word length is integral, this proves (1).

For the uniform extension (2), use the earlier proved bounds:

- For 5 <= n <= 15, 3D <= K+4n+32 and (7/4)n+32 <= 233/4.
- For 16 <= n <= 29, 3D <= K+3n+35 and (3/4)n+35 <= 227/4.
- For 30 <= n <= 31, 3D <= K+(5/2)n+73/2 and n/4+73/2 <= 177/4.

Each implies 3D <= K+(9/4)n+233/4. For larger n, (1) is stronger.
No unknown small diameter is assumed in this extension.

## 3. Verification and remaining problem

[combined_extraction_sort.py](combined_extraction_sort.py) requires both
finite certificates to be complete before using them. It additionally
checks all 5912 permutations of lengths 2 through 7 with helpers restored,
including helpers in reverse numerical order, and replays 1220 full words
for reversals, half rotations, and random inputs. The full-word sizes are
every n=32,...,150, and n=250,500,1000. All strengthened cost assertions pass.

The audit, helper bound (6), and replay results are recorded in
[combined_sort_checks_c7_4.json](combined_sort_checks_c7_4.json).
These samples validate the implementation; the complete finite tree
and the argument above establish the theorem for arbitrary n.

The exact diameter, the proposed coefficient 1/6, and a sharp periodic
constant remain unresolved.
