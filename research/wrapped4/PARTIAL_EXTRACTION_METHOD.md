# Extracting a smallest prefix while charging disorder of the remaining tail

This note proves a more flexible local-to-global sorting lemma. A new
all-n numerical bound requires a **complete** finite extraction tree.
The current exploratory trees must not be used as completed certificates
until their coverage audit says `complete: true`.

## 1. The local accounting lemma

Work in a linearly ordered interval of distinct entries. Mark its k
smallest entries. Move the marked entries left across unmarked triples,
in their current positional order, until the unmarked gap before each
marked entry has size zero, one, or two. Each preparation move is an
allowed 4-cycle and removes exactly three true inversions.

Relabel the marked entries 0,...,k-1 in increasing-value order. Relabel
the unmarked entries k,k+1,... in their current positional order. Only
the prefix through the last marked entry is relevant; pad it with further
unmarked entries when a finite word needs a larger support. Let p be this
artificially relabelled permutation, and let J=Inv(p).

Suppose a finite word of t nonwrapped consecutive 4-cycles transforms p
to

\[
(0,1,\ldots,k-1,\text{an arbitrary permutation of the unmarked entries}).
\]

Let R be the ordinary inversion number of that final unmarked tail,
using the artificial labels, and suppose

\[
\boxed{3t+R-J\le ck.}
\tag{1}
\]

**Lemma.** The same word, applied to the actual numerical entries, places
the k smallest entries first in increasing order, and has amortized
inversion charge at most ck.

Indeed, J counts precisely the inversions involving marked entries.
Their true ordering relative to every unmarked entry agrees with the
artificial ordering because they are the k smallest. These J inversions
all disappear. The final unmarked permutation reverses exactly R of
the original pairwise orders among unmarked entries. Regardless of their
actual values, this can add at most R true inversions. Thus the true
inversion decrease Delta is at least J-R, and 3t-Delta<=ck.

If preparation took q moves, the complete extraction decreases true
inversions by at least 3q+J-R and costs q+t. The same inequality follows:

\[
3(q+t)-\Delta_{\rm total}\le ck.
\tag{2}
\]

It is unnecessary to restore the unmarked entries to their original
order: the next extraction simply starts from their new order. The
finite word is supported inside the active interval, so entries already
placed outside that interval are fixed exactly.

## 2. Finite coverage

The abstract residual patterns and refinement rule are the same as for
the earlier stable extraction certificate. There are 162 initial patterns
at k=3. If a pattern is not supplied with a word satisfying (1), mark
the next-smallest entry. It can occupy any unmarked position of the
existing residual prefix, or a position after that prefix with residual
gap zero, one, or two. These give all children after preparation.

A complete finite tree whose leaves satisfy (1) proves that an extraction
always succeeds after boundedly many marked entries. Let W be an upper
bound on the support width of every word and pattern in that tree.
In particular each extraction selects at most W entries.

`verify_partial_extraction.py` independently regenerates the roots and
all refinements, replays every positive word, verifies its ordered
prefix and its tail inversion number R, and checks (1). An unfinished
frontier is reported explicitly. It imports no search code.

The search implementation uses nonnegative charge increments

\[
3+\operatorname{Inv}(p')-\operatorname{Inv}(p)\in\{0,2,4,6\}.
\]

This is merely a way to discover identities; only the replayed words
and complete coverage are required by the proof.

## 3. Conditional improvement of the universal bound

Suppose such a complete tree is available with 0<=c<=7/4. Apply partial
extractions while the active interval contains at least W entries.
Stop early if the interval is already sorted. For a remaining nontrivial
interval of length m0<W, use the existing fully certified stable
extraction sorter, whose bound is

\[
3L_0\le I_0+\tfrac74(m_0+6).
\]

It restores every helper it uses. Assume the ambient cycle has at least
max(W,32) distinct positions, so its helper-support requirement is met.

Summing (2), then applying the final bound and m0<=W-1, gives for an
arbitrary interval of length m with I inversions

\[
\boxed{3L\le I+cm+
\left(\tfrac74-c\right)(W-1)+\tfrac{21}{2}.}
\tag{3}
\]

If the procedure stops earlier with an already sorted interval, (3)
still holds because its unsorted-inversion remainder is zero and its
charged entry count is at most m.

Combine this with the established pivot charge 14 and the two-list merge
bound 3L_merge<=I_merge+(n-1)/2+13. The balanced-lift decomposition has
total adjacent work ell(p)<=K=floor(n^2/4). Therefore

\[
\boxed{3|p|_4\le\ell(p)+\left(c+\tfrac12\right)n+
\left(\tfrac74-c\right)(W-1)+37.}
\tag{4}
\]

In particular a complete c=3/2 certificate would prove linear coefficient
2/3 in the diameter upper bound, and a complete c=5/3 certificate would
prove coefficient 13/18. Neither numerical improvement follows from an
incomplete tree. The currently established global coefficient remains
3/4 until such a complete certificate and constructor have been verified.

## 4. A complete validation at c=7/4, and the remaining gap

The new method now has a complete hybrid certificate at c=7/4:
`partial_extraction_hybrid_c7_4.json`. Its independent audit checks all
1807 cases, 1447 terminal words, and 13210 moves. There are 473 terminal
words with a permuted unselected tail. Every branch terminates by k=23
and the maximum width is W=32.

This gives an alternative proof of the established global C=3/4 bound
via (3)--(4). The final short interval still uses the previously certified
stable sorter; the 1447 words are the main partial-extraction table, not
the only finite data needed by the combined proof.

`partial_extraction_hybrid_c7_4_sort_checks.json` records 398 complete
sorting words through n=1000, 595262 replayed moves, 120 arbitrary-helper
checks, and 3063 batches that actually change the unselected tail order.
These are implementation checks; the full finite coverage and the lemma
prove the all-n assertion.

The smaller-coefficient attempts remain incomplete. The padded c=12/7
tree ends at stage 24 with 1563 children uncovered; the padded c=26/15
tree ends at stage 25 with 389 children uncovered. A further composition
of stable block identities gives additional positive words but does not
close the tree. None of these attempts proves a smaller global C.
