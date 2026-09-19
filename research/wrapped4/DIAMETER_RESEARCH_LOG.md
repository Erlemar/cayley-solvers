# Research notes after the C=3/4 upper bound

The exact diameter remains unresolved. The current strongest universal
upper bound has C=3/4, proved in COMBINED_EXTRACTION_UPPER_BOUND.md. It improves
the previous C=5/6 result. TWISTED_ROTATIONS.md gives a separate exact-family
result, and validated bounded search is available up to n=40.

## Current extension: charge disorder left behind by an extraction

PARTIAL_EXTRACTION_METHOD.md proves a new local accounting lemma. An
extraction may leave its unselected tail permuted. If the residual pattern
has J inversions and the final unselected tail has R inversions, a word
of t generators is valid at coefficient c whenever 3t+R-J<=ck.
For arbitrary actual entries, the true inversion decrease is at least
J-R. Thus these inequalities telescope even when successive extractions
change the tail order.

A COMPLETE finite certificate of width W would give, for
n>=max(32,W),

    3D(n) <= K+(c+1/2)n+(7/4-c)(W-1)+37.

This is a conditional implication until full finite coverage is checked.
The partial c=3/2, c=5/3 and c=12/7 trees are not proofs of improved
numerical universal bounds. The current established coefficient is 3/4.
Independent checker: verify_partial_extraction.py; constructive extension:
partial_extraction_sort.py. Both explicitly reject incomplete coverage
for an all-n claim.

The first c=12/7 beam pass ended at its 600-second limit at stage 22:
73 node-limited cases and 40 not-searched cases remained at that stage.
Its output is partial_extraction_c12_7_beam.json. A wider-beam retry
produced almost no improvement and was stopped after its saved stage 16;
the process was verified terminated. Its positive words remain in
partial_extraction_c12_7_beam5000.json. Extra unmarked padding resolved
11 of a spread-out 30-case pilot (partial_padded_c12_7_pilot.json).
The padded c=12/7 search finished at its 600-second limit at stage 24.
Independent audit: 4128 cases, 3094 words, 36888 moves, 1082 disordered
tails, maximum width 37, and 1563 uncovered next-stage children. A padded
c=26/15 search finished at its 300-second limit at stage 25: 3098 cases,
2416 words, 25879 moves, maximum width 36, and 389 uncovered children.
Both are incomplete. Every failure is inconclusive. All searches are now
terminated; there is no live process to resume.

The method itself now has a COMPLETE hybrid c=7/4 certificate:
partial_extraction_hybrid_c7_4.json. Independent audit: 1807 cases,
1447 words, 13210 moves, 473 disordered-tail words, k<=23, W=32.
The full constructor replays 398 words and 595262 moves through n=1000,
checks 120 arbitrary-helper inputs, and exercises 3063 disordered-tail
batches. This reproduces C=3/4 rigorously; it does not improve C. The
old stable certificate is still used to finish short intervals.

Further positive search tools, not additional all-n theorems:

- partial_prefix_pdb.py / the optional beam heuristic use exact three-token
  projections only for ranking. The 30-case c=12/7 pilot found ten words,
  mostly in the not-yet-searched part of stage 24; it did not resolve most
  earlier difficult cases. File: partial_pdb_c12_7_pilot.json.
- partial_block_composition.py replays stable source words, adds their
  inverse/reflected forms, and composes them on invariant consecutive
  blocks. Arbitrarily valued outside helpers are restored exactly. It
  found 74 words among 1034 unresolved c=12/7 rows, seven at k23, three at
  k22, one at k17, and the remainder at k24. File: partial_blocks_c12_7.json.
- partial_extraction_tree.py now supports translated identity reuse,
  --stable-blocks, --padding, --reuse-only, and --beam-heuristic. The
  composed c=26/15 snapshot was independently audited: 5591 cases, 3425
  words, 48940 moves, k<=29, W=42, and 10007 uncovered next-stage children.
  Its width can exceed the search packing limit because translations of
  positive words remain valid; no search is performed on such large states.
  Do not extend this frontier blindly. New ideas or improved ancestor
  words are needed. File: partial_extraction_blocks_c26_15_snapshot.json.

## One-pivot simplification has a quadratic obstruction

PIVOT_CUT_OBSTRUCTION.md proves that for the half rotation of four
reversed blocks of size b, n=4b, the minimum adjacent work of moving
one pivot directly to its target and then sorting the cut interval is

    F=5b^2-2b+(b mod 2).

Thus this restricted construction cannot have total adjacent work
K+O(n). This is not a lower bound for the true 4-cycle word metric.
The formula was also checked directly for 40 block sizes through n=160.

## Support-five central K-3 search: positive partial evidence only

The first bounded pass over all 9856 support-five roots is complete:
6155 fully anchored words, 2897 partially anchored words, and 804
inconclusive cases. File: gap_d3_support5_words.json. There is no complete
certificate or all-n theorem for this family yet. The 31100 support-six
roots have not received a full search. This supersedes the older
not-yet-launched statements below; C=3/4 remains unchanged.

## Latest theorem: the entire auxiliary layer K-2 is exact for all n>=12

### Further progress: unequal rotations and the noncentral K-3 families

TWO_DEFECT_UNBALANCED.md now proves |tau_a sigma|_4=h(ab-2) for every
3-cycle or noncrossing double transposition sigma, with a,b>=12. Key
corollary of the central certificate: every constructed word has both
monotone anchors in every gap >=3. The only missing finite anchors occur
in gaps of length two; any such gap that grows triggers a covered refinement.
Retain 3,4,5 unmarked positions modulo three, and reduce only the larger
rotation block to c+epsilon, c=min(a,b), epsilon in {-1,0,1}. The retained
core size is <=24 and fits into the central size 2c+epsilon>=23 by congruence.
One-direction inflation restores the original larger block and gives exact
length. Independent replay: 1790 words, 257980 moves through n=151, 4940
one-direction inflations; report two_defect_unbalanced_checks.json.

THIRD_LAYER_NONCENTRAL.md further proves every noncentral member of
ell=K-3 has length h(K-3), for all n>=13. For even n>=16, any gap>=6
can lose three positions and become an odd central defect-two base with
a remaining gap>=3; one positive or negative inflation restores the target.
If no gap>=6 exists, n<=24. All n14 cases and these bounded-gap even cases
give 218 orbits. Six odd n13 transposition bases close the odd boundary;
odd n>=15 follows from the existing one-transposition theorem.
All 224 searched words attain the exact adjacent/parity lower bound.
The independent standard-library checker verify_third_layer_noncentral.py
regenerates exactly these 224 orbits and replays 4480 moves. Constructor
audit: 1346 words and 245838 moves through n151. No search remains running.

Central K-3 pilot: sparse_gap_roots.py enumerates 177291 models / 42661
geometric orbits, maximum base n33, seven cyclic core types. The d2
geometry was completely compared with the independently certified 942
roots; d3 symmetry identities were checked on 300 orbit samples. Report:
sparse_gap_geometry_checks.json. The 60-case anchored pilot found 44 words
at parity-adjusted H(n), while 16 cases hit their bounded search limits.
gap_pilot_d3.json keeps the results. Those 16 cases are INCONCLUSIVE,
and no full 42661-root search has been launched. The central K-3 layer
and all deeper layers remain open in general; C=3/4 is unchanged.

The rest of this section records the preceding complete K-2 theorem.

### Completed central single-4-cycle family in layer K-3

FOUR_CYCLE_CORE_BOUND.md proves |tau_a c|_4<=H(n) for every n>=12,
central a, and arbitrary 4-cycle c. For n=2,3,4 modulo six, parity gives
the exact value h(K-3)=H(n)-1. For the other residues only the two-value
interval {h(K-3),h(K-3)+2} is proved, not the optimality of longer words.

A support-four-only search over 1705 roots first found 1499 anchored
words, 195 partial words, and 11 inconclusive cases. A bounded retry of
the 206 incomplete cases with all 16 orientations found 1702 anchored
words and three partial words. Their missing-growth refinements all
closed: 15 added bases, all fully anchored. Authoritative certificate:
four_cycle_core_certificate.json, complete, 1720 nodes, 1717 leaves,
three refinement nodes, 15 edges, maximum n23. No search is running.

The independent checker verify_four_cycle_core.py regenerates all six
4-cycle cores using marked position subsets: 14982 raw models, exactly
1705 root orbits. It replays 37998 moves and checks all anchors, edges,
closure, and termination. Report four_cycle_core_audit.json is complete.
four_cycle_core_construct.py implements the all-n construction; independent
replay covers 2254 words through n151, 183658 moves, and 150 refinements.

SPARSE_GAP_CERTIFICATION.md records the general finite-root/partial-anchor
coverage lemma and H(n+6)=H(n)+n+3 propagation. This is a sufficient
certificate method, not an assertion that all deficit layers close.
The full d3 root table still has 9856 support-five and 31100 support-six
orbits; no full search for those families has been launched. Their earlier
60-case pilot failures remain inconclusive. Remaining tasks include those
two central K-3 families and, more importantly, the universal diameter
upper bound. The global coefficient is still C=3/4.

TWO_DEFECT_EXACT.md proves ell(p)=K-2 => |p|_4=h(K-2), for every n>=12.
Together with the central-rotation and one-transposition theorems, the
top THREE auxiliary layers have exact lengths h(K-j), j=0,1,2. Their
maximum is H(n). Any counterexample to D(n)=H(n) must have ell<=K-3.
The previously stronger finite conclusion for n=15 (ell<=52) remains.
The diameter itself is still open and the universal C=3/4 bound is unchanged.

Generalized inflation: for p=tau_a sigma with sigma fixing position j,
inflate its token x into four clones. The simulated word ends with cyclic
origin gamma=-3*k_x. Relabel the INITIAL inflated permutation by +gamma;
the same word then sorts it to the fixed identity. A monotone positive
strand of travel a gives tau_a sigma^[j] with b increased by 3 and cost a.
A monotone negative strand of travel b gives tau_(a+3) sigma^[j], cost b.
This includes winding strands correctly; no global rotation is free.

Keep one monotone strand of each direction in every gap that grows.
The certificate checks same-direction anchor pairs cross zero times and
opposite-direction pairs once. These properties survive every inflation,
allowing arbitrary gap growth by triples and any distribution of the
two directions. For central rotations, use equally many positive and
negative inflations to keep the two blocks central.

Finite coverage: a core of defect two has 3 or 4 marked positions. Keep
gaps 0,1 fixed, reduce longer gaps to 2,3,4 modulo 3, then restore enough
triples so the base size n0 is at least 12 and congruent to n modulo 6.
Always 12<=n0<=23. This produces 3533 rooted models and 942 symmetry
orbits. Independent enumeration using marked position subsets and both
noncrossing pairings obtains 6030 models with exactly the same 942 orbits.

940 roots admit all required anchors. Two roots use a partial-anchor word
plus complete growth refinements: n=12, gaps (2,2,2,2), core (0 9)(3 6),
anchors in gaps 1,3; and n=15, gaps (9,0,2,0), core (0 14)(10 11),
anchors in gap 0. If a growing gap lacks anchors, consume one triple
there and one in any other available gap. The remaining total is even.
All required children are present and anchored; after symmetry and pruning
there are five edges and five added nodes. No negative search assertion
is used in the proof.

AUTHORITATIVE certificate: two_defect_inflation_complete.json, with 947
nodes, 945 anchored leaves, two refinement nodes, maximum base size 23.
verify_two_defect_certificate.py is an independent standard-library
coverage and replay checker. Its report two_defect_certificate_audit.json
is complete: it replays 19974 moves in the main words and six additional
12-move boundary words for tau_5(0 d), d=1..6, at n=12. Those boundary
words cover the only cases not included directly in the all-n one-
transposition theorem (which assumed both rotation blocks >=6).

two_defect_construct.py constructs an optimal word from arbitrary central
core/gap data for any n>=12. check_two_defect_construct.py independently
replayed 3893 full words: all 3533 root models, 300 random larger cases
through n=150, and 60 exceptional growth patterns, exercising 47 refinement
steps and 181474 moves. Report: two_defect_construct_checks.json.

Raw searches two_defect_inflation_certificate.json and
two_defect_inflation_refined.json are SUPERSEDED and intentionally retain
incomplete states. An earlier overly strong refinement grew a long-gap
branch to n=27 and hit a node cap; the partial-anchor argument made that
branch unnecessary. Do not resume it or interpret that cap as a lower
bound. Only the complete certificate and its independent audit support
the theorem. No search is running at this checkpoint.

The preceding goal turn was progress: it proved all rotation-plus-one-
transposition distances. This turn proves the entire next layer. Useful
next directions include a bound H(n) (not necessarily h(ell)) for deeper
deficit layers, or using the inflation idea to improve a global sorting
bound. Note that h(ell) already fails for some deficit-three elements at
n=12, although H(12) still bounds them. Do not assume every deeper layer
will attain its adjacent-length lower bound.

## Latest exact theorem: every rotation followed by a transposition

STRAND_INFLATION.md now proves |tau_a t|_4=h(ab-1) for ALL a,b>=6
and every transposition t. The three short-separation families left
by the interval proof have been closed, including unequal a,b.

Key lemma: replace a token x in a wrapped sorting word by four ordered
clones. A move where x is the long strand becomes four moves; a move
where x is a short strand becomes two. Word length increases by the
total travel c_x. The final cyclic-origin shift is exactly -3*k_x,
where k_x=(initial_position+signed_displacement-label)/n is its winding.
Thus a zero-winding strand gives a valid fixed-position sorting word.
This is a dynamic bundle construction, not an invalid static embedding
of a smaller cycle across gaps.

Three explicit bases have distinguished unmarked strands u in the
lower label block and v in the upper label block. They travel only
counterclockwise by b and clockwise by a, respectively, and cross once:
(a,b,d,u,v)=(7,7,2,0,10), (5,5,1,0,7), (5,5,2,3,9), with lengths
16,8,8. Inflating u gives (a+3,b) at cost b; inflating v gives (a,b+3)
at cost a. The distinguished-strand properties persist. This proves
the missing families for all sizes by induction. The earlier retained-
triple separation argument also works with unequal block sizes, which
reduces every remaining case to these three families or a sharp old
interval base.

rotation_twist_inflation.py provides the complete optimal constructor
for tau_a(0 d), a,b>=6 and all circular separations d. The standard-
library audit verify_twist_inflation.py checks the three base identities,
120 arbitrary winding cases (57 nonzero), and independently replays all
13669 targets with 6<=a,b<=32 and every circular separation. It replays
1932366 individual 4-cycle moves. Report: twist_inflation_independent_audit.json.
The separate block-extension check covers 432 targets through a,b=40.

An earlier paired-strand version in twist_inflation.py simultaneously
adds three positions to each block. It now uses bases (n,d)=(8,2),
(10,1),(10,2) and was checked through n=254,256. Opposite-block strands
need not be antipodal. The separate zero-winding construction is the
stronger result and is used in the main proof.

Consequently EVERY element in the auxiliary layer ell=K-1 has exact
4-cycle length h(K-1) for n>=12. The top two auxiliary layers have
maximum H(n); any larger-distance element must have ell<=K-2.
This still does not prove D(n)=H(n), and the universal upper bound
with C=3/4 is unchanged. No search is running at this checkpoint.

The previous goal turn made concrete progress: the all-n ascent/core
theorem and complete n=15 layers were proved and independently checked.
This turn closes the remaining one-transposition cases. Next useful
directions are to apply zero-winding inflation to finite cores with
two or more transpositions, or to derive a stronger global sorting
bound. Avoid treating the now solved one-transposition families as open.

## Latest structural theorem and complete n=15 layers

ROTATION_ASCENT_LEMMA.md proves an all-n stability theorem: every p with
ell=K-delta can be changed to some rotation tau_a by T<=a(n-a)-ell<=delta
arbitrary transpositions, with ell strictly increasing by positive odd
amounts. The proof works with a bounded affine lift f. Any interior
value i<f(i)<i+n admits a crossing exchange i<j<=f(i)<f(j)<=i+n;
a pigeonhole argument proves existence, and the length increment is
1+2*#{i<k<j:f(i)<f(k)<f(j)}. It terminates at a rotation.

Deleting an endpoint f(h)=h or h+n preserves the defect k(n-k)-Phi(f),
where k is the rank. Consequently the defect of tau_a sigma depends
only on the order type of sigma's moved positions when the explicit
rank feasibility conditions hold. Together with ascent this yields a
complete finite-core enumeration for every fixed deficit delta.

The proof gives explicit classifications and counts for the layers
K-1, K-2, K-3. The cores of defect three are six 4-cycles, ten disjoint
3-cycle/transposition cores with the pair consecutive around the five
positions, and five noncrossing perfect matchings on six positions.
For an arbitrary matching with r pairs and c crossings, the core
defect is r+2c.

Independent checks: rotation_ascent_checks.json covers all 46232
permutations at n=2..8 and 278350 exact ascent increments.
rotation_defect_checks.json covers 92457 core formulas against BFS,
125670 bounded lifts, and 251326 endpoint deletions. The general
enumerator adjacent_deficit_layers.py checks full deficit-0..3 layers
against BFS at n=2..8 and stored exhaustive data at n=10,11.

At n=15 all 7282 states of ell=54 have exact 4-cycle length 18, and all
126700 states of ell=53 have exact length 19. There are 151 and 2232
symmetry classes. Every class has an explicit word attaining the
analytic lower bound. Ten ell=53 classes first hit a 50000-node cap;
all were solved at the same budget in the bounded retry, so there is
no unresolved case. Certificates: second_layer_four_n15.json and
adjacent_layer_d3_four_n15.json.

verify_second_layer_certificate.py is a standard-library checker that
does not import the search implementation or the core enumerator.
It independently enumerates bounded arbitrary-transposition balls from
the ascent theorem, filters by balanced adjacent length, checks exact
coverage, and replays all dihedral and inverse transforms of the words:
9060 and 133920 replays respectively. Reports have suffix
_independent_audit.json. Along with the proved ell=56 and ell=55 cases,
all top four layers at n=15 are settled. Any counterexample to D(15)=20
must have ell<=52. This does not prove the diameter itself.

That checkpoint left three all-n short-transposition families open;
they have now been solved by the inflation theorem above. The finite
core enumeration can next inspect deeper layers. Do not infer a general
diameter from the solved top layers.

## Further reuse in the incomplete c=3/2 extraction tree

reuse_extraction_library.py reused the complete c=7/4 library and the
older libraries, found 380 additional usable words (three at k=10 and
377 at k=11), and wrote adaptive_supplement_c3_2_reused.json. Independent
pruning produced adaptive_extraction_c3_2_reused.json and its audit:
33252 cases, 20516 found words, maximum batch 11, support 24. This tree
is still INCOMPLETE; its next frontier has 57990 cases. The helper-charge
bound eight has no all-n implication until coverage closes. Do not
restart that large frontier without further pruning or a better method.

## Latest complete certificate and upper bound

adaptive_extraction_c7_4_pruned.json is COMPLETE: 10815 cases, 9101 words,
maximum selected batch 24 and maximum support 32. Independent coverage
and replay checks pass. Among all 323 terminal patterns of length k,
the maximum k-max(2,last_moved_position+1) is six. This bounds the final
helper overcharge and sharpens the arbitrary-interval lemma to

    3L <= I + (7/4)m + 21/2.

Combining with the c=1/2 merge lemma and the pivot gives

    D(n) <= floor((4 floor(n^2/4) + 9n + 148)/12), n >= 32.

Replacing 148 by 233 covers every n >= 5 using earlier proved bounds.
combined_extraction_sort.py checked all 5912 short permutations through
size seven with helpers restored, plus 1220 complete sorting words at
n=32..150,250,500,1000. All tightened assertions pass. Report:
combined_sort_checks_c7_4.json. There are no running searches at this
checkpoint. The diameter itself remains unresolved.

## Improved merge certificate

merge_extraction_c1_2_wide_trim.json is now COMPLETE: 5794 cases,
2909 replayed words, maximum selected batch 28, maximum support 30.
verify_merge_tree.py independently checked exact tree coverage and all
words. refined_merge_sort.py additionally checked 8188 short binary
merges and 1240 full sorting words through n=1000, with helpers restored.

This proves 3L_merge <= I + m/2 + 13, and hence

    D(n) <= floor((2 floor(n^2/4) + 5n + 73)/6), n >= 30.

Replacing 73 by 109 makes the bound valid for all n >= 5, using the
previous proved bounds for smaller n. Thus C=5/6 is admissible.

The arbitrary-interval c=3/2 tree remains INCOMPLETE. Its original saved
checkpoint is through k=11 (19404 cases at that stage: 10860 found,
1374 exhausted, 7170 node-limited). That search was stopped after saving
k=11. Supplemental identities and independent pruning reduced the tree
to 33281 total cases and a 61278-pattern next frontier. This is now
superseded by the reused/pruned checkpoint described above, whose
next frontier has 57990 patterns.

The c=7/4 search used radius-five exact endgames on supports up to 20,
a beam fallback, and reuse of stored identities. It closed at k=24,
giving the theorem above. compose_extraction_words.py supplies additional
identities by composing words on invariant intervals while restoring
surrounding helpers; every resulting word is independently replayed.

The complete c=7/4 library has now been reused in the incomplete c=3/2
tree wherever the shorter budget is actually met, and solved ancestors
have been pruned. Further improvement is needed before extending its
remaining large frontier: earlier node limits caused avoidable branch
growth.

## All-n classification in the adjacent-swap metric

ADJACENT_EXTREMAL_CLASSIFICATION.md now proves that the only permutations
with ell=floor(n^2/4) are the central rotations (one for even n, two for
odd n). The key new exact formula is

    ell(tau_a followed by a left rotation of l+1 consecutive entries)
      = a(n-a)-l+2 max(0,l-a).

It follows by choosing a balanced rotation lift in which the moved token
crosses only tokens of the opposite displacement, or using the equivalent
suffix move from tau_(a+1). The extremal classification then follows by
the extremal-token deletion lemma in van Zuylen et al. and induction.
This is an auxiliary-metric theorem; it does not yet classify the
4-cycle diameter elements. Independent standard-library BFS checked all
46232 permutations at n=2..8 and all 203 rotation-run formulas there.

## A simple quotient route is too coarse for D(15)=20

color_quotient_scan.py exhaustively computed eccentricities from contiguous
colour blocks on 15 positions: counts (7,8) give 10; (5,5,5) and (4,4,7)
give 13. These are quotient eccentricities, not S_15 diameters.
For the three equal blocks, the stabilizer already contains the three
block reversals, of balanced adjacent length 30 and hence 4-cycle length
at least 10. Thus adding a worst quotient bound and a worst stabilizer
bound cannot prove D(15)<=20: even the optimal separate bounds sum to
at least 23. This rules out that simple proof route, not the candidate
diameter itself. Data: color_quotient_scan.json.

## Earlier interval result, now sharpened by strand inflation

For every rotation tau_a and every transposition t, cyclic adjacent
length is exactly a(n-a)-1. For a,b=n-a >= 6, its wrapped 4-cycle length
lies between h(ab-1) and h(ab-1)+2, where h(k)=floor(k/3)+(k mod 3).
Equality holds if a,b are not the same nonzero residue modulo 3.
For an even half rotation, equality also holds whenever the circular
separation of the transposed positions is at least three.

The proof has 441 nonwrapped base identities and four exact extensions
by an unmarked triple. All 441 identities and 7350 extensions were
independently replayed; a further 735 tests cover the separation
refinement. The default twisted_exchange_check.py verifies stored words
without a search or nonstandard package dependency.

The earlier estimate was at most H(n)+1 for half rotations, where
H(n)=ceil(n^2/12)+1_{3|n}. The new inflation theorem sharpens this to
the exact value h(K-1), at most H(n), for every n>=12. This family
cannot witness a positive linear term in the diameter.

## New adjacent-distance enumeration

adjacent_extremes.py enumerated all permutations at n=10 and n=11.
At n=10 the unique adjacent-distance maximizer is the half rotation,
at length 25. There are 47 states of length 24. Some are eight adjacent
swaps from the nearest rotation, so adjacent-distance deficit does not
directly bound distance to a rotation.

At n=11 the only two maximizers are the rotations by 5 and 6, at length
30. There are 110 states of length 29. Data: adjacent_extremes_n10.json
and adjacent_extremes_n11.json. The maximal layer is now classified for
all n by ADJACENT_EXTREMAL_CLASSIFICATION.md. The next three layers are
now classified in ROTATION_ASCENT_LEMMA.md, as described above.

## Full adjacent matching is a more difficult family

For even n, set

    p_i = (i + n/2 + (-1)^i) mod n.

This is the half rotation followed by all disjoint adjacent pair swaps.
For n divisible by 4, its unique balanced displacements are plus and
minus (n/2-1), alternating around the circle, and its cyclic adjacent
length is n^2/4-n/2. It has no immediately available 4-cycle reducing
that length by three. The alternating displacement structure may be
useful for a sharper lower bound. ALTERNATING_MATCHING.md now proves
the first-move obstruction and the parity-rounded bound
2*ceil((m(m-1)+2)/6), where m=n/2.

At n=24, wide_matching_retry.py tested the full matching at budget 48
with four different search orders, each limited to 3,000,000 nodes.
All four reached the node limit: these are inconclusive, not lower
bounds. At budget 50 a word was found and independently replayed.
That initially gave the individual interval

    46 <= |p|_4 <= 50, with |p|_4 even.

The word and every attempt are retained in
wide_full_matching_n24_retries.json. This interval has now been resolved:
a symmetry-diverse bidirectional beam found a replay-verified 46-letter
word, so the analytic lower bound proves |p_24|_4=46 exactly. The source
is matching_bidirectional_n24_b46_w10000_s2.json. This permutation does
not refute H(24)=49 and is not a diameter element.

At n=36, matching_beam_n36_b108_w10000_s3.json gives a verified word of
length 108, below H(36)=109. The analytic lower bound is 104. Searches
at 104 and 106 that reached their beam limits remain inconclusive.
verify_matching_certificates.py verifies every retained successful
matching word and writes alternating_matching_certificates.json.

The optimal n=24 word has only three one-unit descents, at steps
1,2,4; all other steps reduce ell by three. Its prefix is
3-,5-,7-,0+. A fixed-prefix beam at n=36 did not find a completion,
but that does not prove none exists. Sixteen simple greedy policies
were also tested after this prefix at n=12,24,...,120; none sorted.
For n >= 24 their residual adjacent lengths were generally n or n-6.
See matching_prefix_n36_w5000_s0.json and matching_greedy_checks.json.

Other bounded searches of partial matchings at n=24 and n=36 are in
wide_matching_n24.json and wide_matching_n36.json. Several words fit
within H(n), and many other cases hit their node limits. At n=36 even
the one-transposition case (whose exact distance is already proved)
hit the limit. Thus search difficulty alone is particularly weak
evidence of a counterexample here.

Potential next mathematical direction: study the minimum extra
crossings needed to route the alternating positive and negative
displacement classes with 4-cycles. Binary projection with a recorded
net class flux may provide lower bounds, but allowing alternative
integer lifts and within-class crossings must be handled explicitly.
Ignoring these alternatives would give an invalid lower bound.

Do not try to tile the full matching construction with nonwrapped
6-by-6 block exchanges of cost 12: that base has 42 ordinary inversions,
so it needs at least 14 nonwrapped 4-cycles. Wrapped words do not embed
as interval words after inserting gaps. This was a rejected idea, not
a proved induction.
