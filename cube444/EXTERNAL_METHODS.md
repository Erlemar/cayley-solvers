# cube444: four methods worth stealing, from the 46,296 write-up

Source: <https://github.com/zoli801/cayley444-46296-solution> (MIT; `README_RU.md` is the
full write-up, `REBUILD_FINAL.sh` replays all ten merge milestones byte-exactly).

Read this before building another local rewriter. Their headline conclusion is the one that
should shape our planning:

> A strong portfolio is already locally irreducible at small radii. Late gains come only
> from very diverse external trajectories, or from more expensive D8/D10/coloured-r8
> rewriting.

**We are the diverse-trajectory source.** Their 46,296 is 1,004 unchanged rows of the
46,378 file plus 39 replacements, and **3 of those 39 are ours** (pids 97, 133, 346 --
their step 10, 46,302 -> 46,296, the merge that crossed their stop condition). Our
`cube444_merged_46372.csv` is in their `inputs/` byte-identical.

## The strategic fact that outranks all four ideas

They record the checkpoint SHA-256 behind **all 62 completed public runs**:

```
58af301a4f2b77d503b6e12d450589c64c076624d3e1ff291128c23663ad3164
```

That is byte-identical to `orig_transformer/model.pth` in our inference bundle. Our `s3`
(`cbd24358fafb41f8832951b1ee92e48b22a43ece68df0df60df3b473ba9cc4f3`) is a different,
Q-Bellman-refined model **nobody outside this repo has**. The public field runs the
original checkpoint at ~46-47M effective beam on 2x T4 with a **median 37,450 s/puzzle**;
we run a better scorer at 2^21-2^24 on a v5e-8 at ~10 min/pid.

That asymmetry is why our three pids survived into their final while nearly everything
else they tried returned zero. Spend effort on TPU width and pid COVERAGE, not on another
rewriter -- the rewriting axis is theirs and is close to exhausted.

---

## 1. Induced-edge trajectory graph

`src/trajectory_graph_merge.py`, flag `--induced-edges`.

Every checkpoint of every replay-valid path becomes a vertex keyed by its **exact 96-colour
state**. Real moves contribute edges. Then, for every vertex already in the set, try all 24
moves: if the neighbour state is *also* in the collected set, add that edge **even though no
trajectory ever traversed it**. BFS from solved.

Guarantee: shortest path within that finite graph -- exact, not heuristic. Not a global
optimum.

Yield: `46,718 -> 46,458` direct, `-> 46,456` with induced edges, and it synthesised
**ID704**, a path present in no single source file. Later graphs up to 1,224,304 directed
edges gave nothing more over 46,302.

Why it may be worth a test for us even though [[route-relinking-portfolio-oracle-rejected]]
found recombination worthless on megaminx: that was route-relinking between pairs of paths,
not a shared-state graph with induced edges across the whole corpus. Different mechanism.
Cheap to try -- we already replay-verify everything in `cube444/tpu/merge_all_sources.py`,
so the vertex set is a by-product.

## 2. 24-frame symbolic DP

`external/helpers/shorten_isometry_frames.py`.

Split a path into maximal same-axis blocks. Within a block, the common exponent of all four
layers factors out as a **whole-cube rotation**. The DP carries one of 24 pending
orientations forward through subsequent blocks, conjugating their moves accordingly, and
requires the final frame to be the identity.

Guarantee: optimal inside that factor/carry/conjugate system, not in the whole group.

Yield: on a slack-12 corpus, 18,287 replay-valid paths, **491 source paths shortened**; it
independently found ID686 and drove `46,304 -> 46,302`. At slack-20: 24,325 paths, 1,073
shortened, but strict gain over 46,302 was 0.

This is the cheapest of the four to implement and it composes with anything -- it is a pure
post-process on a path.

## 3. `twsearch --shortenseqs` with a DERIVED cubie definition

`src/make_twsearch_cubie_definition.py`, `src/twsearch_optimize_submission.py`.
Upstream <https://github.com/cubing/twsearch> @ `f90bbc843a30a9fc22d7dd3ca3c441c5a77c7270`.

The definition is derived from the **official group action**, not hand-written geometry: it
discovers centres, paired wings and oriented corners, and every cubie generator is decoded
back to 96 sticker destinations and required to match the official permutation. That check
is what makes it trustworthy.

Yield, their most productive exact rewriter: D8 ID66 47->45, D8-high48 ID505 48->46, D8 on
historical alternatives IDs 559+796 (-4), D10 full history ID850 47->45. D10 run: 1,255
candidates, 4,008 s, 19 alternatives shortened, 1 improved the incumbent.

We already have twsearch experience from IHES (`data/twsearch_cache/`,
`data/twsearch_rank_L22_*.json`), so this is a port, not a new dependency.

## 4. Coloured MITM is STRICTLY STRONGER than labelled rewriting

The conceptual point, and the reason to keep our own coloured tooling:

> `twsearch` rewrites the same **labelled cubie/group element**. That is stricter than
> colour-state equivalence, so coloured MITM can find shortenings that labelled rewriting
> cannot.

Their coloured r8 (`external/helpers/colored_mitm8_changed.cpp`): exhaustive canonical words
of radius 4 on both sides, checking 4+4 bridges between path checkpoints. Canonicalisation
uses only exact same-axis relations (commuting layers, exponents mod 4, deterministic layer
order). A Bloom filter is a no-false-negative prefilter **only** -- after any hash hit all
96 colours are compared. Interval DP picks the best disjoint window set.

Yield: exactly two production gains (ID575 window [31,38) 7->5; ID686 window [13,23) 10->8),
out of scans covering hundreds of IDs and thousands of seconds.

This confirms [[colour-space-window-rewriting]] from the opposite direction, and it is the
reason a colour-cube-native rewriter is not redundant with a permutation solver.

---

## Their zeros -- do not re-run these

Negative results they paid for, on a base at least as strong as ours:

- full-state exact splice r4 / r5 / r6 on a strong base: **0**
- broad diverse cross-r6, >1.4 billion candidate edges: **0**
- coloured suffix/window MITM r10 (3,512,239 canonical words, all length-11/12 windows): **0**
- scalar-agent beams (Zenodo agents 01/11/24, 56 attempts, 61.7M states expanded): **0**
- colour-stabilizer identities, D10 and D12 macro corpora, millions of pair joins: **0**
- Santa-2023 trajectories at r5 / r6 / targeted r7: **0**
- full audit of 373 versions of the two main public notebooks, over 46,314: **0**
- the public Q submission 46,662 did not improve 46,302

Note the last two against our own habits: [[kaggle-kernel-versioned-output]] says a version
sweep keeps paying on kernels still being re-pushed, but for cube444's two main notebooks
that seam is already mined out by someone else.

## Portfolio sources they used that we have not swept

- Kaggle notebook **version history** at scale: 941 versions, 830 with artifacts, 3,692 raw
  rows, 2,607 unique `(ID, path)`.
- GitHub result ledger <https://github.com/TryDotAtwo/cayleypy-beam-results> -- append-only
  verified results, 490 canonical Cube444 records, 459 unique `(ID, path)`, 385 IDs covered.
  Direct merge took the public base 46,718 -> 46,470.
- CC0 dataset `alexandervc/cayleypy-submissions`: 46 full Cube444 submissions, best single
  55,642, row-wise merge 54,754.
- Santa 2023 paths mapped to current IDs by **exact reconstructed initial state**
  (`initial = solved . inverse(solution)`), not by a fragile index offset -- 86 valid
  trajectories over the 43 IDs 1000..1042.
