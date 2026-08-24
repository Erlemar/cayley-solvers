# Verdicts — what has actually been measured

Every post-processing result on record for the four puzzles, so a closed axis is not
re-run. A blank cell means untested, which per DISCIPLINE.md rule 5 is **not** the same as
zero.

Read the *reach* column before reading the result: a zero at reach *R* says nothing about
R+1, and that distinction has already cost this programme one wrong write-up.

---

## IHES picture cube — best 21,870 (leader 21,840)

| method | scope | reach | result |
|---|---|---|---|
| n-way min-merge | 2,311 repo files (1,323 submission-shaped) | — | **21,870 (+0)** |
| n-way min-merge | 2,421 more profile-wide (Temp, Downloads, Documents, Desktop) | — | **21,870 (+0)** |
| single/2-step shortcut PP | full submission | 2 | 30–80 moves out of ~30,000 |
| BFS d5 table window replace | full submission | 5 | same ~80 moves, no more |
| BFS d6 table window replace | full submission | 6 | **0 additional** over d5 |
| commutator-library window replace | full submission | — | **0** |
| relation mining, trajectory crossover, cross-trajectory bridges | full | — | **0** each |
| GPU MITM, ranked cross-MITM | full | — | **0** each |
| plateau relations, Knuth–Bendix | full | — | **0** each |
| twsearch full-path proof | all 51 pids of length ≤ 20, threshold 18 | exact | **51/51 PROVEN OPTIMAL, 0 improvements** |
| twsearch w18 window sweep | 4,989 windows, run to completion | exact | 847 proven, **4,142 timed out**, 0 hits |
| twsearch W20 campaign | 2,670 windows | exact | ~1,100 proven, rest open |

**Cost curve** (twsearch, depth-11 prune table, `-M 8192`):

| task | threshold | 64 threads | 40 threads |
|---|---|---|---|
| length-20 full path | 18 | **49 s** | ~78 s |
| 20-move window (len ≥ 22) | 18 | — | **88 s** |
| length-21 full path | 19 | — | **865 s** |

≈ **11× per extra move of search depth**. Extrapolated: length-22 ≈ 9,500 s/pid (60 h
fleet-wide); length-23 out of reach at any budget available here.

**Parity halves the work**: every generator is an odd permutation of the 12 edges, so path
length mod 2 is invariant — a length-22 path can only shorten to 20, never 21.

**Standing verdict.** 21,870 is a three-way community plateau (CayleyPy, Chekhlov Dmitrii,
fle3n all tied at exactly that total). There is nothing better to merge, and every local
rewrite family is exhausted out to 20-move windows on the region searched. Do **not** re-run
window rewriting, relation mining, MITM, bridge compression, splicing or Knuth–Bendix on
this file. The gap has to be found by a better search, not harvested.

---

## Tetraminx — best 28,094

| method | scope | reach | result |
|---|---|---|---|
| n-way min-merge | every source on disk | — | **always pays**: 28,821 → 28,718 (−103) from sessions never merged |
| Kaggle kernel version sweep | 31 versions, 15 with a submission | — | **−51** (28,359 → 28,308), 13 versions each holding a unique win |
| Kaggle kernel re-sweep | v55–v119, 4 days later | — | **−91** over 72 pids (28,185 → 28,094) |
| Kaggle kernel sweep (other authors) | 276-version and 1-version kernels | — | **0** — a genuine null; their files are mostly long fallback |
| table window reduce | loose 29,622 floor | 6 | **−16** |
| table window reduce | merged 29,615 | 6 | −8 (6 rewrites) |
| table window reduce | our raw beam output, 12 pids | 6 | **0** (already locally irreducible) |
| table window reduce | the 28,821 merge | 6 / 7 / 8 / 9 | **0 at every reach** |
| MITM radius 10, 11 | sample | 10–11 | 0 |
| MITM radius 12, exhaustive | 166,703 queries | 12 | **0** — every window ≤ 12 is exactly geodesic |
| **MITM radius 13** | 67,235 queries, windows 14–18 | 13 | **−8 over 6 pids — the only exact win in the programme** |
| MITM radius 14 | — | 14 | not run: 38 GB, ~7.6 s/query, ~89 h — priced out |
| cross-trajectory splicing | full corpus, Dijkstra | — | **0, and bounded**: 0 of 6,979 exactly-known waypoints have slack |
| Knuth–Bendix completion | 1,844,967 derived rules | ≤ 9 | **0** — subsumed by radius 12 |
| rewrite alternative paths | 1,121 alternatives within +3 | 13 | **0** — sound in principle, no payoff at this yield |

Five of the six radius-13 wins were a 14-move window collapsing to exactly 13 — precisely
the band a reach-12 certificate cannot see. **Zobrist phantoms**: 7 in 46,687 queries at a
433 M-entry table.

---

## cube444 — best public 46,718 (leader 46,298)

| method | scope | reach | result |
|---|---|---|---|
| n-way min-merge | 125 replay-verified CSVs | — | **exactly 46,718 (+0)**; our own best loses on all 1043 pids |
| colour-space window MITM | 264,671 windows | 6 | **0** |
| colour-space window MITM | full file | 8 | **−2** (pid 761) |
| colour-space window MITM | full file | 10 | **−6 more** (pids 290, 540, 1030) → 46,710 |
| exact tail rung (front 5 + 67 M ball around solved) | full file | 11 | **0** |
| suffix re-solve, beam 16k | 6 pids × k ∈ {12,16,20} | — | **beat-k 0/18** — matched at k=12, never beat |

Every window of length ≤ 10 is geodesic, which on a colour cube implies geodesic in
permutation space too. Branching is **19.21** (level sizes 1, 24, 468, 9000, 172914), so
each further rung costs ~19× for ~2 moves.

**The one win worth knowing in detail** — it is the whole argument for matching states
rather than permutations:

```
pid 761  window [37,46)   r2.f2.r2.-d2.-f2.-f2.d2.f2.r2   (9)
                     ->   -d1.-d1.-f1.-d1.f1.-d1.-r2      (7)
permutations equal? False -- differ at 6 facelets, ALL of them centres
states equal?       True
```

The file's own name suggests its author had already run a depth-7 *permutation* rewriter.
The colour frame is what was left.

**Standing verdict.** The file is locally optimal out to ~12 and its slack is global. Do not
spend more GPU on exact rewriting or merging for this puzzle.

---

## Megaminx — 73,441 floor / 75,200 submitted

| method | scope | result |
|---|---|---|
| neural bridge compression | our own loose 77,214, top-50 long + next 62 | **−128** (77,214 → 77,086); 76 wins / ~2,200 attempts = 3.5 % |
| neural bridge compression | community 75,200, top-50 | −38 raw |
| neural bridge compression | community 75,200, hardest pids 990–1000 | **1 win / 576 attempts (0.2 %)** |
| neural bridge compression | 10 random mid pids of the 75,200 floor | −4 (TPU B=1M + sym4 + NISS, 9 pids) |
| neural bridge compression | pid 514 alone, 11.6 h GPU, sym4+NISS+iterate | −3 — **identical to a 23-minute plain beam-65k probe** |
| neural bridge compression | pid 1000, every config incl. TPU B=1M + sym4 + NISS | **0 at every configuration** |
| commutator-library window replace | full submission | 0 |
| window replace via BFS d6 | full submission | 0 additional over d5 |

**The boundary.** V saturates at ~25–30 (the puzzle diameter). For shallow residuals
(window 20–30) the predicted-saving signal is useful; for deep residuals (window 60+ on
hard pids) V predicts ~25 for everything while the true bridge depth is 60–70, so window
selection becomes pure false positives. Phase-2 diversity (sym4 + NISS = 8 sub-solves per
candidate) at TPU width 1M does not recover wins where the V landscape has no information
past the saturation horizon.

**Rule of thumb, as measured.** Path ≥ 80 moves *and* not from a community min-merge *and*
V is the best available → try it, expect 1–5 %. From a community min-merge → ≤ 1 %, smoke
test only. Hardest pid → no-op, spend the compute elsewhere. When it works, a cheap
single-pass beam captures all of it; bigger beam, symmetry, NISS and iteration are wasted.

---

## Cross-puzzle summary

| method | still worth running? |
|---|---|
| n-way min-merge (by content, re-pulled) | **yes, on every puzzle, every time** |
| Kaggle kernel version sweep | **yes**, on any kernel still being re-pushed |
| cheap local passes | yes — free, but expect 0.2–0.3 % |
| ball / table window sweep | yes on **loose** output; measured 0 on every tight merge |
| `--extend` to price a deeper table | **yes** — it is how d8 was rejected without building it |
| pooled bridging | yes if `slack_audit` finds slackful waypoints; otherwise provably 0 |
| exact ladder | only where the cost curve allows: short paths and windows |
| neural bridge | only on long, non-merged, our-own paths |
| suffix re-solve | **yes, as a diagnostic**, before committing to any of the above |
