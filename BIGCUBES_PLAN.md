# Big cubes: 5x5x5 / 6x6x6 (and 7x7x7) -- measured facts + attack plan

Scoping analysis written 2026-08-17. Nothing built yet. Competitions:

| comp | pids | deadline | teams | best on LB | sample_submission |
|---|---|---|---|---|---|
| `cayley-py-555-cube` | 1035 | 2026-11-20 | **3** | **470,424** (Jason Karpeles) | 503,776 |
| `cayley-py-666-cube` | 1012 | 2026-11-20 | **2** | **502,397** (= the sample, -2) | 502,399 |
| `cayley-py-777-cube` | ? | 2026-11-20 | 2 | ? | (rules not accepted -> 403) |

Both leaderboards are empty in substance: the 666 leader IS the sample submission, and
the 555 leader is 6.6% under it. There are two public notebooks total; the only
non-starter one (`nocarbonintelligence/555cube-gsr-pdbseg`) builds "additive PDBs" keyed
on 4 raw sticker indices with a greedy walk -- not a solver.

## 1. What the puzzles actually are (all measured, not assumed)

```
                          555            666
state_size                150            216
generators                30             36        f{0..n-1}, r{0..n-1}, d{0..n-1} + inverses
central_state             identity       identity  <-- SUPERCUBE, every sticker distinct
measured branching        24.092         29.023    (BFS ratio at depth 4)
exact |G| (Schreier-Sims) 6.1983e91      3.1440e149
counting lower bound      66.43 mv       102.20 mv
position orbits           6x24 + 1x6     9x24
```

**The single most important fact: these are supercubes, not colour cubes.**
`central_state == list(range(N))` and all 150 / 216 values are distinct. This is the
opposite of `cube444`, and it flips several conclusions carried over from there:

- `invert_state` **is** available -> inverse symmetry frames, NISS, and reverse-direction
  search are all back (cube444 had none of these).
- Symmetry frames: 48 (24 rotations x mirror) x 2 (inverse) = **96 frames**, vs cube444's 24.
- `num_classes` is **150 / 216**, not 6. But see the orbit-factored encoding below --
  do not use naive one-hot.
- The state pins the group element exactly (true Cayley graph, not a Schreier coset graph).

**Generators are single-layer quarter turns only.** No half turns, no wide moves, no
whole-cube rotations. So `R2` costs 2, `Rw` costs 2, `3Rw` costs 3, `Rw2` costs 4. Any
human/standard big-cube method roughly *doubles* in cost when priced in this metric.
This is the same metric and the same naming convention as the IHES picture cube
(`data/puzzle_info.json`: 72 stickers, `f0..f2/r0..r2/d0..d2`) and as Kaggle's
**Santa 2023** cube_5/5/5 and cube_6/6/6 -- that corpus of public code is directly on-metric.

### The test set is a scramble-length ladder, and the sample is the exact inverse scramble

`sample_submission` paths replay to `central_state` under `new[i] = st[perm[i]]` (60/60
checked) and their lengths are `1, 2, 3, ..., 1000` -- one pid per length, plus 35 (555)
/ 12 (666) duplicates. So:

- The scramble that produced pid p is a **plain uniform random walk** (3.311% of adjacent
  pairs are immediate inverses vs 1/30 = 3.333% expected -- not non-backtracking).
- We are handed the scramble word for free.
- **~92% of pids have L > 80 and are therefore statistically uniform-random deep states.**
  The "difficulty ladder" is only real for the first ~70 pids.

### Score is essentially linear in "moves per random state"

Because the sample is a valid per-pid fallback, total = `sum_p min(L_p, M)` where M is what
our solver achieves on a random state. Measured table (M vs total, min'd against the
commutation-reduced fallback):

| M (mv/random state) | 555 total | 666 total |
|---|---|---|
| 70 | 69,781 | 68,192 |
| 80 | 79,306 | 77,516 |
| 90 | 88,668 | 86,729 |
| 100 | 97,741 | 95,830 |
| 120 | 115,386 | 113,702 |
| 150 | 140,994 | 139,654 |
| 200 | 181,360 | 180,192 |
| 300 | 253,850 | 253,162 |

**Every move/pid saved is worth ~1000 points.** Perfect play is ~66-70k (555) and
~103-110k (666). The current leaders sit at 470k and 502k, i.e. **~7x and ~5x off**.

## 2. Where the difficulty actually lives

Information budget, from the exact orbit decomposition (bits / log2(branching) = moves):

**555** (4.590 bits/move):

| orbit | states | bits | moves-equiv | share |
|---|---|---|---|---|
| x-centres (24) | 24! | 79.04 | 17.2 | 26% |
| t-centres (24) | 24! | 79.04 | 17.2 | 26% |
| wings (24 pieces) | 24! | 79.04 | 17.2 | 26% |
| midges (12) | 12!*2^11 | 39.84 | 8.7 | 13% |
| corners (8) | 8!*3^7 | 26.39 | 5.7 | 9% |
| face centres | 24 (core rotation) | 4.58 | 1.0 | 1% |
| **sum** | | 307.93 | **67.1** | (exact LB 66.43; 3.00 bits of parity slack) |

**666** (4.859 bits/move):

| orbit group | bits | moves-equiv | share |
|---|---|---|---|
| centres (4 orbits x 24) | 316.15 | 65.1 | **63%** |
| wings (2 orbits x 24) | 158.08 | 32.5 | 32% |
| corners (8) | 26.39 | 5.4 | 5% |
| **sum** | 500.62 | **103.0** | (exact LB 102.20; 4.00 bits slack) |

The parity slack coming out at exactly 3.00 and 4.00 bits confirms the decomposition.

**Read this and the strategy falls out: on big cubes the puzzle IS the centres.**
53% of 555 and 63% of 666 is pure centre permutation -- 24-piece orientation-free
orbits. The "Rubik's cube part" (corners + midges) is 22% of 555 and 5% of 666.
Everything we have built across four puzzles is tuned for the corners/edges regime.

### Generator-orbit incidence (555) -- the structure that makes phases work

| orbit | outer {f0,f4,r0,r4,d0,d4} | inner {f1,f3,r1,r3,d1,d3} | middle {f2,r2,d2} |
|---|---|---|---|
| corners (2,2) | yes | - | - |
| midges (0,2) | yes | - | yes |
| t-centres (0,1) | yes | yes | yes |
| x-centres (1,1) | yes | yes | - |
| wings (1,2) x2 | yes | yes | - |
| face centres (0,0) | - | - | yes |

Two exploitable consequences:

1. **x-centres and t-centres never leave their face under outer-layer turns** -- they only
   rotate within the face. So once centres are solved, the constraint on the remainder is
   exactly "each face's net rotation == 0 mod 4", which is *precisely the 3x3x3 supercube
   condition*. **The last phase of a 555 reduction is the IHES picture cube**, which we
   already solve at 21.87 moves/pid.
2. **The 6 face centres are moved only by the 3 middle slices, and they move as a rigid
   body** (24 reachable configurations = the cube rotation group). Odd-cube only. Forget
   this and you get paths that solve everything except a 24-way core rotation.

## 3. Why the existing stack does not just port

The load-bearing question is whether a neural V has usable range at ~70-100 moves.
Evidence both ways:

- **For**: megaminx is |G|=1e68 with counting LB ~52 and we ship **73 moves/pid** = 1.40x LB
  with a V that saturates at 25-30. Scale-collapse is benign (cube444); only
  discrimination/SNR matters. If 1.40x LB transferred: 555 -> 93 mv (~91k), 666 -> 143 mv
  (~133k). That alone would be a 5x and 3.8x lead with essentially existing code.
- **Against**: 1e91 / 1e149 vs 1e68. A random 555 state has ~92 misplaced pieces and one
  move touches at most 11 of them. The saturation ceiling that killed bridge compression
  past d~30 and killed the GT-V (`gt_v_no_saturation`) is the same wall, and here the wall
  sits at roughly *half* the required depth.

**This is cheap to settle and the test set settles it for free.** Because pids are ordered
by scramble length 1..1000, solving pids in order and finding the L* where solve-rate
collapses is a direct, precise measurement of the reachable depth. That is experiment #1.

## 4. The plan, ranked by return

### Tier 0 -- free, do it first (hours)

**T0.1 Commutation reduction of the given scramble words.** All layers of the same axis
commute, so any maximal same-axis run collapses to at most one net turn per layer.
Iterate to a fixed point. **Measured: 555 503,776 -> 456,814 (-9.3%), 666 502,399 ->
463,203 (-7.8%).** Both immediately take #1 on both leaderboards, with ~30 lines of Python
and no solver. Plant the flag, then never think about it again except as the fallback floor.

**T0.2 twips window rewriting on those words.** We already have this working
(`scripts/20/22/24_twsearch_*.py`, `third_party/twips/target/release/twips.exe`, and it
produced the IHES 21,870 at w16-w18). `twips search --experimental-target-pattern` +
`--generator-moves` is exactly a coset solver. Iterated random-offset windows at radius
12-16 over a 450-move word should give another 20-35% with zero modelling. Estimated
landing zone ~300-350k. Also the permanent rescue path for any pid the real solver fails.

### Tier 1 -- the decisive measurement (days)

**T1.1 Parameterized puzzle module** for n in {5,6,7}: `cube_nnn/puzzle.py` +
verifier + symmetry. cube444's tree is already shape-generic; the deltas are
`num_classes`, the supercube sym (rotate only, **no colour relabel needed**), and
`invert_state` coming back.

**T1.2 Orbit-factored encoding -- do not use naive one-hot.** Every position lies in a
closed orbit of 24 (or 6), so the sticker at position i is always one of that orbit's 24.
One-hot over 24 instead of over N:

| | naive N x N | orbit-factored N x 24 | ratio |
|---|---|---|---|
| 555 | 22,500 | 3,600 | 6.25x |
| 666 | 46,656 | 5,184 | 9.0x |
| 777 | 86,436 | 7,056 | 12.25x |

Cheaper first layer, and it is the *correct* factorization -- the group acts orbit-wise.

**T1.3 The ladder probe.** Train a quick V (cube444's recipe: random walks +
Bellman + exact d<=6 anchors), then beam pids in scramble-length order and find L*.
Report solve-rate and achieved length by scramble length. This single number decides
Tier 2 vs Tier 3, so run it before building anything else.

### Tier 2 -- if direct search has range (L* >= ~100)

Straight port of the tetraminx/megaminx production recipe. Everything here is
already-written code:

- **Sym-pooled beam over 96 frames** (48 rotations/mirrors x inverse). `sympooled_beam_validated`:
  one K*B beam beats sequential frames, and on tetraminx *the inverse frame carries the win*.
  cube444 could not do inverse frames at all; here we can.
- **Width.** Width has been the strongest lever on every puzzle so far. TPU kernel port is
  the cube444 port again, with `PACK_SIZE` raised to >=155 (555) / >=221 (666); JAX
  persistent compile cache is mandatory.
- **Exact endgame ball.** d<=6 from solved is 24.09^6 = 1.96e8 hashes (~1.6 GB) for 555 --
  buildable; makes the tail provably optimal where the beam is narrowest.
- **Q-head instead of V** (17.2x cheaper per beam step; what tetraminx uses).
- **Self-improvement**: all ~950 hard pids are i.i.d. samples from one distribution, so
  every solved path is training data for every other pid. AZ/DAgger loop applies cleanly.

### Tier 3 -- coset ladder (needed for 666, probably; maybe for 555)

Phases as **nested orbit sets**, searched with all generators, target = a coset
(orbits 1..k solved). This is not Thistlethwaite by generator restriction -- it is
Kociemba-style coset solving, and the masked-V machinery exists (megaminx
`two_phase_solver`: "reduce into the subgroup via MaskedV").

555 ladder and its per-phase counting bounds:

| phase | target | phase LB | notes |
|---|---|---|---|
| P1 | x-centres + t-centres + face centres | 35.4 mv | the real work; 53% of the puzzle |
| P2 | + wings | 17.2 mv | must land centres too, so effective branching is lower |
| P3 | + midges + corners | 14.4 mv | **= the IHES picture cube**, measured 21.87 mv/pid |

Sum of phase LBs 67.0 vs global LB 66.4 -- the *information* penalty for phasing is
~1%. The real penalty is search inefficiency per phase. Calibration point: on the IHES
phase we achieve 21.9 against a 18.9 LB = **1.16x**. At 1.16x across the board, 555
reduction lands at ~78 mv (~77k) and 666 at ~120 mv (~114k).

Key design tension, stated plainly:

> Longer phases give a shorter total but need a heuristic with long range (the neural V,
> which saturates). Shorter phases are exactly solvable but sum to more moves.
> **Split until each sub-phase's optimum falls inside the heuristic's usable range.**

A wide beam with a noisy V is precisely the tool that lets phases be *longer* than exact
search permits, because it needs only local ordering, not admissibility or range. That is
our edge over a classical reduction solver, and it is the same edge that got megaminx to
73 with a V that saturates at 25-30.

**Phase-boundary softening** is where we beat classical reduction: instead of hard phases,
score with `w1*h_centres + w2*h_wings + w3*h_333` and anneal the weights. Recovers most of
the sum-of-phases gap. (Note: megaminx `subgoal_search_rejected` found gating V every k
steps hurt -- but that was gating a V that already had range, on a puzzle where direct
search worked. Different regime; re-measure, do not assume either way.)

### Tier 4 -- centre-specific machinery (the 63% nobody else will build)

The centres are 2 (555) / 4 (666) orbits of 24 orientation-free pieces. This is closer to
a multi-orbit token-swap problem than to a Rubik's cube, and it has exact structure we can
exploit without any training:

**Exact k-piece PDBs per orbit.** Track the placement of k specific pieces among 24 slots:

| k | entries | @1 byte | @4 bit |
|---|---|---|---|
| 5 | 5,100,480 | 5 MB | 3 MB |
| 6 | 96,909,120 | 97 MB | 48 MB |
| 7 | 1,744,364,160 | 1.7 GB | 872 MB |
| 8 | 29,654,190,720 | 29.7 GB | 14.8 GB |

k=6 fits in VRAM by the handful (8 tables = 776 MB) so the beam can use a *sum of exact
tables* as its centre scorer with no training at all. k=8 fits on the TPU host
(180 cores / 1417 GB RAM -- `tpu_host_is_a_big_cpu_box`), which is the right box for
building these anyway (CPU + RAM bound, not accelerator bound).

Honest caveat: **a sum of k-piece PDBs saturates too.** On a random configuration each
subset is itself near-random, so each table sits near its own mean and the sum is
near-constant. PDBs buy *reliability near the goal*, not long-range gradient. They are the
per-phase endgame and the sub-phase scorer, not the answer to saturation. The answer to
saturation is shorter phases plus beam width.

Also worth pricing: pure-3-cycle macros are the wrong tool here. A random 24-permutation
needs ~11 3-cycles, and an optimal pure 3-cycle in single-slice QTM costs 10-14 moves, so
commutator-style centre solving is ~130 moves/orbit. Search, not algorithms.

### Tier 5 -- always-on

- **N-way per-pid min over every source**, replay-verified (rules 26 / 26b). Sources:
  every beam run, every phase order, every frame, the twips-rewritten words, the raw
  sample. Copy sources into a stable dir; do not `--extra`-glob scratchpads.
- **twips window rewriting on the final paths**, radius 12-16. On a ~100-move path with
  phase seams there is real slack; this is the pass that produced the IHES 21,870.
- **Re-run the worst-achieved pids at higher width.** Unlike cube444 there is no long tail
  of hard pids (all ~950 are the same distribution), so the first pass should be uniform;
  target by *achieved* length afterwards.

## 5. Traps specific to these two puzzles

1. **Supercube, not colour cube.** Do not carry over cube444's `num_classes=6`, its
   "no invert_state", or its colour-relabel symmetry. Do carry over the `num_classes` trap
   itself: constructors default it wrong, pass it explicitly.
2. **555 core rotation.** The 6 face centres are an orbit moved only by `f2/r2/d2`.
   24 reachable configurations. A solver that ignores them will converge to
   "solved except a whole-cube rotation" and never close. 666 has no such orbit.
3. **`PACK_SIZE`** in the TPU beam kernel is 128 and must go to >=155 (555) / >=221 (666)
   / >=299 (777). cube444's HANDOFF already flags this as the one expected edit.
4. **Metric conversion.** Anything imported from the cubing world counts wide and half
   turns as 1. Price every imported algorithm in single-slice QTM before believing its
   move count. `Rw2` is 4 moves here.
5. **Sample submission is a per-pid fallback, always min against it** -- and against its
   commutation-reduced form, which is 9.3% shorter for free.
6. **Rule 28 still binds.** Matched controls; byte-identical A/B results mean an unwired
   flag; score against `sum_p min(L_p, ...)` over the true current best file, not a
   remembered number.

## 6. First week, concretely

| # | task | outcome |
|---|---|---|
| 1 | commutation-reduce both sample files, verify, submit | **#1 on both LBs** (456,814 / 463,203) |
| 2 | accept 777 rules, pull its data | third empty leaderboard, same code |
| 3 | `cube_nnn/` module: puzzle, verifier, 96-frame symmetry, orbit-factored encoder | shared substrate for 555/666/777 |
| 4 | twips window-rewrite pass over the reduced words (radius 12-16) | ~300-350k, plus the permanent rescue path |
| 5 | quick V train (cube444 recipe) + **ladder probe** | L*, the number that picks Tier 2 vs Tier 3 |
| 6 | branch on L* | direct-search push, or build the P1 centres coset solver |

Build everything parameterized on n. 555, 666 and 777 are three trophies from one
codebase, all with the same deadline and all currently unclaimed.

---

# Revision 2026-08-17b: the scorer wall, measured

A ResMLP was trained for 666 and reported as failing: "mixing point of random walks is
~50, so Q saturates at ~72". Two of those three claims do not survive measurement, and the
third is fatal for a different reason than assumed. This section replaces sections 3-4
above where they conflict.

## R1. Random walks are near-geodesic. They do NOT mix at 50.

Exact distances by bidirectional BFS (ball(<=4) from identity = 446,403 states for 555 /
928,804 for 666, intersected with ball(<=4) from the sample; 25 samples per L):

| L | 555 exact-d dist | mean d | d/L | 666 exact-d dist | mean d | d/L |
|---|---|---|---|---|---|---|
| 4 | {4:25} | 4.00 | 1.000 | {2:3, 4:22} | 3.76 | 0.940 |
| 6 | {4:4, 6:21} | 5.68 | 0.947 | {4:6, 6:19} | 5.52 | 0.920 |
| 8 | {4:2, 6:7, 8:16} | 7.12 | 0.890 | {6:5, 8:20} | 7.60 | 0.950 |

Theoretical drift `1 - 2/G` (G = number of generators) is **0.933** for 555 and **0.944**
for 666, from two independent derivations: (a) per step, ~1 of G generators decreases the
distance, so the walk loses 2 steps per backtrack; (b) the random-walk entropy argument,
`drift >= h / log b`, which is near 1 because the growth rate b (24.09 / 29.02) is close to
the generator count G (30 / 36). Measurement matches.

**Consequences:**

- A length-L walk sits at distance ~**0.93L** (555) / ~**0.94L** (666), saturating only at
  the diameter. So walks stay informative to **L ~ 71** (555) and **L ~ 108** (666).
  Training data is available to depth ~108, not 50.
- **No path needs to be 300 moves.** Counting bound 102.2, average optimal ~107-118,
  diameter maybe ~135. If the pipeline is emitting 300+, that is the pipeline, not the
  puzzle. The target is ~130-160 and the ceiling is ~110.
- A Q topping out at 72 is therefore a **labelling / capacity artifact**, not an
  information limit -- most likely `k_max` on the walk generator, or Bellman scale
  collapse (which cube444 measured as *benign*: the more-collapsed checkpoint beamed
  better). See R2 for why fixing it still will not save direct search.
- Free invariant: every generator is an odd permutation (555 `f0` = 11 four-cycles,
  `f1` = 5), so **d(s) == sgn(s) mod 2**. Solution length parity is known before searching.
  Visible in the table above -- every measured distance has the parity of L.

## R2. Why direct search dies anyway -- and it is not the model

At distance d, one move changes the true distance by exactly 1, i.e. by `1/d`. A learned
scorer has a noise floor of order 1-2 moves. So sibling ranking is only possible while
`1/d` exceeds the noise, i.e.

> **the usable band of ANY heuristic is targets at distance <~ 20.**

At d = 110 a single move is a 0.9% change in the objective. No architecture, encoding,
objective, or amount of data fixes that -- it is a property of the distance field on an
expander, not of the network. Mean saturation is the wrong diagnostic (cube444: gate on
discrimination, never absolute scale); the right one is **child-recall by true depth**
(`09b_eval_q_recall_by_depth.py`, and `allneighbor_qhead_rejected`: recall BY DEPTH is
the gate). But the conclusion holds either way: **stop trying to widen the range of one
scalar.**

## R3. The fix: a VECTOR of small distances, not one big one

Distance to a *set* is `log_b(index of the set)`. So pick targets that are huge sets:

| target for 666 | index | distance at b=29.02 |
|---|---|---|
| one 24-piece centre orbit placed | 24! = 6.2e23 | **16.2 mv** |
| two centre orbits placed | (24!)^2 | 32.5 mv |
| all 4 centre orbits | (24!)^4 | 65.1 mv |
| everything (the identity) | 3.14e149 | 102.2 mv |

**Solving one orbit is a 16-move problem.** That is inside the band by a factor of 2.
So: replace one V with range 110 by **k heuristics each with range <= ~18**, one per
nested coset, and never evaluate a "110" anywhere in the search.

**Soft ladder, not hard phases.** Score candidates with `sum_i w_i * h_i(s)` and anneal
`w` from "centres only" toward "everything". Hard phases force each rung to be reached
exactly before the next begins, which collapses the effective branching; the annealed
weights let the beam break an earlier rung by 2-3 and recover, which is where the
sum-of-phases penalty is won back. Every `h_i` stays small, so every `h_i` is learnable.

## R4. Getting the h_i without training: exact projected PDBs

A rung's heuristic only needs the pieces the rung is about. Exact BFS over a projection:

| pieces tracked (of 24 in an orbit) | entries | @1 byte | @4 bit |
|---|---|---|---|
| 6 | 96,909,120 | 97 MB | 48 MB |
| 8 | 29,654,190,720 | 29.7 GB | 14.8 GB |

k=6 tables fit in VRAM by the handful; k=8 fits on the TPU host (180 cores / 1417 GB,
`tpu_host_is_a_big_cpu_box`), which is the right box for building them anyway.
**Sum of 3 disjoint k=8 PDBs covering an orbit** has range ~30 against a rung distance of
16, and each move changes it by a few units out of ~27 -- ~10% per move, i.e. *inside the
band*. The saturation objection from section 4 above applies to the whole-puzzle distance;
it does **not** apply within a rung. These tables need no training and no labels.

## R5. The cost model that predicts the score

Cost of a rung = (bits of the rung) / (bits-per-move of the move set that rung can use).

- Unconstrained moves: `log2(29.02)` = **4.86 bits/move** (666), `log2(24.09)` = 4.59 (555).
- Moves that must preserve an already-solved orbit are words, not single moves. Estimate
  from 12-move commutators `[A,B]` where A and B are each 3 moves long:
  `log2(29^6)/12` = **~2.3 bits/move**.
  *This number is an estimate, not a measurement -- measure it early, it sets everything.*

| 666 | bits | bits/mv | moves |
|---|---|---|---|
| centres (4 orbits, first is free) | 316 | ~4.0 eff | ~80 |
| wings (2 orbits, centre-preserving) | 158 | ~2.3 | ~70 |
| corners (outer turns, high density) | 26 | ~3.6 | ~7 |
| **total, first cut** | 501 | -- | **~157** |

555 the same way: centres ~35 + wings ~34 + the 3x3x3 supercube finish (measured 21.9 on
IHES) = **~92**. Scores: 555 ~90k, 666 ~150k, against leaders 470k and 502k.
With soft blending, portfolio-over-rung-orders and window polish, 555 ~80-85k and
666 ~125-145k look reachable.

**The number that kills fine ladders:** the counting bound demands
`152 pieces / 102 moves` = **1.5 pieces placed per move** for 666. A pure commutator or
piece-at-a-time method places ~0.2 pieces/move (a 3-cycle costs 10-14 moves here), which
caps it at ~750 moves. Use this ratio to reject a rung design before implementing it: if
a rung places fewer than ~0.6 pieces/move, it is too fine.

## R6. twips is a rung engine we already own

`twips search --experimental-target-pattern <partial> --generator-moves <subset>` is
literally a partial-goal optimal solver, and we have the binary plus
`scripts/20/22/24_twsearch_*.py`. Rungs of depth <= ~14 are exactly solvable with no ML at
all. **A pure-twips ladder is a working 666 solver with zero training** -- probably
~180-250 moves, i.e. ~165-215k, already 2.5-3x the current leader. Build that first as the
floor, then replace the deep rungs with the learned soft ladder.

## R7. Priced and rejected

- **Colour-cube first, then fix centres to exact slots.** The residual is a random element
  of `(S4^6)^4`, ~36 three-cycles at ~9 moves each = **~324 moves**. Dead.
- **Minkwitz / stabilizer-chain sifting** (the standard "solve any permutation group"
  answer, in GAP). Guaranteed valid, fully automatic, no training -- and ~1500 moves for
  152 pieces. Keep only as a safety net that guarantees no pid is ever unsolved.
- **Meet-in-the-middle.** Two beams of width B meet with probability ~B^2/|G|; at B=1e7
  and |G|=3e149 that is never. Consistent with `bidir_mitm_rejected`.
- **Widening the range of a single scalar V** (bigger trunk, better encoding, more Bellman,
  transformer). R2 -- it is the distance field, not the model.

## R8. Leads worth chasing

**Santa 2023** used the identical metric, the identical `f0/r0/d0` naming, both colour and
distinct-sticker (supercube) variants, and included cube_5/5/5 and cube_6/6/6. Reported
character of the winning solutions: group theory, coset/ladder decompositions, and heavy
local optimization, "almost all implemented in Rust or C". That corpus is on-metric and
public. Note the Kaggle CLI 2.2.0 `forums` verb has no search (`list`/`topics` only, the
old `-s` flag is gone), so mine it via the competition discussion pages directly.
