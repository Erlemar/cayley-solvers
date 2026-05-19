# Megaminx — Novel paradigm ideas (2026-05-04)

Working document synthesizing 7 user-proposed ideas + research findings on
alternative paradigms beyond "value-model + beam search". Status: brainstorm
with feasibility analysis. Items here are NOT yet on `to_do_shortlist.md`;
when one is promoted to active execution, move the section there and delete
it here.

**Anchor**: current best 79,522, goal <70,000 (-12% moves), stretch 60,000.
Cluster ceiling at 6M-param Bellman MLP confirmed across m05-m38. Inference
levers (sym-ensemble K=4, search-config diversity, T2.2 tail-resolve, T1.6 SA
macro-insert) shipped and producing diminishing returns per pass. The
next break has to come from a structurally different signal class, action
space, or search algorithm — not another knob on the existing recipe.

---

## Tier ranking (leverage x feasibility)

**Tier 1 — concrete, partial scaffolding shipped, high leverage**

1. [Macro-Q shortlist](#1-macro-q-shortlist) — NOVEL
2. [Curated speedcubing macros in beam](#2-curated-speedcubing-macros-in-beam) — T1.1 queued
3. [Frontier replay / DAgger training](#3-frontier-replay--dagger-training) — B9 queued
4. [Policy-guided V + pi search](#4-policy-guided-v--pi-search) — B10 queued

**Tier 2 — paradigm shifts, 1-2 weeks engineering**

5. [Coordinate solver hybrid](#5-coordinate-solver-hybrid) — PDB long-list + T2.4
6. [ReduceFactor DAG path optimization](#6-reducefactor-dag-path-optimization) — E6 queued, under-scoped
7. [Path relinking across solutions](#7-path-relinking-across-solutions) — NOVEL

**Tier 3 — research, multi-week**

8. [Twsearch integration with KPuzzle coordinates](#8-twsearch-integration)
9. [Learned bidirectional / front-to-front D(s, t)](#9-learned-bidirectional--front-to-front)
10. [Tiny per-phase ML specialist](#10-tiny-per-phase-ml-specialist) — NOVEL
11. [Megaminx insertion-finder](#11-megaminx-insertion-finder) — NOVEL

**Skip — confirmed weak or dominated**

- Brute-force commutators (window OR naive macro action) — m05-era confirmed null
- AlphaZero / MCTS at our compute budget — research agent confirmed worse than
  beam in fewest-moves regime; jasonrute/puzzle_cube + McAleer 2018 evidence
- SAT / MILP whole-puzzle — state space too large; only viable as window
  post-proc, where BFS-d6 already saturates
- More single-knob Bellman recipe variants at 6M params — cluster ceiling
  thoroughly explored (m17, m22, m26b, m27 family, m28, m29, m30, m31, m32,
  SWA m05, m38)

---

## 1. Macro-Q shortlist

**One-liner**: Extend the m23 Q-shortlister output dimension from 24 to ~150
(24 primitives + ~125 curated macros) so the beam-expansion shortlist
includes macros, but only when the model thinks they're plausibly the best
move at this state.

**Why genuinely novel**: solves the "V dilution from brute-force macros"
finding (2026-04-30: 6 to 30 d=4 commutators net +9 to +41 moves; V isn't
macro-aware so brute-force macros dilute the candidate pool). Current m23
outputs Q over 24 primitives only. Expanding the action space at the
shortlister level means the gating happens BEFORE V scoring — the macro is
filtered out unless m24 thinks it's competitive. m05's V still scores final
children; the macro just gets a chance to be considered.

**Status**: NOT QUEUED. Distinct from queued T1.1 ("macros in beam") which
adds macros to the action set without shortlist gating.

**Mechanism**:

1. Curated macro set `M = {(name, gen_word, perm_120) ...}`, ~125 entries
   from SpeedCubeDB scrape (Idea 2).
2. Action set `A = primitives ∪ M`, |A| ~ 150.
3. m24 (new Q-head, 12-15M params) trained by distillation. Target:
   `Q*(s, a) = V_m05(apply(s, a)) - cost(a)`. For primitives `cost = 1`,
   for macros `cost = |gen_word|`.
4. Beam expansion: at each parent, take top-α actions by m24, expand only
   those, re-rank children by V_m05. (Same alpha-shortlist logic as m23,
   bigger action vocabulary.)
5. Path reconstruction: when a macro action is chosen, expand to its
   `gen_word` in the output path.

**Why this should help**:

- Macros change the local branching factor *upward* (more actions to choose
  from) but the shortlister keeps effective per-step compute flat
  (top-α stays the same).
- Wins come from macros that solve sub-patterns in 6-12 moves where
  primitive-only beam would take 12-20.
- T1.6 SA's `macro_insert` operator already produced 48 wins on top-100 with
  brute-force commutators at 7.8% accept rate. Curated macros + shortlist
  gating should hit higher accept rate AND apply globally, not only at SA
  positions.

**Expected impact**: -5K to -15K moves vs current 79,522. Lower bound from
T1.6 macro_insert + curation. Upper bound from CayleyPy paper's per-action-
class gains.

**Falsifiable test**:

- Build minimal 30-macro library (top OCLL/EPLL/CPLL by move count).
- Train m24 (~8h on local 4090).
- Evaluate on stratified-5 vs `m05+m23_v2+sym4` baseline.
- Acceptance: >=+3 solves AND mean <= 84.9 (0.95 x m05's 89.4).
- If passes: scale macro set to ~125 and re-run on top-200 long-tail pids.

**Engineering**: ~3 days after macro library exists. Touches:

- `src/cayley/khoruzhii_search.py` (Q-function path, ~lines 107-115)
- `scripts/02_train.py` (Q-distill mode, action_dim parameter)
- New `scripts/41_macro_qhead.py` for distillation pipeline

**Risk**: training data balance. Primitives appear in EVERY transition;
each macro appears in <1% of transitions. Need stratified sampling per
action class or class-weighted loss.

**References**:

- SpeedCubeDB: <https://speedcubedb.com/a/Megaminx>
- Macro infra: `src/cayley/khoruzhii_search.py:144-165` (per codebase
  exploration 2026-05-04)

---

## 2. Curated speedcubing macros in beam

**One-liner**: Scrape ~100-410 named OLL/PLL-style algorithms from
SpeedCubeDB, convert notation, augment beam action set with curated macros
(no Q-shortlist filtering — the simple version).

**Why useful**: brute-force d=4 commutators failed because they're random
algebraic atoms, not compressed domain knowledge. Speedcubing macros encode
"this 7-move sequence solves this corner-orientation case" — domain experts
found these by hand; we get them for free.

**Status**: QUEUED as T1.1 in `to_do_shortlist.md`. Mechanism shipped
(`KhoruzhiiSolver(macros=...)`); data is the missing piece.

**Available data** (verified by research agent, 2026-05-04):

| Set | URL | # algs | Avg moves |
| --- | --- | --- | --- |
| 4LLL (EOLL+OCLL+EPLL+CPLL) | speedcubedb.com/a/Megaminx/MegaminxOLL28 etc. | 39 | ~10 |
| 2LLL (full OLL+PLL) | speedcubedb.com/a/Megaminx/MegaminxOLL | 410 | unknown |
| Andy Klise PDFs | kungfoomanchu.com/guides/andy-klise-megaminx-old.pdf | overlap | various |

**Notation conversion** (critical):

- Standard CFOP `R / R' / R2 / R2'` -> our `R+ / R- / R++ / R--`
- Wide moves `r, u` and slice moves `M, E, S` -> expand or treat as state
  remappings
- Rotations `x, y, z` -> eliminate by applying to subsequent moves
- **Caveat**: on megaminx `U2 != U2'` (per speedsolving wiki) — verify each
  alg roundtrips before accepting
- Face-name mapping: SpeedCubeDB uses {U, F, L, R, BL, BR, DL, DR, DBL, DBR,
  B, D} or similar; confirm against our internal face indices

**Symmetry expansion**: each algorithm has 60 geometric conjugates via
`data/rotations.npy` (A_5 x C_6 = 360, of which 60 are pure rotations
preserving solved state). A 30-alg base library -> 30 x 60 = 1,800 macros.
Most are redundant; dedupe by permutation.

**Engineering**: ~1 day scraping + conversion, ~0.5 day verification (each
alg roundtrips after K applications, where K is its order in the group),
~0.5 day plumbing into `KhoruzhiiSolver`. Used as input to Idea 1.

**Standalone expected impact** (without Idea 1's Q-shortlist gating):
unclear. Brute-force commutators dilute V; curated macros are higher-quality
but still face the dilution problem. Best deployed with Idea 1.

**References**:

- SpeedCubeDB structure verified by research agent 2026-05-04
- `src/cayley/khoruzhii_search.py:144-165` (macro infra)
- `data/rotations.npy` (360 sym-conjugates)
- Speedsolving wiki notation page: speedsolving.com/wiki/index.php/Megaminx_notation

---

## 3. Frontier replay / DAgger training

**One-liner**: Log beam-frontier states from real solver runs, label with
TRUE remaining suffix length, mix into Bellman training at 25-50%.

**Why genuinely useful**: the cluster ceiling at 6M params is robust across
recipe variants — diagnosis is data-bound, not arch-bound. The training
distribution is random walks; the inference distribution is beam frontiers.
RW states are systematically different from frontier states (frontiers are
biased toward "almost-solved" and "previously-pruned-by-V"). Distribution
shift is the most plausible remaining training-side lever.

**Status**: QUEUED as B9 in `to_do_shortlist.md` (~80 lines). Never run.

**Mechanism**:

1. Run current best solver (`m05+m23_v2+sym4+524k`) on ~200 pids; log all
   frontier states `(state, parent_path_len, eventually_solved, true_suffix_len)`.
2. Filter to states where `true_suffix_len` is known (~80% of frontier states
   from solved pids — `true_suffix_len` = exact length of the realized solve
   path from this state to solved).
3. Build dataset of `(state, true_suffix_len)` pairs. Exact integer labels,
   not Bellman bootstrap.
4. Train m_FR with 25%-50% mixin from frontier dataset, 50%-75% from random
   walks (existing m05 recipe).
5. Eval on stratified-5 + acceptance gate.

**Distinct from**:

- m27 BFS-d6 mixin: that was IID training-distribution states with exact
  labels (within d<=6 shell). B9 is OOD-distribution states with exact
  labels. Different distribution shift.
- Bellman round 2 (m17): same distribution, different target. B9 changes
  the distribution.

**Expected impact**: -2% to -10% mean path length. The m29 result (n_back=4
-> 88.98 mean, only sub-89 training-side recipe) hints distribution-shift
mitigations work. Frontier replay is a stronger version of the same
mechanism.

**Falsifiable test**: 1 training run (~8h on local 4090) + strat-5 eval. If
mean improves <=2 vs m05's 89.4, distribution shift wasn't the binding
constraint and we deprioritize the entire B9/B10 cluster.

**Engineering**: ~80 lines per shortlist. Touches:

- `scripts/02_train.py` (mixin loader)
- `src/cayley/bellman.py` (data path)
- New `scripts/42_log_frontier_states.py` (data collection)

**Risk**: solved-frontier states may be EASIER than the bottleneck states
(those that didn't get solved). Mitigation: also collect from PRUNED
frontiers (states V scored low but were near-optimal in retrospect via
T2.2 tail-resolve). Pruned-frontier states teach V where it's overconfident.

---

## 4. Policy-guided V + pi search

**One-liner**: Train a separate policy head pi(a|s); score beam children as
`V(child) + lambda * (-log pi(a|parent))`.

**Why genuinely useful**: V says "this state looks close to solved"; pi says
"this action is statistically used in solves of similar states". Different
signals, complementary at scoring time. Orseau & Lelis 2022 "Policy-Guided
Heuristic Search" gives search-effort guarantees for this class.

**Status**: QUEUED as B10 in `to_do_shortlist.md`.
`scripts/11_train_policy_head.py` exists, never run.

**Distinct from m38 listwise loss** (which regressed by -2 solves and +10
mean): m38 replaced V's MSE objective with ranking. Here pi is an
ADDITIONAL signal added at scoring time, not a replacement objective. V
training is unchanged.

**Distinct from m23 Q-shortlister**: m23 outputs Q(s, a) used to FILTER
beam expansion to top-alpha. pi outputs P(a|s) used to RE-WEIGHT scoring of
all expanded children. Could combine both (m23 filters, pi re-weights).

**Distinct from Idea 1 Macro-Q shortlist**: Idea 1 expands the action space
and gates expansion. Idea 4 keeps the action space at 24 primitives but
re-scores using pi. Orthogonal mechanisms; could compose.

**Mechanism**:

1. Distillation training data: `(state, action_taken, suffix_len_remaining)`
   tuples from VERIFIED best paths + SA-improved paths (T1.6 outputs).
2. Loss: `CE(pi(s), action_taken)` weighted by `1 / (1 + suffix_len)` so
   good moves on short solves count more (these are the moves we most want
   to imitate).
3. Beam scoring: `score(child) = V(child) + lambda * (-log pi(a|parent))`
   for `lambda in {0.05, 0.1, 0.2, 0.5}` sweep.
4. `lambda=0` reduces to current beam; `lambda -> infinity` reduces to
   greedy policy rollout. Sweepable on strat-5 with no retraining.

**Expected impact**: -2% to -8% mean path length. Lower variance than B9
because doesn't change V at all — pure inference-side lever.

**Falsifiable test**: lambda-sweep on stratified-5. If best lambda doesn't
improve mean by >=3 vs m05's 89.4, deprioritize.

**Engineering**: ~2 days. `scripts/11_train_policy_head.py` already exists;
need to verify it works + add scoring change in
`src/cayley/khoruzhii_search.py`.

**References**:

- Orseau & Lelis 2022, "Policy-Guided Heuristic Search with Guarantees",
  NeurIPS

---

## 5. Coordinate solver hybrid

**One-liner**: Use Botz's nested-subgroup phase tables as an admissible
lower-bound heuristic for IDA* / weighted-A* search on TAIL segments
(length 20-50), with V as tie-breaker.

**Why genuinely different**: V is a noisy walk-depth regressor; phase-1
cost is an EXACT integer lower bound on remaining moves. Different signal
class. Beam has no admissibility lever; A* / IDA* exploit it directly.
This is the cleanest non-(model + beam) paradigm in the queue.

**Status**: PDBs and IDA* both on long-list. Botz integration not scoped.
T2.4 Schreier-Sims-Minkwitz adjacent.

**Reference solver**: Alexander Botz's solver at
<https://git.rwth-aachen.de/alexander.botz/megaminx-solver-v2.0> (TU
Darmstadt thesis). Reported avg ~89-95 moves, multiple table-size variants.
HKHLR HPC variant claims **82 mean / 133 max / ~1 sec runtime at 80 GB
RAM** — UNVERIFIED, project-page-only, no peer-reviewed paper.

**Why "tail-only" framing matters**: full-puzzle optimal solver requires
the largest table (2.19 billion states across phase 1+2, ~80 GB).
Tail-only (last 20-50 moves of an existing beam path) is tractable on 24
GB GCP L4 / 16 GB local 4090.

**Mechanism**:

1. Acquire Botz repo. Verify license + accessibility (may require RWTH
   login — UNVERIFIED).
2. Build smallest phase-1 table: all states reachable from solved by 2-gen
   `<U, R>` moves up to depth 24 (Whitmore + Rokicki proved 24 is God's
   number for the 2-gen subgroup).
3. Wrap as `phase1_lower_bound(state) -> int` callable.
4. Two integrations, A/B-able:
   - **A. Beam re-rank**: `score = V(child) + alpha * phase1_lower_bound(child)`.
     Alpha calibrates the two scales (integer LB vs float V). No A* yet.
     Tests whether the signal is *additive* to V before paying for A*.
   - **B. IDA* tail solver**: when current beam path's last K=30 moves'
     starting state has `phase1_lb <= 25`, hand off to IDA* with phase1_lb
     as heuristic.
5. V kept as tie-breaker among IDA*-equivalent paths.

**Why integration A first**: cheaper to test (no new search algorithm) and
gates whether the LB signal is informative at all. If A doesn't move the
needle, B won't either.

**Expected impact**: full integration -5% to -15%. The 82-mean HKHLR claim
is the unverified ceiling if we replicate their 80 GB setup.

**Falsifiable tests**:

- **Day 1**: verify Botz repo accessibility. If gated, fallback to
  building our own phase-1 BFS table directly (the 2-gen `<U, R>` group has
  only ~10^9 states; ~2h on 4090 with `bfs_bytes` format).
- **Day 2-3**: build smallest table, validate against Botz's reported
  per-state distance for known cases.
- **Day 4-5**: integration A (beam re-rank); strat-5 eval.
- **Day 6-10**: integration B (IDA* tail); strat-5 + stratified-200
  long-tail eval.

**Engineering**: 1-2 weeks. New `scripts/43_botz_phase1_table.py` (or
homegrown), integration in `src/cayley/khoruzhii_search.py`, new
`src/cayley/ida_star.py`. Existing `src/cayley/pdb_lookup.py:26-65` has
single-PDB scaffolding to extend.

**Risk**:

- Botz repo accessibility (UNVERIFIED).
- Alpha calibration: integer LB vs float V — needs scale-matching;
  alpha=1 may be wrong by orders of magnitude. Initial sweep
  `alpha in {0.1, 0.5, 1, 2, 5}`.
- Tail-only handoff condition needs tuning (where does beam stop, IDA*
  start?).
- IDA* may explode if heuristic isn't tight enough — phase 1 alone
  gives a weak LB; phase 1 + 2 jointly is much tighter but needs more
  RAM.

**References**:

- Botz repo: <https://git.rwth-aachen.de/alexander.botz/megaminx-solver-v2.0>
- HKHLR project: <https://www.hkhlr.de/en/projects/4006>
- SpeedSolving thread: <https://www.speedsolving.com/threads/development-of-a-megaminx-solver.93202/>
- Whitmore + Rokicki God's number 24 for `<U, R>`:
  <https://www.speedsolving.com/threads/gods-algorithm-for-megaminx-u-r.93480/>

---

## 6. ReduceFactor DAG path optimization

**One-liner**: Build a DAG over candidate window-replacements for an
existing path; Dijkstra finds globally-optimal combined shortcuts. Replaces
current greedy left-to-right window post-processing.

**Why genuinely useful**: current `full_post_process(...)` greedily
replaces each window from left to right with the shortest BFS-d6
equivalent. This misses combined improvements: a 4-move window might
shrink only when an adjacent 6-move window also shrinks (because together
they form a longer subsequence with a much shorter equivalent). DAG +
Dijkstra finds these globally.

**Status**: QUEUED as E6 / item 14 in `IDEAS.md`. Under-scoped — exists as
a one-line entry, never specified concretely.

**Mechanism**:

1. For path `P = [m_1, m_2, ..., m_n]`, build DAG `G` with nodes
   `0..n` (positions in path).
2. Edge `(i, j)` exists for each window length `j - i in {2, 3, ..., 8}`
   where the window's permutation has a known shorter equivalent.
3. Edge weight = length of the shorter equivalent.
4. Dictionary of shorter equivalents:
   - BFS-d6 table (19.4M states) — already shipped at
     `data/bfs_bytes_d6.pkl`
   - Curated macro library (Idea 2)
   - Symmetry-conjugates of both
5. Dijkstra from 0 to n on G; reconstruct shortened path.

**Why different from current pp**: current pp is one-shot greedy
left-to-right with window size W in {4..7}. DAG version considers ALL
windows of all sizes simultaneously and picks the globally optimal cover.

**Expected impact**: -500 to -1500 moves on full-1001 (additive on top of
T2.2 + T1.6). The 0-match brute-force commutator window finding
(2026-04-30) suggests pure-commutator equivalents won't help, but BFS-d6 +
curated macros are different — and macros + BFS-d6 jointly cover a much
larger replacement vocabulary than either alone.

**Falsifiable test**: implement on current 79,522 submission. Run on full
1001. Acceptance: any positive savings count (mechanism is
admissibility-preserving, can never regress per-path).

**Engineering**: ~3-5 days. New `scripts/45_reduce_factor_dag.py`. Reuses
`data/bfs_bytes_d6.pkl` and (eventually) the macro library from Idea 2.

**Risk**: graph size — for n=80 path with windows up to 8, ~8 x 80 = 640
candidate edges per path. Dijkstra is `O(E log V)` = trivial. The
dictionary lookup is the bottleneck; needs efficient
subsequence-permutation hashing. The bytes-keyed BFS table is already
suitable.

---

## 7. Path relinking across solutions

**One-liner**: For pids with multiple valid solutions (from v16/v17/v19a/v19b
sweep diversity), perturb path_A toward path_B at random crossing positions,
SA-accept on length.

**Why genuinely novel**: NOT QUEUED. T1.6 SA operates on a single path; this
is *cross-solution* diversity. Path relinking is a classical metaheuristic
from operations research (Glover 1997, "Tabu Search and Adaptive Memory
Programming"). We have multi-sweep solutions sitting on disk that we
currently only use for per-pid argmin merging — relinking would extract
more from the same data.

**Mechanism**:

1. For each pid, collect all valid solutions across submissions: path_A
   (best so far), path_B, path_C, ... (from v16, v17, v19a, v19b, T1.6 v1
   and v2, etc.).
2. For each pair (A, B) of solutions for a pid:
   - Find positions `i` where the prefix-states `apply(A[:i])` and
     `apply(B[:i])` are within hamming distance <= 6.
   - Define mixed candidate:
     `A[:i] + bridge_path(state_at_A[:i], state_at_B[:i]) + B[i:]`
     where `bridge_path` is BFS-d6 lookup.
   - If shorter than `min(|A|, |B|)`, accept.
3. Iterate over all pairs; take the global min.

**Why this should help**: search-config diversity from multi-sweep showed
-2.8% gains at zero training cost (TPU v16 union v17 union v19a union
v19b). That's per-pid argmin only. Path relinking is the per-pid
INTERPOLATION — should extract more from the same data because A and B
often diverge then re-converge, and the bridge-then-suffix construction
exploits the convergent structure.

**Expected impact**: -500 to -1500 moves on full-1001. Bounded by quality
of the bridge_path subroutine and how often A/B prefix-states fall within
hamming-6.

**Falsifiable test**: implement on top 50 long-pids of current 79,522
submission. If 0 wins, relinking doesn't compose with multi-sweep
diversity (different mechanism from path-min combining).

**Engineering**: ~3-4 days. New `scripts/46_path_relink.py`. Reuses BFS-d6
table for bridges.

**Risk**:

- Bridge construction may dominate runtime (BFS lookup per crossing
  position).
- State hamming-distance threshold needs calibration; too low gives no
  crossings, too high gives spurious bridges that lengthen the result.

**References**:

- Glover & Laguna 1997, "Tabu Search"
- Resende & Ribeiro 2010, "Path-Relinking for combinatorial optimization"

---

## 8. Twsearch integration

**One-liner**: Express megaminx as a proper KPuzzle (corners + edges +
orientations), wire `cubing/twsearch` for short-alg generation, prune-table
caching, and arbitrary-semigroup search.

**Why useful**: twsearch is Tomas Rokicki's general-purpose twisty-puzzle
search engine — same author as the cube20 work. Handles arbitrary KPuzzle
definitions, prune tables, IDA* with multiple heuristics. Designed for
exactly this problem class.

**Status**: long-list. Earlier flat-120-sticker attempt failed
(representation was wrong for the tool).

**Re-diagnosis of earlier failure**: twsearch expects piece coordinates
with permutation + orientation, not flat sticker permutations. Megaminx
KPuzzle definition exists in `cubing/cubing.js` — start from there, not
from our 120-element sticker representation.

**Mechanism**:

1. Verify cubing.js megaminx KPuzzle JSON corresponds to our 24-generator
   action set. (Likely needs a generator-renaming map.)
2. Validate move roundtrips: each of our 24 generators applied K times
   returns to starting state (for face turns, K = 5).
3. Generate small prune tables via twsearch CLI.
4. Use twsearch as solver for:
   - (a) short alg generation for macro library (Idea 2)
   - (b) tail solving (depth <= 30)
   - (c) subgoal transitions (e.g., corners-solved -> all-solved)

**Expected impact**: depends on what we ask twsearch to do. As a macro
generator: enriches Idea 1's action vocabulary. As a tail solver:
alternative to Idea 5's homegrown IDA*. Both are useful but neither
delivers a -10K savings alone.

**Engineering**: 1-2 weeks (mostly KPuzzle definition + integration glue).

**Risk**: HIGHER than Idea 5 because more dependencies (twsearch build,
KPuzzle JSON mapping, generator naming). Recommendation: pursue only if
Idea 5 yields signal AND we want a more general / capable solver back-end.

**References**:

- twsearch: <https://github.com/cubing/twsearch>
- cubing.js KPuzzle: <https://js.cubing.net/cubing/kpuzzle/>

---

## 9. Learned bidirectional / front-to-front

**One-liner**: Forward beam from scramble + backward beam from solved or
subgoal-shells; learn `D(s, t)` for distance between arbitrary pairs of
states; meet in the middle when forward and backward frontiers come close
under D.

**Why useful**: NISS and sym-ensemble both showed different representations
expose different paths — i.e., one-ended search has structural failure
modes. Two-ended search attacks them. Bidirectional heuristic-search
literature (MM by Holte et al. 2017, BAE* by Sturtevant, DBBS by Alcázar)
provides theoretical foundations.

**Status**: long-list (research-tier).

**Mechanism**:

1. Train D(s, t): pairs sampled from random walks of length 2k, target is
   k. ResMLP with concatenated input or siamese-tower.
2. Forward: beam from scramble using V toward solved.
3. Backward: beam from solved (or BFS-d6 shells) using V toward scramble
   (with inverted action set).
4. Meet condition: `D(forward_state, backward_state) <= threshold`.
5. Concatenate forward path + bridge + reversed backward path.

**Risk**: D(s, t) is fundamentally a harder regression than V(s) — the
symmetric s<->t axis doubles the noise dimensions. We're already
cluster-ceiling-bound on V(s); D(s, t) likely worse without an
architectural breakthrough. Multi-week with high uncertainty.

**Defer**: only after Tier 1 and Tier 2 ideas exhaust.

**References**:

- Holte et al. 2017, "MM: A bidirectional search algorithm that is
  guaranteed to meet in the middle"
- Sturtevant et al. 2020, "BAE*"

---

## 10. Tiny per-phase ML specialist

**One-liner**: Train a sub-100MB MLP specialist for the 2-gen `<U, R>` tail
subgroup; use as tail-resolver after the main solver routes the puzzle to
2-gen.

**Why useful**: state space of `<U, R>` is ~10^9, not 10^68. Exact BFS
labels available (full BFS to depth 24 is feasible), no walk-depth noise,
no cluster ceiling. Specialist closes the last 20-30 moves with
near-optimal moves.

**Status**: NOT QUEUED. From research synthesis 2026-05-04.

**Mechanism**:

1. Generate full BFS table from solved over `<U, R>` moves (depth <= 24,
   ~10^9 states). Use bytes-keyed format like `bfs_bytes_d6.pkl`.
2. Train tiny MLP (1-2M params) to predict exact distance.
3. Deploy as tail-resolver: when current solver path's tail is in
   `<U, R>`-reachable shell, hand off the last K=30 moves.

**Expected impact**: -500 to -2000 moves. Gain bounded by % of pids whose
tails fall in `<U, R>`-reachable region (small but non-zero — many human
megaminx solving methods end with a 2-gen `<U, R>` finisher).

**Engineering**: 2-3 days (assuming BFS table fits in 16GB; may need
bytes-keyed format, ~10^9 states at 2 bytes/state = 2 GB).

**Distinct from Idea 5**: Idea 5 uses Botz's external tables as admissible
LB for any state. Idea 10 is a tiny ML model trained on exact labels for a
much smaller subgroup, used as a tail finisher only.

**Risk**: low. The `<U, R>` BFS has been done before publicly (Whitmore +
Rokicki); we just need to redo it in our representation. Falsification is
cheap.

---

## 11. Megaminx insertion-finder

**One-liner**: Build a megaminx-specific insertion-finder tool — for each
segment of an existing skeleton, search d<=6 commutators that reduce
remaining error.

**Why genuinely novel**: 3x3 FMC has Baiqiang Dong's Insertion Finder;
megaminx has nothing equivalent. Research agent confirmed
(2026-05-04): no GitHub repo, no published tool. cubing.js KPuzzle
provides state-model scaffolding to build on.

**Status**: NOT QUEUED. First-of-kind tool.

**Mechanism**:

1. Take an existing skeleton (verified path with some moves).
2. At each gap position `i` between consecutive moves, search insertions
   from a commutator/macro library (length <= 8) that reduce the remaining
   permutation error.
3. Pick the best insertion globally (not greedy left-to-right).
4. Repeat until no insertion improves.

**Why insertions != T1.6 SA**: SA inserts macros at *random* positions and
keeps wins via HC. Insertion-finder searches *systematically* over all
positions x all macros, picks global best. T1.6 finds local minima; IF
finds globally-best single insertion per pass.

**Expected impact**: -5% to -15%. The 3x3 FMC analog is famously powerful
— top FMC solvers depend on it.

**Engineering**: 1-2 weeks. Requires Idea 2 (curated macro library) as
input.

**Defer rationale**: only meaningful AFTER macro library exists. Combine
downstream with Ideas 1 + 2.

**References**:

- 3x3 Insertion Finder: <https://github.com/baiqiangdong/Insertion-Finder>
  (analog reference; not megaminx)
- cubing.js KPuzzle: <https://js.cubing.net/cubing/kpuzzle/>

---

## Recommended starting points

Three independent spikes, ordered by expected leverage. A and B can run on
different machines; C requires Botz repo verification first.

**Spike A (~1 week, biggest single bet — combines ideas 1 + 2)**

Scrape SpeedCubeDB to ~100 curated macros -> train Macro-Q-shortlist (m24)
-> evaluate on stratified-5 with macro-augmented beam. Falsifiable in 1
training cycle. The "different action space" lever and the cleanest path
to the 70K target alone.

**Spike B (~3 days, parallel-track training)**

B9 frontier replay (Idea 3). Cheapest training-side experiment with a
data-bound-vs-arch-bound hypothesis. Single eval gates whether the B9/B10
cluster is worth pursuing further.

**Spike C (~1 week, the genuine "not model + beam")**

Pull Botz's smallest phase table (Idea 5 partial — integration A only);
wire as additional feature into V scoring; A/B vs current beam on
stratified-51. Tests whether admissible LB signal is *additive* to V. If
additive, scope full IDA* tail solver (Idea 5 full).

---

## Queue status reference

Items above mapped to existing `to_do_shortlist.md` and `IDEAS.md`:

| Idea | Existing queue ref | Status |
| --- | --- | --- |
| 1 Macro-Q shortlist | none | NOVEL, not on queue |
| 2 Curated macros in beam | T1.1 | queued, data missing |
| 3 Frontier replay | B9 | queued, never run |
| 4 Policy-guided V + pi | B10 | queued, never run |
| 5 Coordinate solver hybrid | PDB long-list, T2.4 partial | queued, not scoped |
| 6 ReduceFactor DAG | E6 / item 14 | queued, under-scoped |
| 7 Path relinking | none | NOVEL, not on queue |
| 8 Twsearch | long-list | queued, prior failure |
| 9 Bidirectional D(s, t) | long-list | queued, research-tier |
| 10 Per-phase specialist | none | NOVEL, not on queue |
| 11 Insertion-finder | none | NOVEL, not on queue |

When promoting any item to active execution, move the section to
`to_do_shortlist.md` and remove it from this file.
