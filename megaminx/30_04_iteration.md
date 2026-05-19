# Megaminx iteration — 2026-04-30 → 2026-05-01

Working document for the next ~1 week of work. Builds on
`tier1-tier2-tier3-merged.md`; this file extracts the items we're
actually shipping and orders them.

## Session log (this iteration)

### 2026-04-30 — pivot to layers 1-3 + day-0 prep
- ✅ **HKHLR claim verified, gap is essentially zero** (their 82 avg vs
  our 82.40 avg → max −400 moves with min-merge). T2.4 downgraded
  T2 → T3.
- ✅ **T2.1 simple form (identity-word IF) confirmed redundant** with
  BFS-d6 windows: 0 identity sub-paths in current submission at any
  W ∈ {4,6,8,10,12}. Pivoted to T2.2 + T1.6 in the same priority slot.
- ✅ **Soft-Bellman temperature + Polyak/EMA target update** shipped to
  `src/cayley/bellman.py`. New BellmanConfig fields:
  `softmin_temperature` (default 0.0 = hard min) and
  `target_polyak_tau` (default 0.0 = discrete refresh). Smoke-tested.
  Activates next Bellman retrain via config.
- ✅ **Inversion-pair check**: 0 pairs in test set. Defensive.
- ✅ **T2.2 tail re-solve scout** at beam 131k on 30 mixed pids
  (top 10 + rank 100-119): 14 of 30 improved, **−24 moves total** in
  26.9 min walltime. Top-10 pids gave less per-pid than mid-range
  (recent rescue saturated them).

### 2026-05-01 — full-1001 sweep + T1.6 prototype
- 🔄 **T2.2 full-1001 sweep launched** in background: beam 131k,
  K=[20,40], expected wall ~15h, expected gain extrapolated to
  −500 to −800 moves. Output: `submissions/tail_resolve_full1001.csv`.
- ✅ **T1.6 SA/LAHC prototype built** (`megaminx/scripts/27_path_sa.py`):
  three operators with weighted random selection.
  - **CommutingSwap** (cheap, length-preserving): smoke-tested on top 5
    + 10 mid-range pids, both rescued (82,481) and unrescued (88,195)
    baselines. **0 improvements** in either case — BFS-d6 + same-face
    fixpoint absorbs all length-1 reductions reachable through swap-
    chains. Useful as cheap exploration only; the value-add must come
    from the expensive operators.
  - **TailResolve** (expensive, can shorten): generalization of T2.2 to
    random truncation positions in `[N//3, N-K_min]`. Same beam-solver
    pattern as T2.2; structurally proven at scale.
  - **MacroInsert** (expensive, FMC-style, NOVEL vs T2.2): insert a
    random commutator from `data/commutator_table.pkl` (4,200 entries
    at depth 4-6) at a random position, re-solve the suffix. The
    inserted macro creates a non-identity residue the model must
    absorb; if the new residue is closer-to-solved than the natural
    mid-path state, the new suffix is shorter and total < N. Different
    mechanism from T2.2 (insertion CHANGES the residue rather than
    just truncating). Smoke-validated for macro loading (4,200 valid
    commutators) and apply-path correctness; full end-to-end run waits
    for T2.2 to free GPU.
  - Both expensive operators include final-pass `full_post_process` on
    the boundary join (prefix↔macro, prefix↔new-tail) to capture any
    cancellations the operator's local processing missed.
  - Acceptance: HC (delta < 0) by default, SA mode optional with
    linear cooling. Verifier guards every accepted change.
  - Decision needed when T2.2 finishes: run T1.6 on the T2.2 output
    (additive) vs on the original 82,481 (independent A/B).

---

---

## State of play

- **Current best**: 82,481 (Kaggle scored, 2026-04-30)
- **Goal**: total moves <70,000 (−15% from 82,481, ~12,500 moves)
- **Stretch**: 60,000
- **Production stack**: m05 V teacher + m23_v2 sym-aware Q-shortlister
  + `--sym-ensemble K=4` + qshort α=2 + beam 524k (rescue) / 65k
  (strat-5) + TRT FP16 on GCP
- **Cluster ceiling at 6M params**: confirmed (m17, m22, m26, m27 family,
  m28, m29, m30, m31, m32, SWA all land at strat-5 mean 88–97). Recipe-
  side levers exhausted.
- **What broke the ceiling**: sym-ensemble at inference (m05+sym4 →
  strat-5 mean 88.20). Inference-time diversity is doing real work.

---

## Strategic frame

Four orthogonal layers compose; we've pumped only one of them.

| Layer | What it changes | We've done | Untouched / partial |
|---|---|---|---|
| 1. Search-space | Action set / move semantics | (none) | T1.1 macros, **T2.1 insertion-finder**, T2.4 Minkwitz |
| 2. Post-processing | Path rewriting after solve | BFS-d6 windows | **T1.5 auto-rescue sweep**, **T1.6 SA/LAHC**, T2.2 tail re-solve |
| 3. Heuristic | What the V model knows | m05 Bellman warmstart | **T1.3 TTT**, **T1.4 trace mining**, soft-Bellman, Polyak, listwise rank, T2.5 distributional |
| 4. Inference compute | How we search at solve time | sym-ensemble, qshort, 524k, TRT | T1.2 multi-agent (deferred per policy) |

The 70K crossing plausibly needs 3 of 4 stacking, not one mega-lever.
This iteration moves into layers 1, 2, and 3.

---

## Immediate sequence (this session)

### 1. T2.1 status — confirmed redundant; pivoted to T2.2 + T1.6

**Empirical finding (2026-04-30)**: scanning the current best submission
(`merge_plus_sym8_top20.csv`) for contiguous sub-paths whose net
permutation is identity returns **zero matches** at every window size
W ∈ {4, 6, 8, 10, 12}. This means our existing BFS-d6 window-replacement
post-processing already removes every sub-path that evaluates to identity,
up to W=12. The IHES-style insertion-finder mechanism (identity-word
insertion + adjacent-inverse cancellation, which is what
`scripts/insertion_finder_prototype.py` implements) is therefore
**mathematically redundant** with what we already do.

The proper FMC insertion-finder mechanism is fundamentally different:
insert a *non-identity* commutator that rotates a residue (not solved)
into a permutation that combines with `M_post` more efficiently. That
requires a "skeleton + residue" structure beam paths don't have without
a separate residue-extraction step. Multi-day project; deferred.

Diagnostic scripts:
- `megaminx/scripts/24_count_identity_words.py` — enumerated 288
  depth-4 and 17,088 depth-6 non-trivial identity words on megaminx.
- `megaminx/scripts/25_check_identity_subpaths.py` — confirmed 0
  identity sub-paths in the current submission.

**T2.1 reframe**: keep in the long backlog as "post-T1.1 macros
project" — the curated macro library becomes the commutator candidate
set, and FMC IF is built on top with macros + tail-resolve.

### 1b. T2.2 — Tail re-solve post-processing [NEW PRIMARY]

**What**: For each pid's path `M` of length `N`, truncate the last `K`
moves and re-solve the resulting prefix-state with a fresh beam. If the
new tail is shorter than `K`, accept the shorter total path. Strict
additive post-processing — verifier guards correctness; cannot regress.

**Mechanism**:
1. Compute prefix state: `S_K = apply(initial_state, M[:N-K])`.
2. Run beam from `S_K` to find a new tail `M_tail`.
3. If `len(M_tail) < K`, accept `M' = M[:N-K] + M_tail`.
4. Iterate over `K ∈ {5, 10, 20, 40}`; take min across K and the
   original.

**Why this works** (and why it's distinct from BFS-d6 windows):
- BFS-d6 replaces sub-paths with their shortest equivalent for any perm
  reachable in ≤6 moves. Bounded by depth.
- Tail re-solve targets the *last* K-suffix specifically — where beam
  often converges suboptimally because the heuristic noise dominates
  near the solved state. A fresh beam (different RNG, possibly with
  sym-ensemble) tends to find shorter tails.
- The mechanism doesn't care about commutator structure or perm-table
  membership — it's pure "try a different finish".

**Effort**: ~1 day code + ~1-2 h compute.

**Expected gain**: −50 to −200 moves (per `IDEAS_CATALOG.md` E7 from
the IHES project). Could be larger on megaminx since paths are 3× longer
than IHES.

**Acceptance gate**: any net improvement on full-1001 vs 82,481.
Cannot regress (verifier guard).

### 1c. T1.6 — SA/LAHC local search on completed paths [NEW SECONDARY]

**What**: Simulated Annealing / Late-Acceptance Hill Climbing on
completed paths. Random perturbations: swap commuting adjacent moves,
replace short subsequences with same-perm shorter words, insert
identity-then-cancel-cascade attempts. SA acceptance criterion. Verifier
guards correctness.

**Effort**: ~3 days, ~150 LOC, ~90 min full-1001 compute.

**Expected gain**: −1K to −3K moves (1-3% TSP analog).

**Acceptance gate**: cannot regress; any improvement counts.

---

### 2. HKHLR 82-move claim — VERIFIED but GAP IS ZERO [DONE 2026-04-30]

**Source**: https://www.hkhlr.de/en/projects/4006

**Confirmed facts**:
- Algorithm: BFS + Schreier coset decomposition (Minkwitz-style)
- Variants: 102-twist avg (basic, gigabytes RAM) → **82-twist avg
  (optimized, 80 GB RAM, ~1s search)**
- God's number bounds: upper 133, conjectured 114
- **No public source code or downloads available**

**The gap math**:
- Our 82,481 / 1001 = **82.40** average
- HKHLR optimized variant = **82** average
- Gap: 0.4 moves/puzzle = max ~400 moves over full-1001 if perfectly
  replicated and `min()` helps on every puzzle (realistically 200–500)

**Implication**: T2.4 (Schreier-Sims-Minkwitz port) was estimated at
−1K to −5K *under the assumption HKHLR was meaningfully ahead of us*.
With our sym-ensemble production stack at 82.40 average, we've already
matched their result via a different mechanism. **T2.4 expected gain
collapses to −200 to −500 for 3–5 days of port effort.**

**Decision**: downgrade T2.4 from Tier 2 to Tier 3 / skip-list. Replace
in priority slot with T2.2 + T1.6 (already pivoted above).

**Lesson (worth recording)**: when a "category-different" lever's
absolute target equals our current performance, the per-puzzle min
upside is small. Always re-cost the EV when the baseline shifts —
this finding only became visible at 82,481, would have been a real
improvement at our prior 88,195.

---

### 3. DAY-1 #3 — Soft-Bellman temperature [QUICK CODE CHANGE]

**What**: Replace the hard `min_a V_target(s')` in the Bellman bootstrap
with a soft minimum: `−T·logsumexp(−V_target(s')/T)`. Anneal `T → 0`
recovers hard min.

**Why**: hard min is over-confident — it propagates the single-best
child estimate as if it's certain. Soft min averages across the action
distribution, smoothing target noise. This is the same trick that
distinguishes soft-Q-learning from regular Q-learning.

**Effort**: ~10 LOC. Edit `src/cayley/bellman.py`. Add config field
`bellman.softmin_temperature` (default 0.0 = hard min, preserves
backward-compat).

**Files to edit**:
- `src/cayley/bellman.py`: replace the min reduction in target
  computation
- `megaminx/configs/m33_lr2e4.yaml` and any future Bellman config:
  expose the new field

**When it activates**: next Bellman retrain.

**Expected gain**: bounded — won't break the cluster ceiling alone, but
0.1–0.5 mean-path improvement is plausible given the m32
target_update sensitivity we saw.

---

### 4. DAY-1 #4 — Polyak/EMA target update [QUICK CODE CHANGE]

**What**: Replace the hard target refresh every 10 epochs with a smooth
Polyak/EMA update: `target := τ·model + (1−τ)·target` per step
(τ ≈ 0.005, matches DQN literature).

**Why**: m32 (target_update_every=5) regressed (mean 94.65) because
faster discrete refresh over-fits short-term gradient noise. m05's
value of 10 was calibrated empirically. Polyak smooths out the
refresh entirely — should be strictly more stable and remove the
discrete-step-size hyperparameter.

**Effort**: ~10 LOC. Edit `src/cayley/bellman.py`. Add config field
`bellman.target_polyak_tau` (default `null` preserves discrete
behavior; non-null overrides).

**When it activates**: next Bellman retrain.

**Expected gain**: small standalone, but combines well with soft-min
(both are noise-reduction levers on the target).

---

### 5. Inversion-pair check on test set [DEFENSIVE]

**What**: Check if any pair of test pids `(i, j)` represents inverse
scrambles, i.e. `state(j) = invert_state(state(i))`. If they exist, we
can solve one and reverse the path for the other.

**Effort**: 15 min.

**Implementation**:
```python
import numpy as np, csv, sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'megaminx/src')
from megaminx.puzzle import Megaminx
p = Megaminx.load('megaminx/data/puzzle_info.json')
states = []
for row in csv.DictReader(open('megaminx/data/test.csv')):
    states.append(np.array(row['initial_state'].split('.'), dtype=np.int8))
inv_index = {tuple(p.invert_state(s).tolist()): i for i, s in enumerate(states)}
pairs = [(i, inv_index[tuple(s.tolist())]) for i, s in enumerate(states)
         if tuple(s.tolist()) in inv_index and inv_index[tuple(s.tolist())] != i]
print(f"inversion pairs found: {len(pairs)}")
```

**Expected**: almost certainly 0 pairs. Free defensive check.

---

## Next iterations (sketched)

### Days 1–3 — additive post-processing layers

- **T1.6 SA/LAHC local search on completed paths**. ~3 days, ~150
  LOC, ~90 min full-1001 compute. Cannot regress.
- **T1.5 aggressive beam-stack auto-rescue**. ~1 day code, sweep top-50
  longest pids of current submission. Cannot regress.

### Days 3–5 — heuristic data augmentation

- **T1.4 solver-trace mining**. Instrument `03_solve.py` to dump
  `(state, remaining_path_length)` pairs from successful solves;
  also mine all existing submissions in `submissions/*.csv`. Mix into
  Bellman at 10–25%. Distinct from beam-frontier replay (different
  failure mode).
- **T1.3 TTT integration (BFS-d6 exact-label variant)**. Per-puzzle
  5–10 gradient steps on `(s, BFS_d(s))` for states in local BFS-d6
  shell, then beam. ~3h code, ~50 min compute on full-1001.
- **Alexander's beam-frontier Bellman TTT** (proposed 2026-04-30 by
  Alexander C). Per-puzzle: solve with beam → save all beam frontier
  states → take K Bellman gradient steps on those states
  (target = `1 + min_a V_target(neighbor(s, a))`) → re-solve.
  - **Mechanism**: hybrid of T1.3 (per-puzzle TTT) and B9
    (beam-frontier replay, shortlist item 6), but uses Bellman
    bootstrap rather than exact BFS-d6 labels. No label source
    required — works on any state the beam visits.
  - **Why this is the right distribution**: T1.3's BFS-d6-label
    variant trains on `(s, exact_d(s))` for s in the d≤6 shell of the
    *scramble* — i.e. states close to solved. Alexander's variant
    trains on the states beam *actually ranks during search* —
    typically d=10–80 from solved. The objective↔inference mismatch
    is sharper there. Different failure-mode coverage.
  - **Implementation**: implement alongside T1.3; A/B compare on
    strat-5. Run online (per-puzzle reset of weights, take K small
    SGD steps, solve, restore weights for next puzzle). Use a frozen
    target net for the Bellman bootstrap (otherwise unstable +
    self-reinforcing on own mistakes).
  - **Risks**: (1) catastrophic forgetting — mitigated by per-puzzle
    weight reset and small step count (K≤10 at lr=1e-5);
    (2) training on own mistakes — mitigated by frozen target net;
    (3) cluster ceiling — m17/m26/m28/m29/m31/m32 all Bellman-
    variants and plateaued, but those used random-walk distributions
    not beam-frontier; this is a genuine distribution shift.
  - **Effort**: ~1 day code (need a per-puzzle weight checkpoint +
    restore loop in `03_solve.py`, frontier capture from
    `KhoruzhiiSolver`, mini Bellman trainer). Compute: K=10 steps × 24
    actions × 1001 pids ≈ 240K forwards, ~20 min on 4090 + the
    re-solve cost.

These three (T1.4, T1.3 BFS-label, Alexander beam-frontier-Bellman)
are *different label sources* hitting the same heuristic-quality
problem from different angles. Run all three; compare; combine the
winners in the Days 5–8 retrain.

### Days 5–8 — Bellman retrain combining the above

- New training config: m34 (or rename) combining
  - soft-Bellman temperature (annealed schedule)
  - Polyak target update (τ=0.005)
  - solver-trace mixin (10–25%)
  - listwise rank loss aux (T2.3, optional)
- Strat-5 gate: ≥+3 solves vs m05 AND mean ≤84.9 (0.95×89.4)
- If passes gate: re-train m23_v2 against new teacher, re-run
  full-1001 sym-ensemble

### Re-evaluate at ~75–78K target

If the post-processing + new-teacher stack lands us at 75–78K:
- **T1.1 curated speedcubing macros** (multi-day scrape) + T2.1
  insertion-finder using macros as candidates
- **T2.4 Minkwitz fallback** (if HKHLR verifies)
- **T2.5 distributional V head** (different mechanism than ListNet)

If we plateau:
- **T1.2 multi-agent ensemble** — held in reserve per user policy
  ("ensemble for last")

---

## Outdated items in tier1-tier2-tier3-merged.md

Flagged so we don't re-process them:

1. **DAY-1 #1 SWA m05/400-499**: file lists this as a free 30-min win.
   Already ran 2026-04-28: strat-5 51/51 / mean 96.75 (REGRESSES).
   Late-cycle ckpts diverged enough that averaging blurs the heuristic.
   See HANDOFF.md row "SWA m05 ckpts 399/449/499". Do NOT re-run.

2. **T1.2 cost numbers** (lines 86–107): assume 88,195 baseline with
   beam 131k as per-agent solve. Current production (sym4 + qshort +
   524k + TRT) is roughly 4–8× more expensive per agent, so the "8
   agents × 80 min training + ~84h beam" estimate scales to a
   meaningfully larger budget. Re-cost before launching multi-agent.

3. **T1.1 expected −10K to −20K** (line 72): aspirational. Brute-force
   d=4 commutators net +9 to +41 moves (HURT) on 5 hard pids. T1.1
   paying requires curated speedcubing macros AND macro-aware V
   training (or listwise rank). T2.1 insertion-finder is the cleaner
   first move because it doesn't require V re-training.

---

## Acceptance gates

Re-stating the binding gates (CLAUDE.md + HANDOFF.md):

- **Post-processing layer** (T2.1, T1.5, T1.6, T2.2): any net
  improvement on full-1001 vs 82,481. Cannot regress (verifier guard).
- **Training-side model** (m34+, soft-Bellman, Polyak, trace-mixin):
  ≥+3 strat-5 solves AND mean ≤ 0.95 × m05's 89.4 (≤84.9). Most
  variants fail this; cluster ceiling is real.
- **Submission**: `verify_submission` passes AND total < current best
  (82,481).
- **Inference-side mechanism**: any net improvement to total submission
  via min-merge counts, even small ones (sym-ensemble has been shipping
  in 165–2,450 move increments per pass).

---

## Standing artifacts (referenced by this iteration)

- `models/m05_bellman_warm/epoch_0499.pt` — production V teacher
- `models/m23_v2_sym_aware/epoch_0499.pt` — sym-aware Q-shortlister
- `data/rotations.npy` — 360 megaminx symmetries (A_5 × C_6)
- `data/bfs_bytes_d6.pkl` — 19.4M-state BFS at depth 6
- `data/bfs_d6_train.pt` — same as tensor for training mixins
- `data/commutator_table.pkl` — 37K-entry commutator library (depths
  4–8). USED BY T2.1 as insertion candidate set.
- `data/pp_bfs6_fallback.csv` — fallback (414,678)
- `submissions/merge_plus_sym8_top20.csv` — current best (82,481)
- TRT engine on GCP: `~/cayley/megaminx/models/m05_trt_fp16_b16384_sm89.ts`

---

## How this slots into the existing docs

- **`to_do_shortlist.md`**: this iteration consumes items from there
  (5, 4, 8 partial). Update on completion.
- **`tier1-tier2-tier3-merged.md`**: source for the items we picked.
- **`EXPERIMENTS.md`**: log each completed step as a new row.
- **`SUBMISSION_JOURNEY.md`**: append a new submission entry on every
  Kaggle-scored result.
- **`HANDOFF.md`**: update §2 (score progression) and §6 (rejected
  items) on completion of each phase.
