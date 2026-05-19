# Next-phase experiment plan (2026-05-04)

**State of play**: 79,522 (#1 LB, gap to next 405). Training-side track closed
(16+ recipe variants, 0 wins). Path to <70K runs through ensembling, search-space
changes, and additive post-processing.

**Active runs**: m42 lower-q25 full-1001 on GCP (~10h, ensemble fodder).
**Idle**: local 4090, Kaggle GPU slot.

User-selected next experiments: **T2.4 Minkwitz**, **T1.3 (TTT v2)**, **T1.4
(solver-trace MIXIN proper)**. Plan below ranked by EV per engineering hour and
sequenced for parallelism.

---

## Experiment 1: T1.4 — m43 = standard Bellman + 25% solver-trace mixin

**Effort**: ~1 day code + ~8h train + ~80 min strat-5 eval. **Total**: ~24h.
**Expected**: 50/51 / 91-94 cluster (high prior of cluster); breakthrough is unlikely
but possible. Even cluster output = ensemble member.
**Source of strength**: solver-trace data is EXACT distances; 25% of each batch
comes from these pairs, anchoring the V regression to ground truth on the path
states. Distinct from m37 (which used solver-trace as PRIMARY signal and went
catastrophic OOD) — here it's a 25% MIXIN that supplements the standard
walk-depth Bellman.
**Distinct from m22 family**: m22 mixed in BFS-d6 (d≤6 only). Solver-trace
covers d up to ~150 (the longest paths in submissions), so it anchors the
HARD-TAIL distribution that BFS-d6 doesn't reach.

**Implementation**:
1. Add `solver_trace_path` and `solver_trace_fraction` fields to `BellmanConfig`.
2. In the training loop, sample `solver_trace_fraction * batch_size` pairs from
   `data/solver_trace_train.pt` (128,266 pairs already mined). Mix with the
   `(1 - solver_trace_fraction) * batch_size` walk-depth Bellman samples.
3. Concat states + targets per batch. Standard MSE.
4. Pattern follows existing `bfs_d6_path` / `bfs_d6_fraction` mixin code in `bellman.py`.

**Config** (`m43_solver_trace_mixin.yaml`): same as m05_bellman_warm with
`solver_trace_fraction: 0.25`.

**Run on**: Kaggle P100 (8h, fits within 12h kill limit). Same kernel template as m37.

**Acceptance gate**: standard 50/51 + mean ≤84.9. If passes → big result.
If lands at cluster, still log it as ensemble member.

---

## Experiment 2: T2.4 — Minkwitz / Schreier-Sims-Minkwitz classical fallback

**Effort**: 3-5 days port + ~1h to run on all 1001 + post-shorten.
**Expected**: -50 to -500 moves via min-merge (entirely additive).
**Risk**: low (cannot regress; only swaps shorter classical paths in via min-merge).

**Mechanism**:
1. Construct the megaminx group from generators using `sympy.combinatorics.PermutationGroup`.
2. Apply Schreier-Sims-Minkwitz to factor each test scramble into a sequence
   of generators. The factorization length is determined by the SGS depth
   (typically 100-150 moves for megaminx — slightly LONGER than our beam paths
   on average, but with different per-pid distribution).
3. For each pid, take min(beam_path, minkwitz_path) → submission.

**Why I expect it to find at least SOME wins**:
- Beam search is heuristic; it can get stuck on a suboptimal child early
- Classical SGS construction is deterministic; for some pids, its factorization
  happens to be 5-30 moves shorter than the beam's
- HKHLR's 82-move avg matches our 82.4 — so the AVERAGE is close. The TAIL
  distributions differ → per-pid min-merge picks up wins on specific puzzles.

**Implementation skeleton**:
```python
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.named_groups import SymmetricGroup
# Build generator perms
gens = [Permutation(p) for p in puzzle.generators.values()]
G = PermutationGroup(gens)
# Schreier-Sims base + strong generating set
base = G.base
sgs = G.strong_gens
# Factor a scramble into generators
scramble_perm = Permutation(scrambled_state) * Permutation(solved_state).inverse()
factored = G.coset_factor(scramble_perm)  # returns a list of group elements
# Convert back to generator names
path = [our_generator_name(g) for g in factored]
```

**Risks specific to Megaminx**:
- The G.coset_factor in sympy may produce LONG factorizations (100s of moves)
- Need Minkwitz's ALGORITHM (smarter SGS) for short factorizations — ported
  from Knuth or Holt's GAP code, more work.
- Public Kaggle Santa 2023 community has Minkwitz implementations to crib from.

**Run on**: local CPU-only (sympy is CPU; doesn't need GPU). Doesn't compete
with m42 GCP run or any GPU training.

---

## Experiment 3: T1.3 v2 — TTT with self-distilled labels (DAGGER-style)

**Effort**: ~half-day code + ~80 min eval per variant.
**Expected**: marginal (today's TTT v1 was null; v2 changes the LABEL SOURCE
which is the actual binding constraint).
**Risk**: similar to v1 — could regress slightly, could marginally improve.

**Why v1 was weak**: we used Bellman bootstrap on m05 itself for local targets.
m05 is already approximately self-consistent → tiny gradient signal → no real
shift in beam behavior.

**v2 mechanism — "self-distillation"**:
1. Run a CHEAP first-pass beam (e.g., beam=8k, max_steps=60) on the test puzzle.
   This gives a tentative path of length L_tentative.
2. If solved: every state along the path has a STRICT upper-bound label
   `remaining_path_length`. These are TIGHT (much tighter than walk-depth).
3. Fine-tune V on these path states with their true remaining-distance labels.
4. Run a FULL beam (beam=65k) with locally-tuned V → typically shorter path.

**Why it should work**: solver-trace mining (T1.4) globalized this same idea.
T1.3 v2 localizes it: per-puzzle, per-test-state, the model gets EXACT labels
along its own first-pass solution. No bootstrap, no walk-depth noise.

**Variant 2** (if v2 works): aggregate across puzzles — collect all first-pass
trajectories, mix into next training pass. That's just T1.4 again, but with
per-puzzle augmentation.

**Implementation skeleton**:
```python
def ttt_self_distill(puzzle, model, target_model, solver, test_state, ...):
    snapshot = {k: v.clone() for k, v in model.state_dict().items()}
    # 1. Cheap first pass
    cfg_quick = KhoruzhiiSearchConfig(beam_width=8192, num_steps=60, num_attempts=1)
    found, _, raw = solver.solve(test_state, cfg_quick)
    if not found:
        # Fall back to v1 (Bellman-bootstrap walks from test_state)
        return ttt_v1(...)
    path = full_post_process(raw)
    # 2. Build labels from the path
    states_along = []
    labels_along = []
    cur_state = test_state
    L = len(path)
    states_along.append(torch.tensor(cur_state, ...))
    labels_along.append(L)  # remaining distance
    for i, move_name in enumerate(path):
        cur_state = puzzle.apply_move(cur_state, move_name)
        states_along.append(torch.tensor(cur_state, ...))
        labels_along.append(L - i - 1)
    # 3. Fine-tune
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(50):
        # Sample mini-batch from states_along
        ...
    # 4. Full beam
    cfg_full = KhoruzhiiSearchConfig(beam_width=65536, num_steps=150, num_attempts=1)
    found, _, raw_full = solver.solve(test_state, cfg_full)
    # 5. Restore weights
    model.load_state_dict(snapshot)
    return shorter_of(path, post_process(raw_full))
```

**Run on**: local 4090 (after m42 GCP done releases GCP).

---

## Sequencing

| When | Where | What |
|---|---|---|
| Now | GCP L4 | m42 lower-q25 full-1001 (running) |
| Now (parallel) | Local 4090 | Start coding T1.4 mixin (`bellman.py` mixin field + `m43_solver_trace_mixin.yaml`) |
| Tonight | Local CPU | Start prototyping T2.4 Minkwitz with sympy (no GPU contention) |
| Tomorrow AM | Kaggle GPU | Launch m43 training (~8h on P100) |
| Tomorrow PM | Local 4090 | T1.3 v2 self-distill code + eval (after m42 GCP done) |
| Day +2 | Multi | m43 strat-5 + T2.4 minkwitz on full-1001 (CPU) + m42 ensemble min-merge result |

## Decision branches

- **m42 ensemble adds ≥100 moves**: keep doing this for every cluster-V we've trained (m38, m39a, m40 also have checkpoints). Cumulative grinding.
- **T1.4 mixin breaks the gate (51+/51 mean ≤84.9)**: BIG. Re-evaluate everything.
  Most likely cluster though.
- **T2.4 Minkwitz finds wins**: depends on per-pid distribution. Even -50 moves
  is a free win since it's additive and the implementation is one-time.
- **T1.3 v2 self-distill works**: enables per-puzzle quality gain. Could be
  worth full-1001 deployment.

---

## What this plan does NOT include

- **T1.1 curated speedcubing macros**: separate multi-day scoping. Highest
  theoretical impact (-10K to -20K) but engineering-heavy.
- **T1.2 multi-agent ensemble**: deferred per "saved-for-last big bets" policy.
  Could start once cluster-V members accumulate (we already have m05, m17, m26,
  m26b, m29, m38, m39a, m40, m42 — that's 9 V models for ensembling).

These are higher-impact but higher-effort. The current plan is "incremental wins
from mostly-already-done work" while we plan the bigger bets.
