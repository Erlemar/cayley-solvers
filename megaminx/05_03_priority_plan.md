# Priority plan — T1/T2/T3 selection (2026-05-03)

**State of play**: 79,522 (#1 LB, gap to next 405; submitted 2026-05-03 evening).
**Goal**: <70,000 (need −9,522, ~12%). **Stretch**: <60,000.
**Active runs**: m37 strat-5 v3 (Kaggle, RUNNING), m39a (local 4090, pretrain 45%),
m40 (GCP L4, Bellman warmstart epoch 5).

User-selected items to plan: T1.3, T1.4, T1.6, T2.4, T2.5, T2.6, T3.3, T3.4,
T3.5, T3.7, T3.9.

## Two crosscutting filters

**Filter A — already-paid-for?** Eval-only items beat fresh experiments per hour of work.
**Filter B — orthogonal mechanism vs. cluster-ceiling family?** Cluster ceiling is
information-bound at this scale; another Bellman-recipe regularizer almost
certainly lands at 50/51 / 89 again (m38 listwise just did, on top of m17/m22/
m26/m26b/m27/m28/m29/m30/m31/m32/m34/m36 all at the same plateau). Items that
change the *signal source* (T3.3 oracle, T3.7 d=7) or the *inference loop*
(T1.3 TTT, T2.5 distributional, T2.4 Minkwitz) are more likely to actually move.

## Ranked plan (4 phases)

### Phase 1 — already-paid-for or sub-day wins (DO FIRST)

| # | Item | Why first | Effort | Status |
|---|---|---|---|---|
| 1 | **T1.4** (m37 result) | This IS T1.4. Result lands tonight; everything else's prioritization shifts on the answer. | 0 (in flight) | RUNNING on Kaggle |
| 2 | **T3.3 BFS-d6 oracle Q-head eval** | Task #14 marks training done. Untouched orthogonal mechanism (exact labels, no bootstrap). One of few items that could break ceiling. | ~2h eval | trained, not evaluated |
| 3 | **T1.3 TTT** | 3h code + 50 min full-1001 compute. Best EV/hr on the entire list. Untouched. Different mechanism (per-puzzle local fine-tune). | ~3h code + ~50 min eval | not started |

### Phase 2 — orthogonal-mechanism medium bets (1-day-each)

| # | Item | Why this priority | Effort |
|---|---|---|---|
| 4 | **T2.5 distributional V** (QR-DQN, 32 quantiles) | Different *inference* mechanism (not just training). Beam selects on lower quantile (optimistic preference for confident-close states). Plausibly orthogonal to cluster ceiling. | ~1 day code + ~8h retrain |
| 5 | **T2.4 Minkwitz / Schreier-Sims** | HKHLR doesn't beat us standalone (82 ≈ our 82.4 mean), but per-puzzle min-merge could pick up on puzzles where Minkwitz finds a shorter sequence. EV bounded by HKHLR result. | 3-5 days port + 1h run. **DEFER** unless heuristic track stalls. |

### Phase 3 — cluster-ceiling-likely Bellman variants, BUNDLE (1 retrain)

Don't run T2.6, T3.5, T3.9, T3.4 as separate Bellman experiments — m38 already
showed this class doesn't escape the ceiling individually. **Bundle into one
"mega-Bellman" run**:

| # | Item | Bundle role |
|---|---|---|
| 6 | **m41 = m05 + T2.6 PER + T3.5 per-piece aux + T3.9 cross-traj n-back + T3.4 adversarial mining** | One config, one training run. Tests whether the SUM of cluster-ceiling-family tweaks does what individual ones don't. If it still lands at 50/51 / 89, the family is exhausted. Saves 3× compute vs running them serially. | ~1 day code + ~8h training |

### Phase 4 — heavy infrastructure, conditional

| # | Item | Trigger condition | Effort |
|---|---|---|---|
| 7 | **T3.7 BFS-d7 partial shell** | Run only IF T3.3 oracle Q-head shows an improvement. d=7 (~250-500M states, ~30 GB disk) amplifies whatever d=6 demonstrated. | 1-2 days BFS + integration; needs GCP burst (32 GB+ memory) |

### T1.6 — not a new experiment

Already shipped in production (T1.6 v1 → 81,357; T1.6 v2 → 79,946). Further
runs are score-race grinding, not strategic. Schedule on idle GCP/local
between bigger experiments. Top-300 of 79,522 expected −200 to −400 in min-
merge, ~50h L4.

---

## Concrete sequencing

| When | Where | What |
|---|---|---|
| Tonight | Kaggle | m37 strat-5 v3 result lands; multi-seed V (m41 seed=51) on 2nd slot if user approves |
| Tomorrow AM | Local (after m39a + m40 free) | T3.3 oracle Q-head eval — 2h |
| Tomorrow PM | Local | T1.3 TTT implementation + full-1001 eval — ~4h |
| Day +2 | Local or GCP | m41 mega-Bellman bundle (T2.6 + T3.5 + T3.9 + T3.4) — 1 day code + 8h train |
| Week 2 | GCP | T2.5 distributional V — 1 day code + 8h train |
| Week 2 (background) | GCP idle | T1.6 v3 SA on top-300 of 79,522 (~50h) |
| Week 3+ | conditional | T2.4 Minkwitz, T3.7 BFS-d7 |

## What can change this plan

- **m37 PASSES gate** (51+/51 mean ≤84.9): the solver-trace primary signal works
  → bias hard toward T3.3 (also exact-label) and T3.7 (more exact labels). Skip
  Phase 3 mega-Bellman (signal-source > regularizers).
- **m37 FAILS gate**: information-bound theory holds → Phase 3 mega-Bellman
  is now an *informative negative* (one experiment confirms a whole family is
  exhausted, freeing us to focus on T1.3, T2.5, T2.4 inference-side levers).
- **m40 PASSES gate**: revisit the broader L_lip + L_anchor + μ_d composite
  proposal. Run as m40b extension before mega-Bellman.

---

## T1.3 TTT — design sketch

**Mechanism**: for each test puzzle, locally fine-tune V on a neighborhood
sampled around the puzzle's initial state, then beam-search the test puzzle
with the locally-tuned V. **Reset weights between puzzles** (no cross-puzzle
contamination).

**Signal**: Bellman target on locally-sampled states (target net = frozen
m05). The local fine-tune doesn't introduce new ground-truth labels — it
just makes the model more *Bellman-consistent* in the neighborhood of the
test state, which is exactly where beam search wastes its budget.

**Skeleton**:

```python
def ttt_solve(puzzle, model, beam_solver, test_state, k_steps=50, lr=1e-3,
              n_walks_per_state=24, walk_len=8):
    # 1. Snapshot weights (reset point).
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    target_model = copy.deepcopy(model).eval()  # frozen Bellman target

    # 2. Build neighborhood: random walks of length walk_len starting from test_state.
    states, depths = generate_walks_torch(
        puzzle, n_walks=n_walks_per_state, k_max=walk_len,
        seed=hash(test_state) & 0xFFFFFFFF, device='cuda', n_back=1,
        start_state=test_state,  # NEW: walks start FROM test_state, not solved
    )

    # 3. Fine-tune for k_steps.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(k_steps):
        idx = torch.randperm(len(states))[:1024]
        bs = states[idx]
        bd = depths[idx].float()
        target = bellman_targets(target_model, bs, bd, generators, solved_state, ...)
        pred = model(bs)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    # 4. Beam-solve test_state with locally-tuned V.
    found, _, raw = beam_solver.solve(test_state, cfg)
    path = full_post_process(raw)

    # 5. Reset weights for next puzzle.
    model.load_state_dict(snapshot)
    return path
```

**Implementation notes**:
- `start_state` arg on `generate_walks_torch` doesn't exist yet — small
  addition to `cayley/data.py`.
- Per-puzzle wall: 50 SGD steps × ~10ms = 0.5s on a 6M model + beam search
  cost (~3s). Total ~3.5s/puzzle × 1001 = ~60 min full eval on local 4090.
- Memory: snapshot is ~24 MB (model size). Cheap to copy/restore per puzzle.
- Hyperparameters worth varying: `k_steps` (50 vs 100), `walk_len` (8 vs 16),
  `lr` (1e-3 vs 5e-4). Default the cheapest first.

**Acceptance gate**: any net improvement to total submission via min-merge
counts (post-processing-style gate, not training-side). Drop-in to current
production stack.

**Risk**: TTT could *overfit* the local neighborhood and lose the global
heuristic quality the model needs to find solved. Mitigate by capping
`k_steps` and using a small `lr`. If the snapshot/reset doesn't fully restore
(e.g., optimizer state leaks), debug by comparing per-puzzle eval pre/post-
TTT on the same state.

**Eval plan**:
1. Implement on a 5-pid smoke test, verify per-puzzle path lengths drop or
   stay the same vs no-TTT baseline.
2. Strat-5 with TTT enabled — compare to m05 baseline (50/51 / 89.4). Expect
   tied or marginal improvement on easy buckets, bigger gains on hard tail.
3. Full-1001 eval if strat-5 doesn't regress.
4. Min-merge with current best (79,522) to determine actual contribution.

## Open questions

1. Should T3.3 oracle Q-head eval go AHEAD of T1.3, or in parallel? They use
   different code paths; T3.3 needs solver integration to use Q for ranking,
   T1.3 needs the TTT loop. Independent — do whichever frees first.
2. Does m37 result invalidate Phase 3? If m37 passes (solver-trace primary
   works), we'd want to retrain Q-shortlister against m37 (m23 v3) instead
   of running m41. New priority would be Q-shortlister against m37 first.
3. T2.4 Minkwitz: should we even bother given HKHLR floor matches us? Only
   if everything else stalls. Mark as conditional.

## What this plan does NOT include

- T1.1 curated speedcubing macros (deployment cost still load-bearing —
  separate scoping needed).
- T1.2 multi-agent ensemble (8h Kaggle slot per agent; quote: "saved-for-last
  big bets" per to_do_shortlist policy).
- T1.5 / T1.7 / T2.2 (already shipped in production stack).
