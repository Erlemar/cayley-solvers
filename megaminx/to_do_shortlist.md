# Megaminx to-do shortlist

Active items only. Remove an item when it's done.

**Goal: total moves below 70,000. Stretch 60,000.** (Current best 75,200 → 70K
is −6.9% moves, a heuristic gain. Evaluate every item against this anchor:
does it move us toward 70K, or just save GPU hours?)

---

## State of play (2026-05-18)

- **Best submission: 75,200** (Kaggle submitted ✓ on 2026-05-18 as user-authorized policy exception — min-merge of our 75,961 base + new community v4 CSV (75,355 standalone). 290 v4 wins beat our base + 71 base wins kept, −761 moves. File: `merge_v14_plus_min_count_v4.csv`).
- **Previous: 75,961** (Kaggle submitted ✓ on 2026-05-18 as user-authorized policy exception — 3-way min-merge of AZ v4 prod-1001 (79,606 standalone, 134h GCP L4) + our 77,877 + 76,251 community-best: 114 unique AZ v4 wins concentrated in mid buckets 4-7, −290 moves. File: `merge_v13_az_v4_plus_community.csv`).
- **Best standalone work: 77,214** (our pipeline only, post AZ v4 GCP run). Path: 77,877 (m_dd_v0 ∪ m_curr_v3+m_pi_v2) min-merged with AZ v4 prod-1001 (79,606) → -663 moves. New file: `merge_v12_az_v4_plus_our.csv`.
- **Previous: 76,251** (2026-05-12). Min-merge of 76,304 community-best + m_dd_v0_50ep_prod_1001 (84,132 standalone): 20 unique wins, −53 moves.
- **Previous: 78,158** (2026-05-08). Stack: m_fr_v2 V teacher + m_pi_v1 policy. Saved −250 vs 78,408.
- **Older: 78,408** (2026-05-05). Stack:
  79,522 ∪ community-pushed shareable-kernel runs (alexandervc + fedmug forks of
  `cayleypy-megaminx-beam-shareable`, B=1M with K=8 long-tail at pids 800-900);
  324 wins distributed across all buckets. Saved −1,114 vs 79,522 base.
- **TPU-only score (4-sweep union)**: 82,342 — same single model (m05+m23_v2) at
  different K/B configs (v16 K=4 B=131k, v17 K=8 B=262k, v19a-fix K=4 B=1M,
  v19b K=4 B=1M), per-pid min. Better than single-sweep's 84,750. Search-config
  diversity is real even with same underlying model.
- **Best single-model full-1001**: 84,750 (GCP m05 + m23 + 524k + TRT FP16,
  2026-04-30). Gap from single-model floor → submitted best is ~3,400 moves,
  almost entirely from rescue/post-processing layers (sym-ensemble + tail
  re-solve + SA + macro_insert).
- **Production stack**: m05 + m23_v2 (sym-aware Q-shortlister) + `--sym-ensemble K=4`
  + qshort + beam 524k (rescue) or 65k (strat). TRT FP16 engine on GCP for
  full-1001 (sm_89-specific).
- **NEW post-processing layers (shipped 2026-05-02)**:
  - `T2.2 tail re-solve` (`scripts/26_tail_resolve.py`): truncate last K, re-solve
    with beam, take min. Cannot regress. −701 on full-1001 (333/1001 wins).
  - `T1.6 SA local search` (`scripts/27_path_sa.py`): HC over 3 operators —
    `tail_resolve` (random position), `macro_insert` (FMC-style commutator
    insertion + suffix re-solve), `commuting_swap` (cheap, mostly useless alone).
    −363 on top-100 (84/100 wins). 64 wins NEW vs T2.2.
  - **Operator efficacy** (worth keeping): `tail_resolve` 16.1% accept rate,
    `macro_insert` 7.8%, `commuting_swap` 0.2%. Drop commuting_swap in any v2.
- **Cluster ceiling at 6M params confirmed thoroughly** — every recipe variant
  (m17, m22, m26, m27, m28, m29, m30, m31, m32, SWA) lands in 88-97 strat-5
  mean. **Sym-ensemble at inference** is the only mechanism we've found that
  meaningfully breaks sub-89 (m05+sym4 → 88.20).
- See `SUBMISSION_JOURNEY.md` for the full submission-by-submission story.

---

## A. Highest-EV next moves (saved-for-last big bets)

These are the score-race-defining items. The user's stated discipline:
"we know that ensemble will improve scores - so let's leave ensembling to
later, when we run out of ideas."

1. **Full-1001 sym-ensemble K=2 or K=4 + qshort + 524k + TRT on GCP.**
   Mechanism validated at strat-5 (88.20 mean for K=4 no-qshort, 88.41 with
   qshort+m23_v2). Just scaling to full 1001 pids. Cost: K=2 ≈ 20-30h GCP,
   K=4 ≈ 40-60h. Expected: -1500 to -3000 moves. The mega-bet most likely
   to crack 80K alone.

2. **Multi-seed beam ensemble at full-1001.** Different RNG hash_vec
   seeds → different beam trajectories. Orthogonal to sym-ensemble's
   diversity. Cost ~3× a full-1001. Already proven on hard-tail rescues.

3. **T1.1 curated speedcubing macros.** Mechanism shipped (see
   `KhoruzhiiSolver(macros=...)` in `cayley/khoruzhii_search.py`). Brute-force
   d=4 commutators don't help. Need: scrape ~100 named macros from
   speedsolving.com / cubingdb.com / jPerm, map to our generator notation,
   validate. Effort 3-5 days. Expected −10K to −20K (per tier doc).

## B. Moderate ROI (untried, individually -200 to -1000 moves expected)

4. **More aggressive beam-stack rescue.** Currently triggered only on 2
   catastrophic failures (pid 490, 920). Sweep all model failures from
   merged full-1001, run beam-stack on each. Cost ~5min/failure × ~30
   failures = 2.5h. Expected -100 to -300.

5. **Tail-rescue iteration on 82,481.** After today's K=8 micro-test on top
   20, the new long-tail of 82,481 has fresh pids. Another K=4 sym +
   qshort + 524k pass (not yet done) on top 50 long-tail pids. Cost ~3-4h.
   Per A2.2 pattern: -300 to -700.

6. **NISS retry on m05 alone (no qshort).** A3 tried NISS+qshort+m23 and
   regressed because m23 has lower recall on inverted states. NISS without
   qshort is 2× wall but might work — never tested with the m05+sym
   ensemble stack. Cost ~6h strat-5.

## C. Defensive / diagnostic (low expected gain, cluster ceiling already known)

7. **B7 — Per-depth target shrinkage diagnostic.** ~15 lines.
   Histogram `V(s) - V_target(s)` per walk-depth bucket per epoch. Tells
   us whether bias is the binding mechanism. Run before any further bias
   mitigation work. Low cost, informational only.

8. **B8 — BFS-d6 as Dirichlet boundary IN the Bellman target.** Replace
   `1 + min_a V_target` with EXACT distance when state is in d≤6 shell.
   Different from m27's pretraining mixin. ~20 lines. Untried.

9. **B9 — Beam-frontier replay (DAgger-style).** ~80 lines. Log frontier
   states from real `03_solve.py` runs, mix into Bellman epochs at ~25%.
   Fixes the *distribution* (RW vs beam), not the target.

10. **B10 — m24 V + π multi-task.** Script `11_train_policy_head.py`
    exists, never run. Beam scoring `V(child) + λ·(-log π(a|parent))`.

11. **B11 — cayleypy nbt/bfs walks A/B.** Tighter walk-distance labels
    paired with Bellman. Caveat: m21 (tightened target without Bellman)
    regressed.

12. **m33 lr=2e-4 Bellman variation.** Config ready
    (`configs/m33_lr2e4.yaml`). m32 (target_update=5) already regressed;
    m33 is the only un-tried B6 variant. Low expected value.

## D. Long-list / multi-week research (defer)

- **T1.2 multi-agent ensemble** (CayleyPy paper recipe — 8-30 diverse
  agents + selector). 1+ week.
- **Transformer at scale** — m18 converges but 9× slower per-wall. Local
  4090 viable but slow.
- **A* / IDA*** — admissible LB + best-first. Big code change. 1+ week.
- **Macro-Q shortlisting** — Q-head scores 2-6 move macros mined from
  solved paths.
- **Listwise rank loss training** (T2.3) — calibrate V on relative
  ordering, the actual signal beam uses.
- **PDB (Pattern Databases)** — Korf-style on Megaminx subsets.
- **Bidirectional with learned front-to-front scoring.**
- **ReduceFactor DAG post-processing** — non-greedy global shortcut
  combination. High code cost; window post-proc is saturated.

---

## Acceptance gates (binding)

- **New training-side model**: ≥+3 strat-5 solves AND mean ≤ 0.95 ×
  m05's 89.4 (≤84.9). Most variants fail this; cluster ceiling is real.
- **New inference-side mechanism**: any net improvement to total submission
  via min-merge counts, even small ones (sym-ensemble has been shipping
  in 165-2450 move increments per pass).
- **Submission**: `verify_submission` passes AND total < current best (82,481).

---

## Standing artifacts

- `models/m05_bellman_warm/epoch_0499.pt` — production V teacher (6M params).
- `models/m23_q_shortlister/epoch_0499.pt` — Q-shortlister (12.4M, distilled
  from m05 forward states only). Use with non-rotated solves.
- **`models/m23_v2_sym_aware/epoch_0499.pt`** — Q-shortlister with rotation
  augmentation. **Use INSTEAD of m23 when paired with `--sym-ensemble`.**
- `data/rotations.npy` (360, 120) int8 — A_5 × C_6 group, megaminx symmetries.
- `data/bfs_bytes_d6.pkl` — 19.4M-state BFS at depth 6 (gitignored, 2.5 GB).
- `data/bfs_d6_train.pt` — same as tensor for training mixins (2.18 GB).
- `data/commutator_table.pkl` — 37K-entry commutator library (depths 4-8).
  Useful for future macro work; window-replacement post-proc gives 0 wins.
- `data/pp_bfs6_fallback.csv` — fallback path (414,678).
- TRT engine on GCP: `~/cayley/megaminx/models/m05_trt_fp16_b16384_sm89.ts`
  (sm_89-specific; rebuild for other GPUs).
