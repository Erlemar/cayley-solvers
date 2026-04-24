# Comprehensive idea catalog — IHES Picture Cube

Living reference doc covering every idea we've considered, tried, or confirmed negative,
organized by category with status, effort, and expected gain.

**Current submitted best (own-work only)**: `23,672` (submitted 2026-04-20, beam-524K+NISS+int8 tail re-solve on len≥26). Gap to Rokicki (21,840): **1,832 moves (7.7%)**.
**Local own-work ensemble (unsubmitted)**: ~23,672; beam-1M full solve running, first 100 puzzles already -100 vs E6 beam-65K. 
**Community min-merge floor**: `21,972` — our own models currently contribute 0 puzzles to this. User policy: no community-merge submissions until our own work beats the community on at least some puzzles.

**CRITICAL UNEXPLORED**: `khoruzhii/cayleypy-cube` (NeurIPS 2025 Spotlight) has full code + **pre-trained weights on Zenodo for Picture Cube 333**. 98.4% optimality on 3×3×3 with a 26-agent ensemble. We ported his beam searcher (`src/cayley/khoruzhii_search.py`) but **never downloaded his trained weights** — doing so could leapfrog our own training. Demo: qdiag.xyz/cube.

**Gap decomposition** (from synthesis v2): Source A heuristic ranking errors **(50% of gap)** → attack via training (N1 Bellman-aux, N6 CEA loss). Source B beam pruning (30%) → wider effective beam (A11 Q-func, A12 int8, N5 adaptive-beam). Source C no-backtracking (15%) → N3 NRPA or multi-solver ensemble. Source D PP ceiling (5%) → N4 Dijkstra/ReduceFactor.

Companion docs:
- `IDEAS.md` — prioritized backlog (shorter, actionable-next list)
- `DATA_AND_FEATURES.md` — focused deep-dive on data-side ideas
- `EXPERIMENTS.md` — chronological experiment log
- `README.md` — state of play, commands, gotchas

---

## A. Search / inference side

| # | Idea | Status | Effort | Est. gain | Notes |
|---|---|---|---|---|---|
| A1 | **NISS** (inverse scramble search) | DONE 2026-04-19 | done | -432 in ensemble | `scripts/06_niss_solve.py`. Core technique. |
| A2 | **Six-axis ensemble** — apply 6 whole-cube rotations before solving | UNTRIED | 2-3h | -1000 to -3000 | Extends NISS; needs rotation derivations (bundles with F2). |
| A3 | **Targeted wider beam** on long paths (beam 131K + NISS re-solve) | DONE 2026-04-20 | done | -200 cumulative | `scripts/07_target_wider_beam.py`. Diminishing returns below len=24. |
| A4 | **Wider beam on all puzzles** (beam 131K full, 1003 puzzles) | UNTRIED | 4-5h on L4 | ~-100 more | Mostly re-visits already-solved; low ROI. |
| A5 | **Beam 262K+** | NEGATIVE | — | 0 | Same paths as 131K, 7× slower. |
| A6 | **Weighted A*** (f = w·g + h) in khoruzhii searcher | UNTRIED | 3-4h | small, speculative | Fork searcher to blend path cost and heuristic. |
| A7 | **Diversity-weighted beam** — penalize states similar to others | UNTRIED | 1-2 days | unknown | Complex re-impl; could help on plateaus. |
| A8 | **Kociemba warm-start** — feed Kociemba prefix, let beam finish | UNTRIED | 1-2h | -20 to -80 | Only helps on the 5-7 current fallback puzzles. |
| A9 | **Stagnation retry with larger blacklist** | partially implemented | 1h | ± | Already in khoruzhii_search; could tune. |
| A10 | **Multi-attempt with different random seeds** (beam RNG) | UNTRIED | 30 min | small | Re-run failures with different hash seed. |
| A11 | **Q-function / multi-head neighbor scorer** | DONE 2026-04-20 (benchmark pending) | done | TBD | Distillation from E6 trained on Kaggle (`models/qe6/epoch_0999.pt`, 1000 epochs, final distill MSE 0.205). Solver path added in `khoruzhii_search.py` (`use_q_function=True`). Solve benchmark pending local GPU freeing from beam-1M run. |
| A12 | **int8 state encoding** | DONE 2026-04-20 | done | 6× VRAM reduction, beam 2M feasible | `KhoruzhiiSolver(state_dtype=torch.int8)`, default. CPU + GPU verified same paths as int64. Beam 65K was 1.56 GB → beam 524K now 1.74 GB. Enables beam 1M-2M on 16 GB. Running beam-524K targeted (saved -160 moves, 23,672) and beam-1M full solve now. |
| A13 | **Iterative beam mode** (`beam_mode="iterated"` in cayleypy) | UNTRIED | 30 min | escalates beam on hard puzzles in one call | Andrei Smolensky/Ivan (2026-04-16-18). Replaces our post-hoc targeted-beam runs with built-in escalation. |
| A14 | **Distributed multi-GPU beam search** | UNTRIED | 1 day setup | beam 5M+ on 2 GPUs | `TryDotAtwo/cayleypy@feature/bfs-torchrun-distributed` + `torchrun --nproc 2`. NCCL-backed, owner-partitioned frontier. Ivan Litvak (2026-04-19) got 7M beam on 2× T4. Requires 2+ GPUs (GCP quota bump or Kaggle T4×2). |
| A15 | **Hamming heuristic wide-beam baseline** | UNTRIED | 1h | possibly competitive w/o NN | Run Hamming distance as heuristic at beam 500K+; tests whether our small NN + narrow beam is dominated by simple heuristic + very wide beam. Ivan Litvak reported ~3M beam in 5.5 GB on T4 with Hamming. |
| A16 | **beam-1M full solve with int8** | RUNNING 2026-04-20 | ~19h | projected -500 to -1000 | At beam 1M, pid=99 already 2,117 moves vs E6 beam-65K 2,217 (-100 in first 100). Expected finish ~2026-04-21 13:00. |

## B. Model architecture

| # | Idea | Status | Effort | Est. gain | Notes |
|---|---|---|---|---|---|
| B1 | **Small arch [1024,256]×1-block** (E3/E5/E6) | DONE | — | baseline | The "architectural win." 1.6M params. |
| B2 | **Big model [5000,1000] or [2048,1024]×8** | NEGATIVE | — | regression | v2, big_v1 both worse than small arch. |
| B3 | **Transformer / attention encoder** | UNTRIED | 1-2 weeks | unknown | Chat reports transformer beat MLP on n=15. Research cost high. |
| B4 | **GNN on Cayley graph** | UNTRIED | weeks | unknown | Structural prior via graph attention. Research. |
| B5 | **Deeper residual (4-8 blocks) on small width** | UNTRIED | 2-3h | small | Not yet tested at this scale. |
| B6 | **Conditional input** (include last-move embedding) | UNTRIED | 2h | speculative | Pass most-recent-move as extra feature. |

## C. Data generation & label source

| # | Idea | Status | Effort | Est. gain | Notes |
|---|---|---|---|---|---|
| C1 | **Random walks from solved** (baseline) | DONE | — | baseline | Current `generate_walks_torch`. |
| C2 | **Bellman-refined targets** (E6) | DONE 2026-04-19 | done | -42 single, anchor ensemble | `bellman.py`. Self-bootstrapped `1 + min_a f(apply(s,a))`. |
| C3 | **Mix BFS-d5 exact labels** (20%) | DONE (E7) | tested | bundled +10 ensemble | `sample_from_bfs_table` in `data.py`. Separate effect unclear due to bundling with C4+C5. |
| C4 | **Kociemba-path reversal walks** (10%) | DONE (E7) | tested | bundled +10 ensemble | `data/kociemba_walks.pkl`, 37K test-distribution states. |
| C5 | **1/k curriculum weighting** | NEGATIVE (E7) | — | -512 single-solve | Hurt accuracy on hard puzzles. Drop next retrain. |
| C6 | **n_back=2-8 controlled** | UNTRIED | 30 min | small | Sweet spot between n_back=1 (baseline) and n_back=40 (bad). |
| C7 | **n_back=40 alone** | NEGATIVE | — | regression | MSE 14.4 → 15.84. |
| C8 | **k_max > 30** | NEGATIVE (E4 k_max=45) | — | no gain | Picture cube's "useful" walk length capped at ~30. |
| C9 | **Shortcutting walks** — reject repeat-state walks | UNTRIED | 2-3h | small | Don't emit redundant states within a walk. |
| C10 | **Label from BFS-d6/d7 as training supplement** | DONE (d6) | — | 0 extra beyond d5 | Too sparse for training signal. |
| C11 | **Retry E7 without 1/k curriculum** (C3+C4 only) | UNTRIED (HIGH PRIORITY) | 1h | unknown, could be positive | Isolate whether C3/C4 help once C5 is removed. |

## D. Training recipe & regularization

| # | Idea | Status | Effort | Est. gain | Notes |
|---|---|---|---|---|---|
| D1 | **Fast recipe** (bf16 + compile + batch 16K + fused AdamW) | DONE | — | 3× speedup | Canonical. |
| D2 | **MSE vs L1 vs Huber** | partially | 1h | small | MSE current; L1 only tested bundled. |
| D3 | **Bigger batch (32K-64K)** | UNTRIED | 30 min | small | More stable grads; might allow higher LR. |
| D4 | **Warmup + cosine decay** | partial | 30 min | small | Currently straight cosine. |
| D5 | **Longer training** (>4000 ep on E3 arch) | NEGATIVE (E5 @ 8000ep = -96 only) | — | diminishing | Converged by ~3000. |
| D6 | **Weight decay** (L2 reg) | UNTRIED | 15 min | small | Currently 0. |
| D7 | **Dropout at MLP layers** | UNTRIED | 30 min | speculative | May prevent overfitting longer trains. |
| D8 | **Ranking / pairwise loss** (instead of MSE) | UNTRIED (HIGH EV) | 2-3h | better fit for beam search objective | Beam uses only relative ordering of neighbors, not absolute values. Triplet / listwise / BPR losses fit that directly. Chat consensus: "Bellman > biased random walk" suggests updates that correct relative ordering help most. |
| D9 | **Depth curriculum** k_max=50 → 65 (iterative) | UNTRIED | retrain | possibly novel regime | Alexander C (2026-04-02): trained to plateau at k_max=50, then continued at k_max=65. Our past k_max=45 bundled regression may not apply to an isolated curriculum ramp. |
| D10 | **Wider first layer [1024, 512] × 1** | UNTRIED | retrain | in-between E6 (1.6M) and E10 (6.6M) | 2.5M params. Mentioned in chat as stable config. |

## E. Post-processing

| # | Idea | Status | Effort | Est. gain | Notes |
|---|---|---|---|---|---|
| E1 | **Pair cancellation + state-hash shortcut** | DONE | — | -30 to -80 per submission | Standard. |
| E2 | **2-step shortcut** (18×18 move pairs) | DONE | — | small | Marginal. |
| E3 | **BFS-d5 window replacement** | DONE | — | -14 on raw, 0 on PP'd | Good for raw. |
| E4 | **BFS-d6 window replacement** | NEGATIVE | done | 0 | Same as d5; beam paths avoid d6-exclusive perms. |
| E5 | **Commutator library window replacement** | NEGATIVE | done | 0 | d4+d6 commutators: 4.5K new perms, 0 hits on real paths. |
| E6 | **BFS-d7 window replacement** | INFEASIBLE | — | theoretical | 150M+ states, 20+ GB RAM. |
| E7 | **Tail re-solve PP** — short beam to find shorter tail of path | UNTRIED (HIGH EV) | 2-3h | -50 to -200 | Bigger than window replacement for long paths. |
| E8 | **Insertion finder (classical FMC tool)** | PARTIAL 2026-04-20 | 3-4 days remaining | unknown | `cs0x7f/insertionfinder` → 404 (moved). Built `xuanyan0x7c7/insertionfinder-legacy` on Kaggle with Boost.Regex (`cayley-insertion-finder-port` kernel v6). Binary + `--init` work. **Tool is 3x3x3-specific (54 stickers, corners+edges only)**; picture cube integration requires (1) converting our f0/r0/d0... move notation to 3x3x3 Singmaster (R/U/F...), (2) running IF on the 3x3x3 projection of each path, (3) center-orientation repair. Substantial follow-up. |

## F. Symmetry / structural priors

| # | Idea | Status | Effort | Est. gain | Notes |
|---|---|---|---|---|---|
| F1 | **24× rotational symmetry augmentation** | DONE 2026-04-20 (negative) | done | +24 moves ensemble only | E9 model (`models/e9/epoch_3999.pt`, loss 8.48). Solo solve 25,060 (vs E6's 24,932). Symmetry derivation in `src/cayley/symmetry.py`. Random per-sample rotation. Training still converges similarly but doesn't beat E6 solo; marginal ensemble value. |
| F2 | **6-axis search ensembling** (= A2) | UNTRIED | 2-3h | -1000 to -3000 | Bundles with F1 code (symmetry permutations already derived). Can use the 24 rotations to solve 24 equivalent states and keep shortest. |
| F3 | **Piece decomposition encoder** | DONE 2026-04-20 (negative) | done | 0 ensemble (worse solo) | E8 model with `src/cayley/piece_features.py`. Loss 8.92 (vs E5's 8.41). Solo solve not run to completion (28/30 vs E6's 29/30 at beam 16K on hard puzzles). 2.5× slower inference. Orientation encoding may lose info; retrying with Bellman on top is a future option. |
| F4 | **Orbit-averaged predictions** at inference | UNTRIED | 2-3h | -100 to -500 | Average predictions over symmetric copies of input. Bundle with F1/F2 infra. |

## G. Classical / hybrid solvers

| # | Idea | Status | Effort | Est. gain | Notes |
|---|---|---|---|---|---|
| G1 | **Kociemba fallback** | DONE | — | baseline | 38,440 moves for 1003 puzzles; our floor. |
| G2 | **twsearch with piece decomposition** | PARTIAL 2026-04-20 | ~1 week | close most of gap | **Repo renamed `cubing/twsearch` → `cubing/twips`** (Rust project, not C++). Built on Kaggle (`cayley-twsearch-ihes` kernel v5). Picture cube **flat 72-sticker KPuzzle JSON** (`data/picture_cube.kpuzzle.json`) built + roundtrip-verified, but twips panics on it (`step != 0` in step_by.rs iterator). Correct integration requires CORNERS+EDGES+CENTERS decomposed representation with **group-compatible orientationDelta algebra** (our extracted orientation is slot-dependent — verified roundtrip fails). Substantial further work. |
| G3 | **Kociemba sym-coordinate pruning table** | UNTRIED | 1 week+ | unknown | 3.5B → 64K via FlipUDSlice sym-coord. Admissible heuristic for hybrid use. |
| G4 | **Pattern databases** (PDBs) for corners/edges/centers | UNTRIED | 3-4 days | admissible heuristic | Compatible with IDA* or as beam re-ranker. |
| G5 | **Depth-6+ MITM** | partial (d6 done) | 2-3h | small | d7 needs 150M states, infeasible. |
| G6 | **Allprefix Kociemba search** | seen in notebooks | 1-2 days | unknown | Per `kaggle_research/kaggle_nb_allprefix`. |
| G7 | **cubing.js solver integration** | seen in notebooks | 1-2 days | unknown | Per `kaggle_research/kaggle_nb_cubingjs`. |

## H. Ensembling strategies

| # | Idea | Status | Effort | Est. gain | Notes |
|---|---|---|---|---|---|
| H1 | **Min-merge across seeds** | DONE | — | diminishing past 3-5 | `ens_s0-s3` already exists. |
| H2 | **Multi-seed (5-10) at same recipe** | partial | 1 day | small per seed | Overnight on Kaggle/GCP/local in parallel. |
| H3 | **Diverse recipes ensemble** (E3 + E5 + E6 + E7) | DONE | — | compound -500+ | Current pattern. |
| H4 | **MITM BFS ensemble** | DONE | — | +44 solves on weak model | Useful for weak models; marginal on strong. |
| H5 | **NISS per-model** (not just E6-NISS) | UNTRIED | 2-3h per model | -50 to -200 | E3-NISS and E5-NISS could add diversity. |
| H6 | **Anti-correlation seed selection** | UNTRIED | 1-2 days | small | Choose seeds that disagree most. Research. |

## I. Infrastructure / compute

| # | Idea | Status | Notes |
|---|---|---|---|
| I1 | **Kaggle Notebooks** (2 GPU + 3-5 CPU concurrent) | **SET UP 2026-04-20** | Dataset `artgor/cayley-project-snapshot` + 5 kernel templates at `/tmp/kernels/`. Known quirks: P100 sm_60 needs `torch==2.4.1` + `compile_model=False`, private-dataset mount path is `/kaggle/input/datasets/<owner>/<slug>/`. See `reference_kaggle_pipeline.md`. |
| I2 | **GCP A100 40/80GB** | available after quota request | Currently 0 quota. Request for 24× aug / big training. |
| I3 | **GCP H100** | unknown quota | Would need request + likely higher cost. |
| I4 | **Spot / preemptible L4** | UNTRIED | 60% cheaper but needs checkpointing. |
| I5 | **Local 4090 + GCP L4 + 2× Kaggle concurrently** | AVAILABLE | 4-GPU fleet for ensemble runs. |
| I6 | **Multi-GPU single VM** (2-4 L4) | UNTRIED | Would need quota bump. For big-batch training. |

---

## Ranked next-step recommendations

### Tier 1 — biggest expected gain per day of work
1. **A11 Q-function / multi-head neighbor scorer** — 8× inference speedup. **In progress (Kaggle)**.
2. **A12 int8 state encoding** — 4-8× beam capacity. **DONE 2026-04-20: 6× VRAM reduction verified, enables beam 524K on 16GB GPU.**
3. **D8 Ranking loss** — direct fit for beam-search objective. 2-3h dev + retrain.
4. **G2 twsearch with piece decomposition** — 1 week dev, could close most of the gap. Highest ceiling.
5. **N1 Bellman consistency auxiliary loss** (during main training, not post-hoc) — 1-2 days, -200 to -500 moves.
6. **N2 Test-time Bellman refinement** (per-puzzle fine-tune) — 1-2 days, -50 to -150 on hardest puzzles.
7. **N4 ReduceFactor graph shortening** (Dijkstra on solution graph) — 2-3 days, -100 to -500 (Santa 2023 saved 9K+ on big cubes).
8. **N5 Distance-adaptive beam width** (wide early, narrow late) — 1 day, -50 to -200.
9. **N6 CEA loss** (near-admissible heuristic, never overestimate) — 2-3 days, -100 to -300.
10. **N3 NRPA** (Nested Rollout Policy Adaptation — different search algorithm) — 1-2 days, -50 to -200 via diversity.

### Tier 2 — reliable incremental gains
6. **A2 Six-axis ensemble search** — bundles with F1; 2-3h extra for another -1K moves.
7. **H5 NISS per-model** (E3, E5, E7) — multiple hours, ensemble diversity.
8. **A8 Kociemba warm-start** — 1-2h, helps only the 5-7 fallback puzzles.
9. **A10 Multi-attempt with different beam RNG seeds** — 30 min, small.

### Tier 3 — experimental / research
10. **B3 Transformer encoder**, **B4 GNN** — unknown payoff, weeks of work.
11. **G3 Kociemba sym-coordinate** — admissible heuristic path, 1 week+.
12. **A7 Diversity-weighted beam** — 1-2 days, unknown.

---

## Suggested execution plan

**Next session (2-4h work, plausible -200 to -500 moves)**:
1. C11 retry (E8 = E7 without curriculum). Test: E8 solo + ensemble delta.
2. E7 tail re-solve PP implementation and test on current best.

**Following session (1-2 days, plausible -1000+ moves)**:
3. F1 24× symmetry derivation + E9 training with symmetry-augmented data.
4. F3 Piece decomposition encoder + E10 training (independent of F1 but can combine after).
5. H5 NISS on E3 and E5 if GPUs are free while training.

**Multi-day push (1 week+, potentially closes most of gap)**:
6. G2 twsearch integration.

**Avoid** (confirmed negatives): A5, B2, C5 (alone), C7, C8, D5, E4, E5, E6.

---

## Confirmed negatives (session 2026-04-21/22 additions)

One-line do-not-retry entries (detail in EXPERIMENTS.md):

- **PDB corners alone** (2026-04-21): `data/pdb_picture_cube.pkl`, 12 MB, corners max depth 8, edges_6 max depth 8, centers max depth 3. Used as admissible lookup via `src/cayley/pdb_lookup.py`. Combined with neural value as `value = max(neural, pdb)`. **0 gain** on real beam search paths — 20+ move paths mean corner PDB (depth 8) is always ≤ neural estimate. Would need deeper PDBs (weeks of build time).
- **Three-step PP shortcut** (2026-04-21): `shortcut_three_step` in `src/cayley/post_process.py`. Enumerate 18³ = 5832 three-move sequences per window position, replace if a shorter 3-move sequence hits the same state. **0 gain** on raw or PP'd paths; 2-step already extracts what's there.
- **NRPA level-2 iters=50** (2026-04-21): Pure Python + policy-less NRPA on 100 hardest puzzles. **0/100 solved** in CPU budget. Would need ML-guided policy init + higher level/iters. Deferred.
- **Decomposed KPuzzle** (2026-04-21): `scripts/build_picture_cube_kpuzzle.py` with CORNERS+EDGES+CENTERS orbits + orientationDelta. Roundtrip test fails (f2·f2⁻¹ leaves slot-4 at ori=2 mod 3) because our extracted orientation is slot-dependent, not group-compatible. Used flat 72-sticker version instead (verified), but twips panics on that (`step != 0` in step_by.rs). Correct integration needs group-compatible orientation algebra — substantial follow-up work.
- **twips on flat KPuzzle** (2026-04-21): `data/picture_cube.kpuzzle.json` 41 KB, `numOrientations=1`. Twips (Rust successor to twsearch) iterator panics on this representation; needs decomposed KPuzzle with proper orientation algebra.
- **Same-recipe seed variants** (2026-04-21/22): E9 symmetry-aug, E11 data-mix, E12 Bellman-aux, E6 seeds s11/s12/s13 each contribute only +14-30 moves in ensemble. **Diminishing returns** — future model work should focus on architecturally distinct models (transformer, GNN) or direct search-side wins (wider beam, Q-function).
