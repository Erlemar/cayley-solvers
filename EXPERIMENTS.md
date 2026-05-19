# Experiment Log — IHES Picture Cube

All experiments run on this project (`C:\Users\and-l\cayley`). Prior classical / Kociemba
experiments live in `C:\Users\and-l\kaggle_research\cayleypy-ihes-cube\experiment_log.md`
under that project's numbering (Experiments 1–7 there).

**Hardware**: RTX 4090 Laptop (16 GB VRAM), CUDA driver 12.6, Windows 11, Python 3.14.
**Framework**: PyTorch 2.11.0+cu128, cayleypy 0.1.0.
**Dataset**: 1003 scrambled picture-cube puzzles, 72-facelet permutation state, 18 generators.
**Metric (Kaggle)**: total move count across all puzzles. Lower = better.
**Target (Rokicki, leader)**: 21,840.

Columns below: the config file, model checkpoint path, commit-equivalent notes, hyperparams,
wall time, loss values and (where available) solve rate / Kaggle score.

---

## Experiment 8a — onehot ResMLP (medium config)

- **Config**: [`configs/medium.yaml`](configs/medium.yaml)
- **Model**: `src/cayley/model.py` `ResMLPDistance`, encoding=onehot, hidden=[700, 643], 4 ResBlocks. 7,406,597 params. Input one-hot: 72 × 72 = 5184 dims.
- **Training recipe**: MSE loss, Adam lr 1e-3 with cosine decay, batch 4096, 1M samples/epoch, non-backtracking walks with `n_back=1` (exclude inverse of immediately previous move), k_max=30. 200 epochs.
- **Checkpoint**: `models/medium/epoch_0199.pt` (29.6 MB × 10 checkpoints).
- **Runtime**: ~14 min (single contention-free run). ~4.2s/epoch steady state.
- **Loss curve**:
  - epoch 19: 15.56, epoch 59: 14.99, epoch 99: 14.86, epoch 139: 14.73, epoch 179: 14.53, epoch 199: **14.57** (final)
  - RMS error ~3.8 moves on labels 1–30.
- **Solve (beam 4096, simple mode, Kociemba fallback)**: 802/1003 solved by model, 201 fallback. Wall time 2h42m. Final submission file `submissions/combined_model_koc.csv` (30,770 moves).
- **Kaggle**: **30,770** (confirmed). 28% improvement over the v13 Kociemba baseline (42,718).
- **Analysis**: Strong baseline. The first-100 distribution showed 30 avg moves/puzzle; puzzles 500+ degraded sharply (100+ moves/puzzle) — these were almost all fallbacks. The Kociemba fallback (38,440 CSV) carries most of the late-puzzle load.

## Experiment 8b — embedding encoder (same trunk, smaller input)

- **Config**: [`configs/embed.yaml`](configs/embed.yaml)
- **Model**: `ResMLPDistance` with `encoding="embedding"`, `embed_dim=16`, same [700, 643]×4 trunk. 4,585,349 params. Input = nn.Embedding(72, 16) flattened → 1152 dims (4.5× smaller than onehot).
- **Training recipe**: identical to 8a (MSE, Adam, cosine LR, batch 4096, 1M samples, n_back=1, k_max=30), 200 epochs.
- **Checkpoint**: `models/embed/epoch_0199.pt`.
- **Runtime**: ~12 min. ~3.6s/epoch steady state. Only 1.3 GB VRAM vs 2.8 for onehot.
- **Loss curve**:
  - epoch 19: 15.61, epoch 59: 15.10, epoch 99: 14.75, epoch 139: 14.59, epoch 179: 14.53, epoch 199: **14.40** (final)
  - Marginally better than onehot (14.40 vs 14.57).
- **Beam sweep on 30 stratified puzzles** (max_steps=60):
  - beam 2048: **22/30** solved, avg 29.8 moves, 180s
  - beam 8192: **28/30** solved, avg 27.4 moves, 2004s (11× slower)
  - beam 16384: OOM'd at 25 GB theoretical alloc
- **Solve (beam 4096, Kociemba fallback)**: **IN PROGRESS** — first 300 puzzles 8,623 vs onehot's 14,795 at same point. 42% better per-puzzle on early puzzles. Projected total ~20–25K if trend holds through hard puzzles.
- **Analysis**: Embedding encoder is both smaller and slightly more accurate. The 4.5× input reduction opens the door to wider beams without OOM. We chose NOT to submit at beam 8192 because per-puzzle time ballooned to 36s on hard puzzles (would take 10h). Beam 4096 gives substantial improvement with manageable runtime.

## Experiment 8c — v2 (big arch + n_back=40 + L1) — **REGRESSION**

- **Config**: [`configs/v2.yaml`](configs/v2.yaml)
- **Goal**: test the collective "chat recipe" from the Jan-2025 Telegram: `n_back=40` non-backtracking, L1 loss, bigger first projection [5000, 1000], AdamW wd=1e-5.
- **Model**: embedding, hidden [5000, 1000], 4 ResBlocks. 18,804,153 params (4× bigger than embed).
- **Training recipe**: L1 loss, AdamW lr 1e-3, cosine decay, batch 4096, 1M samples/epoch, **n_back=40**, k_max=30, 250 epochs.
- **Checkpoint**: `models/v2/epoch_0249.pt`.
- **Runtime**: ~44 min. ~10s/epoch steady state.
- **Loss curve**:
  - epoch 0: 5.25, epoch 24: 2.86, epoch 49: 2.81, epoch 99: 2.75, epoch 149: 2.72, epoch 199: 2.69, epoch 249: **2.70** (final)
  - Note: L1 and MSE values are not directly comparable (different scales).
- **Solve (beam 2048 & 4096, 30 stratified puzzles)**:
  - beam 2048: **10/30** solved, avg 31.2, 298s (vs embed 22/30 at 180s — worse and slower)
  - beam 4096: **13/30** solved, avg 32.8, 3,178s (11× slower for a worse result)
- **Analysis**: REGRESSED against the embed baseline. All three changes (n_back=40, L1, bigger model) bundled together produced a worse heuristic. Needed a clean diagnostic.

## Experiment 8d — v3 diagnostic (n_back=40 only)

- **Config**: [`configs/v3.yaml`](configs/v3.yaml)
- **Goal**: isolate whether n_back=40 alone is the culprit in 8c's regression.
- **Model**: embedding, hidden [700, 643], 4 ResBlocks (same as embed baseline). 4,585,349 params. **Only change vs 8b is `n_back=40`.** MSE loss retained. 100 epochs (shorter — enough to see trend).
- **Checkpoint**: `models/v3/epoch_0099.pt`.
- **Runtime**: ~6 min.
- **Loss curve**:
  - epoch 24: 16.58 (vs embed 15.56)
  - epoch 49: 16.10 (vs embed 15.10)
  - epoch 74: 15.87 (vs embed 14.98)
  - epoch 99: 15.84 (vs embed 14.75)
  - **Consistently 0.5–1.1 worse than the n_back=1 baseline.**
- **Solve**: not run — the loss curve alone was conclusive.
- **Analysis**: **`n_back=40` alone hurts training.** Hypothesis: with 18 generators grouped in 9 inverse pairs, banning the inverse of the last 40 actions quickly saturates to "ban everything," and when all pairs are banned my implementation falls back to free sampling. The result is a strange mix of walks that either constrain too little or too much, producing a distribution the NN struggles to generalize from. The Jan-2025 chat's Kaggle 10,281 run likely combined n_back=40 with other changes we haven't identified.

## Experiment 8e — embed full-solve at beam 4096 — **COMPLETE**

- **Model**: same as 8b (`models/embed/epoch_0199.pt`).
- **Solver**: CayleyPy `BeamSearchAlgorithm.search_simple` (advanced mode doesn't support return_path in this library version — a known limitation), beam 4096, max_steps 60, inference_chunk_size 2048.
- **Script**: `scripts/02_solve.py` with `--fallback data/kociemba_fallback.csv`.
- **Runtime**: 8,194s = 2h16m. 8.2s/puzzle avg.
- **Result**: **829 solved by model, 174 fallback, 30,120 total moves**.
- **Per-100-puzzle trajectory** (much flatter than onehot which collapsed after puzzle 400):

  | slice | embed moves | onehot moves | embed better % |
  |---|-------------|---|---|
  | 0–99    | 2,525       |  3,013 | 16% |
  | 100–199 | 3,060       |  4,740 | 35% |
  | 200–299 | 3,038       |  7,042 | 57% |
  | 300–399 | 3,048       |  9,284 | 67% |
  | 400–499 | 2,972       |  8,736 | 66% |
  | 500–599 | 3,122       | 12,268 | 75% |
  | 600–699 | 2,988       | 15,862 | 81% |
  | 700–799 | 3,074       | 19,574 | 84% |
  | 800–899 | 3,140       | 20,940 | 85% |
  | 900–999 | 3,048       | 25,678 | 88% |

- **Output**: `submissions/embed_b4k_full.csv`.
- **Kaggle submission**: submitted 2026-04-17 18:30 UTC, awaiting score.
- **Analysis**: The embed model **solves hard puzzles that onehot punted to Kociemba**. Result: ~30/puzzle across the entire difficulty range (vs onehot's 30-260 per puzzle at the tail). Total improvement: -650 moves (30,770 → 30,120, 2.1%). Smaller than the early-puzzle delta suggested because the Kociemba fallback already provided a strong floor for onehot's unsolved puzzles — the real delta is on the ~150 puzzles where onehot fell back and embed solved. Fallback count dropped 201 → 174 puzzles.

## Side experiments (no new training required)

### Post-processing on combined_model_koc.csv

- **Pair cancellation + state-hash single-move shortcut**: `scripts/post_process_submission.py`. 12 paths shortened, -26 moves → 30,744. Negligible.
- **+ 2-step shortcut** (`shortcut_two_step`, tries 18×18 = 324 move pairs at each position for longer jumps): 15 paths shortened, -34 moves → 30,736. Still tiny. Why: beam-search solutions are already close to locally optimal; pair cancellation rarely applies.

### v13 Kociemba fallback download

- Downloaded `artgor/cayleypy-ihes-supercube-baseline` latest output via Kaggle API (`data/kociemba_fallback.csv`). Validates 1003/1003 at 38,440 moves — 10% better than the v13 on Kaggle (42,718), which is from an older kernel version. This replaced the previous `sample_fallback.csv` (500,620) as the fallback floor.

## Experiment 9 — E6 Bellman refinement (warm-start from E5)

- **Config**: [`configs/e6_bellman.yaml`](configs/e6_bellman.yaml); driver `scripts/05_bellman_refine.py`.
- **Goal**: replace walk-depth labels with self-bootstrapped Bellman targets `y = 1 + min_a target_net(apply(s, a))` to eliminate the walk-depth-upper-bound bias. Warm-start from E5's 8,000-epoch checkpoint.
- **Model**: E5 architecture unchanged — embed [1024, 256] × 1 ResBlock, ~1.6M params.
- **Training recipe**: MSE target, Adam lr 5e-4 (half the E5 LR — refining not exploring), cosine decay, batch 10,000, 1M samples/epoch, k_max=26, n_back=1, bf16+compile+fused AdamW, 500 epochs. Target network is a frozen deepcopy refreshed every 10 epochs. Clipping: target ≤ walk_depth and ≥ 0.
- **Checkpoint**: `models/e6/epoch_0499.pt`.
- **Runtime**: 500 epochs in ~52 min (~6.4s/epoch steady, compile cost ~60s on epoch 0). Each batch runs the target net on all 18 children per state (~18× the forward cost of baseline), then one forward+backward on the trainable model.
- **Loss curve (Bellman-MSE)**:
  - epoch 0: 2.21 (warm-start predictions still aligned to walk-depth; large gap vs Bellman target)
  - epoch 10 (first target refresh): 1.01
  - epoch 33: 0.38, epoch 80: 0.18, epoch 200: 0.18, epoch 500: **0.17** (final)
  - Note: Bellman target values are much smaller than walk-depth — the scale is not comparable with E5's MSE 8.41.
- **E6 vs E5 on 30 hard puzzles at beam 16K** (pids 400-429):
  - E5: solved 27/30, total 709, per-puzzle median delta 0
  - E6: solved **29/30**, total 768, per-puzzle median delta 0
  - E6 adds coverage (+2 puzzles) at the same avg length on the common set. Gain comes from hard puzzles that E5 couldn't crack.
- **Full solve at beam 65K** (khoruzhii searcher, bf16): 998/1003 solved, total **24,932** in 6,923s (1h55m). E5 at same settings was 997/1003 @ 24,974. Net: **-42 moves** single-model; +1 coverage.
- **Submission output**: `submissions/e6_khoruzhii_b65k.csv`.

## Experiment 10 — NISS (inverse scramble search) on E6

- **Goal**: break the directional anisotropy of our beam search by also solving the inverse permutation σ⁻¹ for each puzzle. A valid solve path for σ is `invert_path(solve(σ⁻¹))`. FMC-community standard technique since 2009.
- **Code**: added `invert_state` and `invert_path` methods to `PictureCube` in `src/cayley/puzzle.py` (5 unit tests in `tests/test_puzzle.py`). Standalone `scripts/06_niss_solve.py` that inverts each test state, runs the same KhoruzhiiSolver, converts the path back, verifies, falls back to Kociemba on failure.
- **Solo run**: 997/1003 solved, total **24,948** in 6,929s (E6 forward was 24,932). Effectively on par with forward as a standalone submission.
- **Ensemble contribution — the real win**: combining NISS output with the existing candidates (E6, E5, E3, prior ensembles) dropped the combined total from 24,502 → **24,070** — **-432 moves** purely from adding NISS.
- **Interpretation**: NISS rarely produces the shortest path on any single puzzle, but it fills in gaps — on ~430 puzzles the inverse-direction beam happens to find a different short path that's strictly better than what forward solves produced. Confirms the "directional anisotropy" hypothesis.
- **Submission output**: `submissions/niss_e6_b65k.csv` (solo), `submissions/ens_e6_niss.csv` (combined), `submissions/ens_e6_niss_pp.csv` (final submitted).

## Experiment 11 — targeted wider-beam tail re-solves (B3 family)

- **Goal**: find shorter paths for the longest-path puzzles in our baseline by running beam 131K + NISS with E6 on targeted pids.
- **Script**: `scripts/07_target_wider_beam.py` — accepts `--min-length N` or explicit `--pids a,b,c`. For each target, tries forward + NISS with a wider beam; keeps the shorter of existing baseline and new attempts.
- **Runs (local 4090 Laptop, 2026-04-20)**:
  - `--min-length 27` (40 puzzles): 24 improved, **-50 moves** → baseline 24,018.
  - `--min-length 26` (148 puzzles, covers the above): 33 improved, **-74 moves** → 23,944.
  - `--pids <288 pids with baseline len == 25>`: 37 improved, **-76 moves** → 23,868.
- **Diminishing returns curve**: improvement rate drops from 60% (len≥27) to 22% (len=26) to 13% (len=25). At len=24 baseline paths are already near-optimal for this beam/model combo.
- **Submissions**: `submissions/{wide_b131k_tail,wide_b131k_tail26,wide_b131k_len25}_pp.csv`. Each submitted to Kaggle (2026-04-20): **24,018 → 23,944 → 23,868**.

## Experiment 12 — E8 piece-decomposition encoder

- **Goal**: replace the 72-facelet one-hot/embedding input with a structural prior: 52 small-vocab piece features (corner_id+ori, edge_id+ori, center_id+ori). Cube pieces move as units, so this is the same invariant twsearch uses. 46-52 ints × small vocab embedded vs 72 large-vocab.
- **Derivation** (`src/cayley/piece_features.py`): 8 corners × 3 stickers + 12 edges × 2 + 6 centers × 4 = 72 (verified disjoint + partition). Orientation = index of canonical-first sticker within the slot's read order. Cube invariants `sum(corner_ori) % 3 == 0` and `sum(edge_ori) % 2 == 0` hold after random scrambles — tests in `tests/test_piece_features.py` (6/6 pass). GPU-vectorized extractor (`TorchFeatureExtractor`) verified against numpy reference on 100 random states.
- **Model change**: added `encoding="piece"` branch to `ResMLPDistance` with 6 separate embedding tables. Smaller input dim (832 vs 1152 for embedding encoder) → fewer params (1.25M vs 1.58M).
- **Training (E8, `configs/e8_piece.yaml`)**: same recipe as E5 (K_max=26, batch 10k, 4000 epochs, bf16+compile+fused AdamW). ~0.5s/epoch steady state (compile cache hit), ~40min total.
- **Loss curve**: 47.3 (ep 0) → 9.52 (500) → 9.25 (1000) → 8.91 (3000) → **8.92 final (3999)**. Plateaus ~8.9 vs E5's 8.41 at same epochs. Structural prior **hurts** the representation at this scale.
- **Solve (beam 16K on pids 400-429)**: E8 28/30 at total 757 vs E6 29/30 at 768. Fewer solves, shorter on common. Inference 2.5× slower (unfused piece-extractor + embedding path).
- **Decision**: do not run full beam 65K solve — marginal ensemble value not worth the compute. Likely salvageable only with Bellman refinement on top (untested).

## Experiment 13 — E9 24× rotational symmetry augmentation

- **Goal**: augment training data 24× by applying a random whole-cube rotation to each sample (labels invariant under rotation — rotating solved gives solved).
- **Derivation** (`src/cayley/symmetry.py`): BFS from identity using the 3 whole-cube rotations `R_f = f0·f1·f2`, `R_r = r0·r1·r2`, `R_d = d0·d1·d2` yields exactly 24 distinct permutations (the cube's rotational symmetry group). Tests in `tests/test_symmetry.py` (7/7 pass).
- **Training (E9, `configs/e9_symmetry.yaml`)**: E5 recipe + `augment_symmetry: true`. 4000 epochs at ~1.5s/epoch (on GCP L4). Random per-sample rotation applied after random-walk generation.
- **Loss curve**: 9.00 (500) → 8.88 (1000) → 8.60 (2000) → 8.54 (2500) → 8.48 final (3999). Converges to ~E5 level, not clearly better.
- **Solve at beam 65K**: 25,060 moves (vs E6 solo 24,932 — E9 is 128 worse). 997/1003 solved.
- **Ensemble value**: E9 beats the prior own-work best on 12 puzzles, saving **24 moves** when combined → new own-work best **23,832** (from 23,858). Modest but positive.

## Experiment 14 — E11 (data mix without curriculum), E6 multi-seed, PDB (Kaggle parallel)

- **Infrastructure (2026-04-20)**: set up Kaggle parallel-training pipeline. Dataset `artgor/cayley-project-snapshot` (34MB zipped). 5 kernel templates pushed. Key quirks resolved: P100 sm_60 requires `torch==2.4.1` + `compile_model=False`; private datasets mount at `/kaggle/input/datasets/<owner>/<slug>/`. See `reference_kaggle_pipeline.md`.
- **Kernels launched**:
  - `cayley-e11-data-mix-no-curriculum` (GPU): E5 warmstart + BFS-d5 20% + Kociemba walks 10% + no curriculum. 1500 epochs. Running.
  - `cayley-e6-multi-seed-bellman` (GPU): 3 seeds (11/12/13) of E6 Bellman refinement, each 500 epochs sequential. Running.
  - `cayley-pattern-database-build` (CPU): **COMPLETE**. Output `data/pdb_picture_cube.pkl` (12 MB) — corner, edge-of-6, center BFS tables for admissible heuristic. Usable as re-ranker or future IDA*.
  - `cayley-twsearch-ihes` (CPU): FAILED. `cubing/twsearch` tarball URL serves the renamed `twips` Rust archive; needs further investigation.
  - `cayley-insertion-finder-port` (CPU): FAILED. `cs0x7f/insertionfinder` returns HTTP 404 on both `main` and `master` branches. Needs correct upstream URL.

## Experiment 16 — int8 state encoding in the Khoruzhii searcher (2026-04-20)

- **Motivation**: Community chat (Vlad Kuznetsov, Ivan Litvak, April 2026) noted beam buffers dominate VRAM at wide beams. State values are 0–71, fit in int8/uint8. Current int64 wastes 8 bytes/position.
- **Change**: added `state_dtype: torch.dtype = torch.int8` parameter to `KhoruzhiiSolver.__init__` (default int8). `V0`, initial state, and all beam buffers now int8. Generator permutations (indices) stay int64 for gather efficiency. Hashing casts to int64 internally.
- **CPU verification**: paths match int8 vs int64 on a short-solve puzzle.
- **GPU benchmark (5 varied puzzles, E6, beam 65K)**:
  - int64: identical paths, 33.4s, **peak VRAM 1.56 GB**
  - int8: identical paths, 40.7s, **peak VRAM 0.26 GB** (6× reduction; +22% wall time)
- **Beam-scaling on pid 500** (int8):
  - 131K: 10.6s, 26 moves, 0.47 GB
  - 262K: 19.3s, 26 moves, 0.89 GB
  - **524K: 32.5s, 24 moves, 1.74 GB** (shortened by 2 vs beam 131K)
  - 1M: 67.3s, 24 moves, 3.43 GB (no further shortening)
  - 2M: 178.4s, 24 moves, 6.81 GB (no further shortening)
- **Conclusion**: int8 unlocks beam 524K–1M on our 16 GB GPU (previous cap was 131K). Sweet spot for many puzzles appears to be beam 524K.

## Experiment 17 — beam-524K targeted re-solve on len ≥ 26 (2026-04-20)

- **Script**: `scripts/07_target_wider_beam.py` with `--beam 524288 --try-niss` at E6.
- **Target**: 108 puzzles from `ens_own_through_e9_pp.csv` with baseline length ≥ 26.
- **Result**: 74 improved, 33 same-or-worse, 1 failed. **-160 moves saved, total 23,672 (1003/1003 valid)**.
- **Pattern**: 69% hit rate (much higher than beam-131K's 22% on same size band earlier). Average save per improved puzzle: ~2 moves.
- **Runtime**: 8,000 s (~2h 13m). ~74s per puzzle (forward + NISS each at beam 524K).
- **Submission**: `submissions/wide_b524k_tail26_pp.csv` submitted to Kaggle 2026-04-20 as new own-work best **23,672** (improves prior 23,858 by -186).

## Experiment 18 — E11 data-mix-no-curriculum (Kaggle, 2026-04-20)

- **Hypothesis**: E7 added BFS+Kociemba data + 1/k curriculum and regressed. Isolating by **dropping curriculum** tests whether the data supplements help on their own.
- **Config**: 1500 epochs, warmstart from E5, BFS-d5 mix 20%, Kociemba walks 10%, NO curriculum, MSE loss, E5 architecture. Trained on Kaggle P100 (torch 2.4.1 pin + `compile_model=False`).
- **Result**: final train loss 5.93 (headline lower than E5's 8.41, but misleading — BFS samples with distance 0-5 dominate the mean).
- **Solve (local, beam 16K, pids 400-429)**: E6: 29/30 total 768. **E11: 22/30 total 572** (shorter on common but 7 fewer solves).
- **Full solve (beam 65K)**: **25,034 moves** (E6 was 24,932; E11 -102 worse solo).
- **Ensemble contribution**: +14 moves saved on 7 puzzles. New own-work ensemble: 23,832.
- **Decision**: E11 has ensemble value but single-solve coverage regression suggests BFS-heavy labels narrow the model to near-goal accuracy at the expense of hard-puzzle generalization.

## Experiment 19 — E6 multi-seed Bellman refinement (Kaggle, 2026-04-20)

- **Config**: 3 seeds (11, 12, 13) × 500 epochs × E6 Bellman recipe. Sequential in one GPU kernel.
- **Results**: final Bellman-MSE 0.1715–0.1734 (vs E6 original 0.1693). Near-identical.
- **Checkpoints**: `models/e6_s{11,12,13}/epoch_0499.pt`. Solves pending local GPU freeing from beam-1M.

## Experiment 20 — Q-function distillation from E6 (Kaggle, 2026-04-20)

- **Hypothesis**: Vlad Kuznetsov (CayleyPy chat 2026-04-04) reported 8× beam-search inference speedup by training a model that outputs all 18 neighbor values in one forward pass. For megaminx: beam 2^21 in the wall-time of 2^18; avg path 83 vs 91.
- **Architecture**: E6 trunk + 18-output head (via new `output_dim` param on `ResMLPDistance`). Distillation targets: Q(s, a) ≈ V_teacher(apply(s, a)).
- **Training**: 1000 epochs on Kaggle GPU, MSE distill loss, cosine LR, bf16.
- **Result**: final distill MSE 0.205. Checkpoint `models/qe6/epoch_0999.pt`.
- **Solver integration**: `KhoruzhiiSolver(use_q_function=True)` replaces per-child V forward with one Q forward per parent, selects top-B by `q_flat[idx1]`, materializes children only for top-B.
- **Benchmark**: pending local GPU freeing from beam-1M run.

## Experiment 21 — Pattern Database build (Kaggle, 2026-04-20)

- **Script**: CPU Kaggle kernel `cayley-pattern-database-build`.
- **Output**: `data/pdb_picture_cube.pkl` (12.2 MB) with three BFS-derived distance tables:
  - `corners`: 8! = 40,320 states, max depth achieved.
  - `edges_6` (partial, first 6 edge slots): bounded state count.
  - `centers`: 6! = 720 states.
- **Status**: build complete. Admissible-heuristic integration into our solver/search not yet written.

## Experiment 22 — beam-1M full solve on int8 (RUNNING 2026-04-20)

- Queued automatically after Experiment 17 via `/tmp/queue_beam1m.sh`.
- Full solve: 1003 puzzles × E6 beam 1,048,576 + int8 encoding.
- **Partial**: pid=99 at 2,117 moves (E6 beam-65K was 2,217 — **-100 moves in first 100**). Projection: -500 to -1000 total.
- ETA: ~19h (~67s per puzzle). Expected completion 2026-04-21.

## Experiment 23 — twsearch / insertion-finder build (Kaggle, 2026-04-20, PARTIAL)

- **twsearch → twips**: `cubing/twsearch` was renamed to `cubing/twips`, rewritten in Rust. Kaggle kernel `cayley-twsearch-ihes` v5 builds Rust twips successfully. `twips --help` shows commands including `search`, `solve-known-puzzle`, `gods-algorithm`, `scramble-finder`. For picture cube, flat 72-sticker KPuzzle JSON built + roundtrip-verified (`data/picture_cube.kpuzzle.json`), but `twips search` panics on it ("step != 0" in Rust step_by iterator, likely due to our `numOrientations=1` flat-orbit representation). Proper integration requires CORNERS+EDGES+CENTERS decomposed KPuzzle with group-compatible orientationDelta — our extraction is slot-dependent (verified roundtrip test: `f2·f2⁻¹` leaves slot-4 orientation at 2 mod 3 instead of 0).
- **Insertion finder**: `cs0x7f/insertionfinder` (referenced in IDEAS) returns 404. Used `xuanyan0x7c7/insertionfinder-legacy` (archived, C++11 + Boost.Regex) — kernel v6 builds cleanly (`apt install libboost-regex-dev`). Binary + `--init` works. Tool is 3×3×3-specific (no centers); picture cube integration needs (1) move notation translation f0/r0/d0 → R/U/F, (2) 3×3×3-only projection of paths, (3) center-orientation repair after insertions. Deferred.

## Experiment 15 — community submission min-merge (reference baseline, not submitted)

- **Goal**: measure how competitive our own-work submissions are by comparing against publicly-shared solutions from other competitors.
- **Data sources**: `alexandervc/cayleypy-submissions` (233 MB), `arabidopsisthalian/subm-from-s-to-24` (132 KB), `olegpushs/marged-2025-11-14` (900 KB), `arabidopsisthalian/{ihes-v2,ihes-ruslan,script-ihes,ihes-test-1000-cude-...}`, `adaluodaa/ihes-path`, `alexandervc/cayleypy-submits-parts`, `olegpushs/sub-ihes`. 727 CSVs, 299 valid covering all 1003 puzzles.
- **Best single community submissions**: `submission_22496_Cheldieva.csv` (22,496), `submission_22526_Cheldieva.csv` (22,526), `submission 23088 V2.csv` (23,088), `submission_ 23286_Cheldieva_fromB9_toB24_nrd10.csv` variants.
- **Min-merge of all community CSVs (excluding our work)**: **21,972 moves** (1003/1003 valid).
- **Our contribution**: 0 puzzles where our own-work best beats the community min-merge. Every puzzle our models solve, community solves at least as short.
- **User policy**: do not submit the community-merged result; continue developing until our own work produces at least some improvements over community on individual puzzles.

## Side experiments — post-processing extensions (2026-04-19)

All confirmed **zero-gain** against BFS-d5 on both raw beam outputs and already-pp'd submissions.

### Commutator library (`src/cayley/commutator.py`)

- Enumerated commutators: all `[A, B]` depth-4 (324 words), all `X [A, B] X⁻¹` depth-6 (5,832 words), all `[A B, C]` depth-6 (5,832 words). Deduplicated by permutation → **4,996 unique perms**, of which 4,563 are NOT in BFS-d5.
- Usage: merged with BFS-d5 into a combined `BfsTable` via `merge_tables`, used in `reduce_factor_via_bfs_table`.
- Tested on `ens_e5_all_pp.csv` (24,618, already-pp'd) and `e5_khoruzhii_b65k.csv` (24,974, raw): **0 additional moves saved** over d5-only PP in both cases.
- Diagnosis: real beam paths rarely produce net-window-permutations that land in commutator perms (~5K perms in ~10²² state space). Window replacement with commutator library is theoretically sound but empirically empty for our scale.

### BFS-d6 table (`data/bfs_table_d6.pkl`)

- Full BFS from identity to depth 6: **10,932,894 states** built in 802s, pickle 1.77 GB on disk, ~5 GB RAM on load.
- Scripts: `scripts/build_bfs_table.py --depth 6` (existing, worked unchanged).
- Tested on the same submissions: **also 0 additional moves saved** over d5-only PP. Even with 2,000× the coverage of the commutator library, real path windows still don't hit d6-exclusive perms.
- Implication: **window-replacement post-processing is saturated** for our current solver. Additional d6 savings would require either (a) a much deeper BFS like d7/d8 (150M+ states, infeasible locally), or (b) a different PP approach that modifies the path structure (e.g., tail re-solving with a short-beam search for the last 15 moves of long paths).

---

## Kaggle submissions

| Date (UTC) | File | Model / settings | Public score | Notes |
|---|---|---|---|---|
| 2026-04-16 | (legacy) | Kociemba v13, multi-prefix + center lookup | **42,718** | Pre-ML baseline. |
| 2026-04-17 | `combined_model_koc.csv` | onehot [700,643]×4 model, beam 4096 + Kociemba fallback | **30,770** | First ML submission. 28% better than v13. |
| 2026-04-17 | `embed_b4k_full.csv` | embed [700,643]×4 model (4.6M params), beam 4096 + Kociemba fallback | **30,120** | -650 moves. Embed solves ~27 more puzzles before fallback. |
| 2026-04-18 | `fast_b4k_full.csv` | fast model [700,643]×4 (500ep bf16+compile+batch16k), beam 4096 + Kociemba fallback | **29,710** | -410 moves. Speed optimizations validated (3× training speedup). |
| 2026-04-18 | `fast_b8k_pp.csv` | fast model, beam 8192 + BFS-d5 pp | **28,224** | -1,486. Wider beam helps. |
| 2026-04-18 | `ens_s0_s1_pp.csv` | 2-seed ensemble + pp | **27,790** | -434. Ensemble starts paying. |
| 2026-04-18 | `ens_s0123_pp.csv` | 4-seed ensemble + pp | **27,366** | -424. Diminishing returns on seeds. |
| 2026-04-18 | `ens_mitm_s123_pp.csv` | MITM BFS depth-6 + 3-seed ensemble + pp | **27,106** | -260. MITM adds +44 model solves. |
| 2026-04-18 | `ens_e3_all_pp.csv` | **E3 small arch** (1024/256/1-blk, 4000ep, K_max=26) + khoruzhii searcher @ beam 65k + full ensemble + pp | **24,998** | **-2,108 — architectural win.** E3 model solves 996/1003 at beam 65k. |
| 2026-04-18 | `ens_e5_all_pp.csv` | E5 (E3 arch × 8000 ep) + full ensemble + pp | **24,618** | -380. Marginal. |
| 2026-04-19 | `ens_e6_niss_pp.csv` | **E6 Bellman refinement** (500ep self-bootstrapped) + **NISS-E6** (inverse scramble) + full ensemble + BFS-d5 pp | **24,068** | **-550.** E6 alone beats E5 by 42 single-solve; NISS adds -432 in ensemble. Gap to leader 2,778 → 2,228. |
| 2026-04-20 | `ens_e6_niss_b131k_pp.csv` | + beam-131K+NISS re-solve on 40 tail puzzles (len≥27) | **24,018** | -50. 24/40 improved at wider beam. |
| 2026-04-20 | `wide_b131k_tail26_pp.csv` | + beam-131K+NISS on 148 len≥26 puzzles | **23,944** | -74. 33/148 improved. |
| 2026-04-20 | `wide_b131k_len25_pp.csv` | + beam-131K+NISS on 288 len=25 puzzles | **23,868** | -76. 37/288 improved. |
| 2026-04-20 | `ens_b131k_plus_e7_pp.csv` | + E7 data-bundle ensemble (BFS+Kociemba+curriculum) | **23,858** | -10. Marginal ensemble value from E7; E7 solo 25,444 is -512 worse than E6 (curriculum regresses). |
| 2026-04-20 | `wide_b524k_tail26_pp.csv` | + int8 state encoding + beam-524K+NISS re-solve on 108 len≥26 puzzles | **23,672** | **-186**. 74/108 improved (69% hit rate at 4× wider beam). int8 unlock: peak VRAM 1.74 GB for beam 524K. |

---

## Training run summary (all configs & outcomes)

| Config | Arch | ep | Batch | K_max | Loss | Notes | Final loss |
|---|---|---|---|---|---|---|---|
| medium | onehot [700,643]×4 | 200 | 4096 | 30 | MSE | Baseline | 14.57 |
| embed | **embed** [700,643]×4 | 200 | 4096 | 30 | MSE | +embedding encoder | **14.40** ✓ |
| v2 ❌ | embed [5000,1000]×4 | 250 | 4096 | 30 | L1 | n_back=40, AdamW | 2.70 (L1 scale) — regressed |
| v3 ❌ | embed [700,643]×4 | 100 | 4096 | 30 | MSE | n_back=40 only | 15.84 — worse |
| fast | embed [700,643]×4 | 500 | 16384 | 30 | MSE | + bf16+compile+fused | **14.14** |
| big_v1 ❌ | embed [2048,1024]×8 | 999 | 16384 | 30 | MSE | curriculum + big arch | 5.2 (curr. scale) — regressed |
| small_e1 | embed [1024,256]×1 | 500 | 16384 | 30 | MSE | khoruzhii arch | 14.33 |
| small_e2 | embed [1024,256]×1 | 2000 | 16384 | 30 | MSE | E1 × 4 epochs | 13.51 |
| **small_e3** ✓ | embed [1024,256]×1 | 4000 | 10000 | 26 | MSE | khoruzhii hyperparams | **8.50** |
| small_e4 | embed [1024,256]×1 | 2000 | 16384 | 45 | MSE | longer walks | 46.1 (k=45 scale) |
| **small_e5** ✓ | embed [1024,256]×1 | 8000 | 10000 | 26 | MSE | E3 × 2 epochs | **8.41** |
| **e6 (Bellman)** ✓ | embed [1024,256]×1 | 500 | 10000 | 26 | MSE | self-bootstrapped target, warm-start E5 | **0.17** (Bellman scale) |
| e7 ❌ | embed [1024,256]×1 | 4000 | 10000 | 26 | MSE | BFS 20% + Kociemba 10% + curriculum | 2.36 (curriculum scale — incomparable) |
| e8 ❌ | **piece** [1024,256]×1 | 4000 | 10000 | 26 | MSE | piece-decomposition encoder | 8.92 (worse than E5's 8.41) |
| e9 | embed [1024,256]×1 | 4000 | 10000 | 26 | MSE | 24× rotational symmetry aug (random per-sample) | 8.48 |
| e11 (Kaggle) | embed [1024,256]×1 | 1500 | 10000 | 26 | MSE | BFS-d5 mix 20% + Kociemba walks 10%, NO curriculum, warmstart E5 | 5.93 (BFS-heavy mean distorts headline; coverage regressed vs E6) |
| e6_s11/s12/s13 (Kaggle) | embed [1024,256]×1 | 500 | 10000 | 26 | MSE | Bellman refinement, seeds 11/12/13 (warmstart E5) | 0.1715 / 0.1721 / 0.1734 (Bellman scale — near original E6 0.1693) |
| qe6 (Kaggle) | embed [1024,256]×1, **output_dim=18** | 1000 | 8000 | 26 | distill MSE | Q-function distillation from E6 teacher | 0.205 (distill MSE on 18 neighbor-value outputs) |

Checkmarks = landmark results. ❌ = regressed vs prior best.

---

## Key findings so far (decision log)

### What WORKED

1. **Embedding encoder beats one-hot** — loss 14.40 vs 14.57; cuts input from 5,184 → 1,152 dims (4.5× smaller).
2. **bf16 inference**: 1.2× speedup, *identical* paths to fp32. Safe. (compile=harmful for search, causes recompile loops.)
3. **bf16+batch 16K+torch.compile+fused AdamW** for training: confirmed 3× speedup (4.2s/epoch → 1.4s).
4. **Wider beam helps dramatically — on the right model**: beam 8192 → 65K gives solved 947 → 997/1003. But only when heuristic is good.
5. **MITM BFS depth 6**: 11M-state frontier from solved, `return_all_hashes=True` required. Adds 6-7 moves of effective beam reach. Single run +44 more model-solved puzzles vs plain (903 → 947).
6. **khoruzhii/cayleypy-cube searcher port** (`src/cayley/khoruzhii_search.py`): self-contained 150-line beam. fp16 value buffer, tree-based path storage, stagnation restarts. Fits beam 2^16–2^18 on 16GB where cayleypy's library OOMs at 2^14. **This unlocked the big jump.**
7. **Smaller architecture + longer training is better**: khoruzhii config (hd1=1024, hd2=256, 1 res block, ~1.6M params) at 4000 epochs beat our [700,643]×4 model (4.6M, 500ep). 3.5× faster per epoch too. Shallower nets = easier gradient flow at this size.
8. **Ensemble pays** up to ~3-5 models at same recipe then diminishes. Combining models with different architectures + solve configurations (MITM, beam widths) helped more than pure seed variation.
9. **K_max=26 with 10,000 batch (khoruzhii's setup)** matched their hyperparams exactly — gave the best individual model result (loss 8.50, avg 22.1 moves on 10 puzzles).

### What DID NOT WORK

1. **`n_back=40` alone** (tested isolated in v3): MSE goes 14.40 → 15.84. The Jan-2025 chat's claim of Kaggle 10,281 with this setting must have combined it with things we never identified.
2. **L1 loss bundled with larger arch + `n_back=40`** (v2): severe regression (10/30 at beam 2048 vs 22/30 for baseline). Not isolated — can't say L1 alone is bad.
3. **Curriculum (1/k weighted loss) + big architecture** (big_v1, 23.7M params): worse than our fast model at beam 4096 (8/10 vs 9/10).
4. **`torch.compile` at inference**: recompile loop on every unique beam batch size, 5.8× slower. Keep compile for training only.
5. **8000 epochs vs 4000** (E5 vs E3): -96 moves full solve. Not worth the 2× compute. Converged by ~3000 epochs.
6. **k_max=45 walks** (E4): no improvement on 100-puzzle b16k test (avg 26.1 same as E3). Picture cube's "useful" walk length is ≤ ~30.
7. **Single- and two-step shortcut post-processing**: saves only 30-80 moves out of 30K (0.2-0.3%). Model solutions are already tight.
8. **BFS depth-5 table for post-processing**: same, 80 moves savings. Real gain would need depth 7+ (infeasible at 150M states).
9. **Beam 16,384 with cayleypy library**: OOMs at 25 GB theoretical alloc on 16GB card. Must use khoruzhii searcher.
10. **Beam 262,144 with khoruzhii**: same paths as beam 131K, 7× slower. Diminishing returns past 2^17 for this model.

### Technical caveats

1. **CayleyPy's `advanced` mode doesn't return paths** (library limitation) — must use `simple` mode. Dual-mode trick from competitor chat is wall-time-only.
2. **`BfsResult.layers_hashes` is empty by default** — must pass `return_all_hashes=True` to `graph.bfs()` for MITM to work.
3. **Fresh `CayleyGraph` per solve has different random hash vectors** — broke state tracking. Must build graph once, reuse.
4. **Checkpoint key prefix `_orig_mod.`** when `torch.compile` was used during training. `load_model_checkpoint` strips it.

---

## Files & paths reference

- Plan: [`C:\Users\and-l\.claude\plans\this-will-be-a-purrfect-shore.md`](../.claude/plans/this-will-be-a-purrfect-shore.md)
- Ideas backlog: [`IDEAS.md`](IDEAS.md)
- Previous project's log (Kociemba experiments 1–7): `C:\Users\and-l\kaggle_research\cayleypy-ihes-cube\experiment_log.md`
- Key code:
  - `src/cayley/model.py` — ResMLPDistance (onehot + embedding variants, inference chunking)
  - `src/cayley/data.py` — non-backtracking random walks (numpy + torch, configurable n_back)
  - `src/cayley/training.py` — training loop with amp/compile/fused flags
  - `src/cayley/search.py` — CayleyPy wrapper, reusable Solver
  - `src/cayley/post_process.py` — cancellation + shortcut (1-step, 2-step)
  - `src/cayley/submit.py` — submission builder with fallback min-merge
- Scripts: `01_train.py`, `02_solve.py`, `combine_submissions.py`, `post_process_submission.py`, `beam_sweep.py`
