# Experiment log — Megaminx

Kaggle: [CayleyPy Megaminx](https://www.kaggle.com/competitions/cayley-py-megaminx).
User: `andlukyane`. Deadline 2026-08-31. 16 teams.

**Hardware (default)**: RTX 4090 Laptop (16 GB VRAM, 80 W thermal cap), CUDA 12.8, Windows 11, Python 3.14.
**Framework**: PyTorch 2.11.0+cu128, cayleypy 0.1.0.
**Dataset**: 1001 scrambled Megaminx puzzles ordered by scramble depth (pid 0 = 1-move scramble, pid 1000 = 1000-move). 120-element permutation state. 24 generators.
**Metric**: total moves across all puzzles, lower = better.
**Baselines**:
- Raw sample_submission.csv: 500,572
- Post-processed sample (same-face run reduction, 8.54% free): 457,810
- + BFS-d5 window replacement (9.24% more): 415,521
- + BFS-d6 window replacement (0.20% more): **414,678** (current floor, 2026-04-24)
- Top leaderboard: 79,971 (Kuznetsov), 81,946 (DrozdovDan), 93,606 (Rokicki). Rest: ~413–500K.

---

## Summary table

| ID   | Date       | What changed                                 | MSE   | Stratified score  | Leaderboard     | Notes |
|------|------------|----------------------------------------------|-------|-------------------|-----------------|-------|
| m01  | 2026-04-24 | First port: ResMLP [1024,256]×1, k_max=40, 200 ep | 9.76  | 19/20 easy solved | —               | Smoke only; hard puzzles OOD |
| m02  | 2026-04-24 | Widen k_max=40→80, 2000 ep, same arch        | 66.16 | 2/51 @ beam16k; 0/31 @ beam32k+NISS+BFS-d5 | — | Heuristic-noise-bound; stratified 3/bucket with full innovation stack saves 0 moves → confirms model quality is the binding constraint, not search |
| pp0  | 2026-04-24 | pp_bfs5_fallback.csv — no model              | —     | —                 | **415,521 (rank #8/16)** | Baseline; pure post-processing of sample (same-face + BFS-d5 window replacement) |
| m03  | 2026-04-24 | [2048,512]×2 k_max=80 4000 ep (Kaggle P100)  | 63.74 | —                 | pending         | 8.5h wall (P100 @ 7.6s/ep vs my estimate 0.25s/ep; big underestimate) |
| m04  | running    | Bellman-from-scratch (Kaggle P100)           | —     | —                 | pending         | Pearcatcher recipe: discount 0.999, 120 target refreshes, no RW pretraining |
| m07  | 2026-04-24 | [2048,512]×2 k_max=80 4000 ep seed=10 (4090) | 64.22 | 0/31 @ beam 32k   | —               | 80 min on laptop vs 8.5h on P100. Canonical replica — MSE matches m03 within noise. |
| m08  | 2026-04-24 | [2048,512]×2 k_max=100 seed=20 (GCP L4)      | 139.05| pending           | pending         | Wider training horizon; RMSE 11.8 (vs m07's 8.0) — wider k_max inflates absolute MSE |
| m04  | 2026-04-25 | Bellman-from-scratch (Kaggle P100, 9.4h)     | 0.097*| 15/21 NISS-off    | —               | *Bellman target MSE, not comparable to walk-depth MSE. Mean predicted distance 15.5. Solve rate slightly below m07 NISS-off (18/21). |
| m09  | 2026-04-24 | [2048,512]×2 k_max=80 Muon lr=2e-2 1000 ep   | 65.47 | —                 | —               | Muon converges 4× faster than AdamW (m07) but plateau is similar |
| m10  | 2026-04-25 | Muon lr=1e-2 (LR sweep)                      | 64.92 | —                 | —               | LR sweep variant |
| m11  | 2026-04-25 | Muon lr=5e-3 (LR sweep)                      | 65.23 | —                 | —               | LR sweep variant |
| m12  | 2026-04-25 | Muon lr=5e-2 (LR sweep)                      | 66.81 | 9/21 NISS-off     | —               | Best ckpt (ep 99) had MSE 59.06 — *lowest of any model* — but stratified eval shows only 9/21 solves (worse than m07's 18/21). Lower MSE != better ordering. Walk-depth MSE is a noisy proxy for solve rate. |

---

## m01 — first port (fast recipe, k_max=40)

- **Config**: `configs/m01_fast_k40.yaml`
- **Model**: `cayley.model.ResMLPDistance`, encoding=embedding, embed_dim=16, hidden=[1024, 256], 1 ResBlock. 2,366,849 params.
- **Recipe**: MSE, Adam lr 2e-3 + cosine decay, batch 16384, 1M samples/epoch, non-backtracking walks (n_back=1), k_max=40, 200 epochs, bf16 + torch.compile + fused AdamW.
- **Checkpoint**: `models/m01_fast_k40/epoch_0199.pt`.
- **Runtime**: 100 s (0.5 s/epoch after compile + 20 s first-epoch compile).
- **Loss**: 139 → 9.76 by ep 199. RMSE ≈ 3.1 on target range [1, 40] — comparable to IHES small_e3 tier.
- **Solve 1 (smoke, beam 32k, pid 0-19, ma=2)**: 19/20 by model, total 500,554 (–18 vs sample). Submission: `submissions/m01_b32_l20.csv`.
- **Solve 2 (stratified 10/bucket, beam 32k, ma=2)**: 2/101 by model, 40 min wall time. Submission: `submissions/m01_b32_s10.csv` → 500,564 total. Per-bucket: bucket 0 had 3/10 solves averaging 27.3 moves; all other buckets 0/10.
- **Analysis**:
  - Model is competent only inside its training distribution (distance ≤ 40). Any puzzle with scramble depth > 40 (≈bucket 1 onward) is out-of-distribution — beam search drifts.
  - Easy puzzles solved nearly optimally where model engages.
  - Beam 32k at max_steps=80 ma=2 costs ~2s/easy, ~150s/hard — most time goes to unsolvable-by-model hard cases that exhaust all steps.

## m02 — widen training depth (k_max=80, 2000 epochs)

- **Config**: `configs/m02_k80_long.yaml`
- **Model**: same arch as m01 (2.37M params).
- **Recipe**: same as m01 except `k_max=80`, `n_epochs=2000`, checkpoint every 100 ep.
- **Checkpoint**: `models/m02_k80_long/epoch_1999.pt` (dir name still `k80_long/` until current full solve finishes; rename later).
- **Runtime**: ~20 min (0.6 s/epoch after compile).
- **Loss**: 859 → 66.16 by ep 1999. RMSE ≈ 8.1 on target range [1, 80]. Relatively worse variance-explained than m01 — wider depth range is genuinely harder.
- **Solve 1 (stratified 5/bucket, beam 16k, ma=1)**: 2/51 by model (only bucket 0-99). Total 457,802 (essentially = pp_fallback floor 457,810 minus 8 moves). Submission: `submissions/m02_b16_s5.csv`. Wall time 225 s.
- **Solve 2 (full 1001, beam 32k, ma=2)**: **IN PROGRESS** — started 2026-04-24 10:22. Output will be `submissions/full_k80_b32.csv` (will rename to `m02_b32_a2.csv`).
- **Analysis**:
  - Wider k_max alone did not fix the medium-hard regime: going from 2/101 (m01 at beam 32k) to 2/51 (m02 at beam 16k) is consistent with "beam size is the binding constraint, not training depth". Model heuristic noise (RMSE 8) is larger than the per-step distance delta (≈1 out of 24 neighbors is a good move, 23 are sideways or away).
  - Bucket 0 model-averages (m01: 27.3, m02: 15.0) suggest the k_max=80 model is MORE accurate where it solves — it's just unable to escape beam drift on deep puzzles.
  - Implication: next lever is (i) bigger model for less heuristic noise, AND/OR (ii) much wider beam (65k+), AND/OR (iii) Q-function heads to score 24 neighbors in one pass (more discriminative).

---

## Same-face post-processing (applies to all submissions)

- Every Megaminx face has rotation order 5: X^5 = identity, X^4 = -X, X^3 = -X·-X.
- `megaminx.post_process.reduce_same_face_runs` + `cancel_adjacent_inverses` run to fixpoint.
- On raw sample_submission.csv: -42,762 moves (8.54%) with all 1001 still solving → `data/pp_fallback.csv` is our new fallback floor (457,810).

---

## BFS-d6 window replacement (2026-04-24)

- Extended the BFS table to depth 6. Tuple-keyed dict wouldn't fit in 16 GB; introduced
  `megaminx.bfs_bytes.BfsBytesTable` with bytes-keyed storage (120 bytes per state vs
  ~3.6 KB for tuple-of-int). Built 19,352,405 states in 6.5 min (2.55 GB pickle).
- Applied to pp_bfs5_fallback: -843 moves (0.20%), new floor 414,678.
- Per-puzzle time: 0.19 s (vs 1.77 s for d5 the first time). Total 184 s for 1001 puzzles.
- Did NOT submit — rank stays at #8 (414,678 > #7's 414,305 by 373 moves). Reserving
  today's submission slots for m03 + innovation stack.
- Post-processing is now saturated at this d-level pair. Further gains require a better
  base path (i.e., a model) or ReduceFactor DAG (non-greedy combinatorial shortcutting).

## BFS-d5 window replacement (2026-04-24)

- Built Megaminx BFS-d5 table: 1,376,945 states, 352 MB on disk, 16 s to build
  (cayley.bfs_table on a 4090 Laptop, canonical pruning on). Layer counts
  24/408/6208/90144/1280160 match the public Megaminx growth function exactly.
- Applied `reduce_factor_via_bfs_table` (sliding-window replacement, max_window=6) to
  pp_fallback: -42,289 moves (9.24%) → new floor 415,521 at `data/pp_bfs5_fallback.csv`.
- Cost: ~30 min wall (1.77 s per puzzle, CPU-bound).
- Note: window replacement hits the first improvement per pass and restarts; the 6 largest
  savings all came from late-pid (scramble-heavy) puzzles. Easy puzzles (pid <~400) gain
  almost nothing — their paths are already near-optimal after same-face reduction.

---

## Decision log

- **2026-04-24**: started project by porting shared `cayley.*` modules via duck-typing (Megaminx class mirrors PictureCube). No library fork needed.
- **2026-04-24**: confirmed test.csv is ordered by scramble walk length (sum 1..1000 ≈ 500,500). Stratified 10/bucket replaces first-N as smoke-test default to cover full difficulty range.
- **2026-04-24**: adopted m0N_<slug> naming for configs, models, submissions; EXPERIMENTS.md as canonical ledger; `git init` + private GitHub repo at https://github.com/Erlemar/cayley-solvers.
- **2026-04-24**: chose NOT to submit 457,802 (matches pp_fallback floor) — wastes a daily Kaggle submission slot with no leaderboard signal. Submitted 415,521 (pp_bfs5_fallback) instead after +BFS-d5 window replacement → rank #8/16.
- **2026-04-24**: multi-agent research + literature synthesis. Anchored priorities on DeepCubeA / CayleyPy paper + our IHES wins, not public-kernel votes (top-voted kernels are educational, not LB-coupled).
- **2026-04-24**: policy — don't submit public community-merged results (carried over from IHES).
- **2026-04-24**: innovations-first strategy: verify IHES-proven techniques (NISS, int8, adaptive beam, BFS-d5 post-proc, Bellman) on Megaminx before leaning on canonical big-arch baseline.
- **2026-04-24** (user feedback): Kaggle m03 ran 7h+ without visible progress (Kaggle only exposes stdout on completion). Left running; user decided to not interrupt. Added backlog items T1 (generous early stopping) + T2 (warm-restart scheduler) so future long runs stop themselves when plateaued.
- **2026-04-24**: m03 completed at 8.5h (P100 at 7.6 s/epoch — 30× slower than my estimate of 0.25 s/epoch). Final MSE 63.74, essentially tied with our local m07 (64.22). Kaggle quota hit (30h/week) — m05/m06 push blocked; pivot to training them locally on 4090.
- **2026-04-25** (user feedback): standardize stratified eval at 5/bucket (51 puzzles), not 2/bucket. 2/bucket was noisy at the high-solve-rate end — m04 (15/21) vs m07 (18/21) differed by less than the sample's expected variance. 5/bucket gives ±3 noise floor on 51 puzzles.
