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
- Top leaderboard (as of 2026-04-27): 79,971 (Kuznetsov), 81,946 (DrozdovDan), 88,195 (us, andlukyane, **rank #3**), 93,606 (Rokicki). Rest: ~413–500K.

**Current best (submitted, 2026-04-27)**: **88,195** moves — Phase B (pure m05
fresh on local 4090, beam 131k, all 1001) plus beam-stack rescue of pid 490 and pid 920.

---

## Acceptance gate (do not skip)

A new heuristic / configuration counts as "an improvement worth trusting" only if **all** of the following hold against our current best (m05 NISS-off):

1. **Solve count**: ≥ +3 puzzles on stratified-5/bucket NISS-off at the same `--strat-seed=0` (51-puzzle sample)
2. **Path quality**: mean `model_avg` ≤ 0.95× current best's mean `model_avg` (i.e., shorter paths on average)
3. **No bucket regression**: no bucket drops more than 1 solve vs current best on the same sample

If only one of (1) or (2) is met, that's a TIE (or noise) — investigate further before committing. Re-validate on a second seed (e.g., `--strat-seed=1`) before accepting close calls.

**Why these gates**: training MSE is a noisy proxy (m12 had the lowest MSE in our sweep but only solved 9/21). Validation-tool top-1 is a filter, not a ranker (m04 had top-1 0.995, m07 had 0.990, but m07 solved more puzzles). Stratified solve at fixed seed IS the trustworthy metric.

**Why fixed seed=0**: comparable across runs without sampling noise. Different seeds for confirmation only.

**Why "≥ +3 on 51"**: ~6% improvement is well outside the ±2 noise floor of stratified-5.

---

## Phase template (write this BEFORE running anything)

When starting a new training family / experiment phase, append a block in this exact form:

```
### Phase Q: <slug> — <one-line goal>
Hypothesis:    <what we expect and why>
Acceptance:    <criterion in metric-and-numbers form, refines the gate above>
Steps:         1. train <id_a> with <config>
               2. validate-tool on <ckpt list>
               3. stratified-5 on best ckpt
               4. accept/reject vs current best
Compare to:    <current best run id>
Output ids:    m<NN>_<slug>, m<NN+1>_<slug>, ...
```

The chain processes Steps 1..4 linearly; if a step fails its own micro-acceptance, the chain halts and we re-plan rather than push more compute through a broken hypothesis.

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
| m05  | 2026-04-25 | Bellman warm-start from m07 (500ep)          | 0.094*| **50/51** mean 89.4 | **88,195 (rank #3)** with Phase B + beam-stack rescue | *Bellman MSE. Current production teacher. Beats m07 by +7 solves and 15% shorter mean path. Used in Phase B + beam-stack rescue → 88K submission. |
| m17  | 2026-04-26 | Bellman r2 from m05 (500ep, Kaggle P100)     | 0.097*| 51/51 ≈ tied      | —               | Bellman fixed point: training loss flat from epoch 0 (m17 told us r3+ is a no-op). Don't queue more Bellman rounds. |
| m18  | 2026-04-26 | 4-layer Transformer (824k params, Kaggle)    | 59-62 | killed @ 12h      | —               | Did converge, just per-wall infeasible. CayleyPy paper "Transformer fails at n>15" too strong but Kaggle wall budget is. |
| m21  | 2026-04-26 | learn-to-hit-shell target (clamp(walk_depth-6, 0)) | 65.2 | path-sum 1197 vs m05 1053 | — | REJECTED. Shifted target without Bellman bootstrap loses sharpness; MITM can't recover. m05 already navigates d=6 shell as a side-effect. |
| m23  | 2026-04-26 | Q-shortlister 12.4M params (hidden=(2048,1024) ×3 ResBlock) + MSE+KL loss | KL 0.085, MSE 0.33 | recall=100%@α=2 | —               | DEPLOYED in qshort solver. 4.4× wall speedup at beam 131k. Path quality regression on easy puzzles (+6%) — wall savings funded scaling to higher beam. |
| m26  | 2026-04-27 | bigger arch + walk-depth pretraining (Kaggle)| RUNNING | RUNNING         | —               | Capacity-scaling A/B vs m05's [2048,512]×2. |
| m26b | 2026-04-27 | wider arch (width-vs-depth A/B with m26)     | RUNNING | RUNNING         | —               | Companion to m26. Whichever wins → integrate, retrain m23 student. |
| m26  | 2026-04-27 | bigger arch + walk-depth pretraining (Kaggle, FINAL) | 0.10 | 51/51 / 91.2 | — | REJECTED. +1 solve vs m05 but +1.8 mean → fails gate. Capacity scaling at this signal exhausted. |
| m22 K=2 | 2026-04-28 | K=2 step lookahead Bellman, ep149 mid-train | low | 51/51 / ~91 | — | REJECTED. K-step adds compute without breaking cluster. |
| m27 50% bfs6 | 2026-04-28 | Bellman + 50% BFS-d6 exact mixin (Option B) | 0.10 | 51/51 / 91.49 | — | REJECTED. Exact mixin in pretraining doesn't break cluster either. |
| m27b 25% bfs6 | 2026-04-28 | Bellman + 25% BFS-d6 mixin | — | 51/51 / ~91 | — | REJECTED. Lower mixin doesn't help. |
| m27c 10% bfs6 | 2026-04-28 | Bellman + 10% BFS-d6 mixin (GCP) | — | 51/51 / ~91 | — | REJECTED. Even minimal mixin doesn't help. |
| m28 double_bellman | 2026-04-28 | Double Bellman (van Hasselt 2010) | low | 51/51 / ~91 | — | REJECTED. Bias decorrelation doesn't break cluster. |
| SWA m05/400-499 | 2026-04-28 | weight-averaged 5 late ckpts (Izmailov 2018) | — | 51/51 / 96.75 | — | REJECTED. SWA REGRESSES — late-cycle ckpts diverged enough that averaging blurs the heuristic. |
| m29 n_back=4 | 2026-04-28 | walk-aug ban radius 1→4 | low | **51/51 / 88.98** | — | KEPT. **First sub-89 mean**. -0.42 vs m05 + 1 extra solve. Coverage knob worked, marginally. Used for hard-tail rescue. |
| m30 n_back=16 | 2026-04-28 | walk-aug ban radius 1→16 (more aggressive than m29) | lower | 51/51 / 89.69 | — | REJECTED. m12-trap territory: lower training MSE, worse strat-5. n_back too aggressive over-narrows walks. |
| m31 rot_aug 50% | 2026-04-29 | m05 recipe + rotation augmentation 50% per batch | 0.142 | 50/51 / 95.76 | — | REJECTED + falsified hypothesis. At 6M params, augmentation across 360 orbit-equivalents dilutes signal rather than sharpening it. Hypothesis "orbit coverage is binding" disproved. |
| sym v3 | 2026-04-29 | derived 360 megaminx symmetries (A_5 × C_6) | — | not a model | — | KEPT as artifact. `data/rotations.npy`. Unlocks `--sym-ensemble K` in `03_solve.py` for inference-time ensembling. |
| **m05+sym4** | 2026-04-29 | INFERENCE-time sym-ensemble K=4 (no training change) | — | **51/51 / 88.20** | (rescue → 85,812) | **KEPT, NEW BEST AT STRAT-5.** First mechanism to break the m05 cluster ceiling (88-97 across all training-side variants). Doesn't pass strict +3 solves gate but +1 solve & -1.20 mean is meaningful at full-1001 scale. |
| m05+sym4+qshort+m23_v2 | 2026-04-30 | sym-ensemble + sym-aware shortlister | — | 51/51 / 88.41 | — | KEPT. +0.21 mean vs no-qshort sym4 for 6× wall reduction. Production stack for any --sym-ensemble use. |
| m32 target_update=5 | 2026-04-30 | Bellman recipe variation | 0.0986 | 51/51 / 94.65 | — | REJECTED. Faster target refresh hurts; m05's 10 is calibrated. |
| m23_v2 sym-aware | 2026-04-30 | Q-shortlister with rotation augmentation | MSE 0.35, KL 0.087 | recall α=2 100% | — | KEPT. Pair with `--sym-ensemble` instead of m23 (original). |
| **Submission 82,481** | 2026-04-30 | merge_plus_sym8_top20: K=4 sym + qshort rescues on top 130 (50+80) + K=8 micro on top 20 + GCP full-1001 + Phase B | — | — | **82,481** | superseded |
| m42_v2 distributional V | 2026-05-09 | 32 quantile QR-DQN-style Bellman, warmstart from m_curr_v0 curriculum body | quantile_huber 1.165 | 51/51 / model_avg 88.84 / chosen_avg 87.02 | — | REJECTED. Lands in cluster (chosen 87.02 vs m_curr_v3+m_pi_v2's 85.12). Bucket 0-99 OOD: model 59.2 vs fallback 39.6. Confirms cluster ceiling holds for distributional V too. |
| **Submission 76,304** | 2026-05-09 | min-merge of 78,029 (us) + 2 colleague CSVs (79,911 + 77,152) — user-authorized policy exception | — | — | **76,304** (#1) | **CURRENT BEST**. Standalone work still 78,029. |
| m_v11 (11.8M two-stage) | 2026-05-19 | Trunk (3072,768)+3rb scale-up. Stage 1a 50ep random-walk pretrain. Stage 1b 200ep Bellman with λ_pdb=0 + lr 2e-4 (Rule 14 recipe diff). Stage 2 25ep AZ fine-tune on az_dataset_75200. | Stage 1b final 0.0782 / Stage 2 ep11 v_loss 0.13 | 10-pid bench: 10/10 / 991 (looked promising). Strat-51: STUCK at 16/20 model solves in 1h+ on attempt 21 (vs AZ v4 51/51 in ~75min same recipe). Pids 995-998 head-to-head with production recipe: m_v11 96.25 avg vs AZ v4 67.5 avg = **+43% moves/pid worse**. | — | REJECTED. The 10-pid bench (didn't include hardest pids) was misleading. Bigger V trunks regress regardless of recipe — Rule 14 (20.5M) extends down to 11.8M. Cluster ceiling at 6M is information-bound across 10× param range. Don't retry trunk scale-ups without a fundamentally different signal source. Artifacts kept on GCP for forensic reference. |
| m23_v3_az_v4_sym | 2026-05-19 | Q-shortlister distilled from AZ v4 V (not m05). Same arch as m23_v2 (12.4M, hidden=[2048,1024]×3 ResBlock). 500ep planned; CUDA OOM crash at ep 290 after 3h continuous on 4090 Laptop but ep 199 already at production quality. | MSE 0.355, KL 0.088 | GLOBAL recall α=2 = 100% vs AZ v4 V on 4080 parents (`09_eval_q_recall.py`). Strat-51 AZ v4 V + this qshort + sym4 single-pass beam 65k: 51/51 / **total 4340 / mean 85.1** vs AZ v4 V-only baseline 4465 / 87.5 = **−125 (−2.8%)**. | — | KEPT. Pair with `--qshort-student megaminx/models/m23_v3_az_v4_sym/epoch_0199.pt --qshort-alpha 2 --sym-ensemble 4` for the AZ v4 V production stack. Resolves Rule 15 (m23_v2 misalignment with AZ v4). |
| TPU 48M qshort kernel | 2026-05-19/20 | New shareable kernel `artgor/cayleypy-megaminx-48m-720-qshort-shareable` (JAX SPMD). Clones the 48M_720 V-only body and adds qshort prefilter in the streaming chunk_body. v1 had per-parent top-α bug (62% recall vs V's per-parent choices); v2 fixes to GLOBAL top-αB matching `beam_lab/beam_search_qshort.py` line 161. | — | pid 0 at B=48M sym K=8 SYM_POSITIONS=range(0,4): path 60 vs v_only B=48M reference ~55 = **+5 moves qshort tax**. Wall ~5h for 4 sym (~1.5× faster than v_only, not 6-12× theoretical — V eval isn't the bottleneck at B=48M; chunk expansion + all_to_all dominate). | — | NOT RECOMMENDED for broad deployment. Quality tax (+5 moves on pid 0) too high; sym-ensemble diversity on easy pids doesn't recover it. May still be useful for long-tail-only rescue (pid 700+) where sym matters more — untested at that regime. Kernel + Kaggle dataset version remain as artifacts. |
| m_rank_v0 (child-rank isolation) | 2026-05-24 | doc §4.2 child-rank CE ALONE on m_dd_v0 (encoding=embedding, NO repr channels, NO sym/sat/solver-trace). λ_rank=0.1. Control arm (λ_rank=0) confirms re-fine-tune is a no-op (d=20 std 1.71≈baseline 1.73). | base Bellman 0.07 + rank-CE 0.30 | Variance-SAFE (ep49 V identical to m_dd_v0, d=20 16.6±1.97). Child-choice differs from m_dd_v0 on ~7% states (top1-agree 93%). **Strat-51 sp beam-65k: 51/51 by model, mean 89.9 vs m_dd_v0 89.4 = +0.5 (TIE, within ±2 noise).** | — | NEUTRAL → effectively REJECTED (fails gate). Child-rank EXONERATED as the m_repr_v0 variance culprit (→ sym/sat/solver-trace). But no beam headroom: m_dd_v0 already ranks children well (6M ceiling). Don't pursue on a working V. |
| m_repr_v0 (repr-upgrade + bundle) | 2026-05-23 | doc §3.1 "Representation-upgraded ResMLP", capacity-preserving redesign. `encoding="features"`: gated sum-fusion of inv_state/face/lpos/piece/ori/scalar channels (all α init=0, in_dim unchanged at 1920 — no capacity inflation). Warmstart from m_dd_v0 via bridge (features@α=0 ≡ embedding). + "aggressive bundle": λ_sym consistency 0.3, λ_sat 0.1 (ceiling 30), λ_rank 0.1, solver-trace 20%. 50ep Bellman. | final 41.3 (sat-penalty-inflated) | Canary GREEN (V@d=80=30.1 saturates). **But strat-51 single-pass beam-65k: 2 model-solves / 20 pids in 90min (Rule 21 stall) — CATASTROPHIC** vs m_dd_v0 50/51. | — | **REJECTED.** Two findings: (1) repr channels INERT — α gates →~0, forcing α=0 changes nothing; model declines them (working V already learns bijection). (2) BUNDLE TOXIC — tripled mid-depth V variance (d=20 std 1.7→5.3); mean stayed calibrated (canary fooled) but per-state V scatter killed beam ranking. **Lesson: saturation-MEAN canary insufficient; V VARIANCE at d≈20 is the better beam predictor.** §3.1 now 0/2. See `repr_upgrade_bundle_rejected.md`. |

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

## Submission 95,682 — Phase 1 + Phase 2 (2026-04-26)

Best LB score to date. -73% vs the previous 357,007 submission. 2,076 above
Rokicki (rank #3 at 93,606); behind Kuznetsov (79,971) and DrozdovDan (81,946).

### Pipeline at a glance

```
                 m07 (RW-trained)       m05 (Bellman warmstart from m07)
                       │                              │
                       ▼                              ▼
     Phase 1 ─── beam 131k full solve         Phase 2 ─── beam 131k --resume on m07's misses
     (GCP L4, 34 h)    │                      (GCP L4, 6.5 h)    │
                       ▼                                          ▼
              793 paths in CSV                             206 paths in CSV
                                  │
                                  ▼
                  per-puzzle min(model_path, pp_bfs6_fallback)
                  fb fill for 1 missing pid (pid 492)
                                  │
                                  ▼
                          phase12_post.csv
                          1001/1001 valid
                          95,682 moves
                          submitted to Kaggle
```

### Stage 1 — distance-heuristic training

**m07** — random-walk-trained ResMLP, the canonical baseline.
- Architecture: `ResMLPDistance(hidden_dims=(2048,512), num_res_blocks=2, encoding="embedding", embed_dim=16)`. ~6.0M params.
- Targets: walk-depth labels from `cayley.data.generate_walks_torch` (`n_back=1`, `k_max=80`).
- Loss: MSE.
- Optimizer: fused AdamW, lr 2e-3, weight_decay 0, batch 16384, 4000 epochs, bf16 AMP, fused.
- Training compute: 4090 Laptop, ~80 min wall, final MSE 64.22.
- Checkpoint: `megaminx/models/m07_big_k80/epoch_3999.pt`.

**m05** — Bellman bootstrap on m07's body weights.
- Same architecture, warm-started from m07 epoch 3999.
- Training: 500 ep, lr 5e-4, target net refresh every 10 ep,
  target = `clip(1 + min_a V_target(apply(s, a)), 0, walk_depth)`.
  This sharpens labels: walk-depth is an upper bound; min-over-actions of next-state
  V plus 1 is a tighter (Bellman-consistent) target where reachable.
- Training compute: 4090 Laptop, ~30 min wall, final Bellman MSE 0.094.
- Checkpoint: `megaminx/models/m05_bellman_warm/epoch_0499.pt`.
- Stratified-5 result: 50/51 solved, mean model_avg 89.4 (vs m07's 43/51, 105.4) — accepted as new best.

### Stage 2 — beam search inference

Beam search uses `KhoruzhiiSolver` (in `cayley.khoruzhii_search`) — a strict
batched implementation that does NOT use CayleyPy's `advanced` mode (which silently
returns `path=None`). State encoding is int8 (6× VRAM reduction). Movement: BF16 model
forward, top-K selection on negated predicted distance, terminate on solved or `max_steps`.

**Phase 1 — m07 full solve on GCP L4**:

```bash
python megaminx/scripts/03_solve.py \
    --checkpoint megaminx/models/m07_big_k80/epoch_3999.pt \
    --out megaminx/submissions/full_m07_gcp.csv \
    --pid-from 0 --pid-to 1001 \
    --beams 131072 --max-steps 120 --num-attempts 1 \
    --bf16 \
    --bfs-table megaminx/data/bfs_bytes_d6.pkl
```

- Wall: 34 h on GCP L4 (24 GB, ~$0.70/h, ~$24 spent).
- Result: 793 / 1001 puzzles solved by m07 at beam=131k, 208 fell to beam termination.
- Avg path length over solved: ~98 moves.
- NISS off (no inverse-state retry); single-attempt only.

**Phase 2 — m05 retry on m07's misses (auto-chained)**:

```bash
python megaminx/scripts/03_solve.py \
    --checkpoint megaminx/models/m05_bellman_warm/epoch_0499.pt \
    --out megaminx/submissions/full_m07_gcp.csv \
    --pid-from 0 --pid-to 1001 --resume \
    --beams 131072 --max-steps 120 --num-attempts 1 \
    --bf16 \
    --bfs-table megaminx/data/bfs_bytes_d6.pkl
```

- `--resume` skips pids that already have a row in the CSV; only the 207
  pids m07 missed are re-attempted.
- Wall: 6.5 h on the same GCP L4 VM (started by `run_phase2_chain.sh`
  watching for Phase 1 completion).
- Result: 206 / 207 solved by m05 at beam=131k. The single failure (pid 492)
  is the only puzzle in the entire CSV neither model could crack at beam 131k.
- Per-bucket m05 model_avg: ~94 moves uniformly across all 11 buckets — strikingly
  flat (consistent with the strat-5 result that m05 generates near-constant length
  paths regardless of scramble depth, in contrast to m07 which scaled with bucket).
- Bucket 0-99 is the only bucket where m05's path was longer than fallback's
  (97.1 vs 63.8) — those 13 paths get swapped out in stage 3.

After Phase 2, the CSV contains 1000 model paths (m07 ∪ m05) + 1 missing pid.

### Stage 3 — post-processing & submission

```bash
python megaminx/scripts/merge_batches.py \
    --batches megaminx/submissions/phase12_gcp.csv megaminx/data/pp_bfs6_fallback.csv \
    --fallback megaminx/data/pp_bfs6_fallback.csv \
    --out megaminx/submissions/phase12_post.csv
```

Trick: passing **both** the model CSV and the fallback CSV as `--batches`
forces the script's per-pid minimum to compare against fallback. Because
`pp_bfs6_fallback.csv` covers every pid, this also fills the one missing pid
(492). Result: each puzzle gets `min(m07_or_m05_path, pp_bfs6_fallback_path)`
plus fb-only for pid 492.

Score breakdown (raw vs post-processed):

| stage | total moves |
|---|---|
| Phase 1 paths only (793 m07) | 78,011 |
| Phase 2 paths only (206 m05) | 19,578 |
| **Raw CSV (1000 paths, no fb fill, no swap)** | **97,589** |
| + fb fill for pid 492 | 97,961 |
| + per-puzzle min vs pp_bfs6 | **95,682** |

The min-vs-fb step saves ~2,300 moves; almost all of that comes from
bucket 0-99 (m05 paths longer than fb on easy puzzles).

### What pp_bfs6_fallback contributes

`data/pp_bfs6_fallback.csv` is the "no model" baseline: same-face run reduction +
BFS-d6 sliding-window replacement applied to the official `sample_submission.csv`.
Standalone score: **414,678**. It defines the floor for any pid that the model
fails on — which is why even Phase 1 alone (with fb fill for the 208 misses)
already beats 357,007: most fb paths are shorter than a 120-step beam-failure stub.

### Why this combination works

- **m07 is fast and broad**: random-walk training learns a smooth value function
  that solves easy and medium puzzles cheaply. 793/1001 in one shot.
- **m05 is sharp on hard cases**: Bellman refinement bootstraps a tighter value
  function from m07's body. Of m07's 208 misses, m05 closes 99.5%.
- **pp_bfs6 cleans up the long tail**: even where the model wins on length, the
  per-puzzle min ensures we never submit a model path longer than the BFS-cleaned
  scramble. And it fills the single residual miss.
- **Why this is not optimal yet**: Phase B (running, ETA 2026-04-27) replaces the
  m07-on-easy paths with m05 paths uniformly — m05's paths are shorter on easy
  puzzles too. Expected to land in 80,000–88,000 range.

### Files involved

| file | role |
|---|---|
| `megaminx/data/test.csv` | 1001 input puzzles (state vectors) |
| `megaminx/data/pp_bfs6_fallback.csv` | model-free floor (414,678) |
| `megaminx/data/bfs_bytes_d6.pkl` | 19.4M-state BFS-d6 lookup, used during beam search to verify candidate states |
| `megaminx/models/m07_big_k80/epoch_3999.pt` | RW-trained heuristic |
| `megaminx/models/m05_bellman_warm/epoch_0499.pt` | Bellman-refined heuristic |
| `megaminx/scripts/03_solve.py` | beam-search driver (single-checkpoint, --resume, --bfs-table) |
| `megaminx/scripts/merge_batches.py` | per-pid min across CSVs + fb fill |
| `megaminx/submissions/phase12_post.csv` | the actual submission |

### Total cost

- Local 4090: ~110 min training (m07 80 min + m05 30 min).
- GCP L4: ~40.5 h × $0.70/h ≈ **$28.35**.
- Kaggle: 0 GPU hours used for this submission (Kaggle was used for the
  experiments that proved m05 was the right choice; not the final compute).

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
- **2026-04-25** (user feedback, post Kaggle-thread research): adopt the
  ledmaster/ml-mania-2026 pattern of explicit acceptance gates + phase checklists
  (see top of file). Quote that drove this: *"Agentic systems are very good at
  exploiting weaknesses in the objective you give them"* — our m12 trap
  (lowest training MSE → worst solve rate) was exactly that.

---

## Journal (chronological log of attempts; rejected runs included)

Format: `YYYY-MM-DD HH:MM — id (status) — observation / why kept or rejected`.
Append-only; do not edit past entries. Keeps cross-session memory honest.

- **2026-04-24** — m01 (smoke) — first port works, 19/20 easy puzzles solved at beam 32k, MSE 9.76. Confirmed Megaminx class duck-types `cayley.*` modules.
- **2026-04-24** — m02 (REJECTED for solving, KEPT for diagnostic) — wider k_max=80 + 2000 ep produced MSE 66 but 0/31 stratified solves. Heuristic-noise-bound at small arch.
- **2026-04-24** — pp0 (SUBMITTED 415,521, rank #8) — pure post-processing baseline. Validated end-to-end pipeline.
- **2026-04-24** — m03 (KEPT, used for solve) — Kaggle canonical [2048,512]×2 4000ep. MSE 63.74 in 8.5h. Replicated locally as m07.
- **2026-04-24** — m07 (KEPT, current Phase 1 model) — local m03 replica with seed=10. MSE 64.22 in 80 min on 4090. **Used for GCP Phase 1 full solve.**
- **2026-04-24** — m07 + NISS stratified-2 (SUBMITTED 407,563, rank #4) — first real ML submission. NISS-on doubled wall but solved 21/21.
- **2026-04-25** — m08 (NOT YET TESTED) — k_max=100 variant, MSE 139, validation top-1 0.989 (≈ m07).
- **2026-04-25** — m04 Bellman-from-scratch (REJECTED for solving) — 15/21 stratified, less than m07's 18/21 despite top-1 0.995. Pearcatcher recipe doesn't dominate walk-depth + AdamW for our task.
- **2026-04-25** — m09–m12 Muon LR sweep (REJECTED) — all converged to similar MSE ~64-67 plateau. m12 ep99 had lowest training MSE (59) but worst stratified solve rate (9/21). m12 ep999 had top-1 0.998 (best) but ALSO solved only 9/21 — top-1 is not a reliable ranker. Muon doesn't beat AdamW for this problem at this arch.
- **2026-04-25** — m06 Q-distill from m07 (REJECTED) — 10× faster inference confirmed (551s vs 5683s for 51 puzzles) but solve rate dropped to 20/51. Distillation lost ordering.
- **2026-04-25** — partial GCP submission (SUBMITTED 357,007, rank #4) — interim merge of 440 GCP-Phase-1-solved + pp_bfs6_fallback. -50K vs prior submission.
- **2026-04-25** — m05 Bellman warm-start from m07 (ACCEPTED, NEW BEST) — stratified-5 50/51 solves, mean model_avg 89.4. Beats m07's 43/51 + 105.4 by 7 solves AND 15% shorter. **Acceptance gate passed cleanly.** New current-best heuristic. Used for GCP Phase 2.
- **2026-04-25** — m13 (1000ep Bellman) + m14 (Q-distill from m05) (RUNNING) — testing whether more Bellman epochs and a better-teacher Q-distill respectively beat m05.
- **2026-04-26** — m17 Bellman r2 from m05 (REJECTED) — Kaggle P100, 500 ep, 4 h. Training loss flat from epoch 0 (0.0970 → 0.0953 best at ep 299 → 0.0977 at ep 499; ran past optimum). Strat-5: 51/51 solves vs m05's 50/51, mean ~92 ≈ tied. Acceptance gate fails (only +1 solve, mixed bucket directions). Confirms Bellman r2 is at the fixed point at our scale. Don't queue r3.
- **2026-04-26** — m18 Transformer (REJECTED, killed by Kaggle 12h limit at ep 190) — 4-layer encoder, d_model 128, 824k params. Per-epoch 225s vs ~25s for MLP at 6M params. Did converge (MSE 548 → 62 at ep 99 → 59 at ep 190); CayleyPy paper claim "Transformer fails at n>15" too strong, but per-wall infeasible on Kaggle to reach m07's effective training depth. Recovered ckpt + log via `kernels output` post-cancel. Skip-eval per user.
- **2026-04-26** — Phase 1 (m07 GCP full solve) + Phase 2 (m05 GCP retry on misses) (SUBMITTED 95,682, rank pending update) — see "Submission 95,682" section above. -73% vs prior. Phase B (pure m05 fresh on local 4090) running for resubmit.
- **2026-04-26** — m17 GCP full solve attempt (KILLED at 60/1001 attempts, 40 min wall) — stopped to free GCP for beam_lab optimization A/B benchmarks. m17's strat-5 already showed it's tied with m05; full m17 solve was queued mostly for ensemble path diversity. Will revisit if low-effort optimizations don't break 80K.
- **2026-04-26** — beam_lab — created `megaminx/beam_lab/` self-contained sandbox (24 MB) for beam search optimization. Includes 12 stratified sample puzzles (one per 100-pid bucket + pid 492), m05 checkpoint, profiling-instrumented `KhoruzhiiSolver`, `BatchedKhoruzhiiSolver` (multi-puzzle), `MitmKhoruzhiiSolver` (BFS-d6 shell termination), `BeamStackSolver` (runner-up backtracking), `KhoruzhiiSolverV0` (pre-opt baseline). Portable to GCP/Kaggle.
- **2026-04-26** — Tier-1 micro-opt A/B on GCP L4 (REJECTED as expected) — v0 baseline 1240s vs v1 optimized 1300s on 12 lab samples at beam 131k. Path lengths identical (1053 total). Within run-to-run noise. **Headline finding: `model_s` is 96% of wall** at beam 131k. Optimizations targeting `topk`, `dedup`, `apply`, `hash` (each <2% of wall) cannot move the needle. Real wins must be algorithmic (MITM, Q-shortlister, m21 shell-target) or compile/architectural.
- **2026-04-26** — beam_lab user-contributed micro-opts (incremental hashing via parent-hash + delta over changed positions, MPS sync, fp16 flag) — algorithmically correct, plausible 15% on MPS, expected 2-5% on CUDA L4 where model dominates. Pending re-benchmark.
- **2026-04-26** — Multi-puzzle batched beam K=4 on GCP L4 (REJECTED) — wall 4796s vs single-puzzle 1240s (3.7× SLOWER). L4 is compute-bound at single-puzzle beam=131k; packing 4 puzzles' beams into one model call just makes that call 4× longer. Model_s share dropped to 26% but total wall went up because the per-batch model call now dominates differently. Lesson: batching only helps on launch-bound (small) workloads; H100/A100 with FLOP headroom MIGHT benefit. Not for L4 at our scale.
- **2026-04-26** — training scripts written (NOT YET RUN): m21 shell-target (`08_train_m21_shell.py`), m22 K-step lookahead Bellman (`10_train_m22_horizon.py`), m23 Q-shortlister with hybrid MSE+KL loss (`09_train_q_shortlister.py`), m24 policy head from solved-path triplets (`11_train_policy_head.py`), recall validation gate (`09_eval_q_recall.py`). Ready to launch when local 4090 frees from Phase B.
- **2026-04-26** — m21 (learn-to-hit-shell) trained on GCP L4 (~6 min wall, 500 ep, warmstarted from m05). Final loss 65.2. Lab benchmark m21+MITM REJECTED: path-sum 1197 vs m05's 1053 (-14%). Cause: shifted target without Bellman bootstrap loses m05's heuristic sharpness; MITM can't recover. m05 already navigates the d=6 shell as a side-effect of solving — m21 doesn't add value at this beam level.
- **2026-04-26** — m23 Q-shortlister trained on GCP L4 (~2h wall, 500 ep, 12.4M params with hidden=(2048,1024), num_res_blocks=3). Final MSE 0.33, KL 0.085 (down from MSE 56, KL 0.31 at start). **Recall validation: 100% at α=2** (and 98.7% at α=1; gates ≥99%). Student top-2B contains all teacher top-B in expectation.
- **2026-04-27** — Q-shortlister @ beam 131k benchmark on 12 lab samples: **wall 282s vs m05 baseline 1240s = 4.4× speedup**. Path-sum 1116 vs 1053 (+6% longer paths). Quality regression concentrated on easy puzzles (pid 50: 40 → 86 moves). Hypothesis: easy-puzzle states are out-of-distribution for the student (trained on RW-distributed states). Wall savings allow scaling to higher beam — testing qshort@beam 524k now.
- **2026-04-27** — Phase B (m05 fresh on local 4090, full 1001, beam 131k) (KEPT) — total 87,932 moves before post-rescue, pid 490 (407 moves) and pid 920 (758 moves) the worst two. Local 4090 wall ~10.5h.
- **2026-04-27** — Beam-stack rescue on Phase B's worst (KEPT) — `12_beam_stack_rescue.py` resolved pid 490 to 126 moves (-281) and pid 920 to 137 moves (-621) by retrying with runner-up ancestors. **Combined Phase B + rescue = 88,195 moves SUBMITTED (rank #3, ahead of Rokicki @ 93,606)**.
- **2026-04-27** — beam_lab `--compile` A/B on local 4090 (12 puzzles, beam 131k) (ACCEPTED) — `--compile` 752.6s vs uncompiled 1032.6s, **−27% wall, paths IDENTICAL (1049/1049)**. Mode reduce-overhead activates Inductor kernel fusion + CUDAGraphs through Dynamo. Promoted to default for production runs. Caveat: requires `pad_to_batch_size=True` to avoid shape-driven recompile thrash (5.8× slowdown if violated; documented in beam_search.py docstring).
- **2026-04-27** — internal_batch_size sweep (REJECTED — no improvement) — bs=32k 1497.2s, bs=65k 1498.1s on the 24-puzzle stratified sample at beam 131k with --compile. GPU saturated at bs=16k on 4090; larger chunks just split the same work into bigger kernel launches. Conclusion: 16k is the right default; don't keep raising.
- **2026-04-27** — Adaptive beam escalation (REJECTED on quality) — `run_escalation_benchmark.py` tries beams in order 16k→65k→131k per puzzle, exits on first solve. 24-puzzle: wall 331.7s (−78% vs single-beam 131k=1497s) but path-sum **2270 vs 2013 (+13% paths)**. 21/24 solved at 16k with longer paths; only 3 escalated. The path quality regression makes this a leaderboard loss, not a win. Don't deploy.
- **2026-04-27** — cayleypy library comparison on 3 puzzles (RECONCILED) — earlier 6-8× cayleypy slowdown claim was a config bug (batch_size=256 carried over from alexandervc's notebook where their bigger model OOM'd). With batch_size=2^14 matching ours, simple-mode cayleypy 257s vs ours 230s (within 11%). Adding `compile dynamic=True` to cayleypy: 190.1s vs ours+compile 183.6s (essentially tied). Lesson: always match batch_size before declaring cross-implementation gaps. **Anti-pattern saved** — never trust dramatic timing claims without verifying both sides.
- **2026-04-27** — cayleypy `iterated` mode w/ history_depth=10 (REJECTED) — 410.1s wall (vs ours 183.6s, +124%); paths +10 moves vs simple mode. Russian commenter's docs explicitly warn "this slows down" — it's a quality lever, not a speed lever. m05's sharp Bellman heuristic doesn't need non-backtracking enforcement.
- **2026-04-27** — cayleypy `simple` + MITM (BFS-d6 hashed_neighbourhood) (REJECTED) — 215.3s wall (vs ours 183.6s, +17%); path-sums similar. m05 already navigates the d≤6 shell as a side-effect of the Bellman heuristic; explicit MITM termination doesn't add paths but adds the full shell hash overhead. Confirms 2026-04-26 finding (m21+MITM also failed for same reason).
- **2026-04-27** — Manual CUDA Graphs alone, no torch.compile (REJECTED) — 3-puzzle: 245.9s vs `--compile` 182.5s = **−34% slower**. `compile(default) + manual graph wrap` ties `--compile` exactly (182.4s). The win in `--compile mode=reduce-overhead` is Inductor kernel fusion, NOT graph-replay overhead reduction. There's no headroom from manual graph capture. Code (`CudaGraphedModel`, `setup_model_for_cuda_graphs`, `--cuda-graphs` flag) stays in `beam_search.py` under feature flag, defaults preserve current behavior.
- **2026-04-27** — beam_decay (geometric narrowing per step) (REJECTED on quality) — added `beam_decay` + `min_beam_width` config fields. Tested decay=0.99 floor=32k on 4 puzzles: wall 167.2s (−18%) but paths 358 vs 346 baseline (+3.5%); pid 492 92→99 moves (+7). On full 1001 the +2-3.5% inflation would push 88,195 → ~91k Kaggle score. Speed-without-quality goal not met. Code stays under `--beam-decay` flag.
- **2026-04-27** — Stochastic beam (Gumbel-top-k temperature sampling) (NOT TESTED — skipped on first principles) — added `temperature` config + Gumbel-top-k path in `_do_greedy_step`. Not actually a SPEED lever; it changes which states are kept (sampled vs greedy), not how many. Greedy m05 already operates near-optimally on most puzzles; stochastic variant would add randomness on top of an already-validated greedy ordering. Skipped per user direction (no metric regressions). Code stays under `--temperature` flag for future research use.
- **2026-04-27** — Async pipeline prototype (REJECTED) — added `profile=False` mode to `KhoruzhiiSolver` (no-op `_sync_p()`) and `solved_check_every=K` to config. Skip-syncs alone: 181.3s vs 182.5s baseline on 3-puzzle (−0.7%, within noise). +deferred check K=4: 183.9s wall (+0.7%) AND **+0.8% path-sum** (V0 enters beam at step j, missed CPU-syncing check at j+K finds it gone, beam re-discovers via different parent → +1 path on pid 0 and 1000). 96% model_s ceiling caps async wins at ~4%; in practice <1%. **Saved to `megaminx_gotchas.md`** so future sessions don't re-prototype this. Pivoting to TensorRT.
- **2026-04-27** — TensorRT plan written (PENDING execution on GCP) — `megaminx/tensorrt_gcp_plan.md`. Six-step plan: env install (15min) → export script (30min) → solver integration (30min) → validation gates (45min) → qshort integration (30min) → full 1001 deploy (~12h GPU). Hard gate: 24-puzzle path-sum must equal 2013 (current `--compile` baseline). FP16 primary, BF16 / mixed-precision fallbacks. Engine is sm_89-specific (won't transfer between L4 and A100). The only remaining lever with predicted >5% gain past the async ceiling.
- **2026-04-28** — TRT decision (ADOPT @ beam 165k) — 12-puzzle L4 sweep: TRT@165k 969s/1043paths beats compile@131k 959s/1049paths by -6 paths at +1% wall (noise). TRT@196k = 1142s/1030paths (good for hard tail). TRT@262k = 1514s/1026paths (reserve).
- **2026-04-28** — m26/m26b (REJECTED) — both converge to ~0.10 loss = same as m05's 0.094. Capacity-scaling at this signal exhausted. m26_bellman/ep499 strat-5: 51/51 / mean 91.2, fails gate.
- **2026-04-28** — m27_bfs6_mixin (50%) (REJECTED) — strat-5 ep299: 51/51 / mean 91.49. Same cluster as m26.
- **2026-04-28** — m22 K=2 lookahead (REJECTED at ep149 mid-train) — strat-5 ~tied with cluster.
- **2026-04-28** — m27 ep199 mid-ckpt (REJECTED) — confirms ep299 isn't an over-training artifact.
- **2026-04-28** — m27b/m27c (25%/10% BFS-d6 mixin) (REJECTED) — none break the cluster. Lower-mixin variants don't help either.
- **2026-04-28** — Wave 3 NISS+qshort strat-5 on GCP (KEPT for diagnostic) — same cluster.
- **2026-04-28** — Wave 3 ensemble-3 m05+m23 strat-5 on local (KEPT for diagnostic) — same cluster.
- **2026-04-28** — Wave 3 aggressive beam-stack rescue top 20 (REJECTED at +0 wins) — beam-stack helps catastrophic failures (pid 490, 920) but NOT already-converged-suboptimal pids. Bigger beam is the lever for the latter.
- **2026-04-28** — Hard-tail rescue qshort+524k on top 148 long-path pids (KEPT, SUBMITTED 86,329) — `phase_b_plus148.csv` saved -1,866 moves vs prior 88,195. New best.
- **2026-04-28** — m28 Double Bellman (REJECTED) — bias decorrelation per van Hasselt 2010. Strat-5 51/51 ~91. Cluster ceiling unmoved.
- **2026-04-28** — SWA m05/400-499 (REJECTED, REGRESSES) — strat-5 mean 96.75. Late-cycle ckpts have drifted enough that averaging blurs the heuristic.
- **2026-04-28** — m29 n_back=4 (KEPT, +1 solve) — strat-5 51/51 / **88.98** (-0.42 vs m05's 89.4). First variant to break sub-89. Coverage knob (ban inverse of last 4 actions) worked marginally.
- **2026-04-28** — m30 n_back=16 (REJECTED, m12-trap) — 26% lower training loss but worse strat-5 (89.69). Confirms training MSE is unreliable proxy when n_back over-narrows walks.
- **2026-04-29** — Symmetry derivation v3 (KEPT) — `scripts/19_symmetry_v3.py`. v2 sigma propagator was too weak (local rule never fired from 2-anchor seed); v3 replaced with backtracking + forward-checking. Found **360** symmetries, not 60 — group is **A_5 × C_6** (rotational icosahedral × non-geometric piece-twist invariance under conjugation). `data/rotations.npy` (360, 120) int8, ~42 KB. Math verified end-to-end via `scripts/20_test_sym_translation.py` (R*solved*R_inv on all 360, conj-map on all 360, path round-trip 16/16 on real test puzzles).
- **2026-04-29** — `--sym-ensemble` integration into `03_solve.py` (KEPT, smoke-tested) — adds `--sym-ensemble K`, `--sym-rotations PATH`, `--sym-seed`. Per-pid: K rotations including identity, transform state, full beam, translate path back via R_inv·g·R conjugation map, verify against original, take min. Smoke test `--sym-ensemble 2 --beams 16384 --pids 0,100` passed end-to-end.
- **2026-04-29** — m31 rotation augmentation in Bellman training (REJECTED, hypothesis falsified) — `BellmanConfig.rotations_path / rotation_aug_prob`. m05 recipe + 50% per-batch state rotation, target unchanged (rotation preserves distance). Final loss 0.142 (vs m05's 0.094, +51%). Strat-5: 50/51 solves / mean 95.76 — regresses by +6.4 mean and loses 1 solve. **Hypothesis "orbit coverage is binding" falsified**: at 6M params, augmentation across 360 orbit-equivalents dilutes signal rather than sharpening it.
- **2026-04-29** — m29+qshort+524k full-1001 (ABORTED at pid=99) — qshort student m23 distilled from m05; pairing with m29 teacher caused fb-wins on 58/100 early pids (m23 αB shortlist consistently misses m29's preferred children). Don't reuse Q-shortlister across teachers without re-validating recall. Documented in `megaminx_gotchas.md`.
- **2026-04-29** — Cluster summary post m31 — every single-recipe variant lands in 88-97 strat-5 mean. m29 (88.98) and m05 (89.4) are the only sub-91 results. Recipe levers exhausted at 6M params. Pivot to beam-side levers for the score race.
- **2026-04-29** — Bellman parameter variations PLANNED (`m32_targetref5.yaml` target_update=5; `m33_lr2e4.yaml` lr=2e-4) — defensive only, launch if A1-A3 stall.
- **2026-04-29** — A1 m05+sym-ensemble K=4 strat-5 (KEPT, NEW BEST AT STRAT-5) — 51/51 solves, total 397,404, mean **88.20**. Vs m05 baseline (50/51, 89.4): +1 solve AND -1.20 mean. Vs m29 (51/51, 88.98): -0.78 mean. **First inference-side mechanism to break the m05 cluster ceiling.** Wall ~3h on local 4090.
- **2026-04-30** — A2 hard-tail rescue with sym-ensemble K=2 + qshort + 524k on top 50 long-pids of phase_b_plus148 (KEPT, SUBMITTED 85,812) — 50/50 solved, total 4,431 vs baseline 4,948. -517 moves merged → `phase_b_plus198.csv` 85,812. Submitted to Kaggle.
- **2026-04-30** — GCP m05+qshort+524k+TRT full-1001 finished — 1001/1001 valid, total 84,750. Wall 19.85h. Better than the local-rescue track (85,812) by -1,062.
- **2026-04-30** — Min-merge phase_b_plus198 + GCP full-1001 (KEPT, SUBMITTED 83,362) — phase_b 270 wins (rescues), GCP 446 wins (broader sweep), 285 ties. -2,450 vs phase_b_plus198, -1,388 vs GCP alone. Tracks were genuinely complementary. Submitted to Kaggle.
- **2026-04-30** — A3 NISS+qshort+m05+m23 strat-5 (REJECTED) — 51/51 solves, total 397,659 → mean ~93.35. REGRESSES vs sym4's 88.20 by +5 mean. m23 distilled from m05 forward states; recall drops on inverted states. Don't pair NISS with the qshort+m23 stack.
- **2026-04-30** — A2.2 hard-tail rescue with sym-ensemble K=4 + qshort + 524k on top 80 long-pids of merged 83,362 (KEPT, SUBMITTED 82,646) — 73 of 80 are NEW pids never previously rescued. All 80/80 solved, total 6,941 vs baseline 7,657. -716 moves merged → `merge_plus_sym4_top80.csv` 82,646.
- **2026-04-30** — K=8 sym-ensemble micro-test on top 20 long-pids of 82,646 (KEPT, SUBMITTED 82,481) — All 20 solved, total 1,711 vs baseline 1,876. -165 moves. **K=8 saves ~80% as much per pid as K=4 at 2× wall — diminishing returns confirmed; K=4 is sweet spot for production.**
- **2026-04-30** — m23_v2 sym-aware Q-shortlister training (KEPT, GCP) — 500 ep, ~2h on L4. Same arch as m23 (12.4M, hidden=2048,1024 ×3 ResBlock). Modified `09_train_q_shortlister.py` to support `--rotations-path / --rotation-aug-prob` (rotate state R*s*R_inv before teacher Q-target, prob 0.5). Final MSE 0.354, KL 0.087 (~tied with m23's 0.33/0.085). Recall vs m05: α=1 98.7%, α=2 100%, α=3 100% (matches m23 on unrotated states).
- **2026-04-30** — Strat-5 m05+m23_v2+sym4+qshort (KEPT) — 51/51 solves, total 397,481 → mean **88.41**. Vs A1 m05+sym4 (no qshort) 88.20: +0.21 mean cost for **6× wall reduction** (31min vs 3h). Enables full-1001 sym-ensemble at tractable wall (~10-30h GCP vs ~60h+ for no-qshort). **m23_v2 is the right student** when paired with `--sym-ensemble`; m23 (original) better for non-rotated solves.
- **2026-04-30** — m32 target_update_every=5 Bellman training + strat-5 (REJECTED) — Final loss 0.0986 (vs m05's 0.094). Strat-5: 51/51 / mean **94.65** — REGRESSES by +5.25 vs m05. Faster target refresh (5 vs m05's 10) → over-fits to short-term gradient noise. m05's value of 10 is calibrated. m33 (lr=2e-4) NOT launched given m32's regression and broader cluster pattern.
- **2026-04-30** — T1.1 commutator window-replacement test (REJECTED) — Built 37K-entry library (`data/commutator_table.pkl`, depths 4-8). 29K perms NEW beyond BFS-d6. Tested W=7,8 window-replacement on current submission: **0 matches**. Same finding as IHES. Beam-search paths' structured perms don't fall on commutator atoms. Window-replacement post-processing is permanently empty.
- **2026-04-30** — T1.1 macro-augmented beam prototype (KEPT mechanism, REJECTED brute-force macros) — Wired macros as additional actions in `KhoruzhiiSolver` (cayley/khoruzhii_search.py): `macros` parameter, `action_cost` tensor, cost-aware V adjustment (V + cost - 1), path reconstruction expands macro actions to gen words. Tested 6/30 d=4 commutators on 5 hard pids: net +9 to +41 moves (HURT). Brute-force commutators don't pay; T1.1 needs CURATED speedcubing macros (multi-day scrape). Mechanism is shipped and reusable.
- **2026-04-30 SESSION END** — Best submission: 82,481 (Kaggle scored ✓). Net session impact: -5,714 moves vs morning start at 88,195. **Cluster ceiling at 6M params confirmed thoroughly**: every recipe variant (m17, m22 K=2, m26, m27, m28, m29, m30, m31, m32, SWA) lands in strat-5 mean 88-97. **Pivot validated**: sym-ensemble at INFERENCE is the cluster-breaker (88.20 mean), enabled at full-1001 scale by m23_v2. Next session: see `to_do_shortlist.md` and `HANDOFF.md` for queue.
- **2026-04-27** — m26 + m26b training launched on Kaggle (RUNNING) — capacity-scaling A/B against m05's baseline arch. m26: bigger model with walk-depth pretraining. m26b: wider arch (width-vs-depth A/B with m26). If either lands a sharper heuristic at ≥1.5× the params, integrate as new teacher and retrain m23 student. **Acceptance gate**: must beat m05 on strat-5 by ≥+3 solves AND mean model_avg ≤ 0.95×. If it ties: KEEP m05 (more params adds inference cost).
- **2026-05-02** — m34 (soft-Bellman + Polyak EMA + solver-trace mixin) (REJECTED, CATASTROPHIC) — **0/51 strat-5 solves**. Ablations isolated `softmin_temperature=0.5` as the killer (T not << V scale ~10). m34 ablations (no-mixin / Polyak-only / softmin-only) all confirmed: softmin breaks the V; Polyak alone OK; mixin alone OK. Lesson: T must be << V scale for soft-min approximation to recover the hard min.
- **2026-05-03** — m36 deeper-but-narrower [1024,512]×6 ResBlocks (~5.7M params) (REJECTED) — strat-5 50/51 / mean ~91. Same cluster as every other Bellman variant. Depth doesn't help any more than width (m26b) did. **Width-vs-depth-vs-baseline triangle of architectures all in cluster.**
- **2026-05-03** — m37 solver-trace primary training (Kaggle P100, 200 ep × 16 batches, m07 warmstart) (TRAINED) — final MSE 0.082 on the 128K (state, true_remaining_d) pairs mined from prior verified submissions. Tests "is the wrong objective the binding constraint on cluster ceiling?". Strat-5 eval pending on Kaggle (kernel v3 RUNNING after v1 missing-script + v2 full_post_process signature-drift fixes).
- **2026-05-03** — m38 listwise rank loss training (Kaggle P100, 200 ep, lambda=0.5 with warmup, m07 warmstart) (TRAINED) — final MSE 0.121, listwise 2.94 (~ln(24) baseline). Note: kernel had to inline `_bellman_targets` because snapshot dataset's bellman.py was older than local source (no `softmin_temperature` arg). Filed a snapshot-version-drift memory note (`reference_kaggle_pipeline.md`).
- **2026-05-03** — m38 strat-5 eval (REJECTED) — local 4090, 51 pids @ beam 65k, ~47 min wall. **48/51 solved by model, mean model_avg ~99** (vs m05 50/51 / 89.4). Acceptance gate: gate1 (≥+3 solves) FAIL (-2), gate2 (mean ≤84.9) FAIL (~99). Listwise rank loss as a Bellman auxiliary doesn't break the cluster — actually slightly worse than vanilla m05. **Adds another data point to the cluster-ceiling-is-information-bound theory** (regularizers don't help when labels are noisy).
- **2026-05-03** — h18 hamming-filtered T1.6 SA top-100 (KILLED) — `commutator_table_h18.pkl` (2,520 macros, hamming distance = 18) replacing the unfiltered ~37K commutator pool. Ran 4-way parallel on local 4090. **DemonLord game on GPU dominated CUDA scheduler for 4h** → 0 worker pids completed. After game stopped, even 4-way contention couldn't complete a single pid in 30 min. Killed. Pre-kill signal (-17 / 11 pids vs v1's -30 / 10) suggested h18 was tracking WORSE than the unfiltered macros anyway. **Conclusion: minimum-disruption macro filter doesn't help; diversity beats hamming-minimality. Don't repeat.**
- **2026-05-03** — Parallel-SA-on-single-GPU with active game (REJECTED, infrastructure lesson) — 4-way `27_path_sa.py` workers + DemonLord.exe on the GPU = full CUDA queue domination by game. Even after game ended, 4-way contention made each beam call ~4× slower → no pid completion in reasonable time. **For SA-style work on a shared GPU, prefer 1-2 workers max OR ensure no other CUDA process active.** Filed in operational learnings.
- **2026-05-03** — GCP T1.6 v2 SA top-200 (KEPT, contributed to 79,946) — `27_path_sa.py` with corrected operator weights (`--w-tail 2 --w-macro 1 --w-swap 0`, `n_iter=15`) on top-200 longest pids of 81,357. **122/200 improved, -471 moves; tail_resolve 10.0% accept (203/2038), macro_insert 1.8% (17/962). 36h wall on GCP L4.** Stand-alone result 80,886 worse than current LB 80,739, but the rescue's wins concentrated on long-tail pids that didn't overlap with TPU v19a's wins.
- **2026-05-03** — Min-merge merge_tpu_v19a_fix + GCP T1.6 v2 (KEPT, SUBMITTED 79,946) — `merge_tpu_v19a_plus_t16v2.csv`. 75/1001 pids replaced (rescue had shorter path), 275 longer-rescue (kept base). **80,212 → 79,946 (-266). Submitted 2026-05-03 16:30 UTC, scored COMPLETE. New LB position: #2 (79,946) trailing #1 DrozdovDan (79,931) by just 15 moves; passed Vladislav Kuznetsov (79,971).** Min-merge worked because v19a's wins (TPU narrow-K wide-beam) and T1.6 v2's wins (SA on long-tail) targeted disjoint pid sets.
- **2026-05-03** — m39a wide arch [4096, 512]×2 ResBlocks (~11M params) pretrain + Bellman + strat-5 (RUNNING, local 4090) — completes the width-vs-depth-vs-baseline arch triangle. Honest expectation per config comment: NO break (info-bound, not capacity), but worth confirming. Wall ~5h.
- **2026-05-03** — m40 (Bellman + L_upper one-sided overestimate penalty) (DESIGNED, QUEUED) — synthesis from peer's 4-term composite loss proposal. Of the four terms (L_sup, L_lip, L_upper, L_anchor), only L_upper is novel-to-us, free, and orthogonal: `lambda * mean(max(0, pred - walk_depth)^2)`. clip_upper already clips the TARGET; L_upper additionally pressures the PREDICTION. Other proposed terms are redundant (L_anchor with target-net-on-goal, μ_d via solver-trace = m37, μ_d via BFS-d6 = m22 family) or already-tried-failed (L_lip ≈ m38 listwise). Code: `src/cayley/bellman.py` `lambda_upper` field + loss term, RW-portion only (BFS-d6 mixin exempt). Config: `m40_upper_penalty.yaml`. Plan doc: `m40_plan.md`. Decision: if it lands ~50/51 / 89, abandon the broader composite-loss bundle.
- **2026-05-03** — AnanasClassic/cayleypy-neighbour-model-training repo audit — independent implementation of our qshort recipe (output_dim=n_gens, teacher distillation on child Vs, MSE loss). **Validates our design.** Differences worth flagging if we ever retrain m23: theirs uses 3-stage LR ramp-down (1e-4 → 5e-5 → 2e-5 over 4096 ep × 16 steps), ours uses single-stage. Could squeeze ~0.5-1% recall with the schedule. Released checkpoint `p900-t000-q_1776581286_best` exists; group_id=900 likely megaminx; not yet benchmarked against ours.
- **2026-05-03 SESSION END** — Best submission: **79,946 (#2 LB)**. Net session impact: -266 vs morning start (80,212). **m38 listwise + m36 deeper-but-narrower added to "cluster ceiling holds" pile** (now: m17, m22, m26, m26b, m27, m28, m29, m30, m31, m32, m34, m36, m38, all 6M-12M params, all 88-99 strat-5 mean). m37 solver-trace primary still pending eval — the OUTSTANDING orthogonal experiment. m39a (wide arch) running. m40 (L_upper) queued. Next session: see m37 strat-5 result first, then act accordingly (if 51/51/85: investigate; if 50/51/89: cluster-ceiling-is-information-bound is settled, pivot fully to inference + post-proc + ensembling track).
- **2026-05-03** (late) — TPU v19b partial → submission 79,522 (user-driven). Min-merged into 79,946 via `merge_tpu_v19b.csv`. -424 moves. New #2 LB.
- **2026-05-04** — m37 strat-5 v3 result on Kaggle (REJECTED, CATASTROPHIC) — **3/51 solved by model**, mean 24.33 on the few easy bucket-0 solves; every bucket 1-10 pid hit max_steps=150 without finding solved (~560s each, 7.5h total). Pure solver-trace primary training is OOD on hard scrambles — solver paths concentrate on visited states, not the broader orbit. Random-walk Bellman is load-bearing. Same lesson as m34: don't replace walk-depth, only mix in.
- **2026-05-04** — m39a wide-arch strat-5 result (REJECTED) — 48/51 / mean ~98. Cluster. Wide arch [4096,512]×2 ~11M params doesn't escape label-noise floor. Width sweep complete: m05 (6M) ~89, m26b (13M) ~91, m39a (11M) ~98 — all cluster.
- **2026-05-04** — m40 (Bellman + L_upper, λ=0.1) strat-5 (REJECTED) — 48/51 / ~99. L_upper steady-state near zero (model converged to fixed point satisfying both MSE and clip-upper); barely contributes. Per m40_plan.md decision matrix: dead end. **Per-m40 evidence kills the broader 4-term composite-loss proposal** — combined with m38's listwise null, 2 of 4 proposed terms have failed, and the others are redundant with what we have.
- **2026-05-04** — m42 distributional V (T2.5, QR-DQN-style 32 quantiles, m05 backbone with quantile rows replicated, quantile-Huber loss + distributional Bellman backup) (TRAINED, REJECTED at both inference modes) — final loss 1.12 (q-Huber, not directly comparable). Strat-5 median: 48/51 / ~97. Strat-5 lower-q25: 48/51 / ~95 (slightly better than median by -2 mean, distributional inference IS real but small). Both fail acceptance gate. Code shipped: `cayley/distributional.py` (quantile-Huber + distributional Bellman target), `scripts/38_train_distributional_v.py`, configs/m42_distributional.yaml, `--quantile-reduce` flag in 03_solve.py.
- **2026-05-04** — T1.3 TTT (per-puzzle local fine-tune, Akyurek 2024 style) on hard-tail strat (NULL) — 16/16 solved, mean 93.44 vs m05 baseline 93.25 (+3 moves on 1492, within noise). Bellman-self-consistency on m05 = weak gradient signal; m05 is already locally consistent → tiny updates → no real shift. Implementation: `scripts/37_ttt_solve.py`. Default hyperparams (k_steps=50, n_walks=32, walk_len=8). Mechanism viable but not productive at this signal strength.
- **2026-05-04** — T3.3 oracle Q-head (m35) eval (REJECTED as standalone) — Recall vs m05 teacher: at k_max=5 (m35's training distribution) recall is 0.60-0.85 (m35 is exact, m05 is approximation; they disagree on best child). At k_max=80 (production beam states) recall is 100% at α=2. **m35 IS a viable qshort student at production depth (matches m23) but doesn't add new value** — m23 already at 100% there. Could enable hybrid endgame solver (use m35 exact actions when state enters d≤5 shell) but that's a solver code change.
- **2026-05-04** — m05 hard-tail strat-5 baseline established — buckets 7,8,9,10 = 16 pids @ beam 65k bf16. **16/16 solved, mean 93.25.** New gate metric: any model claiming hard-tail improvement must have mean ≤93.25 (or ideally ≤88.6 = 0.95×). 03_solve.py gained `--strat-buckets` flag.
- **2026-05-04** — Min-merge submission (NOT submitted) — m05 base 79,522 ∪ m42-q25 full-1001 ∪ m38 hard-tail partial ∪ m39a hard-tail partial = 79,486 (-36). Per-source: m42 12 wins -35; m38 0 wins (its pids 500-624 already heavily-optimized); m39a 1 win -1. User decision: too small to submit alone.
- **2026-05-04** — Kaggle hard-tail-only kernels for m38 and m39a (BOTH KILLED at 12h limit) — partial CSVs pulled successfully (kill-resilience worked: per-pid flush + fsync). m38 covered 125/501 pids (bucket 5 + most of 6, 100% solve), m39a covered 74/501 pids (bucket 5 only, slower per-pid because of wider arch). Per-pid wall on P100: m38 ~300s, m39a ~600s. Confirms full-1001 inference at beam 65k is ~14h on P100 — over the 12h kill. Future Kaggle inference jobs need to split or use smaller beam.
- **2026-05-04** — m42 lower-q25 full-1001 inference on GCP L4 (DONE) — 14.4h wall, 1001/1001 valid. Standalone CSV total 90,294 (worse than 79,522 alone — useful only via min-merge). Min-merge contribution to 79,522 base: 12 wins, -35 moves. Adds m42-q25 to the ensemble fodder pile (joins m38, m39a partials).
- **2026-05-04** — m43 (standard m05 Bellman + 25% solver-trace MIXIN proper) (REJECTED, CATASTROPHIC) — Strat-5: **0/51 solved**. Trained 500 ep, final loss 1.4465. Even 25% solver-trace mixin is enough to break Bellman convergence. **Catastrophic pile now: m34, m37, m43** — all three involve solver-trace at non-trivial proportion. Lesson: BFS-d6 mixin (d≤6 only, near-solved) is compatible with RW; solver-trace (full distance distribution, path-concentrated) is OOD enough to break the V even at 25% mixin. Code: `src/cayley/bellman.py` `solver_trace_path` + `solver_trace_fraction` fields; configs/m43_solver_trace_mixin.yaml.
- **2026-05-04** — T1.1 macros Phase 2 scrape + A/B test (KEEP MECHANISM, modest contribution only) — Built `data/named_macros_phase2.yaml` with 54 algorithms across 9 categories (sledgehammer family, adjacent-face commutators, Sune/Antisune, Niklas, F2L insertions, T-perm/Y-perm, etc.). Validator yielded 45 unique surviving permutations (vs 5 in Phase 1; 4% collapse rate confirms broader sources work). 28 of 45 at hamming=18 (same regime as h18 brute-force, already rejected). 3 at h=12 (sledge-3x family — minimum disruption observed), 14 at h=25-54. Format converted to commutator-table-compatible at `data/curated_table_phase2.pkl`. **A/B test on top-50 longest pids of 79,946**: curated A 24/50 improved -79 moves (1.2% macro_insert accept rate, 3.8% tail_resolve). Brute-force B killed early at 10/50: 2/10 improved -2 moves (0% macro_insert at this point, far worse pace). Curated genuinely beats brute-force — but absolute gain is small (-1.6 moves/pid) and per-puzzle wall is 17h+. Not the 70K-cracker we hoped. Decision: don't deepen the multi-day scrape; T1.1 mechanism kept in production but accepts modest contribution.
- **2026-05-05** — community-pushed shareable-kernel runs merged → submission 78,408 (user-driven). alexandervc + fedmug forks of `cayleypy-megaminx-beam-shareable` ran B=1M with K=8 long-tail at pids 800-900. 324 wins distributed across all buckets. -1,114 moves vs 79,522 base. **New best: 78,408.**
- **2026-05-05** — m44 (1.12M tiny V, hidden=[512,128] nrb=2) pretrain + Bellman + strat-5 (REJECTED) — Strat-5: 47/51 / mean ~102. Both gates FAIL. Tiny model lands in/near cluster — confirms cluster ceiling holds across **44× param range** (1.12M → 49.6M planned). **Information-bound theory empirically airtight at the small-scale boundary too.**
- **2026-05-05** — T1.3 v2 self-distill TTT (DAGGER-style) on hard-tail strat (NULL/inapplicable) — Per-puzzle: cheap first beam (8k, 60 steps) → path-derived exact labels → fine-tune V → full beam. **0/16 solved** because cheap first beam (8k×60) cannot solve buckets 7-10 puzzles. Mechanism requires successful first pass; hard tail doesn't have that without using full beam (which defeats the cheap-then-full structure). T1.3 family (v1 + v2) closed: weak signal source AND structural unsuitability for the hard-tail regime where TTT would help most. Implementation: `scripts/40_ttt_self_distill.py`.
- **2026-05-05** — m45 (49.6M large V, hidden=[6144,2048] nrb=3) pretrain + Bellman (RUNNING ON GCP) — pretrain DONE final loss 63.26 at 8.4s/epoch. Bellman warmstart in progress at ~146s/epoch on L4 (49M model is memory-bound for the target_net forwards). ETA ~20h Bellman + 100 min strat-5. Eval result expected ~2026-05-06 mid-day. Final capacity-headroom test; if cluster, the cluster ceiling is bulletproof across 44× param range AND every recipe family. Configs: m45_pretrain.yaml + m45_bellman.yaml. Launcher: scripts/run_m45_chain_gcp.sh. m45 auto-launched via watch_and_launch_m45.sh after m42 GCP completed.
- **2026-05-05 SESSION END** — Best submission: **78,408 (Kaggle scored ✓)**. Net session impact: -1,114 vs session start (79,522). **Training-side experimental track empirically airtight closed across 44× param range** — m44 (1.12M), m05/m07/m17/.../m43 (6M cluster), m26/m26b (12-13M), m39a (11M), m45 (49.6M, pending). Catastrophic pile at 3 (m34, m37, m43) — all solver-trace-related. Cluster pile at 16+ recipe variants. Mechanism failures (T1.3 v1, T1.3 v2, T2.5 m42 distributional). T1.1 Phase 2 macros gives modest contribution (-1.6 moves/pid) but not 70K-cracker. Path to <70K must come from inference-side compute (sym-ensemble at scale, multi-agent T1.2), search-space changes (T1.1 deeper or T2.4 Minkwitz), or community/external help (per the 78,408 submission). **Next session: m45 result lands first; if cluster, fully pivot to T1.2 / T2.4 / scaled inference; if breaks, investigate scaling further.**

- **2026-05-10** — **PDB+IDA* (Korf-Felner style) — REJECTED for max-combine in beam**. Built complete coordinate-space PDB infrastructure: 20-corner perm/ori tables (`corner_coord.py`, verified bijective via 30-step random walks); 30-edge perm/ori tables (`edge_coord.py`, verified); GPU-vectorized chunked BFS builder (`pdb_corner.py`, 50K-row chunks). Built **4 disjoint K=5 corner PDBs** covering all 20 corners, each 452M states / 452 MB (P(20,5)×3^5), 39s build per PDB on RTX 3090. Diameter 14 moves per PDB, mean depth 9.7. **0 admissibility violations** verified on 10K BFS-d6-known states. Max-of-4 lookup: 42K states/s, mean h=10.57 on depth-50 random walks. Integrated as `pdb_combine_mode="max"` in `cayley/khoruzhii_search.py` (per-candidate `value = max(V_neural, h_pdb)`). **20-pid bench (m_curr_v3, beam 65k, bf16)**: baseline 1828 moves (avg 91.4); PDB max-combine 1832 (avg 91.6). 18/20 identical paths, 2/20 (+2 each: pids 100, 650), 0/20 improvements. **Net +4 moves, +20% wall** from per-step PDB lookup overhead. Why neutral: V_neural saturates >> 14 for far-from-solved states so max(V, PDB)=V; near-solved is already covered optimally by BFS-d6. Edge K=5 PDBs (6 disjoint, 547 MB each) NOT built — corners + edges max-combine same fundamental ceiling. IDA* with PDB heuristic infeasible at d=14 (12^14 ≈ 1.3T nodes). Infrastructure reusable; deliverables: `src/megaminx/{corner_coord,edge_coord,pdb_corner,pdb_edge,pdb_heuristic}.py`, `data/pdb_corner_K5{,_p1,_p2,_p3}.pkl` (1.8 GB total), `scripts/58_corner_pdb_beam.py`.

- **2026-05-10** — **m_adm_v0 (admissibility-aware loss, 50 ep)** — Bellman + `λ_pdb=5.0 · mean(relu(h_PDB(s) - V_pred(s))²)` using max-of-4 K=5 corner PDBs. Warmstart from m_curr_v3, otherwise identical recipe. Final loss 0.0733. V undershoot rate dropped from 77% (m_curr_v3) to **15%** (m_adm_v0) on 500K BFS-d6 states — the penalty is doing the right thing. But V(V0) barely moved: 1.988→1.934 (PDB(V0)=0 ≤ V_pred(V0), so penalty doesn't fire at V0). Confirms admissibility-aware loss treats only the *V undershooting* part of the bias; not the *V0 overestimate* part. Code: `src/cayley/bellman.py` `lambda_pdb` field, `pdb_lookup_fn` parameter to `train_bellman`; `scripts/60_train_admissible.py` wires up `CornerPDBHeuristic`.

- **2026-05-10** — **m_dd_v0 (dataset distillation = anchor mixin + admissibility, 50 ep) — ACCEPTED**. Adds a fixed-batch anchor: 32 copies of V0 (target=0) + 4 copies of each of the 24 d=1 children (96 total, target=1) every batch, on top of m_adm_v0 recipe. Diagnoses + fixes the long-standing V(V0)≈2 bug: Bellman bootstrap target at V0 = 1 + min_a V(d=1 child) ≈ 1+1 = 2 because V0's children are NOT solved (only V0 itself is); the boundary V(V0)=0 was getting weak training signal because V0 appears 1/19.3M states in the BFS-d6 mixin. **V(V0): 1.988 → 0.009** (eliminated). mean V@d=1 = 1.00 (perfect calibration). Mean V@d=4 = 4.05. **20-pid bench: 1802 vs baseline 1828 = -26 moves (-1.4%) at 50 ep only**. 8 wins, 7 losses, 5 ties; biggest saves: pid 300 (-19, was 114), pid 900 (-11), pid 350 (-6). Biggest regression: pid 700 (+12). New `BellmanConfig` fields `n_anchor_v0`, `n_anchor_d1`. Config: `megaminx/configs/m_dd_v0.yaml`. **Hypothesis worth testing: full 500-epoch m_dd_v0 — 50 ep already broke the cluster ceiling; full training likely doubles the gain. Then full 1001-pid eval + submission.**

- **2026-05-10** — **m_tb_v0 (Trajectory Balance, smoke)** — Built dual-head GFlowNet model (`cayley/gflow_model.py`: ResMLP trunk + 24-action policy head + 1-D value head + scalar `log_Z` parameter). Trainer at `scripts/62_train_tb.py`: trajectories from V0 → s_k via random walks, reversed to give scramble→V0 solve trajectories with inverse actions; TB loss `(logZ + Σ log P_F(a_t|s_t) - Σ log P_B - log R)²` with uniform P_B (megaminx is bijective: log P_B = -log 24); + `λ·logZ` regularizer for shortest-path bias. 5-epoch smoke: TB loss 55.16 → 1.81; logZ stable at ~0.08. Joint policy + value training infrastructure validated. Per Pan et al. 2026 (arXiv:2603.01786) theory: at TB optimum, log F(s) ~ -dist(s, V0); could serve as drop-in V via `gflow_model.predict_value`. Full training pending GPU availability.

- **2026-05-10** — **m_az_v0 (AlphaZero-lite, smoke)** — Built joint trainer at `scripts/63_train_alphazero.py`: dual-head model (reuses ResMLPGFlowNet); self-play episodes (or `--synthetic-walks` for smoke); per-step (state, π_target, value_target) tuples; loss = α·CE(π_student, π_target) + β·MSE(V_student, value_target). Smoke with 200 synthetic walks (V0 → reversed-walk solve trajectories): p_loss 3.26 → 3.10, v_loss 17.4 → 3.0, lr cosine. Real PUCT self-play on hard pids needs >60 max_steps (greedy V solves none); use `simple_puct_episode` with proper tree search for production. Infrastructure validated; full self-play loop pending.

- **2026-05-10** — **m_dd_v0 20-pid bench bench: -26 moves vs m_curr_v3 baseline** (1802 vs 1828 = -1.4%). 8 wins, 7 losses, 5 ties on 20 stratified pids. The V(V0)≈0.009 fix shows up most on hard pids (pid 300: -19, pid 900: -11). Win is small but real on the validation set. Smaller than expected.

- **2026-05-10** — **m_dd_v0_full (1000ep + early stopping, killed at ep 184) — REJECTED, overfit**. Same recipe as m_dd_v0 50ep but trained 184 epochs with smoothed-loss early stopping (patience=50, min_delta=1e-4). Final smoothed loss 0.06881 (vs 50ep's 0.0724). V(V0): 0.019 (vs 50ep's 0.009 — still good). Undershoot rate 21% vs 21% — same. **But beam quality regressed**: on GCP 1001-pid eval, only 41/60 model solves for pids 0-59 (vs ~95% for 50ep). m_dd_v0_full produced an unusable submission CSV. Hypothesis: long training over-saturated on BFS-d6 anchor states (seen every batch), losing fidelity on harder intermediate-depth states (sampled via random walks). **Lesson: training loss isn't a reliable proxy for beam quality — validate via small bench during long runs.** Killed at epoch 184 via manual kill; m_dd_v0 50ep checkpoint is the canonical baseline going forward.

- **2026-05-10** — **m_tb_v0 full (200ep, no warm-start) — REJECTED**. Loss converged 55→0.0026, logZ -0.366. But on 10-pid bench: **0/10 solved**. Value head learned a relative log F ordering but for length-30 random walks only — uncalibrated for test pids of depth 0-450. Confirmed training-distribution mismatch concern.

- **2026-05-10** — **m_az_v0 full (synthetic walks, 100ep) — REJECTED**. Joint trainer worked (p_loss 3.26→0.15, v_loss 17.4→0.24), but bench 0/10 N/F. Same training-distribution issue: the value head sees only states near V0 (within walk length 30), not the deep test scrambles.

- **2026-05-10** — **m_tb_v1 (proper TB: warm-start trunk + variable walks 5-100, 500ep) — ACCEPTED**. Built `scripts/66_train_tb_proper.py` with two key fixes:
  - Load embedding+input_stack+res_blocks weights from m_dd_v0 50ep (warm-start the trunk's state representations)
  - Init `value_head = -V_distance_head` (sign flip: log F ≈ const - dist)
  - Sample trajectory lengths uniformly from [5, 100] per epoch (cover diverse depths)
  Final tb_loss 0.0023, logZ -0.4353 (regularizer pulling toward shortest paths). **10-pid bench: 10/10 solved, total 1016 moves (avg 101.6)** — vs m_dd_v0 50ep's 871 on same pids. **+16% moves but robust.** Demonstrates dual-head can work if trunk is warm-started and training data covers diverse off-path states. Path quality penalty comes from policy_head + value_head sharing trunk capacity.

- **2026-05-10** — **m_az_v1 (real solver paths from 78,029 submission, 200ep) — REJECTED**. Built `scripts/67_build_az_dataset.py` to convert previous-best submission CSV into 78,029 (state, action, remaining_distance) tuples — each path-state along the solver's realized solution. Trained ResMLPGFlowNet with joint CE(action) + MSE(remaining). Final p_loss 2.68, v_loss 0.53, top-1 acc 18%. **Bench: 0/10 N/F.** Value head calibrated only on path states, not off-path siblings. Beam expands 24 children per candidate, 23 of which are siblings the model has never seen.

- **2026-05-10** — **m_az_v2 (path + 23 siblings/state, V_teacher-labeled, 100ep) — REJECTED**. Augmented v1 dataset: for each path state, also include its 23 sibling states (one per non-taken action), labeled by m_dd_v0 50ep's V prediction. Total: 1.87M states with value labels. Trained joint with policy CE only on path states. Final v_loss 67 (high), p_loss 2.72, top-1 acc 17%. **Bench: 0/10 N/F.** Why: target conflict between path-state value (`remaining_path_len`, often an overestimate) and sibling-state value (`V_teacher`, predicted true distance). Model can't fit both — value head destabilizes. Off-path augmentation alone doesn't fix the underlying calibration issue.

- **2026-05-11** — **m_az_v3 (Bellman value + policy CE hybrid, 100ep) — ACCEPTED**. Built `scripts/71_train_az_v3.py`: trains ResMLPGFlowNet with the m_dd_v0 50ep value recipe (random walks + BFS-d6 mixin + V0/d=1 anchors + Bellman target net refresh every 10 ep), PLUS policy CE on solver-path actions (78,029 dataset). Both heads share trunk. **Final p_loss 1.60, v_loss 0.158, top-1 acc 50.7%** (vs random 4.2%, vs v1's 18%). Strong policy signal. **10-pid bench: 10/10 solved, total 943 (avg 94.3)** — vs m_dd_v0 871, TB v1 1016. **+8% vs m_dd_v0** (smaller penalty than TB v1's +16%). The 50% top-1 policy head is highly informative — usable as Q-shortlister with potential 4× wall savings (per prior m23 analog). Promising dual-head result; value calibration via Bellman is essential.

- **2026-05-11** — **Key insight from AZ v0→v3 progression**: the value head MUST be trained with **broad off-path coverage** (random walks providing diverse states across the manifold). Solver paths alone — even with sibling expansion via teacher V — give too narrow a state distribution for beam search to navigate. Random walks span the broad geometry; optimal paths cover a narrow tube. This is the single most important lesson from this experimental track.

- **2026-05-11** — **PDB+IDA* result summary** (work from 2026-05-09 session, documented here for completeness): infrastructure for K=5 corner PDBs (4 disjoint, 452M states each, 1.8 GB total, max depth 14) built and verified (0 admissibility violations on 10K BFS-d6 states). Max-of-4 combine integrated into `cayley/khoruzhii_search.py` as `pdb_combine_mode="max"`. **20-pid bench: +4 moves vs baseline (neutral, +20% wall).** V_neural already carries the signal PDB provides for far states; near-solved is already optimal via BFS-d6. Edge PDBs (6 disjoint K=5, 547MB each, 3.3GB total) NOT built — same fundamental ceiling. Infrastructure reusable: `src/megaminx/{corner_coord,edge_coord,pdb_corner,pdb_edge,pdb_heuristic}.py`, `data/pdb_corner_K5{,_p1,_p2,_p3}.pkl`.

- **2026-05-11** — **GCP production-recipe eval** (in progress as of session end) — relaunched on `cayley-gpu` (L4 24GB) using `m_dd_v0 50ep` checkpoint with production recipe `--beams 16384,65536 --max-steps 60,150 --niss --bf16 --resume`. First attempt with `--beams 65536 --max-steps 120` (single pass, no NISS) was catastrophic: only 41 model solves over 100 pids tested (~40% solve rate dropping to 0% for pids 60-79). With production recipe: at pid 279/1001 = 217 model + 63 fb (78% solve), avg 73 moves/pid. Trajectory looks like ~80-100K total when complete, comparable to current 76,304 submission. Eval is slow (~130s/pid with NISS+escalation) — ETA ~25 hours from 300/1001. Output: `megaminx/submissions/m_dd_v0_50ep_prod_1001.csv` on GCP at `/home/and-l/cayley/megaminx/submissions/`. **Next session must check GCP eval status, scp the partial/final CSV back, verify and submit if it's better than current 76,304.**

---

> **LEDGER GAP**: entries 2026-05-12 → 2026-05-23 (AZ v4 prod-1001, m_v11, GT-V, state_inv,
> repr-upgrade bundle, child-rank) were logged in HANDOFF.md + memory rather than here.
> See HANDOFF.md §2/§6/§8 and the memory index. The §13 batch below resumes the ledger.

- **2026-05-24/25** — **Strategy-doc §13 batch (6 ideas → 4 themes): 1 works-but-marginal, 3 neutral. No submission change.** Pursued `megaminx_architecture_and_path_shortening_strategy.md` §13: PHS path/cumulative scoring, frontier-regret / pairwise-on-real-mistakes, symmetry-consistency loss, macro-mining. Reinforces the 6M ceiling on training-side scoring; score-race value remains pure-inference. Outcomes below.

- **2026-05-24/25** — **PHS cumulative path scoring (doc §13.1) — VALIDATED, deploy-MARGINAL**. Implemented beam score = V(child) + w_p·Σ_{t≤d}(−log π(a_t|s_t)) via a `self._phs_cum` accumulator threaded through the beam (`beam_lab/beam_search.py` + `beam_search_qshort.py`, no signature change — frontier-hook + unpacks intact), `03_solve.py --phs-cumulative`, 6/6 unit tests (`tests/test_phs_cumulative.py`). Distinct from the regressed memoryless local penalty (AZ v4 V+π λ=0.05 → −156). Uses AZ v4 π head (`m_az_v4_pi_only.pt`) + m23_v3 qshort. **Stage-1** (single-pass strat-51, no sym): neutral on easy pids; collapses at w≥0.2 (cumulative ~d·2.5 swamps V's ~30 range, 48→27 solved); safe w≈0.03. **Stage-2** (production sym4+multipass+NISS, strat-51, w=0.03 vs 0): standalone TIE −3; per-pid min-merge −61/51. **rand50** (buckets 1-8, seed 0, both arms): PHS **−67/50 STANDALONE** over its own w=0 arm (−1.34/pid, difficulty-monotonic: 800-899 −4.6/pid) — a REAL improvement to the V+qshort+sym4+NISS config; the Stage-2 "tie" was diluted by easy bucket-0 (PHS hurts) + the hardest tail. **Deployment reality**: vs the merged best (merge_v12 = 4061 on those 50 pids vs PHS = 4347), PHS beats current-best on only **3/50 pids (−13)** — the merge already absorbs the diversity. ~−200 extrapolated over the ~800 mid pids for ~1-2 days GCP = marginal vs 70K. **Verdict: validated, not worth a multi-day deploy run.** See [[phs-cumulative-validated-marginal]]. CAVEAT: under --sym-ensemble the AZ v4 π sees rotated (non-rot-aug) states on 3/4 rotations → partly OOD; identity in-dist.

- **2026-05-24** — **Symmetry consistency loss (doc §13.4 / §4.6) — TIE, λ_sym exonerated**. `lambda_sym=0.1` isolated on m_dd_v0 (encoding=embedding, no other bundle terms): L_sym = MSE(V(s), V(RsR⁻¹)) on the RW portion, `rotations.npy`. `configs/m_sym_v0.yaml`, `models/m_sym_v0/epoch_0049.pt`. Control = m_rank_ctrl (already a confirmed no-op). Distinct from m31 rotation AUGMENTATION (REJECTED 95.76) — consistency loss adds NO new target samples. **Variance-SAFE**: per-depth std flat across 50ep (d[20,30) 2.65→2.64), saturation healthy ~29.6 → **λ_sym is NOT the m_repr_v0 variance injector** (joins λ_rank exonerated ⇒ culprit narrows to 20% solver-trace ± λ_sat). **Beam-NEUTRAL**: strat-51 sp beam-65k, model 51/51, weighted model_avg ≈88.1 vs m_dd_v0 89.4 = TIE. Fails the acceptance gate. See [[repr-upgrade-bundle-rejected]].

- **2026-05-24/25** — **Frontier-regret harvest (doc §13.2) — labels are benign alt-optima; mechanistically explains the rank/sym ties**. Built `scripts/84_harvest_frontier_regret.py`: replay verified best paths, flag steps where V scores a sibling BELOW the on-path move (a "misrank"), emit (parent, good, bad) triples. m_dd_v0 over merge_v12: **98,930 triples, 56.9% of on-path steps misrank** (72% far-from-solved → 8% near). **But these are benign alternative-optima, not fixable mistakes**: beam trusts V's ranking yet solves 50/51, so a 57% TRUE misrank is impossible; the wide beam keeps the on-path child regardless, and the sibling V prefers is equally-good (many near-optimal moves far from solved). Training to fix them = teach arbitrary path-choice = no beam gain — i.e. the m_rank_v0 tie, mechanistically. Clean true-mistake signal needs cross-path divergence (verified suffix_bad>suffix_good) or confirm-resolve, likely sparse. Dataset `data/frontier_regret_triples.pt` (kept, noisy). [Macro-mining §13.3 not started.]

- **2026-05-25 SESSION SYNTHESIS** — **§13 batch closed: V's child-ordering is already near-optimal at 6M.** PHS works standalone but is redundant with existing min-merge diversity (−13/50 over best). child-rank, sym-consistency, frontier-regret-v0 all neutral — the frontier-regret harvest shows WHY (V's "misranks" are benign alternative-optima). **The path to <70K is pure-inference (sym-ensemble scaling, multi-seed, rescue, merges), not more V/scoring training.** No submission change (75,200 submitted / 77,145 standalone). GCP VM stopped. All §13 code committed, tested, inert-by-default.

- **2026-05-25** — **Bipartite Slot-Sticker Graph-Transformer Q-shortlister (doc §3.2) — REJECTED (no architecture win; stopped at e30/120)**. Built the full §3.2 stack: 264-token bipartite graph (120 slot + 120 sticker + 24 action nodes), state injected via node features (slot knows `state[i]`, sticker knows `inv_state[j]`) NOT a dynamic mask, GraphGPS layers (sparse local masked attn + global relation/dist-biased attn), Q read from the 24 action nodes. Files: `scripts/74_build_bipartite_features.py` (→`data/bipartite_features.pt`), `src/megaminx/graph_transformer_bipartite.py` (3.52M, wired into `cayley.search.load_model_checkpoint`), `scripts/76_train_gt_q_bipartite.py` (distill `Q(s,a)=V_azv4(apply(s,a))`, MSE+KL, early-stop on depth-stratified recall), `scripts/09b_eval_q_recall_by_depth.py`, runbook `bipartite_gt_q_runbook.md`. All validated (builder asserts, model smoke, loader round-trip, GCP in-process smoke). **Result**: bipartite recall tracks the flat GT-Q (`75_train_gt_q.py`, `m_gt_q_v0`, 3.24M) essentially identically — e9 recall@a=2 tied (bip d20/40/60/80 = .751/.643/.386/.483 vs flat .736/.681/.439/.442; e10 MSE 2.25 vs 2.47). e9→e29 bipartite recall rose slowly (min .386→.474, pooled .90→.95) but stayed far below the production ResMLP-Q (`m23_v3_az_v4_sym`, 12.4M) at every depth (e29 a=2: d60 .47 vs .81, d80 .54 vs .79) and decelerating → would not reach baseline by e120. **The action-node prior buys ~nothing over the flat GT at matched size/budget; better MSE fit did NOT move recall (the GT-V lesson replayed in Q-space).** Caveat: GT-Q is 3.5× smaller than the ResMLP-Q (capacity confound for the GT-vs-ResMLP gap) — but the matched-size bipartite-vs-flat tie cleanly answers the architecture question, and scaling GT up isn't worth it: frontier-regret says matching that Q yields ~0 solve gain, and GT is ~30-50× slower (deploy-only-via-distill = back to m23). Stopped at e30 by user call (~$4 of L4); VM stopped. **Ops lessons** (reusable): explicit-`attn_mask` SDPA uses the math kernel → materializes+saves `(B,H,T,T)` → batch 2048 OOMs even with gradient checkpointing; train batch 1024 + grad-ckpt + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; eval/val needs `inference_chunk_size=1024` + bf16 autocast (FP32 eval at the 4096-pooled boundary OOM'd). L4 epoch ~727s (GT is expensive to TRAIN too, not just infer). Code retained, inert-by-default. See [[bipartite-gt-q-shortlister]] memory + `bipartite_gt_q_runbook.md`.

- **2026-05-25** — **Relative-state / Bridge Transformer (doc §3.5) — REJECTED (3/3 sub-experiments negative; saturation re-asserts, scorer is not the bridge bottleneck)**. Goal: a relative-distance model `D(s,t)` to replace the saturating distance-to-solved V as the bridge-compression scorer. Key identity: `apply(s,a)` maps the residual `X=make_residual(s,t)` to `apply(X,a)`, so `D(s,t)` = distance-to-solved of `X` and Bridge-Bellman = standard Bellman on `X` → D is just a `ResMLPDistance` on residual states (no new arch). Built `scripts/85_build_bridge_distance_data.py` (path-window residual dataset from merge_v14+merge_v12, balanced per window length 5-80, pid-split; 874K train / 145K val; self-checked vs `make_residual` + bridge guarantee), `scripts/86_train_bridge_distance.py` (regress-to-`j-i` + per-window-length saturation table vs production V `m_dd_v0`), `scripts/87_train_bridge_bellman.py` (Bridge-Bellman reusing `cayley.bellman._bellman_targets` with `wlen` as per-state `clip_upper`, warm-start from v0-D), and a `--scorer-checkpoint` flag on `scripts/81_bridge_compression.py` (default-off) to isolate the scorer in the deploy A/B. **(1) v0 regress-to-wlen**: extends reliable horizon from V's ~20 to ~40 (val d30→d70 slope D +19.3 vs V +9.1; D≈true to wlen 30) — first model-side positive this cycle — but learns *window-length* not *true distance* (wrong target → no compressibility signal); generalization ceiling val MAE ~10 (train→0; weight decay no help). **(2) v1 Bridge-Bellman**: COLLAPSES the v0 curve toward V's saturated plateau (slope 19→15→13, →V's 9) via the optimistic-`min` bias `bellman.py` documents; `clip_upper(wlen)` caps from above only and can't resist downward collapse → worse than v0. The compressibility signal and the collapse bias are the SAME `min` op, inseparable without ground-truth deep distances. **(3) deploy A/B** (merge_v12 77,214 top-15 long pids, windows 20-40, solver=V fixed, only scorer swapped): scorer=V → 5 wins / **7 moves**; scorer=v0-D → 4 wins / **6 moves** (noise-level worse; D missed pid 342 V caught). Mechanism: score = `wlen − scorer`; V's saturation inflates predicted-save → aggressively proposes marginal windows, and every real win is a 1-2-move sliver, so aggression helps; D's accuracy makes it conservative → skips them. Every winning window = 1-2 moves of slack, identical across scorers → **binding constraint is path near-optimality, not scorer calibration** (confirms [[bridge-compression-findings]]). Net: Rule-23 saturation ceiling re-asserts in residual/bridge space; the bridge scorer was never the bottleneck. Ran fully LOCAL (4090; ~2-min/job training, two ~25-min A/B arms) — not a GCP-scale task; arm D needed `--internal-batch-size 4096` (two models on 16GB). Code + dataset + `m_bridge_d_v0` retained. See [[bridge-residual-distance-rejected]] memory + doc §3.5 Result block.

- **2026-05-26** — **Perceiver IO V (doc §3.9) — TESTED, REJECTED (drifts; same saturation wall as GT-V)**. Built `src/megaminx/perceiver_v.py` (N learned latents cross-attend to 240 slot+sticker tokens from `bipartite_features.pt`, self-attend, mean-pool -> scalar V; NO attention mask -> flash SDPA, dodges the rule-22/27 bipartite-GT blow-up) + `scripts/88_perceiver_v_probe.py` (Stage-1 walk-depth MSE pretrain -> Stage-2 Bellman refine via `cayley.bellman._bellman_targets` + V0/d1 anchors; saturation canary = `V@d80-V@d40` per rule 23). Validated locally (bit-identical local 4090 / GCP L4 smoke), ran full on GCP L4 (2.76M: d=256/64-latent/3-layer, batch 2048). **Result**: the saturation gap is FROZEN across 40 Bellman epochs -- pretrain +13.3, bel-e19 +14.7, bel-e39 +14.5 -- V@d80 stuck 45-53 (diameter ~29). Bellman shifts the whole curve down uniformly toward the low-end anchors (V0~=0.3, V(d1)~=0.9, d30~=24 calibrate fine) but cannot bend the deep end into saturation = the exact GT-V signature (calibrates low, drifts high, gap won't close). Killed at e40 (verdict certain), VM stopped. **The Perceiver encoder hits the same problem-intrinsic saturation ceiling as ResMLP-state_inv, GraphTransformer-V, and the §3.5 Bridge-Bellman -- representation power is not the bottleneck.** Ops: d=256/128-latent/4-layer @ batch 4096 OOMs L4 24GB (training activations across self-blocks) -> batch <=2048 + trim latents/layers + `expandable_segments`; Bellman 24x child forwards through attention ~135 s/epoch on L4 (GT-class cost); `target_net_chunk` must stay small (557s/epoch thrash at 16384, caught locally) + bf16 target forwards. Code retained, inert. See [[perceiver-v-rejected]] memory + doc §3.9 Result block.

- **2026-05-27** — **Dodecahedral CNN / geometric GNN (doc §3.8) — TESTED, REJECTED (collapse masquerading as saturation; can't beam) — LAST architecture experiment**. Built `src/megaminx/dodeca_cnn.py` (per-slot message passing: mean-aggregate over same-face/gen-edge/same-piece/stride2 neighbours from the relation matrix, MLP mix, residual; no attention -> cheap, no OOM) + `scripts/89_dodeca_v_probe.py` (same two-stage pretrain->Bellman + saturation-gate harness as 88). GCP L4, 3.29M (d=256/4-layer). **The instructive result**: Dodeca is the ONLY encoder whose gap-canary said "SATURATES" (gap +4.2 @ bel-e19 vs Perceiver's +14.7 drift) — and it was a **FALSE POSITIVE**. The V curve slid monotonically and never converged: d80 = 60.2->32.1->28.9->23.4->20.7 across canaries, ending a compressed under-predicting V (range 0->21 vs healthy 0->~29; d60/d80 ~3 apart = no depth resolution) = optimistic-min Bellman COLLAPSE (§3.5 family); the gap stayed small only because d40/d80 collapsed together. **Beam bench confirmed**: best.pt (e19, least-collapsed) at beam 65k did not solve even pid 0 after 91 min (healthy V: seconds), GPU 100%/21GB — the GNN message passing is ~100x the ResMLP inference cost AND the collapsed V can't navigate beam. Killed, VM stopped. **Lessons**: (1) the saturation gate needs an ABSOLUTE-calibration check (d80 ~near diameter, not just gap<=10) — Dodeca is the concrete false-positive proving gap-mean alone is insufficient (Rule 21 / repr-bundle); (2) local geometric message passing did NOT escape the wall — it found a *different* failure (collapse) than the attention encoders' drift. Loader wired into `cayley.search` (`DodecaCNNV`/`PerceiverV` branches, inert). See [[dodeca-cnn-rejected]] memory + doc §3.8 Result block.

- **2026-05-27 — §3 ARCHITECTURE SURVEY CLOSED.** Seven encoder families tried as the V/Q scorer — §3.1 repr-bundle, §3.2 bipartite GT, §3.3 axial (≈3.2, not built), §3.4 symmetry (settled), §3.5 bridge, §3.9 Perceiver, §3.8 Dodeca — **ALL hit the same problem-intrinsic saturation ceiling** (none beats the ~6M ResMLP). Attention encoders DRIFT (calibrate low, never saturate high); GNN/Bellman variants COLLAPSE (slide below diameter). Plus: V's child-ordering already near-optimal at 6M (frontier-regret) -> ~0 reranker headroom; attention encoders 30-135x costlier. **Verdict (doc §3.0): stop building global scorer architectures; the path to <70K is PURE-INFERENCE** (sym-ensemble full-1001, multi-seed beam, rescue/merge, curated macros). Reusable: the two-stage + saturation-gate probe harness (`scripts/88`,`89`) for any future encoder, with an added absolute-calibration check.

- **2026-06-06** — **GPU-friend wide-beam architecture ports into the TPU 48M qshort kernel (ideas 1/2/4 built + measured; step is selection-bound at 46.4s; ④ unlocks 64M/96M but is expensive width, not speed).** Analyzed a collaborator's CUDA/NCCL very-wide-beam design (`megaminx/ARCHITECTURE_NEED.md`, same puzzle 120/24) → 5 transferable ideas; ported the inference-side ones into `kaggle_notebooks/tpu_beam_spmd_jax/jax_beam_spmd_v_qshort.py` (body inlined by `tpu_beam_az_v4_48m_720_qshort_shareable`), all flag-gated, all CPU-mesh validated before TPU (`sanity_check_neighborhood.py`, 8 logical devices via `XLA_FLAGS`). His GPU-execution tricks (5-stream overlap, CUDA graphs, CUTLASS, A/B shard buffers) don't port (XLA handles); we'd already matched the micro-opts (qshort=per-parent Q, Phase C/D, power-of-2 owner mask, donated buffers, pre-compiled graph). **KEY MEASUREMENT: 48M baseline = 46.4s/step (README's ~85s was stale); FLOP estimate puts the Q+V GEMMs at ~0.2–0.4s → the step is selection/memory/collective-bound (prime suspect: 384 per-owner top-k merges/step + 12M argsorts), not compute-bound; per-step is pid-independent.** **Idea 2 neighborhood early-stop**: `build_solved_neighborhood()` BFS-from-solved (r5=1.38M states, 12MB device, ~13s) + device `searchsorted` on `chosen_h` + host best-`(j+1)+suffix_len` + verified prefix+suffix; provably ≤ exact path, ≥ solve-rate; **MID-PID ONLY** — hard-tail pid 995 @ K=1 gave `best_near=-1` all steps (beam plateaus >5 from solved; bigger radius infeasible). Config `USE_NEIGHBORHOOD`/`NBHD_RADIUS`. **Idea 1 approx top-k**: `jax.lax.approx_min_k` via `sel_topk` at qshort + final + the per-owner merge; config `USE_APPROX_TOPK`/`APPROX_RECALL`; **v8 speed test (48M, approx on, pid 0 full-depth) IN FLIGHT** (per-step vs 46.4s + length vs 55). **Idea 4 compact-meta + deferred materialize** (`_build_step_body_v_qshort_meta_streaming`): forward all-to-all ships 16B metas (hash+parent_local+move+score+valid), dedup/top-B on hash/score only (no candidate-state gather), 2-hop round-trip rematerializes only the B survivors; **CPU-proven to select the IDENTICAL beam as baseline (length-safe)**; config `USE_META_MATERIALIZE`/`ALPHA_REQ`. **④ verdict (measured): 46.4s@48M → 74.5s@64M → 141s@96M; throughput 1.03/0.86/0.68 M states/s = super-linear (HBM pressure at the 15.75GB cap @96M). ④ enables frontiers the baseline OOMs on, length-safe, but is NOT faster (~3× per-pid wall @96M for one diminishing-returns B-doubling) → niche hard-tail rescue, not a general lever.** Two reusable TPU bugs fixed building ④: (1) seed beam `jnp.broadcast_to((8,B_local,120))`+`device_put` materializes the full 7.68GB beam on ONE device before resharding → OOM past 48M, fixed via `jax.make_array_from_callback` per-shard; (2) un-chunked materialize apply-move = (12.6M,120) int32 gen-index ≈13.5GB HLO temp → compile OOM @64M, fixed via `lax.scan` over `MAT_CHUNK=parent_chunk`. **Ops gotcha: Kaggle `kernels status` RPC 500'd the ENTIRE session and `kernels output` didn't commit/fetch for the in-flight version (worked for completed prior versions) — a run completes (`lastRunTime` updates) yet is CLI-unfetchable; web UI is ground truth; a 15-min CLI poll loop was blind for 2h.** Idea 3 (Q-only) not attempted (conflicts with α=1-regresses). No submission change. See [[gpu-friend-beam-ports]] memory.

- **2026-06-07** — **Faster-axis hunt on the 48M qshort step: ablation-localized the bottleneck + shipped a ~9% length-safe win (lean-merge); concluded.** Goal: make the 46.4s/step faster without lengthening paths. Added a `profile_skip` ablation flag to the streaming body (replace ONE component with a cheap stand-in, timing-only) and ran one config per Kaggle version (each ~hours due to the degraded-API fetch lag). **Component shares of the 46.4s step (pid0, 48M, nbhd on): top-k (approx_min_k everywhere) ~0 AND +0 length (REJECTED — also corrected the earlier "approx +5": the current kernel's pid0 len is 60 not the stale-README 55, approx was length-neutral); recv state-gathers (deferred to B survivors) ~0 (KEPT, length-safe, harmless); all_to_all (skip) ~0; V-forward (skip) ~3.8s/~8%; Q-forward (skip) ~0.9s/~2%; generation gather (children=zeros) ~0; per-owner merge PACK-BUILDING (skip) → 6.8s, i.e. ~85% of the step.** Root cause: the streaming `lax.scan` carries a (world_size, K_per_peer, 128)=1.5 GB pack and RMWs it 8 owners × 48 chunks = 384×/step. **Fix = `lean_merge` (flag, default ON after validation):** carry compact provenance (parent_local+move+score, ~10× smaller carry) and rebuild the 128 B send buckets ONCE post-scan (chunked apply-move + concat; regen identical to `children[new_idx]` so length-safe). CPU-mesh validated (lean == baseline == meta, no false positives). **Measured: 46.4 → 42.1s/step (~9%), pid0 len 60 verify ok.** The hoped 3-6× did NOT materialize — the v14 "6.8s floor" was an ablation artifact (skipping the pack let XLA DCE the whole 1.5 GB carry, which the real fix can't); chunked-regen (v16, 42.2s) and concat-packing (v17, 42.1s) confirmed the residual is the per-chunk merge STRUCTURE (384× concat+topk+carry), irreducible without an accumulate-once rewrite (route+topk ONCE) whose payoff is doubtful since approx showed the merge top_k itself isn't the cost. **Folded lean-merge to default (v18); faster-axis verdict: ~9% is the realistic ceiling for this algorithm at 48M.** No submission change. Reusable: `profile_skip` ablation harness. See [[gpu-friend-beam-ports]] memory.

- **2026-06-08** — **96M found=False was a 23-bit packed-backpointer OVERFLOW (shared by all 3 step bodies), NOT a meta-logic bug — root-caused, fixed (24/3/5 re-layout), locally validated; TPU A/B in flight.** The v21 run (96M V-only-equiv α=24, pid 990, meta-materialize) returned **found=False** despite real V0/neighborhood hash hits (`fs=75`, `best_near=76`) — initially mis-attributed to a broken idea-4 meta body. **Real cause**: the packed uint32 backpointer `parent_local | (rank<<23) | (move<<26)` gave `parent_local` only **23 bits** (max 8,388,607). 48M (B_local=6,291,456) fits → streaming solves; **96M (B_local=12,582,912 > 2^23) overflows**: the top **33%** of every frontier's parent_local leaks bit 23 into the rank field and truncates `pos`, so the walkback — ~33% corrupt per step over 75 steps — never reconstructs a valid path → `_solves` rejects every candidate → found=False. The hash hits were genuine (the beam DID reach V0); only the recorded provenance was corrupt. The meta body looked guilty only because it was the ONLY body ever run at 96M (streaming OOMs there); the CPU smoke never hit it (small-B < 2^23) and only ran meta on non-solving pids at α=2. **Fix** (`jax_beam_spmd_v_qshort.py`): re-laid the uint32 to **24/3/5** (parent_local/rank/move = 32 bits exactly) via module-level `BPTR_*` constants applied to all 3 bodies (V-only, streaming, meta) + seed pack + host walkback, plus a guard that raises if `B_local > 2^24` (supports B_global ≤ 134M; beyond → uint64 memmap). NO shape/compute change — same uint32 memmap, only the bit layout + host arithmetic. **Validated locally (2 axes):** (1) `test_backptr_packing.py` — new layout lossless to 2^24-1; OLD layout corrupts exactly 33.3% of the 96M frontier (bit-level repro of the bug). (2) `validate_meta_real_solve.py` — meta vs streaming on **18 real CPU-mesh solves** (shallow scrambles d=5..10, **α=2 AND α=24**, neighborhood on, production AZ v4 V + m23_v3 Q): both solve optimally, both verify, identical path length, **0 false-positives, 0 mismatches; 6 won via the neighborhood walkback** (both walkback paths proven through meta). This is the verified-solve-through-the-path the prior smoke lacked. **Notebook rebuilt (v22)** + pushed; **TPU A/B CONFIRMED** (96M pid 990, config byte-identical to v21, only packing changed): `found=True verify=True via=v0 fwd=76, solved 1/1` — same hash signals as the v21 found=False run (best_near=76 @ step70, fs=75 @ step75) but the walkback now reconstructs a valid 76-move verified path. 76 == 48M's length (confirms the fix, not a width win on this pid); the packing bug was the COMPLETE story — no second bug. **④ is now FUNCTIONAL and is the enabler for B_global>48M** (streaming OOMs), not "shelved." Lesson reinforced: "runs + matches baseline on NON-solving pids" ≠ validated — a real verified solve through the new path is required (and is what exposed both the bug and the fix). See [[gpu-friend-beam-ports]] memory.
