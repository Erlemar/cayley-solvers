# Progress Report — IHES Picture Cube

**Project**: Kaggle CayleyPy IHES SuperCube — 1003 scrambled picture-cube puzzles, total-move-count metric (lower is better).

## Headline

| Date (UTC) | Total moves | Gain vs prior |
|---|---|---|
| 2026-04-16 (baseline) | **42,718** | — |
| 2026-04-22 (final) | **23,224** | **-19,494 (-46%)** |

Each step below contributed to moving the submitted score down. Every step was verified end-to-end (valid paths on 1003/1003 puzzles) before submitting.

---

## Submitted progression

| # | Date | Score | Delta | What changed |
|---|---|---|---|---|
| 1 | 2026-04-16 | 42,718 | — | Baseline: Kociemba two-phase solver, v13 multi-prefix + center lookup. |
| 2 | 2026-04-17 | 30,770 | -11,948 | **Neural V-heuristic + beam search.** ResMLP with onehot encoding ([700,643]×4 ResBlocks, 7.4M params), beam 4096 on 18-gen cube, Kociemba fallback on unsolved. |
| 3 | 2026-04-17 | 30,120 | -650 | **Embedding encoder** swapped in (`nn.Embedding(72, 16)`, 4.6M params). Smaller input (1152 vs 5184 dims), same trunk. Fewer fallbacks (174 vs 201). |
| 4 | 2026-04-18 | 29,710 | -410 | **Fast recipe**: bf16 + torch.compile + fused AdamW + batch 16K → 3× training speedup at the same accuracy. |
| 5 | 2026-04-18 | 28,224 | -1,486 | **Wider beam** (4096 → 8192) + BFS-depth-5 window-replacement post-processing. |
| 6 | 2026-04-18 | 27,790 | -434 | **2-seed ensemble** (min-merge across seeds). |
| 7 | 2026-04-18 | 27,366 | -424 | 4-seed ensemble. |
| 8 | 2026-04-18 | 27,106 | -260 | **MITM** (meet-in-the-middle BFS depth 6 from solved state) + 3-seed ensemble. Adds 44 model-solved puzzles. |
| 9 | 2026-04-18 | 24,998 | **-2,108** | **Architectural win: small arch + khoruzhii beam searcher.** Embed [1024,256]×1 ResBlock (1.6M params), K_max=26, trained 4000 epochs. Replaced CayleyPy's beam with our port of Khoruzhii's NeurIPS-2025 searcher → beam 65,536 feasible. Model E3 solves 996/1003 with the wider beam. |
| 10 | 2026-04-18 | 24,618 | -380 | **E5 = E3 × 2 epochs** (8000 total). |
| 11 | 2026-04-19 | 24,068 | -550 | **E6 Bellman refinement + NISS.** Self-bootstrapped Bellman targets `y = 1 + min_a V_target(apply(s, a))` starting from E5 (500 epochs, target-net refreshed every 10 epochs). **NISS** (Niss: solve the inverse scramble σ⁻¹, convert path back) added as ensemble member. NISS alone solo is ~par with forward, but contributes -432 moves in ensemble (different directional coverage). |
| 12 | 2026-04-20 | 24,018 | -50 | **Targeted wider-beam tail re-solve**: for every puzzle with baseline path length ≥ 27, re-run beam 131,072 + NISS, keep the shorter. 24/40 puzzles improved. |
| 13 | 2026-04-20 | 23,944 | -74 | Same pattern, extended to length ≥ 26 (148 puzzles). 33/148 improved. |
| 14 | 2026-04-20 | 23,868 | -76 | Extended to length = 25 (288 puzzles). 37/288 improved. Diminishing-returns curve observed: 60% hit rate at len≥27, 22% at len=26, 13% at len=25 → plateau at len ≤ 24 for this beam. |
| 15 | 2026-04-20 | 23,858 | -10 | Added an additional same-recipe model (data-bundle ensemble) — marginal. |
| 16 | 2026-04-20 | 23,672 | **-186** | **int8 state encoding** in khoruzhii searcher → 6× VRAM reduction (beam 65K goes from 1.56 GB to 0.26 GB). This unlocked **beam 524K** on our 16 GB GPU (previously capped at 131K). Re-ran tail re-solve at beam 524K on 108 len≥26 puzzles: 74/108 improved (69% hit rate, up from 22% at beam 131K). Average save per improved puzzle: ~2 moves. |
| 17 | 2026-04-22 | 23,322 | -350 | **Beam-1M full solve + Q-function distillation.** Two big components: (a) the int8 encoding further enabled a full 1003-puzzle solve at beam 1,048,576 with our E6 V-heuristic (wall ~19h, large-search gains on mid-difficulty puzzles). (b) **QE6 Q-function**: distilled the 18-neighbor scorer from E6 into a single 18-output head (`output_dim=n_gen=18` on `ResMLPDistance`) — instead of running 18 separate V forwards per beam step, one Q forward outputs all 18 neighbor scores. Gave ~8× beam-search speedup, unlocking wider-beam solves in the same wall time. |
| 18 | 2026-04-22 | **23,224** | **-98** | **QE6 beam-4M full solve (GCP L4, ~9h) + beam-4M tail re-solve on len≥25**, min-merged with prior pipeline + BFS-d5 PP. Widest beam we ran; picked up residual improvements on mid-difficulty puzzles the beam-1M pass missed. Post-processed output unchanged — the min-merge paths are already at the BFS-d5 window ceiling. |

---

## Summary of the key wins by category

- **Neural heuristic quality** (baseline → E6 Bellman): ~14,000 moves of improvement. The small-arch [1024,256]×1 trunk plus Bellman self-bootstrapping is the V-function that carried us below 25K.
- **Search-side engineering** (beam search port, int8, Q-function): ~2,700 moves from architecture-independent search infrastructure. The Khoruzhii beam searcher port, int8 state buffers, and Q-function distillation each compound with every subsequent model improvement.
- **Ensemble techniques** (multi-seed, MITM, NISS): ~1,200 moves from consuming directional/seed diversity without training new base models.
- **Targeted tail re-solves at progressively wider beams**: ~500 moves extracted by spending wider-beam compute only on the longest-path puzzles — cheap to run once the wider-beam search infra exists.

## Technical details

### Model architecture

All production models share the same small trunk. `ResMLPDistance` (in [`src/cayley/model.py`](src/cayley/model.py)):

- **Encoder**: `nn.Embedding(num_classes=72, embed_dim=16)` on the 72-sticker input → flatten to 1152 dims. One-hot was tried first and regressed slightly.
- **Trunk**: `Linear(1152 → 1024) → ResBlock(1024 → 256 residual) → Linear(256 → head)`.
- **Head**: `output_dim=1` for V-heuristic models (E3/E5/E6/E9), `output_dim=18` (= n_generators) for the Q-heuristic model (QE6).
- **Parameter count**: **1,579,649** (V head) / **1,584,018** (Q head). ~1.6 M params — small enough to keep inference chunks large without OOM, which is what makes wide beams feasible.
- **Precision**: trained in bf16 (`torch.autocast`) with fp32 master weights; served in bf16 at inference.

We tried bigger (`[5000,1000]×4`, 18 M params; `[2048,1024]×8`, ~23 M) and worse; the 1.6 M version trained faster and generalised better on this problem.

### Heuristic type

Two flavours, both act as **per-state scores**; neither is a policy in the RL sense.

- **V-heuristic** (E3, E5, E6, E9): `V(s) → ℝ`, predicting distance-to-solved. Beam search calls `V` once per child (18 child-forwards per parent). The winning model E6 is a **Bellman-refined V-heuristic**.
- **Q-heuristic** (QE6): `Q(s) → ℝ^18`, predicting `V(apply(s, a))` for each of the 18 actions `a`. Beam search calls `Q` *once per parent* and gets all 18 child scores at once — ~8× inference speedup. Not a policy (no softmax, no action-sampling); it's an action-conditioned value.

### Target generation

Three sources, chronologically:

1. **Random-walk depth** (E3, E5): generate a walk from solved of length `k ≤ K_max=26`, label each intermediate state with its step index. Produces an *upper bound* on `d(s, solved)`. `n_back=1` during the walk (don't pick the inverse of the previous move) to avoid trivial back-and-forth. Label noise: a walk might be non-optimal, so the label is an upper bound.
2. **Bellman self-bootstrap** (E6): warm-start from E5, then `target(s) = 1 + min_a target_net(apply(s, a))` where `target_net` is a frozen deep-copy of the training network, refreshed every 10 epochs. Eliminates the walk-depth upper-bound bias. Clipping: `0 ≤ target ≤ walk_depth` (can't exceed the known upper bound from the walk).
3. **V → Q distillation** (QE6): `Q(s)[a] ← V_teacher(apply(s, a))` where `V_teacher = E6 frozen`. Same random walks as training data; labels are a 18-vector per state (one teacher forward per child). MSE loss across all 18 components.

### Training time (one RTX 4090 Laptop, 16 GB VRAM, bf16 + torch.compile)

| Model | Epochs | Samples/epoch | Steady time/epoch | Total wall | Notes |
|---|---|---|---|---|---|
| E3 | 4,000 | 1,000,000 | ~0.5s | ~35 min | Warm-from-scratch, the architectural-win run. |
| E5 | 8,000 | 1,000,000 | ~0.5s | ~70 min | E3 × 2 more epochs; minor improvement. |
| E6 Bellman | 500 | 1,000,000 | ~6.4s | **~52 min** | 18× more forwards per step (target net on all children). Warm-from E5. |
| E9 symmetry-aug | 4,000 | 1,000,000 | ~1.5s (GCP L4) | ~100 min | Random whole-cube rotation per sample. |
| QE6 distill | 1,000 | ~500,000 | ~3s (P100 Kaggle) | ~50 min | Distill from frozen E6 teacher. Output dim 18. |

Compile cache hit on epoch 0 is ~60 s; all subsequent runs reuse it.

### Inference / solve time

The main bottleneck. Beam search is 18× gather + 1 model forward per step, repeated up to `max_steps=60`, over 1003 puzzles.

| Solve | Model | Beam | Wall (1003 puzzles) | Hardware | Notes |
|---|---|---|---|---|---|
| E3 solo | E3 | 65,536 | ~2 h 45 min | RTX 4090 Laptop | First sub-25K score. |
| E6 solo | E6 | 65,536 | ~2 h | RTX 4090 Laptop | Bellman uplift. |
| E6 NISS solo | E6 | 65,536 | ~2 h | RTX 4090 Laptop | Solves σ⁻¹ in parallel. |
| Beam-524K tail | E6 | 524,288 | ~2 h 15 min | RTX 4090 Laptop | 108 puzzles only. int8 state tensor. |
| Beam-1M full | E6 | 1,048,576 | ~19 h | RTX 4090 Laptop | Full 1003 puzzles; int8 mandatory. |
| QE6 b1M | QE6 | 1,048,576 | ~2 h 15 min | RTX 4090 Laptop | Q-function 8× faster than V at same beam. |
| QE6 b4M tail | QE6 | 4,194,304 | ~2 h 30 min | RTX 4090 Laptop | 189 len≥25 puzzles only. |
| QE6 b4M full | QE6 | 4,194,304 | **~9 h** | GCP L4 (24 GB) | Final big solve; ~32 s/puzzle. |

Post-processing (pair-cancel + BFS-d5 window replacement) on a full submission: ~30 s for 1003 paths.

Ensemble combine: ~5 s (takes the min across candidate CSVs per puzzle).

## Repository touch-points for each win

- Khoruzhii beam searcher port: [`src/cayley/khoruzhii_search.py`](src/cayley/khoruzhii_search.py)
- Bellman refinement training: [`src/cayley/bellman.py`](src/cayley/bellman.py), [`scripts/05_bellman_refine.py`](scripts/05_bellman_refine.py)
- NISS (inverse scramble) support: `invert_state` / `invert_path` in [`src/cayley/puzzle.py`](src/cayley/puzzle.py), runner [`scripts/06_niss_solve.py`](scripts/06_niss_solve.py)
- Targeted tail re-solve: [`scripts/07_target_wider_beam.py`](scripts/07_target_wider_beam.py)
- int8 encoding: `state_dtype` parameter on `KhoruzhiiSolver` ([`src/cayley/khoruzhii_search.py`](src/cayley/khoruzhii_search.py))
- Q-function distillation: `output_dim` parameter on `ResMLPDistance` ([`src/cayley/model.py`](src/cayley/model.py)), Q-path in `KhoruzhiiSolver`, distillation trainer [`scripts/06_train_qfunction.py`](scripts/06_train_qfunction.py)
- Post-processing pipeline: [`scripts/post_process_submission.py`](scripts/post_process_submission.py), BFS-d5 window table [`data/bfs_table_d5.pkl`](data/bfs_table_d5.pkl), ensemble combine [`scripts/combine_submissions.py`](scripts/combine_submissions.py)
- Submit bundle: `/submit` and `/verify-and-submit` slash commands in `.claude/commands/`
