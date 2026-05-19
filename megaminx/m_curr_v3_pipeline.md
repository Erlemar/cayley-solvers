# `m_curr_v3` — full training & inference pipeline

Complete reference for how the **m_curr_v3** V-model is trained and used in
production. Three training stages (m07 → m_curr_v0 → m_curr_v3) feeding a
three-model inference stack on TPU.

The final V checkpoint is `models/m_curr_v3/epoch_0499.pt` (Bellman loss
0.0721 — lowest of any V before m_dd_v0). In production it pairs with
`m23_v2` (Q shortlister) and `m_pi_v2` (policy prior) — submission
**78,029** (standalone personal best on our pipeline pre-m_dd_v0); 76,304
after min-merge with two colleague CSVs.

---

## Stage 0 — `m07_big_k80`: base V with walk-depth labels

**Script:** `megaminx/scripts/02_train.py`
**Config:** `configs/m07_big_k80_local.yaml`

**Data:** every epoch generates **1,000,000 random walks** from solved state
(`generate_walks_torch` in `cayley.data`), each walk up to `k_max=80` long.
`n_back=1` → non-backtracking (cannot apply the inverse of the previous
move). Label per state = step index in walk — an **upper bound** on the
true distance (random walks usually wander further than the shortest path).

**Architecture:** `ResMLPDistance`, **6,045,569 params**:

- Per-position embedding `dim=16` for each of 120 sticker positions,
- Flatten → MLP `hidden_dims=(2048, 512)` with **2 residual blocks**,
- Scalar output (predicted distance).

**Hyperparameters:**

| key | value |
|---|---|
| epochs | 4000 |
| batch | 16384 |
| samples / epoch | 1,000,000 |
| `k_max` | 80 |
| `n_back` | 1 |
| lr | 2e-3 cosine |
| optimizer | AdamW (fused), `weight_decay=0` |
| loss | MSE |
| precision | bf16 autocast |
| compile | `torch.compile` (4090 + Py3.14 triton-windows) |
| seed | 10 |

**Runtime:** ~1-2h on 4090 Laptop.

**Output:** `models/m07_big_k80/epoch_3999.pt`. Predicts walk-depth correctly
but is systematically biased high (it's the upper-bound label, not d).

---

## Stage 1 — `m_curr_v0`: curriculum k_max + EMA

**Script:** `megaminx/scripts/48_train_curriculum.py`

**What changes vs Stage 0:**

1. **Warmstart** from `models/m07_big_k80/epoch_3999.pt`.
2. **Curriculum `k_max`** (`48_train_curriculum.py:162-169`):
   - first **5000 epochs** at `k_max=35` (warmup — focus on near-solved
     states where the value landscape is most informative),
   - after warmup: each epoch picks `k_max` uniformly from **{50, 70, 80, 100}**.
3. **EMA model snapshot** (`tau=1e-3` per step). The EMA copy is the production
   checkpoint (`best_ema.pt`).
4. **Held-out validation set:** 2000 walks at `val_k_max=80` with fixed seed
   = 160,000 states. Validation runs every 100 epochs on the **EMA model**.
5. **Early stopping:** `patience=1000` epochs without val_mse improvement, with
   a grace period until `warmup_epochs + patience` (cannot stop during warmup).
6. **Same** MSE on walk-depth (still upper-bound targets — **no Bellman yet**).

**Hyperparameters:** batch 8192, lr=5e-4 cosine, samples_per_epoch=500k, seed=480.

**Training trajectory (`models/m_curr_v0_training.log`):**

- Warmup k=35 through epoch 4999 → val_mse_ema on k=80 stays ~430 (model only
  sees k=35 walks),
- Mix-k from epoch 5000 → val_mse_ema drops 391 → **69.29 at epoch 5199**,
- Early-stop at epoch 6199 (best EMA = 69.29 at epoch 5199).

**Output:** `models/m_curr_v0/best_ema.pt` — curriculum V body, backbone for
every `m_curr_v*`. Best standalone curriculum chosen_avg on strat-5: **85.41**.

---

## Stage 2 — `m_curr_v3`: Bellman + frontier replay + BFS-d6 anchor

**Script:** `megaminx/scripts/05_bellman_refine.py` (calls
`cayley.bellman.train_bellman` in `src/cayley/bellman.py`).
**Config:** `configs/m_curr_v3.yaml`.
**Warmstart:** `models/m_curr_v0/best_ema.pt`.

Drop walk-depth labels and switch to DeepCubeA-style self-bootstrapped targets
with three state streams per batch.

### Three data streams per batch (8192 samples)

From the training log header:

```
[bellman]   19,352,405 BFS-d6 states; per-batch mixin = 819/8192 (10%)
[bellman]   300,000 frontier states; per-batch mixin = 2048/8192 (25%)
```

Total per batch of 8192:

- **5325 RW** (65%): fresh random walks at `k_max=80`, `n_back=1`,
- **2048 frontier** (25%): from `data/frontier_states.pt`,
- **819 BFS-d6** (10%): from `data/bfs_d6_train.pt`.

### Target sources per stream

| stream | state source | target | label source |
|---|---|---|---|
| RW | fresh walk | `clip(1 + min_a V_target(apply(s,a)), 0, walk_depth)` | **bootstrap** via target net |
| frontier | `frontier_states.pt` | same Bellman bootstrap, synthetic `clip_upper` cap = 200 | **bootstrap** via target net |
| BFS-d6 | `bfs_d6_train.pt` | **exact distance** 0..6 | ground truth (BFS from solved) |

The Bellman target uses a **frozen target net** — a copy of the trainable
model, refreshed by hard copy every **10 epochs**
(`target_update_every_epochs=10`, see `src/cayley/bellman.py:692-703`). Target
net forward passes are chunked at `target_net_chunk=4096` for memory.

### Frontier states — what they are and how they're collected

**Built by** `megaminx/scripts/42_log_frontier_states.py` (DAgger-style
state-distribution mixin):

- Runs **vanilla V-only beam search** on 100 pids drawn from buckets 4..7 (medium
  difficulty), `beam=16384`, `max_steps=80`.
- Logs all surviving states at every beam step.
- Deduplicates by state bytes across pids, caps at 200k/pid raw → **300,000
  unique states** in the final set.
- **States only, no labels** — target is computed at training time via the
  Bellman bootstrap (`src/cayley/bellman.py:613-635`).
- Idea: random walks cover the neighbourhood of solved well but poorly cover
  states the beam actually visits at non-trivial depth. Frontier replay shows
  the model the states its heuristic will be evaluated on.
- Structurally **different from m43 solver-trace mixin** (which used realized
  suffix-length labels and went OOD-catastrophic). Here the labels stay
  Bellman; only the state distribution changes.

### BFS-d6 anchor

`data/bfs_d6_train.pt` — 19.35M states at distance 0..6 from solved with
**exact** distances. Built by `megaminx/scripts/14_build_bfs_d6_dataset.py`
from `bfs_bytes_d6.pkl` (full BFS shell, 2.38 GB on disk in `BfsBytesTable`
bytes-keyed format).

Per-depth counts:

| d | states |
|---|---|
| 0 | 1 |
| 1 | 24 |
| 2 | 408 |
| 3 | 6,208 |
| 4 | 90,144 |
| 5 | 1,280,160 |
| 6 | 17,975,460 |
| **Σ** | **19,352,405** |

Mixing 10% of these into every batch with exact labels anchors V to ground
truth on the d≤6 shell — closes the bootstrap circularity (without anchors a
Bellman fixed-point can drift arbitrarily; the m07 V has V(V0)≈2 instead of 0
exactly because of this).

### Loss and optimizer

- **MSE** averaged across the full combined batch (8192).
- AdamW (`weight_decay=0`), lr=**5e-4** cosine, fused, bf16+`torch.compile`,
  batch=8192, **500 epochs** (no early stop in this config — `patience=0`).
- All extras off: `clip_upper=True, clip_lower=True`, `softmin_temperature=0`,
  `lambda_pdb=0`, `lambda_upper=0`, `rotation_aug_prob=0`,
  `double_bellman=False`, `n_anchor_v0=0, n_anchor_d1=0`.

### Loss curve

From `models/m_curr_v3_training.log`: loss drops **10.94** (epoch 0) → **0.0721**
(epoch 499). Lowest Bellman loss of any V before m_dd_v0 was introduced.

**Output:** `models/m_curr_v3/epoch_0499.pt` — production V teacher.

---

## Comparison of the m_curr_v* lineage

There is no `m_curr_v1` (the version was skipped). Lineage v0 → v2 → v3, all
identical 6M params, same architecture, differ only in training recipe:

| | warmstart | target signal | mixins | purpose |
|---|---|---|---|---|
| **m_curr_v0** | `m07_big_k80/epoch_3999.pt` | curriculum: warmup `k_max=35` 5000ep → mix-K {50,70,80,100} | — (random walks, `n_back=1`) | base "curriculum body" — best curriculum chosen_avg 85.41 |
| **m_curr_v2** | `m_curr_v0/best_ema.pt` | Bellman + target-net refresh / 10 ep | + frontier 25% | Option B: does Bellman keep the curriculum advantage? |
| **m_curr_v3** | `m_curr_v0/best_ema.pt` | Bellman (same) | + frontier 25% **+ BFS-d6 anchor 10%** | Option C / kitchen-sink — adds BFS-d6 boundary on top of v2 |

Technically v3 = v2 + the `bfs_d6_fraction: 0.10` line in `configs/m_curr_v3.yaml`.

---

## Inference — qshort + policy + sym, on TPU v3-8

**Notebook:** `megaminx/kaggle_notebooks/tpu_beam_m_curr_v3/cayleypy-tpu-beam-m-curr-v3.ipynb`
(built from `build_notebook.py`).

### Three models on every rank

| model | params | role | applied to |
|---|---|---|---|
| `m_curr_v3` (teacher V) | 6M | reranker — final V(child) | shortlist of size αB |
| `m23_v2` (student Q) | 12.4M | cheap shortlister — Q(s, a) for all a per forward | parents B |
| `m_pi_v2` (policy π) | 6M | prior — log π penalizes unlikely actions | parents B |

All in bf16. Weights live in a Kaggle dataset, loaded into
`M05_STATE_DICT_CPU` / `M23V2_STATE_DICT_CPU` / `M_PI_STATE_DICT_CPU`
(`build_notebook.py:593-613`).

### Parallelism

- **TPU v3-8**, `xmp.spawn` across 8 ranks (`build_notebook.py:680-707`).
- pid-rot pairs (e.g. `1001 pids × K=2 rotations = 2002` pairs) are sharded by
  `i % world_size == rank`.
- Each rank runs a full beam search per (pid, rotation). Per-iteration partial
  save `rank_{N}_partial.json` so a 9h Kaggle kill loses at most one solve.

### Sym K=2 rotations

Megaminx has 60 icosahedral rotation symmetries. For each pid we solve **K
rotated copies** of the state:

- If `R` is a symmetry and `g` a generator, then `R⁻¹ · g · R` is also a
  generator → paths in the rotated frame conjugate back to paths in the
  original frame (`conj_idx` mapping).
- Solve each rotated copy independently, take the **min path length** across
  K — diversity gain because V's landscape has different local minima under
  different rotations.

### Beam loop (per step)

Source: `build_notebook.py:423-569`. Parents `(B, S)` → next parents `(B, S)`:

1. **Expand:** all 24 neighbours `(B·24, S)`.
2. **Hash + dedup:** `(neighbours · hash_vec).sum()`, sort, mask duplicates.
3. **Cheap shortlist (student + policy):**
   - `student_q = student(states)` → `(B, 24)` Q-values,
   - `policy_q = policy(states)` → `(B, 24)` logits → `log_pi`,
   - `student_score = Q + λ·(-log π)`, mask dups → BIG, then **topk αB smallest**
     = shortlist (`α=2` typically, so 2·B child indices).
4. **Expensive rerank (teacher V on shortlist):**
   - `teacher_v = teacher(shortlist_states)` (αB forwards) → rescore
     `V + λ·(-log π)` → **topk B smallest** = new parents.
5. **Tree bookkeeping:** `tree_idx[j] = parent_idx`, `tree_move[j] = move_idx`,
   `min_v_log[j] = min(teacher_v)`.
6. **Solve check:** `chosen_h == V0_hash` (precomputed hash of solved state).
   On first hit save `(found_step, found_pos)`.
7. `xm.mark_step()` — TPU XLA compile boundary.

Everything is **static-shape** — XLA recompiles on any shape change, so every
forward is padded up to `INTERNAL_BS` via `_model_predict_chunked`. The first
iteration is 5-10× slower than the rest (compile warmup).

When `found_step ≥ 0`, the path is reconstructed by walking `tree_idx` /
`tree_move` backward from `(found_step, found_pos)`.

### Production recipe (for actual submission)

The notebook is a SMOKE TEST (single-pass). Production solves use
**multi-pass + NISS**:

```
--beams 16384,65536 --max-steps 60,150 --niss --bf16 --resume
```

- First pass `beam=16384, max_steps=60` — quickly clears easy pids.
- Second pass `beam=65536, max_steps=150` — grinds hard pids.
- **NISS** (Niemiec inverse-state search): also runs beam from the inverse
  permutation, hoping to find a shorter path.
- Sym K=2 + ensemble merge (`megaminx-fork-merge`) across fork notebooks,
  then min-merge with the m_dd_v0 GCP CSV.

### Reference numbers (m_curr_v3 + m_pi_v2 stack)

| run | result |
|---|---|
| single-pass TPU, beam 2¹⁵, ~1.5h on v3-8 | 93,123 |
| single-pass TPU, beam 2¹⁶, ~3h on v3-8 | 88,803 |
| single-pass production-like, multi-pass without NISS | 88,195 |
| **multi-pass + NISS + rescue top-200** | **78,029** (standalone PB) |
| 78,029 min-merged with 2 colleague CSVs | 76,304 (#1 leaderboard) |
