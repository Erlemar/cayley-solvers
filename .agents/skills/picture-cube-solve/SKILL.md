---
name: picture-cube-solve
description: Use this skill for any task in the cayley IHES Picture Cube project — training, solving, evaluation, ensembling, submission. Covers the full ML+search pipeline.
---

# picture-cube-solve

Use when working in `C:\Users\and-l\cayley\` on the Kaggle IHES SuperCube competition.
This skill captures the proven pipeline, known failure modes, and ordered entry points
so a new session can be productive within minutes.

## Current state (read first)

- Best submitted: **24,618** (`submissions/ens_e5_all_pp.csv`).
- Rokicki (leader): 21,840. Gap: 2,778.
- Best single checkpoint: `models/small_e5/epoch_7999.pt` — 1.6M params, MSE 8.41,
  full-set solve 24,974.

Detailed logs in the project root: `README.md` (resume guide), `EXPERIMENTS.md`,
`IDEAS.md`, `DATA_AND_FEATURES.md`. Read the latter three for context before making
decisions.

## The proven pipeline

### Training (fast recipe)

```bash
.venv/Scripts/python.exe -u scripts/01_train.py \
    --config configs/small_e5_long.yaml \
    --output models/<new_name> \
    > models/<new_name>_training.log 2>&1 &
```

Fast recipe = bf16 + batch 10000-16384 + `torch.compile` + fused AdamW. Expect
~0.4s/epoch for the small `[1024,256]×1` architecture, ~1.4s/epoch for `[700,643]×4`.

### Solving (khoruzhii searcher)

```bash
.venv/Scripts/python.exe -u scripts/02_solve.py \
    --checkpoint models/<model>/epoch_<N>.pt \
    --out submissions/<tag>.csv \
    --beam 65536 --max-steps 50 --bf16 \
    --searcher khoruzhii \
    --fallback data/kociemba_fallback.csv \
    > submissions/<tag>.log 2>&1 &
```

`--searcher khoruzhii --bf16 --beam 65536` is the default for max quality. Beam 65k
uses ~1.6 GB VRAM for small models. Beam 131k works too (~3 GB) but is 7× slower for
marginal gains.

### Post-processing + submit via `/submit`

Bundled into the `/submit <description>` slash command. See `.Codex/commands/submit.md`.

## Gotchas (do not rediscover)

1. **Use `.venv/Scripts/python.exe`** — Windows venv convention.
2. **`torch.compile` is harmful for beam search inference** (5.8× slowdown from
   recompile loops). Only enable for training.
3. **CayleyPy's `advanced` mode returns `path=None`.** Use `simple` or `KhoruzhiiSolver`.
4. **`graph.bfs()` requires `return_all_hashes=True`** for MITM to work.
5. **Reuse one `CayleyGraph`** per session — fresh instances have different hash
   vectors.
6. **Never use `sample_fallback.csv`** (500K moves); use `data/kociemba_fallback.csv`.
7. **`Monitor` max timeout = 3,600,000 ms (1h).** Use `persistent: true` for
   longer watches.
8. **Compiled checkpoints have `_orig_mod.` prefix** on state dict keys. The
   `load_model_checkpoint` helper strips this; direct `load_state_dict` calls fail.

## Anti-patterns (confirmed regressions — do not retry)

- `n_back=40` alone in random walks (MSE 14.40 → 15.84).
- Big [2048,1024]×8 model + curriculum (worse than small fast model).
- L1 + big arch + n_back=40 bundled.
- k_max>30 in walks.
- >4000 training epochs on the E3 architecture (diminishing returns).

## Highest-EV untried ideas (from `IDEAS.md` items 0a-0c and 1)

1. **NISS (invert scramble, solve, reverse path)** — expected -0.5 to -1.5 moves per
   scramble, ~50 lines of code. Start here for a quick win.
2. **Commutator-insertion post-processing** — 1-3 moves/scramble, ~1 day effort. Build
   a ~500-entry library of 3-cycle commutators (edge / corner / center) and try
   inserting each at every position of a solution.
3. **Bellman refinement** — code ready in `src/cayley/bellman.py`; launch with
   `.venv/Scripts/python.exe -u scripts/05_bellman_refine.py --config configs/e6_bellman.yaml --output models/e6`.
   500 ep ~ 17 min on 4090. Expected -500 to -2000 moves on single model.

## Kaggle credentials

Token is in project memory (`ref_kaggle_credentials.md`). The `/submit` slash command
exports it automatically. For ad-hoc submits:

```bash
export KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN
```
