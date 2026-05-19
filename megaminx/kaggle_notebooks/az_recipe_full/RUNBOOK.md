# RUNBOOK — `cayleypy-megaminx-az-recipe`

Operational notes for building, publishing, and updating the public AZ-recipe
notebook for the CayleyPy Megaminx competition.

## What this notebook is

A single Kaggle GPU kernel that demonstrates the full AlphaZero recipe used
to reach `m_az_v4` (51/51 strat-5, mean 87.5 — first sub-89 6M-param model on
this project), from a true zero start:

1. 1-epoch walk-depth cold-start (provides Bellman's required warm-start)
2. 10-epoch `m_dd_v0` Bellman + PDB λ=5
3. AZ (state, action) policy dataset built from `merge_v9_with_77152.csv` (76,304 moves)
4. 10-epoch AZ dual-head training (checkpoint every 2 epochs)
5. Per-pid selection matrix: 5 ckpts × 4 rotations × 51 strat-5 pids
6. Verify + write submission CSV

EPOCHS_* default to 10 for the recipe demo; user scales to 50 / 100 via the
commented production cell.

## Prerequisites

- Kaggle CLI installed: `pip install kaggle` (we already have it in `.venv/Scripts/kaggle.exe`).
- `KAGGLE_USERNAME` + `KAGGLE_KEY` env vars set, or `~/.kaggle/kaggle.json` populated.
  Per project memory, our project-scoped token lives in `.claude/settings.local.json`
  (gitignored). Export via `export KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN` before each push.
- Local Python venv with the cayley + megaminx packages importable
  (we use `.venv/Scripts/python.exe` on Windows).

## Files in this directory

| File | Purpose |
|---|---|
| `build_notebook.py` | source of truth — generates the `.ipynb` and `kernel-metadata.json` |
| `cayleypy-megaminx-az-recipe.ipynb` | generated artifact (committed so non-Python reviewers can read on GitHub) |
| `kernel-metadata.json` | Kaggle kernel config (generated) |
| `RUNBOOK.md` | this file |

## Kaggle dataset to assemble

The notebook reads from a Kaggle dataset named `artgor/megaminx-az-recipe-artifacts`.
Assemble it once locally, then `kaggle datasets create -p <dir>`.

### Required files (all at the top level of the dataset)

| File | Source path in repo | Size |
|---|---|---|
| `puzzle_info.json` | `megaminx/data/puzzle_info.json` | 12 KB |
| `test.csv` | `megaminx/data/test.csv` | 376 KB |
| `pp_bfs6_fallback.csv` | `megaminx/data/pp_bfs6_fallback.csv` | 1.2 MB |
| `rotations.npy` | `megaminx/data/rotations.npy` | 43 KB |
| `bfs_d6_train.pt` | `megaminx/data/bfs_d6_train.pt` | 2.2 GB |
| `frontier_states.pt` | `megaminx/data/frontier_states.pt` | 36 MB |
| `pdb_corner_K5.pkl` | `megaminx/data/pdb_corner_K5.pkl` | 432 MB |
| `pdb_corner_K5_p1.pkl` | `megaminx/data/pdb_corner_K5_p1.pkl` | 432 MB |
| `pdb_corner_K5_p2.pkl` | `megaminx/data/pdb_corner_K5_p2.pkl` | 432 MB |
| `pdb_corner_K5_p3.pkl` | `megaminx/data/pdb_corner_K5_p3.pkl` | 432 MB |
| `corner_tables.pkl` | `megaminx/data/corner_tables.pkl` | (small) |
| `merge_v9_with_77152.csv` | `megaminx/submissions/merge_v9_with_77152.csv` | 234 KB |
| `cayley_src/` directory | see below | ~1 MB |

Total: ~5 GB.

### Assembling `cayley_src/`

The notebook expects two source roots under `cayley_src/`:

```
cayley_src/
├── src/
│   └── cayley/           (whole package — bellman, training, model, gflow_model,
│                          search, khoruzhii_search, verify, data, post_process,
│                          puzzle, optimizers, bfs_table, ...)
└── megaminx/
    └── src/
        └── megaminx/     (whole package — puzzle, pdb_heuristic, pdb_corner,
                           post_process, bfs_bytes, mitm_solver, ...)
```

Bash snippet to assemble (run from repo root):

```bash
mkdir -p dataset_payload/cayley_src/src dataset_payload/cayley_src/megaminx/src
cp -r src/cayley dataset_payload/cayley_src/src/
cp -r megaminx/src/megaminx dataset_payload/cayley_src/megaminx/src/
# (Skip test files / build artifacts if you want a leaner dataset.)
```

### Dataset metadata

Create `dataset_payload/dataset-metadata.json`:

```json
{
  "title": "Megaminx AZ recipe artifacts",
  "id": "artgor/megaminx-az-recipe-artifacts",
  "licenses": [{"name": "CC0-1.0"}]
}
```

### Push the dataset

```powershell
$env:KAGGLE_API_TOKEN = $env:KAGGLE_API_TOKEN  # ensure set
# First time only:
.venv/Scripts/kaggle.exe datasets create -p .\dataset_payload --dir-mode zip
# Updating an existing dataset (bumps version):
.venv/Scripts/kaggle.exe datasets version -p .\dataset_payload -m "..." --dir-mode zip
```

## Build + push the notebook

```powershell
# 1. Regenerate the .ipynb + kernel-metadata.json from build_notebook.py:
.venv/Scripts/python.exe megaminx/kaggle_notebooks/az_recipe_full/build_notebook.py

# 2. Push to Kaggle (defaults to is_private: true):
cd megaminx/kaggle_notebooks/az_recipe_full
.venv/Scripts/kaggle.exe kernels push -p .

# 3. Monitor:
.venv/Scripts/kaggle.exe kernels status artgor/cayleypy-megaminx-az-recipe
# Expect KernelWorkerStatus.RUNNING -> COMPLETE in ~60-90 min at EPOCHS=10.
```

## Pre-publish verification (local smoke + Kaggle dry-run)

1. **Local smoke** — make sure all 20 cells parse and the non-training cells run.
   The script below executes setup + imports + Stage C + Stage E1 + Stage E2
   (~5 seconds total, no training):

   ```powershell
   .venv/Scripts/python.exe -c "
   import json
   with open('megaminx/kaggle_notebooks/az_recipe_full/cayleypy-megaminx-az-recipe.ipynb') as f:
       nb = json.load(f)
   ns = {'__name__': '__main__'}
   for i, cell in enumerate(nb['cells']):
       if cell['cell_type'] != 'code': continue
       src = ''.join(cell['source'])
       if any(tag in src for tag in ['Stage A -', 'Stage B -', 'Stage D -', 'Stage E3 -', 'Stage F -', 'Interpret', 'Notebook complete']): continue
       if '# # =====' in src or 'using shipped frontier_states.pt' in src: continue
       exec(src, ns)
   "
   ```

2. **(Optional) Full local run** — to actually train end-to-end, use `papermill`:

   ```powershell
   .venv/Scripts/pip.exe install papermill jupyter
   $env:KAGGLE_TEST_PIDS = "11"  # truncate Stage E to 11 pids for the smoke
   .venv/Scripts/python.exe -m papermill `
     megaminx/kaggle_notebooks/az_recipe_full/cayleypy-megaminx-az-recipe.ipynb `
     out_smoke.ipynb
   Remove-Item Env:\KAGGLE_TEST_PIDS
   ```

   Wall on a local 4090: ~10 min total at EPOCHS=10 + 11 pids in Stage E.
   Confirm the final cell prints `all paths valid` and the "best CHECKPOINT
   per pid" histogram has ≥ 2 distinct ckpt indices.

3. **Kaggle dry-run** — push as `is_private: true` (the default), Save & Run All,
   wait for completion. Inspect:
   - Wall < 90 min total.
   - Stage F's verify cell printed `n_valid == n_total`.
   - Stage interpret cell's "best CHECKPOINT per pid" histogram shows spread.

4. **Flip to public** — edit `kernel-metadata.json` and set `"is_private": false`,
   then re-push. Or change the default in `build_notebook.py`'s `meta` dict and
   rebuild.

## Updating the notebook

Always edit `build_notebook.py` (the source of truth) — never edit the .ipynb
directly. Rebuild and re-push:

```powershell
.venv/Scripts/python.exe megaminx/kaggle_notebooks/az_recipe_full/build_notebook.py
cd megaminx/kaggle_notebooks/az_recipe_full
.venv/Scripts/kaggle.exe kernels push -p .
```

## Scaling guidance (in-notebook)

The notebook's last code cell is a commented production-knob block. Defaults to
`EPOCHS_PRETRAIN=10, EPOCHS_AZ=10, BEAM=16384, K_SYM=4`. To reach the project's
production result (51/51 strat-5 / mean 87.5), bump to `EPOCHS_PRETRAIN=50,
EPOCHS_AZ=100, BEAM=65536, K_SYM=8`, full-1001 pids with multi-pass
`--beams 16384,65536` + NISS — does NOT fit in a single 9h kernel (~17h). Either:

- Run locally with `.venv/Scripts/python.exe megaminx/scripts/03_solve.py` after
  training the model in this notebook + downloading the checkpoint.
- Split into two chained kernels (this notebook ends with checkpoints; a second
  kernel does Stage E + F on the full-1001 set with the production beam config).

## Known gotchas

- **`/kaggle/working` on Windows** — Python's `Path("/kaggle/working").exists()`
  returns True on Windows because `/` is resolved against the current drive
  (`C:\kaggle\working`, which Kaggle's installer may have created). The setup
  cell now keys off whether DATA was found under `/kaggle/input/`; do not
  shortcut this check.

- **`compile_model=False`** on Kaggle — Kaggle's torch image may lag the
  current release. `torch.compile` sometimes fails on the Megaminx ResMLP
  trunk in those images. We disable it; accept the ~25% wall-time hit. To
  enable in production, run a `compile=True` test on a fresh image first.

- **bf16 on T4** — T4 (sm_75) has no native bf16 hardware. The notebook
  detects this via `torch.cuda.get_device_capability()` and falls back to
  fp16 autocast automatically. On Ampere+ GPUs (P100 is sm_60, also no bf16;
  L4 / A10 / A100 are sm_80+ and use bf16).

- **Frontier states from a true cold start** — `frontier_states.pt` is
  produced by running a beam search with a previous V model on real pids.
  We ship it pre-built (only 36 MB) because regenerating from a 1-epoch
  cold-start V is not meaningful. The notebook's Stage B0 markdown cell
  explains, with a commented regeneration snippet.

- **`merge_v9_with_77152.csv` credits** — the AZ policy dataset is built
  from a min-merge of three submissions including two from collaborators.
  The notebook's credits cell has `TODO_FILL_IN` placeholders; replace with
  real names before flipping the kernel to public.

- **`bfs_d6_train.pt` (2.2 GB)** — load time ~10 s. We load once at the top
  of Stage D (the AZ training loop) and once internally inside `train_bellman`
  for Stage B; total ~20 s. Don't loop-load.

## Quotas

- **GPU quota**: Kaggle allows ~30 hours/week of GPU; the demo run takes
  ~1 h, so 20-30 demo runs per week. Production scaling would consume most
  of a week's quota in one go.

- **Dataset size**: 5 GB is well under the 50 GB Kaggle dataset cap.
