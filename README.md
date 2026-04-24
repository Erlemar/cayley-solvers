# cayley — IHES Picture Cube solver

Neural distance heuristic + GPU beam search for the Kaggle
[CayleyPy SuperCube: Solve Optimally IHES puzzle](https://www.kaggle.com/competitions/cayleypy-ihes-cube).

## Status (as of 2026-04-18)

- **Current best Kaggle score: 24,618** (from `submissions/ens_e5_all_pp.csv`).
- Progression: `42,718 (v13 Kociemba) → 30,770 (first ML) → 28,224 → 27,106 → 24,998 → 24,618`.
- Gap to leader (Rokicki, 21,840): **2,778 moves (12.7%)**.

## Start here in a new session

1. Read `EXPERIMENTS.md` for the current state, training runs, and decision log.
2. Read `IDEAS.md` for prioritized untried ideas. Items `0a`-`0e` at the top are NEW
   (forum-mined, highest-EV).
3. `DATA_AND_FEATURES.md` — focused data/feature engineering ideas (orthogonal to search).
4. If picking up Bellman: `src/cayley/bellman.py` + `configs/e6_bellman.yaml` is ready
   to launch, warm-starts from `models/small_e5/epoch_7999.pt`.

## Quick reproduction commands

```bash
# Environment: Python 3.14 + torch 2.11.0+cu128 + cayleypy 0.1.0 + triton-windows
# On Windows, always use .venv/Scripts/python.exe (not plain python)

# Re-verify the current best submission (should print 1003/1003 valid):
.venv/Scripts/python.exe -c "
import sys; sys.path.insert(0, 'src')
from cayley.puzzle import PictureCube
from cayley.verify import verify_submission
p = PictureCube.load('data/puzzle_info.json')
r = verify_submission(p, 'data/test.csv', 'submissions/ens_e5_all_pp.csv')
print(f'valid: {r.n_valid}/{r.n_total}, total: {r.total_moves}')
"

# Launch Bellman refinement (code ready, not yet run):
.venv/Scripts/python.exe -u scripts/05_bellman_refine.py --config configs/e6_bellman.yaml --output models/e6

# Typical solve command for the current best recipe:
.venv/Scripts/python.exe -u scripts/02_solve.py \
    --checkpoint models/small_e5/epoch_7999.pt \
    --out submissions/my_run.csv \
    --beam 65536 --max-steps 50 --bf16 \
    --searcher khoruzhii \
    --fallback data/kociemba_fallback.csv

# Post-process any submission:
.venv/Scripts/python.exe scripts/post_process_submission.py \
    --in submissions/my_run.csv --out submissions/my_run_pp.csv \
    --bfs-table data/bfs_table_d5.pkl

# Combine multiple runs (ensemble = min per puzzle):
.venv/Scripts/python.exe scripts/combine_submissions.py \
    --candidates submissions/a.csv submissions/b.csv \
    --fallback data/kociemba_fallback.csv \
    --out submissions/combined.csv

# Submit to Kaggle (token expires — re-ask user if this fails):
export KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN
.venv/Scripts/kaggle.exe competitions submit \
    -c cayleypy-ihes-cube -f submissions/combined_pp.csv -m "description"
```

## Required data files

All in `data/` — do not delete:

| File | Size | Purpose |
|---|---|---|
| `puzzle_info.json` | 5 KB | 18 generators + solved state |
| `test.csv` | 214 KB | 1003 scrambled puzzles |
| `sample_submission.csv` | 1.8 MB | original baseline (used as worst-case fallback) |
| `kociemba_fallback.csv` | 1.7 MB | 38,440-move Kociemba output; our actual fallback floor |
| `bfs_table_d5.pkl` | 126 MB | 790K-state BFS table for post-processing |

## Key checkpoints

| Path | Arch | Epochs | Final loss | Best solve |
|---|---|---|---|---|
| `models/fast/epoch_0499.pt` | embed [700,643]×4 | 500 | 14.14 MSE | 29,710 single / 28,224 +pp |
| `models/embed/epoch_0199.pt` | embed [700,643]×4 | 200 | 14.40 MSE | 30,120 |
| `models/ensemble_s{1,2,3}/epoch_0499.pt` | fast arch | 500 | ~14.2 | - |
| `models/big_v1/epoch_0999.pt` | big [2048,1024]×8 | 999 | - | regressed |
| `models/small_e1/epoch_0499.pt` | khoruzhii [1024,256]×1 | 500 | 14.33 MSE | - |
| `models/small_e2/epoch_1999.pt` | small | 2000 | 13.51 MSE | - |
| **`models/small_e3/epoch_3999.pt`** | small K_max=26 | 4000 | **8.50 MSE** | 25,070 single |
| `models/small_e4/epoch_1999.pt` | small K_max=45 | 2000 | 46.1 (k=45 scale) | - |
| **`models/small_e5/epoch_7999.pt`** | small K_max=26 | 8000 | **8.41 MSE** | **24,974 single** |

## Gotchas (do not forget)

1. **Always pass `return_all_hashes=True`** when building a BFS for MITM, or
   `BfsResult.layers_hashes` is empty and the search silently degrades.
2. **Don't use CayleyPy's `advanced` beam mode** — it doesn't return paths. Use `simple`
   or the khoruzhii searcher.
3. **`torch.compile` is HARMFUL for inference** — variable beam batch sizes trigger
   8+ recompiles, 5.8× slowdown. Use for training only.
4. **Checkpoints trained with `compile_model: true`** have `_orig_mod.` prefix on state
   dict keys. `load_model_checkpoint` strips this automatically but direct
   `load_state_dict` calls will fail.
5. **Build `CayleyGraph` once per session** — fresh instances have different random hash
   vectors. The `Solver` class already does this; use it.
6. **BFS-d5 post-processing is bounded at max_window=12** by default (see
   `post_process.py`) — saves 30-80 moves per submission, not more. Real gains would
   require depth-6 BFS (~6 GB memory).
7. **Windows + Python 3.14 requires** `triton-windows` package (not `triton`) for
   `torch.compile`. Already installed in `.venv`.

## Layout

- `data/` — competition data + precomputed tables
- `src/cayley/` — library code
  - `puzzle.py` — `PictureCube` loader + state ops
  - `data.py` — non-backtracking random walks (numpy + torch)
  - `model.py` — `ResMLPDistance` (one-hot + embedding, chunked inference)
  - `training.py` — fast-recipe training loop (bf16 + compile + fused AdamW)
  - `bellman.py` — **Bellman refinement training (ready, not yet run)**
  - `search.py` — CayleyPy wrapper (`Solver` class) + `load_model_checkpoint`
  - `khoruzhii_search.py` — **self-contained 150-line beam with fp16 values; use `KhoruzhiiSolver`**
  - `post_process.py` — pair-cancel + state-hash shortcut + BFS-d5 window replacement
  - `bfs_table.py` — build/load BFS lookup tables
  - `submit.py` — min-across-candidates submission builder with fallback
  - `verify.py` — path verification
- `scripts/` — CLI entrypoints
- `configs/` — YAML hyperparameter configs
- `models/` — checkpoints (gitignored)
- `submissions/` — CSVs (gitignored, some kept for ensemble)
- `tests/` — pytest unit tests

## Plan & other references

- Project plan: `C:\Users\and-l\.claude\plans\this-will-be-a-purrfect-shore.md`
- Cross-project research notes: `C:\Users\and-l\kaggle_research\cayleypy-ihes-cube\experiment_log.md`
- Khoruzhii's reference repo (cloned): `C:\Users\and-l\AppData\Local\Temp\cayleypy_cube\cayleypy-cube\`
