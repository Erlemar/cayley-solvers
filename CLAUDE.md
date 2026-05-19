# Project: IHES Picture Cube solver

Kaggle competition: [CayleyPy SuperCube](https://www.kaggle.com/competitions/cayleypy-ihes-cube).
Current best score: **24,618**. Leader: 21,840 (Rokicki).

**Read `README.md` first** for full state, commands, and gotchas. **Read `EXPERIMENTS.md`**
for what was tried and what worked. **Read `IDEAS.md`** for the prioritized untried ideas.

## Non-negotiable rules for this project

1. **Always use `.venv/Scripts/python.exe`** (not `python`, `python3`, or bare `.venv/python`) —
   Windows venv + Python 3.14 convention. Same for `.venv/Scripts/kaggle.exe` and
   `.venv/Scripts/pip.exe`.

2. **Don't enable `torch.compile` for beam-search inference without padding to a
   fixed batch size.** Naive `model(candidates)` recompiles on every shape change
   (5.8× slowdown — confirmed). The fix, used in `megaminx/beam_lab/beam_search.py`:
   `_model_predict(..., pad_to_batch_size=True)` pads every forward to
   `internal_batch_size`, plus `setup_model_for_compile(model, batch_size)` pre-warms
   the compiled graph at that shape. With both pieces in place, `--compile`
   (mode=reduce-overhead) gives ~−27% wall on the 12-puzzle benchmark with
   identical paths. The IHES `src/cayley/khoruzhii_search.py` does NOT yet have
   the padding hook — the original ban still applies there until ported. Training
   already has fixed shapes, so compile-for-training is always fine.

3. **Don't use CayleyPy's `advanced` beam mode** — it silently returns `path=None`.
   Use `simple` or our `KhoruzhiiSolver` (in `src/cayley/khoruzhii_search.py`).

4. **Always pass `return_all_hashes=True` to `graph.bfs(...)`** when you need the result
   for MITM search. Default is False; without it `layers_hashes` is empty and MITM
   degrades to plain beam without warning.

5. **Reuse one `CayleyGraph` per session, not per puzzle.** Each fresh graph has
   different random hash vectors, which corrupts state tracking across calls. Use the
   `Solver` class in `src/cayley/search.py`.

6. **Never use `sample_fallback.csv` as fallback.** It's sample-quality (500K+ moves).
   Always use `data/kociemba_fallback.csv` (38,440 moves).

7. **`Monitor` tool max timeout is 3,600,000 ms (1h).** For multi-hour solves, use
   `persistent: true` (runs until session ends) or accept the timeout and re-arm.

7b. **Use PowerShell `Get-Process` instead of `tasklist /FI` in Bash.** Bash MINGW
   translates `/FI "filter"` to `C:/Program Files/Git/FI` (silent path mangling
   — command runs but with wrong args). For Windows process queries, prefer:
   `Get-Process python -ErrorAction SilentlyContinue | Format-Table Id,WS,StartTime`
   via the `PowerShell` tool. `tasklist` without `/FI` (e.g. via `tasklist | grep`)
   works fine in Bash.

7c. **Bash sessions do not persist cwd across calls.** Each `Bash` tool call
   starts fresh from the project root, so `cd subdir && cmd` in one call does
   NOT carry `subdir` over to the next call. For tools that require being inside
   a specific directory (e.g. `kaggle models instances create -p .`), either
   chain everything in a single `Bash` call with `&&`, or use absolute paths in
   every invocation. Confirmed 2026-05-17: cost ~3 wasted calls during the
   Kaggle Models push session.

7d. **Kaggle CLI on Windows always needs `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`.**
   Without these, the CLI hits `cp932 codec can't decode byte 0x94` errors when
   it reads metadata JSON files containing non-ASCII bytes (em-dashes, smart
   quotes, anything non-ASCII). Export both before any `kaggle models` /
   `kaggle kernels` / `kaggle datasets` call:
   `export KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN PYTHONUTF8=1 PYTHONIOENCODING=utf-8`.
   Also note: `kaggle kernels push` requires the kernel title to slugify to the
   `id` field; long titles with em-dashes silently 400 with "title does not
   resolve to specified id".
   **PowerShell sessions need the token re-exported per invocation** —
   Bash inherits the allowlisted `export` line across calls, but each
   PowerShell tool call starts a fresh session. A 401 on
   `kaggle datasets create|version|push` or `kaggle kernels push` from
   PowerShell is almost always `$env:KAGGLE_API_TOKEN` unset, NOT a
   write-scope issue. Prefix every call:
   `$env:KAGGLE_API_TOKEN="<token>"; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"; <kaggle.exe ...>`.

8. **Never use `sed -i` for in-place edits on this machine.** Windows MINGW's
   sed silently truncates files to 0 bytes (one occurrence cost ~5 min recovering
   `run_benchmark.py`). Use the `Edit` tool, or `sed 'pattern' file > file.new && mv file.new file`.

9. **Before adding any new megaminx experiment, check `megaminx/to_do_shortlist.md`.**
   That file is the active to-do list; an idea already there or already superseded
   by something on the list shouldn't be re-proposed. The 70K-score goal is the
   anchor — evaluate every new suggestion against "does this move us toward 70K
   or just save GPU hours?"

10. **Never use literal `%` in argparse `help=` strings** — Python 3.14 became
    strict about `%` in help text (it's reserved for C-style format specs).
    A help string like `"~30% by difficulty"` raises
    `ValueError: badly formed help string` at argparse construction time.
    Cost in this project: 4-hour m39a queue had to restart because the strat-5
    step crashed at parse_args(). **Fix**: escape as `%%` or rephrase
    ("30 percent" not "30%"). Audit new `help=` strings before commit.

11. **(Megaminx)** **Production full-1001 solve recipe is multi-pass + NISS**:
    `--beams 16384,65536 --max-steps 60,150 --niss --bf16 --resume`. Single-pass
    `--beams 65536 --max-steps 120` is for SMOKE TESTS ONLY — confirmed
    2026-05-11 that single-pass fails for ~50% of pids past depth 60 (m_dd_v0
    50ep on GCP: 41/79 model solves, jumping to 0/20 for pids 60-79). Cost of
    skipping: ~9h of GCP time wasted on an unsubmittable CSV. The 88,195 and
    78,029 submissions both used the multi-pass + NISS recipe.

12. **(Megaminx)** **Long V-model training (>50 epochs) requires per-N-epoch
    beam bench validation, not just smoothed-loss early stopping.** Training
    loss is NOT a reliable proxy for beam quality. Confirmed 2026-05-11:
    m_dd_v0_full trained 184 epochs with smoothed-loss early stop
    (min_delta=1e-4, patience=50). Final loss was lower (0.0688 vs 50ep's
    0.0724) but beam quality regressed severely — solve rate dropped to 0%
    for pids 60-79 that m_dd_v0 50ep handled fine. Hypothesis: long training
    over-saturated on BFS-d6 anchors (sampled every batch) while losing
    fidelity on harder intermediate-depth states (sampled via random walks).
    **Fix**: for V models, validate against a 5-10 pid beam bench every ~50
    epochs during long runs. Use `megaminx/scripts/61_eval_v_at_solved.py`
    for quick V(V0)/calibration check, plus `58_corner_pdb_beam.py --no-pdb`
    for short bench. Stop manually if beam regresses, even if loss is still
    falling. Canonical baseline going forward: `models/m_dd_v0/epoch_0049.pt`.

13. **(Megaminx)** **Before relaunching a model variant, check the prior run's
    actual hyperparameters — don't trust script defaults.** Confirmed
    2026-05-11: AZ v3 was launched with `--rw-batch-size 8192 --policy-batch-size 1024`
    but `71_train_az_v3.py` defaults to `4096 / 512`. Re-launching AZ v4 with
    defaults gave 4× the gradient steps per epoch → policy memorized (top-1
    99% by ep 99, v_loss 5× start), V calibration destroyed. **Fix**: every
    training script echoes the resolved config dict at startup; read the
    prior run's `*_training.log` header before launching a "matched"
    variant. The line `training: {...}` or `epochs: N rw_batch: M ...`
    has everything you need.

14. **(Megaminx)** **Bigger V trunks under the m_dd_v0 recipe regress.**
    Confirmed 2026-05-11: tried `hidden_dims=(4096,1024)+4rb=20.5M`
    params (3.4× the 6M baseline) with the exact m_dd_v0 recipe (PDB λ=5,
    V0/d=1 anchors, frontier 25%, BFS-d6 10%, target update every 10) for
    200 ep. Final training loss matched m_dd_v0 (0.0748 vs 0.0724) but
    bench regressed at every checkpoint: best (ep49) 9/10 / 930 vs 6M
    baseline 10/10 / 871; worst (ep99) 7/10 / 733. Avg path is consistently
    15-17 moves longer per solved pid. The 6M cluster ceiling extends to
    V-only at bigger trunks under this recipe — don't retry without a
    substantively different recipe (lower λ_pdb, lower lr, rotation
    augmentation, or no PDB penalty). See `bigger_v_trunk_regression.md`.

15. **(Megaminx)** **m23_v2 qshort doesn't compose with non-m05 V models.**
    Confirmed 2026-05-11: AZ v4 V + m23_v2 qshort on strat-5 → 48/51 by
    model, 4,705 total (vs AZ v4 V-only's 51/51, 4,465 — **+240 moves
    regression**). m23_v2 was distilled from m05's forward-state V landscape;
    its top-k action ordering reflects m05, not AZ v4. **Fix**: when swapping
    in a new V model for the production stack, either (a) drop qshort and
    rely on sym-ensemble + NISS for inference gains, or (b) re-distill a
    qshort from the new V's forward states first (m23-v3 style). Don't
    naively reuse m23_v2 with a different V.

16. **(Megaminx)** **K_SYM + SYM_POSITIONS is the canonical sym-ensemble
    shard pattern in the shareable beam notebooks.** Two knobs, not one:
    - `K_SYM` = full deterministic ensemble size. Defines the canonical
      rotation list via `np.random.default_rng(SYM_SEED=0).choice(
      non_identity, size=K_SYM-1, replace=False)` with identity prepended.
    - `SYM_POSITIONS` = `range(0, 4)` (or any subset of `[0, K_SYM)`).
      This kernel runs exactly those positions of the canonical list.

    Both shards of a split run MUST set the same K_SYM. Different K_SYM
    values produce different rotation lists at the same positions, making
    cross-shard merge meaningless. Confirmed 2026-05-18: K_SYM=8 split
    `[range(0,4), range(4,8)]` ⇒ disjoint shards, union = full K=8 set.
    K_SYM=4 single-kernel reproduces the OLD K_SYM=4 rotation set exactly.

    Notebooks using this pattern: `tpu_beam_az_v4_32m_shareable/`,
    `tpu_beam_az_v4_32m_720_shareable/`, `tpu_beam_az_v4_48m_720_shareable/`.
    Output files are tagged `_k{K_SYM}_sym{min}_{max}[_niss]` so two
    collaborators' downloads don't collide. Result records store both
    `sym_pos` (logical) and `rot_idx` (absolute idx in rotations.npy /
    rotations_720.npy) for unambiguous cross-shard analysis. See
    [[megaminx_rotations_720]] for the 720-set details.

17. **(Megaminx)** **GCP cayley-gpu is provisioned for solving, not training.**
    The VM has `03_solve.py` working end-to-end (inference), but the training
    stack drifted local-only over time. Before launching ANY training on GCP,
    sync these from local: (a) trainer scripts not yet shipped (commonly
    `60_train_admissible.py`, `67_build_az_dataset.py`, `71_train_az_v3.py`),
    (b) modern `src/cayley/bellman.py` (post-2026-05-05; ~600 added lines for
    BFS-d6 mixin / Double Bellman / rotation aug / V0+d1 anchors), (c) the full
    `megaminx/src/megaminx/` package — local has 12 files (incl. `pdb_heuristic.py`,
    `pdb_corner.py`, `pdb_edge.py`, `corner_coord.py`, `edge_coord.py`,
    `decomposition.py`); GCP often has 6. Confirmed 2026-05-18 during m_v11
    launch: 4 separate transfer rounds, each triggered by a fresh
    `ModuleNotFoundError`. **Module imports happen at load time**, so even
    `lambda_pdb=0` doesn't save you — the `from megaminx.pdb_heuristic import
    CornerPDBHeuristic` at the top of `60_train_admissible.py` fires
    unconditionally. Use `/megaminx-gcp-sync` to do the audit + transfer in one
    pass, BEFORE the tmux launch.

18. **(Megaminx)** **Bigger-trunk scale-ups can't warmstart from a different
    shape.** `warmstart_from_v_model` in `71_train_az_v3.py` uses
    `load_state_dict(..., strict=False)`, which silently skips mismatched-shape
    `input_stack` / `res_blocks` tensors. A "25-ep fine-tune at (3072,768) warm
    from AZ v4 (2048,512)" is really "25-ep from random init except embedding
    layer." 25 epochs from scratch is nowhere near convergence — you'll get
    noise or a partially-trained model. **Fix**: for any trunk-shape change, use
    a two-stage pipeline: (a) from-scratch random-walk MSE pretrain at the new
    shape (`02_train.py`, ~50 ep at ~1.2 s/ep for 11M params), then (b) Bellman
    refine warm from (a) (`60_train_admissible.py`). Confirmed 2026-05-18
    during m_v11 launch planning — the AZ-style fine-tune is then Stage 3
    warm from (b)'s best beam-bench checkpoint, with shape-matched warmstart.

## Conventions

- All submissions go in `submissions/`. Always verify with `verify_submission` before
  sending to Kaggle — a typo in a generator name produces a silent-fail score.
- Training configs in `configs/<name>.yaml`, checkpoints in `models/<name>/`, logs in
  `<models_or_submissions>/<name>_training.log` or `<name>_solve.log`.
- Submission workflow: solve → `post_process_submission.py` → `combine_submissions.py`
  (if ensembling) → `kaggle submit`. See `/submit` slash command for the bundled version.
- Training uses the "fast recipe": bf16 + batch 16384 + torch.compile + fused AdamW.
  Overrides in `configs/fast.yaml`, `configs/small_e*.yaml`.
- Inference uses: `--searcher khoruzhii --bf16 --beam 65536` for the current best pipeline.

## Kaggle submission

Token is in project memory. Export then submit:

```bash
export KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN
.venv/Scripts/kaggle.exe competitions submit -c cayleypy-ihes-cube -f <file> -m "<desc>"
```

Do not commit the token to git (it's in `.claude/settings.local.json` which is
project-local and appropriately gitignored, but the `CLAUDE.md` rule above is fine
since the repo isn't published).

## Anti-patterns (confirmed to regress — do not retry without new justification)

- **`n_back=40` alone** (random-walk non-backtracking depth): confirmed regression
  (MSE 14.40 → 15.84). The Jan-2025 chat's claim of Kaggle 10,281 with this setting must
  be bundled with other changes we never identified.
- **Big model + curriculum** (23.7M params + 1/k weighted loss): worse than small model.
- **L1 + `n_back=40` + bigger arch bundled**: full regression (10/30 at beam 2048).
- **>4000 training epochs on the E3 architecture**: diminishing returns past 3000 ep.
- **k_max > 30** in random walks: no improvement — picture cube's effective diameter.
- **Single/2-step state-hash post-processing alone**: <100 moves savings on 25K submissions.
