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

8. **Never use `sed -i` for in-place edits on this machine.** Windows MINGW's
   sed silently truncates files to 0 bytes (one occurrence cost ~5 min recovering
   `run_benchmark.py`). Use the `Edit` tool, or `sed 'pattern' file > file.new && mv file.new file`.

9. **Before adding any new megaminx experiment, check `megaminx/to_do_shortlist.md`.**
   That file is the active to-do list; an idea already there or already superseded
   by something on the list shouldn't be re-proposed. The 70K-score goal is the
   anchor — evaluate every new suggestion against "does this move us toward 70K
   or just save GPU hours?"

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

Do not commit the token to git (it's in `.Codex/settings.local.json` which is
project-local and appropriately gitignored, but the `AGENTS.md` rule above is fine
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
