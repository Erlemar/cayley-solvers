# Project: CayleyPy competition solvers

FOUR solvers share this repo and the `src/cayley/` library. **Read `README.md` first**
for the router, then the HANDOFF of the puzzle you are working on.

| Puzzle | Best | Standing | Docs |
|---|---|---|---|
| **Professor Tetraminx** | **28,094** | **#1 — next is CayleyPy 28,398, +304 (2026-08-09)** | `tetraminx/HANDOFF.md` |
| cube444 | 48,738 | #2 public LB when submitted | `cube444/HANDOFF.md` |
| Megaminx | 73,441 floor / 75,200 submitted | — | `megaminx/HANDOFF.md` |
| IHES Picture Cube | 21,870 on disk / 21,972 submitted | leader Rokicki 21,840; **21,870 is a 3-way community plateau** | `EXPERIMENTS.md`, `IDEAS.md` |

Rules below are numbered globally; those tagged **(Megaminx)** are puzzle-specific but
the mechanism usually generalises. Rules 1-10, 19-20, 24, 26, 26b and 28 apply to
everything. **Read `EXPERIMENTS.md`** (IHES) / the relevant `HANDOFF.md` for what was
tried and what worked, and **`IDEAS.md`** for prioritized untried ideas.

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
   **Background Bash poll loops are NOT a substitute** — the harness kills them
   after ~20-60 min (confirmed 3x on 2026-07-12 watching Kaggle TPU kernels).
   For multi-hour watches use `Monitor` with `persistent: true` and a script
   that emits only on state CHANGE and exits on terminal states.
   **The default is NOT persistent and the failure is silent** — a watch armed
   with the default `timeout_ms` on a Kaggle kernel simply stops after an hour
   and the terminal-state notification never arrives, which is indistinguishable
   from "still running". Confirmed 2026-08-23: five watches this session used the
   default; the only reason a 2-4 h beam run was not lost is that it was re-armed
   with `persistent: true` minutes before the cap. Any Kaggle GPU/TPU kernel watch
   is a multi-hour watch — set `persistent: true` at arm time, not after.

7b. **Use PowerShell `Get-Process` instead of `tasklist /FI` in Bash.** Bash MINGW
   translates `/FI "filter"` to `C:/Program Files/Git/FI` (silent path mangling
   — command runs but with wrong args). For Windows process queries, prefer:
   `Get-Process python -ErrorAction SilentlyContinue | Format-Table Id,WS,StartTime`
   via the `PowerShell` tool. `tasklist` without `/FI` (e.g. via `tasklist | grep`)
   works fine in Bash.

7b-ii. **PowerShell process queries SELF-MATCH, and torch workers inherit the
   parent's command line.** `Get-CimInstance Win32_Process | Where-Object
   { $_.CommandLine -like "*pat*" }` matches the **pwsh.exe running that query**
   (the query text contains the pattern), so counts read one too high and a
   "kill until clean" loop never converges. Separately, a torch child worker
   inherits the parent's full command line, so one `30_solve.py` run shows as
   2+ processes; killing the "duplicate" kills a worker, the parent `.bat`
   advances to its next command, and the job looks like it is respawning.
   Cost 2026-08-02: ~5 wasted calls plus one healthy run killed by mistake.
   **Fix**: filter on `$_.Name -eq 'python.exe'` AND check `ParentProcessId`
   (`parent=cmd.exe` is a real run; `parent=python.exe` is a worker). This is
   the PowerShell twin of 7g.

7c. **The Bash tool PERSISTS cwd across calls — a stray `cd` poisons every later
   call.** The tool description states "working directory persists between calls,"
   and this is the current behavior: a `cd subdir` (even `cd subdir 2>/dev/null`)
   leaves the NEXT Bash call running from `subdir`, not the project root. Symptom:
   `.venv/Scripts/python.exe: No such file or directory` (exit 127) or a
   `ModuleNotFoundError` from a later call. **Always use ABSOLUTE paths in Bash and
   avoid bare `cd`**; if a tool must run inside a dir (e.g.
   `kaggle models instances create -p .`), chain everything in ONE `Bash` call with
   `&&`. (A 2026-05-17 note claimed the opposite — that cwd does NOT persist; the
   current harness DOES persist it — confirmed 2026-06-17, cost ~3 calls when a
   `cd megaminx` leaked into later `python -c` invocations. Either way, absolute
   paths are the safe habit.)
   **The habit that actually works**: never write `.venv/Scripts/python.exe`
   relatively — always `/c/Users/and-l/cayley/.venv/Scripts/python.exe` — and when a
   command genuinely must run inside a directory, wrap it in a SUBSHELL:
   `( cd "C:/path" && cmd )`. The subshell cannot leak cwd to the next call. Hit 3x
   again on 2026-08-23 (exit 127 each time) with this rule already written; the
   `&&`-chain advice above is easy to forget mid-edit, the subshell is not.

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

7e. **Windows-native tools need Windows-style paths `C:/Users/...`, NOT MINGW
   `/c/Users/...`.** This covers `python -c` path STRINGS *and `gcloud`*. Python's
   `sys.path.insert` / `open()` don't resolve MINGW paths → `ModuleNotFoundError` /
   `FileNotFoundError`. MINGW `/c/...` is only for bash-native tools (ssh, scp, cp,
   ls). The venv python as the *executable* works either way
   (`.venv/Scripts/python.exe` or `/c/Users/and-l/cayley/.venv/Scripts/python.exe`);
   it's the path arguments passed into Python that must use `C:/...`. Confirmed
   2026-06-17 (1 wasted call).
   **`gcloud` behaves the same and fails QUIETLY** — `gcloud storage cp
   "/c/Users/.../tpubundle/*" gs://...` returns `ERROR: The following URLs matched
   no objects or files` rather than a path error, so it reads like an empty source
   dir. Use `"C:/Users/.../tpubundle/*"`. Confirmed 2026-08-01.

7e-ii. **`~` in a shell variable expands on the LOCAL host, even when the string
   is destined for a remote command.** `CKDIR=~/cayley/models` becomes
   `/c/Users/and-l/cayley/models` locally, and sending that to a remote `stat`
   via ssh silently never matches — a watcher waited forever on a checkpoint
   that already existed (2026-08-02). Use LITERAL remote paths
   (`/home/and-l/...`) in any string that will be evaluated on another machine.
   Sibling of 7e (Windows-vs-MINGW path strings in `python -c`).

7f. **Kaggle kernel versions: pull outputs BEFORE the next push, and don't trust
   the status endpoint alone.** `kernels output`/`status`/`logs` serve ONLY the
   LATEST version, so pushing v(N+1) before downloading vN's output hides vN
   behind the UI (cost: a full seq-baseline run's results on 2026-07-12). Pull
   first — it stays much cheaper than recovery — but the loss is NOT permanent;
   see (d). Related session mechanics (all confirmed 2026-07-13/14): (a) ONE
   batch TPU session per account — `kernels push` errors with "Maximum batch
   TPU session count of 1 reached" while any version is queued/running, and CLI
   2.2.0 exposes NO cancel (the SDK does have
   `ApiCancelKernelSessionRequest` — a CLI-surface gap, untested);
   (b) a push can produce a **phantom COMPLETE version** whose log is a
   6-second nbconvert with ZERO cells executed while the real batch session
   survives underneath and still blocks pushes — trust push session errors +
   the UI Session history over `kernels status`; (c) `kaggle kernels files
   <ref>` lists output filenames without downloading — cheap completion probe.

   (d) **Old versions' outputs ARE recoverable — via the SDK, not the CLI**
   (confirmed 2026-08-05; this REVERSES the earlier "REST rejects
   `versionNumber`" claim). `ApiDownloadKernelOutputRequest` has a working
   `version_number` field, even under CLI 2.2.0:

   ```python
   from kagglesdk import KaggleClient
   from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
   with KaggleClient(api_token=tok) as c:
       req = ApiDownloadKernelOutputRequest()
       req.owner_slug, req.kernel_slug = "artgor", "<kernel-slug>"
       req.file_path, req.version_number = "submission.csv", 8
       redirect = c.kernels.kernels_api_client.download_kernel_output(req)
   # then urlopen(redirect.url)
   ```

   Do NOT route this through the CLI: 2.2.0's `kernels pull` accepts
   `owner/slug/N` and **silently ignores it** (`/1`, `/3`, `/999` all returned
   byte-identical bytes to latest — the rule-28 unwired-flag signature), and
   2.2.2's `kernels output` documents the suffix but downloaded nothing at all.
   Read the two 404s to find the version ceiling: from
   `www.kaggleusercontent.com` = version EXISTS but has no such file
   (failed/running/different filename); from `api.kaggle.com/.../
   DownloadKernelOutput` = no such version. **A kernel's own version history is
   a merge source** (rule 26b) and is invisible to every `kaggle kernels` read
   command: on `cayleypy-tetraminx-tpu-beam-q` (31 versions, 15 with a
   `submission.csv`), latest-only was worth -5 moves and the best single
   version -7, but the union of all versions was **-51** across 13 versions
   each holding a unique win -- 10x (28,359 -> 28,308). Puller:
   `scripts/25_pull_kernel_versions.py --kernel <owner>/<slug> --file
   submission.csv --out-dir <dir> --max-version 45`, then fold `<dir>` into
   `90_merge_all.py --extra`.
   **RE-SWEEP: a version sweep is not a one-shot harvest.** The same kernel,
   swept again 4 days later at 119 versions (v55-v119 new), was worth another
   **-91 over 72 pids** (28,185 -> 28,094, 2026-08-09). Any kernel still being
   re-pushed keeps accruing recoverable moves; re-probe the ceiling each time
   rather than assuming the last sweep drained it. **But version COUNT is a bad
   proxy for merge value** -- in that same pass
   `alexandervc/cayleypy-rw-models2-tetraminx` (276 versions) and
   `markcelliott/frames-saturate-at-two-tpu` (1) both contributed **0**, a
   genuine null (0 bad-alphabet rows, paths replay-verify) because their
   `submission.csv` is mostly long fallback: beam-quality pids (<=31 moves)
   anywhere in history were 320/1000 for ours vs 53 and 37 for theirs. Check
   that ratio before spending 200+ requests on a kernel. Cheap ceiling probe:
   binary-search the two 404 sources above (~10 requests, not a blind scan).

7g. **`pgrep -f <pat>` / `pkill -f <pat>` SELF-MATCH the checking command's own
   arg list** — the running `bash -c "... pgrep -f gcp_beam ..."` contains the
   pattern, so `pgrep -f gcp_beam` always finds ITSELF (false "still running")
   and `pkill -f gcp_beam` can kill the shell before it relaunches. Cost 3+
   wasted calls on 2026-07-24 (a self-kill of the launcher, a false "STILL
   RUNNING"). **Fix**: check with `ps -eo cmd | grep '[g]cp_beam' | grep -v grep`
   (bracket-first-char defeats the self-match) or match a path-qualified string
   the checker doesn't contain (`ps -eo etimes,cmd | grep 'tpu-env/bin/python
   gcp_beam'`). Never trust a bare `pgrep -f`/`pkill -f` on a pattern that also
   appears in the command doing the matching. Sibling of 7b/7c (Windows process
   management gotchas).
   **BRACKETING THE PATTERN IS NECESSARY BUT NOT SUFFICIENT** — tripped 3x on
   2026-08-01 *with this rule already written*. `[g]cp_beam` protects the pattern
   token, but the self-match is against the WHOLE command line, so it still fires
   whenever the name appears UNBRACKETED anywhere else in the same command:
   a file path (`gcloud storage cp .../gcp_beam_tetraminx.py ~/tetra/`), a launch
   line (`bash ~/supervise_generic.sh`), even a `grep FRAME_SPEC
   ~/supervise_generic.sh`. All three killed their own ssh shell mid-command, and
   two more produced FALSE "1 process running" counts that nearly caused a double
   launch on one TPU. **The only reliable form is two calls**: read the pids
   (`pgrep -f "[p]attern"`, harmless even if it self-matches), then
   `kill <explicit numeric pids>` in a SEPARATE call that mentions no pattern at
   all. Never put a kill and a launch/copy of the same binary in one command.

8. **Never use `sed -i` for in-place edits on this machine.** Windows MINGW's
   sed silently truncates files to 0 bytes (one occurrence cost ~5 min recovering
   `run_benchmark.py`). Use the `Edit` tool, or `sed 'pattern' file > file.new && mv file.new file`.

9. **Before adding any new megaminx experiment, check `megaminx/to_do_shortlist.md`.**
   That file is the active to-do list; an idea already there or already superseded
   by something on the list shouldn't be re-proposed. The 70K-score goal is the
   anchor — evaluate every new suggestion against "does this move us toward 70K
   or just save GPU hours?" And before EVALUATING whether a forward-looking
   idea would help, grep `megaminx/EXPERIMENTS.md` (and the cube `EXPERIMENTS.md`)
   for a model-ID verdict and skim the relevant config dataclass (e.g.
   `bellman.py`): many ideas are already run+rejected with the hook still present
   as a default-off flag (e.g. symmetry/rotation augmentation on the V head =
   m31, REJECTED). Confirmed 2026-06-10.

10. **Never use literal `%` in argparse `help=` strings** — Python 3.14 became
    strict about `%` in help text (it's reserved for C-style format specs).
    A help string like `"~30% by difficulty"` raises
    `ValueError: badly formed help string` at argparse construction time.
    Cost in this project: 4-hour m39a queue had to restart because the strat-5
    step crashed at parse_args(). **Fix**: escape as `%%` or rephrase
    ("30 percent" not "30%"). Audit new `help=` strings before commit.

11. **(Megaminx)** **Production full-1001 solve recipe is multi-pass + sym-ensemble 4**:
    `--sym-ensemble 4 --beams 16384,65536 --max-steps 60,150 --bf16 --resume`. Single-pass
    `--beams 65536 --max-steps 120` is for SMOKE TESTS ONLY — confirmed
    2026-05-11 that single-pass fails for ~50% of pids past depth 60 (m_dd_v0
    50ep on GCP: 41/79 model solves, jumping to 0/20 for pids 60-79). Cost of
    skipping: ~9h of GCP time wasted on an unsubmittable CSV. **`--niss` DROPPED
    2026-05-27**: ablation (AZ v4 V + sym4, hardest pids 991-1000, fp32 on Kaggle
    GPU) found no-NISS solves 10/10 -- sym4's rotation diversity already covers
    what NISS would add (0 rescues; a rare 1-2 move shave at 2x wall). The 88,195
    and 78,029 submissions used the older pre-sym4 multi-pass + NISS recipe.
    See [[niss-redundant-with-sym4]].

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
    **TPU clause (confirmed 2026-06-12):** on the TPU JAX beam, the production
    driver is **`gcp_beam_v_only.py` (V-only)**. `gcp_beam_v6e.py` runs
    V+qshort (m23_v3 Q) and INFLATES paths with the AZ v4 V — pid 992 = **96
    (qshort 96M) vs 78 (V-only 48M)**, where V-only at half the width + no sym
    beat the merged floor (85) by 7. Even the re-distilled m23_v3 qshort hurts.
    Do NOT use `gcp_beam_v6e.py` for solves. NOTE: every TPU beam in the current
    submission was qshort-built, so a V-only re-run likely improves many hard
    pids. See [[megaminx-v6e-beam-port]] + `megaminx/findings_vonly_qshort_2026-06-12.md`.

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

19. **Don't trust `git check-ignore` to audit ignore rules against an existing
    tree.** It silently passes (empty output, exit 0) on files that are
    *already tracked*, even when those files match the current `.gitignore` —
    because `check-ignore` reports what *would* be ignored if untracked, and
    tracked files override ignore rules. To find drift (committed files that
    should now be ignored), use:
    `git ls-files | grep -E '\.(csv|pt|pkl|log|...)$'`. Then
    `xargs -d '\n' git rm --cached --quiet` to untrack. Confirmed 2026-05-19
    during the private-repo cleanup — ~2 confused tool calls before realizing
    `check-ignore` wasn't the right audit tool.

20. **`gh` CLI lives at `C:\Program Files\GitHub CLI\gh.exe`** and is
    authenticated as `Erlemar` via Windows keyring (token scopes: `repo`,
    `workflow`, `gist`, `read:org`). On a fresh shell call by full path;
    on persistent shells PATH picks it up after the first new session post-
    install. No `GITHUB_TOKEN` env-var plumbing is needed — gh reads from the
    keyring transparently. The cayley project has two private remotes under
    Erlemar: `cayley-solvers` (origin, full history) and `cayley` (new as of
    2026-05-19); choose one as canonical before doing serious work on both.

21. **(Megaminx)** **10-pid bench is NOT a sufficient acceptance gate for a new V
    model.** A V model that passes V calibration (V(V0)≈0, V(d=1)≈1) AND a
    hard-spread 10-pid bench (e.g., 0,100,200,300,500,600,700,800,900,950 at
    beam 65k --no-pdb --bf16) can STILL fail strat-51 catastrophically.
    Confirmed 2026-05-19: m_v11 11.8M Stage 2 ep11 had V(V0)=0.012, V(d=1)=1.00,
    10-pid bench 10/10 / 991 — but strat-51 stuck at 16/20 in 1h+, and pids
    995-998 with the production recipe were +43% moves/pid worse than AZ v4.
    The 10-pid bench's spread doesn't probe the failure modes the V hits on
    the hardest puzzles. **Binding gate**: strat-51 (`--stratified 5
    --strat-seed 0`) with PRODUCTION recipe (`--sym-ensemble 4 --beams
    16384,65536 --max-steps 60,150 --niss --bf16`) — NOT the single-pass
    beam 65k that `/megaminx-eval-v` uses by default. If strat-51 takes >2h or
    gets stuck on any single pid, that IS the failure signal — kill and
    don't promote to full-1001. Cost of skipping: ~6h of GCP monitoring
    + ~3.5h debugging during m_v11 launch.

22. **(Megaminx)** **`torch.compile(model, dynamic=False)` hangs indefinitely on
    custom SDPA-with-additive-attn_mask transformers** (GraphTransformer-style).
    Confirmed 2026-05-19 during the GT bellman launch: the first batch's compile
    spun CPU for 5+ minutes with no error, no progress, no traceback. GPU memory
    didn't grow. Killing the process and restarting with `compile_model: false`
    in the YAML let training proceed at full eager-bf16 speed (~150 s/epoch on
    L4 for d=192/3L/4h). For inference, `torch.compile(model, dynamic=True)`
    DOES work on the same architecture (no hang) but only gives ~1.5× speedup at
    small batches and is neutral at large batches — not worth the 57-second
    compile cost. **Rule**: don't set `compile_model: true` for any trainer that
    uses a custom transformer layer with SDPA `attn_mask`. For ResMLP families
    compile is still safe (and used by all canonical recipes). Companion to
    Rule 2 (which covers inference-side compile-without-padding for ResMLPs).

23. **(Megaminx)** **V@d=high should saturate at the puzzle diameter, not keep
    growing.** Confirmed 2026-05-19/20 (graph-transformer V experiment).
    Working ResMLP V models saturate near megaminx's true diameter (~21-30) for
    states past d=40 on random walks. The GraphTransformer V, despite matching
    BFS-d6 calibration, kept growing past d=40 (V@d=80=42) — and FAILED beam
    search at b=65536 in every configuration tested (direct, V-distilled into
    a same-arch 6M ResMLP, AZ-style aux loss with λ=0.5). Saturation IS the
    load-bearing property: it's what lets beam navigate around walk-depth
    diversity at the same true distance. **Quick check for a new V model**:
    if `V@d=80 - V@d=40 > 10` on RW-generated states, the V landscape is
    drifting (predicting walk-depth, not true distance). Beam will fail
    even if BFS-d6 calibration looks good. See `gt_v_no_saturation.md`
    memory for the full story and the one variant (GT-Q distillation) that
    hasn't been falsified.

24. **All Python `print`/logging output must be ASCII-only, and every `open()`
    must pass `encoding="utf-8"`** — the Windows console + default file codec is
    cp932. Confirmed 2026-05-24: a diagnostic log line containing `V̄`/`Δ̄`
    (combining accents) crashed a training run mid-epoch with
    `UnicodeEncodeError: 'cp932' codec can't encode`; separately, reading an
    em-dash-containing YAML config with a bare `open(path)` raised
    `UnicodeDecodeError: 'cp932' codec can't decode`. The repo's scripts already
    pass `encoding="utf-8"` on reads — match that everywhere, and for ad-hoc
    `python -c` set `PYTHONUTF8=1` (or just keep output ASCII). Sibling of rules
    7d (Kaggle CLI cp932) and 10 (argparse `%`). Don't use Unicode glyphs
    (em-dash, ±, arrows, accented chars) in any string that gets printed.

25. **(Megaminx)** **Bridge compression is V-saturation-bound, not beam-bound.**
    The mechanism (`scripts/81_bridge_compression.py`, `scripts/82_generate_residuals_for_tpu.py`,
    `kaggle_notebooks/tpu_beam_bridge_residuals_b1m_shareable/`, `scripts/83_splice_tpu_residuals.py`)
    is shipped + tested end-to-end. Works on loose paths (our 77,214 → 77,086,
    ~3-4% win rate, ~1 move/pid on top-50 long), but **fails on community
    min-merged paths past d≈30** because V saturates at ~25-30 → the
    predicted-save score collapses to false-positives on deep residuals. Confirmed
    2026-05-22/24: pid 1000 returned 0 wins at every config tested (local beam 65k,
    GCP L4 beam 256k, local beam 131k + sym4 + NISS, TPU B=1M + sym4 + NISS from
    a 70-move base). Phase 2 qshort regressed on residuals (off-distribution for
    m23_v3's training set) — drop qshort in any bridge configuration. **Rule**:
    don't run bridge on already-min-merged community paths or hardest pids; reserve
    it for OUR-pipeline paths with len ≥ 80 OR for cheap fallback-rescue passes.
    Full story: [[bridge-compression-findings]] memory.

26. **(Megaminx)** **Before claiming a bridge / rescue / post-processing win,
    compare against the n-way per-pid min over ALL available CSVs.** Confirmed
    2026-05-24: bridge on community 75,200 reported "+17 moves saved" (75,200 →
    75,183), but the newer community csv `min_count_per_id_before (8) (1).csv`
    (75,266) already had shorter paths for 5 of the 6 "winning" pids. After
    `min(75200, 75266, bridge)` the incremental contribution from bridge was
    actually **only 1 pid (−6) = pid 991**. The 17-vs-1 misattribution was caught
    only when the user flagged it. **Rule**: when reporting an improvement on
    base B, also compute the per-pid min over every other CSV that covers the
    same pids (community submissions, prior our-CSVs) and compare against THAT
    floor. Otherwise you're claiming credit for moves the community already
    contributed.

26b. **The n-way min is only as good as its SOURCE SET, and that set is not
    stable.** Seven merges on 2026-08-03 scanned 121+85, 112+78, 122+78, 128+78
    and finally 131+121 files — silently. Before quoting a merged total:
    (a) **re-pull live machines every time** — one early `scp ~/out/*.json` went
    stale and later merges only refreshed the watched file; the final full
    re-pull took JSONs 78 -> 121 and found moves. A stale copy caps the merge
    with no warning. (b) **Copy sources into a stable dir** (`submissions/`,
    `results/<box>/`) instead of `--extra`-globbing session scratchpads under
    `AppData/Local/Temp/...`, which are temp dirs whose set changes between runs
    — `submission_31.csv` surfaced only on the 6th merge and was worth −3.
    (c) **Search by CONTENT, not name or location**: `submission_dad.csv` sat in
    `megaminx/` holding TETRAMINX data, and three `submission_publ*.csv` sat in
    an unrelated `rogii-wellbore-geology-prediction/` folder — together −22
    moves. Test the header AND that the move alphabet matches the puzzle's
    generators (a megaminx file passes the header test and fails the second).
    Attribution corollary: results from OTHERS RUNNING OUR PUBLISHED NOTEBOOK
    are downstream of our own solver, not a third-party source — miscounting
    them as external understated our own contribution ~3x that day.

27. **(Megaminx)** **Custom transformers with explicit-`attn_mask` SDPA are
    memory-heavy — size the batch for the `(B,H,T,T)` score matrix, not param
    count.** `F.scaled_dot_product_attention(q,k,v,attn_mask=...)` falls back to
    the math kernel, which materializes AND saves-for-backward the full
    `(B, n_heads, T, T)` attention matrix per call. For the bipartite GT (T=264,
    8 heads, 2 SDPA/layer x 4 layers) that is ~4.25 GiB *per SDPA* at batch 2048
    -> OOM on a 24 GB L4 **even with gradient checkpointing** (backward recompute
    holds 2 matrices/layer). Confirmed 2026-05-25 (3 GCP OOM round-trips).
    **Fixes**: train at batch <=1024 + `grad_checkpoint=True` +
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; for eval/beam build the
    model with `inference_chunk_size<=1024` AND run under bf16 autocast (FP32 eval
    doubles the matrix -- the pooled-4096 validation OOM'd this way). The model is
    only ~3.5M params but ~30-50x slower per forward than the ResMLP, and L4 epoch
    time was ~727s (GT is expensive to TRAIN too, not just infer). Sibling to
    Rule 22 (compile-hang on the same SDPA-with-mask path). Full story:
    EXPERIMENTS.md 2026-05-25 / `bipartite_gt_q_runbook.md`.

28. **Run the MATCHED control before quoting any A/B delta — and treat
    byte-identical results as an unwired flag, not a null result.** A total is
    only meaningful against the same pid set, machine, flags AND checkpoint;
    any other difference gets silently attributed to the variable under test.
    Cost 2026-08-02: a full day of local `history_depth=0` runs compared against
    TPU `history_depth=1` baselines produced two conclusions that had to be
    retracted — "V-consistency is falsified" (against its own matched baseline it
    is **-5**) and "the soup is redundant with history_depth" (history was worth
    only ~1 move on that path, far too little to absorb a -4). Three corollaries:
    (a) if the control does not exist, run it — it is cheaper than the wrong
    conclusion; (b) **never generalise an interaction from one measured pair to an
    unmeasured one** — "hd+soup don't stack" was wrongly extended to hd+consistency,
    which measured **90% additive**, and the user caught it; (c) if a config change
    gives byte-identical per-pid results, the flag is not wired — `30_solve.py
    --history-depth` was parsed and never forwarded, caught only because an "hd=1"
    run reproduced the hd=0 control exactly. Score against the true standing best
    with `tetraminx/scripts/56_compare_vs_final.py`, not a remembered floor
    (see rule 26).

29. **`--help` is NOT a safe probe — a script without argparse ignores argv and runs
    its real workload.** Confirmed 2026-08-23: `12_mitm_oracle.py --help` executed a
    full meet-in-the-middle oracle against the d<=5 ball and blocked a 10-minute Bash
    call before the harness killed it. The same sweep also mis-reported the script as
    "hanging" when it was working correctly. **Check first**:
    `grep -q add_argument <script> || echo "no argparse -- do NOT probe with --help"`.
    Scripts in this repo that take NO arguments and run on import:
    `10_derive_symmetry`, `12_mitm_oracle`, `36_phase0_diag`, `42_commute_reduce`,
    `50_verify` (positional only).

30. **After fixing an allocation-shape bug, fix the PATTERN, not the instance you
    measured.** Confirmed 2026-08-23 on cube555: an endgame lookup widened states to
    int64 over the whole candidate shortlist (`(N,150)` int64 = 4.69 GiB) and OOM'd a
    T4. It was chunked and the run re-pushed — and failed again with the SAME class of
    error 12.00 GiB, because the identical widening in the table CONSTRUCTION
    (10,739,017 x 150 x 8) had never been looked at. Two Kaggle sessions for one bug.
    **Fix**: after any such fix, grep every sibling and size it —
    `grep -n '\.long()\|to(torch.int64)' <file>` — and record the result. A bug found
    by measurement tells you the shape of a class, not the location of one defect.

31. **"It fits locally" is not evidence it fits on the target — Windows silently
    spills CUDA past physical VRAM.** WDDM lets an allocation overflow into host RAM,
    so a 12 GiB tensor SUCCEEDS on the 16 GB 4090 and hard-fails on Kaggle. Two
    consequences, both confirmed 2026-08-23: (a) a Kaggle T4 has **14.56 GiB** usable
    against the 4090's ~15.99 GiB, so even without spilling the margin is ~1.4 GiB;
    (b) `torch.cuda.max_memory_allocated` can report a figure LARGER than the card —
    a 25.55 GiB reading on a 16 GB GPU was dismissed as broken instrumentation when it
    was in fact pointing straight at the offending allocation. **Treat an impossible
    peak as a signal, not noise**, and size against the target's VRAM. Also reset peak
    stats BEFORE a task, not after, or the first reading folds in one-time setup.

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
- **(Megaminx) Symmetry/rotation augmentation on the V head** (m31, 2026-04-29):
  REJECTED -- 50/51 / 95.76 vs m05 89.4 (+6 mean). At 6M params it dilutes the
  distance signal rather than sharpening it; falsified "orbit coverage is binding."
  Symmetry aug belongs on the Q-shortlister (m23_v2, KEPT), not the V head. The
  `lambda_sym` consistency-loss variant (`bellman.py`, default off) is the only
  clean-untried angle -- confounded inside the rejected m_repr_v0 bundle.
