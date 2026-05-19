# TPU sweep runbook — `cayleypy-tpu-beam-smoke`

Operational notes for running a new TPU sweep next session. The technical
"why Kaggle TPU is weird" content lives in
`~/.claude/projects/.../memory/reference_kaggle_tpu_pjrt.md` — read THAT
first if anything fails in setup. This file is the project-side runbook:
sweep history, what's saturated, what to try next, and the exact
push/monitor/merge commands.

## Current state (2026-05-03)

- **Submitted best**: 79,522. The 4 TPU sweeps below are baked into it.
- **TPU-only union (4 sweeps)**: 82,342. Each sweep contributed wins in
  different bucket ranges — search-config diversity is cheap and reliable.
- **Notebook last pushed**: v19b (config: K=4 B=1M chunk=32768, pids
  500-1000). Hit Kaggle's 9h kill at ~pid 905; pids 905-1000 were not
  covered.
- **Dataset on Kaggle**: `artgor/megaminx-tpu-artifacts` v1 — `m05_epoch_0499.pt`,
  `m23_v2_epoch_0499.pt`, `rotations.npy`. Don't update unless changing model.

## TPU sweep history (this session)

Each row is a kernel push. Per-pair walls are CACHED (excluding compile).

| ver | K | B | chunk | scope (pids) | per-pair | wall | covered | wins into merge | notes |
|---|---|---|---|---|---|---|---|---|---|
| v16 | 4 | 131k | 4096 | full-1001 | 17s | 2.7h | 1001 | 103 | broad sweep, baseline |
| v17 | 8 | 262k | 4096 | full-1001 | 37s | 9h | ~819 | 385 | hit 9h kill at 82% |
| v17b | 8 | 262k | 4096 | missing pids | 37s | 2.4h | ~182 | +43 | follow-up kernel |
| v19a (broken) | 4 | 1M | **4096** | 0-499 | **42s/step** | killed early | 0 | — | per-chunk dispatch cliff |
| v19a-fix | 4 | 1M | 32768 | 0-499 | 150s | 10h17m | ~499 | 247 | 1.25s/step, normal |
| v19b | 4 | 1M | 32768 | 500-1000 | 150s | 9h00m | ~405 | 266 | hit 9h kill, pids ~905-1000 missing |

**Total this session**: 6 successful kernels, 4 distinct (K, B) configs,
−2,820 moves vs the entering-session 82,481.

## Quick start: push a new sweep next week

1. **Edit `build_notebook.py`** — three knobs:
   - Line 32: `TEST_PIDS = list(range(...))` — which pids this kernel runs.
   - Line 288: `K_SYM = 4` — number of rotations from `rotations.npy`.
   - Line 545: `B_TEST = 1048576` — beam size.
   - Line 550: `INTERNAL_BS = 32768` — chunk size. **Pick so total chunks =
     B*N_GEN/INTERNAL_BS lands in 500-2000 range. Outside this range you
     hit the per-chunk dispatch cliff (35× perf hit).**

2. **Build the notebook** (regenerates the .ipynb from the .py):
   ```bash
   .venv/Scripts/python.exe megaminx/kaggle_notebooks/tpu_beam_smoke/build_notebook.py
   ```

3. **Push to Kaggle**:
   ```bash
   export KAGGLE_API_TOKEN=KGAT_630ac26efca89d28c5b2d496b238b71c
   cd megaminx/kaggle_notebooks/tpu_beam_smoke
   .venv/Scripts/kaggle.exe kernels push -p .
   ```
   Returns a URL; the kernel starts running immediately.

4. **Monitor** (don't poll faster than ~10 min):
   ```bash
   .venv/Scripts/kaggle.exe kernels status artgor/cayleypy-tpu-beam-smoke
   ```
   Expect `KernelWorkerStatus.RUNNING` → `COMPLETE` (or hit 9h kill).

5. **When done**: `/megaminx-tpu-merge <out_name>` (slash command at
   `.claude/commands/megaminx-tpu-merge.md`). Then `/megaminx-submit ...`.

## Cost reference (chunk_size matters)

`INTERNAL_BS` heuristic — keep `chunks_per_forward = B*N_GEN/INTERNAL_BS`
in 500-2000:

| B | INTERNAL_BS | chunks/forward | per-pair (cached) |
|---|---|---|---|
| 131k | 4096 | ~768 | 17s |
| 262k | 4096 | ~1500 | 37s |
| 524k | 16384 | ~768 | ~70s (projected) |
| 1M | 32768 | ~732 | 150s |
| 2M | 65536 | ~732 | ~300s (projected) |
| 4M | 131072 | ~732 | ~600s (projected) |

A 9h kernel covers approximately:
- B=131k K=4 full-1001 ≈ 2.7h (overrun room)
- B=262k K=8 full-1001 ≈ 9h (tight, expect 9h kill at ~80%)
- B=1M K=4 full-1001 ≈ 18h (split into TWO kernels, 500 pids each)
- B=2M K=4 full-1001 ≈ 36h (TOO BIG — scope to top-200 hard-tail)

## What's saturated (don't re-run these exact configs)

These (K, B, pid-range) combinations are already merged into our 79,522 best.
Re-running them at the same model checkpoint is wasted compute:

- K=4 B=131k full-1001 (v16)
- K=8 B=262k full-1001 (v17 + v17b)
- K=4 B=1M pids 0-499 (v19a-fix)
- K=4 B=1M pids 500-905 (v19b before kill)

Re-running with **a different model checkpoint** (e.g., earlier epoch) WOULD
add diversity — but requires a new dataset version upload first.

## Next sweep priorities (planned, EV-ranked)

The bucket-niche finding from this session: each (K, B) config wins different
pid ranges. Diversifying further has clear EV at zero training cost. Best
candidates:

### v19c — fill v19b coverage gap (highest EV, smallest scope)

- **Config**: K=4 B=1M chunk=32768, pids `range(905, 1001)` (~96 pids)
- **Wall projection**: 96 × 4 / 8 ranks = 48 pairs/rank × 150s = 2h
  (+ ~30 min compile = ~2.5h kernel). FITS comfortably in 9h.
- **Expected gain**: 30-50 wins (v19b found 266 wins on pids 500-905; the
  long-tail rate was rising near pid 900 → roughly proportional gain on 905-1000).
- **Why first**: smallest kernel, fills a known gap, no exploration risk.

### v20 — K=8 B=524k on top-200 hard-tail (medium scope, new diversity)

- **Config**: K=8 B=524k chunk=16384, pids = top-200-longest from current best.
- **Wall projection**: 200 × 8 / 8 = 200 pairs/rank × ~70s = 3.9h
  (+ compile ~30 min = ~4.5h). FITS.
- **Expected gain**: K=8 with deeper beam than v17 (524k vs 262k); finds wins
  v17 missed because 262k was too narrow. Likely 50-150 wins, since v17 K=8
  was the single biggest contributor (385 wins).
- **Why second**: untried slice of (K, B) plane, plausibly beats v17 on its
  own bucket range.

### v21 — K=2 B=2M on top-100 hard-tail (exploratory)

- **Config**: K=2 B=2M chunk=65536, pids = top-100-longest.
- **Wall projection**: 100 × 2 / 8 = 25 pairs/rank × ~300s = 2.1h (+ compile
  → ~2.6h). FITS comfortably.
- **Expected gain**: K=2 has the broadest per-rotation depth at any compute
  budget. Untested; could find new long-tail wins or could underperform K=4
  diversity. Pure exploration.
- **Why third**: speculative — only valuable if v19c and v20 don't crack 78K.

### v22+ — beyond

- Different model checkpoint (m05_epoch_0399 instead of 0499) for trajectory
  diversity. Requires uploading a new dataset version. ~1h prep.
- Hybrid CPU+TPU for B>2M targeted on top-50 hardest. **Don't preemptively
  port.** First exhaust TPU-only.

## Kaggle TPU quota

- **Weekly quota: 20h** (rolling 7-day window, NOT calendar week).
  Confirmed via `kaggle kernels push` error "Maximum weekly TPU quota of
  20.00 hours reached." 2026-05-03.
- Each 9h kernel ≈ 9h of quota; budget ~2 full kernels per week, more if
  some hit early kills or are scoped short.
- GPU quota is separate: max **2 concurrent batch GPU sessions** (also
  confirmed via push error). TPU and GPU quotas don't share.
- Reset is rolling 7-day window. Check live status at
  https://www.kaggle.com/settings → Accelerator usage.
- If quota exhausted: pushed TPU kernels are **rejected at push time**
  (not queued). Wait until your oldest TPU run rolls out of the 7-day
  window before retrying.

## Common gotchas (this session's burns)

- **Don't push without rebuilding the .ipynb first.** `build_notebook.py`
  generates the notebook; pushing the stale .ipynb sends an old config.
- **MINGW `/tmp/<dir>` ≠ Windows `C:\Users\...\Temp\<dir>`** — the
  `/megaminx-tpu-merge` skill handles this; if you script around it
  manually, use `cygpath -w` or hardcode the Windows path.
- **9h kill is real and approximate.** v17 and v19b hit it at exactly 9h00m;
  v19a-fix got 10h17m as a bonus. Plan for 8.5h, treat 10h+ as a gift.
- **Per-iteration partial JSON saves are essential.** All `_mp_fn` workers
  write `rank_<r>_partial.json` after every (pid, rotation) pair. After a
  kill, `kaggle kernels output` recovers everything; we lose only the
  in-flight pair.
- **chunk_size BEFORE beam scaling.** v19a wasted ~5h because chunk=4096 hit
  the per-chunk dispatch cliff at B=1M. Always recompute INTERNAL_BS when
  bumping B.
- **Don't import `xm`/`xr`/`xmp` at module top.** Defer all torch_xla submodule
  imports into `_mp_fn`. Any device-creating call in the parent process
  blocks fork. (Reference memory has the gory details.)

## Post-kernel ritual (the slash command)

After a kernel completes (or hits 9h kill), the merge ritual is bundled in
`/megaminx-tpu-merge`:

```
/megaminx-tpu-merge <out_name>
```

Pulls Kaggle output, aggregates per-rank JSONs, maps move indices to names,
takes per-pid min over rotations, merges with current best CSV, verifies.
Reports per-bucket win distribution. Used 5+ times across v15-v19b without
rework. Then submit with `/megaminx-submit`.

## File map

```
megaminx/kaggle_notebooks/tpu_beam_smoke/
├── RUNBOOK.md              # this file
├── build_notebook.py       # generates the .ipynb; edit knobs here
├── cayleypy-tpu-beam-smoke.ipynb  # generated; what gets pushed
└── kernel-metadata.json    # Kaggle kernel ID + dataset link

.claude/commands/
├── megaminx-tpu-merge.md   # post-kernel ritual
└── megaminx-submit.md      # verify + Kaggle submit + update 3 docs

~/.claude/projects/.../memory/
└── reference_kaggle_tpu_pjrt.md  # technical: why Kaggle TPU is weird
```
