---
name: ihes-tpu-run
description: Configure, push, watch, and merge one run of the IHES cube TPU beam kernel (artgor/cayleypy-ihes-cube-tpu-beam). Use when the user asks to run the IHES TPU beam on some pids / with some model / at some width. Encodes the patch-push-monitor-pull-merge loop and every Kaggle session gotcha learned 2026-07.
---

# IHES TPU beam run

One full cycle of the kernel at `kaggle_notebooks/tpu_beam_ihes_shareable/`.
Ask the user only for what's not implied: pids, model, mode, width.

## Step 1 — rebuild + patch config

```bash
C:/Users/and-l/cayley/.venv/Scripts/python.exe C:/Users/and-l/cayley/kaggle_notebooks/tpu_beam_ihes_shareable/build_notebook.py
```

Then patch Cell 1 of the generated `cayleypy-ihes-cube-tpu-beam.ipynb`
(load JSON, edit `cells[1]["source"]` lines, assert the exact number of
replacements — see the patch scripts pattern in session scratchpads). Knobs:

- `PID_LIST = [..]` (explicit pids) or `START_PID`/`END_PID`
- `K_SYM` + `SYM_POSITIONS` — MUST satisfy positions within `[0, K_SYM)`;
  full group = `K_SYM = 48, SYM_POSITIONS = range(0, 48)`
- `use_pooled` (1 = one K*B beam/pid; root count barely affects wall) and
  `use_niss` (pooled: doubles roots; seq: doubles calls)
- `V_CHECKPOINT`: `e6_epoch_0499.pt` (reliable) | `az_cube_v1_v_only.pt`
  (E6-equal, decorrelated) | `az_cube_v2_v_only.pt` (high-variance, best
  upside tail — prefer for tie-breaking sweeps)
- `B_GLOBAL`: 32M default (29-34s/step, ~13 min/beam-call); 64M only where
  32M plateaued (84s/step, superlinear); memmap = NUM_STEPS*B*4 <= ~19 GB

Wall budget: ~40 beam calls per 9h kernel at 32M. Cell 3 prints the
projection and warns above 8.5h.

## Step 2 — CPU smoke if the KERNEL CODE changed (config-only changes skip)

Run the `local_smoke.py` pattern: execute the notebook cells in one
namespace with `XLA_FLAGS=--xla_force_host_platform_device_count=8`,
`/kaggle` paths redirected to scratchpad, tiny-B override, assert pid 1
solves at length 8 verified. Never push changed kernel code without this.

## Step 3 — push (guarded)

BEFORE pushing: pull any not-yet-downloaded outputs of the current latest
version (`kernels output` serves ONLY latest — pushing strands them).

```powershell
$env:KAGGLE_API_TOKEN='<token from kaggle-api-credentials memory>'; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING='utf-8'; & 'C:\Users\and-l\cayley\.venv\Scripts\kaggle.exe' kernels push -p 'C:\Users\and-l\cayley\kaggle_notebooks\tpu_beam_ihes_shareable'
```

"Maximum batch TPU session count of 1 reached" = a session is queued/running
(possibly invisible to `kernels status` — phantom-version desync); wait for
it, don't fight it. No CLI cancel exists.

## Step 4 — watch with Monitor (persistent), NOT background bash

Background bash loops get killed by the harness in ~20-60 min. Use the
Monitor tool, `persistent: true`, script that polls
`kaggle kernels status artgor/cayleypy-ihes-cube-tpu-beam` every 300-600s,
prints only on CHANGE, exits on COMPLETE/ERROR/CANCEL. Queue is typically
2-4h. A COMPLETE with a ~6s log and zero cells = phantom version; the real
session may still be running (check via push error / UI Session history).

## Step 5 — pull + merge

```powershell
& kaggle.exe kernels output artgor/cayleypy-ihes-cube-tpu-beam -p '<repo>/kaggle_notebooks/tpu_beam_ihes_shareable/_kernel_output_vN'
```

```bash
C:/Users/and-l/cayley/.venv/Scripts/python.exe C:/Users/and-l/cayley/scripts/merge_tpu_outputs.py \
  --tpu "kaggle_notebooks/tpu_beam_ihes_shareable/_kernel_output_vN/share_cube_submission_*.csv" \
  --floor submissions/floor_merged_20260712.csv \
  --out submissions/floor_plus_vN.csv
```

The merge post-processes, verify-gates, and attributes wins per-pid against
the floor (never claim a win the floor already had). Report wins/ties/worse.
Remember: pid path-length PARITY is fixed per pid (all generators are odd
permutations) — misses come as +2, and a win means -2.

## Context that caps expectations

The 2026-07 campaign concluded the 21,872 public floor is at/beyond
beam-class reach (24-band exhausted, 23-band sampled 0/36; see
EXPERIMENTS.md Exp 25-28). Re-probing with these configs has ~zero expected
yield — new runs make sense only with a fundamentally different method or a
new floor from outside.
