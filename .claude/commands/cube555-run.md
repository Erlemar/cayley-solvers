---
description: Run one cube555 Kaggle beam cycle end to end — rebuild the notebook, verify the dataset is actually live, push, watch persistently, pull, merge against the best submission, and verify. Encodes the ordering mistakes that cost four sessions on 2026-08-23.
---

# /cube555-run

One full cycle of the cube555 Kaggle beam. Takes an optional argument for the pid
selection (default: whatever the notebook ships with).

    /cube555-run                 # ship the notebook's default queue
    /cube555-run deepest:4       # 4 deepest pids
    /cube555-run 1034,1020       # explicit list

**Run the steps in this order.** The ordering is the point — every step below exists
because doing it later (or not at all) cost a Kaggle session on 2026-08-23.

## 1. Rebuild the notebook

```bash
P=/c/Users/and-l/cayley/.venv/Scripts/python.exe
$P /c/Users/and-l/cayley/cube555/gpu/build_notebook.py          # our 2xT4 PyTorch beam
# or, for the TPU kernel:
$P /c/Users/and-l/cayley/cube555/tpu/build_notebook.py
```

Both accept `--pids`; use it rather than hand-editing cells.

## 2. Test the notebook's OWN cells locally before pushing

```bash
$P /c/Users/and-l/cayley/cube555/gpu/test_notebook_local.py
```

This executes the generated cells against a staged `/kaggle/input` with synthetic
shallow pids. It has caught mount-path bugs, staging bugs and merge bugs that would
otherwise have been found only by burning a session. Expect `VERDICT : PASS`.

## 3. Push the dataset, then VERIFY IT IS LIVE BY CONTENT

`datasets version` returns when the UPLOAD finishes; Kaggle processes the version
asynchronously afterwards. A `kernels push` 20 s later mounts the PREVIOUS version and
the run fails with a bug you already fixed.

```bash
export KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN PYTHONUTF8=1 PYTHONIOENCODING=utf-8
K=/c/Users/and-l/cayley/.venv/Scripts/kaggle.exe
$K datasets version -p C:/Users/and-l/cayley/cube555/gpu/kaggle_dataset -m "<what changed>" --dir-mode zip

# then confirm the change is actually served, by CONTENT not by exit code:
$K datasets download -d artgor/cube555-gpu-code -f gpu_beam.py -p /tmp --unzip
grep -q "<a marker unique to this change>" /tmp/gpu_beam.py || echo "NOT LIVE YET -- wait"
```

Skip the dataset push entirely if no dataset file changed.

## 4. Push the kernel

```bash
$K kernels push -p C:/Users/and-l/cayley/cube555/gpu/push
```

A push immediately launches a run — do not push unless you intend to spend a session.
Check `machine_shape` is `NvidiaTeslaT4` in the metadata (bare `enable_gpu` can land on
a P100 whose sm_60 the image's torch does not support; see CLAUDE.md rule on this).

## 5. Watch with `persistent: true`

Kernel runs routinely exceed an hour, and a default `Monitor` dies silently at the cap —
indistinguishable from "still running". Always:

```
Monitor(persistent=true, command=<poll kernels status, emit on state CHANGE,
        break on COMPLETE|ERROR|CANCEL|FAILED>)
```

## 6. Pull the output

```bash
$K kernels output artgor/cayleypy-cube555-2xt4-beam -p <dir>
```

If this is slow, something large is being staged into `/kaggle/working` and captured as
kernel output — stage big intermediates (the 1.9 GB `anchors_d5.pt`, any repo clone) in
`/tmp` instead. For one specific file out of a large output, use the SDK
`ApiDownloadKernelOutputRequest` (see CLAUDE.md 7f(d)); the CLI cannot select files.

## 7. Merge against the BEST file and verify independently

The kernel's own `submission.csv` is only useful if it merged against our best floor.
Always re-verify with the handoff's own verifier, which is independent of our code:

```bash
( cd C:/Users/and-l/cayley/cube555_pull/cube555_handoff_2026_08_22 && \
  $P cube555/scripts/50_verify.py <merged.csv> )
```

Expect `1035/1035` and `VERDICT : PASS`. Note an EMPTY csv passes vacuously (`0/0`), so
check the row count and the total, not just the verdict. Score against the current best
(187,724 as of 2026-08-23), never a remembered figure.

## Reading the result

- `not_found` tasks cost ~2.5x a solve (they run all `max_steps`), so a queue's wall
  time is dominated by its failures — size the queue by the failures you expect.
- Report per-frame lengths, not just the best: frames differ by a lot (24 moves on one
  pid, and one frame rescued a pid the other missed entirely), and that spread is the
  evidence for how many frames to run next.
