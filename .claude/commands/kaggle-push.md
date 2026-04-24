---
description: Push a Kaggle kernel with the required env vars and poll its status to completion
argument-hint: <kernel-dir-under-/tmp/kernels/>
---

# Kaggle kernel push + poll

Push the Kaggle kernel at `/tmp/kernels/$1/` (or the path the user passed) with the
required env-var prelude, then poll status until terminal state (COMPLETE / ERROR).

## Required env vars

Always set these three before any kaggle CLI call:

```bash
export KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
```

Without `PYTHONUTF8` + `PYTHONIOENCODING`, Windows bash raises `cp932 codec can't decode`
on any non-ASCII character in the kernel source.

## Workflow

1. Push the kernel:
   ```bash
   /c/Users/and-l/cayley/.venv/Scripts/kaggle.exe kernels push -p /tmp/kernels/$1
   ```
   Parse the returned URL to get the Kaggle slug (may differ from the id in
   `kernel-metadata.json`; the slug comes from the title).

2. Poll every 30s (up to ~5 min) for status:
   ```bash
   /c/Users/and-l/cayley/.venv/Scripts/kaggle.exe kernels status artgor/<slug>
   ```
   Stop when status is `COMPLETE` or `ERROR`.

3. If status is `ERROR`, fetch the log to report the failure:
   ```bash
   /c/Users/and-l/cayley/.venv/Scripts/kaggle.exe kernels output artgor/<slug> -p /tmp/kernels_out/<slug>
   cat /tmp/kernels_out/<slug>/*.log | python -c "
   import json, sys
   for e in json.loads(sys.stdin.read()):
       d = e['data'].rstrip()
       if d: print(f'{e[\"time\"]:6.1f}s  {d[:200]}')"
   ```

4. If status is `COMPLETE` and outputs include model checkpoints, pull them with
   `kaggle kernels output -p <dir>` and report the local paths.

## Known issues to pre-emptively handle

- **Kernel id vs title slug mismatch**: on first push, Kaggle derives the slug from the
  title. If `kernel-metadata.json`'s `id` doesn't match, subsequent pushes return 409.
  Fix: after first push, pull kernel metadata and rewrite `id` to the actual slug.

- **Private dataset mount path**: kernel code should probe both
  `/kaggle/input/<slug>/` and `/kaggle/input/datasets/<owner>/<slug>/`.

- **P100 sm_60 + current PyTorch**: if the kernel is GPU, its `pip install` must pin
  `torch==2.4.1` and its config must set `compile_model=False`.

- **`git clone` blocked in kernels**: use `curl -fSL <codeload tarball>` instead.

See `reference_kaggle_pipeline.md` (in user memory) for the full playbook.
