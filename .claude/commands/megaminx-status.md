---
description: One-shot status across GCP solves, Kaggle kernels, and local runs for the megaminx competition.
---

# /megaminx-status

Bundles the status checks we run repeatedly during long solves. Reports compact
progress for each active workstream so you can decide what to do next.

## Execution

Do the following in parallel where possible (one Bash call per workstream):

### 1. GCP VM (Phase 1, Phase 2, or any cayley-gpu run)

```bash
IP=35.229.121.83  # or read from ~/.ssh/cayley_vm_ip if set
ssh -i /c/Users/and-l/.ssh/google_compute_engine \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    and-l@$IP \
    "ls -la /home/and-l/cayley/megaminx/submissions/ 2>&1 | tail -8; \
     echo '---active solves---'; \
     ps -ef | grep -E 'python.*03_solve' | grep -v grep | head -3; \
     echo '---tail of newest log---'; \
     ls -t /home/and-l/cayley/megaminx/submissions/*.log 2>/dev/null | head -1 | \
       xargs -I{} bash -c 'echo \"-- {} --\"; tail -8 {}'; \
     echo '---chain log---'; \
     cat /home/and-l/run_phase2_chain.log 2>&1 | tail -10"
```

Report: which scripts running, latest progress line (`pid=N attempts=N/M`), any chain status.

### 2. Local 4090 (Phase B or any local solve)

```bash
ls -la megaminx/submissions/*.log 2>&1 | tail -5
echo '---active processes---'
tasklist 2>&1 | grep -i python | head -5
echo '---GPU utilization---'
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>&1 | head -2
echo '---tail of newest log---'
ls -t megaminx/submissions/*.log 2>/dev/null | head -1 | xargs tail -8
```

Report: GPU util, memory, latest progress line.

### 3. Kaggle (training kernels and probes)

```bash
export KAGGLE_API_TOKEN=KGAT_630ac26efca89d28c5b2d496b238b71c
.venv/Scripts/kaggle.exe kernels list --user artgor --mine -p 1 -s 5 2>&1 | head -20
```

For any kernel showing `running`, also fetch:
```bash
.venv/Scripts/kaggle.exe kernels status artgor/<slug> 2>&1
```

Report: which kernels running / done / cancelled, last-completed time.

## Output format

Print a compact 3-section block:

```
GCP (35.229.121.83):
  - <script>: pid=N attempts=N/M (M-N remaining, ETA Xh)
  - chain: <last status line>

Local (4090):
  - <script>: pid=N attempts=N/M (GPU XX% util, X MiB)

Kaggle:
  - <slug>: <status> (started Xh ago)
```

End with a one-line "next obvious action" suggestion (e.g. "wait Xh for Phase B",
"submit current Phase 1 csv", "Kaggle kernel needs ckpt download").

## Gotchas

- `kaggle kernels output` while RUNNING returns nothing — only `status` works mid-run.
- After a kernel finishes (even cancelled/killed), `output` returns the partial /kaggle/working/ contents.
- GCP IP changes when the VM is stopped+started — re-fetch from `gcloud compute instances list` if SSH fails.
- 03_solve.py exits non-zero at end of full-range solves due to a benign `UnboundLocalError` on `report` (fixed in main, but pre-fix runs still affected). If you see exit code 1 on a finished solve, check the CSV row count, not the exit code.
