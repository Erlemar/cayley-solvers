---
description: Pull the latest cayleypy-tpu-beam-smoke kernel output, aggregate per-rank JSONs, merge with current best, verify, and prep for submission.
---

# /megaminx-tpu-merge

Bundles the recurring TPU post-completion ritual we run after every kernel
finishes (or hits Kaggle's ~9h kill). Used 5+ times across v15 → v19a-fix.

## Usage

- `/megaminx-tpu-merge <out_name>` — pulls Kaggle output, aggregates ranks,
  merges with current best, writes `megaminx/submissions/<out_name>.csv`.
- `/megaminx-tpu-merge merge_tpu_v19b` — typical use after v19b kernel.
- `/megaminx-tpu-merge merge_tpu_v19b base=megaminx/submissions/...csv` —
  explicit base CSV.

## Defaults

- Kernel slug: `artgor/cayleypy-tpu-beam-smoke`.
- Working dir: `C:\Users\and-l\AppData\Local\Temp\<out_name>`.
- Base: latest `merge_*.csv` in `megaminx/submissions/` (smallest move count).
- Output: `megaminx/submissions/<out_name>.csv`.
- Standalone (TPU-only) output: `megaminx/submissions/<out_name>_standalone.csv`.

## Execution

### 1. Pull kernel output

```bash
export KAGGLE_API_TOKEN=KGAT_630ac26efca89d28c5b2d496b238b71c PYTHONUTF8=1 PYTHONIOENCODING=utf-8
mkdir -p /tmp/<out_name>
.venv/Scripts/kaggle.exe kernels output artgor/cayleypy-tpu-beam-smoke -p /tmp/<out_name> 2>&1 | tail -3
```

If `KernelWorkerStatus.RUNNING`: ABORT — kernel hasn't finished yet.

### 2. Aggregate per-rank JSONs + map to move names

NOTE: MINGW `/tmp/<dir>` maps to `C:\Users\and-l\AppData\Local\Temp\<dir>` —
Python on Windows needs the Windows-native path.

```bash
PYTHONUTF8=1 .venv/Scripts/python.exe -c "
import json, glob, csv, sys
sys.path.insert(0, 'megaminx/src'); sys.path.insert(0, 'src')
from megaminx.puzzle import Megaminx
puz = Megaminx.load('megaminx/data/puzzle_info.json')
MOVE_NAMES = list(puz.move_names)

base = r'C:\Users\and-l\AppData\Local\Temp\<out_name>'
records = []
for path in sorted(glob.glob(base + r'\rank_*_final.json')) + sorted(glob.glob(base + r'\rank_*_partial.json')):
    with open(path) as f:
        records.extend(json.load(f))

# Dedupe (final supersedes partial; same data anyway)
seen = set(); unique = []
for r in records:
    key = (r['pid'], r['rot_idx'])
    if key in seen: continue
    seen.add(key); unique.append(r)

# Group by pid, take min over verified rotations
by_pid = {}
for r in unique:
    if not r.get('verify_ok'): continue
    by_pid.setdefault(r['pid'], []).append(r)
print(f'records: {len(unique)}  pids covered: {len(by_pid)}')

best_per_pid = {pid: min(recs, key=lambda r: r['path_len'])['path_idx_orig']
                for pid, recs in by_pid.items()}

# Write standalone TPU CSV
with open('megaminx/submissions/<out_name>_standalone.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['initial_state_id', 'path'])
    for pid in range(1001):
        if pid in best_per_pid:
            names = [MOVE_NAMES[m] for m in best_per_pid[pid]]
            w.writerow([pid, '.'.join(names)])
        else:
            w.writerow([pid, ''])
print(f'wrote standalone CSV')
"
```

Report: total records, pids covered, distribution of rotations-per-pid (helps
sanity check that K coverage is reasonable).

### 3. Merge with current best

```bash
PYTHONUTF8=1 .venv/Scripts/python.exe -c "
import csv
from collections import Counter

def load_paths(p):
    return {int(r['initial_state_id']): r['path'].split('.') if r['path'] else []
            for r in csv.DictReader(open(p))}

best = load_paths('<base>')
tpu = load_paths('megaminx/submissions/<out_name>_standalone.csv')

bucket_wins = Counter(); bucket_savings = Counter()
wins = 0; total_savings = 0
for pid in range(1001):
    bp = best.get(pid, [])
    tp = tpu.get(pid, [])
    if tp and len(tp) < len(bp):
        wins += 1; total_savings += len(bp) - len(tp)
        bucket_wins[pid // 100] += 1
        bucket_savings[pid // 100] += len(bp) - len(tp)

base_total = sum(len(v) for v in best.values())
print(f'current best: {base_total} moves')
print(f'TPU wins: {wins} pids, saved {total_savings} moves')
print()
print('per-bucket wins:')
for b in sorted(bucket_wins.keys()):
    print(f'  bucket {b}: {bucket_wins[b]:3} wins, {bucket_savings[b]:5} saved')

with open('megaminx/submissions/<out_name>.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['initial_state_id', 'path'])
    for pid in range(1001):
        bp = best.get(pid, [])
        tp = tpu.get(pid, [])
        chosen = tp if (tp and len(tp) < len(bp)) else bp
        w.writerow([pid, '.'.join(chosen)])
print(f'wrote merge CSV (delta -{total_savings})')
"
```

### 4. Verify merged

```bash
PYTHONUTF8=1 .venv/Scripts/python.exe -c "
import sys
sys.path.insert(0, 'megaminx/src'); sys.path.insert(0, 'src')
from megaminx.puzzle import Megaminx
from cayley.verify import verify_submission
from pathlib import Path
puz = Megaminx.load(Path('megaminx/data/puzzle_info.json'))
r = verify_submission(puz, Path('megaminx/data/test.csv'),
                      Path('megaminx/submissions/<out_name>.csv'))
print(f'verify: {r.n_valid}/{r.n_total} valid, total {r.total_moves:,}')
sys.exit(0 if r.all_valid else 1)
"
```

Anything other than `1001/1001 valid`: STOP. Investigate before submitting.

### 5. Report and suggest next step

Print:
```
TPU merge ready: megaminx/submissions/<out_name>.csv = <new_total> (delta -<savings> vs <base_total>)
Suggested next: /megaminx-submit <out_name>.csv "<description>"
```

DO NOT submit automatically.

## Gotchas

- **Windows path conversion**: MINGW `/tmp/foo` ≠ Windows `C:\Users\...\Temp\foo`.
  Use Python on the Windows path explicitly. `cygpath -w /tmp/foo` works.
- **Empty path rows in standalone CSV** are expected for pids the kernel didn't
  reach. The merge with current best fills them in.
- **Partial pids (fewer than K rotations completed)** are still useful — take
  min over what's available. Often 3/4 rotations is enough.
- **`verify_ok=False` rotations** happen rarely; the conjugation-translated path
  failed CPU verification. Skip those when taking min; other rotations cover
  the pid.

## Cost reference

Aggregation + merge typically takes 30s-2min depending on JSON size:

| kernel size | aggregate | merge |
|---|---|---|
| 5 pids smoke | ~2s | ~1s |
| 500 pids partial | ~30s | ~5s |
| 1001 pids full | ~1min | ~10s |
