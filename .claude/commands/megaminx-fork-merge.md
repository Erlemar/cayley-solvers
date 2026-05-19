---
description: Discover forks of cayleypy-megaminx-beam-shareable, pull each fork's output, aggregate per-pid wins across all forks, merge with current best.
---

# /megaminx-fork-merge

Bundles the multi-fork discovery + merge ritual for the shareable TPU kernel.
Used after collaborators run our `cayleypy-megaminx-beam-shareable` notebook
on their own TPU quotas. Distinct from `/megaminx-tpu-merge` (which targets
one specific kernel slug we own); this one searches for ALL public forks of
the shareable kernel.

## Usage

- `/megaminx-fork-merge <out_name>` — auto-pull from forks, aggregate, merge.
- `/megaminx-fork-merge <out_name> file=<path>` — skip the auto-pull;
  user has provided a pre-merged CSV at `<path>` (e.g., from Telegram or
  a Kaggle submission they assembled themselves). Just min-merge with current
  best.
- `/megaminx-fork-merge <out_name> base=<path>` — explicit base CSV
  (default: latest `merge_*.csv` by smallest move count).

## Defaults

- Search query: `cayleypy-megaminx-beam-shareable`
- Excluded slug: `artgor/cayleypy-megaminx-beam-shareable` (our own kernel —
  forks only)
- Per-fork working dir: `C:\Users\and-l\AppData\Local\Temp\<out_name>_<author>`
- Base: latest `merge_*.csv` in `megaminx/submissions/` (smallest move count)
- Standalone (community-only) output: `megaminx/submissions/<out_name>_community_standalone.csv`
- Final merged output: `megaminx/submissions/<out_name>.csv`

## Execution

### Mode A: file=<path> provided

Skip discovery + per-fork pull. Jump straight to step 4 (merge with base)
using the provided file as the community standalone CSV. Verify and report.

### Mode B (default): auto-pull from forks

#### 1. Discover forks

```bash
export KAGGLE_API_TOKEN=KGAT_630ac26efca89d28c5b2d496b238b71c PYTHONUTF8=1 PYTHONIOENCODING=utf-8
.venv/Scripts/kaggle.exe kernels list --search "cayleypy-megaminx-beam-shareable" --page-size 50 2>&1
```

Parse output. Filter out `artgor/cayleypy-megaminx-beam-shareable` (our own).
The remaining `<author>/<slug>` entries are forks. Print the list of forks
found with their last-run dates.

#### 2. Pull each fork's output

For each fork `<author>/<slug>`:

```bash
mkdir -p /tmp/<out_name>_<author>
.venv/Scripts/kaggle.exe kernels output <author>/<slug> -p /tmp/<out_name>_<author> 2>&1 | tail -3
```

If the dir is empty after pull: fork is queued/error/no-prior-runs. Skip
silently (don't retry — empty means no recoverable output yet).

For each fork that yields files: verify presence of `rank_*_final.json` or
`rank_*_partial.json`. Read fork's `run.log` to capture its (pid_range,
K_SYM) config — useful for the per-fork attribution report.

#### 3. Aggregate per-fork into a community standalone CSV

For each fork's per-rank JSONs, take per-pid min over verified rotations.
Across forks, take per-pid min again (fork-level union via min). Write the
combined CSV to `megaminx/submissions/<out_name>_community_standalone.csv`.

```bash
PYTHONUTF8=1 .venv/Scripts/python.exe -c "
import json, glob, csv, sys, os
sys.path.insert(0, 'megaminx/src'); sys.path.insert(0, 'src')
from megaminx.puzzle import Megaminx
puz = Megaminx.load('megaminx/data/puzzle_info.json')
MOVE_NAMES = list(puz.move_names)

# Collect from all fork dirs (one dir per fork, named <out_name>_<author>)
fork_dirs = sorted(glob.glob(r'C:\Users\and-l\AppData\Local\Temp\<out_name>_*'))
print(f'fork dirs: {[os.path.basename(d) for d in fork_dirs]}')

community_best = {}  # pid -> (path_idx_orig, source_author)

for fdir in fork_dirs:
    author = os.path.basename(fdir).replace('<out_name>_', '')
    records = []
    for path in (sorted(glob.glob(fdir + r'\rank_*_final.json'))
                 + sorted(glob.glob(fdir + r'\rank_*_partial.json'))):
        with open(path) as f:
            records.extend(json.load(f))
    seen = set(); unique = []
    for r in records:
        key = (r['pid'], r['rot_idx'])
        if key in seen: continue
        seen.add(key); unique.append(r)
    by_pid = {}
    for r in unique:
        if not r.get('verify_ok'): continue
        by_pid.setdefault(r['pid'], []).append(r)
    print(f'  fork {author}: {len(unique)} records, {len(by_pid)} pids covered')
    for pid, recs in by_pid.items():
        best = min(recs, key=lambda r: r['path_len'])
        if pid not in community_best or best['path_len'] < len(community_best[pid][0]):
            community_best[pid] = (best['path_idx_orig'], author)

# Write community standalone CSV
with open('megaminx/submissions/<out_name>_community_standalone.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['initial_state_id', 'path'])
    for pid in range(1001):
        if pid in community_best:
            path_idx, _author = community_best[pid]
            names = [MOVE_NAMES[m] for m in path_idx]
            w.writerow([pid, '.'.join(names)])
        else:
            w.writerow([pid, ''])

# Per-author attribution
from collections import Counter
c = Counter()
for pid, (_path, author) in community_best.items():
    c[author] += 1
print()
print('per-author best contributions:')
for author, n in c.most_common():
    print(f'  {author}: {n} pids')
print(f'total community-covered pids: {len(community_best)}')
"
```

### 4. Merge with current best

(Same as step 3 of `/megaminx-tpu-merge`, but reads the community standalone
CSV.)

```bash
PYTHONUTF8=1 .venv/Scripts/python.exe -c "
import csv
from collections import Counter

def load_paths(p):
    return {int(r['initial_state_id']): r['path'].split('.') if r['path'] else []
            for r in csv.DictReader(open(p))}

best = load_paths('<base>')
comm = load_paths('megaminx/submissions/<out_name>_community_standalone.csv')

bucket_wins = Counter(); bucket_savings = Counter()
wins = 0; total_savings = 0
for pid in range(1001):
    bp = best.get(pid, [])
    cp = comm.get(pid, [])
    if cp and (not bp or len(cp) < len(bp)):
        wins += 1; total_savings += len(bp) - len(cp)
        bucket_wins[pid // 100] += 1
        bucket_savings[pid // 100] += len(bp) - len(cp)

base_total = sum(len(v) for v in best.values())
print(f'current best: {base_total} moves')
print(f'community wins: {wins} pids, saved {total_savings} moves')
print()
print('per-bucket community wins:')
for b in sorted(bucket_wins.keys()):
    print(f'  bucket {b}: {bucket_wins[b]:3} wins, {bucket_savings[b]:5} saved')

with open('megaminx/submissions/<out_name>.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['initial_state_id', 'path'])
    for pid in range(1001):
        bp = best.get(pid, [])
        cp = comm.get(pid, [])
        chosen = cp if (cp and (not bp or len(cp) < len(bp))) else bp
        w.writerow([pid, '.'.join(chosen)])
print()
print(f'wrote merge: megaminx/submissions/<out_name>.csv')
print(f'new total: {base_total - total_savings}')
"
```

### 5. Verify merged

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

### 6. Report and suggest next step

Print:
```
Community fork merge ready: megaminx/submissions/<out_name>.csv = <new_total> (delta -<savings> vs <base_total>)
Forks contributing: <list>
Suggested next: /megaminx-submit <out_name>.csv "<description>"
```

DO NOT submit automatically.

## Gotchas

- **Empty per-fork dir = "no completed runs yet"**, not API error. Don't
  retry; skip the fork and move on. Confirmed 2026-05-05 on alexandervc's
  fork (queued state with no recoverable output).
- **Some forks won't have run our latest version** — they may have run
  v1/v3/v4 (B=2M/1.5M, OOM-prone). Their per-rank JSONs may be partial
  or contain fewer pids than expected. The aggregation handles this
  gracefully (per-pid min over verified results).
- **User-provided CSV path** (`file=<path>`) is the right mode when the
  user has already merged forks themselves (e.g., from a Telegram share or
  a Kaggle-submitted file they downloaded). Don't auto-pull in that case —
  it could miss data the user already has.
- **Mode A is faster but lossier than Mode B**: Mode A only sees what's
  recoverable via Kaggle API (latest complete version per fork). If a fork
  ran successfully but then queued a new version, prior-version output is
  lost. The user-provided file in Mode B can include data we'd otherwise
  miss.

## When to run

After collaborators have had time to run the shareable kernel. Cadence is
"as the user requests" — not on a schedule.

## Cost reference

| forks found | per-fork pull | aggregate + merge |
|---|---|---|
| 1-2 | ~10s each | ~5s |
| 5+ | ~10s each | ~15s |

Mode B (file path provided) skips the per-fork pull entirely: ~5s total.
