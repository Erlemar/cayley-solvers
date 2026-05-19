---
description: Run the top-N long-pid sym-ensemble rescue + merge cycle on the current best megaminx submission.
---

# /megaminx-rescue

Automate the canonical "rescue the long-tail" cycle that we run after each new
best submission. Extracts the top-N longest paths from the current best CSV,
runs sym-ensemble + qshort + beam 524k on those pids, merges the rescued
paths into the base, verifies the merged CSV, and prepares it for submission.

## Usage

- `/megaminx-rescue 50` — rescue top 50 long-pids of current best submission with K=4 sym (default).
- `/megaminx-rescue 80 K=8` — rescue top 80 with K=8 (diminishing returns past K=4).
- `/megaminx-rescue 50 base=path/to/sub.csv K=4` — explicit base CSV.

## Defaults

- Base submission: latest in `megaminx/submissions/` matching `merge_*.csv` (highest path count = lowest score).
- K (sym-ensemble): 4.
- Beam: 524288.
- Max-steps: 150.
- Teacher: `megaminx/models/m05_bellman_warm/epoch_0499.pt`.
- Student (qshort): `megaminx/models/m23_v2_sym_aware/epoch_0499.pt` (sym-aware, pair with --sym-ensemble).
- Output prefix: `merge_plus_sym{K}_top{N}`.
- Rotations: `megaminx/data/rotations.npy` (default; auto-loaded by `--sym-ensemble`).

## Execution

### 1. Identify base submission and extract top-N pids

```bash
PYTHONUTF8=1 .venv/Scripts/python.exe -c "
import csv
sub = '<base>'
rows = list(csv.DictReader(open(sub)))
rows.sort(key=lambda r: -len(r['path'].split('.')))
print(','.join(r['initial_state_id'] for r in rows[:<N>]))
"
```

Report the pid list, length range of top-N, and sum of path lengths.

### 2. Launch the rescue solve (background)

```bash
nohup .venv/Scripts/python.exe megaminx/scripts/03_solve.py \
  --checkpoint megaminx/models/m05_bellman_warm/epoch_0499.pt \
  --qshort-student megaminx/models/m23_v2_sym_aware/epoch_0499.pt \
  --qshort-alpha 2 \
  --out megaminx/submissions/<output>.csv \
  --pids <pid_list> \
  --beams 524288 --max-steps 150 --bf16 \
  --sym-ensemble <K> \
  > megaminx/submissions/<output>.log 2>&1 &
```

### 3. Monitor with tight filter

```bash
# expected wall: K=2 ≈ 137s/pid, K=4 ≈ 280s/pid, K=8 ≈ 560s/pid
tail -F megaminx/submissions/<output>.log | grep -E --line-buffered \
  "attempts=20/|attempts=40/|attempts=60/|attempts=80/|Traceback|Error|verify:|stats:|wrote |OOM"
```

(Use Monitor tool with `persistent: true`.)

### 4. After completion: merge

```bash
.venv/Scripts/python.exe megaminx/scripts/16_merge_rescue.py \
  --base <base>.csv \
  --rescue megaminx/submissions/<output>.csv \
  --out megaminx/submissions/<merged>.csv
```

Report: `replaced X/N (saved Y moves)`, new total, regressions count (must be 0).

### 5. Verify merged

The merge script verifies inline and prints `verify: 1001/1001 valid, total <T>`.
If anything other than 1001/1001 valid: STOP, do not proceed to submit.

### 6. Tell user the result

Print: "ready to submit <merged>.csv (current_best - delta = new_total)".
Suggest `/megaminx-submit <merged>.csv "<description>"` as next step.

DO NOT submit automatically. The user controls Kaggle slot usage.

## Gotchas

- **Pair m23_v2 (sym-aware) with `--sym-ensemble`**, not the original m23 (distilled
  on forward-state V; lower recall on inverted/rotated states). See
  `megaminx_gotchas.md` "Q-shortlister student must match teacher" entry.
- **K=4 is the sweet spot.** K=8 saves ~80% as much per pid at 2× wall.
- **Beam max-steps = 150 minimum** (real path tail is ~99-103 in current submissions).
  Lower max-steps causes silent failures (model never converges).
- The merge takes per-pid min of base vs rescue. If rescue is uniformly worse than
  base for a pid, base is kept. 0 regressions is the expected outcome with K≥2.
- Rotation index 0 (identity) is always one of the K rotations selected.

## Cost reference (4090 Laptop, beam 524k)

| K | wall/pid | 50 pids | 80 pids |
|---|---|---|---|
| 2 | ~140s | ~2h | ~3h |
| 4 | ~280s | ~4h | ~6h |
| 8 | ~560s | ~8h | ~12h |

Pick K based on remaining session time and previous-pass exhaustion.
