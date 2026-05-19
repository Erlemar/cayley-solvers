---
name: "source-command-megaminx-eval-v"
description: "Strat-5 acceptance gate for a megaminx V model. Runs 51-pid stratified eval at beam 65k, compares to m05 baseline (50/51 / mean 89.4), reports pass/fail."
---

# source-command-megaminx-eval-v

Use this skill when the user asks to run the migrated source command `megaminx-eval-v`.

## Command Template

# /megaminx-eval-v

Standard strat-5 evaluation for a candidate V model. Encapsulates the
canonical "is this V better than m05?" check we run after every retrain
or recipe ablation.

## Usage

`/megaminx-eval-v <CHECKPOINT_PATH>`

If the checkpoint is omitted, evaluate the production V (`m05_bellman_warm`)
as a sanity check on the eval pipeline itself.

## Acceptance gate (binding for any new V)

A new V counts as "an improvement worth trusting" only if **all** hold:
1. **Strat-5 solve count**: ≥ +3 puzzles vs m05 on `--stratified 5
   --strat-seed 0` (51 puzzles).
2. **Path quality**: mean `model_avg` ≤ 0.95 × m05's 89.4 = **≤ 84.9**.
3. **No bucket regresses by more than 1 solve.**

Failing any single criterion = **not acceptable**. If 2 of 3 pass, that's
a TIE (or noise) — re-validate on a second seed (`--strat-seed 1`)
before committing.

## Execution

```bash
CKPT="${1:-megaminx/models/m05_bellman_warm/epoch_0499.pt}"
NAME="$(basename $(dirname $(dirname $CKPT)))_strat5"
OUT="megaminx/submissions/${NAME}.csv"
LOG="megaminx/submissions/${NAME}.log"

# Verify checkpoint exists
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

echo "=== Strat-5 eval ==="
echo "  checkpoint: $CKPT"
echo "  output:     $OUT"
echo ""

# Always launch with python -u to avoid stdout buffering when piped to nohup later
PYTHONUTF8=1 .venv/Scripts/python.exe -u megaminx/scripts/03_solve.py \
    --checkpoint "$CKPT" \
    --out "$OUT" \
    --beams 65536 --max-steps 150 \
    --stratified 5 --strat-seed 0 \
    --bf16 \
    2>&1 | tee "$LOG"

# Optional hard-tail variant — strat-5 on buckets 7-10 only (16 puzzles, where
# the model actually fails). Use when full strat-5 saturates on solve count.
# Add `--strat-buckets 7,8,9,10` to the command above.

# After completion, parse the final summary lines and compare to baseline
echo ""
echo "=== Acceptance gate vs m05 baseline (50/51 / mean 89.4) ==="
.venv/Scripts/python.exe -c "
import re, sys
from pathlib import Path
log = Path('$LOG').read_text()
m_solves = re.search(r'solved_by_model.*?(\d+).*?fallback.*?(\d+)', log)
m_total = re.search(r'total_moves.*?([\d,]+)', log)
m_buckets = re.findall(r'(\d+-\d+)\s+\d+\s+(\d+)/(\d+)\s+([\d.\-]+)', log)
if not m_solves or not m_total:
    print('Could not parse log; manual review required.')
    sys.exit(1)
solved = int(m_solves.group(1))
fallback = int(m_solves.group(2))
total_moves = int(m_total.group(1).replace(',', ''))
n_total = solved + fallback
# Mean model_avg over solved-by-model pids only:
solved_avgs = [float(b[3]) for b in m_buckets if int(b[1]) > 0 and b[3] != '-']
mean_avg = sum(solved_avgs) / max(len(solved_avgs), 1) if solved_avgs else 999
gate1 = solved >= 50 + 3        # >=+3 vs m05's 50
gate2 = mean_avg <= 84.9         # 0.95 * m05's 89.4
print(f'  solves: {solved}/{n_total} (m05: 50/51, +{solved - 50} delta)')
print(f'  mean model_avg: {mean_avg:.2f} (m05: 89.4, target <=84.9)')
print(f'  total_moves: {total_moves:,}')
print(f'  gate 1 (>=+3 solves): {\"PASS\" if gate1 else \"FAIL\"}')
print(f'  gate 2 (mean <=84.9): {\"PASS\" if gate2 else \"FAIL\"}')
print()
if gate1 and gate2:
    print('  >>> ACCEPTANCE GATE PASSED. Promote to production candidate.')
elif solved == 0:
    print('  >>> CATASTROPHIC FAIL. V is broken. Investigate recipe.')
else:
    print('  >>> NOT ACCEPTABLE. Either tie or regression vs m05.')
"
```

## When to use

After every:
- Bellman retrain or recipe ablation (m17, m22, m26..., m34, ...)
- New V architecture trial (m18 transformer, m26b wider, ...)
- New training data mixin (BFS-d6 anchoring, solver-trace mining, ...)
- New optimizer (Muon experiments, ...)

Don't use for:
- Inference-side mechanisms (sym-ensemble, qshort, beam tuning) — those
  need full-1001 to capture per-pid min-merge effects.
- Q-shortlister recall checks — different metric, see
  `09_eval_q_recall.py` (target ≥99% at chosen alpha).

## Common results to expect

| Model | Strat-5 result | Notes |
|---|---|---|
| m05 (production) | 50/51 / 89.4 | Reference baseline |
| m17 / m22 / m26 / m26b / m27 / m28 | 51/51 / 91-97 | Cluster ceiling — same recipe family |
| m29 | 51/51 / 88.98 | Only sub-89 result; n_back=4 walks |
| m31 (rotation aug) | 50/51 / 95.76 | Regresses — orbit dilution at 6M |
| m34 (soft-min + Polyak + solver-trace) | **0/51** | Catastrophic — recipe broke V |

If a new model lands at **0/51**, the recipe is fundamentally broken
(not just a tuning issue) — usually a label-scale mismatch, target-net
desync, or loss-formulation bug. Don't retrain; ablate.
