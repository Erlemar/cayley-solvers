---
description: Quick 10-pid beam bench for a megaminx V model checkpoint. Use during long training runs to validate beam quality without waiting for full strat-5. ~10 min on local 4090.
---

# /megaminx-bench

Fast V-quality sanity check. Runs beam-search on 10 stratified pids and
reports solve count + total moves. Designed for **per-N-epoch validation
during long V training runs** — catches the case where loss is still
falling but beam quality is regressing (e.g., m_dd_v0_full lesson,
2026-05-11).

## Usage

`/megaminx-bench <CHECKPOINT_PATH> [<NAME>]`

If `<NAME>` is omitted, derives from the checkpoint dirname and epoch.

## When to use

- **Mid-training validation**: during any V training >50 epochs, run this
  every ~50 epochs against the latest `epoch_NNNN.pt`. If `total_moves`
  rises by >5% vs the previous bench, stop training and use the earlier
  checkpoint.
- **Quick V sanity check**: after a recipe tweak (anchor weights, λ_pdb,
  PDB on/off), before committing to a full strat-5.
- **NOT for production submission decisions** — use `/megaminx-eval-v`
  (strat-5, 51 puzzles) for those.

## Execution

```bash
CKPT="${1:-megaminx/models/m_dd_v0/epoch_0049.pt}"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

# Derive name: e.g., "m_dd_v0_e0049_bench"
DIR_NAME="$(basename $(dirname $(dirname $CKPT)))"
EPOCH_NUM="$(basename $CKPT .pt | sed 's/epoch_//')"
NAME="${2:-${DIR_NAME}_e${EPOCH_NUM}_bench}"
OUT="megaminx/submissions/${NAME}.csv"
LOG="megaminx/submissions/${NAME}.log"

# 10 pids spanning the difficulty range. Avoids both trivial pids (very low)
# and the unsolvable-without-niss tail (very high). These match what we
# used during m_dd_v0 / TB / AZ bench iterations.
PIDS="0,100,200,300,400,500,600,700,800,900"

echo "=== /megaminx-bench ==="
echo "  checkpoint: $CKPT"
echo "  pids:       $PIDS"
echo "  output:     $OUT"
echo ""

PYTHONUTF8=1 .venv/Scripts/python.exe -u megaminx/scripts/03_solve.py \
    --checkpoint "$CKPT" \
    --out "$OUT" \
    --pids "$PIDS" \
    --beams 65536 --max-steps 120 \
    --bf16 \
    2>&1 | tee "$LOG"

echo ""
echo "=== Bench summary ==="
.venv/Scripts/python.exe -c "
import csv, re
from pathlib import Path
rows = list(csv.DictReader(open('$OUT')))
n_total = len(rows)
log = Path('$LOG').read_text()
m_solves = re.search(r'solved_by_model.*?(\d+).*?fallback.*?(\d+)', log)
m_total = re.search(r'total_moves.*?([\d,]+)', log)
if not m_solves or not m_total:
    print('Could not parse log; check $LOG manually.')
else:
    solved = int(m_solves.group(1))
    fallback = int(m_solves.group(2))
    total_moves = int(m_total.group(1).replace(',', ''))
    print(f'  solves: {solved}/{n_total} (fallback: {fallback})')
    print(f'  total_moves: {total_moves:,}')
    print()
    print('Compare to your previous bench at the same pids. If total_moves')
    print('rose by >5%% vs the previous epoch checkpoint, training has')
    print('started regressing — revert to the previous checkpoint.')
"
```

## Reference baselines (10-pid bench, beam 65536, max_steps 120)

| Model | Solved | Total moves | Notes |
|---|---|---|---|
| m_dd_v0 50ep | 10/10 | 871 | Canonical baseline (Bellman + anchor + PDB) |
| TB v1 500ep | 10/10 | 1016 | +16% vs baseline — dual-head trade-off |
| AZ v3 100ep | 10/10 | 943 | +8% vs baseline — joint policy+value |
| m_dd_v0_full 184ep | (regressed) | (catastrophic) | Why this command exists |

Use these as rough targets. A new V model should land within ±10% of
m_dd_v0 50ep's `total_moves`. Much worse → recipe issue. Better than 871
on this 10-pid set is a real signal; promote to `/megaminx-eval-v` for
the full strat-5 verdict.

## Why this exists

Confirmed 2026-05-11: m_dd_v0_full trained 184 epochs with smoothed-loss
early stop. Training loss kept falling (0.0724 → 0.0688) but on the GCP
full-1001 eval, the long-trained model REGRESSED catastrophically vs the
50ep checkpoint (0/20 solves for pids 60-79). The smoothed-loss min_delta
was too noisy to catch the regression; the gradient on BFS-d6 anchors
over-saturated while harder intermediate-depth distance fidelity
degraded.

**Lesson encoded as a slash command**: training loss is NOT a reliable
proxy for beam quality. Long V runs MUST be validated via beam bench,
periodically. This command runs in ~10 min on local 4090 — cheap insurance.

See CLAUDE.md rule 12 and `megaminx_gotchas.md` "V-model training >50
epochs needs per-bench validation".
