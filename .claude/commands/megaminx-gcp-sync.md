# /megaminx-gcp-sync

Audit and sync the megaminx training stack on GCP cayley-gpu before launching
a training job. Avoids the multi-round ModuleNotFoundError ritual.

## Usage

`/megaminx-gcp-sync [<trainer-script-basename>]`

Optional arg: which trainer you're about to launch (e.g. `60_train_admissible.py`).
If omitted, sync the canonical set (60, 67, 68, 70, 71 trainers + bench scripts
+ bellman.py + full megaminx package).

## Why this exists

CLAUDE.md rule 17: GCP cayley-gpu is provisioned for **solving**
(`03_solve.py`), not training. Training stack has drifted local-only. A
naive `tmux new-session ... python3 60_train_admissible.py ...` will fail
because:

1. `60_train_admissible.py` may not be on GCP (added locally after the
   last sync)
2. `src/cayley/bellman.py` on GCP is missing ~600 lines (BFS-d6 mixin,
   Double Bellman, rotation aug, V0/d1 anchors)
3. `megaminx/src/megaminx/` on GCP often has 6 .py files; local has 12
   (missing `pdb_heuristic.py`, `pdb_corner.py`, `pdb_edge.py`,
   `corner_coord.py`, `edge_coord.py`, `decomposition.py`)
4. Even with `lambda_pdb=0`, the script imports `pdb_heuristic` at
   load time → ModuleNotFoundError

Confirmed 2026-05-18: four separate transfer rounds without this audit
during the m_v11 launch.

## Execution

```bash
TARGET_SCRIPT="${1:-}"

echo "=== GCP training-stack audit ==="

# 1. Check what trainers exist on GCP
echo ""
echo "--- trainer scripts on GCP ---"
gcloud compute ssh cayley-gpu --zone=us-east1-b --command="
ls ~/cayley/megaminx/scripts/ 2>/dev/null | grep -E '^(02|05|58|60|61|65|67|68|70|71)_' || echo 'no recent trainers found'
"

echo ""
echo "--- bellman.py size on GCP (expect ~1163 lines for modern) ---"
gcloud compute ssh cayley-gpu --zone=us-east1-b --command="wc -l ~/cayley/src/cayley/bellman.py 2>/dev/null"
echo "  local: $(wc -l < src/cayley/bellman.py) lines"

echo ""
echo "--- megaminx package files on GCP (expect 12) ---"
GCP_COUNT=$(gcloud compute ssh cayley-gpu --zone=us-east1-b --command="ls ~/cayley/megaminx/src/megaminx/*.py 2>/dev/null | wc -l" | tail -1)
LOCAL_COUNT=$(ls megaminx/src/megaminx/*.py 2>/dev/null | wc -l)
echo "  GCP: $GCP_COUNT, Local: $LOCAL_COUNT"

# 2. Sync the canonical training stack (idempotent — overwrites)
echo ""
echo "=== Syncing training stack ==="

# Trainer + helper scripts
TRAINERS="megaminx/scripts/02_train.py \
megaminx/scripts/58_corner_pdb_beam.py \
megaminx/scripts/60_train_admissible.py \
megaminx/scripts/61_eval_v_at_solved.py \
megaminx/scripts/67_build_az_dataset.py \
megaminx/scripts/71_train_az_v3.py"

# Local-only-modified cayley modules (the training-relevant ones)
CAYLEY_SRC="src/cayley/bellman.py"

# Full megaminx package (cheap to ship — small .py files)
MEGAMINX_SRC="megaminx/src/megaminx/corner_coord.py \
megaminx/src/megaminx/decomposition.py \
megaminx/src/megaminx/edge_coord.py \
megaminx/src/megaminx/pdb_corner.py \
megaminx/src/megaminx/pdb_edge.py \
megaminx/src/megaminx/pdb_heuristic.py"

echo "  trainers..."
gcloud compute scp $TRAINERS \
    cayley-gpu:/home/and-l/cayley/megaminx/scripts/ --zone=us-east1-b 2>&1 | tail -3

echo "  cayley src..."
gcloud compute scp $CAYLEY_SRC \
    cayley-gpu:/home/and-l/cayley/src/cayley/bellman.py --zone=us-east1-b 2>&1 | tail -2

echo "  megaminx package..."
gcloud compute scp $MEGAMINX_SRC \
    cayley-gpu:/home/and-l/cayley/megaminx/src/megaminx/ --zone=us-east1-b 2>&1 | tail -3

# 3. If a specific trainer was passed, do an import-test
if [[ -n "$TARGET_SCRIPT" ]]; then
    echo ""
    echo "=== Import-test for $TARGET_SCRIPT ==="
    gcloud compute ssh cayley-gpu --zone=us-east1-b --command="
cd ~/cayley && python3 -c \"
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'megaminx/src')
# Replicate the trainer's imports without running it
import importlib.util
spec = importlib.util.spec_from_file_location('t', 'megaminx/scripts/$TARGET_SCRIPT')
m = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(m)
    print('IMPORT OK')
except SystemExit:
    print('IMPORT OK (script tried argparse — expected)')
except Exception as e:
    print(f'IMPORT FAILED: {type(e).__name__}: {e}')
\"
"
fi

echo ""
echo "=== Sync done ==="
echo "Next step: tmux new-session -d -s <name> 'python3 megaminx/scripts/<trainer> --config ...'"
```

## When to use

- BEFORE launching any training run on GCP, regardless of which trainer
- Whenever GCP has been idle >1 week and you're about to train there
- After modifying `src/cayley/bellman.py` locally
- After adding new files in `megaminx/src/megaminx/`

## When NOT to use

- For inference / solving (`03_solve.py`) — the solving stack is kept in sync
  separately via the production-solve workflow
- After every single small edit — only when you're about to launch training
  (the sync is cheap but `gcloud ssh` round-trip is ~5s × calls)

## Reference

- CLAUDE.md rule 17 — training-vs-solving gap on GCP
- CLAUDE.md rule 18 — warmstart shape mismatch (related: two-stage scale-up needs both stages on GCP)
- `memory/megaminx_gotchas.md` — "GCP cayley-gpu has the SOLVING stack, not the TRAINING stack"
- `memory/reference_gcp_cayley_vm.md` — VM setup + gcloud scp remote-path gotcha
