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
megaminx/scripts/09_eval_q_recall.py \
megaminx/scripts/58_corner_pdb_beam.py \
megaminx/scripts/60_train_admissible.py \
megaminx/scripts/61_eval_v_at_solved.py \
megaminx/scripts/67_build_az_dataset.py \
megaminx/scripts/71_train_az_v3.py \
megaminx/scripts/72_build_graph_features.py \
megaminx/scripts/73_build_action_relabel.py \
megaminx/scripts/75_train_gt_q.py \
megaminx/scripts/77_eval_gt_vs_baseline.py \
megaminx/scripts/78_distill_v_from_gt.py \
megaminx/scripts/79_train_az_v_aux_gt.py \
megaminx/scripts/80_bench_az_v_head.py"

# Local-only-modified cayley modules (the training-relevant ones)
CAYLEY_SRC="src/cayley/model.py \
src/cayley/training.py \
src/cayley/bellman.py \
src/cayley/search.py"

# Full megaminx package (cheap to ship — small .py files)
MEGAMINX_SRC="megaminx/src/megaminx/corner_coord.py \
megaminx/src/megaminx/decomposition.py \
megaminx/src/megaminx/edge_coord.py \
megaminx/src/megaminx/pdb_corner.py \
megaminx/src/megaminx/pdb_edge.py \
megaminx/src/megaminx/pdb_heuristic.py \
megaminx/src/megaminx/graph_transformer.py"

# GT-specific static data files (built by 72_/73_ scripts; small, idempotent)
GT_DATA="megaminx/data/graph_features.pt \
megaminx/data/action_relabel.pt"

echo "  trainers..."
gcloud compute scp $TRAINERS \
    cayley-gpu:/home/and-l/cayley/megaminx/scripts/ --zone=us-east1-b 2>&1 | tail -3

echo "  cayley src..."
gcloud compute scp $CAYLEY_SRC \
    cayley-gpu:/home/and-l/cayley/src/cayley/ --zone=us-east1-b 2>&1 | tail -3

echo "  megaminx package..."
gcloud compute scp $MEGAMINX_SRC \
    cayley-gpu:/home/and-l/cayley/megaminx/src/megaminx/ --zone=us-east1-b 2>&1 | tail -3

echo "  GT data files..."
for f in $GT_DATA; do
    gcloud compute scp "$f" \
        "cayley-gpu:/home/and-l/cayley/$f" --zone=us-east1-b 2>&1 | tail -1
done

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
- After modifying any of `src/cayley/{bellman,training,search,model}.py` locally
- After adding new files in `megaminx/src/megaminx/`
- After regenerating `megaminx/data/{graph_features,action_relabel,solver_trace_train}.pt`

## Important — file-class coverage

This sync covers code + small static tables. It does NOT auto-sync large
binary files. Sync these manually before training that needs them:

- **AZ policy dataset** (~10 MB): `megaminx/data/az_dataset_76304.pt` or
  `az_dataset_75200.pt`
- **Solver-trace data** (~15 MB): `megaminx/data/solver_trace_train.pt`
- **Frontier states** (~36 MB): `megaminx/data/frontier_states.pt`
- **BFS-d6 dataset** (~2.4 GB, big): `megaminx/data/bfs_d6_train.pt` —
  already on GCP from earlier sync; only re-push if rebuilt
- **Warmstart checkpoints** (~5-25 MB each): `m_dd_v0/epoch_0049.pt`,
  `m_az_v4_v_only_e99.pt`, `m_gt_v0_bellman/epoch_*.pt`, etc. — push the
  ones the specific run warmstarts from

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
