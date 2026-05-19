#!/bin/bash
# Local queue after m36 + h18 macros land:
#   1. m39a pretrain (4000 ep walk-depth, ~3h on 4090)
#   2. m39a Bellman warmstart (500 ep, ~30 min)
#   3. m39a strat-5 eval (51 pids, ~80 min)
#
# Total wall: ~5h.
#
# Run only when local 4090 is free (no m36 or h18 in flight).

set -euo pipefail
PY=/c/Users/and-l/cayley/.venv/Scripts/python.exe

mkdir -p megaminx/models/m39a/pretrain megaminx/models/m39a/bellman

echo "=== m39a step 1: pretrain (4000 ep) ==="
PYTHONUTF8=1 $PY -u megaminx/scripts/02_train.py \
    --config megaminx/configs/m39a_pretrain.yaml \
    --output megaminx/models/m39a/pretrain \
    2>&1 | tee megaminx/models/m39a/pretrain_training.log

# Discover the latest pretrain checkpoint
PRETRAIN_CKPT=$(ls -1 megaminx/models/m39a/pretrain/epoch_*.pt 2>/dev/null | sort | tail -1 || true)
if [[ -z "$PRETRAIN_CKPT" ]]; then
    echo "ERROR: no pretrain checkpoint found"
    exit 1
fi
echo "pretrain done: $PRETRAIN_CKPT"

# Patch Bellman config with the actual checkpoint path
sed "s|^  warmstart_path: .*|  warmstart_path: \"$PRETRAIN_CKPT\"|" \
    megaminx/configs/m39a_bellman.yaml > megaminx/configs/m39a_bellman_run.yaml

echo "=== m39a step 2: Bellman (500 ep) ==="
PYTHONUTF8=1 $PY -u megaminx/scripts/05_bellman_refine.py \
    --config megaminx/configs/m39a_bellman_run.yaml \
    --output megaminx/models/m39a/bellman \
    2>&1 | tee megaminx/models/m39a/bellman_training.log

BELLMAN_CKPT=$(ls -1 megaminx/models/m39a/bellman/epoch_*.pt 2>/dev/null | sort | tail -1 || true)
if [[ -z "$BELLMAN_CKPT" ]]; then
    echo "ERROR: no Bellman checkpoint found"
    exit 1
fi
echo "Bellman done: $BELLMAN_CKPT"

echo "=== m39a step 3: strat-5 eval ==="
PYTHONUTF8=1 $PY -u megaminx/scripts/03_solve.py \
    --checkpoint "$BELLMAN_CKPT" \
    --out megaminx/submissions/m39a_strat5.csv \
    --beams 65536 --max-steps 150 \
    --stratified 5 --strat-seed 0 \
    --bf16 \
    2>&1 | tee megaminx/submissions/m39a_strat5.log

echo ""
echo "=== m39a queue complete ==="
echo "Compare to m05 baseline: 50/51 / mean 89.4"
echo "Acceptance gate: >= +3 solves AND mean <= 84.9"
