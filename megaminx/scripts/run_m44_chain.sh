#!/bin/bash
# m44 (1.12M tiny V) full pipeline: pretrain -> Bellman -> strat-5.
# Launch on local 4090 once it's free.

set -uo pipefail
PY=/c/Users/and-l/cayley/.venv/Scripts/python.exe

mkdir -p megaminx/models/m44/pretrain megaminx/models/m44/bellman

echo "$(date) === m44 step 1: pretrain (4000 ep) ==="
PYTHONUTF8=1 $PY -u megaminx/scripts/02_train.py \
    --config megaminx/configs/m44_pretrain.yaml \
    --output megaminx/models/m44/pretrain \
    2>&1 | tee megaminx/models/m44/pretrain_training.log

PRETRAIN_CKPT=$(ls -1 megaminx/models/m44/pretrain/epoch_*.pt 2>/dev/null | sort | tail -1 || true)
[[ -z "$PRETRAIN_CKPT" ]] && { echo "ERROR: no pretrain ckpt"; exit 1; }
echo "$(date) pretrain done: $PRETRAIN_CKPT"

# Patch Bellman config with the actual checkpoint path.
sed "s|^  warmstart_path: .*|  warmstart_path: \"$PRETRAIN_CKPT\"|" \
    megaminx/configs/m44_bellman.yaml > megaminx/configs/m44_bellman_run.yaml

echo ""
echo "$(date) === m44 step 2: Bellman (500 ep) ==="
PYTHONUTF8=1 $PY -u megaminx/scripts/05_bellman_refine.py \
    --config megaminx/configs/m44_bellman_run.yaml \
    --output megaminx/models/m44/bellman \
    2>&1 | tee megaminx/models/m44/bellman_training.log

BELLMAN_CKPT=$(ls -1 megaminx/models/m44/bellman/epoch_*.pt 2>/dev/null | sort | tail -1 || true)
[[ -z "$BELLMAN_CKPT" ]] && { echo "ERROR: no Bellman ckpt"; exit 1; }
echo "$(date) Bellman done: $BELLMAN_CKPT"

echo ""
echo "$(date) === m44 step 3: strat-5 eval ==="
PYTHONUTF8=1 $PY -u megaminx/scripts/03_solve.py \
    --checkpoint "$BELLMAN_CKPT" \
    --out megaminx/submissions/m44_strat5.csv \
    --beams 65536 --max-steps 150 \
    --stratified 5 --strat-seed 0 \
    --bf16 \
    2>&1 | tee megaminx/submissions/m44_strat5.log

echo ""
echo "$(date) === m44 chain done ==="
echo "Compare to m05 baseline: 50/51 / mean 89.4"
echo "Acceptance gate: >= +3 solves AND mean <= 84.9"
