#!/bin/bash
# m45 (49.6M large V) full pipeline on GCP: pretrain -> Bellman -> strat-5.
# Run on GCP L4 once m42 lower-q25 finishes there.
# Run from ~/cayley/ on the GCP instance (paths are remote-style absolute).

set -uo pipefail
ROOT=/home/and-l/cayley
cd $ROOT

mkdir -p megaminx/models/m45/pretrain megaminx/models/m45/bellman

echo "$(date) === m45 step 1: pretrain (4000 ep, ~10-12h) ==="
PYTHONUTF8=1 python3 -u megaminx/scripts/02_train.py \
    --config megaminx/configs/m45_pretrain.yaml \
    --output megaminx/models/m45/pretrain \
    2>&1 | tee megaminx/models/m45/pretrain_training.log

PRETRAIN_CKPT=$(ls -1 megaminx/models/m45/pretrain/epoch_*.pt 2>/dev/null | sort | tail -1 || true)
[[ -z "$PRETRAIN_CKPT" ]] && { echo "ERROR: no pretrain ckpt"; exit 1; }
echo "$(date) pretrain done: $PRETRAIN_CKPT"

sed "s|^  warmstart_path: .*|  warmstart_path: \"$PRETRAIN_CKPT\"|" \
    megaminx/configs/m45_bellman.yaml > megaminx/configs/m45_bellman_run.yaml

echo ""
echo "$(date) === m45 step 2: Bellman (500 ep, ~2h) ==="
PYTHONUTF8=1 python3 -u megaminx/scripts/05_bellman_refine.py \
    --config megaminx/configs/m45_bellman_run.yaml \
    --output megaminx/models/m45/bellman \
    2>&1 | tee megaminx/models/m45/bellman_training.log

BELLMAN_CKPT=$(ls -1 megaminx/models/m45/bellman/epoch_*.pt 2>/dev/null | sort | tail -1 || true)
[[ -z "$BELLMAN_CKPT" ]] && { echo "ERROR: no Bellman ckpt"; exit 1; }
echo "$(date) Bellman done: $BELLMAN_CKPT"

echo ""
echo "$(date) === m45 step 3: strat-5 eval (~100 min) ==="
PYTHONUTF8=1 python3 -u megaminx/scripts/03_solve.py \
    --checkpoint "$BELLMAN_CKPT" \
    --out megaminx/submissions/m45_strat5.csv \
    --beams 65536 --max-steps 150 \
    --stratified 5 --strat-seed 0 \
    --bf16 \
    2>&1 | tee megaminx/submissions/m45_strat5.log

echo ""
echo "$(date) === m45 chain done ==="
