#!/bin/bash
# Post-training pipeline: launch 1001-pid eval, then TB + AZ full training.
# Designed to run after m_dd_v0_full training finishes (best.pt available).

set -e
PROJECT_ROOT=$(dirname $(dirname $(readlink -f "$0")))
cd "$PROJECT_ROOT/.."
PYTHON=".venv/Scripts/python.exe"
M_DD_BEST="megaminx/models/m_dd_v0_full/best.pt"

if [ ! -f "$M_DD_BEST" ]; then
    echo "ERROR: $M_DD_BEST not found. Run training first."
    exit 1
fi

echo "=== 1001-pid eval of m_dd_v0_full/best.pt ==="
nohup $PYTHON megaminx/scripts/03_solve.py \
    --checkpoint "$M_DD_BEST" \
    --out megaminx/submissions/m_dd_v0_full_1001.csv \
    --beams 65536 --max-steps 120 \
    --bf16 \
    --resume \
    > megaminx/models/m_dd_v0_full_eval.log 2>&1 &
EVAL_PID=$!
echo "Eval started (PID $EVAL_PID); log: megaminx/models/m_dd_v0_full_eval.log"

# Sleep to let eval warm up (compile + load PDBs etc.) before adding GPU contention
sleep 60

echo "=== Full TB training ==="
nohup $PYTHON megaminx/scripts/62_train_tb.py \
    --output megaminx/models/m_tb_v0_full \
    --epochs 200 \
    --n-trajs-per-epoch 16384 \
    --batch-size 256 \
    --traj-len 30 \
    > megaminx/models/m_tb_v0_full.log 2>&1 &
TB_PID=$!
echo "TB started (PID $TB_PID); log: megaminx/models/m_tb_v0_full.log"

# Wait for TB to finish before AZ (avoid 3-way GPU contention)
wait $TB_PID
echo "TB done"

echo "=== Full AZ training ==="
nohup $PYTHON megaminx/scripts/63_train_alphazero.py \
    --v-checkpoint "$M_DD_BEST" \
    --output megaminx/models/m_az_v0_full \
    --synthetic-walks \
    --n-self-play 10000 \
    --max-walk-len 30 \
    --epochs 100 \
    --batch-size 512 \
    > megaminx/models/m_az_v0_full.log 2>&1 &
AZ_PID=$!
echo "AZ started (PID $AZ_PID); log: megaminx/models/m_az_v0_full.log"

wait $EVAL_PID
echo "Eval done"
wait $AZ_PID
echo "AZ done"

echo "All done. Best m_dd checkpoint: $M_DD_BEST"
echo "Submission: megaminx/submissions/m_dd_v0_full_1001.csv"
