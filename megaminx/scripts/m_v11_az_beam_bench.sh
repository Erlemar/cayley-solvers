#!/bin/bash
# Beam bench Stage 2 (m_v11_az) ckpts using 65_bench_dual_head.py.
# Stage 2 V is wrapped in ResMLPGFlowNet (dual-head); 61_eval_v_at_solved.py
# misloads it (V head is value_head.*, not head.*) → garbage V-cal results.
# This script uses the proper dual-head adapter.
#
# Bench 3 ckpts (ep 11, 19, 24) on the same 10-pid set as Stage 1b for direct comparison.
# Reference: Stage 1b ep 199 = 9/10, total 910, avg 101.1.

set -u
cd /home/and-l/cayley

PIDS="0,100,200,300,500,600,700,800,900,950"
LOG=/home/and-l/cayley/megaminx/models/m_v11_az_beam_bench.log

echo "[start $(date -Is)]" > "$LOG"
echo "  pids: $PIDS" >> "$LOG"
echo "  beam: 65536, max-steps: 120, --bf16, mode=az" >> "$LOG"
echo "  reference: Stage 1b ep 199 = 9/10, total 910, avg 101.1" >> "$LOG"
echo "" >> "$LOG"

for EPOCH in 11 19 24; do
    EPSTR=$(printf "%04d" $EPOCH)
    CKPT="megaminx/models/m_v11_az/epoch_${EPSTR}.pt"
    OUT="megaminx/submissions/m_v11_az_bench_ep${EPOCH}.csv"

    echo "=========================================================================" >> "$LOG"
    echo "=== ep $EPOCH start $(date -Is) ===" >> "$LOG"
    echo "=========================================================================" >> "$LOG"

    python3 megaminx/scripts/65_bench_dual_head.py \
        --gflow-checkpoint "$CKPT" \
        --mode az \
        --pids "$PIDS" \
        --out "$OUT" \
        --beam 65536 --max-steps 120 \
        --bf16 \
        >> "$LOG" 2>&1

    echo "" >> "$LOG"
done

echo "[done $(date -Is)]" >> "$LOG"
echo "done" > /tmp/mv11_az_beam_done
