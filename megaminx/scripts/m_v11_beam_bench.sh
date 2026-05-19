#!/bin/bash
# Bench 4 candidate Stage 1b ckpts on a 10-pid set to pick the Stage 2 warmstart.
# Run in tmux on cayley-gpu: tmux new-session -d -s mv11_beam 'bash megaminx/scripts/m_v11_beam_bench.sh'
#
# 10 pids span all difficulty buckets. Beam 65536, max-steps 120, --no-pdb (per Rule 12 bench convention).
# Baseline: AZ v4 ep24 on its 10-pid set = 10/10, 874 total, mean 87.4.

set -u
cd /home/and-l/cayley

PIDS="0,100,200,300,500,600,700,800,900,950"
LOG=/home/and-l/cayley/megaminx/models/m_v11_beam_bench.log

echo "[start $(date -Is)]" > "$LOG"
echo "  pids: $PIDS" >> "$LOG"
echo "  beam: 65536, max-steps: 120, --no-pdb --bf16" >> "$LOG"
echo "  baseline: AZ v4 ep24 = 10/10, 874 total, mean 87.4" >> "$LOG"
echo "" >> "$LOG"

for EPOCH in 99 124 149 199; do
    EPSTR=$(printf "%04d" $EPOCH)
    CKPT="megaminx/models/m_v11_bellman/epoch_${EPSTR}.pt"
    OUT="megaminx/submissions/m_v11_bench_ep${EPOCH}.csv"

    echo "=========================================================================" >> "$LOG"
    echo "=== ep $EPOCH start $(date -Is) ===" >> "$LOG"
    echo "=========================================================================" >> "$LOG"

    python3 megaminx/scripts/58_corner_pdb_beam.py \
        --v-checkpoint "$CKPT" \
        --pids "$PIDS" \
        --out "$OUT" \
        --beam 65536 --max-steps 120 \
        --no-pdb --bf16 \
        >> "$LOG" 2>&1

    echo "" >> "$LOG"
done

echo "[done $(date -Is)]" >> "$LOG"
echo "done" > /tmp/mv11_beam_done
