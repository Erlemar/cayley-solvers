#!/bin/bash
# Polls m_v11_bellman/ for new checkpoints and runs V-calibration on each.
# Exits when training completes (signaled by /tmp/mv11b_done existing).
#
# Records to: megaminx/models/m_v11_bench_results.log
# Run with: tmux new-session -d -s mv11_bench 'bash megaminx/scripts/m_v11_bench_loop.sh'

set -u
cd /home/and-l/cayley

LOG=/home/and-l/cayley/megaminx/models/m_v11_bench_results.log
CKPT_DIR=/home/and-l/cayley/megaminx/models/m_v11_bellman
SEEN_FILE=/tmp/m_v11_seen_ckpts
DONE_MARKER=/tmp/mv11b_done

: > "$SEEN_FILE"
echo "[bench_loop start] $(date -Is)" > "$LOG"
echo "  watching: $CKPT_DIR" >> "$LOG"
echo "  baseline: AZ v4 ep24 V(V0)=0, V(d=1)=1, V(d=6)=6 (ground truth)" >> "$LOG"
echo "" >> "$LOG"

while true; do
    for ckpt in "$CKPT_DIR"/epoch_*.pt; do
        # No matches yet — glob returns the literal pattern
        [[ -e "$ckpt" ]] || continue
        # Already benched
        grep -Fxq "$ckpt" "$SEEN_FILE" && continue

        EPOCH=$(basename "$ckpt" .pt | sed 's/epoch_0*//')
        echo "=== bench ckpt epoch_${EPOCH} ($(date -Is)) ===" >> "$LOG"
        python3 megaminx/scripts/61_eval_v_at_solved.py \
            --checkpoints "$ckpt" \
            >> "$LOG" 2>&1
        echo "" >> "$LOG"

        echo "$ckpt" >> "$SEEN_FILE"
    done

    # Exit when training signals done
    if [[ -f "$DONE_MARKER" ]]; then
        echo "[bench_loop end] $(date -Is) — training done" >> "$LOG"
        break
    fi

    sleep 60
done
