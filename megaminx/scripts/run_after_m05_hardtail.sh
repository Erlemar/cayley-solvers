#!/bin/bash
# Run T3.3 oracle Q-head recall eval + T1.3 TTT smoke after m05 hardtail finishes.

set -uo pipefail
PY=/c/Users/and-l/cayley/.venv/Scripts/python.exe

# Wait for m05 hardtail completion marker
echo "$(date) waiting for m05 hardtail eval ..."
until grep -qE 'stats:.*solved_by_model' megaminx/submissions/m05_hardtail_strat5.log 2>/dev/null; do
    sleep 30
done
echo "$(date) m05 hardtail done"
tail -10 megaminx/submissions/m05_hardtail_strat5.log

# T3.3 recall test - in-distribution (k_max=5, where m35 was trained)
echo ""
echo "=== T3.3 recall at k_max=5 (m35's training distribution) ==="
PYTHONUTF8=1 $PY -u megaminx/scripts/09_eval_q_recall.py \
    --teacher megaminx/models/m05_bellman_warm/epoch_0499.pt \
    --student megaminx/models/m35_oracle_q/epoch_0499.pt \
    --n-parents 2048 --k-max 5 \
    --alpha 1,2,3,4,6 \
    --effective-B 1024 \
    --bf16 \
    2>&1 | tee megaminx/submissions/t33_recall_k5.log

# T3.3 recall test - production distribution (k_max=80, realistic beam states)
echo ""
echo "=== T3.3 recall at k_max=80 (production beam states) ==="
PYTHONUTF8=1 $PY -u megaminx/scripts/09_eval_q_recall.py \
    --teacher megaminx/models/m05_bellman_warm/epoch_0499.pt \
    --student megaminx/models/m35_oracle_q/epoch_0499.pt \
    --n-parents 2048 --k-max 80 \
    --alpha 1,2,3,4,6 \
    --effective-B 1024 \
    --bf16 \
    2>&1 | tee megaminx/submissions/t33_recall_k80.log

# T1.3 TTT smoke test - 5 hard-tail pids only
echo ""
echo "=== T1.3 TTT smoke (5 hard-tail pids) ==="
PYTHONUTF8=1 $PY -u megaminx/scripts/37_ttt_solve.py \
    --checkpoint megaminx/models/m05_bellman_warm/epoch_0499.pt \
    --out megaminx/submissions/m05_ttt_smoke.csv \
    --beams 65536 --max-steps 150 \
    --pids 800,850,900,950,1000 \
    --bf16 \
    2>&1 | tee megaminx/submissions/m05_ttt_smoke.log

echo ""
echo "$(date) all queued evals done"
