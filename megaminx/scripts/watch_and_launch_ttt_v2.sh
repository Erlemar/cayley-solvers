#!/bin/bash
# Watcher: polls for m44 chain to finish, then launches T1.3 v2 (self-distill TTT)
# on hard-tail strat (16 pids) for cheap signal.
set -uo pipefail
PY=/c/Users/and-l/cayley/.venv/Scripts/python.exe

echo "$(date) waiting for m44 chain to finish..."
until grep -q "m44 chain done" megaminx/models/m44_chain.log 2>/dev/null; do
    sleep 60
done
echo "$(date) m44 done. Launching T1.3 v2 self-distill TTT on hard-tail strat..."

PYTHONUTF8=1 $PY -u megaminx/scripts/40_ttt_self_distill.py \
    --checkpoint megaminx/models/m05_bellman_warm/epoch_0499.pt \
    --out megaminx/submissions/m05_ttt_v2_hardtail.csv \
    --beams 65536 --max-steps 150 \
    --quick-beam 8192 --quick-max-steps 60 \
    --stratified 5 --strat-seed 0 --strat-buckets 7,8,9,10 \
    > megaminx/submissions/m05_ttt_v2_hardtail.log 2>&1
echo "$(date) T1.3 v2 hard-tail done"
