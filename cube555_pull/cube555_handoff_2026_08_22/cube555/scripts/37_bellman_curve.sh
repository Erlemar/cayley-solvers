#!/usr/bin/env bash
# Complete the Bellman coverage/length trade curve.
#
# Endpoints already measured on these exact 20 pids, --frames 0, 2^21, --max-steps 300:
#   pretrained  (0 refreshes)  14/20 solved, mean 196.8   <- bench/wl_pretrain.csv
#   bell40      (40 refreshes)  6/20 solved, mean 157.0   <- bench/wl_w21.csv
# Discrimination at remaining 61-70 falls monotonically along the same axis
# (0.246 / 0.228 / 0.225 / 0.217), so the two intermediates should land between the
# endpoints on coverage AND length. If one dominates for our objective, it is the
# operating point -- and it also tells a training run WHERE on this axis to aim.
#
# Only the checkpoint varies. Same pids, same frame, same width, same steps.
set -u
cd /home/artgor/cayley
PY=.venv/bin/python
B=cube555/bench
PIDS=$(cat /tmp/sweep_pids.txt)
COMMON="--beams 2097152 --max-steps 300 --bf16 --compile --history-depth 4 \
        --no-backtrack --endgame-depth 5 --internal-batch-size 262144 --frames 0"

echo "=== bell12  (q555_a_bell/bellman.pt, 6000 steps = 12 refreshes) ==="
date '+%H:%M:%S'
$PY cube555/scripts/30_solve.py --checkpoint cube555/models/q555_a_bell/bellman.pt \
    --pids "$PIDS" $COMMON --out $B/wl_bell12.csv

echo; echo "=== bell20  (q555_a_bell2/bellman_010000.pt, 10000 steps = 20 refreshes) ==="
date '+%H:%M:%S'
$PY cube555/scripts/30_solve.py --checkpoint cube555/models/q555_a_bell2/bellman_010000.pt \
    --pids "$PIDS" $COMMON --out $B/wl_bell20.csv

echo; echo "CURVE_DONE"; date '+%H:%M:%S'
