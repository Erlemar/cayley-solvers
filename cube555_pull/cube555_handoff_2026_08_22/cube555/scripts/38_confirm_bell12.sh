#!/usr/bin/env bash
# Confirmation: bell12 vs bell40 on a DISJOINT, UNBIASED pid set.
#
# The first 20-pid set was drawn from pids bell40 had already solved during the campaign,
# which biases toward bell40 -- and bell40 still lost (6/20 vs 12/20, 157.0 vs 135.2).
# This set is sampled uniformly from all 705 fully-mixed pids (baseline >= 250) that were
# NOT in the first set, with no filter on solve history: 20 are currently beam-solved and
# 4 have never been solved by anything, so coverage is measured honestly rather than on a
# pool pre-screened for solvability.
#
# Selecting the best of four checkpoints on one 20-pid set is exactly the setup where a
# best-of-five spot-check on this project later came back statistically tied. Same pids,
# same frame, same width, same steps; only the checkpoint varies.
set -u
cd /home/artgor/cayley
PY=.venv/bin/python
B=cube555/bench
PIDS=$(cat /tmp/confirm_pids.txt)
COMMON="--beams 2097152 --max-steps 300 --bf16 --compile --history-depth 4 \
        --no-backtrack --endgame-depth 5 --internal-batch-size 262144 --frames 0"

echo "=== CONFIRM bell12 (6k steps) ==="; date '+%H:%M:%S'
$PY cube555/scripts/30_solve.py --checkpoint cube555/models/q555_a_bell/bellman.pt \
    --pids "$PIDS" $COMMON --out $B/cf_bell12.csv

echo; echo "=== CONFIRM bell40 (20k steps, DEPLOYED) ==="; date '+%H:%M:%S'
$PY cube555/scripts/30_solve.py --checkpoint cube555/models/q555_a_bell2/bellman_020000.pt \
    --pids "$PIDS" $COMMON --out $B/cf_bell40.csv

echo; echo "CONFIRM_DONE"; date '+%H:%M:%S'
