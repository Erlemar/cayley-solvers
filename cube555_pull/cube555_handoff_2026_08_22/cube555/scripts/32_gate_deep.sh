#!/bin/bash
# Matched three-way gate on FULLY MIXED pids: control vs Bellman vs blend.
# Same pids, same width, same max-steps, same history depth, one invocation (rule 5).
set -u
cd /home/artgor/cayley
PY=.venv/bin/python
PIDS=354,714,1014
COMMON="--beams 4194304 --max-steps 300 --bf16 --compile --history-depth 4 --internal-batch-size 262144"
A=cube555/models/q555_a/epoch_011718.pt
B=cube555/models/q555_b/epoch_011718.pt
BELL=cube555/models/q555_a_bell/bellman.pt

echo "### control: arm A alone"
$PY cube555/scripts/30_solve.py --checkpoint $A --pids $PIDS $COMMON \
    --out cube555/bench/deep_ctrl.csv 2>&1 | grep --line-buffered -E "pid=|NO SOL|beam-only|wall"
echo "### bellman-refined A"
$PY cube555/scripts/30_solve.py --checkpoint $BELL --pids $PIDS $COMMON \
    --out cube555/bench/deep_bell.csv 2>&1 | grep --line-buffered -E "pid=|NO SOL|beam-only|wall"
echo "### blend A+B"
$PY cube555/scripts/30_solve.py --checkpoint $A --blend $B --pids $PIDS $COMMON \
    --out cube555/bench/deep_blend.csv 2>&1 | grep --line-buffered -E "pid=|NO SOL|beam-only|wall"
echo GATE_DEEP_DONE
