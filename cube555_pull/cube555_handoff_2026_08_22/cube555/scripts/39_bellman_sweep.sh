#!/usr/bin/env bash
# Locate the Bellman optimum: 2k / 4k / 6k / 8k, trained as ONE run with periodic saves,
# then benched on the SAME 24 disjoint pids used for the bell12-vs-bell40 confirmation.
#
# Why one run rather than four: every checkpoint then shares an init, a seed, and a
# hyperparameter set, so step count is the only variable. It also gives a free control --
# the 6k checkpoint here is a direct replicate of q555_a_bell/bellman.pt. If they bench
# the same, bell12's advantage was step count; if they differ, that run's args differed
# and "6k is the optimum" was never established, only "that checkpoint is better".
#
# Known so far on these 24 pids (--frames 0, 2^21, --max-steps 300):
#     bell12 (6k)   12/24  mean 140.9      cf_bell12.csv
#     bell40 (20k)  10/24  mean 174.5      cf_bell40.csv
#     paired n=6, bell12 -28.7 moves, shorter on 5/6
#
# 22_bellman.py saves at every multiple of --save-every, and the final save is bellman.pt,
# so --steps 8000 --save-every 2000 yields exactly 2000/4000/6000 + bellman.pt at 8000.
# Training is cheap: the original 6k run took 20.5 min, so 8k is ~30 min. The bench is the
# expensive half at ~4.5 h.
set -u
cd /home/artgor/cayley
PY=.venv/bin/python
OUT=cube555/models/q555_a_sweep
B=cube555/bench
PIDS=$(cat /tmp/confirm_pids.txt)
COMMON="--beams 2097152 --max-steps 300 --bf16 --compile --history-depth 4 \
        --no-backtrack --endgame-depth 5 --internal-batch-size 262144 --frames 0"

echo "########## TRAIN: 8000 steps from the pretrained init, saving every 2000"
date '+%H:%M:%S'
$PY cube555/scripts/22_bellman.py \
    --init cube555/models/q555_a/epoch_011718.pt \
    --output $OUT --steps 8000 --save-every 2000

echo; echo "checkpoints produced:"; ls -la $OUT

for SPEC in "bellman_002000.pt 2k" "bellman_004000.pt 4k" "bellman_006000.pt 6k" "bellman.pt 8k"; do
  set -- $SPEC
  CK=$OUT/$1; TAG=$2
  if [ ! -f "$CK" ]; then echo "!! MISSING $CK -- skipping $TAG"; continue; fi
  echo; echo "########## BENCH $TAG ($1)"; date '+%H:%M:%S'
  $PY cube555/scripts/30_solve.py --checkpoint "$CK" --pids "$PIDS" $COMMON \
      --out $B/sw_$TAG.csv
done

echo; echo "SWEEP2_DONE"; date '+%H:%M:%S'
