#!/usr/bin/env bash
# Map the Bellman curve BELOW 2k, where the optimum must be.
#
# Measured so far on 24 disjoint pids (--frames 0, 2^21, --max-steps 300):
#     2k  18/24  130.1     4k  9/24 141.4     6k 11/24 148.7     8k 9/24 143.9    20k 10/24 174.5
# Monotone decreasing from 20k down to 2k, and 2k is the LOWEST point measured -- so the
# curve has not turned yet. The pretrained parent (0 steps) scored 14/20 mean 196.8 on a
# different pid set, i.e. clearly worse. A peak therefore exists somewhere in 0 < n < 2000
# and nobody has looked there.
#
# All four Bellman points come from ONE training run so the curve is internally consistent;
# mixing runs would blur it, because two nominally identical runs differ by ~10.5 moves per
# pid (measured: sw_6k vs cf_bell12 differ on 8/8 overlapping pids). The 2000-step endpoint
# of this run is also a third replicate against sw_2k, which extends the noise estimate.
# The pretrained arm pins the left endpoint on THESE pids for the first time.
set -u
cd /home/artgor/cayley
PY=.venv/bin/python
OUT=cube555/models/q555_a_sub
B=cube555/bench
PIDS=$(cat /tmp/confirm_pids.txt)
COMMON="--beams 2097152 --max-steps 300 --bf16 --compile --history-depth 4 \
        --no-backtrack --endgame-depth 5 --internal-batch-size 262144 --frames 0"

echo "########## TRAIN 2000 steps, saving every 250"; date '+%H:%M:%S'
$PY cube555/scripts/22_bellman.py \
    --init cube555/models/q555_a/epoch_011718.pt \
    --output $OUT --steps 2000 --save-every 250
echo; ls -la $OUT

# Ascending, so an early stop still yields a usable left-hand curve.
for SPEC in "bellman_000250.pt 250" "bellman_000500.pt 500" "bellman_001000.pt 1000" "bellman.pt 2000"; do
  set -- $SPEC
  CK=$OUT/$1; TAG=$2
  if [ ! -f "$CK" ]; then echo "!! MISSING $CK -- skipping $TAG"; continue; fi
  echo; echo "########## BENCH $TAG ($1)"; date '+%H:%M:%S'
  $PY cube555/scripts/30_solve.py --checkpoint "$CK" --pids "$PIDS" $COMMON --out $B/sb_$TAG.csv
done

echo; echo "########## BENCH pretrained (0 steps) -- left endpoint on THESE pids"; date '+%H:%M:%S'
$PY cube555/scripts/30_solve.py --checkpoint cube555/models/q555_a/epoch_011718.pt \
    --pids "$PIDS" $COMMON --out $B/sb_0.csv

echo; echo "SUBSWEEP_DONE"; date '+%H:%M:%S'
