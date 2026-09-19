#!/bin/bash
# Iso-cost sweep: same node budget (width x steps ~ 2.1e9), three shapes.
# The question for a SOLVE-RATE goal is whether a fixed budget is better spent wide-and-
# short (less wandering per step) or narrow-and-long (more chances to stumble in).
set -u
cd /home/artgor/cayley
PY=.venv/bin/python
CK=cube555/models/q555_a_bell2/bellman_020000.pt
PIDS=354,714
COMMON="--bf16 --compile --history-depth 4 --no-backtrack --endgame-depth 5 --internal-batch-size 262144"
for CFG in "1048576 2000 w20s2000" "2097152 1000 w21s1000" "4194304 500 w22s500"; do
  set -- $CFG
  echo "### width $1  max-steps $2"
  $PY cube555/scripts/30_solve.py --checkpoint $CK --pids $PIDS \
      --beams $1 --max-steps $2 $COMMON --out cube555/bench/iso_$3.csv 2>&1 \
      | grep --line-buffered -E "pid=|NO SOL|wall" | sed 's/ | found.*//'
done
echo ISO_DONE
