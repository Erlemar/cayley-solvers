#!/usr/bin/env bash
# Matched width -> path-length sweep.
#
# The only variable is beam width. Same pids, same frame (0 = identity, so no frame
# transform at all), same checkpoint, same max-steps, same flags. --beams ESCALATES
# inside 30_solve.py (it breaks on the first width that solves), so each width MUST be
# its own invocation -- passing a list would measure escalation, not width.
#
# max-steps is 300 in every arm, not 250: a beam finds a length-L solution at step L, so
# a cap only truncates. 300 keeps the cap non-binding for the NARROW arm, where paths are
# expected to be longest, and removes it as a confound.
#
# Arm 1 (2^21, the deployed width) doubles as the screen: arms 2 and 3 run only the pids
# it solved, so every comparison is paired.
set -u
cd /home/artgor/cayley
PY=.venv/bin/python
CK=cube555/models/q555_a_bell2/bellman_020000.pt
B=cube555/bench
COMMON="--bf16 --compile --history-depth 4 --no-backtrack --endgame-depth 5 \
        --internal-batch-size 262144 --frames 0 --max-steps 300"
PIDS=$(cat /tmp/sweep_pids.txt)

echo "=== ARM 2^21 (deployed width, also the screen) ==="
date '+%H:%M:%S'
$PY cube555/scripts/30_solve.py --checkpoint $CK --pids "$PIDS" --beams 2097152 \
    $COMMON --out $B/wl_w21.csv

# Pair the later arms to whatever arm 1 actually solved.
SOLVED=$($PY - <<'PYEOF'
import csv
try:
    rows=list(csv.DictReader(open("cube555/bench/wl_w21.csv",encoding="utf-8")))
except FileNotFoundError:
    rows=[]
print(",".join(r["initial_state_id"] for r in rows))
PYEOF
)
if [ -z "$SOLVED" ]; then
  echo "!! ARM 2^21 SOLVED NOTHING -- stopping, later arms would be unpaired"
  exit 1
fi
echo "paired set: $(echo $SOLVED | tr ',' '\n' | wc -l) pids -> $SOLVED"

echo; echo "=== ARM 2^22 (2x deployed) ==="
date '+%H:%M:%S'
$PY cube555/scripts/30_solve.py --checkpoint $CK --pids "$SOLVED" --beams 4194304 \
    $COMMON --out $B/wl_w22.csv

echo; echo "=== ARM 2^20 (half deployed) ==="
date '+%H:%M:%S'
$PY cube555/scripts/30_solve.py --checkpoint $CK --pids "$SOLVED" --beams 1048576 \
    $COMMON --out $B/wl_w20.csv

echo; echo "=== PROBE 2^23 on one pid -- does it fit in 80 GB at all? ==="
date '+%H:%M:%S'
FIRST=$(echo $SOLVED | cut -d, -f1)
$PY cube555/scripts/30_solve.py --checkpoint $CK --pids "$FIRST" --beams 8388608 \
    $COMMON --out $B/wl_w23_probe.csv || echo "2^23 FAILED (expected: OOM at ~108 GB)"

echo; echo "SWEEP_DONE"; date '+%H:%M:%S'
