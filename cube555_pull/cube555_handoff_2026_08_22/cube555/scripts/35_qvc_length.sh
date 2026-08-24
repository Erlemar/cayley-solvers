#!/usr/bin/env bash
# qv-consistency as a LENGTH lever, matched against the width sweep's 2^21 arm.
#
# qvc was rejected on 555 under a COVERAGE objective: it scored 157/155 vs a 187/179
# control (-15% length) but lost 1 of 3 pids, and a lost pid falls back to a ~960-move
# baseline. Under a LENGTH objective, and with a min-merge that keeps the existing lam=0
# path for anything qvc drops, the downside is zero and the upside is ~25 moves/pid.
# That earlier result is n=2. This is the matched test.
#
# CONTROL: cube555/bench/wl_w21.csv -- same checkpoint, same 2^21 width, same --frames 0,
# same --max-steps 300, produced by 34_width_length.sh. Only --qv-consistency differs.
#
# No --fallback on purpose: a pid missing from the output means qvc FAILED on it, which is
# exactly what we need to count. --fallback would write the baseline path and silently
# pollute the length comparison with ~500-move rows.
set -u
cd /home/artgor/cayley
PY=.venv/bin/python
CK=cube555/models/q555_a_bell2/bellman_020000.pt
B=cube555/bench

# Wait for the width sweep. Poll the UNIT STATE, never the process table -- a pgrep -f
# from this script would match this script's own command line and spin forever.
echo "waiting for cube555-wl.service ..."; date '+%H:%M:%S'
while [ "$(systemctl --user is-active cube555-wl.service)" = "active" ]; do sleep 60; done
echo "width sweep is $(systemctl --user is-active cube555-wl.service); proceeding"; date '+%H:%M:%S'

if [ ! -s "$B/wl_w21.csv" ]; then
  echo "!! $B/wl_w21.csv missing or empty -- no control exists, refusing to run treatment"
  exit 1
fi

# Pair to exactly what the control solved.
PAIRED=$($PY - <<'PYEOF'
import csv
rows=list(csv.DictReader(open("cube555/bench/wl_w21.csv",encoding="utf-8")))
print(",".join(r["initial_state_id"] for r in rows))
PYEOF
)
echo "paired set ($(echo $PAIRED | tr ',' '\n' | wc -l) pids): $PAIRED"

echo; echo "=== TREATMENT: qv-consistency 0.30, everything else identical to the control ==="
date '+%H:%M:%S'
$PY cube555/scripts/30_solve.py --checkpoint $CK --pids "$PAIRED" \
    --beams 2097152 --max-steps 300 --bf16 --compile --history-depth 4 --no-backtrack \
    --endgame-depth 5 --internal-batch-size 262144 --frames 0 \
    --qv-consistency 0.30 --out $B/wl_qvc030.csv

echo; echo "QVC_DONE"; date '+%H:%M:%S'
