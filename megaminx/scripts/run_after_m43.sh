#!/bin/bash
# After m43 training completes, run:
#   1. m43 strat-5 eval (full)
#   2. T1.6 SA A/B test: curated phase2 macros vs brute-force, on top-50 of 79,522
set -uo pipefail
PY=/c/Users/and-l/cayley/.venv/Scripts/python.exe

# Wait for training completion
echo "$(date) waiting for m43 training to finish ..."
until grep -q "final loss:" megaminx/models/m43_solver_trace_mixin_training.log 2>/dev/null; do
    sleep 30
done
echo "$(date) m43 training done"

# Step 1: m43 strat-5 eval
echo ""
echo "=== m43 strat-5 eval ==="
PYTHONUTF8=1 $PY -u megaminx/scripts/03_solve.py \
    --checkpoint megaminx/models/m43_solver_trace_mixin/epoch_0499.pt \
    --out megaminx/submissions/m43_strat5.csv \
    --beams 65536 --max-steps 150 \
    --stratified 5 --strat-seed 0 \
    --bf16 \
    > megaminx/submissions/m43_strat5.log 2>&1
echo "$(date) m43 strat-5 done"
tail -20 megaminx/submissions/m43_strat5.log

# Step 2: A/B macro test — top-50 longest pids of 79,522 (or fall back to merge_tpu_v19a_plus_t16v2)
# We use --pids derived from current best submission's longest paths.
# Since 79,522 final isn't in the local repo, use 79,946 base for pid selection (close enough).
BASE_CSV=megaminx/submissions/merge_tpu_v19a_plus_t16v2.csv
TOP_PIDS=$($PY -c "
import csv
with open('$BASE_CSV') as f:
    rows=[(int(r['initial_state_id']), len(r['path'].split('.'))) for r in csv.DictReader(f)]
rows.sort(key=lambda x: -x[1])
print(','.join(str(p) for p,_ in rows[:50]))
")
echo ""
echo "=== A/B test: top-50 longest pids ==="
echo "TOP_PIDS=$TOP_PIDS"

# Run A: curated phase2 macros
echo ""
echo "--- A: curated phase2 macros ---"
PYTHONUTF8=1 $PY -u megaminx/scripts/27_path_sa.py \
    --base $BASE_CSV \
    --out megaminx/submissions/t16_curated_phase2_top50.csv \
    --checkpoint megaminx/models/m05_bellman_warm/epoch_0499.pt \
    --bfs-table megaminx/data/bfs_bytes_d6.pkl \
    --commutator-table megaminx/data/curated_table_phase2.pkl \
    --pids $TOP_PIDS \
    --n-iter 30 --beam 131072 --K-min 15 --K-max 60 \
    --mode hc --w-swap 0 --w-tail 2 --w-macro 1 --max-macro-len 25 \
    --bf16 --verbose \
    > megaminx/submissions/t16_curated_phase2_top50.log 2>&1
echo "$(date) curated A done"

# Run B: brute-force (existing) macros — same pids, same hyperparams
echo ""
echo "--- B: brute-force macros ---"
PYTHONUTF8=1 $PY -u megaminx/scripts/27_path_sa.py \
    --base $BASE_CSV \
    --out megaminx/submissions/t16_bruteforce_top50.csv \
    --checkpoint megaminx/models/m05_bellman_warm/epoch_0499.pt \
    --bfs-table megaminx/data/bfs_bytes_d6.pkl \
    --commutator-table megaminx/data/commutator_table.pkl \
    --pids $TOP_PIDS \
    --n-iter 30 --beam 131072 --K-min 15 --K-max 60 \
    --mode hc --w-swap 0 --w-tail 2 --w-macro 1 --max-macro-len 25 \
    --bf16 --verbose \
    > megaminx/submissions/t16_bruteforce_top50.log 2>&1
echo "$(date) brute-force B done"

# Compare
echo ""
echo "=== A vs B comparison ==="
$PY -c "
import re
for tag, path in [('curated', 'megaminx/submissions/t16_curated_phase2_top50.log'),
                  ('bruteforce', 'megaminx/submissions/t16_bruteforce_top50.log')]:
    text = open(path).read()
    saved = re.search(r'total savings: (\d+) moves', text)
    macro_acc = re.search(r'macro_insert accept rate: ([\d.]+)% \((\d+)/(\d+)\)', text)
    tail_acc = re.search(r'tail_resolve accept rate: ([\d.]+)% \((\d+)/(\d+)\)', text)
    n_imp = re.search(r'improved: (\d+)/(\d+) pids', text)
    print(f'  {tag}:')
    print(f'    improved: {n_imp.group(1)}/{n_imp.group(2)}' if n_imp else '    improved: ?')
    print(f'    total saved: {saved.group(1) if saved else \"?\"} moves')
    print(f'    macro_insert: {macro_acc.group(1) if macro_acc else \"?\"}% ({macro_acc.group(2) if macro_acc else \"?\"}/{macro_acc.group(3) if macro_acc else \"?\"})')
    print(f'    tail_resolve: {tail_acc.group(1) if tail_acc else \"?\"}% ({tail_acc.group(2) if tail_acc else \"?\"}/{tail_acc.group(3) if tail_acc else \"?\"})')
"
echo ""
echo "$(date) all queued tasks done"
