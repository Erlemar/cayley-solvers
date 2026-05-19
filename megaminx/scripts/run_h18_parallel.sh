#!/bin/bash
# Parallel h18 macros T1.6 — 4 workers across 100 pids on a single GPU.
# Each worker writes its own output CSV; merge step overlays improvements onto base.

set -uo pipefail

PY=/c/Users/and-l/cayley/.venv/Scripts/python.exe
BASE=megaminx/submissions/merge_tpuv16_t22_t16.csv
OUT=megaminx/submissions/t16_h18_top100.csv
WORK=megaminx/submissions/t16_h18_workers
mkdir -p $WORK

PIDS_ALL="212,408,615,236,275,423,516,521,558,583,689,768,799,118,184,202,230,235,306,314,321,349,353,367,378,380,398,407,411,448,452,476,494,534,540,574,590,603,683,688,715,726,752,753,761,775,780,789,795,801,815,860,890,920,923,948,975,987,994,113,117,120,180,203,204,210,232,234,292,296,298,300,304,319,328,344,403,414,421,432,434,458,461,463,464,481,487,506,508,511,517,520,527,528,529,535,541,555,580,593"

# Split into 4 chunks of 25 pids each.
IFS=',' read -ra A <<< "$PIDS_ALL"
W0=$(IFS=,; echo "${A[*]:0:25}")
W1=$(IFS=,; echo "${A[*]:25:25}")
W2=$(IFS=,; echo "${A[*]:50:25}")
W3=$(IFS=,; echo "${A[*]:75:25}")

echo "Worker 0: 25 pids ($W0)"
echo "Worker 1: 25 pids ($W1)"
echo "Worker 2: 25 pids ($W2)"
echo "Worker 3: 25 pids ($W3)"

run_worker() {
    local W=$1
    local PIDS=$2
    PYTHONUTF8=1 $PY -u megaminx/scripts/27_path_sa.py \
        --base $BASE \
        --out $WORK/w${W}.csv \
        --checkpoint megaminx/models/m05_bellman_warm/epoch_0499.pt \
        --bfs-table megaminx/data/bfs_bytes_d6.pkl \
        --commutator-table megaminx/data/commutator_table_h18.pkl \
        --pids "$PIDS" \
        --n-iter 30 --beam 131072 --K-min 15 --K-max 60 \
        --mode hc --w-swap 0 --w-tail 2 --w-macro 1 --max-macro-len 6 \
        --bf16 --verbose \
        > $WORK/w${W}.log 2>&1
}

T0=$(date +%s)
run_worker 0 "$W0" & PID0=$!
run_worker 1 "$W1" & PID1=$!
run_worker 2 "$W2" & PID2=$!
run_worker 3 "$W3" & PID3=$!

echo "Launched: PID0=$PID0 PID1=$PID1 PID2=$PID2 PID3=$PID3"
echo "Logs: $WORK/w{0,1,2,3}.log"

wait $PID0; RC0=$?
wait $PID1; RC1=$?
wait $PID2; RC2=$?
wait $PID3; RC3=$?

T1=$(date +%s)
echo ""
echo "All workers done in $((T1-T0)) seconds (return codes: $RC0 $RC1 $RC2 $RC3)"

# Merge: each worker's output has all 1001 pids, but only its assigned pids
# may differ from base. Take only those rows.
echo "Merging worker outputs into $OUT..."
$PY - <<'PYEOF'
import csv
from pathlib import Path

base = {}
with open("megaminx/submissions/merge_tpuv16_t22_t16.csv") as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        base[int(row[0])] = row[1]

worker_pids = {
    0: "212,408,615,236,275,423,516,521,558,583,689,768,799,118,184,202,230,235,306,314,321,349,353,367,378".split(","),
    1: "380,398,407,411,448,452,476,494,534,540,574,590,603,683,688,715,726,752,753,761,775,780,789,795,801".split(","),
    2: "815,860,890,920,923,948,975,987,994,113,117,120,180,203,204,210,232,234,292,296,298,300,304,319,328".split(","),
    3: "344,403,414,421,432,434,458,461,463,464,481,487,506,508,511,517,520,527,528,529,535,541,555,580,593".split(","),
}

improved = dict(base)
for w in range(4):
    out_path = Path(f"megaminx/submissions/t16_h18_workers/w{w}.csv")
    if not out_path.exists():
        print(f"WARN: missing worker output {out_path}")
        continue
    pids_w = set(int(p) for p in worker_pids[w] if p)
    n_changed = 0
    with open(out_path) as f:
        rdr = csv.reader(f)
        next(rdr)
        for row in rdr:
            pid = int(row[0])
            if pid in pids_w and row[1] != base[pid]:
                improved[pid] = row[1]
                n_changed += 1
    print(f"  worker {w}: {n_changed} pids improved out of {len(pids_w)}")

out_path = "megaminx/submissions/t16_h18_top100.csv"
with open(out_path, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["initial_state_id", "path"])
    for pid in sorted(improved):
        wr.writerow([pid, improved[pid]])

base_total = sum(len(p.split(".")) for p in base.values())
new_total = sum(len(p.split(".")) for p in improved.values())
print(f"merged 1001 pids; total moves {base_total:,} -> {new_total:,} ({new_total - base_total:+d})")
PYEOF

echo ""
echo "=== h18 parallel run complete ==="
