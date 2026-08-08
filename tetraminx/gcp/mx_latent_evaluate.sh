#!/bin/bash
# Post-training evaluation for the standalone latent relational Q scorer.
# Safe to rerun after a spot interruption: path CSVs use --resume.

set -euo pipefail

ROOT=/home/and-l/cayley
RESULTS="$ROOT/tetraminx/results/mx_latent_relational"
PIDS=0,50,100,200,300,400,500,600,700,800,900,950,990,995,999

mkdir -p "$RESULTS"
cd "$ROOT"

for EPOCH in 0500 1000 1500; do
    CKPT="$ROOT/tetraminx/models/mx_latent_relational/epoch_${EPOCH}.pt"
    test -f "$CKPT"

    printf '=== %s epoch %s shared 262k decision-quality probe ===\n' \
        "$(date -u +%FT%TZ)" "$EPOCH"
    python3 -u tetraminx/scripts/52_eval_q.py \
        --q-model "$CKPT" \
        --n 262144 \
        --k-min 2 \
        --k-max 40 \
        --tilt 0.5 \
        --seed 7 \
        | tee "$RESULTS/probe_ep${EPOCH}.txt"

    printf '=== %s epoch %s Q@1M, one-frame comparison gate ===\n' \
        "$(date -u +%FT%TZ)" "$EPOCH"
    python3 -u tetraminx/scripts/30_solve.py \
        --checkpoint "$CKPT" \
        --out "$RESULTS/q1m_f1_ep${EPOCH}.csv" \
        --floor none \
        --beam 1048576 \
        --max-steps 45 \
        --sym-frames 1 \
        --history-depth 0 \
        --pids "$PIDS" \
        --chunk-size 32768 \
        --bf16 \
        --no-merge \
        --resume
done

# The proxy has disagreed with beam quality elsewhere in this project, so select
# the practical four-frame checkpoint by measured one-frame path total, not MSE.
BEST_EPOCH=
BEST_TOTAL=
for EPOCH in 0500 1000 1500; do
    read -r COUNT TOTAL < <(
        awk -F, 'NR > 1 && $2 != "" {n=gsub(/\./, ".", $2); total += n + 1; count++} END {print count + 0, total + 0}' \
            "$RESULTS/q1m_f1_ep${EPOCH}.csv"
    )
    if [ "$COUNT" -ne 15 ]; then
        printf 'epoch %s one-frame gate incomplete: %s/15 paths; excluded from selection\n' \
            "$EPOCH" "$COUNT"
        continue
    fi
    if [ -z "$BEST_EPOCH" ] || [ "$TOTAL" -lt "$BEST_TOTAL" ]; then
        BEST_EPOCH=$EPOCH
        BEST_TOTAL=$TOTAL
    fi
done

if [ -z "$BEST_EPOCH" ]; then
    printf 'no checkpoint completed all 15 one-frame paths; four-frame gate not started\n' >&2
    exit 1
fi

CKPT="$ROOT/tetraminx/models/mx_latent_relational/epoch_${BEST_EPOCH}.pt"
printf '=== %s best one-frame checkpoint: epoch %s, total %s; four-frame gate ===\n' \
    "$(date -u +%FT%TZ)" "$BEST_EPOCH" "$BEST_TOTAL"
python3 -u tetraminx/scripts/30_solve.py \
    --checkpoint "$CKPT" \
    --out "$RESULTS/q1m_f4_h1_ep${BEST_EPOCH}.csv" \
    --floor none \
    --beam 1048576 \
    --max-steps 45 \
    --sym-frames 4 \
    --history-depth 1 \
    --pids "$PIDS" \
    --chunk-size 32768 \
    --bf16 \
    --no-merge \
    --resume

printf 'EVAL_DONE %s\n' "$(date -u +%FT%TZ)"
touch "$RESULTS/EVAL_DONE"
