#!/bin/bash
# Evaluate every 100-epoch latent-relational checkpoint on the shared Q@1M gate.
# Safe to rerun after interruption: completed stages have sidecar markers and
# path solves additionally use --resume.

set -euo pipefail

ROOT=/home/and-l/cayley
RESULTS="$ROOT/tetraminx/results/mx_latent_relational"
MODEL_DIR="$ROOT/tetraminx/models/mx_latent_relational"
PIDS=0,50,100,200,300,400,500,600,700,800,900,950,990,995,999
SWEEP_DONE="$RESULTS/SWEEP_DONE"

mkdir -p "$RESULTS"
cd "$ROOT"

[ -f "$SWEEP_DONE" ] && exit 0

for N in $(seq 100 100 1500); do
    EPOCH=$(printf '%04d' "$N")
    CKPT="$MODEL_DIR/epoch_${EPOCH}.pt"
    PROBE="$RESULTS/probe_ep${EPOCH}.txt"
    PATHS="$RESULTS/q1m_f1_ep${EPOCH}.csv"
    PROBE_DONE="$RESULTS/.probe_ep${EPOCH}.done"
    PATHS_DONE="$RESULTS/.q1m_f1_ep${EPOCH}.done"

    test -f "$CKPT"

    # The original evaluation completed these stages before this sweep was
    # requested. Adopt the existing artifacts instead of recomputing them.
    if [ "$EPOCH" = 0500 ] || [ "$EPOCH" = 1000 ] || [ "$EPOCH" = 1500 ]; then
        [ -s "$PROBE" ] && touch "$PROBE_DONE"
        [ -s "$PATHS" ] && touch "$PATHS_DONE"
    fi

    if [ ! -f "$PROBE_DONE" ]; then
        printf '=== %s epoch %s shared 262k decision-quality probe ===\n' \
            "$(date -u +%FT%TZ)" "$EPOCH"
        python3 -u tetraminx/scripts/52_eval_q.py \
            --q-model "$CKPT" \
            --n 262144 \
            --k-min 2 \
            --k-max 40 \
            --tilt 0.5 \
            --seed 7 \
            | tee "$PROBE"
        touch "$PROBE_DONE"
    fi

    if [ ! -f "$PATHS_DONE" ]; then
        printf '=== %s epoch %s Q@1M, one-frame checkpoint sweep ===\n' \
            "$(date -u +%FT%TZ)" "$EPOCH"
        python3 -u tetraminx/scripts/30_solve.py \
            --checkpoint "$CKPT" \
            --out "$PATHS" \
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
        touch "$PATHS_DONE"
    fi
done

SUMMARY="$RESULTS/checkpoint_sweep_summary.csv"
SUMMARY_TMP="$SUMMARY.tmp"
printf 'epoch,solved,total,status\n' > "$SUMMARY_TMP"

BEST_EPOCH=
BEST_TOTAL=
for N in $(seq 100 100 1500); do
    EPOCH=$(printf '%04d' "$N")
    read -r COUNT TOTAL < <(
        awk -F, 'NR > 1 && $2 != "" {n=gsub(/\./, ".", $2); total += n + 1; count++} END {print count + 0, total + 0}' \
            "$RESULTS/q1m_f1_ep${EPOCH}.csv"
    )
    STATUS=incomplete
    if [ "$COUNT" -eq 15 ]; then
        STATUS=complete
        if [ -z "$BEST_EPOCH" ] || [ "$TOTAL" -lt "$BEST_TOTAL" ]; then
            BEST_EPOCH=$EPOCH
            BEST_TOTAL=$TOTAL
        fi
    fi
    printf '%s,%s,%s,%s\n' "$N" "$COUNT" "$TOTAL" "$STATUS" >> "$SUMMARY_TMP"
done
mv "$SUMMARY_TMP" "$SUMMARY"

if [ -z "$BEST_EPOCH" ]; then
    printf 'no checkpoint completed all 15 one-frame paths\n' >&2
    exit 1
fi

printf '=== %s sweep winner: epoch %s, one-frame total %s; four-frame evaluation ===\n' \
    "$(date -u +%FT%TZ)" "$BEST_EPOCH" "$BEST_TOTAL"
python3 -u tetraminx/scripts/30_solve.py \
    --checkpoint "$MODEL_DIR/epoch_${BEST_EPOCH}.pt" \
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

printf 'SWEEP_DONE %s best_epoch=%s one_frame_total=%s\n' \
    "$(date -u +%FT%TZ)" "$BEST_EPOCH" "$BEST_TOTAL"
touch "$SWEEP_DONE"
