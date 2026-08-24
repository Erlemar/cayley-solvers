#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BEAM_SIZE="${BEAM_SIZE:-2097152}"
NUM_STEPS="${NUM_STEPS:-150}"
NUM_ATTEMPTS="${NUM_ATTEMPTS:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16384}"
GPU_IDS="${GPU_IDS:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"

mkdir -p results

exec python3 -u solve_ensemble_submission.py \
  --B "$BEAM_SIZE" \
  --num-steps "$NUM_STEPS" \
  --num-attempts "$NUM_ATTEMPTS" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --gpu-ids "$GPU_IDS" \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  --progress-db "results/progress-shard${SHARD_INDEX}.sqlite3" \
  --output "results/submission-shard${SHARD_INDEX}.csv" \
  --compile \
  --compile-skip-dynamic-cudagraphs
