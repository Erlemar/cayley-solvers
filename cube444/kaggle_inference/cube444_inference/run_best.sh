#!/bin/bash
# Best measured cube444 configuration, 2026-08-09.
#
#   scorer   s3          Q-Bellman-refined PieceTransformer (models/s3)
#   blend    0.4         ResMLP x16 rescorer -- the bundle default, and worth -72 moves
#   endgame  6           exact BFS tail, 67,041,676 states (291 s build, 26.8 GB RAM)
#
# Held-out 54 pids @ B=65536: 2851 vs a 2609 floor (+9.3%), 2 wins, merge gain 4.
# See reports/cube444_stage3_heldout_2026-08-09.md for how to read those numbers --
# in particular, judge by per-pid MIN against the floor, never the standalone mean.
#
# Usage:  ./run_best.sh [extra args passed through to the solver]
#   ./run_best.sh --pids 8,199,399            # a few pids
#   ./run_best.sh --limit 50                  # first 50
#   ./run_best.sh --B 1048576 --num-steps 100 # wide beam
#   ./run_best.sh --num-shards 4 --shard-index 0   # shard a full pass across boxes
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${CUBE444_PY:-python}"

cd "$HERE/solver"
exec "$PY" solve_ensemble_submission.py \
    --transformer-info   models/s3/model.json \
    --transformer-weights models/s3/model.pth \
    --mlp-info    models/mlp_x16/model.json \
    --mlp-weights models/mlp_x16/model.pth \
    --mlp-weight 0.4 \
    --tail-bfs-depth 6 \
    --baseline-submission "$HERE/submissions/cube4_submission_46662.csv" \
    --progress-db results/best.sqlite3 \
    --output      results/best.csv \
    "$@"
