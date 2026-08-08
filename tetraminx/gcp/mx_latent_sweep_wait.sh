#!/bin/bash
# Queue the 100-epoch checkpoint sweep behind the currently active evaluation.

set -u

ROOT=/home/and-l/cayley
RESULTS="$ROOT/tetraminx/results/mx_latent_relational"
EVAL_DONE="$RESULTS/EVAL_DONE"
SWEEP_DONE="$RESULTS/SWEEP_DONE"
SWEEP_LOG="$ROOT/tetraminx/logs/mx_latent_checkpoint_sweep.log"

[ -f "$SWEEP_DONE" ] && exit 0

while [ ! -f "$EVAL_DONE" ]; do
    sleep 30
done

exec bash /home/and-l/mx_latent_checkpoint_sweep.sh >> "$SWEEP_LOG" 2>&1
