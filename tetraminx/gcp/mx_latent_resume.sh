#!/bin/bash
# Idempotently launch or resume the standalone latent relational sparse-Q run.
# This is invoked manually for the first launch and by the VM startup script
# after a spot preemption. The persistent boot disk retains all checkpoints.

set -u

ROOT=/home/and-l/cayley
CFG="$ROOT/tetraminx/configs/mx_latent_relational.yaml"
OUT="$ROOT/tetraminx/models/mx_latent_relational"
LOG="$ROOT/tetraminx/logs/mx_latent_relational.log"
SESSION=mx_latent

mkdir -p "$OUT" "$(dirname "$LOG")"

# Already running: do nothing.
tmux has-session -t "$SESSION" 2>/dev/null && exit 0

EVAL=/home/and-l/mx_latent_evaluate.sh
EVAL_LOG="$ROOT/tetraminx/logs/mx_latent_relational_eval.log"
EVAL_DONE="$ROOT/tetraminx/results/mx_latent_relational/EVAL_DONE"
SWEEP=/home/and-l/mx_latent_checkpoint_sweep.sh
SWEEP_LOG="$ROOT/tetraminx/logs/mx_latent_checkpoint_sweep.log"
SWEEP_DONE="$ROOT/tetraminx/results/mx_latent_relational/SWEEP_DONE"

# Training may have completed just before a preemption interrupted evaluation.
# In that case, resume evaluation directly instead of exiting.
if grep -q '^done$' "$LOG" 2>/dev/null; then
    [ -f "$SWEEP_DONE" ] && exit 0
    if [ -f "$EVAL_DONE" ]; then
        tmux new-session -d -s "$SESSION" \
            "exec bash $SWEEP >> $SWEEP_LOG 2>&1"
    else
        tmux new-session -d -s "$SESSION" \
            "bash $EVAL >> $EVAL_LOG 2>&1 && exec bash $SWEEP >> $SWEEP_LOG 2>&1"
    fi
    exit 0
fi

CKPT=$(find "$OUT" -maxdepth 1 -type f -name 'epoch_*.pt' -print 2>/dev/null \
    | sort -V | tail -1)

ARGS="--config $CFG --output $OUT --set training.checkpoint_every_epochs=10"
if [ -n "$CKPT" ]; then
    ARGS="$ARGS --resume $CKPT"
fi

printf '=== %s launch from %s ===\n' "$(date -u +%FT%TZ)" "${CKPT:-scratch}" >> "$LOG"
tmux new-session -d -s "$SESSION" \
    "cd $ROOT && python3 -u tetraminx/scripts/51_train_sparse_q.py $ARGS >> $LOG 2>&1 && bash $EVAL >> $EVAL_LOG 2>&1 && exec bash $SWEEP >> $SWEEP_LOG 2>&1"
