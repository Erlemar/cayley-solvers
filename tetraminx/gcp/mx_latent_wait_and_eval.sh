#!/bin/bash
# Companion for the uninterrupted first boot. A later spot boot does not need
# this watcher because mx_latent_resume.sh chains evaluation after training.

set -u

ROOT=/home/and-l/cayley
TRAIN_LOG="$ROOT/tetraminx/logs/mx_latent_relational.log"
EVAL_LOG="$ROOT/tetraminx/logs/mx_latent_relational_eval.log"

while ! grep -q '^done$' "$TRAIN_LOG" 2>/dev/null; do
    if ! tmux has-session -t mx_latent 2>/dev/null; then
        printf 'TRAINING_SESSION_MISSING %s\n' "$(date -u +%FT%TZ)" >> "$EVAL_LOG"
        exit 1
    fi
    sleep 300
done

exec bash /home/and-l/mx_latent_evaluate.sh >> "$EVAL_LOG" 2>&1
