#!/usr/bin/env bash
# Second pipeline leg, informed by the 2026-07-23 measurements.
#
#   nohup bash cube444/scripts/run_stages2.sh > cube444/logs/driver2.log 2>&1 &
#
# What changed vs run_stages.sh:
#   * Stage 1 (BFS anchors) is done -- skipped.
#   * Stage 2 small is done -- skipped.
#   * Stage 3 small CONTINUES from ep124 rather than restarting. The first leg
#     was stopped early on a hypothesis (scale collapse is harmful) that the
#     depth sweep then falsified: ep124 beats ep24 at every depth.
#   * Stage 2 mid is cut from 1000 to 400 epochs -- the pretrain loss plateaus
#     at ~epoch 180, so the extra 600 were pure wall time.
#   * Stage 3 mid runs 400 epochs instead of 300, since quality was still
#     improving at the end of the first run.
#
# Order: cheap-and-known-good first (small continuation, ~70 min), then the
# untested capacity lever (15M mid, ~4.5 h).

set -u
cd /home/and-l/cayley || exit 1
mkdir -p cube444/models cube444/logs
export PYTHONUNBUFFERED=1

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*"; }

run() {
    local label="$1" logf="$2"; shift 2
    log "START  $label  -> $logf"
    if "$@" > "$logf" 2>&1; then log "OK     $label"; else
        log "FAILED $label -- see $logf"; tail -20 "$logf"; return 1; fi
}
soft() {
    local label="$1" logf="$2"; shift 2
    log "START  $label  -> $logf"
    if "$@" > "$logf" 2>&1; then log "OK     $label"; else
        log "GATE-FAIL $label (informational, continuing)"; fi
}

log "=== leg 2 start ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run "S3 bellman-small-cont" cube444/logs/03_bellman_small_cont.log \
    python3 cube444/scripts/03_bellman.py \
        --config cube444/configs/c_bellman_small_cont.yaml \
        --output cube444/models/c_bells2 || exit 1

soft "gate bellman-small-cont" cube444/logs/04_gate_bells2.log \
    python3 cube444/scripts/04_eval_v.py \
        --checkpoint cube444/models/c_bells2/epoch_0399.pt --bf16

run "S2 pretrain-mid" cube444/logs/02_pretrain_mid.log \
    python3 cube444/scripts/02_train.py \
        --config cube444/configs/c_v0_pretrain.yaml \
        --output cube444/models/c_v0 --epochs 400 || exit 1

run "S3 bellman-mid" cube444/logs/03_bellman_mid.log \
    python3 cube444/scripts/03_bellman.py \
        --config cube444/configs/c_bellman.yaml \
        --output cube444/models/c_bell || exit 1

soft "gate bellman-mid" cube444/logs/04_gate_bell.log \
    python3 cube444/scripts/04_eval_v.py \
        --checkpoint cube444/models/c_bell/epoch_0399.pt --bf16

log "=== leg 2 done ==="
