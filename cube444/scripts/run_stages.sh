#!/usr/bin/env bash
# Sequential driver for the 4x4x4 V pipeline on the GCP L4 box.
#
#   nohup bash cube444/scripts/run_stages.sh > cube444/logs/driver.log 2>&1 &
#
# Order is deliberately SMALL-FIRST: the 3.30M pipeline completes end-to-end in
# ~2 h and gives a gate verdict early, de-risking the 15M run behind it.
#
# Each stage logs to its own file under cube444/logs/. The 04_eval_v gate on the
# Stage-2 checkpoints is informational only (a random-walk-MSE model is EXPECTED
# to fail the saturation gate -- that is exactly what Bellman fixes), so those
# calls are not allowed to abort the driver.

set -u
cd /home/and-l/cayley || exit 1
mkdir -p cube444/models cube444/logs
export PYTHONUNBUFFERED=1

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*"; }

run() {  # run <label> <logfile> <cmd...>
    local label="$1" logf="$2"; shift 2
    log "START  $label  -> $logf"
    if "$@" > "$logf" 2>&1; then
        log "OK     $label"
    else
        log "FAILED $label (exit $?) -- see $logf"
        tail -20 "$logf"
        return 1
    fi
}

soft() {  # like run(), but a non-zero exit is reported and tolerated
    local label="$1" logf="$2"; shift 2
    log "START  $label  -> $logf"
    if "$@" > "$logf" 2>&1; then
        log "OK     $label"
    else
        log "GATE-FAIL $label (informational, continuing) -- see $logf"
    fi
}

log "=== 4x4x4 pipeline start ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- Stage 1: exact BFS anchors -------------------------------------------
run "S1 bfs-anchors" cube444/logs/01_bfs.log \
    python3 cube444/scripts/01_build_bfs.py \
        --max-exact 5 --d6-sample 20000000 \
        --out cube444/data/bfs_anchors.pt || exit 1

# ---- SMALL pipeline (3.30M) -----------------------------------------------
run "S2 pretrain-small" cube444/logs/02_pretrain_small.log \
    python3 cube444/scripts/02_train.py \
        --config cube444/configs/c_v0s_pretrain.yaml \
        --output cube444/models/c_v0s || exit 1

soft "gate pretrain-small" cube444/logs/04_gate_v0s.log \
    python3 cube444/scripts/04_eval_v.py \
        --checkpoint cube444/models/c_v0s/epoch_0999.pt

run "S3 bellman-small" cube444/logs/03_bellman_small.log \
    python3 cube444/scripts/03_bellman.py \
        --config cube444/configs/c_bellman_small.yaml \
        --output cube444/models/c_bells || exit 1

soft "gate bellman-small" cube444/logs/04_gate_bells.log \
    python3 cube444/scripts/04_eval_v.py \
        --checkpoint cube444/models/c_bells/epoch_0299.pt

# ---- MID pipeline (15.0M) --------------------------------------------------
run "S2 pretrain-mid" cube444/logs/02_pretrain_mid.log \
    python3 cube444/scripts/02_train.py \
        --config cube444/configs/c_v0_pretrain.yaml \
        --output cube444/models/c_v0 || exit 1

soft "gate pretrain-mid" cube444/logs/04_gate_v0.log \
    python3 cube444/scripts/04_eval_v.py \
        --checkpoint cube444/models/c_v0/epoch_0999.pt

run "S3 bellman-mid" cube444/logs/03_bellman_mid.log \
    python3 cube444/scripts/03_bellman.py \
        --config cube444/configs/c_bellman.yaml \
        --output cube444/models/c_bell || exit 1

soft "gate bellman-mid" cube444/logs/04_gate_bell.log \
    python3 cube444/scripts/04_eval_v.py \
        --checkpoint cube444/models/c_bell/epoch_0299.pt

log "=== 4x4x4 pipeline done ==="
