---
description: Fetch GCP cayley-gpu VM external IP, ssh in, and show progress of any running megaminx solve (or training) job. Quick "is my GCP job alive and how far along is it?".
---

# /check-gcp

Quick status check on the GCP cayley-gpu VM. Use any time you need to know:
- Is the VM up?
- Is the solve/training job still running?
- What pid is the eval at? What's the model-solve vs fallback ratio?
- Time remaining (rough estimate)?

## Usage

`/check-gcp [<log-tail-lines>]`

Default tail lines: 30. Pass an integer to override (e.g., `/check-gcp 100`).

## Execution

```bash
TAIL_LINES="${1:-30}"

# Fetch external IP from GCP. Cache it for next runs (the IP changes if VM is
# stopped/started, but persists across the run).
echo "=== Fetching GCP cayley-gpu external IP ==="
GCP_IP=$(gcloud compute instances describe cayley-gpu \
    --zone=us-east1-b \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>&1)
GCP_STATUS=$?

if [[ $GCP_STATUS -ne 0 ]] || [[ -z "$GCP_IP" ]]; then
    echo "FAILED to get IP. Possible causes:"
    echo "  - VM stopped: run 'gcloud compute instances start cayley-gpu --zone=us-east1-b'"
    echo "  - gcloud auth expired: run 'gcloud auth login'"
    echo "  - Wrong project: run 'gcloud config set project <project-id>'"
    echo ""
    echo "  raw error: $GCP_IP"
    exit 1
fi

echo "  IP: $GCP_IP"
echo ""

# Probe what's running on the VM via gcloud compute ssh (more reliable on Windows
# than direct ssh — handles key registration + host fingerprint automatically).
echo "=== VM status ==="
gcloud compute ssh cayley-gpu --zone=us-east1-b --command="
    echo '--- uptime ---'
    uptime
    echo ''
    echo '--- python processes ---'
    ps -ef | grep -E 'python|03_solve|train' | grep -v grep | head -5 || echo 'no python processes'
    echo ''
    echo '--- GPU ---'
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not available'
    echo ''
    echo '--- latest log (checks both submissions/ and models/) ---'
    LATEST_LOG=\$(ls -t ~/cayley/megaminx/submissions/*.log ~/cayley/megaminx/models/*_eval.log ~/cayley/megaminx/models/*_solve.log 2>/dev/null | head -1)
    if [[ -z \"\$LATEST_LOG\" ]]; then
        echo 'no log found'
    else
        echo \"log: \$LATEST_LOG\"
        echo \"mtime: \$(stat -c '%y' \$LATEST_LOG | cut -d. -f1)\"
        echo ''
        echo '--- last $TAIL_LINES lines ---'
        tail -$TAIL_LINES \$LATEST_LOG
        echo ''
        echo '--- progress summary ---'
        grep -E '^pid=' \$LATEST_LOG | tail -1 || echo 'no pid= line yet'
        echo \"model_solves: \$(grep -c 'model:' \$LATEST_LOG 2>/dev/null || echo 0)\"
        echo \"fallbacks:    \$(grep -c 'fb:' \$LATEST_LOG 2>/dev/null || echo 0)\"
    fi
    echo ''
    echo '--- newest CSV (production output progress) ---'
    LATEST_CSV=\$(ls -t ~/cayley/megaminx/submissions/*_prod_1001.csv ~/cayley/megaminx/submissions/*_full*.csv 2>/dev/null | head -1)
    if [[ -n \"\$LATEST_CSV\" ]]; then
        echo \"csv:   \$LATEST_CSV\"
        echo \"rows:  \$(wc -l < \$LATEST_CSV) / 1002 (header + 1001 pids when complete)\"
        echo \"mtime: \$(stat -c '%y' \$LATEST_CSV | cut -d. -f1)\"
    fi
"
SSH_STATUS=$?

if [[ $SSH_STATUS -ne 0 ]]; then
    echo ""
    echo "SSH FAILED (exit $SSH_STATUS). Possible causes:"
    echo "  - VM not booted yet (just started?): wait 30-60s and retry"
    echo "  - gcloud auth expired: run 'gcloud auth login'"
    echo "  - Wrong project: run 'gcloud config set project <project-id>'"
fi

echo ""
echo "=== Quick commands ==="
echo "  Full ssh:        ssh andlukyane@$GCP_IP"
echo "  Pull latest CSV: scp andlukyane@$GCP_IP:~/cayley/megaminx/submissions/<name>.csv megaminx/submissions/"
echo "  Push fix:        scp <local-file> andlukyane@$GCP_IP:~/cayley/<dest>"
echo "  Stop VM (save \$): gcloud compute instances stop cayley-gpu --zone=us-east1-b"
```

## When to use

- Status check after kicking off any GCP solve (avoids manually fetching IP each time).
- Before going to bed / leaving the desk: confirm the run will still be working.
- After resuming a session: catch up on what GCP has done since last check.
- After a long polling silence: confirm the job isn't dead.

## When NOT to use (memory anti-pattern)

**Don't poll every 5-10 minutes** during a multi-hour run. User has noted
this is friction. The job emits progress every ~30-60s; checking once
per hour or per natural session break is fine. The `Monitor` tool with
heartbeat is the appropriate tool for "wake me when X happens".

## Reference

- VM full setup, restart, cost-management notes:
  `memory/reference_gcp_cayley_vm.md`
- Production solve recipe (required for full-1001 eval) is in CLAUDE.md
  rule 11. Don't accept single-pass results from GCP without verifying
  the recipe matches.
