#!/bin/bash
# Watcher: polls GCP for m42 lower-q25 to finish, then launches m45 chain remotely.
set -uo pipefail

REMOTE_LOG=/home/and-l/cayley/megaminx/submissions/m42_lowerq25_full1001.log

echo "$(date) waiting for GCP m42 lower-q25 to finish..."
until gcloud compute ssh cayley-gpu --zone=us-east1-b --tunnel-through-iap \
        --command="grep -qE 'verify:.*total' $REMOTE_LOG 2>/dev/null && echo DONE" \
        2>/dev/null | grep -q DONE; do
    sleep 600
done
echo "$(date) GCP m42 done. Launching m45 chain remotely..."

# Launch m45 chain on GCP via nohup so the SSH session can disconnect.
gcloud compute ssh cayley-gpu --zone=us-east1-b --tunnel-through-iap --command="cd /home/and-l/cayley && chmod +x megaminx/scripts/run_m45_chain_gcp.sh && nohup bash megaminx/scripts/run_m45_chain_gcp.sh > megaminx/models/m45_chain.log 2>&1 &"
echo "$(date) m45 chain launched on GCP"
