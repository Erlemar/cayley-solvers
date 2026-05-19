#!/bin/bash
# Watcher: polls for m43_chain.log to finish, then launches m44 chain on local 4090.
set -uo pipefail

echo "$(date) waiting for m43 chain to finish..."
until grep -q "all queued tasks done" megaminx/submissions/m43_chain.log 2>/dev/null; do
    sleep 60
done
echo "$(date) m43 chain done. Launching m44..."

bash megaminx/scripts/run_m44_chain.sh > megaminx/models/m44_chain.log 2>&1
echo "$(date) m44 chain finished"
