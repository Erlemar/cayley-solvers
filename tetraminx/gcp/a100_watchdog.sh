#!/bin/bash
# Restart any preempted spot A100. Spot VMs with termination-action=STOP remain
# stopped until explicitly started; each VM's metadata startup hook resumes its
# own newest checkpoint.

for VM in tetra-a100 tetra-a100b tetra-latent; do
    STATUS=$(gcloud compute instances describe "$VM" \
        --zone us-central1-a --format='value(status)' 2>/dev/null)
    if [ "$STATUS" = "TERMINATED" ]; then
        printf '%s %s TERMINATED -> starting\n' "$(date -u +%FT%TZ)" "$VM" \
            >> /home/and-l/a100_watchdog.log
        gcloud compute instances start "$VM" --zone us-central1-a \
            >> /home/and-l/a100_watchdog.log 2>&1
    fi
done
