#!/bin/bash
# GCE metadata startup script. The spot VM uses termination-action=STOP, so an
# external watchdog starts it again; this hook then resumes the newest checkpoint.

sleep 45
su - and-l -c "bash /home/and-l/mx_latent_resume.sh" \
    >> /var/log/mx_latent_resume_boot.log 2>&1 || true
