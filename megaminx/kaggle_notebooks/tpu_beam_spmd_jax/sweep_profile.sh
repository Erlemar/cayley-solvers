#!/bin/bash
# Phase-ablation profile of the v6e beam step at B=16M. Each profile_skip value
# stubs one phase (WRONG beam, timing only); step-002 device time vs baseline
# = that phase's share of the per-step wall.
cd /mnt/data/v6e
for ps in none argsort gen a2a packbuild qfwd vfwd; do
  arg=""
  if [ "$ps" != "none" ]; then arg="--profile-skip $ps"; fi
  echo "########## profile_skip=$ps ##########"
  ~/tpu-env/bin/python gcp_beam_v6e.py --b-global 16777216 --start-pid 0 --end-pid 1 \
    --num-steps 3 --progress-every 1 $arg --out /mnt/data/out/ps_$ps.json 2>&1 \
    | grep -E 'cfg\]|compile\] done|step 00'
done
echo "########## DONE_SWEEP ##########"
