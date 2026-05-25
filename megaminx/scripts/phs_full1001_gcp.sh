#!/bin/bash
# PHS cumulative w=0.03 -- FULL-1001 production run (doc 13.1).
# Diversity contributor for min-merge: Stage-2 strat-51 showed standalone TIE (-3)
# but per-pid min-merge -61/51 pids. This full run produces the diverse per-pid paths
# to min-merge into the stack. Inference-side -> cannot regress under min-merge.
# ETA ~30-40h on L4 (~109s/pid measured on strat-51 arm A, qshort-accelerated).
# --resume makes it crash-safe (per-pid CSV writes).
set -u
cd ~/cayley
PYTHONUNBUFFERED=1 PYTHONUTF8=1 python3 -u megaminx/scripts/03_solve.py \
  --checkpoint megaminx/models/m_az_v4_v_only.pt \
  --qshort-student megaminx/models/m23_v3_az_v4_sym/epoch_0199.pt \
  --policy-model megaminx/models/m_az_v4_pi_only.pt --lambda-policy 0.03 --phs-cumulative \
  --sym-ensemble 4 --beams 16384,65536 --max-steps 60,150 --niss --bf16 --resume \
  --out megaminx/submissions/phs_full1001_w0.03.csv > megaminx/models/phs_full1001_w0.03.log 2>&1
echo "DONE $(date)" >> megaminx/models/phs_full1001_w0.03.log
