#!/bin/bash
# PHS cumulative -- Stage-2 BINDING GATE (doc 13.1).
# Production recipe (sym4 + multi-pass + NISS) strat-51: w=0.0 baseline vs w=0.03 treatment.
# This is the only config that solves hard pids, where PHS's path-preservation mechanism
# (preserve policy-likely paths when V misranks) would actually show -- Stage-1 single-pass
# was structurally blind to it.
#
# CAVEAT: under sym-ensemble the AZ v4 pi head sees ROTATED states (it was not rotation-
# augmented) on 3/4 rotations, so the cumulative term is partly OOD noise there; only the
# identity rotation is in-distribution. If the treatment regresses, this pi/sym interaction
# is a candidate cause (follow-up: identity-only policy, or a sym-aware pi). The test is
# still the right DEPLOYMENT question: "does adding PHS to the production stack help?"
#
# Compare via the per-bucket model_avg in each log (weighted mean over the 51 strat pids),
# NOT the 1001-total (which is dominated by ~950 constant fallbacks).
set -u
cd ~/cayley
V=megaminx/models/m_az_v4_v_only.pt
Q=megaminx/models/m23_v3_az_v4_sym/epoch_0199.pt
PI=megaminx/models/m_az_v4_pi_only.pt
RECIPE="--sym-ensemble 4 --beams 16384,65536 --max-steps 60,150 --niss --bf16 --stratified 5 --strat-seed 0 --resume"
SUMMARY=megaminx/models/phs_stage2_summary.txt

echo "PHS Stage-2 (production recipe sym4+multipass+NISS, strat-51) started $(date)" > "$SUMMARY"

echo "=== arm A: w=0.0 baseline $(date) ===" | tee -a "$SUMMARY"
PYTHONUNBUFFERED=1 PYTHONUTF8=1 python3 -u megaminx/scripts/03_solve.py \
  --checkpoint "$V" --qshort-student "$Q" $RECIPE \
  --out megaminx/submissions/phs_stage2_w0.0.csv > megaminx/models/phs_stage2_w0.0.log 2>&1
echo "w=0.0   $(grep 'stats:' megaminx/models/phs_stage2_w0.0.log | tail -1)" | tee -a "$SUMMARY"

echo "=== arm B: w=0.03 treatment $(date) ===" | tee -a "$SUMMARY"
PYTHONUNBUFFERED=1 PYTHONUTF8=1 python3 -u megaminx/scripts/03_solve.py \
  --checkpoint "$V" --qshort-student "$Q" --policy-model "$PI" --lambda-policy 0.03 --phs-cumulative $RECIPE \
  --out megaminx/submissions/phs_stage2_w0.03.csv > megaminx/models/phs_stage2_w0.03.log 2>&1
echo "w=0.03  $(grep 'stats:' megaminx/models/phs_stage2_w0.03.log | tail -1)" | tee -a "$SUMMARY"

echo "DONE $(date)" | tee -a "$SUMMARY"
