#!/bin/bash
# PHS w=0.03 vs w=0 baseline on 50 random mid-difficulty pids (buckets 1-8, seed 0).
# Fast (~3h) controlled diversity measurement on an INDEPENDENT sample (validates the
# strat-51 -61/51 per-pid min-merge finding). Production recipe sym4+multipass+NISS,
# --resume (crash-safe). Both arms run the SAME 50 pids so the per-pid min-merge is clean.
set -u
cd ~/cayley
V=megaminx/models/m_az_v4_v_only.pt
Q=megaminx/models/m23_v3_az_v4_sym/epoch_0199.pt
PI=megaminx/models/m_az_v4_pi_only.pt
PIDS=102,104,106,112,117,122,126,131,157,163,170,198,233,237,303,305,314,332,336,405,419,434,480,484,485,518,519,526,531,564,579,585,592,594,617,619,634,659,669,696,718,730,739,762,766,777,796,820,843,877
RECIPE="--sym-ensemble 4 --beams 16384,65536 --max-steps 60,150 --niss --bf16 --resume --pids $PIDS"
SUMMARY=megaminx/models/phs_rand50_summary.txt

echo "PHS rand50 (buckets 1-8, seed 0) started $(date)" > "$SUMMARY"

echo "=== arm A w=0.0 baseline $(date) ===" | tee -a "$SUMMARY"
PYTHONUNBUFFERED=1 PYTHONUTF8=1 python3 -u megaminx/scripts/03_solve.py \
  --checkpoint "$V" --qshort-student "$Q" $RECIPE \
  --out megaminx/submissions/phs_rand50_w0.0.csv > megaminx/models/phs_rand50_w0.0.log 2>&1
echo "w=0.0   $(grep 'stats:' megaminx/models/phs_rand50_w0.0.log | tail -1)" | tee -a "$SUMMARY"

echo "=== arm B w=0.03 treatment $(date) ===" | tee -a "$SUMMARY"
PYTHONUNBUFFERED=1 PYTHONUTF8=1 python3 -u megaminx/scripts/03_solve.py \
  --checkpoint "$V" --qshort-student "$Q" --policy-model "$PI" --lambda-policy 0.03 --phs-cumulative $RECIPE \
  --out megaminx/submissions/phs_rand50_w0.03.csv > megaminx/models/phs_rand50_w0.03.log 2>&1
echo "w=0.03  $(grep 'stats:' megaminx/models/phs_rand50_w0.03.log | tail -1)" | tee -a "$SUMMARY"

echo "DONE $(date)" | tee -a "$SUMMARY"
