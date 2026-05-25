#!/bin/bash
# PHS cumulative scoring -- Stage-1 relative sweep (doc section 13.1).
#
# Single-pass strat-51, NO sym-ensemble / NO NISS, to isolate the cumulative
# policy-cost effect. w=0.0 is the AZ v4 V + m23_v3 qshort baseline (no policy);
# w>0 adds --policy-model + --lambda-policy w + --phs-cumulative.
#
# Binding gate is Stage 2 (production recipe at the winning w); this is the fast
# relative signal: is there ANY w that beats the w=0.0 control on total moves?
#
# Run on GCP cayley-gpu in tmux:
#   tmux new-session -d -s phs 'bash ~/cayley/megaminx/scripts/phs_sweep_gcp.sh'
set -u
cd ~/cayley

V=megaminx/models/m_az_v4_v_only.pt
Q=megaminx/models/m23_v3_az_v4_sym/epoch_0199.pt
PI=megaminx/models/m_az_v4_pi_only.pt
OUTDIR=megaminx/submissions
LOGDIR=megaminx/models
SUMMARY=$LOGDIR/phs_sweep_summary.txt

mkdir -p "$OUTDIR"
echo "PHS Stage-1 sweep (single-pass strat-51 beam 65k, no sym) started $(date)" > "$SUMMARY"
echo "V=$V  Q=$Q  PI=$PI" >> "$SUMMARY"
echo "" >> "$SUMMARY"

for w in 0.0 0.01 0.03 0.05 0.1 0.2; do
  OUT=$OUTDIR/phs_sweep_w${w}.csv
  LOG=$LOGDIR/phs_sweep_w${w}.log
  if [ "$w" = "0.0" ]; then
    POL=""
  else
    POL="--policy-model $PI --lambda-policy $w --phs-cumulative"
  fi
  echo "=== w=$w starting $(date) ===" | tee -a "$SUMMARY"
  PYTHONUNBUFFERED=1 PYTHONUTF8=1 python3 -u megaminx/scripts/03_solve.py \
    --checkpoint "$V" --qshort-student "$Q" $POL \
    --stratified 5 --strat-seed 0 --beams 65536 --max-steps 120 --bf16 \
    --out "$OUT" > "$LOG" 2>&1
  RC=$?
  STATS=$(grep "stats:" "$LOG" | tail -1)
  if [ $RC -ne 0 ]; then
    echo "w=$w  EXIT=$RC  (FAILED -- see $LOG)  $STATS" | tee -a "$SUMMARY"
  else
    echo "w=$w  $STATS  (csv=$OUT)" | tee -a "$SUMMARY"
  fi
done

echo "" >> "$SUMMARY"
echo "DONE $(date)" | tee -a "$SUMMARY"
