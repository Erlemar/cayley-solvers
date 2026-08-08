# Megaminx AZ4 Training Assets

Support files for the notebook **CayleyPy AZ4 Trainer Megaminx**
(https://www.kaggle.com/code/artgor/cayleypy-az4-trainer-megaminx), which reproduces the
AZ v4 model for the CayleyPy Megaminx competition (strat-5 sample: 51/51 solved, mean 87.5
moves standalone V) and lets you retrain any stage of its pipeline with your own settings.

## The 5-stage lineage these files support

| Stage | What | Warm-start input (this dataset) |
|---|---|---|
| 1 pretrain | random-walk MSE distance regression, 4000 ep | (from scratch) |
| 2 curriculum | k-mix curriculum + EMA + val early-stop | `m07_big_k80_epoch3999.pt` |
| 3 bellman | anchored Bellman + frontier 25% + BFS-d6 10%, 500 ep | `m_curr_v0_best_ema.pt` |
| 4 bellman_dd | + V0/d1 anchor mixin, 50 ep | `m_curr_v3_epoch0499.pt` |
| 5 az | dual-head (policy CE + Bellman V) fine-tune, stop ~ep24 | `m_dd_v0_epoch0049.pt` |

Reference outputs of stage 5: `m_az_v4_epoch0024.pt` (dual-head) and
`m_az_v4_v_only.pt` (extracted value head as a plain distance model — this is
the checkpoint we run beam search with).

## Files

- `*.pt` — PyTorch checkpoints (`{"state_dict", "model_config", ...}`), all
  6M-param ResMLP `(2048,512) x 2 res blocks, embedding dim 16`.
- `bfs_d6_train.pt` — 19,352,405 unique megaminx states within distance 6 of solved,
  with exact distances (`{"states": int8 (N,120), "distances": int8 (N,)}`). Used as the
  exact-target mixin in stages 3/4/5 and for value-calibration checks.
- `frontier_states.pt` — 300,000 deduplicated beam-search frontier states (states
  only; Bellman targets are computed on the fly). Used as the frontier mixin in stages 3/4.
- `submission_73731.csv` — best publicly shared community solutions (total 73,731
  moves over 1001 puzzles), used to build the policy dataset.
- `az_dataset_73731.pt` — `{"states", "actions", "values"}`: one
  (state, action-taken, remaining-distance) tuple per move of every path in the CSV above
  (73,731 tuples). Policy CE data for stage 5. The notebook contains the builder cell, so
  you can regenerate this from any submission CSV.

## Notes

- The exact historical stage-4 run also used a pattern-database lower-bound penalty
  (lambda_pdb=5). The notebook's stage 4 ships with lambda_pdb=0 (the PDB files are large
  and the effect is secondary); `m_dd_v0_epoch0049.pt` here IS the original lambda=5
  artifact, so the default stage-5 path is unaffected.
- Original AZ v4 was trained on a 76,304-move min-merged solutions set; this dataset ships
  the better, fully public 73,731 set instead. More-consistent merged paths converge
  faster: watch the policy top-1 accuracy and stop early (the notebook explains).
