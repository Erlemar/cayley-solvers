# CayleyPy AZ4 Trainer - Megaminx

Train **AZ v4**, a strong 6M-param value+policy model for the
[CayleyPy Megaminx competition](https://www.kaggle.com/competitions/cayley-py-megaminx) -
end-to-end, with every stage's hyperparameters exposed in one config cell.

On a 51-puzzle stratified sample (beam 65536, our production solver), the extracted AZ v4
value head solves **51/51 with mean path 87.5**, vs 89.4 for the previous best pure-V model
of the same size. It was the first 6M model in our pipeline to break the "89-moves cluster"
without inference-side tricks.

This notebook mirrors the structure of
[CayleyPy RW-ModelBaselines Megaminx AdamW](https://www.kaggle.com/code/ogurtsov/cayleypy-rw-modelbaselines-megaminx-adamw)
by @ogurtsov (config-first, modular, ends with beam search + submission), but replaces the
single random-walk regression stage with the full 5-stage AZ v4 pipeline. The **model pools
of both notebooks are merged**: `CFG['model']['model_type']` switches between our ResMLP
(the AZ v4 architecture) and the community `PilgrimAttnRes` family - and any registered
model can be trained through any subset of the stages. Model training is self-contained
PyTorch (all code inlined below); beam search / solving uses the
[cayleypy](https://github.com/cayleypy/cayleypy) library, so results are directly comparable
with the other community baselines.

## The 5-stage pipeline

```
stage 1 "pretrain"    random-walk MSE distance regression        4000 ep   (from scratch)
stage 2 "curriculum"  mixed-k curriculum + EMA + val early-stop  ~6k ep    (warm from 1)
stage 3 "bellman"     Bellman bootstrap + frontier + BFS-d6      500 ep    (warm from 2)
stage 4 "bellman_dd"  + exact anchors at solved / depth-1        50 ep     (warm from 3)
stage 5 "az"          dual-head: policy CE + Bellman value       ~25 ep    (warm from 4)
                      -> export value head = the beam-search model
```

Why so many stages? Each fixes a failure mode of the previous one:

- **pretrain** learns "random-walk depth", an upper bound on true distance.
- **curriculum** improves mid-depth calibration by mixing walk lengths (and EMA smooths it).
- **bellman** replaces the walk-depth label with the self-consistent target
  `1 + min_a V(child)` (DeepCubeA-style), plus two data mixins: 25% beam-frontier states
  (the distribution beam search actually visits) and 10% exact-distance states (all 19.3M
  states within 6 moves of solved).
- **bellman_dd** anchors `V(solved)=0` and `V(depth-1)=1` exactly in every batch (the
  bootstrap alone leaves V(solved) near 2).
- **az** adds a policy head (cross-entropy on a strong community solutions set) while
  keeping the value recipe - the shared trunk learns from both signals, and the VALUE head
  comes out better than any pure-V model we trained. Counter-intuitively you must stop
  early (~epoch 24): past that the trunk memorizes the policy data and the value head
  degrades. Watch the printed top-1 accuracy - when it passes ~35-40% you are past the
  sweet spot.

**Default run: the FULL pipeline, from scratch.** The whole chain needs ~25-30h on a T4 -
several Kaggle sessions - so the notebook stops training gracefully at a wall budget
(default 8.5h), saves all checkpoints plus a `models/chain_state.json` progress marker,
and **the next run of the same notebook continues automatically from its own previous
output** (which is attached as an input). Just press "Save & Run All" again until the log
says `CHAIN COMPLETE`; the final session trains the az stage, exports the value head, and
runs the calibration + beam checks.

**Quick mode**: set `stages_to_run = ['az']` to skip stages 1-4 and warm-start from the
shipped stage-4 checkpoint instead - that reproduces AZ v4 in ~30 minutes on a T4.

Assets (checkpoints + datasets) are in
[megaminx-az4-training-assets](https://www.kaggle.com/datasets/artgor/megaminx-az4-training-assets).
Add it via "Add Input" if it is not already attached, together with the competition data.
