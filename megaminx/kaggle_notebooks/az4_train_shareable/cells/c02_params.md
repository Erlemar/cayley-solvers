# Params

Everything is driven by the single `CFG` dict in the next cell. Groups:

- **manage**: `stages_to_run` (any contiguous subset of
  `['pretrain','curriculum','bellman','bellman_dd','az']`; default = all five),
  `quick_test` (tiny smoke run), `experiment`, `seed`, `precision`, `compile`.

  **`experiment` is the chain identity.** Multi-session runs resume only a previous
  chain whose `(experiment, model, seed)` all match - to start a NEW training from
  scratch, just give it a new `experiment` label; the old chain is left alone and
  ignored. Ordinary stage hyperparameters (lr, epochs, mixin fractions, ...) are
  deliberately NOT part of the identity: change only a later stage and the chain
  correctly reuses your completed earlier stages. Convention: one experiment label =
  one recipe.
- **multi-session chaining**: `max_wall_hours` / `eval_reserve_minutes` / `auto_resume`.
  A Kaggle run killed by the session limit saves NO output, so training stops gracefully
  at the wall budget instead, writes every checkpoint plus `models/chain_state.json`, and
  the next run continues from its own previous output (the notebook's output is attached
  as one of its inputs). Progress carries across runs per stage AND per epoch within a
  stage; the chain state is keyed to (model_type, params, seed) - change any of those and
  it starts fresh.
- **model (the MODEL POOL)**: `CFG['model']['model_type']` selects the backbone, and each
  registered model keeps its own parameter block in the same dict. Out of the box:
  - `'ResMLPDistance'` - our architecture (embedding + LayerNorm ResMLP). The default
    params `(2048, 512) x 2 blocks, embed 16` = the 6.0M AZ v4 trunk.
  - `'PilgrimAttnRes'` - the community baseline family from @ogurtsov's modular notebook
    (one-hot + BatchNorm; `block_type='residual'` or `'attn_res'`).

  **Any model runs through ALL five stages**: stages 1-4 train it as a plain distance
  model, and stage 5 wraps the same backbone with policy + value heads. To add your own:
  define the class in the model-pool cell (needs `forward`, `features`, `feature_dim`),
  register it in `MODEL_REGISTRY`, and add its parameter block under `CFG['model']`.

  Two caveats: (1) the shipped warm-start checkpoints exist only for the default 6M
  ResMLP - any other model/shape must be trained from `'pretrain'` up (the notebook
  enforces this with a clear error); (2) in our experiments bigger ResMLP trunks
  REGRESSED under this exact recipe (a 20.5M model lost to the 6M one at every
  checkpoint), and the whole recipe was tuned on the ResMLP - treat other backbones as
  experiments and judge them with the canary + bench cells, not the training loss.
- **optimizer / loss factories** (same pattern as the community notebook): each stage has
  `'optimizer': {'name': ..., 'params': {...}}` resolving any `torch.optim` class (lr
  still comes from the stage's `'lr'` key), and the two regression stages
  (`pretrain`/`curriculum`) also take `'loss': {'name': ..., 'params': {...}}` - any
  `torch.nn` loss plus `PinballLoss` / `LogCoshLoss`. The Bellman/az losses are part of
  the algorithm and stay fixed.
- **per-stage sections** (`pretrain`, `curriculum`, `bellman`, `bellman_dd`, `az`): epochs,
  batch sizes, learning rates, mixin fractions, anchor counts, target-net refresh cadence.
  Defaults are the exact values that produced AZ v4. Each stage's `warmstart` is `'auto'`
  (previous stage's fresh output if you trained it this session, else the shipped
  checkpoint) or a path.
- **eval**: calibration canary + small cayleypy beam benchmark.
- **solve**: full solving block writing `submission.csv`. Off by default;
  `list_states_to_solve` is a short list by default - set it to `[]` to solve ALL 1001
  puzzles (many hours at useful beam widths).

Approximate wall-clock per stage (Kaggle T4, fp32; the az number is measured, the rest
are scaled from our local runs; local RTX 4090 with bf16+compile is ~4x faster):

| stage | epochs | est. T4 wall |
|---|---|---|
| pretrain | 4000 | ~6-8 h |
| curriculum | early-stops ~6k | ~7-10 h |
| bellman | 500 | ~8-12 h |
| bellman_dd | 50 | ~1-1.5 h |
| az (+ eval) | 30 | ~0.5 h (measured) |
| full chain | | ~25-30 h = 3-4 sessions via auto-resume |

Key facts baked into the defaults (learned the hard way):

- **Training loss is NOT the model-selection signal.** For the `az` stage the best value
  head is at ~epoch 24 even though both losses keep "improving" long after. Checkpoints are
  saved every `checkpoint_every` epochs; the canary + bench cells below are the judge.
- **`az.lr_t_max` stays 200 even though default `epochs=30`.** The original run used a
  200-epoch cosine schedule and early-stopped; shortening T_max would change the LR at
  epoch 24 and give a different model.
- **Precision `'auto'` = bf16 only on Ampere+ GPUs, else fp32.** P100 has no bf16; T4
  emulates it with a catastrophic slowdown (we measured ~40x on inference). fp32 is fine
  for a 6M model.
- **`rw_batch_size=8192 / policy_batch_size=1024` matter.** Running the az stage at half
  batch (the old script defaults) gives 4x the gradient steps per epoch and the policy
  memorizes before the value head is ready.
