# Notes, pitfalls, and what to try

Things we learned building AZ v4 (each of these cost us GPU-days - read before changing
the config):

1. **Select checkpoints by beam quality, never by training loss.** The az stage's value
   head peaks around epoch 24 and then degrades while both losses keep falling. The same
   holds for very long V-model training in general: a 184-epoch stage-4 variant had lower
   loss than the 50-epoch one and solved 0/20 of the puzzles the 50-epoch one solved.
2. **Batch sizes are part of the recipe.** Halving `rw_batch_size`/`policy_batch_size`
   quadruples gradient steps per epoch: the policy memorizes (top-1 90%+) and the value
   head is destroyed. If you change them, re-tune the stopping epoch from the printed
   top-1 accuracy (sweet spot was ~28%).
3. **Bigger trunks are not free wins.** A 20.5M trunk under this exact recipe was worse
   than the 6M one at every checkpoint. Also, warm-starting across a shape change silently
   skips mismatched layers (you would be fine-tuning a mostly random net) - retrain from
   stage 1 at the new shape instead.
4. **The saturation canary is load-bearing.** Every architecture we tried whose V kept
   growing on deep walks (instead of flattening at ~25-32) failed beam search regardless
   of how good its training loss and shallow calibration looked. Check the canary output
   before spending hours on beam runs.
5. **Better policy data converges faster.** AZ v4 was originally trained on a 76,304-move
   merged solutions set; this notebook ships a stronger public 73,731 set. More-consistent
   paths mean fewer ambiguous (state, action) pairs, so expect the sweet spot at a
   somewhat different epoch - watch top-1 accuracy and compare a couple of checkpoints
   with the bench cell.
6. **Ideas that did NOT work** (so you can skip them): symmetry/rotation augmentation on
   the value head; training the value head on solver paths only (beam expands off-path
   children the model has never seen - random-walk coverage is essential); pairing this V
   with a Q-shortlister distilled from a different V.
7. **Non-ResMLP backbones are UNVALIDATED under this recipe.** The model pool lets you
   push `PilgrimAttnRes` (or your own model) through all five stages, but every
   hyperparameter default here was tuned on the ResMLP. In our broader architecture
   survey, seven different encoder families matched the ResMLP on training loss and then
   failed the saturation canary and/or beam search - so run the canary cell before
   spending hours, and expect to retune per stage.

Improving on AZ v4's value head is genuinely hard - most of our subsequent gains came
from inference (wider beams, symmetry ensembles, min-merging across runs), not from
better models. If you find a config that beats it on the bench cell, please share it in
the competition discussion!

*Assets dataset: [megaminx-az4-training-assets](https://www.kaggle.com/datasets/artgor/megaminx-az4-training-assets).
Structure inspired by @ogurtsov's
[modular baselines notebook](https://www.kaggle.com/code/ogurtsov/cayleypy-rw-modelbaselines-megaminx-adamw).
Happy training!*
