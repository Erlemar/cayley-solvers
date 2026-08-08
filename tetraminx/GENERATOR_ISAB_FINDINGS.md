# Generator-ISAB experiment findings

Date: 2026-08-02  
Status: training and checkpoint sweep complete; every reported path replay-verified

## Executive result

Generator-ISAB fixed the severe regression of the earlier latent-relational model, but it did not close the one-frame gap to the PieceTransformer.

On the fixed 15-pid evaluation set, the best one-frame checkpoint was epoch 1300:

| model/configuration | total moves | mean/pid | recorded wall time |
|---|---:|---:|---:|
| merged community/ours floor | **423** | **28.20** | -- |
| PieceTransformer, Q@1M x1 | **441** | **29.40** | 6,305 s |
| **Generator-ISAB epoch 1300, Q@1M x1** | **462** | **30.80** | 1,021 s |
| ResMLP, Q@1M x1 | **468** | **31.20** | 143 s |
| Generator-ISAB epoch 1300, Q@1M x4 frames | **435** | **29.00** | 4,019 s |
| ResMLP, Q@4M x4 frames | **420** | **28.00** | 2,318 s |

Thus, at the same one-frame Q@1M search width, Generator-ISAB:

- beats ResMLP by 6 moves;
- remains 21 moves behind PieceTransformer;
- closes 6 of the 27 moves, or 22.2%, of the ResMLP-to-Transformer gap;
- remains 39 moves behind the merged 423 floor.

The four-frame Generator-ISAB run reaches 435, which is 27 moves better than its own one-frame result and 6 moves better than the recorded one-frame Transformer result. It is not a like-for-like model comparison: it spends four inference frames rather than one.

The recorded wall times also came from different machines and thermal conditions. ResMLP and PieceTransformer were measured on a laptop RTX 4090 under different thermal states; Generator-ISAB was measured on a GCP A100. The raw numbers are useful operational observations, but they are not a controlled latency benchmark. A matched-device benchmark is still required.

## Model tested

The experiment retained the strong ResMLP value function and added a small, standalone relational action-value correction:

`Q(s, a) = Q_resmlp(s, a) + delta_Q_relational(s, a)`

The inference model does not require a Transformer teacher or distillation.

Key properties:

- The epoch-0 model exactly reproduces the frozen ResMLP epoch-1500 checkpoint.
- The 50 physical piece tokens are preserved; they are not compressed into a small generic latent bottleneck.
- Two ISAB blocks use 12 inducing tokens, model width 128, four attention heads, and feed-forward width 256.
- Exact generator-aware action descriptors and a moved-piece readout condition the correction on the candidate action.
- The correction head is zero-initialized, making the starting point a safe ResMLP identity.
- Total parameters: 5,805,209.
- Trainable correction parameters: 796,929.
- The ResMLP base checkpoint SHA256 is `417b74408c7f144b0664b3acb6512e3a2a3e56691049c30bffc01a410e2cbffa`.

The detailed design and original hypotheses are in [GENERATOR_ISAB_PLAN.md](GENERATOR_ISAB_PLAN.md).

## Training and evaluation protocol

Training used:

- 1,500 epochs;
- 256 steps per epoch;
- batch size 2,048;
- non-backtracking random walks at depths 2 through 40 with tilt 0.5;
- exact depth-at-most-5 anchors and symmetry augmentation;
- bf16, compiled training, and fused AdamW;
- checkpoints every 10 epochs.

Training took approximately 50.1 seconds per epoch on a GCP A100 40 GB GPU.

Every 100th checkpoint was evaluated using the same protocol:

- pids `0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 990, 995, 999`;
- one shared 262,144-sample Q probe;
- Q beam width 1,000,000;
- one frame for the checkpoint sweep;
- no community floor during search;
- exact depth-6 endgame table with 27,779,749 states;
- replay verification of every returned path.

The best complete one-frame checkpoint was then evaluated using four frames: `(0, false)`, `(0, true)`, `(1, false)`, and `(1, true)`.

## Complete checkpoint sweep

All 15 checkpoints produced 15 valid paths. Lower path totals are better.

| epoch | probe pair accuracy | probe top-1 | probe Q-gap | Q@1M total | mean/pid | wall time |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.8870 | 0.5088 | 1.391 | 469 | 31.27 | 1,044 s |
| 200 | 0.8870 | 0.5083 | 1.391 | 470 | 31.33 | 1,048 s |
| 300 | 0.8876 | 0.5076 | 1.391 | 463 | 30.87 | 1,024 s |
| 400 | 0.8874 | 0.5077 | 1.391 | 464 | 30.93 | 1,027 s |
| 500 | 0.8875 | 0.5087 | 1.392 | 467 | 31.13 | 1,037 s |
| 600 | 0.8874 | 0.5076 | 1.393 | 463 | 30.87 | 1,024 s |
| 700 | 0.8876 | 0.5081 | 1.395 | 471 | 31.40 | 1,051 s |
| 800 | 0.8880 | 0.5080 | 1.394 | 471 | 31.40 | 1,050 s |
| 900 | 0.8880 | 0.5088 | 1.396 | 468 | 31.20 | 1,041 s |
| 1000 | 0.8875 | 0.5087 | 1.395 | 463 | 30.87 | 1,024 s |
| 1100 | 0.8878 | 0.5083 | 1.399 | 472 | 31.47 | 1,054 s |
| 1200 | 0.8882 | 0.5083 | 1.399 | 469 | 31.27 | 1,044 s |
| **1300** | **0.8878** | **0.5080** | **1.400** | **462** | **30.80** | **1,021 s** |
| 1400 | 0.8881 | 0.5091 | 1.398 | 469 | 31.27 | 1,043 s |
| 1500 | 0.8879 | 0.5090 | 1.397 | 468 | 31.20 | 1,041 s |

The average total across the 15 checkpoints is 467.27. This is only 0.73 moves better than the ResMLP total of 468. The 6-move improvement at epoch 1300 is therefore a real verified result on this set, but not a stable improvement throughout training.

The epoch-1300 one-frame path lengths were:

`1, 35, 34, 34, 33, 34, 33, 33, 32, 34, 33, 31, 33, 31, 31`

The four-frame path lengths were:

`1, 30, 33, 31, 32, 31, 31, 29, 31, 30, 31, 31, 32, 31, 31`

## What worked

### 1. Preserving physical-piece tokens was the correct repair

The previous latent-relational model compressed the state too aggressively and performed much worse:

| model | best complete one-frame total | four-frame total |
|---|---:|---:|
| earlier latent-relational model | 505 at epoch 300 | 485 |
| Generator-ISAB | **462 at epoch 1300** | **435** |

Generator-ISAB improves the comparable one-frame result by 43 moves and the four-frame result by 50 moves. This strongly supports retaining the 50 physical-piece identities and their exact generator geometry. The earlier generic latent bottleneck discarded information that beam search needed.

### 2. A zero-initialized residual over ResMLP was a safe starting point

Epoch 0 exactly matched the known ResMLP. The new branch did not have to relearn the full value function and could specialize in relational corrections. This avoided the catastrophic regression seen in the first latent design.

### 3. Generator-aware conditioning matters

The model now knows which physical pieces and permutations define each action. The improvement over the previous latent model is consistent with the hypothesis that action value cannot be inferred reliably from a generic state summary plus an arbitrary action embedding.

### 4. Multiple frames remain a powerful source of search diversity

Four frames improve the selected checkpoint from 462 to 435, a 27-move gain. The model therefore contains useful but frame-dependent ranking information. However, four independent frames multiply its already substantial inference cost.

## What did not work well enough

### 1. The relational correction did not reproduce the Transformer's one-frame advantage

The best Generator-ISAB checkpoint remains 21 moves behind the PieceTransformer at Q@1M x1. The experiment recovered ResMLP quality and added a modest best-checkpoint gain; it did not demonstrate Transformer-level ranking with MLP-like speed.

### 2. Probe metrics and validation loss do not select good beam-search checkpoints

The offline metrics are nearly flat while path quality moves by up to 10 moves:

- epoch 1400 has the best probe top-1 accuracy, 0.5091, but scores 469 paths;
- epoch 1300 has lower top-1 accuracy, 0.5080, but the best path total, 462;
- epoch 1100 has a large Q-gap, 1.399, but the worst path total, 472;
- epoch 300 is already within one move of the best result, long before training completes.

The Q probe measures average action prediction. Beam search is controlled by a small number of ranking mistakes near the beam cutoff over a long trajectory. Those objectives are not equivalent. Checkpoints must be selected using replay-verified path evaluation, not validation loss, pair accuracy, top-1 accuracy, or Q-gap alone.

### 3. More epochs do not monotonically improve paths

Epoch 300 scores 463, epoch 1300 scores 462, and the final epoch returns to the ResMLP total of 468. Continuing the same objective for longer mostly changes calibration and local rankings; it does not reliably improve the search policy.

### 4. The current training distribution is misaligned with inference failures

Training is dominated by random-walk states and an absolute Q regression objective. Once a strong frozen ResMLP handles common/easy patterns, the useful residual signal is concentrated in ambiguous states and near-tied actions encountered on real beam frontiers. Uniform random walks rarely reproduce that conditional distribution.

At greater walk depths, path-derived Q targets are also noisy and multimodal: several actions can be equally good, and the sampled reverse path is not necessarily the uniquely best continuation. Absolute MSE penalizes acceptable alternatives and spends capacity on Q-scale calibration rather than the relative ordering that beam search needs.

### 5. The hard moved-piece readout may be too restrictive

An action directly reads only the pieces it moves. Yet whether an action is valuable can depend on how those pieces relate to every other piece. The ISAB state tokens can exchange global information, but the final hard mask remains an information and gradient bottleneck. A soft generator bias over all tokens may retain the speed advantage while exposing the action query to the full state.

### 6. The correction may be under-capacity in the wrong places

Only 0.797 million parameters are trainable, versus roughly 3.44 million parameters in the PieceTransformer. Simply scaling the branch would erode the latency goal, but the current two-block, width-128, 12-inducing-token design may not have enough capacity to model the Transformer’s useful pairwise ranking corrections.

### 7. The latency target has not yet been demonstrated

The end-to-end Generator-ISAB run was much slower than the recorded ResMLP run. Some of this comparison is confounded by hardware and thermal conditions, but there is no evidence yet that this architecture is close enough to ResMLP latency. The planned controlled raw-forward benchmark was not completed.

## Interpretation

The central architecture hypothesis was partly supported:

- **Supported:** relational computation can be added without destroying ResMLP quality when physical pieces, exact generator structure, and a zero-initialized residual path are preserved.
- **Not yet supported:** this small ISAB correction is sufficient to match Transformer path quality at substantially lower latency.

Most of the apparent success is recovery from the previous lossy bottleneck. Relative to the actual incumbent ResMLP, the average checkpoint is essentially neutral and the selected best checkpoint gains only six moves on 15 pids. The model should therefore be treated as an informative prototype and possible diversity scorer, not as a new default.

## Recommended next experiment

Keep the standalone residual architecture, but change the data and objective before making the model larger.

### Stage A: collect beam-frontier training data

1. Run the frozen ResMLP and Generator-ISAB on real search frontiers across a much larger pid set.
2. Save states near the retained/pruned beam cutoff, with extra weight on states where models disagree or where action margins are small.
3. Label all 24 actions using stronger offline search targets, exact short-horizon values where available, or a stronger teacher used only during training.
4. Balance examples by depth, frontier rank, and difficulty so easy random-walk states cannot dominate.
5. Keep ordinary random-walk and exact-anchor batches as a separate coverage stream rather than diluting frontier batches with a tiny path-label mix-in.

### Stage B: optimize beam-critical action ranking

Train the residual primarily on centered action advantages and ranking:

- subtract each state’s mean or best action value before computing the main loss;
- use pairwise or listwise losses on the best actions and near-cutoff hard negatives;
- keep a smaller absolute-Q calibration loss so the values remain usable by the existing search;
- retain exact anchors and symmetry consistency as auxiliary losses;
- preserve zero initialization and apply a trust-region/L2 penalty to the correction so it cannot freely damage the ResMLP ordering.

This directly targets the quantity beam search consumes: action ordering on difficult frontier states.

### Stage C: one controlled architecture ablation

Compare the current hard moved-piece readout against a soft global action readout:

- every action may attend to every piece token;
- exact moved-piece membership and generator role are added as attention biases/features rather than used as a hard mask;
- keep the same parameter count and training data for the first A/B test.

If frontier ranking improves without excessive latency, then test either 16 inducing tokens or a modest width increase. Do not scale depth and width simultaneously; the latency budget must remain measurable.

### Stage D: optional late joint tuning

If the residual-only model plateaus, unfreeze only the final ResMLP block for a short second phase at a learning rate 10 to 30 times smaller than the relational branch. This permits the base representation to expose useful relational features without sacrificing the safe warm start.

### Acceptance gates

Before replacing ResMLP, require all of the following:

- replay-verified improvement on a broader, fixed pid suite, not only the current 15 pids;
- a one-frame total near or below the PieceTransformer under the same beam/search configuration;
- controlled raw-forward and end-to-end latency on the same GPU, same precision, same batch shapes, and matched warm-up/thermal state;
- a useful quality-versus-wall-time point against wider ResMLP search, since the current project evidence shows that spending compute on ResMLP beam width is very effective.

Until then, keep epoch 1300 only as a diversity candidate or conditional reranker. Do not replace the ResMLP deployment: the recorded ResMLP Q@4M x4 result is both better (420 versus 435) and faster (2,318 versus 4,019 seconds), even before demanding a controlled latency comparison.

## Reproducibility and artifacts

- Training log: [logs/mx_generator_isab.log](logs/mx_generator_isab.log)
- Evaluation log: [logs/mx_generator_isab_eval.log](logs/mx_generator_isab_eval.log)
- Checkpoint sweep summary: [results/mx_generator_isab/checkpoint_sweep_summary.csv](results/mx_generator_isab/checkpoint_sweep_summary.csv)
- Best one-frame paths: [results/mx_generator_isab/q1m_f1_ep1300.csv](results/mx_generator_isab/q1m_f1_ep1300.csv)
- Best four-frame paths: [results/mx_generator_isab/q1m_f4_h1_ep1300.csv](results/mx_generator_isab/q1m_f4_h1_ep1300.csv)
- Best checkpoint: `models/mx_generator_isab/epoch_1300.pt`
- Final checkpoint: `models/mx_generator_isab/epoch_1500.pt`

Checkpoint hashes:

- epoch 1300: `9b247c94a716dc2ae3ecf5c43daa27e68a3822188e72c82a5a6394ab35fe851d`
- epoch 1500: `b033ef1d51aa5bc8d443a9fac9aceb6a51b2b700188f1ebc5f1af077e9fce2dd`

Independent local replay verification was run with:

```powershell
.venv\Scripts\python.exe tetraminx\scripts\53_verify_checkpoint_sweep.py `
  --results tetraminx\results\mx_generator_isab `
  --four-frame-epochs 1300
```

It confirmed all 15 paths at every 100-epoch checkpoint and all 15 four-frame epoch-1300 paths, with zero invalid paths and the totals reported above.
