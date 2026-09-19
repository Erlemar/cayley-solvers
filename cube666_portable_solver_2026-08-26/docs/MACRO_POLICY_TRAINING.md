# 6x6x6 stage-specific macro-policy training

## Autonomous beam outcome

The factorized policy/value model now solves normalized 6x6x6 bulk states on
its own with beam search.  After deterministic exact corner and parity
normalization, every remaining macro decision is made by the learned policy
and learned value.  Exact permutation code is used only to apply an action,
detect the identity goal, and replay-verify the final 216-sticker path.  The
solver does not call KMCoders, classical greedy bulk reduction, or an exact
finisher controller.

The key architecture change was to exploit the exact decomposition into six
independent 24-piece clusters.  One shared local residual MLP is applied to all
six clusters.  It predicts 4,048 directed 3-cycle logits per cluster and one
cluster-distance value; the six policy blocks are concatenated and the six
values are summed.  This replaced a monolithic 144-piece representation that
fit its training walks but did not guide an independent beam reliably.

Independent synthetic controls at exact macro depths 1, 17, 34, 51, and 68
all solved with beam 128 / branch 16.  The first four used the exact minimum
number of macros; depth 68 used 70.  All five paths replay-verified.

Held-out competition puzzles also passed the full end-to-end gate:

| State | Initial exact macro cost | Learned macro steps | Primitive moves | Full replay |
| --- | ---: | ---: | ---: | --- |
| 200 | 64 | 69 | 709 | solved |
| 210 | 66 | 70 | 711 | solved |
| 220 | 65 | 69 | 723 | solved |

The local checkpoint is
`models/cube666_finisher_factorized_v1/checkpoint.pt`.  Its independent beam
reports are:

- `models/cube666_finisher_factorized_v1/beam_synthetic_b128.json`
- `models/cube666_finisher_factorized_v1/beam_puzzles_200_210_220_b128.json`

A compatible larger follow-up was trained in the paired marimo notebook on the
95 GiB NVIDIA RTX PRO 6000 Blackwell Server Edition.  The notebook reconstructs
the same canonical action order from first principles and stores its checkpoint
at `/marimo/storage/cube666_finisher_factorized_pro6000_v1.pt`.  That run used
250,000 local geodesic states, width 384, four residual blocks, batch 4,096,
and 12,000 updates.  It completed in 19.29 seconds after compilation and reached:

- independent cluster-value MAE: 0.4991 macro moves;
- top-64 exact reducing-action coverage: 100% over 2,000 independent states;
- autonomous learned-beam solve rate: 5/5 at total depths 1, 17, 34, 51, 68;
- beam step counts: 1, 17, 34, 51, 70 respectively.

The 15.84 MiB checkpoint was downloaded to
`models/cube666_finisher_factorized_pro6000_v1/checkpoint.pt`.  Its SHA-256 is
`b09663eef1b669948f58103e1753594206b9460bc0d13aa1c5763c5be1dc2bd2`,
and its action-library digest is
`07680841ce213d77d3525fd3bff581d4188b9dac2257283524a55c8ac9eb32b5`.

### Full autonomous acceptance gate

The Pro 6000 model passed an independent 1,000-state synthetic gate in the
marimo kernel: 1,000/1,000 solved and replay-verified, mean initial exact cost
33.838, mean learned path length 34.280, maximum overhead four macros, and
86.667 seconds elapsed.

On 25 stratified full competition puzzles, both compatible checkpoints solved
and replay-verified every case with beam 128 / branch 16 / depth 80.  The Pro
model was better on both path length and runtime:

| Checkpoint | Solved | Mean macro steps | Mean primitive moves | Max macro overhead | Elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local width-256 | 25/25 | 69.88 | 723.12 | 9 | 42.26 s |
| Pro 6000 width-384 | 25/25 | 68.24 | 704.56 | 7 | 21.79 s |

The complete 1,012-puzzle sweep found a small hard tail at the fast setting:
1,009/1,012 solved and replay-verified.  Puzzles 7, 440, and 879 all stopped at
residual exact cost 12.  A wider rescue (beam 512 / branch 32 / depth 112)
solved all three in 66, 68, and 68 macros respectively, with full 216-sticker
replay verification.  Combined coverage is therefore **1,012/1,012**.

The runner now performs this two-tier retry autonomously.  Its adaptive gate
was exercised on one ordinary puzzle plus all three hard puzzles: 4/4 solved,
with rescue invoked only for the three failures.  Evidence is stored in:

- `models/cube666_finisher_factorized_pro6000_v1/beam_puzzles_all1012_b128.json`
- `models/cube666_finisher_factorized_pro6000_v1/beam_puzzles_rescue3_d112_b512_br32.json`
- `models/cube666_finisher_factorized_pro6000_v1/beam_puzzles_adaptive_gate4.json`

Use the accepted autonomous configuration with:

```powershell
.venv\Scripts\python.exe cube666\scripts\16_run_finisher_policy_beam.py `
  --checkpoint models\cube666_finisher_factorized_pro6000_v1\checkpoint.pt `
  --action-library cube666\training\finisher_policy_v1\action_library.json `
  --all-puzzles --beam-width 128 --branch-width 16 --max-steps 80 `
  --rescue-beam-width 512 --rescue-branch-width 32 --rescue-max-steps 112 `
  --policy-nll-weight 0.02 --progress-every 50 --summary-only `
  --out models\cube666_finisher_factorized_pro6000_v1\beam_puzzles_adaptive_all1012.json
```

### Reproduce the autonomous local baseline

```powershell
.venv\Scripts\python.exe cube666\scripts\15_build_finisher_policy_teacher.py `
  --train-samples 100000 --eval-samples 5000 --max-depth 68 `
  --out-dir cube666\training\finisher_policy_v1

.venv\Scripts\python.exe cube666\scripts\09_train_macro_policy.py `
  --teacher cube666\training\finisher_policy_v1\train.npz `
  --action-library cube666\training\finisher_policy_v1\action_library.json `
  --out-dir models\cube666_finisher_factorized_v1 `
  --architecture factorized --steps 10000 --batch-size 512 `
  --hidden-dim 256 --residual-blocks 3 `
  --value-weight 20 --cluster-value-weight 20

.venv\Scripts\python.exe cube666\scripts\16_run_finisher_policy_beam.py `
  --checkpoint models\cube666_finisher_factorized_v1\checkpoint.pt `
  --action-library cube666\training\finisher_policy_v1\action_library.json `
  --puzzle-indices "200,210,220" `
  --beam-width 128 --branch-width 16 --max-steps 80
```

## Earlier KMC-prefix outcome

A first replay-verified macro-policy model has been trained.  It is useful as a
short KMC-derived prefix before the existing exact greedy bulk phase; it is not
yet a replacement for that phase.

The three-puzzle held-out end-to-end pilot improved the established classical
greedy pipeline by **12.67 moves per puzzle on average**:

| State | Classical greedy | Learned prefix + greedy | Delta |
| --- | ---: | ---: | ---: |
| 200 | 381 | 371 | -10 |
| 210 | 373 | 347 | -26 |
| 220 | 367 | 365 | -2 |
| **Mean** | **373.67** | **361.00** | **-12.67** |

Every path was replayed against all 216 stickers and solved exactly.

## What was trained

- Stage: corner-solved, even-parity bulk center/wing reduction only.
- Input: six exact 24-piece cluster permutations, losslessly one-hot encoded.
- Policy: direct logits over an auditable vocabulary of 1,070 macros.
- Auxiliary targets: total and per-cluster exact unrestricted 3-cycle costs.
- Inference: model top-256 followed by exact algebraic re-ranking.  A prediction
  is never applied without exact scoring.
- Architecture: 3-block residual MLP, width 384.
- Training: 1,500 fixed-shape updates, batch 256, bf16, `torch.compile`, fused
  AdamW on the local RTX 4090 Laptop GPU.

The checkpoint is `models/cube666_kmc_macro_policy_grouped_v1/checkpoint.pt`.

## Teacher construction

Synthetic random-walk/exact-greedy labels were tested first and rejected on an
independent seed.  They did not generalize to unseen trajectories.

The successful teacher is extracted from replayed KMCoders paths.  Consecutive
points where corners are solved and all six physical clusters have even parity
define a legal bulk macro.  Only segments that strictly reduce the exact
residual by at least two units and use at most 20 primitive moves are retained.
Every macro is replay-analyzed; its inverse is added to keep the action table
closed.

Current corpus:

- 275 stdout logs scanned;
- 269 valid paths replayed;
- 762 improving transitions;
- 736 distinct normalized states;
- 1,070 unique macro effects including inverses.

Artifacts are in `cube666/training/kmc_macro_teacher_v1/`.

## Leakage-safe evaluation

Validation holds out complete source puzzle IDs by `state_id % 10`, rather than
random rows from the same trajectory.  On fold 0 (137 states):

- top-64 retained 94.55% of the demonstrated KMC residual reduction;
- top-64 reduced 99.27% of states;
- top-64 median regret was 0;
- top-256 achieved 106.08% of the demonstrated reduction on average;
- top-256 multi-step rollout reduced exact residual from 40.96 to 19.36 on
  average before the classical greedy/finisher phases.

The demonstration gate passed.  The full report is
`models/cube666_kmc_macro_policy_grouped_v1/eval_fold0_rollout.json`.

## Reproduce

```powershell
.venv\Scripts\python.exe cube666\scripts\13_extract_kmc_macro_teacher.py `
  --out-dir cube666\training\kmc_macro_teacher_v1

.venv\Scripts\python.exe cube666\scripts\09_train_macro_policy.py `
  --action-library cube666\training\kmc_macro_teacher_v1\action_library.json `
  --teacher cube666\training\kmc_macro_teacher_v1\teacher.npz `
  --group-ids cube666\training\kmc_macro_teacher_v1\source_state_ids.npy `
  --validation-fold 0 --validation-folds 10 `
  --out-dir models\cube666_kmc_macro_policy_grouped_v1 `
  --steps 1500 --batch-size 256 --hidden-dim 384 --residual-blocks 3

.venv\Scripts\python.exe cube666\scripts\14_solve_macro_policy_pilot.py `
  --checkpoint models\cube666_kmc_macro_policy_grouped_v1\checkpoint.pt `
  --action-library cube666\training\kmc_macro_teacher_v1\action_library.json `
  --indices "200,210,220" --top-k 256
```

## Next gate

Do not run the model over all 1,012 puzzles yet.  First:

1. Run all ten group-held-out folds and at least 25 end-to-end held-out puzzles.
2. Append hybrid rollout states with exact re-labeling (DAgger), then retrain.
3. Expand to at least 5,000 distinct normalized teacher states across KMC seeds.
4. Require replay validity, mean hybrid improvement of at least 5 moves versus
   classical greedy, and no regression on at least 80% of held-out puzzles.
5. Only after that gate, solve all puzzles and compare a verified merged
   submission against the current best.

The old primitive-q666 achieved/W window experiment remains blocked because its
checkpoint/archive is neither local nor visible in the connected Drive.  It is
independent of this macro-policy path and should remain secondary unless those
artifacts are recovered.
