# Frontier-regret and almost-search-free pilot (2026-08-08)

## Outcome

The useful v0 is **not search-free**, but it establishes a real hybrid regime:

1. a frozen-trunk, separately trained 24-way policy greedily commits 0--5 moves;
2. a confidence gap of 0.20--0.25 triggers one 16K beam handoff;
3. the untouched deployed Transformer/ResMLP blend performs the recovery;
4. the existing exact d<=6 table closes the path.

On 12 unseen pids (450, 500, ..., 950, 999), one frame, identical hardware:

| method | solved | moves on 9 common solves | extra solve | wall |
|---|---:|---:|---:|---:|
| deployed blend, full 16K beam | 9/12 | 301 | -- | 148 s |
| learned policy, gap 0.25, one 16K handoff | **10/12** | 310 (+9) | pid 500 in 33 | **112 s** |

Thus the hybrid gained one solve and reduced wall about 24%, but cost about one
move per common solved puzzle.  It is a compute/coverage lead, **not yet a path-
quality or submission win**.

## What was implemented

- `tetraminx/src/tetraminx/search_free.py`
  - adjusted-Q policy scoring;
  - bounded root-action lookahead and regret targets;
  - greedy rollout with cycle/no-backtrack protection;
  - confidence-triggered local lookahead;
  - confidence-triggered compact-beam goal handoff;
  - separate policy and recovery solvers.
- `tetraminx/scripts/70_build_frontier_regret.py`
  - balanced greedy / uncertain-frontier / beam-cutoff harvesting;
  - bounded-lookahead targets;
  - verified full-beam multi-positive root-action targets;
  - exact-endgame override and replay checks;
  - optional student-policy roll-ins for later DAgger rounds.
- `tetraminx/scripts/71_train_frontier_policy.py`
  - listwise regret/action-set policy loss;
  - symmetry transport;
  - puzzle-level validation split;
  - random-walk source-checkpoint preservation;
  - frozen-trunk/head-only policy fitting.
- `tetraminx/scripts/30_solve.py`
  - `--greedy-recovery` mode;
  - local or goal-handoff recovery;
  - `--policy-checkpoint` separation from the recovery scorer.
- `src/cayley/khoruzhii_search.py`
  - optional propagation of successful goal states to their depth-1 root actions.
- `tests/test_tetraminx_search_free.py`
  - root-cost masking and replayable greedy-goal regression tests.

## Experiments

### 1. Bounded-lookahead regret: rejected as the teacher

A balanced 360-state dataset used a 1,024-wide, four-step teacher.  The deployed
blend's teacher-best action was top-1 only 16.7% and top-4 46.4%, but repeated
model-predictive rollout did not solve.  Even B=4,096/H=8 failed pid 50 after
changing 27 of 45 greedy decisions.

Fine-tuning on these labels worsened held-out regret (1.113 -> best trained 1.160),
and a 16K beam comparison was byte-for-byte neutral: 5/6, 169 moves for both
baseline and trained scorer.  Short-horizon self-distillation changes decisions
without supplying valid long-horizon credit.  Do not scale this target.

Artifacts:

- `tetraminx/data/frontier_regret_v0.pt`
- `tetraminx/models/frontier_policy_v0/`

### 2. Verified full-beam root labels: useful

A 16K full teacher solved 22/24 queried states from eight pids.  Nineteen rows had
one successful root action and three had two.  The deployed blend's local top-1
matched no successful root; its mean successful-root rank was 9.82.  Teacher
solution costs ranged 28--37.

Full-network fine-tuning improved only one of five held-out labels and was neutral
at 16K beam, so it was rejected.  A frozen representation with only the 6,168-
parameter action head trained produced a small but usable policy; epoch 30 was the
first checkpoint with held-out top-4 0.40 versus 0.20 at baseline.

Artifacts:

- `tetraminx/data/frontier_verified_v0.pt`
- `tetraminx/models/frontier_verified_policy_v0/` (full fine-tune; rejected)
- `tetraminx/models/frontier_verified_policy_head_v0/epoch_0030.pt` (policy v0)

### 3. Confidence/handoff sweep

First unseen six pids (450--700):

| mode | solved | moves | wall | verdict |
|---|---:|---:|---:|---|
| full 16K beam | 5/6 | 169 | 85 s | control |
| original policy, gap .35, 16K handoff | 5/6 | 172 | 91 s | no gain |
| learned policy, gap .35, 16K handoff | 5/6 | 171 | 59 s | faster, same coverage |
| learned policy, gap .20/.25, 16K handoff | **6/6** | 207 | **49 s** | best hybrid |
| learned policy, gap .10, 16K handoff | 4/6 | 159 on solved | 56 s | greedy prefix too long |
| learned policy, gap .25, 8K handoff | 5/6 | 181 | 33 s | recovery too narrow |

The useful point is an early 0--5-move prefix followed by 16K recovery.  Longer
autonomous prefixes are unsafe with v0.

## Recommended next experiment

Run one genuine DAgger expansion, not more epochs:

1. Select at least 40 new pids disjoint from both the eight teacher pids and the
   12 evaluation pids.
2. Use `epoch_0030.pt` as `--rollin-checkpoint`.
3. Query the first gap<=0.25 state, plus one pre-trigger state and one post-trigger
   or beam-cutoff state per pid.
4. Label with verified 16K root actions.  Keep failed teacher searches as missing,
   never negative labels.
5. Refit only the action head and select by puzzle-held-out successful-root top-4.
6. Re-run the fixed 12-pid control plus the 40-pid ablation set at 8K/16K.

Scale to full inference only if the next model improves one of:

- coverage at equal 16K work without increasing common-path moves;
- equal coverage at 8K;
- common-path length at the current ~24% wall reduction.

Do not submit or replace the deployed beam from this v0 result.

## Reproduction commands

Verified DAgger harvest (illustrative next round):

```powershell
.venv\Scripts\python.exe -u tetraminx\scripts\70_build_frontier_regret.py `
  --checkpoint tetraminx\models\mx_tf_az\epoch_1500.pt `
  --blend tetraminx\models\mx_resmlp_az\best.pt --blend-weights 0.8 0.2 `
  --rollin-checkpoint tetraminx\models\frontier_verified_policy_head_v0\epoch_0030.pt `
  --out tetraminx\data\frontier_verified_v1.pt --pids <DISJOINT_PIDS> `
  --max-states-per-pid 3 --verified-teacher-beam 16384 `
  --verified-teacher-steps 45 --history-depth 1 --qv-consistency 0.3 --bf16
```

Best v0 hybrid:

```powershell
.venv\Scripts\python.exe -u tetraminx\scripts\30_solve.py `
  --checkpoint tetraminx\models\mx_tf_az\epoch_1500.pt `
  --blend tetraminx\models\mx_resmlp_az\best.pt --blend-weights 0.8 0.2 `
  --policy-checkpoint tetraminx\models\frontier_verified_policy_head_v0\epoch_0030.pt `
  --greedy-recovery --confidence-gap 0.25 --recovery-to-goal `
  --recovery-beam 16384 --max-handoffs 1 --history-depth 1 `
  --qv-consistency 0.3 --bf16 --out <OUTPUT.csv>
```
