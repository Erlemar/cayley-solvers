# 07 — AlphaZero dual head (built, never run)

Stage 4. The code is complete and shipped here; **it was never actually run** on this
puzzle. Read `03_TRAIN_RESMLP_V.md` first — this builds on a trained V trunk.

## Why it was not run, and why that is not a verdict

We shipped the pure V because two findings said a policy head would not help *inference*:
the capacity test, and "V-only wins on TPU". Neither of those is a measurement of the AZ
head itself — nobody trained one. So this is **untested, not rejected**. It is a genuine
open lever, and unlike the transformer question it is cheap: one dataset build plus one
fine-tune from an existing trunk.

Be aware of the provenance change, though. Everything else in this pipeline is
self-generated + exact. **The policy targets come from a community solution CSV** — human
and other-solver moves enter training here for the first time. That is fine for the
competition, but it means results from this stage are not comparable to the "zero external
data" story the rest of the work carries. Say so when you report it.

## Step 1 — build the policy dataset

```bash
python cube444/scripts/07_build_az_dataset.py \
  --submission cube444/submissions/cube4_submission_46662.csv \
  --out cube444/data/az_dataset_46662.pt
```

The community file is shipped at `code/submissions/cube4_submission_46662.csv`.
**Verified on the origin machine: 1043/1043 valid, 46,662 moves, mean 44.738, VERDICT
PASS.** It is the strongest cube444 solution we have seen — better than the 46,718 quoted
elsewhere in this package, and only 364 above Rokicki's 46,298.

Each pid contributes `path_len` training tuples: at step `i` along the realized solution
the state is the scramble with the first `i` moves applied, the action is the move taken
at step `i`, and the value target is `path_len - i`. So this file yields ~46,662 pairs —
comparable to the megaminx `az_dataset_76304.pt` that produced the AZ v4 breakthrough
there.

**There is no chicken-and-egg**: Stage 4 can run before you have any beam output of your
own. Once your own solver beats the community per-pid, rebuild the dataset from your own
CSV and re-run.

`--sym-augment K` expands the set K-fold through the rotation group
(`state -> sym(state, R_k)`, `action -> move_relabel[k, action]`). It is **off by
default and should stay off** unless you are testing it deliberately: on megaminx,
symmetry augmentation on the V head was measured to *dilute* the distance signal
(m31, rejected). It belongs on a Q-shortlister, not a value head.

## Step 2 — train the dual head

```bash
python cube444/scripts/08_train_az.py \
  --output cube444/models/c_az_v0 \
  --warmstart-trunk cube444/models/c_bells2/epoch_0399.pt \
  --policy-dataset cube444/data/az_dataset_46662.pt \
  --bfs-anchors cube444/data/bfs_anchors.pt \
  --epochs 100 --rw-batch-size 8192 --policy-batch-size 1024 \
  --anchor-v0 32 --anchor-d1 4 --bfs-fraction 0.10 --k-max 60
```

**The batch sizes are load-bearing — do not accept the script defaults blindly.** On
megaminx, AZ v3 was launched at `8192 / 1024` and a "matched" AZ v4 relaunch that took the
defaults (`4096 / 512`) got **4x the gradient steps per epoch**, memorised the policy
(top-1 99% by ep99), and destroyed V calibration. Echo the resolved config at startup and
compare it against the previous run's log header before calling two runs comparable.

Keep `--anchor-v0 32 --anchor-d1 4`: the same exact-anchor requirement as Stage 2, for the
same reason (`V(solved)` drifts to ~2 without them).

## What to check

The value head pays a measured **8-16% penalty** versus a pure V of the same size on
megaminx — that is the expected cost of sharing a trunk, not a bug. So:

1. Gate the **value head** exactly as in `03`: discrimination, not absolute scale; beam
   bench every ~50 epochs; do not early-stop on SNR.
2. Judge the **policy head** by whether it improves the *beam*, not by top-1 accuracy.
   High top-1 on community moves means it learned to imitate that solver, which is not
   the same as searching better. A policy at 99% top-1 is a memorisation warning.
3. Compare against the pure V at **matched wall clock** and as a **per-pid min against
   the floor** (`05_BEAM_SEARCH.md`) — never as a standalone mean.

If the dual head does not beat the pure V on the merge, that is a real answer and worth
recording: it converts an assumption we shipped on into a measurement.
