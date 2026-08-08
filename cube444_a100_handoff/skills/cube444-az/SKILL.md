---
name: cube444-az
description: Build the AlphaZero policy dataset from a solution CSV and train the cube444 dual head. Use when asked to train an AZ model, a policy head, or to use community/solver solutions as policy targets for the 4x4x4 colour cube.
---

# cube444 — AlphaZero dual head

Full detail in `07_TRAIN_AZ.md`. **Built but never run** on this puzzle — untested, not
rejected. Cheap: one dataset build plus one fine-tune from an existing V trunk.

## Step 1 — policy dataset

```bash
python cube444/scripts/07_build_az_dataset.py \
  --submission cube444/submissions/cube4_submission_46662.csv \
  --out cube444/data/az_dataset_46662.pt
```

Shipped at `code/submissions/cube4_submission_46662.csv`. Verified 1043/1043,
**46,662 moves, mean 44.738, VERDICT PASS** — the best cube444 solution we have, 364
above Rokicki.

Each pid yields `path_len` tuples (state after `i` moves, action at step `i`, value
`path_len - i`). No chicken-and-egg: this runs before you have any beam output. Rebuild
from your own CSV once your solver beats it per-pid.

**Leave `--sym-augment` off** unless testing it deliberately — symmetry augmentation on a
value head was measured to dilute the distance signal on megaminx (m31, rejected).

## Step 2 — train

```bash
python cube444/scripts/08_train_az.py \
  --output cube444/models/c_az_v0 \
  --warmstart-trunk cube444/models/c_bells2/epoch_0399.pt \
  --policy-dataset cube444/data/az_dataset_46662.pt \
  --bfs-anchors cube444/data/bfs_anchors.pt \
  --epochs 100 --rw-batch-size 8192 --policy-batch-size 1024 \
  --anchor-v0 32 --anchor-d1 4 --bfs-fraction 0.10 --k-max 60
```

**Pass the batch sizes explicitly.** Script defaults are `4096 / 512`; on megaminx a
"matched" relaunch that took the defaults got 4x the gradient steps per epoch, memorised
the policy (top-1 99%) and destroyed V calibration. Echo the resolved config and diff it
against the prior run's log header before calling two runs comparable.

`--anchor-v0 32 --anchor-d1 4` are not optional (`V(solved)` drifts to ~2 without them).

## Judging it

* Value head: expect an **8-16% penalty** vs a pure V of the same size — that is the cost
  of a shared trunk, not a bug. Gate on discrimination, bench a beam every ~50 epochs,
  do not early-stop on SNR.
* Policy head: judge by whether the **beam** improves, never by top-1 accuracy. High top-1
  on community moves means it imitates that solver; **99% top-1 is a memorisation
  warning**.
* Compare against the pure V at **matched wall clock**, as a **per-pid min against the
  floor**.

## Provenance caveat — state it when reporting

Everything else in this pipeline is self-generated + exact. The policy targets here come
from a community solution, so external moves enter training for the first time. Results
from this stage are not comparable to the "zero external data" story the rest of the work
carries.
