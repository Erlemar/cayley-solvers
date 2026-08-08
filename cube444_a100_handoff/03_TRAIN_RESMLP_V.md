# 03 — ResMLP value network: the proven recipe

This is the **known-good path**. It produced `c_bells2/epoch_0399.pt`, the model behind
our 53,426 submission. Get this working end-to-end before attempting `04`.

## The deployed model

```
ResMLPDistance, 3.29M params
  state_size    96
  num_classes   6           <- NOT 96. see 06_GOTCHAS #1
  encoding      onehot
  hidden_dims   (2048, 512)
  num_res_blocks 2
  head          single scalar (pure V, NOT AlphaZero dual-head)
```

**A 15M trunk is not better than 3.3M** — matched-epoch, identical solve rate, 3x slower.
Deploy the 3.3M. Width of the *beam* is the lever, not width of the net. (This is the
same result as megaminx Rule 14, now confirmed in the one-hot regime.)

## Stage 1 — random-walk pretrain

```bash
python cube444/scripts/02_train.py --config cube444/configs/c_v0_pretrain.yaml
```

MSE against walk depth, `k_max=60`. Run 400 epochs. This alone does not beam well; it is
a warm start for Stage 2.

## Stage 2 — Bellman refinement (this is where the quality comes from)

```bash
python cube444/scripts/03_bellman.py --config cube444/configs/c_bellman.yaml
```

`configs/c_bellman.yaml` is shipped and is the exact deployed recipe. The load-bearing
parts:

```yaml
bellman:
  warmstart_path: cube444/models/c_v0/epoch_0399.pt
  target_update_every_epochs: 10
  bfs_d6_path: cube444/data/bfs_anchors.pt
  bfs_d6_fraction: 0.10      # 10% of every batch gets EXACT labels
  n_anchor_v0: 32            # solved -> 0, in every batch
  n_anchor_d1: 4             # the 24 d=1 children -> 1, in every batch
  clip_upper: true           # walk depth is an upper bound; enforce it
  clip_lower: true
  lambda_pdb: 0.0            # no PDB on this puzzle
  frontier_fraction: 0.0     # needs solver traces; enable on a second pass
```

**`n_anchor_v0` / `n_anchor_d1` are not optional.** Without exact anchors in every batch
the Bellman bootstrap settles at `V(solved) ~ 2` — the net never learns that solved is
zero, and every beam is then steering by a broken origin.

Apply the `02_DATA.md` label fix to the walk stream feeding this. The MSE objective is
exactly the one that label noise damages most.

## What "working" looks like — gate on these, in this order

### V scale-collapse is BENIGN here. Do not gate on it.

Bellman drives `V@depth80` from 43 down to 17. On megaminx and the IHES cube that
signature meant a dead model (see the graph-transformer and dodecahedral-CNN failures).
**Here the MORE-collapsed checkpoint beams BETTER.** The absolute scale is free to
collapse; what matters is whether the net can still *discriminate* between children.

Gate on **discrimination / SNR**, never on absolute scale:

```bash
python cube444/scripts/04_eval_v.py --checkpoint <your Stage 2 checkpoint>
```

### More Bellman keeps helping long after the loss and the SNR plateau

Measured: solve rate went **12/18 -> 18/18 across 100 epochs with flat SNR**. SNR is a
*catastrophe detector*, not a progress meter. Do not early-stop on it. Use the depth sweep
and the periodic beam bench instead.

### Bench every ~50 epochs, do not trust loss

Run a small beam (see `05_BEAM_SEARCH.md`) on a fixed set of ~18 pids every 50 epochs.
Training loss is not a reliable proxy for beam quality on any puzzle in this family, and
we have been burned by it twice (a 184-epoch megaminx model with lower loss beamed
strictly worse than its 50-epoch checkpoint).

## Expected numbers

| stage | what to expect |
|---|---|
| Stage 1 @ ep400 | beams poorly; a warm start only |
| Stage 2 @ ep100 | ~12/18 on the eval set |
| Stage 2 @ ep200 | improving with flat SNR — keep going |
| Stage 2 @ ep399 | **18/18**; this is the deployed checkpoint |

If you are at ep200 and still under 10/18, something is wrong with anchors or with
`num_classes` — check both before spending more epochs.

## What was built but deliberately NOT run

**Stage 4, the AlphaZero dual-head** (`07_build_az_dataset.py`, `08_train_az.py`). Its
policy data would have come from the community CSV, and both the capacity test and the
"V-only wins" TPU finding said the policy head would not help inference. We shipped the
pure V. If you revisit it, note that using the community CSV as policy data changes the
provenance story — everything else here is self-generated.
