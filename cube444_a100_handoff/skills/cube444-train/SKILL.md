---
name: cube444-train
description: Train a cube444 value network from scratch on this A100 — Stage 1 random-walk pretrain then Stage 2 Bellman refinement, with the label-conflict fix and the acceptance gates. Use when asked to train, retrain, or continue training a V model for the 4x4x4 colour cube.
---

# cube444 — train a V model

Reproduces the recipe behind `c_bells2/epoch_0399.pt` (3.29M ResMLP V, the model behind
our 53,426 submission). Full detail in `03_TRAIN_RESMLP_V.md`; this is the operating loop.

## Before the first run, once

```bash
python cube444/scripts/test_symmetry.py        # MUST print SYMMETRY TABLES OK
python cube444/scripts/20_label_fix.py --self-test
python cube444/scripts/01_build_bfs.py \
  --out cube444/data/bfs_anchors.pt --max-exact 5 --d6-sample 20000000
```

Assert the level sizes `1, 24, 552, 12144, ...`. Wrong d1/d2 = wrong generator
convention; stop.

## Stage 1 — pretrain

```bash
python cube444/scripts/02_train.py --config cube444/configs/c_v0_pretrain.yaml
```

400 epochs. Beams poorly on its own; it is a warm start only. Do not evaluate it as a
solver.

## Stage 2 — Bellman (this is where quality comes from)

```bash
python cube444/scripts/03_bellman.py --config cube444/configs/c_bellman.yaml
```

Check before launching:
* `warmstart_path` points at the Stage 1 checkpoint you actually produced
* `bfs_d6_path` exists
* `n_anchor_v0: 32`, `n_anchor_d1: 4` — **not optional**, without them `V(solved)` drifts
  to ~2 and every beam steers by a broken origin
* `num_classes: 6` — see gotcha #1

Echo the resolved config at startup and read it back. Do not trust script defaults when
"matching" a previous run.

## Acceptance gates, in this order

1. **Do NOT gate on V scale.** Bellman drives `V@d80` 43 -> 17 and the *more* collapsed
   checkpoint beams *better* on this puzzle. This is a genuine exception to the saturation
   rule from megaminx/IHES.
2. Gate on discrimination: `python cube444/scripts/04_eval_v.py --checkpoint <ckpt>`
3. **Bench a real beam every ~50 epochs** on a fixed ~18-pid set. Loss is not a proxy —
   a 184-epoch megaminx model with lower loss beamed strictly worse than its 50-epoch one.
4. **Do not early-stop on SNR.** Solve rate went 12/18 -> 18/18 across 100 epochs with
   flat SNR. SNR is a catastrophe detector, not a progress meter.

Expected: ~12/18 at ep100, 18/18 at ep399.

## If you changed the objective or the labels

Run the **matched control** — same seed, epochs, config, everything except the one
variable. Quote the delta only against that. An objective change (including the
`02_DATA.md` label fix) takes effect on a fresh run only; never warm-start from an
unfixed checkpoint and attribute the difference to the fix.

## Launch long runs detached

Training runs outlive an interactive shell. Launch with `nohup ... &` writing to a log,
checkpoint every 25 epochs, and poll the log — do not hold the session open.
