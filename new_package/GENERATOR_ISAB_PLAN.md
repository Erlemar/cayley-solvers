# Residual Generator-ISAB: Architecture and Training Plan

## Goal

Close the path-quality gap between the Tetraminx ResMLP and PieceTransformer while
retaining inference much closer to the ResMLP. The deployed scorer is one standalone
model. A teacher is not required at inference and is deliberately excluded from the
first controlled training run.

## Why this differs from the failed latent model

The previous latent model compressed 50 physical pieces to eight latent tokens before
the pieces could exchange relational information. All action scores were then decoded
from that shared bottleneck. It discarded piece-specific detail early, had much less
capacity than either baseline, and represented actions with learned IDs rather than
the exact generator geometry.

Generator-ISAB preserves all 50 piece tokens through the relational stack. Each block
uses 12 inducing tokens as a low-rank communication channel:

1. inducing tokens attend to all physical pieces;
2. every physical piece attends back to the updated inducing tokens;
3. the updated 50-token set continues to the next block.

This costs `O(P * L)` attention per block rather than `O(P^2)`, with `P=50` and
`L=12`, but does not make the latent bottleneck the only retained state.

## Model

The complete output is

`Q(s, a) = Q_resmlp(s, a) + delta_Q_relational(s, a)`.

### Base path

The base is byte-compatible with the trained `tq0` ResMLP: 88 categorical facelets,
16-dimensional embeddings, widths 2048 and 512, two residual blocks, and 24 Q outputs.
Its module names and tensor shapes are unchanged, so checkpoint loading needs no
translation.

### Piece path

The verified block-system layout converts 88 facelets into 50 physical pieces. A
piece token combines its ordered sticker values, physical position, and piece type.
The first run uses width 128, four heads, two induced-attention blocks, 12 inducing
tokens, and width-256 feed-forward sublayers.

### Generator-aware readout

For every one of the 24 actions the model stores the exact gather permutation from
`puzzle_info.json`. Its action query is constructed from learned source/destination
facelet factors over the moved facelets. This distinguishes inverse generators and
within-piece orientation changes. Each action query attends only to the physical
pieces moved by that generator, then receives a global induced-token summary.

The scalar correction head is initialized to exactly zero. After loading `tq0`, the
whole model therefore reproduces `tq0` exactly before training.

## Phase 1: controlled architecture test

- Warm-start from `tetraminx/models/tq0/epoch_1500.pt`.
- Freeze the ResMLP base.
- Train only the relational correction for 1,500 epochs.
- Match the earlier sparse-Q data recipe: non-backtracking random walks of length
  2-40, depth tilt 0.5, symmetry-expanded labels, exact depth-5 anchors, batch 2048,
  256 steps per epoch.
- Use bf16 autocast, fixed-shape compilation, fused AdamW, learning rate `3e-4`,
  weight decay `3e-3`, and gradient clipping at 1.0.
- Save every 50 epochs.

This phase isolates the architecture. Because the base is fixed and corrections start
at zero, the model cannot lose ResMLP information representationally, although an
over-trained correction can still worsen ranking. Checkpoint selection must therefore
use search-relevant validation, not final-epoch training loss.

## Evaluation and selection

Run the same 15 pids used for all prior comparisons:

`0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 990, 995, 999`.

For checkpoints every 100 epochs:

1. run the fixed 262,144-sample Q probe;
2. run replay-verified one-frame Q@1M search;
3. select by total valid path length, not validation MSE;
4. run four-frame Q@1M on the best checkpoint;
5. compare with ResMLP 468, PieceTransformer 441, and merged-best 423.

Also benchmark raw model-forward throughput and end-to-end search wall time using the
same hardware, precision, batching, and beam settings.

## Phase 2 only if the architecture plateaus

The next controlled change is frontier-matched dense supervision. Collect states from
actual ResMLP/Generator-ISAB beam frontiers, label all 24 successors with a stronger
offline scorer or search-backed targets, and fine-tune the relational correction on a
mixture of random-walk, exact-anchor, and frontier batches. This is optional training
data generation; the resulting Generator-ISAB remains a standalone inference model.
