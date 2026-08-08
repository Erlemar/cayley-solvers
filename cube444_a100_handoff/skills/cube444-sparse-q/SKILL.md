---
name: cube444-sparse-q
description: Run the sparse-Q / PieceTransformer 2x2 experiment for cube444 — the open question of whether the transformer's win is its architecture or its training objective. Use when asked to train a Q-head, a transformer, or to settle the architecture-vs-objective question on the 4x4x4 colour cube.
---

# cube444 — sparse-Q and PieceTransformer (the 2x2)

Full detail in `04_TRAIN_SPARSE_Q.md`. This is the experiment plan.

## The question

A transformer beats our ResMLP-V beam on **916/1043 pids (87.8%)**, median margin 6,
uniformly across all difficulty bands — ~5 moves/pid. Two things differ at once: the
**architecture** and the **sparse-Q objective**. Nobody separated them.

|  | MSE on walk depth | sparse-Q |
|---|---|---|
| ResMLP | baseline, already have it | **cell B — run first** |
| PieceTransformer | cell C | cell D |

**Run B first.** If a ResMLP-Q on sparse-Q labels captures most of the gap, the answer is
"objective" and you get it at **8.3x the beam width**. If B is flat and D wins, it is the
architecture — at **0.24x** width. Run C only if B and D disagree.

## The objective

Walk length `k` from solved, pivot at index `p`. Label exactly two of the 24 columns:

```
Q(s, undo_last) = p - 1
Q(s, next_move) = p + 1
```

MSE on those two only. **Keep the absolute MSE term** — the beam takes a global top-B over
all (parent, action) pairs, so Q must be comparable across parents; a pure ranking loss
does not give that.

Why it beats depth-MSE: at large `k` the conditional variance of true distance is huge, so
MSE shrinks toward the mean and local discrimination collapses (measured: mean
`V(next)-V(undo)` falls 1.96 -> 0.37 from depth 2 to 21). The sparse-Q difference has
**zero conditional variance** — always exactly 2 — so the gap cannot be flattened for loss.

Apply the `02_DATA.md` label fix; the ~7.25% of pairs asserting a nonexistent gap sit
inside this objective's primary signal.

## Port from

* `code/reference/tetraminx_51_train_sparse_q.py` — working trainer (sampler, objective)
* `code/reference/tetraminx_models.py` — `PieceTransformerQ`, `ResMLPQ`, `build_model`

Three edits, all the same trap:
1. **`num_classes = 6`** (tetraminx defaults it to `state_size` — silent, builds a worse model)
2. `output_dim = 24`
3. Strip anything using `invert_state` — undefined on a colour cube

## Piece tokens: derive, never hardcode

`code/scripts/78_piece_model.py` derives the cubie partition from the 24 move
permutations alone: facelets grouped by **move signature** give exactly 8 corner triples,
24 wing pairs, 24 centre singletons. For a full-cube scorer keep all three -> **56 tokens
+ CLS**. (`78_piece_model.py` drops centres because it targets the reduced 3x3x3 phase;
for an unreduced scorer that is wrong — centre placement is most of the early solve.)

It validates against published 3x3x3 QTM level counts
`1, 12, 114, 1068, 10011, 93840, 878880, 8221632, 76843595`. Keep that test alive.

## The bar the transformer must clear

| scorer | beam step @ B=65536 | vs V |
|---|---|---|
| ResMLP V | 128.7 ms | 1.0x |
| ResMLP Q-head | 15.5 ms | **8.3x wider beam** |
| PieceTransformer Q | 529 ms | **0.24x** |

A Q-head scores all 24 children from **one parent forward**. The transformer must beat
ResMLP-V at **~1/34 the beam width** to pay for itself. **Compare at matched wall clock,
not matched epochs** — a transformer 3 moves/pid better per epoch can still lose per hour.

## Reporting

All four cells as **per-pid min against the floor**, matched wall clock, same pid set,
same beam config. A cell that lifts the standalone mean but adds nothing to the merge is
not a win.
