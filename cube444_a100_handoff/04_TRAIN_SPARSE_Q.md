# 04 — Sparse-Q and the PieceTransformer: the open question

**This is the research.** Everything in `03` is reproduction; this is the part with 5
moves/pid behind it.

## The question, stated precisely

Vlad Kuznetsov's transformer beats our ResMLP-V beam on **916/1043 pids (87.8%)**, median
margin 6, uniformly across every difficulty band. Two things differ between his model and
ours **at the same time**:

1. **the architecture** — a piece-token transformer instead of a one-hot ResMLP
2. **the objective** — sparse-Q rw-middle labels instead of MSE on walk depth

Nobody has separated them. **Run the 2x2.** It is four training runs and it settles which
of the two is worth building on.

|  | MSE on walk depth | sparse-Q |
|---|---|---|
| **ResMLP** | = our `c_bells2` baseline (already have it) | **cell B** |
| **PieceTransformer** | cell C | **cell D** (= Vlad's configuration) |

Cell **B** is the cheap one and the most informative: if a ResMLP-Q on sparse-Q labels
captures most of the gap, the answer is "objective", and you get it at **8x the beam
width** instead of 0.24x (see throughput below). If B is flat and D wins, it is the
architecture, and you are paying a 34x width penalty for it.

Run B first. Run C only if B and D disagree.

## The sparse-Q objective

Random walk of length `k` from solved, pick a pivot at index `p`. Label **exactly two** of
the 24 action columns on the pivot state:

```
Q(s, undo_last_move) = p - 1
Q(s, next_move)      = p + 1
```

MSE on those two entries only; all other columns are unlabelled and contribute nothing.

**Why it beats walk-depth MSE.** Our V loss is `MSE(V(s), k)`. At large `k` the conditional
variance of the true distance given `s` is huge, so the Bayes-optimal prediction shrinks
toward the mean and **local discrimination collapses**. Measured on an IHES V model over
16,384 pivots: mean `V(next) - V(undo)` falls from **1.96 at depth 2 to 0.37 at depth 21**
(it should be 2 at every depth), and top-1 accuracy falls 0.87 -> 0.09.

The sparse-Q labels have **zero conditional variance in their difference** — it is always
exactly 2 — so MSE cannot buy loss by flattening the gap. The absolute level is free to
saturate; the gap is not. Same failure family as the saturation problems we hit on
megaminx, but attacked at the *label* instead of the architecture.

**Keep the absolute MSE term.** Do not replace it with a pure pairwise ranking loss: our
beam takes a **global top-B over all (parent, action) pairs**, so Q must be comparable
*across* parents. A ranking loss gives you within-parent order and nothing else.

Apply the `02_DATA.md` fix here too — the ~7.25% of pairs that assert a nonexistent gap
sit inside this objective's primary signal.

## The PieceTransformer, derived for this cube

Tokens are **physical cubies**, each token being the concatenation of embeddings of the
stickers currently in that cubie's slot, plus piece-position and piece-type embeddings.
Dense unmasked attention. Vlad's IHES config: `d_model=256, 4 layers, 8 heads, ff=1024`,
3.51M params.

**Do not hardcode the cubie layout — derive it.** `code/scripts/78_piece_model.py` already
does this for the 4x4x4 and validates against external ground truth. The derivation:

* A layer turn moves a piece entirely or not at all, so **the set of generators that move
  a facelet is constant across that facelet's piece.** Partition the 96 facelets by that
  *move signature* and you get exactly **8 triples (corners), 24 pairs (wings), 24
  singletons (centres)** — no geometry hardcoded.
* Under outer layers only, the two wings of an edge move together, so grouping wings by
  their outer-only signature gives the 12 edges.

So for a full-cube scorer: **8 + 24 + 24 = 56 tokens** (+ CLS = 57), covering all 96
facelets. Note `78_piece_model.py` *drops* centres, because it targets the reduced 3x3x3
phase where centres are uniform and colour-invisible. **For a scorer on the unreduced cube
you must keep them** — centre placement is most of what the early solve is doing.

`78_piece_model.py` validates the derived piece representation by reproducing the
published 3x3x3 QTM level counts `1, 12, 114, 1068, 10011, 93840, 878880, 8221632,
76843595`. Keep that test alive when you adapt it.

## What to port, and from where

`code/reference/` has the working tetraminx implementations:

* `tetraminx_51_train_sparse_q.py` — the sparse-Q trainer (812 lines). Runnable structure,
  sampler, objective, logging.
* `tetraminx_models.py` — `PieceTransformerQ` (line ~279), `ResMLPQ` (line ~60),
  `build_model` / `model_from_config` (line ~771).

Porting to cube444 is three edits, all of which are the same trap:

1. **`num_classes = 6`**, not `state_size`. Tetraminx is a permutation puzzle, so its
   models default `num_classes = state_size`. This is gotcha #1 and it will not error —
   it will just train a much larger, worse embedding.
2. **`output_dim = 24`** (the 24 generators).
3. **No `invert_state`.** Any inverse-frame or NISS augmentation must be stripped; it is
   not defined on a colour cube.

## Throughput: the bar the transformer has to clear

Measured on a 4090 Laptop (fp16). The A100 will be faster in absolute terms; the *ratios*
are what matter.

| scorer | child-scores/s | beam step @ B=65536 | vs V |
|---|---|---|---|
| ResMLP V (1.58M) | 12.5 M | 128.7 ms | 1.0x |
| **ResMLP Q-head** (same trunk) | ~225 M equiv | **15.5 ms** | **8.3x wider beam** |
| PieceTransformer Q (3.51M) | 2.2-2.5 M | 529 ms | **0.24x** |

A Q-head scores all 24 children with **one forward on the parent** instead of 24 forwards
on the children — that is where the 8.3x comes from, and the non-model step cost (gather +
hash + unique) is only 9 ms at B=65536, so it is not amortised away.

**The transformer must beat ResMLP-V at ~1/34 the beam width to pay for itself.** Do not
skip this arithmetic when you read the 2x2 results — a transformer that is 3 moves/pid
better at equal *epochs* may still be worse at equal *wall clock*. Compare at matched
budget, not matched steps.

Folding the transformer's input stage into one `embedding_bag` gives only 1.0-1.2x
(verified, max abs diff 6e-7). The 4 attention layers are the cost and the model already
runs at ~20 TFLOP/s, so 5x-slower-than-ResMLP is a floor, not an implementation defect.

## Honest counter-evidence

On megaminx, three training-side ranking probes (listwise rank loss, symmetry-consistency,
frontier-regret) **all tied** — V's child ordering was already near-optimal at 6M params
and the apparent "misranks" were benign alternative optima.

Those results constrain *"make V rank better"*. They do **not** constrain *"replace 24 V
forwards with 1 Q forward"*, which is the actual lever here. And they were measured on a
permutation puzzle where the scorer was *not* the bottleneck — which on this cube it
demonstrably is. Do not let them talk you out of cell B, but do expect the *ranking* part
of the win to be smaller than the *throughput* part.

## Success criteria

Report all four cells as **per-pid min against the current floor** at **matched wall
clock**, on the same pid set, with the same beam config. A cell that improves the
standalone mean but adds nothing to the merge is not a win (see `05_BEAM_SEARCH.md`).
