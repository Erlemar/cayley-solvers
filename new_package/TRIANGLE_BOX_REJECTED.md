# Permutation-invariant constraint losses on the 24 Q entries — REJECTED (family)

Two variants built and measured, `tq5_box` and `tq6_profile`. Both lose to the plain
`tq1` control at every pivot depth. The second one identifies a mechanism that
closes the whole family, so this is a category result rather than two data points —
see "The category result" at the bottom.

---

## Variant 1: the triangle-inequality box

Measured 2026-07-31. Treatment `tq5_box` vs control `tq1_sparse_q_paths`, configs
identical apart from the `box_*` block, same seed, run concurrently to stay
checkpoint-matched. **Killed at epoch 50: worse than the control at every pivot
depth, with a mechanism that explains why the premise was wrong.**

## The idea

`PathLabels` supervises exactly **one of 24** entries per state; the other 23 are
unconstrained at every depth, which is where the deep-band gap collapses (0.216 in
the 30-40 band on tq0). But for any state and any action the triangle inequality
gives, for free,

```
d(s) - 1  <=  d(s.a)  <=  d(s) + 1
```

and a path label hands us `d(s)`. So all 24 entries can be pinned to a
depth-dependent interval using only a fact that is certainly true, and — unlike a
ranking penalty — asserting nothing about *which* action is best, which is what tied
`m_rank_v0` on megaminx.

Implemented as two quadratic hinges on the path rows (`box_upper_weight`,
`box_lower_weight`, `box_slack`), with the lower one softer because path lengths are
upper bounds on `d(s)`.

## Pre-flight: the constraint is real

Verified against the exact d<=7 table before any training, ~370k children:

| d(s) | at d-1 | at d | at d+1 | outside box |
|---|---|---|---|---|
| 1 | 0.042 | 0.042 | 0.917 | 0 |
| 4 | 0.056 | 0.057 | 0.887 | 0 |
| 7 | 0.067 | 0.068 | 0.865 | 0 |

Zero violations at every depth, and ~89% of children sit at exactly `d+1`. The
constraint is sound and its interior is heavily skewed. The premise checked out;
the *inference from it* did not.

## Result: worse at every depth

`52_eval_q.py`, 16,384 rw-middle pivots, k<=40, tilt 0.5, both arms at epoch 50.

| band | gap TREAT | gap CTRL | top1 TREAT | top1 CTRL |
|---|---|---|---|---|
| 1-4 | 1.819 | **1.891** | 0.759 | 0.759 |
| 5-9 | 1.255 | **1.494** | 0.480 | **0.494** |
| 10-14 | 0.831 | **1.276** | 0.284 | **0.316** |
| 15-19 | 0.448 | **0.852** | 0.186 | **0.201** |
| 20-24 | 0.237 | **0.505** | 0.109 | **0.120** |
| 25-29 | 0.148 | **0.334** | 0.072 | **0.091** |
| 30-40 | 0.067 | **0.125** | 0.068 | 0.069 |

The box did do what it was built to do — `bviol` (fraction of entries outside the
interval) fell 0.641 -> 0.587 against the control at epoch 50 — and the model still
got worse. Pooled metrics understated the damage (gap 0.957 vs 1.202); the by-depth
table is what makes it unambiguous. Another instance of the standing lesson that
pooled recall is a mirage.

## Why it was wrong — a centering force, not a separating one

**The box is satisfied perfectly by the constant prediction `Q(s,a) = d(s)` for all
24 actions.** That has gap 0. A two-sided hinge says nothing about the interior, and
the cheapest way to satisfy both sides is to sit in the middle — so the box supplies
*centering* pressure exactly where the discrimination signal lives, and none of the
separating pressure it was supposed to add.

The argument that motivated it — "flattening toward a saturation constant violates a
depth-dependent interval" — was true but irrelevant: the model was not flattening to
a depth-*independent* constant. `s(undo)` already tracked depth correctly (21.2 at
band 20-24, 24.9 at 30-40). The real defect is the *spread within* an already
roughly-correct depth, and that lives strictly inside the box where the hinge is
silent. Meanwhile the upper hinge actively pushes down every entry above `d+1`,
compressing the top of the distribution. Net effect: measured gap compression.

## The obvious repair is also unsound

Use the measured skew — 89% of children at exactly `d+1` — as a soft target on the
23 off-path entries. That *is* a separating force. But ~4.6% of entries are genuinely
distance-reducing moves that are not the path move, and labelling those `d+1` when
the truth is `d-1` teaches the model to hide **alternative optima** — reintroducing
precisely the pathology that tied `m_rank_v0` (56.9% of its disagreements were benign
alternative optima). Not run.

## Interim rule (which then produced variant 2)

**A constraint whose feasible set contains the collapsed solution cannot prevent
collapse, however certain the constraint is.** Check the degenerate prediction
against the constraint before writing the loss. This became a required pre-flight.

A first repair — a *budget* ("at most ~2 of 24 entries below `d - 0.5`") — fails the
very same check: the collapsed prediction `Q(s,a) = d` has ZERO entries below the
threshold, so a one-sided count is trivially satisfied. Not run.

---

## Variant 2: the sorted-profile loss

Constrain the SORTED vector of the 24 outputs instead of each entry. Sorting is
permutation-invariant, so it still never says which action wins — the property that
motivated the box — but a flat prediction now mismatches a step-shaped target, so
collapse is penalised. This one passes the pre-flight.

Measured mean sorted child profile at d=7, relative to `d(s)`, 4k states:

```
-1.00 -0.50 +0.38 +0.49 +0.89 +0.90 +0.99 +0.99   then +1.00 x16
```

Best constant against it scores **MSE 0.247 > 0** — separating, unlike the box.
Only the depth-robust ranks were pinned, because the profile is measured at d<=7 and
applied out to d~31 while the reducing-move count drifts up (mean 1.00 at d=1 ->
1.60 at d=7; p50 1, p90 2, max 4): rank 1 -> `d-1` (certain at every depth, a
geodesic always exists), ranks 9..24 -> `d+1` (safe even at double the measured
count), ranks 2..8 free.

**Result at epoch 50 — better than the box, still worse than the control everywhere:**

| band | tq6 profile | tq5 box | **control** |
|---|---|---|---|
| 1-4 | 1.783 | 1.819 | **1.891** |
| 5-9 | 1.298 | 1.255 | **1.494** |
| 10-14 | 0.960 | 0.831 | **1.276** |
| 15-19 | 0.563 | 0.448 | **0.852** |
| 20-24 | 0.312 | 0.237 | **0.505** |
| 25-29 | 0.190 | 0.148 | **0.334** |
| 30-40 | 0.061 | 0.067 | **0.125** |

The mechanism fix was real — the profile beats the box at every mid/deep band, which
is what the centering-vs-separating analysis predicted. It just does not clear the
control.

## The category result

**Sorting routes the gradient by the model's OWN current ranking.** A profile target
says "the k-th smallest output should be `d+1`" but never says *which action* the
k-th smallest is; that assignment comes from the model's present ordering. So the
loss pushes whatever the model currently ranks low even lower and whatever it ranks
high even higher — it **entrenches the existing ordering instead of correcting it.**

That predicts exactly the damage profile observed: mild where the ordering is already
right (band 1-4, 1.783 vs 1.891) and severe where the ordering is near-random (band
25-29, 0.190 vs 0.334, and top1 there is 0.066 against a 0.042 chance floor).

Generalising: **an auxiliary loss that is invariant to permuting the 24 outputs
carries no information about the assignment, and the assignment is the entire
problem.** It can only reshape the value distribution, and the gradient it does
supply is self-reinforcing. The alternative-optima immunity that motivated both
variants and permutation invariance are the same property — so the immunity and the
usefulness cannot be had together this way.

Discrimination among actions requires a loss that NAMES actions using information we
actually have. That is precisely what `path_mse` (the on-path entry), the sparse-Q
rw-middle pair, and the exact BFS anchors already do — and it is why they work.
**The permutation-invariant constraint family is closed. Do not propose variant 3.**

## Artifacts

`box_*` and `profile_*` knobs remain in `51_train_sparse_q.py`, all default 0.0, so
the control path is the previous behaviour bit-for-bit. `tq5_box.yaml` and
`tq6_profile.yaml` are retained and marked REJECTED. Checkpoints:
`tetraminx/models/tq5_box/epoch_0050.pt`, `tetraminx/models/tq6_profile/epoch_0050.pt`,
`tetraminx/models/tq1_ctrl/` (control, continued to 400 as a local tq1 baseline the
repo otherwise lacked).

Reusable: the pre-flight harness — measure the exact child-distance distribution
against the d<=7 table, then check whether the collapsed prediction satisfies the
proposed constraint. It costs ~2 minutes and it would have killed both variants
before either trainer was written.

Ops note: concurrent `torch.compile` runs must not share `TORCHINDUCTOR_CACHE_DIR` —
the race produces a half-written kernel that surfaces as
`InductorError: SyntaxError: source code string cannot contain null bytes`.
