# 666: what to do next -- inference first, then training

Written 2026-08-21. **Supersedes `HANDOFF_666.md` section 4 ("The model line is CLOSED")**
for the reason in Part 0. Companions: `HANDOFF_666.md` (state + post-processing),
`CUBE666_TEACHER_POSTMORTEM.md` (teacher line), `CUBE555_PROGRESS.md` (the working
sibling), `BIGCUBES_PLAN.md` (structure + counting bounds).

---

## Part 0. Why this document exists

`CUBE555_PROGRESS.md` reports the SAME stack solving 555 end to end: 84 percent of
fully-mixed states, 554/1035 pids, mean 169.7 moves, 456,814 -> 218,626 -> ~199,928.
555 is 307.93 bits, counting bound 66.43, fully-mixed distance ~72. So a beam in this
codebase bridges **72 moves**. 666 needs ~110.

Before accepting 110 as the wall, put the two models side by side on the only depth
statistic measured on both:

| | 555 (solves) | 666 arm A (solves nothing) |
|---|---|---|
| Q ceiling | 42 | **72** |
| deep true distance | ~72 | ~110 |
| `min_a Q` at true depth 60 | ~40 (saturated) | **68.9** |
| `min_a Q` at true depth 80 | saturated | 72.2 |

The 666 model is the better-calibrated one on absolute scale. That does not prove it is
better -- scale is not discrimination, and this project has closed several directions on
exactly that distinction -- but it does mean the two were never compared on anything that
could tell them apart.

And 555 does not work because its model is strong. Its **per-frame** solve rate is ~33
percent; the 84 percent comes from `1 - 0.67^k` over 6 symmetry frames. Four search-side
multipliers carry it, and it is not established that 666 ever had them:

| 555 production | 666 as documented in HANDOFF_666 |
|---|---|
| width **2^21 - 2^22** | tested **2^16 -> 2^20** |
| 6 symmetry frames as independent retries | no frames protocol recorded |
| `invert` legal, 48 x 2 = **96** trajectories | not recorded as used |
| `history_depth=4` (-14 percent path) | unknown; the flag was once **unwired** on a sibling driver |
| endgame `d<=5` ball as goal set | word ball built, use as goal set not recorded |
| warm Bellman **0/3 -> 2/3** | Bellman monotone destructive 13 -> 7 -> 6 -> 1 |
| `max_steps` **250** | **UNKNOWN, and this is the dangerous one** |

The step cap matters more than it looks. 555 solves at mean 169.7, **min 128, max 250**,
against a cap of 250: the solved-length distribution is truncated at the cap, and the
achieved path is **2.4x** the true distance. The same inflation on 666 is 110 x 2.4 = 264
steps, and 666's scorer is probably relatively worse, so 300-450. A run capped below that
returns `0/N` whether the beam was descending or dead -- and `0/N` has already been read
as a capability verdict once on this puzzle (`CUBE666_TEACHER_POSTMORTEM.md`).

**So: Part 1 before Part 2. Do not retrain until the deployed model has been run at full
power with the descent instrumented.**

---

# PART 1 -- INFERENCE WITH THE EXISTING MODEL

Model: `models/q666_a_final_ARMA_deployed.pt` (v1 ResMLPQ, 26.4M, 3M updates).
Floor to beat: `FINAL_submission.csv`, 362,371, mean 358.6, median 447, max 562,
md5 `48abf12b37361e83317363142e612fe2`, replay-verified 1012/1012.

## 1.1 What a solve is worth

From `BIGCUBES_PLAN` section 1 (`total = sum_p min(L_p, M)`, min'd against the
commutation-reduced fallback 463,203):

| model solves every pid at M moves | 666 total |
|---|---|
| 150 | 139,654 |
| 200 | 180,192 |
| 300 | 253,162 |

Today: 362,371. **A model that solves at 3x the counting bound is worth about -110,000** --
larger than the external solver file was worth (-89,072), and 56x everything the models
have contributed so far (-1,956 gross). Partial coverage scales roughly linearly:
554/1012 pids at 300 moves is worth roughly -60,000.

Score every claim as a per-pid min against `FINAL_submission.csv`, not against a remembered
number (rules 26 / 26b / 28).

## 1.2 Config audit -- do this before spending any GPU time

For each knob below, find the line in the 666 solve driver that **consumes** it, not the
line that parses it. A parsed-and-never-forwarded flag is the exact failure that produced a
byte-identical A/B on tetraminx (rule 28c), and it is indistinguishable from "the knob does
nothing".

1. **`max_steps`** -- what value did the previous 666 runs use? Record it. If it was <= 250,
   every "0 solves" in `HANDOFF_666` section 4 is confounded and none of those nulls stand.
2. **Beam width** -- confirm the largest width the driver has actually been RUN at on 666,
   not the largest it accepts.
3. **Symmetry frames** -- is there a frames loop at all? How many, drawn from what set?
   666 has 48 (24 rotations x mirror). Verify the generator relabel table round-trips:
   apply frame `k` to the solved state, then `relabel[k, a]` for every `a`, and confirm it
   equals applying `a` first and then frame `k`. All 36 columns, all 48 frames, 0 failures.
4. **`invert`** -- 666 is a supercube (`central_state == identity`, all 216 values distinct),
   so `invert_state` is well defined and legal. 48 frames x invert = **96 trajectories**.
   Smoke-test the DIRECTION: invert a known-good path, translate it back, replay it.
   `TRANSLATED PATH DOES NOT SOLVE` means the direction bug, not "invert does not help"
   (`CUBE555_PROGRESS` mistakes 5 and 7 are both this class).
5. **`history_depth`** -- wired end to end? Run 0 vs 4 and confirm the per-pid results are
   NOT byte-identical. Expect ~-14 percent path length at 4; it saturates at 4 because all
   generators are odd, so the graph is parity-bipartite and revisits only occur at even lags.
6. **Endgame ball** -- is `d<=5` used as the GOAL SET, or is the goal a single state?
   666 `ball(<=4)` = 928,804 exact; `ball(<=5)` is about 26.9M. Membership must compare the
   **state**, not only a 64-bit hash: at ~27M entries and ~1e11 lookups a false positive is
   EXPECTED, and one killed a 29-hour 555 run at pid 800. Make the residual non-fatal.
7. **`torch.compile`** -- padded to a fixed chunk size (rule 2). Verify byte-identical output
   against eager before trusting it. Expect ~1.35x.
8. **`--fallback FINAL_submission.csv`** -- so every ATTEMPTED pid is recorded. `--resume`
   skips pids present in the output CSV and failures are never written there, which silently
   re-grinds the same hard pids (555 mistake 1: four relaunches, an apparent "0/3 in 59 min"
   collapse that was repeated work).

## 1.3 Probe A -- the descent curve

One run. Three deepest pids. One frame. Maximum width and steps. The purpose is not a solve;
it is a curve that **cannot return an ambiguous zero**.

Config:

    width          largest that fits. 666's per-step expansion is ~1.7x 555's
                   (216 bytes/state vs 150, and 36 children vs 30). If 555 fits 2^22
                   with chunked expansion, expect 2^21 - 2^22 here.
    max_steps      500
    frames         1  (frame 0, identity)
    invert         off
    history_depth  4
    endgame        d<=5 ball as goal set
    compile        on, padded
    env            PYTHONUNBUFFERED=1

Instrumentation, logged EVERY step -- this is the whole point of the probe:

    step, min over beam of min_a Q(child), 10th percentile of that,
    best-so-far, distinct states in the beam after dedup, wall seconds

Log once per run: the parity check. Every 666 generator is an odd permutation, so
`d(s) = sgn(s) mod 2`. Print the target's parity and confirm the endgame ball only ever
matches states of the right parity. A mismatch is a bug in the ball or in the replay
direction, not a near miss.

## 1.4 Reading the curve -- three branches, none ambiguous

**(a) `min_a Q` descends monotonically and the beam reaches the ball.**
The model works. Go to 1.5. The previous nulls were a search-config artifact.

**(b) `min_a Q` descends and then PLATEAUS at some value V, flat for 50+ steps.**
The model has range down to V and no further. **V is the number that was missing from every
measurement taken on this puzzle so far.** Two moves, in order:
  - raise width one notch and re-run. If the plateau drops, width was still buying something
    and 2^20 was simply too small.
  - if the plateau holds, the usable range is `110 - V` moves. Go to Part 2, and the Part 3
    fallbacks become live.

**(c) `min_a Q` never descends, or descends for <10 steps and then random-walks.**
The scorer has no depth signal. Run the positive control (1.6) to rule out the harness,
then go to Part 2.

**Do not report "0 solves" without the curve.** That is what closed this line the first time.

## 1.5 If it descends: the campaign

**Order deepest-first.** Saving per pid is `base_p - achieved`, so deepest-first front-loads
the entire gain and any stopping point is near-optimal.

**Spend frames, not width.** 555 measured fresh frames at 0.595 moves/GPU-s on the deepest
pids, against a log-linear return from width at 2x the cost per attempt; arm B (width
escalation) was never run because arm A dominated it. With per-frame rate `r`, coverage after
`k` frames is `1 - (1-r)^k`:

| r per frame | k=6 | k=24 | k=96 |
|---|---|---|---|
| 0.01 | 0.06 | 0.21 | 0.62 |
| 0.03 | 0.17 | 0.52 | **0.95** |
| 0.05 | 0.26 | 0.71 | **0.99** |
| 0.33 (555's rate) | 0.91 | ~1 | ~1 |

**666 does not need to be good. It needs `r` above a few percent.** You have 96 trajectories
(48 frames x invert); 555 gets 84 percent out of 6 at r=0.33.

**Expect decay across rounds.** 555 measured 0.837 -> 0.600 -> 0.500, factor ~0.77 per round:
frames are not fully independent, and the residue is enriched in genuinely hard pids. Two
555 pids have now failed 18 frames each. Budget with the BLENDED per-pid cost (rescues exit
early at ~2-3 frames, failures pay all of them), not the failure cost.

**Early-exit on first solve** within a frame set, so only pids that need extra frames pay.

**Merge and verify before quoting anything.** `43_shortcut -> 41_merge -> 50_verify`,
expecting 1012/1012 `PASS`, then a per-pid min against `FINAL_submission.csv` and every
other source (rule 26b: re-pull live machines every time, copy sources into a stable dir,
search by CONTENT not filename). An empty CSV passes verification vacuously (`0/0 -> PASS`).

**Then post-process -- but measure first.** Run pathkit's suffix diagnostic before the
stack: `beat-k 0` means the slack is global and local rewriting is finished on that file.
Beam output is tight: on 555, commutation reduction on beam paths returned **+0.00 percent**
and shortcut splicing **-0.74 percent** -- all of the post-processing gain there was on raw
scramble words, none on beam output. Do not assume 666 beam output behaves like the solver
file. If the diagnostic is nonzero, run `67_window_segmented.py` (r=4, segmented with 2r
overlap) and `66_neural_bridge.py`.

## 1.6 Positive control -- only if 1.4 returned (c)

Run the **555 model on a 555 pid through the 666 driver** (or the shared driver at n=5).
If that solves, the harness is sound and the 666 model is the problem. If it does not, fix
the driver before touching training. This is the pattern that saved the AZ arm in the
teacher postmortem ("positive control reproduced 8/24, so the harness is sound").

## 1.7 Traps

- A background job in the agent's own cgroup dies with the session. `nohup` survives SIGHUP
  but not scope teardown; use `systemd-run --user` to place it in `app.slice`.
- Buffering is set by the LAST pipe stage. A long run writing to a file block-buffers at
  8 KB, which at ~90 bytes/line is one flush every ~7.7 h. `PYTHONUNBUFFERED=1`.
- Checkpoints saved before `policy_head` existed omit it from `model_config`; `load_model`
  infers the head from the presence of `p_head` weights.
- `pathkit --scan` is `nargs="*"`; passing the flag twice silently drops the first root and
  reports "scanned 0 files".
- Always run `pathkit.cli selftest` first. Every method here can return 0, and so can a
  broken harness.
- Replay-verify the COPY, not only the source. This mount has failed silently before.

---

# PART 2 -- TRAINING A NEW MODEL

Start this when Part 1 returns **(b)** with a plateau that width does not move, or **(c)**.
If Part 1 returns **(a)**, finish the campaign first: a solved pid is worth more than a
better model.

## 2.1 The target, stated as a number

The gate is not loss, not saturation, and not absolute scale. It is **child-recall by true
depth at the beam cut** (09b-style), and 555 now supplies the calibration:

    555 Q top-1 at depth:  0.246 / 0.182     (chance = 1/30 = 0.033)
    555 V(child) top-1:    0.145 / 0.153     (the cheap head is also the better one)
    666 chance             1/36 = 0.028
    666 arm A at depth:    NEVER MEASURED

**Measure arm A's first.** It costs no training and it may end the question: if arm A sits
near 0.15 at depth 100, the model was never the problem and Part 1 (b) is a search issue.

Protocol: sample states by near-geodesic walk (drift 0.944, so a length-L walk sits at
~0.94L true distance), score all 36 children, count how often the certified-decreasing child
(the undo move) is `argmin`. **Report BY DEPTH, never pooled** -- pooled recall is a mirage
because near-solved states have the lowest Q and fill the global top-B, masking the deep
collapse (`allneighbor_qhead_rejected`). Also report the sibling gap `Q(next) - Q(undo)`,
whose true value is exactly 2.0 at every depth. On IHES E6 that gap fell from 1.96 at depth
2 to **0.37 at depth 21** under absolute-distance MSE. Preventing that collapse is what the
label recipe below is for.

## 2.2 Architecture -- port 555, do not scale it

Capacity is NOT the binding constraint: null on 555 (24.8M vs 14.2M) and null on 666
(26.4M vs 119.9M). Keep the shape.

    embedding      216 sticker classes -> 24 dims      (555: 150 -> 24)
    input_stack    216 x 24 = 5,184 -> 1,024           (555: 3,600 -> 1,024)
    res_blocks     10 x (two 1024 x 1024 linears)
    q_head         1,024 -> 36                          (555: -> 30)
    v_head         1,024 -> 1

**One change from 555, which its own doc recommends for the next run: orbit-factored
one-hot instead of the learned embedding.** Every 666 position's sticker is one of that
position's orbit of 24, so one-hot over 24 gives `216 x 24 = 5,184` dims -- identical width
and cost to embed24, but lossless where embed24 is a learned rank-24 factorisation. Plain
one-hot over 216 classes is the thing to avoid (on 555 at 150 classes: 44.1M params,
0.69 Mstate/s vs embed24's 24.8M and 1.34 Mstate/s -- 2x slower for no gain).

The 36-wide Q head is load-bearing, not an optimisation: it scores all 36 children from one
forward pass on the parent, so a beam step costs `B` evaluations instead of `B x 36`. That
is why a 2^21-2^22 beam runs on one A100 where the CayleyPy paper needed 69 agents at 2^24.

## 2.3 Labels -- exactly three sources, and the restriction IS the recipe

A fourth source was measured and rejected on the sibling puzzles. Note that the pre-flight
check for constraint losses passes for the rejected variants too, so passing it means nothing.

**1. Sparse-Q random-walk middles.** Non-backtracking walk of length `k` from solved, ONE
pivot `p` inside it, label two of the 36 columns: `Q(s, undo) = p - 1`, `Q(s, next) = p + 1`.
`pivot_tilt = 0.5` to bias the pivot deeper. `k ~ U[2, K_MAX]`.

> **666 delta: `K_MAX = 120`.** 555 chose 80 from its counting bound of 66.4 (ratio 1.20);
> 666's counting bound is 102.2, so `1.20 x 102.2 = 123`. Round to 120. Choose it from the
> counting bound, not from the oracle. Walks stay informative to L~108 here (drift 0.944).

This objective is the point of the exercise. Absolute-distance MSE is Bayes-optimal-shrinking:
at large `k` the conditional variance of true distance is large, so the optimum flattens the
local gaps -- and the local gaps are the only thing the beam consumes. The sparse-Q
**difference** has zero conditional variance (always exactly 2), so MSE cannot buy loss by
flattening it; the absolute level is free to saturate, the gap is not.

Do NOT lower `K_MAX`. The teacher arm's `k_max = 40` compressed the Q ceiling from 72 to 35
and demonstrably cost solve rate (137 vs 152 on identical windows).

Cheap, grounded refinement: where the pivot or a labelled child is inside the exact ball,
replace the walk label with the exact distance. On tetraminx, 25.9 percent of walk pivots
were mislabelled (always `p >= d`, never below) and ~7.25 percent of labelled pairs assert a
gap that does not exist. Sparse-Q survives this because it is relative, but the ties and
inversions sit inside the primary objective.

**2. Exact BFS anchors.** `d<=4`, all 36 columns exact, 256 rows per step, weight 1.0,
**in every batch**. 666 `ball(<=4)` = 928,804 states (already measured, `BIGCUBES_PLAN` R1).
Load-bearing: without exact anchors in every batch, the Bellman bootstrap settles at
`V(solved) ~ 2`.

> **666 delta to A/B, not to assume: `d<=5` anchors** (~26.9M states). You already build a
> d5 word ball (`32_build_word_ball.py`), and 666's larger diameter argues for the deeper
> anchor. Price it as an experiment with a matched control.

**3. Symmetry expansion.** 4 random frames of 48, via `Q(sym(s,k), relabel[k,a]) == Q(s,a)`.
Draw frames **randomly per sample**. A greedy coverage table keyed on `(undo, next)` hands
the same frames to a given action pair every step and buys column coverage with zero state
diversity -- do not do that. Verify the relabel table round-trips BEFORE training.

**Loss = `sparse + 1.0 * anchor + 1.0 * value`.**

**Budget.** 555 used 3,000,064 updates (11,719 epochs x 256 steps x 2,048 fresh states =
6.14B fresh states), lr 3e-4 cosine to 0, 23.1 h solo on one A100. Match it. Arm A already
had 3M updates, so matching exactly means you are testing the LABELS -- which is the point.
Hold everything else fixed (rule 28: one variable per arm; the teacher arm changed two and
the verdict had to be retracted).

## 2.4 Bellman -- the biggest 555 lever and the biggest 666 hazard

On 555, 6k steps of warm-started Q-Bellman took a checkpoint from **0/3 to 2/3** solved at
~130 moves against a control that solved nothing; deployed at 20k. Warm-starting is the
whole trick -- the same procedure from near-scratch failed.

On 666 the same nominal procedure was **monotone destructive**: beam wins 13 -> 7 -> 6 -> 1.
`CUBE555_PROGRESS` names why it is dangerous: *"On 666, `E[target]` held steady while the
ceiling fell."*

Recipe:

    warm start from the best sparse-Q checkpoint, NEVER from scratch
    expand all 36 children, regress on 1 + min_a' Q_target(child),
      clamped to 0 where the child is solved
    target net refreshed every 500 steps
    lr 2e-5, batch 768
    anchors still in every batch, at weight 2.0 -- they are the only thing pinning
      absolute scale against a bootstrap that would otherwise drift
    watch E[target], NOT the loss. The loss improves while the bootstrap eats itself.

**Mandatory gate on 666:** run the depth-calibration probe from 2.1 **before and after, at
2k-step intervals**. Stop at the first interval where top-1-at-depth falls at any depth
>= 40, even if `E[target]` and the loss look fine. That single check would have caught the
13 -> 7 -> 6 -> 1 collapse at step 2k instead of step 20k.

If plain Bellman still degrades: try **Limited-Horizon Bellman** (search several steps and
back up the best frontier value, sampling from real search regions) instead of the one-step
target. This is the other change `CUBE555_PROGRESS` flags for a next run.

## 2.5 Acceptance gates, in order

1. `V(solved) ~ 0` and `V(d=1) ~ 1`. Failing here means the anchors are not in every batch.
2. **top-1 by depth** vs chance 0.028 and vs 555's 0.246 / 0.182. Plus the sibling gap
   `Q(next) - Q(undo)`, true value 2.0 at every depth.
3. **Probe A from 1.3** on the 3 deepest pids: does `min_a Q` descend past arm A's plateau?
   This is the only gate that spends the model on the real job.
4. Only then a frames campaign.

**Do not promote on loss. Do not promote on saturation ratio.** `CUBE555_PROGRESS` mistake 6:
top-1 was bit-identical across qv-consistency lambdas so the term was called inert; the beam
then moved 187 -> 157 and lost a pid. Offline metrics do not close search questions.

## 2.6 Do not add

- **qv-consistency.** On 555 it trades solve rate for path length (3/3 at mean 187 vs 2/3 at
  mean 157; a lost 555 pid falls back to a ~960-move baseline, net **-714**). Mechanism: V's
  target is the walk index `p` and Q's undo column is `p - 1`, so the two are trained to be
  the same function offset by one (measured `V - min Q = +1.64 +/- 0.65`) and the term
  reduces to a pure cross-parent reweighting. It DID win 2 pids in the 555 merge -- keep it
  as a merge member if already trained, never as the primary. Judge a member on merge
  contribution, not on its standalone verdict.
- **More capacity.** Null on both puzzles.
- **Transformers / attention encoders.** 11-50x the ResMLP at matched training rows.
- **Path labels, sorted-profile / permutation-invariant features, teacher tubes, on-path
  imitation.** All measured and rejected; on-path imitation hit 0.977 accuracy and was pure
  memorisation (4 of 8 "solves" came in at exactly teacher length; pid 511 reproduced its
  420-move path precisely).
- **Bellman from scratch.**
- **A fourth label source.**

---

# PART 3 -- IF BOTH PARTS FAIL

Only if Part 1 returns (c) and a retrained model still plateaus far above the endgame ball.

1. **Window compression.** `CUBE555_PROGRESS` names this the number one idea for 666
   *precisely because the global solve fails there*. On 555 it is worthless: a window longer
   than 169 compresses to 169, so the optimal window is the whole path (pid 1034: global
   saves 799 in one run; four 250-move windows save 298 for four runs). **The logic inverts
   exactly when the global solve fails.** Windows of 200-300 moves cut from the 362,371
   file, each solved as a bounded relative-distance problem, is the direct way to spend a
   model whose range R is below 110.
2. **The rung ladder.** `rung_o5` / `rung_o6` already solve one 24-piece orbit **24/24 in
   18.8-22.3 moves** at 2^16, and width is a lever there (2^11 -> 2^16 = 42 -> 100 percent).
   Composition failed with a **sum** of per-orbit heuristics: with rungs 1-2 solved every
   single move raises the sum, so the beam sits in the stabiliser (`kept 24/24/0`). Score
   the **joint** coset distance with a mask-conditioned head instead, and give the beam
   macro actions so it can cross the ~12-move stabiliser valley in one step (`29^12 = 1e17`
   against a beam of 2^16 -- it cannot be crossed blindly). **Before building any of it,
   measure the effective bits/move of a constrained rung.** `BIGCUBES_PLAN` R5 uses an
   ESTIMATE of 2.3 there and says outright that this number sets everything; if it comes in
   near 1.4, a strict ladder lands at 350-450 moves and is not worth building.
3. **Corners exact BFS.** `8! x 3^7 = 88,179,840` states, 26.39 bits. A table, not a model.
4. **Ask for a second solver file.** On this puzzle one supplied solver CSV was worth
   -89,072 against -1,956 from every model pass combined.

---

# Appendix: 666 numbers

    state_size        216           supercube: central_state == identity, all 216 distinct
    generators        36            f0..f5, r0..r5, d0..d5 and inverses; single-slice QTM
    branching         29.023        log2 = 4.859 bits/move
    |G|               3.1440e149    Schreier-Sims
    counting bound    102.20 moves
    rw drift          0.944 = 1 - 2/36; a length-L walk sits at ~0.94L, informative to L~108
    parity            every generator is odd, so d(s) == sgn(s) mod 2 (free check + prune)
    symmetry          48 frames (24 rotations x mirror); x invert = 96 trajectories
    ball(<=4)         928,804 exact states
    ball(<=5)         ~26.9M
    orbits            4 centre x 24 = 316.15 bits
                      2 wing   x 24 = 158.08 bits
                      corners       =  26.39 bits
                      total         = 500.62 bits
    test set          1012 pids, scramble-length ladder, sample = the inverse scramble
    current best      362,371 (mean 358.6, median 447, max 562), verified 1012/1012

Calibration from the sibling puzzle (`CUBE555_PROGRESS.md`):

    555 counting bound 66.43, fully-mixed distance ~72
    555 achieved       mean 169.7 = 2.4x the true distance
    555 per-frame rate ~0.33, 84 percent over 6 frames
    555 Q top-1 at depth 0.246 / 0.182 against chance 0.033
    555 max_steps 250, and solved lengths truncate at exactly 250
