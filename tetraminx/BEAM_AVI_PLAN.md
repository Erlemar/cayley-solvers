# Beam-AVI on the Tetraminx transformer -- analysis and plan

Source method: `BEAM_AVI_METHOD.md` (cube444, -4.2% below the community floor).
Written 2026-08-26. Competition deadline **2026-08-29 22:00** (~84 h).

---

## 0. Verdict up front

**Worth running, and the best-argued training idea left on this puzzle** -- because
tetraminx has already run the method's own NEGATIVE CONTROL and recorded it as a dead
end.

`tv0_bellman` / `tv1_bellman` are random-walk-distribution Bellman refinement. More of
it made beam quality monotonically WORSE (ep24 = 478, ep99 = 481, tv1 ep124 = 496 on
the 14-pid bench) and HANDOFF concluded "V refinement has plateaued, width is the
lever". cube444 measured that same curve as arm **C20** (+528 vs its own start, losing
46 of 50 pids) and identified the cause as the *distribution*, not the algorithm.

So our record does not say "Bellman does not work here". It says "random-walk Bellman
does not work here" -- which is the sentence cube444 wrote before switching the state
source to the beam and taking -214.

**But the headroom is small and the clock is short.** Counting bound is avg >= 26.7
with a realistic optimum near 28.0; we are at 28.094/pid. Realistic remaining is
~100-400 moves total, and Rokicki's coset solver is *behind* us. This is a
defend-and-extend play with a hard kill gate, not an open-ended research program.

---

## 1. The method, stripped to its mechanism

Three claims, in decreasing order of how much they matter:

1. **The state distribution is the entire effect.** Bootstrapping on beam-frontier
   states works; bootstrapping on random-walk states at the same budget destroys the
   model. Everything else in the write-up is plumbing around this one fact.
2. **The Bellman backup is already on the GPU and being thrown away.** A width-B beam
   forwards B parents, keeps the top B, and at the next step forwards exactly those
   survivors. `min_a' Q(child, a')` is computed by the search anyway, so the target
   `Q(s,a) <- 1 + min_a' Q_target(s.a, a')` costs +0.2% wall.
3. **Training never sees a target network.** Step 2 writes a flat table
   `(state, action, target)`; step 3 is ordinary supervised sparse-column masked MSE.
   Staleness is bounded to one round because the buffer is replaced wholesale.

Plus two guards: exact BFS anchors on all 24 columns pin the absolute scale (a beam
takes a GLOBAL top-B over (parent, action), so cross-parent level comparability is
load-bearing -- soften it and cube444 went 18/18 -> 5/18 solved), and an unbiased
`--full-expand` stream to correct the survivors-only optimism.

---

## 2. Does the diagnosis transfer? Yes -- we measured it independently

cube444 s2: random-walk index labels are exact when shallow and ~14 moves adrift when
deep; pair accuracy falls to 0.526 (chance) in the 44-50 band where the whole test set
lives.

Our own numbers, from HANDOFF and `51_train_sparse_q.py::PathLabels`:

| band | `tq0` ep1500 top1 | gap |
|---|---|---|
| 10-14 | 0.537 | 1.555 |
| 20-24 | 0.249 | 0.857 |
| **30-40** | **0.105** | **0.302** |

959 of our 1,000 test pids are near-diameter (28-31). The deep band is where the beam
lives and it is where the label collapses -- gap 0.302 against 1.555 shallow. HANDOFF
already records the mechanism claim behind the sparse-Q label as **WRONG** for exactly
this reason ("the Q's gap collapses"), and `PathLabels` exists as a partial patch for
it.

Same defect, same location, independently measured. The diagnosis transfers.

---

## 3. Four ways tetraminx is a BETTER fit than cube444

**3.1 The fragile part of the harvest does not exist here.** cube444 pairs step j
parents with step j+1 survivors by *position*, and `HarvestProbe.on_step` has to drop
the batch whenever the driver filtered anything in between -- a silent-mislabel hazard
the write-up flags as "undetectable downstream". On a permutation puzzle we do not need
the pairing at all. Every generator has order 3 and `inverse_idx` / `inv_move_tbl`
already exist, so at step j+1 the parent is recovered from the child by ONE gather:

```
child = parent[:, gen[m]]        =>        parent = child[:, gen[inv[m]]]
```

The row `(parent, m, 1 + min_a q_{j+1}[i,a])` is therefore **entirely local to step
j+1** -- no cross-step state, no ordering assumption, nothing to drop. It also makes
the TPU port trivial: the JAX kernel already carries `in_move` per slot for
non-backtracking.

**3.2 The terminal case is a whole shell, not a single point.** cube444 grounds the
recursion on `apply(s,a) is solved`. We have exact tables to d<=6 in the beam
(`bfs_endgame.npz`, already loaded by the driver) and d<=7 offline. Whenever a child is
in the table, `Q(parent, m) = d(child)` is EXACT and replaces the bootstrap outright.
**MEASURED 2026-08-26, and mostly wrong as originally written.** Round 0 fired only
**38 exact overrides across 20 pids** -- about 2 per solve, not "the last several
layers of every harvest". The cause is structural and should have been foreseen: the
beam HALTS the instant any survivor reaches the table, so only a handful of states are
ever inside it and the overrides cannot be numerous by construction. Continuing past
first contact would change the search and is not worth it.

The shell advantage is real but it lives entirely in the **anchor term of the loss**
(256 rows/step, all 24 columns exact, `anchor_weight` 2.0) -- which is the same
mechanism cube444 used, so it is not a differentiator. Keep the override; it is free
and correct. Do not claim it as an edge.

**3.3 The trainer already consumes the shard format.** `51_train_sparse_q.py` has:

- `PathLabels` -- loads `{states, actions, targets}` and applies masked MSE on the one
  named column. That IS the sparse shard, byte-for-byte in shape.
- `BakedAnchors` / `BFSAnchors` -- the 24-column anchor term at `anchor_weight`.
- `Symmetries.conjugate` + `expand_labels` -- verified action-relabelling transport.
- the exact `sparse = sq.sum() / msk.sum()` masked-mean shape the method needs.

The dense (24-column) stream is the only genuinely new consumer, and it is the same
shape as the anchor stream. This is a ~1-day build, not a rewrite.

**3.4 Generation is cheap here.** Measured on the local 4090: Q@1M, 1 frame, 15 pids =
147 s (~10 s/pid); Q@524k = 80 s (~5 s/pid). cube444 paid 24 s/scramble at B=65536 on
an A100. So we can afford to harvest at **2^20**, four times our deployment width's
quarter rather than a sixteenth of it, and a 50-scramble round still costs ~10-25 min
locally.

---

## 3b. WIDTH: harvest at 2^20, never at 65k

cube444 harvested at B=65536 and deployed at 2^20 -- a 16x gap it could afford because
65k was still a competent solver on that puzzle. **That does not carry over.** Our own
measurements say a 65k beam is a qualitatively different and much worse solver than the
one we deploy:

| config | 15-pid total | /pid |
|---|---|---|
| 65k, 1 frame | 506 | 33.73 |
| 1M, 1 frame | 468 | 31.20 |
| 4M, 1 frame (deployment) | 419 | 27.93 |

A 65k beam is +5.8 moves/pid off deployment. The states it visits are not the states
the deployed beam visits, and "the beam IS the sweep" is the entire premise of the
method -- harvesting off-distribution reintroduces exactly the defect AVI is supposed
to remove.

**Therefore: probe at 2^20, harvest at 2^20 (fall back to 2^18 only if a round exceeds
~25 min), gate at 4M.** Width is the first ablation axis if round 1 is ambiguous.

---

## 4. Four risks, stated honestly

**4.1 The headroom is real but unmeasured.** CORRECTED 2026-08-26: the live score is
**27,665** (team CayleyPy, 2026-08-26 03:10), not the 28,094 this file and HANDOFF both
quoted -- an hourly auto min-merge pipeline has been submitting continuously and its
output is not mirrored on this machine. Consequences:

* 27.665/pid **falsifies HANDOFF's own "realistic true optimum ~28"** in the
  puzzle-facts table. That estimate is unreliable; the true optimum is by definition
  <= 27.665.
* Against the counting bound of 26.7 the remaining headroom is <= ~965 moves, which is
  an upper bound and not a forecast.
* Lead over Rokicki (28,481) is **816 moves**, so this is extending a comfortable lead,
  not defending a threatened one.
* Any local CSV is a stale merge floor. Nothing in `tetraminx/submissions/` is newer
  than 2026-08-21. Do not quote a score from disk.

**4.2 "Every route to 8 units lands on 417"** (HANDOFF 2026-08-02). Width 4M -> 8M,
checkpoint diversity, model soup and min-merge all converge on the same 15-pid total.
That is the competing hypothesis: the residual gap is SEARCH-side, not scorer-side, and
a better Q converts to nothing. The counter is that every one of those scorers was
trained on the same random-walk objective family, so the convergence is evidence
*within* one class. The risk is real, and Phase 0 exists to price it.

**4.3 The gate is coarse.** A 15-pid bench resolves ~+-2 moves/pid; cube444's own rule
is that deltas under ~2% are noise. On our 15-pid set (band 417-432) that means we need
**>= 8-10 moves** at matched width to call a win. The 800-epoch table (band 424-431, no
trend across 800 epochs while loss fell monotonically) shows how easy it is to read
noise as signal here.

**4.4 Our deployed scorer is a blend, not the transformer alone.** Deployment is
`0.8 * mx_tf_az/epoch_1500 + 0.2 * mx_resmlp_az` with `qv_consistency 0.3` and
`history_depth 1`. Retraining the transformer leaves the ResMLP partner stale, and
`qv_consistency` reads the AZ VALUE head against Q on an absolute scale -- so the value
head must be kept calibrated through AVI or the deployed configuration silently
degrades. See 6.3.

---

## 5. The plan

Gates are hard. Each phase ends with a number that either buys the next phase or stops
the line. Run the version-history sweep (7) in parallel throughout -- it is the
zero-risk hedge.

### Compute allocation

| where | what | why |
|---|---|---|
| local 4090 (16 GB, idle) | Phase 0 probe, the whole AVI generate/train loop | the loop is tightly coupled; shards never leave the box, and instrumenting the PyTorch step is a few lines |
| **GCP v6e-8** (QR, us-east5-a) | gate at 4M, Phase 3 conversion sweep | deployment width, and the parts that are embarrassingly parallel over pids |

The TPU needs **no kernel edit** -- the gate and the sweep are ordinary
`jax_beam_spmd_v_only.py` runs, and the 8-rank correctness gate already passed
(pids 990/991/992 at 1M = 32/31/34, matching the 4-chip runs). Provision with the
recipe in HANDOFF (`gcloud alpha compute tpus queued-resources create tetra-v6e8-qr
--zone=us-east5-a --accelerator-type=v6e-8 --runtime-version=v2-alpha-tpuv6e
--provisioning-model=flex-start`), native ssh with a project-level key, and
`46_verify_checkpoint.py` on the converted params before trusting any TPU number.

Deliberately NOT porting the harvest into the JAX kernel: it is a half-day build, and
its only benefit is generating at 4M instead of 2^20. Revisit only if Phase 2 passes
and there is time left.

### Phase 0 -- GO/NO-GO probe (~2 h, local 4090, **B = 2^20**)

New: `tetraminx/scripts/72_path_rank_probe.py`.

Replay each pid's banked 28-30 move path from `FINAL_tetraminx_28094.csv` through the
beam at **B = 2^20** (see 3b -- 65k is off-distribution and would answer the wrong
question) and, at every step, record the **cross-parent percentile** of the on-path
candidate among all `B * 24` scored candidates. Report it banded by remaining distance
(the deep band 25-31 is the one that decides). 30-50 pids at ~10-30 s each.

Mechanics: the on-path state must be force-included in the parent set at each step,
because the beam will drop it -- score the union `beam_parents + on_path_parent`, take
the percentile of the (on-path parent, on-path action) pair, then let the beam continue
from its OWN top-B. Assert the probed and unprobed paths are byte-identical, as cube444
did; a probe that perturbs the search measures itself.

* Null is 0.5. cube444 measured 0.034 (z = 30.2) and called that ample.
* **Near 0.5 in the deep band => STOP.** There is nothing to bootstrap from, and the
  whole line dies for ~2 h of cost.
* Also report the fraction of steps where the on-path candidate SURVIVES into the top
  B. cube444 found the beam beats the reference path while dropping it -- if that
  happens here too it is a good sign, not a bad one, but record it so it is not later
  misread as a bug.

Cheap, and the only honest way to price risk 4.2 before spending build time.

#### RESULT 2026-08-26: GO. 40 deep pids, B=2^18, deployed stack

`results/path_rank_probe_b18.json`. Two corrections to how this must be read, both of
which change the verdict:

**(a) Cluster by pid, not by step.** The probe's own printed z of 22.4 treats 641 steps
as independent; steps within a pid are strongly correlated. Clustered on 40 pids the
overall figure is **mean 0.221, z = 7.7**. Still decisive, but a third of the naive
number -- quote the clustered one.

**(b) Condition on whether the beam still holds the on-path parent.** It does for only
16 percent of steps; the beam abandons the reference path at a median remaining
distance of 23. Unconditioned, the mid bands look dead:

| remaining d | mean | z_clust | ALIVE | DEAD |
|---|---|---|---|---|
| 22-24 | 0.327 | 3.1 | **0.111** (n=71) | 0.656 |
| 19-21 | 0.422 | **1.2** | **0.055** (n=30) | 0.544 |
| 16-18 | 0.371 | 1.9 | 0.081 (n=8) | 0.392 |
| 7-12 | 0.050 | 43.6 | -- | 0.050 |
| 0-6 | 0.0002 | 6133 | -- | 0.0002 |
| all | 0.221 | 7.7 | **0.102** (n=134) | 0.246 (n=698) |

The apparent dead zone at 19-21 is an artifact of the COMPARISON POPULATION. Once the
beam has dropped the reference state we are scoring an abandoned state against states
the beam selected; conditioned on the parent actually being in the beam, 19-21 is the
STRONGEST band at 0.055, with a 60 percent survival rate against a 4.17 percent chance
baseline. The honest headline is the conditioned **0.102**, the same regime as
cube444's 0.034 -- not the contaminated 0.221.

**The dead-vs-alive gap is a second, independent result.** 0.246 against 0.102 is the
optimism selection bias made visible: the beam's kept states score BETTER than a
known-good state at the same depth, because the model under-predicts at depth and the
beam selects hardest for exactly that error. That is the pathology the Bellman backup
corrects, and it is why `E[target]` rising is the healthy sign (s8). The probe
therefore confirms not just that ranking signal exists, but that the specific defect
AVI removes is present here.

Caveats kept on the record: conditioned deep bands have modest n (30 / 8 / 2), and the
reference was `FINAL_tetraminx_28065.csv`, ~0.4 moves/pid stale against the live 27,665.
Neither is close to flipping 0.055 vs a 0.5 null.

### Phase 1 -- build the harvest + AVI loop (~6 h, no GPU contention)

`tetraminx/scripts/73_gen_harvest.py` -- generation with harvest:

* Drive `KhoruzhiiSolver` in q_mode with the **deployed stack unchanged**: blend
  0.8/0.2, `qv_consistency 0.3`, `history_depth 1`, exact d<=6 endgame, 1 frame. The
  state distribution has to be the deployment distribution; that is the entire method.
* At each step, from `q_all` (B, 24) on the survivors:
  - `tgt = 1 + q_raw.min(dim=1).values` using **RAW Q** -- before the qv-consistency
    adjustment and before the non-backtracking ban. Both are search heuristics; neither
    changes the MDP, and the back-move is a legitimate member of the Bellman min.
  - `parent = states[:, gen[inv[in_move]]]`, one gather (3.1).
  - override with the exact table value wherever the child is in `bfs_endgame.npz`
    (3.2).
  - subsample ~8k rows/step with `torch.where` -- no `.nonzero()` / `.item()` / bool
    indexing in the hot path (cube444 s9 lost a step per iteration to a device sync
    there). At B = 2^20 that is a 1/128 sample of ~26M available rows per pid, so
    50 pids x ~25 steps x 8k = **10M rows/round**, matching cube444's 9.5M at four
    times the width. ~930 MB/shard as `uint8` states; overwritten each round.
* Second stream, `--full-expand 256`: sample 256 frontier parents **uniformly, without
  reference to Q**, forward their 24 children, emit a dense 24-column target row. Cost
  is `256*24/B` -- +0.6% at 2^20, +2.3% at 2^18. This is the debiasing stream; without
  it the harvest only ever learns that surviving children were good.
* Also emit the value target `V(parent) = 1 + min_a q_j[parent, a]` on the same rows,
  for risk 4.4.
* Write `runs/avi/shards/rNNN.pt` as `{sparse: (states u8, actions i8, targets f32),
  dense: (states u8, targets f32 (M,24)), value: (targets f32)}`.

`tetraminx/scripts/74_train_avi.py` -- reuse `51_train_sparse_q.py` wholesale:

* Swap `SparseQSampler` for a shard reader; keep `BakedAnchors` at `anchor_weight`
  >= 2.0 (cube444 uses 2.0 and warns explicitly against softening the level term).
* Use **random-single-frame** symmetrisation, not `expand_labels`' full expansion --
  shard rows are plentiful, so augmentation should cost 1 row, not 14. This also keeps
  the step inside 16 GB: HANDOFF records that batch 512 x 14 sym rows peaks at 17.3 GB
  and WDDM then PAGES instead of OOMing (340 ms -> 3,617 ms, no error). Budget ~1024
  sparse + 128 dense + 256 anchor = ~1,408 rows/step.
* Loss: `masked_sparse + 1.0 * dense + 2.0 * anchor + value_w * value`, target clamped
  to [0, 40], lr 2e-5, wd 3e-3, grad clip 1.0, warm from `mx_tf_az/epoch_1500.pt`.
* Scramble seed **derived from the round number**, and the 15 gate pids passed as an
  explicit **exclusion list** (cube444 s5.2 -- a count-based holdout left 47 of 54 gate
  pids inside the training set).

### Phase 2 -- run rounds, gate at round 1 and round 5 (~8 h)

10 rounds x (50 pids @ **B = 2^20** generation + 9,000 steps @ batch 1024)
~= 20-35 min/round. Drop generation to 2^18 if a round exceeds ~25 min.

**Kill gate at round 1**: 15-pid set on the **v6e-8 at 4M x 1 frame** -- deployment
width, exact endgame, hd=1, blend and qv-consistency as deployed -- against a control
run in the SAME session on the same box (CLAUDE.md 28). Reference band is 417-432 with
4M ep1000 = 419. cube444's r005 was already -154 of its final -214, so one round should
show direction.

* `>= -8 moves` -> continue.
* `-3..-8` -> ambiguous; run to round 5 and re-gate.
* `>= 0` -> **STOP**, fall back to 7.

**Diagnostics, and how to read them:**

* `E[target]` **rising is the healthy sign here**, not divergence. Our V drifts DOWN at
  depth (V@d80 = 18.7-23.9 against a true ~28), i.e. the model under-predicts, so the
  bootstrap should climb toward 28-31. cube444 s8 flagged its own 13.1 -> 32.5 climb as
  divergence and was wrong; the FLAT control was the pathological one. We have the
  counting bound and the exact d<=7 table to sanity-check the absolute scale, which
  cube444 did not.
* Gate a MID-RUN checkpoint, not only the last (cube444 s8: solve rate fell in the
  final quartile while `E[target]` was still climbing).

### Phase 3 -- HAND OFF THE CHECKPOINT (not a sweep)

**Scope decision 2026-08-26: converting the model into moves is NOT this session's
job.** Teammates run the wide beam on their own hardware. We deliver a model and the
evidence that it is better; they spend the days of beam wall-clock.

Two consequences that change what we build:

* **No Phase 3/4 conversion sweep, no submission, no merge.** The hourly auto
  min-merge pipeline is explicitly out of scope. Drop the GCS staging and QR-lease
  bookkeeping with it.
* **The gate stops being an internal sanity check and becomes the deliverable.** A
  teammate about to spend days of TPU on this checkpoint needs a number they can
  trust, so the gate has to be matched-control (CLAUDE.md 28) and quoted with its
  resolution, not as a point estimate.

**Handoff package:**

| item | why the teammate needs it |
|---|---|
| `models/avi/rNNN.pt` + `model_config` | the checkpoint itself |
| `46_verify_checkpoint.py` PASS on the JAX-converted params | they run the TPU kernel, not PyTorch -- a bad `load_params_from_pt` conversion would look like "the model is worse" |
| matched-control gate table, both arms same box/flags/conversion | the reason to spend the wall-clock |
| recommended inference config | see below -- AVI moves Q's LEVEL, so the deployed knobs may need re-picking |
| **`--chunk-size 4096`** | 9.2x on any PyTorch-side transformer beam; see the comment in `30_solve.py` |

**Re-gate the deployed knobs, do not inherit them.** `qv_consistency 0.3` and the
0.8/0.2 blend were tuned against `mx_tf_az/epoch_1500`. AVI changes Q's absolute scale
(that is the point -- see the `E[target]` note above), so both need re-picking against
the new checkpoint rather than being carried over. Report the gate at
`qv_consistency` in {0, 0.3} and blended vs transformer-only, and hand over whichever
wins WITH its number.

**Timing:** teammates need lead time before the 2026-08-29 22:00 deadline, so the
gated checkpoint wants to exist by **2026-08-28 morning**, not the final evening.

---

## 5b. RESULT 2026-08-26: five rounds, NEGATIVE

`results/gate_avi_b18.json`. B=2^18 x 1 frame, qv 0.3, hd 1, blend, 15 held-out gate
pids, all six arms in ONE invocation on the same box (CLAUDE.md 28).

| arm | total | vs base | W | L | T | p |
|---|---|---|---|---|---|---|
| incumbent | **441** | -- | -- | -- | -- | -- |
| r000 | 446 | +5 | 2 | 6 | 7 | 0.29 |
| r001 | 447 | +6 | 2 | 6 | 7 | 0.29 |
| r002 | 447 | +6 | 3 | 7 | 5 | 0.34 |
| r003 | 445 | +4 | 3 | 5 | 7 | 0.73 |
| r004 | 447 | +6 | 3 | 7 | 5 | 0.34 |

No arm is individually significant, but all five sit on the same side and the per-pid
deltas are near-identical across arms (pid 800 = +2 everywhere; 200/300/700/990
consistently positive; 900/999 consistently -1). That reproducibility makes it a real
systematic effect rather than scatter. cube444's round 5 was already -154 of its final
-214, so this is a clear negative against the expected effect size.

**Mechanism, visible BEFORE the gate ran** (`77_level_check.py` + variance decomposition):

| ckpt | V_hat | deep gap01 | between-state sd | deep n_reducing |
|---|---|---|---|---|
| incumbent | 17.151 | 0.438 | 6.827 | 1.90 |
| r000 | 17.311 | 0.438 | 6.843 | 2.20 |
| r004 | 17.732 | **0.270** | **6.230** | **2.65** |

The level rose (healthy) but the model FLATTENED: top-1 discrimination halved in the
deep band, cross-parent spread contracted 9 percent, and the model came to believe 2.65
children reduce distance against a true ~1.6. Since the beam takes a global top-B
across (parent, action), cross-parent spread is exactly what it runs on. Same pathology
as [[allneighbor_qhead_rejected]] ("saturation re-asserts in Q-space") and the scale
compression in [[tabfm_v_rejected]].

**Falling loss and rising V_hat both looked healthy and were both wrong.** Only the
replay-verified beam total counted -- the incumbent's own 800-epoch sweep taught this
same lesson and it had to be relearned here.

### KNOWN DESIGN BUG in this run -- target/student mismatch

Generation used the deployed BLEND (0.8 transformer + 0.2 ResMLP) as the target net,
but training updated the TRANSFORMER ALONE on those targets. The student was therefore
fit to a target it cannot represent -- distillation toward the blend -- and at inference
that transformer is blended with the ResMLP again, double-counting the partner. The
method specifies target net == the model being trained. This is an implementation error,
not a property of Beam-AVI, and it is the single most likely cause of the flattening.

**Any retry must fix this first, as a single-variable change** (CLAUDE.md 28): generate
with the transformer alone, train the transformer, gate solo AND blended. Do not also
change lr/steps/anchor weight in the same run or the result will not be attributable.

### THE MISMATCH WAS NOT THE CAUSE -- falsified 2026-08-26/27

A second full 5-round loop with `--solo` (target net == the model being trained, the
method as specified) flattens IDENTICALLY. Matched diagnostic, same fixed 8,192-state
population:

| ckpt | V_hat | deep gap01 | between-sd | deep n_red |
|---|---|---|---|---|
| incumbent | 17.226 | 0.4379 | 6.827 | 1.90 |
| blend r000 | 17.383 | 0.4379 | 6.843 | 2.20 |
| blend r001 | 17.544 | 0.2949 | 6.729 | 2.49 |
| SOLO r000 | 17.401 | **0.4611** | 6.737 | 2.24 |
| SOLO r001 | 17.677 | **0.2665** | 6.468 | 2.69 |

Solo r001 is if anything flatter than blend r001. The target-mean shift from fixing the
mismatch was only +0.22 moves on an identical pid sample (17.047 -> 17.271), which had
already signalled the mismatch was too small to explain a gap01 collapse.

### FINAL GATE, B=2^20, deployment width -- NEGATIVE

`results/gate_avi_solo_b20.json`. 1 frame, qv 0.3, hd 1, solo, 15 held-out gate pids,
three arms in ONE invocation.

| arm | total | vs base | W | L | T | p |
|---|---|---|---|---|---|---|
| incumbent | **427** | -- | -- | -- | -- | -- |
| solo r000 (1 round) | 429 | +2 | 3 | 3 | 9 | 1.00 |
| solo r004 (5 rounds) | 434 | +7 | 1 | 7 | 7 | 0.07 |

**Dose-response, and it descends.** One round is an exact tie; five rounds are clearly
worse. That monotone relationship is the strongest single result here -- far more
convincing than any one delta -- and it tracks the flattening measurements precisely.

**VERDICT: Beam-AVI does not transfer to the tetraminx transformer. Do not retry
without a mechanism that prevents the Q-space flattening.** Note what does NOT rescue
it: since ONE round at 4,000 steps is already only a tie, a gentler recipe (fewer
steps, higher anchor weight) can at best interpolate between the incumbent and that
tie. The curve starts at zero and descends; no setting of the training-pressure knob is
positive.

What was genuinely established, and is worth keeping:

* The **go/no-go probe passed** -- deep ranking signal is real (conditioned percentile
  0.055-0.102 against a 0.5 null). So the failure is NOT "no signal to bootstrap from";
  it is that fitting the bootstrap flattens the Q landscape faster than it sharpens it.
  That is a sharper statement than cube444's precondition can express, and it means the
  precondition probe is necessary but NOT sufficient.
* The flattening signature (`deep gap01`, `between-state sd`) **predicted both gate
  results before either gate ran**. `77_level_check.py` plus the variance decomposition
  is a ~2-minute test that would have saved ~9 hours of beam. Run it first on any future
  bootstrapping experiment on this puzzle.
* `E[target]` and training loss both looked healthy throughout and were both wrong,
  again. Only a replay-verified beam total counted.

## 6. Traps specific to us

**6.1 Use raw Q for the target.** `qv_consistency` rewrites `child_v` to
`Q + lam*|Q - (V-1)|`, and non-backtracking sets banned children to `BIG_F32`. Harvest
the pre-adjustment, pre-ban tensor. Harvesting the adjusted score would train the model
on its own penalty term -- a self-confirming loop with no fixed point at the true
distance.

**6.2 Only the 24 SPATIAL frames transport a Q row.** Inverse antisymmetry gives us 48
search frames, but it transports V, not Q: `Q(s,a) = d(s.a)` maps to
`d((s.a)^-1) = d(a^-1 . s^-1)`, a LEFT multiplication, which is not of the form
`d(s^-1 . a')`. Use `Symmetries.conjugate` (24) for sparse rows; the inverse frame is
fine for generation diversity and for value rows only.

**6.3 Keep the value head calibrated or `qv_consistency 0.3` silently degrades.** It
reads `|Q(s,a) - (V(s)-1)|` on an absolute scale. AVI moves Q's level; if V does not
move with it, the deployed penalty starts charging correct children. Supervise
`V(s) = 1 + min_a Q_target(s,a)` on the same harvested rows (free from the same
forward) and gate BOTH `qv_consistency 0.3` and `0.0` so the term stays attributable.

**6.4 The blend partner goes stale.** `mx_resmlp_az` was trained on the random-walk
objective. Gate transformer-only AND blended; if blended regresses while solo improves,
drop the blend rather than assuming it still pays. Do not change blend weights and the
checkpoint in the same run (CLAUDE.md 28).

**6.5 This is NOT `tq1` and NOT path labels.** `tq1` (near-optimal-path labels) was
REJECTED at ep560, and cube444 rejects path labels too. The AVI target contains no path
length and no walk index -- it is `1 + min_a Q_target(child, a)` on beam states. If the
implementation ever reads a path length, it has drifted into the rejected design.

**6.6 Verify the flag is wired before quoting an A/B.** If an AVI checkpoint returns
byte-identical per-pid results to the control, the checkpoint is not being loaded
(CLAUDE.md 28c -- `30_solve.py --history-depth` was parsed and never forwarded, caught
only this way).

**6.7 Run `46_verify_checkpoint.py` on every converted AVI checkpoint before quoting a
TPU number.** The gate crosses a framework boundary (PyTorch -> `jax_model.py`
`load_params_from_pt`), and a silently mis-converted head would read as "AVI regressed"
rather than as a port bug. Cheap, and it also protects rule 28's matched-control
requirement -- control and AVI arm must both go through the same conversion.

---

## 7. The hedge: run this in parallel, it costs no GPU

The `cayleypy-tetraminx-tpu-beam-q` version-history sweep paid **-51** at 31 versions
and another **-91** at 119. If the kernel has been re-pushed since 2026-08-09, re-probe
the ceiling (binary search the two 404 sources, ~10 requests) and sweep with
`scripts/25_pull_kernel_versions.py`. Zero risk, known yield, independent of everything
above -- and it is the fallback if Phase 0 or Phase 2 kills the AVI line.

---

## 8. Build list

| file | status | what |
|---|---|---|
| `scripts/72_path_rank_probe.py` | NEW | Phase 0 go/no-go at 2^20 |
| `scripts/73_gen_harvest.py` | NEW | beam + Bellman harvest + full-expand |
| `scripts/74_train_avi.py` | NEW | shard reader on top of `51_train_sparse_q.py` |
| `src/cayley/khoruzhii_search.py` | HOOK | expose raw `q_all` + `in_move` per step |
| `configs/avi0.yaml` | NEW | lr 2e-5, anchor_w 2.0, dense_w 1.0, single-frame sym |
| `scripts/30_solve.py` | reuse | local gate |
| `scripts/47_beam_tpu.py`, `jax_beam_spmd_v_only.py` | reuse, NO EDIT | 4M gate + Phase 3 sweep on v6e-8 |
| `scripts/46_verify_checkpoint.py` | reuse | verify converted params before any TPU number |
| `scripts/90_merge_all.py` | reuse | merge |

Cumulative time to the round-1 kill gate: **~8-10 h**. That is the real bet.
