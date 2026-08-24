# Reply to `CUBE666_ACTION_PLAN.md` and `CUBE666_SOLVING_GUIDANCE.md`

2026-08-21. Written after running the experiment those documents ask for. Read alongside
them; this file does not replace them, it answers them.

**Summary of the reply.** The central recommendation — *run the existing model at full
inference power and instrument the descent before retraining* — was correct, worth the GPU
time, and I ran it. The instrumented run does not support the documents' conclusion. It
shows that `min_a Q` descends by a factor of six while the search provably does not
approach solved, and that the two levers the documents lean on hardest, **beam width and
symmetry frames, are measured counterproductive and unfounded respectively.** Along the way
three load-bearing factual claims in the action plan turned out to be false, one of them
central. The prize is real and I verified the arithmetic exactly; the mechanism proposed to
capture it is not.

---

## 1. What I accepted, verified, and acted on

I did not take the documents on trust in either direction. Everything below was recomputed.

### 1.1 The expected-value case is exactly right

Recomputed from `FINAL_submission.csv` (362,371, replay-verified 1012/1012):

| model solves every pid at | doc claims | my computation |
|---|---|---|
| 150 | 139,654 | **139,654** |
| 200 | 180,192 | **180,184** |
| 300 | 253,162 | **253,126** |

Exact to rounding. A model solving at 300 moves is worth **−109,245**, larger than the
external solver file (−89,072) and 56x everything our models have contributed (−1,956
gross). Partial coverage scales close to linearly: the deepest 30 percent of pids at 300
moves is −62,279.

**This is why I ran the probe rather than arguing.** At that prize a few GPU-hours is
correct even at low prior probability, and the documents deserve credit for reframing the
priority. The conclusion below is not "this was not worth testing".

### 1.2 "Inference before training" is the right order

Also correct, and a fair criticism of how this project spent its time: roughly 60 GPU-hours
went into training arms and comparatively little into inference configuration. I accept
the ordering and followed it.

### 1.3 "Do not report 0/N without a curve" is right, and I had already been burned twice

`CUBE666_TEACHER_POSTMORTEM.md` records one retraction from exactly this failure; the
truncated Bellman runs are a second. The discipline is sound and I adopted it.

---

## 2. Three factual errors, checked against code and logs

### 2.1 "666 arm A at depth: NEVER MEASURED" — false, and it is the deciding number

`CUBE666_ACTION_PLAN.md` s2.1 states this and builds its Part 2 around measuring it. It
exists, in `cube666_findings_2026-08-17.md` s4.1 and in the report of 2026-08-21:

| depth | 666 arm A top-1 | vs chance (0.028) |
|---|---|---|
| 40 | 0.090 | 3.2x |
| 60 | 0.043 | 1.5x |
| 72 | 0.035 | 1.25x |
| 85 | 0.029 | **1.04x** |
| 122 | 0.030 | 1.07x |

against the document's own 555 calibration of **0.246 / 0.182 at chance 0.033 = 5.5-7.5x**.

I audited whether this is the random-walk label artifact identified on cube444 (past mixing,
both children are uniformly random, the true gap is ~0 and any scorer reads chance by
construction). It is not, at the depth that matters. 666 mixing is at
`log2(3.144e149) / log2(29.023) = 496.6 / 4.859 = 102` moves, so the depth-85 row sits
**below** mixing, the undo label is valid there, and 1.04x chance is a real measurement.
Only the depth-122 row is subject to that artifact, and I do not rely on it.

This matters because it is the quantity the documents' own frames argument consumes. The
argument is `1 - (1-r)^k` over 96 trajectories, which needs `r` above a few percent.
Frames multiply a nonzero per-frame rate; they cannot multiply zero.

### 2.2 The `max_steps` confound does not apply to the definitive null

The action plan's sharpest procedural point: 555 solves at mean 169.7 against true distance
~72 (2.4x inflation) truncated at a cap of 250, so scaling to 666 needs 264+ and a run
capped below that returns 0/N regardless. Good reasoning. It does not apply here.

`75_qbeam_pids.py` sets `num_steps = submission_length + 120`:

| pid | baseline | steps actually run | doc's recommended cap |
|---|---|---|---|
| 927 | 562 | **682** | 350-400 |
| 852 | 545 | **665** | 350-400 |
| 665 | 544 | **664** | 350-400 |

So the 0/3 at 2^18 and 2^20 ran at nearly twice the cap the document asks for. That null
stands. The document is right about two older runs — the frame portfolio at 220 and the
2^24 width probe at 250 — but the frame portfolio's pids were rw-40/50 (true distance
38-47), where 220 steps is ~5x, so it is not confounded either.

### 2.3 The parity claim is false, and two derived claims fall with it

`CUBE666_ACTION_PLAN.md` s1.3 and s1.2.5: *"every 666 generator is an odd permutation, so
`d(s) = sgn(s) mod 2`"*, used as a free check and prune, and as the reason `history_depth`
saturates at 4 (*"the graph is parity-bipartite and revisits only occur at even lags"*).

Measured directly over all 36 generators:

```
12 ODD, 24 EVEN
```

The 12 outer face turns (layers 0 and 5 on each axis) are odd — they carry their own 36
face stickers as 9 extra 4-cycles on top of the 6 four-cycles on adjacent faces, giving
15 odd cycles. The 24 inner-slice turns move only 6 four-cycles and are **even**.

Consequences: the Cayley graph is **not** parity-bipartite, there is no free parity prune,
and the stated justification for `history_depth=4` does not hold. I caught this because the
probe's own parity line contradicted a known 1-move solve on pid 12, which is the value of
printing a check you believe rather than asserting it.

### 2.4 Two smaller ones

- **"No frames protocol recorded."** `43_frame_portfolio.py` exists and its docstring
  already draws the ensemble-vs-portfolio distinction the guidance makes. Run at 12 frames
  x 2^18 x `history_depth=4`: **0 solves in 20 frame-attempts.**
- **"`history_depth` unknown; the flag was once unwired on a sibling driver."** It is wired
  and consumed at `src/cayley/khoruzhii_search.py:544,557-558`, and was run at 4.

### 2.5 Two proposals are already-measured re-runs

- `CUBE666_SOLVING_GUIDANCE.md` s11 proposes orbit-factored one-hot (216 x 24 = 5,184
  lossless features) as the headline new architecture. **Built and run**: `ResMLPQ2`, arms
  D and E at 119.9M. Statistically tied with arm A (L40 1/24 vs 3/24, p = 0.61).
- s12 proposes an active-orbit curriculum 1 -> 2 -> 4 -> 6 -> 7 -> 8 -> 9 to locate the
  scaling cliff. **Already located**: 1 orbit 24/24 in 18.8-22.3 moves, 2 orbits 31-33
  percent, 3 orbits 0 percent. The cliff is between 1 and 3, not near 6-9.

The genuinely new parts of s11 are the auxiliary per-orbit heads and the projected label
streams, which have not been tested.

### 2.6 The evidence base is not present here

`CUBE555_PROGRESS.md`, `HANDOFF_666.md` and `BIGCUBES_PLAN.md` are cited as primary
evidence and do not exist on this machine. Every 555 number in both documents — 84 percent
coverage, 554/1035 pids, mean 169.7, per-frame 0.33, Q top-1 0.246/0.182 — is therefore
unverified here. I have treated them as accurate and argued against the conclusion anyway;
if any are wrong the case weakens further, not strengthens.

---

## 3. The experiment: Probe A, and why I changed its instrument

### 3.1 What was asked for

`CUBE666_ACTION_PLAN.md` s1.3: three deepest pids, one frame, maximum width, 500 steps,
`history_depth=4`, exact ball as goal set, logging per step the frontier minimum of
`min_a Q`, its 10th percentile, best-so-far, distinct states after dedup, wall seconds.
Three outcomes, none ambiguous: (a) descends and reaches the ball, (b) descends then
plateaus at V, giving usable range `110 - V`, (c) never descends.

I built this as `cube666/scripts/80_descent_probe.py`.

### 3.2 Positive controls first

Per s4, and because a pipeline has many ways to return zero while broken:

| pid | baseline | probe result |
|---|---|---|
| 14 | 3 | ball hit at step 0, exact depth 2, **total 3** |
| 13 | 2 | ball hit at step 0, exact depth 1, **total 2** |
| 12 | 1 | ball hit at step 0, exact depth 0, **total 1** |
| 22 | 11 | ball hit at step 6, exact depth 4, **total 11** |

Exact agreement with the baseline on all four. The harness solves, the goal set works, the
splice arithmetic is right. On pid 22 the descent is visible and healthy: fixed stickers
47 -> 68 -> 78 -> 93 -> 105 -> 113 over six steps.

### 3.3 Why the specified metric is circular, and what I added

**The problem.** The beam selects the 2M lowest-Q children out of 75M candidates. So
`min_a Q` over the resulting frontier is an extreme order statistic *of the quantity being
optimised*. It falls whether or not the search approaches solved. A first run of the probe
read `min_a Q` 29 -> 14 by step 90 on a state certainly ~100 moves out; that is the beam
locating the model's largest underestimates, not descent. Reading a plateau in it as
"usable range `110 - V`" therefore assumes the calibration it is supposed to test.

**The fix.** I added the number of stickers sitting in their solved slot. The model cannot
game it, and it is model-independent. Crucially it has a computable baseline: the 216
stickers live in 9 orbits of 24, so a uniformly random group element has

```
9 orbits x 1 expected fixed point = 9.0 fixed stickers
```

and this is confirmed empirically (`d122 = 9.0`). Any frontier at or below 9.0 is no closer
to solved than a random state.

**Calibration, at matched sample size.** A max over millions of states is a far more
extreme order statistic than a max over hundreds, so I calibrated with 524,288 random walks
rather than a token sample:

```
mean:  d0=216  d5=93  d10=46  d20=18  d30=12  d45=9  d60=9  d80=9  d100=9  d122=9
max:   d0=216  d5=192 d10=216 d20=142 d30=100 d45=62 d60=42 d80=34 d100=32 d122=33
```

Note the mean saturates at ~9 by depth 45, so this instrument has resolution only below
that — which is exactly the band the question lives in. I also had to fix my own first
version of the read-back helper, which compared a frontier max against the calibration
**mean** and so reported "implied depth <=10" for a frontier that was nowhere near it. That
is the same order-statistic error the column exists to catch, reappearing in its own
interpretation.

### 3.4 Configuration as specified

Arm A: width 2^21, 3 deepest pids. Arm B: width 2^22, deepest pid, for the width-sensitivity
check s1.4(b) asks for. Both: 500 steps, `history_depth=4`, exact d<=4 ball (928,804 states)
as the goal set with hash-to-find and full-state-compare-to-accept, per-step JSONL so a
preempted run keeps its science, `PYTHONUNBUFFERED=1`.

---

## 4. Results

All four runs went the full 500 steps. **Zero ball hits in any run.**

| pid | width | `min_a Q` start -> end | fix_mean at step 20 -> end | ball |
|---|---|---|---|---|
| 927 | 2^21 | 74.6 -> **13.1** | 8.3 -> 14.1 | 0 |
| 852 | 2^21 | 66.6 -> **11.2** | 9.4 -> 7.5 | 0 |
| 665 | 2^21 | 78.1 -> **11.2** | 11.9 -> 6.3 | 0 |
| 927 | 2^22 | 74.6 -> **12.0** | 8.6 -> **1.3** | 0 |

Random-state baseline: **9.0**.

### 4.1 The plateau is real and it is fictitious

By the document's taxonomy every run is branch (b): a clean plateau, flat for 100+ steps,
at V ~ 11-13. Their rule then gives usable range `110 - 12 = 98` moves, which would be
excellent news. It is false, and the probe is what shows it:

**If any state in a 1.85-million-wide frontier were genuinely 13 moves from solved, the
beam would enter the exact d<=4 ball within about 9 more steps. It ran 500 and never did,
on any pid, at either width.**

The plateau value is not the model's range. It is the size of the model's error.

### 4.2 Width is anti-productive at exactly the widths recommended

| step | fix_mean @ 2^21 | fix_mean @ 2^22 |
|---|---|---|
| 20 | 8.3 | 8.6 |
| 60 | 8.4 | **2.2** |
| 100 | 7.0 | **1.8** |
| final | 14.1 | **1.3** |

Doubling the width drives the frontier to 1.3 fixed stickers — **7x worse than a random
state** — while `min_a Q` descends just as confidently. The wider beam does not average the
model's error out; it searches harder for the states where the error is largest. Arm A
reaches the same place more slowly (fix_mean peaked at 14.1 on pid 927 but ended at 7.5 and
6.3 on the other two, both below random).

This is the direct opposite of s1.5 ("width is still a useful rescue lever") and of the
I4/I5 escalation to 2^21 and 2^22 in the staged table.

### 4.3 The frames argument loses its mechanism

Coverage `1 - (1-r)^96` requires a nonzero per-frame success rate. The descent curve shows
the failure is not "the beam narrowly misses a solve some fraction of the time" — it is
that the search moves *away* from solved while reporting arrival. There is no partial
success to compound. Combined with 1.04x chance at depth 85 and the existing 0-for-20 frame
portfolio, I do not believe frames convert this.

I want to be precise about what is and is not established: 0/20 only bounds `r < 14%` at
95 percent confidence, so `r = 0.03` is not excluded by that run alone. The descent curve
is the stronger evidence, and it is indirect on this specific point.

---

## 5. What I now believe

**The model does not have a range problem that search configuration can fix. It has a
calibration failure at depth that search configuration amplifies.** Arm A is a good
near-geodesic local optimiser out to ~30-35 moves — the probe's own controls demonstrate
that cleanly, solving pids 12/13/14/22 at exactly their baseline lengths. Past its horizon
its Q is not merely uninformative, it is confidently wrong, and a wide beam is an efficient
machine for finding the states about which it is most wrong.

That reframes the failure. "At chance past depth 85" understates it: at chance would be
harmless, and a wide beam over a harmless prior would drift. What actually happens is worse
than chance, because the selection is adversarial to the scorer's error structure.

This is the fourth instrument/decision-metric dissociation on this project and the largest:
`min_a Q` fell 74.6 -> 11.2 while the frontier ended below a random state.

---

## 6. What I would and would not do next

**Would not:** Part 2's T0-T5 ladder. Its headline architecture (orbit-factored one-hot) is
already measured null at 119.9M, and its curriculum target is already measured dead at 3
orbits. Nor further width escalation, which is now measured counterproductive.

**Would, cheaply:** the frame portfolio at depth — 12+ frames on pid 927 at 2^18 with the
goal set wired — because it is the documents' strongest surviving claim, it costs about an
hour, and my argument against it is indirect. I expect the same answer.

**Would, if anything is to be retrained:** the only lever that addresses the actual defect
is a label source carrying true distance information in the 60-110 band. Every closed lever
traces to labels that go flat past mixing, and none of the proposed architecture,
curriculum or search changes touch that. Limited-horizon Bellman (guidance s15) is the one
training proposal aimed at the right target, but it inherits the same problem: it backs up
values from a network that is confidently wrong at depth, and s15.2's "take the minimum over
a bounded frontier" is precisely the optimistic-selection operation this probe just measured
failing at 2M-wide scale.

**Would, for score:** the documents' own Part 3 item 1. Window compression of 200-300 move
segments is the correct way to spend a model whose reliable range is 30 and whose target is
110, and it is the only thing that has ever paid here (−1,956 gross). The logic genuinely
does invert when the global solve fails, and that observation is one of the best in either
document.

---

## Appendix: artifacts

| item | path |
|---|---|
| probe | `cube666/scripts/80_descent_probe.py` |
| curves | `cube666/logs/descent_A_2p21.jsonl`, `descent_B_2p22.jsonl` |
| logs | `cube666/logs/descent_A_2p21.log`, `descent_B_2p22.log` |
| frame portfolio | `cube666/scripts/43_frame_portfolio.py`, `logs/frames_test.log` |
| whole-pid wide beam | `cube666/scripts/75_qbeam_pids.py`, `logs/qbeam_wide.log` |
| depth probe | `cube666/scripts/53_probe_compare.py` |
| full training account | `../cube666_model_training_report_2026-08-21.md` |

Reproduce the headline in ~15 minutes on one GPU:

```bash
PYTHONUNBUFFERED=1 python cube666/scripts/80_descent_probe.py \
    --width 4194304 --max-steps 500 --pids 1 --offset 0 --device cuda:0
```

Watch the `fixmn` column against the printed calibration line, not `minq`.
