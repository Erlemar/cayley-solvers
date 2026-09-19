# 666: put the model where it works

2026-08-22. Written after `CUBE666_REBUTTAL_AND_PROBE_RESULTS.md`, whose probe settles the
question the earlier plan was arguing about.

**Supersedes `CUBE666_ACTION_PLAN.md` Parts 1 and 2** (see s9 for the specific corrections).
Its Part 3 item 1 -- window compression -- is promoted here from fallback to primary.
Companions: `HANDOFF_666.md`, `CUBE666_TEACHER_POSTMORTEM.md`, `CUBE555_PROGRESS.md`.

---

## 0. The plan in one paragraph

You already have a model that solves 666 instances at true distance <= 30-35 essentially
optimally: pids 12/13/14/22 return at exactly baseline length, 4/4 at depths 10-30. What
does not exist, and on the current evidence will not, is a scorer with range at depth
85-110. So stop moving the competence boundary toward 110 and start handing the model
problems that already sit inside 35. The existing 358.6-move paths are 3.26x inflated
against a true distance of ~110, which means a 100-move window of one of those paths spans
only ~31 true moves -- inside competence. Re-solving windows is therefore the model's job,
and the only unmeasured quantity in the whole argument is how much of that inflation is
local. **Measure that first (s3); it needs no training.**

---

## 1. What is now closed, and with what

| fact | value | source |
|---|---|---|
| competence boundary | ~32-35 true moves | solves d=30, fails d=40; controls at d=1,2,4 exact |
| required for end-to-end | ~110 | counting bound 102.2, mixing at 102 |
| top-1 at depth 40 / 60 / 72 / 85 | 0.090 / 0.043 / 0.035 / **0.029** | chance = 1/36 = 0.028 |
| 555 comparison | 0.246 / 0.182 at chance 0.033 | 555 solves end to end at ~72 |
| wide beam at 2^22 | `fix_mean` -> **1.3** vs random 9.0 | descent probe, 500 steps, 0 ball hits |

Measured null: capacity (26.4M -> 119.9M), architecture v1 -> v2, orbit-factored one-hot
(ResMLPQ2 arms D/E, tied with arm A, p = 0.61), label distribution (flat vs tilt), beam
width 2^16 -> 2^22, symmetry frames (0 solves in 20 frame-attempts), Bellman (13 -> 7 -> 6
-> 1, monotone), teacher paths (0.977 on-path accuracy = memorisation).

The probe's central result is not "no range". It is worse and more useful: **past ~35 the Q
is confidently wrong, and a wider beam is an efficient machine for finding the states it is
most wrong about.** `fix_mean` ending 7x below a random state at 2^22 is Goodhart under
selection, not drift. That is why width is anti-productive here and why nothing on the
scorer-range axis is worth further spend.

At matched absolute depth the gap is worse than it looks: 666 reads 0.090 at depth 40 where
555 reads 0.246, and depth 40 is 0.39 of 666's mixing point against 0.60 of 555's. Weaker
scorer, relatively shallower probe, deeper puzzle.

---

## 2. The arithmetic that says where the points are

`FINAL_submission.csv` is 358.6 moves mean against a true distance of ~110. **The path is
3.26x inflated.** The model's own inflation is 1.0 at d <= 11.

Re-solving a window of the path is worth doing exactly when

    iota(d_W) < rho_local

where `rho_local` is the existing path's inflation over that stretch and `iota(d)` is the
model's inflation on a distance-`d` instance. The window length is bounded by competence:

    W <= 35 * rho_local

With the global 3.26 that permits W ~ 100. If the path is locally tight at 1.3 it permits
W ~ 45 and the margin disappears. **`rho_local` is the entire question and it has never
been measured.**

Two things say it is not tight:

- **The r=3 sweep found 2,314 moves inside 6-move windows; r=4 found 7,636 inside 8-move
  windows.** The yield grew 3.3x for one extra unit of radius. There is real slack at the
  smallest scales, and the trend points outward.
- **Exact ball rewriting dies of cost right there.** r=5 was priced at ~1 move/GPU-min
  against r=4's 26. The ball is exponential in radius; **a beam is roughly linear in W.**
  So the beam takes over exactly where exact rewriting becomes unaffordable, at W = 20-200,
  which no method has touched on this file.

Note also that total compression is approximately `L * iota / rho_eff(W)`, and `rho_eff`
rises with W because larger windows capture slack that spans more of the path. Competence
caps W. **That is the whole reason the training section (s6) exists**: every move of
competence past 35 widens the window and raises `rho_eff`.

---

## 3. W1 -- the achieved/W curve. Do this first. No training.

For `W` in {20, 40, 60, 80, 120, 160, 200}: sample ~20 windows per W from
`FINAL_submission.csv`, uniformly at random along each path, drawn from the deepest ~300
pids. For each window, form the group element of the subword and solve it with arm A.

Config per window solve: width 2^18-2^21, `max_steps = 3*W + 60`, `history_depth=4`, exact
`d<=4` ball (928,804 states) as the goal set with hash-to-find and full-state-compare-to-
accept, `PYTHONUNBUFFERED=1`. Each instance is at distance <= 35 by construction, so these
are short runs.

**Plot `mean(achieved)/W` against W.** That single curve answers everything:

| reading | meaning | action |
|---|---|---|
| any point < 1.0 | direct verified win, no theory needed | that W is deployable now |
| curve falls with W then rises | the optimum W is at the minimum | deploy at the minimum |
| curve flat at ~1.0 everywhere | the file is locally geodesic; slack is global | s5/s6 only, or stop |
| curve > 1.0 and rising | competence boundary is inside the window span | reduce W, or s6 |

Also record, per window: whether the solve completed, achieved length, and true window span
estimated as `achieved_best_over_all_W` for that stretch. The saving available on a path is
`sum over tiling windows of (W - achieved)`.

**Splice direction trap.** The ball's argmin descent yields the word that *solves* `w`, i.e.
`w^-1`; the window needs `w`, the reverse-inverse. That bug failed 9 of 18 paths on 555.
Replay-verify every spliced path independently before counting a single move.

**Illustrative only** -- `rho_eff` is what W1 measures, these are not predictions:

| W | rho_eff | iota | path | approx total |
|---|---|---|---|---|
| 100 | 2.5 | 1.4 | 358 -> ~200 | ~180k |
| 60 | 2.0 | 1.3 | 358 -> ~233 | ~205k |
| 40 | 1.6 | 1.2 | 358 -> ~269 | ~230k |

Against 362,371 today. Even the pessimistic row is the largest remaining lever, and it is
larger than everything the models have contributed to date (-1,956 gross).

---

## 4. W2 -- pessimistic aggregation. No training. Hours.

The measured pathology is selection on the left tail of scorer error. The standard fix is
not more signal, it is a selection rule robust to that tail.

**4a. Two aggregations over the 48 symmetry frames**, computed from the same forward passes:

    Q_avg(s,a) = mean_k Q(sym(s,k), relabel[k,a])     # variance reduction
    Q_max(s,a) = max_k  Q(sym(s,k), relabel[k,a])     # pessimistic; suppresses underestimates

**4b. Recompute top-1 by depth for both on the existing eval set** (`53_probe_compare.py`),
against arm A's single-frame 0.029 at depth 85. Minutes, no beam.

- `Q_avg` top-1 stays ~0.03 -> the error is systematic bias, frames are dead in every form,
  and the frames question is closed by measurement rather than by inference from 0/20.
- `Q_avg` top-1 rises to ~0.10 -> a weak signal exists and is noise-masked, and the fix is a
  scorer change, not a retry protocol.

**Check first whether arm A was trained with a symmetry-expansion label source at all.** If
it was, it may already be near-equivariant, the across-frame spread will be small, `Q_avg`
collapses to `Q`, and this whole section resolves in one line -- which also explains the
0-for-20.

**4c. Re-run the descent probe with `Q_max` as the beam's scorer**, one pid, 2^21, 500
steps. The acceptance criterion is not a solve. It is whether `fix_mean` stays at or above
**9.0**. If it does, maximisation bias is confirmed as the mechanism, and that result
applies to every scorer in this repo, not just 666.

**Important distinction, because the existing null does not cover it:** frames-as-retries
(`43_frame_portfolio.py`, 12 separate beams) lets each beam select its own left-tail error
and does nothing about the pathology. Frame *aggregation inside the scorer* attacks it
directly. The 0-for-20 tests the first and not the second.

---

## 5. W3 -- the slack-2 lower-tail hinge. Training. Gated on a free pre-flight.

The probe is now the strongest evidence for this term that exists on any puzzle here: a wide
beam locating states where Q is spuriously low is precisely what an all-column lower bound
suppresses.

### 5.1 Pre-flight, before any GPU-day is spent

On arm A, unchanged: score ~100k random-walk states across depth bands and histogram

    Q(s,undo) - min_a Q(s,a)      at thresholds 0 / 2 / 4 / 8, by depth

and separately: **of the columns the beam actually selects into its top-B, what fraction sit
more than 2 below their parent's undo column?** If the tail is not there, or the false-low
columns are not the ones winning beam slots, the arm is inert by construction and should not
run. This is `measure_the_constraint_before_optimizing_it` applied to our own hypothesis.

### 5.2 The loss

    q_undo    = q.gather(1, undo[:, None]).detach()
    loss_low  = torch.relu(q_undo - 2.0 - q).mean()

Justification is the triangle inequality, not a label: any two children of the same parent
are at most 2 primitive moves apart, so `|d*(s_a) - d*(s_u)| <= 2` unconditionally. It is a
**self-consistency constraint on the model's own output vector**, which is why it survives
non-geodesic random-walk prefixes entirely.

Three implementation points that are not obvious from the formula:

- **Stop-gradient on the reference is required.** Violations must push suspicious columns
  up, not let the loss cheat by pulling the undo value down.
- **The reference must be a supervised column.** The complete pairwise version collapses to
  one line -- `relu(q.max(1).values - q.min(1).values - 2)` -- and is strictly more general,
  but with no supervised anchor the cheapest way to satisfy it is to move an unsupervised
  value, and the gradient will find that. `Q(s,undo)` is pinned by the sparse-Q MSE.
- **Do not add `next` as a second anchor.** It is tighter and looks free, but `Q(s,next)` is
  pinned to `p+1`, so under walk slack (`p > d*`) the floor `Q(next) - 2` sits above the
  truth and suppresses good children.

### 5.3 Why NOT the zero-margin "undo is argmin" version

It assumes the undo child is the minimum, which is false whenever the walk prefix is
non-geodesic. Worse, the error is **depth-concentrated**: `P(undo descending)` is
`(1 + drift)/2 = 0.972` in the linear regime, but falls toward 0.5 as the walk saturates
near the diameter -- and `pivot_tilt` deliberately oversamples that band. So the damage
lands on exactly the deepest training rows, and it would validate clean on shallow probes.
Do not run it, not even as a control; a regression there confirms an arithmetic error at the
cost of a full arm.

### 5.4 Weighting, arms, gates

Set the hinge weight by gradient-norm ratio, ~5-10 percent of the sparse-Q gradient norm.
The hinge is sparse, so its norm is near zero at init and can spike as the model drifts --
re-measure the ratio at ~10 percent of training rather than fixing it once. Note `.mean()`
over 36 columns with 1-2 violating dilutes the gradient ~20x; the norm-ratio weighting
absorbs this automatically, which is why it is the right way to set it.

Arms: H0 existing objective; H2 slack-2 lower hinge; H3 margin sweep at 4 (the right margin
under a compressed Q is open -- a true 2-move gap maps to a predicted gap well under 1 at
depth, so a flat 2 is a much looser rail there than near the goal). Skip the symmetric upper
term: the beam selects on *low* Q, the upper tail is never sampled, and compressing it only
eats headroom the sparse-Q gap term is trying to build.

Gate on the s7 metrics, and specifically **not** on undo top-1, which improves mechanically
under any of these hinges.

---

## 6. The training target, restated

Not 110. Not 85. **Raise top-1 from 0.090 at depth 40 to ~0.15-0.20 at depth 45.**

That is a 2x improvement in a band that still carries 3.2x-chance signal, rather than a 7x
improvement at 85 where there is none. And it pays through s2: competence 35 -> 45 widens
the deployable window by a factor of `rho`, which raises `rho_eff(W)`, which shortens the
path. On the illustrative table in s3 the difference between W=40 and W=100 is roughly 50k
points.

**Re-tilt the label mass into depth 25-55. Do not truncate `k_max`.** The teacher arm's
`k_max=40` compressed the Q ceiling 72 -> 35 and cost 137 vs 152 solves on identical
windows, because a residual beam wanders out of its window's own depth band and needs
headroom above it. That arm was confounded with teacher sources, so `k_max` is untested in
isolation -- but the mechanism is sound and there is no reason to fight it. Keep `k_max` at
120 and move the pivot distribution.

---

## 7. Metrics to bank

**The general rule the probe earned:** a metric the search optimises cannot be the search's
progress indicator. `min_a Q` over the frontier is an extreme order statistic of the
quantity the beam minimises; it fell 74.6 -> 11.2 while the frontier ended below a random
state.

| gate | value | use |
|---|---|---|
| `fix_mean` over the frontier | random baseline **9.0** = 9 orbits x 1 expected fixed point | acceptance test for any 666 scorer: must stay >= 9.0 over 500 steps at production width |
| `achieved/W` | < 1.0 | the only direct, theory-free win condition for windows |
| top-1 **by depth** | chance 0.028; 555 reference 0.246/0.182 | scorer quality. Never pooled -- shallow states fill the global top-B and mask the deep collapse |
| frontier **max** fixed stickers | vs matched-n calibration max | a beam needs one lineage; the mean hides a thin thread. Read this off the existing JSONL |

Calibration for `fix_mean`, 524,288 random walks:

    mean:  d0=216  d5=93  d10=46  d20=18  d30=12  d45=9  d60=9  d80=9  d100=9  d122=9
    max:   d0=216  d5=192 d10=216 d20=142 d30=100 d45=62 d60=42 d80=34 d100=32 d122=33

Resolution exists only below d~45, which is exactly the band this plan operates in.

---

## 8. Do not do

- Another full-puzzle training arm at any capacity or architecture. Six levers, six nulls.
- Further width escalation. Measured anti-productive: `fix_mean` 1.3 at 2^22 vs 7.0-14.1 at
  2^21, both against a random baseline of 9.0.
- Frames as retries. Frame *aggregation* (s4) is a different intervention and is worth
  hours; another portfolio run is not.
- A strict nested coset ladder. Rung cost is cumulative-bits/4.859, so a 3-rung reduction
  has a counting bound of 266 moves and realistically lands at 350-450 -- which is where the
  external solver already is. The measured cliff at 3 orbits is the same wall from the other
  side.
- Bellman on the full puzzle. It backs up values from a network that is confidently wrong at
  depth, and "minimum over a bounded frontier" is precisely the optimistic-selection
  operation the probe measured failing at 2M-wide scale.
- Teacher paths, on-path imitation, path labels as a distance target.

---

## 9. Corrections to `CUBE666_ACTION_PLAN.md`

Three load-bearing claims in that document are false and it is still in the repo:

1. **"Every 666 generator is odd, so `d(s) = sgn(s) mod 2`."** False. Measured 12 odd /
   24 even. The correct rule: a slice turn's ring is `4n` stickers = `n` four-cycles, so its
   sign is `(-1)^n`; an outer turn adds the face's own `(n^2 - [n odd])/4` four-cycles.
   - n=5: inner 5 odd, outer 11 odd -> all odd, bipartite, parity free
   - n=6: inner 6 even, outer 15 odd -> **12 odd / 24 even, not bipartite**
   - n=7: inner 7, outer 19 -> all odd, bipartite
   - n=4: inner 4, outer 8 -> all even, sgn constant, useless

   **The invariant holds iff n is odd.** So it is real and usable on 555 and 777 and does
   not exist on 444 or 666. The 555 `history_depth=4` justification stands; the 666 one does
   not, and 666's saturation at 4 is now unexplained.

2. **"666 arm A top-1 at depth: NEVER MEASURED."** False. It exists in
   `cube666_findings_2026-08-17.md` s4.1: 0.090 / 0.043 / 0.035 / 0.029 / 0.030 at depths
   40 / 60 / 72 / 85 / 122. It is the deciding number and it was already on record.

3. **The `max_steps` confound.** `75_qbeam_pids.py` sets `num_steps = submission_length +
   120`, so the definitive nulls ran at 682 / 665 / 664 steps -- nearly twice the recommended
   cap. That null stands.

Also already-built and measured, contrary to that document: the frames protocol
(`43_frame_portfolio.py`), `history_depth` wiring
(`src/cayley/khoruzhii_search.py:544,557-558`), orbit-factored one-hot (ResMLPQ2 arms D/E),
and the orbit curriculum (cliff located between 1 and 3 orbits, not 6-9).

---

## Appendix: numbers

    state_size        216           supercube: central_state == identity, all 216 distinct
    generators        36            f0..f5, r0..r5, d0..d5 and inverses; single-slice QTM
    branching         29.023        log2 = 4.859 bits/move
    |G|               3.1440e149    Schreier-Sims
    counting bound    102.20 moves; mixing at 102
    rw drift          0.944 = 1 - 2/36
    parity            NOT bipartite (12 odd / 24 even) -- see s9
    symmetry          48 frames (24 rotations x mirror); x invert = 96 trajectories
    ball(<=4)         928,804 exact states
    orbits            4 centre x 24 = 316.15 bits; 2 wing x 24 = 158.08; corners 26.39
                      total 500.62 bits
    test set          1012 pids, scramble-length ladder, sample = the inverse scramble
    current best      362,371, mean 358.6, median 447, max 562, verified 1012/1012
                      md5 48abf12b37361e83317363142e612fe2
    path inflation    358.6 / 110 = 3.26x

Score if the whole file were replaced at M moves/pid (recomputed and confirmed against
`FINAL_submission.csv`):

    M = 150  ->  139,654
    M = 200  ->  180,184
    M = 300  ->  253,126

Model competence, measured:

    d <= 11   exact (pids 12/13/14/22 return at baseline length)
    d = 30    solves, 4/4
    d = 40    fails, 0/4 at 2^16 / 2^18 / 2^20
    d > 35    confidently wrong; wide beam drives the frontier below a random state
