# Cayley-graph solvers: what we learned, where the value is, and what to try next

Date: 2026-09-04. Sources: every HANDOFF / EXPERIMENTS / plan doc in this repo, the
project memory, `cayley-py-666-cube/cube666_new_report_different_machine.md`, the live
Kaggle leaderboards (pulled today via the CLI), and the external literature
(arXiv 2502.13266, 2502.18663, 2603.01786, 2606.04860, 2607.13219).

This is a synthesis, not a new measurement. Every number below is either quoted from a
repo doc (path given) or pulled live today. Where I extrapolate I say so.

---

## 0. Standing today (live, 2026-09-04)

| Competition | Best we hold | Leader | Gap | Deadline | Where the remaining value is |
|---|---|---|---|---|---|
| Professor Tetraminx | team CayleyPy 27,626 (#1, we merged into it); own file 28,094 | Rokicki 28,481 | +855 lead | closed 2026-08-29 | closed; lessons only |
| Megaminx | team CayleyPy 68,548 (#1, merged) | Rokicki 93,606 | lead | closed 2026-08-31 | closed; lessons only |
| cube444 | team CayleyPy 45,978 (#1, merged); own PP file 46,710 | Rokicki 46,298 | +320 lead | 2026-09-22 | QTM-native phase 2; arch-vs-objective A/B |
| IHES picture cube | own 21,972 (#5, separate team); the 21,870 public plateau file is on disk | Rokicki 21,840; CayleyPy / Chekhlov / fle3n tied 21,870 | -132 own, -30 vs plateau | 2026-09-22 | submitting the on-disk 21,870 ties #2 (your call, it is a community merge); exact len-22 sweep is the only real lever |
| cube555 | own 187,780 (#3); community file PP'd to 109,288 on disk | Diener 109,512 | -78k own | 2026-11-20 | **largest known-achievable gain in the portfolio** (see 2.1) |
| cube666 | own 169,769 (#2); 171,019 / 170,051 also on disk | Diener 166,421 | -3.3k | 2026-11-20 | model-only solver is the research goal (see 4) |
| Christopher's Jewel | 16,490 | 8 teams tied at 16,490 | tie | 2026-09-21 | nothing; the plateau is almost certainly optimal |
| cube777 | not entered | 5 teams | - | 2026-11-20 | cheap classical placing if wanted (cube_nnn + KMC port) |

Also open with few teams: pancake (14 teams), glushkov (9), rapapport-m2 (6),
reversals (2), transposons (2), all closing 2026-09-21/22. Our stack is puzzle-agnostic
by duck-typing; see 5.3.

---

## 1. What the programme has established (the laws, with the number behind each)

These are the cross-puzzle facts that survived matched controls. Anything proposed below
has to be consistent with them.

1. **Width is the first lever, until it is not.** Tetraminx 1M -> 4M = -13 on 15 pids,
   4M -> 8M = -2; on the tight tail 4M/8M/32M = 311/306/301 with no plateau
   (`tetraminx/HANDOFF.md:649-652, 1359-1363`). cube444: top-100 longest at 2^22 = 81/100
   improved, 0 worse. But on cube555 width is **non-monotonic** (2^20 2/6, 2^21 6/6,
   2^22 3/6, 2^23 0/1) and on cube666 it is **anti-productive**: at 2^22 the frontier's
   fixed-sticker mean fell to 1.3 against a random-state baseline of 9.0, because "the
   beam is an efficient machine for finding the states the scorer is most wrong about"
   (`CUBE666_REBUTTAL_AND_PROBE_RESULTS.md:228-266`).

2. **The blind-zone law.** For a scorer and target set, M = mixing depth
   (log_b of the target-set index), V = deepest depth at which child ranking is still
   >= ~2x chance, Z = start distance - V. Every big-cube measurement fits
   "a beam solves iff Z <~ 10, and inflation grows with Z"
   (`CUBE666_MODEL_RESEARCH_2026-08-24.md:81-101`). 666 flat: M ~102, V ~75-80, Z 25-30,
   0 solves at any width. 555 flat: M ~72, V >= 65, Z 0-7, 84% solved at 2.35x inflation.
   CayleyPy paper 555: Z ~0, 1.28-1.39x. Corollary: **V/M is the one number that decides
   whether a puzzle is solvable by this family.** Ours is ~0.9 on 555 and ~0.78 on 666.

3. **Saturation is load-bearing, and its variance is the canary.** Every encoder family
   whose V keeps growing past the diameter (GT-V, Perceiver, `state_inv`) failed the beam;
   V variance at d~20 > ~2.5 predicts collapse (`megaminx/HANDOFF.md:370`). Seven encoder
   families closed on megaminx.

4. **Probe metrics do not select checkpoints.** Four models won on pair accuracy / top-1 /
   recall / calibration and delivered zero moves. Only a replay-verified beam total on a
   matched pid set counts; deltas under ~2% are noise. The 666 "different machine" report
   rediscovered this independently ("MAE and cross-parent ordering can improve while the
   correct next action is evicted").

5. **Bootstrapping works only where the scorer already has signal, and only on the
   search's own distribution.** cube444 Beam-AVI on beam states: -4.2% below the floor,
   37W/4L/13T; the same budget on random-walk states: +528, 2W/46L
   (`BEAM_AVI_METHOD.md:226-255`). Tetraminx (scorer at ceiling): Q-space flattens, 5 rounds
   lose 1W/7L. cube666 flat (scorer blind past 85): monotone destructive (13 -> 7 -> 6 -> 1).

6. **The inverse frame carries the frame ensemble; frames saturate early on permutation
   puzzles but are the cheap diversity on big cubes.** Tetraminx: inverse 414 vs forward
   416, any pair = triple = all four. cube555: "fresh frames beat width" at 0.595
   moves/GPU-s, 6/10 deepest failures rescued. Colour cubes (444) have no inverse frame at
   all; odd supercubes (555/777) have 96 trajectories.

7. **Architecture and objective matter on permutation puzzles, and the winner is
   PieceTransformer + sparse-Q rw-middle.** Tetraminx transformer 424 vs ResMLP 461-468
   (-8%); on 444 Vlad's transformer beat our V-beam on 916/1043 pids, 0 losses, uniformly
   across depth bands (`cube444/EXPERIMENTS.md:626-645`). Whether that is the architecture or
   the objective is **still undecided** and decision-critical.

8. **k_max past the mixing depth degrades the ranking label itself.** Past the diameter
   "undo the last move" is no longer distance-reducing (`tetraminx/TRANSFORMER_AZ_RECIPE.md:
   240-244`). 666 arm A used k_max = 122 against M = 102; the paper uses 0.9-1.0 M.

9. **Post-processing has a measurable floor and we are on it everywhere.** Every window
   <= 12 geodesic on tetraminx (166,703 exact queries), <= 11 on 444, <= 44 at reach 7 on
   555; the only exact wins in the whole programme were radius 13 on tetraminx (-8) and
   reach 6-9 on 555 (-224). Windows on beam output are tight; windows on classical/sample
   output are loose (r=3: -28 on a random word vs -2,314 on solver output). Min-merge
   always pays and the risk is the source set, not the merge.

10. **Classical macro structure does not appear in beam paths, and FMC-style insertion
    fails on long-3-cycle puzzles.** 0 commutator matches on IHES and megaminx; megaminx
    3-cycles cost 12-20 moves. On 666 the reverse holds: the classical KMC (corners + parity
    + SA over inner-slice words + 3-cycle insertion) is 10-15% better than anything neural
    and the neural line's total score contribution is 6 moves.

11. **Training axes close in a noise band.** Tetraminx: 9 checkpoints in 424-431 with
    no trend; d7 anchors (15.7x data) identical; bigger V trunks regress on megaminx.
    Do not extend a budget without a margin larger than the observed spread.

---

## 2. Where the remaining value is, by deadline (and what to do first)

### 2.1 cube555: the biggest gap between us and a number that is known to be achievable

Our own 555 model line solves 554/1035 pids at mean 169.7 moves against a counting bound
of 66.4 (2.55x). The CayleyPy paper reports **92.16 on the same states** (1.39x bound),
and `CUBE666_MODEL_RESEARCH_2026-08-24.md:37-45` already states "our 555 stack is ~1.9x
worse than the published CayleyPy stack on identical states". Diener leads at 109,512
(105.8/pid). A paper-quality scorer at our TPU width would land around **95k, #1 by ~14k**,
and 555 is the cheap proxy for every 666 question (same recipe, M = 72 vs 102).

The difference between our recipe and theirs is enumerated in that doc: dense
walk-depth labels on every walk state (we label 2 pivots per walk), K_max = 0.9-1.0 M
(ours 1.1 M), BatchNorm, 8B states/agent, beam 2^24 (ours 2^21), and **29-69 independently
trained agents min-merged**. The unresolved question is how much of the 1.9x is the scorer
and how much is that 500x search budget. That is the single most valuable experiment in
the portfolio and it is cheap:

- **A0** our sparse-Q ResMLP as is (control, 2^21, 6 frames, 20 fixed pids).
- **A1** paper recipe verbatim (dense V, K_max 65, BN, 4M ResMLP) at the same 2^21.
- **A2** sparse-Q + PieceTransformer (the tetraminx recipe, 8 corners + 24 + 24 wings +
  24 x 3 centre orbits... i.e. piece tokens) at matched wall.
- **A3** A1 with 4 independent seeds min-merged (tests the agent-ensemble mechanism at 4x
  instead of 69x).

Gate is V (recall-by-depth) plus inflation on the 20 pids; if A1 alone closes most of the
gap the lever is labels; if only A3 does, the lever is agents and the right response is
crowd-sourcing them (2.4).

### 2.2 cube444 (18 days): team leads by 320

The named live lever is the **QTM-native phase-2 solver**: the 374 reduction-family paths
in the public file finish their 3x3x3 in 19.91 QTM mean, so a QTM phase 2 drops the
two-phase floor from 48.2 to ~42-45 (`cube444/EXPERIMENTS.md:789-795`); everything else
(G1 membership, coordinates, PDBs, 24 frames) is built and verified. Second: settle the
arch-vs-objective question with a direct A/B on 444 (sparse-Q on our ResMLP vs on the
transformer) rather than guessing. Third: the unswept portfolio sources (public kernel
version histories, `cayleypy-beam-results`, `alexandervc/cayleypy-submissions`), with the
beam-quality ratio check first (rule 7f).

### 2.3 IHES (18 days): 30 moves above the plateau, all in the length-21/22 tier

We are listed separately at 21,972 (#5); the 21,870 plateau file on disk is a community
min-merge, so whether to submit it is an attribution decision, not a technical one.
Above the plateau: all 51 pids of length <= 20 are proven optimal; length-21 146/171 proven. Every miss is
exactly +2 (all 18 generators odd). What is left is a refutation-or-improvement sweep of
the length-22 tier at threshold 20, which costs ~9,500 s per pid on 64 threads, or ~1,400 s
with the 64 GiB quarter-turn pruning table that gave 6.7x. **The v6e TPU host has 180
cores and 1.4 TB RAM** (memory `tpu_host_is_a_big_cpu_box`); a depth-12/13 table and the
whole length-22 tier fit there. Nothing else on IHES is worth a run (rule: do not re-run
window rewriting, relation mining, MITM, bridge, splicing, Knuth-Bendix on this file).

### 2.4 Crowd-source the agent portfolio through public kernels

The paper's 69-agent min-merge is the mechanism we cannot afford ourselves but the
community will run for us: the tetraminx kernel's version history was worth -51 then -91
(`tetraminx/HANDOFF.md:99-124`). Ship the 555 (and later 666) beam as a public Kaggle
kernel where every push draws a fresh agent seed / frame subset, and sweep its versions
with `scripts/25_pull_kernel_versions.py`. Check the beam-quality ratio before spending
requests on other authors' kernels.

---

## 3. New approaches and improvements, ranked (answer to question 1)

Each item: what, why the evidence supports it, cost, and the falsifier. Items marked
[UNTRIED] appear in the docs as untried; [NEW] is not in any doc.

**3.1 [UNTRIED] Close the recipe gap on big cubes first (2.1).** Highest EV in the
portfolio because the target number is known. 1-2 GPU-days. Falsifier: none of A1-A3
beats A0 by > 10% on 20 pids.

**3.2 [NEW] Cluster-factorised PieceTransformer-Q for n x n x n cubes.** The 666 state
is a product of 9 invariant 24-sticker orbits (`cube666/README.md`). A flat 152-token
transformer costs ~4x tetraminx per state and attention is already at 2.5% MXU occupancy
(`TORCHTPU_FINDINGS.md`). Instead: one shared encoder over the 24 slot tokens of a
cluster (identity embedding + cluster-type embedding), producing one summary token per
cluster; a small cross-cluster transformer over the 7 summaries (+CLS); Q head with 36
outputs and an AZ value head from the same trunk. This encodes exactly the structure the
classical solver exploits (moves couple clusters; within-cluster state is a permutation)
at ~1/50th of full attention. Add **auxiliary per-orbit heads** (each cluster's own
walk-depth / cycle count) as training targets only, never as the scorer (the sum-of-orbits
scorer sits in the stabiliser trap: `kept 24/24/0`). Falsifier: recall-by-depth no better
than the flat ResMLP at matched wall.

**3.3 k_max = 0.9 M as a rule, with pivot mass re-tilted into the band that decides
survival.** Tetraminx measured the label degrading past the diameter; 666 arm A trained
20% of its pivots past mixing. `CUBE666_WINDOW_PLAN.md:234-248` restates the target as
0.090@40 -> 0.15-0.20@45. Free.

**3.4 [UNTRIED] Reconcile walk labels with the exact table where they overlap.**
`57_label_consistency.py` measured 37.1% of tetraminx pivots inside the d<=6 table and
25.9% of those with pivot index != true distance (mean gap 1.717). The fix is three
lookups per row. It is an objective change, so it needs its own matched control, but it
is the cheapest untested training-side gain and it transfers to every puzzle with an
exact ball (all of them). Falsifier: 15-pid beam total inside the 424-431 noise band.

**3.5 [UNTRIED] The slack-2 hinge.** Any two children of one parent are <= 2 apart
(triangle inequality), so `relu(q_undo - 2 - q).mean()` with stop-gradient on the
reference is a sound ranking regulariser; `top1_margin_weight` is shipped at 0.0 in every
config and has never been tested. Do NOT use the zero-margin "undo is argmin" version
(`CUBE666_WINDOW_PLAN.md`). Pre-flight: the violation histogram on a trained model is
free and tells you whether there is anything to fix.

**3.6 [UNTRIED] Variance reduction inside the scorer where width is anti-productive.**
On 666 the beam converges on scorer errors. Three cheap search-side arms, each gated on
the descent probe's `fix_mean >= 9.0` (random-state baseline) plus d<=4 ball hits:
frame-averaged Q (`Q_avg` / `Q_max` over 8-48 symmetry frames per candidate, distinct
from frames-as-retries which is a measured null), independent-checkpoint ensemble with
conservative `mu - beta*sigma` scoring, and lineage-smoothed EMA scoring. None has been
run on any puzzle as a scorer-side measure.

**3.7 [UNTRIED, IDEAS A7] Diversity-stratified beam.** Bucket candidates by a cheap
exact signature (per-cluster cycle-count vector on cubes; per-orbit progress on
megaminx-like puzzles) and cap each bucket's share of the beam, so one false basin cannot
take the whole width. Directly targets the failure the "different machine" report
measured ("a score that repeatedly prefers false low-value basins") and the 555 width
non-monotonicity. Test on the 666 descent probe (pids 927/852/665) and on the 555 pids
that 2^22 lost. Cost: one kernel change.

**3.8 Port the sym-pooled beam to 555/666.** Validated on IHES and megaminx (-16 net on
24 tail pids, pid 991 101 -> 82 at equal budget), never ported to the cube kernels.
Odd supercubes have 96 trajectories, so the pool has more to reallocate over than on
tetraminx (where frames saturate at 2).

**3.9 Beam-AVI only where its precondition holds.** The go/no-go probe (on-path
candidate percentile vs the 0.5 null) is necessary but not sufficient (tetraminx passed,
then lost). Use AVI on the beam distribution inside rungs / residuals where the scorer
already ranks well and is not at ceiling: 444 (validated), the 666 visible zone as a
visibility extender (3.13), never on a flat blind scorer.

**3.10 `history_depth` and exact-ball goal sets are free and should be on everywhere.**
h=1 is worth ~-8 on tetraminx (saturates at 1); h=4 is -14% on 555 (saturates at 4
because the graph is bipartite); `30_solve.py --history-depth` still defaults to 0.

**3.11 Exact-proof sweeps on the big CPU host** (2.3) for any puzzle where the remaining
gap is in a provable tier.

**3.12 Kernel version sweeps** for 555/666 (2.4).

**3.13 [NEW] AVI as a visibility extender, gated per round.** On 666 the flat scorer has
signal to depth ~75-80 and none past 85. Bootstrap targets only from states at depth <=
V (where the 444 result says it sharpens), harvest the beam's own frontier there, retrain,
re-measure V with the descent probe, and stop the moment V does not move. This is the
444 mechanism applied as a curriculum instead of a fixed point; the tetraminx flattening
happened at a ceiling this scorer is nowhere near. Falsifier: V unchanged after one round.

**Stop doing (measured):** KMC-imitation scalar V and its distillations (the other
machine's whole line; see 4.1); window rewriting on beam output; Beam-AVI on the flat 666;
GT / Perceiver / GNN V families; bigger V trunks under the same recipe; NISS as a
standalone; per-parent top-alpha shortlisting; progressive top-k in the TPU kernel;
subgoal search; bidirectional MITM; route relinking; FMC insertion on long-3-cycle puzzles.

---

## 4. cube666: a model-only solver (answer to question 2)

### 4.1 What the "different machine" report establishes, and what it does not

The report (`cayley-py-666-cube/cube666_new_report_different_machine.md`) is a
well-controlled negative result for one lineage: a state-only scalar V (DenseValueBN)
trained by imitation on 1.06M states from 8,874 KMC paths over 493 roots, with the label
U(s) = shortest observed KMC suffix, then DAgger, symmetry, full-trajectory labels, pair
losses, an early-prefix curriculum, a q96 relabelling campaign, and C2500-distilled 36-way
Q heads. Every variant improved an offline metric; none solved PID 502 at width 8,192 /
depth 400; the known 195-move path is evicted at depth 3-4 in every checkpoint; mean V
falls from 137 at depth 0 to 27 at depth 128 while true completion still costs 182-200.

Three things it gets right and we should keep: (a) U(s) is an upper bound, an observed
action is a positive but an unobserved one is not a negative; (b) sibling ordering is a
different quantity from calibration and is what beam search consumes; (c) the
recommendation to train a **native 36-action Q** and to gate on within-parent ranking and
reference-path survival before any wide beam.

What it does not establish, and where its recommendation goes wrong: KMC completion cost
is **non-Markov in the 216-sticker state** (`EXPERIMENTS.md:905-909, 1283-1288`: nine
pairs of rough words reaching the same state finish 2-6 moves apart) and phase-structured
(2.66 bits/move against a 4.86 ceiling), so "labels tied to actual downstream completion
cost" from a classical solver are the wrong target for a ranking scorer even if we could
afford them at scale; the corpus is four orders of magnitude smaller than what the working
recipes use (6-8B states); and the failure it measures is the same one our random-walk
line measured from the other side: **the flat scorer is blind at the depths where the
scramble lives**. Its q96 finding that beam-top vs cutoff completions differ by only +3
moves on average is the blind zone seen through labels. Use its assets (C2500, q96 pairs)
as an evaluation set for reference-path survival, not as training data.

### 4.2 The constraint in numbers

- 216 distinct stickers, 36 quarter-turn generators, branching 29.02, |G| = 3.1e149,
  counting bound **102.2 moves**, mixing at ~102; the graph is not bipartite (12 odd / 24
  even generators). Nine invariant 24-sticker orbits: 4 centre, 2 wing, 1 corner-ish
  (500.6 bits total, centres 316 of them).
- Test set (checked today): pids 12-1011 are a scramble-length ladder L = 1..1000; ~92%
  have L > 80, i.e. are effectively random states. Our 169,769 file costs 186.8/pid on the
  800 deep pids; Diener ~182. On the ladder below L ~ 100 our paths are ~0.92 L (sample
  post-processing), which is the only regime the flat model currently touches.
- Flat scorer (arm A): top-1-by-depth 0.090 / 0.043 / 0.035 / 0.029 at 40 / 60 / 72 / 85
  vs chance 0.028; 555 has 0.246 / 0.182. V ~75-80, start ~105-110, **Z = 25-30, 0 solves
  at any width**, and width makes it worse.
- **A complete model-only solver already exists**: the factorised 3-cycle finisher
  (`cube666/MACRO_POLICY_TRAINING.md:73-77`) solves 1,012/1,012 replay-verified, at ~705
  moves/pid (4.2x the KMC file). So "capable of solving fully" is done; the question is
  "at a competitive length", which means < ~165/pid to lead and < ~140 to matter.
- Classical reference: KMC = exact corners (10.35 mean) + GF(2) parity (~1) + SA over
  inner-slice words + 20k-wide 3-cycle insertion; state 500 = 11 + 67 + 126 = 193. Its
  slack is mostly global (the decomposition into cycle units): primitive-move re-solves of
  KMC windows at spans 34/58 by the flat model gave 0 (`submissions/cube666_model_fullpath_*`),
  while the 176,889 -> 171,019 harvest came from macro-level re-selection (the
  `macro_context` line) and radius-3/4 exact rewriting of loose classical words.

### 4.3 The programme, staged and gated

**Stage 0 - calibrate on 555, where the answer is known (2.1).** Nothing below is worth
running until we know whether our recipe or our search budget is the 1.9x. If A1/A2 reach
V/M >= 0.9 on 555 at 2^21, port the winning recipe to 666 with k_max = 92 (0.9 M). The
CayleyPy repo ships a 666 config (K_max 150) that nobody has reported on; run it as an arm.

**Stage 1 - a scorer with the cube's structure (3.2) trained at scale.** Cluster-
factorised transformer, sparse-Q rw-middle labels with k <= 92, exact d<=5 anchors (26.9M
states, every batch), 48 rotations x inversion as augmentation, the slack-2 hinge, per-
orbit auxiliary heads. Train on TPU (TorchTPU einsum path is 1.08x an A100 on tetraminx)
to >= 5B states; the compute is the same order as the 555 model (6.14B, 23 h on one
A100) once the per-cluster encoder replaces flat attention. Gate: V on the full puzzle.
V >= 95 opens the direct route (predicted 1.2-1.3x bound = 125-135/pid, ~130k, which
would lead by ~35k); V in 85-95 goes to Stage 2; V <= 85 closes the flat family on 666
for this recipe and Stage 2 is the only route.

**Stage 1b - extend V from inside (3.13).** One AVI round on beam states at depth <= V,
re-measure V, stop on no movement.

**Stage 2 - a two-rung ladder whose rungs are each inside the measured V/M.** Two
candidate chains; the first is [NEW], the second is the docs' P2.

- *Colour-relaxation ladder.* Rung 1: solve the **colour-cube projection** of the state
  (relabel the 216 stickers to 6 colours). This target set is the colour-stabiliser
  subgroup S; its coset graph has ~390 bits (the supercube's 500.6 minus ~110 bits of
  within-face centre identity), so **M ~ 80**, the same regime as 555 (M = 72) and far
  below the flat 666 (102). Train it exactly like a colour cube (walk-depth labels on
  colour patterns; 48 frames, no inverse frame because S is not normal, and colour
  symmetry needs the recolouring conjugation the 444 code already has). Rung 2: the
  residual is an element of S (~110 bits, counting ~23 moves, realistically 40-60 because
  it is commutator-bound); solve it to the identity with the **existing flat supercube
  scorer**, whose visibility of ~78 covers a start of 40-60 (Z < 0). Predicted total at
  paper-quality rung 1: ~112 + ~50 = ~160/pid (~162k, ahead of Diener); at our current
  555-quality rung 1 (2.4x) it fails, which is why Stage 0 comes first. Cheap gates:
  (i) rung-2 alone, on residual states built by conjugating random walks into S, must
  solve 20/20 at < 70 moves with the current arm A; (ii) rung-1 V/M on the colour graph.
  Note this uses the colour cube as a *relaxation target*, not as the puzzle model; the
  deep-research report was rightly rejected for the latter.
- *Corners-first six-rung orbit ladder (docs' P2).* Rung scorers D_k = distance to
  "clusters 1..k solved", trained on the ladder-start distribution and never as a sum of
  per-cluster values. Counting says 6 x ~21 + 11 ~ 137 at rung-1 quality, ~180 (a loss)
  at 28/rung; the two-cluster gate (P2-a/b) that decides it has never been run. Corners
  can be a learned scorer (a 2x2x2-supercube-sized problem) to keep the line model-only.

**Stage 3 - search-side robustness for whichever scorer survives (3.6, 3.7, 3.8).**
Frame-averaged Q, stratified diversity by cluster cycle-count signature, ensemble
`mu - beta*sigma`, `history_depth`, d<=5 ball as the goal set, sym-pooled allocation over
96 trajectories. Gate on the descent probe (`fix_mean >= 9.0`, ball hits on pids 927 /
852 / 665 / 502).

**Stage 4 - what "model-only" means, and the fallback.** Corners by a learned scorer,
rungs by learned scorers, endgame by the exact d<=5 ball (a table, not a solver), paths
replay-verified: that is model-only in the same sense as tetraminx. The factorised
finisher is the model-only fallback for any pid the ladder misses (complete, 705/pid).
The 3-cycle library it uses is classical knowledge; if that is disqualifying, the
fallback is the flat beam plus the ladder with no macro stage, and coverage must be
measured.

### 4.4 Score insurance that is not model-only (for the 2026-11-20 leaderboard)

The docs' P3' route: the model as an anytime *prefix* optimiser with the exact finisher
as terminal oracle, `A(p,a) = C(p) - C(pa)`, every beam state terminal, never worse than
the fallback, ceiling ~160k because the finisher's 2.0-2.3 bits/move stays in every
path. Plus the KMC parameter gate, 96-trajectory KMC merges, and consolidating the three
incumbents on disk (169,769 submitted, 171,019 and 170,051 in docs) into one ledger. The
Diener gap of 3.3k is a classical-tuning gap; the model programme above is the only
thing that can change the slope.

---

## 5. Other suggestions (answer to question 3)

**5.1 Make the visibility probe a mandatory gate for any new puzzle.** Before any beam
campaign: recall-by-depth from `53_probe_compare.py`, the descent probe with the
random-state `fix_mean` baseline, and the reference-path survival table from the other
machine's report. Together they give V, M, and Z in an hour; they would have saved the
2^21-2^22 666 campaigns and the entire imitation lineage.

**5.2 Unify the two 666 lineages.** Two machines reached the same wall from opposite sides
(random-walk labels blind past 85; imitation labels non-Markov and off-distribution) with
no shared ledger. One `cube666/LEDGER.md` with one incumbent, one gate set, and both
evidence bases; retire the imitation line explicitly so it is not re-run.

**5.3 Enter the cheap open competitions with the generic stack.** Pancake (14 teams),
glushkov (9), rapapport-m2 (6), reversals (2), transposons (2) close 2026-09-21/22. Each
is a permutation Cayley graph; `Solver` + sparse-Q + exact ball + frames + min-merge
duck-types onto them, and the generator sets are small. Pancake has strong combinatorial
baselines (Gates-Papadimitriou 5n/3 + O(1)); the others have almost no competition. Half
a day each to find out whether there is a placing; skip any whose leader is already at a
provable bound.

**5.4 cube777.** Not entered, 5 teams. `cube_nnn` verified the structure machinery on 666
(coordinates 0/400 mismatches, 88M corner PDB filled exactly) and 777 has the parity
invariant (n odd). A KMC-style port is a classical placing, not research; only worth it if
a collaborator wants it.

**5.5 Tooling debts with measured payoff.** TorchTPU beam has an unexplained 2.35x
(4.96 s benched vs 11.65 s in situ) and two unapplied wins (fused multiply-reduce hash
4.4x, endgame index rebuild 641 ms -> ~10 ms); `BlendedQ` silently disables
`--qv-consistency`; `22_export_az_v_only.py` drops the policy head; the IHES searcher still
lacks the compile padding hook. Each is a known silent-degradation source (memory
`dual_codepath_drift`).

**5.6 Write it up.** The blind-zone law with the bits/move ledger, the yield-graded-by-
path-length rule, "width beats scorer until Z > 10", and the exact-rewriting floors are a
coherent empirical account that no current paper states; the tetraminx blog is the
narrative half. The competitions close in three months and the community is small enough
that a write-up is also the cheapest way to recruit the agent portfolio (2.4).

---

## 6. If you only run three things this month

1. **The 555 recipe calibration (2.1 / 3.1)** - decides the scorer-vs-budget question for
   both big cubes and is the only route to a large leaderboard move before 2026-11-20.
2. **The 666 rung-2 gate (4.3, Stage 2, gate i)** - one afternoon with the existing arm A
   on residual states; it tells you whether the colour-relaxation ladder is alive without
   training anything.
3. **The IHES length-22 exact sweep on the TPU host (2.3)** - the 30 moves are in a
   provable tier and the hardware is idle between beam runs.
