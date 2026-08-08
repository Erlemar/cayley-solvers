# GFN-Pathfinding (Morozov et al. 2026) — analysis + megaminx application plan

Date: 2026-07-17. Sources analyzed:

- **Repo**: https://github.com/GreatDrake/gfn-pathfinding (MIT; JAX/equinox; cloned to
  scratchpad this session). Ships pretrained weights (`rubik2_model.eqx` 26 MB,
  `rubik3_model.eqx` 104 MB) and the exact test sets (incl. the DeepCubeA 1000-cube set).
- **Paper**: "Learning Shortest Paths with Generative Flow Networks", Morozov, Maksimov,
  Tiapkin, Samsonov (HSE / Ecole polytechnique), arXiv:2603.01786 (2 Mar 2026). 14 pp.
  NOTE: our EXPERIMENTS.md 2026-05-10 entry cites this same arXiv ID as "Pan et al. 2026"
  — misattributed; correct is Morozov et al. Same group as "Revisiting Non-Acyclic
  GFlowNets in Discrete Environments" (ICML 2025, arXiv:2502.07735), whose theory this
  builds on.
- **Video**: https://youtu.be/21Lf0IwoLMI — Russian-language research talk, "Poisk
  kratchaishikh putei v bol'shikh grafakh s pomoshch'yu generativnykh potokovykh setei"
  (= the paper's content; no method details beyond the paper).

---

## 1. What the method actually is

### 1.1 Construction (the clever part)

Given a graph G and a goal vertex (solved state), build a **non-acyclic GFlowNet
environment** where:

- **s0 = the SOLVED state** (the GFlowNet source, not the scramble);
- transitions = **reversed** edges of G (for a Cayley graph with involutive-closed
  generators this is the same 24-neighbor structure);
- **every state is terminal**: each state gets an extra "stop" edge to the sink sf;
- **reward R(s) = 1 uniformly** over all states, so Z = |V| exactly (known for a puzzle
  group — no learned logZ).

Then the two policies get natural puzzle meanings:

- **P_F** (forward) starts at solved and *scrambles* — it must reach every state of the
  group with equal probability, deciding when to stop (its stop logit doubles as the
  state flow: `F(s) = R(s)/P_F(sf|s) = 1/P_F(sf|s)`, so no separate flow net is needed);
- **P_B** (backward) is *the solver*: from any state it walks back toward s0.

### 1.2 Theorem (the load-bearing part)

**Theorem 3.4**: under reward matching, P_B minimizes the expected trajectory length
E[n_tau] **iff every trajectory with nonzero probability is a shortest path** between s0
and its endpoint. And E[n_tau] = (total flow)/Z, so *minimizing total flow == training
the solver policy to be exactly geodesic, for every state simultaneously*. Shortest-path
optimality is re-expressed as a probabilistic/flow property — no distance function, no
value regression anywhere.

### 1.3 Training algorithm (from `train.py`, verified against the paper)

Per step:
1. Sample a batch of on-policy trajectories from P_F starting at s0 (solved), stop action
   masked, fixed length `Nmax` (**smaller than the diameter is fine**: they use Nmax=12
   for 2x2x2 vs God's number 14, Nmax=24 for 3x3x3 vs QTM 26).
2. Loss = **Trajectory Balance summed over ALL prefixes** (each prefix + stop is a
   complete trajectory since every state is terminal), with the **true logZ plugged in**:
   `sum_i (logZ + sum_{t<=i} log P_F - sum log P_B - log F(s_i))^2`
   where `log F(s_i) = -log P_F(sf|s_i)`.
3. Plus **flow regularization** `lambda * sum_i F(s_i)` — this is the E[n_tau]
   minimizer that forces geodesics (code: `reg_coef * exp(logsumexp(log_flows[1:]))`).
4. Both P_F and P_B trained jointly; single ResMLP trunk (6 residual LN blocks, one-hot
   input), two linear heads. 1024 hidden for 2x2 (~7M), 2048 for 3x3 (~25M). AdamW 3e-4,
   grad-clip 100, batch 2048, 1M iters for 3x3. eps_explore available but 0 by default.

**lambda tuning rule** (README + Fig. 3): larger lambda = shorter paths, too large =
model fails to find any path. Rubik2 lambda=1e-2, rubik3 lambda=5e-7 (note the scale
drop as Z grows — see 4.3 below). Rule of thumb: pick the largest lambda that still
solves after a short smoke.

### 1.4 Inference

**Beam search on cumulative log P_B** (sum of log-probabilities along the path — a
text-generation-style beam, NOT a value-guided beam), with CayleyPy's dedup heuristic.
W=1 = greedy argmax. **One forward pass per state scores all children at once** (policy
logits), vs value-guided beams needing a forward pass per child — 12x fewer NN evals on
the cube, 24x on megaminx.

### 1.5 Results (Table 1, QTM, vs CayleyPy Cube = Chervov/Khoruzhii NeurIPS 2025 — our
KhoruzhiiSolver's direct lineage)

| Beam W | GFN 3x3 len / solve | CayleyPy 3x3 len / solve |
|---|---|---|
| 2^6  | 25.33 / 1.0 | x / 0.687 |
| 2^9  | 23.49 / 1.0 | 24.34 / 1.0 |
| 2^12 | 22.42 / 1.0 | 22.44 / 1.0 |
| 2^15 | 21.70 / 1.0 | **21.61** / 1.0 |
| 2^18 | 21.24 / 1.0 | **21.15** / 1.0 |

2x2: GFN reaches the optimal 10.64 at W=2^6; CayleyPy needs 2^10. Greedy (W=1) solves
100% of 2x2. Runtime at 3x3 W=2^18: 1.74 s/cube (25M params) vs 6.19 s (4M params) on
H200 — the 12x-fewer-evals effect.

**The honest headline: the GFN wins decisively at small-to-mid beam widths and roughly
TIES (slightly loses to) the V-guided beam at W >= 2^15 on 3x3.** Swap puzzle n=20
(2.4e18 states, diameter 190): learns the EXACTLY optimal policy having seen 1e9 states
— but swap's optimal policy is a local rule (swap any descending adjacent pair), so the
cube numbers are the honest anchor for megaminx, not swap.

---

## 2. Rule-9 reconciliation: what we already ran, and why this is still untried

We knew of this paper in May (EXPERIMENTS.md 2026-05-10 cites the arXiv ID) and built
"TB-inspired" trainers. **Every load-bearing design choice differed from the paper:**

| Design axis | Our m_tb_v0/v1 (May) | The paper |
|---|---|---|
| Trajectories | fixed random walks (off-policy), reversed | **on-policy from P_F**, starting at solved |
| Reward | R=1 at reaching V0 (goal-terminal only) | **R=1 at EVERY state** (uniform over group) |
| logZ | learnable scalar, penalized by lambda*logZ | **fixed to true log|G|** (known exactly) |
| P_B | uniform 1/24 (not learned) | **learned** (it IS the solver) |
| Geodesic pressure | lambda*logZ (distorts reward matching) | **flow reg lambda*F(s)** (provably geodesic) |
| Inference | log F used as a V in OUR value beam | **beam on cumulative log P_B** (policy beam) |

Verdicts on record: m_tb_v0 REJECTED (0/10); **m_tb_v1 ACCEPTED-but-worse (10/10, +16%
moves vs m_dd_v0)**; m_az_v0/v1/v2 rejected, m_az_v3/v4 accepted as dual-head (policy
usable, V needs Bellman). None of these tested the paper's actual mechanism. The +16%
is NOT evidence against the paper's recipe — our variant had no geodesic-forcing term
and never used the policy at inference.

Also NOT the same as: compass-report #6 "GFlowNet-triage" (that was GFN for per-pid
compute routing — rejected for unrelated reasons); the section-3 architecture survey
closure (that closed *V/Q encoder architectures*; this is a different **objective +
inference mechanism**, explicitly the "fundamentally different mechanism" carve-out).

**Relevant supporting evidence we already have:**
- **PHS cumulative** (section-13, validated 2026-05-24/25): score = V + w*sum(-log pi)
  along the path gave a real standalone gain (-67/50 on rand50, difficulty-monotonic).
  The GFN paper is the pure-policy limit of that same axis — the cumulative-log-policy
  signal is REAL in our production stack. Our pi there was AZ v4 (imitation of our own
  paths); the GFN P_B would be a principled, geodesic-trained pi.
- **Route-relinking oracle** (2026-06-14): all OUR solver families are correlated (0
  sole wins beyond best merge) because they all share V-geometry. Community CSVs (an
  independent solver family) contributed huge min-merge wins. A GFN arm shares *nothing*
  with our V stack — it is the exact kind of decorrelated arm min-merge rewards.
- **All-neighbors Q-head REJECTED** (Dir-1, 2026-06-14) is NOT a counterexample: that
  head was *distilled from the saturated V*, so it inherited saturation. The GFN policy's
  deep-state signal comes from flow conservation, not V regression. Whether it escapes
  the saturation wall is exactly the measurable question (Gate 2 below).

---

## 3. Honest fit assessment for megaminx

### 3.1 Why it could genuinely matter (mechanisms, not vibes)

1. **It sidesteps the documented wall.** Our meta-thesis: V saturates at ~25-30
   (problem-intrinsic across 7 encoder families), so nothing V-shaped ranks deep states.
   The GFN never ranks by distance — it only needs the 24-way *direction* choice to stay
   sharp at depth, trained by local flow-balance constraints that propagate global
   geodesic information. This is the first proposal since the survey closed that attacks
   deep states without a distance regression.
2. **Decorrelated min-merge arm** (inference-side, cannot regress; acceptance gate says
   any net min-merge improvement counts). Hard tail is where floor headroom is biggest
   (community ~69 vs our ~80 on the worst pids).
3. **24x cheaper node scoring** on GPU beams (one forward scores all 24 children). Our
   local/GCP value-beam wall is dominated by NN evals, so a policy beam buys ~an order
   of magnitude more width at equal wall. (NOT on the TPU 48M kernel — that is
   selection-bound; V is ~8% of step. The TPU port argument is diversity, not speed.)
4. **Composition with proven levers**: sym-ensemble (rotate state, solve, un-rotate —
   policy beams inherit our best lever unchanged); PHS-style mixed scoring
   `V + beta*sum(-log P_B)` (infra exists, `--phs-cumulative`); and a **V-agnostic
   qshort**: m23_v2 fails with any non-m05 V (Rule 15) because it is distilled from a
   specific V's landscape — a GFN P_B shortlister is tied to no V at all.
5. **Two-phase synergy** (the in-flight strongest lever): Phase 2 lives in subgroup H
   (12 TOP gens, diameter ~45, |H| << |G| and exactly computable). A GFN finisher for H
   attacks the dominant ~53-move Phase-2 leg, in a smaller graph where the method's
   scale risk is lower.

### 3.2 Why it could fail (equally concrete)

1. **The paper's own Table 1 crosses over**: at W >= 2^15 the V-guided beam is slightly
   *better* on 3x3. Our production widths are 2^16..2^20+ (TPU: up to 2^25+). There is
   no evidence GFN beats a tuned V-beam at our widths — the realistic win is per-pid
   diversity + width economics, not dominance.
2. **Scale extrapolation is unproven**: megaminx is |G| ~= 1.01e68 (vs cube 4.3e19),
   branching 24 (vs 12), diameter unknown but plausibly ~2x the cube's. The paper's own
   conclusion lists "scalability to extremely large graphs" as future work — we would BE
   that experiment. Uniform-reward coverage of 1e68 states is a much harder ask for P_F.
3. **Nmax must reach typical test-state depth.** They got away with Nmax slightly below
   diameter on the cube. For megaminx nobody knows the diameter; if effective random
   -state distance is ~40-60, training cost per iter grows ~2.5x over rubik3 and the
   flow-learning problem deepens. Needs an Nmax sweep (40/60/80), not an assumption.
4. **Policy sharpness at depth is the whole bet.** If P_B at d>=40 goes near-uniform
   (the policy-space analog of V saturation), beams stop being guided and the arm dies
   exactly where we need it. This is measurable early and cheaply (Gate 2) — it is the
   designated kill criterion.
5. **6M-cluster deja vu**: the ceiling might reappear in policy space. Mildly against:
   m23 (12.4M Q) never showed the V ceiling; the paper used 25M for 3x3 successfully.

### 3.3 Implementation gotchas found by reading the code (would bite a naive port)

- **float32 overflow in the flow regularizer.** `reg = lambda * exp(logsumexp(log_flows))`
  where F(near-s0) ~ Z/24. Rubik3: Z ~ 4.3e19 — fits float32 (barely; their lambda=5e-7
  also acts as a scale-tamer). Megaminx: F(d=1) ~ 1e67 — **exp() = inf in float32, NaN
  on step 1**. Fixes (pick one): regularize in log domain (penalize
  `logsumexp(log_flows)` directly — same argmin direction, different geometry), enable
  x64 for the reg term only (we already run JAX x64 on the TPU beams), or drop the first
  k prefix states from the reg sum (their flows are forced by conservation anyway; code
  already drops s0 itself via `[1:]`).
- **true logZ**: megaminx group order ~= 1.0067e68 -> ln Z ~= 156.7. Compute it exactly
  at impl time via Schreier-Sims on our 24 sticker permutations (sympy
  `PermutationGroup(...).order()`, degree 120 — instant). Same trick for subgroup H in
  the two-phase variant.
- **State encoding**: their one-hot(120 x 120) input = 14,400-dim — fine but fat; our
  embedding encoding (ResMLPGFlowNet already has it) is the natural swap-in.
- The repo is JAX/equinox with a fully jitted training loop (fori_loop) — ports
  naturally to Kaggle v5e-8 / zakhar's v4-8 (`~/envs/jax-tpu`) with our existing
  JAX-on-TPU playbook. A PyTorch smoke on the local 4090 can reuse the ResMLPGFlowNet
  trunk + `66_train_tb_proper.py` scaffolding (swap the loss + trajectory sampling +
  learned-logZ removal).
- Usual local rules apply if implemented here: ASCII-only prints (Rule 24), no literal %
  in argparse help (Rule 10), venv python path, Rule 12 periodic beam bench during long
  runs, Rule 21 strat-51 binding gate before any full-1001.

---

## 4. Proposed application plan (gated, cheap-first)

### Gate 0 — validate the artifact chain (~0.5 day, local, zero training)
Run their shipped `rubik3_model.eqx` through `eval.py` on the DeepCubeA test set
(CPU/4090; JAX-CPU is fine at W=2^9). Confirms we can load/run/port their stack and
reproduces Table 1 numbers. Also de-risks the eqx->our-stack weight translation.

### Gate 1 — megaminx smoke train (~1-2 days, 4090 or v4-8)
Port the trainer to megaminx (24 gens, embedding encoding, exact logZ, log-domain flow
reg). Hidden 1024-2048, Nmax 40, batch 512-2048, ~100-200k iters.
- **PASS**: greedy P_B recovers EXACT shortest paths on the BFS-d6 shell (we have exact
  distances for 19.4M states — a verification luxury the paper didn't have), and solves
  depth-20-40 random walks at high rate.
- **KILL**: optimality broken inside d<=6 after convergence, or unfixable instability in
  the reg term (try log-domain + lambda sweep per their rule-of-thumb first).

### Gate 2 — the money measurement: policy sharpness at depth (~1 day)
This is the saturation question transplanted to policy space, and the real go/no-go:
- **Depth profile**: greedy + W=4 + W=256 solve rate and length by scramble depth
  (d = 10..80). The V-analog check was "V@d80 - V@d40 > 10 = drift"; here it is "does
  P_B's top-1/top-4 stay decisive past d~30 or go uniform?" Also run an 09b-style
  recall-by-depth of P_B's child ordering vs verified-path moves.
- **Bench**: bare policy-beam at W=2^12..2^16 on the 11-pid two-phase spread + a
  strat-51 subset vs m_dd_v0/AZ-v4 value-beam at 65k. Given Table 1, expect GFN to need
  much less width for similar length; the question is the level it plateaus at.
- **KILL**: policy goes uniform past d~35 (the wall wins again) or plateau length is
  >15-20% above our per-pid floor on mid pids (then even min-merge value is thin).

### Gate 3 — deploy as a diversity arm (only if Gate 2 passes)
- Full-1001 policy-beam pass is CHEAP (24x fewer NN evals; local 4090 or one Kaggle GPU
  session plausibly suffices at W=2^14-2^16 + sym4) -> min-merge vs the 73,614 floor.
  Cannot regress; every hard-tail win is real.
- **Composition A/Bs** (each cheap, reuses existing infra):
  a. PHS-mixed scoring: `V + w*sum(-log P_B)` with w sweep — swap GFN P_B for AZ v4 pi
     in the validated `--phs-cumulative` path.
  b. GFN-P_B as global-top-alphaB qshort for the V arms (use the GLOBAL variant per the
     qshort-recall memory; gate on recall-by-depth, not pooled recall).
- Rule 26 discipline: report wins only vs the n-way per-pid min over ALL CSVs.

### Gate 4 (parallel option) — two-phase Phase-2 GFN finisher
Train the same recipe inside subgroup H (12 gens, exact |H| via Schreier-Sims, diameter
~45). Smaller graph = lower scale risk; directly attacks the two-phase stack's dominant
leg. Natural if the two-phase inference-lever work (sym-ensemble/preselect) is already
being run.

### Sidebar — the cube competition
The paper's exact benchmark is 3x3 and its baseline is our solver family's ancestor.
A picture-cube GFN arm (group order exactly known, same construction) is the same port
with smaller scale risk, and the IHES comp (24,618 vs Rokicki's 21,840) also pays for
decorrelated min-merge arms. If Gate 1 stalls on megaminx scale, downshift there first.

---

## 4b. Implementation log (2026-07-17, same day)

Gates 0-1 executed. Code: `src/cayley/gfn_pathfinding.py` (env construction, dual-head
ResMLP policy, prefix-TB + flow reg loss, policy beam with dedup),
`megaminx/scripts/120_train_gfn_pathfinding.py` (tasks rubik2 + megaminx),
`megaminx/scripts/121_gfn_depth_profile.py` (Gate-2 sharpness-by-depth probe).

- **GATE 0 PASSED — exact reproduction.** PyTorch port, paper hyperparams (h=1024,
  batch 128, Nmax 12, lambda=1e-2 flow-mode, lr 3e-4, clip 100): beam-256 on their
  shipped 100-cube test set reached **10.64 avg = the BFS-verified optimum** by ~275k
  iters (solve rate 1.0 throughout; greedy 97% at 325k). Checkpoint:
  `models/gfn_rubik2_v0/ckpt_latest.pt`. The port is faithful.
- **Megaminx v0 (lambda=2.4e-55, no explore): MODE COLLAPSE.** By 20k iters both
  P_F and P_B fully deterministic (H=0.000 at every depth, top1p~1.0), training data
  collapsed to one scramble tube, on-path logF climbing toward logZ — a self-consistent
  degenerate fixed point of on-policy prefix-TB. P_B pointed at the WRONG parent at
  21/24 depth-1 states despite total confidence.
- **eps_explore=0.1 breaks the collapse** (H(P_F) ~2.6-3.0 after restart) but rates
  stay barely above chance through 8k iters in all of lambda {2.4e-55, 1e-50, 1e-46}.
- **KEY SIZING INSIGHT: lambda must bite at the flow levels the model VISITS, not at
  the theoretical near-root flow.** Observed training flows sit at logF ~ 22-50 for a
  long phase; lambda·exp(logF) is numerically ZERO there for all lambdas sized against
  F(d1_true)=e^153 (2.4e-55..1e-46) — so the geodesic/anti-collapse pressure never
  fired in any early arm, and apparent differences between them were eval noise (n=24
  d1 states + chaotic SGD divergence). In the paper's rubik3 run, lambda=5e-7 starts
  biting ~1/3 of the way up the flow climb (logF~14.5 of 45). Megaminx equivalent:
  lambda ~ e^-50 ~ 2e-22 (bite at logF~50), or use logflow mode (bounded gradient,
  pressure at all scales). Probes D (flow 2e-22) / E (logflow 0.3) test this.
- Ops: local multi-hour runs MUST be launched detached (Start-Process cmd /c; the
  harness kills background Bash at ~20-60 min — took out both first launches);
  4 concurrent CUDA processes on the 16GB laptop 4090 produced one transient
  cudaErrorIllegalAddress crash — keep to <=3.
- **Corrected-lambda probes D (flow 2e-22) / E (logflow 0.3): NULL at the 8k-iter
  horizon** (2026-07-18). All five probe arms (lambda spanning 33 orders + two reg
  modes) end at chance-level greedy exact-d, healthy entropies, weak-but-real beam
  signal at d2-d3. Budget reality: 8k x 1024 = 8M trajectories vs the paper's rubik3
  run at 2B trajectories (1M iters x batch 2048) — we probed at 0.4 pct of their
  budget on a graph with 3.5x the logZ. Verdict: full-megaminx GFN training is a
  TPU-scale job (est. 1-3 days on v4-8/v5e-8 with the JAX port), not a laptop job.
  No local probe can cheaply falsify the full-scale question.
- **Decisive local falsifier: megaminx 2-face subgroup (megaminx2f) — MECHANISM
  WORKS, CONVERGES SLOWLY.** Gens {U,-U,F,-F}, order 8.0e12 (Schreier-Sims),
  lnZ=29.71 (between rubik2's 18.3 and rubik3's 45.2), diameter ~25. Nmax=24,
  lambda=6.6e-5 (bite at 34 pct of the flow climb = rubik3's ratio), eps=0.1,
  batch 1024, 60k iters (`megaminx/models/m_gfn2f_v0`). Result: real ranking
  policy on genuine megaminx moves/states — beam-64 solves 100 pct of d<=4 at
  OPTIMAL length, 62 pct of d6, 12 pct of d8/d12; radius extends ~+2 depth per
  45k iters; no collapse, no divergence. Greedy argmax stays soft (the
  fixed-true-logZ prefix-TB equilibrium leaves a persistent residual, rms ~
  0.26*logZ, and compresses flows ~3x — ALSO true of the working rubik2 run, so
  flow calibration is not the deliverable; P_B ordering is). Extrapolated
  full-diameter competence ~5-10x the 60k budget. `--learn-logz` variant (init
  true, 10x lr): **verdict reversed after the full 25k arc — it is the BETTER
  recipe at this scale.** logZ relaxes 29.7 -> ~3.7, TB reaches exact balance
  (~0.003) for the first time, and the policy snaps: broad-coverage transient
  peak at 12.5k (greedy d6 20 pct), then geodesic refinement (optimality up,
  radius recedes). Final profile beats fixed-Z's 60k endpoint at 2.4x less
  compute (P_B top1p 0.66-0.87 everywhere, greedy d3 0.45 vs 0.00, same beam-64
  radius). Mechanism: the geodesic theorem needs only positive reward, not
  uniform — the system chooses its own effective reward, making TB satisfiable.
  Cost: uniform coverage abandoned (radius recedes post-peak). Recipe for future
  runs: learnable logZ + keep BOTH the tb->0-snap checkpoint (best radius) and
  the end checkpoint (sharpest); the radius wall d~8-12 is budget-bound in both
  variants.

## 4c. TPU campaign (2026-07-19/20) — the definitive scale test

Full megaminx at paper-plus budget on v6e-8 (`megaminx/gfn_tpu/gfn_train_tpu.py`,
29M params, batch 4096, 10 it/s = rubik3's entire 2B-trajectory budget in ~28h).
Three configurations run to verdict:

1. **learn-Z + linear-logF reg**: fell into a newly characterized DEGENERATE EXACT
   SOLUTION of prefix-TB — both policies uniform + every flow = Z zeroes the
   residual identically while solving nothing. The paper's un-normalized
   lam*sum(F) prices this point out of existence; normalized variants do not.
2. **exp(logF/2) float32 homolog**: sculpts the flow field perfectly (steep
   profile, deep flows O(10)) yet produces ZERO deployable ordering — beam-256
   0/8 at d=6 after 430k iters = 1.76B trajectories.
3. **EXACT paper objective** (float32 via fixed-shift factorization
   lam*e^100 * mean(exp(logF-100)); fixed true Z): the only configuration that
   produced real ordering — beam-256 solves 50 pct of d=4, 25 pct of d=5, 8 pct
   of d=6 at 250k iters = 1B trajectories, and **solves test pids 1/2/3 at
   optimal lengths 1/2/3 (beam-1024, verified) — the first GFN solves of real
   competition pids**. pid 0 and everything deeper: unreachable.

**Measured scaling law: radius +1 depth per ~250M trajectories, decelerating.**
Competitive depth (30-70) is tens-of-billions-to-astronomically many
trajectories away. rubik3 (lnZ 45) fits in 2B; megaminx (lnZ 156.6) does not
fit in any purchasable budget. **Final answer: the mechanism transfers, the
scaling does not — GFN cannot produce a competitive megaminx solution.** The
negative is about scale, not implementation: the paper's own objective, faithfully
realized, at more than the paper's own budget.

Failure-mode taxonomy (all first-observed and characterized in this project):
mode collapse without exploration; inert lambda (bite-window mis-sizing);
degenerate uniform+flat-at-Z TB fixed point; ordering-free flow shaping under
normalized regularizers; fixed-Z squeeze vs learnable-Z effective-reward drift.
Artifacts: gs://mm-tpu-staging-0977634337/{gfn_paper_run,gfn_full_run}/.

## 5. Verdict

**Genuinely new mechanism, correctly aimed at both of our documented walls (V
saturation; arm correlation), with a cheap and decisive gate sequence — worth Gates 0-2
now.** It is NOT a likely straight upgrade over the tuned V-stack at our beam widths
(the paper's own Table 1 says so), so the 70K-anchored expectation should be: a
decorrelated arm contributing hard-tail min-merge wins + possible qshort/PHS composition
gains, with a small chance the depth-profile reveals policy-space immunity to the
saturation wall — which would be the first crack in that wall from any direction, and
would justify scaling to a full production arm on TPU.

Priority vs in-flight work: below finishing the two-phase inference levers and the
queued TPU deploy items (those have known positive EV); above all remaining training
-side section-13/B-list items (three ties in a row say training-side-at-6M via V is
exhausted — this is the only proposal that changes the objective rather than the
encoder).
