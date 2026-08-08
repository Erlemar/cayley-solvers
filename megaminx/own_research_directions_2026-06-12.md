# Own research directions (not replications), 2026-06-12

Companion to `chat_brainstorm_2026-06-12.md`. That doc covered catching up with
what Ivan/Vlad built. This one develops the chat's UNBUILT research threads --
RL on the search, one model serving both distance and search, symmetry losses --
into experiments that are ours, designed around the objections raised in the
chat and around our own negative results (m_rank_v0/m_sym_v0 ties, m31, m42,
frontier-regret v0 "benign alt-optima" lesson).

Recurring design rules used below:
- New LABEL TYPES, not re-weightings of the same RW/Bellman signal (the two
  Section-13 ties say re-weighting is exhausted at 6M).
- Deploy as a small residual head on a frozen production trunk (score =
  V + lambda * delta), so the working V cannot be destroyed; lambda starts tiny.
- Gates: strat-51 production recipe; d~20 V-variance canary (the real beam
  predictor, per m_repr_v0); for inference-side mechanisms, per-pid min-merge
  contribution at EQUAL WALL vs the mechanism it replaces.

---

## Direction 1 -- search-aware training ("train the system, not the net")

Chat seeds: Ivan's thesis (the unit that solves is net+beam, so train that
unit; reward states whose subtrees survive), Alexander's "penalize the net when
a node's descendants drop out of the beam within K steps" and his active
learning on stagnation states, Fedor's measurement (V~1.9-2.1 at true distance
6-8 near the end of stalls), Nikita's negative results (hard negatives didn't
help; MCTS-in-train too slow), Vlad's objection (survival reward = the model's
own preferences fed back = self-reinforcing suboptimum), Smolensky's objection
(dedup destroys credit: only one parent of a merged child gets recorded).

The objections are correct and they shape the designs: every label below is
grounded OUTSIDE the current model's preferences (wider search, exhaustive
micro-BFS, or terminal verified length) -- never in raw survival.

### 1A. Cross-width expert iteration (verified regret v2)

The missing ingredient of `new_iteration.md` Track 2 was a teacher stronger
than the production beam. We now have one for free: every giant-beam run
(96M-700M, ours or the community's) paired with a cheap narrow run (65k) on the
same pid yields GOLD mistakes of the narrow beam.

**Status 2026-06-13 — giant-beam label source exists, but teacher quality is
configuration-dependent.** The v6e-8 256M alpha24 pid991 run completed and
verified at length 74 (identity-only; community floor is 69), so it can provide
wide-vs-narrow trace data but is not an oracle-quality target on this pid. The
512M/8-chip kernel now mechanically fits and runs, but the first alpha4 qshort
smoke was both slow (~42 min/step) and poorly calibrated (`min_v=27.875` at
step 2). Treat 512M alpha4 traces as engineering smoke only; harvest 1A labels
from 512M runs only after alpha8/12 or alpha24 short smokes show competitive
early `min_v`.

Harvest: log per-step frontier hash sets in the narrow run (65k x ~70 steps x
8B ~ 36MB/pid). Backtrace the wide beam's shorter solution path; find the first
depth t where its state s_t is absent from the narrow frontier. At t-1 the
parent was in both; the on-path child was evicted. Emit:
  (parent, good_child = evicted on-shorter-path child,
   boundary set = the narrow beam's kept states nearest the cutoff at t).
This is verified regret -- the narrow beam demonstrably lost moves by this
decision -- not the benign alt-optima that poisoned frontier-regret v0.
Symmetric harvest for the over-ranked side: narrow-beam states kept >= K steps
inside stagnation basins that never appear within distance k of ANY known
solution path.

Train: frozen trunk + delta head; hinge loss so good_child outranks the
boundary set by margin; keep a V-distill term. Deploy score = V + lambda*delta.

Cost: harvesting is a byproduct of runs we already do; training is hours on
the 4090. Gate: held-out hard-pid eviction-recall first, then strat-51, then
min-merge contribution.

### 1B. Certified lower-bound anchors from stagnation basins (cheapest, soundest)

Fedor's failure mode is an ABSOLUTE calibration error near solved, and it is
certifiable: if V(s) < 3 but solved is not in BFS_r(s), then d(s) >= r+1 --
provable with a micro-BFS (r=3: ~14k states; r=4: ~300k; both trivial).

Pipeline: harvest states from production logs where the frontier-min V stayed
< 3 for >= 10 consecutive steps without solving; certify each with micro-BFS;
add hinge anchors max(0, (r+1) - V(s))^2 into the standard Bellman refine via
the existing n_anchor mixin (the exact pattern that fixed the V(V0)~2 bug).
We have never used certified lower bounds as targets; nobody in the chat
proposed certification at all (Alexander's version re-labels with a wide beam,
which is noisy and expensive).

Cost: a day of plumbing + one 50ep refine. Gate: Fedor-style near-solved
calibration plot (V vs certified bound) + strat-51 + the stall-rate on the 10
worst stagnation pids from the logs.

**Status 2026-06-12 — harvester + certifier + trainer hook SHIPPED; first
calibration result is a finding in itself.**
- `scripts/89_harvest_stagnation_anchors.py`: thin beam loop, harvests
  frontier states with V < 6.5, certifies via `bfs_bytes_d6` (exact d <= 6 /
  proven d >= 7), prints the calibration table, saves the anchor dataset.
- First harvest (AZ v4 V, 8 hard pids, beam 8192, all solved, ~no stalls):
  36,327 unique states -> 5,177 exact + 31,150 certified-LB anchors
  (`data/stagnation_anchors_v0.pt`). **AZ v4 does NOT have the Fedor
  pathology: V in [0,3) is 0 percent provably-wrong (the V0/d1 anchor fix
  likely already covers it). Its real failure is a systematic OPTIMISM BAND:
  V in [4,5) -> 59 percent provably d>=7; V in [5,6) -> 99 percent; mean
  certified gap >= 1.36 moves.** That band is the beam's endgame ranking
  zone — misranking there evicts true d<=6 states before the kernel's
  radius-5 neighborhood early-stop can rescue them.
- Trainer hook wired into `src/cayley/bellman.py`, flag-gated, default off:
  `stag_anchor_path` + `n_anchor_stag` (exact rows ride the main MSE) +
  `n_anchor_stag_lb`/`lambda_stag_lb` (one-sided hinge relu(lb - V)^2 —
  never pushes V down). Smoke config `configs/stag_anchor_smoke.yaml`
  validated end-to-end (2 ep, both pools loaded, loss falling).
- NEXT (gated): pick the refine recipe per Rule 13 (read the target model's
  training log header first); note the harvest SELECTION is model-specific —
  re-harvest with the model being refined (labels themselves are
  model-independent certified facts). Gates: calibration table improves in
  [4,6.5), d~20 variance canary flat, then 10-pid bench, then strat-51.

### 1C. REINFORCE through the beam (the honest moonshot, variance-engineered)

The only arm that literally trains net+beam end-to-end. Make selection
stochastic ONLY in the contested band: comfortable top stays deterministic
top-k; the ~1-2k candidates straddling the cutoff are sampled via Gumbel-top-k
at temperature tau (gives exact log-probs). Terminal reward R = -length
(production beams solve ~always, so the reward is dense per run). Gradient:
sum over band decisions of grad log p(keep) * (R - b), with b from GRPO-style
leave-one-out baselines across 8 stochastic replicas of the same pid (replicas
at beam 8k fit the 4090 together). Optional potential shaping with
Phi = -min_frontier V (policy-invariant). Update the delta head only.

Smolensky's dedup-credit problem: v0 accepts the bias (credit to the recorded
parent only); v1 emits dedup merge lists from beam_lab (cheap) and splits
credit across merged parents.

Why this can find what 1A cannot: it credit-assigns through selection dynamics
with no teacher, so it can improve decisions at the frontier of what ANY beam
solves (where no wider teacher exists). Why it can fail: variance. Kill gate:
on the 12-puzzle lab bench at beam 4k x 8 replicas, >= 1% mean-length gain
over frozen-V with p < 0.05 within 2 GPU-days, else kill without appeal.

### 1D. Learned beam schedule (Liuda's #94/#82, formalized)

From solved-run logs, fit P(state at score-gap g behind the frontier-best at
depth t ends up on the final path). Replace fixed width B with "keep everything
within learned gap g*(t), capped" -- per-depth adaptive width, possibly wide
early / tight late (her even/odd and second-chunk ideas are crude versions).
Pure inference, log-driven, no training; one afternoon in beam_lab, A/B at
equal mean wall.

---

## Direction 2 -- one model serving the search, not just distance

Chat seeds: Alexander's multi-output one-forward blend (+ rank-1 BatchEnsemble
arXiv:2002.06715, TabM), Ivan's "one kernel, no logic, just matmuls"
constraint, Andrey-the-skeptic's "different losses on one trunk is not an
ensemble", Smolensky's "shared trunk -> minimal diversity".

What our search actually consumes: (a) within-frontier RANKING, (b) absolute
proximity for stopping, (c) DIVERSITY across runs (sym4/multi-seed is our only
ceiling-breaker). A scalar V serves (a)+(b); nothing serves (c) inside a
single forward. That is the gap to attack.

### 2A. BatchEnsemble heads + head-voted top-k (an internal ensemble per forward)

One trunk, M=4 heads, decorrelated structurally (answers Smolensky): per-head
rank-1 elementwise trunk perturbations (BatchEnsemble -- still one fused
matmul), per-head bootstrap resampling, and per-head symmetry-coset
augmentation (see 3B). One forward emits 4 scores.

Beam use is the novel part -- not averaging but VOTING: each head nominates
B/4 candidates, frontier = union (+ shared-min fill). This implants multi-seed
ensemble diversity INSIDE one beam at ~1x cost, where today diversity costs us
K full runs. Solutions per head tracked; per-pid min across heads.

Gate vs history (m31, m42 landed in-cluster): judged at EQUAL WALL against
{V + multi-seed x4} and {V + sym4} on the strat-51 hard slice, by per-pid min.

### 2B. Sym-variance uncertainty head (free labels, three uses)

For any state, the spread of V across its rotation orbit is computable on the
fly during training (V on 4-8 conjugates) -- a label that costs nothing and is
OUTSIDE the model's single-state preferences. Train an aux head u(s) to
predict it. Uses, all inference-side:
  (i) cutoff tie-breaking at the beam boundary (prefer low-u when scores tie);
  (ii) adaptive symmetry allocation -- run sym-ensemble (or more rotations)
       only for pids/states with high u along the early path: spends the 4x
       sym budget where rotation outcomes actually diverge;
  (iii) stagnation early-warning (Fedor's basins should be high-u).
Nobody in the chat proposed PREDICTING orbit-variance; Vlad penalizes it away
(his models can't use this signal -- ours can, precisely because we keep V
non-invariant).

---

## Direction 3 -- symmetry in the loss, and its inference dual

Chat seeds: Smolensky's "penalty for differences across symmetric images --
was it ever studied systematically?"; Vlad: in-batch 3-4 syms + variance
penalty, and his own caveat that consistency training and sym-ensemble
inference PULL OPPOSITE WAYS; Alexander: "blend across syms for more stable
estimates"; our m_sym_v0 tie and m31 rejection.

The caveat is the key insight: OUR production depends on sym4 diversity, so
copying Vlad's penalty would attack our main inference mechanism. The right
question is economic: is symmetry worth more as MODEL ACCURACY (invariance) or
as SEARCH DIVERSITY (ensemble)? Spend it deliberately, not both ways at once.

### 3A. Orbit-mean self-distillation (projection, not penalty)

True distance is exactly invariant under conjugation by the 60 rotations, so
the invariant part of V is the signal and the orbit-variance is pure error.
Orbit-averaging = orthogonal projection onto the invariant subspace = a
mathematically clean denoiser. Train student (same 6M arch) on targets
mean_k V_teacher(R_k s R_k^-1) over 4-8 sampled rotations (+ optional short
Bellman after). Differs from Vlad (penalty term fighting the optimizer), from
m31 (rotated states as independent noisy samples), from m_sym_v0 (pairwise MSE
to one rotation, lambda=0.1): distillation moves the TARGET, not the loss
landscape. Directly attacks our documented killer (d~20 per-state V variance).

Then run the experiment that settles the economics AT EQUAL WALL on the
strat-51 hard slice:
  arm A: current V + sym4 ensemble        (diversity spending)
  arm B: orbit-distilled V + 4x width     (accuracy spending)
  arm C: orbit-distilled V + 4 seeds      (accuracy + cheap diversity)
If B/C >= A, sym-ensemble retires and its 4x wall is freed forever; if A wins,
we have proof diversity is the binding asset and 2A/3B rise in priority.

### 3B. Coset-consistency heads (structured diversity; pairs with 2A)

Partition the 60 rotations into M=4 cosets. Head m is trained sym-consistent
WITHIN coset m only (its augmentation/penalty never crosses cosets). Each head
becomes stable inside its class and systematically different across classes:
sym4's empirical diversity (per-rotation outcome spread ~9 moves) compiled
into one forward, principled instead of seed-random. Evaluate inside 2A's
voting harness at equal wall.

### 3C. Sym-pooled beam (pure inference; implement first)

Today sym4 = 4 independent searches of width B with a FIXED budget each.
Instead: ONE beam of width 4B whose start frontier is the 4 rotated copies of
the scramble, each state tagged with its rotation id (6 bits; walkback
translates moves by the tagged rotation -- the R_inv g R plumbing already
exists). The beam then allocates width across rotations DYNAMICALLY every
step: hopeless rotations shrink, promising ones grow. This strictly
generalizes Vlad's prescreen proposal (his is a one-shot staged version) and
dominates fixed allocation whenever V is comparable across subtrees. Guard
against V-bias hogging with per-rotation floor quotas (>= B/2 slots each).
Near the solved end, rotated subtrees converge into each other's orbits and
the shared dedup pool folds them automatically -- small free width bonus.

Notes: orbit-FOLDING the whole frontier (dedup by canonical orbit
representative) is NOT worth it mid-search -- orbit collisions are
measure-zero for random scrambles away from solved; the pooled version above
captures all the realizable value without canonical-hash machinery.

Cost: days in beam_lab + the JAX kernel (backptr packing gains 6 bits/entry --
mind the 23-bit overflow precedent). Risk: low; same total wall as sym4 by
construction. This is the single fastest-to-value original idea in this doc.

**Status 2026-06-12 — IMPLEMENTED (beam_lab) + first A/B positive.**
`beam_lab/beam_search_sympooled.py` (SymPooledSolver) + driver
`scripts/88_sympooled_ab.py`.
- v0 (single global top-K*B over raw V across frames) FAILED the smoke test:
  V is not comparable across rotation frames (per-frame bias + near-solved
  noise), the pool defected from the leading root late in the run (occupancy
  flipped [3072,1024]->[1024,3072]) and solved nothing where seq solved
  easily. Lesson recorded in the module docstring.
- v1 two-level allocator: width split by softmax of per-root PROGRESS
  (EMA of minV_r(0) - minV_r(t) -- baseline subtraction cancels frame bias);
  within-root selection stays plain top-w_r by V. K=1 reduces bit-exactly to
  the production solver (self-test asserts this).
- Smoke A/B (AZ v4 V, K=2, B=2048 each vs pooled 4096 total, pids
  150/250/350/450): pool 2 wins / 1 tie / 0 losses, -18 moves over the 3
  both-solved pids, ~10 percent less wall, and pool SOLVED pid 450 which seq
  failed at both rotations -- dynamic width = the productive frame ran at up
  to ~2x its seq width.
- K=4 hard-slice A/B (B=4096 each vs pooled 16384 total, pids
  650/750/850/950/991): pool -9 total, 2 wins / 1 tie / 2 losses (+1 each),
  wall -18 percent (604s vs 738s). Headline: pid 991 = 97 vs seq-best 107
  (-10) -- the pool concentrated width on the productive rotation. Seq had a
  rotation return None on 3/5 pids; the pool never failed (dead frames'
  width redistributes). Losses are +1 noise on pids where seq's best rotation
  was already comfortable at 4096.
- **pool+inverse (8 roots = 4 rot x {orig, inverse}, SAME 16384 total): 504
  vs pool 516 vs seq 525. Inverse roots won 3/5 pids; pid 991 = 92 (-15 vs
  seq sym4 at identical budget).** +2/+3 dilution on the two easiest pids.
  This revives NISS at zero extra wall (Rule 11 dropped it for 2x wall):
  the allocator feeds inverse frames only when they are productive.
- **PRODUCTION-WIDTH GATE PASSED (24 tail pids, 16k/rot vs 8 roots @ 65k
  total): pool -16 net (8W/8T/8L), wall -7 pct, headline 991: 101 -> 82
  (-19) where all four seq rotations were mediocre.** Loss decomposition:
  one bounded mechanism (width committed to a frame whose final path is
  2-5 longer; path-length deltas invisible in mid-run V-descent) — max +5,
  never catastrophic. Inverse frames won 6/8. Verdict: equal-wall the pool
  is a modest free win on the ordinary tail and a LARGE win on stubborn
  pids; the static ensemble's independent full-width mins remain strong
  when any single rotation is good. DEPLOY: (a) inverse-frame axis to TPU
  notebooks (no kernel change); (b) beam_lab pooled+inverse rescue passes
  now; (c) JAX allocator port for 48M scale (GO, after a+b). Records:
  EXPERIMENTS.md 2026-06-12 entries; results JSONs in `results/`.

---

## Suggested sequencing

1. 3C sym-pooled beam (inference-only, equal-wall win or clean null).
2. 1B certified stagnation anchors (cheapest training fix, sound labels).
3. 1A cross-width expert iteration (labels accumulate from every giant run).
4. 3A orbit-mean distillation + the A/B/C economics experiment.
5. 2A+3B BatchEnsemble/coset heads with head-voted top-k (one training run).
6. 2B uncertainty head (rides along any training run above).
7. 1C beam-REINFORCE (moonshot; only with its kill gate armed).
8. 1D learned beam schedule (filler afternoon project, log-driven).

What I would NOT do from the chat's research talk: Ivan's frozen-interpreter
transformer (fascinating, zero score-path), diffusion as a global scorer
(our own new_iteration.md already demoted it correctly), raw survival reward
(Vlad's objection stands -- 1A/1B/1C are the grounded versions), K-step
composite-move V (m22 rejected it; only the curated-macro variant on the
shortlist remains).
