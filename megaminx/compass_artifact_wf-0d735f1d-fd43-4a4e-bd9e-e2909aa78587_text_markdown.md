# Novel Experimental Directions for a Learned-Heuristic + Search Megaminx Solver

## TL;DR
- The single highest-EV, literature-proven lever you have not fully exploited is a **Q-style "all-neighbors-in-one-forward-pass" value/Q head** (DeepCubeAQ / Q* search): it removes the 12–24× per-node forward-pass tax that DeepCubeA-style and CayleyPy-style beams pay, converting directly into a ~24× larger *effective* beam at fixed GPU compute — exactly the lever (bigger beams) that already works for you, and precisely what frontier Megaminx competitors do (a sub-80k Kaggle submission used "a model that predicts distances for all neighbors at once" at beam 2^27 on one A100).
- Your two best *signal-side* (not capacity-side) bets are a **distributional/quantile value head** (rank the beam by a low quantile rather than the mean — directly attacks your diagnosed beam-quality predictor, per-state V-variance at mid-depth) and an **ensemble of several diverse ~1–2M-param value nets instead of one 6M net** (the deep-ensemble literature shows diversity *helps small nets and hurts large ones*, and the ensemble's disagreement is the same diversity your symmetry-ensemble already monetizes).
- The three ideas you are weighing rank: **(A) unify distance+search into one model** is real (GFlowNet pathfinding, arXiv:2603.01786) but only *wins at small beams* and is neutral-to-slightly-worse at the large beams you live in — so adopt only its Q-head/amortization slice, not the full learned-search; **(B) RL beyond value iteration** is worth it *only* if it changes the label/signal (distributional or flow-based), not as "more RL"; **(C) train 100–500 epochs** is already falsified by your own 184-vs-50-epoch result and the overfitting literature — do not pursue.

## Key Findings

**1. The structural advantage frontier competitors have is throughput per node, not a better heuristic.** DeepCubeA and CayleyPy-style beams call the network once *per child*, i.e. 12× (Rubik) or 24× (Megaminx) forward passes per expanded node. Agostinelli et al.'s Q* / DeepCubeAQ (arXiv:2102.04518) computes transition-cost-plus-heuristic for *all* children in a *single* forward pass; the same paper reports a 157-fold increase in action space "incurs less than a 4-fold increase in computation time when performing AQ* search," and that DeepCubeAQ "achieves up to 129 times faster performance and 1228 times fewer nodes generated compared to A* search." The GFlowNet pathfinding paper independently confirms "approaches like DeepCubeA require 12 times more neural network evaluations … with the same beam width" because its policy "outputs … logits corresponding to every neighbor with a single forward pass." On the SpeedSolving Megaminx thread, the sub-80k Kaggle competitor explicitly used "an improved model that predicts distances for all neighbors at once" at beam 2^27 on a single A100. This is the lever.

**2. Your diagnosis that per-state V-variance at mid-depth (not mean calibration) predicts beam quality maps cleanly onto distributional RL.** QR-DQN (Dabney et al. 2018, arXiv:1710.10044) approximates the return as "a uniform mixture of N Diracs" with N "empirically set as 32," and the fully-parameterized quantile function line (Yang et al., arXiv:1911.02140; IQN/FQF) lets you query the τ-quantile for any τ∈U([0,1]). A heuristic that outputs quantiles lets you rank the beam by, e.g., the 10th-percentile distance — an optimistic-but-calibrated signal that is a different, lower-variance ordering than the mean even when the mean is identical. This is a genuinely new *signal source* (constraint 4 asks exactly for that), not another loss term on the same scalar target.

**3. The deep-ensemble literature directly supports replacing one 6M net with several diverse small nets.** Abe et al., "Pathologies of Predictive Diversity in Deep Ensembles" (arXiv:2302.00704), a study of "nearly 600 neural network classification ensembles," finds diversity interventions "can improve the performance of small neural network ensembles … [but] they harm the performance of the large neural network ensembles most often used in practice." CMU's Multi-Heuristic A* line ("Efficient Search with an Ensemble of Heuristics") shows many "weak" heuristics in independent searches beat one combined heuristic. Since you are capacity-saturated at 6M (information-bound), an ensemble of diverse small heuristics is the regime the theory says wins — and inter-model disagreement supplies the search diversity your symmetry ensemble currently has to manufacture.

**4. GFlowNet pathfinding (arXiv:2603.01786) is the strongest concrete instantiation of "learned search," but its advantage is concentrated at small beams.** On 3×3×3 (1000-cube DeepCubeA test set, QTM), their flow-regularized trajectory-balance GFlowNet vs CayleyPy: at beam 2^6 they solve 100% at avg length 25.33 while CayleyPy solves only 68.7%; at 2^9, 23.49 vs 24.34; but at 2^18 CayleyPy is slightly *shorter* (21.15 vs 21.24). Their own text: "better performance across smaller beam search widths from 1 to 2^9, while showing comparable results for larger values." Their 3×3×3 model is 25M params (vs CayleyPy 4M) — a *policy/flow* model, not a value model, so it is not subject to your value-model capacity regression. Training: hidden size 2048, λ=5×10⁻⁷, 1M iterations batch 2048 on an H200 in JAX (the public repo lists train_steps=2M — a discrepancy to note).

**5. Group theory offers an exact, model-independent label source and an exact tail solver.** Minkwitz's algorithm (1998) and Schreier-Sims strong generating sets produce *guaranteed* factorizations of any permutation into generators (long, but exact and instant). For Megaminx this gives (a) a never-fail fallback, (b) exact short-word macros for the last few cosets, and (c) — most interestingly — exact distance/word labels grounded *outside* the model's own preferences, which is the only kind of new loss the constraint list says has non-trivial prior. Korf's 1997 pattern-database work (three PDBs: 8 corners + two groups of 6 edges) with IDA* produced "the first optimal solutions to random instances of Rubik's Cube" at a "median optimal solution length [of] 18 moves" (Korf & Felner, *Artificial Intelligence* 134:9–22, 2002).

**6. Saturation is naturally respected by policy/ranking/bounded heuristics, not by unbounded value regressors.** Your requirement that V flatten past true diameter is automatically satisfied by (a) a policy/flow head (no magnitude to grow), (b) a quantile head with bounded support, and (c) bidirectional designs where the far region only needs *ranking*.

## Details — Prioritized Surviving Directions

### Direction 1 — Q-head: all 24 child values in one forward pass *(High-EV)*
**Thesis:** Stop paying the 24× forward-pass tax; output all child diffusion-distances (or transition-cost + child-h) in a single pass, spending the saved compute on a larger effective beam.
**Mechanism:** Identical trunk, output layer of 24 heads (one per move) predicting the value of the *resulting* child (equivalently a Q over moves). Beam expansion then needs one forward per *parent*, not per child. Per Q* (arXiv:2102.04518), a 157× action-space blow-up cost <4× compute and yielded up to 129× faster search with 1228× fewer nodes generated; here the win is the inverse — ~24× fewer evals at fixed beam, i.e. ~24× larger beam at fixed wall-clock.
**Why it beats your walls:** It does nothing to model capacity (sidesteps the 6M ceiling — it's a throughput change). Saturation unaffected (still ranking). It *multiplies* the pure-inference-scaling lever you already trust (constraint 7) rather than competing with it.
**Minimal experiment:** Retrain the existing 6M trunk with a 24-way output head, distilling from your current V on child states (cheap, no new labels). Measure nodes/sec and total move count at matched wall-clock vs your current per-child beam.
**Falsifier:** If matched-wall-clock total score does not drop (throughput gain eaten by worse per-node ranking from the shared-trunk multi-head), abandon. Watch for value miscalibration across heads.
**Compute:** One retrain on the 24GB GPU (days); inference unchanged. Cheap.
**Citations:** Agostinelli et al., Q*/DeepCubeAQ (arXiv:2102.04518) — single-pass child costs, 157×-action-space-for-<4×-compute, 129× faster / 1228× fewer nodes; GFlowNet pathfinding (arXiv:2603.01786) — confirms the 12× eval-reduction mechanism; SpeedSolving Megaminx thread — frontier competitor used an all-neighbors model at beam 2^27 for sub-80k.
**Novelty vs tried list:** Your "shortlister" prunes 24→α but still evaluates survivors per-child; this collapses the value evaluation itself into one pass. Different mechanism.
**Confidence:** High; **demonstrated elsewhere** (Q* and the Megaminx frontier both).

### Direction 2 — Distributional / quantile value head *(High-EV, cheap)*
**Thesis:** Rank the beam by a low quantile of the predicted distance distribution, not the mean — attacking the variance you identified as the true beam-quality predictor.
**Mechanism:** Replace the scalar head with 32 quantile outputs (QR-DQN's canonical N) trained by quantile-regression Huber loss. At inference, score candidates by, e.g., the τ=0.1 quantile (optimistic) or by mean − κ·IQR. The inter-quantile spread is itself a per-state uncertainty estimate you can use to down-weight high-variance mid-depth states in beam selection. FQF/IQN (arXiv:1911.02140) generalize this to arbitrary τ if you want a continuous knob.
**Why it beats your walls:** Directly operationalizes constraint 3 (variance, not mean). New signal, not a new loss on the old target (escapes constraint 4). Quantile support is bounded → respects saturation (constraint 2). Same trunk → 6M ceiling untouched.
**Minimal experiment:** Add 32 quantile heads to the current trunk; train on the same random-walk targets; sweep scoring quantile τ∈{0.05…0.5}. Compare total move count and, as a diagnostic, correlate per-state inter-quantile range with beam survival of the optimal child.
**Falsifier:** If no τ beats mean-ranking at matched beam, and quantile spread does not correlate with mid-depth beam errors, the variance story is wrong — drop it.
**Compute:** Trivial architectural delta; one retrain. Very cheap.
**Citations:** Dabney et al. QR-DQN (arXiv:1710.10044) — N=32 Diracs, quantile-regression; Yang et al. FQF (arXiv:1911.02140) — sampled-τ quantile querying with bounded, rankable support.
**Novelty:** None of your rejected loss tweaks changed the *target object* from a scalar to a distribution. This does.
**Confidence:** Medium-high; **proven** in RL, **speculative** for this beam-ranking use.

### Direction 3 — Ensemble of diverse small value nets *(High-EV, composes with symmetry)*
**Thesis:** Several diverse ~1–2M nets, scored by min (or by disagreement-aware aggregation), beat one 6M net and supply the diversity your symmetry ensemble currently fabricates.
**Mechanism:** Train K∈{4,8} small nets differing by seed, random-walk depth distribution, input symmetry frame, and architecture family. At inference, beam-score by min-over-models (optimistic) or rank by a disagreement-penalized score. Because models disagree, the union of their top-B children covers more good basins at the same node budget.
**Why it beats your walls:** The capacity ceiling is per-net; you are buying *signal diversity*, which Abe et al. (arXiv:2302.00704), across ~600 ensembles, show is the lever for small nets specifically and a *liability* for large ones. Variance is reduced where models agree and flagged where they disagree (constraint 3). It is the *economically honest* version of your symmetry trade-off (constraint 6): diversity created at training time across models, not destroyed by a consistency loss.
**Minimal experiment:** Train 4 × 1.5M nets with deliberately varied data/symmetry; compare (a) min-ensemble vs single 6M at matched *total* params and matched inference FLOPs, and (b) ensemble-on-top-of-symmetry vs symmetry-alone.
**Falsifier:** If 4×1.5M ≈ 1×6M at matched FLOPs and adds nothing on top of symmetry, the diversity is redundant — stop.
**Compute:** K cheap trainings (each smaller than current); inference is K small passes ≈ one big pass. Fits 24GB; embarrassingly parallel on TPU.
**Citations:** Abe et al., Pathologies of Predictive Diversity (arXiv:2302.00704) — diversity helps small, harms large, across ~600 ensembles; Multi-Heuristic A* / "Efficient Search with an Ensemble of Heuristics" (CMU); Neural Ensemble Search (arXiv:2006.08573).
**Novelty:** Your rejected experiments all kept *one* model; this is a committee, and the theory specifically predicts the small-net regime wins.
**Confidence:** Medium-high; **demonstrated** in classification/planning, **speculative** transfer to your beam.

### Direction 4 — Bidirectional beam with exact backward frontier / coset PDB *(Speculative, cheap-ish)*
**Thesis:** Grow an *exact* backward frontier from solved (BFS table or pattern database over the last cosets) and run the learned forward beam to meet it — so the near-solved region is exact and V only ranks the far region.
**Mechanism:** Precompute an exact BFS set / additive PDB covering all states within radius r of solved (or within a chosen coset, Thistlethwaite-style). Forward beam guided by V; terminate when any beam node lands in the backward table, then append the exact tail. A learned front-to-front meeting-point scorer (NBS/MM lineage) prioritizes beam nodes likely to connect.
**Why it beats your walls:** Splits near/far regimes (constraint 2): saturation is fine because the far region is pure ranking and the near region is exact, never relying on V's flattened tail. The exact tail removes the calibration burden near solved, where small V errors are most costly. Composes with Directions 1–3.
**Minimal experiment:** Build the radius-r exact backward set (RAM-limited), add "node ∈ backward set" termination + exact splice to the current beam, measure tail-length reduction on the hardest 100 instances.
**Falsifier:** If the reachable backward radius is too small to be hit before the beam would solve anyway (meeting at depth ≈ full solution), the gain is nil — abandon.
**Compute:** PDB/BFS precompute is memory-bound (one-time); inference adds a hash lookup per node. Cheap at runtime.
**Citations:** Korf (AAAI 1997) + Korf & Felner additive/disjoint PDBs (*Artif. Intell.* 134:9–22, 2002) — first optimal Rubik solutions, median 18; general additive abstractions for TopSpin/Pancake (arXiv:1111.0067) — the "no nice structure" puzzles most like Megaminx; NBS/MM near-optimal bidirectional search; IDBiHS restricted-memory bidirectional (ICAPS).
**Novelty:** Your post-processing does small-radius BFS *window replacement*; this uses an exact backward *frontier as a search terminator* — a different use of the same primitive.
**Confidence:** Medium; PDBs **proven** for puzzles, the learned-forward-meets-exact-backward combo **speculative** for Megaminx.

### Direction 5 — Determinantal / diverse beam selection at fixed width *(Speculative, cheap)*
**Thesis:** Select the beam to be high-value *and* mutually diverse (DPP / diverse beam search), so a fixed B covers more distinct basins instead of B near-duplicates.
**Mechanism:** At each beam step, instead of top-B by value, solve an approximate k-DPP / subdeterminant maximization over (value = quality, state-similarity = repulsion). Greedy submodular selection keeps overhead modest; similarity can be Hamming distance on facelets or trunk-embedding cosine.
**Why it beats your walls:** Beam collapse to near-duplicates is a known failure that high V-variance exacerbates; diversity selection is the *search-side* counterpart to Direction 2's model-side fix. Composes with the symmetry ensemble (constraint 7) and the Q-head (Direction 1).
**Minimal experiment:** Drop in greedy DPP selection with a Hamming kernel at one beam width; sweep the quality/diversity weight; measure total moves and beam de-duplication rate.
**Falsifier:** If your beams are already diverse (low duplicate rate at your widths), DPP adds cost with no gain — kill.
**Compute:** Greedy DPP is ~O(B²·d) per step, parallelizable; risk is that at B~1e5–1e8 the kernel dominates — test at small B first.
**Citations:** Determinantal Beam Search (Meister et al., arXiv:2106.07400); k-DPP diverse decoding (Stanford CS224n report); CayleyPy "X-trick" (arXiv:2502.18663) — a domain-specific beam-pruning rule ("don't undo an already-sorted prefix") that "drastically increase[d] the size of solvable graph," an existence proof that *beam-selection* changes can dominate model changes.
**Novelty:** Orthogonal to all rejected model-side tweaks; it changes *which* survivors the beam keeps.
**Confidence:** Medium-low; **proven** in NLP decoding, **speculative** at your beam scale.

### Direction 6 — GFlowNet flow-regularized policy as an amortized triage for the easy majority *(Speculative)*
**Thesis:** Use a flow-regularized trajectory-balance GFlowNet policy to *cheaply* solve easy/medium instances at tiny beams, reallocating saved compute to the hard tail.
**Mechanism:** Train PF/PB with detailed-balance + flow regularization (λ tuned by "largest λ that still finds valid paths"). Solve easy instances near-greedily (the paper solves 47.1% of 3×3×3 at beam 1, 98.4% at beam 8); route only hard instances to the expensive value-beam. Because the score is a SUM over 1001, cheap wins on the easy majority free GPU-hours for the ~70–130-move hard cases.
**Why it beats your walls:** Policy head has no magnitude → saturation-proof. It does not need to *beat* your large beam (it won't, per the data) — it needs to *free compute* so your large beam runs wider where it matters. That reframes Idea A as a budget-allocator, not a replacement.
**Minimal experiment:** Train the GFlowNet on Megaminx (their JAX repo generalizes), measure the fraction of the 1001 solved within ε of your current length at beam ≤64, and the GPU-hours reclaimed.
**Falsifier:** If easy instances are already cheap for your value-beam (likely), reclaimed budget is negligible — abandon. Also falsified if Megaminx's larger group prevents convergence at a λ that still solves.
**Compute:** One GFlowNet training (H200-class ideal; slower on 24GB or a TPU pod). Medium.
**Citations:** Morozov et al., Learning Shortest Paths with GFlowNets (arXiv:2603.01786) — 47.1%@beam1 / 98.4%@beam8 on 3×3×3, 25.33@2^6 (solve 1.0) vs CayleyPy X/0.687, single-pass neighbor logits, λ=5×10⁻⁷; trajectory balance (Malkin et al.).
**Novelty vs ideas A/B:** This is the *defensible* slice of A — amortized search as a triage stage, not the main solver.
**Confidence:** Medium-low for net score; **partially demonstrated** (small-beam wins are real; the budget-reallocation framing is speculative).

### Direction 7 — Exact group-theoretic labels & tail solver (Minkwitz / Schreier-Sims / coset PDB) *(Long-shot, only genuinely-new label source)*
**Thesis:** Generate exact factorizations to (a) supply training labels grounded outside the model's preferences and (b) guarantee a never-fail, splice-able tail.
**Mechanism:** Build a strong generating set; use Minkwitz to factor states near the solved coset into exact words; use these as exact-distance anchors *deeper* than your current depth-1 anchors, and as a fallback solver. Optionally phase the solve (Thistlethwaite-style coset chain) and learn a separate heuristic per phase.
**Why it beats your walls:** Constraint 4 says new losses only have prior if the *label source is new*; exact factorizations are exactly that. Phasing gives Megaminx the two-phase decomposition it structurally lacks (no Kociemba exists).
**Minimal experiment:** Implement Minkwitz on the Megaminx group; measure average word length (likely long) and whether using its outputs as extra anchors at depths 2–6 improves V-variance / beam quality.
**Falsifier:** If Minkwitz words are so long they don't usefully constrain the near-solved metric, or anchors at 2–6 don't move beam quality, drop.
**Compute:** Group machinery is CPU-bound, modest; the risk is engineering effort, not FLOPs.
**Citations:** Minkwitz 1998 (factorization in permutation groups); Schreier-Sims SGS (arXiv:math/0410593); Kalka-Teicher-Tsaban shorter words (arXiv:0804.0629, caveat: Sₙ/Aₙ only); Korf/Felner additive PDBs.
**Novelty:** Genuinely new label provenance vs all rejected loss tweaks.
**Confidence:** Low on direct score impact, **high** that the label source is novel; mechanism **proven** to produce exact words, **speculative** that it improves the learned heuristic.

## Tier Table

| Tier | Direction | One-line why |
|---|---|---|
| **High-EV** | 1. Q-head (all-neighbors single pass) | ~24× effective beam at fixed compute; multiplies your trusted lever; proven by Q* and the Megaminx frontier |
| **High-EV** | 2. Quantile/distributional head | Directly targets your diagnosed V-variance predictor; new signal, bounded support |
| **High-EV** | 3. Diverse small-net ensemble | Theory says diversity wins for small nets; supplies search diversity; reduces variance |
| **Speculative-cheap** | 4. Bidirectional + exact backward frontier/PDB | Near/far regime split; saturation-proof; exact tail |
| **Speculative-cheap** | 5. DPP / diverse beam selection | Fixes beam collapse search-side; composes with symmetry |
| **Speculative** | 6. GFlowNet amortized triage | Frees compute on the easy ~90% for the hard tail |
| **Long-shot** | 7. Group-theory exact labels / tail | Only genuinely-new label source; phased decomposition |

## Recommendations — first three experiments, in order

1. **Q-head all-neighbors value (Direction 1) first.** Highest-confidence, literature-and-frontier-proven; a throughput multiplier that makes *every subsequent experiment cheaper* and directly amplifies the inference-scaling lever you already trust. Benchmark: matched-wall-clock total score on the 1001 set. **Threshold to ramp onto TPU:** ≥3% total-move reduction at matched compute. *Immediately after this lands, re-run your existing inference-scaling sweeps* — the larger effective beam alone may close much of the 75k→70k gap before you touch anything else.
2. **Quantile head (Direction 2) second.** A near-free architectural delta on the (now Q-style) trunk and the sharpest test of your own central diagnosis (variance, not mean). Run it on top of the Q-head. **Threshold:** some τ beats mean-ranking by ≥1% *and* inter-quantile range correlates (ρ>0.3) with mid-depth beam errors; if the correlation fails, you've learned your variance hypothesis needs revisiting — itself valuable.
3. **Diverse small-net ensemble (Direction 3) third.** Orthogonal to 1–2 and compounding; it also lets you A/B the diversity economics directly against your symmetry ensemble. **Threshold:** 4×1.5M min-ensemble beats 1×6M at matched inference FLOPs by ≥2% *and* adds ≥1% on top of symmetry.

Order rationale: (1) changes the cost structure so (2) and (3) run wider for free; (2) is the cheapest high-information test of your own diagnosis; (3) is the most compute-hungry of the three and benefits most from (1)'s throughput gain.

## What I would NOT pursue, and why
- **Train 100–500 epochs and hope (your idea C).** Falsified by your own 184-vs-50 result and corroborated by overfitting evidence; training loss is not a reliable proxy for beam quality. Dead.
- **Bigger value models / staged 12–20M.** Your capacity is information-bound, not parameter-bound; the diverse ensemble (Direction 3) is the correct way to spend more parameters.
- **Full MuZero / Gumbel-MuZero MCTS as the search.** MCTS buys little over a massive beam for a deterministic, single-goal Cayley-graph shortest-path problem; Gumbel-MuZero's advantage is the *few-simulation* regime (it "learns reliably even with 2 simulations"), the opposite of your 1e5–1e8 beam. Skip the tree search; the Gumbel-top-k without-replacement sampling trick could at most be a minor diversity tweak inside Direction 5.
- **Full Stream-of-Search / decision-transformer sequence model over move strings.** Autoregressively generating 70–130-move solutions is expensive and has no evidence of beating a large value-guided beam on this puzzle family; it collides with your throughput needs.
- **Symmetry-consistency training loss.** You already found it pulls against the symmetry *ensemble* you rely on (constraint 6); don't reintroduce it.
- **Another loss term on the existing scalar target** (ranking, hard-negative, frontier-regret variants). Exhausted at 6M; only pursue new *targets* (quantile, Direction 2) or new *label sources* (group theory, Direction 7).

## Caveats
- The GFlowNet pathfinding paper (arXiv:2603.01786) is recent and its strongest results are on 2×2×2/3×3×3 Rubik and a synthetic Swap puzzle, *not* Megaminx; its 3×3×3 model is 25M params (a policy/flow net, not subject to your value-capacity regression), and there is a train_steps discrepancy between paper (1M) and repo (2M). Treat its small-beam dominance as the load-bearing claim and its large-beam parity (21.15 vs 21.24 at 2^18) as the reason not to expect it to beat your big beam.
- The "diversity helps small nets" result (Abe et al., arXiv:2302.00704) is established on image classification (~600 ensembles); transfer to heuristic ranking for beam search is plausible but unproven — hence Direction 3's explicit FLOP-matched falsifier.
- Q*/DeepCubeAQ's headline wins are on Rubik with engineered meta-action spaces; the *throughput* mechanism (single-pass child values) transfers cleanly to Megaminx, but the magnitude of the effective-beam gain depends on your current per-child evaluation overhead — measure it before committing TPU time.
- DPP selection (Direction 5) is demonstrated at NLP beam widths (tens–hundreds); whether the O(B²) kernel is affordable at B~1e5–1e8 is unverified — validate at small B first.
- A few numbers come from a single source each (the sub-80k Megaminx submission's beam 2^27 on A100 from the SpeedSolving forum; the GFlowNet small-beam solve rates from the arXiv paper). They mutually corroborate the "all-neighbors model + big beam" recipe but should be re-verified against the live competition leaderboard before large compute commitments.