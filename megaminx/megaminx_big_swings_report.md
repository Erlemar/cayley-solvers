# Megaminx Solver — Big Structural Swings (Expensive, High-Payoff, Novel)

*Re-run with the objective function flipped: rank by **magnitude of improvement × novelty**, with cost as a secondary constraint. You have TPU-pod access; I treat that as licence to propose searches and training loops that a single-GPU budget would forbid.*

---

## The reframing that drove this pass

Three facts, taken together, told me where to point:

1. **You are a CayleyPy author.** Diffusion-distance value model + RL + V-guided beam + symmetry ensemble *is your own published method* (arXiv:2502.18663 / 2502.13266). So the bar isn't "beat DeepCubeA" — it's "beat the thing you already wrote." Any idea that lands inside the CayleyPy design space is, by definition, not a swing.

2. **The (un)scalability theorem explains every wall on your list at once.** Pendurkar, Huang, Koenig & Sharon (*The (Un)Scalability of Heuristic Approximators for NP-Hard Search Problems*, arXiv:2209.03393) prove that approximating the **completely-informed heuristic to high precision is NP-hard**, and show empirically that the network size needed grows *exponentially* in instance size, with a characteristic "phase 2" where **test loss starts rising exactly at the point where parameters stagnate** — i.e. bigger nets overfit and regress. That is your constraint 1 (capacity dead), constraint 4 (precision-chasing losses neutral), and constraint 5 (longer training overfits) as *one theorem*. Their explicit conclusion: the field should "investigate **other ways of integrating heuristic search with machine learning**" — not keep refining a monolithic value regressor. Your own experiments rediscovered their result. Stop fighting it; route around it.

3. **The merged frontier (~72k) and your system share the same skeleton:** *unidirectional, atomic-move beam over one monolithic V(s).* The Santa-2023/2024 community (the direct ancestor of this competition) converged on exactly this — beam search + SA + graph-optimization post-processing (Kaggle Santa-2023 discussions; maddevs.io write-up). **Nobody in this lineage is doing learned bidirectional meet-in-the-middle, learned phased/coset solving, or hierarchical subgoal search.** That's the open ground.

So the swings below are all **structural reframes of the search**, where the model is demoted to a role it's actually good at (ranking *small* residuals, or ranking *within a small coset*), instead of being asked to do the one thing the theorem says it can't (rank accurately across the full mid-depth of a 10⁶⁸ graph). Every one of them attacks your diagnosed killer — **per-state V-variance at mid-depth (constraint 3)** — not by reducing the variance, but by *never querying V where it's high-variance*.

---

## TIER 1 — The two biggest swings

### Direction 1 — Learned bidirectional meet-in-the-middle, with V reused as a front-to-front heuristic via group composition
**Thesis:** Grow a beam from the scramble *and* a beam from solved; score meetings with your *existing* V evaluated on the residual permutation between frontier nodes. Each side only spans half the depth, and every decision-critical V query lands in the near-solved regime where V is calibrated and low-variance.

**The mechanism, concretely.** Your state is a permutation; moves are right-multiplication by generators; the generator set is symmetric (each face turn's inverse is also a move), so the Cayley graph is undirected and `d(x,y) = |x⁻¹y|` = word length of the residual. Critically, **word length of any element z is exactly distance-to-solved of z**, because solved = identity. Therefore:

> For a forward-frontier node `f` (reached from scramble `s`) and a backward-frontier node `b` (reached from solved `e`), the true remaining gap between them is `d(f,b) = |f⁻¹b| ≈ V(f⁻¹b)`.

`f⁻¹b` is one O(120) permutation compose. So **your current 6M model is already a front-to-front ("all-pair") meeting heuristic** — you just never call it that way. The pipeline:

1. Forward beam from `s`, backward beam from `e`, each to depth ≈ D/2.
2. Find candidate near-meeting pairs `(f,b)` with small `V(f⁻¹b)`. Avoid the |F|×|B| blowup with an **approximate-nearest-neighbour index** (FAISS) over a learned embedding φ where `‖φ(x)−φ(y)‖ ≈ d(x,y)` — distill φ from V's penultimate layer, or train it directly (the goal-conditioned state-embedding recipe of Venkattaramanujam et al., arXiv:2205.01965, learns exactly such a metric embedding so that L2 ≈ graph distance).
3. For the top candidate pairs, verify with `V(f⁻¹b)`; for the small-residual winners, **close the gap exactly** with a narrow BFS / your existing beam on the residual `f⁻¹b` (which is now a near-solved instance — easy, exact).
4. Concatenate `w_f · (closing word) · reverse(w_b)`; run your mature post-processing.

**Why it beats your specific walls:**
- **Mid-depth V-variance (constraint 3) — the headline.** A unidirectional beam to 100 moves must push through ~100 layers of compounding mid-depth ranking error, exactly where V is high-variance. Bidirectional pushes ~50 each side and *meets in the middle*; the only V calls that decide the solution are on **small residuals** `f⁻¹b`, i.e. in V's well-calibrated, low-variance near-solved band. You're not reducing the variance — you're *refusing to depend on the region where it lives.*
- **Saturation (constraint 2) — sidestepped entirely.** V is never queried on far states. Its flat tail past the diameter is irrelevant; you only ever ask "how close are these two *already-close* nodes."
- **6M ceiling (constraint 1) — untouched.** Reuses the existing model. No capacity ask.
- **Composes with your #1 lever.** Symmetry ensemble, multi-seed, and bigger beams all still apply, per-side.

**Minimal experiment / falsifier.** Implement plain bidirectional with exact hash-collision meeting first (sanity; collisions will be ~0 at depth in a 10⁶⁸ space — that's the point, it motivates step 2). Then add embedding-ANN near-meeting + residual closing. Test on your hardest 100 instances at a fixed *total* node budget vs your unidirectional beam. **Falsified if:** at matched budget, the bidirectional total move count is not lower — e.g. because near-meeting pairs with closeable residuals never appear before the forward beam would have solved anyway (would mean the frontiers don't "aim" at each other without an explicit attractor; fix attempt: bias each side's beam by `V(f⁻¹·(nearest opposite-frontier node))`, a learned front-to-front *guidance* term, before declaring defeat).

**Compute.** 2× beam memory + an ANN index over each frontier per step. Heavy engineering (frontier bookkeeping, residual compose, FAISS on-GPU), but squarely inside your TPU regime; the model cost is *zero new training* in the basic version. This is the cheapest-to-try of the Tier-1 ideas because it reuses everything.

**Key citations.**
- Sturtevant & Chen 2016 (bidirectional *brute-force* found **optimal** Rubik solutions where unidirectional search couldn't) — proof that bidirectionality is the lever, here done *unlearned*; you'd be adding the learned meeting score.
- Holte et al. MM / Eckerle et al. 2017 "Sufficient Conditions for Node Expansion in Bidirectional Heuristic Search" (cs.du.edu) / BAE* / MEET (IJCAI-25, ijcai.org/proceedings/2025/0999.pdf) — front-to-end vs **front-to-front** (= all-pair) heuristic theory; confirms `V(f⁻¹b)` is the principled meeting estimator.
- DESP (goal-conditioned cost network for bidirectional retrosynthesis) — a worked example of learned bidirectional with a front-to-front net in a huge implicit graph.
- Venkattaramanujam et al., arXiv:2205.01965 — learn a state embedding whose L2 ≈ graph distance; the enabling trick for ANN-based near-meeting.

**Novelty vs your already-tried list.** Your post-processing does *small-radius BFS window replacement* (local, on a fixed path); this makes the **entire search bidirectional** and uses V as a *front-to-front* scorer via group composition — a use of V that is absent from DeepCubeA, DeepCubeAQ, and both CayleyPy papers (all unidirectional). The group-composition reuse means it costs no new model.

**Confidence.** Medium-high that it helps the hard tail; the *magnitude* is the uncertainty. Bidirectionality is **demonstrated** (classically) to beat unidirectional on the cube; the **learned near-meeting + residual-closing** form is **speculative** for Megaminx.

---

### Direction 2 — Learned multi-phase coset solver (the Kociemba/Thistlethwaite that "doesn't exist" for Megaminx), with a small heuristic per phase
**Thesis:** No two-phase Megaminx solver exists *not because the structure is missing* — Megaminx has the same orientation invariants as the cube — but because the pruning tables are infeasible at 10⁶⁸. Replace each infeasible pruning table with a *small learned heuristic on that phase's much smaller coset space.* This is the literature's prescribed escape from the (un)scalability trap: split one impossible-to-approximate h\* into several easy ones.

**The structural opening.** Dan Hoey's classical analysis (Cube-Lovers archive, RWTH Aachen) nails it: standard Megaminx has **even edge- and corner-permutation parity, corner orientation 0 (mod 3), edge orientation 0 (mod 2)** — "these are the only invariants," giving `(20!·3²⁰·30!·2³⁰)/24 ≈ 1.007×10⁶⁸` states. **Those are exactly the coordinates Kociemba's Phase 1 drives to zero on the cube** (CubeBench arXiv:2512.23328 gives the clean G₀→H→{e} statement). So a Megaminx subgroup chain is sitting right there:
- **Phase 1:** reduce to `H` = {all edges oriented, all corners oriented} — a restricted-move-set target, EO(mod 2) + CO(mod 3) = 0.
- **Phase 2…k:** solve permutation within progressively smaller cosets, each with a restricted generator set.

Each phase gets **its own small value model + beam**, trained only on its own coset (orders of magnitude smaller than the full group). Kociemba gets near-optimal cube solutions (~20 moves) by running Phase 1 to many suboptimal lengths and picking the `(phase-1 + phase-2)` combination that minimizes the *total* — you'd do the same: enumerate alternative Phase-1 maneuvers, solve each remainder, keep the shortest total.

**Why it beats your specific walls:**
- **6M ceiling / information bottleneck (constraint 1) — the core fix.** The (un)scalability theorem says a *monolithic* h\* needs exponential capacity. A *per-phase* h\* approximates distance within a vastly smaller coset → the information each net must carry collapses. You're not adding parameters; you're **partitioning the signal** so 6M is no longer the binding constraint. This is the one idea that directly contests your "capacity is saturated" assumption by changing what the capacity is spent on.
- **Saturation (constraint 2) — by construction.** Each phase's distance has a *small bounded range*; no net is ever asked to grow past a diameter it can't see.
- **Variance (constraint 3) — attacked where it counts.** The Phase-1 coordinates (EO/CO) are *exact, cheap, low-variance features*; a heuristic over them behaves like Kociemba's exact pruning table, not like a noisy mid-depth regressor.
- **New label source (constraint 4).** The phase decomposition supplies grounded, model-independent sub-targets — exactly the "label source genuinely new" your constraint demands before any loss change is worth trying.

**The honest risk.** Phased solvers trade optimality for tractability: Thistlethwaite is ~45 cube moves vs God's 20. On *easy/medium* instances a phased solve could be **longer** than your unidirectional beam, which would hurt the SUM objective. Three mitigations: (i) use phasing **only on the hard tail** where the beam is already struggling, as a portfolio member; (ii) **over-generate Phase-1** maneuvers and minimize total (the Kociemba trick) to claw optimality back; (iii) let each phase be solved by *your existing beam+V machinery on the restricted move set* — so phasing is a wrapper, not a rewrite.

**Minimal experiment / falsifier.** Don't build the whole chain. Build **only Phase 1 = edge-orientation** (the cheapest coordinate): a small learned heuristic + restricted-move beam that drives EO→0, then hand the EO-solved state to your *current* solver for the rest. On a sample, measure `len(EO maneuver) + len(current solver on remainder)` vs `len(current solver on original)`. **Falsified if** the EO-reduction maneuvers are long *and* the post-EO remainder isn't enough shorter to win — i.e. orientation-first doesn't carve the problem the way it does on the cube. (Watch specifically for whether reaching the oriented subgroup is nearly as hard as solving outright; if so, the decomposition gives no leverage.)

**Compute.** Several *small* models (each cheaper than your current 6M) + restricted-move beams; CPU-side group math (Schreier–Sims SGS to define cosets cleanly). Medium-heavy engineering, modest per-model training. The most *research*-expensive idea here, not the most *FLOP*-expensive.

**Key citations.**
- Kociemba two-phase (kociemba.org/math/twophase.htm) & Thistlethwaite subgroup chain (kociemba.org/math/imptwophase.htm) — the proven mechanism on the cube, incl. the "minimize phase1+phase2 total" near-optimality trick.
- Dan Hoey, Cube-Lovers archive (RWTH Aachen) — Megaminx invariants (EO mod 2, CO mod 3, even parities, 1.007×10⁶⁸) → the decomposition exists.
- Pendurkar et al., arXiv:2209.03393 — the theoretical reason decomposition should beat a monolithic net.
- Chen et al., *Learning Admissible Heuristics for A\*: Theory and Practice*, arXiv:2509.22626 — neural heuristics over **PDB abstractions** for Rubik with generalization bounds and a Cross-Entropy-Admissibility loss; the modern "learned heuristic on an abstract/coset space" template.

**Novelty.** A *learned* two-/multi-phase Megaminx solver does not exist (no Kociemba for Megaminx; CayleyPy is single-phase diffusion-distance). This is the highest-ceiling, highest-novelty item — it's literally building the missing solver.

**Confidence.** Speculative, high-ceiling. Mechanism **demonstrated** on the cube (Kociemba/Thistlethwaite are deployed reality); the learned-per-phase Megaminx instantiation is **unproven**.

---

## TIER 2 — Strong structural ideas, less engineering risk

### Direction 3 — Hierarchical subgoal search (kSubS / AdaSubS), chosen *specifically* because it's robust to the value-noise that is killing you
**Thesis:** Search over learned *subgoals* (states k moves ahead) instead of atomic moves. The reason this is the right tool for *your* problem in particular: hierarchical search is provably far more tolerant of value-function noise than best-first/beam — and value-function noise at mid-depth is your diagnosed bottleneck.

**The targeted evidence.** Kujanpää et al. / the *What Matters in Hierarchical Search for Combinatorial Reasoning Problems?* study (arXiv:2406.03361) ran a Rubik's-cube experiment injecting noise into value estimates: low-level best-first (≡ your beam) degrades steadily and collapses toward random; **subgoal methods with horizon >1 retain ~40% success even with *completely random* values**, and AdaSubS is "nearly not affected." kSubS (arXiv:2108.11204) and AdaSubS (arXiv:2206.00702) are the algorithms; both are demonstrated on Rubik's and Sokoban. If your beam is losing short paths because mid-depth V-variance knocks the optimal child out of the top-B, a search that barely depends on per-node value precision is the mechanism-matched fix.

**Mechanism.** Train a subgoal generator `g(s)` producing a few diverse states ~k moves toward solved (repurpose your trunk to emit a k-ahead state, or a small autoregressive/conditional model trained on your existing solution corpus). Run best-first over the high-level subgoal graph; realize each high-level edge with a short conditional policy / BFS (you already have a shortlister and post-processing BFS — most pieces exist). AdaSubS adds multiple generators at different k and a reachability verifier to prune dead subgoals fast.

**Why it beats your walls.** Constraint 3 head-on (noise robustness). Reduces effective depth on deep scrambles. Doesn't enlarge the value net (constraint 1). The generator's diversity *adds* to your symmetry-ensemble diversity rather than fighting it (unlike symmetry-consistency, constraint 6).

**Honest caveat I won't paper over.** You already solve all 1001; the win must be *shorter* solutions, and subgoal search optimizes *solvability/robustness*, not length. The bet is that variance-induced misses are currently costing you length, and a robust search recovers it. Medium confidence on move-count, higher on hard-tail reliability.

**Minimal experiment / falsifier.** Train a k∈{3,4} generator from your solution corpus; run AdaSubS on a sample with k-fill via your existing machinery; compare post-processed move count to your beam. **Falsified if** subgoal solutions aren't shorter after compression, or the generator can't produce reachable subgoals from deep states (subgoal "achievability" collapses far from solved).

**Compute.** One generator training (medium) + a verifier; inference is cheaper per solve than a giant beam. Fits your budget comfortably.

**Citations.** kSubS (2108.11204); AdaSubS (2206.00702); *What Matters…* (2406.03361 — the noise-robustness result, the load-bearing reason to do this); HIPS / VQ-subgoals (2301.12962) for the generative-subgoal variant.

**Novelty.** A different *search paradigm* from the entire DeepCubeA/CayleyPy atomic-beam line. Not on your tried list.

**Confidence.** Medium. **Demonstrated** on Rubik's (robustness + solving); **speculative** specifically as a move-count reducer for your SUM objective.

---

### Direction 4 — Expert-iteration with an *expensive external corrector* as the label oracle (the relabeling you tried, but with a fundamentally stronger teacher)
**Thesis:** Your "frontier-regret relabeling" came out neutral because the corrector was *the model's own beam* — it can only teach V what V already prefers. Replace the corrector with a 10–100× more expensive search (huge bidirectional beam / deep IDA\*), relabel mid-depth states with its genuinely-shorter distances-to-go, retrain, iterate. The label source is now grounded *outside* the model — the exact condition your constraint 4 says is required.

**Mechanism (ExIt / Bootstrap-Learning-Heuristic loop).** Sample states along your current solutions, concentrated at the mid-depth band where variance bites. For each, run the strongest search you can afford to get an improved cost-to-go `ĥ(s)`. Train V toward `ĥ` (regression or rank). Re-solve, re-sample, repeat. This is Arfaee–Zilles–Holte BLH (relabel states on A\*-found paths with better estimates, supervised-learn, repeat) and Anthony et al.'s ExIt (decompose into expensive "expert" planning + cheap "apprentice" generalization), applied at the *mid-depth band* you've identified as the failure region.

**Why it might beat the wall your other experiments couldn't.** Your constraint-4 diagnosis ("disagreements are equally-good alternatives, not fixable mistakes") was measured against a *weak* corrector. A much stronger search may reveal that at mid-depth the disagreements are *not* all ties — that there are systematically shorter continuations V is missing. If so, distilling them lowers mid-depth variance at the *source*.

**Why it's also the most informative experiment you can run — even if it fails.** The probe "how often does a 100× search beat V's label at mid-depth?" is a **direct test of the saturation hypothesis itself.** If the answer is "rarely," you've *confirmed* the value model is genuinely signal-saturated (constraints 1–5 are bedrock; commit fully to search-side redesign, Directions 1–3). If "often," then V is **under-labeled, not saturated** — which would *reopen* the capacity question (constraint 1 might be a labeling artifact, not a capacity limit) and make ExIt and possibly larger models live again. Either outcome redirects your whole strategy. That's why it should run first (see sequencing).

**Minimal experiment / falsifier.** Take 10⁵ mid-depth (~20-move-out) states; run a much-larger-than-training search for each; histogram `ĥ_strong(s) − V(s)`. **Falsified (and strategically decisive) if** the mass is ~0 — V is saturated, don't bother retraining. If there's real negative mass, do one ExIt round and test beam quality.

**Compute.** The probe is cheap-ish (10⁵ strong searches). Full ExIt label-generation is the most FLOP-expensive idea here (your TPU pods earn their keep). Retraining is standard.

**Citations.** Arfaee, Zilles & Holte 2010 (Bootstrap Learning of Heuristic Functions — the canonical "relabel from search, supervised-learn, repeat"); Anthony, Tian & Barber 2017 (Expert Iteration); DeepCubeA AVI (the weaker self-bootstrapping baseline you're improving on); Pendurkar et al. 2209.03393 (why label *quality*, not capacity, is the lever).

**Novelty.** Distinguished from your tried relabeling solely by **corrector strength and target band** — but that distinction is the whole point, and it's the difference between "teaching V its own opinions" and "distilling a stronger search into V."

**Confidence.** Medium-low on net score, **high on information value** (the falsifier resolves your central uncertainty).

---

## TIER 3 — Worth a slot, lower priority

### Direction 5 — Macro-operators mined from your own solution corpus, as first-class search operators to escape endgame value plateaus
**Thesis.** The "last few pieces" of a permutation puzzle is where atomic greedy/beam value flattens and your post-processing earns its keep; commutators/conjugates are the human answer. Mine high-frequency short move-blocks from your 1001 solutions + search traces (MACRO-FF "SOL"-style extraction, Botea et al. 2005), add the best ones (with inverses) as macro-actions, and run the beam in the **mixed atomic+macro** action space with V trained to score macro children. Macros let the search jump across a V-plateau to a basin the atomic beam can't reach in one greedy step. Since the SUM counts atomic moves, only *move-efficient* macros (clean 3-cycles etc.) qualify — which is exactly what commutators are.
**Walls.** Doesn't touch capacity/saturation; targets the specific endgame-plateau failure. Composes with everything above (especially Direction 2's late phases).
**Experiment / falsifier.** Add top-N mined macros to the action set on the *final 15 moves* of hard solves; measure tail length. Falsified if macros never shorten the tail vs your SA-with-macro-insertion post-processing (which may already capture most of this).
**Citations.** MACRO-FF (Botea, Müller, Schaeffer 2005); options (Sutton, Precup, Singh 1999); MAGIC online macro learning (arXiv:2011.03813).
**Confidence.** Medium-low; **partially demonstrated** (macros help long-horizon planning broadly), overlaps your existing macro-insertion post-processing — hence Tier 3.

### Direction 6 — Icosahedral-equivariant trunk at *fixed* 6M, for signal efficiency not capacity *(flagged, low prior)*
You've rejected 7 encoder families on saturation grounds, so the prior is low — I include it only with the one distinction that might matter: prior rejections were about value *magnitude* growing past the diameter. An equivariant net with a **bounded/ranking output by construction** (so saturation is architectural, not learned) might extract more signal per parameter from the dodecahedral symmetry without the failure mode you saw. **But** your symmetry *ensemble* monetizes the very symmetry an equivariant trunk would fold away (constraint 6 again) — so this likely *trades away your best lever*. I would not run this before Directions 1–4. Listed for completeness; **do not prioritize.**

---

## Tier table

| Tier | Direction | One-line why it's here | Cost | Novelty |
|---|---|---|---|---|
| **Big swing** | 1. Learned bidirectional meet-in-the-middle (V as front-to-front via `f⁻¹b`) | Halves depth; all decisive V calls land in the low-variance near-solved band; reuses your model | Med (engineering) | High |
| **Big swing** | 2. Learned multi-phase coset solver (Megaminx Kociemba) | Splits the un-approximable monolithic h\* into easy per-coset h\*; builds the solver that doesn't exist | Med-high (research) | Highest |
| **Strong** | 3. Hierarchical subgoal search (kSubS/AdaSubS) | Provably robust to the value noise that's killing your beam | Med | High |
| **Strong** | 4. ExIt with expensive external corrector | New label source grounded outside V; *and* a decisive test of the saturation hypothesis | High (FLOPs) | Med (vs your relabeling: stronger teacher) |
| **Worthwhile** | 5. Mined macro-operators | Escapes endgame value plateaus | Low | Med-low |
| **Do-not-prioritize** | 6. Equivariant trunk | Trades away your symmetry-ensemble lever | Med | Low-med |

---

## Recommended sequence of the first three experiments

**1. Direction 4's diagnostic probe first (cheap, decisive).** Before investing in any expensive build, answer the one question that determines strategy: *is V actually saturated at mid-depth, or just under-labeled?* Histogram `ĥ_strong(s) − V(s)` on 10⁵ mid-depth states.
- If ~0 → saturation confirmed → pour everything into **search-side redesign (Directions 1→2→3)** and never touch the value net's capacity again.
- If materially negative → V is under-labeled → run an ExIt round, and reconsider whether your capacity ceiling (constraint 1) was a labeling artifact.
Either way you've spent little and bought a decision.

**2. Direction 1 (bidirectional meet-in-the-middle) second.** Highest expected value *and* lowest activation energy of the Tier-1 pair: it reuses your existing model (zero new training in the basic form), and it's the most direct structural answer to your two diagnosed walls (mid-depth variance + deep-scramble depth). Start with exact-collision bidirectional (to feel the 10⁶⁸ collision-sparsity yourself), then add embedding-ANN near-meeting + residual closing.

**3. Direction 2 (phased coset solver, EO-only feasibility slice) third.** Biggest ceiling, most engineering — so it goes last of the three, and you de-risk it with the cheapest possible slice (edge-orientation Phase 1 + your current solver for the remainder) before committing to a full subgroup chain.

**Why this order.** Experiment 1 is a *diagnostic* that re-prioritizes everything after it — running it first is pure information value. Experiment 2 is the best EV-per-engineering-hour and reuses infra, so it banks a likely win early. Experiment 3 is the moonshot; you only fund its full build after 1 tells you the value model really is the bottleneck (justifying decomposition) and after 2 has already given you a bidirectional substrate that each *phase* can itself exploit.

---

## What I would NOT pursue, and why

- **Diffusion combinatorial-optimization solvers (DIFUSCO, arXiv:2302.08224; T2T).** Structural mismatch. They generate a static `{0,1}` solution object (an edge-selection heatmap) for problems where *the instance graph is the input* (TSP, MIS). Megaminx is pathfinding on a *fixed* 10⁶⁸ Cayley graph; the "solution" is a variable-length move *sequence*, not a subset of a given graph. The diffusion-CO machinery has nowhere to attach. Skip.
- **Autoregressive sequence models over move-strings (decision transformer / stream-of-search).** Generating valid 70–130-move solutions token-by-token in a 10⁶⁸ space is expensive and has no evidence of beating a value-guided beam on this family; collides with the throughput you actually need. (You already flagged this; I concur.)
- **MuZero / learned-model MCTS as the main search.** The dynamics are *known and exact* — learning a model is pure overhead, and MCTS buys little over a massive beam for deterministic, single-goal shortest-path. The one transferable scrap (Gumbel top-k without-replacement sampling) is at most a minor diversity tweak inside Direction 3, not a paradigm.
- **Train 100–500 epochs (your idea C).** Dead. Falsified by your own 184-vs-50 result, by the overfitting "phase 2" in Pendurkar et al. (test loss rises as params stagnate), and by the precision-chasing trap the theorem describes. This is the single clearest "no" on the list.
- **Bigger value model (12M/20M).** Dead by the same theorem — *unless* Experiment 1 surprises you by showing under-labeling, in which case capacity becomes a live question again. Conditional, not categorical.

---

## On your three weighed ideas, with the expensive lens

- **(A) Unify distance + search into one model.** The *spirit* is right — make the **search structure** do the work the model can't — but the win is **structural search redesign (Directions 1 & 3)**, not a differentiable/amortized monolith. A differentiable beam is fiddly and has weak evidence here; a *bidirectional* search that demotes V to scoring small residuals, or a *hierarchical* search robust to V-noise, is the defensible version of "unifying."
- **(B) RL beyond value iteration.** Worth the cost **only** in the form where it changes the *label source*: ExIt with a strong external corrector (Direction 4). Generic "more RL" (PPO/Q-learning on the same self-generated signal) won't escape the saturation you already hit — it's the same teacher wearing a different hat. (GFlowNet-for-diversity, from the prior pass, remains the other legitimate "RL beyond VI" slice.)
- **(C) Train 100–500 epochs and hope.** No. See above.

---

## Caveats and uncertainty

- **The single biggest unknown is Experiment 1's result** — it's load-bearing for the whole ranking. If V is under-labeled rather than saturated, Directions 2 and 4 jump in value and even "bigger model" revives. Run it first precisely because so much hinges on it.
- **Bidirectional (Direction 1) magnitude is unproven for this puzzle.** The mechanism is classically sound and reuses your model, but whether the frontiers find closeable residuals *before* the forward beam would have solved anyway is empirical. The "front-to-front guidance term" is the first fallback if naive meeting under-performs.
- **Phasing (Direction 2) can lengthen easy solves.** Treat it as a hard-tail portfolio member with Kociemba-style total-length minimization, not a global replacement, until the EO-slice proves it carves the problem.
- **Subgoal search (Direction 3) optimizes robustness, not length directly** — its move-count payoff depends on your variance-induced misses being real and recoverable.
- **Megaminx-specific group engineering** (clean coset definitions, Schreier–Sims SGS, restricted-move generators) is non-trivial and is where Direction 2's research cost actually lives — more than the model training.
