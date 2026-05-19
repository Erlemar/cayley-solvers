# A Deep Survey of Training-Data Generation and Target-Refinement Pipelines for Neural Heuristic + Beam-Search Solvers, with Application to Megaminx

**Audience:** Senior ML engineer / Kaggle Grandmaster competing in the **Cayley-Py Megaminx** competition with a ResMLP distance-predictor + Bellman refinement + beam search pipeline (current best 95,682). The goal of this report is to map the entire design space, ground each option in primary literature, and then prioritize ideas by ROI on a 2–3 day RTX-4090 budget.

---

## 1. Executive Summary — top-ROI ideas to try first

The state of the art on this exact problem (Megaminx ≈ 10⁶⁸ states, generator-set Cayley graph, ResMLP + beam search) is essentially defined by two papers from the **CayleyPy project**: Chervov et al., *A Machine Learning Approach That Beats Large Rubik's Cubes* (arXiv:2502.13266, NeurIPS 2025) and Chervov et al., *CayleyPy RL: Pathfinding and RL on Cayley Graphs* (arXiv:2502.18663). Most of the heavy lifting they report on cubes 4×4×4 and 5×5×5 was achieved **without Bellman refinement** — by scaling random walks, beam, and **multi-agent ensembling** — which is the single most important empirical fact for prioritizing your effort.

Tier-1 things to try first (ranked by expected score reduction per GPU-day):

1. **Multi-agent ensembling.** Re-train 8–30 ResMLP agents from different RNG seeds and submit per-puzzle the shortest of any agent's solution. This is the single technique that took CayleyPy from "single-agent good" to "beats every team in Santa 2023" on 4×4×4 and 5×5×5. The 5×5×5 result needed 69 agents; the 4×4×4 result needed 29. They report ensembling 10 agents reduced 5×5×5 average length from 92.9→92.16 and 4×4×4 from 49→46.51, with each agent being individually mediocre. *Expected effect on Megaminx: 3–8% score reduction, scales nearly log-linearly with agent count.*
2. **Push n_back further and use "generalized non-backtracking."** The CayleyPy-RL paper explicitly states they use *"improved nonbacktracking based on the following ideas: 1) we forbid not only 1 step in history but typically 32 steps, 2) we consider a bunch of trajectories and ban states for all of them at once, 3) we ban not only visited states but their neighbors."* They report this lets a single-epoch one-minute-trained MLP solve S₂₀ LRX optimally where ordinary RW fails entirely. The Jan-2025 chat about n_back=40 you saw comes directly from this lineage. **Try n_back ∈ {16, 32, 48, 64} and a per-trajectory "ban neighbors" mask.** Expected effect: large for short epochs, modest at convergence — but it cleans up label noise. *Expected: 1–3%.*
3. **Aggressively increase beam width on hard puzzles.** Both CayleyPy and EfficientCube observe solution length decreases approximately *linearly in log₂(beam width)* up to W ≈ 2²⁴. On a 4090 with 24 GB you can probably push to 2¹⁹–2²⁰ on Megaminx (state size 120). Allocate one full GPU-night to a W=2²⁰ run on the residual hardest puzzles in your current submission. *Expected: 2–5%.*
4. **Mix in Bellman/exact targets on the BFS-d6 shell with explicit sample weighting.** Your current "Option B" is correct but probably under-weighted. CayleyPy-RL Table 1 shows that DQN-style refinement *on top* of warmup-by-RW improves Spearman correlation with true distance from 0.92 → 0.98 at S₁₀, and increases beam-search success rate at S₂₈. Concretely: in each minibatch, draw ~10–20% from the ~19.4 M BFS-d6 states with hard MSE labels and the rest from RW + clipped Bellman. The exact targets anchor the network at small distances where errors are most damaging during the final beam-collapse. *Expected: 2–4%.*
5. **Solver-trace data ("Option C": iterative supervised learning).** Run beam search on training puzzles with the current model, record (state along solution, distance-to-solved-along-found-path) pairs, and use them as new labels capped at the found distance. This is the same idea as **Bootstrap Learning of Heuristic Functions** (Arfaee–Zilles–Holte 2010) and is essentially what DeepCubeA does post-hoc. It directly uses the solver's *upper bound on true distance* — strictly tighter than the random-walk bound. *Expected: 2–6% if combined with Bellman.*

After these, the next tier of ideas with real-but-smaller effect: distributional/quantile heads, n-step Bellman, Polyak averaging of the target, BFS-d7 if memory permits, symmetry deduplication exploiting Megaminx's order-60 icosahedral rotation group. The "radically different" paradigms (LLM-as-solver, MCTS-AlphaZero, IDA* with admissible heads) are mostly ROI-negative on this specific competition within 2–3 days.

The remainder of the report systematically justifies these conclusions and surveys options you could explore on larger budgets.

---

## 2. Theoretical Foundations: why random walks work, why they don't fully

### 2.1 Random-walk-distance vs. true word-metric distance

Place Megaminx in its proper mathematical setting. The "12 face × 2 directions" generators define a **Cayley graph** of a finite group of order ≈ 10⁶⁸, with degree 24. The "true distance" you want the network to predict is the **word metric** d(g) = length of the shortest word in the generators expressing g. The label produced by your random walk is the *length of a particular word* (the random one) — which is by construction an **upper bound** on d(g). It is rarely tight: the random walk visits the state space in a way that has very little to do with distance, except at very small k.

**Why the bound is loose.** Consider a non-backtracking simple random walk of length k starting from the identity on a regular Cayley graph. The number of words of length ≤ k grows roughly as (2d−1)ᵏ where d is half the degree (here d≈12, branching ≈23 with non-backtracking). The number of distinct group elements at distance exactly k is the *spherical growth function* S(k). For Megaminx this growth is approximately 23ᵏ until the diameter is approached, after which S(k) plummets and most random words at length k have *true distance* < k. Concretely, Hessen's HPC-Lichtenberg group-theoretic Megaminx solver (HKHLR project) gives an upper bound of **133** for Megaminx's God's number; the lower bound is much smaller (~45–55) and the diameter is *conjectured* to be in the 50s in QTM. So a random walk of length k=80 (your default k_max) typically lands at **true** distance well below 80. The label is biased upward; the bias grows with k.

A clean way to see this: as k → ∞, the random walk's distribution converges to uniform (the **stationary distribution**) at rate governed by the *spectral gap* λ₂ of the random-walk operator. Once mixed, the *true* distance to identity concentrates near the **typical distance** in the group — for Rubik's 3×3×3 this is 17–18 in QTM (Qu 2024, *Rubik's Cube Scrambling Requires at Least 26 Random Moves*); for Megaminx this is conjectured ≈40–50 in face-turn metric, far below k_max=80. Yet you label every state at the end of an 80-step walk as having distance 80.

So your label is the **diffusion distance**, which CayleyPy explicitly names. Diffusion distance is **monotone-correlated** with word-metric distance for small k but is **not order-preserving** at moderate-to-large k (CayleyPy-RL §2.4: *"diffusion distance is not monotonically dependent on the true distance — one node can be further than another for the true distance, while oppositely ranged for the diffusion distance"*). This is why CayleyPy-RL observes empirically that *enlarging the random walk training set does not always improve the model* — you are perfectly fitting a noisy upper bound that begins to disagree with the true distance.

### 2.2 Mixing time of random walks on Cayley graphs of permutation groups

Foundational: **Diaconis & Shahshahani (1981, Z. Wahrsch. 57:159–179)** showed the random-transposition walk on Sₙ exhibits a *cutoff* at t = (n log n)/2. Since then a large body of work (Berestycki-Şengül 2014; Lulov-Pak; Hermon-Kozma 2020 on Cayley graphs; Diaconis 1988 monograph *Group Representations in Probability and Statistics*) gives mixing-time estimates for many generator sets on Sₙ. For the *Babai* conjecture (Babai 1992), the diameter of any Cayley graph of a non-abelian simple group is bounded by O(log^c |G|); for the *Diaconis* conjecture mixing is O(n³ log n). Megaminx's group is a non-trivial extension involving A₂₀ × A₃₀ × (Z₃)¹⁹ × (Z₂)²⁷ (Xu's group-theoretic analysis, *Solving Megaminx Puzzle With Group Theory*, oss.linstitute.net) — non-abelian and very nearly simple in its alternating factors. So the walk has spectral gap bounded away from 0, mixing time *plausibly* in the hundreds of steps; but well before mixing, the walk has *already lost almost all distance information* relative to the word metric.

Practical takeaway: pick k_max that **just covers the conjectured diameter** (k_max ≈ 130 for Megaminx is safe; 80 is on the short side). But also bias k toward smaller depths since labels are tighter there.

### 2.3 Non-backtracking random walks mix faster (Alon-Benjamini-Lubetzky-Sodin, 2007)

The classical result *"Non-backtracking random walks mix faster"* (Alon, Benjamini, Lubetzky, Sodin, *Combinatorica* 27 2007) proves that on a regular expander the non-backtracking-walk mixing rate is up to 2× the simple random walk mixing rate — the closer to Ramanujan, the larger the speedup. Equivalently, the typical NBRW path of length k visits Θ(k/log log k) distinct vertices vs. log-factor fewer for SRW. **For us, this matters in two ways:**

(a) NBRW labels (k = number of NBRW steps) are *closer to true distance* than SRW labels because there is less self-revisit. CayleyPy-RL §3.3 gives a beautifully clean tree intuition: *"Imagine our graph is a tree, then even for non-backtracking random walk number of steps coincide with the true distance."* In a tree, NBRW = BFS exactly. In a Cayley graph of a free or nearly-free group, NBRW labels are nearly tight upper bounds. For Megaminx (which is locally tree-like at small scales), going from n_back=1 to n_back=32–40 (i.e., banning all states visited in the last 32–40 steps and their neighbors) further closes the gap.

(b) NBRW is essentially "cheap loop-erased": you remove most short cycles, get less label variance, train models faster.

This is the core theoretical reason your team's Jan-2025 finding (n_back=40 helps) works.

### 2.4 The coverage problem (which states the RW under-samples)

Random walk from identity with k_max=80 produces a sample density approximately proportional to the *return probability* Σ_k≤K P^k(e,g), which is heavily skewed:
- **States very close to identity (d ≤ 5):** **massively oversampled** (many short words land there). This is exactly why the BFS-d6 mixin is helpful — it gives clean, optimal labels in the over-sampled region.
- **States at typical/medium distance (d = 25–60 for Megaminx):** approximately uniformly sampled (the bulk of the spherical-shell volume).
- **States near the diameter (d > 90):** **systematically under-sampled.** A random walk of length 80 essentially never lands on a true-distance-100 state. But these are exactly the states the solver's beam search must traverse to crack hard puzzles.

This is the **rare-event** problem familiar from molecular dynamics and spin-glass simulation, and motivates everything in §3 below: tempered MCMC, biased walks, hand-crafted "deep" scrambles.

### 2.5 What is the model actually learning?

Reading the CayleyPy authors directly: *"the neural network's predictions for a given node v estimate the diffusion distance from v to the selected destination node."* The network is a learned smoother of the diffusion distance, not the word metric. Empirically this is enough — beam search compensates for the residual error — *because* the diffusion distance is a monotone scalarization that gives a good gradient pointing toward the goal in the bulk of the state space. The Bellman refinement step partially "lifts" the model from diffusion-distance-predictor toward word-metric-predictor, but does so imperfectly because it relies on the same stale targets.

### 2.6 Why RW alone doesn't suffice for far states

Two reasons compound: (i) labels at large k are loose upper bounds; (ii) the loss is dominated by the bulk, where most samples lie. The model under-fits the tail (states needing 60+ moves). Beam search with width 2¹⁶ is wide enough to forgive moderate errors in the bulk — but in the diameter neighborhood the heuristic must be *precise* about the few correct successors, and it isn't.

This sets up the entire motivation for §3 (better walks), §4 (alternate label sources for far states), and §5–6 (Bellman/curriculum to back-propagate distance information from the goal outward).

---

## 3. Variations on Random Walks

| Variant | How | Expected effect | Difficulty |
|---|---|---|---|
| **n_back tuning (1 → 16/32/40/64)** | Deque of last *n_back* states (or moves), forbid generators that revisit | Strong on label tightness; CayleyPy reports "single epoch in a minute" with n_back=32 vs. unable-to-converge with n_back=1 on hard cases | Trivial |
| **"Generalized non-backtracking" (CayleyPy)** | Ban neighbors of recently-visited states, ban across multiple trajectories | The published recipe; modest extra cost in walk gen | Easy |
| **Self-avoiding walks (SAW)** | Forbid all visited states; truly never revisit | Best label quality; expensive (hash table per walk); diverges from stationary distribution | Medium |
| **Loop-erased random walk (LERW)** | Run walk, erase any loop when one occurs | Equivalent label quality to SAW asymptotically; cheaper to implement | Medium |
| **Biased / model-guided walks** | At each step, pick a generator weighted by ε-greedy on current heuristic — i.e., ε-explore, (1−ε)-go-toward-far | Targets samples to under-explored "far" region, but introduces selection bias (label is no longer correct upper bound under the original measure) | Medium; risky |
| **Hamming-weight-targeted / depth-stratified sampling** | Reject walks whose final state has Hamming distance outside a target band; or sample k from a specified non-uniform prior over [1, k_max] | Directly fixes the imbalance toward k=k_max; trivial weighted resampling | Easy |
| **Beam-search trace mining** | While solving training puzzles, record (state, depth_from_goal_along_found_path) for *every* node on the recovered solution; treat as labels (upper bounds, but tight) | This is *the* DeepCubeA self-imitation / Arfaee-Zilles-Holte bootstrap idea; tight labels, free data | Easy if you store traces |
| **Replica-exchange / parallel-tempering MCMC** | Multiple chains at different "temperatures" β (β=0: uniform random move; β>0: prefer moves increasing predicted distance); periodic swap | Standard tool for sampling rugged landscapes (Swendsen-Wang 1986; Geyer 1991; Earl-Deem 2005; review: arXiv:2008.05367, 2502.19240). Expensive; mostly relevant when "far states" are the bottleneck | Medium |
| **Metropolis-Hastings with energy = predicted distance** | Same as above with single chain | Can stagnate; rarely worth it on its own | Easy |
| **Backward walks from "deeply scrambled" states** | Apply WCA-style scrambles (e.g., the Megaminx WCA scramble length 70) and treat the whole trajectory backwards | Equivalent to forward walks (the inverse of every generator is a generator) — but a *different* measure if you use a different random number of moves | Trivial; equivalent |
| **Stratified sampling by depth** | Sample k uniformly in {1,...,k_max} per walk, not "always k_max with snapshots" | Reweights training distribution toward small k; gives more weight to easy-to-learn region near identity | Easy |
| **Coupling-from-the-past (Propp-Wilson)** | Exact samples from the *stationary* (uniform) distribution on the group | Mostly conceptual: gives you uniform samples, but stationary samples are the worst training data because all are near typical distance | Hard, low ROI |
| **Group-theoretic decompositions / coset walks** | Walk within a coset (e.g., Kociemba's pre-G1 phase for Rubik) | Powerful for two-phase solvers but specialized; for Megaminx the coset structure is much messier (semi-direct A₂₀ × A₃₀ × twists) | Hard |

### 3.1 n_back: the central knob

For n_back=1 your label has variance ~Θ(k); for n_back ≫ 1 the label tracks true distance very closely up to ~ the girth of the local graph. The downside: too aggressive n_back at small graphs makes walks *terminate early* (no admissible move), creating selection bias toward locally-non-degenerate scrambles. On Megaminx (degree 24) you have lots of room: n_back=32 is comfortably below the saturating value. The CayleyPy authors report no benefit from going much above ~32 on Sₙ-LRX (degree 3), so on degree-24 Megaminx the natural ceiling is similar. **Recommendation: ablate n_back ∈ {1, 8, 16, 32, 48} on a small (say ResMLP@1M-param, single-agent, beam=2¹⁶) pipeline; pick the smallest that produces optimal-or-near-optimal solutions on a held-out d=20 set.**

### 3.2 Beam-search trace mining ("self-imitation")

This is essentially Arfaee, Zilles, Holte (2010) *Bootstrap Learning of Heuristic Functions* (ICAPS) and the principle behind DeepCubeA's "iterative deepening" curriculum. The idea is: the solver finds *some* solution (length L) for a previously-unsolved or barely-solved scramble. That solution's k-th node is provably at distance ≤ L−k from solved. These labels are **strictly tighter** than RW labels for the same state. The 2025 *Learning Admissible Heuristics for A\** paper (arXiv:2509.22626) argues you want admissible (lower-bound) heuristics; the bootstrap approach gives upper bounds, which is fine for inadmissible-heuristic beam search. **Practical recipe:** every B beam-search runs, dump the (state, solution_length_remaining) pairs of completed puzzles, sample-mix into the next epoch's training data with weight 0.1–0.3.

### 3.3 Replica exchange / parallel tempering on the puzzle graph

This sits at the boundary of "worth trying" and "research project." The idea: maintain K parallel walks at K temperatures, with high-temperature walks doing pure RW and low-temperature walks doing biased walks (toward predicted-far states). Periodically propose swaps with Metropolis-Hastings acceptance to retain detailed balance. The classical reference is Swendsen-Wang (1986); modern RL adaptation includes *Replica Exchange Stochastic Gradient MCMC* (Deng-Feng-Gao-Liang-Lin, ICML 2020, arXiv:2008.05367) and *Langevin SAC* (parallel tempering for RL). This would *only* help if your bottleneck is sampling deep states; given that Megaminx's diameter is bounded by 133 and a 24-degree non-backtracking walk of length 100 already covers a wide swath of the diameter shell, the gain is likely modest. **Skip on a 2–3 day budget; revisit if you have a multi-week run.**

### 3.4 Symmetry-aware sampling

Megaminx has a 60-element rotational symmetry group (icosahedral I, order 60) that acts on the state space. This means you can *deduplicate* RW samples by canonicalizing each state into its lexicographically-smallest orbit representative. Effective training-set size is ~60× smaller for the same model accuracy, because you're not learning 60 redundant copies of each state's distance. Konen's TD-N-tuple Rubik work (arXiv:2301.12167) explicitly reports symmetry handling "greatly increases sample efficiency." **Implementation:** precompute the 60×120 permutation matrices for I acting on facelets; canonicalize each generated state (one matmul per state) before storing. Memory and time are both ~unchanged, but training is much more efficient. *Expected: equivalent to a 5–10× larger training set.*

---

## 4. Alternatives to Random Walks

| Alternative | Cost | Label quality | Megaminx feasibility |
|---|---|---|---|
| **Full BFS to depth d (exact distance labels)** | |Sphere(d)| states; for Megaminx face-turn metric: 1, 24, 528, ~11.6k, ~250k, ~5.5M, ~120M, ~2.6B at d=0,…,7 (extrapolating from speedsolving.com Megaminx growth tables) | Exact / optimal | **d=6 (≈19.4M, you have it) is comfortable. d=7 (≈250M-2B depending on metric and dedup) is borderline on a 4090. d≥8 needs disk-backed retrograde** |
| **Retrograde / backward BFS from solved (= same as above)** | Same as BFS | Same | Same |
| **Pattern Databases (Korf 1997)** | Memory-bound | Admissible / exact lower bound | Possible but Megaminx has no clean cubie-subset like Rubik's "8 corners" — partial PDB on, say, "20 corners only" or "30 edges only" is feasible (~3¹⁹·20! ≈ 5·10²³ — too big) |
| **Solver trace data (Arfaee-Zilles-Holte 2010)** | Free if you're already searching | Tight upper bound | Highly recommended |
| **A\* / IDA\* with admissible heuristic** | Admissible h needed | Optimal | Lacking a strong admissible h on Megaminx; mostly aspirational |
| **Curriculum learning (DeepCubeA-style ADI)** | Same as RW + Bellman | Iteratively tight | The standard |
| **Inverse moves from solved (= scramble interpretation)** | Same as forward walk | Same | Mathematically identical for Cayley graphs (each generator is invertible and the inverse generator is in the set), so this is a notational re-parametrization |
| **Synthetic data from teacher model (distillation)** | Training a teacher first | Inherits teacher's biases | Useful if you have a single very good agent and want many compute-efficient students; this is what model compression on AlphaCube does |
| **Active learning / uncertainty-guided sampling** | Need ensembles or MC-dropout | Adapts to model | Untested on cube-like, but the DeepCubeA "h-distribution shift" plot suggests sampling near the current h's "boundary of competence" is high-yield |

### 4.1 BFS shells (your "BFS-d6 mixin")

Megaminx face-turn growth function (computed for closely related puzzles; Megaminx face-turn shells from Hessen HPC project): the number of distinct states at depths 0–6 totals ≈ 19.4M. At d=7 it's roughly an order of magnitude bigger. Storage: 19.4M states × 120 bytes/state = ~2.3 GB raw, ~150 MB with bit-packed permutation encoding. Feasible. **At d=7–8 you're in the 100 GB territory** and need disk-backed deduplication (the HKHLR project did 22 GB at d=7 with bit-packing). This is the crucial reservoir of *exact* labels. Use them with high sample weight, especially for the final epochs after Bellman has converged.

### 4.2 DeepCubeA's autodidactic value iteration (DAVI) — what it actually does

McAleer et al. 2018 *Solving the Rubik's Cube Without Human Knowledge* (arXiv:1805.07470) and Agostinelli et al. 2019 *Solving the Rubik's Cube with Deep Reinforcement Learning and Search* (Nature Machine Intelligence 1:356–363):

- **Data:** at each iteration, take the solved state, do K random scrambles (K up to 30 for Rubik's QTM); call each intermediate state x_k.
- **Targets (DAVI):** for each x, expand all 12 (24 for Megaminx) children, evaluate the *current target network* on each child, and set y(x) = 1 + min_a target(child). For x=solved, y=0. **No clipping by walk depth is described in the original paper** (although your "clip(·, 0, walk_depth)" is a defensible improvement that CayleyPy-RL §2.4 also does).
- **Target network refresh:** when training loss drops below a threshold, copy weights → target. Roughly equivalent to your "every 10 epochs" schedule.
- **Curriculum-like emergence:** Figure 3 of the Nature MI paper shows the network first learns easy (k=1–5) cubes, then progressively harder ones; the dashed lines in their plot are the true average cost-to-go and the network catches up shell by shell.
- **Total compute (DeepCubeA):** 10 billion training examples, 36 hours on 4× Pascal GPUs.

Differences from your pipeline: DeepCubeA refreshes when *loss drops*, you refresh on a fixed schedule. DeepCubeA does not have the "BFS-d6 mixin" or anything similar; you do, which is an upgrade.

### 4.3 EfficientCube (Takano 2023, *Self-Supervision is All You Need*, TMLR; ICLR 2024)

The big methodological change in EfficientCube vs. DeepCubeA: **predict the *last move* (a 24-way classification) rather than the cost-to-go (a regression).** Equivalently, predict the inverse-move policy directly. Train via the random-scramble pipeline with categorical cross-entropy. At search time use best-first / weighted-A* with the policy as heuristic.

- Trained with **1 billion examples** (10× less than DeepCubeA) and matched/beat DeepCubeA on 3×3×3, 15-puzzle, 7×7 Lights Out.
- Their scaling-law study (with HTM): 100M solutions (8B examples) gives ~98% solve rate at beam=2¹⁸; 1B solutions saturates quality.
- For Megaminx, an analogous policy-head ResMLP would output 24 logits and you'd train with cross-entropy on the **inverse of the last applied generator** during scramble. CayleyPy authors note (arXiv:2502.13266) their value-prediction approach is **26× faster at solve time and 18.5× cheaper to train** than EfficientCube on 3×3×3, with slightly better solutions, so for *this* problem class value-prediction is empirically the winner — but a policy-head may help diversity in an ensemble.

### 4.4 Self-bootstrap / iterative deepening (Arfaee-Zilles-Holte 2010, *Bootstrap Learning of Heuristic Functions*)

Train a weak heuristic on whatever data you can get. Use it to solve some easy problems. Use those solutions to add tighter labels. Train a better heuristic. Repeat. The 2010 paper showed this works on classical sliding-tile puzzles; Lehnert (2020+, *BootstrappedSearch*) and the CEA paper above are descendants. **For Megaminx**: this is essentially "Solver-Trace Mining" (§3.2) applied iteratively. The core empirical claim: bootstrap iterations 2–5 are highly informative, then plateau.

### 4.5 Pattern Databases on Megaminx — would they work?

Korf 1997 (*Finding Optimal Solutions to Rubik's Cube Using Pattern Databases*, AAAI'97) used 8-corner-cubie + 6-edge-cubie + 6-edge-cubie additive PDBs. Each lookup is a lower bound; sum is admissible. Total memory ~80 MB. For Megaminx the equivalent would be:
- **20-corner orientation/permutation:** 20! × 3¹⁹ ≈ 2.8·10²⁷ — too big.
- **Subset of corners (e.g., 8 corners):** 20!/(12!) × 3⁸ ≈ 4·10¹³ — still too big.
- **Centers / fixed pieces:** Megaminx has only 12 face centers; unlike Rubik's they don't permute under face turns of just a single face (centers spin in 3×3×3 only with slice moves; for Megaminx the centers don't move at all under face turns since the centers *are* the rotation axes). So no PDB on centers.

Korf-style PDBs likely **infeasible at standard sizes for Megaminx** because there's no small "natural" subset of pieces. There's a research opening: a learned, neural, *compressed* PDB (the CEA paper's idea, arXiv:2509.22626) — train a small NN to predict admissible distances on a corners-only or edges-only abstraction. But this is multi-week work.

### 4.6 Kociemba/Thistlethwaite-style two-phase decomposition

For 3×3×3, Kociemba's algorithm reduces to a subgroup G₁ = ⟨U,D,L²,R²,F²,B²⟩, then within G₁ (Thistlethwaite uses 4 stages). For Megaminx, Kociemba has produced lower bounds via stage decomposition into "0/1/2n/2o/3" subgroups (see the SpeedSolving thread cited in the CayleyPy-Megaminx context). A practical Megaminx Kociemba-style solver (Hessen HPC project) achieves average 82–102 moves with seconds of CPU. **This is a strong baseline to beat.** 95,682 / 1001 ≈ 95.6 moves average — meaning you're *between* HKHLR's medium-budget solver (102 avg) and their high-budget solver (82 avg). To beat 82 you must move past coset-decomposition methods, which is exactly what neural-heuristic + beam-search is for.

### 4.7 Generating data via inverse moves applied to solved (mathematical equivalence)

For a Cayley graph, every generator g has an inverse g⁻¹ that is also a generator (or whose application is). So *forward random walk from identity* and *backward random walk from identity following inverses* are statistically identical. There's no benefit to "scramble interpretation" beyond bookkeeping — but it does change one thing: in EfficientCube-style training, the target is **the inverse of the last move applied** (a single supervised label, free of bootstrapping). That framing is enabled by the inversion, not by a different sampling distribution.

---

## 5. Variations on Bellman Refinement

| Variant | Description | Expected effect on your pipeline |
|---|---|---|
| **Hard target updates (your current)** | Copy weights every K epochs | Standard; can oscillate |
| **Polyak / soft target update** | θ_target ← (1−τ)θ_target + τθ_online with τ ≈ 0.005 | Smoother, fewer oscillations; 1–2% improvement typical in DQN |
| **n-step Bellman** | y = n + min_aₙ target(applyⁿ(s,a₁..aₙ)) | Trades bias↑ for variance↓; for n=2 it's a depth-2 BFS over the 24×24=576 children, manageable |
| **TD(λ) / eligibility traces** | Geometric mixing of n-step returns | Mostly relevant for episodic on-policy RL; not a clear win here |
| **Double Q / double V** | Two networks: select action with one, evaluate with other | Reduces *overestimation* bias — but here we have *underestimation* via clipping, so opposite sign. Not obviously useful |
| **Distributional / quantile heads (C51, QR-DQN, IQN)** | Predict distribution over distance, not just the mean | Bellemare-Dabney-Munos 2017; Dabney-Rowland-Bellemare-Munos 2018. **Genuinely useful here**: distance is naturally non-negative integer; quantile head lets the beam search use the lower-quantile (optimistic) estimate to stay admissible-like, or the median for stability. ROI: 1–3% with moderate code complexity |
| **Persistent target vs. randomized resets** | Reset target to online occasionally | Some theoretical justification (avoids local cycle traps); empirical evidence weak |
| **DAVI's exact recipe** | Update target on loss-threshold rather than schedule | Adaptive scheduling; closer to convergence-driven |
| **Search-and-Bellman hybrids** | Use beam search to find true-or-tighter distances on a sample of states; use those as Bellman targets weighted ∞ | Strong: combines bootstrap + Bellman |
| **Mixed exact (BFS-d6) + Bellman targets (your Option B)** | Sample mix during minibatch construction | The right thing; tune ratio |
| **Best-of-K Bellman** | y = 1 + (k-th smallest of children's predictions); say K=3 | Reduces variance from a single noisy min; underexplored but theoretically clean |
| **Mean-teacher / EMA student** | Maintain θ_EMA, train student against EMA's targets, but EMA student weights don't equal target | Stabilization trick from semi-supervised learning (Tarvainen-Valpola 2017) |
| **Bootstrapped DQN ensembles** | K independent online networks; Bellman target is min over their means or one of their k samples | Implicit posterior; combines well with multi-agent |

### 5.1 Practical recipe for your "Option B" mix ratio

Treat a minibatch as drawn from three sources: (a) random walk + clipped Bellman target (most informative for k > 6), (b) BFS-d6 exact label (most informative for d ≤ 6), (c) solver-trace upper bound (tighter than RW). A defensible ratio: **60% (a), 25% (b), 15% (c)**; sample-weight (a) at 1.0, (b) at 2.0–3.0, (c) at 1.5. Importantly, the (b)-states near solved also dominate beam-search collapse at the end of every solution; weighting them up directly improves the last few moves of every solve.

### 5.2 n-step Bellman for Megaminx

Depth-2 BFS expansion is 24² = 576 forward calls per state per Bellman update — heavy. But you can amortize: do depth-2 only every 4th Bellman epoch, or only on a sub-batch. Bias-variance: at n=2 you have less variance (averaging more children is less noisy) but more bias from the same target network. DeepCubeA paper's Methods section reports they tried depth-2 BFS and got *slightly worse* results than depth-1 — so this might be a wash, but worth checking because their setting was different.

### 5.3 Distributional heads

The cleanest formulation for distance is **Cross-Entropy Admissibility loss** (CEA, *Learning Admissible Heuristics for A\**, arXiv:2509.22626, 2025): predict a categorical distribution over [0, D_max] and use a hybrid CE loss with an explicit admissibility-violation penalty. For beam search with greedy selection, having a distribution lets you use **CVaR-style** decision rules (pick the action that minimizes a risk-aware functional) and **decouples** training stability from beam-time decision rule. ROI: 1–3% if you're disciplined about hyperparameters, plus opens the door to distributional ensembling (mix posteriors).

### 5.4 What changed from DAVI to EfficientCube to CayleyPy?

- **DAVI/DeepCubeA (2019):** value head + Bellman + weighted-A* search.
- **EfficientCube (2023):** policy head (predict last move) + categorical-CE + best-first search. **No Bellman.** Same random-scramble data.
- **CayleyPy (2025):** value head trained to predict *diffusion distance* (i.e., *just the random-walk-step-count, no Bellman*) + beam search. Multi-agent ensemble. **They explicitly note Bellman gives only modest improvements over raw diffusion-distance training in their setting** (CayleyPy-RL Table 1, where DQN refinement on top of diffusion warmup raises Spearman from 0.96 to 0.98 at S₈).

The big lesson: **training simplicity + multi-agent + larger beam ≫ refined Bellman**. Your Bellman is helping you, but it's not where the next 10% comes from.

---

## 6. Alternatives to Bellman Refinement

| Alternative | Description | Realistic on 2–3 days? |
|---|---|---|
| **Pure diffusion-distance training** | Just RW labels, no Bellman | Yes; CayleyPy showed it's enough at scale |
| **Pure curriculum from BFS shells outward** | Train to depth 1, expand BFS to depth 2, retrain, etc. | Yes for small d (≤6); after that you fall back to RW |
| **Q-learning** (predict (state, action) → distance) | One forward pass returns 24 cost-to-gos; *Q\* search* (Agostinelli et al. ICAPS-PRL 2024) gets 129× speedup on Rubik's | Yes; modest engineering |
| **Q-distillation from a value model** | Train V, then distill into Q | Your "m06" attempt; can fail if distillation distribution mismatches search distribution |
| **Search-based imitation (DAGGER-like)** | Run beam search, train next policy on its trajectories | Yes; subset of solver-trace mining |
| **AlphaZero-style MCTS-in-the-loop** | Joint policy + value, MCTS for self-play improvement | DeepCube-MCTS (McAleer 2018) used this; Konen 2023 (arXiv:2301.12167) reports MCTS wrapper is essential for 3×3×3. Compute-heavy, code-heavy. Plausibly worth it on 2 weeks, not 2 days |
| **Gumbel AlphaZero / TreeQN** | Gumbel-perturb Q for exploration | Marginally better than vanilla AlphaZero in some settings |
| **PPO/A2C with sparse reward** | Pure policy gradient | *Documented to fail* on Rubik's without bootstrap (Lin-Liang 2024 arXiv:2411.19583 only reports 99.4% on **2×2×2**, the toy version). Skip |
| **Goal-conditioned BC + HER** | (s, g) → distance; HER relabels failures | Andrychowicz 2017 HER; useful when goals are diverse, but Megaminx has *one* goal so HER is less applicable. The relabeling reduces to "what state did I reach after k random moves," which is exactly RW |
| **World-model / model-based planning (MuZero)** | Learn a transition model + planner | The transition model is *exact and known* for puzzles; learning one is wasteful |
| **Value Iteration Networks / Universal Planning Networks** | Differentiable VI as an architecture inductive bias | Theoretical; scaling unclear |
| **Diffusion / flow matching for path generation** | Learn p(path | start, goal) | Cutting-edge; no published cube success |
| **Contrastive learning on (near, far) pairs** | Triplet loss to embed states by distance | Useful as auxiliary loss; not a replacement for the regression head |
| **Energy-based models / score matching** | Learn an energy E(s) ∝ distance | Equivalent up to normalization to value learning; no obvious advantage |

### 6.1 Q* search and Q-distillation

The Q* paper (Agostinelli, Shperberg, Shmakov, McAleer, Fox, Baldi 2024, arXiv:2102.04518; PRL@ICAPS 2024) makes a *deployment-time* argument: A* + value-net needs one forward pass per child (24 forward passes per node). A Q-net returns all 24 costs in one forward pass. They report 129× faster, 1288× fewer nodes generated on Rubik's at the *same* solution quality. **For your 24-action Megaminx this would directly translate** to ~24× fewer GPU calls per beam-search step, allowing ~24× larger beam at same wallclock — and we've already seen that beam-doubling gives ~1 move improvement. So Q* search alone is potentially a 5-move-per-puzzle improvement, i.e., ~5000 score points. **High ROI if your m06 Q-distillation failure can be debugged.** The likely failure mode of Q-distillation: the V model gives correct min over children, but the gradient toward "predict 24 child-distances" creates conflicting targets when multiple actions give the same min — solution is to train Q from scratch with the same Bellman target structure rather than distill from V.

### 6.2 AlphaZero-style on Megaminx

In principle: train (π, V) jointly via MCTS-improved targets. DeepCube (McAleer 2018) used MCTS at solve time but trained via ADI (which is essentially a 1-step BFS). True AlphaZero on Rubik's was attempted by Konen (Towards Learning Rubik's Cube with N-tuple-based Reinforcement Learning, arXiv:2301.12167) and works on 2×2×2 with effort, partially on 3×3×3. Megaminx is a much larger search problem; expect the MCTS budget per training step to dominate and outweigh gains. **Not recommended for 2–3 days.**

### 6.3 LLM-as-solver (GPT-4 / Claude)

Honestly: useless on Megaminx. The state representation is 120 numbers; chain-of-thought solvers fail catastrophically beyond 2–3 cube moves of depth.

---

## 7. Radically Different Paradigms

### 7.1 Group-theoretic solvers (Kociemba/Thistlethwaite for Megaminx)

**Status:** the HKHLR (Hessen HPC) Megaminx project produced a *practical* group-theoretic solver in 2023–2024 with **avg 82 moves** at 80 GB RAM, using Schreier-style coset decomposition — broadly Kociemba's approach. **Megaminx God's number upper bound: 133** (HKHLR), lower bound ≈ 45 (no rigorous match). Pure group-theoretic solvers will outperform any neural solver below ~80 moves average until model + beam scale catches up; they will not improve below ~75 moves average without huge RAM. This sets the practical floor for what neural-heuristic + beam-search must achieve to be competitive: **<82 average × 1001 = <82,082 score**. Your 95,682 indicates you're roughly 16% above optimal heuristic-search performance — squarely in the regime where pipeline improvements matter more than algorithmic paradigm switches.

### 7.2 Pattern Databases (Korf 1997 onward)

See §4.5 — likely infeasible at full scale on Megaminx; partial PDBs (e.g., neural compressed) are research projects.

### 7.3 IDA* with learned admissible heuristic

The 2025 *Learning Admissible Heuristics for A\** (CEA loss) paper (arXiv:2509.22626) shows you can train near-admissible neural heuristics that beat compressed PDBs on 3×3×3, with admissibility violations under 10⁻⁶. This *enables* IDA*, which gives optimal solutions. The cost: IDA*'s linear-memory benefit is moot for you (beam fits in 24 GB), and IDA* on a 24-action graph with God's number ~130 has branching ~23 and depth ~130 — node count is astronomical. **For Megaminx's diameter, IDA*-with-NN-heuristic is essentially infeasible.** Beam search is the right tool.

### 7.4 Bidirectional / meet-in-the-middle search

Forward beam from scrambled + backward BFS shell from solved (your BFS-d6 doubles as the meet-in-the-middle target). Holte-Felner 2017 *MM: A Bidirectional Search Algorithm Guaranteed to Meet in the Middle* (Artificial Intelligence) gives the rigorous theory. Your **MITM-d6 termination** is exactly this and is the right idea. To extend: **store BFS-d7 or d8** if memory allows; the union "meet zone" is much bigger and beam search will hit it earlier, shaving 2–4 moves off long solutions. **Each additional BFS shell is roughly 24× more states; d7 ≈ 470 M states ≈ 5–60 GB depending on packing. d8 needs 100s of GB or disk-streaming dedup.** Borderline but possible on a 4090 + 32 GB RAM machine for d=7.

### 7.5 AlphaZero / MuZero — see §6.2.

### 7.6 LLM-as-solver — see §6.3 (skip).

### 7.7 SAT / ILP / constraint programming

Modeling Megaminx-solving as SAT/ILP at depth 60+ is not competitive; SAT solvers don't handle this scale of group-theoretic structure. CP optimizers fail similarly. Skip.

### 7.8 Sampling-based motion planning (RRT/PRM analogs)

In a discrete combinatorial graph these lose their continuous-sampling advantages. RRT-Connect is roughly bidirectional beam without learned heuristic — strictly worse than what you have. Skip.

### 7.9 Garside normal form / reduced words

Megaminx's group does not have an obvious Garside structure. The Schreier-Sims algorithm (implemented in GAP) gives word decompositions but exponentially long ones (CayleyPy-RL §1.3 quotes the Shamir 1989 result that Schreier-Sims outputs are typically exponentially long; GAP fails on Megaminx-sized groups for our purposes). **Not a competitive paradigm.**

### 7.10 Symmetry exploitation (icosahedral I, order 60)

This *is* worth doing at every level:
- **Data dedup:** canonicalize states modulo I before training (§3.4).
- **Inference dedup:** during beam search, collapse states that are I-equivalent.
- **Solution post-processing:** find the symmetry conjugate of the found solution that is shortest under any orbit-equivalent interpretation (rare improvement, but free).

Konen 2023 (arXiv:2301.12167) reports symmetries "greatly increase sample efficiency" on 3×3×3 (which has order-48 symmetry); the same logic applies to Megaminx's larger 60-element group.

### 7.11 Reinforcement learning from scratch (no walks, no BFS)

Lin-Liang 2024 (arXiv:2411.19583) does "policy gradient without solved-state sampling" — but only on **2×2×2**. The sparse reward problem on Megaminx makes pure-RL essentially impossible. Skip.

### 7.12 Quantum / Grover

Theoretically Grover gives O(√N) for unstructured search, but our search isn't unstructured. Not even theoretical relevance. Skip.

### 7.13 Compressed sensing / sparsity in V

Pan 2021 (cited in CayleyPy-RL §1.3) trains a NN to predict the **non-abelian Fourier transform** of the distance function on Sₙ — exploiting Swan 2017's observation that distance functions are "bandlimited" in the Fourier sense. This is intellectually beautiful and has solved sub-problems but no full Rubik success. **Speculative.**

---

## 8. Cross-Field Connections

| Field | Transferable insight |
|---|---|
| **MCMC mixing-time theory** (Levin-Peres-Wilmer, Diaconis 1988) | Pick k_max ~ mixing time. Use spectral-gap estimates to predict when RW labels stop being informative. *Diaconis-Saloff-Coste log-Sobolev inequalities* give convergence-rate bounds for permutation walks. **The Cheeger / conductance perspective:** the "bottleneck ratio" of a Cayley graph determines how easily a walk escapes a region; for Megaminx the bottleneck is between solved-shell (rich structure) and bulk (homogeneous). |
| **Statistical physics of energy landscapes** (spin glasses, basin-hopping, Wales-Doye 1997 *J. Phys. Chem. A* 101:5111) | The "true distance" defines an energy landscape with one global minimum (solved) and many *basins* (clusters of states reachable via short paths from each other). **Disconnectivity graphs** would visualize this. *Basin-hopping* (perturb + minimize) is essentially what beam search does. **Replica exchange** (§3.3) is the standard rare-event tool. *Frustration*: many local minima of the heuristic are unrelated to true distance — visible in your beam-search failure modes. |
| **Optimal transport / Wasserstein distance** | The "Wasserstein metric" between predicted-distance distribution and true-distance distribution is what QR-DQN and IQN minimize. *Schrödinger bridges*: stochastic interpolation between scrambled and solved distributions — there's a paper waiting to be written here, but no current cube application. |
| **Spectral theory of random-walk operators on Cayley graphs** | The eigenvectors of the walk operator are *irreducible representations* of the group (Diaconis-Shahshahani). Eigenfunction with eigenvalue closest to 1 is essentially what your network learns first — it's the slowest-decaying mode and corresponds to the diffusion distance. Knowing this lets you engineer features (e.g., projections onto specific irreps) that capture the bandlimited structure. Pan 2021's idea precisely. |
| **Algebraic combinatorics** (Babai-Seress conjecture, Helfgott 2014) | Diameter bounds on Cayley graphs of simple groups: Babai conjectures O(log^c|G|). This says Megaminx's diameter ≤ a polynomial in 68×log(10) ≈ 156 — and indeed the proven upper bound is 133. Useful prior on k_max. |
| **Population genetics** (coalescent theory, effective population size) | At first glance: walks "mix" in time τ_mix analogous to coalescence time. The "population of beam-search nodes" is analogous to a population evolving under selection (= heuristic) and drift (= randomness). Beam-width is *effective population size*. This makes precise the CayleyPy-RL §2.3 analogy of "research community size." |
| **Theoretical CS: shortest paths on huge graphs** | Hub labeling, contraction hierarchies (Geisberger-Sanders-Schultes 2008), transit nodes — all rely on graph structure (low treewidth, hierarchical). Cayley graphs on simple groups are *expanders*: no useful hierarchy, no efficient hub labeling. This is why such methods don't apply. |
| **Computational chemistry: rare-event sampling** (replica exchange MD, metadynamics, milestoning) | Same problem of sampling deep / rare states. Metadynamics (Laio-Parrinello 2002) adds a history-dependent bias to the energy to discourage revisiting; this is an analog of your n_back generalization at the level of full state-space regions, not just last few states. **A research-quality idea: metadynamics-style biased walks based on a coarse "milestone" partition of the puzzle state space.** |
| **RL theory: PAC analysis of value iteration** | Bertsekas-Tsitsiklis convergence theorems require *contraction* in the Bellman operator; with NN function approximators this fails (the "deadly triad": function approx + bootstrapping + off-policy). Practical implications: target networks, clipping, n-step returns are all stabilizers. Your pipeline already has these. |
| **Neuroscience of value learning** (TD error, successor representations) | The successor representation φ(s) = E[Σ γ^k φ(s_k)] is a shortcut for value computation under policy changes (Dayan 1993). For a fixed policy (= optimal) it equals the discounted indicator of future visits — close to your value head's purpose. **Auxiliary loss idea**: predict φ(s) (the discounted future-state-occupancy distribution) as a multi-task target. Untested on cubes; speculative. |
| **Quantum computing** | No practical relevance at this scale. |
| **Compressed sensing / sparsity in V** | Pan 2021 above. Speculative. |

---

## 9. Tier list / ROI ranking with compute-budget context

Compute budget assumptions: **2–3 days on RTX-4090 (24 GB VRAM, ~200 W effective, ~50 TFLOPS FP16)**. Existing pipeline trains in roughly 4 hours per epoch × 50 epochs = ~8 days total — so you can afford ~1 retrain plus a few short ablations, or many short ablations and an ensemble.

### Tier 1 — ship within a single GPU-day, very high ROI
1. **Multi-agent ensemble (8–16 agents).** Each takes one short training run (e.g., 6–8 epochs). Submit per-puzzle min-length. Expected: −2% to −5% absolute score (roughly 2k–5k score points). **First thing to do.**
2. **Increase n_back to 32 (with neighbor-banning).** Single-line code change in walk generator. Free.
3. **Sample-weight BFS-d6 mixin up.** Set sample weight 2–3× for d≤6 states. Free.
4. **Beam-width sweep on residual hardest puzzles** (W=2¹⁹, 2²⁰ on the 100 hardest of the 1001). Cost: a few hours. Expected: −0.5% to −1.5%.
5. **Symmetry-canonicalize states** (icosahedral I orbit reps). One-time precompute of permutation matrices, runtime ~free.

### Tier 2 — 1-day each, high ROI
6. **Solver-trace mining + 1 bootstrap iteration.** Run beam search, capture (state, distance-along-found-path), retrain.
7. **Polyak target-network averaging** (τ=0.005) replacing your "every 10 epochs" hard update.
8. **n-step (n=2) Bellman.** Implement ⌈1 day⌉; expected modest improvement.
9. **Q*-style head + Q* search** (one forward call returns 24 costs). If you can debug your m06 failure, this enables ~24× larger effective beam. **Very high upside if it works; medium risk.**

### Tier 3 — 2–3 day projects
10. **BFS-d7 shell** (memory-permitting). Hundreds of millions of states; needs disk-backed dedup. Big payoff because it deepens the meet-in-the-middle zone.
11. **Distributional / quantile-regression head** (CEA loss or QR-DQN style). Code complexity moderate.
12. **EfficientCube-style policy head as a *second* agent in the ensemble** (provides diversity).
13. **Retrograde-BFS-from-each-d=6-state**, using those as "near-solved seeds" for warm-start beam search.

### Tier 4 — research projects, week+
14. **MCTS-AlphaZero** training loop (joint π, V).
15. **Replica exchange / parallel-tempering walks** for far-state coverage.
16. **Neural compressed pattern database** (CEA-style admissible heuristic on a corner-only abstraction).
17. **Two-phase Kociemba-style decomposition + per-phase neural heuristic.**

### Tier 5 — likely ROI-negative on your problem
- Pure RL (PPO/A2C from scratch).
- LLM-as-solver.
- SAT/ILP/CP, RRT/PRM.
- Quantum.

---

## 10. Open Research Questions and Further Reading

### Open questions specifically for this competition
1. **How tight is the *spherical growth function* of Megaminx in face-turn vs. quarter-turn metric?** This determines optimal k_max stratification weights and the BFS-d depth where exact labels become infeasible.
2. **What is the *autocorrelation* of beam-search trajectories on Megaminx?** If many failed beams pass through a small "bottleneck region" (a high-conductance set of states), targeted oversampling there is high-yield.
3. **Does ensembling primarily reduce *variance* across hard puzzles, or does it discover *qualitatively different* solution paths?** If the latter, you can use *trajectory diversity* as a training signal.
4. **What's the right symmetry-aware architecture?** Megaminx's icosahedral group has no convolutional structure on a regular grid; equivariant networks à la Cohen-Welling 2016 would need a custom design over the dodecahedron's face-permutation action.
5. **Can a single network simultaneously learn V and π (joint head)?** EfficientCube says policy alone is enough; CayleyPy says value alone is enough. A shared backbone with both heads might combine their strengths.

### Key references (full bibliography)

**Core neural-heuristic + cube papers**
- McAleer, S., Agostinelli, F., Shmakov, A., Baldi, P. (2018). *Solving the Rubik's Cube Without Human Knowledge.* arXiv:1805.07470. (DeepCube, ADI, MCTS-search.)
- Agostinelli, F., McAleer, S., Shmakov, A., Baldi, P. (2019). *Solving the Rubik's cube with deep reinforcement learning and search.* Nature Machine Intelligence 1:356–363. (DeepCubeA, DAVI, weighted-A*, 10B examples, 36h on 4×Pascal.)
- Takano, K. (2023). *Self-Supervision is All You Need for Solving Rubik's Cube.* TMLR / arXiv:2106.03157. (EfficientCube, predict last move, 1B examples, scaling laws.)
- Chervov, A., Khoruzhii, K. et al. (2025). *A Machine Learning Approach That Beats Large Rubik's Cubes (CayleyPy).* arXiv:2502.13266 / NeurIPS 2025. (ResMLP + diffusion-distance + beam, multi-agent, 4×4×4 and 5×5×5 SOTA.)
- Chervov, A. et al. (2025). *CayleyPy RL: Pathfinding and RL on Cayley Graphs.* arXiv:2502.18663. (DQN + diffusion-distance hybrid, generalized non-backtracking, S₃₀ LRX.)
- Lin, Y., Liang, S. (2024). *Solving Rubik's Cube Without Tricky Sampling.* arXiv:2411.19583. (Pure policy gradient, only 2×2×2.)
- Konen, W. (2023). *Towards Learning Rubik's Cube with N-tuple-based Reinforcement Learning.* arXiv:2301.12167. (Symmetry exploitation, MCTS-wrapper essentiality.)
- Pan, J. (2021). *Solving Rubik's Cube via Non-Abelian Fourier Transform.* (Bandlimited V function.)
- DeepCubeA codebase: github.com/forestagostinelli/DeepCubeA.

**Heuristic search & PDBs**
- Korf, R. (1997). *Finding Optimal Solutions to Rubik's Cube Using Pattern Databases.* AAAI'97. (Original PDB.)
- Korf, R., Felner, A. (2002). *Disjoint pattern database heuristics.* Artificial Intelligence 134:9–22. (Additive PDBs.)
- Felner, A., Korf, R., Hanan, S. (2004). *Additive Pattern Database Heuristics.* JAIR.
- Korf, R., Reid, M., Edelkamp, S. (2001). *Time complexity of iterative-deepening A\*.* AIJ.
- Holte, R., Felner, A., Sharon, G., Sturtevant, N. (2017). *MM: A bidirectional search algorithm guaranteed to meet in the middle.* AIJ. (Bidirectional theory.)
- Arfaee, S. J., Zilles, S., Holte, R. (2010). *Bootstrap Learning of Heuristic Functions.* SoCS / ICAPS-PRL. (Iterative tightening from solver traces.)
- Agostinelli, F., Shperberg, S., Shmakov, A., McAleer, S., Fox, R., Baldi, P. (2024). *Q\* Search: Heuristic Search with Deep Q-Networks.* PRL@ICAPS 2024 / arXiv:2102.04518. (Q-net for batched A*.)
- *Learning Admissible Heuristics for A\*: Theory and Practice.* arXiv:2509.22626 (2025). (CEA loss, near-admissible NN heuristic beats compressed PDBs on 3×3×3.)
- *On Using Admissible Bounds for Learning Forward Search Heuristics.* arXiv:2308.11905 (Truncated-Gaussian model.)

**Distributional RL / value-distribution heads**
- Bellemare, M., Dabney, W., Munos, R. (2017). *A Distributional Perspective on Reinforcement Learning.* ICML 2017. (C51.)
- Dabney, W., Rowland, M., Bellemare, M., Munos, R. (2018). *Distributional Reinforcement Learning with Quantile Regression.* AAAI. (QR-DQN.)
- Dabney, W. et al. (2018). *Implicit Quantile Networks for Distributional RL.* ICML.

**Mixing-time and Cayley-graph theory**
- Diaconis, P., Shahshahani, M. (1981). *Generating a random permutation with random transpositions.* Z. Wahr. verw. Gebiete 57:159–179. (Cutoff at (n log n)/2.)
- Diaconis, P. (1988). *Group Representations in Probability and Statistics.* IMS monograph.
- Levin, D., Peres, Y., Wilmer, E. (2017). *Markov Chains and Mixing Times.* AMS. (Standard textbook.)
- Alon, N., Benjamini, I., Lubetzky, E., Sodin, S. (2007). *Non-backtracking random walks mix faster.* Combinatorica.
- Berestycki, N., Şengül, B. (2014). *Cutoff for conjugacy-invariant random walks on the permutation group.* arXiv:1410.4800.
- Hermon, J., Kozma, G. (2020). *Sensitivity of mixing times of Cayley graphs.* arXiv:2008.07517.
- Kempton, M. (2016). *Non-backtracking random walks and a weighted Ihara's theorem.* arXiv:1603.05553.
- Babai, L. (1992). *Local expansion of vertex-transitive graphs and random generation in finite groups.* STOC.
- Helfgott, H. (2014, 2019). *Surveys on the Babai conjecture and growth in groups.*
- Qu, Y. (2024). *Rubik's Cube Scrambling Requires at Least 26 Random Moves.* arXiv:2410.20630. (Mixing-time on Rubik's specifically.)

**Megaminx specific**
- Xu, G. J. (2018). *Solving Megaminx Puzzle With Group Theory.* Yau-Tomas math award paper. (Group structure: semidirect product of A₂₀×A₃₀ with twist groups.)
- HKHLR Megaminx Solver Project (Hessen HPC, 2023–2024). Avg-82-move solver with 80 GB; God's number ≤ 133 upper bound. www.hkhlr.de/en/projects/4006.
- speedsolving.com/wiki — Megaminx state space 1.01×10⁶⁸ on the 12-color version; Kociemba-style group decomposition discussion threads.

**Replica exchange / parallel tempering**
- Geyer, C. (1991). *Markov Chain Monte Carlo Maximum Likelihood.* Computing Science and Statistics 23:156–163.
- Earl, D. J., Deem, M. W. (2005). *Parallel tempering: theory, applications, and new perspectives.* PCCP 7:3910–3916.
- Deng, W., Feng, Q., Gao, L., Liang, F., Lin, G. (2020). *Non-convex Learning via Replica Exchange Stochastic Gradient MCMC.* ICML / arXiv:2008.05367.

**Hindsight / goal-conditioned RL**
- Andrychowicz, M. et al. (2017). *Hindsight Experience Replay.* NeurIPS.
- Ghosh, D. et al. (2020). *Learning to Reach Goals via Iterated Supervised Learning.* (GCSL, related to iterative bootstrap.)

**Search algorithms & competition reports**
- Kaggle Santa 2023 (The Polytope Permutation Puzzle) leaderboard and discussion threads (kaggle.com/competitions/santa-2023). Top approaches use small-support generators + beam search with Hamming heuristic — explained in CayleyPy-RL §1.3.
- Hessel, M. et al. (2018). *Rainbow: Combining Improvements in Deep Reinforcement Learning.* AAAI.
- van Hasselt, H., Guez, A., Silver, D. (2016). *Deep Reinforcement Learning with Double Q-learning.* AAAI.

**Group-theoretic algorithms**
- Sims, C. (1970). *Computational methods in the study of permutation groups.*
- Schreier-Sims algorithm (in GAP / SymPy).
- Kociemba, H. *Two-Phase Algorithm.* kociemba.org.
- Rokicki, T. et al. (2014). *The Diameter of the Rubik's Cube Group is Twenty.* SIAM J. Discrete Math.

---

### Final Note on Strategy

If I had to compress this report into a single recommended sequence of actions over your 2–3 day budget, it would be:

**Day 1 (morning):** Implement multi-agent ensembling infrastructure. Train 4 agents in parallel (or sequentially on the same GPU at smaller per-agent budget). Verify aggregation. *(Tier-1 #1.)* While that runs, also: generalize n_back to 32 with neighbor-banning; pre-compute the 60 icosahedral symmetry permutations; sample-weight BFS-d6 by 2.5×. *(Tier-1 #2, #3, #5.)*

**Day 1 (evening):** Submit ensemble result. Should be at ~91k–93k score. If yes, double the agent count overnight.

**Day 2:** Implement solver-trace mining (record beam-search successful trajectories, retrain with these as upper-bound labels). Parallelly: try Polyak target updates and at least try (small experiment) the Q-head variant for Q*-search — even a partial success on Q* would let you double effective beam.

**Day 3:** Beam-width sweep on top-100 hardest puzzles at W=2¹⁹–2²⁰. Run BFS-d7 if disk allows. Final ensemble re-aggregation. Submit.

The combined effect of just Tier-1 + solver-trace mining + a beam-width sweep on hard puzzles plausibly puts you at 88k–91k, a ~5% absolute improvement. The Q*-search variant (if it works) is a step-change that could go to ~85k. Beyond that, the curve flattens until you spend serious compute on AlphaZero-style training or a proper neural compressed-PDB.

Good luck.