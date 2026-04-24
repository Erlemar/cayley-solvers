# New Ideas Synthesis — CayleyPy IHES Picture Cube

## Context
Current best: **23,858** (avg 23.8). Rokicki #1: **21,840** (avg 21.8). Gap: **2,018 moves (8.5%)**.
Synthesized from 5 parallel research agents: cross-competition transfer, academic papers, creative approaches, gap analysis, and competition discussions.

---

## ROOT CAUSE ANALYSIS: Where Do the 2 Extra Moves Come From?

The gap decomposes into 4 sources (from gap analysis agent):

| Source | Estimated contribution | Root cause |
|--------|----------------------|------------|
| **A. Heuristic ranking errors** | 0.8-1.2 moves/puzzle | Model MSE ~8.5 → RMS error ~2.9 → frequent mis-ranking of neighbors differing by 1 move |
| **B. Beam pruning of correct paths** | 0.3-0.6 moves/puzzle | Optimal path passes through states the heuristic rates poorly ("valley problem") |
| **C. No backtracking** | 0.3-0.5 moves/puzzle | Beam commits to bad intermediate states; can't undo |
| **D. Post-processing ceiling** | 0.1-0.3 moves/puzzle | Suboptimality is distributed, not concentrated in short windows |

**Key insight**: ~50% of the gap is heuristic quality (fix with better training), ~50% is search limitations (fix with search diversity and wider effective beam).

---

## NEW IDEAS (Not in IDEAS_CATALOG.md)

### TIER 1 — Highest expected impact, 1-3 days each

#### N1. Bellman Consistency as Auxiliary Training Loss ⭐⭐⭐
**Source**: Creative approaches agent, extending existing C2 (Bellman refinement)

Instead of applying Bellman refinement as post-hoc fine-tuning, add it as a **simultaneous auxiliary loss** during main training:
```
L_total = L_MSE(f(s), d_walk) + λ * L_bellman(f(s), 1 + min_a f(apply(s, a)))
```
The min requires 18 forward passes per sample but can be batched. This enforces `f(s) = 1 + min_a f(apply(s,a))` continuously, not just in a few final epochs.

**Why it's different from C2**: C2 applies Bellman as post-training refinement. This integrates it from epoch 1, preventing the model from ever learning inconsistent distance estimates. The consistency constraint propagates accurate values outward from solved.

- **Effort**: 1-2 days (modify training loop to compute 18 neighbor values per batch)
- **Expected gain**: -200 to -500 moves (directly attacks Source A)
- **Complementary**: Yes — improves the model that everything else uses

---

#### N2. Test-Time Bellman Refinement (Per-Puzzle Fine-Tuning) ⭐⭐⭐
**Source**: Creative approaches agent

For each puzzle's scramble S, do 100-500 gradient steps of **self-supervised Bellman consistency** on states near S:
1. Generate random walks from S (length 5-10)
2. For each state S' reached, enforce: `f(S') ≈ 1 + min_a f(apply(S', a))`
3. No ground truth needed — purely self-supervised
4. Fine-tune for 30s, then run beam search with the locally-improved model

**Why it works**: The global model has MSE ~8.5 averaged over all states. For any specific puzzle's neighborhood, local errors may be much larger. TTT corrects these local biases without ground truth.

- **Effort**: 1-2 days
- **Expected gain**: -50 to -150 moves (targets the 100 hardest puzzles)
- **Complementary**: Yes — stacks on top of any model

---

#### N3. NRPA (Nested Rollout Policy Adaptation) ⭐⭐
**Source**: Creative approaches agent

A fundamentally different search algorithm from beam search. NRPA recursively improves a policy by learning from the best sequence found so far. At level 0: biased random rollout. At level k: run level-(k-1) multiple times, update policy after each.

**Why it helps**: NRPA explores DIVERSE paths (unlike beam search which clusters around the heuristic's preferred direction). For puzzles where beam search consistently finds 26+ moves, NRPA's different exploration might find 23-24 move paths.

- **Effort**: 1-2 days to implement, run on top-100 hardest puzzles
- **Expected gain**: -50 to -200 moves
- **Complementary**: Fully orthogonal — min-merge with beam search solutions

---

#### N4. ReduceFactor Graph Shortening (Dijkstra on Solution Graph) ⭐⭐
**Source**: Cross-competition agent (Santa 2023, 1st place)

Build a DAG where nodes = positions in current solution, edges = alternative subsequences. For each window [i,j], compute net permutation and look up shortest known factorization. Use Dijkstra to find globally optimal combination of shortcuts.

**Why it's different from existing E1-E5**: Current post-processing does greedy local optimization. ReduceFactor considers ALL non-overlapping shortcut combinations simultaneously via shortest-path algorithm. This finds globally optimal post-processing.

- **Effort**: 2-3 days
- **Expected gain**: -100 to -500 moves (Santa 2023 saved 9,000+ on large cubes)
- **Complementary**: Pure post-processing, stacks with everything

---

#### N5. Distance-Adaptive Beam Width ⭐⭐
**Source**: Academic research agent (NeurIPS 2025, arxiv 2505.15636)

Instead of fixed beam width, adapt dynamically:
- **Wide beam** in early moves (high uncertainty, many viable paths)
- **Narrow beam** near the goal (heuristic is more accurate, fewer viable paths)
- Width = f(estimated_distance_remaining, heuristic_variance)

**Why it helps**: Fixed beam wastes capacity. At depth 20 (near solved), the heuristic is very accurate and beam 1000 suffices. At depth 5 (far from solved), the heuristic is unreliable and you need beam 100K+. Shifting capacity from endgame to opening gives more effective search.

- **Effort**: 1 day (modify beam search loop to compute per-step width)
- **Expected gain**: -50 to -200 moves (equivalent to wider beam at no memory cost)
- **Complementary**: Yes — direct improvement to beam search

---

#### N6. CEA Loss for Near-Admissible Heuristics ⭐⭐
**Source**: Academic research agent (ICLR 2026, arxiv 2509.22626)

Cross-Entropy Admissibility loss trains heuristics that **never overestimate** distance. Standard MSE produces heuristics that sometimes overestimate (rating a state as farther than it is), causing beam search to deprioritize correct paths.

```
L_CEA = L_CE(f(s), d_true) + λ * max(0, f(s) - d_true)²
```

**Why it helps**: An admissible heuristic guarantees that A* finds optimal solutions. For beam search, near-admissibility means the correct next state is almost never pruned (it's never rated as "too far"), only occasionally tied.

- **Effort**: 2-3 days (modify loss function, may need training adjustments)
- **Expected gain**: -100 to -300 moves (directly attacks Source A)
- **Complementary**: Yes — better model for beam search

---

### TIER 2 — Reliable incremental gains

#### N7. Arbitrary Insertion Timing for Post-Processing
**Source**: Cross-competition agent (Santa 2023, 1st place)

When inserting correction subsequences (e.g., from BFS window replacement), try ALL positions in the existing solution, not just the end. Different positions create different move cancellation opportunities at the insertion boundary.

- **Effort**: 1 day
- **Expected gain**: -50 to -200 moves
- **Complementary**: Pure post-processing

---

#### N8. Online Heuristic Adaptation During Search
**Source**: Cross-competition agent (Santa 2024, 1st place)

During beam search for a specific puzzle, collect (state, move, improvement) tuples and train a lightweight surrogate model to predict which moves are worth evaluating. As data accumulates, the surrogate gets better at filtering.

- **Effort**: 2-3 days
- **Expected gain**: -100 to -300 moves (equivalent to wider beam)
- **Complementary**: Yes — enhances beam search at runtime

---

#### N9. Elastic Pulse / Move Deletion + Repair
**Source**: Cross-competition agent (Santa 2025)

For each move in the solution, try deleting it and measuring how far the resulting state is from solved. If you can repair in < 1 move (using BFS table), you saved a move. Even if most deletions fail, the successes are pure improvement.

- **Effort**: 1 day
- **Expected gain**: -30 to -100 moves
- **Complementary**: Pure post-processing

---

#### N10. Per-Puzzle Compute Budget Allocation
**Source**: Cross-competition agent (Santa 2025)

Sort puzzles by (current_moves - estimated_optimal). Spend 10× more beam search time on the top-50 puzzles with largest gaps. A puzzle at 28 moves (where 22 is achievable) has 6 moves to gain; a puzzle at 22 (where 21 is achievable) has only 1.

- **Effort**: 30 minutes
- **Expected gain**: -50 to -200 moves (redistributes existing compute)
- **Complementary**: Meta-strategy for any solver

---

#### N11. AlphaCube Policy Network as Move-Ordering Prior
**Source**: Academic research agent (EfficientCube/AlphaCube, TMLR 2025)

Train a policy network (predicts next move, not distance) alongside the value network. During beam search, use the policy to **order** which children to evaluate first. The value network scores them, but the policy determines evaluation order, saving time on unlikely moves.

- **Effort**: 2-3 days (train separate network, modify beam search)
- **Expected gain**: Equivalent to 2-4× beam width at same wall-time
- **Complementary**: Yes — enhances beam search

---

### TIER 3 — Experimental / Research

#### N12. Recursive Immediate Restart in Post-Processing
**Source**: Cross-competition agent (Santa 2024, 2nd place)

When window optimization finds an improvement at position i, restart scanning from max(0, i-window) instead of continuing from i+1. Improvements cluster spatially.

- **Effort**: 30 minutes
- **Expected gain**: -10 to -50 moves

#### N13. Double Bridge Perturbation + Repair
**Source**: Cross-competition agent (Santa 2024)

Cut solution at 4 random points, reconnect segments differently, apply short correction. If correction < moves saved by cancellation, net improvement.

- **Effort**: 1 day
- **Expected gain**: -20 to -100 moves (speculative)

#### N14. Solution Recombination Across Seeds
**Source**: Cross-competition agent

For each puzzle with multiple solutions (from different seeds/models), check if any pair shares an intermediate state. If so, splice the shorter prefix from one with the shorter suffix from the other.

- **Effort**: 1 day
- **Expected gain**: Very low (probability of shared state ~10^-18)

---

## HOW THESE IDEAS INTERACT WITH THE EXISTING CATALOG

| New Idea | Attacks Source | Interacts with | Independent of |
|----------|--------------|----------------|----------------|
| N1 Bellman aux loss | A (heuristic) | D8 (ranking loss) — try both | A11, A12, F2 |
| N2 Test-time refinement | A (local heuristic) | All models — applied per puzzle | Everything |
| N3 NRPA | B,C (search diversity) | H1-H6 (ensemble) — adds new solver | Model training |
| N4 ReduceFactor | D (post-processing) | E1-E7 — replaces greedy with global | Search/training |
| N5 Adaptive beam | B (beam pruning) | A11,A12 — stacks with wider beam | Model training |
| N6 CEA loss | A (admissibility) | D8 — alternative loss direction | A11, A12, F2 |
| N7 Insertion timing | D (post-processing) | E7 (tail re-solve) | Search/training |
| N8 Online adaptation | A,B (runtime heuristic) | A11 (Q-func) — complementary | Model training |

---

## RECOMMENDED EXECUTION ORDER

### Day 1 (quick wins, -150 to -500 moves)
1. **N10** Per-puzzle budget allocation (30 min)
2. **N5** Distance-adaptive beam width (1 day)
3. **N12** Recursive restart in PP (30 min)

### Day 2-3 (training improvements, -200 to -800 moves)
4. **N1** Bellman consistency auxiliary loss (1-2 days)
5. **N6** CEA loss variant (alternative to N1 — try both, keep better)

### Day 3-5 (new search algorithms, -100 to -400 moves)
6. **N3** NRPA on top-100 hardest puzzles (1-2 days)
7. **N4** ReduceFactor graph shortening (2-3 days)

### Day 5-7 (inference-time tricks, -50 to -300 moves)
8. **N2** Test-time Bellman refinement on top-50 hardest (1-2 days)
9. **N8** Online heuristic adaptation (2-3 days)

### Cumulative expected impact: -700 to -2000 moves
This could bring the total from 23,858 to approximately **22,000-23,100** — closing 50-100% of the remaining gap to Rokicki.

---

## IDEAS THAT WERE CONSIDERED AND REJECTED

| Idea | Why rejected |
|------|-------------|
| RL policy (PPO/SAC) as solver | Worse than heuristic+search; A11 Q-function captures the useful part |
| Transformer seq2seq (scramble→solution) | Overfits to 1003 puzzles; transformer-as-value is B3, weeks of work |
| Diffusion models for path generation | Research-paper-level, months not days |
| Genetic algorithm on move sequences | Crossover useless in 10^22-state group; degenerates to local search |
| True bidirectional with frontier intersection | 10^22 states → need 10^11 frontier per direction, infeasible |
| Solution interpolation between similar scrambles | Non-abelian group prevents meaningful interpolation |
| Contrastive learning for embeddings | Scalar distance is already optimal 1D embedding for beam search |
| Full hierarchical RL (6-level) | 1-2 weeks, high risk |
