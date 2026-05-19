I think the useful abstraction is:

```text
current method = state sampler + target maker + model + search policy
              = n_back=1 random walks + Bellman target + ResMLP value + beam search
```

The biggest opportunity is probably not “better Bellman” alone. It is changing the **distribution of states and decisions** the model sees.

**Random-Walk Variations**
1. **Mixture of `n_back` values**
   Keep `n_back=1`, but mix in `n_back=4/10/20/40`. Long tabu walks push outward; short tabu walks preserve natural local geometry. I would not switch entirely to `n_back=40`; I’d mix.

2. **Depth curriculum**
   Instead of uniform prefixes `1..80`, sample depth bands intentionally:
   ```text
   20% k<=20, 30% 20<k<=60, 30% 60<k<=120, 20% hard/frontier states
   ```
   Current data overrepresents easy prefixes because every walk contributes all earlier states.

3. **Self-avoiding / geodesic-ish walks**
   Track visited hashes within each walk and avoid revisits. This borrows from self-avoiding walks on Cayley graphs, where avoiding words that reduce to identity is a core object of study. It is more expensive, but maybe useful for smaller offline datasets.

4. **Flat-histogram random walks**
   From statistical physics: Wang-Landau / flat-histogram sampling tries to sample rare energy levels uniformly. Here “energy” could be predicted distance, PDB distance, mismatch count, or Bellman value. This directly attacks your “random walks rarely reach rare deep states” point.

5. **Move-distribution random walks**
   Not all generators need equal probability. Try:
   ```text
   uniform moves
   inverse-frequency moves
   learned-policy moves
   commutator/conjugate-biased moves
   subgroup/coset-biased moves
   ```
   This is where group theory matters: good scramblers are not necessarily good training samplers.

6. **Macro random walks**
   Randomly walk in a macro action set: primitive moves plus selected 2-5 move macros, commutators, conjugates. Korf’s old macro-operator work is directly relevant here.

7. **Random walks from many anchors**
   Instead of only solved → outward:
   ```text
   solved shell → outward
   BFS-d6 states → outward
   beam-failure states → outward
   solved trajectories → perturb outward
   ```
   This makes the training distribution less “radial from solved.”

**Alternatives To Random Walks**
1. **Beam-frontier replay**
   Collect states actually seen by beam search, especially failed beams and near-miss states. This may be more valuable than another billion random-walk states.

2. **Solver trajectory replay**
   Whenever any solver finds a path, every suffix gives a label:
   ```text
   s_t has solution length T-t
   ```
   These are not necessarily optimal labels, but they are search-relevant.

3. **Exact BFS shell anchoring**
   You already have code hooks for BFS-d6. I would use exact near-goal shells aggressively. Pattern databases did this classically for Rubik’s Cube: exact distances for abstractions, then search.

4. **Reverse curriculum**
   Start near solved, expand the start distribution only when the model/search can solve the current band. This is basically the logic behind DeepCubeA / Autodidactic Iteration and reverse-curriculum RL.

5. **Hindsight relabeling**
   From sparse-reward RL: failed trajectories still contain achieved states. For puzzles, any generated path gives exact path-distance-to-any-visited-state. That suggests training goal-conditioned or shell-conditioned models.

6. **Coset/subgroup targets**
   Borrow from Kociemba/two-phase solving: train “distance to useful subgroup/coset shell,” not only distance to solved. For Megaminx-like problems, a learned phase-1 target may be far easier than full solving.

**Bellman Variations**
1. **Double Bellman**
   Current target uses `min_a target_model(child)`. A min over noisy estimates has optimistic underestimation bias. Use model A to choose the child and model B to evaluate it, analogous to Double Q-learning.

2. **Soft Bellman**
   Replace hard min with temperature log-sum-exp:
   ```text
   y = 1 + softmin_a V(child)
   ```
   Start soft, anneal toward hard min. This reduces target collapse from one accidentally low child.

3. **n-step Bellman**
   Instead of one-step:
   ```text
   y = d + min_over_paths_len_d V(leaf)
   ```
   for `d=2..4`. More expensive, but targets align better with beam search.

4. **Prioritized Bellman**
   Sample states with large Bellman residual:
   ```text
   |V(s) - (1 + min V(child))|
   ```
   This is the prioritized sweeping idea: spend updates where dynamic-programming inconsistency is high.

5. **Bounded Bellman**
   Clamp with both upper and lower signals:
   ```text
   lower_bound(s) <= target <= random_walk_depth_or_solution_length
   ```
   Lower bound can come from BFS/PDB abstractions, misplaced-piece features, or cheap invariant distances.

6. **Ranking Bellman**
   Beam search mostly needs correct ordering, not calibrated distance. Add pairwise/listwise loss:
   ```text
   best child should rank above siblings
   successful trajectory states should rank above alternatives
   ```

7. **Q-Bellman**
   Train `Q(s,a)` directly:
   ```text
   Q(s,a) = 1 + V(a(s))
   ```
   Then one forward pass gives all move scores. This connects to Q* / DeepCubeAQ and could massively reduce child evaluation cost.

**Alternatives To Bellman**
1. **Expert iteration**
   Use current beam solver as teacher. Train policy/value from successful and failed searches, then re-solve harder cases.

2. **Policy-guided heuristic search**
   Train both:
   ```text
   V(s): how far?
   pi(a|s): which move?
   ```
   Then use Levin Tree Search or Policy-Guided Heuristic Search instead of pure value beam.

3. **Pattern database + neural residual**
   Classical PDBs give exact abstract lower bounds. Neural net predicts residual/ranking. This is less elegant, but very robust.

4. **MCTS / PUCT**
   More expensive than beam, but may be useful for hard instances or as a teacher to create better datasets.

5. **Cross-entropy / evolutionary macro discovery**
   Learn distributions over move sequences that solve hard states; distill the resulting trajectories into policy/value training.

**Alternatives To The Whole Process**
My ranked bets:

1. **Q-shortlister + teacher rerank**
   Train a cheap `Q(s,a)` to shortlist moves/states; use the expensive current value model only on the shortlist. Highest speed upside.

2. **Bellman on beam-frontier replay**
   Keep your model, but change the training states. This is probably the simplest high-impact quality experiment.

3. **Double + soft + prioritized Bellman**
   Same pipeline, better targets, less value collapse risk.

4. **Bidirectional shell solver**
   Train/search toward a known BFS shell or subgroup instead of solved directly. Meet-in-the-middle search theory says this can cut depth dramatically.

5. **Macro-action hierarchy**
   Search over primitive moves plus learned macros. This reduces effective depth and may improve solve length.

6. **Symmetry-aware model**
   Recent Rubik’s Cube geometric deep learning work argues that explicit cube symmetries improve generalization. For Megaminx, symmetry-aware architecture or augmentation could make every sample count more.

7. **Population annealing beam**
   Treat beam search like a physics population: resample good states, mutate via random walks/macros, anneal selection pressure. This could escape heuristic traps better than deterministic top-k beam.

**Most Concrete Next Experiment**
I would run this sequence:

```text
A. collect beam-frontier/failure states
B. fine-tune Bellman on:
   50% n_back=1 random walks
   20% n_back=10/40 walks
   20% beam-frontier states
   10% exact BFS shell states
C. use Double Bellman or softmin targets
D. evaluate solve rate, path length, and top-k child ranking
```

That is the best “not too wild, likely useful” step.

Sources I leaned on: [DeepCubeA](https://www.nature.com/articles/s42256-019-0070-z), [Autodidactic Iteration](https://openreview.net/forum?id=Hyfn2jCcKm), [large-cube diffusion heuristic + beam search](https://arxiv.org/abs/2502.13266), [Q* / DeepCubeAQ](https://arxiv.org/abs/2102.04518), [Korf pattern databases](https://aaai.org/Papers/AAAI/1997/AAAI97-109.pdf), [learning heuristics by bootstrapping](https://www.sciencedirect.com/science/article/pii/S0004370211000877), [Double Q-learning](https://papers.neurips.cc/paper/3964-double-q-learning), [prioritized sweeping](https://publications.ri.cmu.edu/prioritized-sweeping-reinforcement-learning-with-less-data-and-less-real-time), [self-avoiding walks on Cayley graphs](https://www.combinatorics.org/ojs/index.php/eljc/article/view/v31i4p24), [flat-histogram Monte Carlo](https://academic.oup.com/ptps/article/doi/10.1143/PTPS.138.454/1879216), [population annealing](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.92.063307), [bidirectional NBS](https://www.ijcai.org/proceedings/2017/69), [MM bidirectional search](https://www.sciencedirect.com/science/article/pii/S0004370217300905), [Levin Tree Search](https://deepmind.google/research/publications/21589/), and [Rubik’s Cube symmetry-aware neural model](https://pubmed.ncbi.nlm.nih.gov/40839500/).