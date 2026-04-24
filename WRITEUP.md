# Solving the IHES Picture Cube with a Neural Heuristic + Beam Search

*A technical write-up of our approach to the Kaggle CayleyPy IHES SuperCube competition. We reduced total move count from 42,718 (classical baseline) to ~23,250 (-46%) via a small neural distance-heuristic, a wider beam search, and several search-side engineering wins.*

## 1. Problem

The IHES Picture Cube is a 3×3×3 super-cube variant: every sticker is uniquely labelled, so centre orientations matter (a standard 3×3×3 has colour-identical stickers, so a centre twisted by 90° looks solved; here it doesn't).

Concretely:

- **State**: a permutation of 72 stickers — 8 corners × 3 stickers + 12 edges × 2 + 6 centres × 4.
- **Generators**: 18 moves, structured as 3 axis families (f, r, d) × 3 layer indices (0, 1, 2) × 2 signs (forward and inverse). Each move has cycle structure `(4,4,4,4,4,4)` — a quarter turn of one layer.
- **Task**: 1003 scrambled states; emit one solving path per puzzle.
- **Metric**: total move count across all 1003 paths. Lower is better.
- **Hardware**: a single RTX 4090 Laptop (16 GB VRAM), plus free Kaggle GPU kernels and a GCP L4 VM for overnight jobs.

The classical baseline — a Kociemba two-phase solver adapted to the picture cube — gives 42,718 moves. On 3×3×3 God's number is 20, so the natural lower bound for an "optimal on every puzzle" solver is around 20K total. That's the target range we aimed at.

## 2. High-level approach

A distance-heuristic network plus beam search. The core loop, per puzzle:

1. Learn `V(s) ≈ d(s, solved)` from random-walk samples with `d = walk_depth`.
2. At inference time, run greedy beam search: keep the top-B children by `V(apply(s, a))` at each step.
3. Post-process (pair-cancel + BFS-d5 window replacement) then verify.

Nothing in the high level is novel on its own. The improvements came from making each piece of this loop better — the heuristic, the search, the state representation, and the ensemble.

## 3. The neural V-heuristic

### 3.1 Architecture

The winning architecture is surprisingly small — `ResMLPDistance` with:

- Input: 72 stickers → embedding(num_classes=72, embed_dim=16), flatten to 1152 dims.
- Trunk: `Linear(1152 → 1024) → ResBlock(1024 → 256) → Linear(256 → 1)`. ~1.6 M params.

We tried bigger: `[5000, 1000] × 4` with 18 M params and `[2048, 1024] × 8` with ~23 M params. Both regressed. The small arch trained faster, generalised better, and made wider beams tractable because the inference chunk size could go up.

Key detail: the embedding input (`nn.Embedding(72, 16)`) beat one-hot (1152 vs 5184 input dims) by a small but real margin in accuracy and by ~4× in memory. That matters because beam search's effective width is memory-limited.

### 3.2 Training recipe ("fast recipe")

The training recipe that gave us the winning baseline:

- Loss: MSE on walk-depth labels. We experimented with L1 and Huber; MSE stayed.
- Optimiser: AdamW (fused), lr 1e-3, cosine decay to 1e-5.
- Batch size: 10,000–16,000 (small on-GPU batches + Python-side mini-epochs of 1 M samples).
- Data: random walks from solved with `k_max = 26`, `n_back = 1` (don't pick the inverse of the previous move — this is critical; see below).
- Engineering: `bf16` + `torch.compile(mode="reduce-overhead")` + fused AdamW. 3× training throughput vs plain fp32.
- Epoch count: 4000–8000 on the small arch. Diminishing returns past ~3000.

### 3.3 Bellman refinement (E6)

Walk-depth labels are an *upper bound* on `d(s, solved)` — the walk might be non-optimal. The model learns a ceiling rather than the true distance. Bellman refinement addresses this directly:

```
target(s) = 1 + min_a target_net(apply(s, a))
```

where `target_net` is a frozen periodically-refreshed copy of the training network. Warm-start from the fully-trained E5 checkpoint, refine for 500 epochs with `target_net` refreshed every 10 epochs. Bellman-MSE drops from 2.21 (epoch 0) to 0.17 (epoch 500).

Concretely, this is done once per epoch:

1. Sample a batch of states `s`.
2. Compute children `s'_a = apply(s, a)` for each of the 18 actions.
3. Forward target_net on all `18 B` children; take the min per row.
4. Loss: `MSE(train_net(s), 1 + min_a target_net(s'_a))`.

On hard puzzles (pid 400-429 at beam 16K) E6 solves **29/30** vs E5's 27/30 — Bellman cracks two puzzles E5 can't, at the same average path length on the common set. This was our single biggest accuracy win after the architecture change.

## 4. Search-side engineering

This is where most of the later wins came from. The neural heuristic was already close to its quality ceiling by early phase-2; search-side compounded with every new model.

### 4.1 Khoruzhii beam searcher port

The `cayleypy` reference library ships a `BeamSearchAlgorithm` with two modes:

- `simple`: standard beam, but advanced features like history tracking are limited.
- `advanced`: history-tracking beam, but in the current library version it returns `path=None` even on success — broken for our use.

We ported Rokicki-collaborator M. Khoruzhii's beam searcher (published alongside his NeurIPS 2025 spotlight paper on neural beam search for Rubik's cubes) into `src/cayley/khoruzhii_search.py`. The key differences:

- Explicit priority-queue-free top-K via `torch.topk` on a flat score tensor.
- Beam buffers live in a single contiguous tensor we reuse across steps.
- Inference chunk size is a first-class knob, separate from beam width, so we can cap VRAM during the 18× per-child expansion.

Empirically this got us from a ~8K effective beam cap in the library to 65K on the same GPU — an ~8× beam width improvement before any further memory tricks. Beam 65K with the small arch + khoruzhii searcher was the biggest single submission drop (26K → 25K class).

### 4.2 int8 state encoding

Beam-search memory is dominated by the beam-state tensor: `beam_width × state_size = 65,536 × 72 = 4.7 M ints`. We were storing this as int64 (8 B per sticker), using 38 MB per beam layer. At beam 1 M that's 576 MB — we'd OOM at ~2× further expansion.

Switch: `state_dtype = torch.int8`. Stickers are 0–71, fit in int8 trivially. Generator gather indices stay int64 for speed (tiny; 18 × 72 × 8 B = 10 KB). Hashing casts to int64 internally when needed.

Results:
- Beam 65K peak VRAM: 1.56 GB → **0.26 GB (6×)**.
- Wall-time overhead from casting: +22% on the small beam.
- New beam cap on the same 16 GB GPU: **2 M** (limited by hashing + children expansion, not the beam tensor itself).
- Paths identical to int64 (verified CPU + GPU).

### 4.3 Q-function distillation (QE6)

The V-heuristic is evaluated 18 times per beam step — once per child. That's fine at beam 65K, but at beam 1 M it dominates wall time. A standard trick from the Vlad Kuznetsov megaminx work is to distill a Q-head:

- Train a network with 18 outputs, `Q(s, a) ≈ V(apply(s, a))`.
- At inference, one forward pass gives all 18 neighbor scores — no child expansion on the network side.
- Children are only materialised for the top-B among the `18 B` scores.

Distillation loss is just MSE between `Q(s)[a]` and `V(apply(s, a))` over random walks; 1000 epochs got us to `distill_MSE ≈ 0.21`, which was enough. We added an `output_dim` parameter to `ResMLPDistance` (single-head vs 18-head) and a Q-path to `KhoruzhiiSolver` gated by `use_q_function=True`.

Speedup: roughly **8×** at beam 524K, matching Vlad's report on megaminx. This directly bought us beam 1 M → 4 M in the same wall time, which is where the final score reductions came from.

### 4.4 The ensemble loop

With multiple solvers (E3, E5, E6, E9 symmetry-aug, E6+NISS, QE6, beam-524K, beam-1M variants, and a Kociemba fallback), we min-merge per puzzle:

```
final[pid] = min(len(c[pid]) for c in candidates)
```

Then a post-processing pass: pair cancellation (adjacent inverses collapse), followed by a BFS-depth-5 window replacement (for each 5-move window, if the BFS table has a shorter sequence with the same net permutation, swap it in).

Individual solvers contribute marginally in isolation — the directional diversity is what matters. NISS (solve σ⁻¹ and invert the resulting path back) contributed **-432 moves** purely in the ensemble even though its solo score is near-par with forward.

## 5. NISS — a free move from reversibility

Worth a subsection. Given a scramble `σ`, our beam searches for `σ → solved` in the generator semigroup. But the inverse problem — `σ⁻¹ → solved` — has the same optimal length, and if we solve it then *reverse and invert* the resulting path, we get a solve for `σ`. The beam searches are directionally asymmetric (hash seeds, tie-breaking, which child the greedy step picks) so this gives *different paths at the same expected quality*.

Implementation is three lines — `invert_state` (computes `σ⁻¹` from a permutation) plus `invert_path` (reverses a move sequence and inverses each move). Runs cost the same compute as a forward solve. Contribution to submission: on ~430 puzzles the NISS path is strictly shorter than anything the forward searches found. -432 moves for no model work.

In the FMC community this has been standard since ~2009; we only learned about it mid-competition.

## 6. Tail re-solve at progressively wider beams

Observation: the score is dominated by the tail — the ~200 hardest puzzles account for more than half the total moves. Running a 10× wider beam on *every* puzzle is wasteful; most puzzles are already solved near-optimally at beam 65K.

Strategy: targeted re-solve. For each submission, pick the puzzles with path length ≥ N (for some N), re-run at a wider beam, keep the shorter. We rode this down:

| Threshold | Beam | Puzzles | Improved | Saved |
|---|---|---|---|---|
| len ≥ 27 | 131K  | 40  | 24 | -50 |
| len ≥ 26 | 131K  | 148 | 33 | -74 |
| len = 25 | 131K  | 288 | 37 | -76 |
| len ≥ 26 | 524K  | 108 | 74 | -160 |
| len ≥ 25 | 4 M   | 189 | 32 | -64 |

Hit rate is a good diagnostic of how close we are to ceiling. At beam 131K the rate drops from 60% at len ≥ 27 to 13% at len = 25. At beam 524K we pushed the rate back up to 69% on the same puzzles — strong evidence that many puzzles at length 25–30 genuinely needed a wider beam to find a shorter path, rather than being at their true optimum.

## 7. What we tried that didn't work (and what that taught us)

Quick catalogue, for the record:

- **Bigger models.** Every larger variant regressed. Not a clean result — possibly recipe-dependent, but consistent enough across 5 attempts that we stopped.
- **`n_back = 40`** (exclude all recent moves from random-walk sampling). Known-good in one community chat, but regressed MSE 14.4 → 15.84 here. Possibly good only in combination with other changes we never isolated.
- **k_max > 30** in walks. Picture-cube effective diameter is ~26; longer walks didn't add label signal.
- **Curriculum weighting** (`1/k` per sample). Hurt accuracy on hard puzzles; easy-sample-heavy loss trained a near-goal specialist.
- **1/k sampling + BFS-label mixing together (E7 bundle).** Bundled regressions are ambiguous; when we isolated curriculum (E11), the data supplement was neutral, so the curriculum was the culprit.
- **Piece-decomposition input encoding** (E8). Input to the net was `(corner_ids, corner_oris, edge_ids, edge_oris, centre_ids, centre_oris)` — 52 small-vocab features with the right invariants built in. Looked like a clean structural prior. Trained to worse accuracy than the flat 72-sticker embedding, 2.5× slower inference.
- **24× rotational symmetry augmentation** (E9). Random whole-cube rotation per sample. Converged similarly, but didn't beat E6 solo; +24 moves in ensemble.
- **Commutator-library window replacement** for post-processing. Depth-4 + depth-6 commutators (4.5K new 6-move patterns). Zero hits on actual solution paths — beam paths aren't commutator-structured the way hand-written FMC solutions are.
- **BFS-d6 window replacement.** 11 M states, 1.77 GB table. Zero gain over d5. The d5 → d6 marginal windows are too rare in beam output.
- **twips / twsearch integration.** Went far enough to build a group-compatible KPuzzle JSON (`data/picture_cube_decomp.kpuzzle.json`) — verified by round-trip + 50× random 20-move sequence match. Twips solves 1-move and 3-move scrambles instantly but can't reach 15-move depth without a pruning table. Getting a pattern database into twips is doable but days of work for uncertain payoff given how close to optimal the beam outputs already are.

The common thread: most of the **model-side** ideas give small-to-negative deltas once the small arch + Bellman + n_back=1 baseline is in place. The **search-side** ideas (Khoruzhii, int8, Q-function, wider beam tail re-solves) consistently paid off and compounded.

## 8. Results

Submission progression, summarised:

| Date | Score | Δ | What landed |
|---|---|---|---|
| 04-16 | 42,718 |   | Kociemba classical baseline |
| 04-17 | 30,770 | -11,948 | First neural V + beam 4K + Kociemba fallback |
| 04-17 | 30,120 | -650 | Embedding encoder |
| 04-18 | 28,224 | -1,486 | Wider beam + BFS-d5 PP |
| 04-18 | 27,106 | -1,118 | Ensemble + MITM |
| 04-18 | 24,998 | **-2,108** | Khoruzhii searcher + small arch + beam 65K |
| 04-19 | 24,068 | -550 | E6 Bellman + NISS |
| 04-20 | 23,868 | -200 | Targeted beam-131K tail re-solves |
| 04-20 | 23,672 | -186 | int8 encoding + beam-524K tail re-solve |
| 04-22 | 23,322 | -350 | Beam-1M full solve + Q-function |
| 04-22 | **23,224** | -98 | + Beam-4M full solve (GCP) + beam-4M tail re-solve, min-merged |

Final gap to the leader: ~1,400 moves, roughly 6%. Much of that gap is closed by specialised FMC techniques (commutator-heavy skeletons, reduction methods) rather than more ML, which is where we ran out of tractable ideas.

## 9. Lessons

Three that I'll carry forward:

1. **Small model + big search** beats big model + small search on discrete-state problems where the search has a hard resource ceiling. On this cube, every minute of engineering on the search side (int8, Q-head, port a better beam) outpaced every minute on the model side.

2. **Diminishing returns are visible at the hit-rate level long before they're visible in the total score.** Tracking "what fraction of re-solves produced an improvement" at each beam width was the single most useful diagnostic. When the rate drops below ~15% at a given N, that's the signal to widen the beam rather than training another seed.

3. **Directional ensembles are free money.** NISS is the clearest example — same compute, same model, different beam trajectory, -432 moves in the min-merge. Anything that produces genuinely decorrelated paths (inverses, whole-cube rotations, seed RNG) is worth trying before training more models.

## 10. Code

Repo: `cayley/` (private for now).

Key files:

- `src/cayley/model.py` — `ResMLPDistance` with optional Q-head via `output_dim`.
- `src/cayley/bellman.py` + `scripts/05_bellman_refine.py` — E6 Bellman training loop.
- `src/cayley/khoruzhii_search.py` — beam searcher port; `state_dtype=torch.int8`, `use_q_function=True` flags.
- `src/cayley/puzzle.py` — `invert_state`, `invert_path`, and the cube state model.
- `scripts/06_niss_solve.py` — NISS runner.
- `scripts/07_target_wider_beam.py` — targeted tail re-solve.
- `scripts/02_solve.py` — main solver driver with auto-detection of V vs Q heads from the checkpoint.
- `.claude/commands/{submit,verify-and-submit}.md` — the two submission slash commands.

Dependencies: PyTorch 2.11.0+cu128, Python 3.14, cayleypy 0.1.0.
