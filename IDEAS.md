# Experiment ideas for IHES Picture Cube

Living backlog. Untried items ranked by **expected payoff per hour of work** given what
we've already learned. Tick items off as we run them.

**Current best**: `24,068` (E6 Bellman + NISS-E6 + full ensemble + BFS-d5 pp, submitted 2026-04-19).
**Gap to Rokicki (21,840)**: 2,228 moves (9.3%).

---

## HIGHEST priority untried

NEW (from speedsolving forum mining, 2026-04-18):

0a. [x] **NISS (inverse scramble search)** — DONE 2026-04-19. Solo run was 24,948 (slightly worse than E6 forward 24,932), but combined with the forward ensemble it added **-432 moves** (ensemble dropped 24,502 → 24,070). Implementation: `invert_state` + `invert_path` in `src/cayley/puzzle.py`, standalone `scripts/06_niss_solve.py`. Confirmed directional-anisotropy gain despite similar solo performance.

0b. [ ] **Six-axis ensemble search** — ★★★ extension of NISS. Apply each of the 6 possible whole-cube rotations to the scrambled state before solving, translate the path back after, keep the shortest. Requires deriving the 6 whole-cube rotation permutations (the 3 face-axis rotations + identity + 2 non-trivial compositions). Same implementation pain as 24× symmetry aug (#3) — could ride off that work. **Expected -1 to -3 moves per scramble on top of NISS → 1000–3000 moves.**

0c. [x] **Insertion-finder post-processing with commutator library** — DONE 2026-04-19 as a **confirmed zero-gain**. Built `src/cayley/commutator.py` with 4K unique depth-4 + depth-6 perms; also built `data/bfs_table_d6.pkl` (11M states, 1.77 GB). Both give 0 additional moves over BFS-d5 PP on both raw beam outputs and already-PP'd submissions. Beam-generated paths don't contain net-permutations that land in the d6-exclusive perm set. Window replacement has hit a ceiling — do not re-explore unless the base solver changes.

0d. [ ] **twsearch (Rokicki's tool) with piece decomposition** — ★★ Rokicki's actual 21,840 score likely uses `cubing/twsearch`. We have the `.tws` file already scaffolded from prior v13 work (`kaggle_research/cayleypy-ihes-cube/picture_cube_pieces.tws`). State space reduces from 10^103 flat to 10^22 decomposed. Provides IDA* with pattern-database pruning — admissible, often optimal. Might close most of the gap by itself. High effort (~week) to wire up, but parity with the leader.

0e. [ ] **Kociemba sym-coordinate pruning table** — ★★ admissible heuristic from Kociemba's Huge Optimal Solver. 3.5B-entry table at depth 13, reduced via `FlipUDSlice` sym-coordinate (64,430 classes). For picture cube, add a center-orientation coordinate (243 classes). Use as a **tighter alternative to our ML heuristic**, or as a re-ranker/tail-solver where beam fails. Substantial effort (~week+), but the Kociemba Python reference is reusable.

---

Pre-existing HIGH priority untried:

1. [x] **Bellman-style label refinement.** DONE 2026-04-19. 500-epoch refinement of E5 warm-start with target `y = 1 + min_a target_net(apply(s, a))`. Final Bellman-MSE 0.17. E6 single-solve = 24,932 at beam 65K (E5 was 24,974, -42 moves). Solved 998/1003 vs E5's 997. Combined with other candidates it anchored the 24,068 submission. Code: `src/cayley/bellman.py`, `configs/e6_bellman.yaml`, `scripts/05_bellman_refine.py`. Checkpoint: `models/e6/epoch_0499.pt`.
2. [ ] **Multi-model ensemble (khoruzhii-style)**: train 5-10 models of the E3 architecture with different seeds, keep shortest per puzzle. CayleyPy paper goes 90% → 98.4% optimal. We've seen 3-5% gain per ensemble step already; 10-model might push into single-digit-% region. **~3h for 5 models × 2h solve each → ~15h total. Probably overnight.** On GCP L4, budget ~$8-15 for 3-5 new seeds.
3. [ ] **24× cube symmetry augmentation.** Picture cube's rotational symmetry group has 24 elements. Each training state expands to 24 equivalents with same label. Complex to derive (whole-cube rotations aren't in generator set; center orientations rotate non-trivially), but 24× more training data is a step-change for generalization. **Expected -5-10% loss floor.**
4. [ ] **Wider beam (131K) on E3/E5 only for remaining ~100 puzzles** the ensemble doesn't solve tightly. Adaptive escalation: if the best path we have for a puzzle is longer than avg (say, >35 moves), try beam 131K on it. This targets the tail where move count is high. **Expected -300 to -800 moves for ~1h compute.** L4 24GB VRAM headroom makes 131K+ feasible.
5. [ ] **Kociemba warm-start**: feed Kociemba's solution path to the model and let beam search resume from the end-of-prefix state. For the 7-10 puzzles that fall through to fallback, might find shorter combined paths.

## MEDIUM priority untried

6. [ ] **Non-backtracking depth 2-4 (controlled)**. We only tested n_back=1 and n_back=40 (the latter hurt). The sweet spot might be 2-8. Cheap to test: 3 × 500 epochs × ~3min = 15 min total.
7. [ ] **Deeper BFS lookup table (depth 6)**. 11M states, ~6 GB dict. Would find ~3-8% more window substitutions vs depth 5. Memory-tight but feasible.
8. [ ] **Transformer / attention model**. Chat reported a transformer beat their MLP on n=15. Picture cube has structured facelet positions. ~days of effort, uncertain payoff.
9. [ ] **Weighted A* (`f = w·g + h`)** in the khoruzhii searcher. Currently pure h-greedy. Fork the searcher to add g-weighted cost. Might find shorter paths on drifty states.

## LOW priority / RESEARCH untried

10. [ ] **GNN on Cayley graph**. Structural prior — each facelet has known neighbors via generators. Graph attention. High research cost.
11. [ ] **twsearch with piece decomposition**. Decompose into CORNER 8×3 + EDGE 12×2 + CENTER 6×4 (state space 10^22 vs 10^103 flat). Use IDA* with pattern databases. Rokicki-class tool. ~days to integrate. Potentially closes most of the gap.
12. [ ] **Diversity-weighted beam**. Penalize states similar to ones already in beam. Complex re-implementation.
13. [ ] **Commutator insertion library** (Santa 2023 trick). Find spots where a known commutator shortens solution. Needs a commutator library built; moderate effort.

## Open questions to answer experimentally

- At what model loss does solve rate saturate? Our best single is 24,974 at loss 8.41. Would a model at 7.5 loss add 1-2% or 10%?
- Does Bellman refinement on an already-trained model actually work, or do we need to train from scratch with it?
- Does 131K beam's marginal -0.4 avg vs 65K beam help enough in full-solve to justify 7× compute?
- Could we cheaply refine ONLY the tail of each puzzle (last 10 moves) with a BFS search to solved? Targets the biggest source of non-optimality.

---

## What we LEARNED won't help (do not retry as-is)

- **n_back=40 alone** (v3 diagnostic): MSE 14.4 → 15.84. Walks saturate generator-ban space.
- **Huge model + curriculum** (big_v1, 23.7M params + 1/k weighting): worse than small fast model at same beam.
- **`torch.compile` for beam search inference**: 5.8× slower due to recompile loops on variable shapes.
- **L1 + `n_back=40` + bigger arch bundled** (v2): regressed massively.
- **8000 epochs vs 4000** on E5 vs E3: -100 moves full solve. Diminishing returns past ~3000 ep.
- **k_max=45 walks** (E4): no meaningful improvement over k_max=30.
- **Single- and two-step state-hash shortcut post-processing**: saves 30-80 moves on 30K submissions (tight solutions leave no room).
- **Cayleypy library at beam 2^14+**: OOMs at 25GB. Use khoruzhii searcher instead.
- **Beam 262K** on khoruzhii searcher: same path lengths as beam 131K, 7× slower.
- **Commutator-library window replacement (d4+d6, 5K perms)**: 0 gain over BFS-d5 on real submissions. Library perms don't land in beam-path windows.
- **BFS-d6 table (11M perms, 1.77 GB)**: also 0 gain over BFS-d5. Window-replacement PP has hit a ceiling — further gains require a deeper BFS (d7+, 150M+ states, infeasible) or a fundamentally different PP (e.g., re-solving tails). Keep `data/bfs_table_d6.pkl` around but default to d5 for PP.

---

## Active / shipped pipeline

| Step | Tool | Settings |
|---|---|---|
| Training | `configs/small_e5_long.yaml` | [1024,256]×1-block, 8000ep, batch 10k, K_max=26, bf16+compile+fused AdamW |
| Inference | `--searcher khoruzhii --bf16` | beam 65K, num_steps 50, num_attempts 1 |
| MITM (optional) | `--mitm-depth 6` via cayleypy searcher | 11M BFS from solved, `return_all_hashes=True` |
| Ensemble | `scripts/combine_submissions.py` | min-length per puzzle across all candidates |
| Post-process | `scripts/post_process_submission.py --bfs-table data/bfs_table_d5.pkl` | pair-cancel + 1/2-step shortcut + BFS-d5 window replacement |
| Submit | `kaggle competitions submit …` | always after `verify_submission` passes |

---

## Done / shipped (landmark results only)

- [x] Phase 0: scaffold, puzzle loader, verify pipeline.
- [x] Phase 1: first ML pipeline end-to-end.
- [x] **30,770**: 200ep onehot [700,643]×4 + Kociemba fallback. First ML submission.
- [x] **29,710**: fast recipe (bf16+compile+batch16k+fused). 3× training speedup confirmed.
- [x] **28,224**: fast model at beam 8192 + BFS-d5 pp.
- [x] **27,366**: 4-seed ensemble at beam 4096.
- [x] **27,106**: added MITM BFS-d6 to the ensemble.
- [x] **24,998**: switched to small [1024,256]×1 architecture (E3 config: K_max=26, batch 10k, 4000ep) + khoruzhii searcher port + beam 65K. **Biggest single jump (-2,108)**.
- [x] **24,618**: E5 (8000ep) added to full ensemble + BFS pp.
- [x] **24,068**: E6 Bellman refinement + NISS-E6 added to full ensemble + BFS-d5 pp (submitted 2026-04-19, -550 vs prior best).

Plus supporting infrastructure:
- [x] bf16 inference (1.2× speedup, exact path match).
- [x] MITM BFS via `graph.bfs(return_all_hashes=True)`.
- [x] KhoruzhiiSolver port (`src/cayley/khoruzhii_search.py`) — 150-line self-contained beam with fp16 values, tree-based path storage, stagnation restarts. Fits beam 2^17 on 16 GB.
- [x] BFS-d5 post-processing (790K-state lookup) — small but reliable.
