# Speed optimizations — actionable plan

Ranked list of optimizations for **beam search wall-clock** and **model forward**,
with realistic time estimates and the smallest experiment that validates each.

Reference points (Megaminx, 4090 Laptop / GCP L4):
- m07 training (RW, 4000 ep, from scratch): **80 min**
- m05 training (Bellman warmstart from m07, 500 ep): **30 min**
- Phase 1 beam (m07 full 1001 at beam 131k): 34h GCP L4
- Phase 2 beam (m05 retry on 207 m07 misses): 6.5h GCP L4
- Final score: **95,682** moves (rank ~3-4 of 16, target: break 80K)

## Empirical findings (refreshed 2026-04-27 across L4 + 4090 sessions)

| measurement | value | implication |
|---|---|---|
| `model_s` fraction of wall | **96-97%** at beam 131k (both L4 and 4090) | Optimizations on stages <4% can't move the needle |
| `neighbor_s` fraction | 2% | Incremental hashing's main benefit is memory (377 MB saved), not wall |
| `dedup_s + apply_s + topk_s + hash_s` combined | <2% | Tier-1 micro-opts targeting these are within run-to-run noise |
| v0 baseline (12 samples, m05, beam 131k, L4) | 1240s wall, 1053 path-sum | original reference |
| 4090 baseline (no compile, 12 samples) | 1032s wall, 1049 path-sum | reference for compile A/B |
| 4090 + `--compile` (12 samples) | **752s wall, 1049 path-sum** | **−27% wall, paths IDENTICAL** — the only "free" speed win we found |
| 4090 + `--compile` (24-puzzle stratified) | 1497s wall, 2013 path-sum | reference baseline for any future A/B |
| async pipeline ceiling (theoretical max) | ~4% | bounded by 1 − model_s fraction |
| async pipeline measured (skip syncs) | 0.7% (within noise) | confirms theoretical ceiling |
| `internal_batch_size` 16k vs 32k vs 65k | tied (compute-bound) | 16k is the right default; don't keep raising |
| Q-shortlister (m23) at beam 131k | **4.4× wall** vs m05, +6% paths | wall savings funded scaling beam from 131k to 524k |

**Conclusion**: every speed lever that doesn't shrink the model forward time has
been tested and either deployed or rejected. Real wins from here require either
(a) faster model forward — TensorRT FP16 (see `tensorrt_gcp_plan.md`), or
(b) fewer model evaluations per step — Q-shortlister (already deployed) or
distillation (rejected on quality), or (c) a sharper heuristic that cuts more
puzzles short — m26/m26b capacity scaling (running on Kaggle) or multi-task
training (see `multitask_training_ideas.md`).

## Implementation status (as of 2026-04-27)

- ✅ = code done, awaiting GPU; 🔬 = trained, awaiting eval; 🚀 = deployed/submitted; ❌ = abandoned

| ID | idea | status |
|---|---|---|
| §1 | Learn-to-hit-shell (m21) | ❌ REJECTED 2026-04-26: m21+MITM path-sum 1197 vs m05's 1053 (-14%); shifted target loses Bellman sharpness |
| §2 | Q* shortlister (m23) | 🚀 DEPLOYED 2026-04-27 — recall=100% at α=2, beam 131k 4.4× speedup; deployed in qshort solver; powered the beam-524k full 1001 run |
| §3 | Beam-stack backtracking | 🚀 DEPLOYED 2026-04-27 — rescued pid 490 (407→126) and pid 920 (758→137), enabling 88,195 submission |
| §4 | Limited-horizon Bellman (m22) | ✅ `10_train_m22_horizon.py` (training only; not yet integrated) |
| §5 | Policy-guided beam (m24) | ✅ `11_train_policy_head.py` (training only; beam integration pending) |
| §6 | torch.compile + fixed beam padding | 🚀 DEPLOYED 2026-04-27 — `--compile` flag, mode=reduce-overhead, −27% wall on 12-puzzle, paths IDENTICAL |
| §7 | Larger `internal_batch_size` | ❌ REJECTED 2026-04-27 — 16k=32k=65k all tied at compute-bound on 4090 / L4. 16k is the right default. |
| §8 | Macro-Q shortlisting | ⏳ deferred (Tier B) |
| §9 | Symmetry-aware model | ⏳ blocked on rotation derivation |
| §10 | CUDA Graphs (manual capture) | ❌ REJECTED 2026-04-27: direct capture −34% vs `--compile` on 3-puzzle sample (245.9s vs 182.5s). Inductor's kernel fusion is the real win, not graph-replay overhead reduction; `compile(default)+manual graph wrap` ties `--compile` (182.4s) — no headroom. |
| beam_decay (geometric narrowing) | per-step beam = max(min, B·decay^j) | ❌ REJECTED 2026-04-27 (quality regression): decay=0.99 floor=32k saved 33% wall on 3-puzzle but +2% paths; pid 492 +7 moves (92→99). On 1001 puzzles this would push score ~88,195 → ~91k. Speed-without-quality goal not met. |
| Stochastic beam (Gumbel-top-k temperature) | sample B without replacement from softmax(-V/T) | ❌ NOT TESTED 2026-04-27: same goal as beam_decay (cut search short / explore more) but adds randomness vs greedy top-B. Not a speed lever — it changes which states are kept, not how many. Quality risk on greedy-already-optimal pids. Skipped per user direction (no metric regressions). |
| Async pipeline (skip profile syncs + deferred solved-check) | `profile=False` makes `_sync()` a no-op; `solved_check_every=K` checks every K steps | ❌ REJECTED 2026-04-27: (1) skip-syncs alone: 181.3s vs 182.5s baseline on 3-puzzle (−0.7%, within noise). (2) +`solved_check_every=4`: 183.9s wall (+0.7%) with **+0.8% path-sum** (256 vs 254) — when V0 enters at step j and we check at step j+K, V0 is gone and beam re-discovers it at step j+K via a different parent → +1 path. The 96% model_s ceiling means async wins are bounded at ~4%; in practice <1%. Not worth a structural change. Code stays gated behind `--no-profile` / `--solved-check-every` flags (defaults preserve current behavior). |
| §11 | PDB-lite admissible features | ⏳ deferred (only useful with A*) |
| §12 | Bidirectional front-to-front | ⏳ research project; deferred |
| Adaptive beam escalation | 16k→65k→131k early-exit per puzzle | ❌ REJECTED 2026-04-27 — 24-puzzle wall 331.7s (−78%) but path-sum +13% (2270 vs 2013). Quality regression on easy puzzles where 16k beam misses optimal. |
| TensorRT (FP16) on model forward | export the m05 ResMLPDistance to a TRT engine for the model_s 96% slice | ⏳ PLANNED 2026-04-27 — full plan in `tensorrt_gcp_plan.md`. GCP-only (Windows wheel build fails). 6-step plan, ~2.5h active work, hard gate: 24-puzzle path-sum must equal 2013. The only remaining lever past the async ceiling. |
| MITM | BFS-d6 shell beam termination | ❌ REJECTED 2026-04-27 — cayleypy `simple+MITM` 215.3s vs ours 183.6s (+17%) on 3-puzzle; m05 already navigates d≤6 shell as a side-effect, MITM adds shell-hash overhead without saving paths. |
| Multi-puzzle batch | K=4 lockstep beams | ❌ REJECTED 2026-04-26 — wall 4796s vs single 1240s (3.7× SLOWER on L4); compute-bound, packing K puzzles makes the per-batch model call dominate differently. |
| Incremental hashing | parent-hash + delta on changed positions | ✅ in `beam_search.py`, default on |
| cayleypy `iterated` mode w/ history_depth | non-backtracking enforcement | ❌ REJECTED 2026-04-27 — 410.1s vs ours 183.6s (+124%); paths +10 moves. Russian commenter explicitly said "this slows down" — quality lever, not speed lever. m05's sharp Bellman doesn't need it. |

---

## Tier S — clearest path, smallest cost

### 1. Learn-to-hit-shell (m21)
**Idea**: train value model with target = `clamp(walk_depth − 6, min=0)` instead of
`walk_depth`. The search only needs to reach the BFS-d6 shell, then MITM splices
the exact tail. The model has a smaller dynamic range to learn, doesn't waste
capacity on the d≤6 region we already have exact answers for.

| | |
|---|---|
| Training time | ~30 min (warmstart from m05, 500 ep) |
| Code change | 1 line: subtract 6 and clamp the target |
| Validation | strat-5 path-length comparison vs m05 (both with MITM enabled) |
| Predicted gain | 15–25% wall reduction at same path quality |
| Risk | very low — worst case m21 ≈ m05 |
| Total clock | ~1h end-to-end |

### 2. Q* shortlister with recall validation
**Idea**: train a 24-output Q-head from m05 teacher. Use as a *shortlister* (top-α·B
candidates per beam step) feeding into m05 reranker. Student needs high recall
of teacher's top-B, NOT precise final ranking — much easier than what m06 had to
do (and m06 was a final ranker that lost ranking).

Validation gate: BEFORE running beam, measure
`P(student_top_αB ⊇ teacher_top_B)` on held-out states. Don't proceed unless
recall ≥ 99% at α=4.

| | |
|---|---|
| Training time | ~30-60 min (similar arch to m05 + 24-output head) |
| Code change | new training script + `_q_predict` integration in beam loop |
| Validation | recall measurement (10 min), then strat-5 |
| Predicted gain | **6× model wall reduction** (24·B → α·B candidates per step) → ~5× total wall |
| Risk | medium — recall might be insufficient at α=4 |
| Total clock | ~4h end-to-end |
| Reference | DeepCubeAQ, arXiv 2102.04518 |

---

## Tier A — real wins, medium effort

### 3. Beam backtracking / discrepancy rescue
**Idea**: standard beam keeps top-B at each layer. Beam-stack search also stores
the (B+1)..(2B) "runners-up" at early layers. If the main beam fails, restart
from a runner-up. Specifically targets the residual hard tail (pid 492 type).

| | |
|---|---|
| Training time | none |
| Code change | ~4h (modify solve loop to store runner-up snapshots, retry path) |
| Validation | run on the ~10 m05-fails-at-beam-131k puzzles; success = puzzles solved |
| Predicted gain | rescue 2–5 currently-unsolved puzzles → -2K to -5K total moves |
| Risk | low — easy puzzles unchanged |
| Total clock | ~5h |

### 4. Limited-horizon Bellman (m22)
**Idea**: retrain with K-step lookahead targets instead of 1-step Bellman.
Reduces "heuristic depression regions" where m05 underestimates.

| | |
|---|---|
| Training time | ~90 min (K=3 lookahead → 3× per-epoch cost; 500 ep warmstart) |
| Code change | ~2h (modify Bellman target computation) |
| Validation | strat-5 (with MITM) vs m05 |
| Predicted gain | 5–15% on path quality + smoother search |
| Risk | low |
| Total clock | ~4h |
| Reference | arXiv 2511.10264 |

### 5. Policy-guided beam (PHS / Levin-style)
**Idea**: train a softmax policy head π(a|s) on solved-path triplets (state, move).
Score beam candidates as `value + λ·(-log π(a|s_parent))`. Disambiguates among
same-V candidates.

| | |
|---|---|
| Training time | ~30-60 min (multi-task policy head on m05 body, 500 ep) |
| Code change | ~3h (policy head + beam loop integration) |
| Validation | strat-5 path-length comparison; tune λ |
| Predicted gain | 1–3% absolute, 5–10% on hard puzzles where V plateaus |
| Risk | low — λ=0 reduces to current behavior |
| Total clock | ~5h |
| References | Levin Tree Search (IJCAI 2023), PHS (arXiv 2103.11505) |

### 6. torch.compile with fixed beam padding
**Idea**: pad the beam to always exactly B (even when alive < B). Compile becomes
possible (no recompile loops). Pure system-level optimization.

| | |
|---|---|
| Training time | none |
| Code change | ~3h (padding logic + compile invocation) |
| Validation | benchmark optimized beam_lab vs current at beam 131k |
| Predicted gain | 1.3–1.8× on `model_s` → 25–45% wall reduction |
| Risk | very low — falls back to current behavior if disabled |
| Total clock | ~4h |

### 7. Larger `internal_batch_size` (32k–65k)
**Idea**: trivial bump. L4 has 24GB; we default to 16k. More work per kernel launch.

| | |
|---|---|
| Training time | none |
| Code change | trivial — argparse default |
| Validation | A/B benchmark on 12 lab samples |
| Predicted gain | 5–10% wall (less Python overhead per step) |
| Risk | none — falls back if VRAM limited |
| Total clock | ~30 min including measurement |

---

## Tier B — speculative, larger effort

### 8. Macro-Q shortlisting
**Idea**: extend Q-head to score 2–6 move *macros* mined from solved paths and
BFS-d6 tails, not just primitive 24 moves. Reduces beam STEPS (not just per-step
cost). Quality risk if macro recall is bad.

| | |
|---|---|
| Training time | ~60 min (Q-head with 24 + ~500 macro outputs) |
| Code change | ~2 days (macro mining + integration) |
| Validation | A/B beam-step count at same path quality |
| Predicted gain | 1.5–3× wall on hard puzzles if macros are useful |
| Risk | medium-high — macros might not generalize |
| Total clock | ~3 days |

### 9. Symmetry-aware model (data augmentation)
**Idea**: instead of orbit-dedup at search time (failed twice), bake symmetry into
training. Random rotation/conjugation of training samples. Model learns to
predict same V for s and rot(s).

| | |
|---|---|
| Training time | ~60 min (5 ep × 60 rotations effective augmentation = same epochs but bigger effective dataset) |
| Code change | ~30 min (data augmentation in training loop) |
| Validation | strat-5; model_avg should drop |
| Predicted gain | 5–10% solve rate / path quality |
| Risk | medium — relies on having the 60 rotations derived correctly |
| Total clock | ~1.5 days (mostly: deriving the 60 rotation permutations) |
| Reference | symmetry-aware Rubik work (PubMed) |

### 10. CUDA Graphs
**Idea**: capture one beam-step into a graph; replay each step. Lower overhead
than compile. Requires fixed shape (works with #6 above).

| | |
|---|---|
| Training time | none |
| Code change | ~1 day |
| Validation | benchmark on lab samples |
| Predicted gain | 1.2–1.5× on inner loop (stacks with compile?) |
| Risk | medium — fiddly with dynamic state (alive mask, blacklist) |
| Total clock | ~1.5 days |

### 11. PDB-lite admissible lower bounds
**Idea**: pattern databases on Megaminx subsets (corner positions, single face).
Use as additive lower bound or model feature.

| | |
|---|---|
| Code change | ~4h |
| Validation | benchmark with vs without PDB feature |
| Predicted gain | marginal for beam, real for A*/IDA* |
| Risk | low |
| Total clock | ~5h |
| Reference | Korf 1997, disjoint PDBs |

### 12. Bidirectional with learned front-to-front scoring
**Idea**: train a 2-input model `h(s_forward, s_backward)`. Real research
project. Skip in this competition timeline.

| | |
|---|---|
| Total clock | ~1 week+ |
| Status | deferred |

---

## Recommended execution order

Phased, assuming GPUs free up after Phase B finishes (~tomorrow 11 UTC):

### Phase A — Free wins (0.5 day)
- **#7 Larger `internal_batch_size`** (30 min, immediate measurement)
- **#1 Learn-to-hit-shell m21** (1h training + eval)

### Phase B — Engineering wins (1 day)
- **#6 torch.compile + fixed beam padding** (4h)
- **#3 Beam backtracking** (5h)

### Phase C — Model wins (1-2 days)
- **#2 Q* shortlister with recall gate** (4h)
- **#4 Limited-horizon Bellman m22** (4h)
- **#5 Policy-guided beam** (5h)

### Phase D — Speculative (2-3 days)
- **#8 Macro-Q** (3 days)
- **#9 Symmetry-aware model** (1.5 days, blocked on rotation derivation)
- **#10 CUDA Graphs** (1.5 days)

---

## Stacked impact estimates

If all Phase A + B + C land:

| change | wall reduction (multiplicative) | path-length impact |
|---|---|---|
| baseline (current 95,682) | 1.0× | 95,682 |
| larger batch_size | 0.92× | unchanged |
| m21 learn-to-hit-shell + MITM | 0.78× of remaining | -10% paths |
| torch.compile + padding | 0.65× | unchanged |
| beam-backtracking rescue | unchanged wall, -3K paths | -3K |
| Q* shortlister α=4 | 0.20× model, 0.30× total | unchanged (validated by recall) |
| limited-horizon m22 | unchanged wall, -5% paths | -5% |
| policy-guided beam | unchanged wall, -3% paths | -3% |

Stacked best case (all Tier A): **~70–75K total moves at ~5× faster wall**.
That's well past the 80K target and into chasing rank #1 territory.

Stacked realistic case (only some land): **80–85K**.

---

## Anti-recommendations

- More Bellman rounds beyond m05 (m17 told us r2 is a no-op).
- More Transformer attempts (m18 told us infeasible at our wall budget).
- A* / IDA* in this timeline (1 week+, uncertain win).
- Bidirectional front-to-front learned scoring (research project).
