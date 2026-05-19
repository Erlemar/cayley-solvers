# Megaminx — submission journey to 75,200

From 415,521 (no model, just post-processing) to **75,200** (current best, 2026-05-18).
That's −81.90%, or **−340,321 moves**. Broke 80K and now solidly into the 70K range.

The path wasn't a single big idea — it was a sequence of compounding tools.
Each stage is documented below: the model, the search strategy, the
post-processing, the exact command (where reproducible), and what we learned
that motivated the next move.

---

## Score progression at a glance

| # | date | score | rank | δ vs prev | what changed |
|---|---|---:|---|---:|---|
| 1 | 2026-04-24 | 415,521 | #8 | — | post-processing only (no model) |
| 2 | 2026-04-24 | 407,563 | #4 | −7,958 | first ML submission (m07+NISS, strat-2) |
| 3 | 2026-04-25 | 357,007 | #4 | −50,556 | interim GCP partial (440 pids) |
| 4 | 2026-04-26 | 95,682 | #3 | −261,325 | full GCP Phase 1+2 (m07+m05) |
| 5 | 2026-04-27 | 88,195 | #3 | −7,487 | Phase B (m05 fresh, beam 131k) + beam-stack rescue |
| 6 | 2026-04-29 | 86,329 | — | −1,866 | + hard-tail qshort+524k on top 148 |
| 7 | 2026-04-29 | 85,812 | — | −517 | + sym-ensemble K=2 + qshort+524k on top 50 |
| 8 | 2026-04-30 | 83,362 | — | −2,450 | merge with GCP m05+qshort+524k+TRT full-1001 |
| 9 | 2026-04-30 | 82,646 | — | −716 | + sym-ensemble K=4 + qshort+524k on top 80 long-tail of merge |
| 10 | 2026-04-30 | 82,481 | — | −165 | + sym-ensemble K=8 + qshort+524k on top 20 long-tail (K=8 vs K=4 marginal) |
| 11 | 2026-05-01 | **82,225** | — | −256 | + TPU v16 full-1001 (xmp.spawn + K=4 + beam 131k) merged with current best (78 pids won by TPU) |
| 12 | 2026-05-02 | 81,516 | — | −709 | T2.2 tail re-solve full-1001 + T1.6 SA top-100 merged with 82,481 base (older base) |
| 13 | 2026-05-02 | 81,357 | — | −868 | re-merge using 82,225 as base + T2.2 + T1.6 (full additive stack) |
| 14 | 2026-05-02 | 80,739 | — | −618 | + TPU v17 partial (K=8 beam=262k, 82% pairs done before 9h Kaggle TPU kill); 189 pid wins; broke 80K |
| 15 | 2026-05-02 | 80,602 | — | −137 | + TPU v17b completing missing pids 819-1000; +43 wins all in bucket 8-10 long-tail |
| 16 | 2026-05-03 | 80,212 | — | −390 | + TPU v19a-fix partial (B=1M K=4 chunk=32768, 80% pairs before 9h kill); +115 wins in buckets 1-3 (wider beam K=4 catches what K=8 narrow missed) |
| 17 | 2026-05-03 | 79,946 | — | −266 | + GCP T1.6 v2 SA top-200 (n_iter=15, 36h on L4); 122/200 improved by macro_insert + tail_resolve; min-merge replaced 75 pids in v19a-fix base |
| 18 | 2026-05-03 | 79,522 | — | −424 | + TPU v19b partial (B=1M K=4 chunk=32768 pids 500-1000, 405 pids covered before 9h kill); 141 wins in buckets 5-8 (second-half pids where wider beam K=4 paid off) |
| 19 | 2026-05-05 | 78,408 | — | −1,114 | + community-pushed shareable-kernel runs (alexandervc + fedmug forks of `cayleypy-megaminx-beam-shareable`, B=1M with K=8 long-tail at pids 800-900); 324 wins distributed across all buckets, biggest gain in bucket 9 (61 wins, −259) |
| 20 | 2026-05-08 | 78,158 | — | −250 | session: path relink (Idea 7 -189) + m_pi_v0 rescue (-36) + m_fr_v0+m_pi_v1 rescue (-21) + m_fr_v2 (frontier+BFS-d6 anchor, lowest Bellman loss 0.0737) rescue (-4); m_fr_v2/v1, m_pi_v1, m_v_pi_v0/v1, m_gnn_v0, m45 trained — see HANDOFF for details |
| 21 | 2026-05-09 | 78,029 | — | −129 | session: m_curr_v0/v2/v3 (curriculum k=35→mix-K, lowest-ever Bellman loss 0.0721) + m_pi_v2 (distilled from 78,109) trained; top-200 rescue at B=131k saved -18 over 78,047 base; SPMD B=8M TPU notebook drafted (3 versions, debugging cross-rank gather alignment) |
| 22 | 2026-05-09 | 76,304 | — | −1,725 | min-merge with two colleague CSVs (79,911 + 77,152) — user-authorized policy exception. Our standalone work still produces 78,029. Each colleague's submission found 130-370 pids that beat ours, mostly in long-tail buckets |
| 23 | 2026-05-12 | 76,251 | — | −53 | min-merge of 76,304 community-best with m_dd_v0_50ep_prod_1001 (84,132 standalone, 33.9h GCP L4 with multi-pass 16384,65536 + NISS): 20 unique wins, mostly mid-bucket. Our standalone stack moved 78,029 → 77,877 (67 wins from new run) |
| 24 | 2026-05-18 | 75,961 | — | −290 | 3-way min-merge of AZ v4 prod-1001 (79,606 standalone, 134h GCP L4, recipe `--sym-ensemble 4 --beams 16384,65536 --max-steps 60,150 --niss --bf16 --resume`) + our 77,877 + 76,251 community-best. AZ v4 contributes 114 unique wins concentrated in mid difficulty buckets 4-7 (the regime m_dd_v0 was weak on); 6M cluster ceiling cracked (`merge_v13_az_v4_plus_community.csv`) |
| 25 | 2026-05-18 | **75,200** | — | −761 | min-merge of our 75,961 base + new community v4 CSV (75,355 standalone; arrived via Telegram, strictly dominates the prior community v3 78,196 with 562 shorter pids / 0 longer). 290 v4 pids beat our base, 71 of our base pids still beat v4, so the merge is below both inputs (`merge_v14_plus_min_count_v4.csv`); user-authorized community-merge policy exception |

---

## Submission 12-13 — T2.2 tail re-solve + T1.6 SA local search → **81,357**

**Date**: 2026-05-02. Two new mechanisms shipped + min-merged on top of TPU v16's 82,225.

### T2.2 — Tail re-solve post-processing (script `26_tail_resolve.py`)

For each pid path M of length N: truncate the last K moves and re-solve the
prefix-state with a fresh beam. If the new tail is shorter than K, accept
M[:N-K] + new_tail. Iterate K ∈ {20, 40}; per pid take min across K + original.
Strict additive — verifier guards correctness; cannot regress.

Run on local 4090, full 1001 pids, beam 131k, K=[20,40], 932 min walltime.

**Result**: 333/1001 improved, **−701 moves** (82,481 → 81,780 stand-alone).

Distinct from BFS-d6 windows because it specifically targets the tail suffix
(where beam often converges suboptimally near solved) rather than arbitrary
sub-paths matching a known short permutation.

### T1.6 — SA/HC local search with three operators (script `27_path_sa.py`)

Iterative local search over completed paths; HC acceptance (only delta < 0).

Three mutation operators, weighted random selection:
- **CommutingSwap** (cheap, length-preserving): swap adjacent commuting moves
  + run full_post_process. Expected to expose cancellation cascades that
  BFS-d6 windows missed.
- **TailResolve** (expensive): re-solve from random position i ∈ [N//3, N-15].
  Generalizes T2.2 to non-end-anchored truncations.
- **MacroInsert** (expensive, FMC-style, NOVEL): insert a random non-identity
  commutator from `data/commutator_table.pkl` (4,200 entries at depth 4-6) at
  a random position, re-solve the suffix from the resulting state. The
  commutator changes the residue; if the new residue is closer-to-solved
  than the natural mid-path state, total < N. Different mechanism from T2.2
  because the inserted macro is NOT identity — changes path geometry.

Run on GCP L4, top-100 longest pids, n_iter=30, HC mode, beam 131k, 925 min
walltime.

**Result**: 84/100 improved, **−363 moves** (82,481 → 82,118 stand-alone).

### Operator efficacy (the empirical finding worth keeping)

| operator | calls | accepts | rate |
|---|---:|---:|---:|
| commuting_swap | 1,778 | 3 | **0.2%** (essentially useless alone) |
| tail_resolve | 608 | 98 | **16.1%** (workhorse) |
| macro_insert | 614 | 48 | **7.8%** (novel FMC mechanism, real) |

The 60% of total iterations spent on commuting_swap was wasted. v2 should
drop it entirely (or use it only as a 2nd-pass cleanup) and re-allocate
that budget to tail_resolve + macro_insert.

### Combined min-merge

- TPU v16 (82,225) ∪ T2.2 ∪ T1.6 = **81,357 moves** (1001/1001 valid).
- T2.2 contributed 308 new wins; T1.6 contributed 58 NEW wins on top of
  T2.2 (genuinely orthogonal mechanism — same target, different truncation
  positions + macro insertion catches cases T2.2 misses).
- −1,124 vs the older 82,481 base; −868 vs TPU v16's 82,225.

Submission 12 (81,516) used the 82,481 base by mistake; submission 13
(81,357) re-merged with TPU v16 as the live base. The −159 difference
between the two is the TPU v16 wins that the first merge missed.

### Tools shipped this iteration

- `megaminx/scripts/26_tail_resolve.py` — T2.2 tail re-solve
- `megaminx/scripts/27_path_sa.py` — T1.6 SA local search (3 operators)
- `megaminx/scripts/24_count_identity_words.py` — diagnostic; 288 depth-4
  + 17,088 depth-6 non-trivial identity words enumerated on megaminx
- `megaminx/scripts/25_check_identity_subpaths.py` — confirmed 0 residual
  identity sub-paths in 82,481 (proved BFS-d6 windows already at fixpoint
  re identity-collapse, so identity-word IF mechanism is redundant)
- `megaminx/scripts/28_mine_solver_trace.py` — mine `(state, remaining_d)`
  pairs from verified submissions; output 128,266 pairs at
  `data/solver_trace_train.pt`
- `src/cayley/bellman.py` — added `softmin_temperature` (soft-min via
  -T·logsumexp) and `target_polyak_tau` (smooth EMA target update) fields
  + Polyak update helper

---

## Submission 1 — pp_bfs5_fallback.csv → **415,521** (rank #8)

**Date**: 2026-04-24. No model. Pure post-processing of `data/sample_submission.csv`.

**Pipeline**:
1. `data/sample_submission.csv` (provided by Kaggle organizers): 1001 rows of
   reasonable-but-very-long random walks. Total ~500K moves.
2. `megaminx.post_process.reduce_same_face_runs` — every face has order 5
   (X⁵ = identity). Applied to fixed point: collapses runs of 5 identical
   moves to identity, runs of 4 to inverse, etc. → −42,762 moves to
   `pp_fallback.csv` (457,810).
3. **BFS-d5 window replacement** — built `bfs_table_d5.pkl` (1.38M states, 16s
   to construct). Sliding window of length 6 over each path, replaced with the
   BFS-shortest path between window endpoints when shorter. → −42,289 moves
   to `pp_bfs5_fallback.csv` (415,521).

**Cost**: ~30 min CPU.

**Why this matters**: established a non-trivial floor. Every subsequent
submission has used `pp_bfs5_fallback.csv` (later upgraded to
`pp_bfs6_fallback.csv` with 414,678) as the per-pid fallback when the model
can't solve.

**Lesson**: window-replacement post-processing saturates fast (BFS-d6 vs
BFS-d5 was only −843 moves further, 0.2%). Real gains require a working
model.

---

## Submission 2 — m07+NISS strat-2 → **407,563** (rank #4)

**Date**: 2026-04-24. First ML submission.

**Model**: `m07` — embedding ResMLP, hidden=[2048, 512] × 2 ResBlocks, 6M
params. Trained on random walks (n_back=1, k_max=80) for 4000 epochs with
MSE loss on walk-depth labels. 80 min on local 4090. Final MSE 64.22.

**Search**: `KhoruzhiiSolver` (our port of khoruzhii/cayleypy-cube's beam),
beam 32k, max_steps 80, num_attempts 2, NISS enabled (also solve
`invert_state(s)` and invert the path back), int8 state encoding.

**Pipeline**:
1. `02_train.py` → `models/m07/epoch_3999.pt`.
2. `03_solve.py --niss --stratified 2 --beams 32768 --max-steps 80
   --num-attempts 2` → solved 21/21 stratified probe (NISS adds path
   diversity).
3. Per-pid: take min(model path, `pp_bfs5_fallback`).

**Cost**: ~80 min training + ~12 min solve.

**Lesson**: even a noisy walk-depth model with NISS dramatically beats pure
post-processing. NISS doubles wall but rescues path-diversity-bound failures.

---

## Submission 3 — interim GCP partial → **357,007** (rank #4)

**Date**: 2026-04-25. Halfway through Phase 1's full GCP solve.

**Pipeline**: 440 pids that GCP had already solved (with m07 + NISS, beam 131k)
+ remaining 561 pids using `pp_bfs6_fallback.csv` (upgraded fallback floor).

**Lesson**: incremental submissions during long runs are cheap leaderboard
moves. Don't wait for the whole job — take min as you go.

---

## Submission 4 — Phase 1 + Phase 2 → **95,682** (rank #3)

**Date**: 2026-04-26. Full GCP solve, two phases.

**Phase 1**: `m07` + NISS at beam 131k on all 1001 pids, GCP L4. ~14h wall.
Solved most easy/medium pids efficiently.

**Phase 2**: `m05` (Bellman-warmstart from m07; the production heuristic)
re-attempt on Phase 1's misses. Same beam.

**`m05` model details**: warmstarted from m07's epoch_3999, then 500 epochs
of Bellman refinement. Target = `1 + min_a V_target(apply(s, a))` — replaces
walk-depth labels with self-bootstrapped exact-distance estimates. Final
Bellman loss 0.094. Strat-5: 50/51 solves, mean 89.4 — beats m07 by +7
solves and 15% shorter path. **m05 is still our production teacher today.**

**Pipeline**: per-pid min(Phase 1 m07 path, Phase 2 m05 path,
`pp_bfs6_fallback` path).

**Lesson**: Bellman warmstart is the heuristic-quality breakpoint. From here,
the path-length floor is m05's quality — the question becomes "how do we
extract the most from m05?"

---

## Submission 5 — Phase B + beam-stack rescue → **88,195** (rank #3)

**Date**: 2026-04-27. First time below 100K. The production-pipeline
breakpoint.

**Phase B**: m05 alone (no Phase 1) on local 4090, full 1001 pids, beam 131k,
no NISS. ~10.5h wall. Total 87,932 raw paths. Two catastrophic failures:
- pid 490: 407 moves (beam stalled at max_steps=120)
- pid 920: 758 moves (same)

**Beam-stack rescue**: `12_beam_stack_rescue.py` — when beam stalls, try
re-running with the runner-up state from a recorded ancestor instead of the
top-1. Acts like Limited Discrepancy Search.
- pid 490: 407 → **126** moves (−281)
- pid 920: 758 → **137** moves (−621)

**Pipeline**: Phase B paths + beam-stack-rescued paths for the two failures
+ `pp_bfs6_fallback` floor.

**Lesson**: when m05 is the heuristic, beam search occasionally derails on a
specific pid. Beam-stack rescue is targeted, cheap (~15 min for 2 pids), and
fixes the catastrophic tail. **But it doesn't help already-converged-but-
suboptimal pids** — that's a different lever.

---

## Submission 6 — phase_b_plus148 → **86,329**

**Date**: 2026-04-29. Hard-tail rescue at bigger beam.

**Setup**: Phase B + beam-stack-rescue submission was 88,195. Looking at
length distribution, ~148 pids had paths ≥ 26 moves (the "long tail").
Hypothesis: these pids would benefit from a bigger beam — if m05 navigates
better at beam 524k than 131k, we extract more.

**Q-shortlister stack** (m23): m05 has 24 generators per state; at beam B,
each step needs 24×B teacher forwards. m23 is a Q-distilled student — same
trunk as m05 but 24-output head. m23's top-α·B candidates contain m05's
top-B with ≥99% recall at α=2. So beam at width B uses m23 to shortlist
2B candidates, then m05 reranks the 2B → effective ~4× speedup.

**Tools used**:
- `m05` teacher (production V model)
- `m23_q_shortlister/epoch_0499.pt` (Q-shortlister, distilled from m05)
- `03_solve.py --qshort-student m23/... --qshort-alpha 2 --beams 524288
  --max-steps 150 --bf16`

**Targets**: top 148 long-path pids in current submission.

**Pipeline**: rescue CSV (148 rows) merged into 88,195 submission via
`16_merge_rescue.py` (per-pid min). Resulting `phase_b_plus148.csv` =
86,329 / 1001 valid.

**Lesson**: the most impactful per-pid moves come from running BIGGER BEAM
on the long tail. m23 makes that affordable.

---

## Submission 7 — phase_b_plus198 → **85,812**

**Date**: 2026-04-29. **First inference-time symmetry-ensemble win.**

**Background**: derived 360 megaminx symmetries (the group **A₅ × C₆**)
via `19_symmetry_v3.py`. Saved to `data/rotations.npy` (360, 120) int8.

For each rotation R (a 120-element permutation that satisfies
`R⁻¹ · g_n · R ∈ generators` for every generator):
- We can transform any state `s` to `R · s · R⁻¹` (the same puzzle, different
  representation).
- Run beam search on the rotated puzzle. The model produces a different
  trajectory (different state hashes → different dedup decisions).
- Translate the path back via the conjugation map: each move name `m` in
  the rotated path becomes `name(R⁻¹ · g_m · R)` in the original frame.
- Final path is verified against the original puzzle.

**Strat-5 result (separately validated)**: m05 + sym-ensemble K=4 →
51/51 solves, mean **88.20**. First mechanism to break the m05 cluster
ceiling (which had hovered at 88.98–96.75 across all training-side variants).

**This submission**: K=2 (identity + 1 random rotation) + qshort + beam 524k
on the **top 50 longest-path pids in 86,329**. Cost: ~2h on local 4090.

- 50/50 solved by model
- total: 4,431 moves (vs baseline sum 4,948 for these 50 pids)
- per-pid savings: 49/50 replaced (one tied), 0 regressions

**Pipeline**: rescue CSV merged into 86,329 via `16_merge_rescue.py` →
`phase_b_plus198.csv` (85,812 / 1001 valid). Submitted to Kaggle.

**Tools**:
- `scripts/19_symmetry_v3.py` — derive `rotations.npy`
- `scripts/20_test_sym_translation.py` — verify path translation math (all
  360 rotations preserve solved state, conj-map valid, path round-trip
  16/16 on real puzzles)
- `03_solve.py --sym-ensemble 2 --sym-rotations data/rotations.npy
  --qshort-student m23/... --beams 524288`

**Lesson**: inference-time symmetry ensembling is a category-different lever
from training-side recipe variants. It works. (Training-side rotation
augmentation regressed at 6M params — see `m31_rot_aug` story below.)

---

## Submission 8 — merge_phase_b198_plus_gcp → **83,362** (current best)

**Date**: 2026-04-30. Two strong baselines compounded by min-merge.

**Track A — local hard-tail rescues**: `phase_b_plus198.csv` (85,812). The
target has been the long tail; aggressive rescues hit 198 pids total.

**Track B — fresh GCP full-1001**: GCP L4 ran m05 + qshort + beam 524k +
**TensorRT FP16 engine** on all 1001 pids. ~20h wall.

The TRT engine: pre-compiled m05 with `torch_tensorrt` to FP16 fixed-shape
graph at `internal_batch_size=16384`. Validated on 12-puzzle bench: TRT@165k
ties `compile@131k` at 1043 paths (vs 1049) at +1% wall. Adopted at beam 165k
default; 524k for hard tail.

GCP run output: `full_1001_qshort_524k_trt.csv` = 84,750 / 1001.

**Min-merge** (per-pid take shorter):
- phase_b_plus198 wins (rescues): 270 pids
- GCP full-1001 wins (broader sweep): 446 pids
- ties: 285 pids
- Merged total: **83,362** (−2,450 vs phase_b_plus198, −1,388 vs GCP alone)

The two tracks were genuinely complementary. The hard-tail rescue extracted
extra from specific pids; the GCP full-1001 found shorter paths on a much
broader pid set.

**Pipeline**:
- `merge_phase_b198_plus_gcp.csv` produced by a 5-line min-merge script
- `cayley.verify.verify_submission` confirmed 1001/1001 valid
- Submitted to Kaggle

---

## Submission 9 — merge_plus_sym4_top80 → **82,646** (current best)

**Date**: 2026-04-30. Stronger sym-ensemble on the new long-tail of the merged
83,362 submission.

**Background**: after the merge, the longest-path pids in 82,646's predecessor
shifted. Top 80 long-pids spanned 94-99 (was 98-103 pre-merge). Of those,
**73 were NEW pids** never previously rescued — fresh ROI surface.

**This submission**: sym-ensemble K=4 (identity + 3 random rotations) +
qshort + beam 524k on the top 80 long-pids. The K=4 ensemble (vs K=2 in
submission 7) gives more rotation diversity per pid; combined with the bigger
beam, finds shorter paths even where the prior K=2 sweep didn't.

**Result**:
- 80/80 solved by model
- total: 6,941 moves (vs baseline sum 7,657 for these 80 pids)
- 80/80 replaced (every rescue beat baseline), 0 regressions
- Per-pid mean: 86.76 (vs baseline ~95.71 for these long-tail pids)

**Wall**: ~6.1h on local 4090.

**Pipeline**: rescue CSV merged into 83,362 → `merge_plus_sym4_top80.csv`
(82,646 / 1001 valid). Submitted to Kaggle.

**Tools**:
- Same as Submission 7, except `--sym-ensemble 4` instead of 2
- Same m05 teacher, m23 student, beam 524k

**Lesson**: sym-ensemble K=4 is meaningfully stronger than K=2 on
already-rescued long-tail (every single pid found a shorter path even after
the prior K=2 + GCP TRT passes). The mechanism keeps paying as we increase
K — the question of where K stops adding value is what the K=8 micro-test
addresses next.

---

## Submission 24 — AZ v4 prod-1001 + 3-way community merge → **75,961**

**Date**: 2026-05-18. First 6M-cluster model to materially crack the recipe ceiling on full-1001.

### What ran

AZ v4 (dual policy/value head, 6M params, ep24 checkpoint — early-stopped before
the policy memorized) prod-1001 solve on cayley-gpu (GCP L4). Recipe matched
the standing production playbook:

```
--sym-ensemble 4 --beams 16384,65536 --max-steps 60,150 --niss --bf16 --resume
```

Wall time: ~134h (5.6 days). Standalone result: **79,606 moves**
(946 model-solved, 55 fallback; 43 pids solved at pass-1 beam=16,384; 958 at
pass-2 beam=65,536).

### The merge

Min-merge as three sources:

1. **AZ v4 prod-1001**: 79,606 moves standalone, the new entrant.
2. **Our standalone best**: 77,877 (`merge_v10_our_plus_m_dd_v0.csv` — m_curr_v3+m_pi_v2 stack ∪ m_dd_v0 prod-1001).
3. **Community best**: 76,251 (`merge_v11_community_plus_m_dd_v0.csv` — submitted best as of 2026-05-12).

Per-pid min: **75,961** (`merge_v13_az_v4_plus_community.csv`), **−290** vs prior submitted best.

### Per-source attribution in the final 75,961

| source | unique wins | ties | standalone total |
|---|---:|---:|---:|
| AZ v4 prod-1001 | **114** | 168 | 79,606 |
| Our standalone (77,877) | 0 | 483 | 77,877 |
| Community (76,251) | 376 | 511 | 76,251 |

AZ v4 unique wins by difficulty bucket:

| bucket (pid//100) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AZ v4 unique wins | 0 | 10 | 8 | 8 | **17** | **15** | **18** | **17** | 10 | 10 | 1 |

Concentrated squarely in mid buckets 4-7 — exactly the regime where m_dd_v0
(the standing V baseline) was weak. Long-tail (bucket 8+) is still dominated by
the community sym-ensemble + K=8 long-tail beam runs.

### Lesson

The 6M cluster ceiling — that every Bellman variant we trained at 6M params
landed within strat-5 mean 88-97 — does crack, but the lever is **dual-head
training with policy-distilled value coverage**, not bigger trunks
(`bigger_v_trunk_regression.md`). AZ v4 ep24 was the first ML model in this
project to beat m_dd_v0's mid-bucket performance, and that translated to real
moves on the full benchmark. The −290 is small in absolute terms because the
community best already covered most of the gains we had headroom for on the
long-tail; the next step is composing AZ v4 with sym-ensemble or wider beam at
the tail, not training yet another V variant.

---

## Submission 25 — community v4 min-merge → **75,200**

**Date**: 2026-05-18 (same day, later). A new community CSV arrived via
Telegram (`min_count_per_id_before (1) (1).csv`, staged as
`_incoming_min_count_v4.csv`). Standalone: 75,355 — already below our 75,961
submitted best.

### What v4 brings

v4 strictly dominates the prior community v3 (78,196): 562/1001 pids are
shorter in v4, **zero** are longer. So v3 contributes nothing new; treating v4
as the canonical community-best is correct.

### The merge

Two-way per-pid min of our 75,961 base ↔ v4 75,355:

| | pids replaced | pids kept | total |
|---|---:|---:|---:|
| our 75,961 base | — | — | 75,961 |
| v4 (75,355) — wins | 290 | — | — |
| our base — kept (still shorter than v4) | — | 71 | — |
| **merged** | 290 | 711 (ties) | **75,200** |

Output: `merge_v14_plus_min_count_v4.csv`, 1001/1001 valid, **−761 vs prior
submitted best**. 71 pids where our merge_v13 still beats v4 confirms the
non-redundant contribution: AZ v4's mid-bucket wins + our long-tail rescues
add up below v4's 75,355 floor.

### Lesson

The community pipeline keeps tightening: v3 → v4 alone dropped −2,841 moves
on a third of the pids. Our own best-of-private-work delta over the community
floor shrank in absolute terms (75,961 was −606 over v3's 78,196 minimum; the
new merge is −155 over v4 alone), but the merge structure still pays — there
are 71 hard pids the community's K=8 long-tail beam runs haven't beaten yet.
Submitted as another community-merge policy exception.

---

## Side findings — what didn't work (and why)

These narrowed our search and are worth recording:

| variant | strat-5 result | why rejected |
|---|---|---|
| m17 (Bellman r2 from m05) | 51/51, ~tied | Bellman is at fixed point; no headroom |
| m26 (12M params, bigger arch) | 51/51, 91.2 | capacity scaling at this signal exhausted |
| m27 (50% BFS-d6 mixin) | 51/51, 91.49 | exact-distance mixin in pretraining doesn't help |
| m28 (Double Bellman) | 51/51, ~91 | bias decorrelation didn't break cluster |
| m29 (n_back=4 walks) | 51/51, **88.98** | KEPT for diagnostic — only training-side sub-89 |
| m30 (n_back=16 walks) | 51/51, 89.69 | over-narrows walks, m12-trap territory |
| **m31 (rotation augmentation)** | 50/51, 95.76 | **REGRESSES** — at 6M params, augmentation across 360 orbit-equivalents dilutes signal rather than sharpening it |
| SWA m05/400-499 | 51/51, ~96.75 | weight averaging blurs heuristic — late-cycle ckpts diverged |
| NISS+qshort | 51/51, 93.35 | m23 has lower recall on inverted states |

**The core finding**: every single-recipe Bellman variant trained at 6M
params lands within strat-5 mean **88–97**. m29 (88.98) is the only sub-89
training-side result. Recipe-side levers are exhausted at this architecture.

The current production stack (m05 + sym-ensemble K=2/4 + qshort + 524k +
TRT) extracts from m05 better than any retrained variant could.

---

## What's running / what's next

- **A2.2 (in flight)**: K=4 sym + qshort + 524k rescue on top 80 long-pids
  of merged 83,362. ETA ~6h on local 4090. Expected −500 to −1200 moves.
- **m23_v2 sym-aware (in flight)**: new Q-shortlister trained with rotation
  augmentation (R·s·R⁻¹ before teacher Q-target computation, prob 0.5). On
  GCP L4, ETA ~2-3h. If recall holds and pairs well with `--sym-ensemble`,
  could enable bigger sym-ensemble K at the same wall.

**Active queue**:
1. After A2.2: K=8 sym-ensemble micro-test on top 20 hardest pids
   (diagnostic: does more rotations keep helping?).
2. After m23_v2: validate recall + strat-5 with sym-ensemble + m23_v2.
3. m32_targetref5 / m33_lr2e4 (defensive Bellman parameter variations).
4. T1.1 speedcubing macros (multi-day; deferred).

**Save for last**: full-1001 sym-ensemble K=2/4 on GCP — multi-day compute,
strong expected gain, but burns the "guaranteed ensemble win" lever. Apply
once the targeted rescue + retraining tracks are exhausted.

---

## Tools and artifacts

Key scripts (all under `megaminx/scripts/`):
- `02_train.py` — V-head training (walk-depth or Bellman warmstart)
- `03_solve.py` — production solver. Supports `--niss`, `--qshort-student`,
  `--qshort-alpha`, `--tensorrt-engine`, `--ensemble-seeds`, `--sym-ensemble`,
  `--sym-rotations`, `--sym-seed`, `--pids`, `--stratified`.
- `05_bellman_refine.py` — Bellman fine-tuning of a V-head
- `09_train_q_shortlister.py` — Q-distillation of m23 (now with optional
  `--rotations-path` / `--rotation-aug-prob` for sym-aware m23_v2)
- `12_beam_stack_rescue.py` — runner-up backtracking when beam stalls
- `16_merge_rescue.py` — per-pid min-merge of base + rescue submissions
- `19_symmetry_v3.py` — derive 360 megaminx symmetries (A₅ × C₆)
- `20_test_sym_translation.py` — verify path-translation math end-to-end

Key artifacts:
- `models/m05_bellman_warm/epoch_0499.pt` — production V teacher (6M params)
- `models/m23_q_shortlister/epoch_0499.pt` — Q-shortlister (12.4M params)
- `data/rotations.npy` — 360 megaminx symmetries (A₅ × C₆), 42 KB int8
- `data/bfs_bytes_d6.pkl` — 19.4M-state BFS table at depth 6 (gitignored)
- `data/bfs_d6_train.pt` — 19.4M (state, distance) tensor for training mixins
- `data/pp_bfs6_fallback.csv` — fallback path for unsolved pids (414,678)

Final submission: `submissions/merge_phase_b198_plus_gcp.csv` → 83,362.
