# Beam search optimization — research notes

A deep dive on what we could change about beam search itself and what we'd
expect each change to deliver. Companion to README.md (which says how to run
the lab); this file says what to put in it.

## Section 0: First, profile

Before chasing any specific optimization, **run the baseline on the 12 sample
puzzles and read the per-stage timing breakdown** (`run_benchmark.py` reports
it). The optimizations below are clustered by which stage they hit; the
expected wins assume that stage is actually a meaningful share of wall time.

Best guesses for our setup (m05 + beam 131k + 4090, no profile yet):

| stage | guess of wall % | what it does |
|---|---|---|
| `model_s` | 50–70% | model forward on `B × n_gen` candidate states per step |
| `neighbor_s` | 15–25% | apply 24 generators to B states; hash the children |
| `dedup_s` | 5–10% | sort + unique on hashes; isin(blacklist) |
| `apply_s` | 5–10% | gather to materialize the chosen B states |
| `topk_s` | 5–10% | argsort over `B × n_gen` values |
| `hash_s` | 1–3% | hash the B chosen states for stagnation log |

If `model_s` ≪ 50% in your profile, **most of Section 2 is irrelevant** —
attack `neighbor_s` or `dedup_s` instead.

---

## Section 1: Search-space pruning (= search less)

These reduce the number of states beam ever has to look at. Often the largest
absolute wins, because they compound with everything else.

### 1.1 MITM with BFS-d6 shell **(highest ROI; code exists)**

We already have `bfs_bytes_d6.pkl` — exact BFS distances for the 19.4M states
within 6 moves of solved. We've never run it end-to-end.

**Mechanism**: at each beam step, before scoring, check if any state's hash is
in the precomputed shell-hash set. On match, retrieve the known-optimal path
from that state and prepend the beam's path. Search terminates ≤ d_shell
layers earlier than naive beam.

**Math**: for a hard puzzle that beam would otherwise solve in D=50 layers,
hitting the shell at D-6 saves the last 6 layers × beam_width worth of state
expansion = ~786k state-evaluations. That's **10–20% wall-clock saved per hard
puzzle.** Concentrated in late-bucket pids where it matters most.

**Quality**: zero loss — BFS-d6 path is exact. Strictly improves or matches
beam's solution.

**Stronger version**: build BFS-d8 shell at solve time (not pre-baked). d=8
has ~5B states, intractable in full. But `bfs_bytes_d6 ∪ {neighbors of d6
states}` = d=7 partial shell, ~250M states. We could store *just the hashes*
(8 bytes × 250M = 2GB) and skip the path reconstruction. On hash-hit, run a
short focused beam from that state to one of the 19M known-d=6 states. This
gets the shell to effective d=8 with manageable memory.

**Smallest experiment**: run `MitmKhoruzhiiSolver` on pid 492 (the puzzle our
production m05 failed at beam 131k). If it solves, that's the cleanest signal
we've ever had.

### 1.2 Bidirectional beam search (forward + backward)

Generalize MITM: instead of one fixed BFS-d6 shell on the back side, expand
*both* a forward beam from the scrambled state and a backward beam from
solved. Each step, check for hash overlap. On match, splice.

**Math**: Megaminx diameter ≈ 25–30 moves (estimate). Forward-only beam at
diameter 25, B=131k:  about 25 × 131k = 3.3M state-evals. Bidirectional at
12.5 each:  2 × 12.5 × 131k = 3.3M total — **same** in the worst case. So
where's the win?

**The win is in the BRANCHING tree, not the linear sum**. Beam search prunes
greedily at each step; the prune is more accurate when there are *more known
goals to score against*. With a backward beam expanding from solved, every
step gives you 24× more "good states" to recognize. Forward-side ranking
becomes "how close are you to ANY state in the backward beam" instead of
"how close are you to the single solved state".

In practice: bidirectional cuts hard-puzzle wall by 2–4× when the heuristic
is mediocre, much less when the heuristic is sharp. Unclear which regime
m05 is in. Worth testing.

**Backward beam doesn't need a model**: backward expansion from solved is a
plain BFS — no neural heuristic needed. So the "backward cost" is essentially
free (just gather/hash, no model forward). The forward beam still pays its
model cost.

**Implementation**: ~2 days. Key choice: when forward and backward beams hash-collide,
which direction's path do we take? Both — concatenate forward to scramble + reversed-inverse
backward path to solved.

### 1.3 Symmetry pruning **(60× theoretical, hard to achieve)**

Megaminx has icosahedral rotational symmetry: 60 rigid rotations of the
solid map sticker positions to other sticker positions while preserving the
puzzle structure. Two states `s` and `R(s)` for any such R have *the same
true distance to solved* and are equivalent under search.

**Theoretical win**: dedup on canonical form (= lex-min over the 60-orbit)
shrinks the effective beam by up to 60×. On hard puzzles where the same
"essential" state recurs across rotations, this is a massive prune.

**Two prior failures**: BFS over generator orbits couldn't reach all sticker
positions; adjacency-preserving propagation was too restrictive. Both
attempted to *derive* the 60 rotations algorithmically.

**Direct approach instead**: enumerate. For each candidate permutation P of
the 120 stickers, check `P(solved)=solved AND P · gen · P⁻¹ ∈ generators` for
all 24 generators. The valid Ps are the 60 we want. This is O(120!) brute-
force which is infeasible, but constrained by the fact that P must permute
faces (12!) and within-face stickers in geometrically consistent way. The
search space is ~60 × 1 (with reasonable canonicalization).

**Concrete plan**: parameterize P by (face_perm, within_face_rotation):
- choose any image of face 0 → 12 choices
- given that, choose orientation of face 0 → 5 choices  
- the rest is determined by structure
- gives 12 × 5 = 60 candidates, each verifiable in O(state_size × n_gen) time

Then dedup pipeline: at each beam step, after candidate generation, compute
*canonical hash* = min over R of hash(R(state)). Cost: 60× more hash ops but
hashing is cheap. If pruning kicks in even partially (say 10× orbit
collapse), we get a 5–10× wall reduction on hard puzzles.

**Smallest experiment**: derive the 60 rotations on Kaggle CPU (no time
pressure). Verify. Plug into `_state_hash`. Measure dedup ratio on a single
hard puzzle.

### 1.4 Better hash function for less-collision dedup

We use linear hash `sum(hash_vec * state)`. It's GPU-fast but has nontrivial
collision rate. Two distinct states with the same hash get incorrectly
deduped — they shouldn't be, but they are.

**Diagnostic**: instrument the solver to count collisions (= cases where two
different states have the same hash but only one survives dedup). If
collisions are < 0.1% of dedup events, this isn't worth touching. If > 1%,
it's costing real moves on hard puzzles.

**Fix**: use a real cryptographic-quality hash. xxhash on 120-byte states is
GPU-implementable and ~5× the cost but ~2^32 less collision rate.

Not high-priority unless diagnostic flags it.

---

## Section 2: Better heuristic (= score smarter, not more)

Reducing model cost or improving model quality. These hit `model_s` directly.

### 2.1 Q-distillation v2 **(10× model speedup if it works)**

Replace V(s) regression with Q(s, a) — one forward outputs all 24 child
scores at once. Beam expansion: 1 model call per parent instead of 24 per
parent. **10× theoretical model_s reduction.**

**Why m06 (our first attempt) regressed**:
1. Distilled from m07 (training MSE 64) — teacher was too soft to give sharp
   ordering.
2. Same student arch as teacher — student couldn't compress 24-output head
   without losing precision.
3. MSE loss on values — but we only care about *ranking*. A small absolute
   error in V flips ordering in tight cases.

**Fix recipe for v2**:
- Teacher = **m05** (Bellman-refined, much sharper).
- Student = wider arch: hidden_dims=(2048, 2048), 4 res blocks, +24-output
  head with 8192 dim before output. ~12M params (2× m05).
- Loss = combination: 0.5 × MSE(student_Q, teacher_V_children) +
  0.5 × KL(softmax(-student_Q / T), softmax(-teacher_V_children / T)) where
  T is a temperature making the softmax peaked. KL term enforces ranking.
- Training data: random walks (cheap, diverse) + BFS-d6 sampled states
  (exact labels for boundary).
- 2000 epochs on 4090 (~6h training).

**Smallest experiment**: train tiny student (1M params), 100 epochs, distilled
from m05. Run strat-5 eval. If solve rate ≥ 45/51 (close to m05's 50/51),
proceed to full training.

**Risk**: 30% chance we lose ranking again and abort. But the upside is
enormous — would let us run beam 524k in the wall time of beam 131k.

### 2.2 Smaller-model-then-bigger-model "cascade"

Use a fast 1M-param distillation in the first half of beam steps (when the
beam is wide and any rough ordering works), switch to m05 (6M params) for
the last half. **~4× speedup on model_s** if model wall is bottleneck.

**Why this might work**: at high search depths the beam is deep and "good
states" are rare — accurate ranking matters most. Early on, the beam is
mostly wandering; rough ranking is fine.

**Smallest experiment**: train tiny model, run beam with switch at step 30
of 60, compare to m05-only. Acceptance: ≤ +1 mean path length, ≥1.5× speed.

### 2.3 Ensemble heuristic (m05 + m07 + m17)

At each step, score with K models, take min (or mean) for ranking. K-1× more
model wall but better ranking → fewer wasted beam expansions.

**Math**: K=3 ensemble means 3× model time per step. If ensemble is even 20%
better at ranking, the beam might solve in 20% fewer steps → 0.8 × 3 = 2.4×
overall slowdown. **Probably a loss for wall time.**

**But for QUALITY**: ensemble might solve puzzles single models can't.
Worth on the residual unsolved tail (say, the ~5–10 hardest puzzles after
single-model). Not as a main strategy.

**Smallest experiment**: ensemble on pid 492 alone. If it solves where m05
alone failed, run on the post-Phase-B unsolved set.

### 2.4 Admissible heuristic clipping

Replace `h(s) = model(s)` with `h(s) = max(model(s), bfs_lower_bound(s))`.
- For states in BFS-d6 shell: bfs_lower_bound = exact distance.
- For states outside: bfs_lower_bound ≥ 7 (we know they're not within 6).

This is the `pdb_lookup` hook in our existing solver — never used. It makes h
admissible-ish, which would enable IDA* if we ever go that direction.

**Wall impact**: might prune some weak candidates earlier. Marginal but
free if we already have BFS-d6 loaded.

---

## Section 3: Better mechanics (= run the same algorithm faster)

Pure engineering. No correctness change.

### 3.1 Multi-puzzle parallel beams **(2–4× free)**

The model forward call is shape-stable, and the GPU is underutilized between
puzzles. Pack K puzzles' beams into one stacked tensor `(K * B, S)`.

**Per-step changes**:
- Neighbor gen: `(K, B, n_gen, S)` flatten to `(K*B*n_gen, S)`. Same code.
- Hash: same.
- Dedup: per-puzzle. Need a per-row puzzle-id tag, dedup hashes within tag.
- Model forward: ONE big call on `(K * B * n_gen, S)`. K× larger, but K× fewer launches.
- Top-K: per puzzle. K argsorts on `(B * n_gen,)`.
- Termination: per puzzle. Alive mask drops puzzles as they solve.

**Speedup**: ~2× at K=2, ~3× at K=4 (Python overhead diminishes; model time
scales near-linearly with batch). VRAM: comfortable up to K=4 on 24GB cards.

**Quality**: zero loss (each puzzle's beam is independent, just batched).

**Implementation effort**: 1–2 days. Test against single-puzzle baseline on
the 12 lab samples.

### 3.2 `torch.compile` with fixed beam shape

Project CLAUDE.md warns against compile because variable beam batch sizes
trigger recompile loops (5.8× slowdown observed in earlier tests). The fix:
**always pad the beam tensor to exactly B** even when alive states < B. The
extra rows are just wasted work, but no recompilation.

**Math**: pad-cost = (B - alive) / B × per-step time. If beam typically runs
near B, pad-cost is small (< 5%). Compile speedup: 1.3–1.8× on the hot loop.
Net: 1.2–1.7× wall reduction.

**Implementation**: 0.5 days. Add a `pad_to_beam=True` flag, fill with copies
of the worst-scoring state (or a canonical "dead" sentinel hashable to a
sentinel value).

### 3.3 CUDA Graphs

Same precondition (fixed shape) as compile. Capture one beam-step into a
graph; replay each step. Lower overhead than compile (no Python re-trace).

**Speedup**: 1.2–1.5× over eager. **Stacks with compile** if you do both?
Usually one or the other.

**Implementation**: 1–2 days. More fiddly than compile because of dynamic
state (alive mask, blacklist).

### 3.4 Larger `internal_batch_size`

We default to 16384. On a 24GB L4 we could go to 32k or 65k. Each beam step
does fewer Python loop iterations.

**Speedup**: 1.1–1.3× from reduced launch overhead. Watch VRAM headroom.

**Implementation**: trivial — just bump the default and benchmark.

### 3.5 Custom CUDA permutation kernel

`_get_neighbors` does an `expand + gather` to apply 24 generators per state.
A fused CUDA kernel `out[i, g, j] = states[i, generators[g, j]]` would be
2–3× faster on this op alone.

**Wall impact**: neighbor_s is maybe 15% of total → 8–12% wall reduction.
Real engineering — write the kernel, validate against torch.gather, plumb in.

**Effort**: 3–5 days. Probably not worth unless other Tier S ideas are
exhausted.

### 3.6 fp16 vs bf16 model

We use bf16 for inference. fp16 has more mantissa bits (10 vs 7), more
ranking precision in the value head. Possibly slight quality gain.

**Speedup**: same on Ampere/Ada — no hardware advantage either way. So this
is purely a quality knob.

**Smallest experiment**: A/B at beam 65k on the lab samples. Trivially easy.

### 3.7 int8 model quantization (PTQ)

Quantize m05 weights to int8 post-training. Model forward 2–3× faster on
GPUs with int8 tensor cores.

**Quality risk**: substantial. Picky calibration needed. DeepCubeA-style
work has succeeded with int8, but the value head is sensitive — small
perturbations flip ordering on tight cases.

**Effort**: 2 days. Worth a try only after Tier S is done.

---

## Section 4: Algorithmic alternatives (= different paradigm)

Bigger lifts. Not "tune beam search"; "replace beam search".

### 4.1 IDA* with neural heuristic

DeepCubeA used this. Iterative-deepening A* with f = g + h. No memory blowup
(unlike A*). Standard for cube-style puzzles.

**Why we haven't tried**: requires h to be reasonably tight (admissible
preferred). Our h overestimates in places, so IDA* might re-expand a lot.
Could augment with admissible clipping (Section 2.4).

**Speedup vs beam**: theoretical optimality vs greedy pruning. In practice
on cubes: similar wall, sometimes shorter paths. Not a 10× thing.

**Effort**: 1 week. Reasonable side-experiment but not a near-term win.

### 4.2 Macro moves from BFS-d6

Treat each 6-move BFS path as a "macro-move" in the beam expansion. Branching
factor goes from 24 to 24 + ~M (number of macros). Path length D becomes
D / avg_macro_length ≈ D / 4.

**Wall impact**: more children per step (slower per step) but fewer steps
overall. Net: depends entirely on macro coverage.

**Risk**: many macros will be "useless" (don't reduce h) but you still pay
to score them. Need a smart filter.

**Effort**: 1 week. Probably not worth in this competition timeline.

### 4.3 Stochastic / sample-based beam

Instead of top-B by greedy h, sample B states with prob ∝ exp(-h(s)/T).
Avoids local minima. DeepCubeA used a temperature schedule.

**Effect**: marginal on Megaminx — our beam already has enough exploration
via the wide width. Maybe helps on the 1–5 puzzles that current beam fails.

**Effort**: 0.5 days. Easy A/B but probably small win.

### 4.4 Restart from best leaf, not initial state

If beam search runs out of steps without solving, restart from the best leaf
seen (lowest h) instead of the initial state. Effective depth doubles for
free, with already-validated mid-state.

**Caveat**: "best leaf" may be a stuck state. Combine with stagnation
detection.

**Effort**: 0.5 days. Easy and worth trying for puzzles that timeout.

---

## Section 3.5: Micro-optimizations applied 2026-04-26

The first round of micro-optimizations from a code-level review landed in the lab.
All preserve quality (no algorithmic change). Listed in implementation order.

### Applied
| # | change | file | wall impact (estimated) |
|---|---|---|---|
| 1 | `torch.argsort(value)[:B]` → `torch.topk(value, B, largest=False, sorted=False)` | `beam_search.py`, `beam_search_batch.py` | -10–30% on `topk_s` |
| 2 | `torch.no_grad()` → `torch.inference_mode()` in `_model_predict` | `beam_search.py` | -3–8% on `model_s` |
| 3 | Tree backpointers GPU-resident (int8 moves, int32 parents); single CPU copy at the end | `beam_search.py`, `beam_search_batch.py`, `beam_search_mitm.py` | **-10–30%** on total — removes 1 CUDA sync per step |
| 4 | Reuse already-computed neighbor hashes for stagnation log (return them from `_do_greedy_step`) | `beam_search.py` | -1–3% on `hash_s` |
| 5 | Implicit `parent = idx1 // n_gen`, `move = idx1 % n_gen` | `beam_search.py`, `beam_search_batch.py` | small; saves ~50 MB allocator pressure per step |
| 6 | Disable `model.inference_chunk_size` (single-level outer chunking) | `beam_search.py::setup_model_for_inference` | -5–15% on `model_s` |

Stacked expected wall reduction: 30–50% on baseline beam=131k. To be confirmed
when GPU is free.

### Pending (next round)
- **Incremental hashing via delta** (Section 1.4 generalization): each Megaminx
  generator changes ~20 of 120 stickers. Compute child_hash from parent_hash via
  delta over changed positions only → 5× fewer ops on the hash phase, plus avoids
  allocating the full `(B, n_gen, S)` neighbor tensor (~377 MB at B=131k). Real
  wall impact ~3–5% but big memory savings let us push beam higher.
- **Streaming model scoring** (Section 1.5): score candidates chunk-by-chunk
  storing only `(value, candidate_id)`, materialize only the top-B chosen states
  at the end. Memory-only win; defer until pushing beam past 131k on 16 GB cards.

### Tier-2 idea added: hybrid Q-student (shortlist + rerank)

Standalone Q-distillation (m06) lost ranking → bad solve rate. As a *shortlister*
(top-α·B with α∈[2,6]) followed by teacher reranking, the Q-head only needs
**high recall of the teacher's top-B**, not perfect ranking.

Math: teacher cost per step drops from 24·B candidates to α·B → ~6× cheaper at
α=4. Quality preserved if recall ≥ 99%.

Procedure:
1. Train Q-head from m05; any architecture (a 2M-param student is plenty).
2. Held-out test: for each (parent, top-B-from-teacher), what fraction is in
   top-α·B-from-Q? Sweep α.
3. If recall ≥ 99% at small α, plug into the beam loop.

This is the cleanest path to the 10× beam speedup without re-attempting the
ranking risk of m06. Worth a dedicated training cycle once the local 4090 frees.

---

## Recommended GCP experiment sequence

Budget: ~$30 GCP, ~24–48h available after m17 finishes (~tomorrow ~14 UTC).

### Phase α — confirm where time goes (1 hour, free)

Run `run_benchmark.py` baseline on the 12 lab samples with m05, beam 131k.
Read the per-stage breakdown. Decide: is `model_s` 50–70% as guessed?

### Phase β — fastest meaningful win (12h, ~$8)

**Implement multi-puzzle parallel beam (Section 3.1) on local 4090**. Test
on lab samples. If 2–3× wall reduction with no quality loss, deploy on GCP
for a full-1001 solve at K=4. Output: `phase_b_parallel.csv`. Submit + LB
check.

Why first: zero quality risk, big wall savings, blocks nothing.

### Phase γ — MITM full-1001 (10h, ~$7)

Run `MitmKhoruzhiiSolver` over all 1001 with m05 + BFS-d6 shell. This is
the cleanest "search less" experiment we've never run end-to-end. Combine
with multi-puzzle batching from Phase β if it lands.

Output: `phase_b_mitm.csv`. Submit. Compare per-bucket avg path lengths to
non-MITM result.

### Phase δ — beam 524k on residual hard puzzles (8h, ~$6)

Identify the worst-50 by current best path length. Run beam 524k (4× wider)
on just those, with whatever Phase β/γ optimizations landed. Submit.

### Phase ε — Q-distillation v2 (12h training local, no GCP)

Train v2 student on 4090 in parallel with above GCP work. If strat-5 ≥ 48/51,
deploy on GCP for a full 1001 solve at much higher beam. If it lands, this
is the path to **70K-range scores**.

### Anti-recommendations (do NOT spend budget on)

- **CUDA Graphs / custom kernels** before profiling confirms they'd hit something.
- **A* / IDA*** — too much code for the LB-improvement in our timeframe.
- **Macro moves / symmetry derivation v3** — research projects; do as side-bets after primary plan, not blocking.
- **More Bellman rounds** (m17 told us r2 was a no-op).
- **More Transformer attempts** (m18 told us infeasible at our wall budget).

---

## Quality gate (mirror project's main acceptance gate)

For ANY beam-search change to be accepted:
1. Strat-5 (51 puzzles, fixed seed=0) wall time strictly lower than baseline.
2. Median path length ≤ baseline median + 1.
3. No individual puzzle path ≥ baseline + 5.
4. No new "found=0" rows.

If those hold, port to `src/cayley/khoruzhii_search.py`, re-run a full strat-5
on the production pipeline, then full 1001.

If they don't hold, don't ship. Even if "average" is fine, single-puzzle
regressions cost moves on the leaderboard.

---

## Summary table

| idea | tier | speedup | quality risk | effort | depends on |
|---|---|---|---|---|---|
| Profile baseline | 0 | — | — | 1h | nothing |
| Multi-puzzle batch (3.1) | S | 2–4× | none | 1–2d | profile |
| MITM with BFS-d6 (1.1) | S | 1.5–3× hard puzzles | none | 0.5d | code exists |
| Larger `internal_batch_size` (3.4) | A | 1.1–1.3× | none | trivial | nothing |
| `torch.compile` fixed shape (3.2) | A | 1.3–1.8× | none | 0.5d | profile |
| Q-distillation v2 (2.1) | A | 10× model | medium | 1d | training time |
| Bidirectional / d=8 shell (1.2) | A | 2–4× hard | none | 2d | hash storage |
| Smaller-then-bigger cascade (2.2) | B | 2–3× | low–med | 1d | distill student |
| Symmetry pruning v3 (1.3) | B | 5–60× | low | 3d | derive 60 R's |
| Better hash (1.4) | C | small | none | 1d | profile diagnostic |
| Restart from best leaf (4.4) | B | 1.0–1.2× | none | 0.5d | nothing |
| Custom CUDA kernel (3.5) | C | 1.1× | none | 5d | profile |
| IDA* (4.1) | C | unclear | low | 7d | research |
| Macro moves (4.2) | D | unclear | medium | 7d | research |
| int8 quantization (3.7) | D | 2× model | high | 2d | nothing blocks |

(Tiers: S = ship-now, A = ship-soon, B = side-bet, C = if-time-permits, D = probably skip.)
