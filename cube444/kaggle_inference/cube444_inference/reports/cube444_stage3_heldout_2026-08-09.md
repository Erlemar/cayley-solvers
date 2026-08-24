# CayleyPy 4x4x4 — Stage 3 (Q-Bellman), deeper anchors, and a held-out gating set

**Run date:** 2026-08-09 · **Box:** A100-SXM4-80GB (PG509-210), 22 cores, 176 GB RAM
**Code:** `~/cube444_tf` · **Logs:** `~/cube444_tf/runs/`
**Warm start for every arm:** `runs/s1L/step004000` (the 944-scoring checkpoint)

> Separate experiment from `cube444_transformer_results_2026-08-09.md`, which was run on a
> different devserver. See §7 for a side-by-side.

---

> **UPDATE, later the same day.** Three search-side levers from the handoff plan were then
> measured (§6c). One of them — the **transformer+ResMLP blend**, which was available the whole
> time and which every number in this report ran with *switched off* — is worth more than the
> entire training chain: **s3 + blend + exact endgame scores 2851 on the held-out 54 against
> s3-alone's 2931**, with merge gain 4 vs 2. Where this report and §6c disagree, §6c is newer.
> The packaged inference bundle is `~/cube444_inference.tar.gz`.

## Headline

1. **Stage 3 (Q-Bellman) is the best arm measured — and the 18-pid gating set ranks it last.**
   On 54 held-out pids it scores 2931 (+12.3% over floor) against the previous best 2959 and
   the shipped model's 3011. On the 18-pid set it scores 966, *worse than shipped*. Gating on
   the 18 alone would have discarded it.
   *(Superseded as "best measured" by s3 + blend + endgame at 2851 — see §6c. s3 remains the
   best **scorer**; the blend is a search-side change on top of it.)*
2. **The beam is exactly deterministic.** `orig` = 964/964/964 and `s1L_4k` = 944/944 across
   beam hash seeds. There is no measurement noise to average out.
3. **The 964 → 944 improvement is real and generalises**, at ~83% of its magnitude, on pids it
   was never selected against.
4. **Deeper exact anchors are actively harmful** — a second, independent confirmation.
5. **Longer training is dead.** 40k extra steps moved the held-out total by 2 moves.
6. **Anchors at depth D need a BFS table complete to D, not D+1.** This drops d≤6 anchors from
   infeasible to a one-minute build. Level 6 of the cube measured exactly for the first time:
   **63,542,526**.

---

## 1. The measurement problem, and why it was worth an hour

The prior sweep produced 944 from five checkpoints of **one unmodified run** scoring
944 / 976 / 980 / 974 / 960 — a 36-move spread with no trend, larger than the 964→944 delta
being built on, with 944 chosen as the argmin over 7 arms. That is the shape of selection on
noise, so it was tested two ways before anything was built on top of it.

### (a) Measurement variance is exactly zero

`bench_beam.py` gained `--search-seed` / `--seeds` plumbing. The beam's Zobrist hash vector is
seeded from it, so a different seed re-rolls dedup and top-k tie-breaking.

| arm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| orig | 964 | 964 | 964 |
| s1L_4k | 944 | 944 | — |

Identical. At ~1.7e8 states hashed the 64-bit collision rate is ~0.08%, so the hash never
actually changes an outcome. **Replicating seeds is pure waste**; the 36-move checkpoint spread
is real variation in the training trajectory, not measurement error. That makes weight EMA the
right response rather than more repeats.

### (b) The improvement generalises — so the real risk was pid selection, not noise

A held-out set was built: **54 pids at depth 48–49**, disjoint from the stratified 18 (STRAT)
and from the default hard 18, never used to select anything.

| arm | STRAT 18 (floor 820) | HELDOUT 54 (floor 2609) | ties vs floor |
|---|---|---|---|
| orig | 964 (+17.6%) | 3011 (+15.4%) | 1 |
| s1L_4k | 944 (+15.1%) | **2959 (+13.4%)** | 4 |

Scaling the STRAT edge predicts 2949; actual is 2959. **83% of the magnitude survives**, and
floor-ties go 1 → 4 on a discrete metric that averaging cannot flatter. The edge is real, with
a modest amount of selection bias as expected from an argmin over 5 checkpoints on 18 pids.

---

## 2. Results

All arms warm-started from `runs/s1L/step004000`. B=65536, 110 steps, 1 attempt, mlp_weight 0.

### STRAT 18 (floor 820) — the set every prior decision used

| arm                   | total   | vs floor | wins | merge gain |
| --------------------- | ------- | -------- | ---- | ---------- |
| s1L_4k *(prior best)* | **944** | +15.1%   | 0    | 0          |
| long_raw              | 950     | +15.9%   | 0    | 0          |
| d6a_ema               | 960     | +17.1%   | 0    | 0          |
| orig *(shipped)*      | 964     | +17.6%   | 0    | 0          |
| **s3 (Bellman)**      | 966     | +17.8%   | 1    | **2**      |
| long_ema              | 968     | +18.0%   | 0    | 0          |

### HELDOUT 54 (floor 2609) — never used for selection

| arm                     | total    | vs floor   | wins | ties | merge gain |
| ----------------------- | -------- | ---------- | ---- | ---- | ---------- |
| **s3 (Bellman)**        | **2931** | **+12.3%** | 1    | 3    | **2**      |
| long_ema                | 2947     | +13.0%     | 0    | 5    | 0          |
| long_raw                | 2957     | +13.3%     | 0    | 5    | 0          |
| s1L_4k                  | 2959     | +13.4%     | 0    | 4    | 0          |
| orig                    | 3011     | +15.4%     | 0    | 1    | 0          |
| d6a_ema *(d≤6 anchors)* | 3023     | +15.9%     | 0    | 0    | 0          |

**The two sets disagree on ranking.** s3 is last on STRAT and first on HELDOUT. STRAT
correctly ranked orig vs s1L_4k (a 2.5% gap), but among the four new arms — all within 1.9% of
each other — its ordering is essentially inverted. **18 pids cannot resolve differences under
about 2%.** Every "which checkpoint is better" call in this project made on 18 pids at a
sub-2% margin should be treated as unsupported.

---

## 3. Stage 3 — Q-Bellman refinement

`bellman_q.py`, 6000 steps, batch 768 (384 rw + 384 path), 256 anchors/step from the new d≤6
set, anchor_weight 2.0, lr 2e-5, target refresh every 500. 21 min.

**This contradicts the earlier finding that Q-Bellman cannot fix depth saturation.** That run
bootstrapped a near-scratch standalone model, so the target net was flat at depth and the
recursion `1 + min_a' Q_target(child)` had nothing to propagate outward. Here it warm-starts
from a trained stage-1 checkpoint and is grounded on 67M exact anchors, and the difference
shows in the one diagnostic that matters for collapse:

- `E[target]` held at **12.8 → 14.5** across 6000 steps. It did not collapse toward 0, which is
  the documented failure mode of the recursion (Q ≡ 0 is also a fixed point).
- Loss 1.21 → 0.44; bellman term 0.79 → 0.31; anchor term 0.21 → 0.07.
- deep_score 0.8475 → 0.8569 peak (0.8515 at the end — still not a valid gate, see §5).

**It is the only arm in this study that beats the 46,662 floor on a pid, and it does so on
both sets** (wins 1 / merge gain 2 on each). That is the metric the ensemble actually banks.

---

## 4. Deeper anchors — built, then convicted

### The premise that was wrong

`build_anchors.py` requires a BFS table complete to D+1 for anchors at depth D, and asserts
every child is found. **That is a conservatism, not a requirement.** A child of a depth-D state
sits at D−1, D or D+1. If the table is complete to D, a child that *misses* the lookup is at
depth > D by completeness and ≤ D+1 by adjacency — hence exactly D+1. **The miss is the label.**

This costs a factor of ~19 in table size and was the entire blocker on going deeper.

### Cost table (levels 0–5 known, 6 now measured, 7+ extrapolated at ratio 19.18)

| d≤D | level | cumulative | hashes (64b) | one-pass expansion | collide |
|---|---|---|---|---|---|
| 4 | 172,914 | 182,407 | 0.00 GB | 0.4 GB | 0.00% |
| 5 | 3,316,744 | 3,499,151 | 0.03 GB | 7.6 GB | 0.00% |
| **6** | **63,542,526** | **67,041,677** | 0.54 GB | 146.6 GB | 0.01% |
| 7 | ~1.22B | ~1.29B | 10.3 GB | 2.8 TB | 4.49% |
| 8 | ~23.4B | ~24.7B | **197.5 GB** | 53.9 TB | 100% |

**d7–d8 as originally asked is off this machine.** d7 anchors need a complete-to-7 table whose
level-7 expansion is 146 GB in one pass (needs two-pass + 128-bit hashing). d8 needs
complete-to-8: 198 GB of hashes against 176 GB of RAM, with a guaranteed 64-bit collision. A
collision does not crash — it silently assigns a wrong *exact* label, the one thing the recipe
treats as ground truth.

**d≤6 is now a one-minute build.** `build_anchors_deep.py` (chunked BFS + exclusion labelling)
produced `data/q_anchors_d6.pt`: **67,041,677 states × 24 = 1.61B exact columns**, stored as
uint8 (targets are 0..7) so the set is 6.4 GB + 1.6 GB and stays GPU-resident.

Verified two independent ways:
- Mean target by depth is `d + 0.901` for **every** d1–d6. The exclusion-labelled level lands
  exactly on the progression set by the five table-labelled levels.
- That offset implies 1.19 of 24 moves reduce depth, so 95.04% of a d6 state's children sit at
  d7. Measured exclusion rate among d6 anchors: 90.1% / (63.54M/67.04M) = **95.06%**.

### And then it lost

| | STRAT 18 | HELDOUT 54 |
|---|---|---|
| init (s1L_4k, no anchors) | 944 | 2959 |
| d6a_ema (anchor_batch 512, d≤6) | 960 (+16) | **3023 (+64)** |

d6a is the **worst arm measured**, worse than the shipped model, and it is consistent across
both pid sets. This is the **second** independent negative for anchors: the `tmx` arm
(anchor_batch 512 at d≤4) scored 988 against s1's 970 on its matched set.

Two arms, two anchor depths, two pid sets, all negative. Combined with the tetraminx
post-mortem — where path labels and the sorted-profile family were also measured rejected — the
pattern is that **adding label sources to a working sparse-Q objective keeps losing on this
puzzle.** Anchors should be treated as refuted for cube444 unless someone has a mechanism
argument for why depth-5-and-shallower exactness should help a beam whose states sit at ~44.

---

## 5. Longer training, EMA, and training speed

### Longer is dead

| | HELDOUT 54 |
|---|---|
| init (4k steps) | 2959 |
| +40,000 steps, raw | 2957 |
| +40,000 steps, EMA | 2947 |

Two moves over 0.95 GPU-hours of extra training. This matches the tetraminx result (beam total
flat from epoch 1200 to 2000) and the s1L sweep's own shape, where the *earliest* checkpoint
(4k) beat 8k/12k/16k/20k. **The training-length axis is closed.**

EMA (decay 0.999) is worth about 10 moves on held-out (2947 vs 2957, 0.34%) and −18 on STRAT.
Below both sets' resolving power: **neutral, not a win.**

### Training speed — profiled, one real lever

`prof_train.py`, batch 4096. The step is 130.4 ms:

| component | ms/step | share |
|---|---|---|
| model fwd+bwd+opt | 127.4 | **98%** |
| random-walk sampler | 1.9 | 1% |
| AdamW `fused=True` | 127.5 | a wash |
| **torch.compile** | **81.8** | **1.56x** |

- Throughput is flat from batch 4096 → 16384 (32.1k → 33.4k pivots/s): the GPU is already
  saturated and a bigger batch buys nothing.
- The sampler's sync-heavy rejection loop is 66% of a pool refill but **1% of the step** — not
  worth touching.
- `--compile` is applied and confirmed live at **85.2 ms/step (1.53x)**.

### The "2.9x input-stage speedup" — tested, and it does not transfer

The tetraminx study (`RESMLP_VS_TRANSFORMER.md` §5) found the embedding gradient was **67% of
the whole training step** and that recasting the input stage as `one_hot(vals) @ table` took it
450 → 162 ms. The mechanism is real and the setup here looked *more* vulnerable, not less: an
embedding gather's backward is a scatter-add, and at batch 4096 this model contends 688k
gradients over an **18-row** table where theirs had 88.

It was implemented (`fold_input`, `qtrain/model.py`) and benchmarked (`bench_fold.py`).

**Implementation note.** Their fold pushes `piece_projection` into the value table, which
changes the parameter set. That is not usable here — every checkpoint has to load into the
bundle's solver with `strict=True`. Instead the folded table `T[c + C·j] = emb[c + C·j] @ Wⱼᵀ`
is **rebuilt from the live parameters each forward**, so autograd still routes gradients into
`local_value_embedding` and `piece_projection` separately and the `state_dict` is untouched.
Exact, with TF32 off: forward max rel **3.8e-7**, **argmin agreement 1.0000**. (Gradients differ
by 1.3e-4 relative, identically for both parameters — fp32 accumulation order between a
scatter-add and a GEMM, not an algebra error.)

| | ms/step | speedup |
|---|---|---|
| eager | 125.6 | 1.00x |
| eager + fold | 118.4 | **1.06x** |
| compile | 82.4 | 1.52x |
| compile + fold | 77.7 | **1.62x** |

**1.06x, not 2.9x.** The freeze-the-input-stage diagnostic — their own localisation method,
repeated here rather than trusted — says why:

    input stage trainable   126.0 ms
    input stage FROZEN      121.3 ms   ->  the input-stage gradient is 3.7% of the step

**3.7%, not 67%.** Amdahl caps the achievable win at ~1.04x, and the measured 1.06x (which also
picks up the vanished 704 MB intermediate in the forward) is already at that ceiling. Their
number was measured on a laptop 4090 against a 16 GB memory wall; on an 80 GB A100 the
scatter-add never becomes the bottleneck, so the finding is machine-specific, not model-specific.

**Where it is still worth having:** inference. Forward-only, 38.64 → 35.77 ms at batch 4096 and
151.17 → 139.40 ms at 16384, a consistent **~8%**. Against the ~91 GPU-hour estimate for a full
1043-pid 1M-width pass that is ~7 GPU-hours, for a change that is provably the same function.
Enable it in the beam; it is roughly free in training.

---

## 6. What changed in the code

| file | change |
|---|---|
| `build_anchors_deep.py` | **new.** Chunked BFS past level 5; exclusion labelling; uint8 targets; 128-bit hashing option; `--plan-only` cost table |
| `prof_train.py` | **new.** Step-time breakdown: sampler vs model vs optimiser vs compile vs batch scaling |
| `bench_beam.py` | `--search-seed` / `--seeds`, per-seed and across-seed aggregation with an explicit noise-floor line |
| `train.py` | `--compile`, `--ema` (saves `step<N>_ema` alongside `step<N>`), `fused=True` AdamW, anchors kept in stored dtype |
| `bellman_q.py` | anchors kept in stored dtype (a d≤6 set as int64 would be 51 GB) |
| `qtrain/sampler.py` | `ExactAnchors` casts per batch instead of holding the whole set as int64/float32 |
| `run_stage3_chain.sh` | **new.** The full chain, with the reasoning for the ordering in the header |

Artifacts: `data/q_anchors_d5.pt` (3.5M states), `data/q_anchors_d6.pt` (67.0M states).

---

## 6b. The 1M beam on the 6-pid comparison set

B=2^20 (1,048,576), 100 steps, 1 attempt, mlp_weight 0, no compile — flags matched to the
earlier 1M run. Pids `8,199,399,599,799,999`, floor 241. Note pid 8 is trivial (floor 7); the
other five are 46–48.

| arm | total | vs floor | wins | ties | merge gain | wall |
|---|---|---|---|---|---|---|
| **s3 (Bellman)** | **243** | **+0.8%** | 1 | 3 | **2** | 1493 s |
| s1L_4k | 255 | +5.8% | 1 | 1 | 2 | 1576 s |
| orig *(pretrained)* | 259 | +7.5% | 0 | 0 | 0 | 1605 s |

Per-pid (`*` beats floor, `=` ties it):

```
  pid  floor      orig    s1L_4k        s3   best
    8      7        7=        7=        7=      7
  199     46       50        48        48      48
  399     46       52        52        48      48
  599     48       50        52        48=     48
  799     48       50        46*       46*     46
  999     46       50        50        46=     46
  TOT    241       259       255       243    243
```

Three things worth pulling out.

1. **The ordering is monotone in the training chain and matches HELDOUT 54 exactly:**
   s3 < s1L_4k < orig, at both widths and on both sets. `orig → s1L_4k → s3` each strictly
   improves. That is the cleanest confirmation available that the improvements are real and
   compound, and it is the one setting where the 18-pid gate's disagreement is absent.
2. **The gap widens with width.** At B=65536 s3 was +2 against orig on STRAT; at B=2^20 it is
   −16 on these pids (−6.2%). Consistent with the tetraminx conclusion that the transformer
   earns its keep in the width-capped regime, and it means the 65k gating numbers *understate*
   s3.
3. **s3 alone equals the 3-arm min-merge (243).** It is best-or-tied on every single pid, so on
   this set checkpoint diversity contributes **zero** — the opposite of the earlier finding that
   a per-pid min over 7 checkpoints beat the best single one by 28 moves. Worth re-testing
   before committing to a merge-first plan: the diversity result may have been an artifact of
   having no single strong checkpoint.

Caveat: 6 pids, one of them trivial. This is a comparison point against the other reports, not
a gating set — §2's resolution argument applies with more force here, not less.

---

## 6c. The three search levers from the handoff plan

The handoff's `05_BEAM_SEARCH.md` names four levers this project never ran. Locating them first
turned out to matter more than running them, because they are spread across **three different
solvers** — and none of them live in `solve_ensemble_submission.py`, which produced every number
above.

| lever | implemented in | status |
|---|---|---|
| **blend** transformer + ResMLP | `solve_ensemble_submission.py`, `--mlp-weight` (**default 0.4**) | live all along, and **we had it off** |
| **exact endgame** `tail_bfs_depth` | `Searcher.get_solution`, inherited by `QSearcher` | driver never passed it — 2-line fix |
| **history_depth** | `KhoruzhiiSolver.BeamConfig` | absent here; implemented for this test |
| **V-consistency** `qv_consistency_lambda` | `khoruzhii_search.py:176,342` | **not testable** — gated on `has_value_head`, and our PieceTransformer has none. Would be a silent no-op. Needs a dual-head retrain. |

### Results, s3, B=65536

STRAT 18 (floor 820) then the held-out 54 (floor 2609):

| arm | STRAT 18 | Δ | held-out 54 | Δ | wins | merge gain |
|---|---|---|---|---|---|---|
| s3 standalone | 966 | — | 2931 | — | 1 | 2 |
| + endgame d5 | 966 | 0 | — | | 1 | 2 |
| + endgame d6 | 964 | −2 | — | | 1 | 2 |
| + history_depth 1 | 982 | **+16** | — | | 1 | 2 |
| **+ blend 0.4** | **944** | **−22** | **2859** | **−72** | **2** | **4** |
| **+ blend 0.4 + endgame d6** | — | | **2851** | **−80** | **2** | **4** |

**The blend is worth more than the whole training chain.** `orig → s1L_4k → s3` bought
3011 → 2931 (−80) across three training stages and ~2 GPU-days; switching on a flag that was
already defaulted to 0.4 buys another −72, and the endgame a further −8.

`bench_beam.py` was overriding `--mlp-weight` to 0.0, so **every beam number in this report and
in `cube444_s3_bellman_recipe` is transformer-standalone.** They remain valid as scorer
comparisons — all arms had it off — but they are not the best achievable configuration.

### The 18-pid set inverted the blend verdict too

On STRAT 18 the blend looked like a trap: best mean (944) but **merge gain collapsed 2 → 0**,
which under rule 4 is the metric that matters. On the held-out 54 it does the opposite — **gain
doubles, 2 → 4, wins 1 → 2**. Same failure mode as §1: 18 pids cannot resolve this. That set has
now produced a wrong verdict twice, on s3 and on the blend.

### Endgame: fires, correct, and mostly not needed

`tail_bfs_depth=5` scoring *identically* to 0 looks exactly like gotcha #4's dead flag. It is
not: diffing the solution **strings** shows **7 of 18 paths change, all to the same length**.
The exact tail is computed and substituted; the beam's own tail was already optimal, because at
depth 2–6 the Q head has pair accuracy 0.968 and top-1 0.866. There is nothing there to win.

Depth 6 (67,041,676 states, 291 s build, 26.8 GB) does find something — −2 on STRAT, −8 on the
54 — and it cannot lengthen a path by construction. The better reason to run it is **wall
clock**: it terminates early enough that beam time on STRAT went 451 s → ~241 s, so a full
1043-pid pass gets *cheaper* (~6.7 h + build vs ~7.7 h) despite the extra work.

Depth 7 needs a complete-to-7 table with a 146 GB one-pass expansion. Off this machine.

### history_depth: inverted semantics, measured, harmful

`KhoruzhiiSolver` treats `history_depth=0` as weakest (dedup within the current layer only) and
N>0 as *adding* cross-layer dedup. This searcher accumulates `visited_hashed` over **every**
layer already — permanently at `history_depth=∞`, the strongest setting. So N=1 here is a
**relaxation**, not the CayleyPy feature.

Implemented as a windowed cap and smoke-tested first (gotcha #4): at B=16384 it went 2/3 → 1/3
solves. At full width, **982 vs 966**, 2 pids shorter and 3 longer — a genuine perturbation, not
a no-op. Letting the beam re-select old states costs more width than it buys. Leave it at 0.

### Also worth knowing

`solve_ensemble_submission.py` has **no symmetry ensemble at all** — no `--sym-ensemble`, no
recolour code. The handoff calls it the main quality knob (~−2 moves per doubling of K; 1 frame
solves 67% of pids, 4 frames 100%). It exists only in `KhoruzhiiSolver`. Every absolute total in
this report is therefore from a solver missing the lever the handoff rates highest, and is not
comparable to any number `05_solve.py` produces.

---

## 7. Comparison with `cube444_transformer_results_2026-08-09.md`

Different devserver, different experiment (from-scratch training). Where the two touch, they
mostly agree — and the disagreements are the interesting part.

### Agreements

| claim | that report | this run |
|---|---|---|
| No offline metric tracks beam quality | confirmed 3x | confirmed again — s3 has a middling deep_score and the best beam; d6a has a *better* deep_score (0.8663 EMA) than s3 and the worst beam |
| Checkpoint diversity beats checkpoint selection | per-pid min over 7 ckpts = −28 moves | reinforced: s3 is the only arm with merge gain, and it is the arm a single-metric selection would drop |
| The tetraminx recipe does not transfer | `tmx` lost | `d6a` (its anchor component, scaled up) lost harder |
| Training-speed: embedding gradient dominates | 67% of step, 2.9x available | **does not transfer** — implemented and measured at 3.7% of the step and 1.06x here; see §5. Their number was a laptop-4090-at-a-memory-wall result, not a property of the model |

### Disagreements and corrections

1. **"Second seed on the sweep" (its open item 4) is not needed and never was.** The beam is
   exactly deterministic in the hash seed — 5 runs, zero variance. What that item was reaching
   for is real, but the fix is **more pids, not more seeds**. Its own §"Caveat" and the
   per-pid variance section are pointing at pid-selection error, which seeds cannot touch.

2. **Stage 3 should not have been deprioritised** (its open item 1). The cited evidence — a
   parallel session where Q-Bellman failed to fix depth saturation — does not transfer, because
   that run bootstrapped from near-scratch where the target net is flat at depth. Warm-started
   and anchor-grounded, stage 3 is the best arm here.

3. **Its "Recommended next step" (min-merge diverse checkpoints at 2^20) is still the right
   call, but the shortlist changes.** It proposes `s1`, `s1L_4k`, `s1L_20k`. On held-out pids
   `s3` dominates all three and is the only arm contributing merge gain, so it belongs in any
   merge set; `d6a` should be excluded outright.

4. **Its headline numbers are 18-pid numbers.** Given the rank inversion measured here, any
   ordering it states at a sub-2% margin should be re-checked on ≥54 pids before being
   defended. This does not touch its large-margin results (e.g. transformer vs the 47M MLP).

---

## 8. Recommended next step

**Revised after §6c.** The training axis is closed; the search axis was not even open when this
list was first written. In cost order:

0. **Run everything with `--mlp-weight 0.4` and `--tail-bfs-depth 6` from now on.** This is free
   and worth −80 moves on the held-out 54 — more than the entire training chain delivered. It is
   the packaged default in `~/cube444_inference.tar.gz` (`run_best.sh`).
1. **Sweep `--mlp-weight` (0.2 / 0.4 / 0.6).** ~72 min on the 54. 0.4 is the bundle author's
   default, tuned against *their* transformer, not `s3`. Unswept, and it multiplies into every
   run downstream.
2. **Re-measure the blend at width.** All lever numbers are B=65536. `s3`'s edge over `orig`
   *grew* with width (+2 at 65k → −16 at 2^20); the blend's may not. ~30 min on the 6-pid set.
3. **Bench `runs/s3/best` (step 2000).** ~25 min, never run, and the metric that selected it is
   known to be anti-correlated here.
4. **Then merge-first, at width.** `s3`+blend+endgame, `s1L_4k`, `long_ema` at B=2^20 on the
   deepest pids, per-pid min against the floor.
5. **Retire the 18-pid gate for fine comparisons.** It has now produced a wrong verdict twice —
   on `s3` and on the blend. Use the 54 for anything under ~2%; keep the 18 as a smoke test only.
3. ~~**Take the 2.9x input-stage speedup**~~ **Superseded — it is 1.06x here, see §5.** Turn
   `fold_input` on for beam inference (~8%, ~7 GPU-hours on a full 1M pass) and stop looking
   for a large training-speed win: at 3.7% input-stage gradient and a GPU already saturated at
   batch 4096, `torch.compile`'s 1.53x is essentially all there is. Original wording: it is the
   only remaining cheap lever, and everything it would accelerate is currently flat.
4. **Stop adding label sources.** Anchors are now 0-for-2 and path labels and the sorted-profile
   family are already refuted. The wins here came from *refinement* of the existing objective
   (Bellman), not from new supervision.

---

## Appendix — reproduce

```bash
PY=/home/artgor/cube444_a100_handoff/.venv/bin/python
cd ~/cube444_tf

# anchor cost table; then the d<=6 build (exclusion labelling, ~1 min)
$PY build_anchors_deep.py --plan-only
$PY build_anchors_deep.py --max-exact 6 --anchor-depth 6

# training-step breakdown
$PY prof_train.py

# the whole chain: held-out controls, stage 3, longer+EMA, deeper anchors, both benches
./run_stage3_chain.sh
```

Held-out 54-pid gating set (depth 48–49, floor 2609), disjoint from both 18-pid sets:

```
462,466,488,520,532,570,576,606,616,658,668,696,734,842,864,892,996,59,63,67,71,75,95,
109,115,117,121,123,127,145,149,163,169,175,189,201,207,211,221,229,233,237,267,285,
287,289,295,305,327,345,347,355,359,373
```
