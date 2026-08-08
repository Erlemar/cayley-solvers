# CayleyPy 444 Cube — data analysis + AZ recipe port plan

Competition: https://www.kaggle.com/competitions/cayley-py-444-cube
Deadline **2026-09-22 21:00 UTC** · 24 teams · Community/Kudos
Status as of 2026-07-23: **we have no submission yet.**

---

## 1. What the data actually is

Three files, same shape as the IHES cube competition (`puzzle_info.json`,
`test.csv`, `sample_submission.csv`), but the puzzle is structurally different.

### 1.1 The puzzle

| property | value |
|---|---|
| state size | **96** stickers (6 faces x 16) |
| alphabet | **6 colors, 16 of each** — a *color* cube, **not** a picture/super cube |
| generators | **24** = axes {f, r, d} x layers {0,1,2,3} x {CW, CCW} |
| metric | quarter turns only, every one of the 12 layers turns independently |
| convention | `new = old[gen]`, moves joined by `.` — **verified**, all 1043 sample paths solve |
| target | the exact `central_state` array |

**The single most important structural fact: the state is a *coloring*, not a
permutation.** IHES cube and megaminx both have `solved_state = [0,1,...,N-1]`
and every state is a bijection. Here 16 stickers share each color, so:

- The state does **not** determine the underlying group element — it pins it only
  up to the stabiliser of the solved coloring (the 4 same-colored centers of each
  face are interchangeable: `4!^6 = 191,102,976`). This is a **Schreier coset
  graph**, not a Cayley graph.
- **`invert_state` / NISS is not well-defined here.** Don't port it. (Rule 11
  drops NISS anyway, but on megaminx it was at least *definable*.)
- Model input is tiny: one-hot is `96 x 6 = 576` dims, versus `120 x 120` for
  megaminx. That changes the parameter budget completely (see 4.2).

### 1.2 The test set — a pure difficulty ladder

1043 rows, `initial_state_id` 0..1042, each with a `comment` naming its origin:

- **ids 0..999** — `generated rw, len=N` for **N = 1..1000, exactly one per length**.
- **ids 1000..1042** — 43 `santa 2023 id = 150..199` instances (the Kaggle Santa
  2023 4x4x4 puzzles). Their hamming profile (min 71 / max 87 / mean 80.6) is
  **identical to random**, so they are not a special sub-problem.

`sample_submission.csv` is the **exact inverse of each generating random walk**
(total 525,714 moves). So the trivial baseline is handed to you, and every
puzzle has a known upper bound of `rw_len`.

### 1.3 Random walks mix long before the ladder ends

Mean hamming-to-solved over 4096 parallel walks:

```
len:    1     5    10    16    22    26    30    36    50   100   1000
ham: 16.0  53.2  70.7  77.2  79.2  79.5  79.9  80.1  80.0  80.1   80.1
```

Chance level for 6 colors is `96 x 5/6 = 80.0`. **Walks are fully mixed by
length ~26-30.** Everything from id ~49 onward is statistically a uniformly
random state — the "ladder" is only real for the first ~40 ids.

### 1.4 Graph structure (measured)

BFS from solved, exact:

```
d=0            1        d=3        9,000      branch 19.231
d=1           24        d=4      172,914      branch 19.213
d=2          468        d=5    3,316,744      branch 19.181   (9.6 s)
```

**Branching factor 19.18, dead stable.** (24 generators minus move/inverse
cancellation minus the 4 co-axial layers commuting.)

State count: `8! x 3^7 x 24! x 24!/(4!^6) = 1.7763e47` sticker arrays
(= the familiar `7.4e45` visually-distinct positions x 24 orientations).

**Counting lower bound on the typical optimal solution: `log_19.18(1.78e47) = 36.8`.**
Confirmed the other way: ball(5)=3.5e6 grows past N at d = 36.7. So essentially
every random state needs **>= 37 quarter-slice moves**, and no method can average
below that.

### 1.5 Two facts worth exploiting

**(a) The 96 slots split into exactly 4 closed orbits of 24** — corners,
wing-set-A, wing-set-B, centers. No generator ever moves a sticker between
classes. This gives clean sub-problems: the corner orbit alone is
`8! x 3^7 = 88.2M` states — a *fully enumerable, admissible* pattern database
(~88 MB at 1 byte/entry).

**(b) All 24 whole-cube rotations are usable as symmetries.** `R_f = f0.f1.f2.f3`
etc. generate a group of order 24; for **all 24** there is a color relabelling
that maps the rotated solved state back to `central_state`. So a sym-ensemble of
24 frames is available (rotate slots, then relabel colors). Mirror reflections
would extend this to 48.

---

## 2. What the competition actually rewards

Score = **total moves over all 1043 puzzles**, lower is better.

| entry | score | mean/puzzle |
|---|---|---|
| **Tomas Rokicki (leader)** | **46,298** | 44.39 |
| webmaking | 53,580 | 51.37 |
| DrozdovDan | 54,416 | 52.17 |
| **community automerge (33 subs)** | **54,754** | 52.50 |
| Beam 2^24 CayleyPy-Cube SmallM | 55,642 | 53.35 |
| Beam 2^22 Repeat4 SmallM | 55,854 | 53.55 |
| Beam 2^22 SmallM | 58,514 | 56.10 |
| Beam 2^20 SmallM | 67,380 | 64.60 |
| sample_submission (inverse walks) | 525,714 | 504.0 |

### 2.1 Where the points are

Per-difficulty profile of the community merged 54,754 (verified 1043/1043):

```
rw_len band    n    mean solution length
    1-5        5      3.00        <- optimal-ish
   11-15       5     11.40
   21-25       5     20.60
   26-30       5     28.00        <- still == rw_len: beam not beating the walk
   36-40       5     38.00        <- still == rw_len
   41-50      10     44.90        <- beam starts winning
   51-75      25     53.40
  101-200    100     53.82
  401-700    300     53.99        <- flat plateau
  701-1000   300     53.83
  santa2023   43     54.28
```

- 30 puzzles (rw_len <= 30) contribute **427 moves total**.
- The other **1013 puzzles contribute 54,327**, mean **53.63**, and they are all
  drawn from the same distribution.

**=> The whole competition is "lower the mean solve length on a random 4x4x4
state." 1 move/puzzle = ~1,013 leaderboard points.** Nothing else matters:
the easy tail, the santa puzzles, and post-processing tricks are all rounding error.

### 2.2 The gap, and why width alone won't close it

```
counting lower bound      36.8
Rokicki                  ~45.3   (+8.5 over bound)
community merged         ~53.6   (+16.8)
```

Organiser baseline scaling (hard-band means, same model, same code):

```
beam 2^20 -> 66.0
beam 2^22 -> 57.3    (-8.7)
beam 2^24 -> 54.5    (-2.8)
```

Returns are collapsing by 3x per 4x width. Extrapolating, **pure width with that
model asymptotes around 52** — it cannot reach 45. Note also that
**2^22 "Repeat4" (55,854) ~= single 2^24 (55,642)**: four diverse runs at 1/4 the
width match one wide run. Diversity is ~4x cheaper than width, and we get 24
diverse frames free from the rotation group.

**Conclusion: Rokicki's 45.3 is not a width result.** It is almost certainly a
classical multi-phase solver with large pruning tables. Our realistic band with a
strong learned V + wide beam + sym-24 is **~50-52 (=> ~51,000-53,000)**, which is
2nd place territory. Beating 46,298 would require the structural (multi-phase /
PDB-guided) route, not more of the same.

---

## 3. Does our stack even run on this puzzle?

Yes — measured, not assumed. Everything is already shape-parameterized
(`state_size` and `num_classes` are independent constructor args in
`ResMLPDistance` / `ResMLPGFlowNet`; `cayley.data` derives shapes from
`puzzle.solved_state`; `bellman.py` has no hardcoded sizes; no code assumes
`solved_state == arange(N)`).

Smoke test on the 4090 laptop with a 15-line `Cube444` duck-typed puzzle class:

```
GeneratorTable.from_puzzle    OK  (24, 96)
generate_walks_torch          OK  491,520 states, k_max=60, 0.39 s
```

**The only missing piece was `inverse_name`** on the puzzle class.

### 3.1 Measured costs on the 4090 laptop (RTX 4090 Laptop, 17 GB)

Model sizes at `state_size=96, num_classes=6, encoding=onehot` (in_dim 576):

| hidden / res-blocks | params | train (500k-sample epoch) | inference |
|---|---|---|---|
| (2048, 512) x2 | **3.30M** | 0.63 s | **2.90 M states/s** |
| (4096, 1024) x4 | **15.01M** | 2.21 s | 0.98 M states/s |
| (8192, 2048) x4 | 55.18M | 5.58 s | — |

**Training is essentially free** — a 1000-epoch Bellman run on the 15M model is
~37 minutes. This is a completely different economics from megaminx, because the
576-dim one-hot input makes the first layer cheap.

**Inference is the entire bottleneck.** Beam cost = `B x 24 x ~50 steps`:

| beam B | 3.3M model, all 1043 puzzles |
|---|---|
| 2^16 (65k) | 7.9 h |
| 2^18 | 31 h |
| 2^20 | 126 h |
| 2^22 | 503 h |
| 2^24 | **2,012 h** |

A single 4090 cannot run a competitive width. **This is a TPU job**, exactly like
megaminx and IHES — port the JAX SPMD beam kernel.

---

## 4. The AZ recipe port

Our AZ recipe is a 3-stage pipeline (megaminx `m_curr_v3` -> `m_dd_v0` ->
`m_az_v4`). It ports almost unchanged. What follows is the concrete plan with the
deltas called out.

### 4.0 Stage 0 — infrastructure (half a day)

1. `cube444/src/cube444/puzzle.py` — `Cube444` dataclass, `STATE_SIZE=96`,
   `N_GENERATORS=24`, `MOVE_SEPARATOR="."`. Copy `megaminx/src/megaminx/puzzle.py`
   and **delete `invert_state` / `invert_path`-based NISS** (not well-defined,
   1.1). Keep `inverse_name`.
2. `cube444/scripts/verify.py` — replay a submission CSV against `central_state`.
   (Already validated: sample 1043/1043, community merge33 1043/1043 = 54,754.)
3. **Baseline submission immediately**: submit the community 54,754 as the floor,
   then min-merge every future run into it per-pid (Rule 26 — always compare
   against the n-way per-pid min, never against a single base).
4. Symmetry tables: build `rotations_24.npy` (96-slot perms) + `color_maps_24.npy`
   (the 6-color relabel per rotation). Verified buildable — all 24 work.

### 4.1 Stage 1 — BFS anchor tables (1 hour)

- **Full BFS to d=5: 3,499,151 states, 9.6 s.** Do it, store exact labels.
- **d=6 = ~63.6M states.** Enumerable with chunked expansion (~6.1 GB as
  `uint8[96]`). Store a random **20M subsample** (~1.9 GB) as
  `cube444/data/bfs_d6_train.pt` — same role and size as megaminx's
  `bfs_d6_train.pt`.

These are the exact-label anchors that prevent the `V(V0)~2` bug
(`v_v0_bug_and_anchor_fix`). d=7 (~1.2B) is out of reach — don't try.

### 4.2 Stage 2 — V pretrain (random-walk MSE), ~1 h

`02_train.py` equivalent. **Model config:**

```yaml
model:
  state_size: 96
  num_classes: 6
  encoding: onehot        # <- NOT embedding. 576 dims, lossless, cheaper.
  hidden_dims: [4096, 1024]
  num_res_blocks: 4       # 15.0M params
training:
  n_epochs: 1000
  samples_per_epoch: 500_000
  batch_size: 16384       # bigger than megaminx's 8192 — the model is cheap here
  k_max: 60               # ~1.6x the ~37 diameter (megaminx used 80 for diam ~29)
  n_back: 1               # n_back=40 is a confirmed regression (anti-patterns)
  lr: 5.0e-4
  loss: mse
  amp: true
  compile_model: true     # ResMLP — safe (Rule 22 only bans it for SDPA transformers)
```

**Two size notes.** (a) `encoding: onehot` because with 6 symbols an
`nn.Embedding(6,16)` is strictly worse — 1536 input dims for zero extra
information. (b) Rule 14's "bigger V trunks regress" was measured on megaminx
with a 1920-dim embedding input; here the input is 576-dim and the state space is
`1.8e47`, so 15M is the *starting* point, not a violation. But size it against
**beam throughput**, not accuracy: 15M runs at 0.98 M states/s vs 3.3M at 2.90
M/s. At fixed GPU-seconds, `width x model-size = const`, and section 2.2 says
width is worth a lot. **Train both 3.3M and 15M and pick on
score-at-fixed-wall, not on loss.**

Stage 2 output is only a warm start — its labels are badly biased (a length-60
walk usually lands at true distance ~38).

### 4.3 Stage 3 — Bellman refine (the load-bearing stage), ~2 h

`60_train_admissible.py` / `bellman.py`, mirroring `m_dd_v0.yaml`:

```yaml
bellman:
  warmstart_path: cube444/models/c_v0/epoch_0999.pt
  target_update_every_epochs: 10
  clip_upper: true
  clip_lower: true
  bfs_d6_path: cube444/data/bfs_d6_train.pt
  bfs_d6_fraction: 0.10
  n_anchor_v0: 32
  n_anchor_d1: 4
  lambda_pdb: 0.0         # no PDB yet; it was neutral on megaminx anyway
  frontier_fraction: 0.0  # needs solver traces — enable on the 2nd pass
```

**Acceptance gates before this V goes anywhere near a beam** (these are the
scar-tissue gates, and they matter more here than on megaminx because the color
encoding is a new regime):

1. `V(solved) ~ 0` and `V(d=1) ~ 1`.
2. **Saturation, absolute**: `V@d=80` must land in **[36, 44]** — near the real
   diameter — and `V@d=80 - V@d=40 <= 3`. Rule 23 gives the gap test; the
   `dodeca_cnn_rejected` post-mortem adds that a gap test *alone* false-positives
   when V collapses uniformly, so **the absolute band is mandatory**.
3. **Mid-depth variance**: std of `V` at `d=20` should be small and comparable to
   the Stage-2 model's. `repr_upgrade_bundle_rejected` found V *variance* at
   d~20 is a better beam predictor than the saturation mean.
4. A 10-pid beam bench — **but read Rule 21**: a 10-pid bench is *not* a
   sufficient gate. The binding gate is a stratified ~50-pid eval at the
   production recipe.

### 4.4 Stage 4 — AZ dual-head (`71_train_az_v3.py`), ~1 h

**The policy dataset already exists.** The community merged 54,754 CSV is
downloaded and verified at `cube444/community/submission_54754_merge33.csv`;
expanding it gives **~54,754 (state, action) pairs** — almost exactly the size of
megaminx's `az_dataset_76304.pt` (76K) that produced the AZ v4 breakthrough. So
there is **no chicken-and-egg**: Stage 4 can run the day Stage 3 finishes.
Rebuild the dataset from our own beam output whenever we beat the community
per-pid.

Config, matched to the AZ v4 run (read from `m_az_v4_training.log`):

```
--warmstart-trunk cube444/models/c_bellman/<best>.pt
--policy-dataset  cube444/data/az_dataset_54754.pt
--bfs-d6-path     cube444/data/bfs_d6_train.pt
--rw-batch-size 8192 --policy-batch-size 1024      # <- Rule 13: NEVER take defaults
--alpha 1.0 --beta 1.0 --lr 5e-4
--k-max 60 --n-back 1
--anchor-v0 32 --anchor-d1 4 --bfs-d6-fraction 0.10
--target-update-every-epochs 10
--hidden-dims 4096,1024 --num-res-blocks 4
--epochs 100 --checkpoint-every 5                  # <- 5, not 20. See below.
```

**Be honest about what Stage 4 buys.** On megaminx:

- `m_az_v3` at 100 epochs was **worse** than the pure V (bench 943 vs 871).
- `m_az_v4` **early-stopped at ep24** was the first 6M model to beat pure V
  (strat-5 51/51, mean 87.5 vs 89.4).

So the dual head helps *only* in a narrow early window, and over-training makes
the policy memorize and destroys V calibration (Rule 13). Hence
`--checkpoint-every 5` and a beam bench every ~10 epochs (Rule 12: training loss
is not a proxy for beam quality).

**And the policy head itself is not expected to be usable at inference.** Rule 15
(qshort doesn't compose with a different V), `allneighbor_qhead_rejected`, and
the V-only-wins-on-TPU finding all point the same way. **Stage 4's value is
multi-task regularization of the trunk to get a better V head — nothing more.**
Plan the beam as V-only.

### 4.5 Stage 5 — inference, where the score is actually made

- **V-only beam.** No qshort (Rule 15), no NISS (Rule 11, and undefined here).
- **Sym-ensemble 24.** Use the pooled `K x B` beam over rotated+recolored copies
  rather than K sequential runs — `sympooled_beam_validated` measured pooled
  beating sequential at equal budget. Caveat: on megaminx 6/8 of the pooled wins
  came from *inverse* frames, which **do not exist here** (1.1), so expect a
  smaller gain than megaminx saw.
- **Diversity over width**: the leaderboard's own "Repeat4 at 2^22 ~= 2^24"
  data point says 4 diverse runs beat 4x width. With 24 rotation frames this is
  the cheapest lever we have.
- **TPU.** Port the megaminx/IHES JAX SPMD kernel. 96 x uint8 per state is
  *cheaper* than megaminx's 120, so the B ceiling should be at least as good as
  megaminx's 8M on v5e-8. Note Rule 2: any `torch.compile` for beam inference
  needs the fixed-batch padding hook, which `khoruzhii_search.py` still lacks.
- **Always min-merge per-pid against every CSV we hold** (community 54,754 + all
  our runs) before claiming a win — Rule 26.

---

## 5. Ordering, and what to expect

| no. | step | est. wall | actual | status |
|---|---|---|---|---|
| 0 | `Cube444` class + verifier + symmetry | 0.5 d | ~2 h | **DONE** |
| 1 | BFS d5 exact + d6 subsample | 1 h | **2 min** | **DONE** — 25.3M states |
| 2 | Stage-2 RW pretrain (3.3M) | 2 h | **8 min** | **DONE** |
| 3 | Stage-3 Bellman + gates | 3 h | ~30 min to ep124 | **DONE** (leg 2 extends it) |
| 4 | beam 2^16 sanity on hard pids | 2 h | ~1 h | **DONE** — see below |
| 5 | 15M mid pipeline | — | ~5 h | **DONE** — no capacity gain, deploy 3.3M |
| 6 | Stage-4 AZ dual-head, early-stopped | 2 h | — | queued (optional) |
| 7 | TPU beam **port** + CPU validation | — | ~2 h | **DONE** — `cube444/tpu/` |
| 8 | **v6e provision + wide run w/ sym-4** | days | — | **turnkey; the actual score** |

Training turned out to be far cheaper than budgeted (the whole 3.3M pipeline is
~40 min on one L4), which sharpens the original conclusion rather than changing
it: **this is an inference-budget problem, not a training problem.**

**Measured 2026-07-23** — 12 real hard pids, beam 2^16, sym-ensemble 1:

| config | solve rate | mean length when solved |
|---|---|---|
| ours (c_bells ep124, **1 frame**) | 75% | **70.44** |
| organiser 2^16 **Repeat3** SmallM | 83.5% | 71.88 |

Level with the organiser baseline on length at one third the frames, from ~40 min
of training. The V is sound. The remaining gap to the 54,754 floor (mean 53.9) is
~16 moves/puzzle, and the organiser's own width curve says that gap is worth
about 2^16 -> 2^24 of beam. Confirms the plan's central claim: **buy width, not
model.**

Realistic landing zone: **~50-52 mean => ~51,000-53,000**, i.e. clear of the
community floor and the 2^24 baseline, in the webmaking/DrozdovDan band.

Reaching **46,298 needs a different idea**, not a bigger beam. The two candidates
this analysis surfaces:

- **Corner PDB (88.2M entries, exact, admissible)** from the corner orbit — cheap
  to build, gives a certified lower bound to max-combine with V. Tempered by
  `pdb_corner_findings`: max-combine with a strong V was *neutral* on megaminx.
- **Two-phase / orbit-staged search** using the 4-orbit factorization, reusing
  the `two_phase.py` MaskedV machinery from megaminx. This is the structural
  route and the one most likely to be what Rokicki is doing.

Both are separate bets, to be opened only after the AZ+beam baseline is on the
board.

---

## 6. Gotchas specific to this puzzle

1. **State is not a permutation.** No `invert_state`, no NISS, no inverse frames
   in the sym-ensemble. Don't copy those code paths across.
2. **`num_classes=6`, not `state_size`.** Every model constructor in the repo
   defaults `num_classes` to the state size. Pass it explicitly or you will
   silently build a 96x96 one-hot.
3. **`sat_ceiling` defaults to 30.0** in `BellmanConfig` (megaminx's diameter).
   For this puzzle it must be ~40 if the saturation loss is used at all.
4. **The linear beam hash** (`sum(hash_vec * state)`) still works with values
   0..5, but the dynamic range per slot is 6 instead of 120 — worth a collision
   check at large B before trusting a wide TPU run.
5. **Move separator is `.`** — same as megaminx/IHES. `path.split()` on
   whitespace silently returns 1 "move" per row and produces nonsense totals.
