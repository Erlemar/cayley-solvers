# Professor Tetraminx — state of play

Competition: [cayley-py-professor-tetraminx-solve-optimally](https://www.kaggle.com/competitions/cayley-py-professor-tetraminx-solve-optimally)
Deadline **2026-08-29 22:00**. Community comp, Kudos. 26 teams.

## 2026-08-27 UPDATE -- read this before anything below it

**EVERYTHING BELOW IS STALE ON SCORE AND TEAM.** Live Kaggle state, checked
2026-08-26: we are team **CayleyPy** at **27,665** (submitted 2026-08-26 03:10 by an
hourly auto min-merge pipeline), #1, with Rokicki second at 28,481 -- a **816-move**
lead. The sections below say "us 28,094" and list CayleyPy as a *rival* at 28,398;
those are the same team now. Two consequences:

* **27.665/pid falsifies this file's own "realistic true optimum ~28"** in the
  puzzle-facts table. The true optimum is by definition <= 27.665; against the 26.7
  counting bound the remaining headroom is <= ~965 moves, upper bound, not a forecast.
* **Nothing in `tetraminx/submissions/` is newer than 2026-08-21**, and the 27,665
  artifacts are not on this machine. Never quote a score from local disk (rule 26b(a)).

### 9.2x FREE ON EVERY LOCAL TRANSFORMER BEAM -- `--chunk-size 4096`

`30_solve.py --chunk-size` defaulted to 32768. The PieceTransformer scores 88 tokens
through SDPA under an explicit `attn_mask`, which falls back to the math kernel and
materializes a `(chunk, n_heads, 88, 88)` score matrix. Peak transient is linear in
CHUNK and independent of beam width -- **8.79 GiB at 32768 vs 1.12 GiB at 4096** -- and
on a 16 GB Windows card the big one does not OOM, WDDM silently spills it to host RAM.
The only symptom is wall-clock:

| chunk | s/step at B=65536 | peak |
|---|---|---|
| 4096 | **1.0** | 1.12 GiB |
| 32768 | **9.2** | 8.79 GiB |

Throughput is flat 2048-8192 (~65k rows/s). Default changed to 4096. Results are
UNCHANGED -- the forward is bit-identical (max_abs_diff exactly 0) across chunk
1024/2048/4096/8192/16384 and `_state_hash` is stride-invariant, because nothing in the
scoring path reduces across rows. Rules 27 + 31, together.

**How it was found:** transformer-only, the 2-model blend, and blend-without-qv all
timed 209.8 s on the same pid -- identical to 0.1 s. A cost that does not move with the
variable is not caused by that variable (rule 28), which pointed at the forward.

### GOTCHA: the test set is ORDERED BY DIFFICULTY at the start

**pids 0-35 are the shallow block** -- mean reference length **15.2** against **28.6**
for pids 40-999 (pid 0 is a 1-move scramble). 34 of the 46 pids with a <=26-move
reference live in 0-35. Any "stratified sample" built as a contiguous low-pid block is
therefore a shallow sample and will report comfortable numbers that mean nothing. Cost:
one 40-pid probe run had to be killed and relaunched. Use a stride across 40-999.

### BEAM-AVI REJECTED (2026-08-26/27) -- full write-up in `BEAM_AVI_PLAN.md`

Approximate value iteration on the Q head over BEAM-frontier states (the cube444
`BEAM_AVI_METHOD.md` recipe, which took that puzzle 4.2 percent below its floor).
Two independent 5-round loops, two gates, ~9 h of beam. **Negative.**

| gate | arms | result |
|---|---|---|
| B=2^18, blend, 6 arms | incumbent 441 | every AVI arm +4 to +6, same per-pid pattern |
| **B=2^20, solo, 3 arms** | **incumbent 427** | r000 (1 rd) 429, 3W/3L/9T, p=1.0; r004 (5 rd) 434, 1W/7L, p=0.07 |

Dose-response descends: one round ties, five rounds lose. Mechanism is **Q-space
flattening** -- deep-band top-1 gap halves (0.438 -> 0.270), cross-parent sd contracts
6.83 -> 6.23, believed distance-reducing children rise 1.90 -> 2.65 against a true
~1.6. Since the beam takes a global top-B across (parent, action), cross-parent spread
is exactly what it runs on. Same family as [[allneighbor_qhead_rejected]].

**The go/no-go probe PASSED** (deep conditioned percentile 0.055-0.102 vs a 0.5 null),
so the failure is not absence of signal -- fitting the bootstrap flattens the landscape
faster than it sharpens it. **The precondition probe is necessary but not sufficient.**

**Cheap test that would have saved ~9 h:** `77_level_check.py` + the variance
decomposition predicted BOTH gate outcomes before either gate ran, in ~2 minutes. Run it
first on any future bootstrapping experiment here. Training loss and `E[target]` both
looked healthy throughout and were both wrong -- again.

New scripts, all reusable: `72_path_rank_probe.py` (cross-parent rank probe),
`73_gen_harvest.py` (free Bellman harvest off the beam), `74_train_avi.py`,
`75_avi_loop.py`, `76_gate_avi.py` (paired matched-control gate with a sign test),
`77_level_check.py`. Plus a read-only `step_probe` hook in
`src/cayley/khoruzhii_search.py`.

## SUBMITTED 28,094 -- 2026-08-09, **#1** (next: CayleyPy 28,398, +304)

**Kaggle public score 28,094**, `submissions/FINAL_tetraminx_28094.csv`, 1000/1000
replay-verified independently of the merge. Leaderboard at submission time:

| rank | team | score |
|---|---|---|
| **1** | **us** | **28,094** |
| 2 | CayleyPy | 28,398 |
| 3 | Chekhlov Dmitrii | 28,456 |
| 4 | Tomas Rokicki | 28,481 |
| 5 | webmaking | 28,843 |

**The field is moving daily now** -- CayleyPy went from absent to 28,398 in one
morning. Re-check the leaderboard before assuming any bar still holds.

### 28,185 -> 28,094 (-91): the SAME kernel's version history, swept a SECOND time

A version sweep is not a one-shot harvest -- it pays again every time the kernel
gets re-pushed. `cayleypy-tetraminx-tpu-beam-q` was swept on 2026-08-05 at 31
versions (worth -51). Four days later it had **119**; pulling v55-v119 (51 held a
`submission.csv`) was worth another **-91 over 72 pids**, with wins spread thin
across v061/v063/v066/v070/v084/v100/v113/v115/v118 and others. The 30-move tail
is what moved: len-30 bucket 148 -> 92, len-29 484 -> 511, len-28 266 -> 289.

Two community kernels swept in the same pass contributed **exactly 0**:

| kernel | versions | beam-quality pids (<=31) | contribution |
|---|---|---|---|
| `artgor/cayleypy-tetraminx-tpu-beam-q` | 119 | 320 / 1000 | **-91** |
| `alexandervc/cayleypy-rw-models2-tetraminx` | 276 | 53 / 1000 (median len 401) | 0 |
| `markcelliott/frames-saturate-at-two-tpu` | 1 | 37 / 1000 (median len 501) | 0 |

That is a genuine null, not a parse failure -- both replay-verify against tetraminx
with 0 bad-alphabet rows (rule 26b(c) check). They are RW-model / short-run
notebooks whose `submission.csv` is mostly long fallback, so **version COUNT is a
bad proxy for merge value**: 276 alexandervc versions were worth nothing while 51
of ours were worth 91. Sweep depth should follow whether the kernel is a deep beam.

Ceiling probe (binary search on the two distinct 404 sources -- `kaggleusercontent`
= version exists / no such file, `api.kaggle.com` = no such version) finds a
kernel's max version in ~10 requests instead of scanning blind.

### 28,359 -> 28,308 came ENTIRELY from old Kaggle kernel VERSIONS (-51)

Not from the beam. `cayleypy-tetraminx-tpu-beam-q` has 31 versions; 15 still hold a
`submission.csv` and 16 a results JSON. Every `kaggle kernels` read command serves
ONLY the latest, but the SDK's `ApiDownloadKernelOutputRequest.version_number` is
version-addressable (see CLAUDE.md 7f(d), which this REVERSED -- the old rule said
they were stranded forever). Measured:

| source | saving vs 28,359 |
|---|---|
| latest version alone (all `kernels output` can give) | -5 |
| best single version alone (v8) | -7 |
| **union of all versions** | **-51** across 13 versions |

13 versions each held a unique win and they were perfectly additive. Puller:
`scripts/25_pull_kernel_versions.py`, then `90_merge_all.py --extra`.
**Every re-push of a beam kernel leaves recoverable moves behind** -- sweep the
version history of any kernel that was pushed repeatedly with different
seeds/frames/checkpoints.

Also refreshed in that pass: `tetra-tfsweep` had 50 output JSONs vs 42 mirrored
locally (8 never pulled). They added nothing on top of 28,308, but the stale-mirror
failure mode is exactly what rule 26b(a) warns about -- re-pull live boxes every time.

### Earlier same-day history (2026-08-04/05)

| total | source |
|---|---|
| 28,456 | Chekhlov Dmitrii took first place from our 28,467 |
| 28,455 | full-corpus content-filtered min-merge -- pid 761 30->29, from OUR OWN 28,467 |
| 28,447 | first 18 pids of the tier-30 beam sweep (5 wins, -8) |
| 28,426 / 28,387 / 28,359 | tier-30 beam sweep continuing (23 / 50 / 71 wins) |

The 28,455 is the rule-26b lesson landing again: the leader's file was better on 999
rows, and the single row where ours was shorter was worth first place. Always merge
against every CSV that covers the same pids, including our own older ones.

### THE ONLY LIVE LEVER IS THE BEAM, AND IT IS THE TIER-30 SWEEP

The length distribution collapsed: **zero pids at 32+**, 6 at 31, 324 at 30, 370 at 29.
The old yield curve (-1.56/-0.92/-0.28 per pid at len 32/31/30, memory
`beam_yield_graded_by_path_length`) was measured against a much looser floor and its
TIERS NO LONGER EXIST -- do not reuse those numbers. Re-measured 2026-08-04 on the
len>=30 tier with the deployed stack at beam 2M:

```
hit rate 5/18 = 27.8%,  0.444 moves/pid
pid  89: 30 -> 27      pid 112: 31 -> 29      pid 57: 30 -> 29
pid  77: 30 -> 29      pid 431: 31 -> 30
```

Sweep command is `scripts/../run_shard.sh`-style: `30_solve.py --checkpoint
mx_tf_az/epoch_1500.pt --blend mx_resmlp_az.pt --blend-weights 0.8 0.2
--qv-consistency 0.3 --history-depth 1 --sym-frames 2 --beam 2097152 --bf16 --no-merge
--resume`. Harvest with `scripts/63_harvest_shards.py` (replays every row before it is
allowed to lower the floor -- a concurrently-written shard can end in a torn line).

**Next tier is len 29 (370 pids)**, untouched.

### FIXED 2026-08-04: `--blend` silently disabled `--qv-consistency` in PyTorch

`BlendedQ` hard-coded `has_value_head = False`, so the deployed combination
(blend + qv-consistency) ran on the TPU path only and raised SystemExit locally. It now
mirrors `jax_model.apply_qv`: Q blended across members, V taken from the FIRST member
with a value head (NOT averaged -- the members' value heads are not calibrated to a
common scale). Same dual-code-path drift class as the streaming step body.

## POST-PROCESSING: every approach tried, and its verdict

One table so nobody re-derives a dead end. All are exhaustive over the stated scope, all
replay-verify against the original scramble before emitting, all have `--self-test`.

| approach | script | scope actually run | verdict |
|---|---|---|---|
| Exact window lookup, d<=6 | `42_window_reduce.py` | all windows <=6 | 0 -- every such window is geodesic |
| Symmetry-canonical dictionary | `58_symmetry_rewrite.py` | 836k orbits, 48-fold | **0** -- 61.8% of windows found, 0 shorter |
| MITM radius 11 (numpy) | `59_mitm_rewrite.py` | 120 windows, ~4 s/query | 0 (sample too small to conclude) |
| MITM radius 12 (GPU) | `61_mitm_gpu.py` | 166,703 queries, windows 7-15 | 0 -- proves every window <=12 geodesic |
| **MITM radius 13 (GPU)** | **`64_mitm_r13.py`** | **67,235 queries, windows 14-18** | **-8 moves / 6 pids -- THE ONLY WINNER** |
| Cross-trajectory splicing | `60_splice_graph.py` | full corpus, Dijkstra | 0 -- and BOUNDED, see below |
| Knuth-Bendix completion | `62_knuth_bendix.py` | 1,844,967 derived rules | 0 -- subsumed by radius 12 |
| Rewrite alternative paths | `65_rewrite_alternatives.py` | 1,121 alts within +3 | 0 -- sound, no payoff at this yield |
| n-way per-pid min-merge | `63_harvest_shards.py`, `90_merge_all.py` | every source, by CONTENT | **always pays** -- see rules 26 / 26b |

Two results are structural rather than empirical, and they are the ones worth knowing:

* **Splicing is bounded, not merely empty.** A splice through waypoint x beats the
  incumbent only if `cost_from_start(x) + d(x) < incumbent`, because
  `d(x^-1 y) + cost_to_end(y) >= d(x^-1 y) + d(y) >= d(x)`. Over the 6,979 waypoints
  where d(x) is exactly known, ZERO have slack. Adding more trajectories cannot help.
* **KB is subsumed by construction.** Completion converged to `|lhs| <= 9` despite a
  bound of 12, so every rule it derives sits inside what radius 12 already tested
  exhaustively.

The live frontier is radius 13 windows of length 14+, and it is nearly exhausted: 8
moves from 67k queries, with 17-18 windows contributing nothing.

### RADIUS 13 IS THE FIRST EXACT METHOD THAT PAYS -- 8 moves, 2026-08-05

**Corrects the section below.** After radii 10 (leader), 11 and 12 all returned 0, this
file said "exact rewriting is closed / do not build another local rewriter." **That was
wrong by exactly one radius.** `scripts/64_mitm_r13.py` (front B6 x the d7 table,
`d(w) <= 13 iff exists a in B6 with a^-1.w in B7`) found 8 moves over 6 pids on the
28,308 baseline -- the only exact post-processing win in the whole programme:

```
pid  32: window [ 9,25)  16 -> 13   (-3)      pid 303: window [13,27)  14 -> 13
pid  86: window [ 6,20)  14 -> 13             pid 399: window [ 0,14)  14 -> 13
pid 108: window [15,29)  14 -> 13             pid 759: window [ 4,18)  14 -> 13
67,235 queries over 4 shards -> 0 / 4 / 3 / 1 moves
```

**Five of six hits are a 14-window collapsing to exactly 13** -- precisely the band a
reach-12 certificate cannot see and a reach-13 one certifies trivially. The lesson is
methodological: "N radii returned 0" is evidence about those radii, NOT about N+1. State
untested bands as untested.

Cost control that made it affordable, both from facts already in hand:
* Windows of length <= 13 are ALREADY proven geodesic -- the radius-12 sweep tested
  L=13 and found nothing, so d >= 13, and d <= 13 is trivial. Only L >= 14 is new.
* For L >= 14 we know d >= 13, so d(a^-1 w) >= 13 - |a|, forcing |a| = 6 and depth
  exactly 7. The shallow front cannot contribute, so `--shell-only` (26.0M instead of
  27.8M) is provably exact -- and is guarded to refuse when `--min-window <= 13`.

`--max-window 18` bought nothing beyond pid 32's 16-window: no window of length 17-18
fired in 67k queries. **Radius 14 is out of reach**: it needs B7 as permutations
(433M x 88 = 38 GB, ~7.6 s/query -> ~89 h) or a d8 table already rejected by measurement.

Build notes: the VMs build both tables from `puzzle_info.json` alone -- d7 in ~5 min on
12 cores via `03b_build_bfs_deep.py --hash-out`, the B6 front in ~100 s -- so do NOT
upload the 6.5 GB of tables (I wasted an upload learning this). Inverting 27.8M
permutations must be a SCATTER (`AINV[i][P[i][j]] = j`) in uint8 chunks; `np.argsort`
needs the array as int64 and returns int64 indices, 36 GB, and OOMs.

### (superseded above) radius 12, exhaustive, 0 moves

`scripts/61_mitm_gpu.py` is the GPU port of the meet-in-the-middle join, ~35 ms/query vs
~4 s in numpy. `bfs_endgame_d7.npz` (433,385,579 states) was already on disk, so front
B5 x the d7 table certifies **radius 12** rather than 11:

```
d(w) <= 12  iff  exists a in B5 with a^-1.w in B7

windows 13-15:   46,687 queries -> saved 0   (2 phantoms)
windows  7-12:  120,016 queries -> saved 0   (7 phantoms)
                166,703 exact queries, 0 moves
```

**Read the two bands differently -- they prove different things.** For a window of
length L <= reach=12 the join returns the OPTIMAL word, so 0 improvements means every
such window is EXACTLY GEODESIC. For L = 13-15 the join can only certify d <= 12, so 0
means d > 12 there; those windows could still be non-geodesic with d in [13, L-1], which
radius 12 cannot see.

So the provable statement is: **every window of length <= 12 in the submission is
geodesic.** That subsumes the leader's exhaustive radius-10 sweep and our radius-11
sample, and extends the earlier "all 44,390 windows of length <=6 are geodesic" by six
levels. It subsumes the leader's exhaustive radius-10 sweep and our radius-11 sample.
The conclusion drawn here at the time -- "do not build another local rewriter" -- was
WRONG; see the radius-13 section above.

**GOTCHA -- 64-bit Zobrist collides at 433M entries.** A probe false-positives at
~2.3e-11, and one query is 1.77M probes, so a full sweep sees a handful of PHANTOM hits:
states the table reports at a depth they are not at. The phantom is DETERMINISTIC (same
hash on CPU and GPU), so re-checking the depth cannot catch it -- it surfaces as
`descend` finding no child one level down, i.e. an assertion crash mid-sweep. The join
now walks the 32 best candidates and skips any that fails to descend or fails the
permutation guard. Seven phantoms in 46,687 queries. **This scales with the front**: at
radius 13 each query probes 26.0M instead of 1.77M and the counts rose to 13-117 per
shard. All were caught by the guard; none reached output. It is another argument against
radius 14 beyond the 38 GB.

### FALSIFIED 2026-08-05: rewriting ALTERNATIVE paths (`scripts/65_rewrite_alternatives.py`)

We min-merge first and post-process the winner. That order is LOSSY in principle: the
rewriter is path-dependent (it can only shorten a window it sees), so a shorter path can
be geodesic everywhere and immune, while a longer one contains a window that collapses
hard. Merging first discards the longer path before the rewriter sees it. The correct
order is rewrite-all-then-merge; merging after is free.

Tested directly. An alternative k moves longer wins iff `s_alt > k + s_best`:

```
1,121 alternatives within +3 of best, rewritten at radius 13
0 beat their pid's best, 0 moves saved
```

**Sound mechanism, no payoff at this yield.** With ~8 moves per 1,000 paths, a SPECIFIC
alternative saving >=2 is rare enough that 1,121 draws produced none. It would need a
much larger path portfolio or a higher-yield rewriter. Keep the script -- the reasoning
is right and it becomes live if either changes.

Sizing note: count alternatives against the CURRENT baseline. Measured against a looser
best there were 2,657 within +3; against 28,308's tighter paths only 1,121 qualify. I
quoted the stale figure after the baseline changed.

### FALSIFIED 2026-08-04: Knuth-Bendix completion (`scripts/62_knuth_bendix.py`)

The one part of "relation mining" that `58_symmetry_rewrite.py` cannot do: DERIVE
relations never observed, by resolving overlaps between known rules. Seeded exhaustively
to length 3 (hence confluent to 3 by construction), completed to |lhs| <= 12:

```
1,844,967 rules, longest lhs 9
derived reducing rules by |lhs|:  4:640  5:23,936  6:16,384  7:609,792  8:53,760  9:1,133,056
applied to the 28,455 file: 28,455 -> 28,455  (saved 0)
```

**Completion converged to SHORT rules** -- nothing above |lhs|=9 despite a bound of 12,
because overlaps of short rules resolve to short rules in this monoid. Every derived
rule therefore sits well inside the radius-12 MITM reach, so the method is strictly
subsumed by `61_mitm_gpu.py`: it cannot express a shortening that sweep did not already
test exhaustively. To be informative at all it would need |lhs| > 13, and the rule count
already hit 1.84M at round 1, so that bound is not reachable this way.

Together with 58 (mining, 0 shorter) this closes relation-mining in both halves.

### FALSIFIED 2026-08-04: cross-trajectory splicing (`scripts/60_splice_graph.py`)

Waypoint graph over every trajectory we own, with trajectory-step edges, exact bridges
whenever d(x^-1 y) <= 6, and endgame-descent edges. **0 bridges found**, 0 moves. This
is not bad luck, it is a bound: a splice through waypoint x beats the incumbent only if

    cost_from_start(x) + d(x) < incumbent

because d(x^-1 y) + cost_to_end(y) >= d(x^-1 y) + d(y) >= d(x). Over the 6,979 waypoints
where d(x) is exactly known (inside the d<=6 ball), **0 have any slack**. Note the graph
already subsumes the leader's "trajectory crossover": states shared between trajectories
dedup to one vertex, so Dijkstra switches trajectories for free, and it still finds
nothing. The remaining slack is not reachable by recombining routes we already have.

## SUBMITTED 28,467 -- FIRST PLACE, 2026-08-03

**Kaggle public score 28,467** (`submissions/FINAL_tetraminx_28467.csv`, submission
55219253, status COMPLETE). Rokicki's 28,481 is beaten by **14**. 1000/1000 pids,
replay-verified against the original `test.csv` states, **0 invalid**.

An earlier 28,475 went up the same day (submission 55214679) and also cleared the bar.

**Submission policy: the 28,481 rule is SATISFIED.** The standing instruction was "do
not submit until we BEAT 28,481"; that condition is met, so further submissions are
routine rather than gated. The public floor and community CSVs are still used for
training data and per-pid merging only -- they never get uploaded on their own.

**Always merge with `scripts/90_merge_all.py`** (via `/tetraminx-merge` or
`/tetraminx-submit`), never by hand -- hand-merges have silently under-counted twice.
And note the source set is NOT stable between runs: seven merges on 2026-08-03 scanned
121+85, then 112+78, 122+78, 128+78 and finally 131+121 files. Re-pull live machines
every time, copy sources into a stable dir, and search for stray results by CONTENT
(header + move alphabet) -- three result files worth -22 moves were sitting in an
unrelated `rogii-wellbore-geology-prediction/` folder and one in `megaminx/`.
See CLAUDE.md rules 26 / 26b.

### Score progression, 2026-08-03

| total | source |
|---|---|
| 28,559 | session start (previous best on disk) |
| 28,517 | our own beam JSONs from sibling sessions, never folded in |
| 28,500 | top-20 long-path run, 16M x 2 frames |
| 28,493 / 28,484 | public/community CSVs (`submission_publ`, `publ1`, `publ2`) |
| 28,481 | a community GitHub result set -- exactly level with the leader |
| **28,475** | `submission_dad.csv` -- **first submission, bar beaten** |
| 28,471 | the completed top-20 run |
| 28,468 | a sibling-session file that surfaced on the 6th merge |
| **28,467** | full re-pull of all TPU JSONs -- **final submission** |

Of the -92, roughly -62 came from runs we executed and -22 from **others running our
published notebook** -- which is our own solver on hardware we did not pay for, not a
third-party source.

### The deployed inference stack

```
PieceTransformer Q head (ep1500, 3.44M params, AZ dual-head)
  + ResMLP+AZ blended 0.8 / 0.2        (~1.07x cost, -2)
  + qv-consistency lambda 0.3          (free, -4, additive with history)
beam 16M sharded across 8 TPU cores, history_depth 1
2 frames: k0-forward + k1-inverse      (frames SATURATE at 2)
exact d<=6 endgame splice -> replay-verify -> n-way per-pid min
```

**Do NOT describe the TPU path as "progressive top-k" -- it is not.** The JAX kernel
materializes ALL 24 children (`neighbors = states[:, all_moves]`), hashes all of them
for owner routing, and only then takes a per-owner top-K. `q_mode` saves the MODEL
FORWARD only (one forward on the parents instead of 24 on the children), which is the
**17.2x** "Q exhaustive" column of the table below, not the 21.9x "Q + progressive
top-k" column. Progressive top-k -- score first, hash only the top ~alpha*B, double
until B uniques -- exists ONLY in the PyTorch searcher
(`src/cayley/khoruzhii_search.py::_do_greedy_step_q_topk`).

### TRAINING AXIS CLOSED 2026-08-04 -- 800 more epochs bought nothing

Cell 4 (transformer + AZ) ran to the extended 2000-epoch budget, every 100th checkpoint
beam-evaluated at Q@1M x1 forward frame on the 15 stratified pids -- the same protocol as
the ep700..ep1500 curve, so all of it is comparable.

| ep | 1200 | 1300 | 1400 | **1500** | 1600 | 1700 | 1800 | 1900 | 2000 |
|---|---|---|---|---|---|---|---|---|---|
| total | 426 | 428 | 431 | **424** | 426 | 427 | 425 | 428 | 428 |
| avg>1 | 30.36 | 30.50 | 30.71 | **30.21** | 30.36 | 30.43 | 30.29 | 30.50 | 30.50 |

Band 424-431, mean ~426.4, **no trend across 800 epochs**, while training loss kept
falling (13.70-13.86 over the same stretch). Eight checkpoints past ep1500 and not one
beat it.

**Read this as: ep1500's 424 is the low tail of a noise distribution, not a peak.** The
extend-to-2000 rule fired on ep1500 beating ep1200 by 2 -- a gap this spread cannot
resolve. Deploy ep1500 because it is the best measured, but do not believe the model was
still improving.

Two consequences:
* **Do not extend the budget again on this objective.** The next training gain has to come
  from changing WHAT is taught (the walk-label / BFS reconciliation, item 4 in the list
  above), not for how long.
* **Loss is not a stopping signal here** -- it fell monotonically while beam quality did
  not move. Same lesson as megaminx `m_dd_v0_full` (184 epochs, lower loss, 0% solve rate
  on hard pids). Only a replay-verified beam total counts.

### PROGRESSIVE TOP-K PORTED TO THE KERNEL 2026-08-03 -- MEASURED SLOWER, KEEP OFF

`pre_topk_mult` (default 0 = off) selects the global top-M by Q BEFORE materializing any
child, so only M children are built and hashed instead of all `n_gen * B_local`.
Implemented in the non-streaming body; the streaming body RAISES rather than silently
ignoring the flag.

**Correct, and strictly slower at every width tested:**

| width | pre_topk | result | wall |
|---|---|---|---|
| 1M (3 pids) | off | 91 | 37 s/pid |
| 1M (3 pids) | 4 | **91 -- identical paths** | 50 s/pid (**+33%**) |
| 32M (pid 50) | off | 29 | 1480 s |
| 32M (pid 50) | 4 | 29 | 3107 s (**+110%**) |

**The premise was false.** The port was built to relieve "the binding memory constraint",
citing this doc's own estimate that 32M needs ~8.8 GB/rank and is "likely too big". That
estimate was never measured. **32M runs UNCHUNKED with no pre-topk and no
RESOURCE_EXHAUSTED.** Measure the constraint before optimizing it.

Why it loses: `states[:, all_moves]` is ONE regular strided gather that XLA fuses well.
The port replaces it with two IRREGULAR gathers (`take` by parent, `take_along_axis` by
move) plus a serial chain top-k -> gather -> hash -> per-owner top-k. Trading fused
regular work for unfused irregular work is a bad deal whenever memory is not binding --
and it never was.

Keep the code: it is verified correct and would matter if a future puzzle/width really
does hit the children ceiling. Do NOT enable it as an optimization.

**Byproduct worth having: 32M unchunked WORKS**, at 1480 s/pid/frame (2x the 16M cost).
The "~16M practical ceiling" in the section below is superseded. Almost certainly
uneconomic given the width curve (1M->4M -13, 4M->8M -2), and single-frame 32M returned
pid 50 = 29 against the 27 already banked -- but the ceiling is now measured, not assumed.

Narrative write-up: [`BLOG_tetraminx_progress.md`](BLOG_tetraminx_progress.md).

It is the n-way per-pid min over 30 submission CSVs AND the raw beam JSONs in
`results/top100/`. Rebuild it with that same sweep rather than trusting any single
file -- 103 moves once sat unclaimed purely because results were scattered across
sibling session scratchpads and VMs. `merged_all_sessions.csv` (28,718) is kept
UNCHANGED as the fixed comparison baseline; re-banking into it would silently reset
every "moves saved" number to zero.

## NEXT EXPERIMENTS (priority order, as of 2026-08-02 evening)

Every inference lever tested on 2026-08-02 except the cross-architecture blend collapsed
into one redundancy class with `history_depth` (see the checkpoint-averaging section).
Training and search-config levers are the ones still paying. Cheapest-first:

1. **Frames on the transformer -- NEVER RUN above 1 frame.** ISAB gained **-27** going
   1 -> 4 frames on this exact 15-pid set, and the frames section below measures frames
   as ~72% of all wins on the ResMLP with the INVERSE antisymmetry carrying it
   (k0i credit 57.1 vs k0f 29.4). Worse: every transformer run to date used
   `frame_list(1,...) = [(0, False)]`, i.e. the WEAK FORWARD frame. Test `--frame-spec
   0i` first (same 1-frame cost), then 4 frames.
2. **`--history-depth 4`** (user request, 2026-08-02). The 40-pid ablation below measured
   at 1M: h0 +12, h1 -1, **h4 -8**, h8 -8 vs floor -- so **h1 -> h4 is a further -7 moves**
   for +50 s/call, and h8 buys nothing over h4. EVERYTHING run on 2026-08-02 used h1.
   Untested: h4 on the transformer Q head, and h4 at 4M. Expect the gain to SHRINK at
   4M -- width and history_depth overlap (see the CAUTION section), and every 1M-measured
   gain this session shrank at 4M (the ep1000->ep1200 training gain went -6 at 1M, -3 at 4M).
   Also note `30_solve.py --history-depth` was a DEAD FLAG until 2026-08-02; any pre-fix
   local run claiming h>0 was really h0.
3. ~~**Finish the epoch budget**~~ -- **DONE 2026-08-04, AXIS CLOSED.** Trained to 2000
   and evaluated every 100 epochs. See "TRAINING AXIS CLOSED" below: nine checkpoints
   across ep1200-2000 sit in a 424-431 band with NO trend, so ep1500's 424 is the low
   tail of noise, not a peak. More epochs of the same objective do not buy beam quality.
   A further extension needs a CHANGE (item 4 is the concrete one), not more of the same.
4. **NEXT TRAINING RUN: reconcile walk labels with the exact BFS table.** MEASURED
   2026-08-02 by `scripts/57_label_consistency.py` (200k pivots, k in [2,40], tilt 0.5):

   | quantity | value |
   |---|---|
   | pivots whose state is IN the d<=6 table | **37.1%** |
   | of those, walk index p != true d(s) | **25.9%** (always p >= d, never below) |
   | labelled UNDO child: label != true distance | 21.9% |
   | labelled NEXT child: label != true distance | 31.0% |
   | ordering `d(undo) < d(next)` CORRECT | **92.75%** |
   | ordering TIE (asserted gap of 2 is fictional) | 6.14% |
   | ordering INVERTED (teaches the wrong ranking) | **1.11%** |
   | mean TRUE gap `d(next)-d(undo)` | **1.717** (label always asserts 2) |

   Root cause: non-backtracking forbids only the immediate inverse, so a walk is NOT
   geodesic -- with an order-3 generator, `g, g` has walk index 2 at true distance 1.
   The anchor stream then teaches exact distances for the SAME shallow region that the
   walk stream labels with walk index. Two absolute scales, one model, overlapping
   states.

   Why sparse-Q survives this: it is a RELATIVE objective, so a uniform offset in p
   shifts both labels equally and the ordering is right 92.75% of the time. The residual
   harm is (a) an absolute-scale conflict with the anchors and (b) ~7.25% of labelled
   pairs asserting a gap that does not exist -- scaled by the 37% in-table rate, about
   **2.7% of all walk samples carry a false gap** and ~0.4% teach an inverted ranking.
   The 6.14% ties are the SAME benign-alternative-optima failure that killed megaminx's
   rank-loss variant, sitting inside our primary objective rather than an optional one.

   Fix (cheap -- the endgame table is already on-device for the anchors): hash each walk
   pivot and its two labelled children (3 lookups/row vs the 12,288/step the anchors
   already do) and when in-table **replace the walk labels with exact distances**. That
   upgrades 37% of walk rows from approximate to ground truth. Cheaper A/B variant:
   simply DROP in-table pivots from the walk stream, since anchors already cover d<=5
   densely and exactly. Cannot be applied mid-run -- it is an objective change.

   Rank this ABOVE ISAB Stage A/B: same "fix the training signal" thesis, but with a
   measured defect, a known-cheap fix, and no data-collection cost, whereas Stage A is
   still gated on item 6.

5. **4M tier re-runs.** Only 4M+ banks anything against `FINAL_tetraminx_28561.csv`;
   1M runs bank exactly 0 (see the comparison script `scripts/56_compare_vs_final.py`).
6. **NISS** (user request, 2026-08-03) -- after the 16M run and the ep1500 eval.
   NOT the same as the inverse frame: the inverse frame runs a COMPLETE independent
   search on s^-1 and min-merges the two solutions, while NISS builds ONE solution from
   both ends, appending inverse-side moves to the far end. They exploit the same
   forward/inverse asymmetry, which is why megaminx measured NISS as REDUNDANT once
   frame diversity was in place: 0 rescues on pids 991-1000, a rare 1-2 move shave, at
   2x wall (CLAUDE.md rule 11, [[niss_redundant_with_sym4]]). Never implemented on
   tetraminx. Prior is poor -- our inverse frames are worth ~1.4 moves AND run ~9%
   faster than forward -- but it has not been measured here.

7. **Sym-pooled beam (port from megaminx 3C)** -- the better bet of the two, and NOT
   yet ported. One K*B beam over all rotated+inverse copies POOLED, instead of K
   sequential B-beams. Megaminx measured pooled beating sequential sym4 at EQUAL budget
   on stubborn pids (991: 101 -> 82), with inverse frames carrying 6/8 wins -- the same
   pattern the 2026-08-03 16M run shows. Directly relevant: that run spends 4 x 16M as
   four independent searches; pooled, the same budget is one 64M beam over all frames.

   ALREADY SETTLED, do not re-try: **window reduction** (`42_window_reduce.py`) gave
   **0 rewrites** on the 28,821 merge -- the paths are locally optimal at the 6-move
   scale -- and **bridge compression** is the same local-rewrite family, already
   V-saturation-bound on megaminx past d~30. Also rejected on megaminx and not worth
   porting: route-relinking / portfolio oracle (recombination ceiling = 2 moves over 998
   pids), subgoal search, bidirectional MITM, FMC insertion (3-cycles cost 12-20 moves
   here), PHS cumulative (validated but marginal).

8. **Frontier-regret probe on the Q head** -- the GATE for ISAB's Stage A/B. Megaminx ran
   the equivalent probe and found child-ordering near-optimal with "misranks" that were
   benign alternative optima, which closed that whole line; and tetraminx's own path-label
   test (`tq1`) was rejected. Measure before investing days in frontier data collection.

### 2026-08-02: GENERATOR-ISAB RECOVERS RESMLP QUALITY, BUT DOES NOT CLOSE THE TRANSFORMER GAP

The 1,500-epoch standalone Generator-ISAB experiment and every-100-epoch checkpoint
sweep are complete. All reported paths replay with **0 invalid** on the fixed 15-pid
set. Best one-frame checkpoint is **epoch 1300: 462 moves at Q@1M x1**, versus
ResMLP 468, PieceTransformer 441, and the merged floor 423. Its four-frame result is
435. The average over all 15 checkpoint totals is 467.27 and epoch 1500 returns to
468, so this is a small, search-sensitive best-checkpoint gain rather than a stable
new incumbent.

The design successfully repaired the earlier lossy latent model (505 -> 462 at one
frame) by preserving all 50 physical-piece tokens, retaining a frozen ResMLP floor,
using exact generator geometry, and zero-initializing the relational correction.
However, the random-walk absolute-Q objective and offline probe metrics are poorly
aligned with beam-critical frontier rankings. Keep epoch 1300 only as a diversity
candidate; do not replace ResMLP. Next experiment should use real beam-frontier hard
negatives, centered advantage/ranking losses, and an A/B test of hard moved-piece
readout versus soft global generator-biased readout. Full results, caveats, artifacts,
and the proposed phase-2 protocol are in
[`GENERATOR_ISAB_FINDINGS.md`](GENERATOR_ISAB_FINDINGS.md).

Length histogram of the final merge:
`{27: 41, 28: 184, 29: 372, 30: 319, 31: 39, 32: 2}` -- the >=31 tier is nearly
exhausted (was `{31: 118, 32: 9}`), and the mass has moved down to 29/30.

### 2026-08-02: WIDTH BEATS DIVERSITY AT MATCHED BUDGET (the deployment rule)

Two levers were pitted against each other at **identical compute**, same 15 stratified
pids (`0,50,100,200,300,400,500,600,700,800,900,950,990,995,999`), cell-4
(transformer + AZ head) checkpoints, Q@1 frame, history_depth 1, v6e-8.
Community floor on these 15 = **423**.

`avg` = total/15; `avg>1` excludes pid 0 (a 1-move puzzle) and is the honest
per-puzzle number. ALWAYS report total AND average -- a total is only meaningful
against the same pid set, and the mean makes a set swap obvious at a glance.

| config | 15-pid | avg | avg>1 | compute |
|---|---|---|---|---|
| 1M x1, ep1000 (or ep900) | 432 | 28.80 | 30.79 | 1 unit |
| 1M x1, per-pid min over ep700/800/900/1000 | 426 | 28.40 | 30.36 | 4 units |
| **4M x1, ep1000** | **419** | **27.93** | **29.86** | 4 units |
| 4M x1, ep900 | 423 | 28.20 | 30.14 | 4 units |
| 4M x1, per-pid min over ep900+ep1000 | 417 | 27.80 | 29.71 | 8 units |
| *community floor on these 15* | *423* | *28.20* | *30.14* | -- |

**Checkpoint diversity SATURATES; width does not.** At 1M the min over
{ep900, ep1000} is already 426 and adding ep700+ep800 to reach 4 units buys
**zero**. One level up, a second 4M checkpoint buys only **-2** (419 -> 417) for
2x the compute -- against width's **-13** (432 -> 419) over the same 4x span.
At 4 units, width beats checkpoint-min by **7 moves**.

**4M x1 = 419 is the first single config of ours to go under the community floor
(423) on this set.** Do not read the cross-config per-pid min (417 over ~6 runs) as
a result -- that is an oracle, not a deployable config (Rule 26 applies to our own
runs, not just community CSVs).

Deployment rule: **spend compute on beam width, one checkpoint.** Multi-checkpoint
ensembling is not worth its cost at any width tested.

### 2026-08-02: the width curve is EXHAUSTED at ~4M, and everything converges on 417

Cell 4 ep1000, 15 pids, 1 frame, history_depth 1, v6e-8:

| width | total | avg | avg>1 | delta | cost |
|---|---|---|---|---|---|
| 1M | 432 | 28.80 | 30.79 | -- | 1x |
| **4M** | **419** | **27.93** | **29.86** | **-13** | 4x |
| 8M | 417 | 27.80 | 29.71 | -2 | 8x |

**Deployment width is 4M.** 8M costs double for -2. Note 8M single (417) EQUALS the
4M per-pid min over two checkpoints (417) at the same 8 units -- width and checkpoint
diversity have converged on the same ceiling. Also equals min(soup@4M, ep1000@4M).
Every route to 8 units lands on 417.

### 2026-08-02: checkpoint AVERAGING is beam-neutral (weight-space AND score-space)

Tested after the min-merge saturation result, on the theory that averaging reduces
scorer variance rather than sampling K times (a genuinely different mechanism).

Weight-space soup of ep800/900/1000 (`scripts/54_model_soup.py`) -- ONE model out, so
it costs 1x at inference:

| TPU, ep1000 vs soup, hd=1 | total | avg | avg>1 |
|---|---|---|---|
| ep1000 @1M | 432 | 28.80 | 30.79 |
| soup @1M | 432 | 28.80 | 30.79 |
| ep1000 @4M | 419 | 27.93 | 29.86 |
| soup @4M | 420 | 28.00 | 29.93 |

**Neutral at both widths WITH history_depth=1** -- but the hd=0 control tells a
different story. Comparing WITHIN each platform (no cross-platform confound):

| comparison | soup | ep1000 | soup delta |
|---|---|---|---|
| local, hd=**0** | 436 (avg 29.07, avg>1 31.07) | 440 (avg 29.33, avg>1 31.36) | **-4** |
| TPU, hd=**1** | 432 (avg 28.80, avg>1 30.79) | 432 (avg 28.80, avg>1 30.79) | **0** |

**The soup is REDUNDANT, not inert.** It buys -4 with history_depth off and nothing with
it on: `history_depth=1` (worth ~-8 on its own here) already removes the same error class.
A first pass at this data claimed "averaging relocates errors instead of reducing them"
-- that was built on a cross-platform comparison and is WRONG; the control refutes it.

Deployment verdict is unchanged (we always run hd=1), but note the pattern: several
inference levers measured at hd=0 look like independent wins and are competing to fix
one thing. Suspect the same of V-consistency (-4 at ep700 hd=0, gone at ep1000) and
possibly band rerank -- both were measured at hd=0 against an hd=1 baseline and neither
has been re-measured at matched hd. Same shape as megaminx NISS-redundant-with-sym4.

The soup still won the sparse-Q probe against every input in EVERY depth band (pair
.9007 vs .8971, top1 .5317 vs .5270, ahead at 20-24 / 25-29 / 30-40). At hd=1 that
bought zero moves -- third instance of better-probe-metrics-not-transferring after GT-V
calibration and all-neighbour-Q recall.

Score-space blending (`models.BlendedQ`, `30_solve.py --blend`) is BUILT and verified
(uniform + weighted averaging exact; `output_dim` forwarded or the solver loads it as a
scalar V and dies mid-beam). Probe puts it within 0.001 of the soup (pair .9016, top1
.5324), so it is predicted neutral at 3x the cost -- NOT run in beam. Costs K forwards
per step, i.e. a K-blend at width B must beat one model at K*B.

### GOTCHA: `30_solve.py --history-depth` defaults to 0, the TPU driver uses 1

Every local A/B on 2026-08-02 was run at hd=0 and compared against TPU baselines at
hd=1, worth ~4 moves on the 15-pid set (soup local 436 vs soup TPU 432, same checkpoint
/ width / pids). That inflated every local regression and hid the true size of every
local gain. It is NOT a platform/precision effect -- it is an unmatched flag.
**Match `--history-depth 1` on any local run being compared to a TPU number**, and see
the standing caution that width and history_depth OVERLAP.

### 2026-08-02: cell 4 (AZ head) keeps training past where cell 3 stops

15-pid Q@1M by epoch -- cell 4 (transformer + AZ) vs cell 3 (transformer, no AZ):

| ep | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| cell 4 total | 459 | 448 | 445 | 438 | 443 | 442 | 436 | 440 | **432** | **432** |
| cell 4 avg | 30.60 | 29.87 | 29.67 | 29.20 | 29.53 | 29.47 | 29.07 | 29.33 | **28.80** | **28.80** |
| cell 4 avg>1 | 32.71 | 31.93 | 31.71 | 31.21 | 31.57 | 31.50 | 31.07 | 31.36 | **30.79** | **30.79** |
| cell 3 total | 444 | 440 | 446 | 447 | -- | 449 | 438 | 443 | -- | -- |
| cell 3 avg | 29.60 | 29.33 | 29.73 | 29.80 | -- | 29.93 | 29.20 | 29.53 | -- | -- |

Cell 3 is flat in 438-449 with no trend; cell 4 descends and is still descending at
ep1000. Confirmed on the 30-pid discriminating set (no trivial pid, so avg>1 = avg):

| 30-pid set | total | avg |
|---|---|---|
| cell 3 ep700 | 951 | 31.70 |
| cell 4 ep700 | 945 | 31.50 |
| cell 4 ep1000 | 939 | 31.30 |
| cell 4 ep900 | **938** | **31.27** |
| cell 4 per-pid min over ep700/900/1000 | 927 | 30.90 |

**The AZ head's main value is trainability, not the value signal** -- cell 4 beat
cell 3 by 6 on 30 pids long before the value head was ever read at inference.

Note the 30-pid mean (31.3) sits ~1.4 above the 15-pid `avg>1` (29.9): the 30-pid
set is genuinely harder, so totals from the two sets are NOT comparable. This is
exactly why the mean gets reported next to the total.

Per-pid noise between adjacent checkpoints is +-3 (pid 800: 30/32/30/33), so a
4-move total on 15 pids is inside noise -- use the 30-pid set to discriminate.

### 2026-08-02: reading the value head at inference -- three configs priced

Cell 4 ep700, 15 pids, local 4090. Q-only baseline **436**, ~4,700 s.

| config | 15-pid | avg | avg>1 | delta | wall | cost |
|---|---|---|---|---|---|---|
| Q-only | 436 | 29.07 | 31.07 | -- | ~4,700 s | 1x |
| **V-consistency, lam=0.3** | **432** | **28.80** | **30.79** | **-4** | 4,763 s | **1.01x** |
| band rerank, alpha=1.3, band_lo=0.9 | 431 | 28.73 | 30.71 | -5 | 6,547 s | 1.4x |
| full rerank, alpha=2 | 429 | 28.60 | 30.57 | -7 | 14,093 s | 3.0x |

`--qv-consistency` scores `Q(s,a) + lam*|Q(s,a) - (V(s)-1)|` with BOTH heads from one
trunk pass -- genuinely free, and it recovers the entire ep700->ep1000 training gain
without training. Band reranks only the marginal `[0.9B, 1.3B)` slice by V.

Not monotone improvements: band LOSES pid 990 (30 -> 31) and consistency loses pid
600 (32 -> 34) while winning pid 400 (30 -> **28**, better than anything else
including 4M). Gains also concentrate -- pid 300 alone is -3 of band's -5.

**Blocker**: all three are PyTorch-only. The JAX TPU kernel has no rerank/consistency
path, so they cannot currently combine with the 4M width that actually wins.

### Long-path beam sweep, 2026-07-29..31 -- the biggest single lever found so far

AZ value head (`taz_v1_v_only.pt`, sha256 e42e0027ba33), 8M beam, 4 frames,
history_depth 1, on v6e-8. Yield is strongly graded by how long the merged path already
was -- longer path, more slack:

| original length | done/pop | win rate | saved | per-pid |
|---|---|---|---|---|
| 32 | 9/9 | 78% | -14 | **-1.56** |
| 31 | 118/118 | 69% | -109 | **-0.92** |
| 30 | 13/287 | 38% | -6 | **-0.46** (in progress) |

FINAL as run (stopped 2026-08-01, TPU released):

| original length | done/pop | win rate | saved | per-pid |
|---|---|---|---|---|
| 32 | 9/9 | 78% | -14 | **-1.56** |
| 31 | 118/118 | 69% | -109 | **-0.92** |
| 30 | 121/287 | 24% | -34 | **-0.28** (mixed configs, see below) |

**Yield is graded by how long the merged path already was** -- longer path, more slack.
That relationship is the single most useful predictor found here: it says where to spend
TPU and it says when to stop.

### Frames: the inverse frame does the work, but the margin is POPULATION-DEPENDENT

Over 171 pids (mostly len 31/32): all-4-frames -143 moves vs k0-forward-only -39, i.e.
**frames produced ~72% of every win**, and 65% of pids had a non-frame-0 frame strictly
better. Best-frame credit k0f 29.4 / **k0i 57.1** / k1f 30.7 / **k1i 53.8** -- the
INVERSE antisymmetry carries it, not the spatial rotation.

Efficiency on that same >=31-heavy population: k0i alone 4.54 moves/TPU-hour, k0i+k1i
2.97, all four 1.84. On that basis the sweep was switched to `--frame-spec 0i`.

**Then measured on length-30 alone, the margin shrank:**

| config | n | saved | wall | moves/TPU-hour |
|---|---|---|---|---|
| 4 frames | 47 | -21 | 21.3h | 0.99 |
| 1 frame (0i) | 73 | -13 | 8.2h | **1.59** |

**1.62x, not the 2.5x extrapolated from the >=31 population.** One frame still wins per
TPU-hour, but it captures a smaller share of the 4-frame gain on tight paths (16% win
rate vs 36%). LESSON: an efficiency ratio measured on loose paths does not transfer to
tight ones -- re-measure per tier before quoting it.

### RESOLVED 2026-08-02: streaming step body ported to feature parity

The cap described below is GONE. `_build_step_body_v_only_packed_streaming` now
implements history_depth / endgame / no-backtrack / q_mode and matches the wrapper
at 9 inner args / 13 returns, so `parent_chunk` -- and therefore 48M/64M -- works.

Two send-side subtleties in the port, both easy to get wrong:
* **q_mode**: score the chunk's PARENTS, not its children. A (parent_chunk, n_gen)
  output flattens to exactly `children`'s order (child m of parent i at i*n_gen+m).
* **no-backtrack**: `in_move` spans the whole local beam, so it must be
  `dynamic_slice`d to the chunk's parent window. Using it whole would misalign
  every chunk after the first -- and would do so silently, banning the wrong moves.

`q_mode` now raises if `pack_v_score` is off (a Q head cannot rescore a bare state
receive-side), matching the non-streaming contract.

**Validation**: `cpu_smoke.py` gained `--parent-chunk` and `--endgame`, and the two
bodies were run on identical puzzles with history + no-backtrack + endgame all
active. Both returned the SAME path (`3F.-2F.-3D` at scramble 7; the 9-move
`2BR.-3BL.-2BL.2F.2D.-4BL.-3D.-3BL.2D` at scramble 11, both stopping in the table at
depth 6 and splicing the same tail). **That is CPU validation: it proves shapes and
logic, NOT the TPU `all_to_all` collectives, which is where this kernel has
historically broken.** A TPU run at 48M is still unproven.

Also fixed: `cpu_smoke.py` had no host-side tail splice, so ANY `--endgame` run
reported `RESULT: FAIL` even when correct -- the beam legitimately stops on a table
node and the raw path is a prefix. It now splices before checking.

### (historical) BEAM WIDTH WAS CAPPED AT ~16M: the streaming step body was four features behind

`jax_beam_spmd_v_only.py` has TWO step-body builders and `parent_chunk` chooses between
them. Above ~16M the children array (`B_LOCAL * 24 * 88` bytes: 13.3 GB/rank at 48M)
does not fit a 16 GB TPU core, so a big beam REQUIRES `parent_chunk` -- and the
streaming builder it selects predates four features the non-streaming one gained:

| | `_..._packed` | `_..._packed_streaming` |
|---|---|---|
| inner args | 9 (incl. `hist_in`, `inmv_in`) | **7** |
| returns | **13** (incl. `new_hist`, `chosen_move`) | **11** |
| history_depth / endgame / no_backtrack / q_mode | yes | **none of them** |

`step_fn` hardcodes `out_specs=(P("cores"),) * 13`. So today, setting `parent_chunk`:

1. raises `TypeError: _build_step_body_v_only_packed_streaming() got an unexpected
   keyword argument 'q_mode'` (hit on Kaggle 2026-08-01 at 48M);
2. if that is patched, fails again on the 11-vs-13 arity mismatch;
3. **if THAT is patched, runs silently WITHOUT history dedup and WITHOUT the exact
   endgame** -- i.e. quietly worse paths, no error. That third failure is the
   dangerous one.

**Practical ceiling without the port** (children unchunked, per rank):
`8M -> 2.2 GB` (validated, every GCP result), `16M -> 4.4 GB` (plausible, UNTESTED),
`32M -> 8.8 GB` (likely too big with states+activations).

The notebook now asserts rather than trusting the reader: setting `PARENT_CHUNK`
together with any of history/endgame/no-backtrack fails immediately with the reason.
**Going past ~16M means porting those four features into the streaming body** -- and
that port then needs TPU validation, not just a CPU smoke test.

### Where the bar stands

Finishing len-30 at one frame projects ~28,531 (+50); at four frames ~28,486 (+5) but
consuming 76h. Neither clearly clears 28,481, and the decay 32 -> 31 -> 30 of
-1.56/-0.92/-0.28 says the len-29 tier (372 pids in the current merge) will yield less
again. **The remaining gap is unlikely to close by more of the same beam.**

Sizing note: the original forecast for this work was -30 to -45 moves; it returned
**-157** against the 28,718 baseline. The forecast was anchored on the pids 990-999
frames run, whose paths were already tight; ordinary merged 31s and 32s were far looser.

This is NOT 28,821 or 28,775 -- those were
what individual files held. No file on disk contained the combined result: merging all
44 result CSVs across every session and machine found 103 further moves, mostly from
another session's `ab/tier30_Q1M_f4.csv` (35 winning pids) and `ab/tier_Q1M_f4.csv` (26)
which had never been folded in. **Re-run that merge before quoting a score.**

Against the external floor (per-pid min of `community_28843` and `floor_public_29622`,
which is 28,843 -- the old public floor adds nothing), our own solving contributes
**-125 over 83 pids**. Gap to the bar: **+237**, so still nothing to submit.

Shape of the remaining gap: length histogram
`{27: 40, 28: 176, 29: 327, 30: 287, 31: 118, 32: 9}`, max 32. Closing 237 means finding
about one move on ~237 of the 405 pids at length >= 30 -- and section "Window reduction"
below shows the paths are locally irreducible out to 9-move windows, so it will not come
from post-processing.

Distance to the bar as of 2026-07-28 (STALE, kept for context): best on disk was 28,821, so **-341 moves**
(and Rokicki can still move). For scale, the whole sparse-Q slack-tier run projects
about −47. This is not a gap the current beam+NN approach closes on its own; see
the "different method class" note in the yield section below.

> ## STATE OF PLAY CORRECTED 2026-07-28 — most of this file is stale
>
> This file's headline numbers ("FINAL RESULT 29,477", "public floor 29,622",
> "floor >= 33 tier = 60 pids") were all superseded by a later session that pulled a
> much tighter community CSV. Replay-verified against `data/test.csv` today, every
> path solving, 1000/1000:
>
> | file | total | valid |
> |---|---|---|
> | `merged_with_28843.csv` | **28,821** | 1000/1000 |
> | `community_28843.csv` | 28,843 | 1000/1000 |
> | `final_best_v2.csv` (our pipeline) | 29,444 | 1000/1000 |
> | `floor_public_29622.csv` (the "floor" quoted below) | 29,622 | 1000/1000 |
>
> 28,821 is the per-pid min over all 9 full-coverage CSVs on disk, and it already
> clears the 28,863 submission gate. **The remaining target is Rokicki's 28,481.**
>
> Two consequences for any new work:
> 1. **Rule 26 applies hard here.** Our whole pipeline contributes just **16 pids /
>    22 moves** to the 28,821 merge. Any "we improved X moves" claim measured
>    against `floor_public_29622` is inflated by ~800 moves and is meaningless.
>    Measure against `merged_with_28843.csv`.
> 2. **The slack tiers below no longer exist.** Length histogram of the merged best
>    at >=30 is `{30: 306, 31: 141, 32: 15}` — **max length is 32, and nothing is at
>    33+**. The old "floor >= 33 = 60 pids" tier is gone. The current target set is
>    the 156 pids at merged-best >= 31.

## Puzzle facts (all verified locally)

| Fact | Value |
|---|---|
| State | 88 facelets, all distinct → permutation puzzle, same class as IHES cube |
| Generators | 24 = axes {D,F,BL,BR} × layers {2,3,4} × 2 directions; every one order 3 |
| Facelets moved | layer 2 → 21, layer 3 → 15, layer 4 → 9 (distinct cycle types per layer) |
| Group order | 1.538e32 (Schreier–Sims) |
| Facelet orbits | 11: sizes [12,12,3,3,3,3,12,12,12,12,4] |
| BFS levels | 1 / 24 / 408 / 6,592 / 105,136 / 1,659,416 / 26,008,172 (d6 cum 27,779,749) |
| Branching | ~15.8 non-backtracking |
| Counting bound | avg optimal ≥ ~26.7; realistic true optimum ~28 |
| Test set | 1,000 pids; 41 shallow (≤26 moves), 959 near-diameter |

### Symmetry group — 24 spatial, 48 usable frames

Facelet-automorphism group = **15,552 = 24 × 648**, where 648 is the centralizer
(commutes with every move → acts trivially on reachable states). The quotient is
the full tetrahedral group Td: 12 rotations (even axis permutation, direction
preserved) + 12 mirrors (odd axis permutation, direction flipped). Same structure
as cube (1152 = 48 × 24) and megaminx (720 = 120 × 6).

With inverse antisymmetry (`solve s^-1`, then reverse + invert the path) that is
**48 independent search frames per puzzle**. Built + verified by
`scripts/01_build_symmetries.py` (480 conjugation round-trips + 20 antisymmetry
checks); the AZ dataset builder then replays all 48,000 augmented variants and
asserts each solves.

Conventions stored in `data/tetra_symmetries_meta.json`:
- `conj(s)[i] = P_inv[s[P[i]]]`
- a path `m_1..m_L` solving `conj(s)` maps to `sigma(m_1)..sigma(m_L)` solving `s`
- to push an original solution INTO a frame, use `sigma^-1`

## Leaderboard

```
Rokicki       28,481   near-optimal / coset solver territory
Khoruzhii     28,863   CayleyPy beam + NN
webmaking     29,555   ┐ July 2026, public "RW-Models2" notebook family:
Arbidos       29,591   ┘ MLPRes1 ~200K params, RW regression only, beam 2^21, fp16
public kernel 29,622   our floor (ka1242/load-tetramix-solution), verified 1000/1000
```

Headroom floor → leader is 1,141 moves = 1.14/pid. The public competitors use no
Bellman refinement, no BFS anchors, no symmetry ensemble and no exact endgame.

## What is built

| Path | What |
|---|---|
| `src/tetraminx/puzzle.py` | `Tetraminx` puzzle (duck-types `PictureCube`, so all of `src/cayley/*` works unchanged) |
| `scripts/01_build_symmetries.py` | 24 symmetries + inverses + move-relabel tables, self-verifying |
| `scripts/02_import_public_floor.py` | public solution → `submissions/floor_public_29622.csv`, replay-verified |
| `scripts/03_build_bfs.py` | BFS d≤6: endgame hash table (`bfs_endgame.npz`) + Bellman anchors (`bfs_d6_train.pt`) |
| `scripts/10_train_v.py` | Stage A random-walk regression |
| `scripts/11_bellman_refine.py` | Stage B Bellman refinement |
| `scripts/12_eval_v.py` | the four acceptance gates |
| `scripts/20_build_az_dataset.py` | floor → 1,421,856 (state, action, dist) samples at 48× augmentation |
| `scripts/21_train_az.py` | AZ dual-head (shared trunk, V + policy), port of the cube/megaminx AZ v4 recipe |
| `scripts/30_solve.py` | beam solve: sym-ensemble frames + exact endgame + floor merge |
| `kaggle_notebooks/tpu_beam_tetraminx/` | JAX SPMD TPU kernel port + `cpu_smoke.py` |

## Models

`tv0_pretrain` — Stage A, 500 ep, 4,996,481 params, hidden (2048,512) + 2 res
blocks, embedding encoding, k_max 32. Final loss 10.38 (MSE vs walk depth).

`tv0_bellman` — Stage B, warm from Stage A ep499, 400 ep planned, BFS anchors 10%
+ V0/d1 anchors, 18.6 s/epoch on the L4.

### Gate results

| Metric | Stage A ep499 | Stage B ep24 | Gate |
|---|---|---|---|
| V(solved) | +3.723 FAIL | **+0.026** | \|V\| < 0.5 |
| V(d=1) | +1.129 | **+0.984** | 0.5 .. 1.5 |
| exact d=5 | +5.833 | **+5.074** | — |
| saturation drift (d80−d40) | +0.59 | **+0.96** | ≤ 3.0 |
| absolute V@d80 | 28.08 | **23.90** | 22 .. 34 |
| std(V@d20) | 3.71 | **2.98** | ≤ 5.0 (puzzle-calibrated) |
| | FAIL (V0) | **PASS** | |

Note this puzzle saturates cleanly even at Stage A — the failure mode that killed
seven megaminx encoder architectures is simply not present here. Stage B's V@d80
sits ~3 BELOW the counting bound, i.e. it is optimistic on deep states, which is
the normal Bellman-bootstrap signature (same as the AZ v4 optimism band). Watch
it across epochs: a continued slide is the dodecahedral-CNN collapse mode.

## Measured beam baselines (tv0_bellman ep24 — 25 of 400 epochs)

Stratified 15-pid sample, 1 frame, exact endgame on, local 4090:

| config | result |
|---|---|
| beam 65,536, bf16 | 14/15 solved (pid 999 failed), **+5.71 moves/pid vs floor**, ~15 s/pid |
| beam 65,536, fp32 | pids 50/100/200 → 36/34/41 vs bf16's 36/37/41; 2.7x wall. Wash at n=3 |
| beam 262,144, bf16 | pids 100/300/900: 37→**34**, 37→**31** (ties floor), 40→**35**. **+7.33 → +2.67 moves/pid** on those three. ~45 s/pid |

bf16 is not the bottleneck; keep it for wall-time.

### CAUTION: the width and history_depth levers OVERLAP — don't add them up

The "4x width = −4.7 moves/pid" figure was measured with `history_depth=0`, before
that option existed. On pid 100 the levers land on the same number:

| config | pid 100 |
|---|---|
| 65K, no history | 37 |
| 65K, `history_depth=10` | **34** |
| 262K, no history | **34** |
| 524K, `history_depth=10` | 33 |

So most of what 4x width bought was just avoiding re-expansion of recently-seen
states, which `history_depth` gets for free. With history on, 65K → 524K (8x) is
worth only about **−1 move** on this pid. **The marginal value of width under the
full config (history_depth + 4 frames) is not yet cleanly measured** — and it is
the number that decides whether TPU 32M is worth its wall-clock over 1M. Measure
it on the TPU by running the same pids at 1M and 8M before committing to 32M.

### THE CENTRAL FINDING SO FAR: V refinement has plateaued, width is the lever

Identical 14-pid comparison, beam 65,536, 1 frame, exact endgame:

| model | Bellman epochs | V@d80 | bench total |
|---|---|---|---|
| tv0 ep24 | 25 | 23.90 | **478** |
| tv0 ep99 | 100 | 18.74 | 481 |
| tv1 ep124 (double-Bellman + deep anchors) | 125 | 21.42 | 496 |

Three models spanning a 5-point range of deep-scale calibration and two different
Bellman recipes bench **within noise of each other**, and the 25-epoch model is the
best of them. Neither more epochs nor the anti-drift fix moves beam quality.
Meanwhile ONE 4x width step moved it 4.7 moves/pid. Conclusions:

* **Stop spending time on the V.** `tv0_bellman/epoch_0024.pt` is the working model.
* The V@d80 downward drift is real but appears *harmless* — plausibly even useful.
  Note the public notebooks train with `PinballLoss(tau=0.2)`, which deliberately
  biases V to UNDER-predict; an admissible/optimistic heuristic may simply rank
  better for beam. Do not spend more effort "fixing" the drift without a bench
  that shows it costs something.
* Per-pid variance across models is large (pid 900: 38→46, pid 995: 31→42), so a
  15-pid bench resolves ~±2 moves/pid. Do not act on smaller differences.

### Search-side levers, measured (tv0 ep24, beam 65,536, pids 50/100/200/300/400)

| lever | total | vs floor |
|---|---|---|
| baseline, 1 frame | 189 | +7.60/pid |
| `--history-depth 10` (cross-layer dedup) | 186 | +7.00/pid — free, keep on |
| `+ --sym-frames 4` | **170** | **+3.80/pid** — 4x wall |

Per-pid with 4 frames, and which frame won:

| pid | 1 frame | 4 frames | winner | floor |
|---|---|---|---|---|
| 50 | 36 | 35 | k=1 fwd | 31 |
| 100 | 34 | 34 | k=0 fwd | 31 |
| 200 | 41 | **36** | k=0 **inv** | 30 |
| 300 | 37 | **34** | k=0 **inv** | 31 |
| 400 | 38 | **31** | k=1 **inv** | 28 |

**−3.2 moves/pid from 4 frames**, comparable to a 4x width step, and 4 of 5 pids
were won by a non-identity frame — 3 of them by an INVERSE frame. Same pattern as
the cube (inverse frames carried 6/8 wins there).

**But frames saturate at 4.** `--sym-frames 8` reproduced the 4-frame result
EXACTLY — same total (170), same per-pid lengths, same winning frames — at 2x the
wall. Frames 5-8 (k=2, k=3) contributed nothing. The productive set is
{identity, identity-inverse, k=1, k=1-inverse}; the 48-frame reserve is NOT the
lever it looked like. **Spend compute on width, not frame count.**

`history_depth` was missing entirely from our beam — it dedups only within a
layer, while CayleyPy's library beam (which the public notebooks use) runs
`history_depth=10`. Added as a default-off config field in
`src/cayley/khoruzhii_search.py`, so the cube/megaminx stacks are unaffected.

### tv0 ep25 vs ep99 — 75 epochs of training bought nothing in beam terms

Same 15 pids, beam 65,536, 1 frame. Loss fell 0.45 → 0.089 (5x) over those epochs.

| | ep24 | ep99 |
|---|---|---|
| the 14 pids both solved | 478 | 481 |
| pid 999 | not solved | 42 |
| wins / losses / ties | — | 6 / 4 / 4 |

So beam quality is **flat**, not regressing, with ep99 buying one extra solve. But
a 5x loss improvement producing zero length improvement is the Rule-12 signature,
and per-pid variance is large (pid 700 34→49, pid 995 37→31), so a 15-pid bench
resolves maybe ±2 moves/pid. Meanwhile V@d80 collapsed 23.90 → 18.74 (see
`configs/tv1_bellman.yaml` for the mechanism and the fix).

## Gotchas found here

1. **The TPU JAX kernel hardcodes `PACK_SIZE`** — the per-record bucket width for
   the all-to-all. Cube 80 (72-byte states), megaminx 128 (120), **tetraminx 96**
   (88 + 4 parent_local + 1 move + 2 optional score = 95). Missing this gives
   `Incompatible types for broadcasting: uint8[64,88] vs uint8[64,80]`.
   The backpointer bit budget needed no change: 5 move bits covers 24 generators.
   The padding-detection test is `sum == 0`, which is size-agnostic despite the
   comment naming 2556.
2. **`internal_bs` must not exceed `B_local`**, or the parent-chunk reshape asks
   for a 0-sized leading dim (`cannot reshape (512,88) into (0,4096,88)`).
3. **`JAX_ENABLE_X64` must be set before importing jax** — the kernel hashes
   states into int64 and otherwise dies with `Python int too large for C long`.
4. **A backgrounded `mkdir -p X && cd Y && setsid nohup A & ... B > X/log`
   races**: the whole AND-list is backgrounded, so a second launch redirecting
   into `X/` can fire before `mkdir` completes. Launch jobs in separate calls.
5. **`std(V@walk-depth-20)` conflates model noise with real distance spread** at a
   fixed walk depth. The sharp version is std at a fixed EXACT distance (0.17 at
   d=5 here). Treat the walk-depth number as a relative canary vs the recorded
   baseline (3.71 Stage A / 2.98 Stage B), not an absolute threshold.

6. **`pkill -f "11_bellman_refine"` over ssh self-matched its own command** and
   killed the session mid-chain, so the `&&`-chained rebuild after it silently
   never ran (CLAUDE.md gotcha 7g, walked into anyway). Kill by a pattern the
   killing command does not itself contain, or bracket the first char.

## Window reduction (`scripts/42_window_reduce.py`) — banked, small

Replaces any path window whose net group element `s_i^-1 . s_j` sits inside the
d<=6 table with the table's optimal word. Results:

* on the merged CSV (mostly FLOOR paths): **29,615 -> 29,607, 6 rewrites, −8 moves**
* on our raw beam output (12 GCP pids): **0 rewrites**

Both directions are informative. The public floor is NOT perfectly tight — it has
a handful of reducible windows, and that −8 is free and already banked. Our beam
paths, by contrast, are already locally irreducible at window <= 14, because the
exact endgame splice makes the tail optimal and the beam dedups states as it
goes. So run this pass on merged output for the free floor slack; do not expect
it to rescue our own paths.

## What "reach 28,863" actually requires

Floor length histogram (mode 30, max 36):

```
26: 7   27: 19   28: 81   29: 189   30: 278   31: 228   32: 104   33: 41   34: 15   35: 3   36: 1
```

Total 29,622, so the submission bar is **−759 moves**. Three ways to get there:

| target set | improvement needed | result |
|---|---|---|
| the 164 pids with floor >= 32 | −3 each | 29,130 (not enough alone) |
| the 392 pids with floor >= 31 | −2 each | **28,838** — clears the bar |
| the 670 pids with floor >= 30 | −2 each | 28,282 |

So this is a grind over several hundred pids, not a handful of hero solves. Since
the floor averages 29.62 against a true optimum near 28, a pid whose floor is
31–32 plausibly has 2–3 moves of headroom — but our beam currently produces
34–41 on such pids, so width + frames have to close 5–10 moves first.

### BOTTOM LINE (measured, 2026-07-26): our solver is ~2.4 moves/pid behind the floor

Our beam produces **~32 moves per pid almost regardless of the puzzle**, while the
public floor ranges 30–36 and averages 29.62. So we win exactly where the floor
happens to be unusually bad, and nowhere else.

| floor | pids measured | our mean | wins | merge yield/pid |
|---|---|---|---|---|
| 34–36 | 19 | 33.0 | 9 | −1.2 |
| 33 | 41 | 31.98 | 20 | −1.02 |
| 32 | 46 | 31.41 | 18 | −0.59 |
| 31 | 10 | 32.80 | **0** | **0.00** |
| 30 | 10 | — | 2 | −0.20 |
| 29 | 2 | — | 0 | 0.00 |

Note the merge-yield column is what matters: every run is min-merged against the
floor, so a LOSS COSTS NOTHING (we keep the floor path). The low tiers are not
harmful, they are nearly empty. Grinding the 836 remaining pids projects to about
−56 moves for ~35 hours of compute.

**AZ dual-head (`taz_v1`) — flat as a scorer, real as a diversity source.**
100 ep on GCP, 5.0M params, policy top-1 48.7%, V head exported by
`scripts/22_export_az_v_only.py`. On 13 common pids at the standard bench config:

| | total |
|---|---|
| tv0 ep24 | 437 |
| taz_v1 | 434 |
| **min-merge of the two** | **421** |
| floor | 368 |

Alone it is a wash (−3, noise). But the two models disagree hard per pid
(995: 37→31, 900: 40→36, against 950: 29→34, 50: 36→41), so min-merging them is
worth **−1.00 move/pid** over the better single model. That is the payoff — a
second independent solver to merge, not a better scorer. Gates: V(solved) −0.013,
V(d=1) 1.028, d=6 5.908, **std(V@d20) 2.12 (best of any model)**, V@d80 19.68
(fails the absolute band like every other model — that gate still does not predict
beam quality here). Merged output remains +4.08/pid worse than the floor, so it
does not change the verdict. Use AZ runs only on floor>=32 pids. The policy head
was NOT wired into search (untested here; historically regresses).

**Diversity is our strongest lever, and still not enough.** Min-merging 9 diverse
configs (different models, widths, frame counts) on 14 common pids gives
**−2.86 moves/pid vs the best single config** — but the merged result is still
438 against the floor's 398 on those pids, i.e. **+2.9/pid worse**. This also
explains the floor's quality: it is almost certainly itself a community min-merge
over many independent solvers, which is the same mechanism.

Realistic ceiling for this approach: **~29,415** against a 28,863 gate.
Everything else has been falsified as a lever — V saturated across 3 models and
2 recipes, frames saturate at 4, width worthless on tight pids (TPU at 1M lost to
the floor), endgame already exact for the last 6 moves. Reaching 28,863 needs a
solver averaging BELOW 29.6/pid — a different method class (two-phase/coset with
large pruning tables, which is near-certainly what 28,481 and 28,863 are), not
more compute on this one.

### Yield is governed by the FLOOR's slack, not by our beam width

| tier | width | pids measured | yield |
|---|---|---|---|
| floor >= 33 | 262K x 4 | 33 | **−1.18/pid** (16/33 wins) |
| floor == 32 | **524K** x 4 | 20 | **−0.60/pid** (7/20 wins) |

The floor=32 band ran at DOUBLE the width and yielded HALF as much. What we are
harvesting is slack in the public solution, and that slack is concentrated in its
worst pids. Extrapolating a rough halving per floor-length step:

```
floor>=33   60 x -1.18 =  -71     floor=30  278 x ~-0.20 = -56
floor =32  104 x -0.60 =  -62     floor=29  189 x ~-0.12 = -23
floor =31  228 x ~-0.35 = -80     floor<=28 141 x ~-0.05 =  -7
                                              TOTAL  ~ -299  ->  ~29,320
```

against a gate of −759. **The grind alone plausibly lands near 29,300.** On pids
where the floor is already near-optimal there is no slack to harvest — beating it
there requires being genuinely better than the community solver, not just finding
its mistakes. `scripts/../data/yield_sample_pids.txt` (40 pids, 10 each at floor
31/30/29/28) replaces the three guessed numbers with measurements; it is armed to
run on GCP the moment top-60 exits (`run_yield_sample.sh` polls for it).

**Measured rate (first 10 of the top-60, 262K x 4 frames): −0.7 moves/pid.**
Read that as the OPTIMISTIC end — the top-60 are precisely the pids where the
floor is worst, so easier tiers will yield less. −0.76/pid across all 1000 is what
the bar needs, so this config alone lands short and the width has to come up.
That is the whole case for the TPU.

**Targeting rule**: sort pids by floor length descending and spend the widest
beam there. The public notebooks work the same way (theirs runs
`list_states_to_solve: range(990,1000)` — 10 pids per session), accumulating
wins into a merged CSV across many runs. Ours merges via
`scripts/41_merge_tpu_results.py`.

## ABLATION RESULT (2026-07-26/27): width is the lever, history is not

40 pids (30 longest floor + 990-999), AZ model, **1 frame**, endgame table on,
v6e-4. This SUPERSEDES the earlier "width is not the lever / ceiling ~29,410"
conclusion, which was drawn from a broken endgame (~29% stalls) and is wrong.

| beam | h | solved | total | vs floor | wins | mean wall |
|---|---|---|---|---|---|---|
| 128k | 1 | 31/40 | 1076 | +58 | 2 | 13 s |
| 128k | 4 | 31/40 | 1077 | +59 | 2 | 15 s |
| 128k | 8 | 31/40 | 1077 | +59 | 2 | 18 s |
| 1M | 1 | 40/40 | 1323 | −1 | 14 | 108 s |
| 1M | 4 | 40/40 | 1316 | −8 | 14 | 158 s |
| 1M | 8 | 40/40 | 1316 | −8 | 14 | 218 s |
| **4M** | **1** | **40/40** | **1270** | **−54** | **25** | 416 s |

* **Width has not saturated.** 128k->1M = −65 on commonly-solved pids, 1M->4M =
  another −53. At 4M we beat the floor by 54 moves on 40 pids at ONE frame
  (production uses 4, separately worth −3.2/pid).
* **history_depth is a LENGTH optimization, not a solvability one.** It never
  changes what is solvable; it shortens paths. Complete cells (40 records each):

  | config | 128k solved / vs floor | 1M solved / vs floor | 1M mean wall |
  |---|---|---|---|
  | h0 (nothing) | 31/40 / +71 | 40/40 / +12 | 94 s |
  | h0 + non-backtracking | 33/40 / +84 | (pending) | — |
  | h1 | 31/40 / +58 | 40/40 / −1 | 108 s |
  | h4 | 31/40 / +59 | 40/40 / −8 | 158 s |
  | h8 | — | 40/40 / −8 | 218 s |

  Cost per move saved at 1M: **h0->h1 = −13 moves for +14 s/call** (best value),
  h1->h4 = −7 for +50 s, h4->h8 = 0 for +60 s. Identical solved-pid sets across
  all depths at both widths. **Default h=1**; h=0 is defensible if wall matters
  more than ~0.3 moves/pid.

  > **RETRACTED (2026-07-27):** an earlier version of this file claimed h0 solves
  > 2/40 and is "BROKEN", and that non-backtracking is a "DEAD END" at 1/40. Both
  > were artifacts of analysing PARTIAL result files -- the monitor pulled each
  > JSON as soon as it was non-empty, but the driver writes after every pid, so
  > those were 4- and 2-record snapshots reported as "/40". See the process note
  > below. h0 solves 31/40 at 128k and 40/40 at 1M.

* **Non-backtracking (free move-mask) is implemented, off by default.** At 128k it
  solved MORE pids than h1 (33 vs 31) but produced longer paths (+84 vs +58) --
  a coverage-for-length trade that is not yet explained. 1M cell pending.

## PROCESS RULE: an analysis must print its denominator

Three findings were reported wrongly this session because result files were read
mid-write. The driver writes its JSON after EVERY pid, so "file is non-empty" is
not "cell complete". The analysis script printed the solved COUNT but never the
RECORD COUNT, so nothing in the output revealed that 2 solved was 2-of-4 rather
than 2-of-40.

1. Any watcher that pulls remote results must gate on the expected record count,
   not on file existence/non-emptiness.
2. Every summary table must carry a `recs` column. If the denominator is not on
   screen, the number is not trustworthy.
* **The gain is CONCENTRATED.** Of the −54: the 30 longest-floor pids give −55,
  and 990-999 give **+1** (2 wins of 1 move each, 991/992 lose). Tight floors
  have no slack to harvest regardless of width. Target by floor length.
* Per-pid variance is large: pid 998 went 33 (1M) -> 29 (4M) on width alone.

Merged into `submissions/final_best_v2.csv`: **29,444** (30 pids won by TPU).

## LOST DATA (2026-07-27) -- flex VM auto-delete

The `h0` / `h0+nbt` cells (isolating the FREE non-backtracking mask from
history_depth) were queued behind the main grid on the same VM and almost
certainly ran, but were never pulled. `--max-run-duration=24h
--instance-termination-action=DELETE` deleted the instance with the results on
its local disk. **Rule: on a flex VM, pull each cell's output as it completes, or
stage to GCS. A queued follow-on job must fit inside the deletion deadline with
margin, and its results must not live only on the instance.**

## WIDTH ABLATION on the hard tail (pids 990-999), 2026-07-28

h=1 exact, endgame on, 1 frame, AZ model. These ten pids have TIGHT floors
(29-33) -- the group where 4M gained nothing in the 40-pid ablation.

| width | total | vs floor | beats floor | beats best-known | wall/pid | machine |
|---|---|---|---|---|---|---|
| 4M | 311 | +1 | 2/10 | 0/10 | 416 s | v6e-4 |
| 8M | 306 | −4 | 2/10 | 0/10 | 802 s | v6e-4 |
| 32M | **301** | **−9** | **5/10** | 1/10 | 1635 s | v6e-8 |

floor = 310, best-known (merged 28,821) = 300.

* **Width has NOT saturated even at 32M.** ~5 moves per doubling on this group,
  monotone, no plateau. pid 999 first broke at 32M (31->30); 994 fell 33->30.
* **But returns are expensive.** 8x width (4M->32M) = 10 moves over 10 pids at
  4x the wall. At 32M a single-frame beam only reaches PARITY with best-known
  (301 vs 300) after ~4.5 h.
* Practical planning fact: **16M on 4 chips costs 1700 s/pid while 32M on 8 chips
  costs 1635 s/pid** -- the 8-chip box runs double the width for slightly less
  wall. For any wide run the v6e-8 is the better buy outright, not just a way to
  fit a larger B_local.

## v6e-8 via Queued Resources works, and is 2.2x faster (2026-07-28)

The 8-chip machine came up cleanly through the QR API in ~6 min
(`WAITING_FOR_RESOURCES` -> `PROVISIONING` -> `ACTIVE`). Do NOT use
`instances create` for single-host `ct6e-standard-8t` -- backend-broken, reaches
STAGING then INTERNAL_ERRORs and self-deletes.

```
gcloud alpha compute tpus queued-resources create tetra-v6e8-qr \
  --project=gen-lang-client-0977634337 --zone=us-east5-a \
  --accelerator-type=v6e-8 --runtime-version=v2-alpha-tpuv6e \
  --node-id=tetra-v6e8 --provisioning-model=flex-start --max-run-duration=24h
```

* `gcloud compute tpus tpu-vm ssh` hits the SAME Windows path bug as
  `compute ssh` (`'C:\...\Google\Cloud' is not recognized`). Use native ssh, and
  push the key to **project-level** metadata (`gcloud compute project-info
  add-metadata`) -- TPU VMs do not take the instance-level metadata trick.
* jax is NOT preinstalled on `v2-alpha-tpuv6e`; pip `jax[tpu]` + torch-cpu.

**8-rank correctness gate PASSED.** pids 990/991/992 at B=1M returned
**32 / 31 / 34**, identical to both 4-chip runs. This exercises a path the puzzle
had never used: all-to-all fan-out to 8 owners instead of 4, rescaled
`K_per_peer`, and the 3-bit BPTR rank field beyond value 1. No kernel edit was
needed (24 PL / 3 rank / 5 move already covers 8 ranks and B_local <= 16.7M).

**Speed:** same work took 42/48/53 s on 8 chips vs 104/99/111 s on 4 -- **2.2x**,
i.e. better than linear in chip count, because wall is super-linear in `B_local`
(halving B_local more than halves per-step cost). Practical consequence: 32M on
8 chips has the same per-rank load as 16M on 4 (B_local 4M, 8.8 GB child array),
which is what makes 32M feasible at all -- on 4 chips it would need B_local 8M
and a 17.7 GB child array.

## TPU kernel v3 (2026-07-26): endgame table + sharded history

Two additions to `jax_beam_spmd_v_only.py`, both in the PACKED body only
(`_build_step_body_v_only_packed`; the streaming body is untouched).

### 1. Exact endgame table -- the fix that mattered

**Diagnosis first.** ~29% of beam calls at B=1M returned NOT FOUND, all at
uniformly 120-121 s = the full 60 steps. The min-V-per-step trace settled it:

```
15.2 14.5 13.9 ... 2.0 1.6 1.6 0.7 0.7 0.7 ... (0.7 for the last 32 steps)
```

The beam descends cleanly to 0.7 and then FLATLINES. It reaches the goal's
neighbourhood and cannot close, because V is confidently wrong exactly there --
a V trap, not a cycling problem. Width does not help (all 1M slots are in the
same basin) and neither does forbidding revisits.

Our GPU beam never hits this because it never solves to the exact goal: it stops
on any node inside the d<=6 table and splices an optimal tail, exiting before the
trap. So the kernel now does the same -- goal test = "inside the d<=6 table",
Zobrist-keyed on device (88 XOR-gathers, reusing `_sorted_contains`), tail
reconstructed host-side in the driver, every path replay-verified (which also
catches any hash false-positive for free).

Result on the pids that stalled: **5 of 6 failures became verified solves**
(211 fwd/inv 37/36, 449 fwd 35, 532 fwd/inv 36/39). Cost ~1.5x wall per call.
The one remaining failure shows the same 0.7 plateau -- there V scores 0.7 for a
state genuinely MORE than 6 moves out, so the trap reaches past the table. A d<=7
table (415M states) would cover more of it.

### 2. Sharded cross-layer history (`history_depth`, default 0 = off)

Owner rank = `owner_hash(state) % world_size` is deterministic, so a state always
routes back to the same rank and **each rank's own history is complete -- no
extra collective**. Carry is `(world_size, history_depth, B_local)` int64, each
row sorted, -1 sentinel; the filter folds into the existing `drop_mask` beside
duplicate/padding masking; after top-k the selection is sorted and the ring
rolled. Threaded through `step_fn`, the shard_map specs (11->12 outputs),
`donate_argnums`, the `lower()` avals and the step loop.

Honest status: it is correct and runs, but it did NOT fix the stalls (that was
the endgame table) and at depth 8 it costs 3.4x wall. Keep it default-off.

### Gotchas from this port

* **`jnp.searchsorted` is unusable inside shard_map on jax 0.6.2.** It lowers to
  a `lax.scan` whose 2-component int32 carry (lo, hi) is core-invariant in and
  core-varying out -> "the varying manual axes do not match". `method=` does not
  help. jax 0.10 on CPU ACCEPTS it, so the CPU smoke passes while the TPU fails.
  Fixed with `_sorted_contains`: a hand-rolled binary search, statically
  unrolled, pure jnp ops, no scan. Works on both versions.
* **Verify the artifact ON THE REMOTE before believing its output.** A
  `tar -cz ... | ssh 'tar -xz && python ...'` one-liner silently failed to
  overwrite, and I diagnosed the same error twice against the ORIGINAL file.
  Use `scp` then `grep -c <new symbol>` on the remote, and clear `__pycache__`.
* `pkill -f <tag>` over ssh self-matches its own command line and kills the
  shell before the launch (CLAUDE.md 7g). Kill by PID via
  `ps -eo pid,cmd | grep "[t]ag" | awk '{print $1}' | xargs -r kill`.

## TPU: 1M vs the floor on TIGHT pids (earlier, pre-endgame)

v1 (`ALPHA=1`) found **0/8**. v2 (`ALPHA=2`, NUM_STEPS 60) found **8/8**, all
replay-verified. `NUM_STEPS` was not the cause — the paths are 31–35 moves, inside
v1's 45. So `K_per_peer = ALPHA * B_LOCAL // world` is load-bearing: at ALPHA=1
each owner receives exactly `B_LOCAL` candidates and its top-k is a NO-OP, leaving
only sender-side per-bucket selection. (I briefly talked myself out of this on the
theory that random hash routing makes per-bucket top-k ~ global top-k. That
reasoning is wrong — the good candidates are not spread uniformly across buckets.
Use ALPHA=2, as the cube kernel does.)

Result at B=1M, 2 frames, ~31 s/call:

| pid | floor | our GPU best | TPU 1M |
|---|---|---|---|
| 990 | 33 | 33 | 33 |
| 991 | 29 | 29 | **31** |
| 992 | 32 | 32 | 32 |
| 993 | 30 | 30 | **31** |
| total | **124** | **124** | **127** |

A 1M-wide TPU beam is **3 moves WORSE than the floor** on these four and no better
than our 65K–524K GPU results. Consistent with the tier finding: these pids have
tight floors, and width cannot manufacture slack that is not there. Throughput is
fine (~250 pids x 4 frames per 9 h session) — width is simply not the lever.
**Do not scale to 32M on this evidence.**

## TPU path

`kaggle_notebooks/tpu_beam_tetraminx/` holds the ported kernel, `cpu_smoke.py`
(CPU shape validation — run it before any Kaggle session) and `build_notebook.py`.
The notebook does NOT inline the kernel: `jax_model.py` and
`jax_beam_spmd_v_only.py` ship inside the asset dataset and the notebook puts the
dataset on `sys.path`, so TPU runs exactly what the CPU smoke validated.

Assemble + publish:

```
.venv/Scripts/python.exe tetraminx/scripts/40_build_tpu_dataset.py --checkpoint <ckpt>
# PowerShell (token must be re-exported per session -- a 401 is almost always this):
$env:KAGGLE_API_TOKEN="..."; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
.venv/Scripts/kaggle.exe datasets create -p tetraminx/kaggle_datasets/tetraminx-tpu-artifacts
```

Plan: validate at `B_GLOBAL = 1M`, then raise to 8M/32M. `PARENT_CHUNK` must be
set (e.g. 131072) once B_GLOBAL >= 32M. Pull each version's output BEFORE pushing
the next — Kaggle serves only the latest version's output.

## FINAL RESULT (2026-07-26): 29,477 — `submissions/final_best.csv`

1000 pids, every path replayed against the competition states and asserted to
solve. **Unsubmitted** per the user's rule (nothing goes up until <= 28,863).

```
public floor                              29,622
GCP top-60      (floor 33-36, 60 pids)       -67   29/60 wins
local tier-2    (floor 32,   104 pids)       -55   35/104 wins
yield sample + diagnostic benches + AZ        -7
window reduction (harvests the FLOOR's slack)-16
                                          --------
                                          29,477   (-145)
```

All runs are complete; nothing is left running and the GCP VM is stopped.
Measured ceiling if the remaining 776 low-tier pids were also ground: ~29,400
(the yield curve says -74 for 35-40 h of GPU). The gate is 28,863.

## Production runs (completed)

Working model is `tv0_bellman/epoch_0024.pt`. Working config is
`--history-depth 10 --sym-frames 4 --bf16` plus the exact endgame, merged against
the floor so output is never worse.

| where | pids | width | status |
|---|---|---|---|
| GCP L4 | top-60 (floor 33–36) | 262K x 4 frames | ~5 min/pid. **First 10 pids: 4 wins, −7 moves (−0.7/pid)** — 159 36→34, 614 35→34, 156 34→33, 431 34→31 |
| local 4090 | tier-2 (104 pids, floor=32) | 524K x 4 frames | detached via `Start-Process cmd /c`, log `logs/local_tier2.log` |
| Kaggle TPU | 990–993 | 1M x 2 frames | queued |

Don't run two wide local beams at once — 524K x 2 processes filled the 4090's
16 GB (15.9/16.4 used) and both thrash.

## SPARSE-Q HEAD (`tq0`) -- launched 2026-07-28 on the L4

An all-neighbours Q head (24 outputs, one per action) trained with Vlad Kuznetsov's
"random-walk middle sparse-Q" objective (github.com/AnanasClassic/cayleypy-training-core,
same author as the two-phase method and the megaminx Q idea). Same trunk as tv0
(2048/512 + 2 res blocks, embedding), so 5,008,280 params against tv0's 4,996,481.

**Why.** Width is the one lever here that has not saturated (128k->1M = -65 on 40
pids, 1M->4M = -53 more) and it is compute-bound at 416 s/pid. A Q head scores all
24 children from ONE forward on the parent instead of 24 forwards on the children.
Measured end-to-end in `KhoruzhiiSolver._do_greedy_step` (L4-class model, fp16):

| B | V per-child | Q exhaustive | Q + progressive top-k |
|---|---|---|---|
| 16,384 | 155.3 ms | 8.5 ms (18.3x) | **6.4 ms (24.4x)** |
| 65,536 | 578.7 ms | 33.7 ms (17.2x) | **26.4 ms (21.9x)** |

**The objective.** Walk k steps from solved, pick pivot p, label exactly two of the
24 actions: `Q(s, undo) = p-1`, `Q(s, next) = p+1`. Our V loss is MSE(V(s), k), and
at large k the conditional variance of true distance is large, so the MSE-optimal
prediction shrinks toward the mean and the local discrimination flattens. Here the
two labels always differ by exactly 2 with zero conditional variance, so the loss
cannot be reduced by flattening the gap. Measured on tv0_bellman ep24 over 8,192
rw-middle pivots, mean `V(next) - V(undo)` by pivot band: 1.85 (1-4), 1.78 (10-14),
1.22 (20-24), **0.44 (30-40)** -- exactly the collapse the sparse-Q label forbids.

**Four additions of ours over upstream** (`51_train_sparse_q.py`):
1. Exact 24-way Q anchors from the BFS tables. Children of a d<=5 state are all
   d<=6, hence all present in `bfs_endgame.npz`, so every one of the 24 outputs
   gets a TRUE distance. 512 anchor rows per step. Data already existed.
2. Symmetry-expanded label coverage over the 24 spatial frames. Achieved width is
   14 rows/sample covering 8-16 of 24 columns -- NOT 24, because the group permutes
   the 4 axes but not the 3 layers, so an action's orbit has 8 members.
3. Depth-tilted pivot sampling (`pivot_tilt`); the upstream scheme is heavily
   shallow-weighted and our paths are 30-36 long.
4. `top1_margin_weight` as a live knob (upstream ships it at 0.0 everywhere).

**k_max is a trap.** Upstream uses k_max 70 against a ~29 diameter, but that only
survives because its pivot scheme keeps the mass shallow. Past the diameter the walk
is near stationary and "undo the last move" is no longer reliably distance-reducing,
so the RANKING label itself degrades -- not just its scale. Tilting pivots deeper
removes that protection, so `tq0` runs k_max 40 / tilt 0.5. k_max x tilt is the first
ablation axis if it underperforms.

**Correction to an earlier claim in this session**: inverse antisymmetry does NOT
give a free Q label pair. Distance is inverse-invariant, but our moves act on the
right, so removing the walk's last move from `s` is a right-multiplication while the
corresponding operation on `s^-1` is a left-multiplication. There IS a valid
one-sided construction (for `inv(s_p)` the good move is the walk's FIRST move `g_1`,
and a negative needs a pre-move prepended to the walk) -- deliberately NOT
implemented in v1 rather than risk a silent label bug.

**Status.** Running in tmux `tq0` on cayley-gpu, 12.1 s/epoch compiled, 1500 epochs
(~5 h), checkpoints every 50 into `tetraminx/models/tq0/`.

### MEASURED A/B at epoch 250 of 1500 (local 4090, 15 stratified pids, exact endgame)

pids 0,50,100,200,300,400,500,600,700,800,900,950,990,995,999. Floor on these = 429
(28.60/pid). `history_depth` was 0 in all cells -- the handoff's h=1 gain is still on
top of these numbers.

| scorer | frames | total | /pid | vs floor | beats floor | wall |
|---|---|---|---|---|---|---|
| V @ 65k (`tv0_bellman` ep24) | 1 | 506 | 33.73 | +5.13 | 0 | 208 s |
| Q @ 524k | 1 | 484 | 32.27 | +3.67 | 0 | 80 s |
| Q @ 1M | 1 | 468 | 31.20 | +2.60 | 0 | 147 s |
| **Q @ 1M** | **4** | **444** | **29.60** | **+1.00** | **1** (+5 ties) | 590 s |

Q@1M beats V@65k on 11 of 15 pids at 1.4x LESS wall. Q@524k is -22 moves at 2.6x less
wall. The handoff's long-standing "our beam produces ~32/pid almost regardless" is now
29.60/pid, and pid 990 is the first stratified-sample pid we have ever taken BELOW the
floor (33 -> 32). Biggest single win: pid 900, V 40 -> Q 30.

**Do not over-read it.** Merge yield on this sample is still ~zero (428 vs the floor's
429) because a stratified sample is dominated by tight floors, and yield is governed by
the floor's slack (see the yield-tier table above). The measured claim is that our
deficit vs the floor fell from +5.13 to +1.00 moves/pid, which should convert far more
of the floor>=32 tier. **Next measurement: the floor>=33 and floor=32 tiers**, where
the old V at 262Kx4 yielded -1.18 and -0.60/pid.

### The mechanism claim was WRONG -- record it

The stated reason for expecting a win was that the sparse-Q label pins the two labels
exactly 2 apart at every depth, so MSE cannot flatten the discrimination gap the way
walk-depth regression does. **It does not hold.** At epoch 250 the Q's gap collapses
FASTER than the V's:

| pivot band | Q gap | V gap | Q top1 | V top1 |
|---|---|---|---|---|
| 1-4 | 1.879 | 1.845 | 0.779 | 0.768 |
| 10-14 | 1.470 | 1.746 | 0.486 | 0.493 |
| 20-24 | 0.773 | 1.183 | 0.215 | 0.227 |
| 30-40 | 0.216 | 0.401 | 0.102 | 0.081 |

Why: a given (s,a) is labelled p-1 when a walk ARRIVES via a^-1 and p+1 when one LEAVES
via a. At depth the walk is near stationary, the two roles become equiprobable, and the
MSE optimum averages them -- the same collapse, reached by a different route.

What actually carries the win is (a) **top-1 parity at ~22x less compute per step**
(aggregate 0.4918 vs 0.4872, at 1/6 of the training budget), and (b) **much better
absolute calibration**: at band 30-40 the Q's score for the undo move is 27.46 against a
true optimum near 28, where the V says 21.65. Calibration matters because the beam takes
a GLOBAL top-B across parents, so scores must be comparable between different parents.

**The Q is not a drop-in at narrow beam.** At a matched 65k it failed to solve 3 of 5
pids. It buys width and it needs width. Do not swap it into existing narrow configs.

### Slack-tier run vs the REAL bar (28,821), Q@1M x 4 frames, `tq0` ep500 -- COMPLETE

156 pids at merged-best >= 31 (15 at 32, 141 at 31), longest-first, 6,693 s.
All 156 paths replay-verified against `data/test.csv`, 0 invalid.

| slice | pids | wins | merge yield |
|---|---|---|---|
| merged-best = 32 | 15 | 4 | -9 (-0.60/pid) |
| merged-best = 31 | 141 | 24 | -37 (-0.26/pid) |
| **total** | **156** | **28** (+60 ties) | **-46** (-0.29/pid) |

**28,821 -> 28,775.** For scale, our entire pipeline previously contributed 22 moves
to the 28,821 merge, so this triples it in one run. Biggest wins: pid 799 32->28,
plus 237 / 375 / 713 / 940 all 31->28.

Note the tier's own raw total (4,924) is only -7 against the OLD public floor on the
same pids -- the yield is entirely in per-pid wins against the merged min, which is
why merge yield is the only number worth quoting. Rule 26.

## ARCHITECTURE COMPARISON MATRIX (2026-07-29/31)

User-requested head-to-head: ResMLP / GFlowNet / PieceTransformer x no-AZ / AZ head,
all trained on the SAME sparse-Q objective at a matched 1500-epoch budget, then beam
evaluated. Code: `tetraminx/src/tetraminx/models.py` (one interface for every cell:
`forward(states) -> (B, 24)`, `value()` when an AZ head exists, `get_model_config()`
round-tripping through `build_model`), configs `configs/mx_*.yaml`.

| cell | arch | AZ | params | where |
|---|---|---|---|---|
| 1 | resmlp | no | 5,008,280 | = `tq0` |
| 2 | resmlp | yes | 5,008,793 | L4 |
| 3 | transformer | no | 3,444,504 | spot A100 |
| 4 | transformer | yes | 3,444,761 | spot A100 #2 |
| 5 | GFlowNet, TRUE trajectory balance | -- | ~3.4M | L4 |

**A 6th cell was dropped as provably redundant.** `ResMLPGFlowNet` trained with
sparse-Q is bit-identical to the plain ResMLP cell -- same trunk, `policy_head` is
the same `Linear(512,24)` as `head`, and a smoke test produced identical loss to 4
decimals. The GFlowNet *architecture* contributes nothing; what makes a GFlowNet a
GFlowNet is the TB objective, so cell 5 trains that instead (`53_train_gfn_tb.py`).

### Cell 5, TRUE trajectory-balance GFlowNet: FALSIFIED 2026-07-31, killed at ep410

Not a crash -- a converged degenerate solution, diagnosed and stopped.

| signal | value | reading |
|---|---|---|
| TB loss | 0.0010 (rms 0.031) | balance satisfied to 3 decimals |
| logF(d1) vs logF(last) | 34.80 vs 35.67 | **flow FLAT over 32 steps of depth** |
| lnZ | 100 -> 34.5 | collapsed |
| within-parent score spread | 12.03 | policy is CONFIDENT, not uniform |
| `P_B` argmax == undo | **0.0260** | **below** the 0.0417 uniform baseline |

Sign error in the `GFlowNetTBQ` adapter was ruled out by probing the RAW policies:
`P_B` argmax 0.0260 / argmin 0.0405, `P_F` argmax 0.0332 -- all ~random in both
directions. The adapter is correct; the policy is empty.

**Mechanism (identifiable, not mysterious).** Prefix-TB over FIXED-LENGTH
trajectories with a learnable lnZ has **no terminal reward**. Nothing in the
objective states that short paths are good; the model need only be self-consistent
with the distribution IT generates, and on-policy sampling closes that loop. A flat
flow with matched P_F/P_B is an exact optimum, and it found one. Training longer
cannot escape it -- TB was already 0.001 at epoch 410 of 1500.

**What would actually be required**: a reward anchor (terminal reward from the BFS
table, or a reward-conditioned GFN). That is a different algorithm, not a
hyperparameter fix. Combined with [[gfn-pathfinding-ported-gated]] ("full megaminx
needs TPU budget", diameter ~41 at laptop scale), the recommendation is to treat
plain TB as closed for this puzzle.

The recipe traps from that memory WERE applied and were not the problem:
`eps_explore=0.05`, two-stage lambda (0 -> 1e-8 at ep300), learnable logZ at 10x lr.

### AZ head: NEUTRAL, not harmful (cells 1 vs 2, both 1500 ep)

Beam A/B, 15 stratified pids, Q@1M, 1 frame: cell 1 = **468**, cell 2 = **461**
(6 wins / 6 ties / 3 losses). That is 0.47 moves/pid, INSIDE this bench's ~±2/pid
resolution, so a tie. It does NOT reproduce the 8-16 pct multi-task penalty that
`az_tb_dual_head_findings` predicted. Note the probe had claimed AZ was worse in
every band -- the beam disagreed.

**The AZ head only changes beam results through trunk interference** unless inference
actually uses the value head (Q shortlist -> V rerank). That second mode is still
unmeasured.

### Transformer efficiency: 2.9x recovered by profiling, not guessing

The transformer was initially 470 s/epoch (8.2 days for 1500). Three fixes, each
verified numerically exact against a saved reference:

1. **Folded input stage** -- push `piece_projection` into the value table so a token
   is `sum_j table[j, v_j] * mask_j + bias`. Removes the `(B,P,K,D)` intermediate
   (~10 GB at TPU batch sizes, 2.3 GB at training batch). Required, not optional.
2. **Fused-QKV SDPA** replacing `nn.MultiheadAttention` (`_SelfAttn`, same parameter
   NAMES so checkpoints and the JAX loader keep working). 484 -> 450 ms. Small,
   because attention is only **3.2 pct** of this model's MACs at T=51.
3. **One-hot matmul input stage** -- 450 -> **162 ms**, the real win. A gather's
   backward is a scatter-add, and 384k gradients per slot contended atomically for
   an 88-row table. Freezing the input stage alone took the step 472 -> 158 ms, i.e.
   the embedding gradient was **67 pct of the entire step**. Recast as
   `one_hot(vals) @ table` and both directions become dense GEMMs.

Result **470 -> 159 s/epoch**. Diagnosis method worth reusing: a GEMM roofline at our
own matrix shapes (157-202 TFLOP/s on an A100) against achieved throughput (~17)
proved the matmuls were never the limit, and freezing parameter groups localised the
cost in one measurement.

### CORRECTION: the "Rule 22 compile hang" was a misdiagnosis

An earlier note here blamed `torch.compile`. Wrong. The real cause was memory: base
512 peaks at **17.3 GB**, and on a 16.4 GB card **Windows WDDM pages instead of
OOMing**, taking the step from 340 ms to 3,617 ms with no error and no traceback.
`torch.compile` is fine on this model -- 18.6 s warmup on an A100, ~12 pct faster.
The same trap recurred on the INFERENCE side: the default `--chunk-size 32768` gives
the transformer ~855 MB per activation tensor, hit 16.0 GB, and one beam pid took
>16 min; at `--chunk-size 8192` memory is 5.8 GB and a pid takes 360 s. **On this
16 GB card, "no error but inexplicably slow" means paging -- check memory first.**

### TPU: Q-mode ported to the JAX kernel and validated

`jax_beam_spmd_v_only.py` gained `q_mode`: score children from ONE forward on the
PARENTS. `states[:, all_moves].reshape(-1, S)` puts child (i,m) at flat index
`i*n_gen+m`, exactly the flattened layout of a `(B_local, n_gen)` Q output, so nothing
downstream changed. It FORCES `pack_v_score`, because a Q head cannot rescore a bare
state on the receive side (it returns n_gen action scores, not a state value) -- the
send-side score must travel with the candidate.

Validated on the v6e-8 against the GPU, same model and config: pid 50/100/200 =
33/31/34 vs the GPU's 33/33/34, all `verify=True`. **20 s/pid vs the V-only kernel's
42-53 s** at the same width. Note that is 2.3x, not the 22x measured on GPU -- the
TPU step is selection-bound (all-to-all, hashing, top-k), so Amdahl caps what a
cheaper scorer buys. The TPU's edge remains capacity (32M) plus 8 chips.

`jax_model.py` also gained a JAX PieceTransformer (folded input stage) verified
against PyTorch: argmin agreement **1.000** at every precision.

### Spot-instance operations (learned the expensive way)

Spot VMs created with `--instance-termination-action=STOP` **stay stopped after
preemption; GCP never restarts them**. The first preemption cost ~16 idle hours.
Now: a per-VM boot `startup-script` that resumes from the newest checkpoint (use
`su - and-l -c`, NOT `sudo -u`, which failed with exit 126), plus a 5-minute cron
watchdog on the always-on L4 that starts any TERMINATED A100. Checkpoint every 10
epochs, not 50 -- both preemptions rolled back to epoch 50 because nothing between
50 and 90 was ever written.

### Window reduction on the 28,821 merge: ZERO hits -- local post-processing is done

`42_window_reduce.py` over all 1000 paths of `merged_with_28843.csv`:
**28,821 -> 28,821, 0 window rewrites.** Not one window of <=14 moves in any of the
1000 paths can be shortcut by the exact d<=6 table. (Its docstring predicted this for
near-optimal paths; it earned -16 on the old loose 29,622 floor.)

That is a useful negative: the merge is **locally optimal at the 6-move scale**, so
the remaining slack is structural -- it needs a genuinely different route through the
group, not a local rewrite. Do not spend more time on window/bridge post-processing
against this baseline.

## d7 BFS TABLE built 2026-07-28 -- and d8 priced at ZERO by measurement

`scripts/03b_build_bfs_deep.py` (new; `03_build_bfs.py` cannot reach d7 -- it
materialises the final level's states, 405.6M x 88 = 35.8 GB, and argsorts the whole
433M union). Three changes carry the scale: the last level keeps hashes only (nothing
expands it), the table is assembled by scattering depths into the already-sorted union
instead of argsort, and the expand/hash/seen-filter runs in a fork pool where the
frontier is inherited copy-on-write. Validated at d5 against `bfs_endgame.npz`:
identical ztab, all 1,771,577 hashes present at matching depths.

**Built on the v6e-8 HOST, which is a 180-core / 1417 GB machine.** That is the answer
to "can we rent a big enough CPU": we were already renting one, and it dwarfs the
n2-highmem-96 that was going to be priced. Whole build 179 s.

    level sizes: [1, 24, 408, 6592, 105136, 1659416, 26008172, 405605830]
    endgame table: 433,385,579 states <= d7 -> bfs_endgame_d7.npz   (3.90 GB)
    anchors: 29,777,587 states (full d<=6 plus 2M sampled at d7)    (2.65 GB)

Artifacts in `gs://mm-tpu-staging-0977634337/tetra/` and in `tetraminx/data/`.

### Use 1 -- path shortening: ZERO, and the same run prices d8

The table has two consumers, and only the search-side one was evaluated when d7/d8 were
first (wrongly) dismissed on TPU-memory grounds. Both were tested properly:

| table | `--extend` | effective reach | result on the 28,821 merge |
|---|---|---|---|
| d6 | 0 | 6 | 0 rewrites (previously recorded) |
| d7 | 0 | **7** | 0 rewrites |
| d6 | 2 | **8** | 0 rewrites |
| d7 | 2 | **9** | 0 rewrites |

The merge is locally irreducible out to **9-move windows**, not just 6. Note what the
third row does: `42_window_reduce.py --extend k` BFSes k moves out from the window
element and looks the frontier up, so it reaches `table_depth + k` **without building a
deeper table**. Reach 8 is exactly what a d8 table would have delivered here, and it
found nothing. **d8 must not be built for path shortening** -- that is measured, not
extrapolated.

### Use 2 -- exact-Q training anchors: the real (and only) win

`51_train_sparse_q.py` caps anchors at `table_depth - 1`, because all 24 children of an
anchor must be in the table (there is an assert). So the d6 table pinned anchors at
d<=5 = 1,771,577 states. Against `512 x 256 x 1500 = 196,608,000` anchor draws that is
**111 draws per state** -- and the term is duly memorised: the concurrent `mx_resmlp_az`
run on cayley-gpu shows `anchor 0.0425` against `sparse 8.6103` at **epoch 92 of 1500**,
i.e. it stops contributing gradient 6 percent into the run.

d7 lifts the cap to d<=6 = **27,779,749** states (15.7x) = **7.1 draws per state**, the
first time this term has had generalisation pressure on it. That matters because the tq0
post-mortem found it is **absolute calibration**, not the label gap, that carries the Q's
win over V -- and the anchors are the only zero-noise absolute-scale signal in the mix.

**CORRECTION 2026-07-28 -- the first argument written here for skipping d8 was wrong.**
It said d<=7 anchors would be "0.5 draws per state, fewer than one pass" and treated
that as disqualifying. That is backwards: a low draws-per-state means every anchor draw
is a FRESH exact-labelled example, which is the better regime, not the worse one. And
the anchor set is the COMPLETE ball of radius k, so there is nothing to generalise to
inside it -- the value of a deeper table is coverage one level deeper, which has no
natural stopping point. The measured version of the argument is weaker but real: at
matched epochs the d<=6 anchor loss runs ~1.5x the d<=5 one (ep13 .1857 vs .1343,
ep25 .1362 vs .0976, ep50 .0903 vs .0611), so 15.7x the data bought roughly 2x in
epochs-to-saturation, not 15.7x.

### d8 is UNNECESSARY, not merely unprofitable -- the d7 table already reaches d<=7

The generator set is closed under inverse (checked: 0 of 24 generators lack theirs), so
the Cayley graph is undirected and every child of a depth-d state is at d-1, d or d+1.
Therefore for an anchor at depth exactly 7, a child ABSENT from the d<=7 table cannot be
anything but depth 8. **The d7 table determines all 24 exact targets for every one of
the 431M anchors at d<=7, with no deeper build.**

Verified end to end at small scale: baked d<=5 anchors from the **d5** table with misses
resolved to 6, then checked against the true d6 table -- 4,800,000 targets, all exact,
of which **3,986,879 were resolved by the argument alone and every one is genuinely
d=6**. `03b_build_bfs_deep.py --bake-anchor-depth <max_depth>` turns this on
(`bake_targets(..., miss_depth=max_depth+1)`).

So d8 would buy d<=8 anchors -- one level beyond what d7 already gives free -- for
~250 GB peak RAM and a 60 GB artifact. **Still do not build it**, but for this reason,
not the draws-per-state one.

**Baking is also how a big table gets used at all.** Training never needs the table,
only each anchor's 24 child distances. `--bake-out` writes `{states int8 (N,88),
q_targets int8 (N,24), depths int8 (N,)}`, so a 54 GB hash table collapses to a ~7 GB
file that loads anywhere. Without it the d7 table is 3.63 GB of a 23 GB card; a d8
table (54 GB) would fit neither the card nor a 31 GB training host.

Side benefit: the trainer's `hit.all()` assert passing at d<=6 is an independent
completeness check on the new table (every one of the 24 children of every sampled
d<=6 anchor was found).

### `tq3_d7anchor` -- launched 2026-07-28, single-axis A/B against tq0

NOTE the name: `tq1` and `tq2` were already taken by other sessions on cayley-gpu
(`tq1` = tq0 + path supervision, stopped ep574; `tq2` = k_max 32 / pivot_tilt 0.25,
stopped ep854 and looking strong at top1 0.5652 -- probably the better baseline for
any LATER comparison, though its k_max change makes its loss not directly comparable).

`configs/tq3_d7anchor.yaml` is `tq0_sparse_q.yaml` with only the anchor source changed
(`anchor_states`/`anchor_table` are new config keys; `max_anchor_depth` 5 -> 6). Same
trunk, objective, k_max, tilt, lr, seed; params 5,008,280 both. Running in tmux `tq3`
on **`tetra-tq1`** (g2-standard-8 / L4, **us-west4-a** -- us-east1-b, us-central1-a/c,
us-west1-a and us-east4-c were all STOCKOUT for g2).

**EARLY READ AT ep100: 15.7x more anchors is moving nothing.** Matched against tq0
(identical config bar the anchors; do NOT compare against `mx_resmlp_az`, it has an
az_head):

| epoch | tq0 (d<=5) sparse / top1 / gap | tq3 (d<=6) sparse / top1 / gap |
|---|---|---|
| 25 | 9.3588 / 0.4016 / 1.142 | 9.4200 / 0.4005 / 1.126 |
| 50 | 8.9219 / 0.4257 / 1.226 | 8.9102 / 0.4294 / 1.217 |
| 100 | 8.4254 / 0.4505 / 1.281 | 8.4357 / 0.4546 / 1.277 |

top1 differs by +-0.004 across three checkpoints; sparse and gap are marginally worse.
The anchor loss IS higher (0.0627 vs 0.0312 at ep100), so the extra data is real and
harder to fit -- it just is not converting into decision quality on these metrics.
**Caveat**: these are sparse-Q validation metrics on random-walk pivots, NOT the
band-30-40 calibration the tq0 post-mortem says carries the win. That number is not in
the training log, so this is suggestive, not decisive.

### GATE RESULT: ANCHOR DEPTH DOES NOTHING -- axis closed

`52_eval_q.py` on `tq3_d7anchor/epoch_0200.pt` vs `tq0/epoch_0200.pt`, same probe,
**262,144 pivots** (the first pass at n=16384 showed +0.020 in band 30-40 and -0.019 in
band 25-29 -- both ~1,000 recs, i.e. noise; 16x the sample settled it):

| band | recs | tq3 (d<=6) top1 / s(undo) | tq0 (d<=5) top1 / s(undo) |
|---|---|---|---|
| 20-24 | 26,527 | 0.209 / 22.14 | 0.208 / 22.15 |
| 25-29 | 18,290 | 0.139 / 25.40 | 0.140 / 25.45 |
| **30-40** | **14,909** | **0.095 / 27.56** | **0.095 / 27.59** |
| all | 262,144 | 0.4841 | 0.4827 |

Identical on every band, including the deep band and the calibration `s(undo)` that the
tq0 post-mortem named as the mechanism. **15.7x more exact 24-way anchors buys nothing.**

Measured in the regime most favourable to the hypothesis, too: at ep200 tq0's anchor
term is essentially memorised (loss 0.0219) while tq3's is still learning (0.0456), so
tq3 has live anchor gradient that tq0 does not -- and it changes no outcome. The anchor
term is not what constrains this model.

**Consequences: do not run `tq4_baked_d7`. Do not build d8. The anchor-depth axis is
closed.** Total cost of establishing this: ~$6 and 90 minutes, against a d8 build that
would have reached the same conclusion more slowly.

`52_eval_q.py` needed a fix to get here: it assumed the older `ResMLPDistance`
checkpoint layout and died with `unexpected keyword argument 'arch'` on anything the
current `51_train_sparse_q.py` writes. It now dispatches on the `arch` key through
`model_from_config`. The two flavours are weight-compatible (same head keys, same
5,008,280 params) and both expose `output_dim`, so the Q-vs-V kind check works on both.

### `tq4_baked_d7` -- BUILT AND STAGED, deliberately NOT launched

`data/baked_d7_anchors.pt` (8.56 GB, also in GCS): **75,716,141 anchors at d<=7** with
exact 24-way targets -- 27,779,749 at d<=6 plus 47,936,392 sampled from d7, of which
**996,863,356 child targets were resolved to d=8 by the inverse-closure argument**.
Built in 574 s on a throwaway n2-highmem-32 (`tetra-bake`, since deleted).
`configs/tq4_baked_d7.yaml` + `BakedAnchors` in the trainer are in place, so launching
is one command. **Do not launch it** -- the gate above closed the axis.

CONFIRMED AT CONVERGENCE (tq3 ran the full 1500 ep; same 262,144-pivot probe, ep1500):

| band | tq3 (d<=6) top1 | tq0 (d<=5) top1 |
|---|---|---|
| 10-14 | 0.529 | 0.527 |
| 20-24 | 0.244 | 0.241 |
| 30-40 | 0.101 | 0.099 |
| all | 0.5087 | 0.5078 |

**BEAM A/B (the project's real acceptance gate) is also a null.** tq3 ep1500 vs tq0
ep1500, Q@1M x 4 frames, history_depth 1, 15 stratified pids, `--no-merge`:

```
TOTAL  tq3 429   tq0 432   (-3)
tq3 wins: 50(-3) 100(-1) 300(-2) 400(-2) 900(-1)   = -9
tq0 wins: 200 500 950 990 (+1 each) 999(+2)        = +6
ties:     0 600 700 800 995
```

Do NOT bank the -3: it is **6 wins against 5 losses**, a sign test at p ~ 1.0 on the 11
discordant pids. Per-pid beam variance is +-1-2 moves, so 15 pids has far less power than
the 262k-pivot eval that already said "identical". Both arms did contribute 1 pid each to
the all-sessions merge, so the runs were not wasted.

Two latent VERSION-SKEW bugs were found doing this, both from checkpoints written since
the trainer moved to `models.build_model`, and both would hit ANY such checkpoint:
* `52_eval_q.py`: `TypeError: unexpected keyword argument 'arch'` -- now dispatches
  through `model_from_config` when the config carries `arch`.
* `30_solve.py`: `size mismatch for head.weight: [24, 512] vs [1, 512]` -- the new
  configs name the head width `n_actions`, not `output_dim`; now falls back.

### WHICH MODEL RAN WHICH BEAM -- and a logging gap now closed

Two different scorers were used this session, both at **`history_depth = 1`**:

| beam | where | scorer | settings |
|---|---|---|---|
| `v8val` 1M, `frames_b8M_f4` 8Mx4, `wide_b33M` 32M | v6e-8 TPU | **AZ value head, V-only export** (`taz_v1_v_only.pt`) | alpha 2, no_backtrack off, d6 endgame |
| tq3 vs tq0 A/B | L4 GPU | sparse-Q, **no AZ head** | beam 1M, sym-frames 4, bf16 |

Note the TPU scorer DOES come from the AZ dual-head -- its 24-wide policy head stripped
by `22_export_az_v_only.py`, leaving trunk + value head. That is a different question
from the az_head result above: there the value head is an AUXILIARY loss degrading a Q
head; here the value head IS the scorer, and the pre-compaction A/B had it beating the
pure ResMLP V at B=1M.

**The attribution was nearly lost.** `params: 4,996,481` is identical for the AZ V-only
export and the pure ResMLP V (both trunk + scalar head), the launch commands were never
logged, and the TPU VM is deleted -- so the surviving logs alone cannot say which ran.
It was recovered only from the staged file's byte size. Digests, for future attribution:

```
19,995,032 B  sha256:e42e0027ba33  models/taz_v1/taz_v1_v_only.pt   (AZ value head)
19,997,237 B  sha256:4447572792cc  models/tv0_bellman/epoch_0099.pt (pure ResMLP V)
```

`gcp_beam_tetraminx.py` now logs `checkpoint: <path> (<size> B, sha256:<12 hex>)` before
every run. **Do not remove that line** -- param count alone does not identify a scorer.

### The AZ VALUE HEAD costs ~1 point of top1 on the Q -- do not use it here

`mx_resmlp_az` is `tq0` plus `az_head: true` and nothing else (5,008,793 vs 5,008,280
params, same 1500 epochs, same everything). On the shared 262,144-pivot probe at ep1500:

| band | AZ head | no AZ head (tq0) |
|---|---|---|
| 5-9 | 0.614 | **0.622** |
| 10-14 | 0.504 | **0.527** |
| 15-19 | 0.356 | **0.378** |
| 20-24 | 0.228 | **0.241** |
| 25-29 | 0.149 | **0.157** |
| 30-40 | 0.099 | 0.099 |
| all | **0.4973** | **0.5078** |

**-0.0105 top1, -0.0041 pair**, losing every mid band, deep calibration unchanged
(`s(undo)` 27.76 vs 27.74). At 5M params the shared trunk splits capacity between the
24-wide Q head and the value head and the Q pays for it. Same story as the megaminx
dual-head result ([[az-tb-dual-head-findings]], 8-16% penalty) except the cost lands on
the POLICY side here rather than the value side.

### Scoreboard on the shared 262k probe (the only comparable numbers)

| model | az head | anchors | top1 | note |
|---|---|---|---|---|
| tq2 ep850 | no | d<=5 | 0.5092 | highest top1 but deep calibration BROKEN, see below |
| tq3_d7anchor ep1500 | no | d<=6 | 0.5087 | ties tq0; the anchor axis is dead |
| **tq0 ep1500** | no | d<=5 | **0.5078** | **the incumbent to beat** |
| mx_resmlp_az ep1500 | yes | d<=5 | 0.4973 | worst |

### `tq2`'s apparent +0.10 was a METRIC ARTIFACT -- and it regresses deep calibration

tq2 (k_max 32, pivot_tilt 0.25) reports top1 **0.5652** in its own training log against
tq0's 0.4616, which looks like a large win on the axis this file nominated as "the first
ablation axis". It is not. That number is measured on tq2's OWN validation pivots, which
run to k_max 32 -- a shallower, easier distribution. On the same k_max-40 probe both
models will actually face in the beam (`52_eval_q.py --n 262144`, matched epoch 850):

| band | tq2 top1 / s(undo) | tq0 top1 / s(undo) |
|---|---|---|
| 10-14 | 0.528 / 11.40 | 0.508 / 11.66 |
| 20-24 | 0.234 / **20.39** | 0.229 / **21.98** |
| 25-29 | 0.150 / **22.79** | 0.152 / **25.37** |
| 30-40 | 0.094 / **24.22** | 0.097 / **27.60** |
| all | 0.5092 | 0.5027 |

Real aggregate gain is **+0.0065, not +0.10**, it LOSES top1 in both deep bands, and
aggregate `pair` is slightly worse (0.8829 vs 0.8835).

**The serious part is `s(undo)`.** At band 30-40 tq2 scores the undo move at **24.22
against a true optimum near 28**, where tq0 says 27.60 -- tq2 under-predicts deep
distance by ~3.5 moves. Capping walks at 32 and tilting shallow means it never sees the
deep scale, so it saturates low. That is the same saturation failure that killed the
GT-V, the state_inv encoding and the dodeca CNN, and it bites harder here because the
beam takes a GLOBAL top-B ACROSS parents -- scores must be comparable between parents at
different depths, and tq2's are compressed exactly where the hard pids live.

**Rule: never compare two sparse-Q runs on their own training logs if k_max or
pivot_tilt differ -- the val distribution moves with them. Re-measure both under one
`52_eval_q.py` probe.**

`BFSAnchors` needed a memory fix to get there: it did `.to(device).long()` on the anchor
states, which at d<=6 is 19.6 GB of GPU, plus an int64 copy of the table's depths. Now
states are host-resident int8 (2.28 GB) with only the sampled batch moved, and table
depths stay int8 (the pair is 3.63 GB on the card). The d6 path is unchanged.

**What to measure when it lands**: the deep-band numbers from the tq0 post-mortem table
(band 30-40 top1, currently 0.102, and the undo-move calibration, currently 27.46 against
a true optimum near 28), then a beam A/B at Q@1M x 4 frames against `tq0` ep500/ep1500.

### Width: 4M beats 1M per pid, but 1M wins per SECOND -- escalate, don't start wide

Q@4M x 4 frames vs Q@1M x 4 frames, same 15 length-32 pids, same `tq0` ep500 model,
so width is the only variable:

| config | raw total | merge yield | wall |
|---|---|---|---|
| merged-best | 480 | -- | -- |
| Q @ 1M x 4 | 480 | -9 | ~42 s/pid |
| Q @ 4M x 4 | **468** | **-14** | 166 s/pid |

4M beats 1M on 8 of 15 pids, ties 7, **loses 0** -- width has still not saturated,
which is what the Q head was bought for. But the yield-per-wall-second runs the other
way: 1M returns -0.0143 moves/s, 4M only -0.0056, i.e. **1M is ~2.5x more efficient
per second**. 4M is deeper, not cheaper.

**Strategy that dominates either pure choice**: sweep breadth-first at 1M to cover
every pid, then escalate only the ties and near-misses to 4M. Do NOT start a fresh
tier at 4M -- that spends 4x the wall on pids a 1M pass would have won anyway.

### `tq0` final (ep1500) is meaningfully better than the ep500 the first tier used

| band | ep500 top1 | ep1500 top1 | ep500 gap | ep1500 gap |
|---|---|---|---|---|
| aggregate | 0.4946 | **0.5095** | 1.359 | 1.392 |
| 10-14 | 0.503 | 0.537 | 1.515 | 1.555 |
| 20-24 | 0.226 | 0.249 | 0.809 | 0.857 |
| 30-40 | 0.091 | **0.105** | 0.248 | 0.302 |

Every band improved, including the deep one that had REGRESSED between ep250 and
ep500 (0.102 -> 0.091 -> 0.105). ep1500 now beats the tv0 V baseline (0.4872) by
+0.022 aggregate. Second confirmation in one day that a flat probe metric does not
mean training has stopped paying -- this run was nearly killed at ep500.

Consequence: the -46 move tier result was produced by a two-thirds-trained model, so
re-running the 156-pid tier with ep1500 should beat it.

### WIDTH WAS THE MISTAKE: every tier was ground at 1M, and 4M yields 2.9x more

2026-08-01. Same 15 len-32 pids, same `tq0` ep1500, same 4 frames. **Only the beam
width differs.**

| config | total | wins vs merged-best | merge yield |
|---|---|---|---|
| Q @ 1M x4 -- what ALL THREE completed tiers used | 480 | 4 | -9 |
| **Q @ 4M x4** | **454** | **13** | **-26 (2.9x)** |

Individual pids move a long way: 476 goes 31 -> **28**, 858 33 -> 29, 347 34 -> 30.
Wall 2,679 s for 15 pids on the L4 (~179 s/pid).

**Implication.** The three completed tiers (156 + 306 + 295 = 757 pids, **-118 moves**)
were all run at 1M x4. Re-running them at 4M is plausibly worth **-120 to -280 more**,
which would move 28,703 toward ~28,450-28,580 against Rokicki's 28,481.

**Caveat on that extrapolation**: this is the LOOSEST tier (merged-best 32), where slack
is greatest by construction. len-29 yielded -0.078/pid at 1M x4 against this tier's
-0.46/pid, so lower tiers will amplify less. The direction is not in doubt; the
multiplier is.

This also settles the architecture question in practical terms: the transformer's entire
case was better quality per node, but the SAME ResMLP got 2.9x more yield by spending
compute on width instead. While width is still buyable, buying it beats a better scorer
-- see `RESMLP_VS_TRANSFORMER.md` section 6 for the two-regime framing.

### Solver-consensus is NOT a slack-targeting signal here (null, don't retry)

Idea: pids where only ONE independent CSV achieves the per-pid min are probably still
improvable, while pids where all solvers agree are probably optimal -- so target the
low-consensus ones. Measured over the 7 independent full-coverage CSVs, against our
tier results:

| consensus | pids | mean len | tier attempted | won | rate |
|---|---|---|---|---|---|
| 1 | 395 | 29.19 | 26 | 4 | 15.4% |
| 7 (all agree) | 560 | 28.47 | 54 | 11 | **20.4%** |

The signal runs BACKWARDS -- we win slightly more often where every solver agrees.
Reason: the CSVs are not independent (`current_best*`, `final_best*` all descend from
our own pipeline plus the same public floor), so "consensus 7" mostly means "even the
weak floor found it", which correlates with the pid being easy, not with it being
optimal. Do not use consensus for targeting.

The useful part of that table: **we beat the merged min on 20 pct of pids where all
seven solvers agreed.** Agreement across these CSVs does not imply optimality, which
argues the length-30 tier (306 pids) is worth grinding too.

### `tq1` -- near-optimal-path labels: REJECTED 2026-07-28, killed at ~epoch 560

Beam bench is unambiguous. Same 15 stratified pids, Q@1M, 1 frame, matched epoch 500:

| model | total | vs old floor |
|---|---|---|
| `tq0` ep500 (walk labels only) | **459** | +2.00/pid |
| `tq1` ep500 (+ path labels) | **505** | +5.07/pid |

**+46 moves on 15 pids = +3.07/pid**, far outside this bench's ~±2/pid resolution.
The probe agreed: tq1 was worse in EVERY depth band, including the 30-40 band the
change was designed to fix (gap 0.248 -> 0.186, top1 0.091 -> 0.074).

**Why it failed, and it is in our own notes.** `az_tb_dual_head_findings.md`:
*"V head requires off-path training coverage (random walks)."* States on near-optimal
solving paths are a measure-zero slice; the beam spends nearly all its time on
OFF-path states. The design reasoning checked whether the path labels were CORRECT
(they are, within ~0.2 moves) and whether the megaminx alternative-optima objection
applied (it does not, since our reference paths are strictly shorter). Both were
beside the point -- the failure is that correct labels on a measure-zero set do not
generalise, and at 3.4 pct of every batch with full MSE weight they diverted capacity
away from the walk distribution that actually matters.

Do not retry path/frontier-imitation labels as a *mixin to the walk objective*. If
revisited at all, it should be as a separate fine-tune stage or a separate model kept
only for merge diversity -- never diluting the random-walk coverage.

### TRAP: training-log metrics are NOT comparable across k_max / pivot_tilt settings

`tq2` (k_max 32, tilt 0.25) showed train top1 **0.554** at epoch 294 against `tq0`'s
**0.488** at epoch 1500 -- apparently a large win for the shallower recipe. It is an
artefact. The training log scores each model on ITS OWN sampling distribution, and
`tq2` samples shallower pivots, which are far easier. On the common `52_eval_q.py`
probe (fixed k-max 40 / tilt 0.5) at matched epoch 250 the two are a dead heat:

| | tq0 ep250 | tq2 ep250 |
|---|---|---|
| aggregate top1 | 0.4918 | 0.4919 |
| aggregate pair | 0.8741 | 0.8781 |
| aggregate gap | 1.346 | 1.321 |
| band 30-40 top1 / gap | 0.102 / 0.216 | 0.084 / 0.192 |
| band 30-40 s(undo) | **27.46** | 24.23 |

So **k_max 40 / tilt 0.5 was the right call**: equivalent on ranking, and materially
better on absolute calibration at depth (27.46 against a ~28 optimum, vs 24.23).
Calibration matters because the beam takes a GLOBAL top-B across parents. Always
compare models on `52_eval_q.py`, never on the training log, whenever the sampler
config differs. (Same family as the "print your denominator" process rule above.)

`tq2` keeps its value as a MERGE-DIVERSITY source, which was the primary purpose.

### `tq2` -- diverse second Q model (running, replaced tq1 on the L4)

Same recipe as `tq0` but seed 1, `k_max` 32 (vs 40), `pivot_tilt` 0.25 (vs 0.5). Two
purposes: (a) min-merge diversity, which is this puzzle's strongest measured lever
(9 diverse configs = -2.86/pid vs the best single; tv0 u taz_v1 = -1.00/pid), and
(b) it ablates the k_max/tilt pair, which was chosen by argument rather than
measurement. tmux `tq2`, ~5 h.

### `tq1` -- near-optimal-path labels (design notes, kept for the record)

The measured weakness of `tq0` is that its gap COLLAPSES at depth (0.216 in the pivot
30-40 band, worse than the V's 0.401), because past the diameter a random walk is near
stationary: a given (s,a) is labelled p-1 when a walk arrives via a^-1 and p+1 when one
leaves via a, those two roles become equiprobable, and the MSE optimum averages them.

`tq1` adds a third label source that does not have that failure: states along
replay-verified solving paths, labelled `Q(s, on_path_action) = remaining - 1`. Our
reference is 28.2 moves/pid against a ~28 counting bound, so the target is accurate to
within ~0.2 at exactly the depths where the walk labels rot.

- Dataset: `data/path_labels_28821.pt`, **1,383,408 samples**, built by repointing the
  existing verified `20_build_az_dataset.py` at `merged_with_28843.csv` (NOT the stale
  29,622 floor the old `az_dataset.pt` was built from). dist range 1..32, 24/24 actions
  used, 48 variants per path, every variant replay-asserted to solve.
- Supervised with **MSE on the on-path entry, not a "this move must rank first"
  penalty**. The ranking form is exactly what tied `m_rank_v0` on megaminx, where the
  frontier-regret probe showed 56.9 pct of such disagreements were benign ALTERNATIVE
  OPTIMA. An accurate value on one entry opens the gap against the walk-labelled
  entries by itself without asserting anything false. `path_rank_weight` keeps the
  falsified form available at 0 for ablation.
- Why this should transfer here when it did not on megaminx: there, the beam already
  solved 50/51 and the reference moves were equal-cost alternatives. Here the reference
  paths are **strictly shorter** than what our solver produces (28.2 vs 29.6 moves/pid),
  so the on-path label carries real information.
- `tq1_sparse_q_paths.yaml` is `tq0` plus the `path_*` block and nothing else, same
  seed, so the pair is a clean single-variable A/B.
- Queued in tmux `tq1` on cayley-gpu; it polls for the `tq0` session to end and then
  starts, so `tq0` stays intact as the control.

### `52_eval_q.py` is NOT a stopping gate -- Rule 12 again, in Q space

Epoch 250 -> 500 on the probe looked like a hard plateau: aggregate top1 0.4918 ->
0.4946 (+0.003), and the deepest band 30-40 went BACKWARDS (0.102 -> 0.091). On that
reading the remaining 1000 epochs were waste and the run should have been killed.

The real beam says otherwise. Same 15 pids, Q @ 1M, 1 frame:

| checkpoint | total | vs floor |
|---|---|---|
| ep250 | 468 | +2.60/pid |
| ep500 | 459 | +2.00/pid |

-9 moves, i.e. -0.60/pid. That is BELOW this bench's stated resolution (~±2 moves/pid
at n=15), so formally ep250 and ep500 are a tie -- but there is no evidence of
regression and the direction is right, so the run continues. The point to carry
forward: the probe metric flattened while the beam was still moving. Use the probe to
compare MODELS, never to decide when to stop training. Checkpoints are kept every 50
epochs precisely so the choice can be made on a beam bench afterwards; note `best.pt`
is selected by validation LOSS and should not be trusted as the deploy checkpoint.

**Acceptance gate** (`52_eval_q.py`, the binding one -- NOT a loss curve):
```
python3 tetraminx/scripts/52_eval_q.py --q-model tetraminx/models/tq0/best.pt \
    --v-model tetraminx/models/tv0_bellman/epoch_0024.pt
```
Reads pair/top1/gap per pivot-depth band for both models. What matters is the 20-40
bands, where the V's gap has collapsed to 0.44-1.22. Then A/B on the 40-pid ablation
set (30 longest floor + 990-999) where 128k/1M/4M numbers already exist, so
"Q at 8x B vs V at 1x B" can be read off directly.

**Deployment.** `KhoruzhiiSolver(use_q_function=True)` already existed; this session
added `q_progressive_topk=True` (default on, Q path only), which takes the top ~2B
candidates by Q and hashes only those instead of all 24B, doubling k until B distinct
survive. Verified against the exhaustive path: identical next-state multiset at
B=64/512/4096, blacklist honoured. The (parent, move) provenance can differ, because
the progressive path keeps the BEST-SCORING representative of a duplicated child
while the old path keeps an arbitrary one -- same states, same depth, so path length
is unaffected.

**AZ head later.** Verified compatible: `ResMLPDistance(output_dim=24)` and
`cayley.gflow_model.ResMLPGFlowNet` share all 25 trunk tensors at identical shapes,
and the 24-wide Q head has exactly the shape of `policy_head`. Adding a `value_head`
costs 514 parameters. Do it as a FROZEN-TRUNK probe first -- our own AZ v3 / TB v1
measurement showed an 8-16 percent value penalty from multi-task heads, so a joint
fine-tune should be kept as a separate model for the merge, not a replacement.

**Byproduct**: `50_derive_piece_layout.py` derives a puzzle's physical-piece partition
from its generators (facelets on one piece share a stabilizer), self-verifying as a
block system. Reproduces the upstream hand-written IHES 26-piece layout exactly; for
tetraminx it emits 50 pieces (4 x size-3, 30 x size-2, 16 x size-1) to
`data/piece_layout.json`. That is the missing input for a PieceTransformer here --
but the transformer is NOT the recommendation: it costs ~5x the ResMLP per child
scored, and width is what this puzzle wants. Train it later as a merge-diversity
model, if at all.

## Next

1. TPU: validate at 1M, then run the SAME pids at 8M to get the clean marginal
   value of width under the full config. That number decides whether 32M earns
   its wall-clock. Do not assume it does.
2. Keep grinding pids by descending floor length, merging every result with
   `scripts/41_merge_tpu_results.py` (accepts CSVs with no JSON).
3. Untested cheap levers: `beam_decay` (wide early, narrow late) and
   `--num-attempts` (stagnation retry, already wired).
4. Worth one diagnostic: run CayleyPy's own library beam at matched width on a
   few pids. If it beats ours materially, there is an algorithmic gap to find —
   `history_depth` was one such gap and it was worth 3 moves/pid.
5. AZ dual-head: the measured evidence argues it will not help (three V models
   spanning two recipes bench within noise — the V is saturated as a beam scorer,
   and AZ's megaminx win came through the V head). But it was an explicit part of
   the approved plan, so run it as a BOUNDED experiment when the L4 frees:

   ```
   python3 tetraminx/scripts/21_train_az.py \
     --output tetraminx/models/taz_v1 \
     --warmstart-trunk tetraminx/models/tv0_bellman/epoch_0024.pt \
     --policy-dataset tetraminx/data/az_dataset.pt \
     --bfs-d6-path tetraminx/data/bfs_d6_train.pt --epochs 100
   ```

   Then bench its V head against `tv0_bellman/epoch_0024.pt` on the standard
   15-pid set. If it does not win, drop it and return the GPU to solving. Do NOT
   wire the policy head in as a child-orderer without a separate A/B — that is
   the configuration that has repeatedly regressed (megaminx qshort, all-neighbor
   Q head).

6. Kaggle's TPU queue is the bottleneck on the width measurement (queued 3h+).
   Two alternatives exist but both spend resources that are not ours to spend
   unilaterally: the user's GCP TPU-Builders v6e credit (earmarked for megaminx)
   and zakhar's shared v4-8 box (access was granted in a megaminx context, and it
   already has a working `~/envs/jax-tpu`). Ask before using either.
