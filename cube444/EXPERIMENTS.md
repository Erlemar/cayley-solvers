# cube444 — experiment log

**For the one-page summary + resume guide, see `HANDOFF.md`.** This file is the
chronological record of what was built, run, and measured; `ANALYSIS_AND_PLAN.md`
is the data analysis + recipe.

Final result: **53,426, #2 public LB** (submitted 2026-07-25) — see `HANDOFF.md`
for the score progression, model provenance, and what transfers to the 5/6/7 cubes.

Hardware: GCP `cayley-gpu` (L4 24 GB, g2-standard-8, us-east1-b) unless noted.
Local box is an RTX 4090 Laptop (17 GB).

---

## 2026-07-23 — Stage 0: puzzle port + infrastructure

Built `cube444/` from scratch. The shared `cayley.*` stack turned out to be fully
shape-parameterized, so the port needed no changes to it beyond one ASCII fix.

**Package** (`cube444/src/cube444/`)
- `puzzle.py` — `Cube444`, 96 slots / 24 generators / 6 colors. Deliberately has
  **no `invert_state`**: the state is a coloring, so it pins the group element
  only up to the `4!^6 = 191,102,976` stabiliser of the solved coloring. NISS and
  inverse sym-frames are unavailable here.
- `symmetry.py` — 24 whole-cube rotations, each with the color relabel `pi_R`
  that restores the solved target: `sym(s, R) = pi_R[s[R]]`. Plus the move
  relabel table and its row-wise inverse for translating solutions back.
- `models.py` — `load_v_model`, dispatching Stage-2/3 `ResMLPDistance` to the
  shared loader and wrapping Stage-4 `ResMLPGFlowNet` in a value-only adapter
  (`cayley.search.load_model_checkpoint` does not know that class).

**Scripts** — `00_verify`, `01_build_bfs`, `02_train`, `03_bellman`, `04_eval_v`,
`05_solve`, `06_merge`, `07_build_az_dataset`, `08_train_az`, `test_symmetry`,
`run_stages.sh`.

**Bugs found and fixed while building**
1. `build_move_relabel` conjugated the wrong way round (`R.g.R^-1` instead of
   `R^-1.g.R`). Caught by `test_symmetry.py`: 89/200 before, 500/500 after. The
   partial pass rate is what makes this dangerous — it "mostly worked".
2. `src/cayley/bellman.py:456` printed a literal `x` multiplication sign in a log
   line — a Rule 24 violation that would crash a long run on a cp932 console.
   Fixed to ASCII.
3. `megaminx/scripts/71_train_az_v3.py:289` hardcodes `num_classes=state_size`.
   Correct for the permutation puzzles, silently wrong here (would build a 96x96
   one-hot instead of 96x6). `08_train_az.py` sets `num_classes=6` explicitly.

**Validation** — `test_symmetry.py` 500/500 on both the commutation identity and
the solve-in-frame-k-then-translate-back round trip. `00_verify.py` replays the
community CSV: 1043/1043, 54,754 moves.

---

## 2026-07-23 — Stage 1: exact BFS anchors

`01_build_bfs.py --max-exact 5 --d6-sample 20000000`, **2 min 10 s** on the L4.

```
d=0            1
d=1           24
d=2          468
d=3        9,000
d=4      172,914
d=5    3,316,744      ball(<=5) = 3,499,151
d=6   21,809,264      sampled from ~63.6M (frac 0.343)
```

Total 25,308,415 states, 2.43 GB. Branching factor **19.18**, dead stable across
d=2..5.

Correctness check that fell out for free: the d=6 pass reported exactly
**3,316,744** fresh states when run with `--max-exact 4`, matching the exact d=5
sphere from the full BFS. That confirms the "reachable in d+1 and not in ball(d)
=> distance exactly d+1" shortcut, which is what lets us skip a 6 GB dedup.

---

## 2026-07-23 — Stage 2: random-walk MSE pretrain

`c_v0s_pretrain.yaml`, 3,291,137 params, 1000 epochs in **8 min** (0.5 s/epoch).

Final loss **80.5**, plateaued from ~epoch 180. **This plateau is the expected
floor, not a bug**: with `k_max=60` and a true diameter of ~37-40, walk depth is
an increasingly loose label past the mixing time (~26-30). Predicted floor from
the label noise alone is ~70-85 — measured 80.5 sits inside it. It also means
1000 epochs is overkill for this stage; ~200 would do.

Gate output (informational — a walk-depth model is expected to fail):

```
V(solved)  = 4.04      <- solved never appears in a random walk; V0 anchors fix this
V(d=1..6)  = 1.04, 2.08, 3.21, 4.37, 5.55, 6.70    (MAE < 1, well calibrated)
V@40=41.10  V@80=42.85  gap +1.75                   <- ALREADY SATURATES
std(V@d=20) = 8.56                                  <- high, Bellman should tighten
```

The saturation at ~43 is a useful calibration: it sits just above the counting
lower bound of 36.8, as an upper-bound-trained model should. It also confirms the
`[36, 44]` absolute band in `04_eval_v.py` was well chosen — though that check
was demoted to a WARNING (see below) since beam only needs correct ordering, and
a monotonically compressed V can still beam fine.

**Gate restructure**: `04_eval_v.py` now separates HARD failures (saturation gap
> 3 per Rule 23; mid-depth discrimination `V@30 - V@12 < 6`, which catches the
dodeca-style uniform collapse) from WARNINGS (absolute scale band, mid-depth
variance) whose thresholds are megaminx-derived and not yet calibrated here. A
guessed threshold should not be able to block a good model.

---

## 2026-07-23 — Stage 3: Bellman refine (small, 3.3M)

`c_bellman_small.yaml`, 13.6 s/epoch. Loss 14.4 @ep9 -> 1.79 @ep22 -> 0.134 @ep140.

### V collapses in scale — and it does not matter

| ep | V@80 | V@30-V@12 | std@20 | ratio |
|---|---|---|---|---|
| pretrain | 42.85 | — | 8.56 | — |
| 24 | 26.32 | 11.74 | 3.69 | 3.18 |
| 49 | 20.86 | 8.56 | 2.57 | 3.33 |
| 74 | 18.01 | 6.74 | 2.00 | 3.37 |
| 99 | 17.26 | 6.20 | 1.81 | 3.43 |
| 124 | 16.76 | 5.83 | 1.67 | 3.49 |

V slides monotonically away from the counting bound (36.8) toward ~16. The
mechanism is the optimistic-min bias in `target = 1 + min_a V_target(child)`:
with 24 noisy children the min is biased low and compounds. megaminx never hit
this because `m_dd_v0` carried `lambda_pdb=5.0`, an admissible PDB **lower
bound**, which we set to 0 (no PDB for this puzzle).

**WRONG TURN #1.** I read the slide as the dodeca-style collapse and stopped the
run at ep140 to avoid burning 4 h reproducing it on the 15M trunk. The
controlled test then falsified that:

| depth | ep24 (V@80=26.3) | ep124 (V@80=16.8) |
|---|---|---|
| 20 | 6/6, len 38.3 | 6/6, len **27.0** |
| 25 | 2/6, len 71.0 | **4/6**, len **54.0** |
| 30 | 6/6, len 63.3 | 5/6, len 62.8 |
| 35 | 3/6, len 74.3 | **4/6**, len **72.5** |

19/24 vs 17/24, mean length 54 vs 61. **The heavily "collapsed" checkpoint is the
better beam model.** Beam needs ordering, not scale, and the
discrimination-to-noise ratio *rises* monotonically (3.18 -> 3.49). Scale
collapse here is a benign monotone compression.

Consequence for the gates: `04_eval_v.py`'s absolute-discrimination hard gate was
simply wrong — it would have rejected the better model (it failed ep124 at
5.83 < 6.0). Replaced with a **ratio** gate (`discrimination / std@20 >= 2.0`).
The absolute-scale band was already demoted to a warning. Do not gate a V on its
absolute scale for this puzzle.

Related: megaminx verdicts on the obvious anti-collapse knobs — `double_bellman`
REJECTED (m28, "doesn't break cluster"), `softmin_temperature=0.5` CATASTROPHIC
(m34, 0/51; T must be << V scale), `lambda_upper` REJECTED (m40, 48/51 / ~99).
None are worth reaching for now that the collapse is known to be benign.

### WRONG TURN #2 — a reporting bug in our own solver

`05_solve.py` reported "model-solved 0" on hard pids at beam 65536, which I read
as the beam failing. It was not. The counter only incremented when the beam beat
the fallback; a verified solve that was *longer* than the community path was
logged as `fallback`. Tracing pid 299 directly showed it solving cleanly in **67
steps** (min-V descending 16.4 -> 0.03, no stagnation break).

Fixed: `n_found` (beam produced a verified solution), `n_used` (kept it),
`n_improved` (strictly beat the fallback), and `n_fb` are now separate counters,
and the beam's own mean solution length is reported. A beam that solves
everything 20 moves worse than the floor is a completely different situation from
a beam that solves nothing; one counter hid that.

### Where we actually stand

Beam width sweep, same checkpoint (c_bells ep124), synthetic scrambles:

| depth | beam 16384 | beam 65536 |
|---|---|---|
| 25 | 4/6, len 54.0 | **6/6**, len **44.0** |
| 30 | 5/6, len 62.8 | **6/6**, len 65.7 |
| 35 | 4/6, len 72.5 | **5/6**, len **66.2** |
| 40 | — | 3/4, len 62.7 |
| 50 | — | 3/4, len 70.7 |
| 70 | — | 2/4, len 70.0 |

**Matched-width benchmark, 12 real hard pids, beam 2^16, sym-ensemble 1:**

| | solve rate | mean length when solved |
|---|---|---|
| **ours (c_bells ep124, 1 frame)** | **75%** (9/12) | **70.44** |
| organiser 2^16 **Repeat3** SmallM | 83.5% | 71.88 |

We are level with the organiser baseline on length while using **one** frame
against their three, after ~2.5 h of total training on a single L4 with Bellman
stopped at ep124 of 300. One pid already beat the community floor. The V is
sound; the gap to 54,754 is beam budget, exactly as the plan predicted.

---

## 2026-07-23 — leg 2

`run_stages2.sh`: continue small Bellman from ep124 (400 ep, lr 3e-4) -> gate ->
mid pretrain (400 ep) -> mid Bellman (400 ep) -> gate. Mid pretrain cut from 1000
to 400 epochs since the loss plateaus at ~180.

### The small model kept improving — SNR did not show it

`c_bells2/epoch_0399` gate: V(solved)=0.033, V@80=16.99 (gap +0.35),
discrimination 5.74, std@20 1.614, **SNR 3.56**. HARD GATES PASSED.

SNR has risen monotonically at every checkpoint (3.18, 3.33, 3.37, 3.43, 3.49,
3.51, 3.56) while V's absolute scale converged to ~17 — so the collapse found a
fixed point rather than running away.

**But SNR badly understates progress.** At ep99 of leg 2 the SNR looked flat
(3.49 -> 3.51) and V@80 flat (16.76 -> 16.70), which read as "converged". A depth
sweep on identical scrambles said otherwise:

| depth | c_bells ep124 | c_bells2 ep99 |
|---|---|---|
| 25 | 4/6 | **6/6** |
| 30 | 4/6 | **6/6** |
| 35 | 4/6 | **6/6** |

12/18 -> 18/18. **SNR is a catastrophe detector, not a progress meter.** Use the
depth sweep (`90_debug_beam.py`) to decide whether to keep training.

### Sym-ensemble scaling (the recipe-setting measurement)

12 real hard pids, `c_bells2/epoch_0399`, beam 2^16, max-steps 150:

| frames | solve rate | mean beam length | wall/pid |
|---|---|---|---|
| 1 | 8/12 (67%) | 66.75 (60..74) | 68 s |
| **4** | **12/12 (100%)** | **62.67 (58..72)** | 296 s |

= **-2.04 moves per doubling of K**; projected K=8 -> 60.6, K=16 -> 58.6,
K=24 -> 57.4. Sym-4 completely closes the coverage gap (67% -> 100%).

Head-to-head at comparable effort, we now beat the organiser baseline on **both**
axes:

| config | solve rate | mean length |
|---|---|---|
| organiser 2^16 **Repeat3** SmallM | 83.5% | 71.88 |
| **ours 2^16 sym-4** | **100%** | **62.67** |

### Single-GPU is confirmed not competitive

A full 1043-pid run at 2^16 sym-4 projects to **~63,300** (1003 hard pids x 62.67
+ ~427 easy) against the 54,754 community floor, and costs **~86 h (3.6 d)** on
one L4. 0 of 12 benchmarked pids beat the floor. So the merge-only path yields
almost nothing at this width.

Extrapolating the organiser's own width curve (2^16 -> 2^24 is worth -17.4 moves
for their model) onto our numbers:

- 2^24 sym-1 -> ~49.4 mean
- 2^24 sym-4 -> ~45.3 mean (floor is 53.88; Rokicki's hard-band mean is ~45.3)

Two independent routes agree on ~45: subtracting 17.4 from our 62.67, and
subtracting our 9.2-move edge over the organiser from their 2^24 result of 54.5.
**Caveat: this is an extrapolation from two measured points and assumes the width
and sym gains are additive — both optimistic.** Even half the projected gain
would clear the community floor.

**Conclusion: the model is good enough; beam budget is the entire remaining gap.
The next lever is the TPU port, not more training.**

### Capacity: 15M is NOT better than 3.3M (matched epoch)

Both Bellman checkpoints at ep99, identical scrambles, same seed, beam 16384:

| depth | 3.3M (c_bells) | 15M (c_bell) |
|---|---|---|
| 25 | 4/6, len 40.5 | 6/6, len 63.3 |
| 30 | 4/6, len 70.0 | 4/6, len 98.5 |
| 35 | 4/6, len 61.5 | 2/6, len 53.0 |
| **total** | **12/18** | **12/18** |

Same solve rate; worse lengths where directly comparable. (The 15M's better d=35
length is a selection artefact — it solved only the 2 easiest of the 6.)
Caveats: n=6 per depth is noisy, and the 15M warm-started from a 400-epoch
pretrain vs the small's 1000, though both were past the ~180-epoch plateau.

This extends **Rule 14** (bigger V trunks regress under the `m_dd_v0` recipe)
from megaminx's 1920-dim embedding input to this puzzle's 576-dim one-hot. It was
worth testing — the input regime is genuinely different — but the answer matches
the prior.

**Deployment consequence:** the 3.3M is also ~3x faster at inference, and since
width is the entire lever, 3x speed is 3x the beam at fixed TPU-seconds.

## 2026-07-23 — TPU beam port (complete, CPU-validated)

Ported the megaminx/IHES JAX SPMD beam kernel to the color cube. Lives in
`cube444/tpu/`. The megaminx kernel was already ~90% generic (reads
`puzzle_info.json`, derives all shapes), so the port was small and surgical:

- **`jax_model.py`** — added a one-hot encoder branch beside megaminx's
  embedding. Auto-detects from the checkpoint (no `embedding.weight` -> onehot,
  `num_classes` inferred from `input_stack.0.weight` in_dim / state_size). Only
  the first-layer input differs; ResBlocks/head identical.
- **`jax_beam_spmd_v_only.py`** — the ONLY engine change was 6 packed state-slice
  sites `[:, 0:120]` -> `[:, 0:state_size]`. Parent-local (bytes 120..123), move
  (124), and bf16 score (125..126) stay put; PACK_SIZE=128 has room for
  state_size<=120. The all-zero padding sentinel stays valid (a real color state
  sums to >=240, never 0).
- **`gcp_beam_cube444.py`** — new driver: loads the one-hot V, NO NISS (undefined
  for a coloring), and the color-cube sym-ensemble (`sym(s,R)=pi_R[s[R]]`,
  translate the word back via `move_relabel_inv`, re-verify on host). Prefix-
  nested frames, identical convention to `05_solve.py`.
- **`json_to_csv.py`** — merges shard result JSONs into a verified submission.

**Validation (all on CPU, no TPU needed):**

- `test_parity.py`: JAX one-hot fp32 vs PyTorch = **max|d| 5.7e-06**; bf16
  top-100-smallest-V set overlap **98%** (bf16 tie-break drift is the accepted
  0-2 moves/pid the megaminx kernel already tolerates).
- 8-device SPMD emulation (`--xla_force_host_platform_device_count=8`): pid 4
  solved len 5, **verify=True** — validates model forward, hashing, the packed
  `all_to_all` routing with 96-byte states, tree reconstruction, host verify.
- sym-ensemble 4 through the driver: pid 19 solved in all 4 frames (non-identity
  19/5/21 included), every translated path verify=True, BEST=14 (matches the
  numpy `05_solve.py` result). The frame-translation plumbing is correct.

The port is functionally done. Remaining work is hardware: provision a v6e
flex-start VM (skill `megaminx-tpu-provision`), stage artifacts, run at width.
See `cube444/tpu/RUNBOOK.md`.

### On-TPU smoke — WORKS on real v6e silicon (2026-07-24)

Provisioned `ct6e-standard-4t` (v6e-4) FLEX_START in europe-west4-a:
PENDING->STAGING->RUNNING in ~4 min (no stockout), setup (jax[tpu]+CPU torch+pull)
~2.5 min, `DEVICES 4 TPU v6 lite`. BPTR 25/3/5 needed no edit (tree is uint64;
RANK_BITS=3 covers 4 and 8 chips).

`tpu_diag.py` on-device: host==device int64 hash EXACT, V(solved)=-0.029,
V monotone on a walk, encoding=onehot. All green.

Smoke: **pid 299 (rw_len=300, fully mixed), b_global=2^20, sym-1, num_steps=120:
solved len 56, verify=True, 164 s wall, ~0.3 s/step** (device-bound; copy/write
negligible). Re-verified locally after `scp` (independent replay: 56 moves OK).
= **~84M child-states/s on v6e-4**. VM torn down after ~12 min RUNNING.

pid 299 = 56 here vs the community ~54 -- 2 over floor at a SINGLE frame and
1/16 of the eventual width. sym-4 + more width should clear it.

### Measured throughput -> production economics (v6e-4, per-step linear in B)

| width | s/step | s/pid sym1 | min/pid sym4 | full 1043 sym4 |
|---|---|---|---|---|
| 2^20 | 0.3 | 42 | 2.8 | **2.0 d** |
| 2^22 | 1.2 | 168 | 11.2 | **8.1 d** |
| 2^24 | 4.8 | 672 | 44.8 | 32 d |

Sobering: a full competitive run (2^22-2^24) is 8-32 days on ONE v6e-4. Levers to
cut that: v6e-8 via Queued Resources (~2x + super-linear on wide beams), several
v6e-4 boxes sharding pid ranges, or accept 2^20-2^22 over a couple of days.
Port + throughput are proven; the remaining decision is width x hardware x wall.

### v6e-8 via Queued Resources — 8-rank validated, full run launched (2026-07-24)

User picked v6e-8, width 2^20 (1M) first. Provisioned `v6e-8` via the Queued
Resources API (`cube-v6e8-qr` in us-east5-a; the single-host 8t `instances create`
path is backend-broken -- QR auto-retries): WAITING->PROVISIONING->ACTIVE in
~16 min, TPU-VM, jax[tpu]+torch installed, `DEVICES 8 TPU v6 lite`.

**8-rank packing validated** (mandatory -- wrong RANK_BITS = silent walkback
corruption): BPTR 25/3/5 needs no edit (uint64 tree, RANK_BITS=3 = exactly 8
ranks). Diagnostics green. Smoke pid 299 @ 2^20 sym-1: **solved len 62,
verify=True**, ~0.1-0.2 s/step device (2x the v6e-4 rate). (62 vs the v6e-4's 56
is expected -- a different chip count partitions the beam differently and bf16
tie-breaks diverge; both verify, not a bug.)

**Full run launched**: 1043 pids, b_global=2^20, sym-4, num_steps=140, seed 0 ->
`~/out/full_2p20_sym4.json`. Driver checkpoints per-pid (preemption-safe). QR
max-run-duration 72h. Projection at 2^20 sym-4 ~= 54 mean (near the 54,754 floor);
this run validates the full pipeline at scale + gives per-pid data to decide
whether to go wider. Min-merge with the community floor (Rule 26) when done.

### CRITICAL: the beam recompiles per call -> persistent compilation cache

First launch exposed it via the per-pid milestone check: pid 0 (found at step 0,
pre-loop) was instant, but pids 1-2 each took ~90-105s despite solving in 1-2
steps. The step compute was 1.4s; the other ~90s was COMPILE. The kernel rebuilds
the step_fn closure on every `run_beam` call, so JAX's in-memory jit cache MISSES
every call and re-compiles. At sym-4 that's ~90s x 4 x 1043 ~= **5 days of almost
pure recompile**. megaminx never noticed because its pids take hours (compile is
noise); cube444's fast solves make compile 98% of the wall.

Fix: enable JAX's **persistent (on-disk) compilation cache** in the driver:
```python
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cube_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
```
It is keyed by the computation's HLO hash (shapes+ops), NOT the Python callable
identity, so a fresh jit that lowers to identical HLO HITS the disk cache.
Validated: pid 1 frame 0 compiled 105s (written to a 71-entry cache), then EVERY
later beam call dropped to **2s** -- a ~50x speedup. Full 2^20 sym-4 run now
~2 days (compute-bound: ~0.7s/step x depth x 4 frames), within the 72h cap.
**This belongs in RUNBOOK.md for every future cube444/megaminx TPU run** on a
fast-solving puzzle. The per-step host-copy of the packed backptr tree
(b_local x 128 bytes/step) is the remaining deep-pid cost.

### Early full-run read (102 pids): min-merge beats the floor

Transition band (rw 45-55) looks bad standalone -- our beam OVERSHOOTS the raw
walk (pid 49 rw50: ours 54 vs community 50=raw-walk), because walks are still
near-geodesic there and a learned V can't beat them. But this is where Rule 26
earns its keep. Deep band (rw>50, 52 pids done):

| | ours | community |
|---|---|---|
| mean | 54.35 | 53.38 |
| ours<comm | **17 (33%)** | 13 ties, 22 worse |

Our mean is ~1 worse, yet we WIN outright on a THIRD of deep pids (several by -6:
pid 66 55v61, 69 46v52, 77 52v58) because the beam explores differently. Min-merge
captures exactly those: **52 moves saved on 52 deep pids = ~1.0 move/pid**.

Extrapolated over ~990 deep pids: **~990 moves off 54,754 -> ~53,760**, a real
~1000-move improvement and our first own submission below the community floor.
Wall is ~44 s/pid observed (solves ~30-40 steps, faster than the 168s estimate)
-> full run ~13-24 h, not 2 days. Decision: LET IT FINISH; min-merge + verify at
the end. (Lesson: judge a beam run by the per-pid MIN vs the floor, never by its
standalone mean -- the mean hid a 1000-move win.)

### RESULT: full run complete, 53,756 (our first sub below the floor)

1043/1043 verified, ~16 h wall (survived a local-session interruption -- the
detached `setsid nohup` remote process is independent of the local monitors).

- **TPU standalone**: 55,846 (mean 53.54) -- WORSE than community 54,754 alone.
- **min-merged with community (Rule 26)**: **53,756 (mean 51.54)**, 1043/1043
  VERIFY OK. Our TPU run is strictly best on **314 pids** (30%, 16,365 moves);
  community keeps 729. **-998 vs the 54,754 floor.**

Matches the pid-405 projection (~53,794) almost exactly. Public-LB placement:
Rokicki 46,298 | webmaking 53,580 | **OURS 53,756** | DrozdovDan 54,416 |
floor 54,754 -> **~3rd**, first result under the community floor.

Saved `cube444/submissions/cube444_53756_tpu2p20sym4.csv`. v6e-8 QR torn down.
NOT yet submitted to Kaggle (awaiting go-ahead). Next lever for a deeper cut: the
same run at 2^22 projects to ~50 mean (~8 days on one v6e-8, or shard/wider box).

### Targeted 2^22 rescue on top-100 longest -> SUBMITTED 53,426 (#2 public LB)

Instead of a full 2^22 run (8 days), ran 2^22 sym-4 on just the **top-100 longest
paths** in the 53,756 merged solution (lengths 56-60, mean 56.8) -- the pids with
the most overshoot room. New v6e-8 QR, `--pids` explicit list added to the driver,
`--parent-chunk 131072` to stream neighbor-gen at 2^22 (no OOM), ~1.3s/step,
~5.5h for 100 pids.

Result: **81/100 pids strictly improved** vs the merged floor (0 worse -- a wider
beam with the same V rarely regresses), -330 moves. Several -6 (pid 88/116/210
57->51, pid 642 57->51). 3-way min-merge (community + 2^20-full + 2^22-rescue):
**53,426 (mean 51.22), 1043/1043 VERIFY OK**. Source attribution: community 661
pids, our 2^20 301, our 2^22-rescue 81.

**Submitted to Kaggle -> public score 53,426 (COMPLETE), matching local exactly.**
Public LB: Rokicki 46,298 | **OURS 53,426** | webmaking 53,580 -> **#2**.
Saved `cube444/submissions/cube444_53426_v6e8_2p20full_2p22rescue.csv`. All TPUs
torn down. (Lesson: concentrate expensive width on the longest paths -- 100 pids
at 2^22 bought -330 for ~5.5h vs an ~8-day full 2^22 run.)

### CONFIRMED at ep399 — capacity hypothesis falsified

Final gates: 3.3M `c_bells2/epoch_0399` SNR **3.56** (V(solved) 0.033, V@80 16.99);
15M `c_bell/epoch_0399` SNR 3.49 (V(solved) 0.0095, V@80 16.57). Both pass.

Definitive paired sweep, 32 scrambles per model, depths 25/30/35/40, seed 21:

| depth | 3.3M | 15M |
|---|---|---|
| 25 | 5/8 (len 53.4) | 6/8 (71.3) |
| 30 | 4/8 (63.5) | 2/8 (59.0) |
| 35 | 2/8 (74.0) | 2/8 (69.0) |
| 40 | 5/8 (72.0) | 5/8 (82.0) |
| **total** | **16/32** | **15/32** |

Statistically identical solve rate, generally longer paths for the 15M, at 3x the
inference cost. **Deploy the 3.3M.** Rule 14 now holds in the color-cube regime
too.

### METHODOLOGY WARNING for `90_debug_beam.py`

Absolute solve rates from the synthetic sweep swing wildly with the seed and with
the `--depths` list: `c_bells2` scored 18/18 at depths 25/30/35 (seed 7) but
11/24 on the same depths at seed 21 — because scrambles are drawn sequentially
from one RNG, so adding depth 40 changes every depth's scrambles. n=8 per depth
is far too few for an absolute claim.

**Only PAIRED within-run comparisons (same seed, same depths list, two
checkpoints) are trustworthy.** For absolute numbers use the real hard-pid bench
via `05_solve.py`; the trustworthy figure is c_bells2 ep399 at 2^16 sym-4:
**100% solve, mean 62.67**.

---

## 2026-07-31 — STRUCTURAL: two-phase (reduction) solver, sparse-Q margin, lower bounds

Three linked bets, all built and measured. Headline: **the two-phase decomposition
is sound and fully implemented, but a classical HTM-optimising phase-2 solver makes
it provably unable to beat our current floor.** The blocker is identified precisely
and it is not the decomposition itself.

### Orbit structure (verified, `src/cube444/orbits.py`, `scripts/10_verify_orbits.py`)

Under the FULL 24-generator group the 96 slots split into **4 closed orbits of 24**:
corners, centres, wing-A, wing-B. Under the **12 outer-layer generators** the centres
break into 6 orbits of 4 (one per face) while corners/wing-A/wing-B each stay a
single 24-orbit. Verified, not assumed:

- corners == exactly the 24 slots fixed by every inner-layer move
- no generator mixes corner and non-corner slots (orbit closure)
- `R` is invariant along a 5,000-step outer-only walk

**Phase-1 goal** `R(s)` := every face's 4 centre slots carry **that face's own
colour** AND all 24 (face, edge) wing pairs are monochrome.

The centre condition must be ABSOLUTE, not merely monochrome. Outer moves permute a
face's centres among themselves and can never move one off its face, so a state with
monochrome-but-wrong centres is solved only up to a whole-cube rotation -- and whole-
cube rotations are not generators. Requiring `colour == face` costs nothing and
removes a dead end that would otherwise silently produce unsolvable phase-2 handoffs.

**Corners are free in phase 1**, and corners are a closed orbit, so distance-to-R is
*exactly* a function of the 72 non-corner slots. The phase-1 V is defined on that
72-slot quotient — an exact reduction, not an approximation, and 25% cheaper per
forward. Verified: a corner-scrambled twin has an identical 24-child defect profile.

### Phase 2 = a 3x3x3 (`src/cube444/phase2.py`, `scripts/11_verify_phase2.py`)

Cubie membership is derived from the generators alone — two slots are on the same
cubie iff the same set of generators moves them — giving 8 corners x3, 24 wings x2,
24 centres x1, matching `orbits` exactly. The frame (which face is U/R/F/D/L/B) is
found by searching the 48 candidates allowed by the opposite-pair structure.

**Gotcha**: hkociemba's `verify()` returns the sentinel `CUBE_OK`, which *is* `True`,
not `0` — comparing against 0 rejects every valid cube. And once fixed, all 48 frames
pass, because the facelet map is built FROM the frame so relabelling cancels out.
Frame uniqueness is the wrong property; the right one is that translating a WORD is a
homomorphism, checked over 40 random 12-move words. Any self-consistent frame works.

Verified end-to-end: **12/12 random reduced states solved and replayed to solved.**

### The number that decides it: phase-2 QTM cost

The competition is quarter-turn metric; hkociemba minimises HALF-turn metric, where a
180-degree turn is free. Sweeping length caps and 8x the search time barely moves it:

| config | QTM mean | HTM mean | wall |
|---|---|---|---|
| caps 18-20, t=1.0 | 27.67 | 19.17 | 18s/12 |
| caps 16-20, t=1.0 | 27.33 | 19.17 | 42s/12 |
| caps 15-21, t=3.0 | **27.17** | 19.00 | 157s/12 |
| caps 20-24, t=1.0 | 28.67 | 19.75 | 2s/12 |

Whole-cube symmetry frames do not rescue it either — hkociemba already uses the
48-element symmetry group internally to canonicalise, so rotated frames are not
independent searches:

| frames | QTM mean | wall |
|---|---|---|
| K=1 | 27.25 | 17s/12 |
| K=4 | 27.08 | 75s/12 |
| K=8 | 26.75 | 151s/12 |
| K=24 (all) | **26.42** | 633s/12 |

**QTM is pinned at ~26.4-27.2.** 3x3x3 QTM God's number is 26 and random states
average ~21 QTM-optimal, so the HTM objective costs us ~6 moves per solve. Break-even
would need 25.7 *at the phase-1 information-theoretic optimum* — 38x the phase-2
compute still does not reach it. (All 288 symmetry-translated words were replayed on
the original state and verified, so the sym round-trip itself is sound.)

### Budget check (`scripts/13_lower_bounds.py`)

A reduced colouring *is* a 3x3x3 state, so `|R| = 4.325e19` (x4 for parity classes)
against `N = 1.776e47`. The same counting argument that gives the full-cube bound
gives a phase-1 bound:

- full cube: **d >= 35** (rigorous, b<=24) / **37** (measured, b=19.18)
- distance-to-R: **d >= 20** (rigorous) / **21** (measured)

So `two-phase total >= 21 + phase2`:

| phase-2 engine | two-phase floor | vs our floor 46.73 |
|---|---|---|
| ours (hkociemba HTM), 27.2 | 48.2 | **loses even at phase-1 optimum** |
| a QTM-native solver, 24.0 | 45.0 | wins by 1.7 |
| QTM-optimal, 21.0 | 42.0 | wins by 4.7 |

**Verdict: two-phase cannot beat the single-phase floor with an HTM solver, and the
gap is not close enough for beam tuning to rescue.** The decomposition itself is
fine — with a QTM-native phase 2 the floor drops to ~42, comfortably under Rokicki's
44.39. That is the actionable next step, not more phase-1 training.

### Lower bounds: what is actually certifiable

Per-state certified bounds via orbit-projection BFS (exact uint64 base-6 encoding, so
dedup cannot silently unsound the bound by dropping a state):

| orbit | BFS depth reached | LB mean | exact hits |
|---|---|---|---|
| corners | 9 | 9.67 | 166/1043 |
| centres | 6 | 6.96 | 10/1043 |
| wing-A | 6 | 6.98 | 8/1043 |
| wing-B | 6 | 6.98 | 8/1043 |

**max-over-orbits LB = 9.72 mean (max 10)** against actual path lengths of ~46.7.
Orbit projections are far too small to certify anything useful at this scale — the
binding bound is the aggregate counting bound of 37. Honest conclusion: **a tight
per-pid optimality certificate is not attainable for the 4x4x4 by this route**, and
building the full 88.2M-entry corner PDB would only lift 10 -> 14, changing nothing.

### BASELINE CORRECTION — and the transformer result behind it

`submissions/cube444_48738_merge_tg_hybrid.csv` (dated 2026-07-30) verifies
**1043/1043 at 48,738 (mean 46.73)**. It postdates the 53,426 recorded in HANDOFF.
**Provenance (per the user): a min-merge of our work with a TRANSFORMER solution from
Vlad Kuznetsov.** The real slack to Rokicki is **2.34 moves/pid, not 6.83**.

Differencing it against every CSV we produced gives the attribution:

| | |
|---|---|
| pids where the merge beats everything we had | **916 / 1043 (87.8%)** |
| ties | 127 |
| losses | 0 (by construction — it is a min-merge) |
| moves saved | **4,688** (mean 5.12 per winning pid) |
| win margin | min 2, median 6, max 12 |
| on winning pids | transformer 47.46 vs ours 52.57 |

Win rate by difficulty band: 63% on pids 0-99 (where both find optimal, hence ties)
then **84-94% uniformly across bands 100-999**. This is NOT a tail effect and NOT
seed/ensemble diversity — it beats our pipeline essentially everywhere, and never
loses. Implied standalone is roughly **49,000-49,500 vs our 53,426**, i.e. ~7.5%
better than the entire TPU V-beam pipeline.

**This overturns, FOR THIS PUZZLE, the "architecture is closed / path is
pure-inference" conclusion imported from megaminx and the IHES picture cube.** Those
verdicts (the §3 encoder survey, the width-saturation campaign) were established on
those puzzles; the 4x4x4 was never covered by them. Here the scorer is demonstrably
the bottleneck: ~5 moves/pid, uniformly, against a strong 2^20 sym-4 beam.

**Open and decision-critical: WHICH part of Vlad's approach is doing the work?** His
`cayleypy-training-core` pairs a PieceTransformer with a **sparse-Q rw-middle
objective**, and our own prior analysis concluded the objective was the transferable
win while the transformer was 0.24x the ResMLP's throughput
([[sparse-q-piece-transformer]]). If the cube444 result is objective-driven it is
cheap for us to reproduce — the phase-1 trainer built here already implements the
zero-conditional-variance margin term and has a clean control arm. If it is
architecture-driven, the §3 survey needs reopening for colour cubes. **Do not guess:
ask Vlad, or A/B the objective on cube444 directly.** Both are cheap now.

### Phase-1 masked V + sparse-Q margin

`src/cube444/phase1.py`, `scripts/12_train_phase1.py`, gate `scripts/15_eval_phase1_v.py`.
Two arms identical except `lambda_margin` (0.3 vs 0.0).

**Bug found and fixed — the training distribution did not reach the test
distribution.** Reduction-defect counts: `k_max=30` walks average 33.7 (p10 16); real
test states average 39.7 (p10 36). The phase-1 beam accordingly drove defects 43 -> 8
and then stalled with V collapsed to ~0 — it had walked off the training
distribution. Fix: `k_max=120` (mean 38.4, p10 34), label clipped at 30, and the
margin term restricted to pivots at depth <= 20 where "the gap is 2" is still
defensible. To keep that affordable the sampler now harvests every step of one walk
instead of discarding 119 of 120 (`walk_harvest`), so k_max=120 costs 1.5 s/epoch vs
1.4 s at k_max=30.

**Margin-term A/B.** `spread` = V(deepest) - V(5); `gap` = V(next) - V(undo) at pivot
depth 20 — reported there, not at the deepest depth, because a fully mixed walk has a
TRUE gap near 0, so the ideal of 2.0 only holds while the walk is roughly geodesic.

| arm | v1 spread | v1 gap | v2 spread | v2 gap |
|---|---|---|---|---|
| control pretrain | 12.31 | 1.51 | 19.18 | 2.54 |
| margin pretrain | 12.27 | 1.50 | 18.48 | 2.57 |
| control bellman | 5.98 | 0.73 | 6.36 | 0.94 |
| margin bellman | **6.37** | **0.78** | **8.03** | **1.18** |

The margin term is a **no-op in pretrain** (walk-depth labels already carry the gap)
and **resists Bellman's compression**: +6.5% spread at v1, and **+26% spread / +26%
child gap at v2** once the training distribution is correct. That is exactly the
claimed mechanism — MSE on a zero-conditional-variance difference cannot buy loss by
flattening the child gap — and the effect grows once the distribution is right.

Note Bellman *shrinks* the child gap here (2.54 -> 0.94), the opposite of the
single-phase V where more Bellman kept helping. The phase-1 target is a SET with no
BFS-shell anchors to pin the scale, so the optimistic-min bias compounds unopposed
and the margin term is the only thing pushing back. v2 also fixed saturation:
V@120 = 26.2 against a true distance-to-R bound of ~21 (v1 topped out at 15.5), and
the fraction of deep states the model wrongly calls reduced fell 3.1% -> 0.8%.

**KEEP the margin term for any future set-goal (masked) V.** It is nearly free, it
has a clean control arm, and it is the only mechanism found that opposes Bellman gap
collapse when there are no exact anchors to hold the scale.

**Phase-1 beam reach (local, beam 16384).** Reaches R for scramble depth <= 18 —
and finds reductions SHORTER than the scramble (L=14 -> 10 moves) — but fails past
22, while real test states sit at depth >= 37. That is width-limited (production runs
at 2^20-2^22, 64-256x wider), so it does not by itself condemn phase 1; the budget
argument above is the width-independent one. Full pipeline verified end-to-end:
shallow scrambles solve OPTIMALLY (L=2->2, 4->4, 6->6, 8->8, 10->10).

## 2026-08-05 — POST-PROCESSING the best public file (46,718): the tetraminx ladder, ported

Target: `submissions/leader_46718_d7_rot_targeted_3x3.csv`, the best public solution
(1043/1043 replay-verified, **46,718**, mean 44.792, lengths 42-50). Goal: beat Rokicki's
46,298, i.e. find **418 moves = 0.40/pid**.

**Result: `submissions/leader_46710_postprocessed.csv` — 46,710 (-8), 1043/1043
replay-verified through an independent code path (`73_verify_submission.py`, which goes
via `src/cube444/puzzle.py` rather than the GPU pipeline's own tables). Four pids
improved (290, 540, 761, 1030), each by exactly 2; none got longer.**

### The one structural difference from tetraminx, and it is where the only win came from

Tetraminx replaces a window only by a word with the SAME PERMUTATION. Here the state is a
COLOURING, so a window `w[i:j]` may be replaced by any `u` with `s_i[u] == s_j` -- states
equal, not permutations -- because the suffix from `j` is applied to the same state either
way. That is strictly weaker, so it finds every permutation rewrite PLUS those differing
by an invisible piece permutation (4!^6 = 191M interchangeable centres, and the two wings
of an edge). `scripts/70_window_reduce.py` implements it directly in colour space.

The single hit at reach 8 is exactly such a case:

```
pid 761  window [37,46)   r2.f2.r2.-d2.-f2.-f2.d2.f2.r2   (9)
                     ->   -d1.-d1.-f1.-d1.f1.-d1.-r2      (7)
permutations equal? False -- they differ at 6 facelets, ALL of them centres
states equal?       True
```

A permutation-space MITM rejects that replacement. The `d7` in the file's own name
suggests its author already ran a depth-7 permutation rewriter; the colour frame is what
was left. **For any colour cube (this one, and 5/6/7), match states, not permutations.**

### The ladder, all exhaustive, all replay-verified before emitting

| method | script | scope actually run | result |
|---|---|---|---|
| trivial (cancel / repeat / revisit) | — | whole file | 0 adjacent inverses, 0 repeated states |
| colour MITM reach 6 | `70_window_reduce.py --radius 3` | 264,671 windows, 12 s | **0** |
| colour MITM reach 8 | `--radius 4` | 344,581 windows, 3 min | **-2** (pid 761) |
| colour MITM reach 10 | `--radius 5` | 420,353 windows, 100 min | **-6** (pids 290, 540, 1030) |
| endgame table, reach 5+6=11 | `74_endgame_table.py` | tails 12-20, all 1043 pids | **0** (0 phantom rejects) |
| inner-subgroup reach 14 | `72_subgroup_rewrite.py --subset inner` | 18 runs of length 11-13 | **0** |
| pooled bridge, reach 6 | `71_bridge_merge.py --radius 3` | 6.4 paths/pid, 304k waypoints | **0** |
| n-way per-pid min-merge | content scan, 125 verified CSVs | every CSV in the repo | **0 — see below** |

**The merge axis is closed, and that is a measured fact, not an assumption.** A
content-scan of the whole repo (header + move-alphabet + FULL replay) found 125 valid
cube444 CSVs; the n-way per-pid min over all of them is **exactly 46,718**, with **zero
pids** where any other source is shorter. The IHES picture-cube files pass a naive
alphabet test and were correctly rejected by replay — keep the replay gate (rule 26b(c)).
Fresh Kaggle checks: the organiser's community-merge kernel has not run since 2025-11-29
and every `cayleypy-submits*` dataset is stale (best cube444 content 55,854).

### Structure of the file (why the remaining slack is not local)

Measured branching is **19.21** (level sizes 1, 24, 468, 9000, 172914, 3.32M, 63.5M), so
the counting bound is d = 36.7 and this file sits 8.1 moves above it — but only **0.40
above Rokicki**. The corpus is 42.5% inner-slice moves, and the terminal outer-only run
(the 3x3x3 finish of a reduction solve) is strongly bimodal: 386 paths end on an inner
slice (beam/transformer solutions, no reduction structure) while **374 end with an outer
run of 15-22, mean 19.91** — already at the QTM-optimal average for a random 3x3x3, so
that phase is not a target. It is also beyond any exact reach we can afford: certifying a
20-move 3x3x3 needs an optimal QTM solver, not a ball.

**The file is a merge of TWO solver families, and they are not equally good.** Splitting
on whether a path ends in a >=15-move outer-only run:

| family | pids | mean | structure |
|---|---|---|---|
| reduction + 3x3x3 | 374 | **45.76** | reduction 25.86 (sd 1.43) + 3x3x3 19.91 (sd 1.10) |
| neural beam / transformer | 669 | **44.25** | no phase structure, ends on an inner slice |

`corr(reduction, 3x3x3) = -0.458` — a longer reduction leaves an easier 3x3x3, which
damps the total's spread (sd 1.35 vs 1.80 if the two were independent) and therefore also
damps what best-of-N over reductions can win.

Read it carefully: a per-pid min-merge only keeps a reduction path where it BEAT every
neural attempt, so the 374 are the pids where neural beams degrade — the hard tail. That
is simultaneously where this file is weakest (45.76 vs 44.25) and where the most room
sits: pulling those 374 to ~44 would be ~650 moves, more than the 418 the goal needs.

**This CORRECTS the 2026-07-31 two-phase verdict.** That analysis rejected two-phase
because "phase-2 QTM is pinned at 27.2", giving a floor of 48.2 > 46.73. The 27.2 was a
property of the phase-2 solver used, not of the problem: the 374 reduction-style paths in
this file finish their 3x3x3 in **19.91 QTM on average**, so a QTM-native phase 2 reaches
~20. With reduction measured at ~24.9 here, a realistic two-phase total is ~45 — and
best-of-N over diverse reductions is what would push it under 44.39. Two-phase is NOT
dead; it was priced against a weak phase-2 solver.

Two axes closed by argument rather than experiment:
* **FMC-style "alternative optimal 3x3x3 + boundary cancellation" is structurally
  blocked.** The terminal outer run is maximal by construction, so the move preceding it
  is always an inner slice — which commutes with an outer turn but never cancels against
  one.
* **Splicing/bridging cannot help from our own paths.** Not merely empty: our best merged
  file (48,738) is beaten by the public file on **all 1043 pids**, so it contributes no
  waypoint with slack.

### Heuristic post-processing: the beam matches this file but never beats it

Exactness has a hard reach limit, so the band k = 13-24 was probed with beams instead.
Three independent probes, all **0 wins**:

| probe | script | scope | result |
|---|---|---|---|
| plain beam suffix re-solve | `76_suffix_resolve.py` | 6 pids x k=12,16,20 | 0 wins; **11/18 matched k exactly** |
| beam + exact d6 finish | `77_hybrid_suffix.py` | 17 pids x k=13,16,20,24 | 0 wins / 68 solves |
| 3x3x3 phase, outer beam + exact d8 finish | `77 --subset outer --auto-outer-tail` | 47 terminal 3x3x3 phases | 0 wins; **4 reached, all 4 matched** |

The k=12 column is the load-bearing control: our beam returns **exactly 12** on 6/6 pids,
so it is strong enough to reproduce this file's suffixes — and still never beats one. The
k>=16 misses are our beam's width limit, not the file's slack, which is why `--slack` was
added: a null result at slack=0 cannot distinguish "the file is optimal" from "we cannot
search that deep". Same for the 3x3x3 phase: 43 of 47 could not be reached at all, and
every one that could was matched, never beaten.

Validation worth noting: the outer-subgroup ball reproduces the published 3x3x3 QTM level
counts EXACTLY (1, 12, 114, 1068, 10011, 93840, 878880, 8221632, **76843595**), which
checks the move tables, hashing and dedup against external ground truth.

### Honest verdict on post-processing for this goal

Exact local rewriting yields single-digit moves and is bounded: every window of length
**<= 11** in this file is now proven geodesic (in colour space, which implies permutation
space). Note the bound is reach+1, not reach: a window of length L is non-geodesic iff
d <= L-1, and the join finds it iff d <= reach, so reach 10 covers L = 11. (Tetraminx's
HANDOFF states the conservative reach form; its own data confirms the tighter one -- at
reach 12 the L=13 queries returned nothing, and the first radius-13 win was an L=14
window, exactly the first length reach 12 could not certify.) **418 moves cannot come from here** — each rung costs 19.2x more and has returned
~2 moves. The remaining slack is global, so it needs a better SEARCH, not a rewriter.
The one heuristic post-processing direction with unlimited reach is a **suffix re-solve**
(`76_suffix_resolve.py`): our V scores distance-to-solved, so `s_{L-k} -> solved` is the
only sub-problem it can score (prefixes and interior windows have an arbitrary target and
no heuristic).

### Two-phase attack: Stage A done (`scripts/78_piece_model.py`)

The 3x3x3 piece model is derived from the 24 move permutations ALONE -- no cube geometry
is hardcoded. A layer turn moves a piece entirely or not at all, so partitioning the 96
facelets by which generators move them yields exactly 8 triples (corners), 24 pairs
(wings) and 24 singletons (centres); grouping wings by their OUTER-only signature yields
the 12 edges. Centres never leave their face under outer moves and are colour-identical,
so they drop out entirely.

Validated against EXTERNAL ground truth, not self-consistency -- BFS over the derived
representation reproduces the published 3x3x3 QTM counts exactly (12, 114, 1068, 10011,
93840), and **all 374 phase-2 boundary states in the file are confirmed genuinely
reduced** (wings paired, centres uniform), so the two-phase structure is real and
readable.

Remaining stages: **B** a QTM-optimal 3x3x3 solver (corner PDB 8!*3^7 = 88.2M + two edge
PDBs 12!/6!*2^6 = 42.6M, max-combined, IDA*; or emit a `.tws` for the already-built
`third_party/twips/.../twsearch.exe`), then **C** a diverse reduction generator with a
best-of-N gate on 10 of the 374 hard-tail pids. Only scale to GCP if the gate beats the
file's per-pid value. Note phase-1's masked-V beam is width-limited (reaches R only to
scramble depth ~22 at beam 16k, against test states at depth ~37), so Stage C likely
needs 2^20+ width or a classical centres+edge-pairing search.

## Baselines on disk (all replayed and verified)

| CSV | total | hard-band solve rate | mean when solved |
|---|---|---|---|
| community automerge (33 subs) | **54,754** | 98.8% | 53.88 |
| organiser beam 2^16 Repeat3 SmallM | 136,360 | **83.5%** | **71.88** |
| organiser beam 2^16 SmallM (LB only) | 307,192 | — | — |
| sample_submission (inverse walks) | 525,714 | 0% | — |

"hard band" = the 1003 puzzles with `rw_len > 40` plus the 43 santa ones; "fell
back" = solution length exactly equals the generating walk length.

An n-way merge over **every** public CSV (6 of them, Rule 26) returns exactly
54,754 with 1043/1043 pids attributed to the automerge — it already dominates
every other public submission on every single pid. So the community floor is
54,754 and there is nothing extra to harvest from the other kernels.

**The matched-width benchmark**: we can afford roughly beam 2^16 on one L4.
The organiser ran exactly that config with their own model, so the first
meaningful question is not "did we beat 54,754" (we cannot at this width) but
**"at beam 2^16 with ~3 frames, do we beat 83.5% solve rate / 71.88 mean?"**
That isolates V quality from beam budget.
