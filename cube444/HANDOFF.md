# cube444 — HANDOFF / results summary

Single-page state of the CayleyPy 4x4x4 cube. Detail lives in `EXPERIMENTS.md`
(chronological log), `ANALYSIS_AND_PLAN.md` (data analysis + recipe), and
`tpu/RUNBOOK.md` (how to run the TPU beam). This file is the executive summary +
resume guide. Competition: https://www.kaggle.com/competitions/cayley-py-444-cube
(deadline 2026-09-22).

> **BASELINE CORRECTION (2026-07-31).** `submissions/cube444_48738_merge_tg_hybrid.csv`
> verifies **1043/1043 at 48,738 (mean 46.73)** — a min-merge of our work with a
> **transformer solution from Vlad Kuznetsov**. The rest of this file describes the
> older 53,426 state. **Real slack to Rokicki is 2.34 moves/pid, not 6.83.**
>
> The transformer wins **916/1043 pids (87.8%)**, saving 4,688 moves at a median
> margin of 6, **uniformly across every difficulty band** (84-94% on bands 100-999)
> — not a tail or diversity effect. Implied standalone ~49,000-49,500 vs our 53,426.
> **On this puzzle the scorer is the bottleneck by ~5 moves/pid**, which overturns
> the "architecture is closed / path is pure-inference" conclusion carried over from
> megaminx and the IHES cube (neither survey covered the 4x4x4). Highest-value open
> question: is the win the transformer ARCHITECTURE or the sparse-Q OBJECTIVE it is
> trained with? See EXPERIMENTS.md 2026-07-31.

> **POST-PROCESSING PASS ON THE BEST PUBLIC FILE (2026-08-05).**
> `submissions/leader_46718_d7_rot_targeted_3x3.csv` (46,718, mean 44.792) is the best
> public solution and it **dominates every one of the 125 replay-verified cube444 CSVs in
> this repo on all 1043 pids** — the n-way per-pid min is exactly 46,718, so the merge
> axis is empirically closed. The full tetraminx rewriting ladder was ported to colour
> space (`scripts/70`-`77`) and returns **46,710 (-8)** —
> `submissions/leader_46710_postprocessed.csv`, 1043/1043 verified through an independent
> code path, 4 pids improved by 2 each. See EXPERIMENTS.md 2026-08-05.
>
> **SUBMITTED 2026-08-06, Kaggle public score 46,710** (submission 55289633), matching the
> local total exactly. Improved pids: 290 (49->47), 540 (45->43), 761 (48->46),
> 1030 (48->46).
>
> **The goal of beating Rokicki's 46,298 needs 418 moves = 0.40/pid, and it cannot come
> from post-processing.** Every window of length <= 10 is now proven geodesic; each
> further exact rung costs 19.2x for ~2 moves. Our beam **matches but never beats** the
> file's suffixes at k=12 (6/6), so it contributes nothing even as a merge source. The
> slack is global. The live lever is next-lever #4 below.

## Result: 53,426 — #2 on the public leaderboard (submitted 2026-07-25)

```
1. Rokicki      46,298
2. OURS         53,426   <- Kaggle-scored, verified 1043/1043
3. webmaking    53,580
4. DrozdovDan   54,416
5. community    54,754   (public-kernel floor; our starting point)
```

### Score progression (each step's exact total)

| step | source | total | mean | note |
|---|---|---|---|---|
| baseline | community floor (public kernels) | 54,754 | 52.50 | not ours; the min-merge base |
| our beam @ 1M | 2^20 sym-4, standalone | **55,846** | 53.54 | WORSE than floor alone |
| + min-merge | community + 2^20 | **53,756** | 51.54 | -998; won 314 pids outright |
| + 2^22 rescue | community + 2^20 + 2^22-on-top-100 | **53,426** | 51.22 | -330 more; SUBMITTED |

Final submission file: `submissions/cube444_53426_v6e8_2p20full_2p22rescue.csv`.
Per-source attribution in the final merge: community 661 pids, our 2^20 301, our
2^22 rescue 81.

**The load-bearing lesson**: our model ALONE never beat the community (55,846 >
54,754). It explores differently and wins on ~30% of deep pids; the per-pid
min-merge (Rule 26) is what turned that into the score. **Judge a beam run by the
per-pid MIN vs the floor, never by its standalone mean.**

## The model (what was actually deployed + trained on)

Deployed: **`models/c_bells2/epoch_0399.pt`** — a **pure value network (V)**, not
an AlphaZero dual-head.

- ResMLPDistance, 3.29M params, `encoding=onehot`, `num_classes=6`,
  hidden (2048,512), 2 res-blocks, single scalar head.
- Trained on **self-generated + exact data only — zero community/human solutions**:
  random walks from solved (k_max=60, Bellman-bootstrapped labels) + exact BFS
  anchors (d<=6, 25.3M states) + V0/d=1 exact anchors (the V(solved)~2 fix).
- **Stage 4 (AZ dual-head / policy head) was BUILT but NOT run** (`scripts/
  08_train_az.py`, `07_build_az_dataset.py`). Its policy data would have come from
  the community 54,754 CSV, but the capacity test + "V-only wins on TPU" findings
  said the policy head wouldn't help inference, so we shipped the pure V. The
  community CSV only ever entered at the final min-merge, never in training.
- **15M trunk is not better than 3.3M** (matched-epoch, identical solve rate,
  3x slower) — deploy the 3.3M; width is the whole lever. Extends Rule 14 to the
  one-hot regime.

## Why this puzzle is different (the finding that shaped everything)

It is a **COLOR cube, not a permutation puzzle**: 96 stickers, 6 colors x 16.
The state is a *coloring*, so it pins the group element only up to the
`4!^6 = 191,102,976` stabiliser of the solved coloring — a Schreier coset graph,
not a Cayley graph. Consequences baked into the code:
- **No `invert_state`** -> no NISS, no inverse sym-frames.
- **`num_classes` is 6, not state_size** — every model ctor defaults it wrong;
  `megaminx/scripts/71_train_az_v3.py:289` hardcodes it wrong. Pass explicitly.
- **Symmetry needs a color relabel**: `sym(s,R) = color_map_R[s[rotation_R]]`.
  All 24 whole-cube rotations admit one (`src/cube444/symmetry.py`).

Measured graph facts: branching **19.18** (dead stable), state count 1.78e47 ->
**counting lower bound 36.8 moves** for a random state. Random walks mix by
length ~26-30, so ~1000 of the 1043 test pids are statistically uniform-random
deep states; the "difficulty ladder" is only real for the first ~40.

## Key findings index (where the detail lives)

- **V scale-collapse is BENIGN** — Bellman drives V@80 from 43 -> 17, looks like
  the dodeca failure mode but the MORE-collapsed checkpoint beams BETTER. Gate on
  discrimination/noise (SNR), never absolute scale. -> EXPERIMENTS "V collapses in
  scale", `scripts/04_eval_v.py` thresholds.
- **More Bellman keeps helping** long after loss AND SNR plateau (12/18 -> 18/18
  across 100 epochs with flat SNR). SNR is a catastrophe detector, not a progress
  meter — use the depth sweep. -> EXPERIMENTS "leg 2".
- **Sym-ensemble scaling: ~-2 moves per doubling of K**, and it fully closes the
  coverage gap (1 frame 67% -> 4 frames 100% solve). -> EXPERIMENTS "Sym-ensemble
  scaling".
- **TPU port was surgical**: onehot branch in `jax_model.py` + SIX state-slice
  edits `[:,0:120]->[:,0:state_size]` in the beam kernel + a color-cube sym driver.
  JAX==PyTorch parity 5.7e-6. -> `tpu/RUNBOOK.md`, `tpu/test_parity.py`.
- **JAX persistent compilation cache is MANDATORY** for a fast-solving puzzle —
  the beam recompiles per call (~90s), which is ~5 days of pure recompile without
  the disk cache; with it, first call 105s then 2s/call. -> `tpu/RUNBOOK.md`,
  EXPERIMENTS "CRITICAL".
- **Concentrate expensive width on the longest paths**: 2^22 on the top-100
  longest = -330 moves in 5.5h vs an ~8-day full 2^22 run. 81/100 improved,
  0 worse (wider beam + same V rarely regresses). -> EXPERIMENTS "Targeted 2^22
  rescue".
- Operational (cross-project, now in CLAUDE.md 7g + `memory/reference_gcp_cayley_vm`):
  `pgrep -f`/`pkill -f` self-match; detached-remote-launch holds the ssh during
  interpreter startup (verify separately); log on the remote, never pipe ssh
  stdout; Monitors are local-only (remote job survives a session interruption).

## Production economics (measured)

Per-step is linear in B. Full 1043-pid sym-4 run:

| width | v6e-4 | v6e-8 |
|---|---|---|
| 2^20 | 2.0 d | ~16 h (measured) |
| 2^22 | 8.1 d | ~4 d (rescue-only used here) |
| 2^24 | 32 d | ~16 d |

v6e-8 is provisioned via the **Queued Resources API** (not `instances create` —
the single-host 8t create path is backend-broken). Driver checkpoints per-pid, so
preemption/interruption loses nothing.

## What transfers to the 5x5x5 / 6x6x6 / 7x7x7 cubes

Deadlines 2026-11-20. All are color cubes with the SAME structure — the entire
`cube444/` tree is parameterized on `state_size`/`num_classes` and re-runs with a
new `puzzle_info.json`:
- `src/cube444/puzzle.py`, `symmetry.py`, `models.py` — shape-generic already.
- The TPU kernel's only puzzle-specific hardcode was `state_size` in the packed
  slice (now `[:,0:state_size]`); PACK_SIZE=128 covers state_size up to ~120.
  5x5x5 has 150 stickers -> PACK_SIZE must grow to >=155 (state_size+5); 6x6x6
  (216) and 7x7x7 (294) likewise. That's the one edit to expect.
- The `num_classes=6` trap, the color-cube sym (rotate+recolor), the compile
  cache, and min-merge-vs-floor all apply unchanged.

## Next levers (not yet done)

1. **More rescue** — 2^22 (or wider) on the next-100 longest paths of 53,426;
   cheap, incremental, ~another few hundred moves. Same recipe, new `--pids`.
2. **Full 2^22** — ~4 d on one v6e-8 (or shard pid-ranges across boxes); projects
   to ~50 mean.
3. ~~**Structural**: exact corner PDB as a max-combine LB, or two-phase orbit-staged
   search using the 4 closed slot orbits.~~ **BOTH BUILT AND MEASURED 2026-07-31 —
   see EXPERIMENTS.md.** Two-phase is complete and verified (reduction beam +
   classical 3x3x3 finish) but **cannot beat the floor with an HTM-optimising phase-2
   solver**: phase-2 QTM is pinned at 27.2 (God's number 26; QTM-optimal averages
   ~21) while distance-to-R has a counting bound of 21, so the two-phase floor is
   48.2 > 46.73. The PDB route is worse than it looks — orbit projections certify a
   mean LB of only 9.7 against real lengths of 46.7, and a full 88.2M corner PDB
   would lift that to just 14.
4. **QTM-native phase-2 solver** (the live follow-up). At phase-2 = 21 the two-phase
   floor drops to 42, under Rokicki's 44.39. Everything else for two-phase already
   exists and is verified.

## Reproduce / resume

Local (training): `scripts/00_verify` ... `08_train_az`, configs in `configs/`,
models in `models/`. TPU (inference): `tpu/RUNBOOK.md`, artifacts staged at
`gs://mm-tpu-staging-0977634337/cube444/`, provision via the
`megaminx-tpu-provision` skill. NOT submitted beyond 53,426; all TPUs torn down.
