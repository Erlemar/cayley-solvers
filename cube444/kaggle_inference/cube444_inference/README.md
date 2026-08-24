# cube444 — inference bundle

Self-contained beam-search inference for the CayleyPy 4x4x4 **colour** cube, with the best
scorer measured as of **2026-08-09**. Copy the archive anywhere with a CUDA GPU, install two
packages, run one script.

```bash
tar xzf cube444_inference.tar.gz
cd cube444_inference
pip install "tqdm>=4.66" pandas scipy          # plus torch for your CUDA build
CUBE444_PY=$(which python) ./run_best.sh --pids 8,199,399
```

Nothing outside this directory is referenced. All paths are relative; the only absolute path
is the Python interpreter, via `$CUBE444_PY` (defaults to `python`).

---

## The headline number

`run_best.sh` is s3 + blend 0.4 + endgame 6. On the 54 held-out pids at B=65536:

| config | total | vs floor 2609 | wins | merge gain |
|---|---|---|---|---|
| `orig_transformer` (shipped bundle) | 3011 | +15.4% | 0 | 0 |
| `s1L_4k` | 2959 | +13.4% | 0 | 0 |
| `s3` standalone | 2931 | +12.3% | 1 | 2 |
| `s3` + blend 0.4 | 2859 | +9.6% | 2 | 4 |
| **`s3` + blend 0.4 + endgame 6** | **2851** | **+9.3%** | **2** | **4** |

**−160 moves against the shipped bundle on 54 pids**, and the only configuration that beats
the 46,662 floor outright on more than one pid.

---

## How to judge a run — read this before quoting any number

**Judge by the per-pid MIN against the floor, never by the standalone mean.** This is not a
stylistic preference; it is the difference between shipping and not. History from this project:

```
community floor (public kernels)      54,754
our beam @ 2^20, standalone           55,846   <- WORSE than the floor
community min-merge our-2^20          53,756   <- -998, won 314 pids outright
+ 2^22 rescue on top-100              53,426   <- SUBMITTED, #2 on the LB
```

A run whose standalone mean is worse than the floor can still be the best contribution you
have. Merge before you conclude:

```bash
cd solver && python merge_ensemble_progress.py   # see its --help
```

Every config in this bundle is still *above* the floor standalone. They earn their keep in the
merge, which is what the `wins` and `merge gain` columns above measure.

---

## What's in the box

```
run_best.sh                     the best measured config, one command
solver/
  solve_ensemble_submission.py  main driver (Q-head beam + optional MLP blend)
  bench_beam.py                 A/B harness; judges by per-pid min, prints a matched table
  merge_ensemble_progress.py    n-way per-pid min merge
  test.py                       alternative driver (V/scalar path, more search flags)
  pilgrim/                      searcher, Q-searcher, model factory, utils
  unified_training/             PairQMLP definition -- REQUIRED by the blend
  generators/p002.json          move set
  targets/p002-t000.pt          solved state
  test.csv                      the 1043 puzzle states
  models/
    s3/                the scorer. Q-Bellman-refined PieceTransformer, 3,383,064 params
    s3_step2000/       s3's offline-best checkpoint -- UNBENCHED, see "open" below
    s1L_4k/            the stage-1 checkpoint s3 was refined from (control)
    orig_transformer/  the bundle's original transformer, byte-identical to as-shipped
    mlp_x16/           ResMLP x16 scalar rescorer, 180 MB -- REQUIRED by the blend
submissions/
  cube4_submission_46662.csv    the floor and the merge base (verified 1043/1043)
reports/                        five markdown reports; see index at the bottom
```

Weight provenance (md5):

```
5388573d511140f581cdee7bad6e2d42  orig_transformer/model.pth
9b6ad13a21f6b05d7ee5795ac76d6d1d  s3/model.pth
13fae05d75ca9397be5129b1ebd1585d  mlp_x16/model.pth
```

`s3`, `s1L_4k` and `orig_transformer` are the same architecture and load interchangeably
(`strict=True`). Swapping scorers is just `--transformer-info/--transformer-weights`.

---

## Flags that actually matter

| flag | default in `run_best.sh` | what it does |
|---|---|---|
| `--B` | 2**21 (driver default) | beam width. **The main quality lever.** Cost is linear in B. |
| `--mlp-weight` | **0.4** | blend weight. 0.0 = transformer standalone. Worth −72 moves at 0.4. **Never swept — 0.4 is the bundle author's default, not a tuned value for s3.** |
| `--tail-bfs-depth` | **6** | exact endgame. Terminates as soon as any beam state is within N moves and finishes with the exact BFS path. |
| `--num-attempts` | 2 (driver default) | retries with a different blacklist |
| `--num-shards` / `--shard-index` | — | shard a full pass across boxes; the driver checkpoints per pid, so an interrupt loses nothing |
| `--compile` | off here | see gotcha 3 |
| `--history-depth` | 0 | **leave at 0.** Measured +16 moves at 1. See gotcha 4. |

### On `--tail-bfs-depth`

Depth 6 is a **67,041,676-state** table built at startup: **291 s and 26.8 GB of RAM**, every
process, every run. Budget for it. What you get:

- **−8 moves** on the held-out 54 (−2 standalone at 65k). Structurally incapable of making a
  path *longer* — an exact tail can only replace a suboptimal one.
- **Roughly halves beam wall-time** by terminating early. On the 18-pid set the search itself
  went 451 s → ~241 s. On a long run this pays for the build many times over.

**Depth 5 is worthless** (3.5M states, 21 s build): it fires on 7 of 18 pids and changes *zero*
path lengths. Near solved the Q head is already optimal — pair accuracy 0.968, top-1 0.866 at
depth 2–6 — so there is nothing for an exact table to improve. Use 6 or 0.

Depth 7 is not reachable here: it needs a complete-to-7 table whose one-pass expansion is
146 GB.

---

## Gotchas

These are the ones that cost real time. The full set is in `reports/` and the handoff's
`06_GOTCHAS.md`.

1. **`num_classes = 6`, not `state_size`.** Every model constructor defaults it to `state_size`,
   correct for a permutation puzzle and wrong here. It does not error — it silently builds a
   96-way embedding and a larger, worse model. The bundle passes it correctly; preserve that if
   you touch `pilgrim/factory.py`.

2. **This is a colour cube.** A state is a colouring, so there is **no `invert_state`** — no
   NISS, no inverse frames, no bidirectional search, 24 rotation frames rather than 48. Symmetry
   needs a *recolour*, `sym(s,R) = color_map_R[s[rotation_R]]`; the slot-permutation-only form
   produces a legal-looking colouring and is silently wrong.

3. **Do not `torch.compile` beam inference without padding to a fixed batch size** — measured
   **5.8x slowdown** from reshape recompiles. Training has fixed shapes and is fine.

4. **`--history-depth` has inverted semantics here.** CayleyPy's `history_depth=N` *adds*
   cross-layer dedup. This searcher already dedups against **every** previous layer, so N>0 is a
   **relaxation**. Measured at 1: 982 vs 966, and 2/3 → 1/3 solves at small width. Leave it at 0.

5. **Byte-identical results across a config change usually mean the flag is not wired** — but
   verify before concluding either way. `--tail-bfs-depth 5` scores *identically* to 0 here and
   is nonetheless fully wired: 7 of 18 solution *strings* change, all to the same length. Diff
   the paths, not just the totals.

6. **Verify before quoting or submitting.** Never trust a file's own reported total.

7. **`pgrep -f` / `pkill -f` self-match** the checking command's own arguments. Read the pids
   first, then `kill <numeric pids>`.

---

## Reproducing the tables

`bench_beam.py` runs matched arms and reports per-pid min against the floor:

```bash
cd solver
CUBE444_PY=$(which python) python bench_beam.py \
  --pids 8,199,399,599,799,999 --beam 65536 --steps 110 \
  --arms "s3=models/s3|mlp=0.4|tail=6,orig=models/orig_transformer|mlp=0.4|tail=6" \
  --out /tmp/ab.json
```

Per-arm options are `|mlp=W`, `|tail=N`, `|hist=N`; each arm gets its own result DB, so a
matched A/B of a *search* setting runs inside one invocation. Override the floor with
`$CUBE444_FLOOR`.

**Pid-set resolution matters more than it looks.** The 18-pid stratified set inverts rankings
for anything under ~2%: it ranked `s3` *last* among arms it in fact beats, and it said the blend
destroys merge contribution (gain 2 → 0) when on 54 pids the blend *doubles* it (2 → 4). Use
≥54 pids for any margin under 2%. The beam is deterministic — zero variance across hash seeds —
so disagreements are set-composition, not noise, and re-running with a new seed tells you
nothing.

---

## Open, if you pick this up

1. **`models/s3_step2000/` has never been benched.** It is s3's offline-*best* checkpoint
   (deep_score 0.8569); the shipped `s3/` is step 6000, whose 0.8494 was the *worst* of six
   evals. No offline metric on this puzzle tracks beam quality — the arm with the best
   deep_score of all (0.8663) had the worst beam — so this is a genuine coin flip worth ~25 min.
2. **`--mlp-weight` is unswept.** 0.4 is the bundle author's default, tuned against *their*
   transformer, not s3. A 0.2/0.4/0.6 sweep on 54 pids is ~72 min and applies to everything
   downstream.
3. **The blend and endgame are only measured at B=65536.** s3's edge over `orig` *grew* with
   width (+2 at 65k → −16 at 2^20); whether the blend's does is untested.
4. **No symmetry ensemble on this code path.** The handoff calls sym-ensemble the main quality
   knob (~−2 moves per doubling of K, 1 frame → 67% solve, 4 frames → 100%), but it exists only
   in `KhoruzhiiSolver`, not in this driver. Absolute totals here are not comparable to numbers
   produced by that solver.
5. **A full 1043-pid pass** at B=65536 is ~7.7 h with the blend, and *faster* with endgame 6
   (~6.7 h + build) because of early termination.

---

## Reports

| file | what it is |
|---|---|
| `cube444_stage3_heldout_2026-08-09.md` | main results: the training chain, the measurement problem, deeper anchors, training speed, the 1M run |
| `cube444_s3_bellman_recipe_2026-08-09.md` | how `s3` was trained — full reproducible recipe, diagnostics, and the traps |
| `cube444_transformer_results_2026-08-09.md` | the earlier transformer session |
| `cube444_transformer_training_plan_2026-08-08.md` | the plan those came from |
| `cube444_cellD_standalone_report_2026-08-08.md` | standalone PieceTransformer study |

The two dated `2026-08-09` reports predate the blend and endgame results in this README; where
they disagree, this README is newer.

---

## Environment used for all measurements

```
A100-SXM4-80GB, 176 GB RAM
python 3.12.13   torch 2.6.0+cu124   CUDA 12.4
```

Nothing depends on that specific stack — `torch` and `tqdm` are the only hard requirements
(`pandas`/`scipy` are used by peripheral scripts) — but the wall-clock figures assume an A100,
and endgame depth 6 needs ~27 GB of host RAM.
