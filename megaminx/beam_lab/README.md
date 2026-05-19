# beam_lab — sandbox for beam-search optimization

A **self-contained, portable** micro-benchmark for tuning the beam-search inner
loop. Same solver code as production, ships with the m05 checkpoint and 12
sample puzzles, so it can be tarred up and run on any GPU machine without the
rest of the repo.

Use this folder to prototype optimizations, benchmark them, and (once
validated) graft the winning changes back into `src/cayley/khoruzhii_search.py`.

## Folder layout

```
beam_lab/
├── README.md
├── RESEARCH.md            # deep dive on optimization ideas + tier list
├── requirements.txt       # torch + numpy
├── puzzle.py              # Megaminx class (self-contained)
├── model.py               # ResMLPDistance + load_checkpoint
├── beam_search.py         # KhoruzhiiSolver, KhoruzhiiSearchConfig, Profile
├── beam_search_batch.py   # BatchedKhoruzhiiSolver (multi-puzzle parallel)
├── beam_search_mitm.py    # MitmKhoruzhiiSolver (BFS-d6 shell termination)
├── run_benchmark.py       # CLI driver: --mode {single,batch,mitm}
├── models/m05_bellman_warm/epoch_0499.pt   # checkpoint (24 MB, included)
├── data/
│   ├── puzzle_info.json   # Megaminx generator definitions
│   └── sample_puzzles.csv # 12 puzzles spanning every 100-pid bucket
└── results/               # benchmark CSVs land here
```

Sample puzzles: pids `0, 50, 150, 250, 350, 492, 550, 650, 750, 850, 950, 1000`.
Pid 492 is the only puzzle our production m05 failed at beam 131k — useful
target for "did the optimization actually crack it?".

## Running on a fresh machine

1. **Package** the lab from the source repo:
   ```bash
   tar czf beam_lab.tgz -C megaminx beam_lab/
   ```
   (~24 MB — the m05 checkpoint dominates.)
2. **Copy** to the target machine, e.g. GCP:
   ```bash
   scp beam_lab.tgz user@gcp-vm:~/
   ssh user@gcp-vm 'tar xzf beam_lab.tgz && cd beam_lab && ls'
   ```
3. **Install** deps:
   ```bash
   cd beam_lab
   python -m venv .venv && source .venv/bin/activate    # Linux/macOS
   #   or: python -m venv .venv && .venv\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```
4. **Verify**:
   ```bash
   python verify_install.py
   ```
   Prints what's wired up; runs a tiny end-to-end solve on pid 50 (a few seconds
   on GPU, a couple of minutes on CPU) and verifies the path applies to the
   identity. If you see `[FAIL]` in the output, fix that before running benchmarks.
5. **Smoke-test** (real beam config, GPU only):
   ```bash
   python run_benchmark.py --checkpoint models/m05_bellman_warm/epoch_0499.pt \
       --beam 16384 --max-steps 60 --bf16 --pid 50
   ```

The included checkpoint is **m05** (~6 M params, our strongest production model,
scored 95,682 on the Kaggle leaderboard). To use a different model (m07, m17,
m13), copy that .pt file in — `load_checkpoint` reads `model_config` from the
file so no code changes needed.

GPU recommended (CUDA). Full 12-sample run takes ~1–2 min at beam 32k on a 4090,
~3–5 min at beam 131k. CPU works but is ~50× slower.

> ⚠️ **Don't run two beam-search jobs on the same GPU**. Set
> `CUDA_VISIBLE_DEVICES=...` to pin to a free GPU, or wait until any
> production solve finishes.

## MITM mode (optional)

MITM mode (`--mode mitm`) needs a precomputed BFS-d6 shell file (~2.5 GB) which
is **not bundled** in beam_lab. Build it once with the production code (it's
the same file used by `03_solve.py --bfs-table`):

```bash
# from the main repo:
python megaminx/scripts/build_bfs_bytes.py --max-depth 6 \
    --out megaminx/data/bfs_bytes_d6.pkl
```

Then point the lab at it:

```bash
python run_benchmark.py --checkpoint models/m05_bellman_warm/epoch_0499.pt \
    --mode mitm --bfs-table /path/to/bfs_bytes_d6.pkl \
    --beam 131072 --max-steps 120 --bf16
```

The lab defines a self-contained `BfsBytesTable` that's pickle-compatible with
the production class, so the .pkl unpickles cleanly even without the rest of
the project on `PYTHONPATH`.

## How to run

### 1. Single-beam baseline

```bash
.venv/Scripts/python.exe megaminx/beam_lab/run_benchmark.py \
    --checkpoint megaminx/models/m05_bellman_warm/epoch_0499.pt \
    --beam 131072 --max-steps 120 --bf16
```

You'll get a per-puzzle table like:

```
 pid    beam found  path  total  model  neigh  dedup  apply   topk   hash steps
   0  131072     1     0   0.01   0.00   0.00   0.00   0.00   0.00   0.00     0
  50  131072     1    52  18.42  12.31   2.10   1.85   0.92   0.81   0.43    52
 150  131072     1    87  35.21  23.55   3.94   3.50   1.71   1.56   0.95    87
 ...
```

Followed by an aggregate row showing where wall time goes (`model 65% neighbor 12% ...`).

### 2. Sweep beam sizes

```bash
.venv/Scripts/python.exe megaminx/beam_lab/run_benchmark.py \
    --checkpoint megaminx/models/m05_bellman_warm/epoch_0499.pt \
    --beam 16384,65536,131072,262144 --max-steps 120 --bf16 \
    --out megaminx/beam_lab/results/m05_beam_sweep.csv
```

### 3. Just the hard cases

```bash
.venv/Scripts/python.exe megaminx/beam_lab/run_benchmark.py \
    --checkpoint megaminx/models/m05_bellman_warm/epoch_0499.pt \
    --beam 131072,524288 --max-steps 120 --bf16 \
    --pid 492        # the one m05 fails at beam 131k
```

### 4. A/B against an alternate implementation

The way to test an optimization is to add a flag to `beam_search.py`,
implement the alternative behind it, and benchmark on the same 12 samples.
Compare:
- **wall time** (lower is better)
- **path lengths** for each puzzle (must NOT increase by ≥ 1 move on the median)
- **`model_s` fraction** (sanity check: does the optimization hit where you intended?)

A change that saves wall time but produces longer paths on hard puzzles is a
loss, not a win — beam search is a quality-vs-speed knob already.

## What the columns mean

| column | what it measures |
|---|---|
| `total_s` | wall clock from `solver.solve(state, cfg)` entry to return |
| `model_s` | cumulative time in `_model_predict` (model forward pass) |
| `neighbor_s` | cumulative time generating + hashing the 24 children of each beam state |
| `dedup_s` | cumulative time in unique-hash filtering |
| `apply_s` | cumulative time materializing the chosen `B` next states |
| `topk_s` | cumulative time in `argsort(value)[:B]` |
| `hash_s` | cumulative time in the post-step state-hash log (separate from neighbor hashing) |
| `n_steps` | number of beam-search layers actually expanded before solve / abort |

All timings are real wall-clock seconds — the solver calls `torch.cuda.synchronize()`
before reading `time.time()` so async kernel launches don't make the model look free.

---

## Optimization ideas, ranked by ROI

For the full deep dive (math, smallest-experiments, anti-patterns,
GCP budget plan), see [`RESEARCH.md`](./RESEARCH.md). Brief versions
below.

### Tier 1 — high ROI, code we already have or near-trivial

#### 1. MITM beam search

Beam terminates as soon as it touches a state in the BFS-d6 shell (19.4M
states, depth ≤ 6). Saves up to 6 search depths × beam-width worth of work on
hard puzzles.

We already have `MitmKhoruzhiiSolver` (in `megaminx/src/megaminx/mitm_solver.py`).
Never run end-to-end on full 1001. Expected: **2-3× faster on hard puzzles, 0
quality loss** (BFS path is exact).

To benchmark: import `MitmKhoruzhiiSolver`, give it the BFS-d6 shell hashes,
swap into `run_benchmark.py`. Compare on pid 492 first.

#### 2. Adaptive beam per bucket

Easy buckets (0-99) solve at beam 16k just as well as 131k — we're paying
8× more compute per puzzle for nothing.

Wire `--beams 16384,65536,131072` with **auto-escalation only when smaller
fails**. The solve script already supports the comma syntax, but it always
runs all listed beams sequentially. Change to: stop on first success.

Expected: **2× wall-clock for the full 1001-pid run, 0 quality loss**.

To benchmark: instrument `run_benchmark.py` to take an escalation list and
short-circuit on first solve.

#### 3. Multi-puzzle batching

Currently we solve one puzzle at a time at beam 131k → GPU underutilized
between puzzles (model forward is shape-stable, just smaller). Pack 2-3
puzzles' beams into one big batch → near-linear speedup until VRAM caps.

The change is invasive: the beam tensor shape becomes `(P, B, S)` instead of
`(B, S)`, and the solved-check / dedup logic needs to track per-puzzle
status. But model forward becomes one call instead of P calls, which is most
of the win.

Expected: **1.5-2.5×, 0 quality loss**.

### Tier 2 — model side, medium effort

#### 4. Re-do Q-distillation

m06 (our first Q-distillation attempt) was distilled from **m07** (the
RW-trained baseline). It got 10× faster beam search but lost 30% of solve
rate.

Hypothesis: distill from **m05** (sharper Bellman-refined teacher) with a
bigger student head (m06 used the same arch as the teacher; try widening
the head). 10× model-forward speedup if quality recovers.

To benchmark: train a new Q-distill student, replace `_model_predict` with
the Q-head's `(N, n_gen)` output, swap in.

#### 5. Profile-driven

Run `torch.profiler` on a single beam-131k solve and identify if the
bottleneck is model forward, gather/permute, hashing, or top-k. **Without
the profile we're guessing.** Often points to a single fix.

To benchmark: `run_benchmark.py` already shows where time goes per stage.
Pick the biggest fraction — that's where to attack first.

### Tier 3 — system / numerics

#### 6. `torch.compile` for inference at fixed beam

Project CLAUDE.md says compile breaks at variable beam sizes — it does, but
**if you pad the beam to a fixed shape (always B, never less even when fewer
alive states)**, compile should work. The fixed shape costs a few wasted
tensor cells but no recompilation.

Expected: **1.3-1.8×**, watch for compile-cache eviction at startup.

#### 7. CUDA Graphs

Same principle as compile but lower runtime overhead. Capture a single
beam-step into a graph, replay each step. Requires fixed shapes too.

Expected: **1.2-1.5×**, more code change than compile.

#### 8. fp16 vs bf16

We currently use bf16 for inference. fp16 might give better numeric
precision in the value head (bf16 has only 8 bits mantissa). Expected: ≤ 0.05
on path quality, possibly slight speedup. Easy A/B.

#### 9. Larger `internal_batch_size`

We default to 16384. On 24 GB cards (L4, 4090 desktop, 3090) try 32k or 65k.
Each beam step does fewer Python loop iterations.

Expected: **1.1-1.3×**, watch VRAM.

### Tier 4 — algorithmic, larger investment

#### 10. A* / IDA* instead of beam

Beam is space-bounded but quality-suboptimal. A* with our heuristic uses the
same model but explores more efficiently per FLOP — one open queue, expand
lowest-f-value first.

Big lift to implement; uncertain win. The blocker on cubes historically has
been memory (open queue blows up on hard scrambles), but our BFS-d6 shell
gives us a natural cutoff (when a state has BFS distance ≤ 6, take the BFS
path and stop).

#### 11. Symmetry pruning

Megaminx has order-60 icosahedral rotational symmetry. If we orbit-dedup
during search, the effective branching factor drops by ~60×.

Two failed attempts in our project (BFS over generator orbits + adjacency
preservation). Worth one more careful try — high variance outcome but
60× speedup if it works.

#### 12. Better state hash

Linear hash `sum(hash_vec * state)` is fast but has a high collision rate.
A real hash function (xxhash, MurmurHash) would dedup more accurately and
let us drop more candidates per step.

Effect on speed: collision rate is currently low enough that better hashing
doesn't move much. Mostly a quality / safety thing.

---

## Comparing optimizations

When you find a candidate, run the same sweep with and without and report:

```
baseline (m05, beam 131k):
  total wall  = 142.3 s
  avg path    = 88.2
  model_s %   = 65%

candidate X:
  total wall  = 91.7 s   (-36%)
  avg path    = 88.4     (+0.2)  ← acceptable
  model_s %   = 47%
```

Acceptance gate (mirror the project's main acceptance gate):
- Wall time strictly lower.
- Path length: median ≤ baseline median + 1, no individual puzzle ≥ baseline + 5.
- No new "found=0" rows.

If those hold, port to `src/cayley/khoruzhii_search.py` and re-run a stratified-5
on the full pipeline before claiming it.
