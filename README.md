# cayley — CayleyPy competition solvers

Neural distance/action heuristics + large-scale beam search on GPU and TPU, for the
CayleyPy family of Kaggle puzzle competitions.

This repo now hosts **four** solvers sharing one core library (`src/cayley/`). Each has
its own handoff doc, which is the authoritative state — this file is a router.

## Status (2026-08-03)

| Puzzle | Best | Standing | Where |
|---|---|---|---|
| **Professor Tetraminx** | **28,467** | **#1 — beat Rokicki's 28,481 by 14** | [`tetraminx/HANDOFF.md`](tetraminx/HANDOFF.md) |
| 4×4×4 cube (`cube444`) | 48,738 | #2 public LB at time of submission (53,426) | [`cube444/HANDOFF.md`](cube444/HANDOFF.md) |
| Megaminx | 73,441 working floor (75,200 submitted) | — | [`megaminx/HANDOFF.md`](megaminx/HANDOFF.md) |
| IHES Picture Cube | 24,618 | leader Rokicki 21,840 | [`EXPERIMENTS.md`](EXPERIMENTS.md), [`IDEAS.md`](IDEAS.md) |

Tetraminx is the most developed and the only one currently in first place; its
[`BLOG_tetraminx_progress.md`](tetraminx/BLOG_tetraminx_progress.md) is the best single
narrative of the approach that works.

## Start here in a new session

1. **`CLAUDE.md`** — non-negotiable rules and the anti-pattern list. Read this first;
   several rules exist because ignoring them cost hours.
2. The **`HANDOFF.md` of the puzzle you're working on** (table above). Each carries its
   own score progression, measured ablations, and a prioritised next-experiments list.
3. `EXPERIMENTS.md` / `IDEAS.md` — IHES-cube-specific log and untried ideas.
4. **`pathkit/`** — every path post-processing method (min-merge, window rewriting,
   bridging, exact ladders, neural bridge, suffix re-solve) as one puzzle-agnostic
   package, with the measured verdict for each in
   [`pathkit/VERDICTS.md`](pathkit/VERDICTS.md) and the process rules in
   [`pathkit/DISCIPLINE.md`](pathkit/DISCIPLINE.md). Start with
   `python -m pathkit.cli plan --preset <puzzle> --in <submission.csv>`.

## The approach, in one page

All four solvers share a shape:

1. **Train a heuristic.** Either a value function `V(s)` (distance-to-solved) or an
   **all-neighbours Q head** `Q(s,a)` with one output per generator. The Q head scores
   every child from ONE forward on the parent — **17.2×** cheaper per beam step measured,
   and what current tetraminx work uses. (The PyTorch searcher adds *progressive top-k*,
   hashing only the top candidates, for 21.9×; the JAX/TPU kernel does **not** have it —
   it still materializes and hashes all 24 children before its per-owner top-K.)
2. **Beam search**, as wide as hardware allows. On every puzzle tried so far, **width has
   been a stronger lever than heuristic quality**, up to a saturation point.
3. **Exact endgame table.** Stop the beam when it enters a precomputed BFS ball
   (d ≤ 6 for tetraminx, 27.8M states) and splice the table's optimal descent. The last
   moves become provably optimal, and the beam stops where it is narrowest and least
   reliable.
4. **Symmetry frames.** Solve conjugated and inverted copies of the scramble and keep the
   shortest path. The *inverse* frame does most of the work; frames saturate at 2.
5. **N-way per-pid min** across every result source, replay-verified. This is the actual
   submission — see CLAUDE.md rules 26 / 26b, which exist because scattered results have
   repeatedly hidden 20–100 moves.

## Layout

```
src/cayley/            shared library — puzzle, model, training, beam, verification
  khoruzhii_search.py    PyTorch beam — Q-head path, progressive top-k, qv-rerank/consistency
  bellman.py             Bellman refinement
  model.py               ResMLPDistance
scripts/               IHES cube entrypoints
tetraminx/             Professor Tetraminx solver  (HANDOFF.md, src/, scripts/, kaggle_notebooks/)
megaminx/              Megaminx solver             (HANDOFF.md, beam_lab/, scripts/)
cube444/               4x4x4 cube solver           (HANDOFF.md)
data/                  IHES competition data + precomputed tables
configs/               YAML hyperparameter configs
models/  submissions/  gitignored
```

Each puzzle package duck-types the `PictureCube` interface, so everything in
`src/cayley/` works against all of them unchanged.

## Environment

Python 3.14 + torch 2.11.0+cu128 + cayleypy 0.1.0 + triton-windows.

**Always use `.venv/Scripts/python.exe`** — not `python`, `python3`, or `.venv/python`.
Same for `.venv/Scripts/kaggle.exe` and `.venv/Scripts/pip.exe`. See CLAUDE.md rule 1.

## Hardware paths

| Target | Notes |
|---|---|
| Local RTX 4090 | training + solves; launch anything over ~20 min **detached** (`Start-Process` + `.bat`), not via background Bash |
| GCP L4 / A100 spot | training; A100s need a boot `startup-script` + watchdog cron, since preempted spot VMs never restart themselves |
| GCP TPU v6e-8 | the wide beam — one shared beam sharded across 8 cores via `all_to_all` |
| Kaggle TPU v5e-8 | public/shareable notebooks; ~half a v6e-8 |

## Slash commands

| Command | Does |
|---|---|
| `/tetraminx-eval` | standard 15-pid TPU beam eval, guarded, scored against the standing best |
| `/tetraminx-merge` | n-way per-pid min over every source → verified submission |
| `/tetraminx-submit` | merge → independent replay-verify → submit → confirm the score |
| `/megaminx-*` | megaminx bench / eval / sync / submit helpers |
| `/check-gcp`, `/kaggle-push` | infra |

## Gotchas that bite hardest

These are the short list; **`CLAUDE.md` has the full set with the measurements behind
them.**

1. **`torch.compile` for beam inference is only safe with fixed-shape padding.** Naive
   `model(candidates)` recompiles on every shape change (5.8× slowdown). With
   `pad_to_batch_size=True` plus a pre-warm it is ~−27% wall. Training is always fine.
2. **Reuse one `CayleyGraph` per session** — fresh instances get different random hash
   vectors and corrupt cross-call state tracking.
3. **Probe metrics do not select beam checkpoints.** Four separate models won on pair
   accuracy / top-1 / recall / calibration and delivered **zero** moves in search. Only a
   replay-verified beam total counts.
4. **Run the matched control before quoting any A/B delta** (rule 28). Comparing across
   machines or flag sets manufactured two retracted conclusions in one day. If a config
   change gives *byte-identical* results, the flag is not wired.
5. **The n-way merge is only as good as its source set** (rule 26b). Re-pull live
   machines, copy sources into a stable dir, and search by file *content* — result files
   have turned up in unrelated project folders.
6. **Never use `sed -i`** on this machine — MINGW sed silently truncates files to 0 bytes.

## References

- Cross-project research notes: `C:\Users\and-l\kaggle_research\`
- CayleyPy paper: arXiv:2502.13266 + 2502.18663
- Sparse-Q objective / PieceTransformer origin:
  [`AnanasClassic/cayleypy-training-core`](https://github.com/AnanasClassic/cayleypy-training-core)
