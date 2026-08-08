# cube444 on a fresh A100 — handoff package

You are picking up the **CayleyPy 4x4x4 cube** (https://www.kaggle.com/competitions/cayley-py-444-cube,
deadline **2026-09-22**). This package is self-contained: everything here can be run on a
single A100 with no access to the originating machine.

**Read this file, then `01_SETUP.md`, then follow the numbered docs in order.** Do not
start training before reading `06_GOTCHAS.md` — four of the items there have each cost a
full run.

## The five facts that shape everything

1. **This is a COLOUR cube, not a permutation puzzle.** 96 stickers, 6 colours x 16. A
   state is a *colouring*, so it pins the group element only up to the `4!^6 =
   191,102,976` stabiliser of the solved colouring. It is a Schreier coset graph, not a
   Cayley graph. Consequences: **no `invert_state`** (so no NISS, no inverse symmetry
   frames), and **`num_classes = 6`, not `state_size`** — every model constructor in the
   shared library defaults this wrong. Pass it explicitly, every time.

2. **The scorer is the bottleneck here, unlike our other puzzles.** On megaminx and the
   IHES cube we concluded "architecture is closed, gains are inference-side." **That does
   NOT transfer to this cube.** A transformer from Vlad Kuznetsov beats our beam on
   **916/1043 pids (87.8%)**, saving 4,688 moves at a median margin of 6, *uniformly
   across every difficulty band* — not a tail effect. That is ~5 moves/pid of pure scorer
   quality. Closing it is the single highest-value thing you can do.

3. **The open question you are here to answer**: is that win the transformer
   **architecture**, or the **sparse-Q objective** it is trained with? They were never
   separated. `04_TRAIN_SPARSE_Q.md` is designed as a clean 2x2 to settle it.

4. **Judge a beam run by the per-pid MIN against the floor, never by its standalone
   mean.** Our model alone scored 55,846 when the community floor was 54,754 — worse
   standalone — yet min-merging it took us to 53,756. It explores differently and wins
   ~30% of deep pids. A run that looks like a failure on its own mean can be worth
   hundreds of moves.

5. **Post-processing is exhausted; do not spend time there.** Every window of length <=10
   in the best public file is proven geodesic, the full rewriting ladder returns -8 total,
   and the n-way merge over 125 verified CSVs is exactly the best file. The remaining
   418 moves to beat Rokicki must come from a better scorer or a better search.

## Current state of play

| | |
|---|---|
| Rokicki (leader) | **46,298** |
| best public file | **46,662** (`code/submissions/cube4_submission_46662.csv`, verified PASS, mean 44.738) |
| our own best solve | 53,426 (#2 when submitted 2026-07-25) |
| our deployed model | `c_bells2/epoch_0399.pt`, ResMLP V, 3.29M params |
| gap to close | **364 moves = 0.35/pid** vs the 46,662 file |

**You are training from scratch, so you will not reproduce 53,426 immediately.** The
target for a first end-to-end run is a V model that solves the 18-pid eval set and beams
in the same range as `c_bells2`; `03_TRAIN_RESMLP_V.md` gives the checkpoints and the
numbers to compare against at each stage.

## What is in here

```
README.md                 <- you are here
01_SETUP.md               environment, deps, verification that the port is correct
02_DATA.md                data generation + THE LABEL-CONFLICT FIX (read in full)
03_TRAIN_RESMLP_V.md      the proven 3-stage V recipe, with expected numbers
04_TRAIN_SPARSE_Q.md      ResMLP-Q and PieceTransformer on sparse-Q -- the open question
05_BEAM_SEARCH.md         current best search, and how to judge it
06_GOTCHAS.md             the traps, each one measured
07_TRAIN_AZ.md            AlphaZero dual head -- BUILT but never run; cheap open lever
skills/                   drop into .claude/skills/ on the A100 box
code/                     runnable source; see 01_SETUP.md for layout
code/reference/           tetraminx sparse-Q trainer + PieceTransformer to port from
code/submissions/         the community 46,662 solution (AZ policy targets + merge floor)
```

## Suggested order of work

1. `01_SETUP.md` — get imports green and run `test_symmetry.py`. **Do not skip**; it
   checks the colour-cube symmetry identity that silently corrupts everything if wrong.
2. `02_DATA.md` — build the d<=6 BFS ball (this is also your anchor source AND the
   label-conflict oracle).
3. `03_TRAIN_RESMLP_V.md` — Stage 1 pretrain -> Stage 2 Bellman. This is the known-good
   path; get it working before trying anything new.
4. `04_TRAIN_SPARSE_Q.md` — the actual research. Run the 2x2.
5. `05_BEAM_SEARCH.md` — evaluate whatever you trained, always as a min-merge.
6. `07_TRAIN_AZ.md` — optional, and cheap: one dataset build + one fine-tune from the
   trunk you already have. Untested, not rejected.

If you only have budget for one thing, do **step 4** — it is the open question and the
only lever with 5 moves/pid behind it.
