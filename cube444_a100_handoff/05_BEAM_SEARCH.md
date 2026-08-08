# 05 — Beam search: current best approach, and how to judge it

## Run it

```bash
python cube444/scripts/05_solve.py \
  --checkpoint cube444/models/c_bells2/epoch_0399.pt \
  --out cube444/submissions/run.csv \
  --beams 16384,65536 --max-steps 70,140 \
  --sym-ensemble 4 \
  --bf16 --resume
```

Multi-pass (`--beams a,b`) means: try the narrow beam first, and only escalate to the wide
one for pids it failed. Cheap pids stay cheap.

Useful flags: `--pids` (explicit list), `--stratified N`, `--limit`, `--fallback <csv>`
(seed from an existing solution), `--sym-positions` (shard the symmetry frames across
boxes), `--internal-batch-size` (tune to the A100's memory).

## The symmetry ensemble is the main quality knob

**~-2 moves per doubling of K**, and it closes the coverage gap outright: 1 frame solves
67% of pids, 4 frames solves **100%**.

There are 24 whole-cube rotations. On a colour cube each frame needs the *recolour*, not
just the slot permutation:

```
sym(s, R) = color_map_R[ s[rotation_R ] ]
```

`rotations_24.npy` and `color_maps_24.npy` are shipped. **There are no inverse frames** —
`invert_state` does not exist here — so unlike megaminx/IHES you get 24 frames, not 48.

Never compare raw V scores across frames. Scores are only comparable within a frame.

## Width is the lever

Per-step cost is linear in B. For reference, full 1043-pid sym-4 runs on TPU v6e-8 took
~16 h at 2^20 and ~4 d at 2^22. On a single A100 expect materially longer; **shard by pid
range** and treat each shard as independently mergeable — the driver checkpoints per pid,
so an interrupted run loses nothing.

### Concentrate width on the longest paths

The highest-yield use of an expensive beam: run it **only on the pids whose current path
is longest**. Measured — 2^22 on the top-100 longest gave **-330 moves in 5.5 h**, versus
an estimated ~8 days for a full 2^22 run. 81/100 improved, **0 got worse** (a wider beam
with the same V essentially never regresses).

```bash
# rank current solution by path length, take the worst N, re-solve those only
python cube444/scripts/05_solve.py --checkpoint <ckpt> --out rescue.csv \
  --pids "$(python - <<'EOF'
import csv,io
rows={int(r['id']):len(r['moves'].split('.')) for r in csv.DictReader(io.open('current.csv',encoding='utf-8'))}
print(','.join(str(p) for p in sorted(rows,key=lambda p:-rows[p])[:100]))
EOF
)" --beams 4194304 --max-steps 140 --sym-ensemble 4 --bf16
```

## HOW TO JUDGE A RUN — the rule that matters most here

**Judge by the per-pid MIN against the floor, never by the standalone mean.**

Our own history is the argument:

```
community floor (public kernels)          54,754
our beam @ 2^20, standalone               55,846   <- WORSE than the floor
community min-merge our-2^20              53,756   <- -998, won 314 pids outright
+ 2^22 rescue on top-100                  53,426   <- SUBMITTED, #2 on the LB
```

The model that looked like a failure on its own mean was worth ~1,300 moves in the merge.
It explores differently and wins ~30% of deep pids. **A run whose standalone mean is worse
than the floor can still be your best contribution.** Always merge before you conclude.

```bash
python cube444/scripts/06_merge.py --inputs a.csv b.csv c.csv --out merged.csv
python cube444/scripts/73_verify_submission.py merged.csv          # POSITIONAL, not --submission
python cube444/scripts/73_verify_submission.py merged.csv base.csv # optional 2nd arg = baseline to diff
```

Merge sources by **content**, not filename — check the header *and* that the move alphabet
matches this puzzle's generators. Files from a sibling puzzle pass the header test and
fail the second.

## Always verify before quoting or submitting

```bash
python cube444/scripts/73_verify_submission.py <csv> [baseline.csv]
```

POSITIONAL args, not `--submission`. Replays every path against the original state in
`test.csv`; expect `VERDICT : PASS` with `missing 0 / bad moves 0 / unsolved 0`. An
optional second argument diffs against a baseline. Never trust a file's own reported
total — verify, then quote.

## What is already exhausted — do not spend time here

* **Post-processing / window rewriting.** Every window of length <=10 in the best public
  file is proven geodesic. The full tetraminx rewriting ladder ported to colour space
  returns **-8 total**. Each further exact rung costs 19.2x for ~2 moves.
* **The merge axis.** The n-way per-pid min over 125 replay-verified CSVs is exactly the
  best public file — it dominates all of them on all 1043 pids.
* **Two-phase (reduction + 3x3x3 finish).** Built and verified, but cannot beat the floor
  with an HTM-optimising phase 2: phase-2 QTM is pinned at 27.2 while distance-to-R has a
  counting bound of 21, so the two-phase floor is 48.2 > 46.73. The live variant would be
  a **QTM-native phase-2 solver** — at phase-2 = 21 the floor drops to 42, under Rokicki's
  44.39. Everything else for two-phase already exists and is verified.
* **Corner PDB as a lower bound.** Orbit projections certify a mean LB of only 9.7 against
  real lengths of 46.7; a full 88.2M corner PDB lifts that to just 14. Not competitive.

## Colour-cube specifics that bite in search

* **No `invert_state`** -> no NISS, no inverse frames, no bidirectional tricks.
* A state is a colouring, so **different group elements can be the same state**. Hash on
  the colouring, which is what the code already does — but it means the search graph has
  more collisions than a permutation puzzle of comparable size, and dedup matters more.
* `num_classes = 6` at every model construction site (gotcha #1).
