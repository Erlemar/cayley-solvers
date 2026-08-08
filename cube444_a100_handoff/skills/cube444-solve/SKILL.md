---
name: cube444-solve
description: Run the cube444 beam search on the A100, merge the result against the floor, and verify it. Use when asked to solve, beam, evaluate a checkpoint on real pids, rescue long paths, or produce a submission for the 4x4x4 colour cube.
---

# cube444 — beam search and merge

Full detail in `05_BEAM_SEARCH.md`. This is the operating loop.

## Standard run

```bash
python cube444/scripts/05_solve.py \
  --checkpoint <ckpt> --out cube444/submissions/run.csv \
  --beams 16384,65536 --max-steps 70,140 \
  --sym-ensemble 4 --bf16 --resume
```

`--beams a,b` is multi-pass: narrow first, escalate only for pids that failed.
`--sym-ensemble 4` is the main quality knob — **~-2 moves per doubling of K**, and it
takes coverage from 67% (1 frame) to **100%** (4 frames). There are 24 rotations and **no
inverse frames** (`invert_state` does not exist on a colour cube).

Shard with `--pids` / `--sym-positions` across boxes; the driver checkpoints per pid, so
interruption loses nothing. Use `--resume` always.

## Targeted rescue — the highest-yield use of an expensive beam

Run a very wide beam **only on the pids whose current path is longest**. Measured: 2^22 on
the top-100 longest gave **-330 moves in 5.5 h**, versus ~8 days for a full 2^22 run.
81/100 improved, **0 got worse**.

```bash
python cube444/scripts/05_solve.py --checkpoint <ckpt> --out rescue.csv \
  --pids "<comma list of the 100 longest pids in the current best>" \
  --beams 4194304 --max-steps 140 --sym-ensemble 4 --bf16
```

## ALWAYS judge by the per-pid MIN against the floor

Never by the standalone mean. Our 2^20 run scored **55,846 standalone against a 54,754
floor — worse** — and was worth ~1,300 moves in the merge, because it wins ~30% of deep
pids by exploring differently.

```bash
python cube444/scripts/06_merge.py --inputs run.csv floor.csv <others...> --out merged.csv
python cube444/scripts/73_verify_submission.py merged.csv   # POSITIONAL; expect VERDICT PASS
```

Merge sources by **content**: check the header AND that the move alphabet matches this
puzzle's generators. A sibling puzzle's file passes the header test and fails the second.

## Verify before quoting anything

`73_verify_submission.py <csv> [baseline.csv]` replays every path against `test.csv` and
takes POSITIONAL args (not `--submission`). Never quote a file's own reported total —
verify first, then quote. Report totals as `total / mean` over 1043 pids.

## Do not spend time on

Post-processing (every window <=10 is proven geodesic; the full rewriting ladder is -8),
the merge axis (the best public file dominates all 125 verified CSVs), two-phase with an
HTM phase-2 (floor 48.2 > 46.73), or corner PDBs (certified LB 9.7 vs real 46.7). The
live levers are a better scorer and a QTM-native phase-2.
