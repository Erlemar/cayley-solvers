# CayleyPy 5x5x5 — self-contained beam-search handoff

**Built 2026-08-22.** Everything needed to run the solver on a fresh machine, plus the
reasoning behind the configuration. Read `APPROACH.md` for the method and `RESULTS.md` for
the measured numbers.

Competition scoring: `total = sum_p min(L_p, ours_p)` over 1035 pids. Lower is better.

| | total |
|---|---|
| shipped placeholder | 503,776 |
| leaderboard leader (3 teams) | 470,424 |
| commutation reduction, **no model** | 456,814 |
| **`cube555_submission_187780.csv` (in here, verified)** | **187,780** |
| current top-1 solution for 555 | 117,000 |
| perfect play (counting bound 66.4) | ~69,000 |

---

## 1. Layout — run everything from THIS directory

Scripts resolve `PROJECT = Path(__file__).resolve().parents[1]` (= `cube555/`) and import
from both `cube555/src` and `../src`. **The tree below is load-bearing; do not flatten it.**

    .
    +- src/cayley/              shared solver package (KhoruzhiiSolver, ResMLPDistance)
    +- cube555/
       +- src/cube555/          puzzle package (Cube555, ResMLPQ, load_model)
       +- scripts/              all entry points
       +- data/                 puzzle_info.json, test.csv, sym_*.npy  (see s2)
       +- models/               4 checkpoints
       +- bench/                sample_reduced.csv (the no-model baseline)
       +- submissions/          the verified 187,780 file

## 2. FIRST STEP: rebuild the endgame table

`--endgame-depth 5` needs `cube555/data/anchors_d5.pt`. **It is deliberately NOT shipped**
— it is 1.9 GB and rebuilds in about 10 seconds:

    python cube555/scripts/11_build_anchors.py --depth 5

Do the same with `--depth 4` (80 MB) if you intend to retrain; `22_bellman.py` needs it.

Without this the solver still runs, but you must pass `--endgame-depth 0`, and you will
lose the exact tail splice that widens the goal from 1 state to 10,739,017.

## 3. Environment

    Python 3.12 · torch 2.6.0+cu124 · numpy 2.5.1 · one CUDA GPU

Peak GPU memory at beam width 2^21 is ~27 GB. Memory is dominated by the endgame ball and
the chunked workspace, not the beam itself, so 2^22 also fits in 27 GB — but see
`RESULTS.md`, wider is **worse** here.

## 4. Run the solver

    python cube555/scripts/30_solve.py \
        --checkpoint cube555/models/q555_2k_BEST.pt \
        --pids 1034,1033,1032 \
        --beams 2097152 --max-steps 300 --bf16 --compile \
        --history-depth 4 --no-backtrack --endgame-depth 5 \
        --frames 0,7,19,33,41,47 \
        --internal-batch-size 262144 \
        --fallback cube555/bench/sample_reduced.csv \
        --out cube555/bench/my_run.csv

Every flag above is a measured choice; `APPROACH.md` s4 says why. Notes:

- **`--fallback` matters.** `--resume` skips pids already present in the output CSV, and
  failures are not written there. Without `--fallback`, a relaunch re-grinds the same hard
  pids forever. With it, every *attempted* pid is recorded.
- **`--beams` escalates**, it does not sweep. A comma-separated list is tried in order and
  breaks on the first width that solves. To compare widths, run separate invocations.
- **`--frames` takes the first frame that succeeds**, not the best. Each of the 48 symmetry
  frames is a near-independent ~33-50% draw, so 6 frames gives high coverage cheaply via
  early exit. Taking the *shortest* of 6 rather than the first is untested and is the most
  obvious cheap win left.
- `--invert` is legal on this puzzle (555 is a picture cube) and verified working. It gives
  a second axis: 48 frames x 2 directions = 96 trajectories.

## 5. Verify, always

    python cube555/scripts/50_verify.py <submission.csv>     # expect 1035/1035, VERDICT : PASS

Never trust a run's self-reported total. Note an **empty CSV passes vacuously** (0/0 -> PASS).

## 6. Merge several runs

    python cube555/scripts/41_merge.py out.csv a.csv b.csv c.csv

Takes the per-pid minimum and **replays every path against `test.csv` before writing**, so a
path from a run with a translation bug cannot enter a submission. Judge any model or config
on its *contribution to the merge*, never on its standalone total — a checkpoint with a much
worse standalone mean has repeatedly owned pids nothing else could solve.

## 7. Retrain

    python cube555/scripts/20_train.py       # pretrain from scratch (~23 h)
    python cube555/scripts/22_bellman.py \
        --init cube555/models/q555_pretrained.pt \
        --output cube555/models/my_bell --steps 2000 --save-every 250

**Use ~2,000 Bellman steps, not 20,000.** This is the single most important finding in the
handoff — see `RESULTS.md` s2. Watch `E[target]`, not the loss: `Q == 0` is also a fixed
point of the recursion, so a falling loss is consistent with total collapse.

## 8. Checkpoints

| file | Bellman steps | note |
|---|---|---|
| `q555_2k_BEST.pt` | 2,000 | **best measured** — 18/24 solved, mean 130.1 |
| `q555_6k.pt` | 6,000 | 12/24, mean 140.9 |
| `q555_20k_deployed.pt` | 20,000 | worst of six; produced the shipped 187,780 |
| `q555_pretrained.pt` | 0 | the parent all three were warm-started from |

All four are the same architecture (`ResMLPQ`, 24.76M) and share the pretrained parent.
Keep the older ones: they earn their place in a merge even where they lose standalone.

## 9. Scripts

**Derivation** `10_derive_symmetry` · `11_build_anchors` · `12_mitm_oracle`
**Training** `20_train` · `22_bellman` · `31_eval_path`
**Search** `30_solve` · `32_gate_deep.sh` · `33_isocost_sweep.sh`
**Submission** `40_bench` · `41_merge` · `42_commute_reduce` · `43_shortcut` · `50_verify`
**2026-08-22 experiments** `34_width_length.sh` · `35_qvc_length.sh` · `36_phase0_diag.py`
· `37_bellman_curve.sh` · `38_confirm_bell12.sh` · `39_bellman_sweep.sh` · `40_subsweep.sh`

## 10. Verified on build

    imports resolve from this root                 OK
    q555_2k_BEST.pt loads, 24.8M params            OK  (bellman_step=2000)
    50_verify on the shipped submission            1035/1035, VERDICT : PASS

The beam itself was **not** smoke-tested from this directory, because the source machine's
GPU was busy. Before trusting a long run, build the anchors and solve 2-3 pids.
