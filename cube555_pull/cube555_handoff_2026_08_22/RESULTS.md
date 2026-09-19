# Results — the measured numbers, 2026-08-15 → 2026-08-22

Companion to `README.md` (how to run) and `APPROACH.md` (the method).

---

## 1. Where the submission stands

    shipped placeholder                              503,776
    leaderboard leader (3 teams)                     470,424
    sample_reduced.csv -- commutation, NO model      456,814
    previous verified submission                     218,626
    THIS HANDOFF (verified 1035/1035)                187,780
    current top-1 solution for 555                   117,000
    perfect play (counting bound 66.4)                ~69,000

Composition of the 187,780:

    599 pids model-solved      105,912 moves    (589 genuine beam solves, mean 167.8)
    436 pids on baseline        81,868 moves
    0 Santa pids in the merge -- correct; our ~177 loses to their 93.6

**The gap to the leader is 70,780 and it splits almost exactly in half:** ~35,400 from
coverage (pids still on baseline) and ~35,400 from length (solving them shorter). Solving
*everything* at our old ~169 tops out at **152,383** — still 35k short. Length was the half
nobody was working on.

## 2. THE MAIN FINDING — the deployed checkpoint is badly over-refined

24 disjoint, unbiased pids. `--frames 0`, 2^21, `--max-steps 300`. **Only the checkpoint
varies.**

| Bellman steps | solved | mean length | paired vs 20k |
|---|---|---|---|
| **2,000** | **18/24** | **130.1** | n=7, **−48.6**, 7/7 |
| 4,000 | 9/24 | 141.4 | n=5, −34.8, 5/5 |
| 6,000 | 11/24 | 148.7 | n=4, −31.0, 4/4 |
| 6,000 (replicate) | 12/24 | 140.9 | n=6, −28.7, 5/6 |
| 8,000 | 9/24 | 143.9 | n=4, −24.5, 4/4 |
| 20,000 (**deployed**) | 10/24 | 174.5 | — |

**Every refinement length beats the deployed one, all with the same sign.** Five independent
arms agreeing is much stronger than any single comparison.

- **2k is the best arm**: 18/24 coverage vs 10/24, and −48.6 moves paired on 7/7.
- **The middle of the curve is unresolvable.** 4k/6k/8k land at 9/11/9 solves and 141-149
  mean, while two *replicates of 6k* span 11-12 solves and 140.9-148.7.
- **Min-merge over all six arms: 24/24 solved, mean 132.5.** Every pid falls to at least one
  checkpoint, including one nothing had solved before. The deployed 20k owns **1**.

Every path in the shipped 187,780 came from the worst of the six.

## 3. The noise floor — training is non-deterministic

Two 6k runs with identical init, seed (555) and hyperparameters — the step-0 log lines are
byte-identical — produced **different weights**:

    tensors differing   : 87/89
    elements differing  : 23,524,899 of 24.8M
    max abs weight diff : 5.5e-05

Consequence on the same 24 pids:

    overlap n=8, identical on 0/8, mean |diff| 10.5 moves, sd 11.7, range -10..+28
    disagreed on solvability for 7 of 24 pids

**±10.5 moves per pid and ±1-7 pids of coverage is the floor** for any single-run A/B at
this sample size. The beam itself is fully deterministic given a checkpoint; training is not.
This retroactively softens several deltas: 2k-vs-6k (−18.9) and 6k-vs-20k (−28.7) are near
or inside it, while 2k-vs-20k (−48.6) is several times it and survives comfortably.

## 4. Search-side levers, all closed

**Width — wider is worse.** Same pids, same frame, only width varying:

    solved at frame 0:   2^20 = 2/6    2^21 = 6/6    2^22 = 3/6    2^23 = 0/1

No OOM anywhere; 2^23 ran fine. A wider beam admits candidates the scorer cannot rank, and
they crowd out the good ones in a global top-B. Matched lengths (n=2) decided nothing; the
coverage collapse is the signal.

**qv-consistency — not a length lever.** An earlier n=2 observation suggested −15% length.
On 5 matched pids at λ=0.30: **+8.0 mean, 1 win in 5, and it drops a pid.** Rejected, now for
a second independent reason.

## 5. Model diagnostics

**The model is exact where labels are exact, and pinned everywhere else.**

    exact anchors   d=2  true 1.000  pred 0.996   err -0.004
                    d=3  true 2.000  pred 2.001   err +0.001
                    d=4  true 3.000  pred 2.996   err -0.004

    real test states, pid 200-1035: pred ~44.0-44.2 flat, global max 49.25
    true distance for such states is ~65-70  ->  underestimate ~23 moves

This is **not** a capacity or representation failure — a model that cannot represent depth
does not nail d<=4 to ±0.004. Two mechanisms were proposed and both refuted:

- **min-bias in the backup**: measured **+0.138 moves**. Frame-to-frame noise is only 0.413
  sd, far too small for a min over 30 to bias anything. Measured without ground truth by
  exploiting that the 48 symmetry frames are exact-equal views of one state.
- **unfinished value iteration**: saturation is a **converged fixed point** at 43-44
  regardless of 12/20/40 target refreshes, not a propagation front.

**Compression is benign.** Bellman *increases* it (pretrained 50.75 -> 20k 44.19, further
from the true ~65-70) while being the biggest single win in the project. Absolute calibration
is not what the beam needs. This matches the sibling puzzle's recorded rule: *gate on
discrimination, never absolute scale.*

**Discrimination at depth degrades with refinement** — top-1 on replayed paths, chance 0.033:

    remaining      pretrained   6k      10k     20k
       11-20         0.371     0.380   0.360   0.400
       41-50         0.283     0.277   0.274   0.240
       61-70         0.246     0.228   0.225   0.217

Deep discrimination buys **coverage**; shallow sharpness buys **length**. The checkpoints are
complementary, which is why a merge beats any single one.

**Capacity may have been killed wrongly.** "24.8M vs 14.2M, no difference" was measured the
same way that, on the 4x4x4 sibling, killed capacity on a 15-pid beam delta inside the noise
floor — and on re-audit there capacity was the *best* of six levers at +8.3% deep top-1.
Untested here: `q555_b_bell` (14.2M + 20k Bellman) has never been benched.

## 6. Projection — what a re-solve is worth

Re-solving all 1035 pids with a model returning mean `L`, keeping baseline where shorter
(`total = sum_p min(baseline_p, L)`):

| mean L | total | vs leader |
|---|---|---|
| 120 | 115,386 | **−1,614** |
| 125 | 119,716 | +2,716 |
| **130** (2k measured) | **124,019** | +7,019 |
| 140 | 132,568 | +15,568 |
| 150 | 140,994 | +23,994 |
| 170 (roughly today) | 157,481 | +40,481 |

**Coverage has stopped mattering.** At mean 130: 100% coverage gives 124,019, 95% gives
124,854 — missing 5% of pids costs 835 moves. The 2k checkpoint solves 75% at a *single*
frame, so six frames is ~99.9%.

**Everything now rides on mean length, and the target is ~120.** Also note 176 pids have
baselines <=130, totalling 12,349 — an irreducible floor without near-optimal play.

Honest range: the 130.1 comes from pids that solved at **frame 0**; pids needing frames 3-6
are plausibly harder and longer, so the full-set mean is likely **130-145**, i.e. a landing
zone of **124k-137k**. A large improvement on 187,780, probably not first place.

## 7. Open leads, in order of expected value

1. **Re-solve everything with a 2k-family merge.** No training. Roughly 3 days of GPU at
   ~180 s/pid/arm. Worth ~50-60k moves on the projection above.
2. **Sub-2k checkpoints.** The curve is monotone decreasing all the way to 2k, which is the
   *lowest* point measured, so the optimum has not been bracketed from below. The pretrained
   parent (0 steps) is clearly worse, so a peak exists in 0 < n < 2000. ~7 min of training
   per point. *(A sweep at 250/500/1000/2000 was running when this handoff was built —
   check `cube555/logs/subsweep.log` on the source machine.)*
3. **Best-of-frames instead of first-of-frames.** `30_solve.py` returns the first frame that
   succeeds. Taking the shortest of six is strictly better and has never been measured.
4. **Multi-checkpoint min-merge**, already measured at 24/24 vs 18/24 for the best single arm.
5. **Bench `q555_b_bell`** (14.2M + 20k) and re-test capacity properly.

Not worth it: wider beams, qv-consistency, more Bellman steps, orbit PDBs, the coset ladder,
HTM two-phase, corner PDBs, and any fourth training label source.
