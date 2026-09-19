# Teacher-tube gate: post-mortem. The "0/96" was not what I reported.

2026-08-21. Corrects the verdict written into
`gdrive:artgor_meta/cube666_handoff_2026-08-20/HANDOFF.md` s4, which currently says
"TESTED TODAY, FAILED / decisively worse". **That is wrong and must be replaced by this
file when the gdrive mount is repaired.**

## What was reported

A 6-pid matched bridge gate returned **0 wins / 96 attempts at lambda = 0, 0.3 and 1.0**
against arm A's 5/90, and I concluded the teacher supervision made the model worse at its
deployed job.

## What is actually true

The 0/96 is not the model's capability. Four measurements, all on the SAME states:

| check | arm A | teacher (q666_tw) |
|---|---|---|
| min_a Q at true depth 5 / 10 / 20 / 30 | 3.9 / 9.0 / 19.9 / 30.9 | 3.9 / 9.0 / 19.2 / 28.3 |
| min_a Q at depth 40 / 60 / 80 | 43.1 / 68.9 / 72.2 | **32.2 / 35.0 / 34.9** |
| beam solve, depth 10/20/25/30, beam 2^16 | 4/4 each | 4/4 each, near-identical words |
| **173 IDENTICAL windows: solved** | **152** | **137** |
| **173 IDENTICAL windows: WINS** | **13** | **8** |
| predicted-save over those windows | mean 6.85, sd 7.70 | mean 7.32, sd 7.53 |

So: the Q head is healthy to depth 30, the beam is intact, the window-selection signal has
the same distribution, and on matched windows the model wins **8 vs 13 — 62% of arm A, not
0%**.

The 0/96 came from a 6-pid sample of 96 attempts. At the ~3.5% rate implied by 8/13 of arm
A's 5.6%, `0.965^96 = 3.2%` — an unlucky but unremarkable draw, amplified by the two models
ranking different windows into their top-16. **The gate was underpowered, and I read a
sampling artifact as a capability verdict.**

## The bigger error: the experiment is confounded

Two variables changed at once:

1. teacher-tube + on-path policy supervision (the thing under test);
2. **`k_max` 122 -> 40** for the random-walk source.

The k_max change alone compresses the Q ceiling from 72 to **35**. That is visible in the
table above and it demonstrably costs solve rate — 137 vs 152 on identical windows —
because the beam can no longer rank states it wanders into past depth 35, which is exactly
what happens when a residual search leaves the window's own depth band.

So the 8-vs-13 gap **cannot be attributed to the teacher supervision at all.** It is
consistent with being entirely a k_max artifact.

## Verdict

**Teacher-tube supervision is UNTESTED, not refuted.**

The missing control is the one the project's own rule 5 demands: **arm A fine-tuned with
`k_max=40` and NO teacher sources**, same 300k updates, same lr 1e-4. Run that first. If it
also lands near 8/173, the teacher supervision is neutral and k_max=40 is the whole effect.

Reproduce with `70_train_teacher.py`, setting `--tube-batch 0 --onpath-batch 0` for the
control arm (the loss terms drop out cleanly; verify the policy loss reads 0).

## Two things that ARE established

* **On-path imitation is pure memorisation.** Accuracy hit 0.977 on 364,107 fixed
  (state, move) pairs. Drop that source from any rerun; the tube (unbounded, resampled) is
  the part that cannot be memorised and it plateaued at 0.406 = 14.5x chance at mean
  remaining 217.
* **`k_max=40` is not free.** `ideas_codex` says do not MSE past mixing, and that is sound
  for the LABELS, but truncating the training range also truncates the scorer's usable
  range, and the bridge's beam needs headroom above the window grain. If a future arm uses
  k_max=40, expect a weaker window solver and measure it.

## Process failure to carry forward

I changed two variables in one arm and then reported a verdict from a 6-pid gate. Both
halves are covered by rules I had already written down (matched control; do not
reconfigure on n=2-5). The gate looked decisive because it was 0, and a zero is the most
seductive result to over-read — it is also exactly what a broken or underpowered harness
returns.
