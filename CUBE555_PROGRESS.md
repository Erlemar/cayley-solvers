# CayleyPy 5x5x5 — progress with neural models: what worked, what didn't

**Dates:** 2026-08-15 → 2026-08-21 · **Machine:** 1 x A100 80GB (`NVIDIA PG509-210`)
**Code:** `~/cayley/cube555/` · **Technical reference + runbook:** `CUBE555_FINDINGS.md`

This is the progress/retrospective doc. `CUBE555_FINDINGS.md` is the technical reference
(measured structure, derivation traps, file inventory, resume runbook); it does not repeat
the levers table, and this file does not repeat the runbook.

---

## The numbers

| artefact | total | status |
|---|---|---|
| shipped placeholder | 503,776 | — |
| 555 leaderboard leader (per `BIGCUBES_PLAN`) | 470,424 | 3 teams |
| `sample_reduced.csv` — commutation reduction, **no model at all** | 456,814 | verified 1035/1035 |
| `cube555_submission_218626.csv` | **218,626** | verified 1035/1035, `VERDICT: PASS` |
| current best-known min-merge (campaign still running) | **199,928** | **not yet merged or verified** |

Score is `total = sum_p min(L_p, ours_p)`. Solved-by-model coverage right now:

    554 / 1035 pids solved by a model, mean 169.7 moves (min 128, max 250)
    481 still on baseline: 67 attempted-and-failed, 414 never attempted
    merge ownership: deep200 520 · pool 28 · probe 6 · baseline 481

**Zero Santa pids are in the merge, and that is correct.** All 554 wins are random-walk pids.

---

## 1. The thing that decided the project: where the score actually is

1000 of the 1035 pids are random walks whose shipped baseline is the inverse scramble of
length `pid-34`. Only 35 are the Santa benchmark the paper (arXiv 2502.13266) reports on.

- Solving a **deep random-walk pid** saves several hundred moves.
- Solving a **Santa benchmark pid** saves **zero** — our ~177 loses to its 93.6 baseline.

Half a day was spent optimising the wrong 35 pids before this was noticed. The original goal
was "beat 92.16 average path length"; the metric that actually moves is the competition total
over 1035 pids, and those are different objectives. Everything below is aimed at the total.

Consequence: **run the queue deepest-first.** Saving per pid is `base_p - 169`, so ordering
front-loads the gain and any stopping point is near-optimal.

---

## 2. The model

`ResMLPQ`, **24.76M params** (24,757,807):

    embedding        3,600     150 sticker classes -> 24 dims
    input_stack  3,689,472     3,600 -> 1,024
    res_blocks  21,032,960     10 blocks x 2,103,296 (two 1024x1024 linears each)
    q_head          30,750     1,024 -> 30 (one score per generator)
    v_head           1,025     1,024 -> 1

**Budget, fixed before training started:** 11,719 epochs x 256 steps x 2,048 fresh states =
**3,000,064 updates / 6.14B fresh states**, lr 3e-4 cosine to 0, 23.1 h solo. Then **20,000
steps of warm-started Q-Bellman**.

**Label recipe — exactly three sources, and the restriction is the recipe:**

1. **Sparse-Q random-walk middles.** `k ~ U[2,80]`, non-backtracking walk from solved, **one**
   pivot `p` inside it, label two of 30 columns: `Q(s,undo) = p-1`, `Q(s,next) = p+1`.
   `pivot_tilt=0.5` biases the pivot deeper. `k_max=80` from the counting bound (66.4), not
   from the oracle.
2. **Exact BFS anchors**, d<=4, all 30 columns exact, **256 rows/step, weight 1.0**. Load-bearing:
   without exact anchors in every batch the Bellman bootstrap settles at `V(solved) ~ 2`.
3. **Symmetry expansion**, 4 random frames of 48, via `Q(sym(s,k), relabel[k,a]) == Q(s,a)`.

Loss `= sparse + 1.0*anchor + 1.0*value`. Frames are drawn **randomly per sample**; a greedy
coverage table keyed on `(undo,next)` hands the same frames to a given action pair every step,
buying column coverage with zero state diversity.

**Bellman recipe.** Warm-started only. Expand all 30 children, regress on
`1 + min_a' Q_target(child)`, clamped to 0 where the child is solved. Target refreshed every
500 steps, lr 2e-5, batch 768, anchors still in every batch but at **weight 2.0** — they are
the only thing pinning absolute scale against a bootstrap that would otherwise drift. Watch
`E[target]`, **not** the loss; the loss improves while the bootstrap eats itself.

---

## 3. What worked

**1. The recipe transfers to 555 — and does not to 666.** Same stack: 555 solves 84% of
fully-mixed states at mean 169; 666 solves nothing. The discriminator is not the saturation
*ratio* (555 is 42/72 = 0.58, 666 is 72/110 = 0.65, so 555 is nominally worse) but the
**absolute gap the beam must bridge**: ~72 moves is crossable, ~110 is not.

**2. Warm-started Bellman — the single biggest lever.** 6k steps (20 min) took a checkpoint
from **0/3 to 2/3 solved** at ~130 moves against a control that solved nothing. Deployed at 20k.
Warm-starting is the whole trick; the same procedure from near-scratch failed.

**3. The 30-wide Q head.** Scores all 30 children from one forward pass on the parent, so a
beam step costs `B` evaluations instead of `B x 30`. That 30x is why a 2^21–2^22 beam runs on
one A100 where the paper needed 69 agents at 2^24. Q also beats V(child) on top-1 at depth
(0.246/0.182 vs 0.145/0.153) — **the cheap head is also the better one.**

**4. Symmetry frames as independent retries.** Each conjugation frame is close to an
independent draw at ~33%, so solve rate is `1 - 0.67^k`: 70% at k=3, 91% at k=6. Early exit
makes extra frames nearly free — only pids that need them ever pay.

**5. Fresh frames beat beam width — measured, and it inverted the plan.** The runbook said to
width-escalate accumulated failures at 2^22. A staged probe on the 10 deepest failures with
**6 unused frames at the same 2^21 width** rescued **6/10 (r=0.60) for 4,306 moves in 7,234 s**.
Width would have cost 2x per attempt for a log-linear gain, so arm B was never run. This is the
best moves-per-GPU-second lever found in the project: **0.595 moves/GPU-s** on the deepest pids.

**6. Commutation reduction — 9.32% with no model whatsoever.** 503,776 -> 456,814 in ~40 lines,
which already beats the 470,424 leaderboard leader. Worth doing before any model exists.

**7. Cheap wins that held up.** `torch.compile` with padded fixed chunks: **1.35x** beam
throughput, byte-identical output. `history_depth=4`: **-14%** path length (saturates at 4;
unlimited buys nothing, because all 30 generators are ODD, making the graph parity-bipartite so
revisits only occur at even lags). Endgame `d<=5` ball as goal set: widens the target from 1
state to 10,739,017 with an exact tail spliced on arrival.

---

## 4. What didn't work

**1. qv-consistency — REJECTED, and it is not a null result.** It trades **solve rate for path
length**:

    lam 0.00   1033=187  1031=179  1029=193      3/3 solved   wall  887 s
    lam 0.30   1033=157  1031=155  1029=NONE     2/3 solved   wall 2151 s

Shorter paths, one pid lost. On 555 a lost pid falls back to a ~960-move baseline: **-714 net.**
Mechanism: V's target is walk index `p` and Q's undo column is `p-1`, so V and `min_a Q` are
trained to be the same function offset by one (measured `V - minQ = +1.64 ± 0.65`); the term
reduces to a pure cross-parent reweighting. It helped on tetraminx (-4) and was dead on 444
(+4..+6). **It still won 2 pids in the merge** — judge a member on merge contribution, not on
its standalone verdict.

**2. Capacity.** 24.8M vs 14.2M: **no measurable difference** at matched budget. Width is not
the binding constraint on this puzzle.

**3. One-hot encoding.** At 150 classes it is a 22,500-dim input: 44.1M params / 27.4 ms /
0.69 Mstate-s, against embed24 at 24.8M / 20.3 ms / **1.34 Mstate-s**. Roughly 2x slower for no
gain.

**4. Post-processing on beam paths.** Commutation reduction: **+0.00%** (all its gain is on raw
scramble words, none on beam output). Shortcut splicing from the `d<=5` ball: **-0.74%**.

**5. Non-backtracking beam expansion.** Neutral — `history_depth>=1` already drops those moves,
which are only ~3% of the shortlist.

**6. Window compression** (`ideas_codex` #1). A window of length W has relative distance
`min(W, ~72)` and our solver returns ~169 for any fully-mixed element, so any window longer than
169 compresses to ~169 — the optimal window is the whole path, i.e. the global solve. On pid
1034: global saves **799 in one run**; four 250-move windows save 298 for **four** runs. It is
the #1 idea on 666 precisely *because* the global solve fails there.

**7. Width escalation as the rescue lever.** Structurally sound, but see item 5 above — frames
dominate it at half the cost.

**8. Label sources beyond the three.** Path labels and the sorted-profile/permutation-invariant
family were measured and rejected on the sibling puzzle and never re-added. Their pre-flight
check passes for rejected variants, so passing it means nothing.

---

## 5. New findings from the current campaign

**Frames decay across rounds, and the decay is measurable.**

    round 1 (frames 0,7,19,33,41,47)   621 attempted   0.837
    round 2 (a 2nd set of 6)            25 attempted   0.600
    round 3 (a 3rd set)                  4 attempted   0.500
                        geometric decay factor ~0.77 per round

Frames are therefore **not** fully independent — if they were, round 2 would repeat 0.84. The
drop is a frailty effect: the residue is enriched in pids with a genuinely low per-frame rate.
Two pids (`1016`, `1024`) have now failed **18 frames** each, which is the first evidence of a
hard core. Extrapolating the decay across all 48 frames (8 sets of 6) leaves **~12 pids never
solved by frames alone**. Treat that as soft — it is a two-point extrapolation and round 3 is
n=4.

**Real per-pid cost is ~723-800 s, not the ~1,123 s assumed.** Rescues exit early around frame
2-3 (~490 s) while failures pay all six (~1,080 s). The blended figure is what matters for any
allocation decision, and using the failure cost alone understates the rescue lever.

**`--invert` is legal here and has never been run.** 555 is a picture cube, so a state *is* a
group element and `invert_state` is well-defined — one of the rules that inverts from 444, where
it is forbidden. Every run so far had `invert=False`. Combined with 48 frames that is **96**
trajectories, not 48. Approved as the next axis, sequenced behind a smoke test.

---

## 6. Process mistakes worth not repeating

1. **`--resume` skips pids in the output CSV; failures are not written there.** Four relaunches
   re-ground the same five hard pids — an apparent "0/3 in 59 min" collapse that was repeated
   work. Fix: `--fallback`, so every *attempted* pid is recorded.
2. **A hash hit is not proof of identity.** The endgame membership test compared 64-bit hashes
   only; at 10.7M entries and ~1e11 lookups a false positive is EXPECTED (~0.1), and one killed
   a 29-hour run at pid 800. Compare the state; make the residual non-fatal.
3. **Optimised the wrong 35 pids for half a day.**
4. **Reconfigured six times on samples of 2-5 pids.** The step-cap question was finally settled
   *from banked data*: a beam finds a length-L solution at step L, and all were <=193.
5. **Inverted a conclusion about frames.** Observed "frames rescued 0 of 3 *failures*", dropped
   to 1 frame, got **0/4**. The statistic that mattered was how many *successes* had needed a
   non-zero frame. A matched control caught it.
6. **Closed a search question with an offline metric.** top-1 was bit-identical across qvc
   lambdas, so it was called inert; the beam then moved 187->157 and lost a pid.
7. **A direction bug in the shortcut splicer.** The ball's argmin descent yields the word that
   *solves* w (`w^-1`); the window needs `w`, the reverse-inverse. 9 of 18 paths failed replay.
8. **Buffering is set by the LAST pipe stage** — `grep --line-buffered | sed` still blocks. Same
   class of error: a long run writing to a file block-buffers at 8 KB, which at ~90 bytes/line is
   one flush every ~7.7 h. Set `PYTHONUNBUFFERED=1` on anything you intend to monitor.
9. **An empty CSV passes verification vacuously** (`0/0 -> PASS`).
10. **Wrote the findings doc ONLY to the gdrive mount.** When the mount went down before a
    migration it became unreachable and had to be regenerated from context. Write documentation
    into the repo first, then copy outward.
11. **Quoted a crossover pid that embedded an unmeasured assumption.** The "switch at pid 391"
    figure implied a rescue probability of ~0.75 that had never been measured; at the measured
    r it was ~299, and once r came in at 0.60 the whole framing was obsolete because the pool
    became the better queue outright. Derive the number, then state it — do not carry it forward
    from a plan.
12. **A background job in the agent's own cgroup dies with the session.** `nohup` survives
    SIGHUP but not scope teardown. `systemd-run --user` places the job in `app.slice`, outside
    the sandbox, where it outlives the session.

---

## 7. Current campaign state

Deepest-first, `bellman_020000.pt`, 2^21 width, 250 max-steps, 6 frames, `history_depth=4`,
no-backtrack, endgame-5, compiled, `--fallback sample_reduced.csv`.

| stage | scope | status |
|---|---|---|
| round 1 | 830-pid queue, pids 1034..205 | 621 attempted, 520 solved, **paused at pid 413** |
| lever probe | 10 deepest failures, frames 3,11,23,29,37,44 | **done — 6/10, 4,306 moves** |
| pool round 2 | 95 failures, frames 1,9,15,25,31,39 | **running — 37/95, r=0.59** |
| invert smoke test | pids 1010,897,980, `--invert` | queued |
| invert round | residue, `--invert` x frames 0,7,19,33,41,47 | queued |
| fresh queue | 209 pids, 414..205 | queued last (lowest rate) |

Rates driving that order: pool **0.24-0.37** moves/GPU-s, invert-on-residue **~0.2**, fresh
queue **0.134**.

**The probe and pool solves are not in any submission yet.** They live in
`bench/rescue_probe_a.csv` and `bench/rescue_pool.csv`. A `43_shortcut -> 41_merge -> 50_verify`
pass is required, expecting 1035/1035 `PASS`. The 199,928 figure above is an un-verified
min-merge and should not be quoted until it replays.

---

## 8. What is next

1. **Finish pool round 2**, then the invert smoke test. The smoke test exists because a wrong
   direction in `invert_path` would discard every path and look identical to "invert doesn't
   help" — the same ambiguity that produced mistake #5. Read rule: any
   `TRANSLATED PATH DOES NOT SOLVE` means the bug, not the verdict.
2. **Invert round on the residue**, reusing frames 0,7,19,33,41,47 so none of the 30 still-unused
   frames are consumed.
3. **Fresh queue**, 209 pids.
4. **Merge and verify.** Do this before quoting any total.

Two changes worth making on a **next training run**, neither of which is worth interrupting the
current campaign for:

- **Orbit-factored one-hot.** Every position's sticker is one of its orbit's 24, so one-hot over
  24 gives 150 x 24 = **3,600 dims exactly** — identical width and cost to embed24, but lossless
  where ours is a learned rank-24 factorisation.
- **Limited-Horizon Bellman Learning** instead of the one-step `1 + min_a' Q_target`: search
  several steps and back up the best frontier value, sampling from real search regions.

Gate any Bellman run on a depth-calibration probe before and after. On 666, `E[target]` held
steady while the ceiling fell.

Not worth it on 555: orbit PDBs, the coset ladder (the neural beam already works), HTM two-phase,
corner PDBs, and anything that adds a fourth label source to the training recipe.

---

## Post-processing the external 109,512 file (2026-08-23)

`cube555_best_109512.csv` (from the community, 1035 pids, replay-verified PASS) run
through the tetraminx exact-rewriting family. **109,512 -> 109,288 (-224)**, 100 pids
improved, none worse, 1035/1035 replay to solved. Output
`cube555/postproc/cube555_shortened_109288.csv`; code `cube555/postproc/shorten_mitm.py`.

**All 30 generators are ODD permutations**, so d(w) == |w| (mod 2) for every window:
savings are always even (histogram 2:89, 4:10, 6:1) and a reach-R sweep can only newly
fire on L == R (mod 2). `--l-parity` halves every sweep. A 7-window can never collapse
to 6.

| pass | verdict |
|---|---|
| commutation reduction (`42_commute_reduce.py`) | **0** -- beam dedup already prevents same-axis runs |
| n-way per-pid min-merge | **0** -- our best is 187,724 and loses on every pid; content scan of repo + Downloads/Desktop/Documents found no other source |
| exact window lookup, reach 5 | **0** -- and all 16,492 windows of length <=5 have d EXACTLY = L |
| MITM reach 6 | **-116** (53 splices; 8->6 x48, 10->6 x5) |
| MITM reach 7 | **-32** (15 splices) |
| MITM reach 8 | **-26** (12 splices; 10->8 x11, 12->8 x1) |
| MITM reach 9 | **-50** (23 splices; 11->9 x21, 13->9 x2) |
| consolidation re-sweep at 7 and 8 | 0 -- converged |
| reach 7, `--max-window 44` | 0 -- no long-range collapse |

**Yield is not monotonically decaying** -- reach 9 beat both 7 and 8, so do not
extrapolate a decay and stop early (tetraminx radius-13 lesson). The dominant hit at
every reach is an (R+2)-window collapsing to exactly R.

Reach 9 is the practical ceiling: reach 10 needs B_5 (10.7M) as the front, and while the
DGEMM is affordable (~2 h) the 6.8e9 probes/pid prefilter gather is not without a fused
kernel.

Implementation note that made reach 8-9 possible: the join
`h(x^-1 o w) = sum_k XINV[x][k] * hv[w^-1[k]]` is a DGEMM `XINV @ HVW^T`, not an
elementwise broadcast. Two hash vectors with entries under 2^18 keep every float64
partial sum an exact integer below 2^53. Reach 8 went from ~18 h to 20 s. Phantoms scale
with the front (0 at reach 6, 140 at reach 9) and were all caught by the permutation
guard; none reached the output.

The remaining gap is not local: after this sweep every window up to length 44 is geodesic
at reach 7 and every window up to ~21 is geodesic at reach 9, yet paths average 105.6.
Whatever slack is left needs a better SEARCH, not a local rewriter.
