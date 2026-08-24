# cube666 — handoff, 2026-08-20

Written for a fresh OD. Everything in this folder is what does not survive the migration.

---

## 1. State in one line

**Submission went 463,203 -> 362,371 (-21.8%), replay-verified 1012/1012** (0 invalid,
0 missing, mean 358.6, median 447, max 562).

The file is `FINAL_submission.csv` in this folder, also
`../cube666_submission_362371_2026-08-21.csv`, md5 `48abf12b37361e83317363142e612fe2`,
verified both as written AND as read back from gdrive.

The bridge pass that was mid-flight at first writing COMPLETED: -1,180 over 400 pids
(256 wins / 8,649 attempts), included above.

---

## 2. What is in this folder

| file | contents |
|---|---|
| `code_and_state.tgz` | `cube666/{scripts,src,configs,submissions}` + all top-level planning docs |
| `memory.tgz` | `~/.claude/projects/-home-artgor/memory/` — **local dir, NOT a gdrive symlink, would be lost** |
| `pathkit.zip` | the post-processing toolkit (also only in `~/Downloads`) |
| `models/q666_a_final_ARMA_deployed.pt` | 317 MB. **The deployed model.** v1 ResMLPQ, 26.4M, 3M updates. Everything using a model uses this one |
| `models/rung_o{5,6}.pt` | 49 MB each, orbit-projected rung models (see s5) |

NOT copied, deliberately: `cube666/data` (6.1 GB, rebuildable — `10_derive_symmetry.py`,
`11_build_anchors.py`, `32_build_word_ball.py`) and 58 GB of other checkpoints (arms
C/D/E/t and intermediate arm-A epochs). Only arm A's final matters.

Restore with:
```bash
mkdir -p ~/cayley && tar xzf code_and_state.tgz -C ~/cayley
tar xzf memory.tgz -C ~/.claude/projects/-home-artgor/     # check the path exists first
```

---

## 3. How the -100,832 was earned

| stage | total | delta |
|---|---|---|
| `reduced.csv` at session start | 463,203 | -- |
| **external solver CSV, n-way merged** | 374,131 | **-89,072** |
| ball window sweep r=3 | 371,817 | -2,314 |
| exact d5 word-ball rewrite | 371,791 | -26 |
| **window r=4, segmented** | 364,155 | **-7,636** |
| neural bridge, 400 longest pids | 362,975 | **-1,180** |
| final n-way merge over 17 files | 362,927 | -48 |
| pooled cross-trajectory bridge (2026-08-21) | **362,371** | **-556** |

**The single biggest lever was a solver file the user supplied, not anything we built.**
Our models contributed ~1.9% (the neural bridge, -1,956 gross over 8 passes, and nothing
else -- no model solves a pid); that file contributed 19%. If a classical solver is
obtainable, it dominates every model and post-processing lever. Ask for it first.

### The loose/tight rule — read before running any rewriter

`reduced.csv` was a commutation-reduced RANDOM WORD: locally near-geodesic, all slack
GLOBAL. Solver output carries algorithmic setup/undo redundancy. The same sweep on the two
files:

| | reduced random word | solver output |
|---|---|---|
| window r=3 | -28, 14 splices | **-2,314, 786 splices** |
| suffix diagnostic beat-k | 0/30 | **6/20** |
| neural bridge win rate | 0.8% | 4.5% |

So: **run pathkit's suffix diagnostic first** (minutes). beat-k 0 means the slack is global
and local rewriting is finished on that file; nonzero means the whole stack pays.

---

## 4. The model line is CLOSED. Do not re-run these.

Every 666 full-puzzle model is **at chance from depth ~85**; beam horizon 30-35 against
pids of 95-939 moves, so no model solves a single pid. Six independent levers, all measured:

| lever | result |
|---|---|
| label distribution (flat vs tilt, 6 paired checkpoints, n=8192) | flat WORSE at every depth |
| capacity 26.4M -> 119.9M | r85/r100/r122 unchanged |
| architecture v1 -> v2 (dueling, orbit one-hot, s^-1, pre-norm SwiGLU) | moves depth <=40 only |
| training time (3M updates; arm E 10%->20%) | delta at r72 = -0.02 |
| beam width 2^16 -> 2^20 | length 40: 0/4, 0/4, 0/4 |
| Bellman, FAITHFUL 20k steps (2026-08-21) | beam wins 13 -> 7 -> 6 -> **1**, monotone |

Bellman cannot work in principle here: it needs a target net that already discriminates at
depth, and none of ours does.

**Arm A and arm E are statistically TIED on the beam** (L40 1/24 vs 3/24, p=0.61) despite
arm E being 4.5x the parameters — which is why arm A is the deployed model.

### Teacher-path training — CLOSED, but the reasoning below was wrong twice

> **SUPERSEDED 2026-08-21.** The verdict here ("decisively worse", from a 6-pid gate) was an
> underpowered read AND confounded with a simultaneous `k_max` 122->40 change. It was then
> re-run properly and the line closed for a completely different reason. Full account in
> `../cube666_model_training_report_2026-08-21.md` s6 and `CUBE666_TEACHER_POSTMORTEM.md`.
> The short version:
>
> * The `q666_tw` policy hit **0.977** on-path accuracy and appeared to solve **8/24 pids**
>   at beam width 64. That was **MEMORISATION** — 4 of the 8 came in at exactly teacher
>   length, pid 511 reproduced its 420-move path precisely. Worth zero score.
> * The `cube444_a100_handoff` archive prescribes an **8:1 rw:policy batch ratio** and names
>   the failure outright: *"a policy at 99% top-1 is a memorisation warning."* Mine ran at
>   1.3:1. Retrained at 7.8:1 (`q666_az`, `73_train_az.py`), accuracy plateaued at **0.554**
>   and it solved **0/24**. Positive control reproduced 8/24, so the harness is sound.
> * At the DEPLOYMENT width the AZ trunk gave no Q-head gain either: **0/3 pids at both 2^18
>   and 2^20**, identical to arm A, with and without a policy blend.
>
> Both halves are now closed, properly configured. The reason is structural: teacher lengths
> are upper bounds (`d*(s) <= U(s)`; 562 moves over a distance-110 state is 5.1x wrong), and
> a 364k-state corridor is measure-zero in a group of size 3.14e149.

The original idea: k-step walks off verified-solution states give truthful ACTION labels at
arbitrary depth (only the k-step walk must be truthful, not the 300 moves underneath).
Trained a policy head warm-started from arm A, 300k updates, `k_max=40`.

Matched gate, same 6 pids / windows / seed:

| model | wins / attempts | moves |
|---|---|---|
| **arm A (incumbent)** | **5 / 90 (5.6%)** | **-34** |
| teacher, lambda=0 | 0 / 96 | 0 |
| teacher, lambda=0.3 | 0 / 96 | 0 |
| teacher, lambda=1.0 | 0 / 96 | 0 |

Read this table as a **sampling artifact plus a k_max confound**, not a capability verdict:
on 173 IDENTICAL windows the same model won 8 vs arm A's 13, i.e. 62% of arm A rather than
0%, and `k_max=40` alone compresses the Q ceiling 72 -> 35 and costs solve rate (137 vs 152).
Code is in `70_train_teacher.py` + `src/cube666/teacher.py`; the "tube" in there is my own
invention and appears in no guidance doc — use `73_train_az.py` instead.

---

## 5. Rung models — work at k=1, cap there

A 12M model on the orbit-PROJECTED puzzle solves **a uniformly random 24-slot centre orbit
24/24 in 18.8-22.3 moves at beam 2^16**, and width is a lever again (2^11 -> 2^16 = 42% ->
100%) where on the full puzzle 16x width did nothing.

Composition kills it: 2 orbits 31-33%, 3 orbits 0%. Failures all read `kept 24/24/0` — the
beam sits in the stabiliser of the solved rungs. Not fixed by step budget, `solved_weight`
(0.15/0.35/1.0), width, or a joint composite model on the 48-slot projection.

Why: 1 orbit = 79 bits, mixes 30-35 (inside the envelope). 2 orbits = 158 bits, mixes
~50-60 (past it). The composite over all 9 IS the full puzzle. Resizing bought exactly one
orbit.

Corners are only 26.39 bits (8.8e7 states) — that should be an exact BFS table, not a model.

---

## 6. What I would do next, in order

1. **Get a better/second solver file.** Worth more than everything else combined. A
   consistent 300-move solver would be ~-210,000 against today's floor; the current file's
   max is 562. GitHub is blocked for the agent identity (403 tunnel), so the user must
   download e.g. `dwalton76/rubiks-cube-NxNxN-solver` into `~/Downloads`.
   Caveats: it solves COLOUR cubes and 666 is a supercube (216 distinct stickers, solved =
   identity), leaving ~110 bits of centre residual; and wide/half turns inflate 2-3x in our
   single-slice QTM metric.
2. **Finish the neural bridge sweep** over pids ranked 401-1012 (ranks 1-400 are DONE, -1,180) (`66_neural_bridge.py
   --offset`). ~2.6 moves/pid on r=4-cleaned paths, ~2 min/pid.
3. **Re-run window r=4** on any new solver file (`67_window_segmented.py`) — 16.4 moves/pid
   on loose input, idempotent, ~0.5 min/pid.
4. Corners exact BFS table.

Do NOT re-run: any full-puzzle training arm, Bellman, path labels as a distance target,
window r=5 (priced at ~1 move/GPU-min vs r=4's 26), attention/transformer variants
(measured 11-50x the ResMLP at matched training rows).

---

## 7. Tooling notes that will save time

* **`67_window_segmented.py`** — deep ball sweeps OOM because `build_balls` seeds from every
  path state at once. Segment the path with `2r` overlap: lossless (any window <= 2r fits
  inside a segment) and cuts memory ~9x. This turned an OOM into -7,636.
* **pathkit `--scan` is `nargs="*"`** — passing the flag twice silently drops the first root
  and reports "scanned 0 files".
* **Always run `pathkit.cli selftest` first.** Every method here can return 0, and so can a
  broken harness.
* **Replay-verify everything.** `write_submission` re-reads and re-verifies; I also verify
  independently with our own replay code, and I verified the gdrive COPY (md5 + replay), not
  just the source — this mount has failed silently before.
* Model checkpoints saved before `policy_head` existed omit it from `model_config`;
  `load_model` now infers the head from the presence of `p_head` weights.

Full narrative with every measurement is in `../cube666_findings_2026-08-17.md`
(updates 4-6 cover this session).
