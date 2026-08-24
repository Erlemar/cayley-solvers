# pathkit — path post-processing for Cayley-graph puzzles

Everything we know about making an already-valid solution shorter, as one puzzle-agnostic
package: the methods, the code, the cost of each, and — just as important — the measured
evidence for where each one stops paying.

It works unchanged on all four solvers in this repo (IHES picture cube, tetraminx,
megaminx, cube444) plus any puzzle that ships the same `puzzle_info.json`, on CPU or GPU,
with numpy as the only hard dependency.

```
pathkit/
  README.md        this file -- the methods and when to use them
  DISCIPLINE.md    the eight process rules, each with the incident that produced it
  VERDICTS.md      what has been measured on each puzzle: do not re-run a closed axis
  pathkit/
    puzzle.py      one Puzzle abstraction; permutation vs colour detection
    balls.py       ball construction, hashing, the two group reduces  <- the core
    window.py      ball-collision window shortening (reach 2r)
    table.py       BFS-table window rewriting, and the --extend trick
    bridge.py      pooled cross-trajectory bridging (subsumes merge + splice + window)
    merge.py       n-way per-pid min-merge by CONTENT
    cheap.py       adjacent-inverse cancellation, state-hash shortcut
    ladder.py      resumable exact ladder: hit / none / UNPROVEN
    neural.py      residual extraction + scorer-guided window selection
    suffix.py      suffix re-solve (the diagnostic that tells you when to stop)
    cli.py         one entry point for all of it
    selftest.py    positive controls -- run these before believing any zero
```

## Install and run

No install needed; it is importable from the repo root.

```bash
PYTHONPATH=pathkit .venv/Scripts/python.exe -m pathkit.cli selftest
PYTHONPATH=pathkit .venv/Scripts/python.exe -m pathkit.cli plan --preset ihes --in submissions/best.csv
```

`--preset` resolves `puzzle_info.json` and `test.csv` for `ihes`, `tetraminx`, `megaminx`,
`cube444`, or `demo` (a self-contained toy puzzle needing no data files). Anything else:
`--puzzle-info path/to/puzzle_info.json --tests path/to/test.csv`.

Optional: `pip install torch` enables `--device cuda`. Same code path, same answers —
verified by a backend-parity check in the self-test.

## The decision procedure

Run in this order. Each step is cheap relative to the next, and each one's *zero* tells
you something about whether the next is worth starting.

| # | step | command | typical yield |
|---|---|---|---|
| 1 | **n-way min-merge** | `merge --scan . --base best.csv` | large, and never stops paying |
| 2 | cheap local passes | `cheap --in best.csv` | 0.2–0.3 % |
| 3 | ball window sweep r=3 | `window --radius 3 --device cuda` | a few moves per 1000 paths |
| 4 | table sweep + `--extend 2` | `table-window --table t.npz --extend 2` | 0 on tight files; prices a deeper table for free |
| 5 | pooled bridging | `bridge --scan results/ --radius 3` | subsumes 1 + 3; the only source of cross-path wins |
| 6 | suffix probe | `suffix --ks 12,16,20` | diagnostic — tells you when rewriting is finished |
| 7 | neural bridge | (needs your solver) | only on long, non-merged, our-own paths |

Measured on a real IHES file, for calibration of expectations:

```
$ ... cli merge  --preset ihes --scan submissions --base e11_khoruzhii_b65k.csv
  vs e11_khoruzhii_b65k.csv: 25,034 -> 21,870 (-3164) over 916 improved pids

$ ... cli cheap  --preset ihes --in e11_khoruzhii_b65k.csv
  cheap: 25,034 -> 25,030 (-4 moves, 1 paths changed)

$ ... cli window --preset ihes --in e11_khoruzhii_b65k.csv --radius 3 --device cuda
  window reduce r=3: 25,034 -> 25,010 (-24 moves, 11 splices) | proves every window of
  length <= 6 geodesic; longer windows only certified d > 6            [31 s, 1003 pids]
```

The ratio is the lesson: merging beat every rewriter by two orders of magnitude on the
same file. Rewriting is what you do *after* the merge is genuinely exhausted.

---

## 1. N-way per-pid min-merge — `merge.py`

**Mechanism.** For each puzzle id keep the shortest replay-verified path found in *any*
source: our runs, old runs, other machines, community files, old Kaggle kernel versions.

**Why it keeps paying.** The source set is not stable and is easy to under-scan. On
tetraminx the recorded best was 28,821 while the true verified min over what was already
on disk was 28,718 — 103 moves sat unclaimed because nothing had merged them. A stale copy
of a live machine caps the merge silently, so re-pull each box rather than refreshing only
the file being watched.

**Search by content, not name or location.** A file called `submission_dad.csv` sat in the
megaminx folder holding *tetraminx* data; three `submission_publ*.csv` sat in an unrelated
project folder. Together, 22 moves. The header alone is not a sufficient test — every
puzzle in this family has the same header — so `sniff_submission` also checks the move
**alphabet**, which is what makes pointing a scan at arbitrary folders safe: a foreign
puzzle self-rejects.

```python
from pathkit.merge import merge, compare, format_report
best, report = merge(puzzle, tests, roots=["/repo", "~/Downloads", scratch_dir])
format_report(report)
print(compare(puzzle, best, baseline_paths)["delta"])
```

**The attribution trap.** When reporting an improvement on base *B*, compare against the
per-pid min over every *other* source covering those pids, not against *B*. A megaminx
bridge run reported "+17 moves saved"; against the true n-way floor its unique
contribution was one pid and 6 moves. `compare()` exists to make that the easy thing to do.

**Adjacent lever: old Kaggle kernel versions.** A kernel serves only its *latest* version
through the CLI, but old versions each hold unique wins — one sweep was worth 51 moves and
a re-sweep four days later another 91. Version *count* is a bad proxy for value: a
276-version kernel contributed exactly zero because its paths were mostly long fallback.
Check the fraction of beam-quality paths first. (Puller lives in the main repo at
`scripts/25_pull_kernel_versions.py`; the merge here consumes its output.)

## 2. Cheap local passes — `cheap.py`

Adjacent-inverse cancellation, plus a one-move state-hash shortcut that jumps from any
position to the latest later position holding the same state. Both are O(L) and free.

They are strictly subsumed by a radius-1 ball sweep, so their only reason to exist is that
they need no ball machinery — use them inside a solver loop or on a machine with no GPU.
Expect 0.2–0.3 %. That number is itself informative: beam output is already locally tight,
which is why the interesting methods are the ones that look further than one move.

## 3. Ball-collision window shortening — `window.py`, `balls.py`

**The one idea in the package.** Expand a ball of radius *r* around every state of the
path at once, and look for a state reached from two different positions:

```
x in ball(s_i) at depth d1,  x in ball(s_j) at depth d2,  j > i
    =>  window [i, j) can be replaced by a word of length d1 + d2
    =>  saving = (j - d2) - (i + d1)
```

Per colliding state this is one `max(i - d)` and one `min(i + d)` — a single group reduce,
**no pair enumeration** — so one BFS per path covers *every* window of length ≤ 2r
simultaneously. That is what makes a 1000-path sweep affordable.

**States, not permutations.** The collision test compares state vectors, so on a colour
puzzle it accepts any replacement landing on the same *colouring*. That is strictly weaker
than requiring the same permutation, and it is where cube444's only exact win came from:

```
pid 761  window [37,46)   r2.f2.r2.-d2.-f2.-f2.d2.f2.r2   (9)
                     ->   -d1.-d1.-f1.-d1.f1.-d1.-r2      (7)
permutations equal? False -- differ at 6 facelets, ALL of them centres
states equal?       True
```

A permutation-space MITM rejects that replacement. On a permutation puzzle the two tests
coincide, so always using the state test costs nothing. The self-test demonstrates the gap
on a toy colour puzzle (22 of 25 paths strictly shorter in colour space).

**What a zero proves — read this before quoting a result.**

* window length L ≤ 2r → the join *would* have found any shorter word, so no hit means the
  window is **exactly geodesic**;
* window length L > 2r → no hit only certifies **d(window) > 2r**. It can still be
  non-geodesic with a distance between 2r+1 and L−1.

`certify_report()` prints the band with every result so the second cannot be reported as
the first.

**Cost.** Each rung of *r* multiplies the ball by the branching factor — measured 19.21 on
cube444 (level sizes 1, 24, 468, 9000, 172914). Radius 3 sweeps 1000 IHES paths in 31 s on
a GPU; radius 5 is minutes *per path*. Spend the next rung only on the long tail
(`--min-length`).

**Engineering notes that are load-bearing at scale** (all inherited from the cube444 port):
hash in column groups (a one-shot int64 cast of a child batch is an *N*-fold blow-up and
OOMs a 16 GB card); bucket on the hash low bits before sorting past ~10⁸ entries; stream
the last BFS level without dedup since nothing expands it; and **replay-verify every
splice**, because a 64-bit Zobrist probe false-positives at ~2·10⁻¹¹ and a full sweep runs
enough probes to see a handful. Those phantoms are *deterministic* — re-hashing cannot
catch them, only replay can.

## 4. BFS-table window rewriting — `table.py`

The ball sweep is anchored at the path, so it is rebuilt per path. A BFS table is anchored
at the identity, so it is built once and reused forever. On a permutation puzzle a window
equals the group element `w = s_i⁻¹ · s_j`, which is a single lookup. Colour puzzles have
no inverse state and cannot use this — `window_reduce` asserts on it.

**The `--extend` trick, which matters more than the table depth.** A table of depth *D*
only knows distances up to *D*. But if a short word *v* takes *w* into the table at depth
*d*, then `w = (w·v)·v⁻¹`, giving a word of length `d + |v|`. So BFS *k* moves out from
*w*, look every frontier state up, keep the best — effective reach becomes **D + k with no
deeper table built**. Cost per window is about `n_gen^k` lookups, so k=2 is the practical
limit.

This is how a d8 table was priced at zero on tetraminx without building it:

| table | `--extend` | effective reach | result on the 28,821 merge |
|---|---|---|---|
| d6 | 0 | 6 | 0 rewrites |
| d7 | 0 | 7 | 0 rewrites |
| d6 | 2 | **8** | 0 rewrites |
| d7 | 2 | **9** | 0 rewrites |

Reach 8 is exactly what a d8 table would have delivered, and it found nothing. `BfsTable`
reads this repo's `bfs_endgame*.npz` directly (verified against the real 27.8M-state d6
table) and can also `build()` its own for smaller puzzles.

## 5. Pooled cross-trajectory bridging — `bridge.py`

The generalisation that contains the others. Pool the waypoints of *every* path you own
for a pid and expand a ball around each, so two trajectories that merely pass *near* each
other can be joined:

```
new length = g(x) + d(x, z) + d(z, y) + h(y)
```

with `g` the cheapest cost-from-start over all pooled paths reaching *x*, and `h` the
cheapest cost-to-solved from *y*. One pass subsumes n-way min-merge (x = start,
y = solved), exact crossover splicing (d1 = d2 = 0), within-path window shortening (both
waypoints from one path), and genuine cross-path bridges — the part nothing else does.

**It also fixes an ordering mistake.** The usual pipeline min-merges first and rewrites the
winner. That is lossy: the rewriter is path-dependent, so a *shorter* path can be geodesic
everywhere and immune while a *longer* one contains a window that collapses hard. Merging
first throws the longer path away before any rewriter sees it. Rewrite-all-then-merge is
free; merge-then-rewrite permanently destroys candidates.

**Price it before running it.** A splice through waypoint *x* beats the incumbent only if
`g(x) + d(x) < incumbent`, since `d(x⁻¹y) + h(y) ≥ d(x⁻¹y) + d(y) ≥ d(x)`. On tetraminx,
over 6,979 waypoints with an exactly known `d(x)`, **zero** had slack — no radius and no
extra trajectory could have produced a bridge. `slack_audit()` computes that count in
seconds; if it is zero, stop.

## 6. The exact ladder — `ladder.py`

**A timeout is not a proof.** A sweep reporting "0 improvements" is ambiguous between:

```
HIT       a shorter word was found and verified
NONE      the search COMPLETED and proved no shorter word exists
UNPROVEN  the search hit its time limit, or its reach was shorter than the window
```

An IHES sweep once reported 847 windows "proven optimal" at a 2-second limit. Run to
completion, the same rung was 847 proven and **4,142 timed out** — 83 % of it had proved
nothing, and the completed windows clustered just under 2 s, which is the tell that the
budget, not the puzzle, decided the outcome.

So every verdict is one of the three, streamed to a JSONL journal the moment it closes, and
`--resume` re-opens **only** the unproven ones — a bigger budget refines a run instead of
repeating it. `BallSolver` returns `NONE` only inside its own reach and `UNPROVEN` beyond
it, so the band distinction is enforced in code rather than remembered. `ExternalSolver`
wraps any out-of-process exact solver (twsearch and friends) behind the same contract.

**Cost shape.** Finding a shortening is fast; *proving none exists* costs the whole tree,
and every sweep like this is a refutation. Measured with IDA* plus a depth-11 prune table:
a length-20 path proved to threshold 18 in 49 s on 64 threads; length-21 to threshold 19 in
865 s. **~11× per extra move of depth.** Two consequences: searching for a *bigger* saving
is *cheaper* (threshold L−4 is a far smaller tree than L−2), and short work items are the
affordable ones. `parity_step()` detects the case where every generator is an odd
permutation, in which length parity is invariant and half the ladder disappears.

## 7. Neural bridge compression — `neural.py`

Re-solve a window with the full solver instead of a table. The residual sub-problem is

```
X = inv(s_j)[s_i]
```

because a word solving *X* to the identity has exactly the composite permutation taking
`s_i` to `s_j`. Hand *X* to whatever solver you own; splice anything shorter. It cannot
regress — the worst case is spent compute.

This is the only method that reaches past any table or ball you can afford. It is also the
one with a sharp boundary. Measured on megaminx: **−128 moves at a ~3.5 % win rate** on our
own loose paths; **under 1 %** on already min-merged community paths; **exactly zero** on
the hardest pids at every configuration tried, including a width-1M TPU beam with symmetry
ensembling.

The cause is the **scorer**, not the solver. A distance model saturates near the puzzle
diameter, so for a deep residual it predicts the same value for everything and the
"predicted saving" that selects windows becomes pure false positives. `estimate_saturation()`
measures the horizon and `select_windows()` refuses candidates beyond it, rather than
trusting the score. And when this method works at all it works *cheaply* — on
community-floor mid pids a 23-minute plain beam found the identical −3 that an 11.6-hour
sym+NISS+iterate run did.

## 8. Suffix re-solve — `suffix.py`

Interior windows have an arbitrary target, and a value model trained on distance-to-solved
says nothing about distance-to-`s_j`. The suffix is the exception: solving `s_{L−k}` means
reaching the real solved state, so the existing scorer, beam, table and symmetry machinery
all apply. On colour puzzles, where no residual can be extracted at all, this is the *only*
neural post-processing available.

**Use it as a diagnostic first.** It answers a question no amount of rewriting can — is our
search able to beat this file anywhere?

```
k=12: 6/6 suffixes returned exactly 12    (matched, never beat)
k=16: 3 matched, 1 worse, 2 no solution
k=20: 2 matched, 1 worse, 3 no solution   -> beat-k: 0 / 18
```

That is the best public cube444 file. It says the file is already at our search's optimum
wherever we can search, and the misses at k=16/20 are *our* width limit, not slack in the
file. The result took minutes and redirected the whole effort from post-processing to a
better global search.

---

## Cost model at a glance

| method | scaling knob | cost per rung | reach |
|---|---|---|---|
| cheap passes | — | O(L) | 1 move |
| ball sweep | radius r | × branching factor (19.2 on cube444) | 2r, all windows at once |
| table sweep | table depth D | × branching per level, built once | D |
| table + extend k | k | × `n_gen^k` per window | D + k |
| pooled bridging | radius r | × branching, over pooled waypoints | 2r, across paths |
| exact ladder | search threshold | **~11× per extra move** | exact, one window at a time |
| neural bridge | solver budget | your solver's cost | unbounded, scorer-limited |

## Python API

```python
from pathkit.puzzle import Puzzle
from pathkit.io import load_tests, load_submission, write_submission
from pathkit import window, merge, bridge, table, cheap, suffix

pz    = Puzzle.from_puzzle_info("tetraminx/data/puzzle_info.json")
tests = load_tests("tetraminx/data/test.csv")
paths = load_submission("best.csv", pz)

paths, rep = cheap.sweep(pz, tests, paths)
paths, rep = window.sweep(pz, tests, paths, radius=3, device="cuda", min_length=30)
print(window.certify_report(rep))

write_submission("out.csv", pz, paths, tests)      # writes, re-reads, re-verifies
```

Every method takes `(puzzle, tests, paths)` and returns `(new_paths, report)`; every one
replay-verifies before returning; none can make a path longer.

## Testing

```bash
PYTHONPATH=pathkit .venv/Scripts/python.exe -m pathkit.cli selftest
```

30 checks including the positive controls, backend parity (numpy / torch CPU / CUDA agree),
band semantics, merge safety against foreign and corrupt files, and journal resume. The
positive controls matter more than they look: every method here can return "0
improvements", and that is also exactly what a broken pipeline returns.

## Where the numbers come from

`VERDICTS.md` — every measured result per puzzle, so a closed axis is not re-run.
`DISCIPLINE.md` — the eight process rules, each with the incident that produced it.
