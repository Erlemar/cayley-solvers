# Discipline

Eight rules. Every one of them was paid for. They are listed with the incident that
produced them, because the incident is what makes the rule stick — and each is wired into
the code where that was possible, so following it is the default rather than a memory test.

---

## 1. Run the positive control before believing a zero

Every method here can return "0 improvements". So can a method that is silently unwired,
misconfigured, or pointed at the wrong file. The two are indistinguishable from the output.

**Incident.** Eight IHES post-processing scripts in a row returned exactly zero
(`14_mine_path_relations`, `15_trajectory_crossover`, `16_cross_trajectory_bridges`,
`18_mitm_gpu`, `19_ranked_cross_mitm`, `20_twsearch_exact_improve`, `21_plateau_relations`,
`22_twsearch_window_improve`). Before drawing any conclusion the harness was checked
against an inflated path: `f0` has order 4, so `f0 f0 f0 f0` is the identity — insert it and
confirm the sweep removes exactly it. It did (pid 0, 20 → 16, three windows hit, under a
second). The harness worked; the zeros were real. Had it *not*, eight conclusions would
have been wrong.

**In code.** `selftest.py` leads with the positive controls for the ball sweep, the table
sweep, the cheap passes and the suffix re-solve. Run `cli selftest` first, every time.

**Corollary — byte-identical results are not a null result.** If a config change produces
per-pid results identical to the control, the flag is not wired. `30_solve.py
--history-depth` was parsed and never forwarded; the only reason it was caught is that an
"hd=1" run reproduced the hd=0 control exactly.

---

## 2. A timeout is not a proof

There are three outcomes, not two: **HIT**, **NONE** (search completed, nothing exists) and
**UNPROVEN** (out of time, or out of reach). Collapsing the last two into "no improvement"
converts a budget shortfall into a false optimality claim.

**Incident.** An IHES window sweep at a 2-second limit reported 847 windows proven optimal.
Run to completion, the same rung was 847 proven and **4,142 timed out** — 83 % of the sweep
had proved nothing, and 844 of 1,003 pids still had at least one unproven window. The tell
was visible in the data all along: completed windows clustered at a median of 1.08 s with a
max of 1.98 s against a 2 s limit. When the completion times pile up against the budget, the
budget decided the outcome.

**In code.** `ladder.py` records all three verdicts, streams each to a JSONL journal the
moment its solve block closes, and `--resume` re-opens **only** the unproven ones, so a
longer budget refines a run instead of repeating it.

---

## 3. Know the band a method certifies

A meet-in-the-middle join with reach *R* proves two different things depending on the
window length, and only one of them is "optimal":

* **L ≤ R** — the join returns the optimal word, so no hit means the window is *exactly
  geodesic*;
* **L > R** — no hit certifies only *d(window) > R*; the window may still be non-geodesic
  with a distance between R+1 and L−1.

**In code.** `window.certify_report()` prints the band with every result.
`ladder.BallSolver` returns `NONE` only inside its own reach and `UNPROVEN` beyond it.

---

## 4. Score against the n-way per-pid min, never a remembered floor

An improvement is only real if it survives comparison with the shortest path anyone has
for those pids — including community files, old runs and other machines.

**Incident.** A megaminx bridge run reported "+17 moves saved" against the 75,200 base. A
newer community CSV (75,266) already had shorter paths for five of the six winning pids.
After `min(75200, 75266, bridge)` the incremental contribution was **one pid, −6**. The
17-versus-1 misattribution was caught only because someone questioned it.

**In code.** `merge.compare()` returns the delta *and* the per-pid win list against an
explicit baseline. Quote that, not a number from a handoff doc.

---

## 5. Three zeros do not imply the fourth

Extrapolating a trend of negative results is the failure mode that feels safest.

**Incident.** Tetraminx exact rewriting returned 0 at radius 10, 0 at radius 11 and 0 at
radius 12 (the last one exhaustive, 166,703 queries). That was written up as "every local
rewriting method is closed; do not build another local rewriter". **It was wrong by exactly
one radius**: radius 13 found 8 moves over 6 pids — the only exact post-processing win in
the entire programme. Five of the six were a 14-move window collapsing to exactly 13, the
band a reach-12 certificate cannot see.

Say **untested**, not closed. Radius 14 genuinely *is* out of reach, and note the
difference in how that is known: it was *measured* (B7 as permutations is 433M × 88 = 38 GB
at ~7.6 s/query, ~89 h), not extrapolated.

---

## 6. Rewrite all, then merge — never merge then rewrite

The usual pipeline min-merges first and post-processes the winner. That order is lossy.

A rewriter is **path-dependent**: it can only shorten a window it actually sees, and two
paths to the same solved state pass through different intermediate states. A shorter path
can be geodesic in every window and immune to rewriting, while a longer one contains a
window that collapses hard. Merging first throws the longer path away before the rewriter
ever sees it. Merging *after* rewriting is free and can only help.

An alternative *k* moves longer wins iff `s_alt > k + s_best`. On the tetraminx corpus that
was tested directly on 1,121 alternatives within +3 — 0 beat their pid's best, so the
correction was worth nothing *there*. The order is still the right one; it just has to be
cheap, which is what `bridge.py` makes it: pooled bridging does rewrite-all-then-merge in
one pass.

---

## 7. Replay-verify every splice; hash phantoms are deterministic

A 64-bit Zobrist probe false-positives at ~2.3 × 10⁻¹¹ and one MITM query can be 1.77 M
probes, so a full sweep sees a handful of states reported at a depth they are not at.

The phantom is **deterministic** — the same hash on CPU and GPU — so re-checking the depth
cannot catch it. It surfaces as a descent that finds no child one level down: an assertion
crash mid-sweep, not a wrong answer. Seven phantoms in 46,687 queries at a 433 M-entry
table. Any table past ~10⁸ entries keyed by a 64-bit hash needs this treatment.

**In code.** Every candidate splice in `window.py`, `bridge.py`, `ladder.py` and `neural.py`
is replayed against the real state before it is accepted, so a phantom costs a rejected
candidate rather than a corrupt row. `write_submission()` re-reads and re-verifies the file
it just wrote — verifying the in-memory dict proves nothing about the file you submit.

---

## 8. Measure the constraint before optimising it; price a build before paying for it

**Incident (pricing).** A d8 BFS table was proposed for deeper window rewriting. Before
building it, `--extend 2` on the existing d6 table was run — it BFSes 2 moves out from the
window element and looks the frontier up, reaching exactly what d8 would have reached. It
found **0 rewrites**, and d8 was rejected by measurement rather than by argument. The
same run also showed the merge is locally irreducible out to 9-move windows, not just 6.

**Incident (matched control).** A day of local `history_depth=0` runs was compared against
TPU `history_depth=1` baselines, producing two conclusions that both had to be retracted —
the machine, flags and checkpoint all differed, and the difference was silently attributed
to the variable under test. If the matched control does not exist, run it: it is cheaper
than the wrong conclusion. And never generalise an interaction from one measured pair to an
unmeasured one — "these two do not stack" was wrongly extended to a third combination that
measured 90 % additive.

---

## The short version

Before a run: positive control, matched control, and price the next rung with `--extend`
or `slack_audit` if one exists.

After a run: report the band, report proven and unproven separately, and score the delta
against the n-way per-pid min.

Never: extrapolate a trend of zeros, trust a timeout, merge before rewriting, or accept a
splice you have not replayed.
