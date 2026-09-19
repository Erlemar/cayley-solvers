"""The resumable exact ladder -- and the reason it exists: A TIMEOUT IS NOT A PROOF.

A window sweep that reports "0 improvements" is ambiguous between three very different
outcomes, and collapsing them is the single most expensive mistake in this programme:

    HIT       a shorter word was found and verified
    NONE      the search COMPLETED and proved no shorter word exists
    UNPROVEN  the search hit its time limit, or its reach was shorter than the window

An IHES sweep once reported 847 windows "proven optimal" at a 2-second limit. Run to
completion the same rung was 847 proven and 4,142 timed out -- 83 pct of it had proved
nothing, and the completed windows clustered just under 2 s, which is the tell that the
budget and not the puzzle decided the outcome. Every verdict here is therefore recorded
as one of the three above, streamed to a JSONL journal the moment it closes, and `resume`
re-opens ONLY the unproven ones so a bigger budget refines a run instead of repeating it.

COST SHAPE. Finding a shortening is fast; proving none exists costs the whole tree, and
every sweep like this is a refutation. Measured with an IDA* solver and a depth-11 prune
table: a length-20 full path proved to threshold 18 in 49 s on 64 threads, a length-21
path to threshold 19 in 865 s. That is ~11x per extra move of depth. Two consequences
worth planning around: searching for a BIGGER saving is CHEAPER (threshold L-4 is a far
smaller tree than L-2), and short work items are the affordable ones.

PARITY. If every generator is an odd permutation, path length mod 2 is invariant and a
length-22 window can only shorten to 20, never 21. `parity_step` detects this and halves
the ladder.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .backend import get_backend
from .balls import Zobrist, build_balls, collide_savings, word_to

HIT = "hit"
NONE = "none"
UNPROVEN = "unproven"


class Journal:
    """Append-only JSONL of window verdicts. Survives a kill; `resume` reads it back."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.closed = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("verdict") in (HIT, NONE):
                        self.closed[self._key(rec)] = rec

    @staticmethod
    def _key(rec):
        return (int(rec["pid"]), int(rec["i"]), int(rec["j"]))

    def is_closed(self, pid, i, j):
        return (int(pid), int(i), int(j)) in self.closed

    def write(self, rec):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if rec["verdict"] in (HIT, NONE):
            self.closed[self._key(rec)] = rec

    def hits(self):
        return [r for r in self.closed.values() if r["verdict"] == HIT]

    def counts(self):
        c = {HIT: 0, NONE: 0}
        for r in self.closed.values():
            c[r["verdict"]] = c.get(r["verdict"], 0) + 1
        return c


class BallSolver:
    """Exact window solver by ball collision. Honest about its band.

    Proves NONE only when the window is no longer than its reach (2 * radius); beyond that
    it returns UNPROVEN, because failing to find a word of length <= 2r says nothing about
    whether one of length 2r+1 .. L-1 exists.
    """

    def __init__(self, puzzle, radius, device="cpu"):
        self.puzzle = puzzle
        self.radius = radius
        self.bk = get_backend(device)
        self.zob = Zobrist(puzzle.state_size, puzzle.n_labels, self.bk)

    @property
    def reach(self):
        return 2 * self.radius

    def solve(self, s_from, s_to, max_len):
        """Return (verdict, word). `max_len` is the length to beat (the window length)."""
        pz = self.puzzle
        sources = np.stack([s_from, s_to])
        balls = build_balls(pz, sources, self.radius, bk=self.bk, zob=self.zob)
        cands = collide_savings(balls, min_saving=-(10 ** 6))
        _, src_np, dist_np, par_np, mv_np = balls.to_numpy()
        self.bk.free()
        best = None
        for _sav, e_lo, e_hi in cands:
            if int(src_np[e_lo]) == int(src_np[e_hi]):
                continue                                   # both halves from one source
            a, b = e_lo, e_hi
            if int(src_np[a]) != 0:
                a, b = b, a
            total = int(dist_np[a]) + int(dist_np[b])
            if total >= max_len or (best is not None and total >= best[0]):
                continue
            word = word_to(par_np, mv_np, a) + pz.inv_word(word_to(par_np, mv_np, b))
            if not np.array_equal(pz.apply_word(s_from, word), s_to):
                continue                                   # hash phantom
            best = (total, word)
        if best is not None:
            return HIT, best[1]
        return (NONE if max_len <= self.reach + 1 else UNPROVEN), None


class ExternalSolver:
    """Adapter for an out-of-process exact solver (twsearch and friends).

    `run` receives (from_state, to_state, max_len, time_limit) and must return
    (verdict, word_or_None). Wrap whatever binary you have; the ladder only needs the
    three verdicts, and returning UNPROVEN on a timeout is the contract that keeps the
    journal honest.
    """

    def __init__(self, run, time_limit=60):
        self.run = run
        self.time_limit = time_limit

    def solve(self, s_from, s_to, max_len):
        return self.run(s_from, s_to, max_len, self.time_limit)


def parity_step(puzzle):
    """2 when every generator is an odd permutation (length parity is invariant), else 1.

    CONSERVATIVE. It looks at the parity of the FULL state permutation, so it finds the
    invariant only when it is visible there. A puzzle can still have a parity constraint
    living in a quotient -- on the IHES picture cube every generator is an odd permutation
    of the 12 underlying EDGES while being even on the 72 facelets, so this returns 1 and
    the real ladder can still step by 2. A returned 1 means "no invariant detected", not
    "no invariant exists"; check the puzzle's piece decomposition before assuming.
    """
    if not puzzle.is_permutation:
        return 1
    n = puzzle.state_size
    for g in puzzle.gens:
        seen = np.zeros(n, dtype=bool)
        parity = 0
        for start in range(n):
            if seen[start]:
                continue
            ln, k = 0, start
            while not seen[k]:
                seen[k] = True
                k = int(g[k])
                ln += 1
            parity += ln - 1
        if parity % 2 == 0:
            return 1
    return 2


def run_ladder(puzzle, tests, paths, solver, journal, window_length, time_budget=None,
               order="longest-first", log=print, progress=50):
    """Feed every window of `window_length` to `solver`, streaming verdicts to `journal`.

    Windows are ordered longest-incumbent-first by default, so the pids with the most
    slack are decided before any budget runs out.
    """
    step = parity_step(puzzle)
    items = []
    for pid, w in paths.items():
        L = len(w)
        if L < window_length:
            continue
        for i in range(0, L - window_length + 1):
            items.append((L, pid, i, i + window_length))
    if order == "longest-first":
        items.sort(key=lambda t: (-t[0], t[1], t[2]))
    todo = [it for it in items if not journal.is_closed(it[1], it[2], it[3])]
    log("ladder w%d: %d windows, %d already closed, %d to run (parity step %d)"
        % (window_length, len(items), len(items) - len(todo), len(todo), step))
    reach = getattr(solver, "reach", None)
    if reach is not None and reach < window_length:
        log("  WARNING: solver reach %d < window length %d -- every non-hit will be "
            "UNPROVEN, not proven. Raise the radius to at least %d to close this rung."
            % (reach, window_length, -(-window_length // 2)))

    t0 = time.time()
    done = {HIT: 0, NONE: 0, UNPROVEN: 0}
    for k, (_L, pid, i, j) in enumerate(todo):
        if time_budget is not None and time.time() - t0 > time_budget:
            log("  time budget reached; %d windows left OPEN (not proven)"
                % (len(todo) - k))
            break
        states = puzzle.path_states(tests[pid], paths[pid])
        target = (j - i) - step                        # the only shorter length parity allows
        verdict, word = solver.solve(states[i], states[j], (j - i))
        rec = {"pid": int(pid), "i": int(i), "j": int(j), "len": int(j - i),
               "target": int(target), "verdict": verdict,
               "word": puzzle.format(word) if word else None,
               "t": round(time.time() - t0, 2)}
        journal.write(rec)
        done[verdict] = done.get(verdict, 0) + 1
        if progress and (k + 1) % progress == 0:
            log("  %d/%d  hit=%d none=%d unproven=%d  %.1fs"
                % (k + 1, len(todo), done[HIT], done[NONE], done[UNPROVEN],
                   time.time() - t0))
    log("ladder w%d done: hit=%d proven-none=%d UNPROVEN=%d"
        % (window_length, done[HIT], done[NONE], done[UNPROVEN]))
    return done


def apply_journal(puzzle, tests, paths, journal, log=print):
    """Splice in every journal HIT that replay-verifies. Reports proven and unproven apart.

    Applied per pid, longest saving first, on disjoint windows -- a hit recorded against
    the pre-splice trace is invalid once an overlapping window has been rewritten.
    """
    by_pid = {}
    for rec in journal.hits():
        by_pid.setdefault(int(rec["pid"]), []).append(rec)
    out = {pid: list(w) for pid, w in paths.items()}
    applied = rejected = 0
    for pid, recs in by_pid.items():
        if pid not in out:
            continue
        recs.sort(key=lambda r: -(r["len"] - (r["word"].count(".") + 1 if r["word"] else r["len"])))
        used = np.zeros(len(out[pid]) + 1, dtype=bool)
        for rec in recs:
            i, j = int(rec["i"]), int(rec["j"])
            if not rec["word"] or j > len(out[pid]) or used[i:j].any():
                rejected += 1
                continue
            new_word = puzzle.parse(rec["word"])
            if len(new_word) >= j - i:
                rejected += 1
                continue
            states = puzzle.path_states(tests[pid], out[pid])
            if not np.array_equal(puzzle.apply_word(states[i], new_word), states[j]):
                rejected += 1
                continue
            out[pid] = out[pid][:i] + new_word + out[pid][j:]
            used[i:j] = True
            applied += 1
        assert puzzle.solves(tests[pid], out[pid]), "pid %s: journal splice broke path" % pid
    before = sum(len(v) for v in paths.values())
    after = sum(len(v) for v in out.values())
    counts = journal.counts()
    log("applied %d hits (%d rejected), %s -> %s (%+d)"
        % (applied, rejected, "{:,}".format(before), "{:,}".format(after), after - before))
    log("journal: %d proven-none, %d hits -- anything not listed is UNPROVEN, not optimal"
        % (counts.get(NONE, 0), counts.get(HIT, 0)))
    return out, {"applied": applied, "rejected": rejected,
                 "before": before, "after": after, "saved": before - after,
                 "proven_none": counts.get(NONE, 0)}
