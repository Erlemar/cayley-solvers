"""Exact meet-in-the-middle window shortening at radius 11/12.

WHY THIS AND NOT ANOTHER SWEEP. `42_window_reduce.py` replaces a window whose element
is inside the exact d<=6 ball. A radius-10 sweep (competitor, 2026-08-04) returned zero.
The untested band is 11-12, and it is reachable WITHOUT building d7/d8 tables:

    d(w) <= 11   iff  exists a in B5 with  a^-1 . w  in B6
    d(w) <= 12   iff  exists a in B6 with  a^-1 . w  in B6

B5 is only 1,771,577 elements, so a radius-11 certificate is ~1.8M table probes -- a
couple of seconds vectorised. Radius 12 needs a 27.8M front and is ~16x dearer, so it is
reserved for a small ranked set.

ALGEBRA (convention: apply(s, m) = s[gen[m]], so perm(w1+w2) = perm(w1)[perm(w2)]):
    W = A[B]   =>   B = argsort(A)[W]
so the front is stored as INVERSE permutations and one fancy-index `AINV[:, W]` produces
every candidate B at once. The solved state is the identity, so a state IS its element
and the d6 table can be probed directly.

WHAT THE 48-FOLD CANONICALISATION IS FOR HERE. NOT for the table probe -- the exact ball
is closed under conjugation and inversion (verified 80/80), so canonicalising before a
lookup is a provable no-op. It is for DEDUP: solve each canonical window element once and
apply the answer to every occurrence in every row. That is the whole reason a
~150k-window sweep becomes tractable.

    python tetraminx/scripts/59_mitm_rewrite.py --self-test
    python tetraminx/scripts/59_mitm_rewrite.py \
        --target tetraminx/submissions/submission_28456.csv \
        --front-depth 5 --min-window 12 --max-queries 2000 \
        --out tetraminx/submissions/mitm_out.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "tetraminx" / "data"


class Puzzle:
    def __init__(self, data_dir: Path):
        info = json.loads((data_dir / "puzzle_info.json").read_text(encoding="utf-8"))
        self.names = list(info["generators"].keys())
        self.gen = np.array([info["generators"][n] for n in self.names], dtype=np.int64)
        self.n_gen, self.S = self.gen.shape
        self.ident = np.arange(self.S, dtype=np.int64)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.solved = np.array(info["central_state"], dtype=np.int64)
        assert np.array_equal(self.solved, self.ident), \
            "this script assumes solved == identity so that a state IS its group element"
        self.inv_move = np.zeros(self.n_gen, dtype=np.int64)
        for i in range(self.n_gen):
            for j in range(self.n_gen):
                if np.array_equal(self.gen[i][self.gen[j]], self.ident):
                    self.inv_move[i] = j
                    break
        self.SYM = np.load(data_dir / "tetra_symmetries.npy")
        self.SYMI = np.load(data_dir / "tetra_symmetries_inv.npy")
        self.RELAB = np.load(data_dir / "tetra_move_relabel.npy")
        self.relab_inv = np.argsort(self.RELAB, axis=1)
        self.n_sym = self.SYM.shape[0]
        z = np.load(data_dir / "bfs_endgame.npz")
        self.H = z["hashes"]
        self.DEP = z["depths"].astype(np.int8)
        self.ZT = z["ztab"].astype(np.int64)
        self.max_depth = int(z["max_depth"])

    # ---- basics
    def perm_of(self, word):
        P = self.ident.copy()
        for m in word:
            P = P[self.gen[m]]
        return P

    def inv_word(self, word):
        return [int(self.inv_move[m]) for m in reversed(word)]

    def hash_rows(self, states: np.ndarray) -> np.ndarray:
        """(N, S) int -> (N,) int64 Zobrist, matching the table's own keying."""
        h = np.zeros(states.shape[0], dtype=np.int64)
        for i in range(self.S):
            h ^= self.ZT[i][states[:, i]]
        return h

    def depth_of(self, states: np.ndarray) -> np.ndarray:
        """(N, S) -> (N,) depth in the d<=6 ball, or -1 when outside."""
        h = self.hash_rows(states)
        pos = np.searchsorted(self.H, h)
        np.clip(pos, 0, len(self.H) - 1, out=pos)
        hit = self.H[pos] == h
        out = np.where(hit, self.DEP[pos], -1).astype(np.int64)
        return out

    def descend(self, state: np.ndarray):
        """Optimal word taking `state` to solved, using the exact ball. [] if solved."""
        d = int(self.depth_of(state[None, :])[0])
        if d < 0:
            raise ValueError("state not in the ball")
        word = []
        cur = state.copy()
        while d > 0:
            kids = cur[self.gen]                      # (n_gen, S)
            dk = self.depth_of(kids)
            nxt = int(np.argmax((dk >= 0) & (dk == d - 1)))
            assert dk[nxt] == d - 1, "no descending child -- table inconsistent"
            word.append(nxt)
            cur = kids[nxt]
            d -= 1
        return word

    # ---- 48-fold canonical form (for DEDUP only; see module docstring)
    def canonical(self, g: np.ndarray):
        rows = np.empty((2 * self.n_sym, self.S), dtype=np.uint8)
        for half, h in ((0, g), (self.n_sym, np.argsort(g))):
            G = h[self.SYMI]
            rows[half:half + self.n_sym] = np.take_along_axis(self.SYM, G, axis=1)
        b = int(np.lexsort(rows[:, ::-1].T)[0])
        return rows[b].tobytes(), b % self.n_sym, b >= self.n_sym

    def transport(self, word, k: int, inverted: bool):
        back = self.relab_inv[k]
        out = [int(back[m]) for m in word]
        return self.inv_word(out) if inverted else out


# Known BFS level sizes for this puzzle -- used as a build-time correctness check.
# A hash-dedup bug shows up immediately as a level-size mismatch.
BFS_LEVELS = [1, 24, 408, 6592, 105136, 1659416, 26008172]


def build_ball(pz: Puzzle, depth: int, verbose=True):
    """BFS from identity to `depth`, fully vectorised.

    Returns (P uint8 (N,S) elements, parent int32 (N,), move int8 (N,), level_start list).
    Words are reconstructed on demand from the parent chain rather than stored: at depth 5
    that is 1.77M lists of <=5 ints in Python (~700 MB) versus 9 MB of pointers.

    Dedup is by 64-bit Zobrist. Collisions at this scale are ~1e-6 likely, and any dedup
    error would change a level size, which is asserted against BFS_LEVELS.
    """
    t0 = time.time()
    P = pz.ident.astype(np.uint8)[None, :].copy()
    parent = np.array([-1], dtype=np.int32)
    move = np.array([-1], dtype=np.int8)
    seen = pz.hash_rows(P.astype(np.int64))
    level_start = [0]
    lo = 0
    for d in range(1, depth + 1):
        cur = P[lo:].astype(np.int64)                       # frontier
        kids = cur[:, pz.gen]                               # (n, n_gen, S): cur[i][gen[m]]
        n, g, S = kids.shape
        kids = kids.reshape(-1, S)
        kh = pz.hash_rows(kids)
        # drop anything already seen at an earlier depth, then dedup within the level
        fresh = ~np.isin(kh, seen, assume_unique=False)
        kids, kh = kids[fresh], kh[fresh]
        src = np.repeat(np.arange(lo, lo + n, dtype=np.int32), g)[fresh]
        mv = np.tile(np.arange(g, dtype=np.int8), n)[fresh]
        _, first = np.unique(kh, return_index=True)
        first.sort()
        kids, kh, src, mv = kids[first], kh[first], src[first], mv[first]
        lo = P.shape[0]
        level_start.append(lo)
        P = np.concatenate([P, kids.astype(np.uint8)], axis=0)
        parent = np.concatenate([parent, src])
        move = np.concatenate([move, mv])
        seen = np.concatenate([seen, kh])
        seen.sort()
        if d < len(BFS_LEVELS):
            assert kids.shape[0] == BFS_LEVELS[d], (
                f"depth {d}: got {kids.shape[0]:,}, expected {BFS_LEVELS[d]:,}")
        if verbose:
            print(f"  ball depth {d}: {kids.shape[0]:,} new (cum {P.shape[0]:,})"
                  f"  {time.time()-t0:.0f}s", flush=True)
    return P, parent, move, level_start


def word_of_index(pz: Puzzle, i: int, parent, move):
    """Reconstruct the word for ball element i by walking parent pointers."""
    w = []
    while parent[i] >= 0:
        w.append(int(move[i]))
        i = int(parent[i])
    return w[::-1]


def solve_element(pz: Puzzle, W: np.ndarray, AINV: np.ndarray, parent, move,
                  best_len: int, chunk: int = 300_000):
    """Shortest word for element W via  W = A[B],  |a| from the front, |b| <= 6.

    Returns a word strictly shorter than `best_len`, or None. `B = argsort(A)[W]`, and
    AINV already holds argsort(A), so the whole front is one fancy-index per chunk.
    """
    N = AINV.shape[0]
    found = None
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        B = AINV[s:e][:, W]                                  # (m, S) candidate remainders
        dep = pz.depth_of(B.astype(np.int64))
        cand = np.nonzero(dep >= 0)[0]
        if cand.size == 0:
            continue
        for c in cand:
            i = s + int(c)
            wa = word_of_index(pz, i, parent, move)
            total = len(wa) + int(dep[c])
            if total >= best_len or (found is not None and total >= len(found)):
                continue
            # descend() returns the word taking state B TO SOLVED, i.e. the word for
            # B^-1: B[perm(m1..md)] = e. The word FOR B is that reversed and inverted.
            # (Same note is in 42_window_reduce.py; omitting it silently yields wrong
            # words, which the perm guard below rejects -- so the symptom is "finds
            # nothing", not "corrupts paths".)
            wb = pz.inv_word(pz.descend(B[c].astype(np.int64)))
            candidate = wa + wb
            if not np.array_equal(pz.perm_of(candidate), W):
                continue                                     # guard: never trust the join
            found = candidate
            best_len = total
    return found


def self_test(pz: Puzzle, AINV, parent, move, n: int = 8) -> bool:
    """A random word of length L must come back with a word of length <= L."""
    rng = np.random.default_rng(0)
    ok = True
    for t in range(n):
        L = int(rng.integers(8, 12))
        w = rng.integers(0, pz.n_gen, size=L).tolist()
        W = pz.perm_of(w)
        got = solve_element(pz, W, AINV, parent, move, best_len=L + 1)
        if got is None:
            # Any element of a length-L word is at distance <= L, and the front+ball
            # reach is front_depth+6, so a miss here at L <= reach is a BUG, not a
            # legitimately-out-of-range element.
            print(f"  case {t}: len {L} -> NO certificate  <-- FAIL (within reach)")
            ok = False
            break
        if not np.array_equal(pz.perm_of(got), W):
            print(f"  case {t}: WRONG word"); ok = False; break
        print(f"  case {t}: len {L} -> {len(got)}  {'OK' if len(got) <= L else 'LONGER?!'}")
        if len(got) > L:
            ok = False; break
    print(f"self-test: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--front-depth", type=int, default=5, help="5 => radius 11")
    ap.add_argument("--min-window", type=int, default=12)
    ap.add_argument("--max-window", type=int, default=16)
    ap.add_argument("--max-queries", type=int, default=500)
    ap.add_argument("--endgame-cuts", action="store_true",
                    help="idea 4: only windows whose END is the path end (route INTO the "
                         "endgame), which is where the last real gains came from")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=DATA)
    args = ap.parse_args()

    pz = Puzzle(args.data_dir)
    print(f"building front ball to depth {args.front_depth}...", flush=True)
    P, parent, move, lvl = build_ball(pz, args.front_depth)
    AINV = np.argsort(P.astype(np.int64), axis=1).astype(np.uint8)
    print(f"front: {AINV.shape[0]:,} elements")

    if args.self_test:
        return 0 if self_test(pz, AINV, parent, move) else 1
    if not args.target:
        print("--target required"); return 2

    rows = {}
    with io.open(args.target, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rows[int(r["initial_state_id"])] = [pz.idx[m] for m in r["path"].strip().split(".")]
    print(f"target: {args.target.name}  {len(rows)} pids, {sum(map(len, rows.values())):,} moves")

    # ---- collect windows, canonicalise, DEDUP (solve each orbit once), rank
    t0 = time.time()
    orbit = {}                                    # key -> (best_len, [(pid, i, j, k, inv)])
    for pid, w in rows.items():
        L = len(w)
        for i in range(L):
            if args.endgame_cuts and i + args.max_window < L:
                continue
            Pm = pz.ident.copy()
            for j in range(i, min(L, i + args.max_window)):
                Pm = Pm[pz.gen[w[j]]]
                ln = j - i + 1
                if ln < args.min_window:
                    continue
                key, k, inv = pz.canonical(Pm)
                rec = orbit.setdefault(key, [ln, []])
                rec[0] = min(rec[0], ln)
                rec[1].append((pid, i, j + 1, k, inv))
    print(f"windows: {sum(len(v[1]) for v in orbit.values()):,} occurrences, "
          f"{len(orbit):,} distinct orbits  ({time.time()-t0:.0f}s)")

    ranked = sorted(orbit.items(), key=lambda kv: (-kv[1][0], -len(kv[1][1])))
    ranked = ranked[:args.max_queries]
    print(f"querying top {len(ranked):,} orbits by (length, occurrences)...", flush=True)

    improved, saved_total = {}, 0
    t0 = time.time()
    for n, (key, (blen, occ)) in enumerate(ranked):
        pid, i, j, k, inv = occ[0]
        W = pz.perm_of(rows[pid][i:j])
        got = solve_element(pz, W, AINV, parent, move, best_len=blen)
        if got is not None:
            improved[key] = got
            saved_total += blen - len(got)
            print(f"  HIT orbit len {blen} -> {len(got)} "
                  f"(x{len(occ)} occurrences)", flush=True)
        if (n + 1) % 25 == 0:
            print(f"  {n+1}/{len(ranked)} queried, {len(improved)} hits, "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"\n{len(improved)} orbits improved")

    if not improved:
        print("no shortening found")
        return 0

    # ---- apply, longest-saving first per row, re-verifying every splice
    for key, canon_word in improved.items():
        for (pid, i, j, k, inv) in orbit[key][1]:
            w = rows[pid]
            if j > len(w):
                continue
            W = pz.perm_of(w[i:j])
            repl = pz.transport(canon_word, k, inv)
            if len(repl) >= j - i or not np.array_equal(pz.perm_of(repl), W):
                continue
            rows[pid] = w[:i] + repl + w[j:]
    total = sum(map(len, rows.values()))
    print(f"new total: {total:,}")

    if args.out:
        with io.open(args.out, "w", encoding="utf-8", newline="") as fh:
            wr = csv.writer(fh); wr.writerow(["initial_state_id", "path"])
            for pid in sorted(rows):
                wr.writerow([pid, ".".join(pz.names[m] for m in rows[pid])])
        print(f"wrote {args.out} -- RUN THE REPLAY VERIFIER before submitting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
