"""GPU meet-in-the-middle window shortening, radius = front_depth + table_depth.

WHY THIS EXISTS. `59_mitm_rewrite.py` does the same join in numpy at ~4 s per query,
which caps a sweep at ~120 windows. That sample returned 0 hits, but 120 of ~15k proves
very little. This is the same algebra on the GPU at ~50 ms per query, so the FULL sweep
is minutes instead of a day -- and it can use the d7 table, which lifts the certified
radius from 11 to 12.

    d(w) <= front + T   iff   exists a in B_front  with  a^-1 . w  in B_T

With front=5 and the d<=7 table (433,385,579 states) that certifies radius 12. The
competitor's exhaustive sweep stopped at radius 10 and ours at 11, so 12 is the first
genuinely untested band. A radius-12 certificate can only SHORTEN a window of length
>= 13, which is why --min-window defaults there.

ALGEBRA (unchanged from 59, convention apply(s, m) = s[gen[m]]):
    W = A[B]  =>  B = argsort(A)[W]
so the front is stored INVERTED and one column-gather `AINV[:, W]` yields every
candidate remainder at once. Solved == identity, so a state IS its group element.

The 48-fold canonicalisation of 59 is deliberately NOT used: it exists to dedup
repeated window elements, and on this corpus it was measured to collapse nothing
(14,727 windows -> 14,727 distinct orbits), so it is pure cost here.

Every replacement is checked as a full 88-point permutation before it is accepted, and
each rewritten row is replayed from its original scramble at the end.

    python tetraminx/scripts/61_mitm_gpu.py --self-test --table d7
    python tetraminx/scripts/61_mitm_gpu.py \
        --target tetraminx/submissions/FINAL_tetraminx_28455.csv \
        --table d7 --front-depth 5 --min-window 13 --max-window 16 \
        --out tetraminx/submissions/mitm12.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "tetraminx" / "data"

BFS_LEVELS = [1, 24, 408, 6592, 105136, 1659416, 26008172]


class Puzzle:
    def __init__(self, data_dir: Path, table: str, device: str):
        info = json.loads((data_dir / "puzzle_info.json").read_text(encoding="utf-8"))
        self.names = list(info["generators"].keys())
        self.gen = np.array([info["generators"][n] for n in self.names], dtype=np.int64)
        self.n_gen, self.S = self.gen.shape
        self.ident = np.arange(self.S, dtype=np.int64)
        self.idx = {n: i for i, n in enumerate(self.names)}
        assert np.array_equal(np.array(info["central_state"], dtype=np.int64), self.ident), \
            "this script assumes solved == identity so that a state IS its group element"
        self.inv_move = np.zeros(self.n_gen, dtype=np.int64)
        for i in range(self.n_gen):
            for j in range(self.n_gen):
                if np.array_equal(self.gen[i][self.gen[j]], self.ident):
                    self.inv_move[i] = j
                    break
        self.device = device
        npz = "bfs_endgame.npz" if table == "d6" else "bfs_endgame_d7.npz"
        t0 = time.time()
        z = np.load(data_dir / npz)
        H = z["hashes"]
        D = z["depths"]
        self.ZT = z["ztab"].astype(np.int64)
        self.max_depth = int(z["max_depth"])
        print(f"table {npz}: {H.size:,} states, max_depth {self.max_depth}, "
              f"{H.nbytes / 2**30:.2f} GiB  ({time.time()-t0:.0f}s)", flush=True)
        t0 = time.time()
        self.H = torch.from_numpy(H).to(device)
        self.D = torch.from_numpy(D).to(device)
        self.ZTg = torch.from_numpy(self.ZT).to(device)
        print(f"table -> {device} in {time.time()-t0:.0f}s "
              f"({torch.cuda.memory_allocated(device) / 2**30:.2f} GiB allocated)", flush=True)
        # numpy mirrors for the cheap single-state calls (descend, guards)
        self.H_np, self.D_np = H, D

    # ---- basics (CPU)
    def perm_of(self, word):
        P = self.ident.copy()
        for m in word:
            P = P[self.gen[m]]
        return P

    def inv_word(self, word):
        return [int(self.inv_move[m]) for m in reversed(word)]

    def depth_np(self, states: np.ndarray) -> np.ndarray:
        h = np.zeros(states.shape[0], dtype=np.int64)
        for i in range(self.S):
            h ^= self.ZT[i][states[:, i]]
        pos = np.searchsorted(self.H_np, h)
        np.clip(pos, 0, self.H_np.size - 1, out=pos)
        return np.where(self.H_np[pos] == h, self.D_np[pos], -1).astype(np.int64)

    def descend(self, state: np.ndarray):
        """Optimal word taking `state` to solved, using the exact table."""
        d = int(self.depth_np(state[None, :])[0])
        if d < 0:
            raise ValueError("state not in the ball")
        word, cur = [], state.copy()
        while d > 0:
            kids = cur[self.gen]
            dk = self.depth_np(kids)
            nxt = int(np.argmax(dk == d - 1))
            assert dk[nxt] == d - 1, "no descending child -- table inconsistent"
            word.append(nxt)
            cur = kids[nxt]
            d -= 1
        return word

    # ---- GPU depth probe
    def depth_gpu(self, B: torch.Tensor) -> torch.Tensor:
        """(N, S) uint8 states -> (N,) int64 depth, -1 outside the table."""
        h = torch.zeros(B.shape[0], dtype=torch.int64, device=B.device)
        for i in range(self.S):
            h ^= self.ZTg[i][B[:, i].long()]
        pos = torch.searchsorted(self.H, h)
        pos.clamp_(0, self.H.numel() - 1)
        return torch.where(self.H[pos] == h, self.D[pos].long(),
                           torch.full_like(h, -1))


def build_front(pz: Puzzle, depth: int, verbose=True):
    """BFS from identity to `depth`. Returns (P uint8 (N,S), parent int32, move int8)."""
    t0 = time.time()
    P = pz.ident.astype(np.uint8)[None, :].copy()
    parent = np.array([-1], dtype=np.int32)
    move = np.array([-1], dtype=np.int8)

    def zhash(states):
        h = np.zeros(states.shape[0], dtype=np.int64)
        for i in range(pz.S):
            h ^= pz.ZT[i][states[:, i]]
        return h

    seen = zhash(P.astype(np.int64))
    lo = 0
    for d in range(1, depth + 1):
        cur = P[lo:].astype(np.int64)
        kids = cur[:, pz.gen].reshape(-1, pz.S)
        n = cur.shape[0]
        kh = zhash(kids)
        fresh = ~np.isin(kh, seen)
        kids, kh = kids[fresh], kh[fresh]
        src = np.repeat(np.arange(lo, lo + n, dtype=np.int32), pz.n_gen)[fresh]
        mv = np.tile(np.arange(pz.n_gen, dtype=np.int8), n)[fresh]
        _, first = np.unique(kh, return_index=True)
        first.sort()
        kids, kh, src, mv = kids[first], kh[first], src[first], mv[first]
        lo = P.shape[0]
        P = np.concatenate([P, kids.astype(np.uint8)], axis=0)
        parent = np.concatenate([parent, src])
        move = np.concatenate([move, mv])
        seen = np.concatenate([seen, kh])
        seen.sort()
        if d < len(BFS_LEVELS):
            assert kids.shape[0] == BFS_LEVELS[d], (
                f"front depth {d}: got {kids.shape[0]:,}, expected {BFS_LEVELS[d]:,}")
        if verbose:
            print(f"  front depth {d}: {kids.shape[0]:,} new (cum {P.shape[0]:,})"
                  f"  {time.time()-t0:.0f}s", flush=True)
    return P, parent, move


def word_of_index(i: int, parent, move):
    w = []
    while parent[i] >= 0:
        w.append(int(move[i]))
        i = int(parent[i])
    return w[::-1]


class Joiner:
    """Holds the inverted front on the GPU and answers `shortest word for W`."""

    def __init__(self, pz: Puzzle, front_depth: int):
        P, self.parent, self.move = build_front(pz, front_depth)
        self.pz = pz
        self.flen = np.array([len(word_of_index(i, self.parent, self.move))
                              for i in range(P.shape[0])], dtype=np.int64)
        AINV = np.argsort(P.astype(np.int64), axis=1).astype(np.uint8)
        self.AINV = torch.from_numpy(AINV).to(pz.device)
        self.flen_g = torch.from_numpy(self.flen).to(pz.device)
        self.reach = front_depth + pz.max_depth
        self.phantoms = 0
        print(f"front: {P.shape[0]:,} elements, reach = {front_depth} + "
              f"{pz.max_depth} = {self.reach}", flush=True)

    # A 64-bit Zobrist over a 433M-entry table false-positives about 2.3e-11 per probe,
    # and one query is 1.77M probes -- so a full sweep sees a handful of PHANTOM hits:
    # states the table reports at some depth that are not actually in the ball. The
    # phantom is deterministic (same hash on CPU and GPU), so it cannot be caught by
    # re-checking the depth; it surfaces as `descend` finding no child one level down.
    # Hence: walk candidates in increasing total length and skip any that fails to
    # descend or fails the permutation guard, rather than trusting the argmin.
    N_CANDIDATES = 32

    def solve(self, W: np.ndarray, best_len: int):
        """Shortest VERIFIED word for element W strictly under `best_len`, or None."""
        pz = self.pz
        Wg = torch.from_numpy(W.astype(np.int64)).to(pz.device)
        B = self.AINV[:, Wg]                       # (N, S) candidate remainders
        dep = pz.depth_gpu(B)
        tot = torch.where(dep >= 0, self.flen_g + dep,
                          torch.full_like(dep, 1 << 30))
        k = min(self.N_CANDIDATES, tot.numel())
        vals, idx = torch.topk(tot, k, largest=False)
        vals = vals.tolist()
        idx = idx.tolist()
        for total, i in zip(vals, idx):
            if total >= best_len:
                return None                        # sorted, so nothing later can win
            wa = word_of_index(i, self.parent, self.move)
            Bi = B[i].cpu().numpy().astype(np.int64)
            try:
                # descend() returns the word taking Bi TO solved, i.e. the word for
                # Bi^-1. The word FOR Bi is that reversed and inverted -- omitting this
                # yields wrong words which the guard below rejects, so the symptom is a
                # silent "finds nothing" (it cost 59_mitm_rewrite.py a full null run).
                wb = pz.inv_word(pz.descend(Bi))
            except (AssertionError, ValueError):
                self.phantoms += 1                 # hash collision; try the next one
                continue
            cand = wa + wb
            if len(cand) != total or not np.array_equal(pz.perm_of(cand), W):
                self.phantoms += 1
                continue
            return cand
        return None


def load_csv(p: Path):
    out = {}
    with io.open(p, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            s = (r.get("path") or "").strip()
            if s:
                out[int(r["initial_state_id"])] = s.split(".")
    return out


def self_test(pz: Puzzle, jn: Joiner, n: int = 10) -> bool:
    rng = np.random.default_rng(0)
    ok = True
    for t in range(n):
        L = int(rng.integers(jn.reach - 3, jn.reach + 1))
        w = rng.integers(0, pz.n_gen, size=L).tolist()
        W = pz.perm_of(w)
        t0 = time.time()
        got = jn.solve(W, best_len=L + 1)
        dt = time.time() - t0
        if got is None:
            print(f"  case {t}: len {L} -> NO certificate  <-- FAIL (within reach)")
            ok = False
            break
        if not np.array_equal(pz.perm_of(got), W) or len(got) > L:
            print(f"  case {t}: BAD word ({len(got)} vs {L})")
            ok = False
            break
        print(f"  case {t}: len {L} -> {len(got)}  ({dt*1000:.0f} ms)")
    print(f"self-test: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--table", choices=["d6", "d7"], default="d7")
    ap.add_argument("--front-depth", type=int, default=5)
    ap.add_argument("--min-window", type=int, default=13)
    ap.add_argument("--max-window", type=int, default=16)
    ap.add_argument("--max-queries", type=int, default=0, help="0 = no cap")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    pz = Puzzle(args.data_dir, args.table, args.device)
    jn = Joiner(pz, args.front_depth)

    if args.self_test:
        return 0 if self_test(pz, jn) else 1
    if not args.target:
        raise SystemExit("--target is required unless --self-test")

    rows = {p: [pz.idx[m] for m in mv] for p, mv in load_csv(args.target).items()}
    orig = {p: list(v) for p, v in rows.items()}
    total0 = sum(map(len, rows.values()))
    print(f"target {args.target.name}: {total0:,} moves over {len(rows)} rows")
    if args.min_window <= jn.reach:
        print(f"windows {args.min_window}-{args.max_window} vs reach {jn.reach}: lengths "
              f"<= {jn.reach} get an OPTIMAL answer (geodesic test), longer ones a "
              f"shortening iff d <= {jn.reach}")

    nq = saved = 0
    t0 = time.time()
    for n, pid in enumerate(sorted(rows)):
        w = rows[pid]
        i = 0
        while i < len(w):
            hi = min(args.max_window, len(w) - i)
            if hi < args.min_window:                # no window of the requested size left
                i += 1
                continue
            hit = False
            # No length is a priori hopeless: the join returns words up to `reach`, so it
            # beats ANY window whose true distance is under its own length. For L <= reach
            # that makes the answer optimal (this is how "every window <=6 is geodesic"
            # was measured); for L > reach it is a genuine shortening whenever d <= reach.
            for L in range(args.min_window, hi + 1):
                if args.max_queries and nq >= args.max_queries:
                    break
                W = pz.perm_of(w[i:i + L])
                nq += 1
                got = jn.solve(W, best_len=L)
                if got is not None:
                    print(f"  pid {pid}: window [{i},{i+L}) {L} -> {len(got)}", flush=True)
                    w[i:i + L] = got
                    saved += L - len(got)
                    hit = True
                    break
            if args.max_queries and nq >= args.max_queries:
                break
            if not hit:                             # on a hit, retry the same start
                i += 1
        rows[pid] = w
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(rows)} rows, {nq:,} queries, saved {saved}, "
                  f"{time.time()-t0:.0f}s", flush=True)
        if args.max_queries and nq >= args.max_queries:
            print(f"  stopped at the --max-queries cap ({nq:,})")
            break

    print(f"\n{nq:,} queries, saved {saved} moves, total "
          f"{sum(map(len, rows.values())):,}  ({time.time()-t0:.0f}s)")
    print(f"phantom hits rejected (64-bit Zobrist collisions): {jn.phantoms:,}")

    if saved and args.out:
        starts = {}
        with io.open(args.data_dir / "test.csv", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                starts[int(r["initial_state_id"])] = np.array(
                    [int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
        bad = 0
        for pid, w in rows.items():
            s = starts[pid].copy()
            for m in w:
                s = s[pz.gen[m]]
            if not np.array_equal(s, pz.ident):
                bad += 1
                rows[pid] = orig[pid]
        print(f"replay: {bad} rows failed and were reverted")
        with io.open(args.out, "w", encoding="utf-8", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["initial_state_id", "path"])
            for pid in sorted(rows):
                wr.writerow([pid, ".".join(pz.names[m] for m in rows[pid])])
        print(f"wrote {args.out}: {sum(map(len, rows.values())):,} moves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
