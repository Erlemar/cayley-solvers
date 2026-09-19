"""Radius-13 meet-in-the-middle: front B6 (27.8M perms) x the d<=7 table.

    d(w) <= 13   iff   exists a in B6  with  a^-1 . w  in B7

This is the next band after `61_mitm_gpu.py` (front B5, radius 12). It is 15.7x dearer
per query -- 27.8M probes instead of 1.77M -- and needs the front as ACTUAL PERMUTATIONS
(2.44 GB), not just the hashes the endgame npz stores. That is the whole reason it was
not done first.

WHAT EACH BAND PROVES (the distinction that matters when reading the output):
  * window length L <= reach=13 -> the join returns the OPTIMAL word, so "0 saved" means
    every such window is EXACTLY GEODESIC. Radius 12 already proved this for L <= 12, so
    the new information here is L = 13.
  * window length L >= 14      -> only certifies d <= 13, so "0 saved" means d > 13 and
    the window may still be non-geodesic with d in [14, L-1].

TRIANGLE-INEQUALITY STRUCTURE. Every window of length <= 12 is already known geodesic,
so for any window we test with L >= 14 we have d(w) >= 13, hence
d(a^-1 w) >= d(w) - |a| >= 13 - 6 = 7. A hit therefore requires |a| = 6 AND depth
exactly 7 -- the shallow part of the front cannot contribute. `--shell-only` exploits
this and drops the front from 27.8M to the 26.0M depth-6 shell; it is exact for L >= 14
and must be OFF when testing L <= 13.

BUILD ONCE, REUSE. `--build-front` writes `b6_front.npz` (~2.6 GB: perms uint8, parent
int32, move int8). The depth-6 expansion is 39.8M children, which is why the frontier is
walked in chunks -- materialising it whole is ~28 GB and is what OOMs the naive BFS in
59/61.

    python tetraminx/scripts/64_mitm_r13.py --build-front --front-out b6_front.npz
    python tetraminx/scripts/64_mitm_r13.py --self-test --front b6_front.npz
    python tetraminx/scripts/64_mitm_r13.py --front b6_front.npz \\
        --target tetraminx/submissions/FINAL_tetraminx_28359.csv \\
        --min-window 13 --max-window 16 --out tetraminx/submissions/mitm13.csv
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
    def __init__(self, data_dir: Path, device: str, load_table: bool = True):
        info = json.loads((data_dir / "puzzle_info.json").read_text(encoding="utf-8"))
        self.names = list(info["generators"].keys())
        self.gen = np.array([info["generators"][n] for n in self.names], dtype=np.int64)
        self.n_gen, self.S = self.gen.shape
        self.ident = np.arange(self.S, dtype=np.int64)
        self.idx = {n: i for i, n in enumerate(self.names)}
        assert np.array_equal(np.array(info["central_state"], dtype=np.int64), self.ident)
        self.inv_move = np.zeros(self.n_gen, dtype=np.int64)
        for i in range(self.n_gen):
            for j in range(self.n_gen):
                if np.array_equal(self.gen[i][self.gen[j]], self.ident):
                    self.inv_move[i] = j
                    break
        self.device = device
        z = np.load(data_dir / "bfs_endgame_d7.npz")
        self.ZT = z["ztab"].astype(np.int64)
        self.max_depth = int(z["max_depth"])
        if load_table:
            H, D = z["hashes"], z["depths"]
            print(f"d7 table: {H.size:,} states, {H.nbytes / 2**30:.2f} GiB", flush=True)
            self.H_np, self.D_np = H, D
            self.H = torch.from_numpy(H).to(device)
            self.D = torch.from_numpy(D).to(device)
            self.ZTg = torch.from_numpy(self.ZT).to(device)

    def perm_of(self, word):
        P = self.ident
        for m in word:
            P = P[self.gen[m]]
        return P

    def inv_word(self, word):
        return [int(self.inv_move[m]) for m in reversed(word)]

    def zhash(self, states: np.ndarray) -> np.ndarray:
        h = np.zeros(states.shape[0], dtype=np.int64)
        for i in range(self.S):
            h ^= self.ZT[i][states[:, i]]
        return h

    def depth_np(self, states: np.ndarray) -> np.ndarray:
        h = self.zhash(states)
        pos = np.searchsorted(self.H_np, h)
        np.clip(pos, 0, self.H_np.size - 1, out=pos)
        return np.where(self.H_np[pos] == h, self.D_np[pos], -1).astype(np.int64)

    def descend(self, state: np.ndarray):
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

    def depth_gpu(self, B: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(B.shape[0], dtype=torch.int64, device=B.device)
        for i in range(self.S):
            h ^= self.ZTg[i][B[:, i].long()]
        pos = torch.searchsorted(self.H, h)
        pos.clamp_(0, self.H.numel() - 1)
        return torch.where(self.H[pos] == h, self.D[pos].long(), torch.full_like(h, -1))


def build_front(pz: Puzzle, depth: int, out: Path, chunk: int = 150_000):
    """BFS to `depth`, walking each frontier in chunks.

    The depth-6 level expands 1,659,416 states into 39,825,984 children. Materialising
    that in one array is ~28 GB as int64; chunking keeps the peak to the survivors.
    """
    t0 = time.time()
    P = pz.ident.astype(np.uint8)[None, :].copy()
    parent = np.array([-1], dtype=np.int32)
    move = np.array([-1], dtype=np.int8)
    seen = pz.zhash(P.astype(np.int64))
    lo = 0
    for d in range(1, depth + 1):
        n_front = P.shape[0] - lo
        keep_s, keep_h, keep_src, keep_mv = [], [], [], []
        for s in range(0, n_front, chunk):
            e = min(n_front, s + chunk)
            cur = P[lo + s: lo + e]                      # (m, S) uint8
            kids = cur[:, pz.gen]                        # (m, n_gen, S) uint8
            m = cur.shape[0]
            kids = kids.reshape(-1, pz.S)
            kh = pz.zhash(kids.astype(np.int64))
            pos = np.searchsorted(seen, kh)
            np.clip(pos, 0, seen.size - 1, out=pos)
            fresh = seen[pos] != kh                      # not at an earlier depth
            if not fresh.any():
                continue
            keep_s.append(kids[fresh])
            keep_h.append(kh[fresh])
            keep_src.append(np.repeat(np.arange(lo + s, lo + e, dtype=np.int32),
                                      pz.n_gen)[fresh])
            keep_mv.append(np.tile(np.arange(pz.n_gen, dtype=np.int8), m)[fresh])
        kids = np.concatenate(keep_s); kh = np.concatenate(keep_h)
        src = np.concatenate(keep_src); mv = np.concatenate(keep_mv)
        del keep_s, keep_h, keep_src, keep_mv
        _, first = np.unique(kh, return_index=True)      # dedup within the level
        first.sort()
        kids, kh, src, mv = kids[first], kh[first], src[first], mv[first]
        lo = P.shape[0]
        P = np.concatenate([P, kids]); parent = np.concatenate([parent, src])
        move = np.concatenate([move, mv])
        seen = np.concatenate([seen, kh]); seen.sort()
        if d < len(BFS_LEVELS):
            assert kids.shape[0] == BFS_LEVELS[d], (
                f"depth {d}: got {kids.shape[0]:,}, expected {BFS_LEVELS[d]:,}")
        print(f"  depth {d}: {kids.shape[0]:,} new (cum {P.shape[0]:,})  "
              f"{time.time()-t0:.0f}s", flush=True)
    depths = np.zeros(P.shape[0], dtype=np.int8)
    i = 1
    for d in range(1, depth + 1):
        depths[i:i + BFS_LEVELS[d]] = d
        i += BFS_LEVELS[d]
    np.savez(out, perms=P, parent=parent, move=move, depths=depths)
    print(f"wrote {out}: {P.shape[0]:,} elements, {time.time()-t0:.0f}s", flush=True)


def word_of_index(i: int, parent, move):
    w = []
    while parent[i] >= 0:
        w.append(int(move[i]))
        i = int(parent[i])
    return w[::-1]


class Joiner:
    N_CANDIDATES = 32

    def __init__(self, pz: Puzzle, front: Path, shell_only: bool, chunk: int):
        z = np.load(front)
        P, self.parent, self.move = z["perms"], z["parent"], z["move"]
        dep = z["depths"]
        self.keep = np.nonzero(dep == dep.max())[0] if shell_only else np.arange(P.shape[0])
        self.flen = dep[self.keep].astype(np.int64)      # |a| == BFS depth of a
        # Inverting 27.8M permutations. argsort would need the array as int64 (18.2 GiB)
        # and then returns int64 indices (another 18.2 GiB); for a PERMUTATION the
        # inverse is just a scatter, AINV[i][P[i][j]] = j, so build it in uint8 chunks.
        n = self.keep.size
        AINV = np.empty((n, pz.S), dtype=np.uint8)
        CH = 2_000_000
        for s in range(0, n, CH):
            e = min(n, s + CH)
            blk = P[self.keep[s:e]]
            vals = np.broadcast_to(np.arange(pz.S, dtype=np.uint8), blk.shape)
            np.put_along_axis(AINV[s:e], blk.astype(np.intp), vals, axis=1)
        del P, z
        self.pz = pz
        self.chunk = chunk
        self.AINV = torch.from_numpy(AINV).to(pz.device)
        self.flen_g = torch.from_numpy(self.flen).to(pz.device)
        self.reach = int(dep.max()) + pz.max_depth
        self.phantoms = 0
        print(f"front: {self.keep.size:,} elements"
              f"{' (depth-6 shell only)' if shell_only else ''}, "
              f"reach = {int(dep.max())} + {pz.max_depth} = {self.reach}, "
              f"{AINV.nbytes / 2**30:.2f} GiB", flush=True)

    def solve(self, W: np.ndarray, best_len: int):
        pz = self.pz
        Wg = torch.from_numpy(W.astype(np.int64)).to(pz.device)
        N = self.AINV.shape[0]
        cand_v, cand_i = [], []
        for s in range(0, N, self.chunk):
            e = min(N, s + self.chunk)
            B = self.AINV[s:e][:, Wg]
            dep = pz.depth_gpu(B)
            tot = torch.where(dep >= 0, self.flen_g[s:e] + dep,
                              torch.full_like(dep, 1 << 30))
            k = min(self.N_CANDIDATES, tot.numel())
            v, i = torch.topk(tot, k, largest=False)
            keep = v < best_len
            if keep.any():
                cand_v.append(v[keep]); cand_i.append(i[keep] + s)
            del B, dep, tot
        if not cand_v:
            return None
        v = torch.cat(cand_v); i = torch.cat(cand_i)
        order = torch.argsort(v)[: self.N_CANDIDATES]
        for total, j in zip(v[order].tolist(), i[order].tolist()):
            if total >= best_len:
                return None
            src = int(self.keep[j])
            wa = word_of_index(src, self.parent, self.move)
            Bi = np.argsort(self.pz.perm_of(wa))[W]      # a^-1 . w, recomputed on CPU
            try:
                wb = pz.inv_word(pz.descend(Bi))
            except (AssertionError, ValueError):
                self.phantoms += 1
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-front", action="store_true")
    ap.add_argument("--front-out", type=Path, default=DATA / "b6_front.npz")
    ap.add_argument("--front", type=Path, default=DATA / "b6_front.npz")
    ap.add_argument("--front-depth", type=int, default=6)
    ap.add_argument("--shell-only", action="store_true",
                    help="keep only the deepest shell -- EXACT for windows >= reach+1, "
                         "WRONG for shorter ones (see the triangle-inequality note)")
    ap.add_argument("--target", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--min-window", type=int, default=13)
    ap.add_argument("--max-window", type=int, default=16)
    ap.add_argument("--pids", type=str, default="")
    ap.add_argument("--chunk", type=int, default=4_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.build_front:
        pz = Puzzle(args.data_dir, "cpu", load_table=False)
        build_front(pz, args.front_depth, args.front_out)
        return 0

    pz = Puzzle(args.data_dir, args.device)
    if args.shell_only and args.min_window <= 13:
        raise SystemExit("--shell-only is only exact for windows >= 14; "
                         "raise --min-window or drop --shell-only")
    jn = Joiner(pz, args.front, args.shell_only, args.chunk)

    if args.self_test:
        rng = np.random.default_rng(0)
        ok = True
        for t in range(6):
            L = int(rng.integers(jn.reach - 2, jn.reach + 1))
            w = rng.integers(0, pz.n_gen, size=L).tolist()
            W = pz.perm_of(w)
            t0 = time.time()
            got = jn.solve(W, best_len=L + 1)
            dt = time.time() - t0
            if got is None or not np.array_equal(pz.perm_of(got), W) or len(got) > L:
                print(f"  case {t}: len {L} -> "
                      f"{'NONE' if got is None else len(got)}  <-- FAIL")
                ok = False
                break
            print(f"  case {t}: len {L} -> {len(got)}  ({dt:.2f}s)")
        print(f"self-test: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    if not args.target:
        raise SystemExit("--target is required unless --build-front / --self-test")

    rows = {p: [pz.idx[m] for m in mv] for p, mv in load_csv(args.target).items()}
    orig = {p: list(v) for p, v in rows.items()}
    if args.pids:
        want = {int(x) for x in args.pids.split(",")}
        rows = {p: v for p, v in rows.items() if p in want}
    print(f"target {args.target.name}: {sum(map(len, rows.values())):,} moves "
          f"over {len(rows)} rows", flush=True)

    nq = saved = 0
    t0 = time.time()
    for n, pid in enumerate(sorted(rows)):
        w = rows[pid]
        i = 0
        while i < len(w):
            hi = min(args.max_window, len(w) - i)
            if hi < args.min_window:
                i += 1
                continue
            hit = False
            for L in range(args.min_window, hi + 1):
                W = pz.perm_of(w[i:i + L])
                nq += 1
                got = jn.solve(W, best_len=L)
                if got is not None:
                    print(f"  pid {pid}: window [{i},{i+L}) {L} -> {len(got)}", flush=True)
                    w[i:i + L] = got
                    saved += L - len(got)
                    hit = True
                    break
            if not hit:
                i += 1
        rows[pid] = w
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(rows)} rows, {nq:,} queries, saved {saved}, "
                  f"{time.time()-t0:.0f}s", flush=True)

    print(f"\n{nq:,} queries, saved {saved} moves  ({time.time()-t0:.0f}s)")
    print(f"phantom hits rejected: {jn.phantoms:,}")

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
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
