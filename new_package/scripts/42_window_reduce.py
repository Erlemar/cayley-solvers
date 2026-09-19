"""Shorten paths by replacing any window whose net effect is inside the BFS table.

For a path s_0 -> ... -> s_L, the word spanning positions [i, j) equals the group
element  w = s_i^-1 . s_j  (with the puzzle-info convention apply(s,g) = s . g).
If w is in the d<=6 table at depth d < j - i, the window can be replaced by the
table's optimal word for w:

    table.descend(w) gives m_1..m_d with  w . m_1 ... m_d = e
    =>  w = inv(m_d) ... inv(m_1)      (reverse + invert)

so splicing that in takes s_i to s_j in d moves instead of j - i.

This is the megaminx "bridge compression" mechanism at window scale. It failed
there on already-minimal community paths and worked on loose ones — our beam
output is loose (2–4 moves above optimal), which is the regime where it pays.
The floor paths are near-optimal, so expect few hits on those.

Every rewritten path is replayed and asserted to still solve.

    python3 tetraminx/scripts/42_window_reduce.py \
        --in tetraminx/submissions/current_best.csv \
        --out tetraminx/submissions/current_best_reduced.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]


class Table:
    def __init__(self, path: Path):
        d = np.load(path, allow_pickle=True)
        self.hashes = d["hashes"]
        self.depths = d["depths"]
        self.ztab = d["ztab"]
        self.max_depth = int(d["max_depth"])

    def lookup(self, states: np.ndarray) -> np.ndarray:
        states = np.atleast_2d(states)
        h = np.zeros(states.shape[0], dtype=np.int64)
        for i in range(states.shape[1]):
            h ^= self.ztab[i][states[:, i]]
        pos = np.clip(np.searchsorted(self.hashes, h), 0, len(self.hashes) - 1)
        out = np.full(states.shape[0], -1, dtype=np.int64)
        hit = self.hashes[pos] == h
        out[hit] = self.depths[pos[hit]]
        return out

    def descend(self, state: np.ndarray, gens: np.ndarray) -> list[int]:
        d = int(self.lookup(state[None, :])[0])
        assert d >= 0
        out: list[int] = []
        cur = state
        while d > 0:
            children = cur[gens]
            cd = self.lookup(children)
            out.append(int(np.nonzero(cd == d - 1)[0][0]))
            cur = children[out[-1]]
            d -= 1
        return out

    def shortest_word(self, w: np.ndarray, gens: np.ndarray,
                      inv_idx: np.ndarray, extend: int) -> list[int] | None:
        """Shortest word equal to the group element `w`, exact whenever
        dist(w) <= max_depth + extend.

        The table only knows distances up to d<=6. But if some short word v makes
        w.v land inside the table at depth d, then w = (w.v) . v^-1, giving a word
        of length d + |v|. So BFS `extend` moves out from w, look every frontier
        state up, and keep the best (d + |v|). extend=3 reaches length 9 for the
        cost of 6,592 lookups; extend=4 reaches 10 for 105,136.
        """
        best: tuple[int, list[int]] | None = None
        d0 = int(self.lookup(w[None, :])[0])
        if d0 >= 0:
            best = (d0, [])
        frontier = w[None, :]
        words: list[list[int]] = [[]]
        for depth in range(1, extend + 1):
            if best is not None and best[0] <= depth:
                break                      # cannot beat what we already have
            nxt = frontier[:, gens.reshape(-1)].reshape(-1, w.shape[0])
            nwords = [wd + [g] for wd in words for g in range(gens.shape[0])]
            ds = self.lookup(nxt)
            hits = np.nonzero(ds >= 0)[0]
            for h in hits:
                total = int(ds[h]) + depth
                if best is None or total < best[0]:
                    best = (total, nwords[h])
            frontier, words = nxt, nwords
        if best is None:
            return None
        total, prefix = best
        # w . prefix = t  =>  w = t . prefix^-1, and t's word is descend(t) inverted
        cur = w
        for g in prefix:
            cur = cur[gens[g]]
        tail = [int(inv_idx[m]) for m in reversed(self.descend(cur, gens))]
        word = tail + [int(inv_idx[g]) for g in reversed(prefix)]
        assert len(word) == total
        return word


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "tetraminx" / "data")
    ap.add_argument("--endgame", type=Path, default=None)
    ap.add_argument("--max-window", type=int, default=14)
    ap.add_argument("--extend", type=int, default=0,
                    help="BFS this many moves out from each window element to reach "
                         "words longer than the table depth (0 = table lookup only). "
                         "Cost per window is ~24^extend lookups, so 2 is usually the "
                         "practical limit; only applied to windows >= --extend-min.")
    ap.add_argument("--extend-min", type=int, default=8,
                    help="only spend --extend effort on windows at least this long")
    args = ap.parse_args()

    info = json.loads((args.data_dir / "puzzle_info.json").read_text(encoding="utf-8"))
    move_names = list(info["generators"].keys())
    name_to_idx = {nm: i for i, nm in enumerate(move_names)}
    gens = np.array([info["generators"][nm] for nm in move_names], dtype=np.int64)
    inv_idx = np.array([name_to_idx[nm[1:] if nm.startswith("-") else "-" + nm]
                        for nm in move_names])
    solved = np.array(info["central_state"], dtype=np.int64)
    n = len(solved)

    table = Table(args.endgame or (args.data_dir / "bfs_endgame.npz"))
    print(f"table: {len(table.hashes):,} states <= d{table.max_depth}")

    with open(args.data_dir / "test.csv", encoding="utf-8", newline="") as f:
        states0 = {int(r["initial_state_id"]):
                   np.array([int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
                   for r in csv.DictReader(f)}

    with open(args.inp, encoding="utf-8", newline="") as f:
        paths = {int(r["initial_state_id"]): [name_to_idx[m] for m in r["path"].split(".")]
                 for r in csv.DictReader(f) if r["path"]}

    def trace(s0: np.ndarray, path: list[int]) -> list[np.ndarray]:
        out = [s0]
        cur = s0
        for m in path:
            cur = cur[gens[m]]
            out.append(cur)
        return out

    total_before = sum(len(p) for p in paths.values())
    n_changed = 0
    saved = 0
    for pid, path in paths.items():
        s0 = states0[pid]
        improved = True
        while improved:
            improved = False
            states = trace(s0, path)
            L = len(path)
            for i in range(L):
                hi = min(L, i + args.max_window)
                for j in range(i + 2, hi + 1):
                    # w = s_i^-1 . s_j
                    inv_si = np.empty(n, dtype=np.int64)
                    inv_si[states[i]] = np.arange(n)
                    w = inv_si[states[j]]
                    word = None
                    d = int(table.lookup(w[None, :])[0])
                    if 0 <= d < j - i:
                        word = [int(inv_idx[m]) for m in reversed(table.descend(w, gens))]
                    elif args.extend > 0 and (j - i) >= args.extend_min:
                        cand = table.shortest_word(w, gens, inv_idx, args.extend)
                        if cand is not None and len(cand) < j - i:
                            word = cand
                    if word is not None:
                        path = path[:i] + word + path[j:]
                        saved += (j - i) - len(word)
                        n_changed += 1
                        improved = True
                        break
                if improved:
                    break
        cur = s0
        for m in path:
            cur = cur[gens[m]]
        assert np.array_equal(cur, solved), f"pid {pid}: reduced path does not solve"
        paths[pid] = path

    total_after = sum(len(p) for p in paths.values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid in sorted(paths):
            w.writerow([pid, ".".join(move_names[m] for m in paths[pid])])

    print(f"{len(paths)} pids: {total_before:,} -> {total_after:,} "
          f"({total_after - total_before:+,} moves, {n_changed} window rewrites)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
