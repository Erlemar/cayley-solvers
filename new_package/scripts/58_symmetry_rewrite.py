"""Symmetry-canonical substring rewriting: mine short words from a whole corpus of
paths, then splice them into a target submission wherever a conjugate element appears.

WHY THIS IS NOT `42_window_reduce.py`. That script looks a window's net element up in
the exact d<=6 BFS table. This one adds two things:

  1. CROSS-ROW MINING. Row A may realise group element `g` in 7 moves while row B
     realises it in 9. Nothing in the exact table knows that -- 7 is not "optimal", just
     shorter than 9 -- so a table lookup can never transfer it. A dictionary mined from
     every path we own can.

  2. 48-FOLD CANONICALISATION. Rows almost never contain the *same* element; they
     contain conjugates. Keying by a canonical form over 24 tetrahedral automorphisms x
     group inversion collapses each orbit to one entry, so a word found anywhere is
     usable everywhere in its orbit.

MEASURED FACT THAT SHAPES THE DESIGN (verified 2026-08-04, 80/80 cases): **the exact
BFS table is closed under all 48.** Distance is invariant under conjugation and
inversion, so if conj_k(g) is in the ball at depth d then so is g, at the same depth.
Canonicalising before a TABLE lookup is therefore a provable no-op -- it cannot find a
single hit that the plain lookup misses. Symmetry earns its keep ONLY in the mined
dictionary. Do not "improve" 42_window_reduce by adding symmetry to it.

Algebra, all verified numerically at import time (`--self-test`):
  * state/word:      apply(s, m) = s[gen[m]]      perm(m1..mk) = ((I[g1])[g2])...
  * inversion:       perm(inv(w)) == perm(w)^-1   with inv(w) = reversed, moves inverted
  * conjugation:     perm(relabel_k(w)) == SYM[k][ perm(w)[ SYM_INV[k] ] ]
    (the OTHER orientation, SYM_INV[k][g[SYM[k]]], is WRONG -- checked, fails)

Every rewritten path is replayed against the original scramble before being written.

    python tetraminx/scripts/58_symmetry_rewrite.py --self-test
    python tetraminx/scripts/58_symmetry_rewrite.py \
        --target tetraminx/submissions/submission_28456.csv \
        --corpus tetraminx/submissions tetraminx/results \
        --max-window 10 --out tetraminx/submissions/rewritten.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "tetraminx" / "data"


# --------------------------------------------------------------------------- algebra
class Algebra:
    def __init__(self, data_dir: Path):
        info = json.loads((data_dir / "puzzle_info.json").read_text(encoding="utf-8"))
        self.names = list(info["generators"].keys())
        self.gen = np.array([info["generators"][n] for n in self.names], dtype=np.int64)
        self.n_gen, self.S = self.gen.shape
        self.ident = np.arange(self.S, dtype=np.int64)
        self.idx = {n: i for i, n in enumerate(self.names)}
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

    def perm_of(self, word) -> np.ndarray:
        P = self.ident.copy()
        for m in word:
            P = P[self.gen[m]]
        return P

    def inv_word(self, word):
        return [int(self.inv_move[m]) for m in reversed(word)]

    def conj(self, g: np.ndarray, k: int) -> np.ndarray:
        """perm(relabel_k(w)) for g = perm(w). Verified; the other orientation is wrong."""
        return self.SYM[k][g[self.SYMI[k]]]

    def canonical(self, g: np.ndarray):
        """-> (key bytes, k, inverted) minimising the byte form over the 48 variants.

        VECTORISED: all 24 conjugates at once via take_along_axis, since
        conj_k(g)[j] = SYM[k][ g[ SYM_INV[k][j] ] ]. A per-k Python loop here costs
        ~100 numpy calls per window and makes corpus mining hours instead of minutes.
        """
        rows = np.empty((2 * self.n_sym, self.S), dtype=np.uint8)
        for half, h in ((0, g), (self.n_sym, np.argsort(g))):
            G = h[self.SYMI]                                   # (24, S)
            rows[half:half + self.n_sym] = np.take_along_axis(self.SYM, G, axis=1)
        # lexicographic argmin over rows: lexsort's LAST key is primary, so feed the
        # columns reversed to make column 0 dominant.
        b = int(np.lexsort(rows[:, ::-1].T)[0])
        return rows[b].tobytes(), b % self.n_sym, b >= self.n_sym

    def transport(self, word, k: int, inverted: bool):
        """Map a word from the canonical frame back to the frame of the query element.

        canonical(g) applies (maybe) inversion, then relabel_k. Undo by inverting the
        MOVE-RELABEL MAP itself -- `argsort(RELAB[k])` -- rather than looking up the
        inverse symmetry index. SYM's composition convention is not `SYM[a][SYM[b]]`
        (checked: no b satisfies it for k=9), so index-chasing there is a trap; the
        relabel map is a plain permutation of 24 move ids and inverts unambiguously.
        """
        back = self.relab_inv[k]
        out = [int(back[m]) for m in word]
        return self.inv_word(out) if inverted else out


def self_test(alg: Algebra, n: int = 200) -> bool:
    rng = np.random.default_rng(0)
    ok = True
    for _ in range(n):
        w = rng.integers(0, alg.n_gen, size=int(rng.integers(1, 10))).tolist()
        P = alg.perm_of(w)
        if not np.array_equal(P[alg.perm_of(alg.inv_word(w))], alg.ident):
            print("FAIL inversion"); ok = False; break
        k = int(rng.integers(0, alg.n_sym))
        if not np.array_equal(alg.perm_of([int(alg.RELAB[k][m]) for m in w]), alg.conj(P, k)):
            print("FAIL conjugation"); ok = False; break
        key, kk, inv = alg.canonical(P)
        back = alg.transport(
            [int(alg.RELAB[kk][m]) for m in (alg.inv_word(w) if inv else w)], kk, inv)
        if not np.array_equal(alg.perm_of(back), P):
            print("FAIL transport round-trip"); ok = False; break
    print(f"self-test ({n} cases): {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- corpus
def load_paths_csv(path: Path):
    out = {}
    with io.open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            s = (r.get("path") or "").strip()
            if s:
                out[int(r["initial_state_id"])] = s.split(".")
    return out


def load_paths_json(path: Path):
    out = {}
    try:
        recs = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return out
    if not isinstance(recs, list):
        return out
    for r in recs:
        if isinstance(r, dict) and r.get("found") and r.get("verify_ok") and r.get("path"):
            p = r["path"].split(".")
            pid = int(r["pid"])
            if pid not in out or len(p) < len(out[pid]):
                out[pid] = p
    return out


def gather_corpus(dirs, alg: Algebra):
    """-> list of move-index paths. Keeps LONG paths too: a longer row can still hold a
    shorter WORD for some sub-element, which is the whole point of cross-row mining."""
    seen_files, uniq = 0, set()
    for d in dirs:
        d = Path(d)
        files = ([d] if d.is_file()
                 else [Path(p) for p in glob.glob(str(d / "**" / "*.csv"), recursive=True)]
                      + [Path(p) for p in glob.glob(str(d / "**" / "*.json"), recursive=True)])
        for f in files:
            got = load_paths_csv(f) if f.suffix == ".csv" else load_paths_json(f)
            if not got:
                continue
            seen_files += 1
            for mv in got.values():
                try:
                    idxs = tuple(alg.idx[m] for m in mv)
                except KeyError:
                    break          # not this puzzle's alphabet -- skip the whole file
                uniq.add(idxs)
    # Dedup across files: 148 result files overwhelmingly repeat the same rows, and a
    # duplicated path contributes nothing new to a shortest-word dictionary.
    paths = [list(t) for t in uniq]
    return paths, seen_files


# --------------------------------------------------------------------------- mining
def mine_dictionary(paths, alg: Algebra, max_window: int, progress: int = 200):
    """canonical key -> (length, word in canonical frame).

    Windows are walked incrementally: extending [i, j) to [i, j+1) is one gather, so a
    path of length L costs O(L * max_window) gathers rather than O(L^2 * len) products.
    """
    book: dict[bytes, tuple[int, list[int]]] = {}
    t0 = time.time()
    for n, w in enumerate(paths):
        L = len(w)
        for i in range(L):
            P = alg.ident.copy()
            hi = min(L, i + max_window)
            for j in range(i, hi):
                P = P[alg.gen[w[j]]]
                ln = j - i + 1
                if ln < 2:
                    continue          # single moves are already minimal
                key, k, inv = alg.canonical(P)
                cur = book.get(key)
                if cur is None or ln < cur[0]:
                    sub = w[i:j + 1]
                    canon = [int(alg.RELAB[k][m]) for m in (alg.inv_word(sub) if inv else sub)]
                    book[key] = (ln, canon)
        if progress and (n + 1) % progress == 0:
            print(f"  mined {n+1}/{len(paths)} paths, {len(book):,} orbits, "
                  f"{time.time()-t0:.0f}s", flush=True)
    return book


# --------------------------------------------------------------------------- rewrite
def rewrite_path(word, alg: Algebra, book, max_window: int):
    """Greedy longest-saving-first pass. Returns (new_word, saved)."""
    best = None
    L = len(word)
    for i in range(L):
        P = alg.ident.copy()
        hi = min(L, i + max_window)
        for j in range(i, hi):
            P = P[alg.gen[word[j]]]
            ln = j - i + 1
            if ln < 3:
                continue
            key, k, inv = alg.canonical(P)
            hit = book.get(key)
            if hit is None or hit[0] >= ln:
                continue
            repl = alg.transport(hit[1], k, inv)
            if not np.array_equal(alg.perm_of(repl), P):
                continue                                  # transport guard
            save = ln - len(repl)
            if save > 0 and (best is None or save > best[0]):
                best = (save, i, j + 1, repl)
    if best is None:
        return word, 0
    save, i, j, repl = best
    return word[:i] + repl + word[j:], save


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path)
    ap.add_argument("--corpus", nargs="+", default=[])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-window", type=int, default=10)
    ap.add_argument("--max-paths", type=int, default=0, help="cap corpus size (0 = all)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=DATA)
    args = ap.parse_args()

    alg = Algebra(args.data_dir)
    if args.self_test:
        return 0 if self_test(alg) else 1
    if not self_test(alg, 60):
        return 1
    if not args.target:
        print("--target required"); return 2

    tgt = load_paths_csv(args.target)
    tgt_idx = {p: [alg.idx[m] for m in mv] for p, mv in tgt.items()}
    print(f"target: {args.target.name}  {len(tgt_idx)} pids, "
          f"{sum(len(v) for v in tgt_idx.values()):,} moves")

    paths, nfiles = gather_corpus(args.corpus, alg)
    if args.max_paths:
        paths.sort(key=len, reverse=True)   # longest first: most windows, most slack
        paths = paths[:args.max_paths]
    print(f"corpus: {nfiles} files, {len(paths):,} paths, "
          f"{sum(len(p) for p in paths):,} moves")

    print(f"mining windows up to {args.max_window}...", flush=True)
    book = mine_dictionary(paths, alg, args.max_window)
    print(f"dictionary: {len(book):,} canonical orbits")

    total_saved, changed = 0, []
    for pid in sorted(tgt_idx):
        w = tgt_idx[pid]
        while True:
            w, saved = rewrite_path(w, alg, book, args.max_window)
            if not saved:
                break
            total_saved += saved
            changed.append((pid, saved))
        tgt_idx[pid] = w
    print(f"\nsaved {total_saved} moves over {len(set(p for p, _ in changed))} pids")
    for pid, s in changed[:20]:
        print(f"  pid {pid}: -{s}")

    if args.out and total_saved:
        with io.open(args.out, "w", encoding="utf-8", newline="") as fh:
            wr = csv.writer(fh); wr.writerow(["initial_state_id", "path"])
            for pid in sorted(tgt_idx):
                wr.writerow([pid, ".".join(alg.names[m] for m in tgt_idx[pid])])
        print(f"wrote {args.out} ({sum(len(v) for v in tgt_idx.values()):,} moves)")
        print("NOT verified here -- run the replay verifier before submitting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
