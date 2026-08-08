"""Bounded Knuth-Bendix completion over the move monoid, as a length-reducing rewriter.

WHAT IS NEW HERE vs 58_symmetry_rewrite.py. That script MINES relations: it can only
use a shortening it has literally observed somewhere in the corpus (it found 61.82% of
target windows in an 836k-orbit dictionary and 0 of them shorter). Knuth-Bendix instead
DERIVES relations it has never seen, by resolving overlaps between known rules -- if
u1 -> v1 and u2 -> v2 overlap in a word w, then w reduces two ways, and the difference
is a new rule. That is the one part of "relation mining" the corpus approach cannot do.

THE SYSTEM. Rules are (lhs -> rhs) with perm(lhs) == perm(rhs) and rhs shortlex-smaller.
Seeding is exhaustive rather than heuristic: enumerate EVERY word up to --seed-len,
bucket by group element, and map each non-minimal word to its bucket's shortlex-least
representative. That makes the system complete (hence confluent) up to --seed-len by
construction, so any rule completion later adds is genuinely derived, not overlooked.

WHAT TO EXPECT, STATED UP FRONT. Every window of length <=6 in our submissions was
measured geodesic, and 61_mitm_gpu.py certifies radius 12 exhaustively. A rewriting
rule can only pay if its lhs is LONGER than what those sweeps already cover -- so the
number to look at in the output is not "moves saved" but the rule-length histogram:
completion is interesting only if it produces reducing rules with |lhs| > 13. This
script is built to make that measurable rather than to be believed.

    python tetraminx/scripts/62_knuth_bendix.py --self-test
    python tetraminx/scripts/62_knuth_bendix.py --seed-len 4 --max-rule 10 \\
        --target tetraminx/submissions/FINAL_tetraminx_28455.csv \\
        --out tetraminx/submissions/kb_out.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "tetraminx" / "data"


class Pz:
    def __init__(self, d: Path):
        info = json.loads((d / "puzzle_info.json").read_text(encoding="utf-8"))
        self.names = list(info["generators"].keys())
        self.gen = np.array([info["generators"][n] for n in self.names], dtype=np.int64)
        self.n_gen, self.S = self.gen.shape
        self.ident = np.arange(self.S, dtype=np.int64)
        self.idx = {n: i for i, n in enumerate(self.names)}

    def perm_of(self, word) -> np.ndarray:
        P = self.ident
        for m in word:
            P = P[self.gen[m]]
        return P

    def key(self, word) -> bytes:
        return self.perm_of(word).astype(np.uint8).tobytes()


def shortlex(w) -> tuple:
    return (len(w), w)


class System:
    """A shortlex-reducing rewriting system over move words."""

    def __init__(self, pz: Pz):
        self.pz = pz
        self.rules: dict[tuple, tuple] = {}
        self.by_len: dict[int, list[tuple]] = defaultdict(list)
        self.max_lhs = 0

    def add(self, lhs: tuple, rhs: tuple) -> bool:
        if lhs == rhs or lhs in self.rules:
            return False
        if shortlex(rhs) >= shortlex(lhs):
            lhs, rhs = rhs, lhs
            if lhs == rhs or lhs in self.rules:
                return False
        self.rules[lhs] = rhs
        self.by_len[len(lhs)].append(lhs)
        self.max_lhs = max(self.max_lhs, len(lhs))
        return True

    def reduce(self, w: tuple, cap: int = 10_000) -> tuple:
        """Leftmost-longest reduction to normal form."""
        w = tuple(w)
        for _ in range(cap):
            fired = False
            i = 0
            while i < len(w):
                hi = min(self.max_lhs, len(w) - i)
                for L in range(hi, 0, -1):
                    sub = w[i:i + L]
                    r = self.rules.get(sub)
                    if r is not None:
                        w = w[:i] + r + w[i + L:]
                        fired = True
                        break
                if fired:
                    break
                i += 1
            if not fired:
                return w
        raise RuntimeError("reduction did not terminate")

    # ---- seeding: exhaustive up to `n`, so the system is confluent up to n
    def seed(self, n: int, verbose=True):
        pz = self.pz
        best: dict[bytes, tuple] = {}
        words: dict[bytes, list[tuple]] = defaultdict(list)
        t0 = time.time()
        for L in range(1, n + 1):
            for w in product(range(pz.n_gen), repeat=L):
                k = pz.key(w)
                words[k].append(w)
                if k not in best or shortlex(w) < shortlex(best[k]):
                    best[k] = w
            if verbose:
                print(f"  seed length {L}: {len(best):,} distinct elements "
                      f"({time.time()-t0:.0f}s)", flush=True)
        n_add = 0
        for k, ws in words.items():
            b = best[k]
            for w in ws:
                if w != b and self.add(w, b):
                    n_add += 1
        # the empty word: any word equal to the identity reduces away
        if verbose:
            print(f"  seeded {n_add:,} rules over {len(best):,} elements", flush=True)

    # ---- completion: resolve overlaps into new rules
    def complete(self, max_rule: int, rounds: int = 6, verbose=True):
        """Resolve critical pairs. Overlaps are found through a PREFIX INDEX rather than
        by scanning every ordered pair of rules -- at ~200k seeded rules the quadratic
        form is ~4e10 comparisons and never finishes."""
        derived_hist = defaultdict(int)
        for rnd in range(rounds):
            new = []
            lhss = list(self.rules)
            prefix = defaultdict(list)          # p -> lhss beginning with p
            for b in lhss:
                for k in range(1, len(b)):
                    prefix[b[:k]].append(b)
            t0 = time.time()
            for a in lhss:
                ra = self.rules[a]
                # OVERLAP: a suffix of `a` is a prefix of `b`, so one word contains both
                # redexes and reduces two different ways.
                for k in range(1, len(a)):
                    for b in prefix.get(a[len(a) - k:], ()):
                        if len(a) + len(b) - k > max_rule:
                            continue
                        x = self.reduce(ra + b[k:])
                        y = self.reduce(a[:len(a) - k] + self.rules[b])
                        if x != y:
                            new.append((x, y))
                # INCLUSION: b sits wholly inside a
                for s in range(len(a)):
                    for L in range(1, len(a) - s + 1):
                        if L == len(a):
                            continue
                        b = a[s:s + L]
                        rb = self.rules.get(b)
                        if rb is None:
                            continue
                        x = self.reduce(ra)
                        y = self.reduce(a[:s] + rb + a[s + L:])
                        if x != y:
                            new.append((x, y))
            added = 0
            for x, y in new:
                lo, hi = (x, y) if shortlex(x) < shortlex(y) else (y, x)
                if len(hi) <= max_rule and self.add(hi, lo):
                    added += 1
                    derived_hist[len(hi)] += 1
            if verbose:
                print(f"  round {rnd}: {len(new):,} critical pairs, {added:,} new rules, "
                      f"{len(self.rules):,} total ({time.time()-t0:.0f}s)", flush=True)
            if added == 0:
                break
        return derived_hist


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
    ap.add_argument("--target", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--seed-len", type=int, default=4)
    ap.add_argument("--max-rule", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    pz = Pz(args.data_dir)
    sysm = System(pz)
    print(f"seeding exhaustively to length {args.seed_len} "
          f"({pz.n_gen ** args.seed_len:,} words)", flush=True)
    sysm.seed(args.seed_len)

    if args.self_test:
        rng = np.random.default_rng(0)
        ok = True
        for t in range(200):
            L = int(rng.integers(1, 14))
            w = tuple(rng.integers(0, pz.n_gen, size=L).tolist())
            r = sysm.reduce(w)
            if not np.array_equal(pz.perm_of(r), pz.perm_of(w)):
                print(f"  case {t}: reduction CHANGED the element  <-- FAIL")
                ok = False
                break
            if shortlex(r) > shortlex(w):
                print(f"  case {t}: reduction got LONGER  <-- FAIL")
                ok = False
                break
        print(f"self-test: {'PASS' if ok else 'FAIL'} "
              f"(reduction is element-preserving and non-increasing)")
        if not ok:
            return 1

    print(f"completing to |lhs| <= {args.max_rule}", flush=True)
    hist = sysm.complete(args.max_rule, rounds=args.rounds)
    print(f"\nrules: {len(sysm.rules):,}, longest lhs {sysm.max_lhs}")
    if hist:
        print("DERIVED (not seeded) reducing rules by lhs length:")
        for L in sorted(hist):
            print(f"  |lhs| {L:2d}: {hist[L]:,}")
        print("Only lengths > 13 could beat the exhaustive radius-12 MITM sweep.")
    else:
        print("completion derived NOTHING beyond the seed -- the seeded system was "
              "already confluent at this bound.")

    if not args.target:
        return 0

    rows = {p: [pz.idx[m] for m in mv] for p, mv in load_csv(args.target).items()}
    total0 = sum(map(len, rows.values()))
    saved = 0
    for pid in sorted(rows):
        r = sysm.reduce(tuple(rows[pid]))
        if len(r) < len(rows[pid]):
            print(f"  pid {pid}: {len(rows[pid])} -> {len(r)}", flush=True)
            saved += len(rows[pid]) - len(r)
            rows[pid] = list(r)
    print(f"\napplied to {args.target.name}: {total0:,} -> "
          f"{sum(map(len, rows.values())):,} (saved {saved})")

    if saved and args.out:
        starts = {}
        with io.open(args.data_dir / "test.csv", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                starts[int(r["initial_state_id"])] = np.array(
                    [int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
        bad = 0
        for pid, w in rows.items():
            s = starts[pid].copy()
            for mv in w:
                s = s[pz.gen[mv]]
            if not np.array_equal(s, pz.ident):
                bad += 1
        print(f"replay: {bad} rows failed")
        if bad == 0:
            with io.open(args.out, "w", encoding="utf-8", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["initial_state_id", "path"])
                for pid in sorted(rows):
                    wr.writerow([pid, ".".join(pz.names[m] for m in rows[pid])])
            print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
