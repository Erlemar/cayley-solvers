"""Prefix-split exact ladder: prove/deny "pid p can be done in L-2", one opening at a time.

A depth-D search from a scramble already contains every forced-opening subtree, so
splitting by prefix does NOT reduce total work -- it *distributes* it.  That is the
point: a len-22 full-path proof is ~10 h on one box but ~2.5 min per chunk over the
262 two-move openings, which is the only way the fleet can attack the lengths where
our remaining slack lives.

Soundness: every length-(L-2) solution starts with SOME k moves, and we enumerate
every distinct product of exactly k moves (18 / 262 / 3750 for k=1/2/3 -- deduped by
permutation, since `f0.f1` and `f1.f0` are the same subproblem).  So "every chunk
returned none" is exactly as strong as a monolithic `none`, and one `hit` is a -2.

Openings are tried rarest-first: the shortest path for a pid shares its opening with
only 2.4% of that pid's other stored paths vs 9.0% expected (scripts/26_prefix_structure.py),
so the crowd's favourite opening is the least likely place for a shorter path to live.
Ordering only pays on early exit -- which is the case worth having.

    # ENCODING SELF-TEST first -- forces each pid's OWN opening and must find its tail
    python scripts/27_prefix_split_ladder.py --baseline sub.csv --self-test \
        --journal data/psplit/selftest.jsonl --k 2 --pids 30 32 --threads 6

    # real run: prove/deny -2 for every len-20 pid
    python scripts/27_prefix_split_ladder.py --baseline sub.csv --path-length 20 \
        --k 2 --journal data/psplit/L20_k2.jsonl --threads 8 --resume

    python scripts/27_prefix_split_ladder.py --baseline sub.csv --apply-only \
        --journal data/psplit/L20_k2.jsonl --out submissions/psplit_L20.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.puzzle import PictureCube  # noqa: E402
from cayley.verify import load_submission  # noqa: E402

csv.field_size_limit(10_000_000)

DEFAULT_EXE = (
    PROJECT / "third_party/twips/third_party/twsearch_legacy/build/bin/twsearch.exe"
)
DEFAULT_TWS = PROJECT / "data/picture_cube_pieces_v2_sym.tws"
WINLIBS_BIN = Path(
    r"C:\Users\and-l\AppData\Local\Microsoft\WinGet\Packages"
    r"\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\mingw64\bin"
)
TWS_MOVES = {"f0", "f1", "f2", "r0", "r1", "r2", "d0", "d1", "d2"}


# ------------------------------------------------------- twsearch encoding (from 24_)


def invert_official(move: str) -> str:
    return move[1:] if move.startswith("-") else "-" + move


def official_to_tws(move: str) -> str:
    return move[1:] + "'" if move.startswith("-") else move


def solution_to_scramble(path: list[str]) -> str:
    """Scramble whose solution is exactly this path (identity -> path)."""
    return " ".join(official_to_tws(invert_official(m)) for m in reversed(path))


def tws_token_to_official(token: str) -> str:
    prime = token.endswith("'")
    base = token[:-1] if prime else token
    if base not in TWS_MOVES:
        raise ValueError(f"unknown twsearch move {token!r}")
    return ("-" if prime else "") + base


def perm_of(path: list[str], generators: dict[str, np.ndarray]) -> np.ndarray:
    state = np.arange(72, dtype=np.int64)
    for move in path:
        state = state[generators[move]]
    return state


def undo(prefix: list[str]) -> list[str]:
    """Path that undoes `prefix` (inverse moves, reversed)."""
    return [invert_official(m) for m in reversed(prefix)]


# ------------------------------------------------------- prefix set + crowd ordering


def canonical_prefixes(k: int, generators: dict[str, np.ndarray],
                       names: list[str]) -> list[list[str]]:
    """Every distinct permutation reachable by EXACTLY k moves, one representative each."""
    seen: dict[bytes, list[str]] = {}
    for combo in itertools.product(names, repeat=k):
        key = perm_of(list(combo), generators).tobytes()
        if key not in seen:
            seen[key] = list(combo)
    return list(seen.values())


def crowd_counts(roots: list[Path], k: int, generators: dict[str, np.ndarray]
                 ) -> dict[int, dict[bytes, int]]:
    """{pid: {prefix_perm_bytes: how many stored distinct paths open that way}}.

    Counts DISTINCT paths, so a result copied into 40 merged CSVs still counts once.
    """
    seen: dict[int, set[tuple[str, ...]]] = defaultdict(set)
    for root in roots:
        if not root.exists():
            continue
        for path in ([root] if root.is_file() else root.rglob("*.csv")):
            try:
                with open(path, "r", encoding="utf-8", newline="") as f:
                    rd = csv.reader(f)
                    head = next(rd, None)
                    if not head or head[:2] != ["initial_state_id", "path"]:
                        continue
                    for row in rd:
                        if len(row) < 2 or not row[1].strip():
                            continue
                        try:
                            pid = int(row[0])
                        except ValueError:
                            continue
                        seen[pid].add(tuple(row[1].strip().split(".")))
            except (OSError, UnicodeDecodeError):
                continue
    out: dict[int, dict[bytes, int]] = {}
    for pid, paths in seen.items():
        c: dict[bytes, int] = defaultdict(int)
        for p in paths:
            if len(p) < k:
                continue
            try:
                c[perm_of(list(p[:k]), generators).tobytes()] += 1
            except KeyError:
                continue  # foreign move alphabet
        out[pid] = dict(c)
    return out


# ------------------------------------------------------- journal -> submission


def apply_journal(rows: dict[int, list[str]], journal: Path,
                  generators: dict[str, np.ndarray]) -> tuple[int, int]:
    best: dict[int, list[str]] = {}
    if not journal.exists():
        return 0, 0
    with journal.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("verdict") != "hit":
                continue
            pid = rec["pid"]
            cand = list(rec["prefix"]) + list(rec["word"])
            if pid not in rows:
                continue
            if not np.array_equal(perm_of(cand, generators),
                                  perm_of(rows[pid], generators)):
                print(f"  [reject] pid {pid}: replay mismatch", flush=True)
                continue
            if len(cand) >= len(rows[pid]):
                continue
            if pid not in best or len(cand) < len(best[pid]):
                best[pid] = cand
    saved = 0
    for pid, cand in best.items():
        saved += len(rows[pid]) - len(cand)
        rows[pid] = cand
    return len(best), saved


# ------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--journal", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--k", type=int, default=2, help="forced opening length")
    ap.add_argument("--fixed-prefix", default="",
                    help="dot-separated fixed opening moves; enumerate remaining k moves, using suffix IDs")
    ap.add_argument("--path-length", type=int, help="only pids whose incumbent is this long")
    ap.add_argument("--pids", type=int, nargs="*")
    ap.add_argument("--skip-pids", type=int, nargs="*",
                    help="pids to exclude even when they match the length filter")
    ap.add_argument("--pid-order-file", type=Path,
                    help="optional JSON report from 29_rank_twsearch_targets.py; "
                         "pids listed in its `ranking` array run first")
    ap.add_argument("--target-delta", type=int, default=2,
                    help="search for a solution this much shorter than the incumbent")
    ap.add_argument("--self-test", action="store_true",
                    help="force each pid's OWN opening and search at its own length; "
                         "MUST report a hit or the scramble encoding is wrong")
    ap.add_argument("--max-pids", type=int)
    ap.add_argument("--max-openings", type=int,
                    help="only the N rarest openings per pid -- a SEARCH, not a proof: "
                         "all-none over a subset proves nothing, and is journalled as "
                         "partial coverage")
    ap.add_argument("--opening-offset", type=int, default=0,
                    help="skip the first N openings of the rarest-first order (shards a "
                         "pid across machines)")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--pop-roots", nargs="*", default=["submissions"])
    ap.add_argument("--apply-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    ap.add_argument("--tws", type=Path, default=DEFAULT_TWS)
    ap.add_argument("--cache-dir", type=Path, default=PROJECT / "data/twsearch_cache")
    ap.add_argument("--memory-mib", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--start-prune-depth", type=int, default=11)
    ap.add_argument("--time-limit", type=int, default=120)
    ap.add_argument("--lookahead", type=int, default=8,
                    help="how far the feeder may run ahead of the reader; small keeps "
                         "early-exit-on-hit effective, large keeps the pipe busy")
    args = ap.parse_args()
    args.fixed_prefix = args.fixed_prefix.split(".") if args.fixed_prefix else []

    puzzle = PictureCube.load(PROJECT / "data/puzzle_info.json")
    generators = {n: np.asarray(v, dtype=np.int64) for n, v in puzzle.generators.items()}
    names = list(puzzle.move_names)
    rows = load_submission(args.baseline)
    args.journal.parent.mkdir(parents=True, exist_ok=True)

    if not args.apply_only:
        if len(args.fixed_prefix) >= args.k or any(move not in generators for move in args.fixed_prefix):
            raise SystemExit("fixed prefix must contain valid moves and be shorter than k")
        if args.fixed_prefix and args.self_test:
            raise SystemExit("fixed prefix is incompatible with self-test mode")
        prefixes = [args.fixed_prefix + suffix for suffix in
                    canonical_prefixes(args.k - len(args.fixed_prefix), generators, names)]
        print(f"k={args.k}, fixed={args.fixed_prefix}: {len(names)**(args.k-len(args.fixed_prefix))} suffix move-strings -> "
              f"{len(prefixes)} distinct openings", flush=True)

        pid_filter = set(args.pids or ())
        skip_pids = set(args.skip_pids or ())
        pid_rank: dict[int, int] = {}
        if args.pid_order_file:
            rank_report = json.loads(args.pid_order_file.read_text(encoding="utf-8"))
            pid_rank = {int(entry["pid"]): i
                        for i, entry in enumerate(rank_report["ranking"])}
            print(f"loaded target order for {len(pid_rank)} pids from "
                  f"{args.pid_order_file}", flush=True)
        ordered_pids = sorted(
            rows,
            key=lambda pid: (pid_rank.get(pid, len(pid_rank)), pid),
        )
        scope = []
        for pid in ordered_pids:
            L = len(rows[pid])
            if pid_filter and pid not in pid_filter:
                continue
            if pid in skip_pids:
                continue
            if args.path_length is not None and L != args.path_length:
                continue
            if args.n_shards > 1 and pid % args.n_shards != args.shard:
                continue
            depth = L - args.k if args.self_test else L - args.target_delta - args.k
            if depth < 1:
                continue
            scope.append(pid)
        if args.max_pids:
            scope = scope[: args.max_pids]
        if not scope:
            raise SystemExit("no pids in scope")

        lengths = {len(rows[p]) for p in scope}
        if len(lengths) != 1:
            raise SystemExit(f"mixed incumbent lengths {sorted(lengths)}; one process "
                             f"has one --maxdepth, so run one --path-length at a time")
        L = lengths.pop()
        max_depth = (L - args.k) if args.self_test else (L - args.target_delta - args.k)

        done: set[tuple[int, int]] = set()
        hit_pids: set[int] = set()
        if args.resume and args.journal.exists():
            with args.journal.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("verdict") in ("hit", "none"):
                        if rec.get("prefix") != prefixes[rec["prefix_id"]]:
                            raise SystemExit("resume prefix identity mismatch; use a separate journal")
                        done.add((rec["pid"], rec["prefix_id"]))
                    if rec.get("verdict") == "hit":
                        hit_pids.add(rec["pid"])
            print(f"resume: {len(done):,} chunks settled, {len(hit_pids)} pids already hit",
                  flush=True)

        counts = crowd_counts([PROJECT / r for r in args.pop_roots], args.k, generators)
        work: list[tuple[int, int, list[str], list[str], np.ndarray]] = []
        for pid in scope:
            path = list(rows[pid])
            if args.self_test:
                cand = [(0, path[: args.k])]
            else:
                c = counts.get(pid, {})
                order = sorted(
                    range(len(prefixes)),
                    key=lambda i: (c.get(perm_of(prefixes[i], generators).tobytes(), 0), i),
                )
                cand = [(i, prefixes[i]) for i in order]
                lo = args.opening_offset
                hi = lo + args.max_openings if args.max_openings else len(cand)
                cand = cand[lo:hi]
            for pref_id, pref in cand:
                if (pid, pref_id) in done or pid in hit_pids:
                    continue
                window = undo(pref) + path        # a path that solves prefix-applied state
                work.append((pid, pref_id, pref, window, perm_of(window, generators)))
        if not work:
            print("nothing to do", flush=True)
        else:
            command = [
                str(args.exe.resolve()), "-q", "-si",
                "-M", str(args.memory_mib),
                "-t", str(args.threads),
                "--maxdepth", str(max_depth),
                "--startprunedepth", str(args.start_prune_depth),
                "--timelimit", str(args.time_limit),
                "--cachedir", str(args.cache_dir.resolve()),
                str(args.tws.resolve()),
            ]
            env = os.environ.copy()
            env["PATH"] = str(WINLIBS_BIN) + os.pathsep + env.get("PATH", "")
            mode = "SELF-TEST" if args.self_test else f"-{args.target_delta}"
            print(f"{len(scope)} pids x <= {len(prefixes)} openings = {len(work):,} chunks, "
                  f"L={L} {mode}, maxdepth {max_depth}, {args.threads} threads", flush=True)
            print(" ".join(command), flush=True)

            proc = subprocess.Popen(
                command, cwd=args.exe.resolve().parent.parent.parent, env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1,
            )
            assert proc.stdin is not None and proc.stdout is not None

            processed = [0]
            skipped = [0]
            lock = threading.Lock()
            # Only items actually written to twsearch land here, so block N of the
            # output always corresponds to fed[N] even when early exit skips work.
            fed: list[tuple[int, int, list[str], list[str], np.ndarray]] = []

            def feed() -> None:
                try:
                    assert proc.stdin is not None
                    for item in work:
                        while len(fed) - processed[0] > args.lookahead:
                            time.sleep(0.05)
                        with lock:
                            if item[0] in hit_pids:   # early exit: this pid is solved
                                skipped[0] += 1
                                continue
                            fed.append(item)
                        proc.stdin.write(solution_to_scramble(item[3]) + "\n")
                        proc.stdin.flush()
                    proc.stdin.close()
                except BaseException:
                    pass

            threading.Thread(target=feed, daemon=True).start()

            journal = args.journal.open("a", encoding="utf-8")
            index = -1
            words: list[list[str]] = []
            timed_out = False
            exhausted = False
            solver_log = args.journal.with_suffix('.solver.log').open('a', encoding='utf-8')
            t0 = time.time()
            t_block = t0
            n_hit = n_none = n_timeout = 0
            pid_state: dict[int, dict[str, int]] = defaultdict(
                lambda: {"none": 0, "timeout": 0, "hit": 0})

            def close_block() -> None:
                nonlocal n_hit, n_none, n_timeout
                if index < 0 or index >= len(fed):
                    return
                pid, pref_id, pref, window, target = fed[index]
                good = [w for w in words
                        if len(w) <= max_depth and np.array_equal(
                            perm_of(w, generators), target)]
                if good:
                    best = min(good, key=lambda w: (len(w), w))
                    verdict, payload = "hit", best
                    n_hit += 1
                    with lock:
                        hit_pids.add(pid)
                elif timed_out or not exhausted:
                    verdict, payload = "timeout", None
                    n_timeout += 1
                else:
                    verdict, payload = "none", None
                    n_none += 1
                pid_state[pid][verdict] += 1
                rec = {"pid": pid, "prefix_id": pref_id, "prefix": pref,
                       "incumbent_len": len(window) - args.k, "verdict": verdict,
                       "word": payload, "max_depth": max_depth,
                       "self_test": bool(args.self_test),
                       "secs": round(time.time() - t_block, 2)}
                journal.write(json.dumps(rec) + "\n")
                journal.flush()
                if verdict == "hit":
                    total = args.k + len(payload)
                    print(f"  HIT pid {pid} via {'.'.join(pref)}: "
                          f"{len(window) - args.k} -> {total}", flush=True)

            for raw in proc.stdout:
                solver_log.write(raw)
                solver_log.flush()
                line = raw.rstrip("\r\n")
                if line == "Solving":
                    close_block()
                    index += 1
                    processed[0] = index
                    words, timed_out, exhausted = [], False, False
                    t_block = time.time()
                    if index and index % 200 == 0:
                        rate = index / max(time.time() - t0, 1e-9)
                        left = (len(work) - index) / max(rate, 1e-9)
                        print(f"  {index:,}/{len(work):,}  hits={n_hit} none={n_none} "
                              f"timeout={n_timeout} skipped={skipped[0]}  {rate:.2f} chunk/s"
                              f"  eta {left/60:.1f} min", flush=True)
                    continue
                if line.startswith("Search timed out"):
                    timed_out = True
                    continue
                if line.startswith("No solution found in "):
                    exhausted = line.split()[-1] == str(max_depth)
                    continue
                if index < 0 or not raw.startswith(" "):
                    continue
                try:
                    words.append([tws_token_to_official(t) for t in line.split()])
                except ValueError:
                    continue
            code = proc.wait()
            if code:
                exhausted = False
                timed_out = True
            close_block()
            journal.close()
            solver_log.close()
            print(f"twsearch exit {code}: chunks={index + 1:,}/{len(work):,} hits={n_hit} "
                  f"none={n_none} timeout={n_timeout} wall={time.time() - t0:.0f}s",
                  flush=True)
            if code:
                return 1

            if args.self_test:
                bad = [p for p in scope if pid_state[p]["hit"] == 0]
                if bad:
                    print(f"\nSELF-TEST FAILED for {len(bad)} pids: {bad[:10]} -- the "
                          f"scramble encoding is wrong, do NOT trust any `none`.",
                          flush=True)
                    return 1
                print(f"\nSELF-TEST PASSED: all {len(scope)} pids found their own tail "
                      f"through their own opening.", flush=True)
                return 0

            if args.max_openings or args.opening_offset or args.fixed_prefix:
                print(f"\npartial sweep ({args.max_openings} of {len(prefixes)} openings "
                      f"from #{args.opening_offset}): a pid with no hit is NOT proven "
                      f"optimal -- the remaining openings were never searched.", flush=True)
            else:
                proven = [p for p in scope
                          if pid_state[p]["timeout"] == 0 and pid_state[p]["hit"] == 0
                          and pid_state[p]["none"] == len(prefixes)]
                print(f"\nfully covered (every opening returned none) => PROVEN optimal at "
                      f"L={L}: {len(proven)}/{len(scope)} pids", flush=True)

    n_hits, saved = apply_journal(rows, args.journal, generators)
    out = args.out or (PROJECT / "submissions" / f"{args.journal.stem}_applied.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["initial_state_id", "path"])
        for pid in sorted(rows):
            writer.writerow([pid, ".".join(rows[pid])])
    print(f"applied {n_hits} hits, saved {saved} moves -> {out} "
          f"(total {sum(len(v) for v in rows.values())})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
