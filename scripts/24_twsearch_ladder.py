"""Resumable exact window-shortening ladder for IHES, driven by legacy twsearch.

Same mechanism as `22_twsearch_window_improve.py` (feed every window of a chosen
length to one long-lived twsearch process as an independent scramble, accept only
replay-verified strictly-shorter words), with three additions the longer rungs of
the ladder need:

* **Streaming + resumable.** Each window's verdict is appended to a JSONL journal
  the moment its solve block closes, so a run that is killed at hour three keeps
  everything it proved. `--resume` re-reads the journal and feeds only the
  windows that are still open.
* **Timeout is a distinct verdict.** A window that hits `--time-limit` is NOT
  proven optimal, it is unfinished; it stays open so a later rung can retry it
  with a bigger budget. `22_...` silently folded these in with the clean misses.
* **Priority ordering.** Windows are fed longest-incumbent-first, so the pids
  with the most slack are proven before any budget runs out.

Move-count parity is a state invariant here (every generator is an odd
permutation of the 12 edges), so a window of length L can only shorten to
L-2, L-4, ...  That is why `--max-depth` defaults to L-2 and why the rungs
step by two.

    # prove/deny "this 20-window can be done in 18" for every window
    python scripts/24_twsearch_ladder.py --baseline sub.csv --window-length 20 \
        --journal data/ladder/w20.jsonl --time-limit 60

    # apply everything proven so far to the baseline
    python scripts/24_twsearch_ladder.py --baseline sub.csv --apply-only \
        --journal data/ladder/w20.jsonl --out submissions/w20.csv
"""

from __future__ import annotations

import argparse
import csv
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

from cayley.puzzle import PictureCube
from cayley.verify import load_submission, verify_submission

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


def invert_official(move: str) -> str:
    return move[1:] if move.startswith("-") else "-" + move


def official_to_tws(move: str) -> str:
    return move[1:] + "'" if move.startswith("-") else move


def solution_to_scramble(path: list[str]) -> str:
    """Scramble whose solution is exactly this window (identity -> window)."""
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


def select_non_overlapping(
    path_len: int, candidates: list[tuple[int, int, list[str]]]
) -> list[tuple[int, int, list[str]]]:
    """Max-saving set of non-overlapping rewrites (interval DP)."""
    by_start: dict[int, list[tuple[int, int, list[str]]]] = defaultdict(list)
    for cand in candidates:
        by_start[cand[0]].append(cand)
    score = [0] * (path_len + 1)
    choice: list[tuple[int, int, list[str]] | None] = [None] * (path_len + 1)
    for pos in range(path_len - 1, -1, -1):
        score[pos] = score[pos + 1]
        for start, end, word in by_start.get(pos, ()):
            value = end - start - len(word) + score[end]
            if value > score[pos]:
                score[pos] = value
                choice[pos] = (start, end, word)
    selected = []
    pos = 0
    while pos < path_len:
        cand = choice[pos]
        if cand is None:
            pos += 1
        else:
            selected.append(cand)
            pos = cand[1]
    return selected


def apply_journal(
    rows: dict[int, list[str]],
    journal: Path,
    generators: dict[str, np.ndarray],
) -> tuple[int, int]:
    """Fold every verified hit in the journal into `rows` (in place)."""
    hits: dict[int, list[tuple[int, int, list[str]]]] = defaultdict(list)
    n_hits = 0
    if journal.exists():
        with journal.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:  # torn last line of a killed run
                    continue
                if rec.get("verdict") != "hit":
                    continue
                pid, start, word = rec["pid"], rec["start"], rec["word"]
                path = rows.get(pid)
                if path is None:
                    continue
                end = start + rec["window_len"]
                if end > len(path) or len(word) >= rec["window_len"]:
                    continue
                # The window must still be what the journal saw.
                if path[start:end] != rec["window"]:
                    continue
                if not np.array_equal(
                    perm_of(word, generators), perm_of(path[start:end], generators)
                ):
                    continue
                hits[pid].append((start, end, word))
                n_hits += 1
    saved = 0
    for pid, options in hits.items():
        for start, end, word in reversed(select_non_overlapping(len(rows[pid]), options)):
            saved += end - start - len(word)
            rows[pid][start:end] = word
    return n_hits, saved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--journal", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--window-length", type=int)
    ap.add_argument("--min-path-length", type=int, default=0)
    ap.add_argument("--max-path-length", type=int, default=10**9)
    ap.add_argument("--full-path-only", action="store_true",
                    help="only the window that IS the whole path (needs --window-length L)")
    ap.add_argument("--full-path-any", action="store_true",
                    help="whole path of every pid in the length range, any length. "
                         "twsearch's -si caps each search below its own input "
                         "length, so one process can cover mixed lengths.")
    ap.add_argument("--pids", type=int, nargs="*")
    ap.add_argument("--pid-order-file", type=Path,
                    help="optional JSON report from 29_rank_twsearch_targets.py; "
                         "targets listed in its `ranking` array run first")
    ap.add_argument("--parity", choices=["even", "odd"],
                    help="keep only pids whose incumbent length has this parity. "
                         "Move-count parity is a state invariant, so an even-length "
                         "path can only shorten to an even one -- running the two "
                         "classes at one threshold each avoids searching depths that "
                         "are unreachable by parity.")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--max-windows", type=int)
    ap.add_argument("--apply-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    ap.add_argument("--tws", type=Path, default=DEFAULT_TWS)
    ap.add_argument("--cache-dir", type=Path, default=PROJECT / "data/twsearch_cache")
    ap.add_argument("--memory-mib", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--start-prune-depth", type=int, default=11)
    ap.add_argument("--min-depth", type=int)
    ap.add_argument("--max-depth", type=int)
    ap.add_argument("--time-limit", type=int, default=60)
    ap.add_argument("--random-start", action="store_true",
                    help="randomize native depth-first move order")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    puzzle = PictureCube.load(PROJECT / "data/puzzle_info.json")
    generators = {
        name: np.asarray(values, dtype=np.int64)
        for name, values in puzzle.generators.items()
    }
    rows = load_submission(args.baseline)
    original = {pid: len(path) for pid, path in rows.items()}
    args.journal.parent.mkdir(parents=True, exist_ok=True)

    if not args.apply_only:
        if args.window_length is None and not args.full_path_any:
            raise SystemExit("--window-length is required unless --apply-only")
        wlen = args.window_length or 0
        if args.max_depth is not None:
            max_depth = args.max_depth
        elif args.full_path_any:
            raise SystemExit("--full-path-any needs an explicit --max-depth")
        else:
            max_depth = wlen - 2
        pid_filter = set(args.pids or ())

        done: set[tuple[int, int]] = set()
        if args.resume and args.journal.exists():
            with args.journal.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("verdict") in ("hit", "none"):
                        done.add((rec["pid"], rec["start"]))
            print(f"resume: {len(done):,} windows already settled", flush=True)

        work: list[tuple[int, int, list[str]]] = []
        for pid in sorted(rows):
            path = rows[pid]
            if not (args.min_path_length <= len(path) <= args.max_path_length):
                continue
            if pid_filter and pid not in pid_filter:
                continue
            if args.parity is not None and len(path) % 2 != (args.parity == "odd"):
                continue
            if args.n_shards > 1 and pid % args.n_shards != args.shard:
                continue
            if args.full_path_any:
                if len(path) < 2 or (pid, 0) in done:
                    continue
                work.append((pid, 0, list(path)))
                continue
            if len(path) < wlen:
                continue
            starts = [0] if args.full_path_only else range(len(path) - wlen + 1)
            if args.full_path_only and len(path) != wlen:
                continue
            for start in starts:
                if (pid, start) in done:
                    continue
                work.append((pid, start, path[start : start + wlen]))
        # Longest incumbent first: most slack proven before any budget runs out.
        # Within one length, an evidence-ranked order can put likely wins before
        # expensive refutations.  It changes scheduling only, never acceptance.
        pid_rank: dict[int, int] = {}
        if args.pid_order_file:
            rank_report = json.loads(args.pid_order_file.read_text(encoding="utf-8"))
            pid_rank = {int(entry["pid"]): i
                        for i, entry in enumerate(rank_report["ranking"])}
            print(f"loaded target order for {len(pid_rank)} pids from "
                  f"{args.pid_order_file}", flush=True)
        work.sort(key=lambda item: (
            -len(rows[item[0]]),
            pid_rank.get(item[0], len(pid_rank)),
            item[0], item[1],
        ))
        if args.max_windows is not None:
            work = work[: args.max_windows]
        if not work:
            print("no open windows", flush=True)
        else:
            targets = [perm_of(w, generators) for _, _, w in work]
            command = [
                str(args.exe.resolve()), "-q", "-si",
                "-M", str(args.memory_mib),
                "-t", str(args.threads),
                "--maxdepth", str(max_depth),
                "--startprunedepth", str(args.start_prune_depth),
                "--timelimit", str(args.time_limit),
                "--cachedir", str(args.cache_dir.resolve()),
            ]
            if args.min_depth is not None:
                command.extend(["--mindepth", str(args.min_depth)])
            if args.random_start:
                command.append("--randomstart")
            if args.seed is not None:
                command.extend(["-R", str(args.seed)])
            command.append(str(args.tws.resolve()))
            env = os.environ.copy()
            env["PATH"] = str(WINLIBS_BIN) + os.pathsep + env.get("PATH", "")
            print(
                f"window {wlen}: {len(work):,} open windows, depths "
                f"{args.min_depth or 'auto'}..{max_depth}, "
                f"limit {args.time_limit}s/window",
                flush=True,
            )
            print(" ".join(command), flush=True)

            proc = subprocess.Popen(
                command, cwd=args.exe.resolve().parent.parent.parent, env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1,
            )
            assert proc.stdin is not None and proc.stdout is not None

            def feed() -> None:
                try:
                    assert proc.stdin is not None
                    for _, _, window in work:
                        proc.stdin.write(solution_to_scramble(window) + "\n")
                        proc.stdin.flush()
                    proc.stdin.close()
                except BaseException:
                    pass

            threading.Thread(target=feed, daemon=True).start()

            journal = args.journal.open("a", encoding="utf-8")
            index = -1
            words: list[list[str]] = []
            timed_out = False
            t0 = time.time()
            t_block = t0
            n_hit = n_none = n_timeout = 0

            def close_block() -> None:
                nonlocal n_hit, n_none, n_timeout
                if index < 0 or index >= len(work):
                    return
                pid, start, window = work[index]
                good = [
                    w for w in words
                    if len(w) < len(window)
                    and np.array_equal(perm_of(w, generators), targets[index])
                ]
                if good:
                    best = min(good, key=lambda w: (len(w), w))
                    verdict, payload = "hit", best
                    n_hit += 1
                elif timed_out:
                    verdict, payload = "timeout", None
                    n_timeout += 1
                else:
                    verdict, payload = "none", None
                    n_none += 1
                rec = {
                    "pid": pid, "start": start, "window_len": len(window),
                    "window": window, "verdict": verdict, "word": payload,
                    "max_depth": max_depth, "secs": round(time.time() - t_block, 2),
                }
                journal.write(json.dumps(rec) + "\n")
                journal.flush()
                if verdict == "hit":
                    print(
                        f"  HIT pid {pid} @{start}: {len(window)} -> {len(payload)}",
                        flush=True,
                    )

            for raw in proc.stdout:
                line = raw.rstrip("\r\n")
                if line == "Solving":
                    close_block()
                    index += 1
                    words, timed_out = [], False
                    t_block = time.time()
                    if index and index % 50 == 0:
                        rate = index / max(time.time() - t0, 1e-9)
                        left = (len(work) - index) / max(rate, 1e-9)
                        print(
                            f"  {index:,}/{len(work):,}  hits={n_hit} "
                            f"none={n_none} timeout={n_timeout}  "
                            f"{rate:.2f} win/s  eta {left/60:.1f} min",
                            flush=True,
                        )
                    continue
                if line.startswith("Search timed out"):
                    timed_out = True
                    continue
                if index < 0 or not raw.startswith(" "):
                    continue
                try:
                    words.append([tws_token_to_official(t) for t in line.split()])
                except ValueError:
                    continue
            close_block()
            code = proc.wait()
            journal.close()
            print(
                f"twsearch exit {code}: blocks={index + 1:,}/{len(work):,} "
                f"hits={n_hit} none={n_none} timeout={n_timeout} "
                f"wall={time.time() - t0:.0f}s",
                flush=True,
            )

    n_hits, saved = apply_journal(rows, args.journal, generators)
    out = args.out or (PROJECT / "submissions" / f"{args.journal.stem}_applied.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["initial_state_id", "path"])
        for pid in sorted(rows):
            writer.writerow([pid, ".".join(rows[pid])])
    base = verify_submission(puzzle, PROJECT / "data/test.csv", args.baseline)
    rep = verify_submission(puzzle, PROJECT / "data/test.csv", out)
    for pid in sorted(rows):
        if len(rows[pid]) < original[pid]:
            print(f"  pid {pid}: {original[pid]} -> {len(rows[pid])}")
    print(
        f"journal hits={n_hits}, applied saving={saved}; "
        f"{base.total_moves:,} -> {rep.total_moves:,} "
        f"({rep.total_moves - base.total_moves:+,}); "
        f"{rep.n_valid}/{rep.n_total} valid -> {out}",
        flush=True,
    )
    return 0 if rep.all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
