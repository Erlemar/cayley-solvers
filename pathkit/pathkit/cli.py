"""Command line for every method in the package.

    python -m pathkit.cli selftest
    python -m pathkit.cli plan   --preset tetraminx --in sub.csv
    python -m pathkit.cli merge  --preset tetraminx --scan . --out merged.csv
    python -m pathkit.cli window --preset cube444 --in sub.csv --out out.csv --radius 4
    python -m pathkit.cli ladder --preset ihes --in sub.csv --journal j.jsonl --window 12

Presets resolve puzzle_info.json and test.csv inside this repo; --puzzle-info / --tests
override them for anything else.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

PRESETS = {
    "ihes": ("data/puzzle_info.json", "data/test.csv"),
    "tetraminx": ("tetraminx/data/puzzle_info.json", "tetraminx/data/test.csv"),
    "megaminx": ("megaminx/data/puzzle_info.json", "megaminx/data/test.csv"),
    "cube444": ("cube444/data/puzzle_info.json", "cube444/data/test.csv"),
}


def resolve(args):
    from .io import load_tests
    from .puzzle import Puzzle
    if args.preset == "demo":
        pz = Puzzle.demo_lrx(12)
        import numpy as np
        rng = np.random.default_rng(0)
        tests = {}
        for pid in range(20):
            s, _ = pz.random_scramble(14, rng)
            tests[pid] = s
        return pz, tests
    if args.puzzle_info:
        info, tcsv = Path(args.puzzle_info), Path(args.tests) if args.tests else None
    else:
        if args.preset not in PRESETS:
            raise SystemExit("pass --preset one of %s (or --puzzle-info)"
                             % (", ".join(sorted(PRESETS) + ["demo"])))
        info = REPO / PRESETS[args.preset][0]
        tcsv = Path(args.tests) if args.tests else REPO / PRESETS[args.preset][1]
    pz = Puzzle.from_puzzle_info(info, args.preset)
    tests = load_tests(tcsv) if tcsv and Path(tcsv).exists() else {}
    return pz, tests


def cmd_plan(args):
    """Print the decision procedure for a given file, with the cheap checks run first."""
    from .io import load_submission
    pz, tests = resolve(args)
    paths = load_submission(args.inp, pz)
    lens = sorted(len(w) for w in paths.values())
    total = sum(lens)
    n = len(lens)
    print(pz.describe())
    print("file: %d pids, %s moves, mean %.2f, median %d, max %d"
          % (n, "{:,}".format(total), total / max(n, 1), lens[n // 2], lens[-1]))
    bad = [p for p, w in paths.items() if p in tests and not pz.solves(tests[p], w)]
    print("replay: %d invalid rows" % len(bad))
    print("")
    print("recommended order (each step is cheap relative to the next):")
    print(" 1. merge   -- n-way per-pid min over every source on disk and every live box.")
    print("              Re-pull machines rather than trusting a copy. Scan by CONTENT.")
    print(" 2. cheap   -- adjacent-inverse + state-hash shortcut. Free; expect 0.2-0.3 pct.")
    print(" 3. window  -- ball sweep at r=3, then r=4 on the long tail only.")
    print("              A zero at reach 2r means every window that short is geodesic.")
    if pz.is_permutation:
        print(" 4. table   -- if you have a BFS endgame table, sweep it with --extend 2")
        print("              before considering a deeper table. That is how d8 got priced at 0.")
    else:
        print(" 4. (no table step: colour puzzle, no inverse state -- the ball sweep is it)")
    print(" 5. bridge  -- pool every path per pid and bridge; subsumes 1 and 3 in one pass.")
    print("              Run slack_audit first: zero slackful waypoints means zero bridges.")
    print(" 6. suffix  -- diagnostic. If the solver never beats a suffix it can score,")
    print("              the remaining slack is GLOBAL and rewriting is finished here.")
    print(" 7. neural  -- only for long, non-merged, our-own paths. Measure the scorer")
    print("              saturation horizon first or window selection is false positives.")
    return 0


def cmd_merge(args):
    from .io import load_submission, write_submission
    from .merge import compare, format_report, merge
    pz, tests = resolve(args)
    roots = [Path(r) for r in (args.scan or [REPO])]
    best, report = merge(pz, tests, roots, base=args.base, verbose=args.verbose)
    format_report(report)
    if args.base:
        cmp = compare(pz, best, load_submission(args.base, pz))
        print("vs %s: %s -> %s (%+d) over %d improved pids"
              % (Path(args.base).name, "{:,}".format(cmp["baseline_total"]),
                 "{:,}".format(cmp["merged_total"]), cmp["delta"], len(cmp["wins"])))
        for pid, gain in cmp["wins"][:10]:
            print("    pid %-6s -%d" % (pid, gain))
    if args.out:
        write_submission(args.out, pz, best, tests)
    return 0


def cmd_cheap(args):
    from .cheap import sweep
    from .io import load_submission, write_submission
    pz, tests = resolve(args)
    paths = load_submission(args.inp, pz)
    out, rep = sweep(pz, tests, paths)
    print("cheap: %s -> %s (%+d moves, %d paths changed)"
          % ("{:,}".format(rep["before"]), "{:,}".format(rep["after"]),
             -rep["saved"], rep["changed"]))
    if args.out:
        write_submission(args.out, pz, out, tests)
    return 0


def cmd_window(args):
    from .io import load_submission, write_submission
    from .window import certify_report, sweep
    pz, tests = resolve(args)
    paths = load_submission(args.inp, pz)
    if args.pids:
        keep = set(int(x) for x in args.pids.split(","))
        paths = {p: w for p, w in paths.items() if p in keep}
    out, rep = sweep(pz, tests, paths, args.radius, device=args.device,
                     fixpoint=args.fixpoint, min_length=args.min_length,
                     progress=args.progress)
    print(certify_report(rep))
    print("  swept %d pids (%d skipped as short), %d phantom rejects, %.1fs"
          % (rep["swept"], rep["skipped_short"], rep["phantom_rejects"], rep["seconds"]))
    if args.out:
        write_submission(args.out, pz, out, tests)
    return 0


def cmd_table_window(args):
    from .io import load_submission, write_submission
    from .table import BfsTable, sweep
    pz, tests = resolve(args)
    paths = load_submission(args.inp, pz)
    table = (BfsTable.load(args.table) if args.table
             else BfsTable.build(pz, args.build_depth, log=print))
    print("table: %s states, depth <= %d" % ("{:,}".format(len(table.hashes)), table.max_depth))
    out, rep = sweep(pz, tests, paths, table, max_window=args.max_window,
                     extend=args.extend, progress=args.progress)
    print("table window: %s -> %s (%+d moves, %d rewrites), effective reach %d"
          % ("{:,}".format(rep["before"]), "{:,}".format(rep["after"]), -rep["saved"],
             rep["splices"], rep["effective_reach"]))
    if args.out:
        write_submission(args.out, pz, out, tests)
    return 0


def cmd_bridge(args):
    from .bridge import sweep
    from .io import (iter_candidate_files, load_submission, sniff_submission,
                     write_submission)
    pz, tests = resolve(args)
    base = load_submission(args.inp, pz)
    pools = {pid: [w] for pid, w in base.items()}
    for p in iter_candidate_files([Path(r) for r in (args.scan or [])], exts=(".csv",)):
        if not sniff_submission(p, pz):
            continue
        for pid, word in load_submission(p, pz).items():
            if pid in pools and pz.solves(tests[pid], word):
                pools[pid].append(word)
    print("pooled %.2f paths per pid" % (sum(len(v) for v in pools.values()) / max(len(pools), 1)))
    out, rep = sweep(pz, tests, pools, args.radius, device=args.device,
                     max_paths=args.max_paths, slack=args.slack, progress=args.progress)
    print("bridge: merged floor %s -> %s (%+d beyond the merge, %d pids bridged), %.1fs"
          % ("{:,}".format(rep["before_merged"]), "{:,}".format(rep["after"]),
             -rep["saved_beyond_merge"], rep["bridged_pids"], rep["seconds"]))
    if args.out:
        write_submission(args.out, pz, out, tests)
    return 0


def cmd_ladder(args):
    from .io import load_submission, write_submission
    from .ladder import BallSolver, Journal, apply_journal, run_ladder
    pz, tests = resolve(args)
    paths = load_submission(args.inp, pz)
    journal = Journal(args.journal)
    if not args.apply_only:
        solver = BallSolver(pz, args.radius, device=args.device)
        run_ladder(pz, tests, paths, solver, journal, args.window,
                   time_budget=args.time_budget, progress=args.progress)
    out, rep = apply_journal(pz, tests, paths, journal)
    if args.out:
        write_submission(args.out, pz, out, tests)
    return 0


def cmd_suffix(args):
    from .io import load_submission
    from .suffix import make_table_solver, probe
    from .table import BfsTable
    pz, tests = resolve(args)
    paths = load_submission(args.inp, pz)
    table = (BfsTable.load(args.table) if args.table
             else BfsTable.build(pz, args.build_depth, root=pz.solved, log=print))
    solver = make_table_solver(table, pz, extend=args.extend)
    pids = sorted(paths)[:args.sample]
    probe(pz, tests, {p: paths[p] for p in pids}, solver, ks=tuple(
        int(x) for x in args.ks.split(",")))
    return 0


def cmd_verify(args):
    from .io import load_submission
    pz, tests = resolve(args)
    paths = load_submission(args.inp, pz)
    bad = [p for p, w in paths.items() if p not in tests or not pz.solves(tests[p], w)]
    total = sum(len(w) for w in paths.values())
    missing = sorted(set(tests) - set(paths))
    print("%d rows, %s moves, invalid=%d, missing=%d"
          % (len(paths), "{:,}".format(total), len(bad), len(missing)))
    if bad:
        print("  invalid pids: %s" % bad[:20])
    if missing:
        print("  missing pids: %s" % missing[:20])
    return 1 if (bad or missing) else 0


def cmd_selftest(args):
    from .selftest import run_all
    return run_all()


def build_parser():
    ap = argparse.ArgumentParser(prog="pathkit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, needs_input=True):
        p.add_argument("--preset", default="", help="ihes, tetraminx, megaminx, cube444, demo")
        p.add_argument("--puzzle-info", default="", help="path to puzzle_info.json")
        p.add_argument("--tests", default="", help="path to test.csv")
        if needs_input:
            p.add_argument("--in", dest="inp", required=True, help="submission CSV")
        p.add_argument("--out", default="", help="write result here")
        p.add_argument("--progress", type=int, default=0, help="log every N pids")

    p = sub.add_parser("plan", help="describe a file and print the recommended order")
    common(p)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("merge", help="n-way per-pid min-merge by content")
    common(p, needs_input=False)
    p.add_argument("--scan", nargs="*", help="roots to scan (default: repo root)")
    p.add_argument("--base", default="", help="baseline CSV to attribute against")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=cmd_merge)

    p = sub.add_parser("cheap", help="adjacent-inverse plus state-hash shortcut")
    common(p)
    p.set_defaults(fn=cmd_cheap)

    p = sub.add_parser("window", help="ball-collision window shortening (reach 2r)")
    common(p)
    p.add_argument("--radius", type=int, default=3)
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--fixpoint", action="store_true", help="re-sweep until nothing changes")
    p.add_argument("--min-length", type=int, default=0, help="skip paths shorter than this")
    p.add_argument("--pids", default="", help="comma separated subset")
    p.set_defaults(fn=cmd_window)

    p = sub.add_parser("table-window", help="BFS-table window rewriting with extend")
    common(p)
    p.add_argument("--table", default="", help="bfs_endgame npz; omit to build one")
    p.add_argument("--build-depth", type=int, default=5)
    p.add_argument("--max-window", type=int, default=14)
    p.add_argument("--extend", type=int, default=0,
                   help="BFS this many moves out from the window element; reach becomes "
                        "table depth plus this, with no deeper table built")
    p.set_defaults(fn=cmd_table_window)

    p = sub.add_parser("bridge", help="pooled cross-trajectory bridging")
    common(p)
    p.add_argument("--scan", nargs="*", help="extra CSV roots to pool from")
    p.add_argument("--radius", type=int, default=3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-paths", type=int, default=8)
    p.add_argument("--slack", type=int, default=8)
    p.set_defaults(fn=cmd_bridge)

    p = sub.add_parser("ladder", help="resumable exact ladder with hit/none/unproven")
    common(p)
    p.add_argument("--journal", required=True)
    p.add_argument("--window", type=int, default=12, help="window length to prove")
    p.add_argument("--radius", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--time-budget", type=float, default=None, help="seconds")
    p.add_argument("--apply-only", action="store_true", help="harvest the journal, no search")
    p.set_defaults(fn=cmd_ladder)

    p = sub.add_parser("suffix", help="suffix re-solve diagnostic")
    common(p)
    p.add_argument("--table", default="")
    p.add_argument("--build-depth", type=int, default=5)
    p.add_argument("--extend", type=int, default=1)
    p.add_argument("--ks", default="8,10,12")
    p.add_argument("--sample", type=int, default=20)
    p.set_defaults(fn=cmd_suffix)

    p = sub.add_parser("verify", help="replay-verify a submission")
    common(p)
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("selftest", help="run the built-in correctness tests")
    p.set_defaults(fn=cmd_selftest)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
