"""Built-in correctness tests, led by the POSITIVE CONTROLS.

Every method here can return "0 improvements", and that is also exactly what a broken or
unwired pipeline returns. Before believing a zero, prove the harness can find a win it is
guaranteed to find: inflate a known-good path with a word that is the identity and check
the method removes exactly it. On IHES that control (a generator of order 4, applied four
times) is what established that a long string of zeros was real rather than a plumbing
failure -- and it is cheap enough that there is no excuse for skipping it.

    python -m pathkit.cli selftest
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .backend import get_backend
from .puzzle import Puzzle

_FAIL = []


def check(name, cond, detail=""):
    status = "ok  " if cond else "FAIL"
    print("  [%s] %s%s" % (status, name, ("  -- " + detail) if detail else ""))
    if not cond:
        _FAIL.append(name)
    return cond


def _identity_pad(puzzle, m):
    """A short word that is the identity: move m followed by its inverse."""
    return [m, int(puzzle.inv_move[m])]


def test_puzzle_loading():
    print("puzzle loading")
    p = Puzzle.demo_lrx(12)
    check("lrx inverse map (X is self-inverse)", int(p.inv_move[2]) == 2)
    check("lrx L and R are inverses", int(p.inv_move[0]) == 1 and int(p.inv_move[1]) == 0)
    check("lrx is a permutation puzzle", p.is_permutation)
    c = Puzzle.demo_lrx_coloured(12, 3)
    check("coloured lrx is not a permutation puzzle", not c.is_permutation)
    check("coloured lrx has 3 labels", c.n_labels == 3)
    rng = np.random.default_rng(0)
    s, w = p.random_scramble(20, rng)
    check("random scramble is solved by its word", p.solves(s, w))
    check("inv_word round trip",
          np.array_equal(p.apply_word(p.apply_word(s, w), p.inv_word(w)), s))


def test_positive_control_window():
    print("positive control: ball window sweep")
    from .window import reduce_path
    p = Puzzle.demo_lrx(12)
    rng = np.random.default_rng(1)
    bk = get_backend("cpu")
    exact = 0
    for t in range(8):
        s0, w = p.random_scramble(16, np.random.default_rng(50 + t))
        pad = _identity_pad(p, t % 3)
        infl = list(w[:3]) + pad + list(w[3:])
        out = reduce_path(p, s0, infl, radius=2, bk=bk)
        if len(out) <= len(w) and p.solves(s0, out):
            exact += 1
    check("inflated paths reduced back and still solve", exact == 8, "%d/8" % exact)
    s0, w = p.random_scramble(16, rng)
    out = reduce_path(p, s0, w, radius=2, bk=bk)
    check("clean path never gets longer", len(out) <= len(w))
    check("clean path still solves", p.solves(s0, out))


def test_positive_control_table():
    print("positive control: BFS table window rewriting")
    from .table import BfsTable, window_reduce
    p = Puzzle.demo_lrx(10)
    t = BfsTable.build(p, 5)
    check("table built", len(t.hashes) > 10, "%d states" % len(t.hashes))
    s0, w = p.random_scramble(18, np.random.default_rng(2))
    infl = list(w[:4]) + _identity_pad(p, 0) + list(w[4:])
    out = window_reduce(p, s0, infl, t, max_window=8)
    check("identity padding removed", len(out) <= len(w), "%d -> %d" % (len(infl), len(out)))
    check("path still solves", p.solves(s0, out))


def test_extend_reaches_deeper():
    print("extend trick reaches past the table depth")
    from .table import BfsTable
    p = Puzzle.demo_lrx(10)
    shallow = BfsTable.build(p, 3)
    deep = BfsTable.build(p, 5)
    rng = np.random.default_rng(3)
    agree = tested = 0
    for _ in range(40):
        _s, w = p.random_scramble(5, rng)
        if len(w) < 4:
            continue
        elem = np.arange(p.state_size)
        for m in w:
            elem = elem[p.gens[m]]
        d_deep = deep.shortest_word(p, elem, extend=0)
        d_ext = shallow.shortest_word(p, elem, extend=2)
        if d_deep is None:
            continue
        tested += 1
        if d_ext is not None and len(d_ext) == len(d_deep):
            agree += 1
    check("depth-3 table with extend=2 matches a depth-5 table",
          tested > 0 and agree == tested, "%d/%d" % (agree, tested))


def test_backend_parity():
    print("backend parity")
    from .window import reduce_path
    p = Puzzle.demo_lrx(12)
    results = {}
    for dev in ("cpu", "torch", "cuda"):
        bk = get_backend(dev)
        tot = 0
        for t in range(6):
            s0, w = p.random_scramble(16, np.random.default_rng(200 + t))
            infl = list(w[:3]) + _identity_pad(p, 0) + list(w[3:])
            out = reduce_path(p, s0, infl, radius=3, bk=bk)
            assert p.solves(s0, out)
            tot += len(out)
        results[type(bk).__name__] = tot
    check("all available backends agree", len(set(results.values())) == 1, str(results))


def test_colour_beats_permutation():
    print("colour-space rewriting is at least as strong as permutation-space")
    from .window import reduce_path
    P = Puzzle.demo_lrx(12)
    C = Puzzle.demo_lrx_coloured(12, 3)
    bk = get_backend("cpu")
    worse = strict = 0
    for t in range(25):
        s0P, w = P.random_scramble(18, np.random.default_rng(300 + t))
        s0C = C.solved[s0P]
        if not C.solves(s0C, w):
            continue
        outP = reduce_path(P, s0P, w, radius=3, bk=bk)
        outC = reduce_path(C, s0C, w, radius=3, bk=bk)
        assert P.solves(s0P, outP) and C.solves(s0C, outC)
        if len(outC) > len(outP):
            worse += 1
        if len(outC) < len(outP):
            strict += 1
    check("colour sweep never does worse", worse == 0)
    check("colour sweep finds rewrites permutation space cannot",
          strict > 0, "%d of 25 paths strictly shorter in colour space" % strict)


def test_band_semantics():
    print("band semantics: a miss beyond reach is UNPROVEN, not NONE")
    from .ladder import NONE, UNPROVEN, BallSolver
    p = Puzzle.demo_lrx(12)
    solver = BallSolver(p, radius=2)          # reach 4
    rng = np.random.default_rng(5)
    s, _ = p.random_scramble(12, rng)
    v_short, _ = solver.solve(s, s.copy(), max_len=3)
    check("short window with no shortening reports proven NONE or a HIT",
          v_short in (NONE, "hit"), v_short)
    far = p.apply_word(s, [0] * 9)
    v_long, _ = solver.solve(s, far, max_len=9)
    check("window longer than reach reports UNPROVEN", v_long == UNPROVEN, v_long)


def test_merge_safety():
    print("merge safety: content scan, foreign alphabet, invalid rows")
    from .io import write_submission
    from .merge import merge
    p = Puzzle.demo_lrx(12)
    rng = np.random.default_rng(6)
    tests, good, loose = {}, {}, {}
    for pid in range(8):
        s, w = p.random_scramble(14, rng)
        tests[pid] = s
        good[pid] = w
        loose[pid] = list(w[:2]) + _identity_pad(p, 2) + list(w[2:])
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write_submission(td / "loose.csv", p, loose, tests, log=lambda *a: None)
        write_submission(td / "good.csv", p, good, tests, log=lambda *a: None)
        (td / "foreign.csv").write_text(
            "initial_state_id,path\n0,f0.-f0.r1\n", encoding="utf-8")
        (td / "corrupt.csv").write_text(
            "initial_state_id,path\n0,%s\n" % p.format([0, 0, 0]), encoding="utf-8")
        best, rep = merge(p, tests, [td], verbose=False)
        check("merge picks the shorter file", sum(len(v) for v in best.values())
              == sum(len(v) for v in good.values()))
        check("foreign-alphabet file is skipped by content", rep["files_csv"] == 3,
              "csv files accepted: %d" % rep["files_csv"])
        check("invalid rows are rejected on replay", rep["rejected_invalid"] >= 1,
              "rejected %d" % rep["rejected_invalid"])
        check("all pids covered", not rep["missing"])


def test_bridge_subsumes_merge():
    print("pooled bridging subsumes min-merge and finds crossovers")
    from .bridge import bridge_pid
    p = Puzzle.demo_lrx(12)
    bk = get_backend("cpu")
    s0, w = p.random_scramble(18, np.random.default_rng(7))
    longer = list(w[:2]) + _identity_pad(p, 0) + _identity_pad(p, 2) + list(w[2:])
    out = bridge_pid(p, s0, [longer, list(w)], radius=1, bk=bk)
    check("pool returns at most the incumbent", len(out) <= len(w))
    check("bridged path solves", p.solves(s0, out))
    out2 = bridge_pid(p, s0, [longer], radius=2, bk=bk)
    check("single-path pool still shortens", len(out2) < len(longer))


def test_journal_resume():
    print("ladder journal: unproven stays open, proven does not")
    from .ladder import HIT, NONE, UNPROVEN, Journal
    with tempfile.TemporaryDirectory() as td:
        jp = Path(td) / "j.jsonl"
        j = Journal(jp)
        j.write({"pid": 1, "i": 0, "j": 5, "verdict": NONE, "word": None})
        j.write({"pid": 1, "i": 5, "j": 10, "verdict": UNPROVEN, "word": None})
        j.write({"pid": 2, "i": 0, "j": 5, "verdict": HIT, "word": "L.R"})
        j2 = Journal(jp)
        check("proven window is closed on resume", j2.is_closed(1, 0, 5))
        check("unproven window re-opens on resume", not j2.is_closed(1, 5, 10))
        check("hit is recoverable", len(j2.hits()) == 1)


def test_suffix_probe():
    print("suffix re-solve")
    from .suffix import make_table_solver, resolve_suffix
    from .table import BfsTable
    p = Puzzle.demo_lrx(10)
    t = BfsTable.build(p, 5, root=p.solved)
    solver = make_table_solver(t, p, extend=1)
    s0, w = p.random_scramble(16, np.random.default_rng(8))
    infl = list(w) + _identity_pad(p, 2)
    new, outcome = resolve_suffix(p, s0, infl, 6, solver)
    check("inflated suffix is beaten", outcome == "beat", outcome)
    check("suffix splice solves", p.solves(s0, new))


def test_real_puzzles_load():
    print("real puzzle definitions load and self-verify")
    from .cli import PRESETS, REPO
    ok = []
    for name, (info, _t) in PRESETS.items():
        f = REPO / info
        if not f.exists():
            continue
        pz = Puzzle.from_puzzle_info(f, name)
        s, w = pz.random_scramble(12, np.random.default_rng(9))
        ok.append((name, pz.solves(s, w)))
    check("every available puzzle_info loads and round-trips",
          all(v for _n, v in ok), str(ok))


def run_all():
    _FAIL.clear()
    for fn in (test_puzzle_loading, test_positive_control_window,
               test_positive_control_table, test_extend_reaches_deeper,
               test_backend_parity, test_colour_beats_permutation,
               test_band_semantics, test_merge_safety, test_bridge_subsumes_merge,
               test_journal_resume, test_suffix_probe, test_real_puzzles_load):
        fn()
    print("")
    if _FAIL:
        print("FAILED: %s" % ", ".join(_FAIL))
        return 1
    print("all checks passed")
    return 0
