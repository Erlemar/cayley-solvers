"""Deep window shortening inside a GENERATOR SUBGROUP (4x4x4 colour cube).

WHY. The full generator set branches at 19.2, so a ball of radius r costs 19.2^r and the
affordable reach is ~10 (`70_window_reduce.py --radius 5`). But the corpus is not
generator-uniform: 42.5% of all moves are INNER slices, and the paths contain long runs
that use nothing else (159 runs of length >= 8, longest 13). The inner slices generate a
subgroup that branches at ~10, so the SAME ball budget reaches 4-6 moves deeper on
exactly those runs. Symmetrically the outer layers generate the 3x3x3 group and the
corpus ends in a 3x3x3 finish (372 outer-only runs of length 18-22).

SOUNDNESS. Restricting the search does not restrict where the answer may be spliced: any
word u with s_i[u] == s_j is a legal replacement for the window regardless of which
generators built it. So this is a strictly cheaper probe of a strictly smaller space, and
every hit is replay-verified before it lands.

The window problem is solved in COLOUR space (see `70_window_reduce.py` for why that is
both correct and stronger than permutation matching here).

    python cube444/scripts/72_subgroup_rewrite.py --target <csv> --subset inner --radius 6
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "cube444_window_reduce", Path(__file__).with_name("70_window_reduce.py"))
wr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wr)

SUBSETS = {
    "inner": ["f1", "f2", "r1", "r2", "d1", "d2"],
    "outer": ["f0", "f3", "r0", "r3", "d0", "d3"],
    "slice1": ["f1", "r1", "d1"],
    "slice2": ["f2", "r2", "d2"],
}


def base_name(m: str) -> str:
    return m[1:] if m.startswith("-") else m


def runs_of(word, names, layers, min_run):
    """Maximal [i, j) ranges whose every move's layer is in `layers`."""
    out = []
    cur = None
    for k, m in enumerate(word):
        if base_name(names[m]) in layers:
            cur = k if cur is None else cur
        else:
            if cur is not None and k - cur >= min_run:
                out.append((cur, k))
            cur = None
    if cur is not None and len(word) - cur >= min_run:
        out.append((cur, len(word)))
    return out


def pairwise_shorten(cube, s_start: np.ndarray, s_end: np.ndarray, limit: int,
                     radius: int, chunk: int, stats):
    """Shortest word (<= 2*radius) taking s_start to s_end, or None if none beats `limit`.

    Two balls instead of one per path index. For a 20-move run the all-index sweep would
    hold 21 balls at once, which is what makes reach 14-16 unaffordable there; endpoints
    only is 2 balls, at the cost of not seeing sub-window rewrites.
    """
    src = torch.as_tensor(np.stack([s_start, s_end]), device=cube.device)
    h, s_idx, dist, par, mv = wr.build_balls(cube, src, radius, chunk)
    BIG = torch.iinfo(torch.int32).max // 4
    d32 = dist.to(torch.int32)
    a = torch.where(s_idx == 0, d32, torch.full_like(d32, BIG))
    b = torch.where(s_idx == 1, d32, torch.full_like(d32, BIG))
    cands = wr.group_min(h, a, b, limit=limit)
    par_c, mv_c = par.cpu().numpy(), mv.cpu().numpy()
    del h, s_idx, dist, par, mv, src, a, b, d32
    if cube.device.type == "cuda":
        torch.cuda.empty_cache()
    for total, e_lo, e_hi in cands:
        u = wr.word_to(par_c, mv_c, e_lo) + cube.inv_word(wr.word_to(par_c, mv_c, e_hi))
        if len(u) >= limit:
            continue
        if np.array_equal(cube.apply_word_np(s_start, u), s_end):
            return u
        stats["phantom"] = stats.get("phantom", 0) + 1
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--subset", type=str, default="inner",
                    help="inner | outer | slice1 | slice2 | comma-list of layer names")
    ap.add_argument("--radius", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=150_000)
    ap.add_argument("--min-run", type=int, default=8)
    ap.add_argument("--max-run", type=int, default=99)
    ap.add_argument("--pairwise", action="store_true",
                    help="MITM the run's two endpoints only (2 balls) instead of every "
                         "index -- the only affordable mode for long runs at high radius")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--data-dir", type=Path, default=wr.DATA)
    ap.add_argument("--progress", type=int, default=25)
    args = ap.parse_args()

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    cube = wr.Cube(args.data_dir, dev)
    layers = set(SUBSETS.get(args.subset, args.subset.split(",")))
    keep = [i for i, n in enumerate(cube.names) if base_name(n) in layers]
    sub = cube.restricted(keep)
    print(f"device {dev}  subset {sorted(layers)} -> {sub.n_gen} generators, "
          f"radius {args.radius} (reach {2*args.radius})", flush=True)

    tests = wr.load_tests(args.data_dir)
    rows = wr.load_paths(cube, args.target)
    base_total = sum(len(v) for v in rows.values())
    assert wr.verify_all(cube, tests, rows) == 0, "input does not replay"
    print(f"target {args.target.name}: {len(rows)} pids, {base_total:,} moves", flush=True)

    targets = []
    for pid, w in sorted(rows.items()):
        for (i, j) in runs_of(w, cube.names, layers, args.min_run):
            if j - i <= args.max_run:
                targets.append((pid, i, j))
    targets.sort(key=lambda t: -(t[2] - t[1]))
    print(f"{len(targets)} runs of length {args.min_run}-{args.max_run} "
          f"(longest {targets[0][2]-targets[0][1] if targets else 0})", flush=True)

    g2s = {g: k for k, g in enumerate(keep)}
    stats, saved, t0 = {}, 0, time.time()
    for n, (pid, i, j) in enumerate(targets):
        w = rows[pid]
        if j > len(w):
            continue
        seg = w[i:j]
        if any(m not in g2s for m in seg):
            continue                                  # path changed under an earlier splice
        states = cube.path_states_np(tests[pid], w)
        seg_sub = [g2s[m] for m in seg]
        if args.pairwise:
            u = pairwise_shorten(sub, states[i], states[j], len(seg),
                                 args.radius, args.chunk, stats)
            got = u if u is not None else seg_sub
        else:
            got = wr.reduce_path(sub, states[i], seg_sub, args.radius, args.chunk, stats)
        if len(got) < len(seg):
            repl = [keep[m] for m in got]
            cand = w[:i] + repl + w[j:]
            if np.array_equal(cube.apply_word_np(tests[pid], cand), cube.solved_np):
                saved += len(seg) - len(repl)
                print(f"  HIT pid {pid} run [{i},{j}): {len(seg)} -> {len(repl)} "
                      f"(-{len(seg)-len(repl)})", flush=True)
                rows[pid] = cand
            else:
                stats["reject"] = stats.get("reject", 0) + 1
        if args.progress and (n + 1) % args.progress == 0:
            el = time.time() - t0
            print(f"  {n+1}/{len(targets)} runs, saved {saved}, {el:.0f}s "
                  f"({el/(n+1):.2f}s/run)", flush=True)

    total = sum(len(v) for v in rows.values())
    print(f"\nsubset {args.subset} radius {args.radius}: saved {saved} "
          f"({base_total:,} -> {total:,})")
    print(f"phantom rejects: {stats.get('phantom', 0)}  splice rejects: {stats.get('reject', 0)}")
    bad = wr.verify_all(cube, tests, rows)
    print(f"output replay check: {bad} unsolved")
    if bad:
        print("REFUSING TO WRITE -- replay failed")
        return 4
    if args.out and saved:
        wr.write_paths(cube, rows, args.out)
        print(f"wrote {args.out}")
    elif args.out:
        print("no saving -- not writing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
