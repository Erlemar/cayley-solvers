"""Pooled cross-trajectory bridging for the 4x4x4 colour cube.

GENERALISES tetraminx's `60_splice_graph.py`. That script spliced where two trajectories
share a state exactly (d = 0 crossover). Here every waypoint of every known solution for
a pid is expanded into a ball of radius r, so two trajectories that merely pass NEAR each
other can still be joined by a bridge of up to 2r moves:

    new length = g(x) + d(x, z) + d(z, y) + h(y)

where g is the cheapest cost-from-start over all pooled paths reaching waypoint x, and h
the cheapest cost-to-solved from waypoint y. Per colliding state z that is one amin over
(g + d) and one over (h + d) -- no pair enumeration.

Because the pool holds EVERY path we own for the pid, this subsumes in one pass:
  * n-way per-pid min-merge (rule 26)    -- x = start, y = solved, same path
  * exact crossover splicing             -- d1 = d2 = 0
  * within-path window shortening        -- both waypoints from the same path
  * bridges BETWEEN paths                -- the new part
and it does rewrite-all-then-merge rather than merge-then-rewrite, which the tetraminx
HANDOFF flags as the lossy order (`65_rewrite_alternatives.py`).

Sources are discovered by CONTENT (rule 26b) and every one is replay-verified before its
states are allowed into the pool; every emitted path is replay-verified again.

    python cube444/scripts/71_bridge_merge.py --base cube444/submissions/leader_46718_d7_rot_targeted_3x3.csv \
        --scan-root cube444 --scan-root data/community_subs --radius 3 \
        --out cube444/submissions/leader_bridged.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import os
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "cube444_window_reduce", Path(__file__).with_name("70_window_reduce.py"))
wr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wr)


def discover(cube: wr.Cube, tests, roots, verbose=True):
    """Every CSV under `roots` that parses as cube444 AND replays 100%."""
    alpha = set(cube.names)
    out = {}
    for root in roots:
        for dirpath, _dn, filenames in os.walk(root):
            if ".git" in dirpath or ".venv" in dirpath:
                continue
            for fn in filenames:
                if not fn.endswith(".csv"):
                    continue
                p = Path(dirpath) / fn
                try:
                    with io.open(p, encoding="utf-8", newline="") as fh:
                        head = fh.readline()
                        if "initial_state_id" not in head or "path" not in head:
                            continue
                        probe = fh.readline().strip()
                    body = probe.split(",", 1)[1].strip().strip('"')
                    if not all(m in alpha for m in body.split(".")[:20]):
                        continue
                    rows = wr.load_paths(cube, p)
                except Exception:
                    continue
                bad = 0
                for pid, w in rows.items():
                    if pid not in tests or not np.array_equal(
                            cube.apply_word_np(tests[pid], w), cube.solved_np):
                        bad += 1
                        break
                if bad or not rows:
                    continue
                out[str(p)] = rows
                if verbose:
                    tot = sum(len(v) for v in rows.values())
                    print(f"  source {p.name:60s} {len(rows):5d} pids {tot:8,d}", flush=True)
    return out


group_min = wr.group_min          # shared with 70/72


def bridge_pid(cube: wr.Cube, s0: np.ndarray, paths, radius: int, chunk: int, stats):
    """paths: list of move-index lists, all valid solutions for s0. Returns best word."""
    incumbent = min(paths, key=len)
    best = list(incumbent)
    # pool waypoints: state -> (g, path, idx) and (h, path, idx)
    pool = {}
    for pi, w in enumerate(paths):
        st = cube.path_states_np(s0, w)
        L = len(w)
        for k in range(L + 1):
            key = st[k].tobytes()
            rec = pool.get(key)
            if rec is None:
                pool[key] = [k, pi, k, L - k, pi, k, st[k]]
            else:
                if k < rec[0]:
                    rec[0], rec[1], rec[2] = k, pi, k
                if L - k < rec[3]:
                    rec[3], rec[4], rec[5] = L - k, pi, k
    recs = list(pool.values())
    src = torch.as_tensor(np.stack([r[6] for r in recs]), device=cube.device)
    g_lab = torch.as_tensor(np.array([r[0] for r in recs], dtype=np.int32), device=cube.device)
    h_lab = torch.as_tensor(np.array([r[3] for r in recs], dtype=np.int32), device=cube.device)
    stats["waypoints"] = stats.get("waypoints", 0) + len(recs)

    h, s_idx, dist, par, mv = wr.build_balls(cube, src, radius, chunk)
    a = g_lab[s_idx.long()] + dist.to(torch.int32)
    b = h_lab[s_idx.long()] + dist.to(torch.int32)
    cands = group_min(h, a, b, limit=len(best))
    if not cands:
        return best
    par_c, mv_c, si_c, di_c = (par.cpu().numpy(), mv.cpu().numpy(),
                               s_idx.cpu().numpy(), dist.cpu().numpy())
    for total, e_lo, e_hi in cands:
        r_lo, r_hi = recs[int(si_c[e_lo])], recs[int(si_c[e_hi])]
        w1 = wr.word_to(par_c, mv_c, e_lo)
        w2 = wr.word_to(par_c, mv_c, e_hi)
        cand = list(paths[r_lo[1]][:r_lo[2]]) + w1 + cube.inv_word(w2) + list(paths[r_hi[4]][r_hi[5]:])
        if len(cand) >= len(best):
            continue
        if not np.array_equal(cube.apply_word_np(s0, cand), cube.solved_np):
            stats["phantom"] = stats.get("phantom", 0) + 1
            continue
        best = cand
        break
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--scan-root", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=150_000)
    ap.add_argument("--max-paths", type=int, default=8, help="distinct pooled paths per pid")
    ap.add_argument("--slack", type=int, default=8,
                    help="pool a path only if it is within this many moves of the incumbent")
    ap.add_argument("--pids", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--data-dir", type=Path, default=wr.DATA)
    ap.add_argument("--progress", type=int, default=100)
    args = ap.parse_args()

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    cube = wr.Cube(args.data_dir, dev)
    tests = wr.load_tests(args.data_dir)
    base = wr.load_paths(cube, args.base)
    base_total = sum(len(v) for v in base.values())
    print(f"base {args.base.name}: {len(base)} pids, {base_total:,} moves", flush=True)
    assert wr.verify_all(cube, tests, base) == 0, "base does not replay"

    roots = args.scan_root or [PROJECT / "cube444"]
    print("discovering sources by content...", flush=True)
    srcs = discover(cube, tests, roots)
    print(f"{len(srcs)} verified sources", flush=True)

    pool = {pid: {tuple(w)} for pid, w in base.items()}
    for rows in srcs.values():
        for pid, w in rows.items():
            if pid in pool:
                pool[pid].add(tuple(w))

    todo = sorted(base)
    if args.pids:
        if ":" in args.pids:
            a, b = args.pids.split(":")
            todo = [p for p in todo if int(a) <= p < int(b)]
        else:
            todo = [int(x) for x in args.pids.split(",")]

    stats = {}
    saved = 0
    t0 = time.time()
    npool = 0
    for n, pid in enumerate(todo):
        cands = sorted(pool[pid], key=len)
        inc = len(cands[0])
        cands = [c for c in cands if len(c) <= inc + args.slack][:args.max_paths]
        npool += len(cands)
        best = bridge_pid(cube, tests[pid], [list(c) for c in cands],
                          args.radius, args.chunk, stats)
        if len(best) < len(base[pid]):
            d = len(base[pid]) - len(best)
            saved += d
            print(f"  HIT pid {pid}: {len(base[pid])} -> {len(best)}  (-{d}, "
                  f"{len(cands)} pooled paths)", flush=True)
            base[pid] = best
        if args.progress and (n + 1) % args.progress == 0:
            el = time.time() - t0
            print(f"  {n+1}/{len(todo)} pids, saved {saved}, {el:.0f}s "
                  f"({el/(n+1):.2f}s/pid, {npool/(n+1):.1f} paths/pid)", flush=True)

    total = sum(len(v) for v in base.values())
    print(f"\nradius {args.radius}: saved {saved} moves ({base_total:,} -> {total:,})")
    print(f"waypoints expanded: {stats.get('waypoints', 0):,}  "
          f"phantom rejects: {stats.get('phantom', 0)}")
    bad = wr.verify_all(cube, tests, base)
    print(f"output replay check: {bad} unsolved")
    if bad:
        print("REFUSING TO WRITE -- replay failed")
        return 4
    if args.out and saved:
        wr.write_paths(cube, base, args.out)
        print(f"wrote {args.out}")
    elif args.out:
        print("no saving -- not writing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
