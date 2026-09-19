"""Endgame (tail) shortening against a shared exact ball around SOLVED.

WHY THIS BUYS A RUNG THE FULL SWEEP CANNOT. `70_window_reduce.py` reaches 2r because it
must build a ball around BOTH window endpoints, and each rung costs 19.2x. But every one
of the 1043 paths ends at the SAME state, so a ball around solved is built ONCE and
reused by every pid and every tail index:

    reach(tail) = r (front ball around s_i)  +  D (shared table around solved)

At r=5, D=6 that is reach 11 for tails against reach 10 for arbitrary windows, at a small
fraction of the cost of lifting the whole sweep to r=6 (which would be ~22 h).
Tetraminx's equivalent (`59_mitm_rewrite.py --endgame-cuts`) is where its last real gains
came from, so the tail is the right place to spend the extra rung.

Colour space throughout -- the table is a ball of COLOURINGS, so it already quotients out
the 191M-element centre stabiliser (see `70_window_reduce.py` for why that is both valid
and stronger than permutation matching here).

    python cube444/scripts/74_endgame_table.py --depth 6 --save cube444/data/solved_ball_d6.npz
    python cube444/scripts/74_endgame_table.py --load cube444/data/solved_ball_d6.npz \
        --target cube444/submissions/leader_r5.csv --radius 5 --out cube444/submissions/leader_tail.csv
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


def build_table(cube: wr.Cube, depth: int, chunk: int):
    """BFS ball of COLOURINGS around solved. Returns (sorted hashes int64, depths int8)."""
    dev = cube.device
    t0 = time.time()
    cur = torch.as_tensor(cube.solved_np, device=dev)[None, :]
    H = [cube.hash(cur)]
    D = [torch.zeros(1, dtype=torch.int8, device=dev)]
    seen = H[0].clone()
    for d in range(1, depth + 1):
        # The deepest level is never expanded and never dedup-checked again, so keeping
        # its 64M x 96 states (6.1 GB) or folding it into `seen` is pure cost.
        last = d == depth
        parts_h, parts_s = [], []
        for s in range(0, cur.shape[0], chunk):
            blk = cur[s:s + chunk]
            kids = blk[:, cube.gen].reshape(-1, wr.STATE_SIZE)
            kh = cube.hash(kids)
            pos = torch.searchsorted(seen, kh).clamp_(max=seen.shape[0] - 1)
            sel = torch.nonzero(seen[pos] != kh, as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            khs = kh[sel]
            o = torch.argsort(khs)
            u = torch.ones_like(o, dtype=torch.bool)
            u[1:] = khs[o][1:] != khs[o][:-1]
            pick = sel[o[u]]
            parts_h.append(kh[pick])
            if not last:
                parts_s.append(kids[pick])
            del kids, kh
        if not parts_h:
            break
        kh = torch.cat(parts_h)
        kids = None if last else torch.cat(parts_s)
        del parts_h, parts_s
        o = torch.argsort(kh)
        kh = kh[o]
        u = torch.ones_like(kh, dtype=torch.bool)
        u[1:] = kh[1:] != kh[:-1]
        kh = kh[u]
        if not last:
            kids = kids[o][u]
        H.append(kh)
        D.append(torch.full((kh.numel(),), d, dtype=torch.int8, device=dev))
        n_new = kh.numel()
        if not last:
            seen = torch.sort(torch.cat([seen, kh])).values
            cur = kids
        print(f"  depth {d}: {n_new:,} new (cum {int(seen.numel()) + (n_new if last else 0):,})"
              f"  {time.time()-t0:.0f}s", flush=True)
        del kids
        if last:
            break
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    h = torch.cat(H)
    dep = torch.cat(D)
    o = torch.argsort(h)
    return h[o].contiguous(), dep[o].contiguous()


def probe(h_tab, d_tab, q):
    """Depth of each query hash in the table, or -1."""
    pos = torch.searchsorted(h_tab, q).clamp_(max=h_tab.numel() - 1)
    hit = h_tab[pos] == q
    return torch.where(hit, d_tab[pos].to(torch.int32), torch.full_like(pos, -1, dtype=torch.int32))


def descend(cube: wr.Cube, h_tab, d_tab, state: np.ndarray):
    """Optimal word from `state` to solved using the table. None if not in the ball."""
    st = torch.as_tensor(state, device=cube.device)[None, :]
    d = int(probe(h_tab, d_tab, cube.hash(st))[0].item())
    if d < 0:
        return None
    word, cur = [], state
    while d > 0:
        kids = torch.as_tensor(cur, device=cube.device)[None, :][:, cube.gen].reshape(
            -1, wr.STATE_SIZE)
        dk = probe(h_tab, d_tab, cube.hash(kids))
        nxt = torch.nonzero(dk == d - 1, as_tuple=True)[0]
        if nxt.numel() == 0:
            return None                      # phantom table hit -- caller rejects
        m = int(nxt[0].item())
        word.append(m)
        cur = cur[cube.gen_np[m]]
        d -= 1
    return word


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--subset", type=str, default="",
                    help="restrict generators, e.g. 'outer' -- the outer layers generate "
                         "the 3x3x3 group and branch at 9.37, so the same budget reaches "
                         "~2 plies deeper than the full 19.21-branching set")
    ap.add_argument("--save", type=Path)
    ap.add_argument("--load", type=Path)
    ap.add_argument("--target", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--radius", type=int, default=5, help="front ball around each tail index")
    ap.add_argument("--min-tail", type=int, default=1, help="smallest tail length tested")
    ap.add_argument("--max-tail", type=int, default=16, help="largest tail length tested")
    ap.add_argument("--chunk", type=int, default=150_000)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--data-dir", type=Path, default=wr.DATA)
    ap.add_argument("--progress", type=int, default=100)
    args = ap.parse_args()

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    cube = wr.Cube(args.data_dir, dev)
    if args.subset:
        SUB = {"inner": ["f1", "f2", "r1", "r2", "d1", "d2"],
               "outer": ["f0", "f3", "r0", "r3", "d0", "d3"]}
        layers = set(SUB.get(args.subset, args.subset.split(",")))
        keep = [i for i, n in enumerate(cube.names)
                if (n[1:] if n.startswith("-") else n) in layers]
        cube = cube.restricted(keep)
        print(f"subset {args.subset}: {cube.n_gen} generators", flush=True)

    if args.load and args.load.exists():
        z = np.load(args.load)
        h_tab = torch.as_tensor(z["h"], device=cube.device)
        d_tab = torch.as_tensor(z["d"], device=cube.device)
        print(f"loaded table {args.load.name}: {h_tab.numel():,} states, "
              f"max depth {int(d_tab.max())}", flush=True)
    else:
        print(f"building solved ball to depth {args.depth}...", flush=True)
        h_tab, d_tab = build_table(cube, args.depth, args.chunk)
        print(f"table: {h_tab.numel():,} states", flush=True)
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.save, h=h_tab.cpu().numpy(), d=d_tab.cpu().numpy())
            print(f"saved {args.save}")
    if not args.target:
        return 0

    tests = wr.load_tests(args.data_dir)
    rows = wr.load_paths(cube, args.target)
    base_total = sum(len(v) for v in rows.values())
    assert wr.verify_all(cube, tests, rows) == 0, "input does not replay"
    print(f"target {args.target.name}: {len(rows)} pids, {base_total:,} moves "
          f"(reach {args.radius}+{int(d_tab.max())})", flush=True)

    saved, phantom, t0 = 0, 0, time.time()
    todo = sorted(rows)
    for n, pid in enumerate(todo):
        w = rows[pid]
        L = len(w)
        lo = max(0, L - args.max_tail)
        hi = max(0, L - args.min_tail)
        if hi <= lo:
            continue
        states = cube.path_states_np(tests[pid], w)
        idxs = list(range(lo, hi))
        src = torch.as_tensor(states[idxs], device=cube.device)
        h, s_idx, dist, par, mv = wr.build_balls(cube, src, args.radius, args.chunk)
        dep = probe(h_tab, d_tab, h)
        # saving = (L - dep) - (index + dist), only where the table hit
        pathidx = torch.as_tensor(np.array(idxs, dtype=np.int32), device=cube.device)[s_idx.long()]
        sav = (L - dep) - (pathidx + dist.to(torch.int32))
        sav = torch.where(dep >= 0, sav, torch.full_like(sav, -1))
        best = int(sav.max().item()) if sav.numel() else -1
        if best > 0:
            order = torch.argsort(sav, descending=True)[:32].cpu().numpy()
            par_c, mv_c = par.cpu().numpy(), mv.cpu().numpy()
            pi_c, di_c = pathidx.cpu().numpy(), dist.cpu().numpy()
            applied = False
            for e in order:
                i = int(pi_c[e])
                w1 = wr.word_to(par_c, mv_c, int(e))
                z = cube.apply_word_np(states[i], w1)
                w2 = descend(cube, h_tab, d_tab, z)
                if w2 is None:
                    phantom += 1
                    continue
                cand = w[:i] + w1 + w2
                if len(cand) >= L:
                    continue
                if not np.array_equal(cube.apply_word_np(tests[pid], cand), cube.solved_np):
                    phantom += 1
                    continue
                print(f"  HIT pid {pid}: tail from {i}: {L - i} -> {len(w1)+len(w2)} "
                      f"(total {L} -> {len(cand)})", flush=True)
                saved += L - len(cand)
                rows[pid] = cand
                applied = True
                break
            if not applied:
                pass
        del h, s_idx, dist, par, mv, src, dep, sav
        if cube.device.type == "cuda":
            torch.cuda.empty_cache()
        if args.progress and (n + 1) % args.progress == 0:
            el = time.time() - t0
            print(f"  {n+1}/{len(todo)} pids, saved {saved}, {el:.0f}s "
                  f"({el/(n+1):.2f}s/pid)", flush=True)

    total = sum(len(v) for v in rows.values())
    print(f"\ntail reach {args.radius}+{int(d_tab.max())}: saved {saved} "
          f"({base_total:,} -> {total:,})   phantom rejects {phantom}")
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
