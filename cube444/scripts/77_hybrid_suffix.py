"""Suffix re-solve with an EXACT endgame: heuristic beam forward, d6 table backward.

The plain beam (`76_suffix_resolve.py`) matches the file's suffix at k=12 but starts
failing to find any solution at k=16-20 -- a width limit, since it must cover the whole
suffix heuristically. Here the last 6 plies are removed from its burden: every beam state
is probed against the exact 67M-state ball around solved (`74_endgame_table.py`), so the
beam only has to reach WITHIN 6 of solved rather than all the way, and any hit yields a
provably optimal finish.

    total = (beam steps) + (exact table depth)

A win needs total < k. This is the strongest probe available for the k=13-24 band, which
is beyond exact ball reach (each exact rung costs 19.2x) and is where any remaining slack
in the file must live -- everything <= 10 is already proven geodesic.

    python cube444/scripts/77_hybrid_suffix.py --target <csv> --k 14,18,22 \
        --beam 65536 --table cube444/data/solved_ball_d6.npz --pids 0:30 --report
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

_spec = importlib.util.spec_from_file_location(
    "cube444_window_reduce", Path(__file__).with_name("70_window_reduce.py"))
wr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wr)
_spec2 = importlib.util.spec_from_file_location(
    "cube444_endgame", Path(__file__).with_name("74_endgame_table.py"))
eg = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(eg)

from cube444.models import load_v_model  # noqa: E402


def score(model, states_u8: torch.Tensor, batch: int) -> torch.Tensor:
    """V(state) for (N,96) uint8, in chunks. Lower = closer to solved."""
    out = torch.empty(states_u8.shape[0], dtype=torch.float16, device=states_u8.device)
    with torch.no_grad():
        for i in range(0, states_u8.shape[0], batch):
            chunk = states_u8[i:i + batch].long()
            out[i:i + batch] = model(chunk).flatten().to(torch.float16)
    return out


def hybrid_solve(cube, model, h_tab, d_tab, start: np.ndarray, limit: int,
                 beam: int, batch: int, max_steps: int):
    """Shortest word start->solved of length < limit found by beam+table, else None."""
    dev = cube.device
    cur = torch.as_tensor(start, device=dev)[None, :]
    hist = []                                     # (parent_idx, move) per kept level
    best = None
    for step in range(1, max_steps + 1):
        if best is not None and step >= len(best):
            break
        kids = cur[:, cube.gen].reshape(-1, wr.STATE_SIZE)
        n_par = cur.shape[0]
        par = torch.arange(n_par, dtype=torch.int32, device=dev).repeat_interleave(cube.n_gen)
        mv = torch.arange(cube.n_gen, dtype=torch.int8, device=dev).repeat(n_par)
        h = cube.hash(kids)
        o = torch.argsort(h)
        u = torch.ones_like(o, dtype=torch.bool)
        u[1:] = h[o][1:] != h[o][:-1]
        keep = o[u]
        kids, par, mv, h = kids[keep], par[keep], mv[keep], h[keep]
        # exact finish?
        dep = eg.probe(h_tab, d_tab, h)
        cand = torch.nonzero(dep >= 0, as_tuple=True)[0]
        if cand.numel():
            tot = step + dep[cand]
            j = int(torch.argmin(tot).item())
            total = int(tot[j].item())
            if total < limit and (best is None or total < len(best)):
                e = int(cand[j].item())
                word = [int(mv[e].item())]
                p = int(par[e].item())
                for s in range(step - 1, 0, -1):
                    pp, mm = hist[s - 1]
                    word.append(int(mm[p].item()))
                    p = int(pp[p].item())
                word.reverse()
                z = cube.apply_word_np(start, word)
                tail = eg.descend(cube, h_tab, d_tab, z)
                if tail is not None:
                    full = word + tail
                    if len(full) < limit and np.array_equal(
                            cube.apply_word_np(start, full), cube.solved_np):
                        best = full
        if step == max_steps:
            break
        v = score(model, kids, batch)
        k = min(beam, v.numel())
        idx = torch.topk(v, k, largest=False).indices
        cur = kids[idx]
        hist.append((par[idx], mv[idx]))
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--table", type=Path, default=PROJECT / "data" / "solved_ball_d6.npz")
    ap.add_argument("--checkpoint", type=Path,
                    default=PROJECT / "models" / "c_bells2" / "epoch_0399.pt")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--k", type=str, default="14,18,22")
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--batch", type=int, default=1 << 15)
    ap.add_argument("--max-steps", type=int, default=0, help="0 => k-7")
    ap.add_argument("--pids", type=str, default="")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--subset", type=str, default="",
                    help="restrict the beam's generators (use with a matching --table)")
    ap.add_argument("--auto-outer-tail", action="store_true",
                    help="ignore --k; per pid use the terminal outer-only run (the 3x3x3 "
                         "finish of a reduction solve) as the suffix")
    ap.add_argument("--min-tail", type=int, default=12)
    ap.add_argument("--slack", type=int, default=0,
                    help="search up to k+slack so the run measures the beam's own reach; "
                         "splicing still requires strictly fewer than k")
    ap.add_argument("--progress", type=int, default=10)
    args = ap.parse_args()

    full = wr.Cube(args.data_dir, args.device)
    cube = full
    keep = None
    if args.subset:
        SUB = {"inner": ["f1", "f2", "r1", "r2", "d1", "d2"],
               "outer": ["f0", "f3", "r0", "r3", "d0", "d3"]}
        layers = set(SUB.get(args.subset, args.subset.split(",")))
        keep = [i for i, n in enumerate(full.names)
                if (n[1:] if n.startswith("-") else n) in layers]
        cube = full.restricted(keep)
        print(f"subset {args.subset}: {cube.n_gen} generators", flush=True)
    z = np.load(args.table)
    h_tab = torch.as_tensor(z["h"], device=cube.device)
    d_tab = torch.as_tensor(z["d"], device=cube.device)
    model = load_v_model(args.checkpoint, device=args.device,
                         dtype=torch.bfloat16 if args.bf16 else torch.float32)
    tests = wr.load_tests(args.data_dir)
    rows = wr.load_paths(full, args.target)
    base_total = sum(len(v) for v in rows.values())
    ks = [int(x) for x in args.k.split(",")]
    todo = sorted(rows)
    if args.pids:
        if ":" in args.pids:
            a, b = args.pids.split(":")
            todo = [p for p in todo if int(a) <= p < int(b)]
        else:
            todo = [int(x) for x in args.pids.split(",")]
    print(f"table {h_tab.numel():,} states  beam {args.beam}  k={ks}  pids={len(todo)}",
          flush=True)

    max_dep = int(d_tab.max())
    saved, tried, t0 = 0, 0, time.time()
    for n, pid in enumerate(todo):
        w = rows[pid]
        L = len(w)
        states = full.path_states_np(tests[pid], w)
        if args.auto_outer_tail:
            layers = {"f0", "f3", "r0", "r3", "d0", "d3"}
            t = 0
            while t < L and (full.names[w[L - 1 - t]].lstrip("-")) in layers:
                t += 1
            ks_pid = [t] if t >= args.min_tail else []
        else:
            ks_pid = ks
        for k in ks_pid:
            if k >= L:
                continue
            i = L - k
            # --slack raises the acceptance limit ABOVE k so the run doubles as its own
            # control: a null result at slack=0 cannot tell "the file is optimal" from
            # "our beam is too weak to reach this depth at all". With slack>0 the length
            # actually returned answers that -- if the beam cannot even match k, the
            # null is uninformative. Splicing still requires a STRICT improvement.
            limit = k + args.slack
            steps = args.max_steps or max(1, limit - 1 - max_dep)
            got = hybrid_solve(cube, model, h_tab, d_tab, states[i], limit, args.beam,
                               args.batch, steps)
            tried += 1
            if got is None:
                if args.report:
                    print(f"  pid {pid} k={k}: nothing below {limit}", flush=True)
                continue
            if args.report:
                mark = "WIN" if len(got) < k else ("=" if len(got) == k else "worse")
                print(f"  pid {pid} k={k}: beam+table {len(got)}  {mark}", flush=True)
            if len(got) >= k:
                continue
            if keep is not None:
                got = [keep[m] for m in got]
            cand = w[:i] + got
            if not np.array_equal(full.apply_word_np(tests[pid], cand), full.solved_np):
                print(f"  pid {pid} k={k}: REJECTED (does not replay)")
                continue
            print(f"  HIT pid {pid} k={k}: suffix {k} -> {len(got)} "
                  f"(total {L} -> {len(cand)})", flush=True)
            saved += k - len(got)
            rows[pid] = cand
            break
        if args.progress and (n + 1) % args.progress == 0:
            el = time.time() - t0
            print(f"  {n+1}/{len(todo)} pids, {tried} solves, saved {saved}, "
                  f"{el:.0f}s ({el/(n+1):.1f}s/pid)", flush=True)

    total = sum(len(v) for v in rows.values())
    print(f"\nhybrid suffix: {tried} solves, saved {saved}  ({base_total:,} -> {total:,})")
    bad = wr.verify_all(full, tests, rows)
    print(f"output replay check: {bad} unsolved")
    if bad:
        print("REFUSING TO WRITE -- replay failed")
        return 4
    if args.out and saved and not args.report:
        wr.write_paths(full, rows, args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
