"""Do random-walk sparse-Q labels contradict the exact BFS anchors?

The two training streams label overlapping states by different rules:

  * anchors  -- exact BFS distance for the state and all 24 children (ground truth);
  * walks    -- Q(s, undo) = p-1 and Q(s, next) = p+1, where p is the WALK INDEX.

Those agree only when the walk is geodesic. Non-backtracking forbids only the immediate
inverse, so it does NOT guarantee geodesity: if a generator has order 3 then `g, g` has
walk index 2 but lands at true distance 1. Any such pivot teaches the model a target that
the anchor stream simultaneously contradicts.

This measures, over sampled pivots whose true distance is known from the d<=6 table:
  * how often p != d(s), and by how much;
  * how often the two labelled entries (undo / next) disagree with the child's TRUE
    distance, and in which direction;
  * the same broken down by pivot band, since only shallow pivots are in-table.

It reports rates -- it does not change training. A high rate would justify either
clamping p to the exact distance when the state is in-table, or excluding in-table
pivots from the walk stream and letting anchors cover that depth.

Usage:
  python tetraminx/scripts/57_label_consistency.py --n 200000
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))
sys.path.insert(0, str(PROJECT / "src"))


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=40)
    ap.add_argument("--pivot-tilt", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "tetraminx" / "data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    trainer = _load("t51", "51_train_sparse_q.py")
    dev = args.device
    puzzle = trainer.Tetraminx.load(args.data_dir / "puzzle_info.json")
    names = list(puzzle.move_names)
    gen = torch.tensor([puzzle.generators[n] for n in names], dtype=torch.int64, device=dev)
    inv = torch.tensor([names.index(puzzle.inverse_name(n)) for n in names],
                       dtype=torch.int64, device=dev)
    solved = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=dev)

    z = np.load(args.data_dir / "bfs_endgame.npz")
    hashes = torch.from_numpy(z["hashes"]).to(dev)
    depths = torch.from_numpy(z["depths"]).to(dev)
    ztab = torch.from_numpy(z["ztab"]).to(dev)
    max_depth = int(z["max_depth"])
    print(f"endgame table: {hashes.numel():,} states at d<={max_depth}")

    def lookup(states: torch.Tensor):
        """-> (depth, in_table). Depth is meaningless where in_table is False."""
        h = torch.zeros(states.size(0), dtype=torch.int64, device=states.device)
        for i in range(states.size(1)):
            h = torch.bitwise_xor(h, ztab[i].index_select(0, states[:, i]))
        pos = torch.searchsorted(hashes, h).clamp_max(hashes.numel() - 1)
        hit = hashes.index_select(0, pos) == h
        return depths.index_select(0, pos).long(), hit

    g = torch.Generator(device=dev)
    g.manual_seed(args.seed)
    sampler = trainer.SparseQSampler(gen, inv, solved, args.k_min, args.k_max,
                                     args.pivot_tilt, g)

    tot = in_tab = 0
    p_ne_d = 0
    diff_hist: dict[int, int] = {}
    undo_bad = next_bad = undo_n = next_n = 0
    band_tot: dict[int, int] = {}
    band_in: dict[int, int] = {}
    band_bad: dict[int, int] = {}

    ord_n_acc = [0, 0, 0, 0]   # (both-in-table, correct, tie, inverted)
    gap_acc = [0.0]
    done = 0
    chunk = 16384
    while done < args.n:
        b = min(chunk, args.n - done)
        states, pivots, prev, nxt = sampler.sample(b)
        d, hit = lookup(states)
        tot += b
        in_tab += int(hit.sum())
        # p vs true distance, only where the pivot state is in the table
        if int(hit.sum()):
            pv = pivots[hit]
            dv = d[hit]
            diff = (pv - dv)
            p_ne_d += int((diff != 0).sum())
            for val, cnt in zip(*[x.tolist() for x in torch.unique(diff, return_counts=True)]):
                diff_hist[int(val)] = diff_hist.get(int(val), 0) + int(cnt)
            for band in range(0, 8):
                sel = pv == band
                if int(sel.sum()):
                    band_tot[band] = band_tot.get(band, 0) + int(sel.sum())
                    band_bad[band] = band_bad.get(band, 0) + int((diff[sel] != 0).sum())
        for band in range(0, 8):
            sel = pivots == band
            if int(sel.sum()):
                band_in[band] = band_in.get(band, 0) + int((sel & hit).sum())
        # the two LABELLED children: compare label to the child's true distance
        rows = torch.arange(b, device=dev)
        for tag, act, lab in (("undo", prev, pivots - 1), ("next", nxt, pivots + 1)):
            child = torch.gather(states, 1, gen[act])
            cd, chit = lookup(child)
            if int(chit.sum()):
                bad = (lab[chit] != cd[chit])
                if tag == "undo":
                    undo_n += int(chit.sum()); undo_bad += int(bad.sum())
                else:
                    next_n += int(chit.sum()); next_bad += int(bad.sum())
        # THE DECISION-RELEVANT CHECK. A uniform offset in p is harmless: sparse-Q is a
        # RELATIVE objective, so if p overstates by k both labels shift by k and the
        # ordering survives. What actually damages training is the ORDER inverting --
        # the label asserts undo is 2 closer than next, so if truly d(undo) >= d(next)
        # the model is being taught the wrong ranking on a state where truth is known.
        cu = torch.gather(states, 1, gen[prev])
        cn = torch.gather(states, 1, gen[nxt])
        du, hu = lookup(cu)
        dn, hn = lookup(cn)
        both = hu & hn
        if int(both.sum()):
            a, c = du[both], dn[both]
            ord_n_acc[0] += int(both.sum())
            ord_n_acc[1] += int((a < c).sum())     # correct direction
            ord_n_acc[2] += int((a == c).sum())    # tie: the gap-of-2 is fictional
            ord_n_acc[3] += int((a > c).sum())     # INVERTED
            gap_acc[0] += float((c - a).float().sum())
        done += b

    print(f"\nsampled {tot:,} pivots, k in [{args.k_min},{args.k_max}], tilt {args.pivot_tilt}")
    print(f"pivot state IS in the d<={max_depth} table: {in_tab:,} "
          f"({100.0 * in_tab / tot:.2f}%)  <- the only rows where the streams can be compared")
    if in_tab:
        print(f"of those, walk index p != true distance d(s): {p_ne_d:,} "
              f"({100.0 * p_ne_d / in_tab:.2f}%)")
        print(f"\n  p - d(s)   count     share")
        for k in sorted(diff_hist):
            print(f"  {k:+8d} {diff_hist[k]:8,} {100.0 * diff_hist[k] / in_tab:8.2f}%")
        print(f"\n  {'pivot p':>8} {'in-table':>9} {'p != d':>8} {'rate':>7}")
        for band in sorted(band_tot):
            bt, bb = band_tot[band], band_bad.get(band, 0)
            print(f"  {band:8d} {bt:9,} {bb:8,} {100.0 * bb / bt:6.2f}%")
    if undo_n:
        print(f"\nlabelled UNDO child in table: {undo_n:,}; label != true distance: "
              f"{undo_bad:,} ({100.0 * undo_bad / undo_n:.2f}%)")
    if next_n:
        print(f"labelled NEXT child in table: {next_n:,}; label != true distance: "
              f"{next_bad:,} ({100.0 * next_bad / next_n:.2f}%)")
    n, ok, tie, inv = ord_n_acc
    if n:
        print(f"\nORDERING (both labelled children in table): {n:,} cases")
        print(f"  d(undo) <  d(next)  CORRECT   {ok:,} ({100.0 * ok / n:.2f}%)")
        print(f"  d(undo) == d(next)  TIE       {tie:,} ({100.0 * tie / n:.2f}%)  "
              f"<- label asserts a gap of 2 that does not exist")
        print(f"  d(undo) >  d(next)  INVERTED  {inv:,} ({100.0 * inv / n:.2f}%)  "
              f"<- label teaches the WRONG ranking")
        print(f"  mean true gap d(next)-d(undo) = {gap_acc[0] / n:.3f}  (label always asserts 2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
