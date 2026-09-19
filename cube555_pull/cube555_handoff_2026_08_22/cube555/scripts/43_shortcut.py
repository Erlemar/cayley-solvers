"""Shorten solved paths by splicing exact optimal words into slack windows.

    python cube555/scripts/43_shortcut.py in.csv out.csv [--max-window 20]

A beam path is ~167 moves for a state at true distance ~70, so it is locally very
inefficient. For every window [i, i+L) of the path, the net group element is

    w = s_i^-1 o s_{i+L}          (array form: w = inv(s_i)[s_{i+L}])

because states here ARE group elements (solved == identity) and `apply` right-multiplies.
If w lands in the exact d<=D ball then d(w) is known EXACTLY, and when d(w) < L the window
can be replaced by w's optimal word -- recovered by descending argmin over the ball's
stored exact Q, one depth per move. Longest windows are tried first and the whole pass
repeats to a fixed point, since a splice creates new adjacencies.

This is exact, model-free and cheap: one hash lookup per window. Unlike commutation
reduction (measured +0.00% on beam paths, because the beam's own dedup already prevents
collapsible same-axis runs) this catches slack the beam cannot see -- any detour whose net
effect is a short word.

Every rewritten path is replayed against test.csv before it is written.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cube555.puzzle import Cube555  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--max-window", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    puz = Cube555.load(PROJECT / "data" / "puzzle_info.json")
    names = list(puz.move_names)
    G = np.array([puz.generators[n] for n in names], dtype=np.int64)
    central = np.array(puz.solved_state, dtype=np.int64)
    tests = {
        int(r["initial_state_id"]): np.array(
            [int(x) for x in r["initial_state"].split(",")], dtype=np.int64
        )
        for r in csv.DictReader(open(PROJECT / "data" / "test.csv", encoding="utf-8"))
    }

    blob = torch.load(
        PROJECT / "data" / f"anchors_d{args.depth}.pt",
        map_location="cpu",
        weights_only=False,
    )
    dev = args.device
    ball_states = blob["states"].to(dev)
    ball_q = blob["q"].to(dev)
    ball_depth = blob["depth"].to(dev)
    g = torch.Generator(device=dev)
    g.manual_seed(555)
    hv = torch.randint(
        -(2**62), 2**62, (150,), dtype=torch.int64, device=dev, generator=g
    )
    bh = (ball_states.long() * hv).sum(1)
    order = torch.argsort(bh)
    bh, ball_q, ball_depth = bh[order], ball_q[order], ball_depth[order]
    hv_np = hv.cpu().numpy()
    bh_np = bh.cpu().numpy()
    bq_np = ball_q.cpu().numpy()
    bd_np = ball_depth.cpu().numpy()
    print(f"ball d<={args.depth}: {bh_np.size:,} states")

    def lookup(w: np.ndarray) -> int:
        h = int((w.astype(np.int64) * hv_np).sum())
        pos = int(np.searchsorted(bh_np, h))
        if pos >= bh_np.size or bh_np[pos] != h:
            return -1
        return pos

    def optimal_word(pos: int) -> list[str]:
        """A word FOR the group element at `pos` -- not the word that solves it.

        Descending the ball by argmin-Q from x yields u with x o u = e, i.e. u = x^-1.
        The window needs w itself, so return u reversed with each move inverted. Getting
        this backwards produces a legal-looking path that does not solve; the replay
        guard below caught exactly that on 9 of 18 paths.
        """
        out = []
        cur = None
        while True:
            d = int(bd_np[pos])
            if d == 0:
                return [
                    (m[1:] if m.startswith("-") else "-" + m) for m in reversed(out)
                ]
            a = int(bq_np[pos].argmin())
            out.append(names[a])
            cur = (ball_states[pos].cpu().numpy() if cur is None else cur)[G[a]]
            pos = lookup(cur)
            if pos < 0:
                raise RuntimeError("descent left the ball")

    rows = list(csv.DictReader(open(args.src, encoding="utf-8")))
    before = after = 0
    out_rows, n_splice, n_fail = [], 0, 0
    for r in rows:
        pid = int(r["initial_state_id"])
        mv = [m for m in r["path"].split(".") if m]
        before += len(mv)
        changed = True
        while changed:
            changed = False
            st = [tests[pid].copy()]
            for m in mv:
                st.append(st[-1][G[names.index(m)]])
            inv = [np.argsort(s) for s in st]
            i = 0
            new: list[str] = []
            while i < len(mv):
                hit = None
                for L in range(min(args.max_window, len(mv) - i), 5, -1):
                    w = inv[i][st[i + L]]
                    pos = lookup(w)
                    if pos >= 0 and int(bd_np[pos]) < L:
                        hit = (L, optimal_word(pos))
                        break
                if hit:
                    new.extend(hit[1])
                    i += hit[0]
                    n_splice += 1
                    changed = True
                else:
                    new.append(mv[i])
                    i += 1
            mv = new
        cur = tests[pid].copy()
        for m in mv:
            cur = cur[G[names.index(m)]]
        if not np.array_equal(cur, central):
            n_fail += 1
            mv = [m for m in r["path"].split(".") if m]
        after += len(mv)
        out_rows.append((pid, puz.format_path(mv)))

    with open(args.dst, "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(["initial_state_id", "path"])
        for pid, p in sorted(out_rows):
            w_.writerow([pid, p])
    print(
        f"{args.src.name}: {before:,} -> {after:,} moves ({100.0*(after-before)/max(1,before):+.2f}%)"
    )
    print(f"  splices {n_splice}, replay failures (reverted) {n_fail}")
    print(f"wrote {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
