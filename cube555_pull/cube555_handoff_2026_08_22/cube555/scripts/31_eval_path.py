"""Tie-aware Q-head discrimination by REMAINING DEPTH, on near-geodesic path states.

    python cube555/scripts/31_eval_path.py --checkpoint cube555/models/q555_a/epoch_002399.pt

WHY PATH PIVOTS AND NOT RANDOM WALKS. On 444 the depth diagnostic was built from random
walks and asserted the true gap was 2 at every depth. Walks are fully mixed well before
the diameter, so past mixing both children are uniform-random states at distance ~D, the
true gap is ~0, and a PERFECTLY calibrated scorer scores 0.5. An entire "the model is
blind past depth 40" conclusion was that artifact. Here the pivots come from replaying the
35 shipped santa solutions (~93 moves each, all replay-verified), where the remaining
count is a tight upper bound on true distance at exactly the depths the beam operates at.

TIE-AWARE. A saturated head that outputs the same value for many actions would score well
on a naive argmin. Ties are counted as FAILURES; the symptom of getting this wrong is
pair_acc < top1_acc, which is impossible when both are measured honestly.

Chance is 0.500 for pair and 1/30 = 0.033 for top-1.
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
sys.path.insert(0, str(PROJECT.parent / "src"))

from cube555.models import load_model  # noqa: E402
from cube555.puzzle import Cube555  # noqa: E402

N_SANTA = 35
BANDS = [
    (1, 5),
    (6, 10),
    (11, 20),
    (21, 30),
    (31, 40),
    (41, 50),
    (51, 60),
    (61, 70),
    (71, 80),
    (81, 130),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument(
        "--paths",
        type=Path,
        default=None,
        help="solution CSV to replay (default: the shipped baseline)",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    puz = Cube555.load(PROJECT / "data" / "puzzle_info.json")
    names = list(puz.move_names)
    name_i = {n: i for i, n in enumerate(names)}
    G = np.array([puz.generators[n] for n in names], dtype=np.int64)
    inv_i = np.array([name_i[puz.inverse_name(n)] for n in names], dtype=np.int64)
    tests = {
        int(r["initial_state_id"]): np.array(
            [int(x) for x in r["initial_state"].split(",")], dtype=np.int64
        )
        for r in csv.DictReader(open(PROJECT / "data" / "test.csv", encoding="utf-8"))
    }
    src = args.paths or (PROJECT / "data" / "sample_submission.csv")
    paths = {
        int(r["initial_state_id"]): [m for m in r["path"].split(".") if m]
        for r in csv.DictReader(open(src, encoding="utf-8"))
    }

    states, on_move, undo_move, remaining = [], [], [], []
    for pid in range(N_SANTA):
        if pid not in paths:
            continue
        cur = tests[pid].copy()
        mv = paths[pid]
        prev = -1
        for i, m in enumerate(mv):
            a = name_i[m]
            # the "undo" is the inverse of the move that ARRIVED here: remaining+1
            if prev >= 0:
                states.append(cur.copy())
                on_move.append(a)
                undo_move.append(int(inv_i[prev]))
                remaining.append(len(mv) - i)
            cur = cur[G[a]]
            prev = a
        if not np.array_equal(cur, np.array(puz.solved_state)):
            raise SystemExit(f"pid {pid} path does not solve -- wrong source file")

    X = torch.tensor(np.array(states), dtype=torch.int64, device=args.device)
    on = torch.tensor(on_move, device=args.device)
    un = torch.tensor(undo_move, device=args.device)
    rem = np.array(remaining)
    print(
        f"{X.shape[0]:,} path pivots from {src.name}, remaining "
        f"{rem.min()}..{rem.max()}"
    )

    model = load_model(args.checkpoint, device=args.device, dtype=torch.float32)
    model.return_value = True
    with torch.no_grad():
        q, v = model(X)
    q = q.float()
    v = v.float().cpu().numpy()
    rows = torch.arange(q.shape[0], device=q.device)
    q_on = q[rows, on]
    q_un = q[rows, un]
    other = torch.ones_like(q, dtype=torch.bool)
    other[rows, on] = False
    best_other = q.masked_fill(~other, float("inf")).min(dim=1).values
    # TIE-AWARE: strict inequality, so a saturated head cannot score its ties as skill.
    pair = (q_on < q_un).cpu().numpy()
    top1 = (q_on < best_other).cpu().numpy()
    q_on_np = q_on.cpu().numpy()

    print(
        f"\n{'remaining':>12} {'n':>6} {'pair':>7} {'top-1':>7} {'Q(on)':>8} {'V':>8}"
    )
    print(f"{'chance':>12} {'':>6} {0.5:>7.3f} {1/30:>7.3f}")
    for lo, hi in BANDS:
        sel = (rem >= lo) & (rem <= hi)
        n = int(sel.sum())
        if n == 0:
            continue
        print(
            f"{f'{lo}-{hi}':>12} {n:>6} {pair[sel].mean():>7.3f} {top1[sel].mean():>7.3f} "
            f"{q_on_np[sel].mean():>8.2f} {v[sel].mean():>8.2f}"
        )
    print(
        "\nRead: pair >> 0.5 and top-1 >> 0.033 in the 61-80 band is what the beam needs;"
        "\nQ(on) tracking `remaining - 1` is calibration, which the beam does NOT need."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
