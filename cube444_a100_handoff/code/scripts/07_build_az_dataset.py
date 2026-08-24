"""Build the AZ policy dataset (state, action, remaining_distance) from a submission CSV.

    python3 cube444/scripts/07_build_az_dataset.py \
        --submission cube444/community/submission_54754_merge33.csv \
        --out cube444/data/az_dataset_54754.pt

Each pid contributes path_len tuples: at step i along the realized solution the
state is the initial scramble with the first i moves applied, the action is the
move taken at step i, and the value is path_len - i.

The community merged CSV (54,754 moves) yields ~54,754 pairs -- almost exactly
the size of megaminx's az_dataset_76304.pt that produced the AZ v4 breakthrough.
So there is no chicken-and-egg: Stage 4 can run before we have any beam output
of our own. Rebuild from our own CSV once we beat the community per-pid.

--sym-augment K expands the set K-fold through the rotation group: state ->
sym(state, R_k) and action -> move_relabel[k, action]. This is deliberately OFF
by default. Evidence: on megaminx, symmetry augmentation on the V HEAD was
REJECTED (m31: it dilutes the distance signal at 6M params) but KEPT on the
Q-shortlister (m23_v2). This dataset only feeds the policy head (the value head
is trained by Bellman on random walks, and `values` is used only when
--gamma-path-value > 0), so augmenting here is the supported case -- but run it
as an explicit A/B against the un-augmented baseline, not as a silent default.
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

from cube444.puzzle import Cube444


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--test", type=Path, default=PROJECT / "data" / "test.csv")
    ap.add_argument("--sym-augment", type=int, default=1,
                    help="expand K-fold through the rotation group (1 = off)")
    args = ap.parse_args()

    puz = Cube444.load(PROJECT / "data" / "puzzle_info.json")
    central = np.array(puz.solved_state, dtype=np.int64)
    G = {n: np.array(v, dtype=np.int64) for n, v in puz.generators.items()}
    move_to_idx = {n: i for i, n in enumerate(puz.move_names)}

    with open(args.test, encoding="utf-8") as f:
        tests = {int(r["initial_state_id"]):
                 np.array([int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
                 for r in csv.DictReader(f)}
    with open(args.submission, encoding="utf-8") as f:
        sub = {int(r["initial_state_id"]): r["path"] for r in csv.DictReader(f)}

    states, actions, values = [], [], []
    n_bad = 0
    for pid, path_str in sub.items():
        if pid not in tests or not path_str.strip():
            n_bad += 1
            continue
        moves = path_str.split(".")
        if any(m not in G for m in moves):
            n_bad += 1
            continue
        cur = tests[pid].copy()
        traj_s, traj_a, traj_v = [], [], []
        n = len(moves)
        for i, m in enumerate(moves):
            traj_s.append(cur.copy())
            traj_a.append(move_to_idx[m])
            traj_v.append(n - i)
            cur = cur[G[m]]
        if not np.array_equal(cur, central):
            n_bad += 1
            continue
        states.extend(traj_s)
        actions.extend(traj_a)
        values.extend(traj_v)

    S = np.array(states, dtype=np.int8)
    A = np.array(actions, dtype=np.int64)
    V = np.array(values, dtype=np.float32)
    print(f"base: {len(S):,} (state, action) pairs from {len(sub) - n_bad} verified paths "
          f"({n_bad} skipped)")

    if args.sym_augment > 1:
        rot = np.load(PROJECT / "data" / "rotations_24.npy")
        cmap = np.load(PROJECT / "data" / "color_maps_24.npy")
        relabel = np.load(PROJECT / "data" / "move_relabel_24.npy")
        K = min(args.sym_augment, rot.shape[0])
        aS = [S]
        aA = [A]
        aV = [V]
        for k in range(1, K):
            aS.append(cmap[k][S[:, rot[k]]].astype(np.int8))
            aA.append(relabel[k][A])
            aV.append(V)
        S, A, V = np.concatenate(aS), np.concatenate(aA), np.concatenate(aV)
        print(f"sym-augment K={K}: -> {len(S):,} pairs")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"states": torch.from_numpy(S),
                "actions": torch.from_numpy(A),
                "values": torch.from_numpy(V)}, args.out)
    print(f"wrote {args.out}  ({S.nbytes/1e6:.0f} MB)")
    print(f"  value range {V.min():.0f}..{V.max():.0f}  mean {V.mean():.2f}")
    print(f"  action histogram (min/max over 24): {np.bincount(A, minlength=24).min()} / "
          f"{np.bincount(A, minlength=24).max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
