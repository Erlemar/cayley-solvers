"""Build the Tetraminx AZ policy/value dataset from a solved-path CSV.

For every pid in the floor CSV we replay its path and emit one
(state_before_move, action, remaining_moves) sample per move, augmented by

  * the 24 spatial symmetries (state conjugation + move relabeling), and
  * inverse antisymmetry (solve s^-1 via the reversed-inverted path),

for 48 variants per path.  A 29,622-move floor yields ~1.42M samples.

EVERY variant is replayed and asserted to reach the solved state before its
samples are emitted, so a bad relabeling can never silently corrupt the dataset.

Output (torch .pt):
  states  (N, 88) uint8   state BEFORE the action
  actions (N,)    int64   move index into puzzle_info generator order
  dists   (N,)    int16   remaining moves to solved along this path
  meta    dict

    python3 tetraminx/scripts/20_build_az_dataset.py \
        --floor tetraminx/submissions/floor_public_29622.csv \
        --out tetraminx/data/az_dataset.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]


def apply_symmetry(state: np.ndarray, P: np.ndarray, P_inv: np.ndarray) -> np.ndarray:
    """conj(s)[i] = P_inv[s[P[i]]] -- the relabeling verified in 01_build_symmetries."""
    return P_inv[state[P]]


def invert_state(state: np.ndarray) -> np.ndarray:
    inv = np.empty_like(state)
    inv[state] = np.arange(state.shape[0], dtype=state.dtype)
    return inv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=Path,
                    default=PROJECT / "tetraminx" / "submissions" / "floor_public_29622.csv")
    ap.add_argument("--out", type=Path, default=PROJECT / "tetraminx" / "data" / "az_dataset.pt")
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "tetraminx" / "data")
    ap.add_argument("--trace-out", type=Path, default=None,
                    help="also write the same samples in cayley.bellman's "
                         "solver-trace format (keys: states, distances) so the "
                         "Bellman trainer can anchor the DEEP scale, which BFS "
                         "anchors (d<=6) cannot reach")
    ap.add_argument("--n-syms", type=int, default=24, help="symmetries to use (<=24)")
    ap.add_argument("--no-inverse", action="store_true", help="skip inverse-antisymmetry doubling")
    args = ap.parse_args()

    info = json.loads((args.data_dir / "puzzle_info.json").read_text(encoding="utf-8"))
    gens = {nm: np.array(p, dtype=np.int64) for nm, p in info["generators"].items()}
    move_names = list(gens.keys())
    name_to_idx = {nm: i for i, nm in enumerate(move_names)}
    inv_name = {nm: (nm[1:] if nm.startswith("-") else "-" + nm) for nm in move_names}
    solved = np.array(info["central_state"], dtype=np.int64)
    n = len(solved)

    sym = np.load(args.data_dir / "tetra_symmetries.npy").astype(np.int64)
    sym_inv = np.load(args.data_dir / "tetra_symmetries_inv.npy").astype(np.int64)
    relabel = np.load(args.data_dir / "tetra_move_relabel.npy").astype(np.int64)
    assert 1 <= args.n_syms <= sym.shape[0]

    # relabel[k][m] = sigma: a path solving conj(s) maps to sigma(path) solving s.
    # To push an ORIGINAL solution INTO the conjugated frame we need sigma^-1.
    sigmas_inv = [np.argsort(relabel[k]) for k in range(sym.shape[0])]

    with open(args.data_dir / "test.csv", encoding="utf-8", newline="") as f:
        states_by_pid = {int(r["initial_state_id"]):
                         np.array([int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
                         for r in csv.DictReader(f)}

    with open(args.floor, encoding="utf-8", newline="") as f:
        paths = {int(r["initial_state_id"]): r["path"].split(".")
                 for r in csv.DictReader(f) if r["path"]}

    all_states: list[np.ndarray] = []
    all_actions: list[int] = []
    all_dists: list[int] = []

    def replay_and_emit(s0: np.ndarray, path: list[int]) -> None:
        """path is a list of MOVE INDICES. Replay, assert solved, emit per move."""
        L = len(path)
        cur = s0
        for i, mi in enumerate(path):
            all_states.append(cur.astype(np.uint8))
            all_actions.append(int(mi))
            all_dists.append(L - i)
            cur = cur[gens[move_names[mi]]]
        assert np.array_equal(cur, solved), "variant path does not solve its state"

    t0 = time.time()
    n_variants = 0
    for pid in sorted(paths):
        s0 = states_by_pid[pid]
        fwd = [name_to_idx[m] for m in paths[pid]]
        base: list[tuple[np.ndarray, list[int]]] = [(s0, fwd)]
        if not args.no_inverse:
            inv_path = [name_to_idx[inv_name[move_names[m]]] for m in reversed(fwd)]
            base.append((invert_state(s0), inv_path))
        variants = list(base)
        for k in range(args.n_syms):
            if np.array_equal(sym[k], np.arange(n)):
                continue  # identity already covered by the base variants
            si = sigmas_inv[k]
            for (bs, bp) in base:
                variants.append((apply_symmetry(bs, sym[k], sym_inv[k]),
                                 [int(si[m]) for m in bp]))
        for (vs, vp) in variants:
            replay_and_emit(vs, vp)
        n_variants += len(variants)
        if pid % 250 == 0:
            print(f"  pid {pid}: {len(all_states):,} samples ({time.time() - t0:.0f}s)",
                  flush=True)

    states_t = torch.from_numpy(np.stack(all_states))
    actions_t = torch.tensor(all_actions, dtype=torch.int64)
    dists_t = torch.tensor(all_dists, dtype=torch.int16)
    per_pid = n_variants // max(len(paths), 1)
    n_used = int((torch.bincount(actions_t, minlength=len(move_names)) > 0).sum())
    print(f"\ntotal: {states_t.shape[0]:,} samples from {len(paths)} pids "
          f"x {per_pid} variants ({time.time() - t0:.0f}s)")
    print(f"dist range: {int(dists_t.min())}..{int(dists_t.max())}  "
          f"actions used: {n_used}/{len(move_names)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"states": states_t, "actions": actions_t, "dists": dists_t,
                "meta": {"floor": str(args.floor), "n_syms": args.n_syms,
                         "inverse": not args.no_inverse, "move_names": move_names}},
               args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")

    if args.trace_out is not None:
        # NOTE the labels are path-remaining lengths on a NEAR-optimal floor, so
        # they are upper bounds on true distance (a move or two high), not exact.
        # That is still far better than an unanchored bootstrap, which drifts
        # DOWN without bound at depth. Use as a mixin (<=25%), never as the
        # primary signal -- as a replacement it is an OOD catastrophe (m37).
        assert int(dists_t.max()) <= 127, "distances must fit int8"
        torch.save({"states": states_t, "distances": dists_t.to(torch.int8)},
                   args.trace_out)
        print(f"wrote {args.trace_out} (solver-trace format, "
              f"d range {int(dists_t.min())}..{int(dists_t.max())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
