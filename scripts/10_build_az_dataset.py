"""Build the AZ-cube policy/value dataset from the best-known submission paths.

For every pid in the floor CSV, replay its path and emit (state, action,
dist_to_solved) tuples, augmented by:
  * the 48 cube symmetries (state conjugation + move relabeling), and
  * path inversion (NISS relation: the solution of invert_state(s) is the
    reversed-inverted solution of s),
for 96 variants per path (~2.1M samples from a 21,872-move floor).

Every variant is REPLAYED and asserted to end at solved before its samples are
emitted, so a bad relabeling can never silently corrupt the dataset.

Output (torch .pt):
  states  (N, 72) uint8    state BEFORE the action
  actions (N,)    int64    move index into MOVE_NAMES order of puzzle_info
  dists   (N,)    int16    remaining moves to solved along this path
  meta    dict             floor path, counts, move-name order

Also exports the BFS-d6 anchor tensor (states + exact depths) if --bfs-out
is given.

    .venv/Scripts/python.exe scripts/10_build_az_dataset.py \
        --floor submissions/floor_merged_20260712.csv \
        --out data/az_cube_dataset.pt \
        --bfs-out data/bfs_d6_train.pt
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

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "share" / "cube_symmetries"))

from use_cube_symmetries import apply_symmetry, invert_state, move_relabel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=Path, default=PROJECT / "submissions" / "floor_merged_20260712.csv")
    ap.add_argument("--out", type=Path, default=PROJECT / "data" / "az_cube_dataset.pt")
    ap.add_argument("--bfs-out", type=Path, default=None)
    ap.add_argument("--bfs-table", type=Path, default=PROJECT / "data" / "bfs_table_d6.pkl")
    ap.add_argument("--n-syms", type=int, default=48, help="symmetries to use (<=48)")
    ap.add_argument("--no-inverse", action="store_true", help="skip path-inversion doubling")
    args = ap.parse_args()

    pinfo = json.loads((PROJECT / "data" / "puzzle_info.json").read_text(encoding="utf-8"))
    gens = {nm: np.array(p, dtype=np.int64) for nm, p in pinfo["generators"].items()}
    move_names = list(gens.keys())
    name_to_idx = {nm: i for i, nm in enumerate(move_names)}
    inv_name = {nm: (nm[1:] if nm.startswith("-") else "-" + nm) for nm in move_names}
    n = 72
    solved = np.arange(n, dtype=np.int64)

    sym = np.load(PROJECT / "share" / "cube_symmetries" / "cube_symmetries.npy").astype(np.int64)
    sym_inv = np.load(PROJECT / "share" / "cube_symmetries" / "cube_symmetries_inv.npy").astype(np.int64)
    assert 1 <= args.n_syms <= 48
    gens_int = {nm: [int(x) for x in g] for nm, g in gens.items()}
    # Per-symmetry move relabeling sigma: solving the transformed scramble with
    # move q corresponds to move sigma[q] on the original -- we need sigma^-1
    # to transform an original solution into the transformed frame.
    sigmas_inv = []
    for i in range(args.n_syms):
        sigma = move_relabel(sym[i], sym_inv[i], gens_int)
        sigmas_inv.append({v: k for k, v in sigma.items()})

    states_by_pid: dict[int, np.ndarray] = {}
    with open(PROJECT / "data" / "test.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            states_by_pid[int(row["initial_state_id"])] = np.array(
                [int(x) for x in row["initial_state"].split(",")], dtype=np.int64)

    paths: dict[int, list[str]] = {}
    with open(args.floor, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["path"]:
                paths[int(row["initial_state_id"])] = row["path"].split(".")

    def replay_and_emit(s0: np.ndarray, path: list[str], out_states, out_actions, out_dists):
        """Replay path from s0; assert it solves; emit one sample per move."""
        L = len(path)
        cur = s0
        for i, mv in enumerate(path):
            out_states.append(cur.astype(np.uint8))
            out_actions.append(name_to_idx[mv])
            out_dists.append(L - i)
            cur = cur[gens[mv]]
        assert np.array_equal(cur, solved), "variant path does not solve its state"

    all_states: list[np.ndarray] = []
    all_actions: list[int] = []
    all_dists: list[int] = []
    t0 = time.time()
    n_variants = 0
    for pid in sorted(paths):
        s0 = states_by_pid[pid]
        fwd = paths[pid]
        variants: list[tuple[np.ndarray, list[str]]] = [(s0, fwd)]
        if not args.no_inverse:
            u0 = invert_state(s0)
            inv_path = [inv_name[m] for m in reversed(fwd)]
            variants.append((u0, inv_path))
        base = list(variants)
        for i in range(args.n_syms):
            P, Pinv, si = sym[i], sym_inv[i], sigmas_inv[i]
            if np.array_equal(P, solved):
                continue  # identity already covered by the base variants
            for (bs, bp) in base:
                variants.append((apply_symmetry(bs, P, Pinv), [si[m] for m in bp]))
        for (vs, vp) in variants:
            replay_and_emit(vs, vp, all_states, all_actions, all_dists)
        n_variants += len(variants)
        if pid % 200 == 0:
            print(f"  pid {pid}: {len(all_states):,} samples so far "
                  f"({time.time()-t0:.0f}s)", flush=True)

    states_t = torch.from_numpy(np.stack(all_states))          # (N,72) uint8
    actions_t = torch.tensor(all_actions, dtype=torch.int64)
    dists_t = torch.tensor(all_dists, dtype=torch.int16)
    print(f"\ntotal: {states_t.shape[0]:,} samples from {len(paths)} pids "
          f"x {n_variants // max(len(paths), 1)} variants avg "
          f"({time.time()-t0:.0f}s)")
    print(f"dist range: {int(dists_t.min())}..{int(dists_t.max())}  "
          f"action histogram nonzero: {int((torch.bincount(actions_t, minlength=18) > 0).sum())}/18")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "states": states_t, "actions": actions_t, "dists": dists_t,
        "meta": {"floor": str(args.floor), "n_syms": args.n_syms,
                 "inverse": not args.no_inverse, "move_names": move_names},
    }, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")

    if args.bfs_out is not None:
        from cayley.bfs_table import BfsTable
        table = BfsTable.load(args.bfs_table)
        print(f"BFS table: {len(table.table):,} states, max_depth {table.max_depth}")
        n_bfs = len(table.table)
        bfs_states_np = np.empty((n_bfs, n), dtype=np.uint8)
        bfs_depths_np = np.empty((n_bfs,), dtype=np.int16)
        for i, (state, word) in enumerate(table.table.items()):
            bfs_states_np[i] = state
            bfs_depths_np[i] = len(word)
        bfs_states = torch.from_numpy(bfs_states_np)
        bfs_depths = torch.from_numpy(bfs_depths_np)
        torch.save({"states": bfs_states, "depths": bfs_depths}, args.bfs_out)
        print(f"wrote {args.bfs_out} ({args.bfs_out.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
