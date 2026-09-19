"""Path-state bank for IHES Q-Bellman (stage 3): every state on the floor's solution paths.

The 444 s3 recipe samples half of each Bellman batch from solution-path STATES (their
labels are discarded -- every target comes from the bootstrap and the anchors). Path LABELS
are a measured loss on 444 and tetraminx; path states only put the model on realistic
descent-corridor states.

Stores the raw states s_0 .. s_{L-1} of each replay-verified floor path (the solved end
state is dropped), with the pid and the moves-to-go (an upper bound on the distance; exact
on proven-optimal pids). Gate pids are held out. Symmetry / inverse images are drawn at
sampling time by the trainer.

    python scripts/83_build_ihes_path_bank.py --out data/ihes_path_bank.pt
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the IHES path-state bank.")
    ap.add_argument("--floor", type=Path,
                    default=PROJECT / "submissions" / "ihes_20260912_1410_verified.csv")
    ap.add_argument("--gate", type=Path, default=PROJECT / "data" / "ihes_gate54.json")
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    ap.add_argument("--out", type=Path, default=PROJECT / "data" / "ihes_path_bank.pt")
    args = ap.parse_args()

    info = json.loads((args.data_dir / "puzzle_info.json").read_text(encoding="utf-8"))
    gens = {n: np.array(p, dtype=np.int64) for n, p in info["generators"].items()}
    solved = np.array(info["central_state"], dtype=np.int64)
    held = set(json.loads(args.gate.read_text(encoding="utf-8"))["pids"])

    with open(args.data_dir / "test.csv", encoding="utf-8", newline="") as f:
        start = {int(r["initial_state_id"]): np.array([int(x) for x in r["initial_state"].split(",")],
                                                      dtype=np.int64)
                 for r in csv.DictReader(f)}
    with open(args.floor, encoding="utf-8", newline="") as f:
        paths = {int(r["initial_state_id"]): (r["path"].split(".") if r["path"] else [])
                 for r in csv.DictReader(f)}

    states, togo, pids = [], [], []
    n_moves = 0
    for pid in sorted(paths):
        mv = paths[pid]
        cur = start[pid]
        seq = []
        for m in mv:
            seq.append(cur)
            cur = cur[gens[m]]
        if not np.array_equal(cur, solved):
            raise SystemExit(f"pid {pid}: floor path does not solve")
        n_moves += len(mv)
        if pid in held:
            continue
        L = len(mv)
        for i, s in enumerate(seq):
            states.append(s)
            togo.append(L - i)
            pids.append(pid)
    st = torch.from_numpy(np.stack(states).astype(np.uint8))
    tg = torch.tensor(togo, dtype=torch.uint8)
    pd = torch.tensor(pids, dtype=torch.int16)
    hist = torch.bincount(tg.long()).tolist()
    torch.save({"states": st, "togo": tg, "pid": pd,
                "meta": {"floor": args.floor.name, "floor_moves": n_moves,
                         "held_out": sorted(held), "note": "states only; labels are NOT used"}},
               args.out)
    print(f"floor replay-verified: {len(paths)} pids, {n_moves:,} moves; held out {len(held)} gate pids")
    print(f"wrote {args.out}: {st.size(0):,} states from {len(set(pids))} pids; togo histogram {hist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
