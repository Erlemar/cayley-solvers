"""Merge TPU beam results (JSON) into a submission CSV, with an exact endgame splice.

The TPU kernel solves all the way to the exact solved state, so its tail is only
as good as the V model. Here we do the splice on the host instead: walk each
returned path and, at the FIRST state that is inside the BFS table (d <= 6),
replace everything after it with the table's optimal descent. That can only
shorten the path, costs no TPU time, and makes the last moves provably optimal.

Then take the per-pid min across every record and every extra CSV supplied.
Rule 26: a "win" is only a win against the n-way min over ALL available sources,
not against one base -- so always pass the floor and any prior runs via --extra.

    python3 tetraminx/scripts/41_merge_tpu_results.py \
        --results tetraminx_tpu_results.json \
        --extra tetraminx/submissions/floor_public_29622.csv \
        --out tetraminx/submissions/merged.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]


class EndgameTable:
    def __init__(self, path: Path):
        d = np.load(path, allow_pickle=True)
        self.hashes = d["hashes"]
        self.depths = d["depths"]
        self.ztab = d["ztab"]
        self.max_depth = int(d["max_depth"])

    def lookup(self, states: np.ndarray) -> np.ndarray:
        states = np.atleast_2d(states)
        h = np.zeros(states.shape[0], dtype=np.int64)
        for i in range(states.shape[1]):
            h ^= self.ztab[i][states[:, i]]
        pos = np.clip(np.searchsorted(self.hashes, h), 0, len(self.hashes) - 1)
        out = np.full(states.shape[0], -1, dtype=np.int64)
        hit = self.hashes[pos] == h
        out[hit] = self.depths[pos[hit]]
        return out

    def descend(self, state: np.ndarray, gens: np.ndarray) -> list[int]:
        d = int(self.lookup(state[None, :])[0])
        assert d >= 0
        out: list[int] = []
        cur = state
        while d > 0:
            children = cur[gens]
            cd = self.lookup(children)
            nxt = int(np.nonzero(cd == d - 1)[0][0])
            out.append(nxt)
            cur = children[nxt]
            d -= 1
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*", type=Path, default=[],
                    help="TPU results JSON files (optional -- omit to merge CSVs only)")
    ap.add_argument("--extra", nargs="*", type=Path, default=[],
                    help="existing CSVs to include in the per-pid min")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "tetraminx" / "data")
    ap.add_argument("--endgame", type=Path, default=None)
    args = ap.parse_args()

    info = json.loads((args.data_dir / "puzzle_info.json").read_text(encoding="utf-8"))
    move_names = list(info["generators"].keys())
    name_to_idx = {nm: i for i, nm in enumerate(move_names)}
    gens = np.array([info["generators"][nm] for nm in move_names], dtype=np.int64)
    solved = np.array(info["central_state"], dtype=np.int64)

    with open(args.data_dir / "test.csv", encoding="utf-8", newline="") as f:
        states = {int(r["initial_state_id"]):
                  np.array([int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
                  for r in csv.DictReader(f)}

    eg_path = args.endgame or (args.data_dir / "bfs_endgame.npz")
    endgame = EndgameTable(eg_path) if Path(eg_path).exists() else None
    print("endgame table: "
          + (f"{len(endgame.hashes):,} states" if endgame else "DISABLED"))

    def replay(s0: np.ndarray, path: list[int]) -> np.ndarray:
        cur = s0
        for m in path:
            cur = cur[gens[m]]
        return cur

    best: dict[int, list[int]] = {}
    source: dict[int, str] = {}

    def offer(pid: int, path: list[int], src: str) -> None:
        if not np.array_equal(replay(states[pid], path), solved):
            print(f"  pid {pid} from {src}: path does NOT solve -- rejected")
            return
        if pid not in best or len(path) < len(best[pid]):
            best[pid] = path
            source[pid] = src

    n_spliced = 0
    moves_saved = 0
    for rf in args.results:
        records = json.loads(rf.read_text(encoding="utf-8"))
        for rec in records:
            if not rec.get("found") or not rec.get("path"):
                continue
            pid = int(rec["pid"])
            path = [name_to_idx[m] for m in rec["path"].split(".")]
            if endgame is not None:
                cur = states[pid]
                for i, m in enumerate(path):
                    d = int(endgame.lookup(cur[None, :])[0])
                    if d >= 0 and d < len(path) - i:
                        path = path[:i] + endgame.descend(cur, gens)
                        n_spliced += 1
                        moves_saved += len(rec["path"].split(".")) - len(path)
                        break
                    cur = cur[gens[m]]
            offer(pid, path, f"tpu:{rf.name}:sym{rec.get('sym')}:inv{int(rec.get('inverted', False))}")

    for cf in args.extra:
        with open(cf, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["path"]:
                    offer(int(r["initial_state_id"]),
                          [name_to_idx[m] for m in r["path"].split(".")], f"csv:{cf.name}")

    if endgame is not None:
        print(f"endgame splice: {n_spliced} paths shortened, {moves_saved} moves saved")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid in sorted(best):
            w.writerow([pid, ".".join(move_names[m] for m in best[pid])])

    total = sum(len(v) for v in best.values())
    from collections import Counter
    by_src = Counter(s.split(":")[0] for s in source.values())
    print(f"\nwrote {args.out}: {len(best)} pids, {total:,} moves")
    print(f"winner by source: {dict(by_src)}")
    if len(best) < len(states):
        print(f"WARNING: {len(states) - len(best)} pids have no path -- not submittable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
