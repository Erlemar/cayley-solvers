from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .ball import ExactBall
from .exact_search import solve_ida_star
from .pdb import EdgePatternDatabase, combined_heuristic
from .puzzle import INVERSE_ACTION, SOLVED, apply_path


def random_nonbacktracking_path(rng: np.random.Generator, length: int) -> list[int]:
    out: list[int] = []
    for _ in range(length):
        choices = np.arange(12)
        if out:
            choices = choices[choices != int(INVERSE_ACTION[out[-1]])]
        out.append(int(rng.choice(choices)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball", default="jewel/artifacts/ball_d8_official")
    parser.add_argument("--pdb", action="append", default=[
        "jewel/artifacts/pdb_edges_0_4_official.npy",
        "jewel/artifacts/pdb_edges_5_9_official.npy",
    ])
    parser.add_argument("--depths", default="8,10,12,14,16")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--node-limit", type=int, default=2_000_000)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out")
    args = parser.parse_args()

    ball = ExactBall.load(args.ball)
    pdbs = [EdgePatternDatabase.load(path) for path in args.pdb]
    rng = np.random.default_rng(args.seed)
    rows = []
    for walk_depth in [int(x) for x in args.depths.split(",")]:
        for trial in range(args.trials):
            scramble = random_nonbacktracking_path(rng, walk_depth)
            state = apply_path(SOLVED, scramble)
            initial_h = combined_heuristic(state, pdbs)
            result = solve_ida_star(
                state,
                ball,
                pdbs,
                max_depth=walk_depth,
                node_limit=args.node_limit,
                time_limit=args.time_limit,
            )
            row = {
                "walk_depth": walk_depth,
                "trial": trial,
                "initial_h": initial_h,
                "solved": result.path is not None,
                "optimal_depth": len(result.path) if result.path is not None else None,
                "nodes": result.nodes,
                "seconds": result.seconds,
                "reason": result.reason,
                "thresholds": result.thresholds,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = {
        "rows": rows,
        "solved": sum(row["solved"] for row in rows),
        "total": len(rows),
        "mean_nodes": float(np.mean([row["nodes"] for row in rows])),
        "mean_seconds": float(np.mean([row["seconds"] for row in rows])),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
