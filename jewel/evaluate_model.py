from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .ball import ExactBall
from .benchmark_exact import random_nonbacktracking_path
from .interactive import interactive_solve
from .model import load_transformer
from .pdb import EdgePatternDatabase
from .puzzle import SOLVED, apply_path, inverse_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ball", default="jewel/artifacts/ball_d8_official")
    parser.add_argument("--pdb", action="append")
    parser.add_argument("--depths", default="8,10,12,14,16,18")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--widths", default="1,4,16,64,256")
    parser.add_argument("--max-learned-steps", type=int, default=18)
    parser.add_argument("--first-success", action="store_true")
    parser.add_argument("--heuristic-weight", type=float, default=0.35)
    parser.add_argument("--regret-weight", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--out", default="jewel/results/transformer_v1_random.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_transformer(args.checkpoint, device)
    ball = ExactBall.load(args.ball)
    pdb_paths = args.pdb or [
        "jewel/artifacts/pdb_edges_0_4_official.npy",
        "jewel/artifacts/pdb_edges_5_9_official.npy",
    ]
    pdbs = [EdgePatternDatabase.load(path) for path in pdb_paths]
    rng = np.random.default_rng(args.seed)
    widths = [int(x) for x in args.widths.split(",")]
    rows = []
    for depth in [int(x) for x in args.depths.split(",")]:
        for trial in range(args.trials):
            scramble = random_nonbacktracking_path(rng, depth)
            state = apply_path(SOLVED, scramble)
            result = interactive_solve(
                state,
                model,
                ball,
                pdbs,
                widths=widths,
                max_learned_steps=args.max_learned_steps,
                device=device,
                try_all_widths=not args.first_success,
                heuristic_weight=args.heuristic_weight,
                regret_weight=args.regret_weight,
            )
            valid = result.path is not None and apply_path(state, result.path).rank() == 0
            fallback = inverse_path(scramble)
            chosen_path = result.path if valid and len(result.path) <= len(fallback) else fallback
            row = {
                "walk_depth": depth,
                "trial": trial,
                "solved": valid,
                "solution_length": len(result.path) if valid else None,
                "fallback_length": len(fallback),
                "merged_length": len(chosen_path),
                "delta_vs_fallback": len(chosen_path) - len(fallback),
                "width": result.width,
                "expanded": result.expanded_states,
                "generated": result.generated_states,
                "seconds": result.seconds,
                "reason": result.reason,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    per_depth = {}
    for depth in sorted(set(row["walk_depth"] for row in rows)):
        selected = [row for row in rows if row["walk_depth"] == depth]
        solved = [row for row in selected if row["solved"]]
        per_depth[str(depth)] = {
            "solved": len(solved),
            "total": len(selected),
            "solve_rate": len(solved) / len(selected),
            "mean_length": float(np.mean([row["solution_length"] for row in solved])) if solved else None,
            "mean_merged_length": float(np.mean([row["merged_length"] for row in selected])),
            "wins_vs_fallback": int(sum(row["delta_vs_fallback"] < 0 for row in selected)),
            "mean_seconds": float(np.mean([row["seconds"] for row in selected])),
            "mean_expanded": float(np.mean([row["expanded"] for row in selected])),
        }
    summary = {"checkpoint": args.checkpoint, "rows": rows, "per_depth": per_depth}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(per_depth, indent=2))


if __name__ == "__main__":
    main()
