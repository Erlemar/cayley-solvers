from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .ball import ExactBall
from .gflownet import GFNBackwardPolicy, load_gflownet
from .model import load_transformer
from .official import OfficialPuzzle
from .pdb import EdgePatternDatabase
from .policy_search import TransformerPolicy, phs_search, policy_ida_star, uniform_policy
from .puzzle import INVERSE_ACTION, SOLVED, apply_path, inverse_path


def random_nonbacktracking_path(rng: np.random.Generator, length: int) -> list[int]:
    path: list[int] = []
    for _ in range(length):
        actions = np.arange(12)
        if path:
            actions = actions[actions != int(INVERSE_ACTION[path[-1]])]
        path.append(int(rng.choice(actions)))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gflownet")
    parser.add_argument("--transformer", default="jewel/models/transformer_v4_public/best.pt")
    parser.add_argument("--puzzle-info", default="jewel/data/puzzle_info.json")
    parser.add_argument("--ball", default="jewel/artifacts/ball_d8_official")
    parser.add_argument("--pdb", action="append")
    parser.add_argument("--depths", default="10,12,14")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--algorithms", default="ida_heuristic,ida_policy,phsh,phsstar")
    parser.add_argument("--policy", choices=["gflownet", "transformer", "uniform"], default="gflownet")
    parser.add_argument("--node-limit", type=int, default=250_000)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--policy-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="jewel/results/policy_search_benchmark.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    official = OfficialPuzzle.load(args.puzzle_info)
    ball = ExactBall.load(args.ball)
    pdb_paths = args.pdb or [
        "jewel/artifacts/pdb_edges_0_4_official.npy",
        "jewel/artifacts/pdb_edges_5_9_official.npy",
    ]
    pdbs = [EdgePatternDatabase.load(path) for path in pdb_paths]
    if args.policy == "gflownet":
        if not args.gflownet:
            raise ValueError("--gflownet is required for --policy gflownet")
        model, _, _ = load_gflownet(args.gflownet, device)
        policy = GFNBackwardPolicy(model, official, device=device)
    elif args.policy == "transformer":
        model = load_transformer(args.transformer, device)
        policy = TransformerPolicy(model, device=device)
    else:
        policy = uniform_policy

    algorithms = set(args.algorithms.split(","))
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    for walk_depth in [int(value) for value in args.depths.split(",")]:
        for trial in range(args.trials):
            scramble = random_nonbacktracking_path(rng, walk_depth)
            state = apply_path(SOLVED, scramble)
            incumbent = inverse_path(scramble)
            cases: list[tuple[str, object]] = []
            if "ida_heuristic" in algorithms:
                cases.append(
                    (
                        "ida_heuristic",
                        policy_ida_star(
                            state,
                            ball,
                            pdbs,
                            policy=uniform_policy,
                            ordering="heuristic",
                            incumbent_path=incumbent,
                            max_depth=walk_depth,
                            node_limit=args.node_limit,
                            time_limit=args.time_limit,
                        ),
                    )
                )
            if "ida_policy" in algorithms:
                cases.append(
                    (
                        "ida_policy",
                        policy_ida_star(
                            state,
                            ball,
                            pdbs,
                            policy=policy,
                            ordering="policy",
                            incumbent_path=incumbent,
                            max_depth=walk_depth,
                            node_limit=args.node_limit,
                            time_limit=args.time_limit,
                        ),
                    )
                )
            for variant in ("phsh", "phsstar"):
                if variant in algorithms:
                    cases.append(
                        (
                            variant,
                            phs_search(
                                state,
                                ball,
                                pdbs,
                                policy=policy,
                                variant=variant,
                                max_depth=max(40, walk_depth),
                                node_limit=args.node_limit,
                                time_limit=args.time_limit,
                                policy_batch_size=args.policy_batch_size,
                            ),
                        )
                    )
            for algorithm, result in cases:
                result_dict = asdict(result)
                valid = result.path is not None and apply_path(state, result.path).rank() == 0
                row = {
                    "walk_depth": walk_depth,
                    "trial": trial,
                    "algorithm": algorithm,
                    "valid": valid,
                    "path_length": len(result.path) if result.path is not None else None,
                    **{
                        key: value
                        for key, value in result_dict.items()
                        if key not in {"path", "algorithm"}
                    },
                }
                rows.append(row)
                print(json.dumps(row), flush=True)

    summary: dict[str, dict] = {}
    for algorithm in sorted({row["algorithm"] for row in rows}):
        selected = [row for row in rows if row["algorithm"] == algorithm]
        solved = [row for row in selected if row["valid"]]
        summary[algorithm] = {
            "cases": len(selected),
            "solved": len(solved),
            "optimal_certificates": sum(row["optimal"] for row in selected),
            "mean_length_solved": float(np.mean([row["path_length"] for row in solved])) if solved else None,
            "mean_expanded": float(np.mean([row["expanded"] for row in selected])),
            "median_expanded": float(np.median([row["expanded"] for row in selected])),
            "total_seconds": float(sum(row["seconds"] for row in selected)),
        }
    report = {"args": vars(args), "summary": summary, "rows": rows}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
