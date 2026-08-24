from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict

import torch

from .ball import ExactBall
from .gflownet import GFNBackwardPolicy, load_gflownet
from .model import load_transformer
from .official import OfficialPuzzle, parse_official_state
from .pdb import EdgePatternDatabase
from .policy_search import TransformerPolicy, phs_search, phs_then_exact, policy_ida_star
from .puzzle import apply_path


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--state", help="48 comma-separated official labels")
    source.add_argument("--initial-state-id", type=int)
    parser.add_argument("--test", default="jewel/data/test.csv")
    parser.add_argument("--puzzle-info", default="jewel/data/puzzle_info.json")
    parser.add_argument("--ball", default="jewel/artifacts/ball_d8_official")
    parser.add_argument("--pdb", action="append")
    parser.add_argument("--policy", choices=["gflownet", "transformer"], default="gflownet")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--algorithm", choices=["ida", "phsh", "phsstar", "phsh-exact", "phsstar-exact"], default="phsh-exact")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--node-limit", type=int, default=500_000)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--max-depth", type=int, default=30)
    parser.add_argument("--policy-batch-size", type=int, default=256)
    parser.add_argument("--ida-ordering", choices=["policy", "heuristic", "hybrid"], default="heuristic")
    args = parser.parse_args()

    official = OfficialPuzzle.load(args.puzzle_info)
    if args.state is not None:
        official_state = parse_official_state(args.state)
    else:
        with open(args.test, encoding="utf-8", newline="") as handle:
            rows = {int(row["initial_state_id"]): row for row in csv.DictReader(handle)}
        if args.initial_state_id not in rows:
            raise ValueError(f"initial_state_id not found: {args.initial_state_id}")
        official_state = parse_official_state(rows[args.initial_state_id]["initial_state"])
    state = official.to_structured(official_state)
    ball = ExactBall.load(args.ball)
    pdb_paths = args.pdb or [
        "jewel/artifacts/pdb_edges_0_4_official.npy",
        "jewel/artifacts/pdb_edges_5_9_official.npy",
    ]
    pdbs = [EdgePatternDatabase.load(path) for path in pdb_paths]
    device = torch.device(args.device)
    if args.policy == "gflownet":
        model, _, _ = load_gflownet(args.checkpoint, device)
        policy = GFNBackwardPolicy(model, official, device=device)
    else:
        model = load_transformer(args.checkpoint, device)
        policy = TransformerPolicy(model, device=device)

    if args.algorithm == "ida":
        result = policy_ida_star(
            state, ball, pdbs, policy=policy, max_depth=args.max_depth,
            node_limit=args.node_limit, time_limit=args.time_limit,
        )
        payload = asdict(result)
    elif args.algorithm in {"phsh", "phsstar"}:
        result = phs_search(
            state, ball, pdbs, policy=policy, variant=args.algorithm,
            max_depth=args.max_depth, node_limit=args.node_limit,
            time_limit=args.time_limit, policy_batch_size=args.policy_batch_size,
        )
        payload = asdict(result)
    else:
        variant = "phsh" if args.algorithm == "phsh-exact" else "phsstar"
        combined = phs_then_exact(
            state, ball, pdbs, policy=policy, phs_variant=variant,
            ida_ordering=args.ida_ordering,
            phs_node_limit=args.node_limit, phs_time_limit=args.time_limit,
            phs_policy_batch_size=args.policy_batch_size,
            ida_node_limit=args.node_limit, ida_time_limit=args.time_limit,
            max_depth=args.max_depth,
        )
        result = combined.exact
        payload = {"phs": asdict(combined.phs), "exact": asdict(combined.exact)}

    path = result.path
    valid_internal = path is not None and apply_path(state, path).rank() == 0
    valid_official = bool(
        path is not None
        and (official.apply_path(official_state, path) == official.central_state).all()
    )
    output = {
        "status": "solved" if valid_internal and valid_official else "not_solved",
        "algorithm": args.algorithm,
        "policy": args.policy,
        "optimality_certified": bool(result.optimal),
        "path_length": len(path) if path is not None else None,
        "path": official.format_path(path) if path is not None else None,
        "valid_internal": valid_internal,
        "valid_official": valid_official,
        "search": payload,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
