"""Give a Jewel state to the GFlowNet beam/best-first solver."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .ball import ExactBall
from .gflownet import GFNBackwardPolicy, load_gflownet
from .gfn_beam import (
    gfn_adaptive_beam_search,
    gfn_batched_best_first_search,
    gfn_hybrid_search,
)
from .official import OfficialPuzzle, parse_official_state
from .pdb import EdgePatternDatabase
from .puzzle import apply_path


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--state", help="48 comma-separated official labels")
    source.add_argument("--initial-state-id", type=int)
    parser.add_argument("--checkpoint", default="jewel/models/gflownet_v1/best.pt")
    parser.add_argument("--mode", choices=["beam", "best-first", "hybrid"], default="hybrid")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test", default="jewel/data/test.csv")
    parser.add_argument("--puzzle-info", default="jewel/data/puzzle_info.json")
    parser.add_argument("--ball", default="jewel/artifacts/ball_d8_official")
    parser.add_argument("--pdb", action="append")
    parser.add_argument("--widths", default="1,4,16,64")
    parser.add_argument("--max-learned-steps", type=int, default=18)
    parser.add_argument("--beam-branch-actions", type=int, default=4)
    parser.add_argument("--heuristic-weight", type=float, default=0.2)
    parser.add_argument("--best-first-batch-size", type=int, default=256)
    parser.add_argument("--best-first-node-limit", type=int, default=100_000)
    parser.add_argument("--best-first-time-limit", type=float, default=30.0)
    parser.add_argument("--best-first-max-depth", type=int, default=32)
    parser.add_argument("--best-first-branch-actions", type=int, default=4)
    parser.add_argument("--move-only", action="store_true")
    args = parser.parse_args()

    if args.state is not None:
        official_state = parse_official_state(args.state)
    else:
        with open(args.test, encoding="utf-8", newline="") as handle:
            rows = {int(row["initial_state_id"]): row for row in csv.DictReader(handle)}
        if args.initial_state_id not in rows:
            raise ValueError(f"initial_state_id not found: {args.initial_state_id}")
        official_state = parse_official_state(rows[args.initial_state_id]["initial_state"])

    device = torch.device(args.device)
    official = OfficialPuzzle.load(args.puzzle_info)
    state = official.to_structured(official_state)
    model, _, checkpoint = load_gflownet(args.checkpoint, device)
    policy = GFNBackwardPolicy(model, official, device=device)
    ball = ExactBall.load(args.ball)
    pdb_paths = args.pdb or [
        "jewel/artifacts/pdb_edges_0_4_official.npy",
        "jewel/artifacts/pdb_edges_5_9_official.npy",
    ]
    pdbs = [EdgePatternDatabase.load(path) for path in pdb_paths]
    widths = tuple(int(value) for value in args.widths.split(","))

    if args.mode == "beam":
        result = gfn_adaptive_beam_search(
            state,
            policy,
            ball,
            pdbs,
            widths=widths,
            branch_actions=args.beam_branch_actions,
            max_learned_steps=args.max_learned_steps,
            heuristic_weight=args.heuristic_weight,
        )
    elif args.mode == "best-first":
        result = gfn_batched_best_first_search(
            state,
            policy,
            ball,
            pdbs,
            batch_size=args.best_first_batch_size,
            node_limit=args.best_first_node_limit,
            time_limit=args.best_first_time_limit,
            max_depth=args.best_first_max_depth,
            branch_actions=args.best_first_branch_actions,
            heuristic_weight=args.heuristic_weight,
        )
    else:
        result = gfn_hybrid_search(
            state,
            policy,
            ball,
            pdbs,
            widths=widths,
            max_learned_steps=args.max_learned_steps,
            beam_branch_actions=args.beam_branch_actions,
            heuristic_weight=args.heuristic_weight,
            best_first_batch_size=args.best_first_batch_size,
            best_first_node_limit=args.best_first_node_limit,
            best_first_time_limit=args.best_first_time_limit,
            best_first_max_depth=args.best_first_max_depth,
            best_first_branch_actions=args.best_first_branch_actions,
        )

    valid_internal = bool(
        result.path is not None and apply_path(state, result.path).rank() == 0
    )
    valid_official = bool(
        result.path is not None
        and np.array_equal(
            official.apply_path(official_state, result.path), official.central_state
        )
    )
    if result.path is not None and not (valid_internal and valid_official):
        raise RuntimeError("GFlowNet search returned a path that failed replay")
    payload = {
        "status": "solved" if valid_internal and valid_official else "search_failed",
        "next_move": (
            official.format_path(result.path[:1]) if result.path else None
        ),
        "solution_path": (
            official.format_path(result.path) if result.path is not None else None
        ),
        "solution_length": len(result.path) if result.path is not None else None,
        "optimality_certified": result.optimal,
        "valid_internal": valid_internal,
        "valid_official": valid_official,
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "algorithm": result.algorithm,
        "reason": result.reason,
        "lower_bound": result.lower_bound,
        "width_or_batch": result.width,
        "expanded_states": result.expanded_states,
        "generated_states": result.generated_states,
        "search_seconds": result.seconds,
    }
    if args.move_only:
        print(payload["next_move"] or ("SOLVED" if result.path == [] else "FAILED"))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
