from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .ball import ExactBall
from .gflownet import GFNBackwardPolicy, load_gflownet
from .model import load_transformer
from .official import OfficialPuzzle, parse_official_state
from .pdb import EdgePatternDatabase
from .policy_search import TransformerPolicy, greedy_policy_path, phs_search, policy_ida_star
from .puzzle import apply_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["gflownet", "transformer"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", choices=["phsh", "phsstar", "levin"], default="phsh")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--puzzle-info", default="jewel/data/puzzle_info.json")
    parser.add_argument("--test", default="jewel/data/test.csv")
    parser.add_argument("--baseline", default="jewel/data/public_16490.csv")
    parser.add_argument("--ball", default="jewel/artifacts/ball_d8_official")
    parser.add_argument("--pdb", action="append")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--node-limit", type=int, default=25_000)
    parser.add_argument("--time-limit", type=float, default=5.0)
    parser.add_argument("--policy-batch-size", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=40)
    parser.add_argument("--incumbent-bound", action="store_true")
    parser.add_argument("--model-greedy-incumbent", action="store_true")
    parser.add_argument("--prove", action="store_true")
    parser.add_argument("--proof-node-limit", type=int, default=100_000)
    parser.add_argument("--proof-time-limit", type=float, default=10.0)
    parser.add_argument("--out", default="jewel/results/policy_search_competition.json")
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
        model, _, checkpoint = load_gflownet(args.checkpoint, device)
        policy = GFNBackwardPolicy(model, official, device=device)
        checkpoint_step = int(checkpoint.get("step", -1))
    else:
        model = load_transformer(args.checkpoint, device)
        policy = TransformerPolicy(model, device=device)
        checkpoint_step = -1

    with open(args.test, encoding="utf-8", newline="") as handle:
        tests = {int(row["initial_state_id"]): row for row in csv.DictReader(handle)}
    with open(args.baseline, encoding="utf-8", newline="") as handle:
        baseline_rows = list(csv.DictReader(handle))
    if args.sample_size is not None and args.limit is not None:
        raise ValueError("use either --sample-size or --limit, not both")
    if args.sample_size is not None:
        available = np.arange(args.offset, len(baseline_rows))
        if args.sample_size > len(available):
            raise ValueError("sample size exceeds available rows")
        rng = np.random.default_rng(args.seed)
        selected_indices = np.sort(rng.choice(available, size=args.sample_size, replace=False))
        selected = [baseline_rows[int(index)] for index in selected_indices]
    else:
        stop = (
            len(baseline_rows)
            if args.limit is None
            else min(len(baseline_rows), args.offset + args.limit)
        )
        selected = baseline_rows[args.offset:stop]
    rows: list[dict] = []

    for baseline_row in selected:
        state_id = int(baseline_row["initial_state_id"])
        official_initial = parse_official_state(tests[state_id]["initial_state"])
        state = official.to_structured(official_initial)
        baseline_path = official.parse_path(baseline_row["path"])
        incumbent_candidates: list[list[int]] = []
        if args.incumbent_bound:
            incumbent_candidates.append(baseline_path)
        if args.model_greedy_incumbent:
            greedy = greedy_policy_path(state, ball, policy=policy, max_steps=64)
            if greedy is not None:
                incumbent_candidates.append(greedy)
        incumbent = min(incumbent_candidates, key=len) if incumbent_candidates else None
        phs = phs_search(
            state,
            ball,
            pdbs,
            policy=policy,
            variant=args.variant,
            incumbent_path=incumbent,
            max_depth=args.max_depth,
            node_limit=args.node_limit,
            time_limit=args.time_limit,
            policy_batch_size=args.policy_batch_size,
        )
        candidate = phs.path
        valid_internal = candidate is not None and apply_path(state, candidate).rank() == 0
        valid_official = bool(
            candidate is not None
            and np.array_equal(
                official.apply_path(official_initial, candidate), official.central_state
            )
        )
        exact = None
        if args.prove and candidate is not None:
            exact = policy_ida_star(
                state,
                ball,
                pdbs,
                policy=policy,
                ordering="heuristic",
                incumbent_path=candidate,
                max_depth=args.max_depth,
                node_limit=args.proof_node_limit,
                time_limit=args.proof_time_limit,
            )
        row = {
            "initial_state_id": state_id,
            "baseline_length": len(baseline_path),
            "candidate_length": len(candidate) if valid_internal and valid_official else None,
            "delta": len(candidate) - len(baseline_path) if valid_internal and valid_official else None,
            "valid_internal": valid_internal,
            "valid_official": valid_official,
            "phs_optimal": phs.optimal,
            "phs_expanded": phs.expanded,
            "phs_generated": phs.generated,
            "phs_seconds": phs.seconds,
            "phs_reason": phs.reason,
            "lower_bound": phs.lower_bound,
            "proof_optimal": exact.optimal if exact is not None else None,
            "proof_expanded": exact.expanded if exact is not None else None,
            "proof_seconds": exact.seconds if exact is not None else None,
            "proof_reason": exact.reason if exact is not None else None,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    solved = [row for row in rows if row["candidate_length"] is not None]
    summary = {
        "policy": args.policy,
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint_step,
        "variant": args.variant,
        "count": len(rows),
        "solved_and_verified": len(solved),
        "raw_candidate_score_on_solved": int(sum(row["candidate_length"] for row in solved)),
        "baseline_score_on_solved": int(sum(row["baseline_length"] for row in solved)),
        "baseline_score_all_rows": int(sum(row["baseline_length"] for row in rows)),
        "wins": sum(row["delta"] is not None and row["delta"] < 0 for row in rows),
        "ties": sum(row["delta"] == 0 for row in rows),
        "losses": sum(row["delta"] is not None and row["delta"] > 0 for row in rows),
        "optimality_certificates": sum(bool(row["proof_optimal"] or row["phs_optimal"]) for row in rows),
        "total_phs_expanded": int(sum(row["phs_expanded"] for row in rows)),
        "mean_phs_expanded": float(np.mean([row["phs_expanded"] for row in rows])) if rows else 0.0,
        "total_phs_seconds": float(sum(row["phs_seconds"] for row in rows)),
    }
    report = {"args": vars(args), "summary": summary, "rows": rows}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
