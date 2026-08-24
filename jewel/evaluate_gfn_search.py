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
from .puzzle import SOLVED, apply_path, inverse_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["beam", "best-first", "hybrid"], default="hybrid")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--baseline", default="jewel/data/public_16490.csv")
    parser.add_argument("--puzzle-info", default="jewel/data/puzzle_info.json")
    parser.add_argument("--test", default="jewel/data/test.csv")
    parser.add_argument("--ball", default="jewel/artifacts/ball_d8_official")
    parser.add_argument("--pdb", action="append")
    parser.add_argument("--widths", default="1,4,16,64")
    parser.add_argument("--max-learned-steps", type=int, default=20)
    parser.add_argument("--beam-branch-actions", type=int, default=4)
    parser.add_argument("--heuristic-weight", type=float, default=0.35)
    parser.add_argument("--depth-weight", type=float, default=0.0)
    parser.add_argument("--best-first-batch-size", type=int, default=256)
    parser.add_argument("--best-first-node-limit", type=int, default=100_000)
    parser.add_argument("--best-first-time-limit", type=float, default=30.0)
    parser.add_argument("--best-first-max-depth", type=int, default=32)
    parser.add_argument("--best-first-branch-actions", type=int, default=12)
    parser.add_argument("--raw-any-solution", action="store_true")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--out", default="jewel/results/gflownet_search_evaluation.json")
    parser.add_argument("--submission", default="jewel/results/gflownet_search_merged.csv")
    parser.add_argument("--raw-submission", default="jewel/results/gflownet_search_raw.csv")
    args = parser.parse_args()

    if args.sample_size is not None and args.limit is not None:
        raise ValueError("use either --sample-size or --limit, not both")
    device = torch.device(args.device)
    official = OfficialPuzzle.load(args.puzzle_info)
    model, _, checkpoint = load_gflownet(args.checkpoint, device)
    policy = GFNBackwardPolicy(model, official, device=device)
    ball = ExactBall.load(args.ball)
    pdb_paths = args.pdb or [
        "jewel/artifacts/pdb_edges_0_4_official.npy",
        "jewel/artifacts/pdb_edges_5_9_official.npy",
    ]
    pdbs = [EdgePatternDatabase.load(path) for path in pdb_paths]
    with open(args.baseline, encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    with open(args.test, encoding="utf-8", newline="") as handle:
        test_rows = {row["initial_state_id"]: row for row in csv.DictReader(handle)}

    available = np.arange(args.offset, len(all_rows))
    if args.sample_size is not None:
        if args.sample_size > len(available):
            raise ValueError("sample size exceeds available rows")
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(available, size=args.sample_size, replace=False))
    else:
        stop = (
            len(all_rows)
            if args.limit is None
            else min(len(all_rows), args.offset + args.limit)
        )
        indices = np.arange(args.offset, stop)
    selected_rows = [all_rows[int(index)] for index in indices]
    widths = tuple(int(value) for value in args.widths.split(","))
    rows: list[dict] = []
    merged_paths: dict[str, str] = {}

    for baseline_row in selected_rows:
        state_id = baseline_row["initial_state_id"]
        baseline_path = official.parse_path(baseline_row["path"])
        official_initial = parse_official_state(test_rows[state_id]["initial_state"])
        state = official.to_structured(official_initial)
        reconstructed = apply_path(SOLVED, inverse_path(baseline_path))
        assert state.rank() == reconstructed.rank()
        assert apply_path(state, baseline_path).rank() == 0
        incumbent_length = None if args.raw_any_solution else len(baseline_path)

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
                depth_weight=args.depth_weight,
                incumbent_length=incumbent_length,
                try_all_widths=True,
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
                depth_weight=args.depth_weight,
                incumbent_length=incumbent_length,
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
                depth_weight=args.depth_weight,
                incumbent_length=incumbent_length,
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
                official.apply_path(official_initial, result.path),
                official.central_state,
            )
        )
        candidate = result.path if valid_internal and valid_official else None
        chosen = (
            candidate
            if candidate is not None and len(candidate) < len(baseline_path)
            else baseline_path
        )
        merged_paths[state_id] = official.format_path(chosen)
        row = {
            "initial_state_id": int(state_id),
            "baseline_length": len(baseline_path),
            "candidate_length": len(candidate) if candidate is not None else None,
            "candidate_path": official.format_path(candidate) if candidate is not None else None,
            "delta": len(candidate) - len(baseline_path) if candidate is not None else None,
            "merged_length": len(chosen),
            "valid_internal": valid_internal,
            "valid_official": valid_official,
            "optimal": result.optimal,
            "algorithm": result.algorithm,
            "width_or_batch": result.width,
            "expanded": result.expanded_states,
            "generated": result.generated_states,
            "seconds": result.seconds,
            "reason": result.reason,
            "lower_bound": result.lower_bound,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    solved = [row for row in rows if row["candidate_length"] is not None]
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "mode": args.mode,
        "count": len(rows),
        "solved_and_verified": len(solved),
        "candidate_score_on_solved": int(sum(row["candidate_length"] for row in solved)),
        "baseline_score_on_solved": int(sum(row["baseline_length"] for row in solved)),
        "baseline_score_all_rows": int(sum(row["baseline_length"] for row in rows)),
        "merged_score": int(sum(row["merged_length"] for row in rows)),
        "wins": sum(row["delta"] is not None and row["delta"] < 0 for row in rows),
        "ties": sum(row["delta"] == 0 for row in rows),
        "losses": sum(row["delta"] is not None and row["delta"] > 0 for row in rows),
        "optimality_certificates": sum(bool(row["optimal"]) for row in rows),
        "total_expanded": int(sum(row["expanded"] for row in rows)),
        "mean_expanded": float(np.mean([row["expanded"] for row in rows])) if rows else 0.0,
        "total_seconds": float(sum(row["seconds"] for row in rows)),
        "mean_seconds": float(np.mean([row["seconds"] for row in rows])) if rows else 0.0,
        "failure_ids": [row["initial_state_id"] for row in rows if row["candidate_length"] is None],
    }
    report = {"args": vars(args), "summary": summary, "rows": rows}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    full_run = args.sample_size is None and args.offset == 0 and len(rows) == len(all_rows)
    if full_run:
        submission = Path(args.submission)
        submission.parent.mkdir(parents=True, exist_ok=True)
        with submission.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["initial_state_id", "path"])
            writer.writeheader()
            for row in all_rows:
                writer.writerow(
                    {
                        "initial_state_id": row["initial_state_id"],
                        "path": merged_paths[row["initial_state_id"]],
                    }
                )
        if len(solved) == len(rows):
            raw_submission = Path(args.raw_submission)
            raw_submission.parent.mkdir(parents=True, exist_ok=True)
            with raw_submission.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["initial_state_id", "path"])
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            "initial_state_id": row["initial_state_id"],
                            "path": row["candidate_path"],
                        }
                    )


if __name__ == "__main__":
    main()
