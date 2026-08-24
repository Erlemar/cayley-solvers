from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .ball import ExactBall
from .interactive import interactive_solve
from .model import load_transformer
from .official import OfficialPuzzle, parse_official_state
from .pdb import EdgePatternDatabase
from .puzzle import SOLVED, apply_path, inverse_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline", default="jewel/data/public_16490.csv")
    parser.add_argument("--puzzle-info", default="jewel/data/puzzle_info.json")
    parser.add_argument("--test", default="jewel/data/test.csv")
    parser.add_argument("--ball", default="jewel/artifacts/ball_d8_official")
    parser.add_argument("--pdb", action="append")
    parser.add_argument("--widths", default="1,4,16,64")
    parser.add_argument("--max-learned-steps", type=int, default=18)
    parser.add_argument("--raw-any-solution", action="store_true")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", default="jewel/results/competition_eval.json")
    parser.add_argument("--submission", default="jewel/results/competition_merged.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_transformer(args.checkpoint, device)
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
    official = OfficialPuzzle.load(args.puzzle_info)
    end = len(all_rows) if args.limit is None else min(len(all_rows), args.offset + args.limit)
    selected_rows = all_rows[args.offset:end]
    widths = [int(x) for x in args.widths.split(",")]
    results = []
    merged_paths: dict[str, str] = {}

    for row in selected_rows:
        baseline_path = official.parse_path(row["path"])
        official_initial = parse_official_state(test_rows[row["initial_state_id"]]["initial_state"])
        state = official.to_structured(official_initial)
        reconstructed = apply_path(SOLVED, inverse_path(baseline_path))
        assert state.rank() == reconstructed.rank()
        assert apply_path(state, baseline_path).rank() == 0
        assert np.array_equal(official.apply_path(official_initial, baseline_path), official.central_state)
        result = interactive_solve(
            state,
            model,
            ball,
            pdbs,
            widths=widths,
            max_learned_steps=args.max_learned_steps,
            device=device,
            try_all_widths=True,
            incumbent_length=None if args.raw_any_solution else len(baseline_path),
        )
        valid_internal = result.path is not None and apply_path(state, result.path).rank() == 0
        valid_official = result.path is not None and np.array_equal(
            official.apply_path(official_initial, result.path), official.central_state
        )
        valid = valid_internal and valid_official
        candidate = result.path if valid else None
        chosen = candidate if candidate is not None and len(candidate) < len(baseline_path) else baseline_path
        merged_paths[row["initial_state_id"]] = official.format_path(chosen)
        item = {
            "initial_state_id": int(row["initial_state_id"]),
            "baseline_length": len(baseline_path),
            "candidate_length": len(candidate) if candidate is not None else None,
            "delta": (len(candidate) - len(baseline_path)) if candidate is not None else None,
            "merged_length": len(chosen),
            "width": result.width,
            "expanded": result.expanded_states,
            "seconds": result.seconds,
            "valid": valid,
            "valid_internal": valid_internal,
            "valid_official": valid_official,
        }
        results.append(item)
        print(json.dumps(item), flush=True)

    summary = {
        "checkpoint": args.checkpoint,
        "offset": args.offset,
        "count": len(results),
        "baseline_score": sum(item["baseline_length"] for item in results),
        "candidate_score": sum(item["candidate_length"] for item in results if item["candidate_length"] is not None),
        "merged_score": sum(item["merged_length"] for item in results),
        "solved": sum(item["valid"] for item in results),
        "officially_verified": sum(item["valid_official"] for item in results),
        "wins": sum(item["delta"] is not None and item["delta"] < 0 for item in results),
        "ties": sum(item["delta"] == 0 for item in results),
        "losses": sum(item["delta"] is not None and item["delta"] > 0 for item in results),
        "mean_seconds": float(np.mean([item["seconds"] for item in results])) if results else 0.0,
        "rows": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))

    if args.offset == 0 and len(results) == len(all_rows):
        submission = Path(args.submission)
        submission.parent.mkdir(parents=True, exist_ok=True)
        with submission.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["initial_state_id", "path"])
            writer.writeheader()
            for row in all_rows:
                writer.writerow({"initial_state_id": row["initial_state_id"], "path": merged_paths[row["initial_state_id"]]})


if __name__ == "__main__":
    main()
