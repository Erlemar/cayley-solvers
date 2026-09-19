"""Evaluate a macro model directly on harvested DAgger frontier states."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition  # noqa: E402
from cube666.macro_beam import learned_macro_beam_search  # noqa: E402
from cube666.macro_data import load_macro_action_library  # noqa: E402
from cube666.macro_policy import build_macro_policy_model  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--frontier-dir", type=Path, required=True)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--action-library", type=Path, required=True)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--branch-width", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--exact-cost-weight", type=float, default=8.0)
    parser.add_argument("--policy-nll-weight", type=float, default=0.02)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    _, table = load_macro_action_library(
        args.action_library, puzzle.generators, decomposition
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["action_digest"] != table.digest:
        raise ValueError("checkpoint and action-library digests differ")
    model = build_macro_policy_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    with np.load(args.frontier_dir / "frontiers.npz", allow_pickle=False) as payload:
        cluster_states = payload["cluster_states"].astype(np.uint8, copy=False)
        source_pids = payload["source_pids"].astype(np.int32, copy=False)
    frontier_report = json.loads(
        (args.frontier_dir / "report.json").read_text(encoding="utf-8")
    )
    frontier_rows = frontier_report["rows"]
    if len(frontier_rows) != len(cluster_states):
        raise ValueError("frontier report and state count differ")

    rows: list[dict[str, object]] = []
    for index, initial_state in enumerate(cluster_states):
        query_id = str(frontier_rows[index]["query_id"])
        teacher_path = tuple(
            token
            for token in (
                args.query_dir / "paths" / f"{query_id}.path.txt"
            ).read_text(encoding="utf-8").strip().split(".")
            if token
        )
        case_started = time.perf_counter()
        result = learned_macro_beam_search(
            initial_state,
            model,
            table,
            beam_width=args.beam_width,
            branch_width=args.branch_width,
            max_steps=args.max_steps,
            policy_nll_weight=args.policy_nll_weight,
            exact_cost_weight=(
                args.exact_cost_weight if args.exact_cost_weight > 0 else None
            ),
            model_batch_size=2048,
            fallback_exact_cost=True,
        )
        primitive_moves = sum(len(table.paths[action]) for action in result.actions)
        row = {
            "elapsed_seconds": round(time.perf_counter() - case_started, 4),
            "final_residual": result.final_exact_cost,
            "macro_steps": result.macro_steps,
            "model_primitive_moves": primitive_moves,
            "pid": int(source_pids[index]),
            "query_id": query_id,
            "solved": result.solved,
            "teacher_primitive_moves": len(teacher_path),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    report = {
        "beam_width": args.beam_width,
        "branch_width": args.branch_width,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "exact_cost_weight": args.exact_cost_weight,
        "mean_final_residual": statistics.fmean(
            int(row["final_residual"]) for row in rows
        ),
        "policy_nll_weight": args.policy_nll_weight,
        "rows": rows,
        "solved": sum(bool(row["solved"]) for row in rows),
        "states": len(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
