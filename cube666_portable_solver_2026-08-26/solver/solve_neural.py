"""Solve exact-sticker CUBE666 states with the accepted autonomous neural macro beam."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cube666.classical import (  # noqa: E402
    apply_path,
    build_decomposition,
    parity_repair_path,
    residual_report,
)
from cube666.corners import CornerCoordinateSystem, ExactCornerSolver  # noqa: E402
from cube666.macro_beam import learned_macro_beam_search  # noqa: E402
from cube666.macro_data import (  # noqa: E402
    cluster_costs,
    load_macro_action_library,
    validate_factorized_action_table,
)
from cube666.macro_policy import build_macro_policy_model  # noqa: E402
from cube666.macros import reduce_quarter_turn_path, state_cluster_permutations  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


DEFAULT_DATA = ROOT / "cayley-py-666-cube"
DEFAULT_CHECKPOINT = (
    ROOT / "models" / "cube666_finisher_factorized_pro6000_v1" / "checkpoint.pt"
)
DEFAULT_ACTIONS = (
    ROOT / "cube666" / "training" / "finisher_policy_v1" / "action_library.json"
)


def parse_indices(raw: str) -> list[int]:
    return sorted({int(value) for value in raw.split(",") if value.strip()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--action-library", type=Path, default=DEFAULT_ACTIONS)
    parser.add_argument("--indices", default="0", help="comma-separated zero-based test-row indices")
    parser.add_argument("--all-puzzles", action="store_true")
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--branch-width", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--rescue-beam-width", type=int, default=512)
    parser.add_argument("--rescue-branch-width", type=int, default=32)
    parser.add_argument("--rescue-max-steps", type=int, default=112)
    parser.add_argument("--policy-nll-weight", type=float, default=0.02)
    parser.add_argument("--model-batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "neural_solutions.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs" / "neural_solutions.json")
    return parser.parse_args()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("initial_state_id", "path"))
        writer.writeheader()
        for row in rows:
            if row["solved"]:
                writer.writerow(
                    {"initial_state_id": row["initial_state_id"], "path": row["path"]}
                )
    temporary.replace(path)


def choose_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    return torch.device(raw)


def main() -> None:
    args = parse_args()
    if args.beam_width <= 0 or args.branch_width <= 0 or args.max_steps <= 0:
        raise ValueError("beam dimensions must be positive")
    started = time.perf_counter()
    device = choose_device(args.device)
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    _, table = load_macro_action_library(
        args.action_library,
        puzzle.generators,
        decomposition,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["action_digest"] != table.digest:
        raise ValueError("checkpoint and action-library digests differ")
    if checkpoint["model_config"].get("architecture") == "factorized":
        validate_factorized_action_table(table)
    model = build_macro_policy_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    test_rows = list(puzzle.iter_test_states(args.data_dir / "test.csv"))
    indices = list(range(len(test_rows))) if args.all_puzzles else parse_indices(args.indices)
    if not indices or any(index < 0 or index >= len(test_rows) for index in indices):
        raise IndexError("selected puzzle index is outside test.csv")

    corner_solver = ExactCornerSolver.build(
        CornerCoordinateSystem.discover(puzzle.generators, decomposition)
    )
    rows: list[dict[str, object]] = []
    for ordinal, index in enumerate(indices, start=1):
        case_started = time.perf_counter()
        state_id, full_state = test_rows[index]
        corner_path = corner_solver.solve(full_state, puzzle.solved_state)
        corner_state = apply_path(full_state, puzzle.generators, corner_path)
        parity_path = parity_repair_path(
            residual_report(corner_state, puzzle.solved_state, decomposition).parity_vector,
            decomposition,
        )
        normalized = apply_path(corner_state, puzzle.generators, parity_path)
        clusters = np.asarray(
            state_cluster_permutations(normalized, puzzle.solved_state, decomposition),
            dtype=np.uint8,
        )
        initial_cost = int(cluster_costs(clusters).sum())
        result = learned_macro_beam_search(
            clusters,
            model,
            table,
            beam_width=args.beam_width,
            branch_width=args.branch_width,
            max_steps=args.max_steps,
            policy_nll_weight=args.policy_nll_weight,
            model_batch_size=args.model_batch_size,
        )
        attempts = 1
        effective = (args.beam_width, args.branch_width, args.max_steps)
        if not result.solved and args.rescue_beam_width:
            attempts += 1
            effective = (
                args.rescue_beam_width,
                args.rescue_branch_width,
                args.rescue_max_steps,
            )
            result = learned_macro_beam_search(
                clusters,
                model,
                table,
                beam_width=args.rescue_beam_width,
                branch_width=args.rescue_branch_width,
                max_steps=args.rescue_max_steps,
                policy_nll_weight=args.policy_nll_weight,
                model_batch_size=args.model_batch_size,
            )

        path: tuple[str, ...] = ()
        replay_verified = False
        if result.solved:
            macro_path = tuple(
                move for action in result.actions for move in table.paths[action]
            )
            path = reduce_quarter_turn_path(corner_path + parity_path + macro_path)
            replay_verified = puzzle.apply_path(full_state, path) == puzzle.solved_state
            if not replay_verified:
                raise AssertionError(f"state {state_id}: neural path failed full replay")

        row = {
            "attempts": attempts,
            "beam_width": effective[0],
            "branch_width": effective[1],
            "elapsed_seconds": round(time.perf_counter() - case_started, 6),
            "final_exact_cost": int(cluster_costs(result.final_state).sum()),
            "initial_exact_cost": initial_cost,
            "initial_state_id": state_id,
            "macro_steps": result.macro_steps,
            "max_steps": effective[2],
            "path": ".".join(path),
            "primitive_moves": len(path) if path else None,
            "replay_verified": replay_verified,
            "solved": bool(result.solved),
        }
        rows.append(row)
        if args.progress_every > 0 and (ordinal % args.progress_every == 0 or not result.solved):
            print(json.dumps({key: value for key, value in row.items() if key != "path"}, sort_keys=True))

    solved_rows = [row for row in rows if row["solved"] and row["replay_verified"]]
    report = {
        "action_digest": table.digest,
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "mean_primitive_moves": (
            round(statistics.fmean(int(row["primitive_moves"]) for row in solved_rows), 6)
            if solved_rows
            else None
        ),
        "output": str(args.output.resolve()),
        "replay_verified": len(solved_rows),
        "rows": rows,
        "solved": sum(bool(row["solved"]) for row in rows),
        "total": len(rows),
    }
    write_csv(args.output, rows)
    atomic_text(args.report, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))
    if len(solved_rows) != len(rows) and not args.allow_partial:
        raise SystemExit("one or more states were unsolved; partial outputs were written")


if __name__ == "__main__":
    main()

