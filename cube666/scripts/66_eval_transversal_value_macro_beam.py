"""Use the exact-factor value as a full six-cluster macro-beam heuristic.

The proposal network supplies a small set of short, multi-cluster actions.  A
separately trained stabilizer model ranks their children by summing its broadly
generalized distance estimate over all six 24-piece clusters.  This deliberately
separates solvability supervision from the classical-path proposal distribution.
"""

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

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import (  # noqa: E402
    apply_path,
    build_decomposition,
    parity_repair_path,
    residual_report,
)
from cube666.corners import CornerCoordinateSystem, ExactCornerSolver  # noqa: E402
from cube666.macro_beam import learned_macro_beam_search  # noqa: E402
from cube666.macro_data import cluster_costs, load_macro_action_library  # noqa: E402
from cube666.macro_policy import build_macro_policy_model  # noqa: E402
from cube666.macros import reduce_quarter_turn_path, state_cluster_permutations  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402
from cube666.stabilizer_policy import (  # noqa: E402
    StabilizerModelConfig,
    StabilizerPolicyValue,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube")
    parser.add_argument("--proposal-checkpoint", type=Path, required=True)
    parser.add_argument("--action-library", type=Path, required=True)
    parser.add_argument("--stabilizer-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--incumbent",
        type=Path,
        default=(
            PROJECT
            / "submissions"
            / "cube666_path_context_uniform_v3_hybrid_strictwin_v6.csv"
        ),
    )
    parser.add_argument("--puzzle-indices", required=True)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--branch-width", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--model-batch-size", type=int, default=2048)
    parser.add_argument("--value-batch-size", type=int, default=8192)
    parser.add_argument("--transversal-weight", type=float, default=5.0)
    parser.add_argument("--exact-cost-weight", type=float, default=0.0)
    parser.add_argument("--move-cost-weight", type=float, default=1.0)
    parser.add_argument("--policy-nll-weight", type=float, default=0.02)
    parser.add_argument("--continue-after-solution", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_incumbent_lengths(path: Path) -> dict[int, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            int(row["initial_state_id"]): len(row["path"].split("."))
            for row in csv.DictReader(handle)
        }


def main() -> None:
    args = parse_args()
    if min(
        args.beam_width,
        args.branch_width,
        args.max_steps,
        args.model_batch_size,
        args.value_batch_size,
    ) <= 0:
        raise ValueError("search dimensions must be positive")
    if min(args.transversal_weight, args.exact_cost_weight, args.move_cost_weight) < 0:
        raise ValueError("ranking weights must be non-negative")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    _, table = load_macro_action_library(
        args.action_library,
        puzzle.generators,
        decomposition,
    )

    proposal_payload = torch.load(
        args.proposal_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if proposal_payload["action_digest"] != table.digest:
        raise ValueError("proposal checkpoint and action library digests differ")
    proposal_model = build_macro_policy_model(proposal_payload["model_config"])
    proposal_model.load_state_dict(proposal_payload["model_state_dict"])
    proposal_model.to(device).eval()

    stabilizer_payload = torch.load(
        args.stabilizer_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    stabilizer_config = StabilizerModelConfig(**stabilizer_payload["config"])
    stabilizer_model = StabilizerPolicyValue(stabilizer_config)
    stabilizer_model.load_state_dict(stabilizer_payload["model"])
    stabilizer_model.to(device).eval()
    stabilizer_scale = float(stabilizer_payload["value_scale"])

    @torch.no_grad()
    def ranking_value(states: np.ndarray) -> np.ndarray:
        flat = states.reshape(-1, 24)
        predictions = []
        for start in range(0, len(flat), args.value_batch_size):
            tensor = torch.from_numpy(flat[start : start + args.value_batch_size]).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                _, values = stabilizer_model(tensor)
            predictions.append(values.float().cpu().numpy() * stabilizer_scale)
        transversal = np.concatenate(predictions).reshape(len(states), 6).sum(axis=1)
        if args.exact_cost_weight:
            exact = cluster_costs(states).sum(axis=1).astype(np.float32, copy=False)
        else:
            exact = 0.0
        return args.transversal_weight * transversal + args.exact_cost_weight * exact

    incumbent_lengths = load_incumbent_lengths(args.incumbent)
    requested = [int(value) for value in args.puzzle_indices.split(",") if value.strip()]
    all_states = list(puzzle.iter_test_states(args.data_dir / "test.csv"))
    corner_solver = ExactCornerSolver.build(
        CornerCoordinateSystem.discover(puzzle.generators, decomposition)
    )
    rows = []
    started = time.perf_counter()

    for index in requested:
        case_started = time.perf_counter()
        state_id, state = all_states[index]
        state_id_int = int(state_id)
        corner_path = corner_solver.solve(state, puzzle.solved_state)
        corner_state = apply_path(state, puzzle.generators, corner_path)
        parity_path = parity_repair_path(
            residual_report(
                corner_state,
                puzzle.solved_state,
                decomposition,
            ).parity_vector,
            decomposition,
        )
        setup_path = corner_path + parity_path
        normalized = apply_path(corner_state, puzzle.generators, parity_path)
        clusters = np.asarray(
            state_cluster_permutations(
                normalized,
                puzzle.solved_state,
                decomposition,
            ),
            dtype=np.uint8,
        )
        result = learned_macro_beam_search(
            clusters,
            proposal_model,
            table,
            beam_width=args.beam_width,
            branch_width=args.branch_width,
            max_steps=args.max_steps,
            policy_nll_weight=args.policy_nll_weight,
            move_cost_weight=args.move_cost_weight,
            model_batch_size=args.model_batch_size,
            stop_on_first_solution=not args.continue_after_solution,
            ranking_value_predictor=ranking_value,
        )
        full_path: tuple[str, ...] | None = None
        replay_verified = False
        if result.solved:
            macro_path = tuple(
                move for action in result.actions for move in table.paths[action]
            )
            full_path = reduce_quarter_turn_path(setup_path + macro_path)
            replay_verified = puzzle.apply_path(state, full_path) == puzzle.solved_state
            if not replay_verified:
                raise AssertionError(f"PID {state_id} solved projection but failed full replay")
        incumbent = incumbent_lengths[state_id_int]
        row = {
            "elapsed_seconds": round(time.perf_counter() - case_started, 4),
            "expanded_states": result.expanded_states,
            "final_exact_cost": result.final_exact_cost,
            "generated_states": result.generated_states,
            "incumbent_moves": incumbent,
            "initial_exact_cost": int(cluster_costs(clusters).sum()),
            "macro_steps": result.macro_steps,
            "moves": None if full_path is None else len(full_path),
            "pid": state_id_int,
            "replay_verified": replay_verified,
            "score_delta": None if full_path is None else len(full_path) - incumbent,
            "solved": result.solved,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    solved_rows = [row for row in rows if row["solved"]]
    report = {
        "action_digest": table.digest,
        "beam_width": args.beam_width,
        "branch_width": args.branch_width,
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "exact_cost_weight": args.exact_cost_weight,
        "max_steps": args.max_steps,
        "mean_moves": (
            None
            if not solved_rows
            else statistics.fmean(int(row["moves"]) for row in solved_rows)
        ),
        "mean_score_delta": (
            None
            if not solved_rows
            else statistics.fmean(int(row["score_delta"]) for row in solved_rows)
        ),
        "move_cost_weight": args.move_cost_weight,
        "policy_nll_weight": args.policy_nll_weight,
        "proposal_checkpoint": str(args.proposal_checkpoint.resolve()),
        "replay_verified": sum(bool(row["replay_verified"]) for row in rows),
        "rows": rows,
        "solved": len(solved_rows),
        "stabilizer_checkpoint": str(args.stabilizer_checkpoint.resolve()),
        "total": len(rows),
        "transversal_weight": args.transversal_weight,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
