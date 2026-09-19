"""Classical projected oracle gate for the stabilizer ladder.

For each stage, this evaluates every verified eligible macro exactly and greedily
selects the child with the lowest unrestricted 3-cycle residual.  The first
five stages are tested on uniform random A24 permutations.  The invariant-bound
terminal stage is tested on long random walks in its exact residual subgroup.

This is deliberately model-free.  It answers whether the action graph and a
truthful algebraic objective have enough local reachability to justify neural
training.  Stalls are recorded as failures rather than hidden as timeouts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition, permutation_parity  # noqa: E402
from cube666.macro_data import (  # noqa: E402
    load_macro_action_library,
    three_cycle_units_batch,
)
from cube666.macros import reduce_commuting_quarter_turn_path  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--puzzle-info",
        type=Path,
        default=PROJECT / "cayley-py-666-cube" / "puzzle_info.json",
    )
    parser.add_argument(
        "--action-library",
        type=Path,
        default=(
            PROJECT
            / "cube666"
            / "training"
            / "kmc_macro_teacher_allruns_sym8_v3"
            / "action_library.json"
        ),
    )
    parser.add_argument(
        "--ladder",
        type=Path,
        default=PROJECT / "cube666" / "artifacts" / "stabilizer_ladder_v1.json",
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--terminal-walk-depth", type=int, default=200)
    parser.add_argument("--seed", type=int, default=60666)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT / "cube666" / "reports" / "stabilizer_oracle_v1.json",
    )
    return parser.parse_args()


def even_random_permutation(rng: np.random.Generator) -> np.ndarray:
    permutation = rng.permutation(24).astype(np.uint8)
    if permutation_parity(permutation.tolist()):
        permutation[0], permutation[1] = permutation[1], permutation[0]
    return permutation


def inverse_action_indices(effects: np.ndarray) -> np.ndarray:
    by_effect = {effect.tobytes(): index for index, effect in enumerate(effects)}
    result = np.empty(len(effects), dtype=np.int32)
    for index, effect in enumerate(effects):
        inverse = np.empty_like(effect)
        inverse[effect] = np.arange(24, dtype=np.uint8)
        result[index] = by_effect[inverse.tobytes()]
    return result


def terminal_random_walk(
    rng: np.random.Generator,
    effects: np.ndarray,
    inverse: np.ndarray,
    depth: int,
) -> np.ndarray:
    state = np.arange(24, dtype=np.uint8)
    last = -1
    for _ in range(depth):
        action = int(rng.integers(len(effects)))
        if last >= 0 and action == int(inverse[last]):
            action = (action + 1) % len(effects)
        state = state[effects[action]]
        last = action
    return state


def eligible_actions(
    active_cluster: int,
    locked_clusters: tuple[int, ...],
    active_masks: np.ndarray,
    effects: np.ndarray,
    paths: tuple[tuple[str, ...], ...],
) -> tuple[np.ndarray, tuple[tuple[str, ...], ...]]:
    mask = active_masks[:, active_cluster].copy()
    if locked_clusters:
        mask &= ~active_masks[:, locked_clusters].any(axis=1)
    shortest: dict[bytes, tuple[np.ndarray, tuple[str, ...]]] = {}
    for action in np.flatnonzero(mask):
        effect = effects[action, active_cluster]
        path = paths[int(action)]
        old = shortest.get(effect.tobytes())
        if old is None or (len(path), path) < (len(old[1]), old[1]):
            shortest[effect.tobytes()] = (effect.copy(), path)
    ordered = sorted(shortest.values(), key=lambda item: (len(item[1]), item[1]))
    return (
        np.asarray([item[0] for item in ordered], dtype=np.uint8),
        tuple(item[1] for item in ordered),
    )


def greedy_solve(
    initial: np.ndarray,
    effects: np.ndarray,
    paths: tuple[tuple[str, ...], ...],
    *,
    max_steps: int,
) -> dict[str, object]:
    state = initial.copy()
    identity = np.arange(24, dtype=np.uint8)
    initial_cost = int(three_cycle_units_batch(state[None])[0])
    current_cost = initial_cost
    chosen: list[int] = []
    primitive_path: tuple[str, ...] = ()
    trajectory = [current_cost]

    for _ in range(max_steps):
        if np.array_equal(state, identity):
            break
        children = np.take_along_axis(
            np.broadcast_to(state, effects.shape),
            effects,
            axis=1,
        )
        costs = three_cycle_units_batch(children)
        best = min(
            range(len(effects)),
            key=lambda index: (int(costs[index]), len(paths[index]), paths[index]),
        )
        best_cost = int(costs[best])
        if best_cost >= current_cost:
            return {
                "final_cost": current_cost,
                "initial_cost": initial_cost,
                "macro_steps": len(chosen),
                "primitive_moves": len(primitive_path),
                "solved": False,
                "stall": True,
                "trajectory": trajectory,
            }
        state = children[best]
        current_cost = best_cost
        chosen.append(best)
        primitive_path = reduce_commuting_quarter_turn_path(
            primitive_path + paths[best]
        )
        trajectory.append(current_cost)

    return {
        "final_cost": current_cost,
        "initial_cost": initial_cost,
        "macro_steps": len(chosen),
        "primitive_moves": len(primitive_path),
        "solved": bool(np.array_equal(state, identity)),
        "stall": False,
        "trajectory": trajectory,
    }


def main() -> None:
    args = parse_args()
    if min(args.trials, args.max_steps, args.terminal_walk_depth) <= 0:
        raise ValueError("trial and depth settings must be positive")
    rng = np.random.default_rng(args.seed)
    puzzle = Cube666Puzzle.load(args.puzzle_info)
    decomposition = build_decomposition(puzzle.generators)
    _, table = load_macro_action_library(
        args.action_library,
        puzzle.generators,
        decomposition,
    )
    ladder = json.loads(args.ladder.read_text(encoding="utf-8"))
    identity = np.arange(24, dtype=np.uint8)
    active_masks = np.any(table.effects != identity[None, None, :], axis=2)

    started = time.perf_counter()
    stage_reports = []
    for stage in ladder["stages"]:
        active_cluster = int(stage["active_cluster"])
        locked = tuple(int(value) for value in stage["locked_clusters"])
        effects, paths = eligible_actions(
            active_cluster,
            locked,
            active_masks,
            table.effects,
            table.paths,
        )
        inverse = inverse_action_indices(effects)
        rows = []
        for trial in range(args.trials):
            if int(stage["stage_index"]) == len(ladder["stages"]) - 1:
                initial = terminal_random_walk(
                    rng,
                    effects,
                    inverse,
                    args.terminal_walk_depth,
                )
            else:
                initial = even_random_permutation(rng)
            result = greedy_solve(
                initial,
                effects,
                paths,
                max_steps=args.max_steps,
            )
            result["trial"] = trial
            rows.append(result)

        solved_rows = [row for row in rows if row["solved"]]
        report = {
            "action_count": len(effects),
            "active_cluster": active_cluster,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "locked_clusters": list(locked),
            "mean_macro_steps_solved": (
                round(float(np.mean([row["macro_steps"] for row in solved_rows])), 6)
                if solved_rows
                else None
            ),
            "mean_primitive_moves_solved": (
                round(float(np.mean([row["primitive_moves"] for row in solved_rows])), 6)
                if solved_rows
                else None
            ),
            "rows": rows,
            "solved": len(solved_rows),
            "stage_index": int(stage["stage_index"]),
            "trials": args.trials,
        }
        stage_reports.append(report)
        print(
            json.dumps({key: value for key, value in report.items() if key != "rows"}),
            flush=True,
        )

    payload = {
        "action_library_digest": table.digest,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "format_version": 1,
        "gate_b_passed": all(
            stage["solved"] >= int(np.ceil(0.9 * stage["trials"]))
            for stage in stage_reports
        ),
        "ladder": str(args.ladder.resolve()),
        "max_steps": args.max_steps,
        "seed": args.seed,
        "stages": stage_reports,
        "terminal_walk_depth": args.terminal_walk_depth,
        "trials": args.trials,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({key: value for key, value in payload.items() if key != "stages"}, indent=2)
    )


if __name__ == "__main__":
    main()
