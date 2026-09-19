"""Evaluate exact held-out waypoint reproduction and shortcut discovery."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.bridge import (  # noqa: E402
    BridgeTrajectoryDataset,
    OrbitBridgeConfig,
    OrbitBridgePolicyValueNet,
    bridge_beam_search,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher",
        type=Path,
        default=PROJECT / "cube666" / "training" / "bridge_v1" / "trajectories.npz",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--pids", help="optional comma-separated subset of split PIDs")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--minimum-horizon", type=int, default=4)
    parser.add_argument("--maximum-horizon", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=4_096)
    parser.add_argument("--branch-width", type=int, default=8)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--policy-weight", type=float, default=0.15)
    parser.add_argument("--model-batch-size", type=int, default=8_192)
    parser.add_argument("--seed", type=int, default=668)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def parse_pids(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    return {int(value) for value in raw.split(",") if value.strip()}


def replay_actions(
    state: np.ndarray,
    actions: tuple[int, ...],
    generators: np.ndarray,
) -> np.ndarray:
    current = np.asarray(state)
    for action in actions:
        current = current[generators[action]]
    return current


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.minimum_horizon <= 0:
        raise ValueError("samples and minimum-horizon must be positive")
    if args.maximum_horizon < args.minimum_horizon:
        raise ValueError("maximum-horizon must not be below minimum-horizon")
    dataset = BridgeTrajectoryDataset.load(args.teacher)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    manifest = checkpoint["manifest"]
    if manifest["kind"] != "cube666_goal_conditioned_bridge_v1":
        raise ValueError("checkpoint is not a cube666 bridge model")
    if manifest["source_digest"] != dataset.source_digest:
        raise ValueError("checkpoint and trajectory dataset have different source digests")
    if tuple(manifest["move_names"]) != dataset.move_names:
        raise ValueError("checkpoint and trajectory dataset use different move orders")
    config = OrbitBridgeConfig(**manifest["model_config"])
    model = OrbitBridgePolicyValueNet(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    split_value = 1 if args.split == "validation" else 2
    requested_pids = parse_pids(args.pids)
    candidates: list[tuple[int, int, int]] = []
    for path_index, (pid, split) in enumerate(zip(dataset.path_pids, dataset.path_splits)):
        if int(split) != split_value:
            continue
        if requested_pids is not None and int(pid) not in requested_pids:
            continue
        start = int(dataset.path_offsets[path_index])
        stop = int(dataset.path_offsets[path_index + 1]) - 1
        for state_index in range(start, stop):
            maximum = min(
                int(dataset.remaining_lengths[state_index]),
                args.maximum_horizon,
            )
            if maximum >= args.minimum_horizon:
                candidates.append((int(pid), state_index, maximum))
    if not candidates:
        raise ValueError("no eligible held-out windows match the request")
    rng = np.random.default_rng(args.seed)
    replace = args.samples > len(candidates)
    selected = rng.choice(len(candidates), size=args.samples, replace=replace)

    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for sample_index, candidate_index in enumerate(selected, start=1):
        pid, current_index, maximum = candidates[int(candidate_index)]
        horizon = int(rng.integers(args.minimum_horizon, maximum + 1))
        current = dataset.states[current_index]
        goal = dataset.states[current_index + horizon]
        case_started = time.perf_counter()
        result = bridge_beam_search(
            current,
            goal,
            model,
            dataset.orbit_positions,
            dataset.generator_permutations,
            dataset.inverse_actions,
            beam_width=args.beam_width,
            branch_width=args.branch_width,
            maximum_steps=horizon,
            value_weight=args.value_weight,
            policy_weight=args.policy_weight,
            model_batch_size=args.model_batch_size,
        )
        verified = False
        if result.solved:
            verified = np.array_equal(
                replay_actions(current, result.actions, dataset.generator_permutations),
                goal,
            )
            if not verified:
                raise AssertionError("bridge search returned a path that failed exact replay")
        found_length = len(result.actions) if result.solved else None
        row = {
            "depth": result.depth,
            "elapsed_seconds": round(time.perf_counter() - case_started, 4),
            "expanded_states": result.expanded_states,
            "found_length": found_length,
            "generated_states": result.generated_states,
            "horizon": horizon,
            "pid": pid,
            "sample": sample_index,
            "shortcut": bool(result.solved and found_length < horizon),
            "solved": result.solved,
            "verified": verified,
        }
        rows.append(row)
        if args.progress_every > 0 and (
            sample_index % args.progress_every == 0 or not result.solved
        ):
            print(json.dumps(row, sort_keys=True), flush=True)

    solved_rows = [row for row in rows if row["solved"]]
    by_horizon: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        bucket = f"{4 * (int(row['horizon']) // 4):02d}-{4 * (int(row['horizon']) // 4) + 3:02d}"
        by_horizon[bucket].append(row)
    report = {
        "beam_width": args.beam_width,
        "branch_width": args.branch_width,
        "by_horizon": {
            bucket: {
                "solved": sum(bool(row["solved"]) for row in bucket_rows),
                "solve_rate": statistics.fmean(bool(row["solved"]) for row in bucket_rows),
                "total": len(bucket_rows),
            }
            for bucket, bucket_rows in sorted(by_horizon.items())
        },
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "mean_found_length": (
            statistics.fmean(int(row["found_length"]) for row in solved_rows)
            if solved_rows
            else None
        ),
        "mean_teacher_horizon": statistics.fmean(int(row["horizon"]) for row in rows),
        "model_checkpoint": str(args.checkpoint),
        "rows": rows,
        "samples": len(rows),
        "shortcut_moves": sum(
            int(row["horizon"]) - int(row["found_length"])
            for row in solved_rows
            if int(row["found_length"]) < int(row["horizon"])
        ),
        "shortcuts": sum(bool(row["shortcut"]) for row in rows),
        "solve_rate": statistics.fmean(bool(row["solved"]) for row in rows),
        "solved": len(solved_rows),
        "source_digest": dataset.source_digest,
        "split": args.split,
        "verified": sum(bool(row["verified"]) for row in rows),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(args.out)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
