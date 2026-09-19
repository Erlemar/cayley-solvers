"""Build self-supervised primitive-cost walks over an inverse-closed short macro set."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


CLUSTERS = 6
SIZE = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-effects", type=Path, required=True)
    parser.add_argument("--action-costs", type=Path, required=True)
    parser.add_argument("--action-library", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--maximum-action-cost", type=int, default=14)
    parser.add_argument("--samples", type=int, default=250_000)
    parser.add_argument("--maximum-depth", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=76666)
    return parser.parse_args()


def inverse_effect(effect: np.ndarray) -> np.ndarray:
    inverse = np.empty_like(effect)
    identity = np.broadcast_to(np.arange(SIZE, dtype=np.uint8), effect.shape)
    np.put_along_axis(inverse, effect, identity, axis=-1)
    return inverse


def three_cycle_costs(states: np.ndarray) -> np.ndarray:
    batch = len(states)
    costs = np.empty((batch, CLUSTERS), dtype=np.float32)
    for row in range(batch):
        for cluster in range(CLUSTERS):
            permutation = states[row, cluster]
            seen = np.zeros(SIZE, dtype=bool)
            cycles = 0
            for start in range(SIZE):
                if seen[start]:
                    continue
                cycles += 1
                cursor = start
                while not seen[cursor]:
                    seen[cursor] = True
                    cursor = int(permutation[cursor])
            costs[row, cluster] = (SIZE - cycles) / 2.0
    return costs


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.maximum_depth <= 0 or args.batch_size <= 0:
        raise ValueError("samples, maximum-depth, and batch-size must be positive")
    started = time.perf_counter()
    all_effects = np.load(args.action_effects, allow_pickle=False).astype(
        np.uint8, copy=False
    )
    all_costs = np.load(args.action_costs, allow_pickle=False).astype(
        np.int32, copy=False
    )
    library = json.loads(args.action_library.read_text(encoding="utf-8"))
    all_paths = library["paths"]
    eligible = np.flatnonzero(all_costs <= args.maximum_action_cost)
    effect_to_action = {
        all_effects[action].tobytes(): int(action) for action in eligible
    }
    inverse_original = np.full(len(eligible), -1, dtype=np.int64)
    for position, action in enumerate(eligible):
        inverse_original[position] = effect_to_action.get(
            inverse_effect(all_effects[action]).tobytes(), -1
        )
    closed_mask = inverse_original >= 0
    closed_original = eligible[closed_mask]
    closed_set = set(int(action) for action in closed_original)
    closed_original = np.asarray(
        [
            int(action)
            for action in closed_original
            if int(effect_to_action[inverse_effect(all_effects[action]).tobytes()])
            in closed_set
        ],
        dtype=np.int64,
    )
    original_to_local = {
        int(action): local for local, action in enumerate(closed_original)
    }
    effects = all_effects[closed_original]
    costs = all_costs[closed_original]
    paths = [all_paths[int(action)] for action in closed_original]
    inverse_actions = np.asarray(
        [
            original_to_local[
                int(effect_to_action[inverse_effect(effect).tobytes()])
            ]
            for effect in effects
        ],
        dtype=np.int32,
    )
    identity = np.broadcast_to(
        np.arange(SIZE, dtype=np.uint8), (CLUSTERS, SIZE)
    )
    for action, inverse_action in enumerate(inverse_actions):
        replay = np.take_along_axis(effects[action], effects[inverse_action], axis=-1)
        if not np.array_equal(replay, identity):
            raise AssertionError(f"inverse action {action} failed exact replay")

    rng = np.random.default_rng(args.seed)
    states_out = np.empty((args.samples, CLUSTERS, SIZE), dtype=np.uint8)
    labels_out = np.empty((args.samples, 1), dtype=np.int32)
    counts_out = np.ones(args.samples, dtype=np.int16)
    values_out = np.empty(args.samples, dtype=np.float32)
    clusters_out = np.empty((args.samples, CLUSTERS), dtype=np.float32)
    depths_out = np.empty(args.samples, dtype=np.uint8)
    solution_actions_out = np.full(
        (args.samples, args.maximum_depth), -1, dtype=np.int32
    )
    groups_out = np.arange(args.samples, dtype=np.int32)
    depth_histogram = np.zeros(args.maximum_depth + 1, dtype=np.int64)
    for start in range(0, args.samples, args.batch_size):
        stop = min(start + args.batch_size, args.samples)
        count = stop - start
        walk_states = np.broadcast_to(identity, (count, CLUSTERS, SIZE)).copy()
        depths = rng.integers(1, args.maximum_depth + 1, size=count)
        walk_costs = np.zeros(count, dtype=np.int32)
        last_actions = np.full(count, -1, dtype=np.int32)
        walk_actions = np.full((count, args.maximum_depth), -1, dtype=np.int32)
        for step in range(args.maximum_depth):
            actions = rng.integers(0, len(effects), size=count, dtype=np.int32)
            if step:
                forbidden = inverse_actions[last_actions]
                collision = actions == forbidden
                while np.any(collision):
                    actions[collision] = rng.integers(
                        0, len(effects), size=int(collision.sum()), dtype=np.int32
                    )
                    collision = actions == forbidden
            active = step < depths
            active_rows = np.flatnonzero(active)
            walk_states[active_rows] = np.take_along_axis(
                walk_states[active_rows], effects[actions[active_rows]], axis=-1
            )
            walk_costs[active] += costs[actions[active]]
            last_actions[active] = actions[active]
            walk_actions[active_rows, step] = actions[active_rows]
        residuals = three_cycle_costs(walk_states)
        residual_sum = residuals.sum(axis=1, keepdims=True)
        weights = residuals / np.maximum(residual_sum, 1.0)
        cluster_targets = weights * walk_costs[:, None]
        states_out[start:stop] = walk_states
        labels_out[start:stop, 0] = inverse_actions[last_actions]
        values_out[start:stop] = walk_costs
        clusters_out[start:stop] = cluster_targets
        depths_out[start:stop] = depths
        for solution_step in range(args.maximum_depth):
            active = solution_step < depths
            active_rows = np.flatnonzero(active)
            source_steps = depths[active_rows] - 1 - solution_step
            forward_actions = walk_actions[active_rows, source_steps]
            solution_actions_out[start + active_rows, solution_step] = inverse_actions[
                forward_actions
            ]
        depth_histogram += np.bincount(depths, minlength=len(depth_histogram))
        print(json.dumps({"generated": stop, "samples": args.samples}), flush=True)

    action_digest = hashlib.sha256(
        json.dumps(paths, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "teacher.npz",
        states=states_out,
        teacher_actions=labels_out,
        teacher_action_counts=counts_out,
        search_value_targets=values_out,
        search_cluster_targets=clusters_out,
        walk_depths=depths_out,
        teacher_solution_actions=solution_actions_out,
    )
    np.save(args.out_dir / "source_state_ids.npy", groups_out)
    np.save(args.out_dir / "action_effects.npy", effects)
    np.save(args.out_dir / "action_costs.npy", costs.astype(np.int16))
    np.save(args.out_dir / "inverse_actions.npy", inverse_actions)
    (args.out_dir / "action_library.json").write_text(
        json.dumps(
            {
                "action_digest": action_digest,
                "format_version": 1,
                "paths": paths,
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "action_count": len(effects),
        "action_digest": action_digest,
        "eligible_actions": len(eligible),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "inverse_closed_actions": len(effects),
        "maximum_action_cost": args.maximum_action_cost,
        "maximum_depth": args.maximum_depth,
        "mean_primitive_target": float(values_out.mean()),
        "samples": args.samples,
        "seed": args.seed,
        "target_maximum": float(values_out.max()),
        "target_minimum": float(values_out.min()),
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
