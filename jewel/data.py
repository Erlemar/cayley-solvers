"""Mixed exact and demonstration training data for the Jewel policy."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .ball import ExactBall
from .official import OfficialPuzzle, parse_official_state
from .puzzle import INVERSE_ACTION, SOLVED, apply_action, apply_action_batch, rank_states


def _membership(sorted_values: np.ndarray, queries: np.ndarray) -> np.ndarray:
    loc = np.searchsorted(sorted_values, queries)
    safe = np.minimum(loc, len(sorted_values) - 1)
    return (loc < len(sorted_values)) & (sorted_values[safe] == queries)


def exact_action_masks(
    edge_perm: np.ndarray,
    edge_ori: np.ndarray,
    ring_ori: np.ndarray,
    distances: np.ndarray,
    ball: ExactBall,
) -> np.ndarray:
    masks = np.zeros(len(edge_perm), dtype=np.uint16)
    for depth in np.unique(distances):
        depth = int(depth)
        if depth == 0:
            continue
        selected = np.flatnonzero(distances == depth)
        lower = ball.ranks[depth - 1]
        for action in range(12):
            ep, eo, ro = apply_action_batch(edge_perm[selected], edge_ori[selected], ring_ori[selected], action)
            child_ranks = rank_states(ep, eo, ro)
            hit = _membership(lower, child_ranks)
            masks[selected[hit]] |= np.uint16(1 << action)
    if np.any((distances > 0) & (masks == 0)):
        raise RuntimeError("exact samples without a descending action")
    return masks


def _load_layer_coordinates(root: Path, depth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.load(root / f"edge_perm_d{depth}.npy", mmap_mode="r"),
        np.load(root / f"edge_ori_d{depth}.npy", mmap_mode="r"),
        np.load(root / f"ring_ori_d{depth}.npy", mmap_mode="r"),
    )


def sample_exact_states(
    ball: ExactBall,
    n_balanced: int,
    n_uniform: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = ball.root
    ep_parts: list[np.ndarray] = []
    eo_parts: list[np.ndarray] = []
    ro_parts: list[np.ndarray] = []
    d_parts: list[np.ndarray] = []

    depths = np.arange(1, ball.depth + 1)
    counts = np.full(len(depths), n_balanced // len(depths), dtype=int)
    counts[: n_balanced % len(depths)] += 1
    for depth, count in zip(depths, counts):
        ep, eo, ro = _load_layer_coordinates(root, int(depth))
        idx = rng.integers(0, len(ep), size=int(count))
        ep_parts.append(np.asarray(ep[idx]))
        eo_parts.append(np.asarray(eo[idx]))
        ro_parts.append(np.asarray(ro[idx]))
        d_parts.append(np.full(int(count), depth, dtype=np.uint8))

    sizes = np.asarray([len(layer) for layer in ball.ranks[1:]], dtype=np.float64)
    uniform_depths = rng.choice(depths, size=n_uniform, p=sizes / sizes.sum())
    for depth in depths:
        count = int(np.count_nonzero(uniform_depths == depth))
        if not count:
            continue
        ep, eo, ro = _load_layer_coordinates(root, int(depth))
        idx = rng.integers(0, len(ep), size=count)
        ep_parts.append(np.asarray(ep[idx]))
        eo_parts.append(np.asarray(eo[idx]))
        ro_parts.append(np.asarray(ro[idx]))
        d_parts.append(np.full(count, depth, dtype=np.uint8))
    return np.concatenate(ep_parts), np.concatenate(eo_parts), np.concatenate(ro_parts), np.concatenate(d_parts)


def sample_demonstrations(
    count: int,
    rng: np.random.Generator,
    min_depth: int = 8,
    max_depth: int = 20,
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ep_out = np.empty((count, 12), dtype=np.uint8)
    eo_out = np.empty((count, 12), dtype=np.uint8)
    ro_out = np.empty((count, 6), dtype=np.uint8)
    distances = np.empty(count, dtype=np.uint8)
    masks = np.empty(count, dtype=np.uint16)
    written = 0
    while written < count:
        n = min(batch_size, count - written)
        target_depth = rng.integers(min_depth, max_depth + 1, size=n, dtype=np.uint8)
        ep = np.repeat(SOLVED.edge_perm[None, :], n, axis=0)
        eo = np.repeat(SOLVED.edge_ori[None, :], n, axis=0)
        ro = np.repeat(SOLVED.ring_ori[None, :], n, axis=0)
        previous = np.full(n, -1, dtype=np.int8)
        last = np.zeros(n, dtype=np.int8)
        for step in range(1, max_depth + 1):
            actions = rng.integers(0, 12, size=n, dtype=np.int8)
            blocked = (previous >= 0) & (actions == INVERSE_ACTION[np.maximum(previous, 0)])
            while np.any(blocked):
                actions[blocked] = rng.integers(0, 12, size=int(blocked.sum()), dtype=np.int8)
                blocked = (previous >= 0) & (actions == INVERSE_ACTION[np.maximum(previous, 0)])
            active = target_depth >= step
            for action in range(12):
                idx = np.flatnonzero(active & (actions == action))
                if len(idx):
                    ep[idx], eo[idx], ro[idx] = apply_action_batch(ep[idx], eo[idx], ro[idx], action)
            previous[active] = actions[active]
            last[active] = actions[active]
        sl = slice(written, written + n)
        ep_out[sl], eo_out[sl], ro_out[sl] = ep, eo, ro
        distances[sl] = target_depth
        masks[sl] = np.left_shift(np.uint16(1), INVERSE_ACTION[last].astype(np.uint16))
        written += n
    return ep_out, eo_out, ro_out, distances, masks


def sample_public_trajectories(
    baseline_path: str | Path,
    test_path: str | Path,
    puzzle_info_path: str | Path,
    ball: ExactBall,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Sample deep states from replay-verified public competition paths."""
    official = OfficialPuzzle.load(puzzle_info_path)
    with Path(baseline_path).open(encoding="utf-8", newline="") as handle:
        baselines = list(csv.DictReader(handle))
    with Path(test_path).open(encoding="utf-8", newline="") as handle:
        tests = {row["initial_state_id"]: row for row in csv.DictReader(handle)}

    pool: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]] = []
    for row in baselines:
        official_initial = parse_official_state(tests[row["initial_state_id"]]["initial_state"])
        state = official.to_structured(official_initial)
        actions = official.parse_path(row["path"])
        if not np.array_equal(official.apply_path(official_initial, actions), official.central_state):
            raise ValueError(f"public path fails official replay: {row['initial_state_id']}")
        for step, action in enumerate(actions):
            # The exact ball already supplies complete labels for these states;
            # keep public supervision only in the genuinely deep region.
            if ball.distance_rank(state.rank()) is None:
                pool.append(
                    (
                        state.edge_perm.copy(),
                        state.edge_ori.copy(),
                        state.ring_ori.copy(),
                        len(actions) - step,
                        1 << action,
                    )
                )
            state = apply_action(state, action)
        if state.rank() != 0:
            raise ValueError(f"public path fails structured replay: {row['initial_state_id']}")
    if not pool:
        raise ValueError("no deep public trajectory states were found")
    selected = rng.integers(0, len(pool), size=count)
    return (
        np.stack([pool[i][0] for i in selected]),
        np.stack([pool[i][1] for i in selected]),
        np.stack([pool[i][2] for i in selected]),
        np.asarray([pool[i][3] for i in selected], dtype=np.uint8),
        np.asarray([pool[i][4] for i in selected], dtype=np.uint16),
        len(pool),
    )


def build_mixed_dataset(
    ball_root: str | Path,
    out: str | Path,
    *,
    n_exact_balanced: int = 400_000,
    n_exact_uniform: int = 300_000,
    n_demo: int = 300_000,
    demo_weight: float = 0.25,
    public_baseline: str | Path | None = None,
    public_test: str | Path = "jewel/data/test.csv",
    public_puzzle_info: str | Path = "jewel/data/puzzle_info.json",
    n_public: int = 0,
    public_weight: float = 1.0,
    seed: int = 20260810,
) -> dict:
    ball = ExactBall.load(ball_root)
    rng = np.random.default_rng(seed)
    exact_ep, exact_eo, exact_ro, exact_d = sample_exact_states(
        ball, n_exact_balanced, n_exact_uniform, rng
    )
    exact_masks = exact_action_masks(exact_ep, exact_eo, exact_ro, exact_d, ball)
    demo_ep, demo_eo, demo_ro, demo_d, demo_masks = sample_demonstrations(n_demo, rng)

    public_pool = 0
    if n_public:
        if public_baseline is None:
            raise ValueError("public_baseline is required when n_public > 0")
        public_ep, public_eo, public_ro, public_d, public_masks, public_pool = sample_public_trajectories(
            public_baseline,
            public_test,
            public_puzzle_info,
            ball,
            n_public,
            rng,
        )
    else:
        public_ep = np.empty((0, 12), dtype=np.uint8)
        public_eo = np.empty((0, 12), dtype=np.uint8)
        public_ro = np.empty((0, 6), dtype=np.uint8)
        public_d = np.empty(0, dtype=np.uint8)
        public_masks = np.empty(0, dtype=np.uint16)

    edge_perm = np.concatenate((exact_ep, demo_ep, public_ep))
    edge_ori = np.concatenate((exact_eo, demo_eo, public_eo))
    ring_ori = np.concatenate((exact_ro, demo_ro, public_ro))
    distance = np.concatenate((exact_d, demo_d, public_d))
    action_mask = np.concatenate((exact_masks, demo_masks, public_masks))
    source = np.concatenate(
        (
            np.zeros(len(exact_ep), dtype=np.uint8),
            np.ones(len(demo_ep), dtype=np.uint8),
            np.full(len(public_ep), 2, dtype=np.uint8),
        )
    )
    complete_mask = source == 0
    sample_weight = np.where(
        source == 0,
        1.0,
        np.where(source == 1, demo_weight, public_weight),
    ).astype(np.float32)

    ranks = rank_states(edge_perm, edge_ori, ring_ori)
    split = ((ranks * np.uint64(11_400_714_819_323_198_485)) >> np.uint64(57)).astype(np.uint8)
    # Roughly 95/5 and stable by state, so duplicates cannot cross the split.
    is_validation = split >= 121
    order = rng.permutation(len(edge_perm))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        edge_perm=edge_perm[order],
        edge_ori=edge_ori[order],
        ring_ori=ring_ori[order],
        distance=distance[order],
        action_mask=action_mask[order],
        source=source[order],
        complete_mask=complete_mask[order],
        sample_weight=sample_weight[order],
        is_validation=is_validation[order],
    )
    meta = {
        "count": int(len(edge_perm)),
        "exact": int(len(exact_ep)),
        "demonstration": int(len(demo_ep)),
        "demonstration_weight": float(demo_weight),
        "public": int(len(public_ep)),
        "public_pool": int(public_pool),
        "public_weight": float(public_weight),
        "validation": int(is_validation.sum()),
        "seed": seed,
        "ball_depth": ball.depth,
        "distance_histogram": {str(int(d)): int(np.count_nonzero(distance == d)) for d in np.unique(distance)},
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
