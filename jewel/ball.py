"""Compact exact reverse-ball construction and lookup."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .puzzle import JewelState, SOLVED, apply_action, apply_action_batch, rank_states


@dataclass(slots=True)
class ExactBall:
    root: Path
    ranks: list[np.ndarray]

    @classmethod
    def load(cls, root: str | Path, mmap_mode: str | None = "r") -> "ExactBall":
        root = Path(root)
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        ranks = [np.load(root / f"ranks_d{d}.npy", mmap_mode=mmap_mode) for d in range(meta["depth"] + 1)]
        return cls(root, ranks)

    @property
    def depth(self) -> int:
        return len(self.ranks) - 1

    def distance_rank(self, rank: int) -> int | None:
        value = np.uint64(rank)
        for d, layer in enumerate(self.ranks):
            i = int(np.searchsorted(layer, value))
            if i < len(layer) and layer[i] == value:
                return d
        return None

    def distance(self, state: JewelState) -> int | None:
        return self.distance_rank(state.rank())

    def optimal_action_mask(self, state: JewelState, distance: int | None = None) -> int:
        d = self.distance(state) if distance is None else distance
        if d is None:
            raise ValueError("state is not in exact ball")
        if d == 0:
            return 0
        lower = self.ranks[d - 1]
        mask = 0
        for action in range(12):
            rank = np.uint64(apply_action(state, action).rank())
            i = int(np.searchsorted(lower, rank))
            if i < len(lower) and lower[i] == rank:
                mask |= 1 << action
        if not mask:
            raise RuntimeError(f"no descending action found at exact distance {d}")
        return mask

    def exact_path(self, state: JewelState) -> list[int] | None:
        d = self.distance(state)
        if d is None:
            return None
        out: list[int] = []
        cur = state
        while d:
            mask = self.optimal_action_mask(cur, d)
            action = (mask & -mask).bit_length() - 1
            out.append(action)
            cur = apply_action(cur, action)
            d -= 1
        return out


def _save_layer(root: Path, depth: int, ranks: np.ndarray, ep: np.ndarray, eo: np.ndarray, ro: np.ndarray) -> None:
    np.save(root / f"ranks_d{depth}.npy", ranks)
    np.save(root / f"edge_perm_d{depth}.npy", ep)
    np.save(root / f"edge_ori_d{depth}.npy", eo)
    np.save(root / f"ring_ori_d{depth}.npy", ro)


def build_exact_ball(root: str | Path, max_depth: int, *, overwrite: bool = False) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / "meta.json"
    start_depth = 0
    if meta_path.exists() and not overwrite:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["depth"] >= max_depth:
            return meta
        start_depth = int(meta["depth"])
        ranks = np.load(root / f"ranks_d{start_depth}.npy")
        ep = np.load(root / f"edge_perm_d{start_depth}.npy")
        eo = np.load(root / f"edge_ori_d{start_depth}.npy")
        ro = np.load(root / f"ring_ori_d{start_depth}.npy")
        previous_previous = np.load(root / f"ranks_d{start_depth - 1}.npy") if start_depth else np.empty(0, dtype=np.uint64)
        sizes = list(map(int, meta["sizes"]))
        timings = list(map(float, meta["timings_seconds"]))
    else:
        ep = SOLVED.edge_perm[None, :].copy()
        eo = SOLVED.edge_ori[None, :].copy()
        ro = SOLVED.ring_ori[None, :].copy()
        ranks = rank_states(ep, eo, ro)
        _save_layer(root, 0, ranks, ep, eo, ro)
        sizes = [1]
        timings = [0.0]
        previous_previous = np.empty(0, dtype=np.uint64)

    for depth in range(start_depth + 1, max_depth + 1):
        started = perf_counter()
        n = len(ep)
        cand_ep = np.empty((n * 12, 12), dtype=np.uint8)
        cand_eo = np.empty((n * 12, 12), dtype=np.uint8)
        cand_ro = np.empty((n * 12, 6), dtype=np.uint8)
        for action in range(12):
            sl = slice(action * n, (action + 1) * n)
            cand_ep[sl], cand_eo[sl], cand_ro[sl] = apply_action_batch(ep, eo, ro, action)
        cand_ranks = rank_states(cand_ep, cand_eo, cand_ro)
        unique_ranks, first = np.unique(cand_ranks, return_index=True)
        if len(previous_previous):
            loc = np.searchsorted(previous_previous, unique_ranks)
            seen = (loc < len(previous_previous)) & (previous_previous[np.minimum(loc, len(previous_previous) - 1)] == unique_ranks)
            keep = ~seen
            unique_ranks = unique_ranks[keep]
            first = first[keep]
        next_ep = cand_ep[first]
        next_eo = cand_eo[first]
        next_ro = cand_ro[first]
        previous_previous = ranks
        ranks, ep, eo, ro = unique_ranks, next_ep, next_eo, next_ro
        elapsed = perf_counter() - started
        sizes.append(int(len(ranks)))
        timings.append(elapsed)
        _save_layer(root, depth, ranks, ep, eo, ro)
        meta = {"depth": depth, "sizes": sizes, "timings_seconds": timings, "total_states": int(sum(sizes))}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"depth={depth} layer={len(ranks):,} total={sum(sizes):,} seconds={elapsed:.2f}", flush=True)
    return json.loads(meta_path.read_text(encoding="utf-8"))
