"""Edge pattern databases for Christopher's Jewel."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import prod
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from .puzzle import ACTION_EDGE_FLIP, ACTION_EDGE_SRC, JewelState


ACTION_EDGE_DEST = np.argsort(ACTION_EDGE_SRC, axis=1).astype(np.uint8)


def partial_permutation_count(n: int, k: int) -> int:
    return prod(range(n - k + 1, n + 1))


def rank_pattern(positions: np.ndarray, orientations: np.ndarray) -> int:
    k = len(positions)
    rank = 0
    used: list[int] = []
    for i in range(k):
        pos = int(positions[i])
        digit = pos - sum(prev < pos for prev in used)
        rank = rank * (12 - i) + digit
        used.append(pos)
    ori_bits = sum(int(orientations[i]) << i for i in range(k))
    return (rank << k) | ori_bits


def rank_patterns(positions: np.ndarray, orientations: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.uint8)
    n, k = positions.shape
    rank = np.zeros(n, dtype=np.uint64)
    for i in range(k):
        if i:
            used_less = np.sum(positions[:, :i] < positions[:, i : i + 1], axis=1, dtype=np.uint64)
        else:
            used_less = 0
        digit = positions[:, i].astype(np.uint64) - used_less
        rank = rank * np.uint64(12 - i) + digit
    ori_bits = np.zeros(n, dtype=np.uint64)
    for i in range(k):
        ori_bits |= orientations[:, i].astype(np.uint64) << np.uint64(i)
    return (rank << np.uint64(k)) | ori_bits


def apply_pattern_action(positions: np.ndarray, orientations: np.ndarray, action: int) -> tuple[np.ndarray, np.ndarray]:
    dest = ACTION_EDGE_DEST[action][positions]
    flip = ACTION_EDGE_FLIP[action][dest]
    return dest, np.bitwise_xor(orientations, flip)


@dataclass(slots=True)
class EdgePatternDatabase:
    pieces: tuple[int, ...]
    distances: np.ndarray
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path, mmap_mode: str | None = "r") -> "EdgePatternDatabase":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(tuple(meta["pieces"]), np.load(path, mmap_mode=mmap_mode), path)

    def abstract_rank(self, state: JewelState) -> int:
        inverse = np.empty(12, dtype=np.uint8)
        inverse[state.edge_perm] = np.arange(12, dtype=np.uint8)
        positions = inverse[np.asarray(self.pieces, dtype=np.uint8)]
        orientations = state.edge_ori[positions]
        return rank_pattern(positions, orientations)

    def heuristic(self, state: JewelState) -> int:
        value = int(self.distances[self.abstract_rank(state)])
        if value == 255:
            raise RuntimeError("pattern database contains an unreachable entry")
        return value


def build_edge_pdb(path: str | Path, pieces: Sequence[int], *, overwrite: bool = False) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pieces = tuple(map(int, pieces))
    if not pieces or len(set(pieces)) != len(pieces) or any(not 0 <= p < 12 for p in pieces):
        raise ValueError("pieces must be distinct edge ids in 0..11")
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists() and not overwrite:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    k = len(pieces)
    size = partial_permutation_count(12, k) * (1 << k)
    distances = np.full(size, 255, dtype=np.uint8)
    positions = np.asarray(pieces, dtype=np.uint8)[None, :]
    orientations = np.zeros((1, k), dtype=np.uint8)
    ranks = rank_patterns(positions, orientations)
    distances[ranks] = 0
    layer_sizes = [1]
    timings: list[float] = [0.0]
    depth = 0

    while len(positions):
        started = perf_counter()
        n = len(positions)
        cand_pos = np.empty((n * 12, k), dtype=np.uint8)
        cand_ori = np.empty((n * 12, k), dtype=np.uint8)
        for action in range(12):
            sl = slice(action * n, (action + 1) * n)
            cand_pos[sl], cand_ori[sl] = apply_pattern_action(positions, orientations, action)
        cand_ranks = rank_patterns(cand_pos, cand_ori)
        unique_ranks, first = np.unique(cand_ranks, return_index=True)
        keep = distances[unique_ranks] == 255
        ranks = unique_ranks[keep]
        first = first[keep]
        positions = cand_pos[first]
        orientations = cand_ori[first]
        depth += 1
        distances[ranks] = depth
        elapsed = perf_counter() - started
        layer_sizes.append(int(len(ranks)))
        timings.append(elapsed)
        print(f"pdb={pieces} depth={depth} layer={len(ranks):,} seen={int(np.count_nonzero(distances != 255)):,}", flush=True)

    np.save(path, distances)
    meta = {
        "pieces": list(pieces),
        "size": int(size),
        "depth": depth - 1,
        "layer_sizes": layer_sizes[:-1],
        "timings_seconds": timings[:-1],
        "reachable": int(np.count_nonzero(distances != 255)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def ring_heuristic(state: JewelState) -> int:
    twists = state.ring_ori.astype(np.int16)
    return int(np.minimum(twists, 4 - twists).sum())


def combined_heuristic(state: JewelState, pdbs: Sequence[EdgePatternDatabase]) -> int:
    values = [ring_heuristic(state)]
    values.extend(pdb.heuristic(state) for pdb in pdbs)
    return max(values)

