"""Exact three-token projections, used only to rank positive beam searches."""
from functools import lru_cache

import numpy as np
from numba import njit


@njit(cache=True)
def move_position(pos, start, direction):
    if pos < start or pos >= start + 4:
        return pos
    return start + (pos - start - direction) % 4


@njit(cache=True)
def build(n, groups):
    tables = np.full((len(groups), n * n * n), -1, np.int16)
    queue = np.empty(n * n * n, np.int64)
    for group in range(len(groups)):
        a, b, c = groups[group]
        goal = (a * n + b) * n + c
        tables[group, goal] = 0
        queue[0] = goal
        begin, end = 0, 1
        while begin < end:
            state = queue[begin]
            begin += 1
            a, b, c = state // (n * n), state // n % n, state % n
            distance = tables[group, state] + 1
            for start in range(n - 3):
                for direction in (-1, 1):
                    aa = move_position(a, start, direction)
                    bb = move_position(b, start, direction)
                    cc = move_position(c, start, direction)
                    other = (aa * n + bb) * n + cc
                    if tables[group, other] < 0:
                        tables[group, other] = distance
                        queue[end] = other
                        end += 1
        assert end == n * (n - 1) * (n - 2)
    return tables


@lru_cache(maxsize=4)
def tables(n, k):
    groups = [(i, i + 1, i + 2) for i in range(0, k - 2, 3)]
    if k % 3:
        groups.append((k - 3, k - 2, k - 1))
    groups = np.array(groups, np.int64)
    return groups, build(n, groups)


@njit(cache=True)
def score(state, n, k, groups, distances):
    positions = np.empty(k, np.int64)
    for i in range(n):
        token = int((state[i // 10] >> np.uint64(6 * (i % 10))) & np.uint64(63))
        if token < k:
            positions[token] = i
    total = 0
    for group in range(len(groups)):
        a, b, c = groups[group]
        key = (positions[a] * n + positions[b]) * n + positions[c]
        total += distances[group, key]
    return total
