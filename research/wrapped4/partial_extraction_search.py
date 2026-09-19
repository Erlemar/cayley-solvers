"""Search for placing a smallest prefix, charging disorder of the remaining tail."""
import time
from functools import lru_cache

import numpy as np
from numba import njit, types
from numba.typed import Dict

from wide_search import CACHE_KEY, STATE, apply_packed, pack


@njit(cache=True)
def project(state, k, n):
    words = np.array(state, np.uint64)
    for i in range(n):
        shift = np.uint64(6 * (i % 10))
        value = (words[i // 10] >> shift) & np.uint64(63)
        if value >= k:
            words[i // 10] |= np.uint64(63) << shift
    return words[0], words[1], words[2], words[3]


@njit(cache=True)
def build_prefix_endgame(n, k, radius, capacity):
    distances = Dict.empty(key_type=STATE, value_type=types.int16)
    queue = np.empty((capacity, 4), np.uint64)
    words = np.zeros(4, np.uint64)
    for i in range(n):
        value = i if i < k else 63
        words[i // 10] |= np.uint64(value) << np.uint64(6 * (i % 10))
    identity = (words[0], words[1], words[2], words[3])
    distances[identity] = np.int16(0)
    queue[0] = words
    begin, end = 0, 1
    for depth in range(radius):
        tail = end
        for index in range(begin, end):
            state = (queue[index, 0], queue[index, 1], queue[index, 2], queue[index, 3])
            for action in range(2 * (n - 3)):
                other = apply_packed(state, action, n)
                if other not in distances:
                    assert tail < capacity
                    distances[other] = np.int16(depth + 1)
                    queue[tail] = other
                    tail += 1
        begin, end = end, tail
    return distances


@lru_cache(maxsize=40)
def endgame(n, k):
    return build_prefix_endgame(n, k, 4, 2500000)


@njit(cache=True)
def local_inversions(values, marked):
    total = 0
    selected = 0
    for i in range(4):
        for j in range(i + 1, 4):
            if values[i] > values[j]:
                total += 1
                selected += min(values[i], values[j]) < marked
    return total, selected


@njit(cache=True)
def dfs(state, inv, marked_inv, remaining, allowance, last, depth, n, k,
        failed, path, stats, cap, seed, priority, finish):
    stats[0] += 1
    if stats[0] > cap:
        return -1
    if allowance < 0 or 3 * remaining < marked_inv:
        return 0
    if marked_inv == 0:
        stats[1] = depth
        return 1
    if remaining == 0:
        return 0
    if remaining <= 5:
        projected = project(state, k, n)
        if projected in finish:
            if finish[projected] > remaining:
                return 0
        elif remaining <= 4:
            return 0
    key = (state, last)
    if key in failed and failed[key] >= remaining:
        return 0
    actions = np.empty(2 * (n - 3), np.int64)
    next_inv = np.empty(2 * (n - 3), np.int64)
    next_marked = np.empty(2 * (n - 3), np.int64)
    next_allowance = np.empty(2 * (n - 3), np.int64)
    scores = np.empty(2 * (n - 3), np.int64)
    count = 0
    for action in range(2 * (n - 3)):
        if last >= 0 and action == (last ^ 1):
            continue
        if last >= 0 and abs(last // 2 - action // 2) >= 4 and action < last:
            continue
        values = np.empty(4, np.int64)
        for j in range(4):
            position = action // 2 + j
            values[j] = int((state[position // 10] >> np.uint64(6 * (position % 10))) & np.uint64(63))
        old_i, old_m = local_inversions(values, k)
        new_values = np.empty(4, np.int64)
        for j in range(4):
            new_values[j] = values[(j + (1 if action % 2 == 0 else 3)) % 4]
        new_i, new_m = local_inversions(new_values, k)
        di, dm = new_i - old_i, new_m - old_m
        assert di in (-3, -1, 1, 3)
        aa = allowance - 3 - di
        mm = marked_inv + dm
        if aa < 0 or mm > 3 * (remaining - 1):
            continue
        score = (mm if priority == 0 else inv + di) * (2 * n) + (action + seed) % (2 * n)
        pos = count
        while pos > 0 and scores[pos - 1] > score:
            actions[pos] = actions[pos - 1]
            next_inv[pos] = next_inv[pos - 1]
            next_marked[pos] = next_marked[pos - 1]
            next_allowance[pos] = next_allowance[pos - 1]
            scores[pos] = scores[pos - 1]
            pos -= 1
        actions[pos] = action
        next_inv[pos] = inv + di
        next_marked[pos] = mm
        next_allowance[pos] = aa
        scores[pos] = score
        count += 1
    for index in range(count):
        action = actions[index]
        path[depth] = action
        result = dfs(apply_packed(state, action, n), next_inv[index], next_marked[index],
                     remaining - 1, next_allowance[index], action, depth + 1, n, k,
                     failed, path, stats, cap, seed, priority, finish)
        if result:
            return result
    failed[key] = remaining
    return 0


def solve(p, k, allowance, cap=30000, seed=0, priority=0):
    n = len(p)
    assert n <= 40 and sorted(p) == list(range(n))
    inv = sum(p[i] > p[j] for i in range(n) for j in range(i + 1, n))
    marked_inv = sum(p[i] > p[j] and min(p[i], p[j]) < k for i in range(n) for j in range(i + 1, n))
    budget = (inv + allowance) // 3
    failed = Dict.empty(key_type=CACHE_KEY, value_type=types.int64)
    path = np.zeros(budget + 1, np.int64)
    stats = np.zeros(2, np.int64)
    start = time.monotonic()
    status = dfs(pack(p), inv, marked_inv, budget, allowance, -1, 0, n, k,
                 failed, path, stats, cap, seed, priority, endgame(n, k))
    row = {'status': 'found' if status == 1 else 'node_limit' if status < 0 else 'exhausted',
           'nodes': int(stats[0]), 'seconds': time.monotonic() - start}
    if status == 1:
        word = [(int(a) // 2, 1 if int(a) % 2 == 0 else -1) for a in path[:int(stats[1])]]
        state = list(p)
        for i, d in word:
            block = state[i:i + 4]
            state[i:i + 4] = block[d:] + block[:d]
        assert state[:k] == list(range(k))
        tail_inv = sum(state[i] > state[j] for i in range(k, n) for j in range(i + 1, n))
        charge = 3 * len(word) + tail_inv - inv
        assert charge <= allowance
        row.update(word=word, length=len(word), endpoint=state, tail_inversions=tail_inv, charge=charge)
    return row
