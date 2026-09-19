"""Bounded 4-cycle search with prescribed monotone token directions."""
import time

import numpy as np
from numba import njit, types
from numba.typed import Dict

from affine_insertion import balanced_keys
from general_sort import apply_to
from near_rotation_search import advance_lift, potential
from wide_search import CACHE_KEY, apply_packed, pack


@njit(cache=True)
def allowed(state, action, n, colors):
    for j in range(4):
        pos = (action // 2 + j) % n
        label = int((state[pos // 10] >> np.uint64(6 * (pos % 10))) & np.uint64(63))
        color = colors[label]
        if color == 0:
            continue
        delta = (3 if j == 0 else -1) if action % 2 == 0 else (-3 if j == 3 else 1)
        if color > 0:
            if delta <= 0 or delta > (label - pos) % n:
                return False
        else:
            if delta >= 0 or -delta > (pos - label) % n:
                return False
    return True


@njit(cache=True)
def dfs(state, lift, ell, remaining, last, depth, n, radius, endgame,
        failed, path, stats, cap, commute, seed, colors):
    stats[0] += 1
    if stats[0] > cap:
        return -1
    if ell > 3 * remaining:
        return 0
    if ell == 0:
        stats[1] = depth
        return 1
    if state in endgame:
        if endgame[state] > remaining:
            return 0
    elif remaining <= radius:
        return 0
    if remaining == 0:
        return 0
    key = (state, last)
    if key in failed and failed[key] >= remaining:
        return 0
    actions = np.empty(2 * n, np.int64)
    lengths = np.empty(2 * n, np.int64)
    scores = np.empty(2 * n, np.int64)
    lifts = np.empty((2 * n, n), np.int64)
    states = np.empty((2 * n, 4), np.uint64)
    count = 0
    for action in range(2 * n):
        if last >= 0 and action == (last ^ 1):
            continue
        if last >= 0 and commute[last // 2, action // 2] and action < last:
            continue
        if not allowed(state, action, n, colors):
            continue
        next_lift, next_ell = advance_lift(lift, ell, action, n, True)
        if next_ell > 3 * (remaining - 1):
            continue
        next_state = apply_packed(state, action, n)
        score = next_ell * (2 * n) + (action + seed) % (2 * n)
        pos = count
        while pos > 0 and scores[pos - 1] > score:
            actions[pos] = actions[pos - 1]
            lengths[pos] = lengths[pos - 1]
            scores[pos] = scores[pos - 1]
            lifts[pos, :] = lifts[pos - 1, :]
            states[pos, :] = states[pos - 1, :]
            pos -= 1
        actions[pos] = action
        lengths[pos] = next_ell
        scores[pos] = score
        lifts[pos, :] = next_lift
        states[pos, :] = next_state
        count += 1
    for pos in range(count):
        action = actions[pos]
        path[depth] = action
        next_state = (states[pos, 0], states[pos, 1], states[pos, 2], states[pos, 3])
        result = dfs(next_state, lifts[pos], lengths[pos], remaining - 1, action, depth + 1,
                     n, radius, endgame, failed, path, stats, cap, commute, seed, colors)
        if result != 0:
            return result
    failed[key] = np.int16(remaining)
    return 0


def solve(p, budget, endgame, colors, cap=100000, seed=0, radius=4):
    n = len(p)
    colors = np.array(colors, dtype=np.int8)
    lift = np.array(balanced_keys(list(p)), np.int64)
    ell = int(potential(lift, n))
    assert (budget - ell) % 2 == 0
    failed = Dict.empty(key_type=CACHE_KEY, value_type=types.int16)
    path = np.zeros(budget + 1, np.int64)
    stats = np.zeros(2, np.int64)
    supports = [{(i + j) % n for j in range(4)} for i in range(n)]
    commute = np.array([[not (a & b) for b in supports] for a in supports])
    begin = time.monotonic()
    result = dfs(pack(p), lift, ell, budget, -1, 0, n, radius, endgame,
                 failed, path, stats, cap, commute, seed, colors)
    row = {'status': {1: 'found', 0: 'exhausted_for_this_anchor_assignment', -1: 'node_limit'}[result],
           'nodes': int(stats[0]), 'seconds': time.monotonic() - begin}
    if result == 1:
        word = [(int(a) // 2, 1 if int(a) % 2 == 0 else -1) for a in path[:stats[1]]]
        assert apply_to(p, word) == list(range(n))
        # Independent signed-travel validation, not packed-state pruning.
        from twist_inflation import strand_data
        motion, travel, _ = strand_data(p, word)
        for x, color in enumerate(colors):
            if color > 0:
                assert motion[x] == travel[x] == (x - p.index(x)) % n
            elif color < 0:
                assert -motion[x] == travel[x] == (p.index(x) - x) % n
        row.update(word=word, length=len(word))
    return row
