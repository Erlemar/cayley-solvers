"""Bounded exact search for up to 40 positions using four packed words.

Successful words are replayed without using the packed representation.
Node-limited searches are inconclusive. The heuristic and pruning are
the same as in near_rotation_search.py; packing is widened to six bits.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from numba import njit, types
from numba.typed import Dict

from affine_insertion import balanced_keys
from general_sort import apply_to
from near_rotation_search import advance_lift, potential
from rotation_bounds import diameter_lower


STATE = types.UniTuple(types.uint64, 4)
CACHE_KEY = types.Tuple((STATE, types.int64))


def pack(p):
    assert len(p) <= 40
    words = [0] * 4
    for i, v in enumerate(p):
        words[i // 10] |= int(v) << (6 * (i % 10))
    return tuple(np.uint64(v) for v in words)


@njit(cache=True)
def apply_packed(state, action, n):
    words = np.array(state, dtype=np.uint64)
    values = np.empty(4, np.uint64)
    positions = np.empty(4, np.int64)
    for j in range(4):
        pos = (action // 2 + j) % n
        positions[j] = pos
        shift = np.uint64(6 * (pos % 10))
        values[j] = (words[pos // 10] >> shift) & np.uint64(63)
    for j in range(4):
        pos = positions[j]
        shift = np.uint64(6 * (pos % 10))
        source = (j + 1) % 4 if action % 2 == 0 else (j + 3) % 4
        words[pos // 10] &= ~(np.uint64(63) << shift)
        words[pos // 10] |= values[source] << shift
    return words[0], words[1], words[2], words[3]


@njit(cache=True)
def build_endgame(n, radius, capacity, wrapped=True):
    dist = Dict.empty(key_type=STATE, value_type=types.int16)
    queue = np.empty((capacity, 4), np.uint64)
    identity = np.zeros(4, np.uint64)
    for i in range(n):
        identity[i // 10] |= np.uint64(i) << np.uint64(6 * (i % 10))
    state = (identity[0], identity[1], identity[2], identity[3])
    dist[state] = np.int16(0)
    queue[0] = identity
    begin, end = 0, 1
    for depth in range(radius):
        tail = end
        for index in range(begin, end):
            state = (queue[index, 0], queue[index, 1], queue[index, 2], queue[index, 3])
            for action in range(2 * (n if wrapped else n - 3)):
                next_state = apply_packed(state, action, n)
                if next_state not in dist:
                    assert tail < capacity
                    queue[tail] = next_state
                    dist[next_state] = np.int16(depth + 1)
                    tail += 1
        begin, end = end, tail
    return dist


@njit(cache=True)
def dfs(state, lift, ell, remaining, last, depth, n, radius, endgame,
        failed, path, stats, cap, commute, seed, wrapped):
    stats[0] += 1
    if stats[0] > cap:
        return -1
    if ell > 3 * remaining:
        return 0
    if state in endgame:
        if endgame[state] <= remaining:
            stats[1] = depth
            return 1
        return 0
    if remaining <= radius:
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
    for action in range(2 * (n if wrapped else n - 3)):
        if last >= 0 and action == (last ^ 1):
            continue
        if last >= 0 and commute[last // 2, action // 2] and action < last:
            continue
        next_lift, next_ell = advance_lift(lift, ell, action, n, wrapped)
        if next_ell > 3 * (remaining - 1):
            continue
        next_state = apply_packed(state, action, n)
        score = next_ell * (2 * n) + (action + seed) % (2 * n)
        pos = count
        while pos > 0 and scores[pos - 1] > score:
            actions[pos] = actions[pos - 1]
            lengths[pos] = lengths[pos - 1]
            scores[pos] = scores[pos - 1]
            lifts[pos] = lifts[pos - 1]
            states[pos] = states[pos - 1]
            pos -= 1
        actions[pos] = action
        lengths[pos] = next_ell
        scores[pos] = score
        lifts[pos] = next_lift
        states[pos] = next_state
        count += 1
    for pos in range(count):
        action = actions[pos]
        path[depth] = action
        next_state = (states[pos, 0], states[pos, 1], states[pos, 2], states[pos, 3])
        result = dfs(next_state, lifts[pos], lengths[pos], remaining - 1,
                     action, depth + 1, n, radius, endgame, failed, path,
                     stats, cap, commute, seed, wrapped)
        if result != 0:
            return result
    failed[key] = np.int16(remaining)
    return 0


def solve(p, budget, endgame, radius=4, cap=100000, seed=0, wrapped=True):
    n = len(p)
    assert n <= 40
    lift = np.array(balanced_keys(list(p)) if wrapped else p, np.int64)
    ell = int(potential(lift, n))
    assert (budget - ell) % 2 == 0
    failed = Dict.empty(key_type=CACHE_KEY, value_type=types.int16)
    path = np.zeros(budget + 1, np.int64)
    stats = np.zeros(2, np.int64)
    supports = [{(i + j) % n for j in range(4)} for i in range(n)]
    commute = np.array([[not (a & b) for b in supports] for a in supports])
    begin = time.monotonic()
    result = dfs(pack(p), lift, ell, budget, -1, 0, n, radius, endgame,
                 failed, path, stats, cap, commute, seed, wrapped)
    row = {'n': n, 'permutation': list(p), 'budget': budget,
           'adjacent_length': ell,
           'status': {1: 'found', 0: 'exhausted', -1: 'node_limit'}[result],
           'nodes': int(stats[0]), 'seconds': time.monotonic() - begin}
    if result == 1:
        actions = list(map(int, path[:stats[1]]))
        state = pack(p)
        for action in actions:
            state = tuple(np.uint64(x) for x in apply_packed(state, action, n))
        while int(endgame[state]):
            old = int(endgame[state])
            for action in range(2 * (n if wrapped else n - 3)):
                next_state = tuple(np.uint64(x) for x in apply_packed(state, action, n))
                if next_state in endgame and int(endgame[next_state]) == old - 1:
                    actions.append(action)
                    state = next_state
                    break
            else:
                raise AssertionError('Missing endgame descent')
        word = [(a // 2, 1 if a % 2 == 0 else -1) for a in actions]
        assert len(word) <= budget
        assert apply_to(p, word) == list(range(n))
        row['word'] = word
        row['length'] = len(word)
    return row


def validate():
    """All generator operations and comparison to exact small peripheries."""
    import random
    rng = random.Random(20260926)
    moves = 0
    for n in range(5, 41):
        for _ in range(4):
            p = list(range(n))
            rng.shuffle(p)
            for a in range(2 * n):
                actual = tuple(int(x) for x in apply_packed(pack(p), a, n))
                expected = tuple(int(x) for x in pack(apply_to(p, [(a // 2, 1 if a % 2 == 0 else -1)])))
                assert actual == expected
                moves += 1
    data = json.loads((Path(__file__).parent / 'diameter_exact/n8.json').read_text())
    endgame = build_endgame(8, 4, 1000000)
    for p in data['farthest']:
        assert solve(p, 5, endgame, cap=100000)['status'] == 'exhausted'
        assert solve(p, 7, endgame, cap=100000)['status'] == 'found'
    result = {'packed_moves_verified': moves,
              'n8_peripheral_pairs_verified': len(data['farthest'])}
    (Path(__file__).parent / 'wide_search_validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=24)
    parser.add_argument('--cap', type=int, default=100000)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    if args.validate:
        validate()
        return
    n = args.n
    assert n % 2 == 0 and 6 <= n <= 40
    endgame = build_endgame(n, 4, 5000000)
    print('endgame', n, len(endgame), flush=True)
    rows = []
    for count in range(n // 2 + 1):
        p = [(i + n // 2) % n for i in range(n)]
        for i in range(count):
            p[2 * i], p[2 * i + 1] = p[2 * i + 1], p[2 * i]
        ell = int(potential(np.array(balanced_keys(p), np.int64), n))
        budget = diameter_lower(n)
        if (budget - ell) % 2:
            budget -= 1
        row = solve(p, budget, endgame, cap=args.cap, seed=19)
        row['matching_size'] = count
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ('word', 'permutation')}), flush=True)
        (Path(__file__).parent / f'wide_matching_n{n}.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
