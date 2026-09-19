"""Symmetry-diverse beam with meetings against translated earlier layers.

All symmetries fix the matching permutation and the identity. Only
independently replayed complete words are mathematical certificates.
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
from matching_beam import expand
from near_rotation_search import potential
from wide_search import STATE, apply_packed, build_endgame, pack


@njit(cache=True)
def decode(state, n):
    return np.array([int((state[i // 10] >> np.uint64(6 * (i % 10))) & np.uint64(63))
                     for i in range(n)], np.int64)


@njit(cache=True)
def transform(values, n, sign, shift):
    out = np.zeros(4, np.uint64)
    for j in range(n):
        v = (sign * values[(sign * (j - shift)) % n] + shift) % n
        out[j // 10] |= np.uint64(v) << np.uint64(6 * (j % 10))
    return out[0], out[1], out[2], out[3]


@njit(cache=True)
def canonical(state, n):
    values = decode(state, n)
    best = state
    for shift in range(n):
        sign = 1 if shift % 2 == 0 else -1
        trial = transform(values, n, sign, shift)
        if trial < best:
            best = trial
    return best


@njit(cache=True)
def select_distinct(states, order, n, width):
    seen = Dict.empty(key_type=STATE, value_type=types.int64)
    keep = np.empty(min(width, len(order)), np.int64)
    count = 0
    for index in order:
        state = (states[index, 0], states[index, 1], states[index, 2], states[index, 3])
        key = canonical(state, n)
        if key in seen:
            continue
        seen[key] = index
        keep[count] = index
        count += 1
        if count == width:
            break
    return keep[:count]


@njit(cache=True)
def meeting(states, earlier, p, n):
    opposite = Dict.empty(key_type=STATE, value_type=types.int64)
    for index in range(len(earlier)):
        state = (earlier[index, 0], earlier[index, 1], earlier[index, 2], earlier[index, 3])
        values = decode(state, n)
        for i in range(n):
            values[i] = p[values[i]]
        for shift in range(n):
            sign = 1 if shift % 2 == 0 else -1
            key = transform(values, n, sign, shift)
            opposite[key] = index * n + shift
    for index in range(len(states)):
        state = (states[index, 0], states[index, 1], states[index, 2], states[index, 3])
        if state in opposite:
            value = opposite[state]
            return index, value // n, value % n
    return -1, -1, -1


def prefix(history, depth, index):
    actions = []
    for parents, moves in reversed(history[:depth]):
        actions.append(int(moves[index]))
        index = int(parents[index])
    return [(a // 2, 1 if a % 2 == 0 else -1) for a in reversed(actions)]


def run(n, budget, width, seed, strata=False):
    assert n % 4 == 0
    p = [(i + n // 2 + (1 if i % 2 == 0 else -1)) % n for i in range(n)]
    assert [p[p[i]] for i in range(n)] == list(range(n))
    endgame = build_endgame(n, 4, 5000000)
    states = np.array([pack(p)], np.uint64)
    lifts = np.array([balanced_keys(p)], np.int64)
    lengths = np.array([int(potential(lifts[0], n))], np.int64)
    lasts = np.array([-1], np.int64)
    history = []
    layers = [states]
    trace = []
    begin = time.monotonic()
    word = None
    route = None
    for depth in range(1, budget + 1):
        ns, nl, ne, parents, actions, scores, found = expand(
            states, lifts, lengths, lasts, n, budget - depth, endgame, seed + 1337 * depth)
        if found >= 0:
            word = prefix(history, depth - 1, int(parents[found]))
            a = int(actions[found])
            word.append((a // 2, 1 if a % 2 == 0 else -1))
            state = tuple(np.uint64(x) for x in ns[found])
            while int(endgame[state]):
                d = int(endgame[state])
                for a in range(2 * n):
                    trial = tuple(np.uint64(x) for x in apply_packed(state, a, n))
                    if trial in endgame and int(endgame[trial]) == d - 1:
                        word.append((a // 2, 1 if a % 2 == 0 else -1))
                        state = trial
                        break
                else:
                    raise AssertionError('Missing endgame descent')
            route = 'endgame'
            break
        if not len(scores):
            break
        other_depth = budget - depth
        if 4 < other_depth < depth:
            first, second, shift = meeting(ns, layers[other_depth], np.array(p, np.int64), n)
            if first >= 0:
                word = prefix(history, depth - 1, int(parents[first]))
                action = int(actions[first])
                word.append((action // 2, 1 if action % 2 == 0 else -1))
                backward = prefix(history, other_depth, int(second))
                sign = 1 if shift % 2 == 0 else -1
                transformed = [((i + shift) % n, d) if sign == 1
                               else ((shift - i - 3) % n, -d) for i, d in backward]
                word += [(i, -d) for i, d in reversed(transformed)]
                route = 'symmetry_meeting'
                break
        if strata:
            values = np.unique(ne)
            allocation = max(1, width // len(values))
            parts = []
            for value in values:
                group = np.flatnonzero(ne == value)
                order = group[np.argsort(scores[group])]
                parts.append(select_distinct(ns, order, n, allocation))
            keep = np.concatenate(parts)
        else:
            keep = select_distinct(ns, np.argsort(scores), n, width)
        states, lifts, lengths, lasts = ns[keep].copy(), nl[keep].copy(), ne[keep].copy(), actions[keep].copy()
        history.append((parents[keep].copy(), lasts))
        layers.append(states)
        row = {'depth': depth, 'candidates': len(scores), 'kept': len(keep),
               'min_adjacent_length': int(min(lengths)), 'max_adjacent_length': int(max(lengths))}
        trace.append(row)
        if depth % 5 == 0:
            print(json.dumps(row), flush=True)
    result = {'n': n, 'permutation': p, 'budget': budget, 'width': width, 'seed': seed, 'strata': strata,
              'status': 'found' if word is not None else 'inconclusive', 'trace': trace,
              'seconds': time.monotonic() - begin,
              'scope': 'Heuristic breadth truncation; unsuccessful runs prove no lower bound.'}
    if word is not None:
        assert len(word) <= budget
        assert apply_to(p, word) == list(range(n))
        result.update(word=word, length=len(word), route=route)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=24)
    parser.add_argument('--budget', type=int, default=46)
    parser.add_argument('--width', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--strata', action='store_true')
    args = parser.parse_args()
    result = run(args.n, args.budget, args.width, args.seed, args.strata)
    suffix = '_strata' if args.strata else ''
    name = f'matching_bidirectional_n{args.n}_b{args.budget}_w{args.width}_s{args.seed}{suffix}.json'
    Path(__file__).with_name(name).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('word', 'trace', 'permutation')}), flush=True)


if __name__ == '__main__':
    main()
